// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.
//
// Minimal ovsensors C++ example covering the complete sensor and result lifecycle.
// Snippet markers are referenced by the public skills; keep them intact.

#include <ovrtx/ovrtx.h>
#include <ovsensors/ovsensors.h>
#include <ovsensors/ovsensors_provider_ovrtx.h>
#include <ovstage/ovstage.h>
#include <ovstage/ovstage_api/ovstage_api_utils.h>
#include <ovstage/ovstage_population.h>

#include "example_paths.h"
#include "image.h"

#include <array>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <string>
#include <string_view>

static constexpr ovsensors_map_output_description_t kCpuMap{ OVSENSORS_MAP_DEVICE_TYPE_CPU, 0 };

namespace
{

constexpr char kDefaultSceneFileName[] = "scene.usda";
constexpr char kProductPath[] = "/Render/Products/CameraProduct";
constexpr char kSensorName[] = "camera";

// Application-owned publication ordinals start above a new stage's write floor.
constexpr ovstage_ordinal_t kOpenOrdinal = 1;

// Match scene.usda's 30 Hz cadence so every step is due.
constexpr double kStepDeltaSeconds = 1.0 / 30.0;
constexpr int kStepCount = 4;

ovx_string_t str(const char* s)
{
    return { s, std::strlen(s) };
}

// [snippet:check-error-helper]
// Return true on success; otherwise print the thread-local last-error detail.
template <typename ResultT>
bool check(ResultT const& result, std::string_view operation)
{
    if (result.status == OVSENSORS_API_SUCCESS)
    {
        return true;
    }

    // The detail is borrowed until the next ovsensors call on this thread.
    ovx_string_t detail = ovsensors_get_last_error();
    std::cerr << "ovsensors " << operation << " failed (status " << static_cast<int>(result.status) << ")";
    if (detail.ptr && detail.length > 0)
    {
        std::cerr << ": " << std::string_view(detail.ptr, detail.length);
    }
    std::cerr << "\n";
    return false;
}
// [/snippet:check-error-helper]

bool checkStage(ovstage_api_status_t status, std::string_view operation)
{
    if (status == OVSTAGE_OK)
    {
        return true;
    }
    std::cerr << "ovstage " << operation << " failed (code " << static_cast<unsigned>(status) << ")\n";
    return false;
}

bool waitStageOp(ovstage_instance_t* stage, ovstage_enqueue_result_t enqueued, std::string_view operation)
{
    if (!checkStage(enqueued.status, operation))
    {
        return false;
    }
    ovstage_op_wait_result_t wait{};
    const ovstage_api_status_t code = ovstage_wait_op(stage, enqueued.op_index, OVSTAGE_TIMEOUT_INFINITE, &wait);
    for (size_t i = 0; i < wait.error_op_id_count; ++i)
    {
        const ovx_string_t e = ovstage_get_last_op_error(stage, wait.error_op_ids[i]);
        std::cerr << "ovstage " << operation << " op " << wait.error_op_ids[i].raw_id
                  << " failed: " << std::string_view(e.ptr ? e.ptr : "", e.length) << "\n";
    }
    // Release every operation before destroying the stage.
    const bool completed = checkStage(code, operation) && wait.error_op_id_count == 0;
    const bool released = checkStage(ovstage_release_op(stage, enqueued.op_index), "release_op");
    return completed && released;
}

bool waitPopulationOp(ovstage_instance_t* stage, ovstage_population_enqueue_result_t enqueued, std::string_view operation)
{
    if (!checkStage(enqueued.status, operation))
    {
        return false;
    }
    const ovstage_api_status_t code =
        ovstage_population_wait_op(stage, enqueued.op_index, OVSTAGE_TIMEOUT_INFINITE, nullptr);
    if (code != OVSTAGE_OK)
    {
        const ovx_string_t e = ovstage_population_get_last_error();
        std::cerr << "ovstage " << operation << " failed: " << std::string_view(e.ptr ? e.ptr : "", e.length) << "\n";
        return false;
    }
    return true;
}

// Seal an application-owned ordinal before using it as a publication bound.
bool sealOrdinal(ovstage_instance_t* stage, ovstage_ordinal_t ordinal)
{
    ovstage_write_floor_desc_t floor{};
    floor.ordinal = ordinal;
    floor.scope = OVSTAGE_SCOPE_ALL;
    return waitStageOp(stage, ovstage_advance_write_floor(stage, &floor), "advance_write_floor");
}

std::filesystem::path executableDir(const char* argv0)
{
#ifdef __linux__
    std::error_code error;
    const std::filesystem::path self{ std::filesystem::canonical("/proc/self/exe", error) };
    if (!error)
    {
        return self.parent_path();
    }
#endif
    return std::filesystem::absolute(argv0).parent_path();
}

// [snippet:read-mapped-output]
// Copy pixels while their mapping is live; no borrowed tensor escapes unmap.
bool readOutput(ovsensors_instance_t* instance, const ovsensors_output_t& output, example::Image& image,
                ovsensors_map_handle_t& liveMap)
{
    ovsensors_mapped_t mapped{};
    if (!check(ovsensors_map_output(instance, output.handle, &kCpuMap,
                                   OVSENSORS_TIMEOUT_INFINITE, &mapped), "map_output"))
        return false;
    liveMap = mapped.map;
    bool valid = false;
    try
    {
        if (mapped.status != OVSENSORS_EVENT_COMPLETED)
            throw std::runtime_error("CPU mapping did not complete: " +
                std::string(mapped.error_message.ptr ? mapped.error_message.ptr : "", mapped.error_message.length));
        // This example has no CUDA dependency. A CPU map must already be ready.
        if (mapped.cuda_sync.stream || mapped.cuda_sync.wait_event)
            throw std::runtime_error("CPU mapping unexpectedly requires CUDA synchronization");
        const DLTensor* pixels = nullptr;
        for (size_t i = 0; mapped.tensors && i < mapped.tensor_count; ++i)
        {
            const auto& tensor = mapped.tensors[i];
            if (tensor.name.ptr && std::string_view(tensor.name.ptr, tensor.name.length) == "LdrColor")
            {
                pixels = tensor.dl;
                break;
            }
        }
        if (!pixels)
            throw std::runtime_error("CPU mapping has no LdrColor image tensor");
        image = example::copyImage(*pixels);
        valid = true;
    }
    catch (const std::exception& error)
    {
        std::cerr << "image validation failed: " << error.what() << "\n";
    }
    const ovsensors_cuda_sync_t consumerDone{ 0, 0 };
    const auto result = ovsensors_unmap_output(instance, liveMap, consumerDone);
    const bool unmapped = check(result, "unmap_output");
    if (result.status == OVSENSORS_API_SUCCESS || result.status == OVSENSORS_API_INVALID_HANDLE)
        liveMap = 0;
    return valid && unmapped;
}
// [/snippet:read-mapped-output]

// [snippet:walk-result-set]
// A cadence hold may have no frames, but a due sensor must eventually yield pixels.
bool harvest(ovsensors_instance_t* instance, const ovsensors_step_result_set_t& results,
             example::Image& image, size_t& outputCount, ovsensors_map_handle_t& liveMap)
{
    if (results.group_count == 0 || !results.groups)
    {
        std::cerr << "step produced no sensor groups\n";
        return false;
    }
    for (size_t g = 0; g < results.group_count; ++g)
    {
        const ovsensors_sensor_result_t& group = results.groups[g];
        if (group.status != OVSENSORS_API_SUCCESS)
        {
            std::cerr << "sensor group failed: "
                      << std::string_view(group.error_message.ptr ? group.error_message.ptr : "",
                                          group.error_message.length) << "\n";
            return false;
        }
        for (size_t f = 0; f < group.frame_count; ++f)
        {
            const ovsensors_frame_t& frame = group.frames[f];
            if (!frame.output_count)
            {
                std::cerr << "sensor frame has no outputs\n";
                return false;
            }
            bool foundColor = false;
            for (size_t o = 0; o < frame.output_count; ++o)
            {
                const ovsensors_output_t& output = frame.outputs[o];
                const std::string_view name(output.name.ptr, output.name.length);
                if (name != "LdrColor" && name != "/Render/Vars/LdrColor")
                    continue;
                foundColor = true;
                if (output.validity != OVSENSORS_OUTPUT_VALID)
                {
                    std::cerr << "LdrColor carries no valid sample\n";
                    return false;
                }
                if (!readOutput(instance, output, image, liveMap))
                    return false;
                ++outputCount;
            }
            if (!foundColor)
            {
                std::cerr << "sensor frame has no LdrColor output\n";
                return false;
            }
        }
    }
    return true;
}
// [/snippet:walk-result-set]

} // namespace

int main(int argc, char* argv[])
{
    const std::string defaultScene{ (executableDir(argv[0]) / kDefaultSceneFileName).string() };
    std::string scenePath = defaultScene;
    std::filesystem::path outputPath = "_output/cpp.png";
    for (int i = 1; i < argc; ++i)
    {
        const std::string_view option(argv[i]);
        if (option == "--help")
        {
            std::cout << "Usage: ovsensors_minimal [--output _output/cpp.png] [--scene scene.usda]\n";
            return 0;
        }
        if ((option != "--output" && option != "--scene") || i + 1 == argc || !argv[i + 1][0] ||
            std::string_view(argv[i + 1]).rfind("--", 0) == 0)
        {
            std::cerr << "Expected --output PATH or --scene PATH; use --help for usage\n";
            return 1;
        }
        if (option == "--output")
            outputPath = argv[++i];
        else
            scenePath = argv[++i];
    }
    // [snippet:register-schema-paths]
    // The public wheel loaders initialize USD lazily. Register its schema paths
    // before the first stage call, as in the ovrtx minimal example.
    const ovrtx_config_t config{};
    if (ovrtx_register_schema_paths(&config).status != OVRTX_API_SUCCESS)
    {
        std::cerr << "ovrtx schema registration failed\n";
        return 1;
    }
    // [/snippet:register-schema-paths]

    ovstage_instance_t* stage = nullptr;
    ovsensors_provider_t provider{};
    bool providerCreated = false;
    ovsensors_instance_t* instance = nullptr;
    ovsensors_sensor_t* sensor = nullptr;
    bool attached = false;
    bool initialized = false;
    ovsensors_map_handle_t liveMap = 0;
    ovsensors_op_id_t liveOp = 0;
    bool hasLiveOp = false;

    // Guard each teardown phase so every exit follows the same order.
    auto cleanup = [&](int exitCode) {
        int code = exitCode;
        // Retain failed cleanup handles for an exact retry before ending the stage borrow.
        if (liveMap)
        {
            const auto result = ovsensors_unmap_output(instance, liveMap, { 0, 0 });
            if (!check(result, "unmap_output during cleanup"))
            {
                code = 1;
                if (result.status != OVSENSORS_API_INVALID_HANDLE)
                    return code;
            }
            liveMap = 0;
        }
        if (hasLiveOp)
        {
            if (!check(ovsensors_release_result_set(instance, liveOp), "release_result_set during cleanup"))
                return 1;
            hasLiveOp = false;
        }
        if (sensor != nullptr && !check(ovsensors_destroy_sensor(sensor), "destroy_sensor"))
        {
            return 1;
        }
        sensor = nullptr;
        if (attached && !check(ovsensors_detach_ovstage(instance), "detach_ovstage"))
        {
            return 1;
        }
        attached = false;
        if (instance != nullptr && !check(ovsensors_destroy_instance(instance), "destroy_instance"))
        {
            return 1;
        }
        instance = nullptr;
        // The provider must outlive every instance created from it.
        if (providerCreated)
        {
            if (ovsensors_provider_ovrtx_destroy(&provider) != OVSENSORS_API_SUCCESS)
            {
                std::cerr << "provider destruction failed\n";
                return 1;
            }
            providerCreated = false;
        }
        if (initialized)
        {
            if (!check(ovsensors_shutdown(), "shutdown"))
                code = 1;
        }
        if (stage != nullptr)
        {
            if (!checkStage(ovstage_destroy_instance(stage), "destroy_instance"))
                code = 1;
        }
        return code;
    };

    // [snippet:populate-stage]
    // Populate through ovstage, then seal the publication ordinal.
    ovstage_instance_desc_t stageDesc{};
    stageDesc.name = "ovsensors.example.minimal";
    if (!checkStage(ovstage_create_instance(&stageDesc, &stage), "create_instance"))
    {
        return cleanup(1);
    }

    std::cerr << "Loading " << scenePath << "...\n";
    const ovstage_population_enqueue_result_t populated = ovstage_population_open_usd_from_file(
        stage, str(scenePath.c_str()), kOpenOrdinal, /* time = */ 0.0, OVSTAGE_POPULATION_DOMAIN_RENDERING);
    if (!waitPopulationOp(stage, populated, "population_open_usd_from_file") || !sealOrdinal(stage, kOpenOrdinal))
    {
        return cleanup(1);
    }
    std::cerr << "Stage populated.\n";
    // [/snippet:populate-stage]

    // [snippet:create-provider-and-instance]
    // The provider creates and owns its renderer. A null config uses ovrtx
    // defaults. The provider declares its supported sensor capabilities.
    std::cerr << "Creating provider. The first run compiles and caches shaders, which takes minutes...\n";
    if (ovsensors_provider_ovrtx_create(str(OVSENSORS_EXAMPLE_OVRTX_LIBRARY), nullptr, &provider) != OVSENSORS_API_SUCCESS)
    {
        std::cerr << "ovsensors_provider_ovrtx_create failed\n";
        return cleanup(1);
    }
    providerCreated = true;

    if (!check(ovsensors_initialize(nullptr), "initialize"))
    {
        return cleanup(1);
    }
    initialized = true;

    if (!check(ovsensors_create_instance(&provider, 1, &instance), "create_instance"))
    {
        return cleanup(1);
    }
    if (!check(ovsensors_attach_ovstage(instance, stage), "attach_ovstage"))
    {
        return cleanup(1);
    }
    attached = true;
    std::cerr << "Provider and instance ready.\n";
    // [/snippet:create-provider-and-instance]

    // [snippet:create-sensor]
    // Select this provider explicitly; adoption still derives member order and cadence
    // from the RenderProduct and its authored sensor prims.
    ovsensors_sensor_spec_t spec{};
    spec.name = str(kSensorName);
    spec.product = str(kProductPath);
    spec.provider = &provider;
    if (!check(ovsensors_create_sensor(instance, &spec, &sensor), "create_sensor"))
    {
        return cleanup(1);
    }

    const ovsensors_descriptor_t* descriptor = nullptr;
    if (!check(ovsensors_get_descriptor(sensor, &descriptor), "get_descriptor"))
    {
        return cleanup(1);
    }
    std::cout << "Sensor " << std::string_view(descriptor->name.ptr, descriptor->name.length)
              << ": " << descriptor->rate.num << "/" << descriptor->rate.den << " Hz\n";
    // [/snippet:create-sensor]

    // [snippet:step-fetch-release]
    // Seal, step, fetch, harvest, and release every operation.
    example::Image image;
    size_t totalOutputCount = 0;
    for (int step = 0; step < kStepCount; ++step)
    {
        const ovstage_ordinal_t ordinal = kOpenOrdinal + 1 + static_cast<ovstage_ordinal_t>(step);
        if (!sealOrdinal(stage, ordinal))
        {
            return cleanup(1);
        }

        const std::array stepSet{ sensor };
        const ovsensors_enqueue_result_t stepped =
            ovsensors_step(instance, stepSet.data(), stepSet.size(), kStepDeltaSeconds, ordinal);
        if (!check(stepped, "step"))
        {
            return cleanup(1);
        }

        liveOp = stepped.op_index;
        hasLiveOp = true;
        ovsensors_step_result_set_t results{};
        const ovsensors_result_t fetched =
            ovsensors_fetch_result_set(instance, stepped.op_index, OVSENSORS_TIMEOUT_INFINITE, &results);
        if (fetched.status != OVSENSORS_API_SUCCESS)
        {
            // Terminal operation failure has operation-owned detail; other failures are thread-local.
            if (fetched.status == OVSENSORS_API_OP_FAILED)
            {
                const ovx_string_t detail = ovsensors_get_last_op_error(instance, stepped.op_index);
                std::cerr << "fetch_result_set failed: "
                          << std::string_view(detail.ptr ? detail.ptr : "", detail.length) << "\n";
            }
            else
                check(fetched, "fetch_result_set");
            return cleanup(1);
        }

        const bool valid = harvest(instance, results, image, totalOutputCount, liveMap);

        const bool released = check(ovsensors_release_result_set(instance, liveOp), "release_result_set");
        if (released)
            hasLiveOp = false;
        if (!valid || !released)
        {
            return cleanup(1);
        }
    }
    // [/snippet:step-fetch-release]

    if (totalOutputCount == 0 || image.rgb.empty())
    {
        std::cerr << "no image outputs produced\n";
        return cleanup(1);
    }
    // Complete resource teardown before publishing success or writing the artifact.
    if (cleanup(0) != 0)
        return 1;
    try
    {
        outputPath = std::filesystem::absolute(outputPath);
        example::writePng(image, outputPath);
    }
    catch (const std::exception& error)
    {
        std::cerr << "PNG write failed: " << error.what() << "\n";
        return 1;
    }
    std::cout << "Saved " << image.width << "x" << image.height << " PNG: " << outputPath.string() << "\n";
    return 0;
}
