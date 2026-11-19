// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary

#include "pose_provider.h"
#include <ovsensors/ovsensors.h>
#include <ovstage/ovstage.h>
#include <ovstage/ovstage_population.h>
#include <ovx/path_dictionary/path_dictionary.h>
#include <ovx/path_dictionary/path_dictionary_utils.h>

#include <array>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#if defined(_WIN32)
#include <windows.h>
#endif

namespace fs = std::filesystem;
namespace {
template <size_t N> constexpr ovx_string_t text(const char (&value)[N]) { return {value, N - 1}; }
ovx_string_t text(const std::string& value) { return {value.data(), value.size()}; }
void check(ovsensors_result_t result)
{
    if (result.status == OVSENSORS_API_SUCCESS) return;
    const auto error = ovsensors_get_last_error();
    throw std::runtime_error("ovsensors status " + std::to_string(result.status) + ": " +
        std::string(error.ptr ? error.ptr : "", error.length));
}
void stageCheck(ovstage_api_status_t status)
{
    if (status != OVSTAGE_OK) throw std::runtime_error("ovstage status " + std::to_string(status));
}
void waitStage(ovstage_instance_t* stage, ovstage_enqueue_result_t operation)
{
    stageCheck(operation.status);
    ovstage_op_wait_result_t result{};
    const auto status = ovstage_wait_op(stage, operation.op_index, OVSTAGE_TIMEOUT_INFINITE, &result);
    const auto released = ovstage_release_op(stage, operation.op_index);
    stageCheck(status);
    stageCheck(released);
}
void seal(ovstage_instance_t* stage, ovstage_ordinal_t ordinal)
{
    ovstage_write_floor_desc_t floor{};
    floor.ordinal = ordinal;
    floor.scope = OVSTAGE_SCOPE_ALL;
    waitStage(stage, ovstage_advance_write_floor(stage, &floor));
}
void dictionaryCheck(path_dictionary_instance_t* dictionary, ovx_api_result_t result)
{
    if (result.error.ptr) path_dictionary_release_error(dictionary, result.error);
    if (result.status != OVX_API_SUCCESS) throw std::runtime_error("Path dictionary operation failed.");
}
template <class T> T scalarParameter(const ovsensors_param_t& parameter, ovx_string_t name, uint8_t code)
{
    const auto& tensor = parameter.dl;
    if (!parameter.name.ptr || parameter.name.length != name.length ||
        std::memcmp(parameter.name.ptr, name.ptr, name.length) != 0 || !tensor.data ||
        tensor.device.device_type != kDLCPU || tensor.device.device_id != 0 || tensor.ndim != 0 ||
        tensor.dtype.code != code || tensor.dtype.bits != sizeof(T) * 8 || tensor.dtype.lanes != 1 ||
        tensor.byte_offset != 0)
        throw std::runtime_error("Unexpected CPU scalar parameter: " + std::string(name.ptr, name.length));
    T value;
    std::memcpy(&value, tensor.data, sizeof(value));
    return value;
}
fs::path executableDirectory()
{
#if defined(_WIN32)
    std::wstring buffer(32768, L'\0');
    const auto count = GetModuleFileNameW(nullptr, buffer.data(), static_cast<DWORD>(buffer.size()));
    if (!count || count == buffer.size()) throw std::runtime_error("Cannot locate executable.");
    buffer.resize(count);
    return fs::path(buffer).parent_path();
#else
    return fs::read_symlink("/proc/self/exe").parent_path();
#endif
}

struct Application {
    ovsensors_provider_t provider{};
    ovsensors_instance_t* instance = nullptr;
    ovsensors_sensor_t* sensor = nullptr;
    ovstage_instance_t* stage = nullptr;
    path_dictionary_instance_t* dictionary = nullptr;
    ovx_primpath_list_t paths = OVX_INVALID_PRIMPATH_LIST;
    ovstage_query_handle_t query = OVSTAGE_INVALID_QUERY_HANDLE;
    ovsensors_op_id_t operation = OVSENSORS_INVALID_OP_ID;
    ovsensors_map_handle_t mapping = 0;

    void close()
    {
        if (mapping) { check(ovsensors_unmap_output(instance, mapping, {})); mapping = 0; }
        if (operation) { check(ovsensors_release_result_set(instance, operation)); operation = 0; }
        if (sensor) { check(ovsensors_destroy_sensor(sensor)); sensor = nullptr; }
        // Do not destroy either borrowed owner if instance destruction fails.
        if (instance) { check(ovsensors_destroy_instance(instance)); instance = nullptr; }
        if (query != OVSTAGE_INVALID_QUERY_HANDLE) {
            waitStage(stage, ovstage_release_query(stage, query)); query = OVSTAGE_INVALID_QUERY_HANDLE;
        }
        if (paths != OVX_INVALID_PRIMPATH_LIST) {
            dictionaryCheck(dictionary, path_dictionary_release_path_list_reference(dictionary, paths));
            paths = OVX_INVALID_PRIMPATH_LIST;
        }
        if (stage) { stageCheck(ovstage_destroy_instance(stage)); stage = nullptr; }
        if (pose_provider_destroy(&provider) != OVSENSORS_API_SUCCESS) throw std::runtime_error("Provider destruction failed.");
    }

    void run(const fs::path& scene, const fs::path& output, int samples, const std::string& product)
    {
        ovstage_instance_desc_t stageDesc{"example.custom-provider"};
        stageCheck(ovstage_create_instance(&stageDesc, &stage));
        const auto sceneString = scene.u8string();
        const auto populate = ovstage_population_open_usd_from_file(stage, text(sceneString), 1, 0,
            OVSTAGE_POPULATION_DOMAIN_RENDERING);
        stageCheck(populate.status);
        stageCheck(ovstage_population_wait_op(stage, populate.op_index, OVSTAGE_TIMEOUT_INFINITE, nullptr));
        seal(stage, 1);
        dictionary = ovstage_get_path_dictionary(stage);
        if (!dictionary) throw std::runtime_error("Stage has no path dictionary.");
        // [snippet:provider-create]
        if (pose_provider_create(&provider) != OVSENSORS_API_SUCCESS) throw std::runtime_error("Provider creation failed.");
        check(ovsensors_create_instance(&provider, 1, &instance));
        check(ovsensors_attach_ovstage(instance, stage));
        ovsensors_sensor_spec_t spec{text("pose"), text(product), &provider};
        // Admission reads the populated matrix before the application can write one.
        check(ovsensors_create_sensor(instance, &spec, &sensor));
        check(ovsensors_reset(instance, 0, OVSENSORS_TIMEOUT_INFINITE));
        // [/snippet:provider-create]

        const ovsensors_descriptor_t* descriptor = nullptr;
        check(ovsensors_get_descriptor(sensor, &descriptor));
        if (descriptor->source_count != 1 || descriptor->schema.length != 0)
            throw std::runtime_error("Expected one explicitly selected source.");
        const std::string sourcePath(descriptor->sources[0].ptr, descriptor->sources[0].length);
        const auto sourceView = text(sourcePath);
        dictionaryCheck(dictionary, path_dictionary_create_path_list_from_strings(dictionary, &sourceView, 1, &paths));
        stageCheck(ovstage_query_from_path_list(stage, paths, &query));

        fs::create_directories(output);
        std::ofstream csv(output / "transforms.csv");
        csv.exceptions(std::ios::badbit | std::ios::failbit);
        csv << "publicationOrdinal,simulationTime,sourceIndex";
        for (int i = 0; i < 16; ++i) csv << ",m" << i / 4 << i % 4;
        csv << '\n' << std::setprecision(17);
        for (int sample = 0; sample < samples; ++sample) {
            if (sample == samples / 2) check(ovsensors_reset(instance, 10, OVSENSORS_TIMEOUT_INFINITE));
            const ovstage_ordinal_t ordinal = static_cast<ovstage_ordinal_t>(sample) + 2;
            std::array<double, 16> matrix{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, sample * 0.5, 2, 3, 1};
            int64_t shape = 1;
            DLTensor tensor{matrix.data(), {kDLCPU, 0}, 1, {kDLFloat, 64, 16}, &shape, nullptr, 0};
            ovstage_write_data_t data{};
            data.tensors = &tensor;
            data.tensor_count = 1;
            data.semantic = OVSTAGE_SEMANTIC_MATRIX;
            // [snippet:publish-transform]
            waitStage(stage, ovstage_write_attribute(stage, query, {0, text("omni:xform")}, ordinal,
                data, OVSTAGE_PRIM_MODE_UPSERT));
            seal(stage, ordinal);
            const auto enqueued = ovsensors_step(instance, &sensor, 1, 0.25, ordinal);
            check({enqueued.status});
            operation = enqueued.op_index;
            // [/snippet:publish-transform]
            ovsensors_step_result_set_t results{};
            check(ovsensors_fetch_result_set(instance, operation, OVSENSORS_TIMEOUT_INFINITE, &results));
            if (results.group_count != 1 || results.groups[0].status != OVSENSORS_API_SUCCESS ||
                results.groups[0].frame_count != 1 || results.groups[0].frames[0].output_count != 1)
                throw std::runtime_error("Expected one successful transform frame.");
            const auto result = results.groups[0].frames[0].outputs[0];
            if (result.validity != OVSENSORS_OUTPUT_VALID) throw std::runtime_error("Transform output is invalid.");
            ovsensors_mapped_t mapped{};
            ovsensors_map_output_description_t request{OVSENSORS_MAP_DEVICE_TYPE_CPU, 0};
            check(ovsensors_map_output(instance, result.handle, &request, OVSENSORS_TIMEOUT_INFINITE, &mapped));
            mapping = mapped.map;
            if (mapped.status != OVSENSORS_EVENT_COMPLETED || mapped.tensor_count != 1 || mapped.param_count != 3 ||
                !mapped.tensors || !mapped.params)
                throw std::runtime_error("Expected a completed CPU transform mapping.");

            // [snippet:independent-map]
            check(ovsensors_release_result_set(instance, operation));
            operation = OVSENSORS_INVALID_OP_ID;
            // mapped remains valid after the result tree and output handle have retired.
            const auto* values = static_cast<const double*>(mapped.tensors[0].dl->data);
            const auto publication = scalarParameter<ovstage_ordinal_t>(mapped.params[0], text("publicationOrdinal"), kDLInt);
            const auto time = scalarParameter<double>(mapped.params[1], text("simulationTime"), kDLFloat);
            const auto source = scalarParameter<uint32_t>(mapped.params[2], text("sourceIndex"), kDLUInt);
            csv << publication << ',' << time << ',' << source;
            for (int i = 0; i < 16; ++i) csv << ',' << values[i];
            csv << '\n';
            if (values[12] != matrix[12] || publication != ordinal) throw std::runtime_error("Published transform differs from the observed transform.");
            check(ovsensors_unmap_output(instance, mapping, {}));
            mapping = 0;
            // [/snippet:independent-map]
        }
        csv.close();
        std::ofstream metadata(output / "metadata.json");
        metadata.exceptions(std::ios::badbit | std::ios::failbit);
        metadata << "{\n  \"output\": \"LocalTransform\", \"version\": 2,\n"
            "  \"tensor\": {\"dtype\": \"float64\", \"shape\": [4, 4], \"layout\": \"row-major\"},\n"
            "  \"space\": \"parent-local\", \"translationUnits\": \"stage units (meters in scene.usda)\",\n"
            "  \"source\": \"" << sourcePath << "\", \"providerSelection\": \"explicit\",\n"
            "  \"renderProduct\": \"" << product << "\",\n"
            "  \"params\": {\"publicationOrdinal\": \"int64\", \"simulationTime\": \"float64 seconds\", \"sourceIndex\": \"uint32\"},\n"
            "  \"resetOrigin\": 10, \"resetBeforeSample\": " << samples / 2 << ", \"sampleCount\": " << samples << "\n}\n";
        metadata.close();
    }
};
}

int main(int argc, char** argv)
{
    Application app;
    int result = 0;
    try {
        fs::path scene, output = fs::current_path() / "output" / "custom-provider";
        std::string product = "/World/Pose";
        int samples = 6;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--help") {
                std::cout << "Usage: custom_provider [--scene scene.usda] [--output directory] [--samples 2..10000]\n"
                    "                       [--render-product /World/Pose]\n"
                    "Writes transforms.csv and metadata.json using a CPU stage-transform provider.\n";
                return 0;
            }
            if (i + 1 >= argc) throw std::runtime_error("Missing value for " + arg);
            const std::string value = argv[++i];
            if (arg == "--scene") scene = fs::u8path(value);
            else if (arg == "--output") output = fs::u8path(value);
            else if (arg == "--render-product") product = value;
            else if (arg == "--samples") {
                size_t consumed = 0;
                try {
                    samples = std::stoi(value, &consumed);
                } catch (const std::invalid_argument&) {
                    throw std::runtime_error("--samples must be 2..10000.");
                } catch (const std::out_of_range&) {
                    throw std::runtime_error("--samples must be 2..10000.");
                }
                if (consumed != value.size() || samples < 2 || samples > 10000) throw std::runtime_error("--samples must be 2..10000.");
            } else throw std::runtime_error("Unknown option: " + arg);
        }
        if (scene.empty()) scene = executableDirectory() / "scene.usda";
        if (!fs::is_regular_file(scene)) throw std::runtime_error("Scene does not exist: " + scene.u8string());
        app.run(fs::absolute(scene), fs::absolute(output), samples, product);
        std::cout << "Saved " << samples << " transforms to " << fs::absolute(output) << '\n';
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; result = 1; }
    try { app.close(); }
    catch (const std::exception& error) { std::cerr << "Cleanup failed; borrowed owners remain alive until process exit: " << error.what() << '\n'; result = 1; }
    return result;
}
