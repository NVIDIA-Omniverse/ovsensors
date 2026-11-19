// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary

#include "pose_provider.h"

#include <ovstage/ovstage_api/ovstage_api.h>
#include <ovx/path_dictionary/path_dictionary.h>
#include <ovx/path_dictionary/path_dictionary_utils.h>

#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <new>
#include <string>
#include <vector>

namespace {
template <size_t N> constexpr ovx_string_t text(const char (&value)[N]) { return {value, N - 1}; }
ovx_string_t view(const std::string& value) { return {value.data(), value.size()}; }

struct Failure { ovsensors_api_status_t status; const char* detail; };
[[noreturn]] void fail(ovsensors_api_status_t status, const char* detail) { throw Failure{status, detail}; }
void require(bool condition, ovsensors_api_status_t status, const char* detail)
{
    if (!condition) fail(status, detail);
}

struct Sample {
    std::array<double, 16> matrix{};
    ovstage_ordinal_t ordinal = 0;
    double time = 0;
    uint32_t sourceIndex = 0;
};
struct Entry {
    ovsensors_provider_sensor_t* sensor = nullptr;
    std::shared_ptr<Sample> sample;
    std::string error;
    ovsensors_provider_output_t output{};
    ovsensors_provider_frame_t frame{};
};
struct Operation {
    std::vector<Entry> entries;
    std::vector<ovsensors_provider_sensor_result_t> groups;
};

// A map owns its descriptors and shares only the immutable sample with the operation.
struct Mapping {
    std::shared_ptr<Sample> sample;
    int64_t shape[2] = {4, 4};
    int64_t strides[2] = {4, 1};
    DLTensor matrix{};
    ovsensors_tensor_t tensor{};
    std::array<ovsensors_param_t, 3> params{};
    explicit Mapping(std::shared_ptr<Sample> value) : sample(std::move(value))
    {
        matrix = {sample->matrix.data(), {kDLCPU, 0}, 2, {kDLFloat, 64, 1}, shape, strides, 0};
        tensor = {text("matrix"), &matrix, text("Row-major parent-local transform; translation is row 3, in stage units.")};
        params[0] = {{&sample->ordinal, {kDLCPU, 0}, 0, {kDLInt, 64, 1}, nullptr, nullptr, 0},
                     text("publicationOrdinal"), text("Sealed publication bound supplied by the application.")};
        params[1] = {{&sample->time, {kDLCPU, 0}, 0, {kDLFloat, 64, 1}, nullptr, nullptr, 0},
                     text("simulationTime"), text("Absolute simulation time in seconds, on the reset timeline.")};
        params[2] = {{&sample->sourceIndex, {kDLCPU, 0}, 0, {kDLUInt, 32, 1}, nullptr, nullptr, 0},
                     text("sourceIndex"), text("Index into the sensor descriptor's sources array.")};
    }
};
} // namespace

struct ovsensors_provider_sensor_t {
    std::string source;
    bool alive = true;
};
struct ovsensors_provider_context_t {
    ovstage_instance_t* stage = nullptr;
    char error[1025]{};
    double time = 0;
    bool clockSet = false;
    uint64_t nextId = 1;
    // Retired sensor tokens stay allocated until provider destruction, preventing pointer reuse.
    std::map<ovsensors_provider_sensor_t*, std::unique_ptr<ovsensors_provider_sensor_t>> sensors;
    std::map<uint64_t, std::unique_ptr<Operation>> operations;
    std::map<uint64_t, std::unique_ptr<Mapping>> mappings;
};

namespace {
using Context = ovsensors_provider_context_t;
void error(Context* context, const char* message) noexcept
{
    if (!context) return;
    std::strncpy(context->error, message, sizeof(context->error) - 1);
    context->error[sizeof(context->error) - 1] = 0;
}
template <class F> ovsensors_api_status_t boundary(Context* context, F&& body) noexcept
{
    if (context) context->error[0] = 0;
    try { return body(); }
    catch (const Failure& failure) { error(context, failure.detail); return failure.status; }
    catch (const std::bad_alloc&) { error(context, "Allocation failed."); return OVSENSORS_API_OUT_OF_MEMORY; }
    catch (...) { error(context, "Unexpected exception in the pose provider."); return OVSENSORS_API_INTERNAL; }
}
void attached(Context* context)
{
    require(context && context->stage, OVSENSORS_API_OP_FAILED, "Provider is not attached.");
}
uint64_t nextId(Context& context)
{
    require(context.nextId != std::numeric_limits<uint64_t>::max(), OVSENSORS_API_OP_FAILED, "Handle space exhausted.");
    return context.nextId++;
}
bool liveSensor(Context& context, ovsensors_provider_sensor_t* sensor)
{
    const auto found = context.sensors.find(sensor);
    return found != context.sensors.end() && found->second->alive;
}

void stageCheck(ovstage_api_status_t status)
{
    require(status == OVSTAGE_OK, OVSENSORS_API_OP_FAILED, "ovstage operation failed.");
}
void waitStage(ovstage_instance_t* stage, ovstage_enqueue_result_t operation)
{
    stageCheck(operation.status);
    ovstage_op_wait_result_t result{};
    const auto status = stage->data_plane.vtable->wait_op(stage->data_plane.context, operation.op_index, OVSTAGE_TIMEOUT_INFINITE, &result);
    // Retire tracking even when the asynchronous operation failed.
    const auto released = stage->data_plane.vtable->release_op(stage->data_plane.context, operation.op_index);
    stageCheck(status);
    stageCheck(released);
}
void dictionaryCheck(path_dictionary_instance_t* dictionary, ovx_api_result_t result)
{
    if (result.error.ptr) path_dictionary_release_error(dictionary, result.error);
    require(result.status == OVX_API_SUCCESS, OVSENSORS_API_OP_FAILED, "ovstage path dictionary operation failed.");
}

// These short-lived resources never become sensor state. Copy the payload before closing them.
struct StageRead {
    ovstage_instance_t* stage;
    path_dictionary_instance_t* dictionary;
    ovx_primpath_list_t paths = OVX_INVALID_PRIMPATH_LIST;
    ovstage_query_handle_t query = OVSTAGE_INVALID_QUERY_HANDLE;
    ovstage_read_handle_t read = OVSTAGE_INVALID_READ_HANDLE;
    ovstage_read_group_t group{};
    bool hasGroup = false;
    explicit StageRead(ovstage_instance_t* value) : stage(value), dictionary(value->data_plane.vtable->get_path_dictionary(value->data_plane.context))
    {
        require(dictionary && dictionary->vtable, OVSENSORS_API_OP_FAILED, "Stage has no path dictionary.");
    }
    bool close() noexcept
    {
        bool ok = true;
        auto cleanup = [&](auto&& action) { try { action(); } catch (...) { ok = false; } };
        if (hasGroup) { cleanup([&] { stageCheck(stage->data_plane.vtable->release_groups(stage->data_plane.context, &group, 1)); }); hasGroup = false; }
        if (read != OVSTAGE_INVALID_READ_HANDLE) {
            cleanup([&] { waitStage(stage, stage->data_plane.vtable->release_read(stage->data_plane.context, read)); });
            read = OVSTAGE_INVALID_READ_HANDLE;
        }
        if (query != OVSTAGE_INVALID_QUERY_HANDLE) {
            cleanup([&] { waitStage(stage, stage->data_plane.vtable->release_query(stage->data_plane.context, query)); });
            query = OVSTAGE_INVALID_QUERY_HANDLE;
        }
        if (paths != OVX_INVALID_PRIMPATH_LIST) {
            cleanup([&] { dictionaryCheck(dictionary, path_dictionary_release_path_list_reference(dictionary, paths)); });
            paths = OVX_INVALID_PRIMPATH_LIST;
        }
        return ok;
    }
    ~StageRead() { close(); }
};

std::array<double, 16> readTransform(Context& context, const std::string& source, ovstage_ordinal_t ordinal)
{
    StageRead read(context.stage);
    const auto sourcePath = view(source);
    dictionaryCheck(read.dictionary, path_dictionary_create_path_list_from_strings(read.dictionary, &sourcePath, 1, &read.paths));
    stageCheck(context.stage->data_plane.vtable->query_from_path_list(context.stage->data_plane.context, read.paths, &read.query));
    const auto name = text("omni:xform");
    ovx_token_t token{};
    dictionaryCheck(read.dictionary, path_dictionary_create_tokens_from_strings(read.dictionary, &name, 1, &token));
    ovstage_ordinal_range_t range{};
    range.end_ordinal = ordinal;
    waitStage(context.stage, context.stage->data_plane.vtable->read_attributes(context.stage->data_plane.context, read.query, &token,
        1, range, &read.read));
    std::array<double, 16> matrix{};
    bool foundMatrix = false;
    for (;;) {
        size_t fetchedReadGroupCount = 0;
        const auto status = context.stage->data_plane.vtable->fetch_read_next(context.stage->data_plane.context, read.read,
            OVSTAGE_TIMEOUT_INFINITE, &read.group, 1, &fetchedReadGroupCount);
        if (status == OVSTAGE_ERROR_END_OF_ITERATION) break;
        stageCheck(status);
        read.hasGroup = true;
        const auto& group = read.group;
        require(group.column_count == 1 && group.columns,
            OVSENSORS_API_NOT_SUPPORTED, "Expected one transform column for the source.");
        const auto& column = group.columns[0];
        require(!column.is_delete && !column.is_array && group.prims.count == 1 &&
            column.data.tensor_count == 1 && column.data.tensors && !column.data.mask &&
            column.data.cuda_sync.stream == 0 && column.data.cuda_sync.wait_event == 0,
            OVSENSORS_API_NOT_SUPPORTED, "Expected one fixed CPU transform value for the source.");
        const auto& tensor = column.data.tensors[0];
        const uint32_t row = column.data.index_map ? column.data.index_map[0] : 0;
        require(tensor.data && tensor.shape && tensor.strides && tensor.ndim > 0 &&
            tensor.device.device_type == kDLCPU && tensor.dtype.lanes == 1 &&
            row < static_cast<uint64_t>(tensor.shape[0]), OVSENSORS_API_NOT_SUPPORTED, "Unsupported stage tensor layout.");
        const auto* bytes = static_cast<const unsigned char*>(tensor.data) + tensor.byte_offset;
        if (column.attribute == token) {
            require(!foundMatrix && column.semantic == OVSTAGE_SEMANTIC_MATRIX && tensor.dtype.code == kDLFloat &&
                tensor.dtype.bits == 64 && tensor.ndim == 3 && tensor.shape[1] == 4 && tensor.shape[2] == 4 &&
                tensor.strides[0] == 16 && tensor.strides[1] == 4 && tensor.strides[2] == 1,
                OVSENSORS_API_NOT_SUPPORTED, "Expected omni:xform as a float64 parent-local 4x4 matrix.");
            std::memcpy(matrix.data(), bytes + row * 16 * sizeof(double), sizeof(matrix));
            for (double value : matrix) require(std::isfinite(value), OVSENSORS_API_OP_FAILED, "Transform is nonfinite.");
            foundMatrix = true;
        }
        stageCheck(context.stage->data_plane.vtable->release_groups(context.stage->data_plane.context, &read.group, 1));
        read.hasGroup = false;
    }
    require(read.close(), OVSENSORS_API_OP_FAILED, "Could not release ovstage read resources.");
    require(foundMatrix, OVSENSORS_API_NOT_SUPPORTED, "omni:xform is missing at this publication.");
    return matrix;
}

void apiVersion(const Context*, uint32_t* major, uint32_t* minor, uint32_t* patch) noexcept
{
    if (major) *major = OVSENSORS_VERSION_MAJOR;
    if (minor) *minor = OVSENSORS_VERSION_MINOR;
    if (patch) *patch = OVSENSORS_VERSION_PATCH;
}
const ovsensors_provider_claims_t* claims(const Context* context) noexcept
{
    static const ovsensors_provider_claims_t value{text("example.local-transform"), nullptr, 0};
    return context ? &value : nullptr;
}
ovx_string_t lastError(const Context* context) noexcept
{
    return context ? ovx_string_t{context->error, std::strlen(context->error)} : ovx_string_t{};
}
ovsensors_api_status_t attach(Context* context, ovstage_instance_t* stage) noexcept
{
    return boundary(context, [&] {
        require(context && stage, OVSENSORS_API_INVALID_ARGUMENT, "A context and stage are required.");
        require(!context->stage, OVSENSORS_API_OP_FAILED, "Provider is already attached.");
        require(stage->data_plane.vtable && stage->data_plane.vtable->get_version, OVSENSORS_API_NOT_SUPPORTED, "Stage has no versioned interface.");
        uint32_t major{}, minor{}, patch{};
        stage->data_plane.vtable->get_version(stage->data_plane.context, &major, &minor, &patch);
        require(major == OVSTAGE_VERSION_MAJOR && minor == OVSTAGE_VERSION_MINOR,
            OVSENSORS_API_NOT_SUPPORTED, "Incompatible ovstage API version.");
        require(stage->data_plane.vtable->get_path_dictionary && stage->data_plane.vtable->query_from_path_list && stage->data_plane.vtable->release_query &&
            stage->data_plane.vtable->read_attributes && stage->data_plane.vtable->fetch_read_next && stage->data_plane.vtable->release_read &&
            stage->data_plane.vtable->release_groups && stage->data_plane.vtable->wait_op && stage->data_plane.vtable->release_op,
            OVSENSORS_API_NOT_SUPPORTED, "Stage lacks required read slots.");
        context->stage = stage;
        context->clockSet = false;
        return OVSENSORS_API_SUCCESS;
    });
}
ovsensors_api_status_t detach(Context* context) noexcept
{
    return boundary(context, [&] {
        require(context, OVSENSORS_API_INVALID_ARGUMENT, "A context is required.");
        require(context->operations.empty() && context->mappings.empty(), OVSENSORS_API_OP_FAILED, "Release operations and maps before detach.");
        for (const auto& sensor : context->sensors)
            require(!sensor.second->alive, OVSENSORS_API_OP_FAILED, "Destroy sensors before detach.");
        context->stage = nullptr;
        context->clockSet = false;
        return OVSENSORS_API_SUCCESS;
    });
}
ovsensors_api_status_t createSensor(Context* context, const ovsensors_descriptor_t* descriptor,
    ovstage_ordinal_t ordinal, ovsensors_provider_sensor_t** out) noexcept
{
    if (out) *out = nullptr;
    return boundary(context, [&] {
        require(context && descriptor && out, OVSENSORS_API_INVALID_ARGUMENT, "Descriptor and output are required.");
        attached(context);
        require(descriptor->source_count == 1 && descriptor->sources, OVSENSORS_API_NOT_SUPPORTED, "Exactly one source with a published local transform is supported.");
        require(descriptor->rate.num == 0 && descriptor->rate.den == 0 &&
            descriptor->start.num == 0 && descriptor->start.den == 0,
            OVSENSORS_API_NOT_SUPPORTED, "Only unauthored cadence is supported: one sample per step.");
        const auto source = descriptor->sources[0];
        require(source.ptr && source.length && source.length <= 4096 && source.ptr[0] == '/' &&
            !std::memchr(source.ptr, 0, source.length), OVSENSORS_API_INVALID_ARGUMENT, "Expected an absolute source path.");
        auto sensor = std::make_unique<ovsensors_provider_sensor_t>();
        sensor->source.assign(source.ptr, source.length);
        readTransform(*context, sensor->source, ordinal);
        auto* handle = sensor.get();
        context->sensors.emplace(handle, std::move(sensor));
        *out = handle;
        return OVSENSORS_API_SUCCESS;
    });
}
ovsensors_enqueue_result_t step(Context* context, const ovsensors_provider_step_desc_t* desc) noexcept
{
    uint64_t operationId = OVSENSORS_INVALID_OP_ID;
    const auto status = boundary(context, [&] {
        attached(context);
        require(desc && (desc->sensors != nullptr) == (desc->sensor_count != 0) &&
            std::isfinite(desc->delta_time) && desc->delta_time > 0, OVSENSORS_API_OP_FAILED, "Malformed step.");
        require(context->clockSet, OVSENSORS_API_RESET_REQUIRED, "Reset the clock before the first step.");
        const double end = context->time + desc->delta_time;
        require(std::isfinite(end), OVSENSORS_API_OP_FAILED, "Simulation time overflow.");
        for (size_t i = 0; i < desc->sensor_count; ++i) {
            require(liveSensor(*context, desc->sensors[i]), OVSENSORS_API_INVALID_HANDLE, "Unknown sensor.");
            for (size_t j = 0; j < i; ++j)
                require(desc->sensors[i] != desc->sensors[j], OVSENSORS_API_OP_FAILED, "Duplicate sensor.");
        }
        auto operation = std::make_unique<Operation>();
        operation->entries.resize(desc->sensor_count);
        operation->groups.resize(desc->sensor_count);
        for (size_t i = 0; i < desc->sensor_count; ++i) {
            auto& entry = operation->entries[i];
            entry.sensor = desc->sensors[i];
            try {
                entry.sample = std::make_shared<Sample>();
                entry.sample->matrix = readTransform(*context, entry.sensor->source, desc->ordinal);
                entry.sample->ordinal = desc->ordinal;
                entry.sample->time = end;
                entry.output = {text("LocalTransform"), OVSENSORS_OUTPUT_VALID, nextId(*context)};
                entry.frame = {context->time, end, &entry.output, 1};
                operation->groups[i] = {entry.sensor, OVSENSORS_API_SUCCESS, {}, &entry.frame, 1};
            } catch (const Failure& failure) {
                entry.sample.reset();
                entry.error = failure.detail;
                operation->groups[i] = {entry.sensor, OVSENSORS_API_OP_FAILED, view(entry.error), nullptr, 0};
            }
        }
        const auto id = nextId(*context);
        context->operations.emplace(id, std::move(operation));
        context->time = end; // Empty sensor sets advance the same clock.
        operationId = id;
        return OVSENSORS_API_SUCCESS;
    });
    return {status, operationId};
}
ovsensors_api_status_t fetch(Context* context, uint64_t id, ovsensors_timeout_ns_t,
    ovsensors_provider_result_set_t* out) noexcept
{
    if (out) *out = {};
    return boundary(context, [&] {
        require(context && out, OVSENSORS_API_OP_FAILED, "A result output is required.");
        const auto found = context->operations.find(id);
        require(found != context->operations.end(), OVSENSORS_API_INVALID_HANDLE, "Unknown operation.");
        out->groups = found->second->groups.empty() ? nullptr : found->second->groups.data();
        out->group_count = found->second->groups.size();
        return OVSENSORS_API_SUCCESS;
    });
}
ovsensors_api_status_t mapOutput(Context* context, uint64_t output, const ovsensors_map_output_description_t* desc,
    ovsensors_timeout_ns_t, ovsensors_provider_mapped_t* out) noexcept
{
    if (out) *out = {};
    return boundary(context, [&] {
        require(context && out, OVSENSORS_API_INVALID_ARGUMENT, "A mapping output is required.");
        const auto device = desc ? desc->device_type : OVSENSORS_MAP_DEVICE_TYPE_DEFAULT;
        require((device == OVSENSORS_MAP_DEVICE_TYPE_DEFAULT || device == OVSENSORS_MAP_DEVICE_TYPE_CPU) &&
            (!desc || !desc->sync_stream), OVSENSORS_API_NOT_SUPPORTED, "This provider supports synchronous CPU mappings only.");
        std::shared_ptr<Sample> sample;
        for (const auto& operation : context->operations)
            for (const auto& entry : operation.second->entries)
                if (entry.sample && entry.output.provider_output_handle == output) sample = entry.sample;
        require(static_cast<bool>(sample), OVSENSORS_API_INVALID_HANDLE, "Unknown output.");
        auto mapping = std::make_unique<Mapping>(std::move(sample));
        auto* value = mapping.get();
        const auto id = nextId(*context);
        context->mappings.emplace(id, std::move(mapping));
        *out = {&value->tensor, 1, {}, id, OVSENSORS_EVENT_COMPLETED, {}, text("LocalTransform"),
                text("example.parent-local-transform"), text("Source local transform, without hierarchy composition or physical sensor simulation."),
                2, value->params.data(), value->params.size()};
        return OVSENSORS_API_SUCCESS;
    });
}
ovsensors_api_status_t unmap(Context* context, uint64_t id, ovsensors_cuda_sync_t done) noexcept
{
    return boundary(context, [&] {
        require(context && context->mappings.count(id), OVSENSORS_API_INVALID_HANDLE, "Unknown mapping.");
        require(done.stream == 0 && done.wait_event == 0, OVSENSORS_API_OP_FAILED,
            "CPU reads must finish before unmap; CUDA consumer dependencies are unsupported.");
        context->mappings.erase(id);
        return OVSENSORS_API_SUCCESS;
    });
}
ovsensors_api_status_t release(Context* context, uint64_t id) noexcept
{
    return boundary(context, [&] {
        require(context && context->operations.count(id), OVSENSORS_API_INVALID_HANDLE, "Unknown operation.");
        context->operations.erase(id);
        return OVSENSORS_API_SUCCESS;
    });
}
ovsensors_api_status_t reset(Context* context, double time, ovsensors_timeout_ns_t) noexcept
{
    return boundary(context, [&] {
        attached(context);
        require(std::isfinite(time), OVSENSORS_API_OP_FAILED, "Reset origin must be finite.");
        require(context->operations.empty() && context->mappings.empty(), OVSENSORS_API_OP_FAILED, "Release operations and maps before reset.");
        context->time = time;
        context->clockSet = true;
        return OVSENSORS_API_SUCCESS;
    });
}
ovsensors_api_status_t destroySensor(Context* context, ovsensors_provider_sensor_t* sensor) noexcept
{
    return boundary(context, [&] {
        require(context && liveSensor(*context, sensor), OVSENSORS_API_INVALID_HANDLE, "Unknown sensor.");
        for (const auto& operation : context->operations)
            for (const auto& entry : operation.second->entries)
                require(entry.sensor != sensor, OVSENSORS_API_BUSY, "Release the sensor's operations first.");
        sensor->alive = false;
        return OVSENSORS_API_SUCCESS;
    });
}

const ovsensors_provider_vtable_t vtable{
    apiVersion, claims, lastError, attach, detach, createSensor, step, fetch,
    mapOutput, unmap, release, reset, destroySensor
};
} // namespace

extern "C" POSE_API ovsensors_api_status_t pose_provider_create(ovsensors_provider_t* out)
{
    if (out) *out = {};
    return boundary(nullptr, [&] {
        require(out, OVSENSORS_API_INVALID_ARGUMENT, "A provider output is required.");
        *out = {&vtable, new Context};
        return OVSENSORS_API_SUCCESS;
    });
}
extern "C" POSE_API ovsensors_api_status_t pose_provider_destroy(ovsensors_provider_t* provider)
{
    if (!provider) return OVSENSORS_API_INVALID_ARGUMENT;
    if (!provider->context && !provider->vtable) return OVSENSORS_API_SUCCESS;
    if (provider->vtable != &vtable || !provider->context) return OVSENSORS_API_INVALID_ARGUMENT;
    auto* context = provider->context;
    const auto status = boundary(context, [&] {
        require(!context->stage && context->operations.empty() && context->mappings.empty(), OVSENSORS_API_BUSY,
            "Detach and release all resources before destroying the provider.");
        for (const auto& sensor : context->sensors)
            require(!sensor.second->alive, OVSENSORS_API_BUSY, "Destroy sensors before destroying the provider.");
        return OVSENSORS_API_SUCCESS;
    });
    if (status == OVSENSORS_API_SUCCESS) { delete context; *provider = {}; }
    return status;
}
