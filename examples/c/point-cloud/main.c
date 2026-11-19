// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary

#ifndef _WIN32
#define _POSIX_C_SOURCE 200809L
#endif
#include <ovrtx/ovrtx.h>
#include <ovrtx/ovrtx_config.h>
#include <ovsensors/ovsensors.h>
#include <ovsensors/ovsensors_provider_ovrtx.h>
#include <ovstage/ovstage.h>
#include <ovstage/ovstage_api/ovstage_api_utils.h>
#include <ovstage/ovstage_population.h>
#include "example_paths.h"

#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#ifdef _WIN32
#include <direct.h>
#include <windows.h>
#else
#include <unistd.h>
#endif

#define MAX_ROWS 4096
#define PATH_CAPACITY 4096

struct cloud
{
    float rows[MAX_ROWS][4];
    size_t row_count;
    int32_t delivered_count;
    uint64_t frame_id, timestamp_ns;
    uint32_t coords_type;
    uint16_t frame_of_reference;
    float model_to_app[16];
    double frame_start, frame_end;
};

struct app
{
    ovstage_instance_t* stage;
    ovsensors_provider_t provider;
    ovsensors_instance_t* instance;
    ovsensors_sensor_t* sensor;
    ovsensors_map_handle_t live_map;
    ovsensors_op_id_t live_op;
    int has_live_op, attached, initialized, provider_created;
};

static ovx_string_t str(const char* value)
{
    ovx_string_t result = { value, strlen(value) };
    return result;
}

static int named(ovx_string_t value, const char* name)
{
    return value.ptr && value.length == strlen(name) && memcmp(value.ptr, name, value.length) == 0;
}

static int check(ovsensors_result_t result, const char* operation)
{
    if (result.status == OVSENSORS_API_SUCCESS)
        return 1;
    {
        ovx_string_t detail = ovsensors_get_last_error();
        fprintf(stderr, "%s failed (status %d): %.*s\n", operation, (int)result.status,
                (int)detail.length, detail.ptr ? detail.ptr : "");
    }
    return 0;
}

static int check_stage(ovstage_api_status_t status, const char* operation)
{
    if (status == OVSTAGE_OK)
        return 1;
    fprintf(stderr, "ovstage %s failed (status %u)\n", operation, (unsigned)status);
    return 0;
}

static int seal(ovstage_instance_t* stage, ovstage_ordinal_t ordinal)
{
    ovstage_write_floor_desc_t floor = {0};
    ovstage_op_wait_result_t waited = {0};
    ovstage_enqueue_result_t op;
    int completed, released;
    floor.ordinal = ordinal;
    floor.scope = OVSTAGE_SCOPE_ALL;
    op = ovstage_advance_write_floor(stage, &floor);
    if (!check_stage(op.status, "advance_write_floor"))
        return 0;
    completed = check_stage(ovstage_wait_op(stage, op.op_index, OVSTAGE_TIMEOUT_INFINITE, &waited), "wait_op");
    for (size_t i = 0; i < waited.error_op_id_count; ++i)
    {
        ovx_string_t detail = ovstage_get_last_op_error(stage, waited.error_op_ids[i]);
        fprintf(stderr, "ovstage publication failed: %.*s\n", (int)detail.length, detail.ptr ? detail.ptr : "");
    }
    completed = completed && waited.error_op_id_count == 0;
    released = check_stage(ovstage_release_op(stage, op.op_index), "release_op");
    return completed && released;
}

static int unmap(struct app* app)
{
    ovsensors_cuda_sync_t done = {0, 0};
    ovsensors_result_t result = ovsensors_unmap_output(app->instance, app->live_map, done);
    int ok = check(result, "unmap_output");
    // This known-live handle is consumed on a provider INVALID_HANDLE too.
    if (ok || result.status == OVSENSORS_API_INVALID_HANDLE)
        app->live_map = 0;
    return ok;
}

static int release(struct app* app)
{
    if (!check(ovsensors_release_result_set(app->instance, app->live_op), "release_result_set"))
        return 0;
    app->has_live_op = 0;
    return 1;
}

static int cleanup(struct app* app)
{
    int ok = 1;
    // Exact retries retain their handles. Stop before destroying a still-borrowed stage.
    if (app->live_map && !unmap(app))
    {
        ok = 0;
        if (app->live_map)
            return 0;
    }
    if (app->has_live_op && !release(app))
        return 0;
    if (app->sensor && !check(ovsensors_destroy_sensor(app->sensor), "destroy_sensor"))
        return 0;
    app->sensor = NULL;
    if (app->attached && !check(ovsensors_detach_ovstage(app->instance), "detach_ovstage"))
        return 0;
    app->attached = 0;
    if (app->instance && !check(ovsensors_destroy_instance(app->instance), "destroy_instance"))
        return 0;
    app->instance = NULL;
    if (app->provider_created)
    {
        ovsensors_api_status_t status = ovsensors_provider_ovrtx_destroy(&app->provider);
        if (status != OVSENSORS_API_SUCCESS)
        {
            fprintf(stderr, "provider destruction failed (status %d)\n", (int)status);
            return 0;
        }
        app->provider_created = 0;
    }
    if (app->initialized && !check(ovsensors_shutdown(), "shutdown"))
        ok = 0;
    app->initialized = 0;
    if (app->stage && !check_stage(ovstage_destroy_instance(app->stage), "destroy_instance"))
        ok = 0;
    app->stage = NULL;
    return ok;
}

static const DLTensor* tensor(const ovsensors_mapped_t* mapped, const char* name)
{
    for (size_t i = 0; i < mapped->tensor_count; ++i)
        if (named(mapped->tensors[i].name, name))
            return mapped->tensors[i].dl;
    fprintf(stderr, "Missing tensor %s\n", name);
    return NULL;
}

static const DLTensor* parameter(const ovsensors_mapped_t* mapped, const char* name)
{
    for (size_t i = 0; i < mapped->param_count; ++i)
        if (named(mapped->params[i].name, name))
            return &mapped->params[i].dl;
    fprintf(stderr, "Missing CPU parameter %s\n", name);
    return NULL;
}

static int layout(const DLTensor* dl, uint8_t code, uint8_t bits, int ndim)
{
    uint64_t extent = 0, packed_stride = 1;
    if (!dl || !dl->data || dl->device.device_type != kDLCPU || dl->dtype.code != code ||
        dl->dtype.bits != bits || dl->dtype.lanes != 1 || dl->ndim != ndim || (ndim && !dl->shape))
        return 0;
    // Bound address arithmetic before reading either explicit strides or packed storage.
    for (int axis = ndim - 1; axis >= 0; --axis)
    {
        uint64_t stride;
        if (dl->shape[axis] <= 0 || (dl->strides && dl->strides[axis] <= 0))
            return 0;
        stride = dl->strides ? (uint64_t)dl->strides[axis] : packed_stride;
        if ((uint64_t)(dl->shape[axis] - 1) > (UINT64_MAX - extent) / stride ||
            (uint64_t)dl->shape[axis] > UINT64_MAX / packed_stride)
            return 0;
        extent += (uint64_t)(dl->shape[axis] - 1) * stride;
        packed_stride *= (uint64_t)dl->shape[axis];
    }
    return dl->byte_offset <= PTRDIFF_MAX && extent < (uint64_t)PTRDIFF_MAX / (bits / 8) &&
           dl->byte_offset <= (uint64_t)PTRDIFF_MAX - (extent + 1) * (bits / 8);
}

static void read_element(const DLTensor* dl, size_t row, size_t column, void* destination)
{
    size_t index = 0;
    if (dl->ndim == 1)
        index = row * (dl->strides ? (size_t)dl->strides[0] : 1);
    else if (dl->ndim == 2)
        index = row * (dl->strides ? (size_t)dl->strides[0] : (size_t)dl->shape[1]) +
                column * (dl->strides ? (size_t)dl->strides[1] : 1);
    memcpy(destination, (const char*)dl->data + (size_t)dl->byte_offset + index * (dl->dtype.bits / 8), dl->dtype.bits / 8);
}

static int scalar(const ovsensors_mapped_t* mapped, const char* name, uint8_t code, uint8_t bits, void* value)
{
    const DLTensor* dl = parameter(mapped, name);
    if (!dl || (dl->ndim != 0 && dl->ndim != 1) || !layout(dl, code, bits, dl->ndim) ||
        (dl->ndim == 1 && dl->shape[0] != 1))
    {
        fprintf(stderr, "Invalid CPU scalar parameter %s\n", name);
        return 0;
    }
    read_element(dl, 0, 0, value);
    return 1;
}

// [snippet:read-point-cloud]
static int copy_cloud(const ovsensors_mapped_t* mapped, const ovsensors_frame_t* frame, struct cloud* cloud)
{
    const DLTensor *coordinates, *intensity, *counts, *flags, *transform;
    struct cloud next = {0};
    if (mapped->status != OVSENSORS_EVENT_COMPLETED || mapped->cuda_sync.stream || mapped->cuda_sync.wait_event)
    {
        fprintf(stderr, "CPU point-cloud mapping is not ready: %.*s\n", (int)mapped->error_message.length,
                mapped->error_message.ptr ? mapped->error_message.ptr : "");
        return 0;
    }
    coordinates = tensor(mapped, "Coordinates");
    intensity = tensor(mapped, "Intensity");
    counts = tensor(mapped, "Counts");
    flags = tensor(mapped, "Flags");
    transform = parameter(mapped, "modelToAppTransform");
    if (!layout(coordinates, kDLFloat, 32, 2) || coordinates->shape[0] != 3 ||
        !layout(intensity, kDLFloat, 32, 1) || !layout(counts, kDLInt, 32, 1) || counts->shape[0] != 1 ||
        !layout(flags, kDLUInt, 8, 1) || !layout(transform, kDLFloat, 32, 2) ||
        transform->shape[0] != 4 || transform->shape[1] != 4)
    {
        fprintf(stderr, "Expected CPU Coordinates float32[3,N], Intensity float32[N], Counts int32[1], "
                        "Flags uint8[N], and modelToAppTransform float32[4,4]\n");
        return 0;
    }
    read_element(counts, 0, 0, &next.delivered_count);
    if (next.delivered_count < 0 || next.delivered_count > coordinates->shape[1] ||
        next.delivered_count > intensity->shape[0] || next.delivered_count > flags->shape[0])
    {
        fprintf(stderr, "Counts exceeds a point-channel extent\n");
        return 0;
    }
    if (!scalar(mapped, "frameId", kDLUInt, 64, &next.frame_id) ||
        !scalar(mapped, "timestampNs", kDLUInt, 64, &next.timestamp_ns) ||
        !scalar(mapped, "coordsType", kDLUInt, 32, &next.coords_type) ||
        !scalar(mapped, "frameOfReference", kDLUInt, 16, &next.frame_of_reference))
        return 0;
    for (size_t i = 0; i < 16; ++i)
    {
        read_element(transform, i / 4, i % 4, &next.model_to_app[i]);
        if (!isfinite(next.model_to_app[i]))
            return 0;
    }
    for (int32_t i = 0; i < next.delivered_count && next.row_count < MAX_ROWS; ++i)
    {
        uint8_t flag;
        float* row = next.rows[next.row_count];
        read_element(flags, (size_t)i, 0, &flag);
        // Counts bounds delivered entries; only the VALID bit establishes a return.
        if (!(flag & 0x40u))
            continue;
        for (size_t axis = 0; axis < 3; ++axis)
            read_element(coordinates, axis, (size_t)i, &row[axis]);
        read_element(intensity, (size_t)i, 0, &row[3]);
        for (size_t j = 0; j < 4; ++j)
        {
            if (!isfinite(row[j]))
            {
                fprintf(stderr, "Nonfinite coordinate or intensity in valid return\n");
                return 0;
            }
        }
        ++next.row_count;
    }
    next.frame_start = frame->frame_start_time;
    next.frame_end = frame->frame_end_time;
    if (!isfinite(next.frame_start) || !isfinite(next.frame_end) || next.frame_end < next.frame_start)
        return 0;
    if (next.row_count)
        *cloud = next;
    return 1;
}
// [/snippet:read-point-cloud]

static int harvest(struct app* app, const ovsensors_step_result_set_t* results, struct cloud* cloud)
{
    const ovsensors_map_output_description_t cpu = { OVSENSORS_MAP_DEVICE_TYPE_CPU, 0 };
    if (!results->group_count)
    {
        fprintf(stderr, "No sensor group returned\n");
        return 0;
    }
    for (size_t g = 0; g < results->group_count; ++g)
    {
        const ovsensors_sensor_result_t* group = &results->groups[g];
        if (group->status != OVSENSORS_API_SUCCESS)
        {
            fprintf(stderr, "Sensor group failed: %.*s\n", (int)group->error_message.length,
                    group->error_message.ptr ? group->error_message.ptr : "");
            return 0;
        }
        for (size_t f = 0; f < group->frame_count; ++f)
        {
            const ovsensors_frame_t* frame = &group->frames[f];
            int found = 0;
            for (size_t o = 0; o < frame->output_count; ++o)
            {
                const ovsensors_output_t* output = &frame->outputs[o];
                ovsensors_mapped_t mapped = {0};
                int copied, unmapped;
                if (!named(output->name, "PointCloud") && !named(output->name, "/Render/Vars/PointCloud"))
                    continue;
                found = 1;
                if (output->validity != OVSENSORS_OUTPUT_VALID)
                {
                    fprintf(stderr, "PointCloud has no valid sample\n");
                    return 0;
                }
                if (!check(ovsensors_map_output(app->instance, output->handle, &cpu,
                                               OVSENSORS_TIMEOUT_INFINITE, &mapped), "map_output"))
                    return 0;
                app->live_map = mapped.map;
                copied = copy_cloud(&mapped, frame, cloud);
                unmapped = unmap(app);
                if (!copied || !unmapped)
                    return 0;
            }
            if (!found)
            {
                fprintf(stderr, "Frame has no PointCloud output\n");
                return 0;
            }
        }
    }
    return 1;
}

static int default_scene(char* path, size_t capacity, const char* argv0)
{
    char* slash;
#ifdef _WIN32
    DWORD size = GetModuleFileNameA(NULL, path, (DWORD)capacity);
    if (!size || size >= capacity)
        return 0;
    slash = strrchr(path, '\\');
#else
    ssize_t size = readlink("/proc/self/exe", path, capacity - 1);
    if (size == (ssize_t)(capacity - 1))
        return 0;
    if (size < 0)
    {
        if (strlen(argv0) >= capacity)
            return 0;
        strcpy(path, argv0);
    }
    else
        path[size] = '\0';
    slash = strrchr(path, '/');
#endif
    (void)argv0;
    if (slash)
        slash[1] = '\0';
    else
        path[0] = '\0';
    if (strlen(path) + strlen("point_cloud_scene.usda") >= capacity)
        return 0;
    strcat(path, "point_cloud_scene.usda");
    return 1;
}

static int make_parents(const char* path)
{
    char parent[PATH_CAPACITY];
    strcpy(parent, path);
    for (size_t i = 1; parent[i]; ++i)
    {
#ifdef _WIN32
        if (parent[i] != '/' && parent[i] != '\\')
#else
        if (parent[i] != '/')
#endif
            continue;
#ifdef _WIN32
        if (i == 2 && parent[1] == ':')
            continue;
#endif
        {
            char separator = parent[i];
            int result;
            parent[i] = '\0';
#ifdef _WIN32
            result = _mkdir(parent);
#else
            result = mkdir(parent, 0777);
#endif
            parent[i] = separator;
            if (result != 0 && errno != EEXIST)
                return 0;
        }
    }
    return 1;
}

static int write_cloud(const char* path, const struct cloud* cloud)
{
    char metadata[PATH_CAPACITY];
    FILE* file;
    int ok;
    snprintf(metadata, sizeof metadata, "%s.json", path);
    if (!make_parents(path) || !(file = fopen(path, "wb")))
    {
        fprintf(stderr, "Cannot write CSV %s: %s\n", path, strerror(errno));
        return 0;
    }
    fprintf(file, "x,y,z,intensity\n");
    for (size_t i = 0; i < cloud->row_count; ++i)
        fprintf(file, "%.9g,%.9g,%.9g,%.9g\n", cloud->rows[i][0], cloud->rows[i][1], cloud->rows[i][2], cloud->rows[i][3]);
    ok = !ferror(file);
    if (fclose(file) != 0)
        ok = 0;
    if (!ok)
    {
        fprintf(stderr, "CSV write failed: %s\n", path);
        return 0;
    }
    file = fopen(metadata, "wb");
    if (!file)
    {
        fprintf(stderr, "Cannot write metadata %s: %s\n", metadata, strerror(errno));
        return 0;
    }
    fprintf(file, "{\n  \"rows\": %zu, \"max_rows\": %d, \"delivered_count\": %" PRId32 ",\n"
                  "  \"frame_id\": %" PRIu64 ", \"timestamp_ns\": %" PRIu64 ",\n"
                  "  \"coords_type\": %" PRIu32 ", \"frame_of_reference\": %" PRIu16 ",\n"
                  "  \"frame_start\": %.17g, \"frame_end\": %.17g,\n  \"model_to_app_transform\": [",
            cloud->row_count, MAX_ROWS, cloud->delivered_count, cloud->frame_id, cloud->timestamp_ns,
            cloud->coords_type, cloud->frame_of_reference, cloud->frame_start, cloud->frame_end);
    for (size_t i = 0; i < 16; ++i)
        fprintf(file, "%s%.9g", i ? ", " : "", cloud->model_to_app[i]);
    fprintf(file, "]\n}\n");
    ok = !ferror(file);
    if (fclose(file) != 0)
        ok = 0;
    if (!ok)
        fprintf(stderr, "Metadata write failed: %s\n", metadata);
    return ok;
}

int main(int argc, char** argv)
{
    char scene_path[PATH_CAPACITY];
    const char* scene = NULL;
    const char* output = "_output/points.csv";
    struct app app = {0};
    struct cloud cloud = {0};
    int code = 1;
    for (int i = 1; i < argc; ++i)
    {
        if (strcmp(argv[i], "--help") == 0)
        {
            puts("Usage: ovsensors_point_cloud [--output _output/points.csv] [--scene point_cloud_scene.usda]");
            return 0;
        }
        if ((strcmp(argv[i], "--output") != 0 && strcmp(argv[i], "--scene") != 0) ||
            i + 1 == argc || !argv[i + 1][0] || strncmp(argv[i + 1], "--", 2) == 0)
        {
            fputs("Expected --output PATH or --scene PATH; use --help for usage\n", stderr);
            return 1;
        }
        if (strcmp(argv[i], "--scene") == 0)
            scene = argv[++i];
        else
            output = argv[++i];
    }
    if (!output[0] || strlen(output) + strlen(".json") >= PATH_CAPACITY)
    {
        fputs("Output path is empty or too long\n", stderr);
        return 1;
    }
    if (!scene)
    {
        if (!default_scene(scene_path, sizeof scene_path, argv[0]))
        {
            fputs("Cannot locate bundled scene; supply --scene PATH\n", stderr);
            return 1;
        }
        scene = scene_path;
    }
    // [snippet:point-cloud-setup]
    if (ovrtx_register_schema_paths(NULL).status != OVRTX_API_SUCCESS)
    {
        fputs("ovrtx schema registration failed\n", stderr);
        return 1;
    }
    {
        ovstage_instance_desc_t desc = {0};
        ovstage_population_enqueue_result_t op;
        desc.name = "ovsensors.example.point-cloud";
        if (!check_stage(ovstage_create_instance(&desc, &app.stage), "create_instance"))
            goto finish;
        op = ovstage_population_open_usd_from_file(app.stage, str(scene), 1, 0.0, OVSTAGE_POPULATION_DOMAIN_RENDERING);
        if (!check_stage(op.status, "population_open_usd_from_file"))
            goto finish;
        if (!check_stage(ovstage_population_wait_op(app.stage, op.op_index, OVSTAGE_TIMEOUT_INFINITE, NULL), "population_wait_op"))
        {
            ovx_string_t error = ovstage_population_get_last_error();
            fprintf(stderr, "%.*s\n", (int)error.length, error.ptr ? error.ptr : "");
            goto finish;
        }
        if (!seal(app.stage, 1))
            goto finish;
    }
    {
        // Lidar requires motion BVH, which the renderer disables by default.
        ovrtx_config_entry_t entry = ovrtx_config_entry_motion_bvh(OVRTX_MOTION_BVH_AUTO);
        ovrtx_config_t config = { &entry, 1 };
        ovsensors_api_status_t status;
        fputs("Creating renderer; first startup may compile shaders for several minutes...\n", stderr);
        status = ovsensors_provider_ovrtx_create(str(OVSENSORS_EXAMPLE_OVRTX_LIBRARY), &config, &app.provider);
        if (status != OVSENSORS_API_SUCCESS)
        {
            fprintf(stderr, "Provider creation failed (status %d)\n", (int)status);
            goto finish;
        }
        app.provider_created = 1;
    }
    if (!check(ovsensors_initialize(NULL), "initialize"))
        goto finish;
    app.initialized = 1;
    if (!check(ovsensors_create_instance(&app.provider, 1, &app.instance), "create_instance") ||
        !check(ovsensors_attach_ovstage(app.instance, app.stage), "attach_ovstage"))
        goto finish;
    app.attached = 1;
    {
        ovsensors_sensor_spec_t spec = {0};
        spec.name = str("lidar");
        spec.product = str("/Render/Products/LidarProduct");
        spec.provider = &app.provider;
        if (!check(ovsensors_create_sensor(app.instance, &spec, &app.sensor), "create_sensor"))
            goto finish;
    }
    // [/snippet:point-cloud-setup]
    for (unsigned step = 0; step < 4; ++step)
    {
        ovsensors_enqueue_result_t op;
        ovsensors_step_result_set_t results = {0};
        ovsensors_result_t fetched;
        int copied, released;
        if (!seal(app.stage, 2 + step))
            goto finish;
        op = ovsensors_step(app.instance, &app.sensor, 1, 0.1, 2 + step);
        if (!check((ovsensors_result_t){op.status}, "step"))
            goto finish;
        app.live_op = op.op_index;
        app.has_live_op = 1;
        fetched = ovsensors_fetch_result_set(app.instance, app.live_op, OVSENSORS_TIMEOUT_INFINITE, &results);
        if (fetched.status != OVSENSORS_API_SUCCESS)
        {
            if (fetched.status == OVSENSORS_API_OP_FAILED)
            {
                ovx_string_t error = ovsensors_get_last_op_error(app.instance, app.live_op);
                fprintf(stderr, "fetch_result_set failed: %.*s\n", (int)error.length, error.ptr ? error.ptr : "");
            }
            else
                check(fetched, "fetch_result_set");
            goto finish;
        }
        copied = harvest(&app, &results, &cloud);
        released = release(&app);
        if (!copied || !released)
            goto finish;
    }
    {
        int nonzero_position = 0;
        for (size_t i = 0; i < cloud.row_count; ++i)
            for (size_t axis = 0; axis < 3; ++axis)
                nonzero_position |= cloud.rows[i][axis] != 0.0f;
        if (cloud.row_count && !nonzero_position)
        {
            fputs("All exported XYZ positions are zero\n", stderr);
            goto finish;
        }
    }
    if (!cloud.row_count)
    {
        fputs("No valid point-cloud returns produced\n", stderr);
        goto finish;
    }
    code = 0;
finish:
    if (!cleanup(&app))
        code = 1;
    if (code || !write_cloud(output, &cloud))
        return 1;
    printf("Saved %zu rows from %" PRId32 " delivered entries: %s (metadata: %s.json)\n",
           cloud.row_count, cloud.delivered_count, output, output);
    return 0;
}
