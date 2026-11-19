/* Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

/** @file ovsensors_types.h
 *  Common, pure-C ovsensors types: status and result shapes, handles, the fetched
 *  result tree, and initialization configuration.
 */

#ifndef OVSENSORS_TYPES_H
#define OVSENSORS_TYPES_H

#define OVSENSORS_VERSION_MAJOR 0
#define OVSENSORS_VERSION_MINOR 2
#define OVSENSORS_VERSION_PATCH 0

#ifndef OVSENSORS_DEPRECATED
#    if defined(__cplusplus) && __cplusplus >= 201402L
#        define OVSENSORS_DEPRECATED(msg) [[deprecated(msg)]]
#    elif defined(__GNUC__) || defined(__clang__)
#        define OVSENSORS_DEPRECATED(msg) __attribute__((deprecated(msg)))
#    elif defined(_MSC_VER)
#        define OVSENSORS_DEPRECATED(msg) __declspec(deprecated(msg))
#    else
#        define OVSENSORS_DEPRECATED(msg)
#    endif
#endif

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

/* Shared strings and DLPack tensors. ovx_string_t always carries an explicit byte length;
 * use literal_to_ovx_string(...) or a C compound literal rather than a bare const char*. */
#include <ovx/types.h>
#include <ovx/dlpack/dlpack.h>

/* Compatible forward declarations for the ovstage substrate read by ovsensors. */
#ifndef OVSTAGE_ORDINAL_T_DECLARED
#    define OVSTAGE_ORDINAL_T_DECLARED
typedef int64_t ovstage_ordinal_t;
#endif
#ifndef OVSTAGE_INSTANCE_T_DECLARED
#    define OVSTAGE_INSTANCE_T_DECLARED
typedef struct ovstage_instance_t ovstage_instance_t;
#endif

#ifdef __cplusplus
extern "C"
{
#endif

    /** @defgroup ovsensors_core_types Core object types
     *  @{
     */

    /** Opaque runtime holding borrowed provider bundles and owning sensors, operations,
     *  outputs, and maps. */
    typedef struct ovsensors_instance_t ovsensors_instance_t;
    /** Opaque application sensor handle owned by an ovsensors instance. */
    typedef struct ovsensors_sensor_t ovsensors_sensor_t;
    /** Required provider dispatch table; defined by ovsensors_provider.h. */
    typedef struct ovsensors_provider_vtable_t ovsensors_provider_vtable_t;
    /** Provider-private state. ovsensors never inspects its representation. */
    typedef struct ovsensors_provider_context_t ovsensors_provider_context_t;

    /** @} */

    /** @defgroup ovsensors_handle_types Handle types
     *  @{
     */

    /** Operation identifier, scoped to its issuing ovsensors instance or provider. */
    typedef uint64_t ovsensors_op_id_t;
    /** Output identifier, scoped to one live aggregate operation and its owning instance. */
    typedef uint64_t ovsensors_output_handle_t;
    /** Mapping identifier, scoped to one ovsensors instance. */
    typedef uint64_t ovsensors_map_handle_t;

    /** Sentinel operation identifier. Zero is never a live operation. */
#define OVSENSORS_INVALID_OP_ID ((ovsensors_op_id_t)0)
    /** Sentinel output handle. Test validity, not the handle, to decide whether payload is usable. */
#define OVSENSORS_INVALID_OUTPUT_HANDLE ((ovsensors_output_handle_t)0)
    /** Sentinel map handle. Zero is never a live mapping. */
#define OVSENSORS_INVALID_MAP_HANDLE ((ovsensors_map_handle_t)0)

    /** @} */

    /** @defgroup ovsensors_sync_types Timeout and synchronization types
     *  @{
     */

    /** Cooperative timeout budget in nanoseconds. */
    typedef uint64_t ovsensors_timeout_ns_t;

    /** Poll once without waiting. */
#define OVSENSORS_TIMEOUT_POLL ((ovsensors_timeout_ns_t)0)
    /** Wait without a deadline while the provider continues to make progress. */
#define OVSENSORS_TIMEOUT_INFINITE ((ovsensors_timeout_ns_t)~0ULL)

    /** Exact rational, matching an authored USD @c double2 (numerator, denominator). 0/0 = unset. */
    typedef struct
    {
        int64_t num; /**< Numerator. */
        int64_t den; /**< Positive denominator when authored; zero when unset. */
    } ovsensors_rational_t;

    /** CUDA producer/consumer synchronization, passed by value across a map boundary.
     *  Layout matches @c ovrtx_cuda_sync_t. Both fields zero means no outstanding dependency;
     *  tensor placement is described by DLTensor.device. */
    typedef struct
    {
        uintptr_t stream; /**< CUDA stream: 0 means none, 1 the default stream, greater values a specific stream. */
        uintptr_t wait_event; /**< CUDA event to wait on before access or recycling; 0 means none. */
    } ovsensors_cuda_sync_t;

    /** @} */

    /** @defgroup ovsensors_result_types Result and status types
     *  @{
     */

    /**
     * Status shared by the application API and the provider interface.
     *
     * Numeric values are explicit and ABI-stable. Failure detail is available through the
     * applicable last-error function or per-sensor error message. Last-error detail may be empty
     * if it could not be recorded, including on allocation failure; OVSENSORS_API_TIMEOUT has none.
     */
    typedef enum
    {
        OVSENSORS_API_SUCCESS = 0, /**< The call succeeded; see the entry point's completion semantics. */
        OVSENSORS_API_ERROR = 1, /**< Unspecified failure; see the applicable last-error function. */
        OVSENSORS_API_TIMEOUT = 2, /**< Cooperative timeout budget expired. */
        OVSENSORS_API_INVALID_ARGUMENT = 3, /**< An argument or authored configuration is invalid. */
        OVSENSORS_API_INVALID_HANDLE = 4, /**< A nonzero handle is unknown, expired, or unusable. */
        OVSENSORS_API_BUSY = 5, /**< A live prerequisite prevents the transition; retry later. */
        OVSENSORS_API_NOT_SUPPORTED = 6, /**< The requested schema, device, or output is unsupported. */
        OVSENSORS_API_OUT_OF_MEMORY = 7, /**< A required allocation or resource reservation failed. */
        OVSENSORS_API_OP_FAILED = 8, /**< Asynchronous operation work completed unsuccessfully. */
        OVSENSORS_API_RESET_REQUIRED = 9, /**< Time consumption may be inconsistent; reset before stepping. */
        OVSENSORS_API_NOT_ATTACHED = 10, /**< The call needs a stage; attach one with ovsensors_attach_ovstage(). */
        OVSENSORS_API_INTERNAL = 99 /**< Contract violation, invariant failure, or caught exception. */
    } ovsensors_api_status_t;

    /** Result returned by synchronous ovsensors entry points; detail is thread-local. */
    typedef struct
    {
        ovsensors_api_status_t status; /**< Status of the call. */
    } ovsensors_result_t;

    /**
     * Result returned by an asynchronous enqueue.
     *
     * @c status is the enqueue status, not the completion status: success means the work was
     * accepted, not that it ran. @c op_index is OVSENSORS_INVALID_OP_ID whenever
     * @c status is not OVSENSORS_API_SUCCESS.
     */
    typedef struct
    {
        ovsensors_api_status_t status; /**< Status of the enqueue. */
        /** Live opaque operation id on success; OVSENSORS_INVALID_OP_ID otherwise. */
        ovsensors_op_id_t op_index;
    } ovsensors_enqueue_result_t;

    /** @} */

    /** @defgroup ovsensors_log_types Logging types
     *  @{
     */

    /** Log severity levels. Values and ordering match @c ovstage_log_severity_t. */
    typedef enum
    {
        OVSENSORS_LOG_VERBOSE = -2, /**< Most verbose (debug/trace). */
        OVSENSORS_LOG_INFO = -1, /**< Informational. */
        OVSENSORS_LOG_WARNING = 0, /**< Warning; the operation continues. */
        OVSENSORS_LOG_ERROR = 1, /**< Error; the operation may have failed (fatal reports here too). */
        OVSENSORS_LOG_NONE = 3 /**< Threshold sentinel: as a filter level, disables all logging.
                                *   Never delivered to the callback. */
    } ovsensors_log_severity_t;

    /**
     * Callback for receiving ovsensors log messages.
     *
     * Delivered asynchronously on the logging thread, one invocation at a time. Synchronize
     * access to any state shared with other threads. See ovsensors_set_log_callback().
     *
     * Logging is supplemental; disabling it does not change status or last-error reporting.
     *
     * @param severity Severity of the message. Never OVSENSORS_LOG_NONE.
     * @param timestamp Wall-clock time in seconds since the epoch.
     * @param source Provider the message concerns, identified by its claim name; `{NULL, 0}` for
     *               ovsensors itself. Both @p source and @p message are borrowed for this call only.
     * @param message Message text.
     * @param user_data Context supplied to ovsensors_set_log_callback().
     */
    typedef void (*ovsensors_log_callback_t)(
        ovsensors_log_severity_t severity, double timestamp, ovx_string_t source, ovx_string_t message, void* user_data);

    /** @} */

    /** @defgroup ovsensors_provider_types Provider bundle types
     *  @{
     */

    /**
     * Provider bundle: a dispatch table paired with the state it operates on.
     * Field order and types remain fixed across API versions so callers can query the
     * vtable's first get_api_version slot before accessing the rest of the table.
     *
     * ovsensors never owns or destroys either field.
     * @see ovsensors_destroy_instance() for the bundle-borrow endpoint.
     */
    typedef struct
    {
        const ovsensors_provider_vtable_t* vtable; /**< Borrowed immutable dispatch table. */
        ovsensors_provider_context_t* context; /**< Borrowed provider-private state; caller-owned. */
    } ovsensors_provider_t;

    /** @} */

    /** @defgroup ovsensors_output_types Output, mapping, and result-tree types
     *  @{
     */

    /** Requested tensor representation. The provider selects actual placement and validates support. */
    typedef enum
    {
        OVSENSORS_MAP_DEVICE_TYPE_DEFAULT = 0, /**< Provider's most efficient representation. */
        OVSENSORS_MAP_DEVICE_TYPE_CPU = 1, /**< Read back to CPU memory. */
        OVSENSORS_MAP_DEVICE_TYPE_CUDA = 2, /**< Linear CUDA device memory. */
        OVSENSORS_MAP_DEVICE_TYPE_CUDA_ARRAY = 3 /**< CUDA array: kDLCUDA / kDLOpaqueHandle. */
    } ovsensors_map_device_type_t;

    /** Mapping request forwarded unchanged to the provider. Params remain CPU-resident. */
    typedef struct
    {
        ovsensors_map_device_type_t device_type; /**< Requested representation; DEFAULT lets the provider choose. */
        /** Synchronize production to this CUDA stream: 0 none, 1 default, greater values specific.
         *  When nonzero, access after map on this stream or a stream synchronized with it is ready. */
        uintptr_t sync_stream;
    } ovsensors_map_output_description_t;

    /** Producer status carried by a mapped output. */
    typedef enum
    {
        OVSENSORS_EVENT_PENDING = 0,
        OVSENSORS_EVENT_COMPLETED = 1,
        OVSENSORS_EVENT_FAILURE = 2
    } ovsensors_event_status_t;

    /** Payload validity of a produced output. Zero initialization means unusable payload,
     *  not an absent output. */
    typedef enum
    {
        OVSENSORS_OUTPUT_INVALID = 0, /**< Output exists, but its payload is not a valid sample. */
        OVSENSORS_OUTPUT_VALID = 1 /**< Output payload is a valid sample. */
    } ovsensors_output_validity_t;

    /** One named DLPack tensor in a successful mapping; see ovsensors_unmap_output(). */
    typedef struct
    {
        ovx_string_t name; /**< Constituent name. */
        const DLTensor* dl; /**< DLPack descriptor and payload; CUDA arrays are opaque, not linear memory. */
        ovx_string_t doc; /**< Human-readable description. */
    } ovsensors_tensor_t;

    /** One CPU-resident mapped parameter; see ovsensors_unmap_output(). */
    typedef struct
    {
        DLTensor dl; /**< DLPack descriptor by value; its shape, strides, and CPU payload are borrowed. */
        ovx_string_t name; /**< Parameter name, independent of tensor names. */
        ovx_string_t doc; /**< Human-readable parameter description. */
    } ovsensors_param_t;

    /** One named output in an operation-owned frame; see ovsensors_release_result_set(). */
    typedef struct
    {
        ovx_string_t name; /**< Nonempty output name. */
        ovsensors_output_validity_t validity; /**< Explicit payload validity; never infer it from zeroes. */
        ovsensors_output_handle_t handle; /**< Nonzero. */
    } ovsensors_output_t;

    /** Successful mapped view; see ovsensors_unmap_output() for its borrow endpoint.
     *  Strings in this view and its tensor/parameter descriptors may be empty. Nonempty strings
     *  require a non-NULL pointer to length readable bytes; no terminator is required. */
    typedef struct
    {
        const ovsensors_tensor_t* tensors; /**< Tensor array; NULL allowed only at count zero. */
        size_t tensor_count; /**< Number of entries; may be zero. */
        ovsensors_cuda_sync_t cuda_sync; /**< Producer-ready dependency to honor before payload access. */
        ovsensors_map_handle_t map; /**< Nonzero. */
        /** Producer state: PENDING, COMPLETED or FAILURE; each is valid on map-call SUCCESS. */
        ovsensors_event_status_t status;
        ovx_string_t error_message; /**< Producer error detail; may be empty for any status. */
        ovx_string_t name; /**< Output name. */
        ovx_string_t type; /**< Provider-defined output type. */
        ovx_string_t doc; /**< Human-readable output description. */
        int version; /**< Provider-defined output schema version. */
        /** CPU parameter array, separate from tensors; NULL allowed only at count zero. */
        const ovsensors_param_t* params;
        size_t param_count; /**< Number of entries; may be zero. */
    } ovsensors_mapped_t;

    /** One emit of one sensor: an absolute closed frame window in seconds on the simulation
     *  timeline, and the named outputs it produced.
     *  @see ovsensors_release_result_set() for the operation lifetime. */
    typedef struct
    {
        double frame_start_time; /**< Finite absolute start time in seconds. */
        double frame_end_time; /**< Finite absolute end time, not before the start. */
        const ovsensors_output_t* outputs; /**< NULL exactly when output_count is zero. */
        size_t output_count; /**< Number of entries in outputs. */
    } ovsensors_frame_t;

    /** Create spec for one named sensor and the RenderProduct it adopts. */
    typedef struct
    {
        ovx_string_t name; /**< Instance-unique sensor name, used in diagnostics and refusals. */
        /** The adopted RenderProduct prim path. Ownership is per provider instance;
         *  see ovsensors_create_sensor(). */
        ovx_string_t product;
        /** Optional authoritative selection from the instance's provider set. NULL selects
         *  automatic routing. The bundle is matched by its vtable and context pointer values,
         *  not this pointer's address; borrowed only for ovsensors_create_sensor(). */
        const ovsensors_provider_t* provider;
    } ovsensors_sensor_spec_t;

    /**
     * Reflection of one adopted sensor, fully resolved at admission.
     *
     * Callers receive a borrowed view of the creation-time identity and configuration.
     * @see ovsensors_destroy_sensor() for the borrow endpoint.
     */
    typedef struct
    {
        ovx_string_t name; /**< The name from the spec. */
        ovx_string_t product; /**< The adopted RenderProduct prim path. */

        /** The product's camera-relationship targets in authored order. The provider and output
         *  format define how tensor elements and parameters correspond to these members. */
        const ovx_string_t* sources;
        size_t source_count; /**< Number of entries in sources. */

        /** Applied schema that elected the provider automatically; empty for explicit selection. */
        ovx_string_t schema;
        ovx_string_t provider_name; /**< The provider serving this sensor. */

        ovsensors_rational_t rate; /**< Authored frames per second; 0/0 when unauthored. */
        ovsensors_rational_t start; /**< Authored start offset in seconds; 0/0 when unauthored. */
    } ovsensors_descriptor_t;

    /**
     * One sensor's part of a step result; exactly one group exists per participating sensor.
     *
     * SUCCESS with zero frames means nothing was due. NOT_SUPPORTED or OP_FAILED carries a
     * nonempty @c error_message and does not affect other groups.
     * @see ovsensors_release_result_set() for the operation-owned fields' lifetime.
     */
    typedef struct
    {
        const ovsensors_descriptor_t* descriptor; /**< Borrowed. @see ovsensors_destroy_sensor(). */
        ovsensors_api_status_t status; /**< Exactly SUCCESS, NOT_SUPPORTED, or OP_FAILED. */
        ovx_string_t error_message; /**< Empty on success; nonempty otherwise. */
        const ovsensors_frame_t* frames; /**< NULL exactly when frame_count is zero. */
        size_t frame_count; /**< Zero for a failed group; otherwise zero or more frames. */
    } ovsensors_sensor_result_t;

    /**
     * Borrowed aggregate result for every sensor admitted to one step operation.
     *
     * Contains frames and output handles, not payload.
     * @see ovsensors_release_result_set() for the result-tree lifetime.
     */
    typedef struct
    {
        const ovsensors_sensor_result_t* groups; /**< NULL exactly when group_count is zero. */
        /** One group per sensor captured by the accepted step. Group k belongs to sensor k of the
         *  array passed to ovsensors_step(), whatever order the providers answered in. */
        size_t group_count;
    } ovsensors_step_result_set_t;

    /** @} */

    /** @defgroup ovsensors_config_types Configuration
     *  @{
     */

    /** Incomplete configuration-entry type. No entries are defined in this version. */
    typedef struct ovsensors_config_entry_t ovsensors_config_entry_t;

    /** Config for ovsensors_initialize(). NULL, or an @c entry_count of zero, means defaults. */
    typedef struct
    {
        const ovsensors_config_entry_t* entries; /**< Ignored when entry_count is zero. */
        size_t entry_count; /**< Must be zero in this version. */
    } ovsensors_config_t;

    /** @} */

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* OVSENSORS_TYPES_H */
