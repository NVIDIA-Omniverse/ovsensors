/* Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

/**
 * @file ovsensors_provider.h
 * @brief The ovsensors provider interface: the dispatch table a sensor backend implements.
 *
 * @details
 * ovsensors core owns no provider and links no renderer or physics engine. A provider owns its
 * backend, borrows the shared ovstage, and claims sensors by applied-schema identifiers.
 * An application obtains a bundle from the provider's factory and passes it to
 * @ref ovsensors_create_instance().
 *
 * A provider needs these headers, not the ovsensors library: nothing here is an imported symbol.
 *
 * **Ownership.** Provider bundles are borrowed and provider contexts are never destroyed by
 * ovsensors. See @ref ovsensors_destroy_instance() for the bundle-borrow endpoint.
 *
 * **One bundle, one instance.** A context backs exactly ONE instance for that instance's whole
 * lifetime. Passing the same bundle to a second ovsensors_create_instance() while the first
 * instance is alive is refused with OVSENSORS_API_INVALID_ARGUMENT; the context is free to bind
 * again once @ref ovsensors_destroy_instance() has succeeded.
 *
 * **Slot contract.** Every slot is required, is serialized by its owning instance, must not
 * throw across the C boundary, and must not reenter that instance. Direct callers must serialize
 * calls on each context and must not interleave them with calls from its owning instance.
 *
 * **Enforcement.** ovsensors validates slot statuses, handles, result structure, frame ordering,
 * and mapped descriptors. Malformed result sets make the operation sticky OVSENSORS_API_OP_FAILED;
 * malformed mappings return OVSENSORS_API_OP_FAILED with provider-specific detail.
 *
 * Only @ref ovsensors_provider_vtable_t::fetch_result_set, @ref ovsensors_provider_vtable_t::map_output
 * and @ref ovsensors_provider_vtable_t::reset may return OVSENSORS_API_TIMEOUT. Fetch and reset
 * retain their work for exact retry; map returns an all-zero result and no mapping.
 * Cleanup slots never time out. Handle-consuming cleanup preserves input ownership on failure
 * for an exact retry, except OVSENSORS_API_INVALID_HANDLE, which reports the input is already
 * gone. @ref ovsensors_provider_vtable_t::detach_ovstage has its own terminal borrow contract and
 * is never retried by ovsensors.
 *
 * A slot handed a foreign or retired handle may return OVSENSORS_API_INVALID_HANDLE. Providers
 * must remain usable when driven directly, even though ovsensors does not pass such handles.
 *
 * @version 0.2.0
 * @date 2026-08-18
 */

#ifndef OVSENSORS_PROVIDER_H
#define OVSENSORS_PROVIDER_H

#include "ovsensors_types.h"

#ifdef __cplusplus
extern "C"
{
#endif

    /** @defgroup ovsensors_provider_interface_types Provider interface types
     *  @{
     */

    /** The provider's own handle for one created sensor -- its side of ovsensors_sensor_t. */
    typedef struct ovsensors_provider_sensor_t ovsensors_provider_sensor_t;

    /**
     * One provider's part of an accepted step.
     *
     * Only the delta is supplied: the provider owns its own clock, seeded by
     * the origin given to @ref ovsensors_provider_vtable_t::reset and advanced by each accepted step.
     */
    typedef struct
    {
        ovstage_ordinal_t ordinal; /**< Caller-guaranteed sealed publication every provider reads. */
        double delta_time; /**< Time to simulate, in seconds. Checked finite and greater than zero. */
        /** Unique live sensors of this provider, for this call only. NULL exactly when the count is zero. */
        ovsensors_provider_sensor_t* const* sensors;
        size_t sensor_count; /**< May be zero: every provider is dispatched for every accepted step. */
    } ovsensors_provider_step_desc_t;

    /** One provider-private output in a sensor result group. */
    typedef struct
    {
        ovx_string_t name; /**< 1 to 1024 readable bytes at a non-NULL pointer, e.g. "rgb". */
        ovsensors_output_validity_t validity; /**< Exactly OVSENSORS_OUTPUT_INVALID or OVSENSORS_OUTPUT_VALID. */
        /** Nonzero; see @ref ovsensors_provider_vtable_t::release_result_set for its lifetime. */
        uint64_t provider_output_handle;
    } ovsensors_provider_output_t;

    /** One provider frame in a step result. */
    typedef struct
    {
        double frame_start_time; /**< Finite, absolute seconds. */
        double frame_end_time; /**< Finite; not before the start. */
        const ovsensors_provider_output_t* outputs; /**< NULL exactly when the count is zero. */
        size_t output_count; /**< May be zero. */
    } ovsensors_provider_frame_t;

    /** One provider result group, for one participating sensor context. */
    typedef struct
    {
        ovsensors_provider_sensor_t* sensor; /**< Matching sensor from the step input set. */
        ovsensors_api_status_t status; /**< Exactly SUCCESS, NOT_SUPPORTED, or OP_FAILED. */
        /** Empty on success; otherwise 1 to 1024 readable bytes at a non-NULL pointer. */
        ovx_string_t error_message;
        const ovsensors_provider_frame_t* frames; /**< NULL exactly when the count is zero. */
        size_t frame_count; /**< Zero for a failed group. */
    } ovsensors_provider_sensor_result_t;

    /** Provider-owned result view from a successful @ref ovsensors_provider_vtable_t::fetch_result_set. */
    typedef struct
    {
        /** NULL allowed at count zero; see @ref ovsensors_provider_vtable_t::release_result_set for its lifetime. */
        const ovsensors_provider_sensor_result_t* groups;
        size_t group_count; /**< Exactly one group per sensor in the corresponding step. */
    } ovsensors_provider_result_set_t;

    /** Provider-owned mapped view from a successful @ref ovsensors_provider_vtable_t::map_output.
     *  See @ref ovsensors_provider_vtable_t::unmap_output for its borrow endpoint.
     *  String readability and empty-value rules match @ref ovsensors_mapped_t. */
    typedef struct
    {
        const ovsensors_tensor_t* tensors; /**< Tensor array; NULL allowed only at count zero. */
        size_t tensor_count; /**< May be zero. */
        ovsensors_cuda_sync_t cuda_sync; /**< Producer-ready sync handed to the application. */
        uint64_t provider_map_handle; /**< Nonzero; see @ref ovsensors_provider_vtable_t::unmap_output. */
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
    } ovsensors_provider_mapped_t;

    /** Diagnostic identity and automatic routing claims advertised at instance creation. */
    typedef struct
    {
        ovx_string_t name; /**< Nonempty label used as the core log source, e.g. "ovsensors.rtx"; need not be unique. */
        const ovx_string_t* schema_ids; /**< Unique nonempty applied-schema ids; NULL allowed at count zero. */
        size_t schema_id_count; /**< Zero makes this an explicit-only provider. */
    } ovsensors_provider_claims_t;

    /** @} */

    /** @defgroup ovsensors_provider_vtable Provider dispatch table
     *  @{
     */

    /**
     * Required immutable dispatch table for one provider implementation.
     * The first slot, @ref ovsensors_provider_vtable_t::get_api_version, retains its position and
     * signature across API versions.
     * Unversioned tables do not satisfy this contract.
     */
    struct ovsensors_provider_vtable_t
    {
        /**
         * Report the OVSENSORS_VERSION_MAJOR / MINOR / PATCH contract this provider implements,
         * independent of its backend's product version. Write each non-NULL output; NULL outputs
         * are ignored. This query does not change provider state.
         */
        void (*get_api_version)(const ovsensors_provider_context_t* context,
                                uint32_t* out_major,
                                uint32_t* out_minor,
                                uint32_t* out_patch);

        /**
         * Report identity and the complete schema claim set during ovsensors_create_instance(),
         * before attaching. Claims are unique within the provider.
         *
         * Claim modality- or model-specific schemas, not shared base schemas. Overlapping claims
         * are allowed across providers; automatic sensor creation rejects ambiguous matches.
         * An explicit selection does not require any matching claim.
         *
         * @return Non-NULL claims and reachable data borrowed until the encompassing
         *         ovsensors_create_instance() call returns, or NULL to report a malformed
         *         provider (fails create).
         */
        const ovsensors_provider_claims_t* (*claims)(const ovsensors_provider_context_t* context);

        /**
         * Detail for the most recent non-timeout slot failure on this context. ovsensors
         * copies the bytes before continuing. Return a borrowed view of at most 1024 bytes, valid
         * until the next slot call on this context, or `{NULL, 0}` when there is no detail to
         * give. Malformed or throwing detail is a contract violation, reported as INTERNAL.
         */
        ovx_string_t (*get_last_error)(const ovsensors_provider_context_t* context);

        /**
         * Attach this context to the instance's stage. On failure, core rolls back providers
         * already attached in the same attempt. Do not read authored configuration: no publication
         * bound is supplied.
         *
         * @pre The context is detached.
         * @param stage Borrowed stage; see @ref ovsensors_provider_vtable_t::detach_ovstage for its lifetime.
         * @post Failure leaves no borrow of the supplied stage and no stage resources requiring
         *       the caller's handle to remain valid. Any retained backend references must be
         *       independently owned and safe to release during provider destruction. An
         *       already-attached context returns OP_FAILED without disturbing its existing
         *       attachment. A terminal context refuses attachment without accessing the stage.
         * @return SUCCESS, or NOT_SUPPORTED / OUT_OF_MEMORY / OP_FAILED / INTERNAL with detail.
         *         A direct call with a NULL context or stage returns INVALID_ARGUMENT, which
         *         ovsensors never supplies and does not accept as an attach-time failure.
         */
        ovsensors_api_status_t (*attach_ovstage)(ovsensors_provider_context_t* context, ovstage_instance_t* stage);

        /**
         * End this context's borrow of the attached stage. Do not dereference the caller's stage
         * handle again. Release stage-derived resources or retain them through independently
         * owned backend references that remain safe after the caller releases its stage handle.
         *
         * ovsensors calls every attached provider once, logs failures, never retries, and still
         * succeeds. A failure must not retain a borrow of the caller's stage handle.
         *
         * @pre Every sensor this provider created is destroyed, every provider operation
         *      released, and every mapping unmapped.
         * @post When the preconditions hold, every outcome ends this context's stage borrow.
         *       Failure may leave the context terminal: it must then refuse attachment and
         *       ordinary backend work, without retrying backend detach, while preserving safe
         *       provider destruction. A reusable context may be attached again; reusability is
         *       provider-specific. Must not time out.
         * @return SUCCESS, or OUT_OF_MEMORY / OP_FAILED / INTERNAL with detail. A direct call
         *         with a NULL context returns INVALID_ARGUMENT, which ovsensors never supplies.
         */
        ovsensors_api_status_t (*detach_ovstage)(ovsensors_provider_context_t* context);

        /**
         * Create provider state for one routed sensor. ovsensors has resolved everything:
         * @p descriptor carries the adopted RenderProduct prim, the member prims behind it in
         * authored order, the schema that matched this provider's automatic claim (empty for
         * explicit selection), and the authored cadence. Do not re-resolve these facts -- read
         * any further authored configuration from the attached stage at @p snapshot_ordinal and copy what you retain.
         *
         * @p snapshot_ordinal is a committed-publication read bound, not a historical topology
         * selector; a later step implies no configuration refresh.
         * The provider validates support for the selected product and configuration during
         * creation or execution. Explicit selection is authoritative; core never tries another
         * provider after failure.
         *
         * @param descriptor Borrowed for this call only; copy anything you retain.
         * @param snapshot_ordinal Committed-publication read bound for authored configuration.
         * @param[out] out_sensor Initialized to NULL by ovsensors. Success returns a unique
         *                        sensor handle; failure returns NULL. See
         *                        @ref ovsensors_provider_vtable_t::destroy_sensor for its lifetime.
         * @return SUCCESS, or INVALID_ARGUMENT (malformed authored configuration) / BUSY /
         *         NOT_SUPPORTED / OUT_OF_MEMORY / OP_FAILED / RESET_REQUIRED / INTERNAL.
         *         Timeout is not allowed.
         */
        ovsensors_api_status_t (*create_sensor)(ovsensors_provider_context_t* context,
                                                const ovsensors_descriptor_t* descriptor,
                                                ovstage_ordinal_t snapshot_ordinal,
                                                ovsensors_provider_sensor_t** out_sensor);

        /**
         * Enqueue this provider's sensors for one accepted step. @p step and its unique sensor set
         * are borrowed for this call only. A successful operation retains logical sensor
         * references; see @ref ovsensors_provider_vtable_t::release_result_set for their endpoint.
         * Submission may wait for completion according to the backend configuration.
         *
         * @post Success returns a nonzero provider operation id. Every other status must return
         *       OVSENSORS_INVALID_OP_ID and leave no live operation. Return RESET_REQUIRED when
         *       time may have been consumed inconsistently, INVALID_HANDLE for an unknown sensor
         *       in the set. Timeout, not-supported and invalid-argument are not allowed.
         */
        ovsensors_enqueue_result_t (*step)(ovsensors_provider_context_t* context,
                                           const ovsensors_provider_step_desc_t* step);

        /**
         * Resolve one live operation and expose all participating sensor groups. ovsensors
         * zero-initializes @p out_results. Success returns exactly one group per sensor passed
         * to @ref ovsensors_provider_vtable_t::step, each appearing once in any order. Group status,
         * detail and frame-count rules are defined by @ref ovsensors_provider_sensor_result_t.
         * A successful group with zero frames means the sensor was not due. Frame windows are
         * finite, ordered within a group, and sorted by `(end, start)`.
         *
         * @param op_id Live operation returned by @ref ovsensors_provider_vtable_t::step.
         * @param timeout_ns Remaining cooperative budget: zero polls, INFINITE has no deadline.
         * @param[out] out_results Borrowed on success; see
         *                         @ref ovsensors_provider_vtable_t::release_result_set for its lifetime.
         * @return SUCCESS; TIMEOUT (operation stays pending, result all zero); INVALID_HANDLE
         *         (@p op_id unknown, nothing retained); or a terminal OUT_OF_MEMORY / OP_FAILED /
         *         RESET_REQUIRED / INTERNAL, which also zeroes the result and retains the
         *         operation for release. No other status is allowed.
         */
        ovsensors_api_status_t (*fetch_result_set)(ovsensors_provider_context_t* context,
                                                   ovsensors_op_id_t op_id,
                                                   ovsensors_timeout_ns_t timeout_ns,
                                                   ovsensors_provider_result_set_t* out_results);

        /**
         * Map one provider output using @p map_output_desc, or provider defaults when NULL.
         * ovsensors forwards the request without validating device availability or placement.
         * Success returns a nonzero provider map and zero or more tensors and CPU parameters.
         * `cuda_sync` communicates producer readiness; `sync_stream` requests synchronization
         * during mapping. CUDA arrays remain opaque handles in kDLCUDA/kDLOpaqueHandle tensors.
         * Layout and association with @ref ovsensors_descriptor_t::sources are provider- and
         * output-defined; each tensor's DLPack descriptor specifies its shape and strides.
         *
         * @param provider_output_handle Live output from a successful result set.
         * @param map_output_desc Mapping request, or NULL for provider defaults.
         * @param timeout_ns Remaining cooperative budget: zero polls, INFINITE has no deadline.
         * @param[out] out_mapped Borrowed on success; see
         *                        @ref ovsensors_provider_vtable_t::unmap_output for its lifetime.
         * @post Every non-success result leaves @p out_mapped all zero.
         * @return SUCCESS, TIMEOUT (no mapping; retryable), or another known status with detail.
         *         Unknown values are treated as INTERNAL.
         */
        ovsensors_api_status_t (*map_output)(ovsensors_provider_context_t* context,
                                             uint64_t provider_output_handle,
                                             const ovsensors_map_output_description_t* map_output_desc,
                                             ovsensors_timeout_ns_t timeout_ns,
                                             ovsensors_provider_mapped_t* out_mapped);

        /**
         * End mapped access after honoring @p consumer_done. Must not time out.
         *
         * The returned status determines ownership of the mapping. SUCCESS and INVALID_HANDLE
         * consume @p provider_map_handle: ovsensors retires its record and never presents the
         * handle again. Every other status retains the mapping, and ovsensors presents the same
         * handle again on an exact retry. Its tensor and parameter arrays, all metadata strings,
         * DLPack descriptors with their shape, stride and payload storage, and handle remain stable
         * until SUCCESS or INVALID_HANDLE consumes it, including after
         * @ref ovsensors_provider_vtable_t::release_result_set retires its operation.
         * A provider that has already released the mapping must therefore report INVALID_HANDLE,
         * with the release error in its detail, rather than a retryable status.
         *
         * @return SUCCESS; INVALID_HANDLE for an unknown or released-with-error mapping,
         *         distinguished by detail; or BUSY / OUT_OF_MEMORY / OP_FAILED / RESET_REQUIRED /
         *         INTERNAL.
         */
        ovsensors_api_status_t (*unmap_output)(ovsensors_provider_context_t* context,
                                               uint64_t provider_map_handle,
                                               ovsensors_cuda_sync_t consumer_done);

        /**
         * Release all non-mapped resources owned by one operation. Called once resolution is no
         * longer pending, after either a successful or a terminally failed fetch. Success
         * consumes @p op_id, its logical sensor references, output handles and result storage
         * (groups, frames, output arrays and strings). INVALID_HANDLE reports it already gone;
         * every other failure retains it release-retry-only. Must not time out.
         *
         * Mappings may outlive their operation. ovsensors may call this while mappings issued
         * from @p op_id are live; the provider must not refuse for that reason. See
         * @ref ovsensors_provider_vtable_t::unmap_output for the lifetime of mapped storage.
         *
         * @return SUCCESS, INVALID_HANDLE (@p op_id unknown), or a retryable BUSY / OUT_OF_MEMORY
         *         / OP_FAILED / RESET_REQUIRED / INTERNAL.
         */
        ovsensors_api_status_t (*release_result_set)(ovsensors_provider_context_t* context, ovsensors_op_id_t op_id);

        /**
         * Join this provider's time state to a new origin. No operations or mappings are live.
         * Success completes this provider's part of the current reset attempt.
         *
         * @param time New simulation-time origin, in seconds. Finite.
         * @param timeout_ns Remaining cooperative budget: zero polls, INFINITE has no deadline.
         *                   This budget cannot interrupt synchronous backend submission.
         * @return SUCCESS; TIMEOUT (work retained for exact retry); or OUT_OF_MEMORY / OP_FAILED
         *         / RESET_REQUIRED / INTERNAL, which are TERMINAL: the attempt is over, the
         *         provider retains no pending reset, and a later reset is a fresh attempt rather than a
         *         retry of this one. Only TIMEOUT retains the pending reset.
         */
        ovsensors_api_status_t (*reset)(ovsensors_provider_context_t* context,
                                        double time,
                                        ovsensors_timeout_ns_t timeout_ns);

        /**
         * Destroy one provider sensor after its final operation reference is released. Mappings
         * from this sensor may still be live; see @ref ovsensors_provider_vtable_t::unmap_output
         * for their lifetime. Success consumes @p sensor; INVALID_HANDLE reports it already gone.
         * Every other failure preserves it destroy-retry-only; ovsensors retries only this cleanup
         * and admits no new work for the sensor. Must not time out.
         * @return SUCCESS, INVALID_HANDLE (@p sensor unknown), or a retryable BUSY /
         *         OUT_OF_MEMORY / OP_FAILED / RESET_REQUIRED / INTERNAL.
         */
        ovsensors_api_status_t (*destroy_sensor)(ovsensors_provider_context_t* context,
                                                 ovsensors_provider_sensor_t* sensor);
    };

    /** @} */

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* OVSENSORS_PROVIDER_H */
