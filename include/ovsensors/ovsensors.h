/* Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

/**
 * @file ovsensors.h
 * @brief The ovsensors application API.
 *
 * @details
 * ovsensors simulates sensors over a scene owned by an ovstage instance. The core owns no
 * providers and links no renderer or physics engine: each provider owns its own substrate and
 * claims sensors by their authored applied schema. The application owns the clock and the loop.
 *
 * Explicit initialization and every live instance hold process references. Initialization is
 * optional when no start-up configuration is needed.
 *
 * **Timeouts.** fetch, map, reset and flush_log take one cooperative budget:
 * OVSENSORS_TIMEOUT_POLL supplies a zero wait budget and OVSENSORS_TIMEOUT_INFINITE has no
 * deadline. For instance calls the budget includes waiting for the instance before doing the
 * work. The budget cannot interrupt synchronous provider or backend work. TIMEOUT leaves work
 * live for exact retry. Other entry points wait for the instance without a deadline and may block
 * in the stage or provider; none is suitable for a real-time thread.
 *
 * **Threading.** One call at a time runs against an instance for the call's whole duration;
 * separate instances run concurrently. Instance exclusion is not fair. Provider slots must not
 * reenter their owning instance; detected reentrancy returns OVSENSORS_API_ERROR.
 *
 * **Topology.** Sensor creation observes current topology. Serialize creation with stage
 * publication and keep a live sensor's products, member prims, applied schemas, and authored config
 * immutable; change config by destroying and recreating in a quiescent window. Dynamic state
 * such as transforms may keep being published at the ordinals passed to step.
 *
 * @version 0.2.0
 * @date 2026-08-18
 */

#ifndef OVSENSORS_H
#define OVSENSORS_H

#include "ovsensors_types.h"

#ifdef __cplusplus
extern "C"
{
#endif

    /** @defgroup ovsensors_version Version query
     *  @{
     */

    /** Get the API version used to build the loaded library. Compare it with the
     *  OVSENSORS_VERSION_* header macros to check compatibility. Callable at any time,
     *  including before initialize. Each output is optional. */
    void ovsensors_get_version(uint32_t* out_major, uint32_t* out_minor, uint32_t* out_patch);

    /** @} */

    /** @defgroup ovsensors_error Error reporting
     *  @{
     */

    /** Detail for the last failed synchronous call on this thread. Borrowed, valid only until
     *  the next ovsensors call on the same thread (a fiber switch also invalidates it). No
     *  detail is supplied for OVSENSORS_API_TIMEOUT.
     *  @return Nonempty view, or `{NULL, 0}` when none is available. */
    ovx_string_t ovsensors_get_last_error(void);

    /** Borrowed detail for an operation that failed asynchronously -- distinct from
     *  ovsensors_get_last_error(), since an op can fail long after its enqueue returned success.
     *  Call after a fetch reporting OVSENSORS_API_OP_FAILED.
     *  @see ovsensors_release_result_set() for the borrow endpoint.
     *  @return Nonempty view, or `{NULL, 0}` when @p instance is not one of ours, or @p op_id is
     *          unknown to it or did not fail. */
    ovx_string_t ovsensors_get_last_op_error(ovsensors_instance_t* instance, ovsensors_op_id_t op_id);

    /** @} */

    /** @defgroup ovsensors_logging Logging
     *  @{
     */

    /**
     * Register the process-global log callback, replacing any previous one. Passing NULL for
     * @p callback disables logging. A message's @c source identifies the provider it concerns,
     * or is empty for ovsensors itself.
     *
     * Requires a live reference -- an ovsensors_initialize() or a live instance -- so a caller
     * that wants start-up messages initializes first. Delivery is on a dedicated thread that
     * exists for the lifetime of that reference: ovsensors_initialize() starts it, and the last
     * shutdown or instance destroy flushes and joins it.
     *
     * Messages admitted before replacement use the previous registration; later messages use the
     * replacement. On SUCCESS, the previous registration is no longer in use, so callback resources
     * used only by it may be released. Passing NULL also drains the previous callback before
     * returning. OUT_OF_MEMORY leaves the existing registration unchanged.
     *
     * Messages below @p severity are dropped; OVSENSORS_LOG_NONE disables everything. Filtering
     * is by severity alone.
     *
     * @pre Not called from inside a log callback; doing so returns OVSENSORS_API_ERROR rather
     *      than deadlocking.
     * @return SUCCESS, ERROR (reentrant, or no live reference), OUT_OF_MEMORY, INTERNAL.
     */
    ovsensors_result_t ovsensors_set_log_callback(ovsensors_log_severity_t severity,
                                                  ovsensors_log_callback_t callback,
                                                  void* user_data);

    /** Complete delivery of log messages admitted before this call -- a point-in-time barrier, so
     *  messages admitted concurrently may not be included. Admission is best-effort: queue
     *  overflow drops newest messages and may produce a later aggregate warning. Returns
     *  immediately when no admitted work remains.
     *  @pre Not called from inside a log callback; doing so returns OVSENSORS_API_ERROR immediately.
     *  @return SUCCESS, TIMEOUT (not drained in budget), ERROR (reentrant, or no live reference),
     *          INTERNAL. */
    ovsensors_result_t ovsensors_flush_log(ovsensors_timeout_ns_t timeout_ns);

    /** @} */

    /** @defgroup ovsensors_lifecycle Library lifecycle
     *  @{
     */

    /**
     * Initialize ovsensors, or increase its reference count. Every successful call needs one
     * matching shutdown; live instances hold a reference too, so a process may rely on
     * create_instance alone. Initializing explicitly is what lets a caller install a log
     * callback, or pass configuration, before the first instance exists.
     *
     * @param config Optional; NULL or an entry_count of zero means defaults. Borrowed for the
     *               duration of the call. No keys are defined; a nonzero entry_count is
     *               INVALID_ARGUMENT.
     * @return SUCCESS, INVALID_ARGUMENT (malformed config), OUT_OF_MEMORY, INTERNAL.
     */
    ovsensors_result_t ovsensors_initialize(const ovsensors_config_t* config);

    /**
     * Release one explicit initialization reference. The final process reference flushes pending
     * log messages and tears down the dispatcher; live instances keep their own references.
     *
     * A log callback may release a nonfinal reference. Releasing the final process reference
     * from a callback returns ERROR and leaves that reference outstanding.
     *
     * @return SUCCESS, INVALID_ARGUMENT (no matching initialize), or ERROR (final release from
     *         a log callback).
     */
    ovsensors_result_t ovsensors_shutdown(void);

    /** @} */

    /** @defgroup ovsensors_instance Instance lifecycle
     *  @{
     */

    /**
     * Create an unattached instance over a fixed provider set. Contexts must be distinct.
     * Diagnostic names and schema claims may overlap across providers; automatic sensor
     * creation rejects ambiguous matches. Providers may have no automatic claims.
     * Each provider's get_api_version is queried once during successful creation. Major and minor
     * must match OVSENSORS_VERSION_MAJOR / MINOR; any patch version is compatible. Version checks
     * precede access to the remaining vtable slots, and all providers are validated before claims.
     * Incompatible versions return NOT_SUPPORTED with the provider index and expected and supplied
     * versions in ovsensors_get_last_error().
     *
     * @par Ownership
     * @p providers array storage is borrowed only until this call returns. ovsensors copies each
     * bundle's vtable and context pointers; both pointed-to objects remain borrowed until
     * ovsensors_destroy_instance() succeeds. ovsensors never owns or destroys a provider context.
     *
     * @pre No context appears twice or belongs to another live instance. Both checks precede
     *      every provider slot call; a violation returns INVALID_ARGUMENT naming the offending index.
     * @param providers May be NULL only when @p provider_count is zero, yielding an instance that
     *                  can create no sensors.
     * @param provider_count Number of entries in @p providers.
     * @param[out] out_instance NULL on every failure.
     * @return SUCCESS, INVALID_ARGUMENT, NOT_SUPPORTED, OUT_OF_MEMORY, INTERNAL.
     */
    ovsensors_result_t ovsensors_create_instance(const ovsensors_provider_t* providers,
                                                 size_t provider_count,
                                                 ovsensors_instance_t** out_instance);

    /**
     * Destroy an instance and all sensor state it owns. NULL is a successful no-op. Live
     * operations, mappings, or timed-out reset work produce BUSY without mutation. Detaches if
     * still attached; provider contexts are never destroyed here.
     *
     * @pre If still attached, the stage borrow described by ovsensors_detach_ovstage() remains
     *      active through this call.
     * @pre Except for the NULL no-op, not called from a log callback; doing so returns ERROR
     *      without beginning teardown.
     * Calls overlapping successful destruction either finish before teardown or return
     * INVALID_HANDLE. Callers must still respect borrowed-view lifetimes. Do not reuse a handle
     * after successful destruction.
     *
     * @post On failure after teardown began the handle is destroy-retry-only: call again with
     *       the same pointer and make no other call on it.
     * @post SUCCESS ends every provider-bundle borrow supplied at creation. Only then may the
     *       application destroy those providers; earlier destruction is undefined behavior.
     *
     * Teardown converges provided every provider cleanup eventually succeeds; permanently failing
     * cleanup strands the instance and its stage.
     *
     * @return SUCCESS, INVALID_HANDLE, BUSY, ERROR (called from a log callback), or a retryable
     *         cleanup failure (OUT_OF_MEMORY, OP_FAILED, RESET_REQUIRED, INTERNAL).
     */
    ovsensors_result_t ovsensors_destroy_instance(ovsensors_instance_t* instance);

    /** @} */

    /** @defgroup ovsensors_attach Stage attachment
     *  @{
     */

    /**
     * Attach an externally-owned ovstage instance before creating sensors or stepping. Detach
     * before attaching another stage.
     *
     * Fail-closed: on success every provider supplied at creation is attached to @p stage, and a
     * provider that rejects it fails the whole call, leaving the instance detached with any
     * provider attached during the attempt detached again.
     *
     * The stage must report the same ovstage API major and minor versions used to build
     * ovsensors; patch versions may differ. A version mismatch attaches no providers and reports
     * the expected and supplied versions through ovsensors_get_last_error().
     *
     * @pre Not already attached. @p stage is borrowed; see ovsensors_detach_ovstage() for the
     *      borrow endpoint. The application remains responsible for stage publication.
     * @return SUCCESS, INVALID_ARGUMENT, INVALID_HANDLE, BUSY (already attached), NOT_SUPPORTED
     *         (incompatible ovstage API version or a provider rejected the stage),
     *         OUT_OF_MEMORY, OP_FAILED, INTERNAL.
     */
    ovsensors_result_t ovsensors_attach_ovstage(ovsensors_instance_t* instance, ovstage_instance_t* stage);

    /**
     * Detach the attached stage and require every provider to stop dereferencing it.
     *
     * Once preconditions pass, every provider is called and this function succeeds. Provider
     * failures are logged. Every provider must end its borrow of the caller-owned stage even on
     * failure; any retained backend stage state must have an independent lifetime. A failed
     * provider may require destruction before it can be used again.
     *
     * @pre Every sensor destroyed -- including any left live by a failed ovsensors_destroy_sensor()
     *      -- every operation and mapping released, and no reset pending. A live one produces BUSY
     *      without mutation.
     * @post SUCCESS ends the application-facing stage borrow; the application may destroy the
     *       stage. A successful ovsensors_destroy_instance() performs this detach and has the same
     *       effect. Destroying the stage before either succeeds is undefined behavior. Detaching
     *       when not attached is a successful no-op. The instance may be attached again only if
     *       its providers remain reusable.
     * @return SUCCESS, INVALID_HANDLE, BUSY.
     */
    ovsensors_result_t ovsensors_detach_ovstage(ovsensors_instance_t* instance);

    /** @} */

    /** @defgroup ovsensors_sensor Sensor lifecycle
     *  @{
     */

    /**
     * Adopt one RenderProduct as a sensor. @p spec names the product prim; everything else is
     * resolved here -- member sensor prims from the product's camera relationship in authored
     * order, and cadence from their authored rationals. Read the result with ovsensors_get_descriptor().
     *
     * When spec->provider is NULL, elect exactly one provider and applied schema shared by every
     * member. Multiple matching claims are INVALID_ARGUMENT. An explicit provider must match
     * both pointer values of a bundle supplied to this instance, otherwise INVALID_ARGUMENT.
     * It bypasses automatic election, including schema requirements, and descriptor.schema is
     * empty. Only that provider receives create_sensor; a refusal never falls back to another.
     * Providers validate their supported configuration during creation or execution.
     *
     * Every relationship target must resolve in authored order. A RenderProduct may be adopted
     * once per provider instance; different providers may adopt the same path. A failed sensor
     * destruction retains that provider's product claim until cleanup succeeds. Sensor names
     * remain instance-unique. Creation failure returns no sensor handle.
     *
     * Cadence is read from authored `double2` quotients into exact rationals: lane 0 numerator,
     * lane 1 a positive denominator. Both lanes must be integral and within `int64_t` range; a
     * fractional or out-of-range lane is INVALID_ARGUMENT naming the offending attribute.
     * `frameRate` is in frames per second, with a nonnegative numerator; zero selects
     * provider-defined cadence. `startTime` is a signed offset in seconds. Every member must
     * have exactly matching rate and start pairs, including whether each is authored. Values
     * are neither normalized nor scheduled by ovsensors.
     *
     * @pre The instance is attached.
     * @param instance Instance that will own the sensor.
     * @param spec Borrowed for this call; ovsensors copies what it retains.
     * @param[out] out_sensor Caller owns it; destroy with ovsensors_destroy_sensor(). NULL on
     *                        failure.
     * @return SUCCESS, INVALID_ARGUMENT, INVALID_HANDLE, BUSY (product already claimed by this provider),
     *         NOT_ATTACHED, NOT_SUPPORTED (no automatic claim or the selected provider refuses),
     *         OUT_OF_MEMORY, OP_FAILED, RESET_REQUIRED, INTERNAL.
     */
    ovsensors_result_t ovsensors_create_sensor(ovsensors_instance_t* instance,
                                               const ovsensors_sensor_spec_t* spec,
                                               ovsensors_sensor_t** out_sensor);

    /**
     * Retire a sensor and release its provider state. Success immediately invalidates the handle.
     *
     * @pre No unreleased operation references this sensor, and no reset is pending; either is
     *      BUSY without mutation. Live mappings do not block this call; see
     *      ovsensors_unmap_output() for their independent lifetime.
     * @post SUCCESS ends every descriptor borrow obtained through ovsensors_get_descriptor() or a
     *       fetched result set and invalidates all data reachable through those descriptors.
     *       Provider-cleanup failure leaves the handle destroy-retry-only; retry with the same
     *       pointer or let instance teardown retry it. A provider INVALID_HANDLE completes the
     *       destroy and is logged. Do not retain a handle after success: its address may be reused.
     *
     * @return SUCCESS, INVALID_HANDLE, BUSY, or a retryable provider cleanup status.
     */
    ovsensors_result_t ovsensors_destroy_sensor(ovsensors_sensor_t* sensor);

    /** Borrow the identity resolved when the sensor was created; use the explicit string lengths.
     *  Output is NULL on failure.
     *  @see ovsensors_destroy_sensor() for the borrow endpoint.
     *  @return SUCCESS, INVALID_ARGUMENT, INVALID_HANDLE, INTERNAL. */
    ovsensors_result_t ovsensors_get_descriptor(const ovsensors_sensor_t* sensor,
                                                const ovsensors_descriptor_t** out_descriptor);

    /** @} */

    /** @defgroup ovsensors_simulation Sensor simulation
     *  @{
     */

    /**
     * Enqueue a simulation step for an explicit sensor set, covering
     * `[last_step_time, last_step_time + delta_time]`, where @e last_step_time is the instance's
     * simulation clock -- zero at creation, advanced by each accepted step, set absolutely by
     * ovsensors_reset().
     *
     * History for sensors omitted from @p sensors is discarded.
     *
     * Success snapshots @p sensors and returns one aggregate operation. Whether submission waits
     * for completion depends on the provider and its configuration. Later provider failure is
     * sticky at fetch. Every operation must eventually be released.
     *
     * @pre Attached; every handle live, created on this instance, and unique in the set.
     * @param instance Instance whose simulation clock advances.
     * @param sensors Borrowed. May be NULL only when @p sensor_count is zero, which advances time
     *                and produces no frames.
     * @param sensor_count Number of entries in @p sensors.
     * @param delta_time Seconds. Finite and greater than zero.
     * @param ordinal Already-sealed committed-publication lower bound, not a historical selector.
     *                ovsensors neither seals nor replaces it.
     * @return `{SUCCESS, nonzero}` once admitted; otherwise INVALID_ARGUMENT, INVALID_HANDLE,
     *         NOT_ATTACHED, RESET_REQUIRED, OUT_OF_MEMORY or INTERNAL paired with
     *         OVSENSORS_INVALID_OP_ID.
     */
    ovsensors_enqueue_result_t ovsensors_step(ovsensors_instance_t* instance,
                                              ovsensors_sensor_t* const* sensors,
                                              size_t sensor_count,
                                              double delta_time,
                                              ovstage_ordinal_t ordinal);

    /**
     * Resolve one step operation and expose its per-sensor result set. Chronological
     * predecessors are resolved first even when fetches arrive out of order. This returns
     * structure -- frames and output handles -- not payload; ovsensors_map_output() turns a
     * handle into tensors.
     *
     * SUCCESS means the set is complete, not that every sensor succeeded: inspect every group's
     * @c status. A failed group affects no other group and does not fail the fetch; its failure
     * detail is only its @c error_message and does not populate ovsensors_get_last_error(). A
     * terminal provider fetch failure instead makes the whole operation sticky OP_FAILED and
     * exposes no groups.
     *
     * TIMEOUT leaves resolution live for exact retry. OP_FAILED is sticky; see
     * ovsensors_get_last_op_error() and ovsensors_release_result_set(). Repeated successful fetches
     * may return the same immutable result. Malformed provider output is never exposed and makes
     * the operation sticky OP_FAILED, with detail naming the provider and offending field.
     *
     * @param instance Instance that owns @p op_id.
     * @param op_id Live operation returned by ovsensors_step().
     * @param timeout_ns One total budget, including resolution of chronological predecessors.
     * @param[out] out_results Borrowed on success; see ovsensors_release_result_set() for the
     *                         result-tree lifetime. Every failure zeroes the output.
     *
     * @return SUCCESS, TIMEOUT, OP_FAILED, INVALID_ARGUMENT, INVALID_HANDLE, OUT_OF_MEMORY,
     *         INTERNAL.
     */
    ovsensors_result_t ovsensors_fetch_result_set(ovsensors_instance_t* instance,
                                                  ovsensors_op_id_t op_id,
                                                  ovsensors_timeout_ns_t timeout_ns,
                                                  ovsensors_step_result_set_t* out_results);

    /**
     * Map one fetched output using @p map_output_desc; NULL selects provider defaults.
     * Device selection, availability, and supported representations belong to the provider.
     * TIMEOUT returns no mapping and may be retried.
     *
     * On success, honor `out_mapped->cuda_sync` before reading producer data. The returned views
     * are borrowed; see ovsensors_unmap_output() for their lifetime and the map's consumption
     * rules. Every failure zeroes @p out_mapped. Cleanup left pending by a failed map is retried
     * before the next map, reset, detach, or instance destruction; those calls return BUSY if
     * cleanup still cannot complete.
     *
     * @pre The owning operation is live. It need not stay live afterwards -- see
     *      ovsensors_release_result_set().
     * @return SUCCESS, TIMEOUT, or another applicable status (INVALID_ARGUMENT, INVALID_HANDLE,
     *         BUSY, NOT_SUPPORTED, OUT_OF_MEMORY, OP_FAILED, RESET_REQUIRED, INTERNAL).
     */
    ovsensors_result_t ovsensors_map_output(ovsensors_instance_t* instance,
                                            ovsensors_output_handle_t output,
                                            const ovsensors_map_output_description_t* map_output_desc,
                                            ovsensors_timeout_ns_t timeout_ns,
                                            ovsensors_mapped_t* out_mapped);

    /**
     * End mapped access. @p consumer_done is the dependency the provider must honor before
     * recycling memory; `{0, 0}` declares none. SUCCESS consumes @p map and invalidates every
     * mapped pointer. Once a live mapping has been resolved, a provider INVALID_HANDLE also
     * consumes it because the provider reports that mapping already absent. An INVALID_HANDLE
     * from input validation consumes no mapping. Any other provider failure preserves the live
     * mapping for exact retry. Never times out.
     *
     * The tensor and parameter arrays, all metadata strings, and DLPack descriptors with their
     * shape, stride and payload storage remain valid until the map is consumed, including after
     * ovsensors_release_result_set() retires its operation. This call remains required after release.
     *
     * @return SUCCESS, INVALID_ARGUMENT, INVALID_HANDLE, or a retryable provider cleanup status.
     */
    ovsensors_result_t ovsensors_unmap_output(ovsensors_instance_t* instance,
                                              ovsensors_map_handle_t map,
                                              ovsensors_cuda_sync_t consumer_done);

    /**
     * Release a step operation and every non-mapped resource it owns. Live mappings do not block
     * release or lose validity; unmap remains legal after the operation is gone.
     *
     * Pending chronological or provider work returns BUSY before cleanup begins. Once it begins,
     * the result-set, group, frame, and output views, output handles, and operation-error detail are
     * invalidated. Descriptor objects retain the sensor lifetime; live map handles retain the
     * mapping lifetime.
     *
     * @post On provider-cleanup failure @p op_id is release-retry-only and is retried with
     *       the same identifier; releases that already succeeded are not repeated.
     * @return SUCCESS, INVALID_ARGUMENT, INVALID_HANDLE, BUSY, or a retryable provider cleanup
     *         status.
     */
    ovsensors_result_t ovsensors_release_result_set(ovsensors_instance_t* instance, ovsensors_op_id_t op_id);

    /**
     * Join every provider at a new simulation-time origin, for an episode or environment reset.
     * After success `last_step_time` is @p time and the next step runs forward from there.
     *
     * A reset that times out after provider work begins remains pending: retry with the same
     * @p time; a different target is INVALID_ARGUMENT. While pending, sensor creation/destruction
     * and instance destruction are BUSY, as is detaching an attached stage; step is RESET_REQUIRED.
     *
     * A terminal provider failure returns OP_FAILED and ends the attempt. A later reset is a new
     * attempt and may use a different target. Step remains RESET_REQUIRED until a reset succeeds;
     * teardown remains available subject to its normal preconditions.
     *
     * @pre All operations and mappings released, no sensor in failed cleanup.
     * @param instance Instance whose providers and simulation clock are reset.
     * @param time Seconds. Finite.
     * @param timeout_ns One total budget for the joined attempt. Provider failures normalize to
     *                   OP_FAILED while the copied detail preserves the original status.
     * @return SUCCESS, TIMEOUT, INVALID_ARGUMENT, INVALID_HANDLE, BUSY, OUT_OF_MEMORY,
     *         OP_FAILED, INTERNAL.
     */
    ovsensors_result_t ovsensors_reset(ovsensors_instance_t* instance, double time, ovsensors_timeout_ns_t timeout_ns);

    /** @} */

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* OVSENSORS_H */
