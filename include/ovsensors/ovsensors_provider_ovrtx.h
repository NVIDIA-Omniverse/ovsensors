/* Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

/**
 * @file ovsensors_provider_ovrtx.h
 * @brief An ovsensors provider backed by ovrtx.
 *
 * This provider ships separately from ovsensors core and owns the ovrtx renderer it creates. One
 * ovsensors sensor corresponds to one ovrtx RenderProduct.
 *
 * @par Tensor layout
 * @ref ovsensors_descriptor_t::sources lists the product's camera-relationship targets in authored
 * order. Their association with tensor elements is output-specific. Tensors retain the ovrtx
 * output layout; inspect each DLPack descriptor for shape and strides. For point-cloud outputs
 * with @c Counts and @c Flags, Counts[0] bounds delivered entries; Flags describes entry validity
 * according to that output's contract.
 *
 * Tensors and CPU parameters occupy separate arrays in @ref ovsensors_provider_mapped_t.
 * Parameters remain on the CPU for CUDA mappings. Look up entries by name within each array;
 * tensor order and count are not contractual.
 */

#ifndef OVSENSORS_PROVIDER_OVRTX_H
#define OVSENSORS_PROVIDER_OVRTX_H

#include <ovrtx/ovrtx_types.h>
#include <ovsensors/ovsensors_provider.h>

#ifdef __cplusplus
extern "C"
{
#endif

    /** @defgroup ovsensors_provider_ovrtx ovrtx provider factory
     *  @{
     */

    /**
     * Create a provider and its renderer.
     *
     * The provider advertises its fixed ovrtx modality schemas through
     * @ref ovsensors_provider_vtable_t::claims.
     *
     * Loads the public ovrtx library at the supplied absolute path. The library must export the
     * required ovrtx entry points and match the major/minor version of the ovrtx headers used to
     * build this provider (any patch version). The version is checked before configuration is
     * passed to ovrtx. The provider retains the library through renderer destruction, including
     * failure cleanup; see
     * @ref ovsensors_provider_ovrtx_destroy() for failed-destruction behavior.
     *
     * A non-NULL configuration is passed unchanged to ovrtx_create_renderer(). ovrtx owns
     * configuration validation, defaults and duplicate-key semantics. NULL is passed as an empty
     * configuration, as required by ovrtx.
     *
     * If ovrtx_initialize() has not run, renderer creation also initializes ovrtx;
     * @p config must then include initialization settings as well as renderer settings, and that
     * implicit initialization remains live until renderer destruction.
     *
     * Successful detach permits another attachment. Failed detach makes this provider terminal:
     * it refuses attachment and ordinary backend work, never retries backend detach, and remains
     * destroyable. ovrtx retains any stage state needed for destruction under its own ownership;
     * see @ref ovsensors_provider_vtable_t::detach_ovstage for the stage-borrow endpoint.
     * Use a new provider to resume work after terminal failure.
     *
     * @param ovrtx_library_path Absolute path to the public ovrtx loader library, borrowed for this
     *                           call. Must be nonempty and contain no NUL bytes; a terminating NUL
     *                           is not required. Windows paths use UTF-8.
     * @param config Renderer configuration borrowed for this call. ovrtx copies retained values.
     *               NULL requests ovrtx defaults. Supplied configuration must satisfy ovrtx's contract.
     * @param[out] out_provider Provider bundle to pass to ovsensors_create_instance(). Unchanged on
     *                          failure. See @ref ovsensors_destroy_instance() for its borrow lifetime.
     * @return SUCCESS, INVALID_ARGUMENT for invalid inputs (including a relative path),
     *         OUT_OF_MEMORY, or INTERNAL when the library cannot be loaded, required symbols are
     *         missing, the ovrtx version is incompatible, or renderer creation fails.
     */
    ovsensors_api_status_t ovsensors_provider_ovrtx_create(ovx_string_t ovrtx_library_path,
                                                           const ovrtx_config_t* config,
                                                           ovsensors_provider_t* out_provider);

    /**
     * Destroy the provider and its renderer. NULL and an already-destroyed bundle are successful
     * no-ops.
     *
     * If renderer destruction fails or throws, the library and any retained backend resources
     * remain resident for process lifetime.
     *
     * @param provider Bundle returned by @ref ovsensors_provider_ovrtx_create(), or NULL.
     * @pre The bundle borrow has ended; see @ref ovsensors_destroy_instance().
     * @return SUCCESS, or INTERNAL if renderer destruction failed. The bundle is consumed and
     *         zeroed on either result; destruction cannot be retried.
     */
    ovsensors_api_status_t ovsensors_provider_ovrtx_destroy(ovsensors_provider_t* provider);

    /** @} */

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* OVSENSORS_PROVIDER_OVRTX_H */
