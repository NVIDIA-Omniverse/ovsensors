---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
name: interpreting-step-results
description: >-
  Interpret ovsensors holds, failures, frame windows and mapped outputs; choose CPU or CUDA representations and preserve tensor and parameter lifetimes.
license: LicenseRef-NvidiaProprietary
---

# Interpreting step results

Read the [shared types](../../include/ovsensors/ovsensors_types.h) for result
contracts and the [application header](../../include/ovsensors/ovsensors.h) for
operation and mapping lifetimes. For setup and teardown, use
[application flow](../application-flow/SKILL.md).

## Classify before mapping

A successful fetch returns structure. Group `k` corresponds to sensor `k` passed
to `step`. For each group:

- **Held:** `SUCCESS` and no frames. This is normal cadence behavior.
- **Produced:** `SUCCESS` and one or more frames.
- **Failed:** a failure status and no frames, with that group's error message.

A group failure does not fail the fetch or invalidate other sensors' frames.
Conversely, a terminal fetch failure yields no result tree: read the operation
error and release the operation. Do not confuse these two error channels.

Check every output's validity before mapping; an invalid output can still have a
nonzero handle. After a successful map, check producer status and honor its
readiness dependency. Map-call success alone does not mean the producer completed
successfully. CPU examples require `EventStatus.COMPLETED` and no CUDA dependency
before reading.

> **Source:** [steady-state/main.py](../../examples/python/steady-state/main.py),
> snippets `classify-groups` and `map-output`; function `require_cpu_ready`.
>
> **Source (C++):** [minimal/main.cpp](../../examples/c/minimal/main.cpp), snippets
> `walk-result-set` and `read-mapped-output`.

Frame start/end are absolute simulation-time windows, not step indices. They may
reach back before the step delivering them, overlap, or be instantaneous. Align by
time. The supplied steady-state scene produces a camera every 1/30-second step
and a lidar on every third step; this is that provider/scene's behavior, not a
core scheduling guarantee.

## Choose and consume the representation

Look up tensors by name and inspect DLPack dtype, shape, strides and byte offset.
Tensor order and count are not fixed. `mapping.params` is a separate namespace
from tensors; parameters remain CPU-resident even for GPU payloads. Preserve
output metadata and typed parameters when explaining or exporting a result.

- `Device.DEFAULT` preserves the provider's representation. An image can be an
  opaque CUDA array; its raw handle is not a linear device pointer.
- `Device.CPU` requests host data. `numpy(copy=True)` owns its result;
  `numpy(copy=False)` and `numpy.from_dlpack(tensor)` borrow the mapping's storage.
  Zero-copy NumPy views are read-only. Copy while still mapped when data must survive.
- `Device.CUDA` requests linear device memory. DLPack stream negotiation can order
  the consumer after the producer. Finish GPU work or supply consumer completion
  before unmap, including on exceptions.
- `Device.CUDA_ARRAY` explicitly requests an opaque array. Inspect `cuda_array`,
  `shape`, and `dl_dtype`; NumPy, `data_ptr`, and DLPack export reject this representation.
  CUDA array operations require a separate implementation.

Support belongs to each provider. Never silently retry a different device as if
it were the requested one. `sync_stream=0` requests no stream synchronization;
`1` names the default CUDA stream, and other values pass through a native handle.
This option does not make device memory host-readable. DLPack `stream=-1` leaves
ordering to the caller.

> **Source:** [tensor-interop/main.py](../../examples/python/tensor-interop/main.py),
> snippets `inspect-default`, `cpu-dlpack`, `cuda-dlpack`, and `cpu-params`;
> [reduction.py](../../examples/python/tensor-interop/reduction.py) implements the Warp reduction.

That example inspects DEFAULT without reading its opaque image handle, then uses
an explicit CPU copy and an explicit linear CUDA mapping. Warp performs a real RGB
reduction and synchronizes the device before unmap, including after a launch error.
It does not demonstrate CUDA-array pixel consumption or asynchronous completion
handoff. Parameters come from a separate lidar map because the camera has none.

## Respect independent lifetimes

Result groups, frames and output handles expire at operation release. A live map
retains its own metadata, descriptors, tensors and parameters until consuming
unmap, including after operation release or sensor destruction. Python array and
DLPack objects do not keep a mapping open. A failed non-consuming unmap retains
its storage and handle for exact retry; see the header for consuming statuses.

> **Source:** [custom-provider/main.cpp](../../examples/c/custom-provider/main.cpp),
> snippet `independent-map`; [main.py](../../examples/c/custom-provider/main.py),
> snippet `python-independent-map`.

## Interpret points using their output contract

The ovrtx [provider header](../../include/ovsensors/ovsensors_provider_ovrtx.h)
owns its tensor layout. When `Counts` and `Flags` are present, `Counts[0]` bounds
delivered entries; it does not count valid returns. The examples select valid
entries with `Flags & 0x40`. Coordinates are `float32[3,N]` in the provider layout;
the Python export transposes selected values to owned `float32[N,3]`.

> **Source:** [steady-state/main.py](../../examples/python/steady-state/main.py),
> snippet `copy-points-and-params`; [point-cloud/main.c](../../examples/c/point-cloud/main.c)
> performs bounded C11 export with typed frame parameters.

An empty point array is a data outcome to diagnose, not proof of a successful scan.
Inspect the scene, cadence, mapping status and renderer configuration. For a product
with several sources, use the provider's documented association with
`descriptor.sources`; these single-source examples establish no tiled or batch layout.
