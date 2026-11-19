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
name: provider-authoring
description: >-
  Implement or adapt a native ovsensors provider, including CPU models and custom backends: ABI negotiation, stage reads, claims, result ownership, reset and cleanup.
license: LicenseRef-NvidiaProprietary
---

# Provider authoring

Use the [CPU custom-provider example](../../examples/c/custom-provider/README.md)
as the working implementation. Its C++17 shared library implements the C ABI;
C++ and Python applications load the same factory. Read the
[provider header](../../include/ovsensors/ovsensors_provider.h) for exact signatures,
allowed statuses and ownership. The example's factory is application code, not an
additional ovsensors API.

## Start from the right files

- [pose_provider.cpp](../../examples/c/custom-provider/pose_provider.cpp) — context,
  sensor/operation/mapping records, ovstage reads, all slots and factory implementation.
- [pose_provider.h](../../examples/c/custom-provider/pose_provider.h) — sample-local
  create/destroy factory declarations.
- [main.cpp](../../examples/c/custom-provider/main.cpp) and
  [main.py](../../examples/c/custom-provider/main.py) — publishing transforms,
  initial/later reset, explicit selection and reads after operation release.
- [scene.usda](../../examples/c/custom-provider/scene.usda) — an Xform and an untyped
  prim, each behind its own RenderProduct, meters, identity parent, no authored cadence.
- [CMakeLists.txt](../../examples/c/custom-provider/CMakeLists.txt) — separate
  `sample_pose_provider` and `custom_provider` targets using ovsensors/ovstage SDKs.

Build and run with the example README's commands. The provider target uses public
headers and the supplied ovstage/path-dictionary vtables; it links no ovsensors core,
ovstage, ovrtx or CUDA runtime. The host links the APIs it calls. Python loads the
library with `ctypes` and uses public `ovsensors.ovsensors_provider_t`; keep both
library and factory owner alive through successful instance teardown.

## Establish ABI and routing

All **13 slots** are required. Keep the vtable immutable and the context alive for
its borrowing instance. One context serves one live core instance; serialize direct
calls and do not interleave them with core's calls. Slots must not reenter the
instance or let C++ exceptions cross the C boundary. The example's `boundary`
helper contains exceptions and records diagnostic detail.

`get_api_version` reports the implemented **ovsensors provider ABI**, using
`OVSENSORS_VERSION_MAJOR`, `_MINOR`, and `_PATCH`. It is not a provider implementation,
ovstage, or ovrtx version. Write each non-null output independently. Core checks
major/minor equality before attach and permits patch differences. A mismatch rejects
instance creation; old unversioned tables must be rebuilt, not probed as this layout.

`claims` returns a valid diagnostic identity and zero or more unique applied-schema
IDs. Prefer modality/model schemas rather than common base schemas for automatic
election. Overlapping providers can be selected explicitly. Diagnostic names are
not selectors; applications pass the supplied vtable/context bundle.

The example advertises **zero schema IDs** and both hosts select it explicitly.
It registers no custom schema. Automatic selection matches applied-schema
identifiers against provider claims. Register custom schemas when their
definitions, defaults or inheritance are needed, and populate the inputs your
provider requires. See [schema registration](../registering-sensor-schemas/SKILL.md).
Explicit selection requires no matching claim, but the provider must still
validate its inputs. Ordinary ovstage population preserves RenderProduct `camera`
targets independently of prim type; core also has no source-type test. A provider
must validate the data its backend needs. The sample requires an existing
`omni:xform` matrix and reads it at admission, before any host simulation writes.
Its `--render-product` option selects the Xform or the genuinely untyped source
from the same scene. No missing matrix is replaced by an invented transform.

Schema registration makes definitions resolvable; it does not make rendering
population import arbitrary custom attributes. Keep custom attribute population
separate from schema claims and relationship-target preservation.

## Locate every slot

The function names below are in `pose_provider.cpp`; its final `vtable` initializer
shows their ABI order.

| ABI slot | Example function | Main responsibility |
|---|---|---|
| `get_api_version` | `apiVersion` | ovsensors provider ABI, no state change |
| `claims` | `claims` | Valid identity and explicit-only claim set |
| `get_last_error` | `lastError` | Borrowed detail for the latest slot failure |
| `attach_ovstage` | `attach` | Validate ovstage ABI/slots and borrow the stage |
| `detach_ovstage` | `detach` | End the stage borrow |
| `create_sensor` | `createSensor` | Validate supported descriptor, copy source, read at admission ordinal |
| `step` | `step` | Advance time and create operation-owned groups/frames/outputs |
| `fetch_result_set` | `fetch` | Return operation-owned result structure |
| `map_output` | `mapOutput` | Create independent mapping and stable typed views |
| `unmap_output` | `unmap` | Finish mapped access and consume or retain the handle |
| `release_result_set` | `release` | Retire operation resources while preserving existing maps |
| `reset` | `reset` | Establish a new simulation-time origin |
| `destroy_sensor` | `destroySensor` | Retire sensor after its operation references end |

## Read published input and define outputs

Attach has no publication bound: validate interfaces there, not authored
configuration. `create_sensor` receives the resolved descriptor and an admission
ordinal; copy any borrowed fields retained beyond that call. Validate unsupported
source count, type and cadence explicitly. A later step does not imply that core
refreshes the descriptor.

`readTransform` reads the source's `omni:xform` at the supplied sealed ordinal,
then copies it before releasing all ovstage read/query/group/path-list resources.
The host writes and seals before stepping and makes no later write during this
synchronous read. An ordinal is a committed-publication read bound, not historical
payload storage.

The example supports one source with a published local matrix and unauthored rate/start, producing one sample
per accepted step. `LocalTransform` contains a CPU `float64[4,4]` tensor named
`matrix`: parent-local, row-major, translation in row 3. With its meter scene and
identity parent, translations match the authored positions. This is a transform
probe, not world-pose composition or physical sensor simulation. CPU parameters
are `publicationOrdinal` (`int64`), `simulationTime` (`float64` seconds), and
`sourceIndex` (`uint32`). See `Sample` and `Mapping` for their actual descriptors.

Normal population gives the untyped source an identity local matrix; it authors
no USD transform operations. The host then publishes its simulated movement to
that existing matrix column, just as it does for the Xform source.

Keep result and mapping storage independent. `Operation` owns groups, frames and
output handles; `Mapping` owns descriptors/metadata and shares an immutable
`Sample`. Releasing an operation or destroying a sensor must not invalidate an
existing map. Later steps must not overwrite borrowed samples.

> **Source:** [main.cpp](../../examples/c/custom-provider/main.cpp), snippets
> `publish-transform` and `independent-map`; [main.py](../../examples/c/custom-provider/main.py),
> snippets `python-provider-factory` and `python-independent-map`.

## Preserve time, errors and cleanup

The sample requires reset before initial stepping, advances time on empty sensor
sets too, and demonstrates a later reset with no live operations/maps. Frames use
absolute windows on that timeline. A per-sensor read failure becomes a failed
group with detail; a successful empty group means a hold.

Only fetch, map and reset may time out. Fetch retains its operation; map failure
returns an all-zero result and no mapping. Only a reset timeout retains a pending
reset for exact retry; terminal reset failure ends that attempt. Time budgets are
cooperative and cannot interrupt synchronous work. The CPU sample completes
synchronously and does not manufacture timeout behavior.

`DEFAULT` and `CPU` return completed host data with no CUDA dependency. CUDA,
CUDA-array and nonzero synchronization-stream requests fail explicitly. CPU reads
finish before unmap; unsupported consumer dependencies leave the map live for retry.
A GPU backend must implement producer readiness and consumer completion using the
header's synchronization contract; this CPU example does not demonstrate it.

For handle cleanup, `SUCCESS` consumes ownership and `INVALID_HANDLE` reports it
already gone. Other allowed failures retain the handle for exact cleanup retry;
do not free storage then return a retryable status. Detach is different: once its
preconditions hold, every outcome ends the caller's stage borrow. Retained backend
state must own an independent lifetime. Do not depend on core alone to enforce
these rules when the provider is driven directly.

The sample factory rejects destruction while attached or holding resources.
Retired sensor tokens remain allocated until factory destruction to prevent stale
pointer reuse; redesign that bounded-example tradeoff for unbounded sensor churn.
After adapting the provider, exercise actual stage-derived data, reset and retained
maps through its host, and verify the chosen backend's failure and cleanup paths.
