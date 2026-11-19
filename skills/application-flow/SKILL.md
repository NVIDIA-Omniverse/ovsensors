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
name: application-flow
description: >-
  Build or adapt ovsensors applications in Python or C/C++: stage publication, provider selection, stepping and resource cleanup.
license: LicenseRef-NvidiaProprietary
---

# Application flow

Start with [example setup](../../examples/README.md). Use the
[application header](../../include/ovsensors/ovsensors.h) for contracts and the
runnable sources below for complete error handling.

## Choose the provider and source

The [minimal Python application](../../examples/python/minimal/main.py) selects
the ovrtx bundle explicitly; [steady state](../../examples/python/steady-state/main.py)
uses automatic applied-schema routing. The
[custom provider](../../examples/c/custom-provider/README.md) demonstrates an
explicit-only CPU backend, with C++ and Python hosts.

Explicit selection uses the supplied bundle's vtable and context, not its name.
It bypasses schema election, leaves descriptor `schema` empty, and never falls
back to another provider. Automatic election requires applicable schema claims;
overlapping matches need explicit selection. The shipped ovrtx provider has a
fixed claim set. Multiple bundles do not remove backend attachment limits.

Every sensor adopts one RenderProduct. Explicit selection retains product,
member and cadence validation. Ordinary ovstage population preserves `camera`
relationship targets independently of prim type. Core has no source-type test;
the selected provider validates its required input data. The custom example
accepts an Xform or untyped prim with a populated local matrix. Its
`--render-product` option selects either source from the same scene, and admission
reads that matrix before the host publishes movement.

## Follow the owned lifecycle

1. For ovrtx-authored sensor attributes, register schemas before the first ovstage
   API call. Read [schema registration](../registering-sensor-schemas/SKILL.md)
   when the host controls initialization. The CPU custom example needs no ovrtx schema.
2. Create and populate the application-owned stage. Complete writes and seal their
   ordinal before admitting sensors or stepping. A publication bound is not a
   historical snapshot; do not infer configuration refresh from a later step.
3. Create providers, create an instance with those bundles, then attach the stage.
   An instance initially has no stage. Keep each context and its library alive
   through successful instance destruction; a context serves one live instance.
4. Admit sensors and inspect descriptors when diagnosing routing or cadence.
   Providers own cadence behavior. The custom provider requires an explicit reset
   before its first step; it also demonstrates a later reset with no live work.
5. Step with a sealed ordinal and positive time delta. Own every accepted operation
   until release, including when fetch fails. A fetch timeout leaves work pending
   for retry; a terminal fetch failure still requires release.
6. Inspect groups and map valid outputs using
   [interpreting results](../interpreting-step-results/SKILL.md). Complete reads or
   copy data before unmap. Release operations, destroy sensors, detach/destroy the
   instance, then release its provider and stage owners.

> **Source (Python):** [minimal/main.py](../../examples/python/minimal/main.py),
> snippets `build-stage`, `provider-and-instance`, and `step-and-fetch`.
>
> **Source (C++):** [minimal/main.cpp](../../examples/c/minimal/main.cpp), snippets
> `populate-stage`, `create-provider-and-instance`, `create-sensor`, and `step-fetch-release`.
>
> **Source (automatic routing):** [steady-state/main.py](../../examples/python/steady-state/main.py),
> snippet `create-sensor-set`.

## Preserve failure ownership

Python examples put fetch inside a `try`/`finally` that releases the accepted
operation. The `Instance.results(op)` context manager does not release an operation
when its own entry fails; do not rely on that context alone for failed-fetch cleanup.
Read `release_operation` and `destroy_stage` in the minimal application before
adapting its exception path. Cleanup must preserve the original failure detail.

`BUSY` at detach or instance destruction signals live resources or unfinished
cleanup. Do not destroy the stage while it remains borrowed. The standalone Python
examples terminate without finalizers if cleanup leaves it attached; do not copy
process termination into a reusable library helper.

Mappings are independent of operations: release can retire result structure and
output handles while an existing mapping survives. Sensor destruction is also
permitted after its operation references are gone, even with live maps. Detach
requires all mappings, operations and sensors to be gone. Read each cleanup
function's status contract before retrying or discarding a handle.

Read synchronous last-error detail immediately; a later call can overwrite it.
For terminal fetch failure, use the operation error. Per-sensor failed groups own
their own messages. An instance-creation version rejection concerns the provider's
ovsensors ABI major/minor, which is separate from the backend version.
