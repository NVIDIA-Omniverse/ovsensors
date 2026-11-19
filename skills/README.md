<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: LicenseRef-NvidiaProprietary

NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
property and proprietary rights in and to this material, related
documentation and any modifications thereto. Any use, reproduction,
disclosure or distribution of this material and related documentation
without an express license agreement from NVIDIA CORPORATION or
its affiliates is strictly prohibited.
-->
# ovsensors skills

Use the task guide that matches the work:

- [Application flow](application-flow/SKILL.md): stage publication, provider
  selection, operations, and cleanup.
- [Interpreting step results](interpreting-step-results/SKILL.md): holds/errors,
  frame times, named tensors, parameters, and CPU/CUDA mapping lifetimes.
- [Registering sensor schemas](registering-sensor-schemas/SKILL.md): early schema
  registration and embedding-host initialization.
- [Provider authoring](provider-authoring/SKILL.md): implement a native backend
  using the real CPU provider and its C++/Python hosts.

The [headers](../include/ovsensors/) define contracts. Skills link runnable
[example sources](../examples/) and named functions or paired
`[snippet:name]` / `[/snippet:name]` markers. Read those regions before adapting a
recipe. When moving code, preserve referenced markers or update the referring
skill. Keep executable API recipes in the examples instead of copying them into
another guide. Use the [shared setup](../examples/README.md) for installation.
