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
name: registering-sensor-schemas
description: >-
  Register ovrtx USD sensor schemas before ovstage initialization; diagnose missing authored sensor attributes or cadence without confusing schema claims with registration.
license: LicenseRef-NvidiaProprietary
---

# Registering sensor schemas

ovrtx supplies the USD schema plugins for its authored sensor attributes. Register
their paths before the first ovstage API call initializes USD. Start from the
[shared setup](../../examples/README.md) so the selected loader and schema resources
come from compatible packages.

## Python and native applications

Call `ovrtx.register_schema_paths()` before creating `ovstage.Stage()` or otherwise
calling ovstage. Importing the selected Python bindings alone does not initialize
the USD backend.

> **Source (Python):** [minimal/main.py](../../examples/python/minimal/main.py),
> snippet `register-schema-paths`.

C and C++ applications call `ovrtx_register_schema_paths()` before their first
ovstage API call. The sample links the public ovrtx loader for registration and
passes that same loader to the ovsensors provider factory. The provider's ownership
of its renderer is independent of the application's registration call.

> **Source (C++):** [minimal/main.cpp](../../examples/c/minimal/main.cpp), snippet
> `register-schema-paths`; [CMake setup](../../examples/c/cmake/ovsensors_example.cmake).

## Embedding hosts

An embedding host may initialize USD before the application's entry point.
Inspect the actual loading and initialization order. Register during host startup
or arrange schema plugin paths before launching that host. Linking the public
ovstage loader alone is not evidence that USD has already initialized.

A restart can be necessary for an already initialized host; ordinary standalone
examples require no re-exec. Do not add one without identifying the host constraint.

## Custom providers and diagnosis

Automatic provider selection matches source prims' applied-schema identifiers
against provider claims. Advertising a claim does not define or register a USD
schema. Register custom schemas when your application needs their definitions,
defaults or inheritance, and ensure the provider's required inputs are populated.
Explicit provider selection does not require a matching schema claim; the selected
provider still requires valid inputs.

For custom schema families, call `ovstage.population.register_usd_schemas(...)`
(C: `ovstage_population_register_usd_schemas(...)`) before the first operation that
reads USD schema definitions.

The [CPU custom provider](../../examples/c/custom-provider/README.md) advertises
zero schema IDs and is selected explicitly. Its Xform and untyped-prim targets
need no custom schema and no ovrtx registration.

Ordinary ovstage population preserves RenderProduct `camera` targets independently
of prim type. Each provider validates its required source data. The custom sample
uses populated local matrices, verified before its first simulation write.
An untyped prim does not become a USD Xform by authoring `xformOp:*` attributes;
the sample leaves it untyped and publishes movement through ovstage.

Schema registration makes definitions resolvable. It does not make rendering
population import arbitrary custom attributes, and advertising a provider claim
changes neither population nor schema registration.

For unexpected `0/0` cadence or missing `omni:sensor:*` values, inspect authored
attributes, registered plugins and initialization order. The symptom alone does
not prove late registration. Read `sensor.descriptor` or
`ovsensors_get_descriptor()` after admission. A missing relationship target can
also be an authored-scene or population issue.

Continue with [application flow](../application-flow/SKILL.md) for stage publication
and [interpreting results](../interpreting-step-results/SKILL.md) for cadence holds.
