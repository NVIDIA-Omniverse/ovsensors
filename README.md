# ovsensors

ovsensors simulates sensors over an application-owned ovstage scene. Applications
use a C API or Python bindings; providers implement the sensor backend. The
separate ovrtx provider supplies RTX rendering, while a custom provider can run
entirely on the CPU.

The current API is **0.2.0, pre-release**. Minor releases may break compatibility
before 1.0. Use source examples and runtime packages from compatible releases.

## Start here

- **Render a camera image:** follow the [example setup](examples/README.md), then
  run [Python minimal](examples/python/minimal/README.md) or
  [C++ minimal](examples/c/minimal/README.md). Both save a 256 × 256 PNG without a display.
- **Write your own backend:** build the
  [CPU custom provider](examples/c/custom-provider/README.md). Its C++ and Python
  applications move an Xform or an untyped prim in ovstage and read real local transforms through
  the same provider library. This route needs no ovrtx runtime or GPU.
- **Install Python bindings:** the [Python package guide](python/README.md)
  explains wheels, SDK bindings, and source-only installations.

Binary SDKs and wheels contain the runtime and development artifacts. Obtain
these examples and skills separately from the matching source release; they are
not included in either binary format. ovrtx and ovstage are separate dependencies.

## Examples

| Goal | Example | Result |
|---|---|---|
| Implement a native provider | [Custom provider, C++ and Python](examples/c/custom-provider/README.md) | Stage-derived `float64[4,4]` transforms, typed metadata, reset and retained mappings |
| Save a camera image | [Python minimal](examples/python/minimal/README.md) | Owned CPU image saved as PNG; explicit provider selection |
| Use the C API from C++17 | [C++ minimal](examples/c/minimal/README.md) | Camera PNG using installed SDKs |
| Run cameras and lidar at different cadences | [Python steady state](examples/python/steady-state/README.md) | PNG, point arrays and frame metadata; automatic provider selection |
| Consume a lidar from C11 | [Point cloud](examples/c/point-cloud/README.md) | At most 4,096 valid XYZ/intensity rows in CSV, plus JSON metadata |
| Consume CPU and CUDA tensors | [Python tensor interop](examples/python/tensor-interop/README.md) | NumPy copy, Warp reduction, image and metadata; optional Warp dependency |

Each example owns its small local USD scene. See [common setup](examples/README.md)
for packages and compilers, then use the example's README for commands and outputs.

## How it fits together

A sensor adopts one RenderProduct. Its `camera` relationship identifies the source
prims. Automatic routing matches their applied schemas to provider claims;
applications can instead select a supplied provider bundle explicitly. A provider
name is a diagnostic label, not a selector.

The application publishes stage changes, seals an ordinal, and supplies that
ordinal with each simulation step. Results are pulled:

```text
step → fetch result set → map named output → read or copy → unmap → release
```

A fetched result set contains groups, frames, and output handles. Mapping an output
exposes borrowed DLPack tensors and separate CPU parameters. Check group failures,
output validity, producer status, and synchronization before reading. Successful
empty groups are cadence holds. A mapping can remain alive after its operation is
released; its storage stays borrowed until unmap.

The [application-flow skill](skills/application-flow/SKILL.md) follows the complete
lifecycle. [Interpreting results](skills/interpreting-step-results/SKILL.md) covers
frame times, mapping representations, ownership, and GPU ordering.

## Source layout

- [`include/ovsensors/`](include/ovsensors/) — authoritative application,
  provider, shared-type, and ovrtx factory C headers.
- [`python/ovsensors/`](python/ovsensors/) — ctypes bindings and provider wrappers.
- [`examples/`](examples/) — runnable consumers and the independent CPU provider.
- [`skills/`](skills/README.md) — four task guides for coding agents, including
  [provider authoring](skills/provider-authoring/SKILL.md).

For changes to this source, read [AGENTS.md](AGENTS.md).
See [release notes](CHANGELOG.md) and [version](VERSION.md).

## License and security

See [license](LICENSE) and [security reporting](SECURITY.md).

NOTICE AND DISCLAIMER: This software automatically retrieves, accesses or interacts with external materials. Those retrieved materials are not distributed with this software and are governed solely by separate terms, conditions and licenses. You are solely responsible for finding, reviewing and complying with all applicable terms, conditions, and licenses, and for verifying the security, integrity and suitability of any retrieved materials for your specific use case. This software is provided “AS IS”, without warranty of any kind. The author makes no representations or warranties regarding any retrieved materials, and assumes no liability for any losses, damages, liabilities or legal consequences from your use or inability to use this software or any retrieved materials. Use this software and the retrieved materials at your own risk.
