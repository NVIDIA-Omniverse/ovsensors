# ovsensors source guide

Use [README.md](README.md) for orientation and
[examples/README.md](examples/README.md) for consumer setup.

## Choose the task guide

- [Application flow](skills/application-flow/SKILL.md): stage publication,
  provider selection, sensor lifetime, stepping, and cleanup.
- [Interpreting step results](skills/interpreting-step-results/SKILL.md): holds,
  errors, frame windows, named tensors, parameters, and CPU/CUDA ownership.
- [Registering sensor schemas](skills/registering-sensor-schemas/SKILL.md): early
  ovrtx schema registration and embedding-host initialization order.
- [Provider authoring](skills/provider-authoring/SKILL.md): implement or adapt the
  native provider ABI using the independent CPU example.

## Source and contract ownership

The four [public headers](include/ovsensors/) own the C API contracts. Python
bindings live in [python/ovsensors/](python/ovsensors/). Runnable recipes and local
scenes live in [examples/](examples/); link to their functions or snippet markers
instead of duplicating application code in prose. Keep referenced markers and
relative links correct when moving code.

Public headers remain C-compatible, with opaque handles and no C++ or host-framework
types. Preserve names, layouts, license notices, and documented ownership. A public
symbol rename is an API change. Use shared ovx strings and DLPack types; provider
interface types use the `ovsensors_provider_` prefix. Headers state observable
contracts, while examples explain concrete use.

Providers and stages remain caller-owned. An instance borrows its provider
contexts through successful instance destruction. Mapped payloads and metadata
remain valid until successful unmap or a consuming `INVALID_HANDLE`, including
when their operation or sensor has already been released. Read the owning header
before changing lifetime or error behavior.

## Example changes

Keep setup declared in [examples/pyproject.toml](examples/pyproject.toml) or the
example's CMake project. Select SDKs explicitly; do not make configuration download
a runtime. Keep scenes local, outputs predictable, and headless operation usable.
The [custom provider](examples/c/custom-provider/README.md) has a separate CPU route
using only ovsensors, ovstage, and NumPy for its Python host.

Run the affected example with the selected packages and inspect its actual PNG,
point array, or transform output. Report the platform and any unexecuted paths.
Keep package installation instructions in the shared setup or Python README;
keep each example's CLI and output meaning in its own README.
