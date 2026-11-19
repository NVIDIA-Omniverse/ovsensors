# ovsensors Python bindings

Python bindings for the ovsensors 0.2 C API. A platform wheel includes the native
core and the optional ovrtx provider library. It does not bundle ovstage or ovrtx,
and installing it adds no Python dependencies. Applications select their own
providers, runtimes, and array libraries.

## Install

Use Python 3.10–3.13 and a package matching your platform: Linux x86-64 or aarch64
with glibc 2.35 or newer, or Windows x86-64. Linux x86-64 requires an AVX-capable CPU.
The Linux wheel tags are `manylinux_2_35_x86_64` and `manylinux_2_35_aarch64`;
Windows uses `win_amd64`.

Prebuilt Windows x64 wheels and SDKs require a Microsoft C++ runtime compatible
with MSVC 14.38 (Visual Studio 2022 17.8) or newer. Install the current supported
[Microsoft v14 x64 Redistributable](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170)
unless your application supplies compatible runtime libraries. Installing Python
or ovstage alone does not satisfy this prerequisite.

For a supplied wheel, set `OVSENSORS_WHEEL` to its full original filename, including
the version and platform tags. From your application directory on Linux:

```sh
uv venv --python 3.11 .venv
uv pip install --python .venv "$OVSENSORS_WHEEL"
.venv/bin/python -c "import ovsensors; print(ovsensors.__version__); print(ovsensors.version())"
```

On Windows PowerShell, use `$env:OVSENSORS_WHEEL` in the install command and
`.venv\Scripts\python.exe` to run Python. When a compatible release is available
from your package index, `uv pip install --python .venv 'ovsensors==0.2.*'` selects
that API line instead.

`ovsensors.__version__` is the distribution version, including build qualifiers.
`ovsensors.version()` loads the native core and returns its API version tuple.
The wheel finds its bundled libraries without `PYTHONPATH` or loader overrides.
The 0.2 API is pre-release; a minor release may break compatibility before 1.0.

## Use it

The [source examples](../examples/README.md) provide complete runnable applications:
start with a [camera PNG](../examples/python/minimal/README.md), or the
[CPU custom provider](../examples/c/custom-provider/README.md) when building a
backend without a renderer. Source examples are obtained separately from the
wheel or SDK. NumPy, Pillow, ovstage and ovrtx are example dependencies, not
installation dependencies of this package.

Use normal imports:

```python
import ovsensors
from ovsensors.providers import ovrtx as ovrtx_provider
```

The ovrtx wrapper creates and owns its renderer. Its optional `ovrtx_library_path`
selects an absolute ovrtx loader filename; otherwise it discovers the selected
ovrtx package. Keep the provider alive through successful instance teardown.
A custom native bundle uses public `ovsensors.ovsensors_provider_t`; its library
and factory owner must be retained by the caller.

Mappings expose borrowed tensors and separate CPU parameters. For an owned NumPy
array, the runnable camera example requests `Device.CPU` and calls
`numpy(copy=True)` before unmap. `Device.DEFAULT` can expose an opaque CUDA array;
it is not a linear pixel pointer. The [tensor interop example](../examples/python/tensor-interop/README.md)
shows CPU DLPack copying and linear CUDA consumption with producer ordering and
consumer completion. See the [result guide](../skills/interpreting-step-results/SKILL.md)
for lifetime and status checks.

## SDK and source bindings

An extracted SDK supplies plain bindings under `python/ovsensors/`. Select them
with `PYTHONPATH=/absolute/path/to/sdk/python` (PowerShell:
`$env:PYTHONPATH = 'C:\path\to\sdk\python'`) and use an environment with the
application's dependencies. Keep the SDK together: its bindings find native
libraries in the SDK's `bin/`. No Python project installation is needed.

An editable install from this source directory, `uv pip install --python .venv -e .`,
installs bindings only; it does not build or supply native libraries. For a split
source/native setup, `OVSENSORS_LIBRARY_PATH` selects the absolute directory holding
the compatible core library, and `OVSENSORS_PROVIDER_OVRTX_LIBRARY_PATH` selects
the ovrtx provider library directory. These are ovsensors library locations;
`ovrtx_library_path` selects the separate renderer loader.

A source checkout without generated distribution metadata reports the API base
version through `__version__`. Use the [shared setup](../examples/README.md) for
complete SDK and wheel applications.
