# ovsensors examples

These source examples consume installed ovsensors packages. Keep the whole example
tree from the matching source release; SDKs and wheels do not include it. Run the
commands below from this `examples/` directory. Scenes resolve beside their source
or executable, and output paths are relative to your working directory.

## Choose an example

- [Custom provider — C++ and Python](c/custom-provider/README.md): build a separate
  CPU provider, read an Xform or untyped prim's published local transform, and save CSV/JSON.
  **No ovrtx or GPU is required.** Use its separate minimal setup.
- [Python camera](python/minimal/README.md) and [C++ camera](c/minimal/README.md):
  save a 256 × 256 PNG using explicit provider selection.
- [Python steady state](python/steady-state/README.md): run a camera and lidar at
  different authored rates; save PNG, NumPy points and frame metadata.
- [C11 point cloud](c/point-cloud/README.md): save at most 4,096 valid XYZ/intensity
  rows and typed frame parameters as CSV/JSON.
- [Python tensor interop](python/tensor-interop/README.md): inspect the default
  representation, copy CPU data, and reduce linear CUDA data with Warp.

## Requirements

The renderer examples use **ovsensors 0.2, ovrtx 0.7, and ovstage 0.4**. Select a
compatible set for your OS and architecture; development builds from the same
minor line can still differ. They require an NVIDIA RTX GPU and a driver supported
by the selected ovrtx release, even when reading results on the CPU. First renderer
startup can take several minutes while shaders compile.

Python examples require Python 3.10–3.13 and [uv](https://docs.astral.sh/uv/).
[pyproject.toml](pyproject.toml) declares the compatible package lines plus NumPy
and Pillow. Only tensor interop requires the `tensor-interop` extra (Warp 1.10.1).
Native examples require CMake 3.20+ and C11/C++17 compilers; on Windows, use the
Visual Studio C++ tools. Choose platform-matching SDKs or wheels. Linux packages
require glibc 2.35+, and Linux x86-64 also requires AVX. Windows C/C++ and Python
consumers also need the [Microsoft C++ runtime prerequisite](../python/README.md#install)
for the prebuilt x64 SDKs and wheels.

## Python with supplied wheels

Obtain one ovsensors, one ovrtx and one ovstage wheel from a compatible release or
supplied build. Set `OVSENSORS_WHEEL`, `OVRTX_WHEEL`, and `OVSTAGE_WHEEL` to their
**absolute, original wheel filenames**, including their real version and platform
tags. Do not rename them. These variables are installation inputs, not runtime
library overrides.

On Linux:

```sh
uv venv --python 3.11 .venv
uv pip install --python .venv -r pyproject.toml --prerelease allow \
  "$OVSENSORS_WHEEL" "$OVRTX_WHEEL" "$OVSTAGE_WHEEL"
.venv/bin/python -c "import ovsensors; print(ovsensors.__file__); print(ovsensors.version())"
.venv/bin/python python/minimal/main.py
```

On Windows PowerShell, set the three corresponding `$env:` variables:

```powershell
uv venv --python 3.11 .venv
uv pip install --python .venv -r pyproject.toml --prerelease allow `
  $env:OVSENSORS_WHEEL $env:OVRTX_WHEEL $env:OVSTAGE_WHEEL
.venv\Scripts\python.exe -c "import ovsensors; print(ovsensors.__file__); print(ovsensors.version())"
.venv\Scripts\python.exe python/minimal/main.py
```

Explicit wheel inputs keep the supplied versions selected while checking the
unchanged project requirements. `--prerelease allow` permits development wheels
when those are supplied. This route does not require those versions to be on a
public index. Clear any SDK/source `PYTHONPATH` and native-library overrides before
using the wheel route; its ovsensors libraries come from the installed package.

When all three compatible releases are available from your package index, replace
the install command with:

```sh
uv pip install --python .venv -r pyproject.toml
```

Add `--extra tensor-interop` to either install command for the optional CUDA
example, or install `warp-lang==1.10.1` into the selected environment later.
Ordinary camera and steady-state runs do not import Warp.

The first run saves `_output/python.png`. Open it in any image viewer; a desktop
session is unnecessary. `--show` opts into desktop viewing. All examples accept
`--help`; their READMEs describe scene restrictions and output options.

## Native applications with SDKs

Extract ovsensors, ovstage and ovrtx SDKs into separate directories and keep their
layouts intact. Configure out of source with explicit package locations:

```sh
cmake -S c/minimal -B build/minimal -DCMAKE_BUILD_TYPE=Release \
  -Dovsensors_ROOT=/absolute/path/to/ovsensors-sdk \
  -Dovstage_ROOT=/absolute/path/to/ovstage-sdk \
  -Dovrtx_ROOT=/absolute/path/to/ovrtx-sdk
cmake --build build/minimal --config Release
./build/minimal/ovsensors_minimal --output _output/cpp.png
```

PowerShell uses a backtick for continuation. With a multi-config generator, run
`.\build\minimal\Release\ovsensors_minimal.exe`. To build the C11 example, use
`-S c/point-cloud -B build/point-cloud` with the same package selections, then run
`ovsensors_point_cloud` from that build directory. The [custom provider](c/custom-provider/README.md)
configures separately with only ovsensors and ovstage SDKs.

The renderer example helper uses the three `*_ROOT` options above. To use
dependency wheels instead, retain `ovsensors_ROOT`, omit both dependency roots,
and set `Python3_EXECUTABLE` to the absolute Python executable in the prepared
wheel environment. ovsensors still comes from its SDK; ovstage/ovrtx headers and
libraries then come from those wheels.

CMake uses installed packages; it does not download them. Renderer examples
request ovsensors `core` and `provider_ovrtx` components. Their applications link
ovstage for scene operations and ovrtx for schema registration; the ovrtx provider
receives the selected renderer loader path at runtime.

For your own provider target, request the `headers` component and link
`ovsensors::ovsensors_headers`. Every component needs an ovstage SDK for shared
ovx/DLPack headers; this adds no ovstage native link. Only `provider_ovrtx` also
requires ovrtx headers. See the [custom provider CMake project](c/custom-provider/CMakeLists.txt)
for the separate header-only provider and core/ovstage application dependencies.

## Python with SDK bindings

To exercise the ovsensors SDK instead of its wheel, create a separate environment.
Use the same supplied ovrtx and ovstage wheels, without installing an ovsensors wheel:

```sh
uv venv --python 3.11 .venv-sdk
uv pip install --python .venv-sdk --prerelease allow \
  "$OVRTX_WHEEL" "$OVSTAGE_WHEEL" 'numpy>=2.2,<3' 'pillow>=12.1,<13'
PYTHONPATH=/absolute/path/to/ovsensors-sdk/python \
  .venv-sdk/bin/python python/minimal/main.py
```

For subsequent example commands substitute `.venv-sdk/bin/python` and keep that
`PYTHONPATH` selection. On Windows, install with `$env:OVRTX_WHEEL` and
`$env:OVSTAGE_WHEEL`, set `$env:PYTHONPATH` to the SDK's `python` directory, and run
`.venv-sdk\Scripts\python.exe`. The SDK bindings find ovsensors native libraries
in its adjacent `bin/`. The selected dependency wheels supply their own runtimes.
Install `warp-lang==1.10.1` in this environment only when running tensor interop.

If an SDK moves, configure a fresh CMake build directory and update the explicit
Python path. Existing builds and virtual environments may retain absolute paths.
For package/source binding distinctions, see the [Python README](../python/README.md).
