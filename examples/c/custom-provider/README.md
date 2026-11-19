# Build a CPU stage-transform provider

This example implements the complete ovsensors provider C ABI in a small C++17
shared library. Two applications, `main.cpp` and `main.py`, use that same library.
They move a source prim in ovstage, seal each publication, then save the provider's
observed transforms to `transforms.csv` and the output description to
`metadata.json`. It needs no renderer, GPU, CUDA runtime, or remote scene assets.

The provider reports a **parent-local transform**, not world pose or a simulated
physical sensor. `scene.usda` uses meters and an identity parent. Translation
occupies row 3 of a row-major matrix. The example supports exactly one source with
a published `omni:xform` matrix per RenderProduct and unauthored rate/start: one
sample on every accepted step. Other source cardinalities, missing or unsupported
matrices, and authored cadences are rejected. Admission checks the populated
matrix before the application writes its first update.

The shared scene has an `Xform` at `/World/Transform` and a genuinely untyped
`def "Plain" {}` at `/World/Plain`. Ordinary ovstage population preserves both
RenderProduct targets and supplies their local matrices. The Xform starts at
`(0, 2, 3)`; the plain prim starts with an identity local matrix. The plain prim
authors no transform operations or sensor schema. Subsequent movement comes from
the application's writes to the existing ovstage matrix column. The provider reads
that published data without checking the USD prim type.

## Build and run C++

Use CMake 3.20+, a C++17 compiler, an ovsensors 0.2 SDK, and a population-enabled
ovstage 0.4 SDK from the matching current source release. Linux and Windows/MSVC
build paths are supplied. Copy this whole
directory, including its local USDA and Windows loader template, or keep the
complete [examples tree](../../README.md). No Kit checkout is required.
Windows C++ and Python hosts also need the
[Microsoft C++ runtime prerequisite](../../../python/README.md#install)
for the prebuilt x64 SDKs and wheels.

From this directory, select the extracted SDKs explicitly:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -Dovsensors_DIR=/path/to/ovsensors/lib/cmake/ovsensors \
  -Dovstage_DIR=/path/to/ovstage/lib/cmake/ovstage
cmake --build build --config Release
./build/custom_provider --output "output/my transforms"
./build/custom_provider --render-product /World/PlainPose --output "output/plain transforms"
```

With a Windows multi-config generator, run
`build\Release\custom_provider.exe --output "output\my transforms"`.
The selected SDK DLLs load from their explicit configured paths; keep the sample
provider DLL beside the executable. Native Windows execution still requires
validation on that platform. Linux runtime paths refer to the SDKs selected at
configure time; keep those SDKs in place or reconfigure after moving them.

The application finds its staged `scene.usda` beside the executable, including
when invoked from another directory. `--scene` overrides it. Output paths are
relative to the invocation directory and missing parent directories are created.
`--samples` accepts 2 through 10000; `--help` initializes no native runtime.
`--render-product` defaults to `/World/Pose`, whose source is the Xform. Select
`/World/PlainPose` to exercise the untyped prim with the same provider. The host
uses the admitted descriptor's source path for writes and output metadata.

Only `core` is requested from the ovsensors CMake package. The
`sample_pose_provider` target consumes public headers and dispatches through the
supplied ovstage and path-dictionary vtables. It links neither ovsensors core nor
ovstage, ovrtx, or the shipped ovrtx provider. The `custom_provider` application
links core, ovstage, and this sample library.

## Run Python with the same provider library

Use Python 3.10–3.13, NumPy, the matching ovstage Python package, and the ovsensors
bindings from your selected SDK or wheel. A separate environment avoids installing
the renderer dependencies used by the other [examples](../../README.md):

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install 'ovstage==0.4.*' numpy
```

On Windows activate with `.venv\Scripts\activate`. Install
`'ovsensors==0.2.*'` for the wheel route, or use the SDK bindings below. If your
release packages are supplied as local wheels, pass those wheel filenames to pip
instead of resolving a version from an index. Select matching supported packages;
this example does not download native SDKs during configuration. ovrtx and GPU
tensor packages are unnecessary.

For SDK bindings, set `PYTHONPATH` to that SDK's `python` directory. For an
ovsensors wheel, install it in the selected environment and clear SDK
`PYTHONPATH` overrides. Then run:

```sh
python main.py --provider-library build/libsample_pose_provider.so \
  --output "output/python transforms"
python main.py --provider-library build/libsample_pose_provider.so \
  --render-product /World/PlainPose --output "output/python plain transforms"
```

On Windows use `--provider-library build\Release\sample_pose_provider.dll`.
The script resolves this path to an absolute filename and loads it with standard
`ctypes`. Its only provider binding is the publicly exported
`ovsensors.ovsensors_provider_t` and the two factory declarations in
[`pose_provider.h`](pose_provider.h). The `CDLL` object and bundle remain alive
until instance destruction succeeds. A bare bundle does not retain a native owner
for you; do not destroy the factory owner while any instance still borrows it.

## Expected output

For either RenderProduct, with six samples the CSV has six matrices with x translations
`0, 0.5, 1, 1.5, 2, 2.5`; y and z stay at `2, 3`. Publication ordinals are
`2..7`. The first reset establishes time 0. A second reset before sample 3
establishes time 10, so sample times are
`0.25, 0.5, 0.75, 10.25, 10.5, 10.75`.
`metadata.json` records `renderProduct` and its resolved `source`, so the two runs
remain distinguishable even though their application-authored motion is the same.

Each `LocalTransform` mapping has schema version 2 (signed publication ordinals), a contiguous CPU
`float64[4,4]` tensor named `matrix`, and three CPU scalar parameters:

- `publicationOrdinal`: `int64`, the application's sealed read bound.
- `simulationTime`: `float64`, absolute simulation seconds on the reset timeline.
- `sourceIndex`: `uint32`, index 0 into the descriptor's one-element `sources`.

`DEFAULT` and `CPU` produce the same completed host representation. Explicit
`CUDA`, `CUDA_ARRAY`, and nonzero synchronization streams return `NOT_SUPPORTED`.
There is no silent placement fallback. CPU reads must finish before unmap;
nonzero CUDA consumer dependencies are rejected without consuming the mapping.

## Provider ownership and extension points

[`pose_provider.cpp`](pose_provider.cpp) contains all 13 required slots and the
sample-local factory. Its claims have a diagnostic name and **zero schema IDs**.
Both applications select the bundle explicitly. Explicit selection requires no
matching schema claim; this provider still validates its required inputs. For
automatic selection and custom schema registration, see the
[schema registration guide](../../../skills/registering-sensor-schemas/SKILL.md).

Read the implementation in this order:

1. `attach` checks the ovstage API version and borrows the supplied stage, without
   reading authored data. `createSensor` validates the source matrix at its sealed
   admission ordinal and copies the source path.
2. `readTransform` reads `omni:xform` at the requested publication bound and copies
   the matrix before releasing every ovstage group, read, query, and path list.
   The application performs no later write before this synchronous read finishes;
   a publication bound does not provide historical payload storage.
3. `step` advances the provider's clock, including for an empty sensor set.
   Successful admission creates an operation. A per-sensor read failure becomes
   a failed group with diagnostic detail; it is never reported as an empty sample.
4. `Operation` owns result structure. `Mapping` owns mapped descriptors and shares
   an immutable `Sample`. Both applications release the operation before reading
   the still-live mapping. Mappings also survive sensor destruction.
5. `reset` requires operations and maps to be gone. `destroySensor` requires its
   operations to be released; `detach` requires all resources gone. The factory
   refuses destruction while attached or holding live resources. Retired sensor
   tokens remain allocated until provider destruction to prevent stale pointer
   reuse; this small example is not designed for unbounded sensor churn.

Calls on a context must be serialized and non-reentrant. The provider is
synchronous: polling can complete immediately, and it does not manufacture
timeouts. C++ exceptions are contained at its C boundaries. Factory and slot
failures preserve their documented ownership; inspect `get_last_error` for slot
detail when driving the vtable directly. Follow the ownership and permitted
return statuses in [`ovsensors_provider.h`](../../../include/ovsensors/ovsensors_provider.h)
when replacing the ovstage read with your own domain backend.

The applications release maps, operations, sensors, and their instance before
destroying the stage and provider owner. If instance cleanup fails, those owners
must stay alive. The Python executable terminates without finalizers in that rare
case because the selected ovstage wrapper otherwise destroys its borrowed native
stage during finalization.
