# Export lidar returns from C

This C11 application renders the bundled lidar scene and writes a bounded CSV
of valid XYZ/intensity returns. It uses an explicit CPU mapping and copies the
values before unmapping. Rendering is headless; no viewer or CUDA toolkit is
needed by the application.

Use matching ovsensors, ovstage and ovrtx SDKs, CMake 3.20+, a C11 compiler,
and an NVIDIA RTX GPU with a supported driver. See the [shared setup](../../README.md)
for package selection and the dependency-wheel alternative. Explicit SDK roots
work without Python or a Kit checkout.

From the public source `examples` directory, configure and run on Linux:

```bash
cmake -S c/point-cloud -B build/point-cloud -Dovsensors_ROOT="/path/to/ovsensors-sdk" -Dovstage_ROOT="/path/to/ovstage-sdk" -Dovrtx_ROOT="/path/to/ovrtx-sdk"
cmake --build build/point-cloud --config Release
./build/point-cloud/ovsensors_point_cloud --output "_output/lidar points.csv"
```

On Windows, use a Visual Studio developer shell and Windows SDK paths for the
same CMake commands. With the Visual Studio generator, run:

```powershell
.\build\point-cloud\Release\ovsensors_point_cloud.exe --output "_output/lidar points.csv"
```

Success prints the number of rows and both artifact paths. Counts vary with the
renderer and scene. The default output is `_output/points.csv`; parent directories
are created. `--help` prints usage without initializing the runtime. First startup
can take several minutes while shaders compile.

The CSV has columns `x,y,z,intensity`, containing at most 4096 finite valid
returns from the latest nonempty frame across four 0.1 s steps. It uses the
component-major `Coordinates` float32 `[3,N]`, `Intensity` float32 `[N]`,
`Counts` int32 `[1]` and `Flags` uint8 `[N]` tensors. `Counts[0]` bounds the
delivered range within each allocation; the `Flags & 0x40` validity bit selects
returns. Only the copied rows are inspected; this bounded export is not a full
scan validator. No CPU-to-CUDA fallback occurs.

`<output>.json` records `rows`, `max_rows`, `delivered_count`, the absolute
`frame_start`/`frame_end` times in seconds, and CPU parameters `frame_id`,
`timestamp_ns`, `coords_type`, `frame_of_reference`, and
`model_to_app_transform` (16 values in row order). Parameters are read separately
from payload tensors. These retain the provider's coordinate convention rather
than silently applying a transform. The bundled scene requests Cartesian
coordinates in the sensor frame; XYZ is in meters and intensity is the model's
return intensity.

[point_cloud_scene.usda](point_cloud_scene.usda) contains only local geometry
and explicit nonvisual materials. `--scene PATH` must preserve
`/Render/Products/LidarProduct`, the `PointCloud` RenderVar and the requested
channels/Cartesian coordinates. The renderer configuration enables motion BVH
for lidar. Empty frames may occur while warming up; a run without valid returns,
failed groups, incompatible tensor layouts and failed artifact writes exit
nonzero. Cleanup failures retain handles for an exact retry and stop before
releasing a still-borrowed Stage.

Adapt [main.c](main.c), especially `point-cloud-setup` and `read-point-cloud`.
See [result interpretation](../../../skills/interpreting-step-results/SKILL.md)
for mapping lifetimes and the [public C API](../../../include/ovsensors/ovsensors.h)
for the authoritative cleanup contract.
