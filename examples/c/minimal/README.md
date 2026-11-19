# Save a camera image from C++

This C++17 application uses the ovsensors C API to render a local camera scene
and save a 256 × 256 RGB PNG. It maps `LdrColor` to CPU memory, copies the pixels,
and completes cleanup before writing the image. Rendering is headless.

Use matching ovsensors, ovstage and ovrtx SDKs, CMake 3.20+, a C++17 compiler,
and an NVIDIA RTX GPU with a supported driver. See the
[shared setup](../../README.md) for package selection and the dependency-wheel
alternative. No Kit checkout or Python environment is needed with explicit SDK roots.

From the public source `examples` directory, configure and run on Linux:

```bash
cmake -S c/minimal -B build/minimal -Dovsensors_ROOT="/path/to/ovsensors-sdk" -Dovstage_ROOT="/path/to/ovstage-sdk" -Dovrtx_ROOT="/path/to/ovrtx-sdk"
cmake --build build/minimal --config Release
./build/minimal/ovsensors_minimal --output "_output/camera image.png"
```

On Windows, run the same CMake commands from a Visual Studio developer shell,
using Windows SDK paths. With the Visual Studio generator, run:

```powershell
.\build\minimal\Release\ovsensors_minimal.exe --output "_output/camera image.png"
```

Success prints `Saved 256x256 PNG: ...`. Open that file to see the box and ground.
First startup can take several minutes while shaders compile. `--help` lists
options without initializing the runtime. Parent output directories are created.

[scene.usda](scene.usda) defines `/Render/Products/CameraProduct` and its
`LdrColor` output. `--scene PATH` accepts a replacement with those same names.
The application seals stage publications, explicitly selects the ovrtx provider,
and takes four steps of 1/30 s; change the loop constants when adapting cadence.
Failed groups, unusable pixels and a run without an image return a nonzero exit
status. On cleanup failure it retains retry handles and does not destroy a stage
still borrowed by ovsensors.

The source markers in [main.cpp](main.cpp) show schema registration before
Stage creation, sensor admission, result traversal and CPU mapping. Consult
[application flow](../../../skills/application-flow/SKILL.md) and
[result interpretation](../../../skills/interpreting-step-results/SKILL.md)
when adapting the lifecycle. The provider loads ovrtx privately; this application
also links its public loader for schema registration.

PNG encoding uses Sean Barrett's `stb_image_write.h`; its license is included
in that header.
