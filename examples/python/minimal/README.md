# Render a camera image

Render the local lit cube and ground plane to a 256×256 PNG. Prepare the
[shared Python environment](../../README.md) with an ovsensors SDK or wheel,
matching ovstage/ovrtx packages, and an RTX-capable GPU. From `examples/`:

```sh
.venv/bin/python python/minimal/main.py --output "_output/first image.png"
```

On Windows, use `.venv\Scripts\python.exe` in place of `.venv/bin/python`.

Open `_output/first image.png`. Rendering is headless; `--show` additionally
opens a desktop viewer. The first renderer startup can take several minutes.

[main.py](main.py) registers schemas before creating the stage, seals each
publication, selects the ovrtx provider explicitly, and steps the camera at its
[authored 30 Hz cadence](scene.usda). `LdrColor` is mapped explicitly to CPU as
`uint8[256,256,4]` and copied before unmap. The saved PNG keeps RGB channels.
This example demonstrates CPU consumption; see [tensor interop](../tensor-interop/README.md)
for linear CUDA consumption and provider-selected representations.

`--output PATH` defaults to `_output/python.png`; missing parent directories are
created. `--steps N` defaults to 4 and must be positive. `--scene PATH` accepts a
local scene with the same camera product and output. `--help` creates no GPU context.

Cadence holds are allowed; failed groups, invalid outputs, incomplete producer
work, or a run without an image raise an error. Every accepted step is released,
including when fetching fails. A fetch timeout belongs to that same operation;
a retry must fetch that operation again rather than advance simulation again.
This script uses the default infinite fetch budget.

If cleanup leaves the stage borrowed, the standalone script reports the failure
and exits without Python finalizers. This protects the stage owner until the
process ends; ordinary successful cleanup destroys the stage after detach.
