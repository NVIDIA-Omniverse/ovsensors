# Capture camera and lidar streams

Create two sensors once, step at 30 Hz, and save the latest camera image and lidar
scan. Prepare the [shared Python environment](../../README.md) with matching
ovsensors, ovstage and ovrtx packages and an RTX-capable GPU. From `examples/`:

```sh
.venv/bin/python python/steady-state/main.py --output "_output/sensor run.png"
```

On Windows, use `.venv\Scripts\python.exe` in place of `.venv/bin/python`.

The command writes:

- `sensor run.png`: a 256×256 RGB view of the lit cube and ground plane.
- `sensor run.npz`: `coordinates` as `float32[N,3]` and `intensity` as `float32[N]`.
  Coordinates are Cartesian metres in the configured lidar frame; intensity is
  processed return strength. The local scene produces nonempty finite returns.
- `sensor run.json`: absolute frame windows in seconds, group totals, delivered
  and valid point counts, and named CPU parameters with their dtype, shape and values.
  Lidar parameters include timestamps, coordinate/frame convention, pose and capacity.
  The camera's parameter dictionary may be empty.

The [local scene](scene.usda) authors a 30 Hz camera and 10 Hz lidar with full
scans. Over the default nine steps, the ovrtx provider emits nine camera frames,
three lidar scans and six lidar holds. A hold is successful delivery of zero
frames. Frames have their own absolute windows: camera samples have zero width;
lidar scans span 0.1 seconds. Align data by these windows, not by loop index.

[main.py](main.py) uses automatic provider selection from applied schemas and
`RendererConfig(motion_bvh="auto")` for the lidar pipeline. It requests CPU
mappings, checks producer state, bounds delivered entries by `Counts[0]`, then
filters the ovrtx lidar `Flags` VALID bit (`0x40`). It transposes the provider's
component-major coordinates and copies the selected rows before unmap.
The NPZ contains only valid returns; the JSON keeps delivery and validity separate.

`--steps N` defaults to 9; allow at least three steps for this scene's first scan.
`--output PATH` defaults to `_output/steady-state.png`. `--points PATH` and
`--metadata PATH` override the adjacent NPZ/JSON paths. Output parents are created.
`--scene PATH` requires the same products/channels; `--show` opens the saved PNG
in a desktop viewer. `--help` creates no GPU context.

Failed groups, invalid outputs, failed/incomplete producer data, or a run without
both a camera image and nonempty lidar returns cause an error. Cleanup follows the
[minimal example's ownership rules](../minimal/README.md).
