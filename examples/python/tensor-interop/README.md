# Consume a mapped image on CPU and CUDA

Inspect the provider's default camera representation, copy its CPU image, and
compute the mean RGB byte value on linear CUDA memory with Warp. The same local
scene supplies lidar CPU parameters so their separate namespace is visible.

Prepare the [shared Python environment](../../README.md) with matching ovsensors,
ovstage and ovrtx packages and an RTX-capable GPU. This example alone needs the
`tensor-interop` extra, qualified with **warp-lang 1.10.1**. To add it to an already
prepared SDK or supplied-wheel environment without changing those packages:

```sh
uv pip install --python .venv "warp-lang==1.10.1"
.venv/bin/python python/tensor-interop/main.py --output "_output/interop image.png"
```

On Windows, use `.venv\Scripts\python.exe` in place of `.venv/bin/python`.

The command saves a 256×256 PNG and adjacent JSON containing DEFAULT tensor
metadata, matching CPU/CUDA RGB means, absolute frame windows, and typed lidar
parameters. The mean is in byte units (0–255), excludes alpha and is computed
from the same camera output. It is a data-consumption example, not a performance
benchmark or a test of every possible mapping representation.

The executable recipe in [main.py](main.py) demonstrates:

- `inspect-default`: inspect DLPack device, shape, strides and type before choosing
  a consumer. DEFAULT is provider-selected; the camera may return an opaque CUDA
  array. Its handle is not a linear pointer. This example inspects it without
  reading it and then explicitly requests CPU and linear CUDA mappings.
- `cpu-dlpack`: `np.from_dlpack` borrows the mapped pixels; `.copy()` creates the
  image that survives unmap. Keeping a NumPy or Warp object does not extend the
  ovsensors mapping's borrow.
- `cuda-dlpack`: `wp.from_dlpack` receives the tensor object and negotiates producer
  ordering on Warp's stream. The [small reduction kernel](reduction.py) reads the
  borrowed image without modifying it. Device synchronization completes all
  consumer reads before unmap, including on a launch error. If synchronization
  itself fails, the standalone process exits without returning the live borrow.
- `cpu-params`: lidar parameters are copied from `mapping.params`, separately
  from payload tensors. Parameters are CPU resident for every mapping request;
  camera outputs in this provider have no parameters.

`--steps N` defaults to 3 and inspects the final step; use a multiple of 3 with
the shipped scene so both sensors emit. `--scene PATH` requires the same camera
and lidar products. `--output PATH` defaults to `_output/tensor-interop.png`;
`--metadata PATH` overrides the adjacent JSON. Parents are created. `--help`
works without Warp installed and creates no GPU context.

Mapping failure, invalid output, or disagreement between reductions is an error;
there is no device fallback. GPU execution has been exercised on Linux x86-64;
Windows and ARM runtime qualification are separate. Cleanup otherwise follows
[the minimal example](../minimal/README.md).
