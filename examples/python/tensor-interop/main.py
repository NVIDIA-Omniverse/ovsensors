# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Inspect DEFAULT, copy CPU data, and consume a linear CUDA image with Warp."""

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import ovrtx
import ovsensors
import ovstage
from ovsensors.providers import ovrtx as ovrtx_provider
from PIL import Image

DEFAULT_SCENE = Path(__file__).resolve().parent / "scene.usda"
PRODUCT = "/Render/Products/CameraProduct"
LIDAR_PRODUCT = "/Render/Products/LidarProduct"
OPEN_ORDINAL = 1
STEP_DELTA = 1.0 / 30.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE, help="Local USD camera scene")
    parser.add_argument("--steps", type=int, default=3, help="Warmup steps; last step must emit camera and lidar (default: 3)")
    parser.add_argument("--output", type=Path, default=Path("_output/tensor-interop.png"), help="Owned CPU image PNG")
    parser.add_argument("--metadata", type=Path, help="Result JSON; defaults to the PNG stem plus .json")
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps must be positive")
    if not args.scene.is_file():
        parser.error(f"scene does not exist: {args.scene}")
    args.metadata = args.metadata or args.output.with_suffix(".json")
    if args.output.resolve() == args.metadata.resolve():
        parser.error("--output and --metadata must name different files")

    # Import the optional consumer only after argument parsing; --help creates no GPU context.
    try:
        import warp as wp
        from reduction import sum_rgb
    except ImportError as error:
        raise RuntimeError("Install this example's tensor-interop extra; see its README") from error

    ovrtx.register_schema_paths()
    stage = ovstage.Stage("ovsensors.example.tensor-interop")
    inst = None
    try:
        ovstage.population.open_usd(
            stage, str(args.scene), ordinal=OPEN_ORDINAL, time=0.0, domains=ovstage.PopulationDomain.RENDERING
        )
        stage.advance_write_floor(OPEN_ORDINAL, ovstage.Scope.ALL).wait()
        with ovrtx_provider.provider(ovrtx.RendererConfig(motion_bvh="auto")) as provider, ovsensors.Instance([provider]) as inst:
            inst.attach(stage)
            with inst.sensor("camera", PRODUCT) as camera, inst.sensor("lidar", LIDAR_PRODUCT) as lidar:
                pixels = report = parameter_report = None
                for step in range(args.steps):
                    ordinal = OPEN_ORDINAL + 1 + step
                    stage.advance_write_floor(ordinal, ovstage.Scope.ALL).wait()
                    op = inst.step([camera, lidar], delta_time=STEP_DELTA, ordinal=ordinal)
                    try:
                        # A timeout retries this fetch's op; it must never enqueue another step.
                        results = inst.fetch(op)
                        for group in results:
                            if group.failed:
                                raise RuntimeError(f"{group.name}: {group.error_message}")
                            for frame, output in group.outputs():
                                if not output.valid:
                                    raise RuntimeError(f"Output {output.name!r} carries no valid sample")
                                if step == args.steps - 1 and group.name == "camera":
                                    pixels, report = inspect_output(inst, output.handle, wp, sum_rgb)
                                    report["window_seconds"] = [frame.start_time, frame.end_time]
                                elif step == args.steps - 1 and group.name == "lidar":
                                    parameter_report = read_lidar_params(inst, output.handle)
                                    parameter_report["window_seconds"] = [frame.start_time, frame.end_time]
                    finally:
                        release_operation(inst, op)
                if pixels is None or parameter_report is None:
                    raise RuntimeError("The last step must emit both sensors; use a multiple of 3 steps for this scene")
                report["lidar"] = parameter_report
            inst.detach()
    finally:
        destroy_stage(stage, inst)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels).convert("RGB").save(args.output, format="PNG")
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"RGB mean: CPU={report['cpu_mean_rgb']:.9f}, CUDA={report['cuda_mean_rgb']:.9f} (byte units)")
    print(f"Saved {args.output.resolve()} and {args.metadata.resolve()}", flush=True)


def inspect_output(inst, handle, wp, kernel):
    report = {"consumer": f"warp-lang {wp.__version__}"}
    # [snippet:inspect-default]
    with inst.mapped(handle, device=ovsensors.Device.DEFAULT) as mapping:
        if mapping.status == ovsensors.EventStatus.FAILURE:
            raise RuntimeError(f"DEFAULT producer failed: {mapping.error_message}")
        view = mapping["LdrColor"]
        dtype = view.dl_dtype
        opaque = dtype.code == ovsensors.DLDataTypeCode.kDLOpaqueHandle
        report["default"] = {
            "device": list(view.device), "shape": list(view.shape), "strides": list(view.strides),
            "dtype": {"code": dtype.code, "bits": dtype.bits, "lanes": dtype.lanes},
            "representation": "opaque CUDA array" if opaque else "linear tensor",
            "producer_status": mapping.status.name,
        }
        print(f"DEFAULT: {report['default']}")
        # Metadata inspection requires no payload read. An opaque CUDA-array handle
        # is never a linear pointer and cannot be passed to NumPy/DLPack here.
    # [/snippet:inspect-default]

    # [snippet:cpu-dlpack]
    with inst.mapped(handle, device=ovsensors.Device.CPU) as mapping:
        require_cpu_ready(mapping)
        borrowed = np.from_dlpack(mapping["LdrColor"])
        pixels = borrowed.copy()  # Only this owned copy survives unmap.
        del borrowed
        if pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[2] != 4:
            raise RuntimeError(f"Expected uint8[height,width,4], got {pixels.dtype}{pixels.shape}")
        report["cpu_mean_rgb"] = float(pixels[:, :, :3].mean(dtype=np.float64))
        report["shape"] = list(pixels.shape)
        report["camera_params"] = {
            name: {"dtype": tensor.dtype, "shape": list(tensor.shape), "values": tensor.numpy(copy=True).tolist()}
            for name, tensor in mapping.params.items()
        }
    # [/snippet:cpu-dlpack]

    # [snippet:cuda-dlpack]
    with inst.mapped(handle, device=ovsensors.Device.CUDA) as mapping:
        if mapping.status == ovsensors.EventStatus.FAILURE:
            raise RuntimeError(f"CUDA producer failed: {mapping.error_message}")
        view = mapping["LdrColor"]
        if view.device[0] != ovsensors.DLDeviceType.kDLCUDA or view.dtype != "uint8" or view.shape != pixels.shape:
            raise RuntimeError("CUDA mapping did not provide the requested linear uint8 image")
        # Params remain CPU resident even though this image is on CUDA.
        report["camera_cuda_param_devices"] = {name: list(tensor.device) for name, tensor in mapping.params.items()}
        device = wp.get_device(f"cuda:{view.device[1]}")
        with wp.ScopedDevice(device):
            try:
                # Pass the object, not a raw capsule: Warp requests producer ordering
                # on its consumer stream through TensorView.__dlpack__(stream=...).
                borrowed = wp.from_dlpack(view)
                total = wp.zeros(1, dtype=wp.uint64, device=device)
                wp.launch(kernel, dim=pixels.shape[:2], inputs=[borrowed, total], device=device)
            finally:
                # Complete every queued read, including on a launch error, before unmap.
                try:
                    wp.synchronize_device(device)
                except BaseException:
                    # A failed fence cannot establish that returning this borrow is safe.
                    try:
                        print("Fatal: CUDA completion failed; exiting with the mapping retained.", file=sys.stderr)
                        traceback.print_exc()
                        sys.stdout.flush()
                        sys.stderr.flush()
                    finally:
                        os._exit(1)
            report["cuda_mean_rgb"] = int(total.numpy()[0]) / (pixels.shape[0] * pixels.shape[1] * 3)
            del borrowed
    # [/snippet:cuda-dlpack]
    if not np.isclose(report["cpu_mean_rgb"], report["cuda_mean_rgb"], rtol=0, atol=1e-12):
        raise RuntimeError("CPU and CUDA reductions of the same output disagree")
    return pixels, report


# [snippet:cpu-params]
def read_lidar_params(inst, handle):
    # Camera output has no parameters in this provider. Lidar supplies timestamps,
    # pose, coordinate convention and capacity as named CPU tensors.
    with inst.mapped(handle, device=ovsensors.Device.CPU) as mapping:
        require_cpu_ready(mapping)
        params = {}
        for name, tensor in mapping.params.items():
            if tensor.device[0] != ovsensors.DLDeviceType.kDLCPU:
                raise RuntimeError(f"Parameter {name!r} was not CPU resident")
            values = tensor.numpy(copy=True)
            params[name] = {"dtype": str(values.dtype), "shape": list(values.shape), "values": values.tolist()}
        if not params:
            raise RuntimeError("Expected lidar parameters for the authored PointCloud output")
        return {"params": params}
# [/snippet:cpu-params]


def release_operation(inst: ovsensors.Instance, op: int) -> None:
    """Called once in the step's finally, including when fetch fails before entry."""
    primary = sys.exc_info()[1]
    try:
        inst.release(op)
    except BaseException as cleanup:
        if primary is None:
            raise
        message = f"Operation {op} cleanup also failed: {cleanup}"
        if hasattr(primary, "add_note"):
            primary.add_note(message)
        else:
            print(message, file=sys.stderr)


def require_cpu_ready(mapping: ovsensors.Mapping) -> None:
    """A successful map call does not itself promise successful, ready producer data."""
    if mapping.status != ovsensors.EventStatus.COMPLETED:
        raise RuntimeError(f"Output producer is {mapping.status.name}: {mapping.error_message}")
    if mapping.cuda_sync.stream or mapping.cuda_sync.wait_event:
        raise RuntimeError("CPU mapping still carries a CUDA dependency; refusing an unsynchronized read")


def destroy_stage(stage: ovstage.Stage, inst: ovsensors.Instance | None) -> None:
    """Destroy only after detach/destroy succeeded, or before attach succeeded."""
    if inst is not None and inst.attached:
        # Stage.__del__ destroys unconditionally in the pinned ovstage release.
        # A still-borrowed stage therefore requires process exit without finalizers.
        try:
            print("Fatal: sensor cleanup left the stage borrowed; exiting without finalizers.", file=sys.stderr)
            traceback.print_exc()
        finally:
            try:
                sys.stdout.flush()
                sys.stderr.flush()
            finally:
                os._exit(1)
    stage.destroy()


if __name__ == "__main__":
    main()
