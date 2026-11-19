# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Run camera and lidar sensors with independent cadences on one stage.

The loop distinguishes holds from failures and copies the latest camera image
while its output is mapped. Snippet markers are referenced by public/skills/.
"""

import argparse
import json
import os
import sys
import traceback
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import ovrtx
import ovsensors
import ovstage
from ovsensors.providers import ovrtx as ovrtx_provider
from PIL import Image

DEFAULT_SCENE = Path(__file__).resolve().parent / "scene.usda"
SENSORS = {
    "camera": "/Render/Products/CameraProduct",
    "lidar": "/Render/Products/LidarProduct",
}
OPEN_ORDINAL = 1
STEP_DELTA = 1.0 / 30.0
STEPS = 9


def main() -> None:
    parser = argparse.ArgumentParser(description="Run camera/lidar cadences; save PNG, point NPZ and frame JSON")
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE, help="USD scene to load")
    parser.add_argument("--steps", type=int, default=STEPS, help="Number of steps to run")
    parser.add_argument("--output", type=Path, default=Path("_output/steady-state.png"), help="Destination camera PNG")
    parser.add_argument("--points", type=Path, help="Lidar NPZ destination; defaults to the PNG stem plus .npz")
    parser.add_argument("--metadata", type=Path, help="JSON destination; defaults to the PNG stem plus .json")
    parser.add_argument("--show", action="store_true", help="Also open the image in the system viewer")
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps must be positive")
    if not args.scene.is_file():
        parser.error(f"scene does not exist: {args.scene}")

    args.points = args.points or args.output.with_suffix(".npz")
    args.metadata = args.metadata or args.output.with_suffix(".json")
    if len({path.resolve() for path in (args.output, args.points, args.metadata)}) != 3:
        parser.error("--output, --points and --metadata must name different files")

    # [snippet:register-schema-paths]
    # Register sensor schemas before the first ovstage call initializes USD.
    ovrtx.register_schema_paths()
    # [/snippet:register-schema-paths]
    # [snippet:build-stage]
    stage = ovstage.Stage("ovsensors.example.steady-state")
    inst = None
    try:
        ovstage.population.open_usd(
            stage, str(args.scene), ordinal=OPEN_ORDINAL, time=0.0, domains=ovstage.PopulationDomain.RENDERING
        )
        stage.advance_write_floor(OPEN_ORDINAL, ovstage.Scope.ALL).wait()
        # [/snippet:build-stage]
        print("Creating renderer; the first run compiles and caches shaders...", file=sys.stderr, flush=True)
        # This example's lidar pipeline needs motion BVH; ovrtx leaves it disabled by default.
        config = ovrtx.RendererConfig(motion_bvh="auto")
        with ovrtx_provider.provider(config) as provider, ovsensors.Instance([provider]) as inst:
            inst.attach(stage)
            with ExitStack() as cleanup:
                # [snippet:create-sensor-set]
                # Keep the same admitted sensors in each step to retain their cadence history.
                sensors = [cleanup.enter_context(inst.sensor(name, product)) for name, product in SENSORS.items()]
                descriptors = []
                for sensor in sensors:
                    descriptor = sensor.descriptor
                    print(f"Sensor {descriptor.name!r}: {descriptor.product} ({descriptor.rate.num}/{descriptor.rate.den} Hz)")
                    descriptors.append({"name": descriptor.name, "product": descriptor.product,
                                        "rate_hz": [descriptor.rate.num, descriptor.rate.den]})
                # [/snippet:create-sensor-set]

                totals = [0, 0, 0]
                last_image = last_points = None
                records = []
                for step in range(args.steps):
                    ordinal = OPEN_ORDINAL + 1 + step
                    stage.advance_write_floor(ordinal, ovstage.Scope.ALL).wait()
                    print(f"Step {step + 1}/{args.steps}:")
                    counts, samples = run_step(inst, sensors, ordinal)
                    totals = [a + b for a, b in zip(totals, counts)]
                    records.extend(samples["frames"])
                    if samples["image"] is not None:
                        last_image = samples["image"]
                    if samples["points"] is not None:
                        last_points = samples["points"]
                if last_image is None or last_points is None or not len(last_points["intensity"]) or totals[2]:
                    raise RuntimeError(f"Expected camera and nonempty lidar output without failed groups, got {totals}")
                print(f"{totals[0]} produced, {totals[1]} held, {totals[2]} failed")
            inst.detach()
    finally:
        destroy_stage(stage, inst)

    save_image(last_image, args.output, show=args.show)
    args.points.parent.mkdir(parents=True, exist_ok=True)
    with args.points.open("wb") as stream:
        np.savez_compressed(stream, **last_points)
    metadata = {"step_delta_seconds": STEP_DELTA, "steps": args.steps, "sensors": descriptors,
                "groups": dict(zip(("produced", "held", "failed"), totals)), "frames": records,
                "artifacts": {"camera": str(args.output), "lidar": str(args.points)},
                "points": {"valid_count": len(last_points["intensity"]), "coordinates_shape": list(last_points["coordinates"].shape),
                           "coordinates_dtype": str(last_points["coordinates"].dtype),
                           "intensity_dtype": str(last_points["intensity"].dtype)}}
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Saved {args.points.resolve()} and {args.metadata.resolve()}", flush=True)


# [snippet:classify-groups]
def run_step(inst: ovsensors.Instance, sensors: list, ordinal: int) -> tuple[tuple[int, int, int], dict]:
    """Keep delivered groups, output validity, and producer state separate."""
    op = inst.step(sensors, delta_time=STEP_DELTA, ordinal=ordinal)
    produced = held = failed = 0
    samples = {"image": None, "points": None, "frames": []}
    try:
        results = inst.fetch(op)
        if len(results) != len(sensors):
            raise RuntimeError(f"Expected {len(sensors)} sensor groups, received {len(results)}")
        for group in results:
            if group.failed:
                failed += 1
                print(f"  {group.name}: failed: {group.error_message}")
            elif group.held:
                held += 1
                print(f"  {group.name}: hold")
            else:
                produced += 1
                # [snippet:map-output]
                for frame, output in group.outputs():
                    if not output.valid:
                        raise RuntimeError(f"{group.name}: output {output.name!r} carries no valid sample")
                    with inst.mapped(output.handle, device=ovsensors.Device.CPU) as mapping:
                        require_cpu_ready(mapping)
                        print(f"  {group.name}/{output.name}: {summarise(mapping)}")
                        if "LdrColor" in mapping:
                            samples["image"] = mapping["LdrColor"].numpy(copy=True)
                        elif "Counts" in mapping:
                            samples["points"] = copy_points(mapping)
                        samples["frames"].append({
                            "sensor": group.name, "ordinal": ordinal, "output": output.name,
                            "window_seconds": [frame.start_time, frame.end_time],
                            "valid": output.valid, "producer_status": mapping.status.name,
                            "type": mapping.type, "version": mapping.version,
                            "params": copy_params(mapping),
                            **({"delivered_count": point_count(mapping),
                                "valid_count": len(samples["points"]["intensity"])} if "Counts" in mapping else {}),
                        })
                # [/snippet:map-output]
    finally:
        release_operation(inst, op)
    return (produced, held, failed), samples


# [/snippet:classify-groups]


# [snippet:summarise-tensors]
def point_count(mapping: ovsensors.Mapping) -> int:
    """Counts bounds the component-major Coordinates tensor for this one-source product."""
    counts = mapping["Counts"].numpy(copy=False)
    coordinates = mapping["Coordinates"]
    if counts.size != 1 or counts.dtype.kind not in "iu":
        raise RuntimeError(f"Expected one integer Counts value, got {counts.shape}/{counts.dtype}")
    delivered = int(counts.flat[0])
    if len(coordinates.shape) != 2 or coordinates.shape[0] != 3 or not 0 <= delivered <= coordinates.shape[1]:
        raise RuntimeError(f"Invalid point count {delivered} for coordinates{coordinates.shape}")
    return delivered


def summarise(mapping: ovsensors.Mapping) -> str:
    if "Counts" in mapping:
        return f"{point_count(mapping)} delivered points, coordinates{mapping['Coordinates'].shape}"
    return f"LdrColor {mapping['LdrColor'].shape}"


# [/snippet:summarise-tensors]


# [snippet:copy-points-and-params]
def copy_points(mapping: ovsensors.Mapping) -> dict:
    delivered = point_count(mapping)
    flags = mapping["Flags"].numpy(copy=False)
    intensity = mapping["Intensity"].numpy(copy=False)
    if flags.dtype != np.uint8 or flags.ndim != 1 or flags.size < delivered:
        raise RuntimeError(f"Flags {flags.dtype}{flags.shape} cannot describe {delivered} delivered points")
    if intensity.ndim != 1 or intensity.size < delivered:
        raise RuntimeError(f"Intensity shape {intensity.shape} cannot hold {delivered} delivered points")
    # Counts bounds delivery; Flags' VALID bit identifies usable ovrtx lidar returns.
    valid = (flags[:delivered] & 0x40) != 0
    # The provider stores XYZ as [3, capacity]; save convenient [point, XYZ] rows.
    coordinates = mapping["Coordinates"].numpy(copy=False)[:, :delivered].T[valid].copy()
    intensity = intensity[:delivered][valid].copy()
    if not np.isfinite(coordinates).all() or not np.isfinite(intensity).all():
        raise RuntimeError("Lidar returned nonfinite coordinates or intensity")
    return {"coordinates": coordinates, "intensity": intensity}


def copy_params(mapping: ovsensors.Mapping) -> dict:
    # Parameters have a separate namespace and remain on the CPU for every mapping device.
    params = {}
    for name, tensor in mapping.params.items():
        values = tensor.numpy(copy=True)
        params[name] = {"dtype": str(values.dtype), "shape": list(values.shape), "values": values.tolist()}
    return params


# [/snippet:copy-points-and-params]


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


def save_image(pixels: np.ndarray, output: Path, *, show: bool = False) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(pixels).convert("RGB")
    image.save(output, format="PNG")
    print(f"Saved {output.resolve()}", flush=True)
    if show:
        image.show()


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
