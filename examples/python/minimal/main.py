# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Render an authored camera with ovsensors and save its latest image.

Snippet markers are referenced by public/skills/; keep them intact.
"""

import argparse
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
SENSOR_NAME = "camera"
OPEN_ORDINAL = 1
STEP_DELTA = 1.0 / 30.0  # Match the authored camera cadence.
STEPS = 4


def main() -> None:
    parser = argparse.ArgumentParser(description="Render an ovsensors camera to PNG")
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE, help="USD scene to load")
    parser.add_argument("--steps", type=int, default=STEPS, help="Number of steps to run")
    parser.add_argument("--output", type=Path, default=Path("_output/python.png"), help="Destination PNG image")
    parser.add_argument("--show", action="store_true", help="Also open the image in the system viewer")
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps must be positive")
    if not args.scene.is_file():
        parser.error(f"scene does not exist: {args.scene}")

    try:
        if args.output.is_dir():
            parser.error(f"output is a directory: {args.output}")
        for parent in args.output.parents:
            if parent.exists() or parent.is_symlink():
                if not parent.is_dir():
                    parser.error(f"output parent is not a directory: {parent}")
                break
    except OSError as error:
        parser.error(f"cannot inspect output '{args.output}': {error}")

    # [snippet:register-schema-paths]
    # Register before the first ovstage call initializes USD.
    ovrtx.register_schema_paths()
    # [/snippet:register-schema-paths]

    stage = ovstage.Stage("ovsensors.example.minimal")
    inst = None
    try:
        # [snippet:build-stage]
        ovstage.population.open_usd(
            stage, str(args.scene), ordinal=OPEN_ORDINAL, time=0.0, domains=ovstage.PopulationDomain.RENDERING
        )
        stage.advance_write_floor(OPEN_ORDINAL, ovstage.Scope.ALL).wait()
        # [/snippet:build-stage]

        print("Creating renderer; the first run compiles and caches shaders...", file=sys.stderr, flush=True)
        # [snippet:provider-and-instance]
        # The provider owns its renderer and must outlive its ovsensors instance.
        with ovrtx_provider.provider() as provider, ovsensors.Instance([provider]) as inst:
            inst.attach(stage)
            with inst.sensor(SENSOR_NAME, PRODUCT, provider=provider) as sensor:
                last_image = None
                for step in range(args.steps):
                    ordinal = OPEN_ORDINAL + 1 + step
                    stage.advance_write_floor(ordinal, ovstage.Scope.ALL).wait()
                    image = run_one_step(inst, sensor, ordinal)
                    if image is not None:
                        last_image = image
                if last_image is None:
                    raise RuntimeError(f"No usable image produced over {args.steps} steps")
            inst.detach()
        # [/snippet:provider-and-instance]
    finally:
        destroy_stage(stage, inst)

    try:
        save_image(last_image, args.output, show=args.show)
    except OSError as error:
        raise SystemExit(f"Image output error for '{args.output}': {error}") from None


# [snippet:step-and-fetch]
def run_one_step(inst: ovsensors.Instance, sensor: ovsensors.Sensor, ordinal: int) -> np.ndarray | None:
    """Fetch results, handling sensor failure, cadence holds and invalid output."""
    op = inst.step([sensor], delta_time=STEP_DELTA, ordinal=ordinal)
    image = None
    try:
        results = inst.fetch(op)
        for group in results:
            if group.failed:
                raise RuntimeError(f"Sensor {group.name!r} failed: {group.error_message}")
            if group.held:
                continue  # A successful step with no frame due.
            for _frame, output in group.outputs():
                if not output.valid:
                    raise RuntimeError(f"Output {output.name!r} carries no valid sample")
                # Copy while mapped so the image remains valid after unmap.
                with inst.mapped(output.handle, device=ovsensors.Device.CPU) as mapping:
                    require_cpu_ready(mapping)
                    image = mapping["LdrColor"].numpy(copy=True)
    finally:
        release_operation(inst, op)
    return image


# [/snippet:step-and-fetch]


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
    """Save PNG without a desktop dependency; open a viewer only on request."""
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
