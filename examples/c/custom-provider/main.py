# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

"""Consume the same native pose-provider DSO as main.cpp, without a renderer."""

import argparse
import csv
import ctypes
import json
import os
from pathlib import Path
import sys
import traceback


def cleanup(action):
    """Keep the original error visible if a release also fails."""
    primary = sys.exc_info()[1]
    try:
        action()
    except Exception as error:
        if primary is None:
            raise
        print(f"Cleanup also failed: {error}", file=sys.stderr)


def sample_transform(instance, sensor, ordinal):
    import ovsensors

    operation = instance.step([sensor], delta_time=0.25, ordinal=ordinal)
    try:
        results = instance.fetch(operation)
        if len(results) != 1:
            raise RuntimeError("Expected one sensor group")
        group = results[0]
        if group.failed:
            raise RuntimeError(group.error_message)
        if len(group.frames) != 1 or len(group.frames[0].outputs) != 1:
            raise RuntimeError("Expected one transform output")
        output = group.frames[0].outputs[0]
        if not output.valid:
            raise RuntimeError("Transform output is invalid")
        mapping = instance.map_output(output.handle, device=ovsensors.Device.CPU)
        try:
            if mapping.status != ovsensors.EventStatus.COMPLETED:
                raise RuntimeError(f"Transform producer failed: {mapping.error_message}")
            # [snippet:python-independent-map]
            instance.release(operation)
            operation = None
            matrix = mapping["matrix"].numpy(copy=True)
            expected = {
                "publicationOrdinal": (ovsensors.DLDataTypeCode.kDLInt, 64, 1),
                "simulationTime": (ovsensors.DLDataTypeCode.kDLFloat, 64, 1),
                "sourceIndex": (ovsensors.DLDataTypeCode.kDLUInt, 32, 1),
            }
            parameters = mapping.params
            if parameters.keys() != expected.keys():
                raise RuntimeError("Unexpected transform parameters")
            params = {}
            for name, dtype in expected.items():
                value = parameters[name]
                actual = value.dl_dtype
                if (value.device != (ovsensors.DLDeviceType.kDLCPU, 0) or value.shape != () or
                        (actual.code, actual.bits, actual.lanes) != dtype):
                    raise RuntimeError(f"Unexpected CPU scalar parameter: {name}")
                params[name] = value.numpy(copy=True).item()
            # The NumPy copies remain usable after unmap; output handles do not.
            # [/snippet:python-independent-map]
            return matrix, params
        finally:
            cleanup(mapping.unmap)
    finally:
        # fetch can fail before a results context manager enters its body.
        if operation is not None:
            cleanup(lambda: instance.release(operation))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider-library", type=Path, required=True, help="Built sample provider .so or .dll")
    parser.add_argument("--scene", type=Path, default=Path(__file__).with_name("scene.usda"))
    parser.add_argument("--output", type=Path, default=Path("output/custom-provider-python"))
    parser.add_argument("--samples", type=int, default=6)
    parser.add_argument("--render-product", default="/World/Pose", help="RenderProduct whose source to sample")
    args = parser.parse_args()
    if not 2 <= args.samples <= 10000:
        parser.error("--samples must be 2..10000")
    for path in (args.provider_library, args.scene):
        if not path.is_file():
            parser.error(f"file does not exist: {path}")

    import numpy as np
    import ovsensors
    import ovstage

    # [snippet:python-provider-factory]
    library = ctypes.CDLL(str(args.provider_library.resolve()))
    library.pose_provider_create.argtypes = [ctypes.POINTER(ovsensors.ovsensors_provider_t)]
    library.pose_provider_create.restype = ctypes.c_int
    library.pose_provider_destroy.argtypes = [ctypes.POINTER(ovsensors.ovsensors_provider_t)]
    library.pose_provider_destroy.restype = ctypes.c_int
    bundle = ovsensors.ovsensors_provider_t()
    status = library.pose_provider_create(ctypes.byref(bundle))
    if status != ovsensors.Status.SUCCESS:
        raise RuntimeError(f"Provider creation failed: {status}")
    # Both library and bundle stay in this scope until the instance has been destroyed.
    # [/snippet:python-provider-factory]
    stage = instance = None
    destroyed = False
    try:
        stage = ovstage.Stage("example.custom-provider.python")
        ovstage.population.open_usd(stage, str(args.scene.resolve()), ordinal=1, time=0,
                                   domains=ovstage.PopulationDomain.RENDERING)
        stage.advance_write_floor(1).wait()
        instance = ovsensors.Instance([bundle])
        instance.attach(stage)
        with ovstage.PathDictionary(stage) as paths:
            path_list = None
            query = None
            try:
                xform = paths.intern_token("omni:xform")
                # Admission reads the populated matrix before the application can write one.
                with instance.sensor("pose", args.render_product, provider=bundle) as sensor:
                    descriptor = sensor.descriptor
                    if len(descriptor.sources) != 1 or descriptor.schema:
                        raise RuntimeError("Expected one explicitly selected source")
                    source = descriptor.sources[0]
                    path_list = paths.create_path_list_from_strings([source])
                    query = stage.query_from_path_list(path_list)
                    instance.reset(0)
                    args.output.mkdir(parents=True, exist_ok=True)
                    with (args.output / "transforms.csv").open("w", newline="", encoding="utf-8") as stream:
                        writer = csv.writer(stream)
                        writer.writerow(["publicationOrdinal", "simulationTime", "sourceIndex"] +
                                        [f"m{row}{col}" for row in range(4) for col in range(4)])
                        for index in range(args.samples):
                            if index == args.samples // 2:
                                instance.reset(10)
                            ordinal = index + 2
                            matrix = np.eye(4, dtype=np.float64)
                            matrix[3, :3] = [index * 0.5, 2, 3]
                            tensor = ovstage.make_dltensor(
                                matrix.ravel(), dtype=ovstage.DLDataType(code=ovstage.DLDataTypeCode.kDLFloat,
                                                                       bits=64, lanes=16), shape=[1])
                            stage.write_attribute(query, xform, ordinal=ordinal, tensors=tensor,
                                                  is_array=False, semantic=ovstage.AttributeSemantic.MATRIX).wait()
                            stage.advance_write_floor(ordinal).wait()
                            observed, params = sample_transform(instance, sensor, ordinal)
                            if not np.array_equal(observed, matrix) or params["publicationOrdinal"] != ordinal:
                                raise RuntimeError("Published transform differs from the observed transform")
                            writer.writerow([params["publicationOrdinal"], params["simulationTime"],
                                             params["sourceIndex"], *observed.ravel().tolist()])
            finally:
                try:
                    if query is not None:
                        cleanup(lambda: stage.release_query(query).wait())
                finally:
                    if path_list is not None:
                        cleanup(lambda: paths.destroy_path_list(path_list))
        metadata = {
            "output": "LocalTransform", "version": 2,
            "tensor": {"dtype": "float64", "shape": [4, 4], "layout": "row-major"},
            "space": "parent-local", "translationUnits": "stage units (meters in scene.usda)",
            "source": source, "providerSelection": "explicit", "renderProduct": args.render_product,
            "params": {"publicationOrdinal": "int64", "simulationTime": "float64 seconds", "sourceIndex": "uint32"},
            "resetOrigin": 10, "resetBeforeSample": args.samples // 2, "sampleCount": args.samples,
        }
        (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    finally:
        try:
            if instance is not None:
                instance.destroy()
            destroyed = True
        finally:
            if not destroyed:
                # Stage finalization could otherwise destroy an instance still borrowed by core.
                print("Fatal: instance cleanup failed; preserving the stage and provider until process exit.", file=sys.stderr)
                traceback.print_exc()
                sys.stdout.flush()
                sys.stderr.flush()
                os._exit(1)
            try:
                if stage is not None:
                    cleanup(stage.destroy)
            finally:
                status = library.pose_provider_destroy(ctypes.byref(bundle))
                if status != ovsensors.Status.SUCCESS:
                    raise RuntimeError(f"Provider destruction failed: {status}")
    print(f"Saved {args.samples} transforms to {args.output.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(error, file=sys.stderr)
        raise SystemExit(1)
