# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
"""Inspect installed metadata only: never import ovrtx/ovstage or load a library."""

import importlib.metadata
from pathlib import Path
import sys


def cmake_value(name, value):
    value = str(value).replace("\\", "/")
    if any(token in value for token in (";", "\n", '"', "]==]")):
        raise ValueError(f"unsupported character in {name}: {value!r}")
    return f"set({name} [==[{value}]==])\n"


def discover():
    if sys.prefix == sys.base_prefix:
        raise RuntimeError("Python3_EXECUTABLE must select the examples virtual environment")
    venv = Path(sys.prefix).resolve(strict=True)
    sdk = Path(sys.argv[1]).resolve(strict=True)
    config = sdk / "lib/cmake/ovsensors/ovsensorsConfig.cmake"
    if not config.is_file():
        raise RuntimeError(f"{sdk} is not a built ovsensors SDK (missing {config})")
    values = {"OVSENSORS_EXAMPLE_SDK": sdk}
    for package in ("ovrtx", "ovstage"):
        dist = importlib.metadata.distribution(package)
        prefix = Path(dist.locate_file(package)).resolve(strict=True)
        if not prefix.is_relative_to(venv):
            raise RuntimeError(f"{package} resolved outside the selected virtual environment: {prefix}")
        library_name = f"{package}-dynamic.dll" if sys.platform == "win32" else f"lib{package}-dynamic.so"
        required = [
            prefix / f"include/{package}/{package}.h",
            prefix / "bin" / library_name,
            prefix / f"lib/cmake/{package}/{package}Config.cmake",
        ]
        if package == "ovrtx":
            required.extend(prefix / name for name in ("include/ovx/types.h", "include/ovx/dlpack/dlpack.h"))
        for path in required:
            if not path.is_file():
                raise RuntimeError(f"{package} {dist.version} wheel is missing {path}; no source-tree fallback is used")
        key = package.upper()
        values[f"{key}_WHEEL_PREFIX"] = prefix
        values[f"{key}_WHEEL_VERSION"] = dist.version
        values[f"{key}_WHEEL_LIBRARY"] = (prefix / "bin" / library_name).resolve(strict=True)
    Path(sys.argv[2]).write_text("".join(cmake_value(k, v) for k, v in values.items()), encoding="utf-8")


if __name__ == "__main__":
    try:
        discover()
    except (OSError, ValueError, importlib.metadata.PackageNotFoundError, RuntimeError) as exc:
        sys.exit(str(exc))
