# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Python API for ovsensors.

ovsensors aggregates sensor output from one or more providers and hands it to an
application through a pull model: step, fetch, map, read, unmap, release.

With a provider, a stage, and a sealed publication ``ordinal``::

    import ovsensors

    with ovsensors.Instance([provider]) as inst:
        inst.attach(stage)
        with inst.sensor("camera", "/World/Render/Product") as camera:
            op = inst.step([camera], delta_time=1.0 / 60.0, ordinal=ordinal)
            with inst.results(op) as results:
                for group in results:
                    if group.held:
                        continue
                    for frame, output in group.outputs():
                        with inst.mapped(output.handle, device=ovsensors.Device.CPU) as mapping:
                            data = mapping["Coordinates"].numpy()

Instances, sensors, result sets, and mappings support context-managed lifetimes. Every
live instance holds its own library reference.

Provider and ovstage bundles remain opaque and are passed through without linking their
implementations. Providers are authored through the C interface.
"""

from ._src.bindings import (
    OVSENSORS_TIMEOUT_INFINITE,
    OVSENSORS_TIMEOUT_POLL,
    OVSENSORS_VERSION_MAJOR,
    OVSENSORS_VERSION_MINOR,
    OVSENSORS_VERSION_PATCH,
    library_path,
    ovsensors_provider_t,
)
from ._src.dlpack import DLDataType, DLDataTypeCode, DLDevice, DLDeviceType, DLTensor
from ._src.instance import (
    INFINITE,
    POLL,
    Instance,
    Mapping,
    Sensor,
    flush_log,
    initialize,
    last_error,
    set_log_callback,
    shutdown,
    version,
)
from ._src.types import (
    CudaSync,
    Descriptor,
    Device,
    EventStatus,
    Frame,
    LogSeverity,
    Output,
    OvsensorsError,
    Rational,
    ResultSet,
    SensorResult,
    Status,
    TensorView,
    TimeoutError,
    Validity,
)

# Packages carry their distribution version; source checkouts use the API base.
try:
    from ._version import version as __version__
except ModuleNotFoundError as _exc:
    if _exc.name != f"{__name__}._version":
        raise
    __version__ = f"{OVSENSORS_VERSION_MAJOR}.{OVSENSORS_VERSION_MINOR}.{OVSENSORS_VERSION_PATCH}"

__all__ = [
    "CudaSync",
    "DLDataType",
    "DLDataTypeCode",
    "DLDevice",
    "DLDeviceType",
    "DLTensor",
    "Descriptor",
    "Device",
    "EventStatus",
    "Frame",
    "INFINITE",
    "Instance",
    "LogSeverity",
    "Mapping",
    "OVSENSORS_TIMEOUT_INFINITE",
    "OVSENSORS_TIMEOUT_POLL",
    "OVSENSORS_VERSION_MAJOR",
    "OVSENSORS_VERSION_MINOR",
    "OVSENSORS_VERSION_PATCH",
    "Output",
    "OvsensorsError",
    "POLL",
    "Rational",
    "ResultSet",
    "Sensor",
    "SensorResult",
    "Status",
    "TensorView",
    "TimeoutError",
    "Validity",
    "__version__",
    "flush_log",
    "initialize",
    "last_error",
    "library_path",
    "ovsensors_provider_t",
    "set_log_callback",
    "shutdown",
    "version",
]
