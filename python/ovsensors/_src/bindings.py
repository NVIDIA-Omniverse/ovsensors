# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Raw ctypes declarations for the application-facing ovsensors C API.

Declarations mirror the public headers. Provider vtables remain a C-side concern;
Python carries only their opaque ``ovsensors_provider_t`` bundles.
"""

import ctypes
import operator
import os
import sys
import threading
from pathlib import Path
from typing import Any, Iterable, List, Optional, Tuple

from .dlpack import DLTensor

# ---------------------------------------------------------------------------------
# Scalars and sentinels, from ovsensors_types.h
# ---------------------------------------------------------------------------------

ovsensors_op_id_t = ctypes.c_uint64
ovsensors_output_handle_t = ctypes.c_uint64
ovsensors_map_handle_t = ctypes.c_uint64
ovsensors_timeout_ns_t = ctypes.c_uint64
ovstage_ordinal_t = ctypes.c_int64
OVSTAGE_ORDINAL_MIN = -(1 << 63)
OVSTAGE_ORDINAL_MAX = (1 << 63) - 1

OVSENSORS_INVALID_OP_ID = 0
OVSENSORS_INVALID_OUTPUT_HANDLE = 0
OVSENSORS_INVALID_MAP_HANDLE = 0

OVSENSORS_TIMEOUT_POLL = 0
OVSENSORS_TIMEOUT_INFINITE = 0xFFFFFFFFFFFFFFFF

OVSENSORS_VERSION_MAJOR = 0
OVSENSORS_VERSION_MINOR = 2
OVSENSORS_VERSION_PATCH = 0

# Status codes. Values are pinned by the header; a test asserts they still match.
OVSENSORS_API_SUCCESS = 0
OVSENSORS_API_ERROR = 1
OVSENSORS_API_TIMEOUT = 2
OVSENSORS_API_INVALID_ARGUMENT = 3
OVSENSORS_API_INVALID_HANDLE = 4
OVSENSORS_API_BUSY = 5
OVSENSORS_API_NOT_SUPPORTED = 6
OVSENSORS_API_OUT_OF_MEMORY = 7
OVSENSORS_API_OP_FAILED = 8
OVSENSORS_API_RESET_REQUIRED = 9
OVSENSORS_API_NOT_ATTACHED = 10
OVSENSORS_API_INTERNAL = 99

OVSENSORS_LOG_VERBOSE = -2
OVSENSORS_LOG_INFO = -1
OVSENSORS_LOG_WARNING = 0
OVSENSORS_LOG_ERROR = 1
OVSENSORS_LOG_NONE = 3

# Values match ovrtx_map_device_type_t exactly, so the two cannot silently mean different things.
OVSENSORS_MAP_DEVICE_TYPE_DEFAULT = 0
OVSENSORS_MAP_DEVICE_TYPE_CPU = 1
OVSENSORS_MAP_DEVICE_TYPE_CUDA = 2
OVSENSORS_MAP_DEVICE_TYPE_CUDA_ARRAY = 3

OVSENSORS_EVENT_PENDING = 0
OVSENSORS_EVENT_COMPLETED = 1
OVSENSORS_EVENT_FAILURE = 2

OVSENSORS_OUTPUT_INVALID = 0
OVSENSORS_OUTPUT_VALID = 1


# ---------------------------------------------------------------------------------
# Argument guards for the fixed-width C inputs
# ---------------------------------------------------------------------------------

# Caller-supplied ids, handles, and timeouts are uint64_t; ordinals are int64_t.
_UINT64_MAX = 0xFFFFFFFFFFFFFFFF


def _check_uint64(value: Any, name: str, remedy: str = "") -> int:
    """Validate one caller integer against the ``uint64_t`` field it is about to fill.

    ctypes wraps out-of-range integers. Reject booleans before ``operator.index`` and
    validate the range before crossing the C boundary. Invalid Python values raise
    builtin exceptions because the library has not been called.
    """
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an int, not bool")
    try:
        result = operator.index(value)
    except TypeError:
        raise TypeError(f"{name} must be an int, got {type(value).__name__}") from None
    if result < 0:
        raise ValueError(f"{name} must be non-negative, got {value}{remedy}")
    if result > _UINT64_MAX:
        raise ValueError(f"{name} must fit in uint64 (<= {_UINT64_MAX}), got {value}{remedy}")
    return result


def check_timeout(timeout_ns: Any) -> int:
    """Validate a nanosecond timeout without allowing uint64 wrapping."""
    return _check_uint64(timeout_ns, "timeout_ns", "; pass ovsensors.INFINITE to block")


def check_ordinal(ordinal: Any) -> int:
    """Validate a signed OVStage ordinal before ctypes conversion."""
    if isinstance(ordinal, bool):
        raise TypeError("ordinal must be an int, not bool")
    result = operator.index(ordinal)
    if not OVSTAGE_ORDINAL_MIN <= result <= OVSTAGE_ORDINAL_MAX:
        raise ValueError(f"ordinal must fit in int64, got {ordinal}")
    return result


def check_handle(handle: Any, name: str) -> int:
    """Validate an opaque uint64 id while preserving its argument name in errors."""
    return _check_uint64(handle, name)


def check_sync_stream(stream: Any) -> int:
    """Validate integer storage only; the provider owns CUDA stream semantics."""
    value = _check_uint64(stream, "sync_stream")
    if value > (1 << (ctypes.sizeof(ctypes.c_size_t) * 8)) - 1:
        raise ValueError("sync_stream must fit in uintptr_t")
    return value


class ovx_string_t(ctypes.Structure):
    """Borrowed string view: a pointer and a byte length, from ovx/string_types.h.

    ``{NULL, 0}`` decodes as ``""``. ``ptr`` is a character pointer rather than
    ``c_char_p`` so ctypes cannot truncate at an embedded NUL before ``length`` is used.
    """

    _fields_ = [
        ("ptr", ctypes.POINTER(ctypes.c_char)),
        ("length", ctypes.c_size_t),
    ]

    def __init__(self, value: Optional[str] = None) -> None:
        """Build a view over an owned UTF-8 buffer, kept alive by this object."""
        super().__init__()
        encoded = str(value if value is not None else "").encode("utf-8")
        self._buffer = ctypes.create_string_buffer(encoded)
        self.ptr = ctypes.cast(self._buffer, ctypes.POINTER(ctypes.c_char))
        self.length = len(encoded)

    def text(self) -> str:
        """Decode to str. A NULL or zero-length view reads as the empty string."""
        if not self.ptr or self.length == 0:
            return ""
        return ctypes.string_at(self.ptr, int(self.length)).decode("utf-8", errors="replace")

    def __bool__(self) -> bool:
        return bool(self.ptr) and int(self.length) > 0

    def __str__(self) -> str:
        return self.text()


# ---------------------------------------------------------------------------------
# Opaque handles
# ---------------------------------------------------------------------------------


class ovsensors_instance_t(ctypes.Structure):
    """Opaque. Only ever held as a pointer."""


class ovsensors_sensor_t(ctypes.Structure):
    """Opaque. Only ever held as a pointer."""


class ovstage_instance_t(ctypes.Structure):
    """Opaque ovstage bundle, passed through without linking or dereferencing it."""


ovstage_instance_p = ctypes.POINTER(ovstage_instance_t)


def as_stage_pointer(stage: Any) -> Tuple[ovstage_instance_p, Any]:
    """Return the stage pointer and the original owner that must outlive attachment.

    Accepts an ovstage wrapper, any ctypes pointer or ``c_void_p``, or an integer address.
    Foreign ctypes pointers are cast because ctypes type identity is per declaration.
    A bare address has no Python owner to retain, and pointer liveness cannot be checked.
    """
    owner = stage
    for _ in range(4):  # bounded: a wrapper around a wrapper is fine, a cycle is not
        if isinstance(stage, ovstage_instance_p):
            return stage, owner
        if isinstance(stage, (ctypes._Pointer, ctypes.c_void_p)):
            return ctypes.cast(stage, ovstage_instance_p), owner
        if isinstance(stage, int) and not isinstance(stage, bool):
            return ctypes.cast(ctypes.c_void_p(stage), ovstage_instance_p), owner
        inner = getattr(stage, "_inst", None)
        if inner is None:
            inner = getattr(stage, "_as_parameter_", None)
        if inner is None:
            break
        stage = inner

    raise TypeError(
        f"expected an ovstage handle, got {type(owner).__name__}. Pass an ovstage.Stage, "
        f"a ctypes pointer to its instance, or its address as an int."
    )


class ovsensors_config_entry_t(ctypes.Structure):
    """Opaque and incomplete; no settings exist yet."""


class ovsensors_provider_t(ctypes.Structure):
    """A provider bundle: an immutable vtable and its private context, both borrowed.

    This is the only provider-side Python binding. Both pointers remain opaque; provider
    vtable authoring is a C-side concern.
    """

    _fields_ = [
        ("vtable", ctypes.c_void_p),
        ("context", ctypes.c_void_p),
    ]


def as_provider(provider: Any) -> Tuple[ovsensors_provider_t, Any]:
    """Return a provider bundle and the original owner that must outlive the instance.

    Accepts a wrapper exposing ``bundle`` or ``_as_parameter_``, or a bare bundle. The
    caller retains responsibility for the C-side lifetime represented by a bare bundle.
    """
    owner = provider
    for _ in range(4):  # bounded: a wrapper around a wrapper is fine, a cycle is not
        if isinstance(provider, ovsensors_provider_t):
            return provider, owner
        inner = getattr(provider, "bundle", None)
        if inner is None:
            inner = getattr(provider, "_as_parameter_", None)
        if inner is None:
            break
        provider = inner

    raise TypeError(
        f"expected a provider, got {type(owner).__name__}. Pass the object a provider "
        f"factory returned, or its ovsensors_provider_t bundle."
    )


# ---------------------------------------------------------------------------------
# Value types
# ---------------------------------------------------------------------------------


class ovsensors_result_t(ctypes.Structure):
    """An API status result."""

    _fields_ = [("status", ctypes.c_int)]


class ovsensors_enqueue_result_t(ctypes.Structure):
    """A status plus the operation id it minted.

    ``op_index`` is OVSENSORS_INVALID_OP_ID whenever ``status`` is not SUCCESS -- core
    enforces that pairing, so Python may rely on it.
    """

    _fields_ = [
        ("status", ctypes.c_int),
        ("op_index", ovsensors_op_id_t),
    ]


class ovsensors_rational_t(ctypes.Structure):
    """An authored cadence. ``0/0`` means unauthored -- the provider picks."""

    _fields_ = [
        ("num", ctypes.c_int64),
        ("den", ctypes.c_int64),
    ]


class ovsensors_cuda_sync_t(ctypes.Structure):
    """A CUDA stream and event, as opaque integers.

    On a mapping this is the producer-ready dependency to honor before reading payload;
    passed to unmap it is the consumer-done dependency.
    A CPU mapping reports ``{0, 0}``.
    """

    _fields_ = [
        ("stream", ctypes.c_void_p),
        ("wait_event", ctypes.c_void_p),
    ]


class ovsensors_map_output_description_t(ctypes.Structure):
    """Mapping options forwarded to the provider without device or stream probing."""

    _fields_ = [
        ("device_type", ctypes.c_int),
        ("sync_stream", ctypes.c_size_t),
    ]


class ovsensors_config_t(ctypes.Structure):
    """Initialization config; only ``entry_count == 0`` is supported, and ``entries`` is ignored."""

    _fields_ = [
        ("entries", ctypes.POINTER(ovsensors_config_entry_t)),
        ("entry_count", ctypes.c_size_t),
    ]


class ovsensors_sensor_spec_t(ctypes.Structure):
    """A name, adopted RenderProduct, and optional authoritative provider selection."""

    _fields_ = [
        ("name", ovx_string_t),
        ("product", ovx_string_t),
        ("provider", ctypes.POINTER(ovsensors_provider_t)),
    ]


class ovsensors_descriptor_t(ctypes.Structure):
    """Everything resolution determined about a sensor. Owned by the instance."""

    _fields_ = [
        ("name", ovx_string_t),
        ("product", ovx_string_t),
        ("sources", ctypes.POINTER(ovx_string_t)),
        ("source_count", ctypes.c_size_t),
        ("schema", ovx_string_t),
        ("provider_name", ovx_string_t),
        ("rate", ovsensors_rational_t),
        ("start", ovsensors_rational_t),
    ]


class ovsensors_tensor_t(ctypes.Structure):
    """One named tensor in a mapping. The ``dl`` borrow ends at a SUCCESS or INVALID_HANDLE unmap."""

    _fields_ = [
        ("name", ovx_string_t),
        ("dl", ctypes.POINTER(DLTensor)),
        ("doc", ovx_string_t),
    ]


class ovsensors_param_t(ctypes.Structure):
    """One named CPU-resident parameter, borrowing storage until unmap."""

    _fields_ = [
        ("dl", DLTensor),
        ("name", ovx_string_t),
        ("doc", ovx_string_t),
    ]


class ovsensors_output_t(ctypes.Structure):
    """One named output of a frame. ``handle`` identifies payload for ``map_output``."""

    _fields_ = [
        ("name", ovx_string_t),
        ("validity", ctypes.c_int),
        ("handle", ovsensors_output_handle_t),
    ]


class ovsensors_mapped_t(ctypes.Structure):
    """A successful mapping: borrowed tensors, the sync to honour, and the map handle."""

    _fields_ = [
        ("tensors", ctypes.POINTER(ovsensors_tensor_t)),
        ("tensor_count", ctypes.c_size_t),
        ("cuda_sync", ovsensors_cuda_sync_t),
        ("map", ovsensors_map_handle_t),
        ("status", ctypes.c_int),
        ("error_message", ovx_string_t),
        ("name", ovx_string_t),
        ("type", ovx_string_t),
        ("doc", ovx_string_t),
        ("version", ctypes.c_int),
        ("params", ctypes.POINTER(ovsensors_param_t)),
        ("param_count", ctypes.c_size_t),
    ]


class ovsensors_frame_t(ctypes.Structure):
    """One emitted frame's absolute time window and outputs."""

    _fields_ = [
        ("frame_start_time", ctypes.c_double),
        ("frame_end_time", ctypes.c_double),
        ("outputs", ctypes.POINTER(ovsensors_output_t)),
        ("output_count", ctypes.c_size_t),
    ]


class ovsensors_sensor_result_t(ctypes.Structure):
    """One sensor's part of a step result.

    SUCCESS with zero frames is an ordinary cadence hold; it is never how failure is
    reported.
    """

    _fields_ = [
        ("descriptor", ctypes.POINTER(ovsensors_descriptor_t)),
        ("status", ctypes.c_int),
        ("error_message", ovx_string_t),
        ("frames", ctypes.POINTER(ovsensors_frame_t)),
        ("frame_count", ctypes.c_size_t),
    ]


class ovsensors_step_result_set_t(ctypes.Structure):
    """The aggregate result for every sensor admitted to one step."""

    _fields_ = [
        ("groups", ctypes.POINTER(ovsensors_sensor_result_t)),
        ("group_count", ctypes.c_size_t),
    ]


ovsensors_log_callback_t = ctypes.CFUNCTYPE(
    None, ctypes.c_int, ctypes.c_double, ovx_string_t, ovx_string_t, ctypes.c_void_p
)

# ---------------------------------------------------------------------------------
# Library loading
# ---------------------------------------------------------------------------------

OVSENSORS_LIBRARY_PATH_HINT: Optional[str] = None

_LIBRARY_STEM = "ovsensors"


def _library_filename() -> str:
    if sys.platform == "win32":
        return f"{_LIBRARY_STEM}.dll"
    if sys.platform == "darwin":
        return f"lib{_LIBRARY_STEM}.dylib"
    return f"lib{_LIBRARY_STEM}.so"


def _existing_dirs(paths: Iterable[Path]) -> List[Path]:
    """Return unique existing directories in first-seen order, skipping access errors."""
    out: List[Path] = []
    seen = set()
    for candidate in paths:
        try:
            if not (candidate.exists() and candidate.is_dir()):
                continue
            resolved = candidate.resolve()
        except OSError:
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        out.append(resolved)
    return out


def _search_directories(extra: Optional[Iterable[Path]] = None) -> List[Path]:
    """Ordered search path: explicit hints first, then the environment, then the package."""
    candidates: List[Path] = []
    if extra:
        candidates.extend(Path(p) for p in extra)
    if OVSENSORS_LIBRARY_PATH_HINT:
        candidates.append(Path(OVSENSORS_LIBRARY_PATH_HINT))
    env = os.environ.get("OVSENSORS_LIBRARY_PATH")
    if env:
        candidates.extend(Path(p) for p in env.split(os.pathsep) if p)
    here = Path(__file__).resolve()
    candidates.append(here.parent)
    candidates.append(here.parents[1])
    # Wheel layout: ovsensors/bin/ beside the Python modules.
    candidates.append(here.parents[1] / "bin")
    # Installed layout: <prefix>/python/ovsensors/_src/ and <prefix>/bin/.
    if len(here.parents) > 3:
        candidates.append(here.parents[3] / "bin")
    return _existing_dirs(candidates)


class _LibraryLoader:
    """Load libovsensors lazily and publish it after version and ABI declaration.

    A lock serializes first use. Failed validation or prototype declaration leaves no
    partially configured CDLL reachable.
    """

    def __init__(self) -> None:
        self._lib: Optional[ctypes.CDLL] = None
        self._lock = threading.Lock()

    @property
    def lib(self) -> ctypes.CDLL:
        """The loaded library, loading and declaring prototypes on first use."""
        published = self._lib
        if published is not None:
            return published
        with self._lock:
            if self._lib is None:
                loaded = self._load()
                _check_library_version(loaded)  # before the other 19 are bound to it
                _declare_prototypes(loaded)
                self._lib = loaded  # last, so a half-configured CDLL is never reachable
            return self._lib

    def path(self) -> Optional[Path]:
        """Known loaded path, or None if unloaded or resolved only by the OS loader."""
        loaded = self._lib
        if loaded is None:
            return None
        path = Path(loaded._name)
        return path.resolve() if path.is_absolute() else None

    def _load(self) -> ctypes.CDLL:
        filename = _library_filename()
        tried: List[str] = []
        for directory in _search_directories():
            candidate = directory / filename
            if candidate.exists():
                return ctypes.CDLL(str(candidate))
            tried.append(str(candidate))
        try:
            return ctypes.CDLL(filename)
        except OSError as exc:
            hint = "\n  ".join(tried) if tried else "(no candidate directories existed)"
            raise OSError(
                f"could not load {filename}. Set OVSENSORS_LIBRARY_PATH or "
                f"ovsensors._src.bindings.OVSENSORS_LIBRARY_PATH_HINT.\nTried:\n  {hint}\n{exc}"
            ) from exc


def _check_library_version(lib: ctypes.CDLL) -> None:
    """Refuse a loaded library whose major.minor differs from these bindings'.

    Only ``ovsensors_get_version`` is declared before this check. Patch versions are
    accepted as ABI-compatible. A mismatch raises ``RuntimeError`` because the version
    query itself succeeded and returned no API error status.
    """
    lib.ovsensors_get_version.argtypes = [
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint32),
    ]
    lib.ovsensors_get_version.restype = None

    major, minor, patch = ctypes.c_uint32(), ctypes.c_uint32(), ctypes.c_uint32()
    lib.ovsensors_get_version(ctypes.byref(major), ctypes.byref(minor), ctypes.byref(patch))

    if (major.value, minor.value) != (OVSENSORS_VERSION_MAJOR, OVSENSORS_VERSION_MINOR):
        bindings_version = f"{OVSENSORS_VERSION_MAJOR}.{OVSENSORS_VERSION_MINOR}.{OVSENSORS_VERSION_PATCH}"
        library_version = f"{major.value}.{minor.value}.{patch.value}"
        raise RuntimeError(
            f"ovsensors version mismatch: Python bindings are {bindings_version} but "
            f"{lib._name} is {library_version}. Major and minor must agree; patch need not."
        )


def _declare_prototypes(lib: ctypes.CDLL) -> None:
    """Declare argtypes and restype for all 20 exported entry points.

    Declared in header order so a diff against ``ovsensors.h`` reads straight down.
    """
    instance_p = ctypes.POINTER(ovsensors_instance_t)
    sensor_p = ctypes.POINTER(ovsensors_sensor_t)

    # Keep this function as the complete prototype inventory; redeclaration is idempotent.
    lib.ovsensors_get_version.argtypes = [
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint32),
    ]
    lib.ovsensors_get_version.restype = None

    lib.ovsensors_get_last_error.argtypes = []
    lib.ovsensors_get_last_error.restype = ovx_string_t

    lib.ovsensors_get_last_op_error.argtypes = [instance_p, ovsensors_op_id_t]
    lib.ovsensors_get_last_op_error.restype = ovx_string_t

    lib.ovsensors_set_log_callback.argtypes = [ctypes.c_int, ovsensors_log_callback_t, ctypes.c_void_p]
    lib.ovsensors_set_log_callback.restype = ovsensors_result_t

    lib.ovsensors_flush_log.argtypes = [ovsensors_timeout_ns_t]
    lib.ovsensors_flush_log.restype = ovsensors_result_t

    lib.ovsensors_initialize.argtypes = [ctypes.POINTER(ovsensors_config_t)]
    lib.ovsensors_initialize.restype = ovsensors_result_t

    lib.ovsensors_shutdown.argtypes = []
    lib.ovsensors_shutdown.restype = ovsensors_result_t

    lib.ovsensors_create_instance.argtypes = [
        ctypes.POINTER(ovsensors_provider_t),
        ctypes.c_size_t,
        ctypes.POINTER(instance_p),
    ]
    lib.ovsensors_create_instance.restype = ovsensors_result_t

    lib.ovsensors_destroy_instance.argtypes = [instance_p]
    lib.ovsensors_destroy_instance.restype = ovsensors_result_t

    lib.ovsensors_attach_ovstage.argtypes = [instance_p, ctypes.POINTER(ovstage_instance_t)]
    lib.ovsensors_attach_ovstage.restype = ovsensors_result_t

    lib.ovsensors_detach_ovstage.argtypes = [instance_p]
    lib.ovsensors_detach_ovstage.restype = ovsensors_result_t

    lib.ovsensors_create_sensor.argtypes = [
        instance_p,
        ctypes.POINTER(ovsensors_sensor_spec_t),
        ctypes.POINTER(sensor_p),
    ]
    lib.ovsensors_create_sensor.restype = ovsensors_result_t

    lib.ovsensors_destroy_sensor.argtypes = [sensor_p]
    lib.ovsensors_destroy_sensor.restype = ovsensors_result_t

    lib.ovsensors_get_descriptor.argtypes = [sensor_p, ctypes.POINTER(ctypes.POINTER(ovsensors_descriptor_t))]
    lib.ovsensors_get_descriptor.restype = ovsensors_result_t

    lib.ovsensors_step.argtypes = [
        instance_p,
        ctypes.POINTER(sensor_p),
        ctypes.c_size_t,
        ctypes.c_double,
        ovstage_ordinal_t,
    ]
    lib.ovsensors_step.restype = ovsensors_enqueue_result_t

    lib.ovsensors_fetch_result_set.argtypes = [
        instance_p,
        ovsensors_op_id_t,
        ovsensors_timeout_ns_t,
        ctypes.POINTER(ovsensors_step_result_set_t),
    ]
    lib.ovsensors_fetch_result_set.restype = ovsensors_result_t

    lib.ovsensors_map_output.argtypes = [
        instance_p,
        ovsensors_output_handle_t,
        ctypes.POINTER(ovsensors_map_output_description_t),
        ovsensors_timeout_ns_t,
        ctypes.POINTER(ovsensors_mapped_t),
    ]
    lib.ovsensors_map_output.restype = ovsensors_result_t

    lib.ovsensors_unmap_output.argtypes = [instance_p, ovsensors_map_handle_t, ovsensors_cuda_sync_t]
    lib.ovsensors_unmap_output.restype = ovsensors_result_t

    lib.ovsensors_release_result_set.argtypes = [instance_p, ovsensors_op_id_t]
    lib.ovsensors_release_result_set.restype = ovsensors_result_t

    lib.ovsensors_reset.argtypes = [instance_p, ctypes.c_double, ovsensors_timeout_ns_t]
    lib.ovsensors_reset.restype = ovsensors_result_t


_loader = _LibraryLoader()


def library() -> ctypes.CDLL:
    """The loaded ovsensors shared library, with every prototype declared."""
    return _loader.lib


def library_path() -> Optional[Path]:
    """Absolute loaded-library path, or None if unloaded or the OS-selected path is unknown."""
    return _loader.path()
