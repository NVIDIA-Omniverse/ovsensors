# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Lazy CUDA runtime bindings for synchronizing borrowed DLPack tensors."""

import ctypes
import operator
import sys
import threading
from pathlib import Path
from typing import Iterable, Optional

from . import bindings as _b

_CUDART_MIN_MAJOR = 12
_CUDART_MAX_MAJOR = 20
_prefix, _suffix = ("cudart64_", ".dll") if sys.platform.startswith("win") else ("libcudart.so.", "")
_LIB_CANDIDATES = tuple(f"{_prefix}{v}{_suffix}" for v in reversed(range(_CUDART_MIN_MAJOR, _CUDART_MAX_MAJOR + 1)))
if not sys.platform.startswith("win"):
    _LIB_CANDIDATES = ("libcudart.so", *_LIB_CANDIDATES)

_POINTER_MAX = (1 << (ctypes.sizeof(ctypes.c_void_p) * 8)) - 1
_lock = threading.Lock()
_lib: Optional[ctypes.CDLL] = None


def _packaged_candidates() -> Iterable[Path]:
    """Find runtimes beside the already loaded ovsensors library."""
    core_path = _b.library_path()
    if core_path is None:
        return
    bin_dir = core_path.parent
    for directory in (bin_dir / "plugins" / "gpu.foundation", bin_dir):
        for name in _LIB_CANDIDATES:
            try:
                candidate = (directory / name).resolve(strict=True)
            except OSError:
                continue
            if candidate.is_file():
                yield candidate


def _load() -> ctypes.CDLL:
    global _lib
    if _lib is not None:
        return _lib
    with _lock:
        if _lib is None:
            # Keep messages only: an exception would retain its caller's traceback.
            last_err: Optional[str] = None
            candidates = (*_packaged_candidates(), *_LIB_CANDIDATES)
            for candidate in candidates:
                try:
                    lib = ctypes.CDLL(str(candidate))
                    lib.cudaStreamWaitEvent.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint]
                    lib.cudaStreamWaitEvent.restype = ctypes.c_int
                    lib.cudaStreamSynchronize.argtypes = [ctypes.c_void_p]
                    lib.cudaStreamSynchronize.restype = ctypes.c_int
                    lib.cudaGetDevice.argtypes = [ctypes.POINTER(ctypes.c_int)]
                    lib.cudaGetDevice.restype = ctypes.c_int
                    lib.cudaGetErrorString.argtypes = [ctypes.c_int]
                    lib.cudaGetErrorString.restype = ctypes.c_char_p
                except (OSError, AttributeError) as err:
                    last_err = str(err)
                    continue
                _lib = lib  # Publish only after declaring every prototype.
                break
            else:
                raise BufferError(
                    "Automatic CUDA DLPack synchronization is unsupported: could not load the CUDA runtime. "
                    "Use stream=-1 only if the caller honors Mapping.cuda_sync before reading the tensor. "
                    f"Tried {candidates}: {last_err}"
                )
    return _lib


def _check(lib: ctypes.CDLL, rc: int, fn: str) -> None:
    if rc != 0:
        detail = lib.cudaGetErrorString(rc)
        message = detail.decode(errors="replace") if detail else "unknown CUDA error"
        raise RuntimeError(f"{fn} failed (rc={rc}): {message}")


def _normalize_stream(stream: Optional[int]) -> int:
    if stream is None:
        return 1  # CUDA's explicit legacy stream sentinel, never the ambiguous 0.
    if isinstance(stream, bool):
        raise TypeError("stream must be an integer, not bool")
    try:
        stream = operator.index(stream)
    except TypeError:
        raise TypeError("stream must be an integer or None") from None
    if stream == 0 or stream < -1 or stream > _POINTER_MAX:
        raise ValueError("stream must be -1, 1 (legacy), 2 (per-thread), or a positive pointer-sized integer")
    return stream


def synchronize(device_id: int, producer_stream: int, wait_event: int, consumer_stream: Optional[int]) -> None:
    """Honor both producer dependencies, or bypass synchronization with ``-1``.

    Default producer streams and default-stream operations require the caller to
    select the tensor's CUDA context and device first. Automatic synchronization
    rejects producer stream ``2`` because the original producer thread cannot be
    identified. Consumer stream ``2`` is supported. This helper never changes the
    current device or context.
    """
    consumer_stream = _normalize_stream(consumer_stream)
    if consumer_stream == -1:
        return
    if producer_stream == 2:
        raise BufferError(
            "The per-thread CUDA producer stream cannot be synchronized automatically. "
            "The provider must expose an event or explicit stream instead, or the caller must "
            "honor Mapping.cuda_sync and export with stream=-1."
        )
    block_producer = producer_stream != 0 and producer_stream != consumer_stream
    check_device = producer_stream == 1 or (wait_event != 0 and consumer_stream in (1, 2))
    if not block_producer and not wait_event and not check_device:
        return

    lib = _load()
    if check_device:
        current_device = ctypes.c_int()
        _check(lib, lib.cudaGetDevice(ctypes.byref(current_device)), "cudaGetDevice")
        if current_device.value != device_id:
            raise BufferError(
                f"CUDA default-stream synchronization requires tensor device {device_id}, "
                f"but the current CUDA runtime device is {current_device.value}. "
                f"Select CUDA device {device_id} and the tensor's CUDA context before exporting this tensor; "
                "ovsensors does not change the current device or context."
            )

    # An event need not cover all producer-stream work; honor each dependency.
    if block_producer:
        _check(lib, lib.cudaStreamSynchronize(producer_stream), "cudaStreamSynchronize")
    if wait_event:
        _check(lib, lib.cudaStreamWaitEvent(consumer_stream, wait_event, 0), "cudaStreamWaitEvent")
