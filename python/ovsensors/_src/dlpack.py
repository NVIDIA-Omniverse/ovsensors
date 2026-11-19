# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""ctypes declarations and capsule export for ``ovx/dlpack/dlpack.h``.

Mapped ``DLTensor`` values borrow provider memory until unmap. Exported managed tensors
copy their descriptors but continue to borrow payload. Their storage uses the raw Python
allocator so the C deleter can run without executing Python.
"""

import ctypes
from typing import Any, Tuple

DLPACK_MAJOR_VERSION = 1
DLPACK_MINOR_VERSION = 3

# Versioned capsules borrow const provider memory and therefore advertise read-only.
DLPACK_FLAG_BITMASK_READ_ONLY = 1 << 0

# Prevent zero-width elements and implausibly large wrapped lane counts before exposing
# a view over provider storage.
_MAX_DLPACK_LANES = 255

# DLPack consumers rename a capsule to its ``used_`` form when they take ownership.
_c_str_dltensor = b"dltensor"
_c_str_used_dltensor = b"used_dltensor"
_c_str_dltensor_versioned = b"dltensor_versioned"
_c_str_used_dltensor_versioned = b"used_dltensor_versioned"


class DLDeviceType:
    """DLPack device codes used by these bindings."""

    kDLCPU = 1
    kDLCUDA = 2
    kDLCUDAHost = 3
    kDLCUDAManaged = 13


class DLDataTypeCode:
    """Element type codes."""

    kDLInt = 0
    kDLUInt = 1
    kDLFloat = 2
    kDLOpaqueHandle = 3
    kDLBfloat = 4
    kDLComplex = 5
    kDLBool = 6


class DLDevice(ctypes.Structure):
    """Which device the payload lives on.

    See the `DLPack DLDevice fields
    <https://github.com/dmlc/dlpack/blob/v1.3/include/dlpack/dlpack.h#L125-L136>`_.
    """

    _fields_ = [
        ("device_type", ctypes.c_int),
        ("device_id", ctypes.c_int32),
    ]


class DLDataType(ctypes.Structure):
    """Element type: a code, a width in bits, and a lane count.

    See the `DLPack DLDataType fields
    <https://github.com/dmlc/dlpack/blob/v1.3/include/dlpack/dlpack.h#L184-L215>`_.
    """

    _fields_ = [
        ("code", ctypes.c_uint8),
        ("bits", ctypes.c_uint8),
        ("lanes", ctypes.c_uint16),
    ]

    def lane_count(self, what: str = "tensor") -> int:
        """Return this element's lane count after validating ``[1, 255]``."""
        lanes = int(self.lanes)
        if not 1 <= lanes <= _MAX_DLPACK_LANES:
            raise ValueError(
                f"{what}: DLDataType lanes must be in [1, {_MAX_DLPACK_LANES}], got {lanes} "
                f"(code={int(self.code)} bits={int(self.bits)})"
            )
        return lanes

    def lane_dtype_str(self, what: str = "tensor") -> str:
        """Build a NumPy dtype name for one lane.

        Vector width is represented by the folded shape. NumPy validates whether it supports
        the resulting scalar width when constructing the dtype.
        """
        self.lane_count(what)
        code = int(self.code)
        bits = int(self.bits)
        if code == DLDataTypeCode.kDLBool and bits == 8:
            return "bool"
        prefix = {
            DLDataTypeCode.kDLInt: "int",
            DLDataTypeCode.kDLUInt: "uint",
            DLDataTypeCode.kDLFloat: "float",
        }.get(code)
        if prefix is None or bits % 8 != 0:
            raise ValueError(f"DLPack code={code} bits={bits} has no numpy equivalent")
        return f"{prefix}{bits}"

    def numpy_dtype_str(self) -> str:
        """Build a NumPy dtype name for a scalar element; reject vector elements."""
        if self.lanes != 1:
            raise ValueError(f"vector lanes={self.lanes} has no scalar numpy dtype")
        return self.lane_dtype_str()


class DLTensor(ctypes.Structure):
    """A borrowed view of payload. Does not own its memory.

    See the `DLPack DLTensor fields
    <https://github.com/dmlc/dlpack/blob/v1.3/include/dlpack/dlpack.h#L217-L278>`_.
    """

    _fields_ = [
        ("data", ctypes.c_void_p),
        ("device", DLDevice),
        ("ndim", ctypes.c_int32),
        ("dtype", DLDataType),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)),
        ("byte_offset", ctypes.c_uint64),
    ]

    @property
    def shape_tuple(self) -> Tuple[int, ...]:
        """Shape as a tuple. Empty for a 0-dimensional tensor."""
        return tuple(int(self.shape[i]) for i in range(int(self.ndim)))

    @property
    def strides_tuple(self) -> Tuple[int, ...]:
        """Strides in elements, or an empty tuple when the tensor is compact.

        DLPack allows a NULL strides pointer to mean row-major compact, so an empty
        tuple here means compact rather than zero-dimensional -- check ``ndim`` to
        tell those apart.
        """
        if not self.strides:
            return ()
        return tuple(int(self.strides[i]) for i in range(int(self.ndim)))

    @property
    def is_compact(self) -> bool:
        """True when DLPack's NULL-strides convention says the payload is row-major compact."""
        return not self.strides

    def lane_folded_layout(self, what: str = "tensor") -> Tuple[Tuple[int, ...], Tuple[int, ...], int]:
        """Return ``(shape, strides, lanes)`` with vector lanes as a trailing axis.

        * shape:   ``shape + (lanes,)``
        * strides: ``stride * lanes`` for each source axis, followed by ``1``

        Empty strides retain DLPack's row-major-compact convention. Scalar layouts are
        returned unchanged.
        """
        lanes = self.dtype.lane_count(what)
        shape = self.shape_tuple
        strides = self.strides_tuple
        if lanes == 1:
            return shape, strides, 1
        folded_strides = tuple(stride * lanes for stride in strides) + (1,) if strides else ()
        return shape + (lanes,), folded_strides, lanes


class DLPackVersion(ctypes.Structure):
    """The version a DLPack 1.x versioned capsule carries."""

    _fields_ = [
        ("major", ctypes.c_uint32),
        ("minor", ctypes.c_uint32),
    ]


_DLPACK_DELETER = ctypes.CFUNCTYPE(None, ctypes.c_void_p)


class DLManagedTensor(ctypes.Structure):
    """The pre-1.0 managed tensor. ``dl_tensor`` is at offset 0."""

    _fields_ = [
        ("dl_tensor", DLTensor),
        ("manager_ctx", ctypes.c_void_p),
        ("deleter", _DLPACK_DELETER),
    ]


class DLManagedTensorVersioned(ctypes.Structure):
    """The DLPack 1.x managed tensor, which is the one that can carry the read-only flag."""

    _fields_ = [
        ("version", DLPackVersion),
        ("manager_ctx", ctypes.c_void_p),
        ("deleter", _DLPACK_DELETER),
        ("flags", ctypes.c_uint64),
        ("dl_tensor", DLTensor),
    ]


_MANAGED_BY_CAPSULE_NAME = (
    (_c_str_dltensor, DLManagedTensor),
    (_c_str_dltensor_versioned, DLManagedTensorVersioned),
)

# The DLPack deleter can run without the GIL, so its managed block must use the matched
# PyMem_RawMalloc/PyMem_RawFree allocator pair.
PyMem_RawMalloc = ctypes.pythonapi.PyMem_RawMalloc
PyMem_RawMalloc.argtypes = [ctypes.c_size_t]
PyMem_RawMalloc.restype = ctypes.c_void_p

PyMem_RawFree = ctypes.pythonapi.PyMem_RawFree
PyMem_RawFree.argtypes = [ctypes.c_void_p]
PyMem_RawFree.restype = None

Py_IncRef = ctypes.pythonapi.Py_IncRef
Py_IncRef.argtypes = [ctypes.py_object]
Py_IncRef.restype = None

Py_DecRef = ctypes.pythonapi.Py_DecRef
Py_DecRef.argtypes = [ctypes.py_object]
Py_DecRef.restype = None

PyCapsule_Destructor = ctypes.CFUNCTYPE(None, ctypes.c_void_p)

PyCapsule_New = ctypes.pythonapi.PyCapsule_New
PyCapsule_New.argtypes = [ctypes.c_void_p, ctypes.c_char_p, PyCapsule_Destructor]
PyCapsule_New.restype = ctypes.py_object

PyCapsule_SetContext = ctypes.pythonapi.PyCapsule_SetContext
PyCapsule_SetContext.argtypes = [ctypes.py_object, ctypes.c_void_p]
PyCapsule_SetContext.restype = ctypes.c_int

# Use raw pointers in the destructor to avoid re-incrementing a capsule at refcount zero.
_PyCapsule_IsValid_raw = ctypes.pythonapi["PyCapsule_IsValid"]
_PyCapsule_IsValid_raw.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
_PyCapsule_IsValid_raw.restype = ctypes.c_int

_PyCapsule_GetPointer_raw = ctypes.pythonapi["PyCapsule_GetPointer"]
_PyCapsule_GetPointer_raw.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
_PyCapsule_GetPointer_raw.restype = ctypes.c_void_p

_PyCapsule_GetContext_raw = ctypes.pythonapi["PyCapsule_GetContext"]
_PyCapsule_GetContext_raw.argtypes = [ctypes.c_void_p]
_PyCapsule_GetContext_raw.restype = ctypes.c_void_p

# Resolve before a destructor may enter with an exception already pending.
_PyErr_Clear = ctypes.pythonapi.PyErr_Clear
_PyErr_Clear.argtypes = []
_PyErr_Clear.restype = None


# Use a native raw free as the DLPack deleter; consumers may invoke it without the GIL or
# during interpreter teardown, when a ctypes Python callback is unsafe.
_C_FREE_DELETER = ctypes.cast(PyMem_RawFree, _DLPACK_DELETER)


def _drain_pending_exception() -> None:
    """Clear a pending exception so a ctypes capsule destructor can complete cleanup.

    ctypes cannot preserve that exception across the callback boundary. ``PyErr_Clear``
    is resolved at import time so this function makes no preliminary lookup call.
    """
    _PyErr_Clear()


@PyCapsule_Destructor
def _capsule_destructor(capsule_ptr: int) -> None:
    """Release one exported capsule: drop the retained owner, free an unconsumed block.

    The module-level callback outlives every capsule. Capsule context carries the retained
    owner. A consumer claims the managed block by renaming the capsule, so only a capsule
    still bearing its original name is freed here.
    """
    # Cleanup must proceed even when capsule destruction begins with an exception pending.
    _drain_pending_exception()
    try:
        ctx_id = _PyCapsule_GetContext_raw(capsule_ptr)
        if ctx_id:
            # Keep a local reference until this callback frame exits.
            owner = ctypes.cast(ctx_id, ctypes.py_object).value
            Py_DecRef(owner)
    finally:
        for capsule_name, managed_type in _MANAGED_BY_CAPSULE_NAME:
            if not _PyCapsule_IsValid_raw(capsule_ptr, capsule_name):
                continue
            managed_ptr = _PyCapsule_GetPointer_raw(capsule_ptr, capsule_name)
            if managed_ptr:
                block = managed_type.from_address(managed_ptr)
                if block.deleter:
                    block.deleter(managed_ptr)
            break


def _to_dlpack_capsule(
    dl_tensor: DLTensor,
    manager_ctx: Any,
    *,
    versioned: bool = False,
    readonly: bool = True,
) -> Any:
    """Mint a DLPack ``PyCapsule`` describing ``dl_tensor``, borrowing its memory.

    ``manager_ctx`` is retained only while the capsule exists, not for the lifetime of a
    consumer's view. The caller must therefore keep the mapping alive. The managed tensor,
    shape, and strides are copied into one raw allocation; only payload remains borrowed.

    ``versioned`` selects the DLPack 1.x ``DLManagedTensorVersioned`` layout, which is the
    layout that can carry ``readonly``. Vector lanes become a trailing shape axis.
    """
    shape, strides, _lanes = dl_tensor.lane_folded_layout("cannot export tensor as DLPack")
    ndim = len(shape)
    if versioned and ndim and not strides:
        # DLPack 1.2+ requires explicit strides for every non-scalar tensor.
        compact_strides = [1] * ndim
        for axis in range(ndim - 2, -1, -1):
            compact_strides[axis] = compact_strides[axis + 1] * shape[axis + 1]
            if compact_strides[axis] > (1 << 63) - 1:
                raise BufferError("compact tensor strides exceed DLPack's int64 range")
        strides = tuple(compact_strides)

    if versioned:
        managed_type: Any = DLManagedTensorVersioned
        capsule_name = _c_str_dltensor_versioned
    else:
        managed_type = DLManagedTensor
        capsule_name = _c_str_dltensor

    managed_size = ctypes.sizeof(managed_type)
    axis_bytes = ndim * ctypes.sizeof(ctypes.c_int64)
    mem_ptr = PyMem_RawMalloc(managed_size + axis_bytes + (axis_bytes if strides else 0))
    if not mem_ptr:
        raise MemoryError("failed to allocate a DLManagedTensor for the DLPack export")

    try:
        managed_tensor = managed_type.from_address(mem_ptr)
        if versioned:
            managed_tensor.version.major = DLPACK_MAJOR_VERSION
            managed_tensor.version.minor = DLPACK_MINOR_VERSION
            managed_tensor.flags = DLPACK_FLAG_BITMASK_READ_ONLY if readonly else 0

        # Fold byte_offset into data because some DLPack consumers ignore the field, then
        # export an offset of zero. Preserve NULL for legal empty tensors.
        data_address = int(dl_tensor.data or 0)
        if data_address:
            data_address += int(dl_tensor.byte_offset)

        managed_tensor.dl_tensor.data = data_address
        managed_tensor.dl_tensor.device = dl_tensor.device
        managed_tensor.dl_tensor.ndim = ndim
        managed_tensor.dl_tensor.byte_offset = 0
        managed_tensor.dl_tensor.dtype.code = dl_tensor.dtype.code
        managed_tensor.dl_tensor.dtype.bits = dl_tensor.dtype.bits
        managed_tensor.dl_tensor.dtype.lanes = 1

        shape_ptr = ctypes.cast(mem_ptr + managed_size, ctypes.POINTER(ctypes.c_int64))
        for axis, extent in enumerate(shape):
            shape_ptr[axis] = extent
        managed_tensor.dl_tensor.shape = shape_ptr

        if strides:
            strides_ptr = ctypes.cast(mem_ptr + managed_size + axis_bytes, ctypes.POINTER(ctypes.c_int64))
            for axis, stride in enumerate(strides):
                strides_ptr[axis] = stride
            managed_tensor.dl_tensor.strides = strides_ptr
        else:
            # Legacy compact tensors and scalars may omit strides.
            managed_tensor.dl_tensor.strides = None

        # The raw-free deleter needs no manager state; Python ownership lives on the capsule.
        managed_tensor.manager_ctx = None
        managed_tensor.deleter = _C_FREE_DELETER

        capsule = PyCapsule_New(mem_ptr, capsule_name, _capsule_destructor)
    except BaseException:
        PyMem_RawFree(mem_ptr)
        raise

    Py_IncRef(manager_ctx)  # released by _capsule_destructor
    if PyCapsule_SetContext(capsule, id(manager_ctx)) != 0:
        Py_DecRef(manager_ctx)
        raise RuntimeError("failed to retain the tensor owner on the DLPack capsule")
    return capsule
