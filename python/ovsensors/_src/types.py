# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Python value types over the ovsensors result tree.

Metadata types are snapshots. :class:`TensorView` instead borrows provider memory for
the lifetime of its mapping.
"""

import ctypes
import operator
from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING, Any, Iterator, List, Optional, Sequence, Tuple

from . import bindings as _b
from .dlpack import DLPACK_MAJOR_VERSION, DLDataType, DLDataTypeCode, DLDeviceType, DLTensor, _to_dlpack_capsule

if TYPE_CHECKING:
    import numpy

    # Avoid collision with TensorView.numpy in annotations evaluated by type checkers.
    NumpyArray = numpy.ndarray


class Status(IntEnum):
    """Every status the C API can return."""

    SUCCESS = _b.OVSENSORS_API_SUCCESS
    ERROR = _b.OVSENSORS_API_ERROR
    TIMEOUT = _b.OVSENSORS_API_TIMEOUT
    INVALID_ARGUMENT = _b.OVSENSORS_API_INVALID_ARGUMENT
    INVALID_HANDLE = _b.OVSENSORS_API_INVALID_HANDLE
    BUSY = _b.OVSENSORS_API_BUSY
    NOT_SUPPORTED = _b.OVSENSORS_API_NOT_SUPPORTED
    OUT_OF_MEMORY = _b.OVSENSORS_API_OUT_OF_MEMORY
    OP_FAILED = _b.OVSENSORS_API_OP_FAILED
    RESET_REQUIRED = _b.OVSENSORS_API_RESET_REQUIRED
    NOT_ATTACHED = _b.OVSENSORS_API_NOT_ATTACHED
    INTERNAL = _b.OVSENSORS_API_INTERNAL


class Device(IntEnum):
    """Provider-selected, host, linear CUDA, or opaque CUDA-array mapping."""

    DEFAULT = _b.OVSENSORS_MAP_DEVICE_TYPE_DEFAULT
    CPU = _b.OVSENSORS_MAP_DEVICE_TYPE_CPU
    CUDA = _b.OVSENSORS_MAP_DEVICE_TYPE_CUDA
    CUDA_ARRAY = _b.OVSENSORS_MAP_DEVICE_TYPE_CUDA_ARRAY


class EventStatus(IntEnum):
    """Provider event status reported by a mapped output."""

    PENDING = _b.OVSENSORS_EVENT_PENDING
    COMPLETED = _b.OVSENSORS_EVENT_COMPLETED
    FAILURE = _b.OVSENSORS_EVENT_FAILURE


class Validity(IntEnum):
    """Payload validity of a produced output."""

    INVALID = _b.OVSENSORS_OUTPUT_INVALID
    VALID = _b.OVSENSORS_OUTPUT_VALID


class LogSeverity(IntEnum):
    """Log severities, ordered so a threshold comparison works."""

    VERBOSE = _b.OVSENSORS_LOG_VERBOSE
    INFO = _b.OVSENSORS_LOG_INFO
    WARNING = _b.OVSENSORS_LOG_WARNING
    ERROR = _b.OVSENSORS_LOG_ERROR
    NONE = _b.OVSENSORS_LOG_NONE


class OvsensorsError(RuntimeError):
    """A non-success status from the C API, with whatever detail the library supplied.

    ``op_id`` identifies an operation-scoped failure and is ``None`` for calls that have
    no operation.
    """

    def __init__(self, status: Status, operation: str, detail: str = "", op_id: Optional[int] = None) -> None:
        self.status = status
        self.operation = operation
        self.detail = detail
        self.op_id = op_id
        message = f"{operation} failed: {status.name}"
        if op_id is not None:
            message = f"{message} (op {op_id})"
        if detail:
            message = f"{message} -- {detail}"
        super().__init__(message)


class TimeoutError(OvsensorsError):  # noqa: A001 - deliberately shadows the builtin within this package
    """A cooperative budget expired; the live work may be retried unchanged."""


@dataclass(frozen=True)
class Rational:
    """A numerator/denominator pair. ``0/0`` means unauthored; ``0/1`` is authored zero."""

    num: int
    den: int

    @property
    def authored(self) -> bool:
        """True when the value was authored on the prim."""
        return self.den != 0

    def as_float(self) -> Optional[float]:
        """The value as a float, or None when unauthored."""
        return None if self.den == 0 else self.num / self.den

    def __str__(self) -> str:
        return "unauthored" if self.den == 0 else f"{self.num}/{self.den}"


@dataclass(frozen=True)
class CudaSync:
    """Opaque CUDA stream and event handles for producer-consumer synchronization.

    ``{0, 0}`` means no wait is required.
    """

    stream: int = 0
    wait_event: int = 0

    @property
    def required(self) -> bool:
        """True when there is a dependency the consumer must honour before reading."""
        return self.stream != 0 or self.wait_event != 0

    @classmethod
    def _from_c(cls, raw: _b.ovsensors_cuda_sync_t) -> "CudaSync":
        return cls(stream=int(raw.stream or 0), wait_event=int(raw.wait_event or 0))

    def _to_c(self) -> _b.ovsensors_cuda_sync_t:
        return _b.ovsensors_cuda_sync_t(
            stream=ctypes.c_void_p(self.stream or None), wait_event=ctypes.c_void_p(self.wait_event or None)
        )


@dataclass(frozen=True)
class Descriptor:
    """Creation-time sensor identity and configuration, copied into a snapshot.

    Attributes:
        name (str): Instance-unique sensor name supplied at creation.
        product (str): Adopted RenderProduct prim path.
        sources (Tuple[str, ...]): Product camera-relationship target paths in authored
            order. The provider's output format defines how tensors and parameters
            correspond to these sources.
        schema (str): Applied schema used for automatic provider selection; empty for
            explicit provider selection.
        provider_name (str): Diagnostic name of the provider serving this sensor.
        rate (Rational): Authored frames per second; ``0/0`` when unauthored.
        start (Rational): Authored start offset in seconds; ``0/0`` when unauthored.
    """

    name: str
    product: str
    sources: Tuple[str, ...]
    schema: str
    provider_name: str
    rate: Rational
    start: Rational

    @classmethod
    def _from_c(cls, raw: _b.ovsensors_descriptor_t) -> "Descriptor":
        sources = tuple(raw.sources[i].text() for i in range(int(raw.source_count))) if raw.sources else ()
        return cls(
            name=raw.name.text(),
            product=raw.product.text(),
            sources=sources,
            schema=raw.schema.text(),
            provider_name=raw.provider_name.text(),
            rate=Rational(int(raw.rate.num), int(raw.rate.den)),
            start=Rational(int(raw.start.num), int(raw.start.den)),
        )


class TensorView:
    """A borrowed view of one named tensor from a mapping.

    Keep the mapping alive while reading this view. Copy before unmap if the payload must
    outlive the mapping. :meth:`numpy` requires host memory; the DLPack protocol supports
    CPU and linear CUDA without copying. Opaque CUDA arrays expose :attr:`cuda_array`
    and descriptor metadata; they cannot be exported as linear arrays.

    Provider shape and strides are reported unchanged. Vector lanes are exposed as a
    trailing array axis, so an array has ``shape + (lanes,)`` when ``lanes > 1``.
    """

    __slots__ = ("_name", "_doc", "_dl", "_owner")

    def __init__(self, name: str, dl: "ctypes._Pointer", owner: Any, doc: str = "") -> None:
        self._name = name
        self._doc = doc
        self._dl = dl
        self._owner = owner

    @property
    def name(self) -> str:
        """The provider-reported constituent name, which may be empty."""
        return self._name

    @property
    def doc(self) -> str:
        """The provider's description, copied when the mapping is created."""
        return self._doc

    def _tensor(self) -> DLTensor:
        self._owner._check_live(f"tensor {self._name!r}")
        return self._dl.contents

    @property
    def shape(self) -> Tuple[int, ...]:
        """Shape as reported by the provider, excluding any vector-lane axis."""
        return self._tensor().shape_tuple

    @property
    def strides(self) -> Tuple[int, ...]:
        """DLPack strides in elements, or empty for row-major compact storage."""
        return self._tensor().strides_tuple

    @property
    def dtype(self) -> str:
        """NumPy dtype name for one lane; opaque handles have no NumPy dtype."""
        return self._tensor().dtype.lane_dtype_str(f"tensor {self._name!r}")

    @property
    def dl_dtype(self) -> DLDataType:
        """A snapshot of the provider's DLPack code, bits, and lanes, including opaque types."""
        dtype = self._tensor().dtype
        return DLDataType(dtype.code, dtype.bits, dtype.lanes)

    @property
    def cuda_array(self) -> int:
        """Borrowed CUDA-array handle, valid until unmap; never a linear device pointer.

        Honor :attr:`Mapping.cuda_sync` before using the handle in CUDA operations,
        and supply consumer completion to :meth:`Mapping.unmap` as needed.
        """
        t = self._tensor()
        if int(t.dtype.code) != DLDataTypeCode.kDLOpaqueHandle or int(t.device.device_type) != DLDeviceType.kDLCUDA:
            raise ValueError(f"tensor {self._name!r} is not an opaque CUDA array")
        if not t.data:
            raise ValueError(f"tensor {self._name!r} has a NULL CUDA array handle")
        return int(t.data)

    def _require_linear(self, tensor: DLTensor) -> None:
        if int(tensor.dtype.code) == DLDataTypeCode.kDLOpaqueHandle:
            raise BufferError(
                f"tensor {self._name!r} contains an opaque handle, not linear memory; "
                "use cuda_array for CUDA interop or map with Device.CPU or Device.CUDA"
            )

    @property
    def lanes(self) -> int:
        """Components per element, validated in DLPack's supported range."""
        return self._tensor().dtype.lane_count(f"tensor {self._name!r}")

    @property
    def device(self) -> Tuple[int, int]:
        """``(device_type, device_id)`` as DLPack reports it."""
        t = self._tensor()
        return (int(t.device.device_type), int(t.device.device_id))

    @property
    def data_ptr(self) -> int:
        """Raw address of the payload, including any byte offset."""
        t = self._tensor()
        self._require_linear(t)
        return int(t.data or 0) + int(t.byte_offset)

    def _borrow(self) -> "NumpyArray":
        """Create a zero-copy ndarray over this tensor's strided storage span.

        Negative, padded, transposed, and zero strides are preserved. The backing buffer
        spans from the lowest to highest reachable element, and the ndarray offset places
        logical element zero at ``data_ptr``. Empty tensors are handled before validating
        the data pointer because DLPack permits their pointer to be NULL. Vector lanes are
        folded into a trailing scalar axis before calculating the span.
        """
        import numpy

        t = self._tensor()
        self._require_linear(t)
        if int(t.device.device_type) not in (DLDeviceType.kDLCPU, DLDeviceType.kDLCUDAHost):
            raise ValueError(f"numpy() needs host memory; tensor {self._name!r} is on device {self.device}")

        what = f"tensor {self._name!r}"
        dtype = numpy.dtype(t.dtype.lane_dtype_str(what))
        itemsize = dtype.itemsize
        shape, strides, _lanes = t.lane_folded_layout(what)

        if any(extent == 0 for extent in shape):
            # Empty tensors read no storage, including when data is NULL.
            if not t.data:
                return numpy.empty(shape, dtype)
            return numpy.ndarray(
                shape, dtype, (ctypes.c_ubyte * 0).from_address(int(t.data) + int(t.byte_offset)), 0, None
            )

        if not t.data:
            raise ValueError(f"tensor {self._name!r} has a NULL data pointer")

        base = int(t.data) + int(t.byte_offset)

        if not strides:
            # NULL strides means row-major compact; a scalar shape holds one element.
            span = itemsize * int(numpy.prod(shape)) if shape else itemsize
            return numpy.ndarray(shape, dtype, (ctypes.c_ubyte * span).from_address(base), 0, None)

        low = sum((extent - 1) * stride for extent, stride in zip(shape, strides) if stride < 0)
        high = sum((extent - 1) * stride for extent, stride in zip(shape, strides) if stride > 0)
        span = (high - low + 1) * itemsize
        buffer = (ctypes.c_ubyte * span).from_address(base + low * itemsize)
        byte_strides = tuple(stride * itemsize for stride in strides)
        return numpy.ndarray(shape, dtype, buffer, -low * itemsize, byte_strides)

    def numpy(self, copy: bool = True) -> "NumpyArray":
        """View or copy CPU or CUDA-host payloads as a NumPy array.

        ``copy=True`` returns a writable copy that survives unmap. ``copy=False`` returns
        a read-only view of the provider's strided storage, valid only while the mapping
        is open. Vector lanes form a trailing shape axis. Use DLPack for CUDA device memory.
        """
        array = self._borrow()
        array.flags.writeable = False
        return array.copy() if copy else array

    def __array__(self, dtype: Any = None, copy: Optional[bool] = None) -> "NumpyArray":
        """numpy's array protocol, so ``numpy.asarray(view)`` and ``numpy.array(view)`` work.

        ``copy=None`` follows numpy's copy-if-required semantics. A borrowed result is
        read-only and expires at unmap; a copied result is writable. ``copy=False``
        raises when dtype conversion would require a copy.
        """
        import numpy

        array = self.numpy(copy=False)
        if dtype is not None and numpy.dtype(dtype) != array.dtype:
            if copy is False:
                raise ValueError(
                    f"copy=False, but tensor {self._name!r} holds {array.dtype} and "
                    f"{numpy.dtype(dtype)} was requested; converting requires a copy"
                )
            return array.astype(dtype, copy=True)
        return array.copy() if copy else array

    def __dlpack_device__(self) -> Tuple[int, int]:
        """Return ``(device_type, device_id)`` for the DLPack protocol."""
        return self.device

    def __dlpack__(
        self,
        *,
        stream: Optional[int] = None,
        max_version: Optional[Tuple[int, int]] = None,
        dl_device: Optional[Tuple[int, int]] = None,
        copy: Optional[bool] = None,
    ) -> Any:
        """Export the payload as a DLPack capsule, zero-copy, on CPU or CUDA alike.

        The consumer's tensor borrows storage only until the mapping is unmapped; keep the
        mapping alive or copy within its context::

            with instance.mapped(handle, device=Device.CPU) as mapping:
                borrowed = numpy.from_dlpack(mapping["Coordinates"])
                kept = numpy.from_dlpack(mapping["Coordinates"]).copy()

        Exporting from an unmapped mapping raises :class:`OvsensorsError` rather than
        handing out a capsule over memory the provider has taken back.

        For CUDA, ``stream=None`` or ``1`` selects the legacy default stream, ``2`` the
        per-thread default stream, and larger integers a native consumer stream handle.
        ``0`` is invalid. Producer events are waited on in the consumer stream; a
        different producer stream is synchronized on the host before export. Default
        streams require the tensor's CUDA context and device to be current on the calling
        thread. A per-thread producer stream (``Mapping.cuda_sync.stream == 2``) cannot
        be synchronized automatically because its originating thread is unknown.
        Synchronization lazily loads the CUDA runtime and raises if it is unavailable
        or fails. ``stream=-1`` explicitly skips synchronization: the caller must honor
        :attr:`Mapping.cuda_sync` before reading. Export does not record consumer
        completion; follow :meth:`Mapping.unmap` when returning the borrowed storage.

        ``copy=True`` and cross-device requests are refused because this export cannot
        copy. ``max_version`` selects a versioned DLPack 1.x capsule when supported,
        otherwise the legacy capsule. Vector lanes form a trailing shape axis.

        Versioned exports carry the DLPack read-only flag.
        """
        tensor = self._tensor()
        self._require_linear(tensor)
        if copy is True:
            raise BufferError(
                f"copy=True is not supported; tensor {self._name!r} exports a borrow of provider memory"
            )
        if dl_device is not None:
            try:
                requested_device = (operator.index(dl_device[0]), operator.index(dl_device[1]))
            except (TypeError, IndexError, KeyError, ValueError) as exc:
                raise TypeError("dl_device must be a (device_type, device_id) tuple") from exc
            current_device = (int(tensor.device.device_type), int(tensor.device.device_id))
            if requested_device != current_device:
                raise BufferError(
                    f"dl_device={requested_device!r} does not match tensor {self._name!r} on device "
                    f"{current_device!r}; cross-device copy is not supported"
                )
        use_versioned = max_version is not None and max_version[0] == DLPACK_MAJOR_VERSION and max_version >= (1, 0)
        if int(tensor.device.device_type) in (DLDeviceType.kDLCUDA, DLDeviceType.kDLCUDAManaged):
            from . import cudart

            sync = self._owner.cuda_sync
            cudart.synchronize(int(tensor.device.device_id), sync.stream, sync.wait_event, stream)
            # CUDA calls release the GIL and can block; the mapping must still be live.
            tensor = self._tensor()
        # Retaining the view retains both its mapping owner and DLTensor descriptor.
        return _to_dlpack_capsule(tensor, self, versioned=use_versioned, readonly=True)

    def __repr__(self) -> str:
        try:
            lanes = self.lanes
            vector = f", lanes={lanes}" if lanes != 1 else ""
            return f"TensorView(name={self._name!r}, shape={self.shape}, dtype={self.dtype!r}{vector})"
        except Exception:
            return f"TensorView(name={self._name!r}, <unavailable>)"


@dataclass(frozen=True)
class Output:
    """One named output of a frame. Snapshot; ``handle`` is valid until result-set release."""

    name: str
    validity: Validity
    handle: int

    @property
    def valid(self) -> bool:
        """True when the payload is a usable sample."""
        return self.validity is Validity.VALID

    @classmethod
    def _from_c(cls, raw: _b.ovsensors_output_t) -> "Output":
        return cls(name=raw.name.text(), validity=Validity(int(raw.validity)), handle=int(raw.handle))


@dataclass(frozen=True)
class Frame:
    """One emit of one sensor: an absolute closed window and the outputs it produced.

    Windows may predate the delivering step, overlap, or have zero width. Align frames by
    timestamp rather than call or step index.
    """

    start_time: float
    end_time: float
    outputs: Tuple[Output, ...]

    @property
    def duration(self) -> float:
        """Window length in seconds. Zero for an instantaneous sample, as cameras report."""
        return self.end_time - self.start_time

    def __iter__(self) -> Iterator[Output]:
        return iter(self.outputs)

    def __len__(self) -> int:
        return len(self.outputs)

    def output(self, name: str) -> Output:
        """Look an output up by name. Raises KeyError if absent."""
        for candidate in self.outputs:
            if candidate.name == name:
                return candidate
        raise KeyError(f"no output named {name!r}; have {[o.name for o in self.outputs]}")

    @classmethod
    def _from_c(cls, raw: _b.ovsensors_frame_t) -> "Frame":
        outputs = tuple(Output._from_c(raw.outputs[i]) for i in range(int(raw.output_count))) if raw.outputs else ()
        return cls(start_time=float(raw.frame_start_time), end_time=float(raw.frame_end_time), outputs=outputs)


@dataclass(frozen=True)
class SensorResult:
    """One sensor's part of a step result.

    :attr:`held`, :attr:`failed`, and :attr:`produced` combine status and frame count into
    the common result states.
    """

    descriptor: Descriptor
    status: Status
    error_message: str
    frames: Tuple[Frame, ...]

    @property
    def failed(self) -> bool:
        """True when this sensor failed. Its error_message is the only channel that says why."""
        return self.status is not Status.SUCCESS

    @property
    def held(self) -> bool:
        """True for an ordinary cadence hold. The sensor succeeded, but nothing was due."""
        return self.status is Status.SUCCESS and not self.frames

    @property
    def produced(self) -> bool:
        """True when this sensor succeeded AND emitted at least one frame."""
        return self.status is Status.SUCCESS and bool(self.frames)

    @property
    def name(self) -> str:
        """The sensor's name, from its descriptor."""
        return self.descriptor.name

    def __iter__(self) -> Iterator[Frame]:
        return iter(self.frames)

    def __len__(self) -> int:
        return len(self.frames)

    def outputs(self) -> Iterator[Tuple[Frame, Output]]:
        """Every ``(frame, output)`` pair, flattened -- the usual consumer loop."""
        for frame in self.frames:
            for output in frame.outputs:
                yield frame, output

    @classmethod
    def _from_c(cls, raw: _b.ovsensors_sensor_result_t) -> "SensorResult":
        frames = tuple(Frame._from_c(raw.frames[i]) for i in range(int(raw.frame_count))) if raw.frames else ()
        return cls(
            descriptor=Descriptor._from_c(raw.descriptor.contents),
            status=Status(int(raw.status)),
            error_message=raw.error_message.text(),
            frames=frames,
        )


class ResultSet(Sequence[SensorResult]):
    """The aggregate result of one step: one group per sensor, in the step's order.

    Snapshotted at fetch, so it stays readable after the operation is released.
    Group order matches the order sensors were passed to ``step``, so positional
    correlation works -- but :meth:`by_name` is there so a consumer need not rely on it.
    """

    __slots__ = ("_groups", "_op_id")

    def __init__(self, groups: Sequence[SensorResult], op_id: int) -> None:
        self._groups: Tuple[SensorResult, ...] = tuple(groups)
        self._op_id = op_id

    @property
    def op_id(self) -> int:
        """The operation this result set came from."""
        return self._op_id

    def __getitem__(self, index: Any) -> Any:
        return self._groups[index]

    def __len__(self) -> int:
        return len(self._groups)

    def __iter__(self) -> Iterator[SensorResult]:
        return iter(self._groups)

    def by_name(self, name: str) -> SensorResult:
        """Look a sensor's group up by name rather than by position."""
        for group in self._groups:
            if group.descriptor.name == name:
                return group
        raise KeyError(f"no sensor named {name!r}; have {[g.descriptor.name for g in self._groups]}")

    @property
    def failures(self) -> List[SensorResult]:
        """Every group that failed. Empty when every sensor succeeded."""
        return [g for g in self._groups if g.failed]

    @property
    def produced(self) -> List[SensorResult]:
        """Every group that emitted at least one frame."""
        return [g for g in self._groups if g.produced]

    @classmethod
    def _from_c(cls, raw: _b.ovsensors_step_result_set_t, op_id: int) -> "ResultSet":
        groups = [SensorResult._from_c(raw.groups[i]) for i in range(int(raw.group_count))] if raw.groups else []
        return cls(groups, op_id)

    def __repr__(self) -> str:
        return f"ResultSet(op_id={self._op_id}, groups={len(self._groups)}, failures={len(self.failures)})"
