# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Python lifetime and error handling over the ovsensors C API.

Instances retain provider and attached-stage owners; mappings and tensor views retain
their parents. Context managers enforce release order, result metadata is snapshotted,
and C statuses become exceptions.
"""

import ctypes
import sys
import threading
import warnings
from contextlib import contextmanager
from enum import IntEnum
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

from . import bindings as _b
from .types import (
    CudaSync,
    Descriptor,
    Device,
    EventStatus,
    LogSeverity,
    OvsensorsError,
    ResultSet,
    Status,
    TensorView,
    TimeoutError,
)

INFINITE = _b.OVSENSORS_TIMEOUT_INFINITE
POLL = _b.OVSENSORS_TIMEOUT_POLL

_LogCallback = Callable[[LogSeverity, float, str, str], None]

# Optional provider-owner protocol.
_RETAIN = "_ovsensors_retain"
_RELEASE = "_ovsensors_release"


class _State(IntEnum):
    """Private wrapper lifetime state.

    ``CLEANUP_RETRY_ONLY`` permits only the teardown retry required after cleanup has
    started and failed.
    """

    ACTIVE = 1
    CLEANUP_RETRY_ONLY = 2
    DEAD = 3


def _warn_dropped_resource(message: str) -> None:
    """Report a leaked resource without letting an escalated warning escape ``__del__``."""
    try:
        warnings.warn(message, ResourceWarning, stacklevel=3)
    except Exception:
        try:
            print(f"ResourceWarning: {message}", file=sys.stderr)
        except Exception:
            pass  # stderr gone (interpreter shutdown); nothing left to report through


def _cleanup_without_masking(cleanup: Callable[[], None], propagating: BaseException) -> None:
    """Run teardown without replacing an exception already in flight.

    A cleanup exception is inserted into ``propagating.__context__``. Process-control
    exceptions are not intercepted.
    """
    try:
        cleanup()
    except Exception as cleanup_error:
        cleanup_error.__context__ = propagating.__context__
        propagating.__context__ = cleanup_error


def _notify(owner: Any, hook_name: str) -> None:
    """Call an optional provider-owner lifecycle hook."""
    hook = getattr(owner, hook_name, None)
    if hook is not None:
        hook()


def _failure_detail(instance: Optional["Instance"], op_id: Optional[int]) -> str:
    """Return operation-scoped detail when available, otherwise thread-local detail."""
    if instance is not None and op_id is not None and op_id != _b.OVSENSORS_INVALID_OP_ID:
        scoped = instance._op_detail(op_id)
        if scoped:
            return scoped
    return last_error()


def _check(
    status_value: int,
    operation: str,
    op_id: Optional[int] = None,
    instance: Optional["Instance"] = None,
) -> Status:
    """Raise for non-success, using :class:`TimeoutError` for retryable timeouts."""
    status = Status(status_value)
    if status is Status.SUCCESS:
        return status
    detail = _failure_detail(instance, op_id)
    if status is Status.TIMEOUT:
        raise TimeoutError(status, operation, detail, op_id)
    raise OvsensorsError(status, operation, detail, op_id)


def version() -> Tuple[int, int, int]:
    """The loaded library's ``(major, minor, patch)``."""
    lib = _b.library()
    major, minor, patch = ctypes.c_uint32(), ctypes.c_uint32(), ctypes.c_uint32()
    lib.ovsensors_get_version(ctypes.byref(major), ctypes.byref(minor), ctypes.byref(patch))
    return (major.value, minor.value, patch.value)


def last_error() -> str:
    """Best-effort detail for the last failing call on this thread, copied into a string.

    Empty when no detail is available, including for timeouts.
    """
    return _b.library().ovsensors_get_last_error().text()


def initialize() -> None:
    """Take a library reference that must be balanced by :func:`shutdown`.

    Instances hold their own references; call this explicitly when library state is
    needed before the first instance, such as for early log registration.
    """
    _check(_b.library().ovsensors_initialize(None).status, "initialize")


def shutdown() -> None:
    """Release one explicit initialization reference.

    The final process reference flushes and stops log dispatch; live instances retain
    their own references. Releasing the final reference from a log callback raises
    :class:`OvsensorsError` with status ERROR and leaves that reference outstanding.
    """
    _check(_b.library().ovsensors_shutdown().status, "shutdown")


def flush_log(timeout_ns: int = INFINITE) -> None:
    """Wait for delivery of messages admitted before this call.

    ``timeout_ns`` is a cooperative nanosecond budget. Admission is best-effort; queue
    overflow can drop messages. Requires an explicit initialization reference or a
    live instance. Do not call from a log callback.
    """
    _check(_b.library().ovsensors_flush_log(_b.check_timeout(timeout_ns)).status, "flush_log")


class _LogCallbackRegistration:
    """Serialize registration and keep the active ctypes trampoline alive."""

    def __init__(self) -> None:
        self._trampoline: Optional[Any] = None
        self._lock = threading.Lock()

    def set(self, severity: LogSeverity, callback: Optional[_LogCallback]) -> None:
        """Register or clear the callback atomically with its Python keepalive."""
        with self._lock:
            if callback is None:
                # A default-constructed CFUNCTYPE is the NULL function pointer the C API
                # wants; ctypes refuses a bare None for a declared callback argument.
                null_callback = _b.ovsensors_log_callback_t()
                _check(
                    _b.library().ovsensors_set_log_callback(int(LogSeverity.NONE), null_callback, None).status,
                    "set_log_callback",
                )
                self._trampoline = None
                return

            def _shim(
                sev: int, timestamp: float, source: _b.ovx_string_t, message: _b.ovx_string_t, _user: Any
            ) -> None:
                callback(LogSeverity(sev), float(timestamp), source.text(), message.text())

            trampoline = _b.ovsensors_log_callback_t(_shim)
            # Publish the new keepalive only after C accepts the registration.
            _check(_b.library().ovsensors_set_log_callback(int(severity), trampoline, None).status, "set_log_callback")
            self._trampoline = trampoline


_log_registration = _LogCallbackRegistration()


def set_log_callback(callback: Optional[_LogCallback], severity: LogSeverity = LogSeverity.INFO) -> None:
    """Install the process-global log callback, or clear it by passing ``None``.

    Requires an explicit initialization reference or a live instance. Callbacks run
    serially on a logging thread; synchronize state shared with application threads.
    The arguments are ``(severity, timestamp, source, message)``. ``source`` names the
    provider the message concerns, or is empty for ovsensors itself. Both strings are
    Python copies and may be kept after the callback returns.

    Successful replacement or disabling drains the previous registration before
    returning. Do not call this function, :func:`flush_log`, or :meth:`Instance.destroy`
    from a log callback; :func:`shutdown` also forbids releasing the final reference there.
    """
    _log_registration.set(severity, callback)


class Mapping:
    """A live, context-managed mapping of one output.

    Tensor views borrow provider memory until unmap, so keep the mapping alive while
    reading them. Releasing the operation does not invalidate an existing mapping.

    The metadata below is copied when the mapping is created. String values may be empty.

    Attributes:
        status (EventStatus): Provider event state; any value may accompany a successful
            map. Payload validity is reported separately by :attr:`Output.validity`.
        error_message (str): Provider error detail, which may be empty for any status.
        name (str): Provider-reported output name.
        type (str): Provider-defined output type.
        doc (str): Human-readable output description.
        version (int): Provider-defined output schema version.
    """

    __slots__ = (
        "_instance", "_raw", "_tensors", "_params", "_sync", "_state", "_rollback_pending",
        "status", "error_message", "name", "type", "doc", "version",
    )

    def __init__(self, instance: "Instance", raw: _b.ovsensors_mapped_t) -> None:
        # Allocate the cleanup owner before native code can publish a handle into raw.
        self._instance = instance
        self._raw = raw
        self._state = _State.ACTIVE
        self._rollback_pending = False
        self._tensors: Tuple[TensorView, ...] = ()
        self._params: Tuple[TensorView, ...] = ()

    @property
    def _handle(self) -> int:
        return int(self._raw.map)

    def _populate(self) -> None:
        """Materialize fallible Python views after the native mapping has an owner."""
        raw = self._raw
        self._sync = CudaSync._from_c(raw.cuda_sync)
        self.status = EventStatus(raw.status)
        self.error_message = raw.error_message.text()
        self.name = raw.name.text()
        self.type = raw.type.text()
        self.doc = raw.doc.text()
        self.version = int(raw.version)
        self._tensors = tuple(
            TensorView(raw.tensors[i].name.text(), raw.tensors[i].dl, self, raw.tensors[i].doc.text())
            for i in range(int(raw.tensor_count))
        )
        self._params = tuple(
            TensorView(raw.params[i].name.text(), ctypes.pointer(raw.params[i].dl), self, raw.params[i].doc.text())
            for i in range(int(raw.param_count))
        )

    def _check_live(self, what: str) -> None:
        if self._state is not _State.ACTIVE:
            raise OvsensorsError(Status.INVALID_HANDLE, f"read {what}", "the mapping has already been unmapped")

    @property
    def cuda_sync(self) -> CudaSync:
        """Producer-ready dependency to honor before reading the payload."""
        return self._sync

    @property
    def tensors(self) -> Tuple[TensorView, ...]:
        """The mapped tensors, in the order the provider reported them."""
        self._check_live("tensors")
        return self._tensors

    @property
    def params(self) -> Dict[str, TensorView]:
        """Named CPU-resident parameters, separate from the tensor namespace."""
        self._check_live("params")
        return {param.name: param for param in self._params}

    def tensor(self, name: str) -> TensorView:
        """Look up a tensor by name."""
        self._check_live(f"tensor {name!r}")
        for candidate in self._tensors:
            if candidate.name == name:
                return candidate
        raise KeyError(f"no tensor named {name!r}; have {[t.name for t in self._tensors]}")

    def __getitem__(self, name: str) -> TensorView:
        """Look up a tensor by name; use :attr:`tensors` for positional access."""
        if not isinstance(name, str):
            raise TypeError(
                f"a mapping is indexed by tensor name, got {type(name).__name__}. Use "
                f"mapping.tensors[{name!r}] if you meant a position."
            )
        return self.tensor(name)

    def __contains__(self, name: object) -> bool:
        """True when this mapping contains a tensor of the given name."""
        return isinstance(name, str) and any(t.name == name for t in self._tensors)

    def __iter__(self) -> Iterator[TensorView]:
        return iter(self.tensors)

    def __len__(self) -> int:
        return len(self._tensors)

    def unmap(self, consumer_done: Optional[CudaSync] = None) -> None:
        """Return the mapping. Idempotent, so an explicit unmap inside a ``with`` is safe.

        ``consumer_done`` names a dependency the provider must honour before reusing
        the memory; omit it when reads are already complete.

        SUCCESS and INVALID_HANDLE consume the mapping. Any other status leaves it live
        and readable for an exact retry.
        """
        if self._state is not _State.ACTIVE:
            return
        if not self._handle:
            # A failed native map never acquired storage.
            self._state = _State.DEAD
            self._instance._forget_pending_mapping(self)
            return
        sync = (consumer_done or CudaSync())._to_c()
        status = int(_b.library().ovsensors_unmap_output(self._instance._handle, self._handle, sync).status)
        if status in (Status.SUCCESS, Status.INVALID_HANDLE):
            self._state = _State.DEAD
            self._instance._forget_pending_mapping(self)
        _check(status, "unmap_output")

    def __enter__(self) -> "Mapping":
        return self

    def __exit__(self, exc_type: Any, exc: Optional[BaseException], tb: Any) -> None:
        """Unmap on exit without replacing an exception already in flight."""
        if exc is None:
            self.unmap()  # clean exit: an unmap failure is the only news there is
            return
        _cleanup_without_masking(self.unmap, exc)

    def __del__(self) -> None:
        """Best-effort unmap and warn when a live mapping is dropped.

        No C call is made during interpreter finalization, when provider teardown order is
        unknown. Cleanup is attempted before warning because warnings may be escalated.
        """
        if getattr(self, "_state", None) is not _State.ACTIVE or not self._handle:
            return
        if getattr(self._instance, "_state", None) is not _State.ACTIVE:
            return
        if sys.is_finalizing():
            _warn_dropped_resource(
                f"Mapping (handle={self._handle}) was still mapped at interpreter shutdown; NO "
                "unmap was attempted, because a C call during the final collection can run after "
                "the provider it routes to has been freed. Use a 'with' block or call unmap() "
                "explicitly."
            )
            return
        try:
            self.unmap()
        except Exception:
            pass
        _warn_dropped_resource(
            f"Mapping (handle={self._handle}) was garbage-collected without unmap(); issued a "
            "best-effort unmap to return the provider's storage. Use a 'with' block or call "
            "unmap() explicitly -- an unreturned mapping answers destroy_instance with BUSY."
        )

    def __repr__(self) -> str:
        state = "live" if self._state is _State.ACTIVE else "unmapped"
        return f"Mapping(handle={self._handle}, tensors={len(self._tensors)}, {state})"


class Sensor:
    """A context-managed sensor.

    For example::

        with inst.sensor("camera", "/World/Render/Product") as camera:
            ...

    Destroyed by :meth:`destroy`, context exit, or instance teardown. A provider-cleanup
    failure leaves the handle available only for a destroy retry.
    """

    __slots__ = ("_instance", "_handle", "_name", "_state")

    def __init__(self, instance: "Instance", handle: Any, name: str) -> None:
        self._instance = instance
        self._handle = handle
        self._name = name
        self._state = _State.ACTIVE

    @property
    def alive(self) -> bool:
        """True while this wrapper has no recorded destruction or cleanup-only state.

        This does not query native state. Failed sensor or instance destruction can
        leave it True even when only a destruction retry is permitted.
        """
        return self._state is _State.ACTIVE

    def _require_active(self, operation: str) -> None:
        """Reject wrapper states known to forbid ordinary sensor operations."""
        if self._state is _State.CLEANUP_RETRY_ONLY:
            raise OvsensorsError(
                Status.INVALID_HANDLE,
                operation,
                f"sensor {self._name!r} is destroy-retry-only after a failed destroy_sensor; "
                "call destroy() again, or let instance teardown retry it",
            )
        if self._state is _State.DEAD:
            raise OvsensorsError(Status.INVALID_HANDLE, operation, "the sensor has been destroyed")

    @property
    def descriptor(self) -> Descriptor:
        """Creation-time resolved descriptor, returned as a snapshot."""
        self._require_active("get_descriptor")
        out = ctypes.POINTER(_b.ovsensors_descriptor_t)()
        _check(
            _b.library().ovsensors_get_descriptor(self._handle, ctypes.byref(out)).status,
            "get_descriptor",
        )
        return Descriptor._from_c(out.contents)

    @property
    def name(self) -> str:
        """Creation-time sensor name, available after destruction."""
        return self._name

    def destroy(self) -> None:
        """Destroy the sensor, retrying any earlier cleanup failure.

        BUSY from a live operation or pending reset leaves the sensor unchanged; finish
        that work before retrying. Once provider cleanup begins, any failure, including
        BUSY, permits only another destroy call or instance teardown. The wrapper cannot
        distinguish these BUSY outcomes, so :attr:`alive` does not certify native usability.
        SUCCESS and INVALID_HANDLE end the wrapper lifetime.
        """
        if self._state is _State.DEAD:
            return
        status = int(_b.library().ovsensors_destroy_sensor(self._handle).status)
        if status in (Status.SUCCESS, Status.INVALID_HANDLE):
            self._state = _State.DEAD
            self._instance._forget(self)
        elif status != Status.BUSY:
            self._state = _State.CLEANUP_RETRY_ONLY
        _check(status, "destroy_sensor")

    def __enter__(self) -> "Sensor":
        return self

    def __exit__(self, exc_type: Any, exc: Optional[BaseException], tb: Any) -> None:
        """Destroy on exit without replacing an exception already in flight."""
        if exc is None:
            self.destroy()
            return
        _cleanup_without_masking(self.destroy, exc)

    def __repr__(self) -> str:
        if self._state is _State.ACTIVE:
            return f"Sensor({self._name!r})"
        if self._state is _State.CLEANUP_RETRY_ONLY:
            return f"Sensor({self._name!r}, <destroy-retry-only>)"
        return "Sensor(<destroyed>)"


class Instance:
    """Own an ovsensors instance, its provider references, and its sensors.

    Use it as a context manager::

        with ovsensors.Instance([provider]) as inst:
            inst.attach(stage)
            ...

    See :meth:`destroy` for recovery after a cleanup failure.
    """

    __slots__ = (
        "_handle", "_owners", "_retained", "_stage_owner", "_sensors", "_attached", "_state", "_pending_mappings"
    )

    def __init__(self, providers: Sequence[Any]) -> None:
        """Create an instance over a fixed provider set.

        Provider objects are retained until destruction succeeds. With a bare
        ``ovsensors_provider_t``, the caller must keep the underlying C provider alive.
        """
        pairs = [_b.as_provider(p) for p in providers]
        bundles = [bundle for bundle, _ in pairs]
        owners = [owner for _, owner in pairs]
        self._owners: Tuple[Any, ...] = tuple(owners) + tuple(bundles)
        self._retained: Tuple[Any, ...] = ()
        self._stage_owner: Any = None
        self._sensors: List[Sensor] = []
        self._pending_mappings: List[Mapping] = []
        self._attached = False
        # A failed create must leave the finalizer with no C handle to dispatch on.
        self._state = _State.DEAD

        count = len(bundles)
        array = (_b.ovsensors_provider_t * count)(*bundles) if count else None
        handle = ctypes.POINTER(_b.ovsensors_instance_t)()
        _check(
            _b.library().ovsensors_create_instance(array, count, ctypes.byref(handle)).status,
            "create_instance",
        )
        self._handle = handle
        self._state = _State.ACTIVE

        self._retained = tuple(owner for owner in owners if hasattr(owner, _RETAIN))
        for owner in self._retained:
            _notify(owner, _RETAIN)

    @property
    def attached(self) -> bool:
        """True while an ovstage is attached."""
        return self._attached

    @property
    def sensors(self) -> Tuple[Sensor, ...]:
        """Every sensor this instance currently owns."""
        return tuple(self._sensors)

    def _forget(self, sensor: Sensor) -> None:
        if sensor in self._sensors:
            self._sensors.remove(sensor)

    def _forget_pending_mapping(self, mapping: Mapping) -> None:
        if mapping in self._pending_mappings:
            self._pending_mappings.remove(mapping)

    def _op_detail(self, op_id: int) -> str:
        """Read operation detail without allowing diagnostics to replace the failure."""
        try:
            return _b.library().ovsensors_get_last_op_error(self._handle, op_id).text()
        except Exception:
            return ""

    def _require_active(self, operation: str) -> None:
        """Reject wrapper states known to forbid ordinary instance operations."""
        if self._state is _State.CLEANUP_RETRY_ONLY:
            raise OvsensorsError(
                Status.INVALID_HANDLE,
                operation,
                "the instance is destroy-retry-only after a failed destroy_instance; call "
                "destroy() again with nothing else in between",
            )
        if self._state is _State.DEAD:
            raise OvsensorsError(Status.INVALID_HANDLE, operation, "the instance has been destroyed")

    def attach(self, stage: Any) -> None:
        """Attach an ovstage bundle to a detached instance.

        A failed new attachment leaves it detached. If already attached, BUSY preserves
        the existing attachment.

        ``stage`` is an opaque ``ovstage_instance_t*`` the caller obtained elsewhere;
        ovsensors never links ovstage. An owner object is retained after successful
        attachment and released by :meth:`detach` or :meth:`destroy`; a bare address has
        no owner for Python to retain.
        """
        self._require_active("attach_ovstage")
        handle, owner = _b.as_stage_pointer(stage)
        _check(_b.library().ovsensors_attach_ovstage(self._handle, handle).status, "attach_ovstage")
        self._stage_owner = owner
        self._attached = True

    def detach(self) -> None:
        """End this instance's stage borrow, allowing the caller to destroy the stage.

        Successful instance destruction also ends the borrow. Detach is refused with
        BUSY while any sensor, operation, mapping, or reset attempt is live.
        """
        self._require_active("detach_ovstage")
        _check(_b.library().ovsensors_detach_ovstage(self._handle).status, "detach_ovstage")
        self._attached = False
        self._stage_owner = None

    def create_sensor(self, name: str, product: str, *, provider: Any = None) -> Sensor:
        """Adopt a RenderProduct as a sensor, once per provider instance.

        ``provider=None`` elects from applied-schema claims. An explicit provider must
        belong to this instance and is authoritative: no other provider is tried.
        Different providers may adopt the same product; sensor names stay instance-unique.
        Prefer :meth:`sensor`, which destroys it for you.
        """
        self._require_active("create_sensor")
        # Each ovx_string_t owns its UTF-8 buffer outside the struct; retain both objects
        # across the C call instead of relying on ctypes' private keepalive propagation.
        name_view = _b.ovx_string_t(name)
        product_view = _b.ovx_string_t(product)
        # Conversion checks wrapper state and keeps its bundle alive through admission.
        bundle, _owner = _b.as_provider(provider) if provider is not None else (None, None)
        spec = _b.ovsensors_sensor_spec_t(
            name=name_view, product=product_view, provider=ctypes.pointer(bundle) if bundle is not None else None
        )
        handle = ctypes.POINTER(_b.ovsensors_sensor_t)()
        _check(
            _b.library().ovsensors_create_sensor(self._handle, ctypes.byref(spec), ctypes.byref(handle)).status,
            "create_sensor",
        )
        sensor = Sensor(self, handle, name)
        self._sensors.append(sensor)
        return sensor

    @contextmanager
    def sensor(self, name: str, product: str, *, provider: Any = None) -> Iterator[Sensor]:
        """Create a sensor for the duration of a ``with`` block, destroying it on exit.

        With an attached stage and a sealed publication ``ordinal``::

            with inst.sensor("camera", "/World/Render/Product") as camera:
                op = inst.step([camera], delta_time=1.0 / 60.0, ordinal=ordinal)
                with inst.results(op) as results:
                    ...

        Cleanup failures propagate on normal exit and do not replace an exception already
        in flight. Release operations that reference the sensor before leaving the block.
        """
        made = self.create_sensor(name, product, provider=provider)
        try:
            yield made
        except BaseException as propagating:
            _cleanup_without_masking(made.destroy, propagating)
            raise
        made.destroy()

    def step(self, sensors: Sequence[Sensor], delta_time: float, ordinal: int = 0) -> int:
        """Advance simulation by ``delta_time`` seconds and return an operation id.

        Requires an attached stage, finite positive ``delta_time``, and unique live
        sensors from this instance. An empty sensor set advances time without frames;
        history for omitted sensors is discarded. ``ordinal`` is an already-sealed
        committed-publication lower bound, not a historical selector; ovsensors neither
        seals nor replaces it.

        Result groups follow the supplied sensor order. Every returned operation must
        eventually be released with :meth:`release` or :meth:`results`.
        """
        self._require_active("step")
        count = len(sensors)
        array = (ctypes.POINTER(_b.ovsensors_sensor_t) * count)(*[s._handle for s in sensors]) if count else None
        result = _b.library().ovsensors_step(self._handle, array, count, delta_time, _b.check_ordinal(ordinal))
        op_id = int(result.op_index)
        # A rejected step has no operation-scoped error channel.
        _check(result.status, "step", op_id if op_id != _b.OVSENSORS_INVALID_OP_ID else None, self)
        return op_id

    def fetch(self, op_id: int, timeout_ns: int = INFINITE) -> ResultSet:
        """Fetch one operation's result metadata as a snapshot that outlives the operation.

        A group inside the set may have failed without failing the fetch -- check
        :attr:`ResultSet.failures`. ``timeout_ns`` is one total cooperative nanosecond budget,
        including resolution of earlier operations; use ``POLL`` or ``INFINITE`` as
        needed. A spent budget raises :class:`TimeoutError` with ``op_id`` and leaves
        the operation live for retry. Output handles still expire at release.
        """
        self._require_active("fetch_result_set")
        op_id = _b.check_handle(op_id, "op_id")
        raw = _b.ovsensors_step_result_set_t()
        _check(
            _b.library()
            .ovsensors_fetch_result_set(self._handle, op_id, _b.check_timeout(timeout_ns), ctypes.byref(raw))
            .status,
            "fetch_result_set",
            op_id,
            self,
        )
        return ResultSet._from_c(raw, op_id)

    def map_output(
        self, handle: int, device: Device = Device.DEFAULT, timeout_ns: int = INFINITE, *, sync_stream: int = 0
    ) -> Mapping:
        """Map an output with provider-owned device and CUDA stream semantics.

        ``DEFAULT`` preserves the provider's representation, which can be an opaque
        CUDA array. Request ``CPU`` for NumPy or ``CUDA`` for linear device memory.
        ``sync_stream=0`` requests no stream synchronization, ``1`` the default CUDA
        stream, and other values pass the native handle unchanged. Core does not probe
        devices or streams. Prefer :meth:`mapped` for automatic unmap.

        A Python view-construction failure immediately unmaps the native mapping. If
        that cleanup fails, the original exception retains the cleanup error as context
        and this instance retains the mapping for retry by :meth:`destroy`.
        """
        self._require_active("map_output")
        raw = _b.ovsensors_mapped_t()
        description = _b.ovsensors_map_output_description_t(
            device_type=int(device), sync_stream=_b.check_sync_stream(sync_stream)
        )
        mapping = Mapping(self, raw)
        self._pending_mappings.append(mapping)
        try:
            _check(
                _b.library()
                .ovsensors_map_output(
                    self._handle,
                    _b.check_handle(handle, "handle"),
                    ctypes.byref(description),
                    _b.check_timeout(timeout_ns),
                    ctypes.byref(raw),
                )
                .status,
                "map_output",
            )
            mapping._populate()
        except BaseException as propagating:
            try:
                _cleanup_without_masking(mapping.unmap, propagating)
            finally:
                mapping._rollback_pending = mapping._state is _State.ACTIVE
            raise
        self._forget_pending_mapping(mapping)
        return mapping

    @contextmanager
    def mapped(
        self, handle: int, device: Device = Device.DEFAULT, timeout_ns: int = INFINITE, *, sync_stream: int = 0
    ) -> Iterator[Mapping]:
        """Map an output for the duration of a ``with`` block, unmapping on exit.

        Cleanup failures propagate on normal exit and do not replace an exception already
        in flight.
        """
        mapping = self.map_output(handle, device, timeout_ns, sync_stream=sync_stream)
        try:
            yield mapping
        except BaseException as propagating:
            _cleanup_without_masking(mapping.unmap, propagating)
            raise
        mapping.unmap()  # clean exit: an unmap failure is the only news there is

    def release(self, op_id: int) -> None:
        """Retire an operation. Live mappings taken from it stay valid until unmap.

        Pending work returns BUSY before cleanup begins. Once cleanup begins, output
        handles and operation-error detail expire; copied result metadata stays readable.
        A provider cleanup failure requires another release with the same ``op_id``.
        """
        self._require_active("release_result_set")
        op_id = _b.check_handle(op_id, "op_id")
        _check(
            _b.library().ovsensors_release_result_set(self._handle, op_id).status,
            "release_result_set",
            op_id,
            self,
        )

    @contextmanager
    def results(self, op_id: int, timeout_ns: int = INFINITE) -> Iterator[ResultSet]:
        """Fetch a result set and release the operation on exit.

        The yielded snapshot stays readable after exit. Cleanup failures propagate on
        normal exit and do not replace an exception already in flight. If fetching
        fails before entry, the operation remains caller-owned for fetch retry or release.
        """
        result_set = self.fetch(op_id, timeout_ns)
        try:
            yield result_set
        except BaseException as propagating:
            _cleanup_without_masking(lambda: self.release(op_id), propagating)
            raise
        self.release(op_id)  # clean exit: a release failure is the only news there is

    def reset(self, time: float, timeout_ns: int = INFINITE) -> None:
        """Join every provider at the finite simulation-time origin ``time``, in seconds.

        Requires every operation and mapping to be released and no sensor in failed
        cleanup. ``timeout_ns`` is one total cooperative nanosecond budget. A timeout after provider
        work begins leaves a pending reset: retry with the same ``time``. While pending,
        step reports RESET_REQUIRED; sensor creation/destruction, instance destruction,
        and detaching an attached stage report BUSY.

        A terminal provider failure reports OP_FAILED and ends the attempt. A new reset
        may use a different time, but step remains RESET_REQUIRED until reset succeeds.
        Teardown remains available subject to its normal preconditions.
        """
        self._require_active("reset")
        _check(
            _b.library().ovsensors_reset(self._handle, time, _b.check_timeout(timeout_ns)).status,
            "reset",
        )

    def last_op_error(self, op_id: int) -> str:
        """Sticky failure detail, or empty for an unknown or successful operation."""
        self._require_active("get_last_op_error")
        return _b.library().ovsensors_get_last_op_error(self._handle, _b.check_handle(op_id, "op_id")).text()

    def destroy(self) -> None:
        """Destroy the instance and its sensors. Idempotent after destruction.

        Live operations, mappings, or a pending reset must finish before teardown can
        begin. Provider cleanup can also return BUSY after teardown has begun; then only
        another destroy call is permitted. The wrapper cannot distinguish these BUSY
        outcomes. Other native cleanup failures likewise require a destroy retry.
        Provider owners and the attached stage remain retained until destruction succeeds.
        Do not call from a log callback.

        Cleanup retained after failed Python mapping construction is retried first.
        Failure there leaves instance teardown unstarted. A mapping still being
        constructed, including its immediate cleanup attempt, also makes destruction BUSY.
        """
        if self._state is _State.DEAD:
            return
        while self._pending_mappings:
            pending = tuple(self._pending_mappings)
            if any(not mapping._rollback_pending for mapping in pending):
                raise OvsensorsError(Status.BUSY, "destroy_instance", "a mapping is still being constructed")
            for mapping in pending:
                try:
                    mapping.unmap()
                except OvsensorsError as error:
                    if error.status is not Status.INVALID_HANDLE:
                        raise
        status = int(_b.library().ovsensors_destroy_instance(self._handle).status)
        if status in (Status.SUCCESS, Status.INVALID_HANDLE):
            self._state = _State.DEAD
            for sensor in self._sensors:
                sensor._state = _State.DEAD
            self._sensors.clear()
            self._attached = False
            self._stage_owner = None
            # Successful destruction ends all routing through retained provider owners.
            for owner in self._retained:
                _notify(owner, _RELEASE)
            self._retained = ()
            self._owners = ()
        elif status not in (Status.BUSY, Status.ERROR):
            self._state = _State.CLEANUP_RETRY_ONLY
        _check(status, "destroy_instance")

    def __enter__(self) -> "Instance":
        return self

    def __exit__(self, exc_type: Any, exc: Optional[BaseException], tb: Any) -> None:
        """Destroy on exit without replacing an exception already in flight."""
        if exc is None:
            self.destroy()
            return
        _cleanup_without_masking(self.destroy, exc)

    def __del__(self) -> None:
        """Best-effort destroy and warn when a live instance is dropped.

        No C call is made during interpreter finalization, when provider teardown order is
        unknown. Cleanup is attempted before warning because warnings may be escalated.
        """
        state = getattr(self, "_state", None)
        if state is None or state is _State.DEAD:
            return
        if sys.is_finalizing():
            _warn_dropped_resource(
                "Instance was still live at interpreter shutdown; NO destroy was attempted, "
                "because teardown calls into every provider and one may already have been freed. "
                "Use a 'with' block or call destroy() explicitly."
            )
            return
        try:
            self.destroy()
        except Exception:
            pass
        _warn_dropped_resource(
            "Instance was garbage-collected without destroy(); issued a best-effort destroy "
            "to release the instance, its providers and any attached stage. Use a 'with' "
            "block or call destroy() explicitly."
        )

    def __repr__(self) -> str:
        if self._state is _State.DEAD:
            return "Instance(<destroyed>)"
        if self._state is _State.CLEANUP_RETRY_ONLY:
            return "Instance(<destroy-retry-only>)"
        return f"Instance(sensors={len(self._sensors)}, attached={self._attached})"
