# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Python wrapper for the shipped ovrtx-backed provider.

Example::

    import ovsensors
    from ovsensors.providers import ovrtx as ovrtx_provider

    with ovrtx_provider.provider() as provider:
        with ovsensors.Instance([provider]) as inst:
            inst.attach(stage)

Importing this module loads neither native library nor the ovrtx Python package.
The provider's capabilities are fixed; its vtable and context remain opaque to Python.
See :class:`Provider` for configuration and loader selection.
"""

import ctypes
import importlib.util
import os
import sys
import threading
from pathlib import Path
from typing import Any, List, Literal, Optional, Union

from .._src import bindings as _b
from .._src import instance as _instance
from .._src.bindings import ovsensors_provider_t, ovx_string_t
from .._src.types import OvsensorsError, Status

__all__ = [
    "OVSENSORS_PROVIDER_OVRTX_LIBRARY_PATH_HINT",
    "Provider",
    "library",
    "library_path",
    "provider",
]


# ---------------------------------------------------------------------------------
# Library loading
# ---------------------------------------------------------------------------------

OVSENSORS_PROVIDER_OVRTX_LIBRARY_PATH_HINT: Optional[str] = None

_LIBRARY_STEM = "ovsensors_provider_ovrtx"


def _library_filename() -> str:
    if sys.platform == "win32":
        return f"{_LIBRARY_STEM}.dll"
    if sys.platform == "darwin":
        return f"lib{_LIBRARY_STEM}.dylib"
    return f"lib{_LIBRARY_STEM}.so"


def _search_directories() -> List[Path]:
    """Provider-library search path, in precedence order."""
    candidates: List[Path] = []
    if OVSENSORS_PROVIDER_OVRTX_LIBRARY_PATH_HINT:
        candidates.append(Path(OVSENSORS_PROVIDER_OVRTX_LIBRARY_PATH_HINT))
    env = os.environ.get("OVSENSORS_PROVIDER_OVRTX_LIBRARY_PATH")
    if env:
        candidates.extend(Path(p) for p in env.split(os.pathsep) if p)
    # The provider normally ships beside core; the provider-specific setting still wins.
    core_env = os.environ.get("OVSENSORS_LIBRARY_PATH")
    if core_env:
        candidates.extend(Path(p) for p in core_env.split(os.pathsep) if p)
    # Do not force core to load merely to discover its directory.
    core_path = _b.library_path()
    if core_path is not None:
        candidates.append(core_path.parent)
    here = Path(__file__).resolve()
    candidates.append(here.parent)
    candidates.append(here.parents[1])
    # Wheel layout: ovsensors/bin/ beside the Python modules.
    candidates.append(here.parents[1] / "bin")
    # Installed layout: <prefix>/python/ovsensors/providers/ and <prefix>/bin/.
    if len(here.parents) > 3:
        candidates.append(here.parents[3] / "bin")
    return _b._existing_dirs(candidates)


class _LibraryLoader:
    """Load the provider library lazily and publish it only after declaring its ABI."""

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
                f"could not load {filename}. Set OVSENSORS_PROVIDER_OVRTX_LIBRARY_PATH or "
                f"ovsensors.providers.ovrtx.OVSENSORS_PROVIDER_OVRTX_LIBRARY_PATH_HINT.\n"
                f"Tried:\n  {hint}\n{exc}"
            ) from exc


def _declare_prototypes(lib: ctypes.CDLL) -> None:
    """Declare the two exported entry points with their complete C signatures.

    The config stays opaque here: ovrtx bindings own its layout and conversion.
    ``None`` is passed as NULL; an explicit config borrows ovrtx-owned storage only
    for the duration of the factory call.
    """
    lib.ovsensors_provider_ovrtx_create.argtypes = [
        ovx_string_t,  # ovrtx_library_path, by value
        ctypes.c_void_p,  # const ovrtx_config_t*; NULL selects ovrtx defaults
        ctypes.POINTER(ovsensors_provider_t),  # out_provider
    ]
    lib.ovsensors_provider_ovrtx_create.restype = ctypes.c_int

    lib.ovsensors_provider_ovrtx_destroy.argtypes = [ctypes.POINTER(ovsensors_provider_t)]
    lib.ovsensors_provider_ovrtx_destroy.restype = ctypes.c_int


_loader = _LibraryLoader()


def library() -> ctypes.CDLL:
    """The loaded ovrtx provider shared library, with both prototypes declared."""
    return _loader.lib


def library_path() -> Optional[Path]:
    """Absolute loaded-provider path, or None if unloaded or the OS-selected path is unknown."""
    return _loader.path()


# ---------------------------------------------------------------------------------
# The provider
# ---------------------------------------------------------------------------------


def _resolve_ovrtx_library_path(path: Optional[Union[str, os.PathLike[str]]]) -> Path:
    """Select one loader without importing ovrtx or loading any native library."""
    if path is not None:
        value = os.fspath(path)
        if not isinstance(value, str):
            raise TypeError("ovrtx_library_path must be str or os.PathLike[str]")
        if not value or "\0" in value:
            raise ValueError("ovrtx_library_path must be nonempty and contain no NUL characters")
        return Path(value).resolve()

    remedy = "Pass ovrtx_library_path explicitly to the ovrtx loader binary."
    # Inspect only modules already imported by the application. Importing ovrtx can
    # register USD schemas and load a native library when its opt-in is enabled.
    bindings = sys.modules.get("ovrtx._src.bindings")
    loader = getattr(bindings, "_ovrtx_loader", None)
    schema_paths = sys.modules.get("ovrtx._src.schema_paths")
    selected_paths = set()
    for loaded in (getattr(loader, "_lib", None), getattr(schema_paths, "_loaded_lib", None)):
        if loaded is not None:
            selected = Path(os.fsdecode(loaded._name))
            if not selected.is_absolute() or not selected.is_file():
                raise RuntimeError(f"Cannot resolve the selected ovrtx loader {selected!s}. {remedy}")
            selected_paths.add(selected.resolve())
    if len(selected_paths) > 1:
        found = ", ".join(str(selected) for selected in sorted(selected_paths))
        raise RuntimeError(f"Conflicting selected ovrtx loader paths: {found}. {remedy}")
    if selected_paths:
        return selected_paths.pop()

    # A top-level spec locates the package selected by Python without executing it.
    # Preserve package symlinks until after constructing staged-layout candidates.
    try:
        spec = importlib.util.find_spec("ovrtx")
    except (ImportError, ValueError) as exc:
        raise RuntimeError(f"Cannot locate the selected ovrtx Python package. {remedy}") from exc
    filename = "ovrtx-dynamic.dll" if sys.platform == "win32" else "libovrtx-dynamic.so"
    candidates = set()
    locations = (spec.submodule_search_locations or ()) if spec is not None else ()
    for location in locations:
        package = Path(location)
        for directory in (package / "bin", package.parent.parent / "bin"):
            candidate = directory / filename
            if candidate.is_file():
                candidates.add(candidate.resolve())
    if len(candidates) == 1:
        return candidates.pop()
    if candidates:
        found = ", ".join(str(candidate) for candidate in sorted(candidates))
        raise RuntimeError(f"Ambiguous ovrtx loader paths: {found}. {remedy}")
    raise RuntimeError(f"No ovrtx loader found in the selected ovrtx Python package. {remedy}")


def _to_c_config(config: Any) -> Any:
    """Borrow ovrtx's converter without declaring any ovrtx struct in this package."""
    if config is None:
        return None

    try:
        from ovrtx import Renderer, RendererConfig
    except ImportError as exc:
        raise ImportError("explicit config requires compatible ovrtx Python bindings") from exc

    if not isinstance(config, RendererConfig):
        raise TypeError(
            f"config must be None or ovrtx.RendererConfig, got {type(config).__name__}. "
            "Pass renderer.config to copy its settings; the provider creates its own renderer."
        )
    return Renderer._to_c_config(config)


class Provider:
    """Own an ovrtx provider bundle and the renderer created by its C factory.

    Pass this object to :class:`ovsensors.Instance` to prevent destruction while the
    instance is live. Passing the raw ``bundle`` requires caller-managed lifetime.
    See :meth:`destroy` for cleanup outcomes.
    """

    def __init__(
        self,
        config: Any = None,
        *,
        ovrtx_library_path: Optional[Union[str, os.PathLike[str]]] = None,
    ) -> None:
        """Create a provider with default or explicit ovrtx configuration.

        ``config`` accepts ``None`` for ovrtx defaults or an ``ovrtx.RendererConfig``
        from compatible ovrtx bindings, including ``renderer.config``. ovrtx owns
        conversion and validation; the converted entries are forwarded unchanged.

        ``ovrtx_library_path`` accepts a nonempty str or ``os.PathLike[str]`` with no NUL
        characters; relative paths resolve from the current directory. If omitted,
        reuse the loader selected by ovrtx rendering/schema registration or find a
        unique loader in its Python package's wheel/staged layout. Discovery neither
        searches PATH nor loads ovrtx. Missing, conflicting or ambiguous defaults
        raise RuntimeError. The native factory reports ovrtx loading or renderer-creation
        failures as OvsensorsError.
        """
        c_config = _to_c_config(config)

        c_library_path = _b.ovx_string_t(str(_resolve_ovrtx_library_path(ovrtx_library_path)))

        self.bundle = ovsensors_provider_t()
        self._destroyed = False
        self._retained_instance_count = 0

        # Keep the CDLL alive until destroy() completes.
        self._lib: Optional[ctypes.CDLL] = library()

        # Keep the path and ovrtx-owned config storage alive through the call.
        # The native factory checks runtime generation before reading the config.
        status = int(
            self._lib.ovsensors_provider_ovrtx_create(
                c_library_path,
                None if c_config is None else ctypes.byref(c_config),
                ctypes.byref(self.bundle),
            )
        )
        if status != Status.SUCCESS:
            raise OvsensorsError(Status(status), "provider_ovrtx_create")

    @property
    def destroyed(self) -> bool:
        """True after the provider bundle is consumed; further destruction is a no-op."""
        return self._destroyed

    @property
    def retained_instance_count(self) -> int:
        """Number of live Python instances created with this Provider object."""
        return self._retained_instance_count

    def _ovsensors_retain(self) -> None:
        self._retained_instance_count += 1

    def _ovsensors_release(self) -> None:
        if self._retained_instance_count > 0:
            self._retained_instance_count -= 1

    def destroy(self) -> None:
        """Destroy the provider and its renderer.

        Raises OvsensorsError with BUSY without mutation while a Python instance
        created with this Provider object is live; destroy those instances first.
        Native cleanup failure raises OvsensorsError after consuming the bundle:
        ``destroyed`` is true and cleanup cannot be retried. Repeated calls are no-ops.
        """
        if self._destroyed:
            return

        if self._retained_instance_count > 0:
            held_instance_count = self._retained_instance_count
            holders = "instance still routes" if held_instance_count == 1 else "instances still route"
            raise OvsensorsError(
                Status.BUSY,
                "provider_ovrtx_destroy",
                f"{held_instance_count} ovsensors {holders} through this provider. Destroy the instance "
                f"first -- ovsensors.h orders destroy_instance before the provider's own destroy.",
            )

        lib = self._lib
        status = int(Status.SUCCESS)
        if lib is not None:
            status = int(lib.ovsensors_provider_ovrtx_destroy(ctypes.byref(self.bundle)))

        # The C destroy consumes the bundle even when it reports failure.
        self._destroyed = True
        # Leave a safely rejectable NULL bundle rather than dangling pointers.
        self.bundle.vtable = None
        self.bundle.context = None
        self._lib = None

        if status != Status.SUCCESS:
            raise OvsensorsError(Status(status), "provider_ovrtx_destroy")

    def __enter__(self) -> "Provider":
        return self

    def __exit__(self, exc_type: Any, exc: Optional[BaseException], tb: Any) -> Literal[False]:
        """Destroy on exit without replacing an exception already in flight."""
        if exc is None:
            self.destroy()
            return False
        _instance._cleanup_without_masking(self.destroy, exc)
        return False

    # No finalizer: interpreter shutdown may unload the CDLL before cleanup can call it.

    def __repr__(self) -> str:
        state = "destroyed" if self._destroyed else "live"
        if self._retained_instance_count:
            state = f"{state}, retained by {self._retained_instance_count}"
        return f"Provider({state})"


def provider(
    config: Any = None,
    *,
    ovrtx_library_path: Optional[Union[str, os.PathLike[str]]] = None,
) -> Provider:
    """Create a :class:`Provider` with the same configuration and loader-path arguments."""
    return Provider(config, ovrtx_library_path=ovrtx_library_path)
