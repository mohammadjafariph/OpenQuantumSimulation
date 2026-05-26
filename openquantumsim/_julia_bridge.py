"""Lazy Julia backend bridge.

The bridge intentionally imports `juliacall` lazily so ordinary Python-side
operator work and tests do not pay Julia startup cost.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any, TypedDict

SYSIMAGE_ENV = "OPENQUANTUMSIM_JULIA_SYSIMAGE"
USE_SYSIMAGE_ENV = "OPENQUANTUMSIM_USE_SYSIMAGE"
CACHE_DIR_ENV = "OPENQUANTUMSIM_CACHE_DIR"
JULIACALL_SYSIMAGE_ENV = "PYTHON_JULIACALL_SYSIMAGE"


class SysimageMetadata(TypedDict):
    """Metadata stored next to the user-built Julia sysimage."""

    schema_version: int
    sysimage_path: str
    backend_path: str
    backend_fingerprint: str
    julia_executable: str
    julia_version: str
    platform: str
    machine: str
    created_at: str


class JuliaBridgeUnavailable(RuntimeError):
    """Raised when the Julia backend cannot be loaded."""


_JL: Any | None = None
_BACKEND: Any | None = None


def cache_dir() -> Path:
    """Return the OpenQuantumSim user cache directory."""
    configured = os.environ.get(CACHE_DIR_ENV)
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".cache" / "openquantumsim"


def sysimage_metadata_path() -> Path:
    """Return the metadata path for an auto-discovered sysimage."""
    return cache_dir() / "sysimage.json"


def read_sysimage_metadata() -> SysimageMetadata | None:
    """Read cached sysimage metadata if it exists and is well-formed."""
    path = sysimage_metadata_path()
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    required = set(SysimageMetadata.__annotations__)
    if not isinstance(data, dict) or not required.issubset(data):
        return None
    return data  # type: ignore[return-value]


def write_sysimage_metadata(metadata: SysimageMetadata) -> None:
    """Write metadata for the latest user-built sysimage."""
    path = sysimage_metadata_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")


def backend_path() -> Path:
    """Return the Julia backend package path."""
    package_path = Path(__file__).resolve().parent / "julia" / "OpenQuantumSimJL"
    dev_path = Path(__file__).resolve().parents[1] / "src" / "OpenQuantumSimJL"
    if dev_path.exists():
        return dev_path
    return package_path


def backend_fingerprint() -> str:
    """Return a content hash for the packaged Julia backend."""
    root = backend_path()
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix in {".jl", ".toml"}:
            digest.update(str(path.relative_to(root)).encode("utf-8"))
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
    return digest.hexdigest()


def configured_sysimage_path() -> Path | None:
    """Return the sysimage path that should be passed to JuliaCall, if any."""
    explicit = os.environ.get(SYSIMAGE_ENV)
    if explicit:
        path = Path(explicit).expanduser()
        return path if path.is_file() else None

    if os.environ.get(USE_SYSIMAGE_ENV, "1").lower() in {"0", "false", "no", "off"}:
        return None

    metadata = read_sysimage_metadata()
    if metadata is None or not _metadata_matches_current_backend(metadata):
        return None

    path = Path(metadata["sysimage_path"]).expanduser()
    return path if path.is_file() else None


def active_sysimage_path() -> Path | None:
    """Return the JuliaCall sysimage path configured for this process."""
    path = os.environ.get(JULIACALL_SYSIMAGE_ENV)
    return Path(path) if path else None


def _metadata_matches_current_backend(metadata: SysimageMetadata) -> bool:
    return (
        metadata["schema_version"] == 1
        and metadata["backend_fingerprint"] == backend_fingerprint()
        and metadata["platform"] == sys.platform
        and metadata["machine"] == platform.machine()
        and _juliapkg_meta_matches(metadata)
    )


def _juliapkg_meta_matches(metadata: SysimageMetadata) -> bool:
    """Check the Julia executable/version recorded by juliapkg when available."""
    try:
        from juliapkg.state import STATE  # type: ignore[import-untyped]
    except Exception:
        return True

    meta_path = Path(str(STATE.get("meta", "")))
    if not meta_path.is_file():
        return True
    try:
        juliapkg_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return True
    executable = str(juliapkg_meta.get("executable", ""))
    version = str(juliapkg_meta.get("version", ""))
    return (
        executable == metadata["julia_executable"]
        and version == metadata["julia_version"]
    )


def _maybe_configure_sysimage() -> None:
    """Set JuliaCall's sysimage option before JuliaCall is imported."""
    if JULIACALL_SYSIMAGE_ENV in os.environ:
        return
    path = configured_sysimage_path()
    if path is not None:
        os.environ[JULIACALL_SYSIMAGE_ENV] = str(path)


def get_julia() -> Any:
    """Return the `juliacall.Main` object, importing it on first use."""
    global _JL
    if _JL is not None:
        return _JL
    _maybe_configure_sysimage()
    try:
        from juliacall import Main as jl  # type: ignore[import-untyped]
    except Exception as exc:  # pragma: no cover - depends on local Julia setup
        msg = "juliacall is required to use the Julia backend."
        raise JuliaBridgeUnavailable(msg) from exc
    _JL = jl
    return jl


def load_backend(*, instantiate: bool = False) -> Any:
    """Activate and load the `OpenQuantumSimJL` backend module."""
    global _BACKEND
    if _BACKEND is not None:
        return _BACKEND

    jl = get_julia()
    path = str(backend_path())
    try:
        jl.seval("using Pkg")
        jl.Pkg.activate(path, io=jl.devnull)
        _load_backend_module(jl, instantiate=instantiate)
        _BACKEND = jl.OpenQuantumSimJL
    except Exception as exc:  # pragma: no cover - depends on local Julia setup
        msg = f"Unable to load Julia backend from {path}."
        raise JuliaBridgeUnavailable(msg) from exc
    return _BACKEND


def _load_backend_module(jl: Any, *, instantiate: bool) -> None:
    """Load the backend, instantiating only when requested or needed."""
    if instantiate:
        _instantiate_and_load_backend(jl)
        return

    try:
        jl.seval("using OpenQuantumSimJL")
    except Exception:
        _instantiate_and_load_backend(jl)


def _instantiate_and_load_backend(jl: Any) -> None:
    """Instantiate/load the backend, resolving stale manifests on retry."""
    try:
        jl.Pkg.instantiate(io=jl.devnull)
        jl.seval("using OpenQuantumSimJL")
    except Exception:
        jl.Pkg.resolve(io=jl.devnull)
        jl.Pkg.instantiate(io=jl.devnull)
        jl.seval("using OpenQuantumSimJL")


def backend_available() -> bool:
    """Return whether the Julia backend can be loaded."""
    try:
        load_backend()
    except JuliaBridgeUnavailable:
        return False
    return True
