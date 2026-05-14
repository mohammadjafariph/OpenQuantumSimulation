"""Lazy Julia backend bridge.

The bridge intentionally imports `juliacall` lazily so ordinary Python-side
operator work and tests do not pay Julia startup cost.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


class JuliaBridgeUnavailable(RuntimeError):
    """Raised when the Julia backend cannot be loaded."""


_JL: Any | None = None
_BACKEND: Any | None = None


def backend_path() -> Path:
    """Return the Julia backend package path."""
    package_path = Path(__file__).resolve().parent / "julia" / "OpenQuantumSimJL"
    dev_path = Path(__file__).resolve().parents[1] / "src" / "OpenQuantumSimJL"
    if dev_path.exists():
        return dev_path
    return package_path


def get_julia() -> Any:
    """Return the `juliacall.Main` object, importing it on first use."""
    global _JL
    if _JL is not None:
        return _JL
    try:
        from juliacall import Main as jl  # type: ignore[import-untyped]
    except Exception as exc:  # pragma: no cover - depends on local Julia setup
        msg = "juliacall is required to use the Julia backend."
        raise JuliaBridgeUnavailable(msg) from exc
    _JL = jl
    return jl


def load_backend() -> Any:
    """Activate and load the `OpenQuantumSimJL` backend module."""
    global _BACKEND
    if _BACKEND is not None:
        return _BACKEND

    jl = get_julia()
    path = str(backend_path())
    try:
        jl.seval("using Pkg")
        jl.Pkg.activate(path)
        _instantiate_and_load_backend(jl)
        _BACKEND = jl.OpenQuantumSimJL
    except Exception as exc:  # pragma: no cover - depends on local Julia setup
        msg = f"Unable to load Julia backend from {path}."
        raise JuliaBridgeUnavailable(msg) from exc
    return _BACKEND


def _instantiate_and_load_backend(jl: Any) -> None:
    """Instantiate/load the backend, resolving stale manifests on retry."""
    try:
        jl.Pkg.instantiate()
        jl.seval("using OpenQuantumSimJL")
    except Exception:
        jl.Pkg.resolve()
        jl.Pkg.instantiate()
        jl.seval("using OpenQuantumSimJL")


def backend_available() -> bool:
    """Return whether the Julia backend can be loaded."""
    try:
        load_backend()
    except JuliaBridgeUnavailable:
        return False
    return True
