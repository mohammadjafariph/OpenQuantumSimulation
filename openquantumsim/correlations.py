"""Two-time correlation functions via the quantum regression theorem."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import numpy as np
from numpy.typing import NDArray

from ._julia_bridge import JuliaBridgeUnavailable, load_backend
from .operators import Operator
from .result import Options

Array = NDArray[np.complex128]
FloatArray = NDArray[np.float64]


def correlation_2op_1t(
    H: Operator,
    rho0: Array,
    taulist: Sequence[float],
    a_op: Operator,
    b_op: Operator,
    *,
    c_ops: Sequence[Operator] | None = None,
    options: Options | None = None,
) -> Array:
    """Return ``<A(tau) B(0)>`` using quantum regression.

    This first implementation supports time-independent Lindblad systems.
    """
    opts = options or Options()
    rho0_array = np.asarray(rho0, dtype=np.complex128)
    taus = np.asarray(taulist, dtype=np.float64)
    c_arrays = [op.to_numpy() for op in c_ops or []]
    _validate_correlation_inputs(H, rho0_array, taus, a_op, b_op, c_arrays, "taulist")

    try:
        backend = load_backend()
    except JuliaBridgeUnavailable as exc:
        msg = "correlation_2op_1t requires the Julia backend; run setup_julia.py first."
        raise NotImplementedError(msg) from exc

    raw = backend.correlation_2op_1t(
        H.to_numpy(),
        rho0_array,
        taus,
        a_op.to_numpy(),
        b_op.to_numpy(),
        c_arrays,
        rtol=float(opts.rtol),
        atol=float(opts.atol),
        method=str(opts.method),
        krylov_dim=int(opts.krylov_dim),
    )
    return np.asarray(_field(raw, "correlations"), dtype=np.complex128)


def correlation_2op_2t(
    H: Operator,
    rho0: Array,
    tlist: Sequence[float],
    taulist: Sequence[float],
    a_op: Operator,
    b_op: Operator,
    *,
    c_ops: Sequence[Operator] | None = None,
    options: Options | None = None,
) -> Array:
    """Return ``<A(t + tau) B(t)>`` for each ``t`` and ``tau``."""
    opts = options or Options()
    rho0_array = np.asarray(rho0, dtype=np.complex128)
    times = np.asarray(tlist, dtype=np.float64)
    taus = np.asarray(taulist, dtype=np.float64)
    c_arrays = [op.to_numpy() for op in c_ops or []]
    _validate_correlation_inputs(H, rho0_array, times, a_op, b_op, c_arrays, "tlist")
    _validate_nonnegative_sorted_times(taus, "taulist")

    try:
        backend = load_backend()
    except JuliaBridgeUnavailable as exc:
        msg = "correlation_2op_2t requires the Julia backend; run setup_julia.py first."
        raise NotImplementedError(msg) from exc

    raw = backend.correlation_2op_2t(
        H.to_numpy(),
        rho0_array,
        times,
        taus,
        a_op.to_numpy(),
        b_op.to_numpy(),
        c_arrays,
        rtol=float(opts.rtol),
        atol=float(opts.atol),
        method=str(opts.method),
        krylov_dim=int(opts.krylov_dim),
    )
    return np.asarray(_field(raw, "correlations"), dtype=np.complex128)


def _validate_correlation_inputs(
    H: Operator,
    rho0: Array,
    times: FloatArray,
    a_op: Operator,
    b_op: Operator,
    c_ops: Sequence[Array],
    time_name: str,
) -> None:
    if H.shape[0] != H.shape[1]:
        msg = "H must be square."
        raise ValueError(msg)
    if rho0.shape != H.shape:
        msg = "rho0 must have the same shape as H."
        raise ValueError(msg)
    _validate_nonnegative_sorted_times(times, time_name)
    for op_name, operator in (("a_op", a_op), ("b_op", b_op)):
        if operator.shape != H.shape:
            msg = f"{op_name} must have the same shape as H."
            raise ValueError(msg)
    for collapse in c_ops:
        if collapse.shape != H.shape:
            msg = "collapse operators must have the same shape as H."
            raise ValueError(msg)


def _validate_nonnegative_sorted_times(times: FloatArray, name: str) -> None:
    if times.ndim != 1 or len(times) == 0:
        msg = f"{name} must be a non-empty one-dimensional sequence."
        raise ValueError(msg)
    if np.any(times < 0):
        msg = f"{name} must be non-negative."
        raise ValueError(msg)
    if np.any(np.diff(times) < 0):
        msg = f"{name} must be sorted in ascending order."
        raise ValueError(msg)


def _field(raw: object, name: str) -> object:
    try:
        return getattr(raw, name)
    except AttributeError:
        return cast(object, raw[name])  # type: ignore[index]
