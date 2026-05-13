"""Phase-space distributions for finite Fock-space states."""

from __future__ import annotations

from collections.abc import Sequence
from math import pi
from typing import cast

import numpy as np
from numpy.typing import NDArray
from scipy.special import eval_genlaguerre, gammaln  # type: ignore[import-untyped]

Array = NDArray[np.complex128]
FloatArray = NDArray[np.float64]


def wigner(
    state: Array,
    xvec: Sequence[float] | FloatArray,
    pvec: Sequence[float] | FloatArray | None = None,
) -> FloatArray:
    """Evaluate the Wigner function on an ``x``/``p`` grid.

    The state may be a ket or density matrix in a truncated Fock basis. The
    returned array has shape ``(len(pvec), len(xvec))``.
    """
    rho = _density_matrix(state)
    x, p = _phase_space_axes(xvec, pvec)
    x_grid, p_grid = np.meshgrid(x, p)
    alpha = cast(Array, (x_grid + 1j * p_grid) / np.sqrt(2.0))
    radius2 = np.abs(alpha) ** 2
    gaussian = np.exp(-2.0 * radius2) / pi
    values = np.zeros_like(radius2, dtype=np.float64)

    dim = rho.shape[0]
    laguerre_arg = 4.0 * radius2
    for m in range(dim):
        diag = rho[m, m]
        if abs(diag) > 0:
            element = ((-1) ** m) * eval_genlaguerre(m, 0, laguerre_arg) * gaussian
            values += float(diag.real) * element
        for n in range(m):
            coefficient = rho[n, m]
            if abs(coefficient) == 0:
                continue
            element = _wigner_basis_element(m, n, alpha, laguerre_arg, gaussian)
            values += 2.0 * np.real(coefficient * element)
    return values


def q_function(
    state: Array,
    xvec: Sequence[float] | FloatArray,
    pvec: Sequence[float] | FloatArray | None = None,
) -> FloatArray:
    """Evaluate the Husimi-Q function on an ``x``/``p`` grid.

    The convention is ``Q(alpha) = <alpha|rho|alpha> / pi`` with
    ``alpha = (x + i p) / sqrt(2)``.
    """
    rho = _density_matrix(state)
    x, p = _phase_space_axes(xvec, pvec)
    x_grid, p_grid = np.meshgrid(x, p)
    alpha = cast(Array, (x_grid + 1j * p_grid) / np.sqrt(2.0))
    coherent_values = _coherent_grid(alpha, rho.shape[0])
    values = np.einsum(
        "nxy,nm,mxy->xy",
        coherent_values.conj(),
        rho,
        coherent_values,
        optimize=True,
    )
    return cast(FloatArray, np.real(values) / pi)


def phase_space_grid(
    *,
    xlim: tuple[float, float] = (-5.0, 5.0),
    plim: tuple[float, float] | None = None,
    points: int = 201,
) -> tuple[FloatArray, FloatArray]:
    """Return equally spaced ``x`` and ``p`` axes for phase-space plots."""
    if points <= 1:
        msg = "points must be greater than 1."
        raise ValueError(msg)
    if xlim[0] >= xlim[1]:
        msg = "xlim must be ordered as (min, max)."
        raise ValueError(msg)
    p_limits = xlim if plim is None else plim
    if p_limits[0] >= p_limits[1]:
        msg = "plim must be ordered as (min, max)."
        raise ValueError(msg)
    return (
        np.linspace(xlim[0], xlim[1], points, dtype=np.float64),
        np.linspace(p_limits[0], p_limits[1], points, dtype=np.float64),
    )


def _wigner_basis_element(
    m: int,
    n: int,
    alpha: Array,
    laguerre_arg: FloatArray,
    gaussian: FloatArray,
) -> Array:
    order = m - n
    ratio = np.exp(0.5 * (gammaln(n + 1) - gammaln(m + 1)))
    values = (
        ((-1) ** n)
        * ratio
        * (2.0 * alpha) ** order
        * eval_genlaguerre(n, order, laguerre_arg)
        * gaussian
    )
    return cast(Array, values)


def _coherent_grid(alpha: Array, dim: int) -> Array:
    values = np.empty((dim, *alpha.shape), dtype=np.complex128)
    values[0] = np.exp(-0.5 * np.abs(alpha) ** 2)
    for n in range(1, dim):
        values[n] = values[n - 1] * alpha / np.sqrt(float(n))
    return values


def _density_matrix(state: Array) -> Array:
    array = np.asarray(state, dtype=np.complex128)
    if array.ndim == 1:
        return np.outer(array, array.conj())
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        msg = "state must be a ket vector or square density matrix."
        raise ValueError(msg)
    return array


def _phase_space_axes(
    xvec: Sequence[float] | FloatArray,
    pvec: Sequence[float] | FloatArray | None,
) -> tuple[FloatArray, FloatArray]:
    x = np.asarray(xvec, dtype=np.float64)
    p = np.asarray(xvec if pvec is None else pvec, dtype=np.float64)
    _validate_axis(x, "xvec")
    _validate_axis(p, "pvec")
    return x, p


def _validate_axis(axis: FloatArray, name: str) -> None:
    if axis.ndim != 1 or len(axis) == 0:
        msg = f"{name} must be a non-empty one-dimensional sequence."
        raise ValueError(msg)
    if len(axis) > 1 and np.any(np.diff(axis) <= 0):
        msg = f"{name} must be sorted in strictly ascending order."
        raise ValueError(msg)
