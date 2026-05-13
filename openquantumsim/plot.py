"""Plotting helpers for solver outputs and phase-space distributions."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .phase_space import phase_space_grid, q_function, wigner
from .result import Result

Array = NDArray[np.complex128]
FloatArray = NDArray[np.float64]
AxisInput = Sequence[float] | FloatArray
SeriesInput = Sequence[complex | float] | NDArray[Any]


def expect_plot(
    times: AxisInput,
    values: SeriesInput,
    *,
    ax: Any | None = None,
    label: str | None = None,
    ylabel: str = "expectation",
) -> Any:
    """Plot one expectation-value time series."""
    axis = _axis(ax)
    series = np.asarray(values)
    axis.plot(times, series.real, label=label)
    axis.set_xlabel("time")
    axis.set_ylabel(ylabel)
    if label is not None:
        axis.legend()
    return axis


def plot_expectations(
    result: Result,
    *,
    ax: Any | None = None,
    labels: Sequence[str] | None = None,
    ylabel: str = "expectation",
) -> Any:
    """Plot all expectation-value series stored in a solver result."""
    axis = _axis(ax)
    for idx, values in enumerate(result.expect):
        label = labels[idx] if labels is not None and idx < len(labels) else None
        series = np.asarray(values)
        axis.plot(result.times, series.real, label=label)
    axis.set_xlabel("time")
    axis.set_ylabel(ylabel)
    if labels is not None:
        axis.legend()
    return axis


def plot_state_observable(
    result: Result,
    name: str,
    *,
    ax: Any | None = None,
    ylabel: str | None = None,
) -> Any:
    """Plot a named state-observable series from a solver result."""
    if name not in result.state_observables:
        msg = f"Result has no state observable named {name!r}."
        raise KeyError(msg)
    return expect_plot(
        result.times,
        result.state_observables[name],
        ax=ax,
        label=name,
        ylabel=name if ylabel is None else ylabel,
    )


def plot_wigner(
    state: Array,
    xvec: AxisInput | None = None,
    pvec: AxisInput | None = None,
    *,
    ax: Any | None = None,
    points: int = 201,
    xlim: tuple[float, float] = (-5.0, 5.0),
    plim: tuple[float, float] | None = None,
    cmap: str = "RdBu_r",
    colorbar: bool = True,
) -> Any:
    """Plot a Wigner function as a diverging phase-space heatmap."""
    x, p = _axes_or_default(xvec, pvec, xlim=xlim, plim=plim, points=points)
    values = wigner(state, x, p)
    vmax = float(np.max(np.abs(values))) if values.size else 1.0
    return plot_phase_space(
        values,
        x,
        p,
        ax=ax,
        cmap=cmap,
        colorbar=colorbar,
        vmin=-vmax,
        vmax=vmax,
        title="Wigner function",
    )


def plot_q_function(
    state: Array,
    xvec: AxisInput | None = None,
    pvec: AxisInput | None = None,
    *,
    ax: Any | None = None,
    points: int = 201,
    xlim: tuple[float, float] = (-5.0, 5.0),
    plim: tuple[float, float] | None = None,
    cmap: str = "viridis",
    colorbar: bool = True,
) -> Any:
    """Plot a Husimi-Q function as a positive phase-space heatmap."""
    x, p = _axes_or_default(xvec, pvec, xlim=xlim, plim=plim, points=points)
    values = q_function(state, x, p)
    return plot_phase_space(
        values,
        x,
        p,
        ax=ax,
        cmap=cmap,
        colorbar=colorbar,
        vmin=0.0,
        title="Husimi Q function",
    )


def plot_phase_space(
    values: FloatArray,
    xvec: AxisInput,
    pvec: AxisInput,
    *,
    ax: Any | None = None,
    cmap: str = "viridis",
    colorbar: bool = True,
    vmin: float | None = None,
    vmax: float | None = None,
    title: str | None = None,
) -> Any:
    """Plot a precomputed phase-space array."""
    axis = _axis(ax)
    x = np.asarray(xvec, dtype=np.float64)
    p = np.asarray(pvec, dtype=np.float64)
    image = axis.imshow(
        values,
        origin="lower",
        extent=(float(x[0]), float(x[-1]), float(p[0]), float(p[-1])),
        aspect="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    axis.set_xlabel("x")
    axis.set_ylabel("p")
    if title is not None:
        axis.set_title(title)
    if colorbar:
        axis.figure.colorbar(image, ax=axis)
    return axis


def plot_density_matrix(
    rho: Array,
    *,
    ax: Any | None = None,
    absolute: bool = True,
    colorbar: bool = True,
    cmap: str = "magma",
) -> Any:
    """Plot a density matrix magnitude or real part."""
    axis = _axis(ax)
    matrix = np.asarray(rho, dtype=np.complex128)
    values = np.abs(matrix) if absolute else matrix.real
    image = axis.imshow(values, origin="upper", cmap=cmap)
    axis.set_xlabel("column")
    axis.set_ylabel("row")
    if colorbar:
        axis.figure.colorbar(image, ax=axis)
    return axis


def _axis(ax: Any | None) -> Any:
    if ax is not None:
        return ax
    import matplotlib.pyplot as plt

    _, axis = plt.subplots()
    return axis


def _axes_or_default(
    xvec: AxisInput | None,
    pvec: AxisInput | None,
    *,
    xlim: tuple[float, float],
    plim: tuple[float, float] | None,
    points: int,
) -> tuple[FloatArray, FloatArray]:
    if xvec is None:
        return phase_space_grid(xlim=xlim, plim=plim, points=points)
    x = np.asarray(xvec, dtype=np.float64)
    p = np.asarray(xvec if pvec is None else pvec, dtype=np.float64)
    return x, p
