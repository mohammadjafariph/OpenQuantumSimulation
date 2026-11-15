"""Turn backend solver statistics into errors and warnings.

The Julia backend records what happened (integrator return code, Krylov and
GMRES convergence, drift of conserved quantities). This module decides what
the user is told: a failed solve raises, a suspicious one warns.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping
from typing import Any

from .errors import ConvergenceError, SolverError, SolverHealthWarning

# Drift of an invariant (trace, Hermiticity, positivity, steady-state residual)
# above this floor, or above ``HEALTH_RTOL_FACTOR * rtol`` if that is larger,
# triggers a warning.
HEALTH_TOLERANCE_FLOOR = 1e-6
HEALTH_RTOL_FACTOR = 100.0


def health_tolerance(rtol: float) -> float:
    """Return the drift above which a result is reported as suspicious."""
    return max(HEALTH_TOLERANCE_FLOOR, HEALTH_RTOL_FACTOR * float(rtol))


def check_solver_health(
    solver: str,
    stats: Mapping[str, Any],
    *,
    rtol: float,
    strict: bool = True,
    stacklevel: int = 3,
) -> None:
    """Raise or warn according to the statistics a solver reported.

    A failed solve raises :class:`SolverError` (or :class:`ConvergenceError`
    for an iterative method that did not converge). With ``strict=False`` the
    failure is downgraded to a :class:`SolverHealthWarning` and the caller
    returns the partial result. Drift of conserved quantities always warns.
    """
    failure = _failure(solver, stats)
    if failure is not None:
        error_type, message = failure
        if strict:
            raise error_type(message, solver_stats=stats)
        warnings.warn(message, SolverHealthWarning, stacklevel=stacklevel)
        return

    problems = _drift_problems(stats, health_tolerance(rtol))
    if problems:
        message = (
            f"{solver}: the result may be inaccurate: "
            + "; ".join(problems)
            + ". Tighten Options.rtol/atol or check the model."
        )
        warnings.warn(message, SolverHealthWarning, stacklevel=stacklevel)


def _failure(
    solver: str,
    stats: Mapping[str, Any],
) -> tuple[type[SolverError], str] | None:
    if bool(stats.get("success", True)):
        return None

    retcode = str(stats.get("retcode", "unknown"))
    if stats.get("krylov_converged") is False:
        residual = _as_float(stats.get("krylov_max_residual"))
        return ConvergenceError, (
            f"{solver}: Krylov exponentiation did not converge "
            f"(largest residual {residual:.3e}, krylov_dim="
            f"{stats.get('krylov_dim', '?')}). Increase Options.krylov_dim, use a "
            "finer time grid, or switch to method='ode'."
        )
    if retcode == "GMRESNotConverged":
        residual = _as_float(stats.get("linear_residual"))
        return ConvergenceError, (
            f"{solver}: GMRES did not converge (residual {residual:.3e}, "
            f"krylov_dim={stats.get('krylov_dim', '?')}). Increase "
            "Options.krylov_dim or use method='direct'."
        )
    if retcode == "NonFiniteSolution":
        return SolverError, (
            f"{solver}: the linear solve returned a non-finite state. The "
            "steady state may not be unique."
        )

    failed = stats.get("n_failed_trajectories")
    if isinstance(failed, int) and failed > 0:
        return SolverError, (
            f"{solver}: {failed} of {stats.get('n_traj', '?')} trajectories "
            f"stopped early (return code {retcode!r}). Loosen Options.rtol/atol "
            "or check the model for stiffness."
        )
    return SolverError, (
        f"{solver}: the integrator stopped early with return code {retcode!r}. "
        "Values after the failure are NaN. Loosen Options.rtol/atol or check "
        "the model for stiffness or divergence."
    )


def _drift_problems(stats: Mapping[str, Any], tolerance: float) -> list[str]:
    problems: list[str] = []
    trace_drift = _as_float(stats.get("trace_drift"))
    if trace_drift > tolerance:
        problems.append(f"trace drifted by {trace_drift:.3e}")
    hermiticity_drift = _as_float(stats.get("hermiticity_drift"))
    if hermiticity_drift > tolerance:
        problems.append(f"state is non-Hermitian by {hermiticity_drift:.3e}")
    min_eigenvalue = _as_float(stats.get("min_eigenvalue"))
    if min_eigenvalue < -tolerance:
        problems.append(f"smallest eigenvalue is {min_eigenvalue:.3e}")
    relative_residual = _as_float(stats.get("relative_residual"))
    if relative_residual > tolerance:
        problems.append(
            f"steady-state residual |L rho| / |L| is {relative_residual:.3e}"
        )
    return problems


def _as_float(value: object) -> float:
    """Convert a statistic to float; missing or NaN values count as zero."""
    if value is None:
        return 0.0
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if math.isnan(number) else number
