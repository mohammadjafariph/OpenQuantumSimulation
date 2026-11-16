"""Unit tests for error types, option validation, and health checks.

These run without the Julia backend.
"""

from __future__ import annotations

import warnings

import pytest

import openquantumsim as oqs
from openquantumsim._health import check_solver_health, health_tolerance
from openquantumsim._julia_bridge import call_backend

HEALTHY = {
    "retcode": "Success",
    "success": True,
    "krylov_converged": True,
    "trace_drift": 1e-14,
    "hermiticity_drift": 0.0,
    "min_eigenvalue": -1e-13,
}


def test_error_hierarchy() -> None:
    assert issubclass(oqs.BackendError, oqs.OpenQuantumSimError)
    assert issubclass(oqs.BackendError, RuntimeError)
    assert issubclass(oqs.SolverError, oqs.OpenQuantumSimError)
    assert issubclass(oqs.ConvergenceError, oqs.SolverError)
    assert issubclass(oqs.SolverHealthWarning, UserWarning)
    # Kept for callers that caught the type raised before 0.1.0.
    assert issubclass(oqs.JuliaBridgeUnavailable, NotImplementedError)
    assert issubclass(oqs.JuliaBridgeUnavailable, oqs.BackendError)


def test_healthy_stats_are_silent() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        check_solver_health("mesolve", HEALTHY, rtol=1e-8)
        check_solver_health("mesolve", {}, rtol=1e-8)


def test_integrator_failure_raises_with_stats() -> None:
    stats = {**HEALTHY, "retcode": "Unstable", "success": False}

    with pytest.raises(oqs.SolverError, match="Unstable") as excinfo:
        check_solver_health("mesolve", stats, rtol=1e-8)

    assert excinfo.value.solver_stats["retcode"] == "Unstable"
    assert not isinstance(excinfo.value, oqs.ConvergenceError)


def test_krylov_failure_raises_convergence_error() -> None:
    stats = {
        **HEALTHY,
        "retcode": "KrylovNotConverged",
        "success": False,
        "krylov_converged": False,
        "krylov_max_residual": 3.0e5,
        "krylov_dim": 4,
    }

    with pytest.raises(oqs.ConvergenceError, match="krylov_dim=4"):
        check_solver_health("mesolve", stats, rtol=1e-8)


def test_gmres_failure_raises_convergence_error() -> None:
    stats = {
        "retcode": "GMRESNotConverged",
        "success": False,
        "linear_residual": 0.17,
        "krylov_dim": 30,
    }

    with pytest.raises(oqs.ConvergenceError, match="method='direct'"):
        check_solver_health("steadystate", stats, rtol=1e-10)


def test_failed_trajectories_are_counted_in_the_message() -> None:
    stats = {
        "retcode": "Unstable",
        "success": False,
        "n_failed_trajectories": 3,
        "n_traj": 200,
    }

    with pytest.raises(oqs.SolverError, match="3 of 200 trajectories"):
        check_solver_health("mcsolve", stats, rtol=1e-8)


def test_non_strict_mode_warns_instead_of_raising() -> None:
    stats = {**HEALTHY, "retcode": "MaxIters", "success": False}

    with pytest.warns(oqs.SolverHealthWarning, match="MaxIters"):
        check_solver_health("mesolve", stats, rtol=1e-8, strict=False)


@pytest.mark.parametrize(
    ("field", "value", "fragment"),
    [
        ("trace_drift", 1e-3, "trace drifted"),
        ("hermiticity_drift", 1e-3, "non-Hermitian"),
        ("min_eigenvalue", -1e-3, "smallest eigenvalue"),
        ("relative_residual", 1e-3, "steady-state residual"),
    ],
)
def test_drift_beyond_tolerance_warns(field: str, value: float, fragment: str) -> None:
    stats = {**HEALTHY, field: value}

    with pytest.warns(oqs.SolverHealthWarning, match=fragment):
        check_solver_health("mesolve", stats, rtol=1e-8)


def test_health_tolerance_follows_requested_accuracy() -> None:
    assert health_tolerance(1e-10) == pytest.approx(1e-6)
    assert health_tolerance(1e-3) == pytest.approx(1e-1)

    stats = {**HEALTHY, "trace_drift": 1e-3}
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        check_solver_health("mesolve", stats, rtol=1e-3)


def test_nan_statistics_do_not_trigger_drift_warnings() -> None:
    stats = {**HEALTHY, "min_eigenvalue": float("nan"), "trace_drift": None}

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        check_solver_health("mesolve", stats, rtol=1e-8)


def test_options_defaults_are_valid() -> None:
    options = oqs.Options()

    assert options.strict is True
    assert options.method == "auto"
    assert options.ode_solver == "auto"
    assert options.backend == "cpu"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"method": "magic"},
        {"ode_solver": "euler"},
        {"method": "tsit5", "ode_solver": "vern7"},
        {"method": "krylov", "ode_solver": "vern9"},
        {"backend": "quantum"},
        {"rtol": 0.0},
        {"atol": -1e-9},
        {"krylov_dim": 0},
        {"seed": -1},
    ],
)
def test_options_reject_invalid_values(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        oqs.Options(**kwargs)  # type: ignore[arg-type]


def test_gpu_backend_is_reported_as_not_implemented() -> None:
    with pytest.raises(NotImplementedError, match="gpu"):
        oqs.Options(backend="gpu")


class JuliaError(Exception):
    """Stand-in with the same class name as ``juliacall.JuliaError``."""


def test_call_backend_wraps_julia_errors() -> None:
    def failing(*_args: object) -> None:
        raise JuliaError("DimensionMismatch: sizes differ\nStacktrace:\n [1] foo")

    with pytest.raises(oqs.BackendError, match="sizes differ") as excinfo:
        call_backend(failing, 1, 2)

    assert "Stacktrace" not in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, JuliaError)


def test_call_backend_passes_python_errors_through() -> None:
    def failing() -> None:
        raise KeyError("missing")

    with pytest.raises(KeyError):
        call_backend(failing)


def test_call_backend_returns_the_result() -> None:
    assert call_backend(lambda a, b=0: a + b, 2, b=3) == 5
