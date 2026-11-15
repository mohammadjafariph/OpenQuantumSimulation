"""Solvers must say when a result cannot be trusted."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import openquantumsim as oqs

pytestmark = pytest.mark.physics


def _qubit() -> tuple[oqs.Operator, oqs.Operator, np.ndarray, oqs.Operator]:
    atom = oqs.SpinSpace(0.5, label="atom")
    excited = oqs.basis(atom, "up")
    rho0 = oqs.ket2dm(excited)
    projector = oqs.Operator(rho0, atom, "P_excited")
    return oqs.sigmax(atom), oqs.sigmam(atom), rho0, projector


def _driven_cavity(dim: int) -> tuple[oqs.Operator, list[oqs.Operator], np.ndarray]:
    cavity = oqs.FockSpace(dim, label="cavity")
    a = oqs.destroy(cavity)
    hamiltonian = 2.0 * (a + oqs.create(cavity)) + 1.0 * oqs.num(cavity)
    rho0 = oqs.ket2dm(oqs.basis(cavity, 0))
    return hamiltonian, [np.sqrt(0.5) * a], rho0


def test_healthy_mesolve_reports_invariants() -> None:
    sx, sm, rho0, projector = _qubit()
    times = np.linspace(0.0, 5.0, 51)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = oqs.mesolve(
            0.4 * sx,
            rho0,
            times,
            c_ops=[np.sqrt(0.3) * sm],
            e_ops=[projector],
        )

    stats = result.solver_stats
    assert stats["success"] is True
    assert stats["retcode"] == "Success"
    assert stats["ode_solver"] == "tsit5"
    assert stats["nsteps"] > 0
    assert stats["nfev"] > stats["nsteps"]
    assert stats["trace_drift"] < 1e-9
    assert stats["hermiticity_drift"] < 1e-9
    assert stats["min_eigenvalue"] > -1e-9
    assert stats["n_nonfinite_states"] == 0


def test_integrator_failure_raises_solver_error() -> None:
    sx, _sm, rho0, projector = _qubit()
    hamiltonian = oqs.time_dependent_hamiltonian(
        0.0 * sx,
        [(sx, lambda t: float("nan") if t > 1.0 else 0.4)],
    )

    with pytest.raises(oqs.SolverError) as excinfo:
        oqs.mesolve(hamiltonian, rho0, np.linspace(0.0, 3.0, 13), e_ops=[projector])

    assert excinfo.value.solver_stats["success"] is False
    assert excinfo.value.solver_stats["retcode"] != "Success"


def test_non_strict_mode_returns_partial_result_with_nan_tail() -> None:
    sx, _sm, rho0, projector = _qubit()
    times = np.linspace(0.0, 3.0, 13)
    hamiltonian = oqs.time_dependent_hamiltonian(
        0.0 * sx,
        [(sx, lambda t: float("nan") if t > 1.0 else 0.4)],
    )

    with pytest.warns(oqs.SolverHealthWarning, match="stopped early"):
        result = oqs.mesolve(
            hamiltonian,
            rho0,
            times,
            e_ops=[projector],
            options=oqs.Options(strict=False),
        )

    population = result.expect[0].real
    assert result.solver_stats["success"] is False
    assert len(population) == len(times)
    assert np.allclose(population[:5], np.cos(0.4 * times[:5]) ** 2, atol=1e-6)
    assert np.all(np.isnan(population[5:]))


def test_krylov_non_convergence_raises_convergence_error() -> None:
    hamiltonian, c_ops, rho0 = _driven_cavity(12)

    with pytest.raises(oqs.ConvergenceError, match="krylov_dim=4"):
        oqs.mesolve(
            hamiltonian,
            rho0,
            [0.0, 50.0],
            c_ops=c_ops,
            options=oqs.Options(method="krylov", krylov_dim=4, rtol=1e-10),
        )

    converged = oqs.mesolve(
        hamiltonian,
        rho0,
        [0.0, 50.0],
        c_ops=c_ops,
        e_ops=[oqs.num(oqs.FockSpace(12, label="cavity"))],
        options=oqs.Options(method="krylov", krylov_dim=30, rtol=1e-10),
    )
    reference = oqs.mesolve(
        hamiltonian,
        rho0,
        [0.0, 50.0],
        c_ops=c_ops,
        e_ops=[oqs.num(oqs.FockSpace(12, label="cavity"))],
        options=oqs.Options(ode_solver="vern7", rtol=1e-10, atol=1e-12),
    )
    assert converged.solver_stats["krylov_converged"] is True
    assert np.allclose(converged.expect[0], reference.expect[0], atol=1e-7)


def test_time_dependent_mesolve_rejects_krylov() -> None:
    sx, _sm, rho0, _projector = _qubit()
    hamiltonian = oqs.time_dependent_hamiltonian(0.0 * sx, [(sx, 0.4)])

    with pytest.raises(ValueError, match="krylov"):
        oqs.mesolve(
            hamiltonian,
            rho0,
            [0.0, 1.0],
            options=oqs.Options(method="krylov"),
        )


def test_steadystate_default_is_accurate_and_gmres_failure_is_loud() -> None:
    hamiltonian, c_ops, _rho0 = _driven_cavity(25)

    rho, stats = oqs.steadystate(hamiltonian, c_ops, return_stats=True)
    assert stats["method"] == "direct"
    assert stats["converged"] is True
    assert stats["relative_residual"] < 1e-12
    assert np.isclose(np.trace(rho).real, 1.0)

    generator = oqs.lindblad_superoperator(hamiltonian, c_ops)
    assert np.linalg.norm(oqs.apply_superoperator(generator, rho)) < 1e-9

    with pytest.raises(oqs.ConvergenceError, match="GMRES did not converge"):
        oqs.steadystate(hamiltonian, c_ops, method="iterative-gmres")

    with pytest.warns(oqs.SolverHealthWarning):
        loose = oqs.steadystate(
            hamiltonian,
            c_ops,
            method="iterative-gmres",
            options=oqs.Options(strict=False),
        )
    assert isinstance(loose, np.ndarray)
    assert loose.shape == rho.shape


def test_correlation_failure_raises_convergence_error() -> None:
    hamiltonian, c_ops, rho0 = _driven_cavity(12)
    cavity = oqs.FockSpace(12, label="cavity")

    with pytest.raises(oqs.ConvergenceError):
        oqs.correlation_2op_1t(
            hamiltonian,
            rho0,
            [0.0, 50.0],
            oqs.destroy(cavity),
            oqs.create(cavity),
            c_ops=c_ops,
            options=oqs.Options(krylov_dim=4),
        )


def test_backend_exceptions_surface_as_backend_error() -> None:
    atom = oqs.SpinSpace(0.5, label="atom")
    non_hermitian = oqs.sigmam(atom)
    rho0 = oqs.ket2dm(oqs.basis(atom, "up"))

    with pytest.raises(oqs.BackendError, match="Hermitian") as excinfo:
        oqs.brmesolve(
            non_hermitian,
            rho0,
            [0.0, 1.0],
            a_ops=[(oqs.sigmax(atom), lambda omega: 0.1)],
        )

    assert excinfo.value.julia_type == "ArgumentError"
    assert "Stacktrace" not in str(excinfo.value)


def test_ode_solver_option_is_honoured() -> None:
    sx, sm, rho0, projector = _qubit()
    times = np.linspace(0.0, 4.0, 41)
    results = {
        solver: oqs.mesolve(
            0.4 * sx,
            rho0,
            times,
            c_ops=[np.sqrt(0.3) * sm],
            e_ops=[projector],
            options=oqs.Options(ode_solver=solver, rtol=1e-10, atol=1e-12),
        )
        for solver in ("tsit5", "vern7", "vern9")
    }

    for solver, result in results.items():
        assert result.solver_stats["ode_solver"] == solver
        assert np.allclose(result.expect[0], results["vern9"].expect[0], atol=1e-8)
