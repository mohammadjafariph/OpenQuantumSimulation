"""Backend-free tests: solvers reject bad input before starting Julia.

Every check here runs before the Julia backend is loaded, so a malformed call
fails fast with a clear message on a machine without Julia as well.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from scipy import sparse as sp  # type: ignore[import-untyped]

import openquantumsim as oqs
from openquantumsim import solvers
from openquantumsim.observables import fidelity_observable, state_metrics

QUBIT = oqs.SpinSpace(0.5, label="q")
H2 = 0.5 * oqs.sigmaz(QUBIT)
H3 = oqs.num(3)
RHO2 = oqs.ket2dm(oqs.basis(QUBIT, "up"))
PSI2 = oqs.basis(QUBIT, "up")
TIMES = np.linspace(0.0, 1.0, 3)


# --------------------------------------------------------------------------
# mesolve, brmesolve, steadystate, correlations
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"rho0": oqs.ket2dm(oqs.basis(3, 0))}, "rho0 must have the same shape"),
        ({"tlist": []}, "non-empty"),
        ({"tlist": [1.0, 0.0]}, "ascending"),
        ({"c_ops": [oqs.destroy(3)]}, "same shape as H"),
        ({"e_ops": [oqs.num(3)]}, "same shape as H"),
    ],
)
def test_mesolve_rejects_malformed_input(kwargs: dict[str, Any], message: str) -> None:
    call: dict[str, Any] = {"rho0": RHO2, "tlist": TIMES}
    call.update(kwargs)
    rho0 = call.pop("rho0")
    tlist = call.pop("tlist")

    with pytest.raises(ValueError, match=message):
        oqs.mesolve(H2, rho0, tlist, **call)


def test_mesolve_rejects_krylov_for_time_dependent_hamiltonians() -> None:
    H_t = oqs.time_dependent_hamiltonian(H2, [(oqs.sigmax(QUBIT), 0.1)])

    with pytest.raises(ValueError, match="krylov"):
        oqs.mesolve(H_t, RHO2, TIMES, options=oqs.Options(method="krylov"))


def test_brmesolve_rejects_mismatched_coupling_operator() -> None:
    with pytest.raises(ValueError, match="same shape as H"):
        oqs.brmesolve(H2, RHO2, TIMES, a_ops=[(oqs.destroy(3), lambda _w: 1.0)])


def test_steadystate_rejects_mismatched_collapse_operator() -> None:
    with pytest.raises(ValueError, match="same shape as H"):
        oqs.steadystate(H2, [oqs.destroy(3)])


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"rho0": oqs.ket2dm(oqs.basis(3, 0))}, "rho0 must have the same shape"),
        ({"taulist": []}, "non-empty"),
        ({"taulist": [-1.0, 0.0]}, "non-negative"),
        ({"taulist": [1.0, 0.0]}, "ascending"),
        ({"a_op": oqs.num(3)}, "a_op must have the same shape"),
        ({"b_op": oqs.num(3)}, "b_op must have the same shape"),
        ({"c_ops": [oqs.destroy(3)]}, "collapse operators"),
    ],
)
def test_correlation_2op_1t_rejects_malformed_input(
    kwargs: dict[str, Any],
    message: str,
) -> None:
    call: dict[str, Any] = {
        "rho0": RHO2,
        "taulist": TIMES,
        "a_op": oqs.sigmap(QUBIT),
        "b_op": oqs.sigmam(QUBIT),
    }
    call.update(kwargs)

    with pytest.raises(ValueError, match=message):
        oqs.correlation_2op_1t(
            H2,
            call.pop("rho0"),
            call.pop("taulist"),
            call.pop("a_op"),
            call.pop("b_op"),
            **call,
        )


def test_correlation_2op_2t_checks_both_time_grids() -> None:
    a_op, b_op = oqs.sigmap(QUBIT), oqs.sigmam(QUBIT)

    with pytest.raises(ValueError, match="tlist must be non-negative"):
        oqs.correlation_2op_2t(H2, RHO2, [-1.0, 0.0], TIMES, a_op, b_op)
    with pytest.raises(ValueError, match="taulist must be sorted"):
        oqs.correlation_2op_2t(H2, RHO2, TIMES, [1.0, 0.0], a_op, b_op)


def test_correlation_method_follows_explicit_ode_solver() -> None:
    from openquantumsim.correlations import _correlation_method

    assert _correlation_method(oqs.Options()) == "auto"
    assert _correlation_method(oqs.Options(ode_solver="vern7")) == "ode"
    assert _correlation_method(oqs.Options(method="krylov")) == "krylov"


# --------------------------------------------------------------------------
# mcsolve and single_trajectory
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"psi0": oqs.basis(3, 0)}, ValueError, "psi0 must be a ket"),
        ({"psi0": np.zeros(2)}, ValueError, "nonzero norm"),
        ({"tlist": []}, ValueError, "non-empty"),
        ({"tlist": [1.0, 0.0]}, ValueError, "ascending"),
        ({"n_traj": 0}, ValueError, "n_traj must be positive"),
        ({"c_ops": [oqs.destroy(3)]}, ValueError, "same shape as H"),
        ({"options": oqs.Options(method="krylov")}, ValueError, "krylov"),
    ],
)
def test_mcsolve_rejects_malformed_input(
    kwargs: dict[str, Any],
    error: type[Exception],
    message: str,
) -> None:
    call: dict[str, Any] = {"psi0": PSI2, "tlist": TIMES}
    call.update(kwargs)

    with pytest.raises(error, match=message):
        oqs.mcsolve(H2, call.pop("psi0"), call.pop("tlist"), **call)


def test_single_trajectory_rejects_krylov_and_bad_kets() -> None:
    with pytest.raises(ValueError, match="krylov"):
        oqs.single_trajectory(H2, PSI2, TIMES, options=oqs.Options(method="krylov"))
    with pytest.raises(ValueError, match="psi0 must be a ket"):
        oqs.single_trajectory(H2, oqs.basis(3, 0), TIMES)


def _plan(observables: dict[str, Any], dim: int = 2) -> Any:
    return solvers._mcsolve_state_observable_plan(dim, observables)


def test_state_observable_plan_turns_metrics_into_operators() -> None:
    plan = _plan(
        state_metrics(
            purity=True,
            population_indices=[1],
            bloch_vector=True,
            fidelity_to=PSI2,
        ),
    )

    names = [entry.name for entry in plan.entries]
    assert names == [
        "purity",
        "population_1",
        "bloch_x",
        "bloch_y",
        "bloch_z",
        "fidelity",
    ]
    # Purity of a pure trajectory is a constant; the rest are expectations.
    assert plan.entries[0].constant == 1.0
    assert len(plan.operators) == 5
    np.testing.assert_allclose(plan.operators[0], np.diag([0.0, 1.0]))
    np.testing.assert_allclose(plan.operators[-1], RHO2)
    assert _plan({}).operators == ()


def test_state_observable_plan_accepts_pure_density_matrix_reference() -> None:
    plan = _plan(fidelity_observable(2.0 * RHO2))

    np.testing.assert_allclose(plan.operators[0], RHO2)


@pytest.mark.parametrize(
    ("observables", "error", "message"),
    [
        ({"x": 1.0}, TypeError, "must be callable"),
        ({"": lambda _s: 0.0}, ValueError, "must not be empty"),
        ({"a/b": lambda _s: 0.0}, ValueError, "must not contain"),
        ({1: lambda _s: 0.0}, TypeError, "must be strings"),
        ({"custom": lambda _s: 0.0}, NotImplementedError, "Arbitrary Python"),
        (state_metrics(l1_coherence=True), NotImplementedError, "nonlinear"),
        (state_metrics(population_indices=[5]), ValueError, "outside Hilbert"),
        (
            fidelity_observable(0.5 * np.eye(2)),
            NotImplementedError,
            "pure reference",
        ),
        (fidelity_observable(np.ones(3)), ValueError, "ket length"),
        (fidelity_observable(np.zeros(2)), ValueError, "nonzero norm"),
        (fidelity_observable(np.zeros((2, 2))), ValueError, "nonzero trace"),
        (fidelity_observable(np.ones((3, 3))), ValueError, "ket or density"),
    ],
)
def test_state_observable_plan_rejects_what_it_cannot_aggregate(
    observables: dict[Any, Any],
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        _plan(observables)


def test_state_observable_plan_rejects_bloch_vector_beyond_qubits() -> None:
    with pytest.raises(ValueError, match="two-level"):
        _plan(state_metrics(bloch_vector=True), dim=3)
    with pytest.raises(ValueError, match="0, 1, or 2"):
        solvers._bloch_operator(2, 3)


def test_state_observable_plan_rejects_unknown_metric_kind() -> None:
    def metric(_state: Any) -> complex:
        return 0j

    metric.__oqs_metric_spec__ = {"kind": "mystery"}  # type: ignore[attr-defined]

    with pytest.raises(NotImplementedError, match="mystery"):
        _plan({"m": metric})


def test_state_observable_results_split_constants_and_expectations() -> None:
    plan = _plan(state_metrics(purity=True, population_indices=[0]))
    raw_expect = np.array([[0.1, 0.2], [0.9, 0.8]], dtype=np.complex128)
    raw_std = np.array([[0.0, 0.0], [0.01, 0.02]])
    raw_stderr = raw_std / 10

    observed, std, stderr = solvers._mcsolve_state_observable_results(
        plan,
        raw_expect,
        raw_std,
        raw_stderr,
        1,
        2,
    )

    np.testing.assert_allclose(observed["purity"], [1.0, 1.0])
    np.testing.assert_allclose(std["purity"], [0.0, 0.0])
    np.testing.assert_allclose(observed["population_0"], [0.9, 0.8])
    np.testing.assert_allclose(stderr["population_0"], [0.001, 0.002])


# --------------------------------------------------------------------------
# Payload helpers
# --------------------------------------------------------------------------


class _FakeBackend:
    def __init__(self) -> None:
        self.sparse_calls: list[tuple[Any, ...]] = []

    def sparse_from_csc(self, *args: Any) -> str:
        self.sparse_calls.append(args)
        return "sparse"

    def InterpolatedCoefficient(self, times: Any, values: Any) -> tuple[Any, Any]:  # noqa: N802
        return ("interpolated", len(times))


def test_matrix_payload_sends_sparse_structure_with_one_based_indices() -> None:
    backend = _FakeBackend()
    identity = sp.identity(3, format="csc")

    assert solvers._matrix_payload(backend, identity) == "sparse"
    rows, cols, colptr, rowval, nzval = backend.sparse_calls[0]
    assert (rows, cols) == (3, 3)
    np.testing.assert_array_equal(colptr, [1, 2, 3, 4])
    np.testing.assert_array_equal(rowval, [1, 2, 3])
    np.testing.assert_allclose(nzval, [1, 1, 1])

    # A large, mostly-zero dense array is also sent as sparse; a small or
    # dense one is sent as is.
    assert solvers._matrix_payload(backend, np.eye(10)) == "sparse"
    small = solvers._matrix_payload(backend, np.eye(2))
    assert isinstance(small, np.ndarray)
    full = solvers._matrix_payload(backend, np.ones((10, 10)))
    assert isinstance(full, np.ndarray)


def test_coefficient_payload_accepts_supported_coefficients_only() -> None:
    backend = _FakeBackend()
    interpolated = oqs.InterpolatedCoefficient([0.0, 1.0], [0.0, 1.0])

    assert solvers._coefficient_payload(backend, 2) == 2 + 0j
    assert solvers._coefficient_payload(backend, interpolated) == ("interpolated", 2)
    assert solvers._coefficient_payload(backend, abs) is abs
    with pytest.raises(TypeError, match="unsupported coefficient type"):
        solvers._coefficient_payload(backend, "not a coefficient")


def test_seed_and_max_step_payloads() -> None:
    assert solvers._seed_payload(oqs.Options()) == -1
    assert solvers._seed_payload(oqs.Options(seed=0)) == 0
    assert solvers._seed_payload(oqs.Options(seed=7)) == 7
    assert solvers._max_step_payload(oqs.Options()) == 0.0
    assert solvers._max_step_payload(oqs.Options(max_step=0.5)) == 0.5


def test_backend_state_arrays_are_checked_before_use() -> None:
    with pytest.raises(RuntimeError, match="density states"):
        solvers._density_states(np.zeros((2, 2)), 1)
    with pytest.raises(RuntimeError, match="ket states"):
        solvers._ket_states(np.zeros((2, 2, 1)), 1)

    densities = solvers._density_states(np.zeros((2, 2, 3)), 3)
    kets = solvers._ket_states(np.zeros((2, 3)), 3)
    assert len(densities) == 3
    assert len(kets) == 3


def test_field_reads_attributes_or_keys() -> None:
    class WithAttribute:
        retcode = "Success"

    assert solvers._field(WithAttribute(), "retcode") == "Success"
    assert solvers._field({"retcode": "Failure"}, "retcode") == "Failure"


# --------------------------------------------------------------------------
# Superoperators
# --------------------------------------------------------------------------


def test_apply_superoperator_accepts_kets_and_checks_shapes() -> None:
    generator = oqs.lindblad_superoperator(H2, [oqs.sigmam(QUBIT)])

    from_ket = oqs.apply_superoperator(generator, PSI2)
    from_rho = oqs.apply_superoperator(generator, RHO2)
    np.testing.assert_allclose(from_ket, from_rho)
    # d/dt of the excited population under unit decay is -1.
    assert from_rho[0, 0] == pytest.approx(-1.0)

    with pytest.raises(ValueError, match="square density matrix"):
        oqs.apply_superoperator(generator, np.zeros((2, 3)))
    with pytest.raises(ValueError, match="does not match rho"):
        oqs.apply_superoperator(generator, np.eye(3))


@pytest.mark.parametrize(
    ("tlist", "message"),
    [([], "non-empty"), ([1.0, 0.0], "increasing")],
)
def test_propagator_rejects_bad_time_grids(tlist: list[float], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        oqs.propagator(H2, tlist)
