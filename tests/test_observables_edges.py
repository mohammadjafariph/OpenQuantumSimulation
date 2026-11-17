"""Edge cases and input checks of the state diagnostics."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import openquantumsim as oqs

PLUS = np.array([1.0, 1.0], dtype=np.complex128) / np.sqrt(2.0)
UP = np.array([1.0, 0.0], dtype=np.complex128)
MIXED = 0.5 * np.eye(2, dtype=np.complex128)


def test_expect_accepts_kets_and_density_matrices_only() -> None:
    sz = oqs.sigmaz(2)

    assert oqs.expect(sz, UP) == pytest.approx(1.0)
    assert oqs.expect(sz, oqs.ket2dm(UP)) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="ket vector or density matrix"):
        oqs.expect(sz, np.zeros((2, 2, 2)))


def test_renyi_entropy_limits() -> None:
    # Maximally mixed qubit: every Renyi entropy is one bit.
    for alpha in (0.0, 0.5, 1.0, 2.0, np.inf):
        assert oqs.renyi_entropy(MIXED, alpha) == pytest.approx(1.0)
    # Pure state: zero, and the rank-based alpha=0 limit is zero too.
    assert oqs.renyi_entropy(UP, 0.0) == pytest.approx(0.0)
    assert oqs.renyi_entropy(UP, 2.0) == pytest.approx(0.0)
    assert oqs.renyi_entropy(MIXED, 2.0, base=np.e) == pytest.approx(np.log(2.0))
    with pytest.raises(ValueError, match="non-negative"):
        oqs.renyi_entropy(MIXED, -1.0)


def test_entropy_of_zero_rank_spectrum_is_zero() -> None:
    assert oqs.von_neumann_entropy(np.diag([1.0, 0.0])) == 0.0


def test_trace_norm_handles_general_and_hermitian_matrices() -> None:
    nilpotent = np.array([[0.0, 2.0], [0.0, 0.0]], dtype=np.complex128)

    assert oqs.trace_norm(nilpotent) == pytest.approx(2.0)
    assert oqs.trace_norm(np.diag([1.0, -3.0])) == pytest.approx(4.0)
    assert oqs.trace_norm(np.diag([1.0, -3.0]), hermitian=False) == pytest.approx(4.0)
    with pytest.raises(ValueError, match="square"):
        oqs.trace_norm(np.zeros((2, 3)))


@pytest.mark.parametrize(
    "matrix",
    [
        np.zeros((2, 3)),
        np.array([[0.5, 1.0], [0.0, 0.5]]),
        np.eye(2),
        np.diag([1.5, -0.5]),
    ],
    ids=["not-square", "not-hermitian", "trace-two", "negative"],
)
def test_is_density_matrix_rejects_each_defect(matrix: Any) -> None:
    assert not oqs.is_density_matrix(matrix)


def test_is_density_matrix_accepts_valid_state() -> None:
    assert oqs.is_density_matrix(MIXED)
    assert not oqs.is_hermitian(np.zeros(3))


def test_normalize_state_rejects_degenerate_input() -> None:
    np.testing.assert_allclose(oqs.normalize_state(2.0 * UP), UP)
    np.testing.assert_allclose(oqs.normalize_state(2.0 * MIXED), MIXED)
    with pytest.raises(ValueError, match="zero norm"):
        oqs.normalize_state(np.zeros(2))
    with pytest.raises(ValueError, match="zero trace"):
        oqs.normalize_state(np.zeros((2, 2)))
    with pytest.raises(ValueError, match="ket vector or square"):
        oqs.normalize_state(np.zeros((2, 3)))


def test_fidelity_between_kets_and_density_matrices_in_both_orders() -> None:
    rho_plus = oqs.ket2dm(PLUS)

    assert oqs.fidelity(UP, rho_plus) == pytest.approx(0.5)
    assert oqs.fidelity(rho_plus, UP) == pytest.approx(0.5)
    assert oqs.fidelity(UP, PLUS, squared=False) == pytest.approx(np.sqrt(0.5))
    assert oqs.infidelity(UP, UP) == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("left", "right", "message"),
    [
        (UP, np.ones(3), "kets have incompatible lengths"),
        (UP, np.eye(3) / 3, "ket length does not match"),
        (np.eye(3) / 3, UP, "ket length does not match"),
        (MIXED, np.eye(3) / 3, "incompatible shapes"),
    ],
)
def test_fidelity_rejects_mismatched_dimensions(
    left: Any,
    right: Any,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        oqs.fidelity(left, right)


def test_distances_reject_mismatched_dimensions() -> None:
    with pytest.raises(ValueError, match="incompatible shapes"):
        oqs.trace_distance(MIXED, np.eye(3) / 3)
    with pytest.raises(ValueError, match="incompatible shapes"):
        oqs.hilbert_schmidt_distance(MIXED, np.eye(3) / 3)


def test_bloch_vector_requires_a_qubit() -> None:
    np.testing.assert_allclose(oqs.bloch_vector(PLUS), [1.0, 0.0, 0.0], atol=1e-12)
    with pytest.raises(ValueError, match="two-level"):
        oqs.bloch_vector(np.eye(3) / 3)


def test_evaluate_state_observables_checks_callbacks() -> None:
    states = [UP, PLUS]

    with pytest.raises(TypeError, match="must be callable"):
        oqs.evaluate_state_observables(states, {"x": 1.0})
    with pytest.raises(ValueError, match="must return a scalar"):
        oqs.evaluate_state_observables(states, {"x": lambda state: state})
    with pytest.raises(TypeError, match="must be strings"):
        oqs.evaluate_state_observables(states, {1: lambda _state: 0.0})
    with pytest.raises(ValueError, match="must not contain"):
        oqs.evaluate_state_observables(states, {"a/b": lambda _state: 0.0})


def test_named_metric_wrappers_reject_non_scalar_results() -> None:
    from openquantumsim.observables import _scalar_metric

    wrapped = _scalar_metric(lambda state: state)

    with pytest.raises(ValueError, match="must return a scalar"):
        wrapped(UP)


def test_population_helpers_reject_negative_indices() -> None:
    assert list(oqs.population_observable(1, name="excited")) == ["excited"]
    with pytest.raises(ValueError, match="non-negative"):
        oqs.population_observable(-1)
    with pytest.raises(ValueError, match="non-negative"):
        oqs.population_observables([0, -1])


@pytest.mark.parametrize(
    ("dims", "keep", "message"),
    [
        ((), 0, "at least one subsystem"),
        ((2, 0), 0, "must be positive"),
        ((2, 2), (0, 0), "duplicate"),
        ((2, 2), 2, "outside dims"),
        ((2, 3), 0, "does not match subsystem dimensions"),
    ],
)
def test_partial_trace_rejects_bad_subsystem_descriptions(
    dims: tuple[int, ...],
    keep: Any,
    message: str,
) -> None:
    state = np.zeros(4, dtype=np.complex128)
    state[0] = 1.0

    with pytest.raises(ValueError, match=message):
        oqs.partial_trace(state, dims, keep)


def test_partial_trace_keeping_nothing_returns_the_trace() -> None:
    state = np.zeros(4, dtype=np.complex128)
    state[0] = 1.0

    np.testing.assert_allclose(oqs.partial_trace(state, (2, 2), ()), [[1.0]])


def test_mutual_information_and_negativity_need_disjoint_parts() -> None:
    bell = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.complex128) / np.sqrt(2.0)

    assert oqs.mutual_information(bell, (2, 2), 0, 1) == pytest.approx(2.0)
    with pytest.raises(ValueError, match="disjoint"):
        oqs.mutual_information(bell, (2, 2), 0, 0)
    with pytest.raises(ValueError, match="disjoint"):
        oqs.negativity(bell, (2, 2), [0], [0])


def test_concurrence_requires_two_qubits() -> None:
    with pytest.raises(ValueError, match="two-qubit"):
        oqs.concurrence(MIXED)


def test_density_matrix_conversion_rejects_bad_shapes() -> None:
    with pytest.raises(ValueError, match="ket vector or square"):
        oqs.purity(np.zeros((2, 3)))
