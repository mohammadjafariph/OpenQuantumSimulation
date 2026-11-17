"""Edge cases and input checks of operators, states and Hilbert spaces."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import openquantumsim as oqs


def test_operator_rejects_non_square_or_non_matrix_data() -> None:
    with pytest.raises(ValueError, match="square"):
        oqs.Operator(np.zeros((2, 3)))
    with pytest.raises(ValueError, match="two-dimensional"):
        oqs.Operator(np.zeros(4))


def test_operator_arithmetic_mixes_dense_and_sparse_data() -> None:
    sparse_x = oqs.sigmax(2)
    dense_z = oqs.Operator(np.diag([1.0, -1.0]))

    np.testing.assert_allclose((sparse_x + dense_z).to_numpy(), [[1, 1], [1, -1]])
    np.testing.assert_allclose((dense_z - sparse_x).to_numpy(), [[1, -1], [-1, -1]])
    np.testing.assert_allclose((-dense_z).to_numpy(), np.diag([-1.0, 1.0]))
    np.testing.assert_allclose((dense_z * dense_z).to_numpy(), np.eye(2))
    np.testing.assert_allclose((dense_z / 2).to_numpy(), np.diag([0.5, -0.5]))
    np.testing.assert_allclose(np.asarray(dense_z, dtype=complex), np.diag([1, -1]))
    np.testing.assert_allclose((dense_z @ dense_z).to_numpy(), np.diag([1, -1, -1, 1]))


def test_operator_arithmetic_checks_shapes() -> None:
    with pytest.raises(ValueError, match="shapes differ"):
        oqs.sigmax(2) + oqs.num(3)
    with pytest.raises(ValueError, match="incompatible for multiplication"):
        oqs.sigmax(2) * oqs.num(3)


def test_tensor_needs_operands_and_tracks_spaces_only_when_all_known() -> None:
    qubit = oqs.SpinSpace(0.5)

    with pytest.raises(ValueError, match="at least one operator"):
        oqs.tensor()
    assert oqs.tensor(oqs.sigmax(qubit), oqs.sigmax(qubit)).space is not None
    assert oqs.tensor(oqs.sigmax(qubit), oqs.sigmax(2)).space is None


@pytest.mark.parametrize(
    ("space", "index", "message"),
    [
        (oqs.SpinSpace(0.5), "sideways", "Unknown basis label"),
        (oqs.FockSpace(3), "up", "only for SpinSpace"),
        (oqs.DickeSpace(2), "middle", "Unknown Dicke basis label"),
        (oqs.FockSpace(3), 3, "outside dimension"),
        (oqs.FockSpace(3), -1, "outside dimension"),
    ],
)
def test_basis_rejects_unknown_labels_and_indices(
    space: Any,
    index: Any,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        oqs.basis(space, index)


def test_dicke_basis_labels_and_states() -> None:
    ensemble = oqs.DickeSpace(3)

    assert np.argmax(oqs.basis(ensemble, "all_up")) == 0
    assert np.argmax(oqs.basis(ensemble, "ground")) == 3
    assert np.argmax(oqs.dicke_state(ensemble, 1)) == 2
    with pytest.raises(ValueError, match="excitations must satisfy"):
        oqs.dicke_state(ensemble, 4)


def test_pauli_operators_require_two_levels() -> None:
    with pytest.raises(ValueError, match="two-dimensional"):
        oqs.sigmax(3)
    assert oqs.sigmaz().space is None


def test_thermal_state_populations() -> None:
    rho = oqs.thermal_dm(oqs.FockSpace(40), 0.5)
    populations = np.real(np.diag(rho))

    assert populations.sum() == pytest.approx(1.0)
    assert np.sum(np.arange(40) * populations) == pytest.approx(0.5, rel=1e-6)
    with pytest.raises(ValueError, match="non-negative"):
        oqs.thermal_dm(3, -0.1)


def test_coherent_state_of_zero_amplitude_is_vacuum() -> None:
    np.testing.assert_allclose(oqs.coherent(4, 0.0), [1, 0, 0, 0])


def test_collective_lowering_requires_positive_dimension() -> None:
    from openquantumsim.operators import _collective_lowering_data

    with pytest.raises(ValueError, match="positive"):
        _collective_lowering_data(0)


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: oqs.FockSpace(0), "must be positive"),
        (lambda: oqs.SpinSpace(0.3), "2S \\+ 1"),
        (lambda: oqs.SpinSpace(-0.5), "S >= 0"),
        (lambda: oqs.DickeSpace(-1), "n_spins >= 0"),
        (lambda: oqs.CompositeSpace(()), "at least one subsystem"),
    ],
)
def test_hilbert_spaces_reject_invalid_sizes(factory: Any, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        factory()


def test_composite_spaces_flatten_and_iterate() -> None:
    a, b, c = oqs.FockSpace(2), oqs.SpinSpace(1.0), oqs.DickeSpace(3)

    joined = (a * b) * (b * c)
    assert len(joined) == 4
    assert list(joined) == [a, b, b, c]
    assert joined.dim == 2 * 3 * 3 * 4
    assert oqs.DickeSpace(4).total_spin == 2.0
    with pytest.raises(NotImplementedError):
        _ = oqs.HilbertSpace().dim
