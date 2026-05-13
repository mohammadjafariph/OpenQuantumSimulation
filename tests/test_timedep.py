import numpy as np
import pytest

import openquantumsim as oqs


def test_interpolated_coefficient_is_complex_and_clamped() -> None:
    coefficient = oqs.InterpolatedCoefficient([0.0, 1.0, 2.0], [0.0, 1.0j, 2.0])

    assert coefficient(-1.0) == 0.0
    assert coefficient(3.0) == 2.0
    assert np.isclose(coefficient(0.5), 0.5j)
    assert np.isclose(coefficient(1.5), 1.0 + 0.5j)


def test_interpolated_coefficient_validates_grid() -> None:
    with pytest.raises(ValueError, match="same length"):
        oqs.InterpolatedCoefficient([0.0, 1.0], [1.0])

    with pytest.raises(ValueError, match="strictly ascending"):
        oqs.InterpolatedCoefficient([0.0, 0.0], [1.0, 2.0])


def test_time_dependent_hamiltonian_evaluates_terms() -> None:
    atom = oqs.SpinSpace(0.5, label="atom")
    base = 0.5 * oqs.sigmaz(atom)
    drive = oqs.sigmax(atom)

    H = oqs.time_dependent_hamiltonian(
        base,
        [
            oqs.HamiltonianTerm(drive, lambda t: 0.25 * t),
            (oqs.sigmay(atom), oqs.InterpolatedCoefficient([0.0, 2.0], [0.0, 1.0])),
        ],
    )

    expected = base.to_numpy() + 0.5 * drive.to_numpy() + oqs.sigmay(atom).to_numpy()
    assert H.shape == base.shape
    assert H.space == atom
    assert np.allclose(H.to_numpy(2.0), expected)
    assert np.allclose(H(2.0).to_numpy(), expected)


def test_time_dependent_hamiltonian_validates_terms() -> None:
    atom = oqs.SpinSpace(0.5)
    cavity = oqs.FockSpace(3)

    with pytest.raises(ValueError, match="match base shape"):
        oqs.time_dependent_hamiltonian(oqs.sigmaz(atom), [(oqs.destroy(cavity), 1.0)])

    with pytest.raises(TypeError, match="scalar number"):
        oqs.time_dependent_hamiltonian(
            oqs.sigmaz(atom),
            [(oqs.sigmax(atom), lambda _t: [1.0])],
        )
