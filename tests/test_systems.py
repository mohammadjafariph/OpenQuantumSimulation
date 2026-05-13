import numpy as np
import pytest

import openquantumsim as oqs
from examples.dicke.system import two_ensemble_dicke_system


def test_two_ensemble_dicke_system_shapes() -> None:
    system = two_ensemble_dicke_system(2, 0.1)

    assert isinstance(system.hilbert.spaces[0], oqs.DickeSpace)
    assert isinstance(system.hilbert.spaces[1], oqs.DickeSpace)
    assert system.H.shape == (9, 9)
    assert system.psi0.shape == (9,)
    assert len(system.c_ops) == 3
    assert len(system.e_ops) == 1
    assert system.hilbert.dim == 9
    assert system.metadata["N"] == 2
    assert system.metadata["kappa"] == 0.1
    assert np.isclose(np.linalg.norm(system.psi0), 1.0)
    assert np.allclose(system.e_ops[0].to_numpy(), system.e_ops[0].to_numpy().conj().T)
    assert np.isclose(oqs.expect(system.e_ops[0], system.psi0), 1.0)


def test_two_ensemble_dicke_system_validates_inputs() -> None:
    with pytest.raises(ValueError, match="N must be positive"):
        two_ensemble_dicke_system(0, 0.1)
    with pytest.raises(ValueError, match="kappa must be non-negative"):
        two_ensemble_dicke_system(2, -0.1)


def test_jaynes_cummings_system_shapes_and_observables() -> None:
    system = oqs.jaynes_cummings_system(
        4,
        cavity_frequency=1.0,
        atom_frequency=1.0,
        coupling=0.2,
        cavity_decay=0.05,
        atom_decay=0.03,
    )

    assert isinstance(system.hilbert.spaces[0], oqs.FockSpace)
    assert isinstance(system.hilbert.spaces[1], oqs.SpinSpace)
    assert system.H.shape == (8, 8)
    assert system.psi0.shape == (8,)
    assert system.rho0.shape == (8, 8)
    assert len(system.c_ops) == 2
    assert len(system.e_ops) == 2
    assert system.metadata["cavity_dim"] == 4
    assert np.isclose(np.linalg.norm(system.psi0), 1.0)
    assert np.isclose(oqs.expect(system.e_ops[0], system.psi0), 0.0)
    assert np.isclose(oqs.expect(system.e_ops[1], system.psi0), 1.0)
    assert np.allclose(system.H.to_numpy(), system.H.to_numpy().conj().T)


def test_jaynes_cummings_system_validates_inputs() -> None:
    with pytest.raises(ValueError, match="cavity_dim"):
        oqs.jaynes_cummings_system(0)
    with pytest.raises(ValueError, match="initial_photon"):
        oqs.jaynes_cummings_system(2, initial_photon=2)
    with pytest.raises(ValueError, match="non-negative"):
        oqs.jaynes_cummings_system(2, cavity_decay=-0.1)
