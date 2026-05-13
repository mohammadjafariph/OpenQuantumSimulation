import numpy as np
import pytest
from scipy import sparse as sp  # type: ignore[import-untyped]

import openquantumsim as oqs
from examples.dicke.observables import (
    dicke_k_rdm,
    dicke_mutual_information,
    precompute_dicke_reduction,
    trajectory_dicke_mutual_information,
    two_ensemble_dicke_mutual_information,
)


def test_destroy_create_number() -> None:
    space = oqs.FockSpace(4)
    a = oqs.destroy(space)
    adag = oqs.create(space)
    n = oqs.num(space)

    assert a.shape == (4, 4)
    assert sp.issparse(a.data)
    assert sp.issparse(adag.data)
    assert sp.issparse(n.data)
    assert a.to_numpy()[0, 1] == 1
    assert np.isclose(a.to_numpy()[1, 2], np.sqrt(2))
    assert np.allclose(adag.to_numpy(), a.to_numpy().conj().T)
    assert np.allclose(np.diag(n.to_numpy()), [0, 1, 2, 3])


def test_qubit_basis_and_density_matrix() -> None:
    atom = oqs.SpinSpace(0.5)
    up = oqs.basis(atom, "up")
    down = oqs.basis(atom, "down")

    assert np.allclose(up, [1, 0])
    assert np.allclose(down, [0, 1])
    assert np.allclose(oqs.ket2dm(up), [[1, 0], [0, 0]])


def test_operator_algebra_and_tensor() -> None:
    atom = oqs.SpinSpace(0.5)
    sx = oqs.sigmax(atom)
    sz = oqs.sigmaz(atom)
    op = sx * sx
    tensored = sx @ sz

    assert sp.issparse(op.data)
    assert sp.issparse(tensored.data)
    assert np.allclose(op.to_numpy(), np.eye(2))
    assert tensored.shape == (4, 4)


def test_spin_s_dicke_operators() -> None:
    space = oqs.SpinSpace(1.0)
    jm = oqs.spin_jm(space)
    jx = oqs.spin_jx(space)
    jz = oqs.spin_jz(space)

    assert jm.shape == (3, 3)
    assert np.isclose(jm.to_numpy()[1, 0], np.sqrt(2))
    assert np.isclose(jm.to_numpy()[2, 1], np.sqrt(2))
    assert np.allclose(jx.to_numpy(), jx.to_numpy().conj().T)
    assert np.allclose(np.diag(jz.to_numpy()), [1, 0, -1])


def test_dicke_space_collective_operators() -> None:
    space = oqs.DickeSpace(4)
    jm = oqs.collective_lowering(space)
    jp = oqs.collective_raising(space)
    jx = oqs.collective_x(space)
    jz = oqs.collective_z(space)
    nex = oqs.collective_excitation(space)

    assert jm.shape == (5, 5)
    assert np.allclose(jm.to_numpy(), oqs.dicke_jm(space).to_numpy())
    assert np.allclose(jp.to_numpy(), jm.to_numpy().conj().T)
    assert np.allclose(jx.to_numpy(), jx.to_numpy().conj().T)
    assert np.allclose(np.diag(jz.to_numpy()), [2, 1, 0, -1, -2])
    assert np.allclose(np.diag(nex.to_numpy()), [4, 3, 2, 1, 0])
    assert np.isclose(jm.to_numpy()[1, 0], 2.0)
    assert np.isclose(jm.to_numpy()[2, 1], np.sqrt(6))
    assert np.allclose(oqs.basis(space, "all_excited"), [1, 0, 0, 0, 0])
    assert np.allclose(oqs.basis(space, "ground"), [0, 0, 0, 0, 1])
    assert np.allclose(oqs.dicke_state(space, 2), [0, 0, 1, 0, 0])


def test_expect_entropy_purity() -> None:
    atom = oqs.SpinSpace(0.5)
    up = oqs.basis(atom, "up")
    rho = oqs.ket2dm(up)

    assert oqs.expect(oqs.sigmaz(atom), up) == 1
    assert oqs.von_neumann_entropy(rho) == 0
    assert oqs.purity(rho) == 1


def test_evaluate_state_observables() -> None:
    states = [
        np.array([1.0, 0.0], dtype=np.complex128),
        np.array([1.0, 1.0j], dtype=np.complex128) / np.sqrt(2),
    ]

    values = oqs.evaluate_state_observables(
        states,
        {
            "norm": lambda ket: np.vdot(ket, ket),
            "first_population": lambda ket: abs(ket[0]) ** 2,
        },
    )

    assert np.allclose(values["norm"], [1.0, 1.0])
    assert np.allclose(values["first_population"], [1.0, 0.5])

    with pytest.raises(ValueError, match="must return a scalar"):
        oqs.evaluate_state_observables(states, {"bad": lambda ket: ket})


def test_partial_traces_and_dicke_mutual_information() -> None:
    bell = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.complex128) / np.sqrt(2)
    rho_a, rho_b = oqs.partial_traces(bell, 2, 2)

    assert np.allclose(rho_a, 0.5 * np.eye(2))
    assert np.allclose(rho_b, 0.5 * np.eye(2))
    assert np.allclose(oqs.partial_trace(oqs.ket2dm(bell), (2, 2), 0), rho_a)
    assert np.isclose(oqs.von_neumann_entropy(rho_a), 1.0)
    assert np.isclose(oqs.bipartite_mutual_information(bell, 2, 2), 2.0)
    assert np.isclose(oqs.mutual_information(bell, (2, 2), 0, 1), 2.0)

    product_rho = np.kron(0.5 * np.eye(2), oqs.ket2dm(np.array([1.0, 0.0])))
    assert np.isclose(oqs.mutual_information(product_rho, (2, 2), 0, 1), 0.0)

    n_spins = 2
    top_dicke = np.array([1.0, 0.0, 0.0], dtype=np.complex128)
    rho_top = oqs.ket2dm(top_dicke)
    reduction = precompute_dicke_reduction(n_spins, 1)
    rho_one_spin = dicke_k_rdm(rho_top, reduction)

    assert rho_one_spin.shape == (2, 2)
    assert np.isclose(np.trace(rho_one_spin), 1.0)
    assert np.isclose(dicke_mutual_information(rho_top, n_spins), 0.0)


def test_two_ensemble_dicke_mutual_information_series() -> None:
    n_spins = 2
    product_top = np.zeros((n_spins + 1) ** 2, dtype=np.complex128)
    product_top[0] = 1.0

    mi_a, mi_b = two_ensemble_dicke_mutual_information(product_top, n_spins)
    series_a, series_b = trajectory_dicke_mutual_information(
        [product_top, product_top],
        n_spins,
    )

    assert np.isclose(mi_a, 0.0)
    assert np.isclose(mi_b, 0.0)
    assert np.allclose(series_a, [0.0, 0.0])
    assert np.allclose(series_b, [0.0, 0.0])
