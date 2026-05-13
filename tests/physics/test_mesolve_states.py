import numpy as np
import pytest


def _assert_density_matrices_are_physical(states: list[np.ndarray]) -> None:
    for rho in states:
        assert np.isclose(np.trace(rho), 1.0, atol=2e-8)
        assert np.allclose(rho, rho.conj().T, atol=2e-8)
        eigenvalues = np.linalg.eigvalsh(0.5 * (rho + rho.conj().T))
        assert np.min(eigenvalues) > -2e-8


@pytest.mark.physics
def test_mesolve_save_states_preserves_density_matrix_invariants() -> None:
    import openquantumsim as oqs
    from openquantumsim._julia_bridge import backend_available

    if not backend_available():
        pytest.skip("Julia backend is not available.")

    gamma = 0.2
    atom = oqs.SpinSpace(0.5, label="atom")
    H = 0.15 * oqs.sigmax(atom)
    excited = oqs.basis(atom, "up")
    rho0 = oqs.ket2dm(excited)
    times = np.linspace(0.0, 3.0, 31)

    result = oqs.mesolve(
        H,
        rho0,
        times,
        c_ops=[np.sqrt(gamma) * oqs.sigmam(atom)],
        options=oqs.Options(rtol=1e-9, atol=1e-11, save_states=True),
    )

    assert result.states is not None
    assert len(result.states) == len(times)
    assert all(rho.shape == (2, 2) for rho in result.states)
    assert np.allclose(result.states[0], rho0, atol=1e-12)
    _assert_density_matrices_are_physical(result.states)


@pytest.mark.physics
def test_mesolve_state_observables_without_returning_states() -> None:
    import openquantumsim as oqs
    from openquantumsim._julia_bridge import backend_available

    if not backend_available():
        pytest.skip("Julia backend is not available.")

    atom = oqs.SpinSpace(0.5, label="atom")
    H = 0.0 * oqs.sigmaz(atom)
    excited = oqs.basis(atom, "up")
    rho0 = oqs.ket2dm(excited)
    times = np.linspace(0.0, 0.2, 3)

    result = oqs.mesolve(
        H,
        rho0,
        times,
        state_observables=oqs.state_metrics(
            purity=True,
            fidelity_to=excited,
            population_indices=[0, 1],
        ),
        options=oqs.Options(rtol=1e-9, atol=1e-11, save_states=False),
    )

    assert result.states is None
    assert np.allclose(result.state_observables["purity"], np.ones_like(times))
    assert np.allclose(result.state_observables["fidelity"], np.ones_like(times))
    assert np.allclose(result.state_observables["population_0"], np.ones_like(times))
    assert np.allclose(result.state_observables["population_1"], np.zeros_like(times))


@pytest.mark.physics
def test_mesolve_hamiltonian_rabi_oscillation_matches_analytic_population() -> None:
    import openquantumsim as oqs
    from openquantumsim._julia_bridge import backend_available

    if not backend_available():
        pytest.skip("Julia backend is not available.")

    omega = 0.8
    atom = oqs.SpinSpace(0.5, label="atom")
    H = 0.5 * omega * oqs.sigmax(atom)
    excited = oqs.basis(atom, "up")
    rho0 = oqs.ket2dm(excited)
    excited_projector = oqs.Operator(oqs.ket2dm(excited), atom, "P_excited")
    times = np.linspace(0.0, 8.0, 81)

    result = oqs.mesolve(
        H,
        rho0,
        times,
        e_ops=[excited_projector],
        options=oqs.Options(rtol=1e-9, atol=1e-11, save_states=True),
    )

    expected = np.cos(0.5 * omega * times) ** 2
    assert np.allclose(result.expect[0].real, expected, rtol=2e-6, atol=2e-7)
    assert np.max(np.abs(result.expect[0].imag)) < 1e-12
    assert result.states is not None
    _assert_density_matrices_are_physical(result.states)
