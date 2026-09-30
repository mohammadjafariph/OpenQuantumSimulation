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

@pytest.mark.physics
def test_propagator_superoperator_matches_mesolve() -> None:
    import openquantumsim as oqs
    from openquantumsim._julia_bridge import backend_available

    if not backend_available():
        pytest.skip("Julia backend is not available.")

    gamma = 0.3
    qubit = oqs.SpinSpace(0.5, label="q")
    H = 0.0 * oqs.sigmaz(qubit)
    collapse = np.sqrt(gamma) * oqs.sigmam(qubit)
    times = [0.0, 0.4, 1.2]
    rho0 = oqs.ket2dm(oqs.basis(qubit, "up"))

    result = oqs.mesolve(
        H,
        rho0,
        times,
        c_ops=[collapse],
        e_ops=[oqs.sigmaz(qubit)],
        options=oqs.Options(rtol=1e-9, atol=1e-11, save_states=True),
    )

    current = rho0
    previous_time = 0.0
    for time in times:
        step = oqs.propagator(H, [time - previous_time], c_ops=[collapse])[0]
        current = oqs.apply_superoperator(step.to_numpy(), current)
        previous_time = time
        state = result.states[times.index(time)]
        assert oqs.fidelity(current, state) == pytest.approx(1.0, abs=1e-6)

@pytest.mark.physics
def test_brmesolve_zero_temperature_damping_matches_mesolve() -> None:
    import openquantumsim as oqs
    from openquantumsim._julia_bridge import backend_available

    if not backend_available():
        pytest.skip("Julia backend is not available.")

    kappa = 0.4
    qubit = oqs.SpinSpace(0.5, label="q")
    H = 0.5 * oqs.sigmaz(qubit)
    rho0 = oqs.ket2dm(oqs.basis(qubit, "up"))
    times = [0.0, 0.5, 1.5]

    def spectrum(omega: float) -> float:
        # Decay channels sit at negative Bohr frequencies.
        return kappa if omega < 0 else 0.0

    br = oqs.brmesolve(
        H,
        rho0,
        times,
        a_ops=[(oqs.sigmam(qubit), spectrum)],
        e_ops=[oqs.sigmaz(qubit)],
        options=oqs.Options(rtol=1e-9, atol=1e-11, save_states=True),
    )
    lind = oqs.mesolve(
        H,
        rho0,
        times,
        c_ops=[np.sqrt(kappa) * oqs.sigmam(qubit)],
        e_ops=[oqs.sigmaz(qubit)],
        options=oqs.Options(rtol=1e-9, atol=1e-11, save_states=True),
    )

    np.testing.assert_allclose(br.expect[0].real, lind.expect[0].real, atol=1e-8)
    for br_state, lind_state in zip(br.states, lind.states, strict=True):
        assert oqs.fidelity(br_state, lind_state) == pytest.approx(1.0, abs=1e-8)


@pytest.mark.physics
def test_brmesolve_white_noise_dephasing_rate() -> None:
    import openquantumsim as oqs
    from openquantumsim._julia_bridge import backend_available

    if not backend_available():
        pytest.skip("Julia backend is not available.")

    qubit = oqs.SpinSpace(0.5, label="q")
    H = 0.5 * oqs.sigmaz(qubit)
    rho_plus = oqs.ket2dm(
        (oqs.basis(qubit, "up") + oqs.basis(qubit, "down")) / np.sqrt(2.0)
    )
    times = [0.0, 0.5, 1.5]

    dep = oqs.brmesolve(
        H,
        rho_plus,
        times,
        a_ops=[(oqs.sigmaz(qubit), lambda omega: 0.3)],
        options=oqs.Options(rtol=1e-9, atol=1e-11, save_states=True),
    )

    for time, state in zip(times, dep.states, strict=True):
        # sigma_z dephasing damps coherences at rate 2 * gamma
        assert complex(state[0, 1]).real == pytest.approx(
            0.5 * np.cos(time) * np.exp(-0.6 * time), abs=1e-8
        )
        assert complex(state[0, 1]).imag == pytest.approx(
            -0.5 * np.sin(time) * np.exp(-0.6 * time), abs=1e-8
        )
