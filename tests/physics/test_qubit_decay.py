import numpy as np
import pytest


@pytest.mark.physics
def test_qubit_decay_mesolve_matches_analytic_population() -> None:
    import openquantumsim as oqs
    from openquantumsim._julia_bridge import backend_available

    if not backend_available():
        pytest.skip("Julia backend is not available.")

    gamma = 0.35
    atom = oqs.SpinSpace(0.5, label="atom")
    H = 0.0 * oqs.sigmaz(atom)
    collapse = np.sqrt(gamma) * oqs.sigmam(atom)
    excited = oqs.basis(atom, "up")
    rho0 = oqs.ket2dm(excited)
    projector_excited = oqs.Operator(oqs.ket2dm(excited), atom, "P_excited")
    times = np.linspace(0.0, 6.0, 61)

    result = oqs.mesolve(
        H,
        rho0,
        times,
        c_ops=[collapse],
        e_ops=[projector_excited],
        options=oqs.Options(rtol=1e-9, atol=1e-11),
    )

    expected = np.exp(-gamma * times)
    assert np.allclose(result.times, times)
    assert len(result.expect) == 1
    assert np.allclose(result.expect[0].real, expected, rtol=2e-5, atol=2e-7)
    assert np.max(np.abs(result.expect[0].imag)) < 1e-12
    assert result.solver_stats["requested_method"] == "auto"
    assert result.solver_stats["method"] == "ode"


@pytest.mark.physics
def test_mesolve_krylov_and_ode_methods_agree() -> None:
    import openquantumsim as oqs
    from openquantumsim._julia_bridge import backend_available

    if not backend_available():
        pytest.skip("Julia backend is not available.")

    gamma = 0.2
    omega = 0.3
    atom = oqs.SpinSpace(0.5, label="atom")
    H = 0.5 * omega * oqs.sigmax(atom)
    excited = oqs.basis(atom, "up")
    rho0 = oqs.ket2dm(excited)
    collapse = np.sqrt(gamma) * oqs.sigmam(atom)
    excited_projector = oqs.Operator(oqs.ket2dm(excited), atom, "P_excited")
    times = np.linspace(0.0, 1.0, 11)

    krylov = oqs.mesolve(
        H,
        rho0,
        times,
        c_ops=[collapse],
        e_ops=[excited_projector],
        options=oqs.Options(method="krylov", krylov_dim=20, rtol=1e-9, atol=1e-11),
    )
    ode = oqs.mesolve(
        H,
        rho0,
        times,
        c_ops=[collapse],
        e_ops=[excited_projector],
        options=oqs.Options(method="ode", krylov_dim=20, rtol=1e-9, atol=1e-11),
    )

    assert np.allclose(krylov.expect[0], ode.expect[0], rtol=1e-7, atol=1e-9)
    assert krylov.solver_stats["method"] == "krylov"
    assert krylov.solver_stats["krylov_dim"] == 20


@pytest.mark.physics
def test_steadystate_qubit_decay_returns_ground_state() -> None:
    import openquantumsim as oqs
    from openquantumsim._julia_bridge import backend_available

    if not backend_available():
        pytest.skip("Julia backend is not available.")

    gamma = 0.35
    atom = oqs.SpinSpace(0.5, label="atom")
    H = 0.0 * oqs.sigmaz(atom)
    collapse = np.sqrt(gamma) * oqs.sigmam(atom)
    ground = oqs.basis(atom, "down")
    expected = oqs.ket2dm(ground)

    direct = oqs.steadystate(H, [collapse], method="direct")
    iterative = oqs.steadystate(
        H,
        [collapse],
        method="iterative-gmres",
        options=oqs.Options(rtol=1e-12, krylov_dim=10),
    )

    assert np.allclose(direct, expected, atol=1e-10)
    assert np.allclose(iterative, expected, atol=1e-10)
    assert np.isclose(np.trace(direct), 1.0)
    assert np.allclose(direct, direct.conj().T)
