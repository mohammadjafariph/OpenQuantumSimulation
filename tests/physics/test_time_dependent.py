import numpy as np
import pytest


@pytest.mark.physics
def test_mesolve_time_dependent_constant_drive_matches_rabi() -> None:
    import openquantumsim as oqs
    from openquantumsim._julia_bridge import backend_available

    if not backend_available():
        pytest.skip("Julia backend is not available.")

    omega = 0.8
    atom = oqs.SpinSpace(0.5, label="atom")
    H0 = 0.0 * oqs.sigmaz(atom)
    H = oqs.time_dependent_hamiltonian(
        H0,
        [(oqs.sigmax(atom), lambda _t: 0.5 * omega)],
    )
    excited = oqs.basis(atom, "up")
    rho0 = oqs.ket2dm(excited)
    excited_projector = oqs.Operator(oqs.ket2dm(excited), atom, "P_excited")
    times = np.linspace(0.0, 8.0, 81)

    result = oqs.mesolve(
        H,
        rho0,
        times,
        e_ops=[excited_projector],
        options=oqs.Options(rtol=1e-9, atol=1e-11),
    )

    expected = np.cos(0.5 * omega * times) ** 2
    assert np.allclose(result.times, times)
    assert np.allclose(result.expect[0].real, expected, rtol=2e-6, atol=2e-7)
    assert np.max(np.abs(result.expect[0].imag)) < 1e-12
    assert result.solver_stats["method"] == "time-dependent-ode"
    assert result.solver_stats["time_dependent"] is True


@pytest.mark.physics
def test_mesolve_time_dependent_interpolated_coefficient() -> None:
    import openquantumsim as oqs
    from openquantumsim._julia_bridge import backend_available

    if not backend_available():
        pytest.skip("Julia backend is not available.")

    omega = 0.4
    atom = oqs.SpinSpace(0.5, label="atom")
    coefficient = oqs.InterpolatedCoefficient([0.0, 8.0], [0.5 * omega, 0.5 * omega])
    H = oqs.time_dependent_hamiltonian(
        0.0 * oqs.sigmaz(atom),
        [(oqs.sigmax(atom), coefficient)],
    )
    excited = oqs.basis(atom, "up")
    excited_projector = oqs.Operator(oqs.ket2dm(excited), atom, "P_excited")
    times = np.linspace(0.0, 8.0, 41)

    result = oqs.mesolve(
        H,
        oqs.ket2dm(excited),
        times,
        e_ops=[excited_projector],
        options=oqs.Options(rtol=1e-9, atol=1e-11),
    )

    expected = np.cos(0.5 * omega * times) ** 2
    assert np.allclose(result.expect[0].real, expected, rtol=2e-6, atol=2e-7)
