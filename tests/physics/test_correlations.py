import numpy as np
import pytest


@pytest.mark.physics
def test_correlation_2op_1t_qubit_decay_matches_analytic() -> None:
    import openquantumsim as oqs
    from openquantumsim._julia_bridge import backend_available

    if not backend_available():
        pytest.skip("Julia backend is not available.")

    gamma = 0.4
    atom = oqs.SpinSpace(0.5, label="atom")
    H = 0.0 * oqs.sigmaz(atom)
    excited = oqs.basis(atom, "up")
    rho0 = oqs.ket2dm(excited)
    collapse = np.sqrt(gamma) * oqs.sigmam(atom)
    taus = np.linspace(0.0, 4.0, 41)

    corr = oqs.correlation_2op_1t(
        H,
        rho0,
        taus,
        oqs.sigmap(atom),
        oqs.sigmam(atom),
        c_ops=[collapse],
        options=oqs.Options(rtol=1e-9, atol=1e-11),
    )

    expected = np.exp(-0.5 * gamma * taus)
    assert corr.shape == taus.shape
    assert np.allclose(corr.real, expected, rtol=2e-6, atol=2e-8)
    assert np.max(np.abs(corr.imag)) < 1e-12


@pytest.mark.physics
def test_correlation_2op_2t_qubit_decay_matches_analytic() -> None:
    import openquantumsim as oqs
    from openquantumsim._julia_bridge import backend_available

    if not backend_available():
        pytest.skip("Julia backend is not available.")

    gamma = 0.3
    atom = oqs.SpinSpace(0.5, label="atom")
    H = 0.0 * oqs.sigmaz(atom)
    excited = oqs.basis(atom, "up")
    rho0 = oqs.ket2dm(excited)
    collapse = np.sqrt(gamma) * oqs.sigmam(atom)
    times = np.linspace(0.0, 3.0, 7)
    taus = np.linspace(0.0, 2.0, 9)

    corr = oqs.correlation_2op_2t(
        H,
        rho0,
        times,
        taus,
        oqs.sigmap(atom),
        oqs.sigmam(atom),
        c_ops=[collapse],
        options=oqs.Options(rtol=1e-9, atol=1e-11),
    )

    expected = np.exp(-gamma * times[:, None]) * np.exp(-0.5 * gamma * taus[None, :])
    assert corr.shape == expected.shape
    assert np.allclose(corr.real, expected, rtol=2e-6, atol=2e-8)
    assert np.max(np.abs(corr.imag)) < 1e-12


def test_correlation_validates_inputs_before_backend_load() -> None:
    import openquantumsim as oqs

    atom = oqs.SpinSpace(0.5)
    H = oqs.sigmaz(atom)
    rho0 = oqs.ket2dm(oqs.basis(atom, "up"))

    with pytest.raises(ValueError, match="non-negative"):
        oqs.correlation_2op_1t(
            H,
            rho0,
            [-1.0],
            oqs.sigmap(atom),
            oqs.sigmam(atom),
        )
