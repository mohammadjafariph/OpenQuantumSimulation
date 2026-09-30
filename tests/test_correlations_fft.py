import numpy as np
import pytest

import openquantumsim as oqs


def test_spectrum_correlation_fft_peaks_at_oscillation_frequency() -> None:
    gamma = 0.5
    omega0 = 2.0
    taus = np.linspace(0.0, 20.0, 401)
    corr = np.exp(-0.5 * gamma * taus) * np.exp(-1j * omega0 * taus)

    wlist, spectrum = oqs.spectrum_correlation_fft(taus, corr)

    assert wlist.shape == taus.shape
    assert spectrum.shape == taus.shape
    assert np.all(np.diff(wlist) > 0)
    peak = wlist[np.argmax(np.abs(spectrum))]
    assert peak == pytest.approx(omega0, abs=2.0 * np.pi / (taus[-1] - taus[0]))


def test_spectrum_correlation_fft_real_exponential_is_lorentzian() -> None:
    gamma = 0.4
    taus = np.linspace(0.0, 30.0, 601)
    corr = np.exp(-0.5 * gamma * taus)

    wlist, spectrum = oqs.spectrum_correlation_fft(taus, corr)

    # The FFT estimator computes the discrete-time spectrum of the sampled
    # exponential, dt / (1 - exp((i w - gamma/2) dt)); compare against that
    # exact form inside the sampled band, away from the Nyquist edge.
    dt = taus[1] - taus[0]
    in_band = np.abs(wlist) < 0.25 * np.pi / dt
    assert np.count_nonzero(in_band) > 10
    ratio = np.exp((1j * wlist - 0.5 * gamma) * dt)
    exact_dtft = dt / (1.0 - ratio)
    assert np.allclose(spectrum[in_band], exact_dtft[in_band], rtol=5e-3, atol=5e-3)


def test_spectrum_correlation_fft_validates_inputs() -> None:
    taus = np.linspace(0.0, 1.0, 11)
    corr = np.ones(11, dtype=np.complex128)

    with pytest.raises(ValueError, match="matching taulist"):
        oqs.spectrum_correlation_fft(taus, corr[:-2])

    with pytest.raises(ValueError, match="two samples"):
        oqs.spectrum_correlation_fft(taus[:1], corr[:1])

    with pytest.raises(ValueError, match="increasing"):
        oqs.spectrum_correlation_fft(taus[::-1], corr)

    with pytest.raises(ValueError, match="uniformly spaced"):
        ragged = np.array([0.0, 0.1, 0.3, 0.4, 0.5])
        oqs.spectrum_correlation_fft(ragged, np.ones(5, dtype=np.complex128))