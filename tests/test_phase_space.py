import numpy as np
import pytest

import openquantumsim as oqs


def test_wigner_vacuum_normalization_and_peak() -> None:
    x, p = oqs.phase_space_grid(xlim=(-5.0, 5.0), points=151)
    vacuum = oqs.fock(oqs.FockSpace(8), 0)

    values = oqs.wigner(vacuum, x, p)
    dx = x[1] - x[0]
    dp = p[1] - p[0]

    assert values.shape == (len(p), len(x))
    assert values[len(p) // 2, len(x) // 2] == pytest.approx(1 / np.pi, rel=1e-3)
    assert np.sum(values) * dx * dp == pytest.approx(1.0, abs=2e-5)


def test_wigner_fock_one_has_negative_origin() -> None:
    x = np.linspace(-2.0, 2.0, 81)
    one_photon = oqs.fock(oqs.FockSpace(4), 1)

    values = oqs.wigner(oqs.ket2dm(one_photon), x)

    assert values[len(x) // 2, len(x) // 2] == pytest.approx(-1 / np.pi, rel=1e-3)


def test_wigner_coherent_state_peak_location() -> None:
    alpha = 1.0 + 0.5j
    x = np.linspace(-3.0, 3.0, 121)
    p = np.linspace(-3.0, 3.0, 121)
    state = oqs.coherent(oqs.FockSpace(24), alpha)

    values = oqs.wigner(state, x, p)
    peak_p_idx, peak_x_idx = np.unravel_index(np.argmax(values), values.shape)

    assert x[peak_x_idx] == pytest.approx(np.sqrt(2.0) * alpha.real, abs=0.08)
    assert p[peak_p_idx] == pytest.approx(np.sqrt(2.0) * alpha.imag, abs=0.08)


def test_q_function_vacuum_convention() -> None:
    x, p = oqs.phase_space_grid(xlim=(-6.0, 6.0), points=161)
    vacuum = oqs.fock(oqs.FockSpace(8), 0)

    values = oqs.q_function(vacuum, x, p)
    dx = x[1] - x[0]
    dp = p[1] - p[0]

    assert values.shape == (len(p), len(x))
    assert values[len(p) // 2, len(x) // 2] == pytest.approx(1 / np.pi, rel=1e-3)
    assert np.sum(values) * dx * dp == pytest.approx(2.0, abs=2e-4)
    assert np.min(values) >= 0.0


def test_phase_space_input_validation() -> None:
    with pytest.raises(ValueError, match="square density"):
        oqs.wigner(np.ones((2, 3), dtype=np.complex128), [0.0])

    with pytest.raises(ValueError, match="strictly ascending"):
        oqs.q_function(np.array([1.0, 0.0], dtype=np.complex128), [0.0, 0.0])

    with pytest.raises(ValueError, match="points"):
        oqs.phase_space_grid(points=1)
