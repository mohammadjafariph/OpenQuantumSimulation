import numpy as np
import pytest

import openquantumsim as oqs
from openquantumsim.solvers import _to_python_dict


class _NamedTupleLikeStats:
    nsteps = 12
    retcode = "Success"
    method = "ode"

    def __dir__(self) -> list[str]:
        return ["method", "nsteps", "retcode"]

    def __getitem__(self, _key: object) -> object:
        raise AssertionError("_to_python_dict should not probe missing fields")


def test_solver_stats_conversion_uses_present_fields_only() -> None:
    stats = _to_python_dict(_NamedTupleLikeStats())

    assert stats == {"nsteps": 12, "retcode": "Success", "method": "ode"}


def test_solver_stats_conversion_accepts_mappings() -> None:
    assert _to_python_dict({"retcode": "Success"}) == {"retcode": "Success"}

def test_propagator_unitary_rotation_and_composition() -> None:
    omega = 0.7
    qubit = oqs.SpinSpace(0.5, label="q")
    H = (omega / 2.0) * oqs.sigmaz(qubit)
    times = [0.0, 1.0, 2.5]

    props = oqs.propagator(H, times)
    assert len(props) == 3
    for time, prop in zip(times, props, strict=True):
        u = prop.to_numpy()
        # unitarity
        np.testing.assert_allclose(u.conj().T @ u, np.eye(2), atol=1e-12)
        # exact rotation about z: |up> picks up a phase
        phase = u[0, 0]
        assert phase == pytest.approx(np.exp(-1j * omega * time / 2.0))

    # time-independent composition: U(2.5) = U(1.0) U(1.5)
    u_nested = oqs.propagator(H, [1.5])[0].to_numpy() @ props[1].to_numpy()
    np.testing.assert_allclose(u_nested, props[2].to_numpy(), atol=1e-10)


def test_propagator_superoperator_matches_amplitude_damping_analytic() -> None:
    gamma = 0.35
    qubit = oqs.SpinSpace(0.5, label="q")
    H = 0.0 * oqs.sigmaz(qubit)
    collapse = np.sqrt(gamma) * oqs.sigmam(qubit)
    times = [0.0, 0.5, 2.0]

    props = oqs.propagator(H, times, c_ops=[collapse])
    excited = oqs.ket2dm(oqs.basis(qubit, "up"))
    ground = oqs.ket2dm(oqs.basis(qubit, "down"))
    mixed = 0.5 * np.eye(2, dtype=np.complex128)

    for time, prop in zip(times, props, strict=True):
        superop = prop.to_numpy()

        # trace preservation on a general state
        arbitrary = np.array(
            [[0.4, 0.2j], [-0.2j, 0.6]], dtype=np.complex128
        )
        out_arbitrary = oqs.apply_superoperator(superop, arbitrary)
        assert np.real(np.trace(out_arbitrary)) == pytest.approx(1.0, abs=1e-10)

        # the ground state is a fixed point of amplitude damping
        ground_out = oqs.apply_superoperator(superop, ground)
        np.testing.assert_allclose(ground_out, ground, atol=1e-12)

        # excited-state population decays as exp(-gamma t)
        out = oqs.apply_superoperator(superop, excited)
        assert np.real(out[0, 0]) == pytest.approx(np.exp(-gamma * time), abs=1e-10)
        assert np.real(np.trace(out)) == pytest.approx(1.0, abs=1e-10)

        # amplitude damping polarizes the maximally mixed state toward the
        # ground state (the channel is not unital).
        out_mixed = oqs.apply_superoperator(superop, mixed)
        assert np.real(out_mixed[0, 0]) == pytest.approx(
            0.5 * np.exp(-gamma * time), abs=1e-10
        )
        assert np.real(out_mixed[1, 1]) == pytest.approx(
            1.0 - 0.5 * np.exp(-gamma * time), abs=1e-10
        )

        # coherence of |+x> decays as exp(-gamma t / 2)
        plus = oqs.ket2dm(
            (oqs.basis(qubit, "up") + oqs.basis(qubit, "down")) / np.sqrt(2.0)
        )
        out_plus = oqs.apply_superoperator(superop, plus)
        assert np.real(out_plus[0, 1]) == pytest.approx(
            0.5 * np.exp(-0.5 * gamma * time), abs=1e-10
        )


def test_lindblad_superoperator_and_validation() -> None:
    qubit = oqs.SpinSpace(0.5, label="q")
    H = 0.0 * oqs.sigmaz(qubit)
    collapse = np.sqrt(0.2) * oqs.sigmam(qubit)

    generator = oqs.lindblad_superoperator(H, [collapse])
    assert generator.shape == (4, 4)

    with pytest.raises(ValueError, match="increasing"):
        oqs.propagator(H, [1.0, 0.5])

    with pytest.raises(ValueError, match="non-empty"):
        oqs.propagator(H, [])
