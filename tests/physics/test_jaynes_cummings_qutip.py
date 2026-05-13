import numpy as np
import pytest


@pytest.mark.physics
def test_jaynes_cummings_mesolve_matches_qutip() -> None:
    qutip = pytest.importorskip("qutip")
    import openquantumsim as oqs
    from openquantumsim._julia_bridge import backend_available

    if not backend_available():
        pytest.skip("Julia backend is not available.")

    cavity_dim = 5
    cavity_frequency = 1.0
    atom_frequency = 1.0
    coupling = 0.08
    cavity_decay = 0.015
    atom_decay = 0.025
    times = np.linspace(0.0, 8.0, 81)

    system = oqs.jaynes_cummings_system(
        cavity_dim,
        cavity_frequency=cavity_frequency,
        atom_frequency=atom_frequency,
        coupling=coupling,
        cavity_decay=cavity_decay,
        atom_decay=atom_decay,
    )
    oqs_result = oqs.mesolve(
        system.H,
        system.rho0,
        times,
        c_ops=system.c_ops,
        e_ops=system.e_ops,
        options=oqs.Options(method="ode", rtol=1e-9, atol=1e-11),
    )

    qutip_result = _qutip_jaynes_cummings(
        qutip,
        cavity_dim=cavity_dim,
        cavity_frequency=cavity_frequency,
        atom_frequency=atom_frequency,
        coupling=coupling,
        cavity_decay=cavity_decay,
        atom_decay=atom_decay,
        times=times,
    )

    assert len(oqs_result.expect) == 2
    assert len(qutip_result.expect) == 2
    assert np.allclose(oqs_result.expect[0].real, qutip_result.expect[0], atol=5e-7)
    assert np.allclose(oqs_result.expect[1].real, qutip_result.expect[1], atol=5e-7)
    assert np.max(np.abs(oqs_result.expect[0].imag)) < 1e-12
    assert np.max(np.abs(oqs_result.expect[1].imag)) < 1e-12


def _qutip_jaynes_cummings(
    qutip: object,
    *,
    cavity_dim: int,
    cavity_frequency: float,
    atom_frequency: float,
    coupling: float,
    cavity_decay: float,
    atom_decay: float,
    times: np.ndarray,
) -> object:
    cavity_eye = qutip.qeye(cavity_dim)
    atom_eye = qutip.qeye(2)
    a = qutip.tensor(qutip.destroy(cavity_dim), atom_eye)
    sm = qutip.tensor(cavity_eye, qutip.sigmam())
    sz = qutip.tensor(cavity_eye, qutip.sigmaz())
    n_cavity = a.dag() * a
    excited_projector = qutip.tensor(
        cavity_eye,
        qutip.basis(2, 0) * qutip.basis(2, 0).dag(),
    )
    hamiltonian = (
        cavity_frequency * n_cavity
        + 0.5 * atom_frequency * sz
        + coupling * (a.dag() * sm + a * sm.dag())
    )
    psi0 = qutip.tensor(qutip.basis(cavity_dim, 0), qutip.basis(2, 0))
    collapse_ops = []
    if cavity_decay > 0:
        collapse_ops.append(np.sqrt(cavity_decay) * a)
    if atom_decay > 0:
        collapse_ops.append(np.sqrt(atom_decay) * sm)
    return qutip.mesolve(
        hamiltonian,
        psi0 * psi0.dag(),
        times,
        c_ops=collapse_ops,
        e_ops=[n_cavity, excited_projector],
        options={"rtol": 1e-10, "atol": 1e-12, "store_states": False},
    )
