from pathlib import Path

import numpy as np
import pytest


@pytest.mark.physics
def test_single_trajectory_can_save_ket_states() -> None:
    import openquantumsim as oqs
    from openquantumsim._julia_bridge import backend_available

    if not backend_available():
        pytest.skip("Julia backend is not available.")

    atom = oqs.SpinSpace(0.5, label="atom")
    H = 0.0 * oqs.sigmaz(atom)
    excited = oqs.basis(atom, "up")
    excited_projector = oqs.Operator(oqs.ket2dm(excited), atom, "P_excited")
    times = np.linspace(0.0, 0.2, 3)

    result = oqs.single_trajectory(
        H,
        excited,
        times,
        e_ops=[excited_projector],
        options=oqs.Options(seed=909, max_step=0.05, save_states=True),
    )

    assert result.solver_stats["n_traj"] == 1
    assert result.states is not None
    assert len(result.states) == len(times)
    assert all(state.shape == excited.shape for state in result.states)
    assert all(np.isclose(np.linalg.norm(state), 1.0) for state in result.states)
    assert np.allclose(result.states[0], excited)
    assert np.allclose(result.expect[0].real, np.ones_like(times))


@pytest.mark.physics
def test_single_trajectory_state_observables_without_returning_states() -> None:
    import openquantumsim as oqs
    from openquantumsim._julia_bridge import backend_available

    if not backend_available():
        pytest.skip("Julia backend is not available.")

    atom = oqs.SpinSpace(0.5, label="atom")
    H = 0.0 * oqs.sigmaz(atom)
    excited = oqs.basis(atom, "up")
    times = np.linspace(0.0, 0.2, 3)

    result = oqs.single_trajectory(
        H,
        excited,
        times,
        state_observables=oqs.state_metrics(
            purity=True,
            fidelity_to=excited,
            population_indices=[0],
        ),
        options=oqs.Options(seed=909, max_step=0.05, save_states=False),
    )

    assert result.states is None
    assert np.allclose(result.state_observables["purity"], np.ones_like(times))
    assert np.allclose(result.state_observables["fidelity"], np.ones_like(times))
    assert np.allclose(result.state_observables["population_0"], np.ones_like(times))


@pytest.mark.physics
def test_mcsolve_qubit_decay_tracks_analytic_population() -> None:
    import openquantumsim as oqs
    from openquantumsim._julia_bridge import backend_available

    if not backend_available():
        pytest.skip("Julia backend is not available.")

    gamma = 0.35
    atom = oqs.SpinSpace(0.5, label="atom")
    H = 0.0 * oqs.sigmaz(atom)
    collapse = np.sqrt(gamma) * oqs.sigmam(atom)
    excited = oqs.basis(atom, "up")
    excited_projector = oqs.Operator(oqs.ket2dm(excited), atom, "P_excited")
    times = np.linspace(0.0, 4.0, 41)

    result = oqs.mcsolve(
        H,
        excited,
        times,
        c_ops=[collapse],
        e_ops=[excited_projector],
        n_traj=1500,
        options=oqs.Options(seed=2026, max_step=0.01),
    )

    expected = np.exp(-gamma * times)
    population = result.expect[0].real

    assert result.solver_stats["n_traj"] == 1500
    assert len(result.expect) == 1
    assert len(result.expect_std) == 1
    assert len(result.expect_stderr) == 1
    assert result.expect_std[0].shape == times.shape
    assert result.expect_stderr[0].shape == times.shape
    assert np.all(result.expect_std[0] >= 0.0)
    assert np.all(result.expect_stderr[0] >= 0.0)
    assert np.all(population >= -1e-12)
    assert np.all(population <= 1.0 + 1e-12)
    assert np.mean(np.abs(population - expected)) < 0.025
    assert np.max(np.abs(population - expected)) < 0.07


@pytest.mark.physics
def test_mcsolve_standard_error_shrinks_with_trajectory_count() -> None:
    import openquantumsim as oqs
    from openquantumsim._julia_bridge import backend_available

    if not backend_available():
        pytest.skip("Julia backend is not available.")

    gamma = 0.35
    atom = oqs.SpinSpace(0.5, label="atom")
    H = 0.0 * oqs.sigmaz(atom)
    collapse = np.sqrt(gamma) * oqs.sigmam(atom)
    excited = oqs.basis(atom, "up")
    excited_projector = oqs.Operator(oqs.ket2dm(excited), atom, "P_excited")
    times = np.linspace(0.0, 3.0, 31)

    low = oqs.mcsolve(
        H,
        excited,
        times,
        c_ops=[collapse],
        e_ops=[excited_projector],
        n_traj=300,
        options=oqs.Options(seed=44, max_step=0.01),
    )
    high = oqs.mcsolve(
        H,
        excited,
        times,
        c_ops=[collapse],
        e_ops=[excited_projector],
        n_traj=1200,
        options=oqs.Options(seed=44, max_step=0.01),
    )

    expected_ratio = np.sqrt(300 / 1200)
    mask = low.expect_std[0] > 0.05
    stderr_ratio = np.median(high.expect_stderr[0][mask] / low.expect_stderr[0][mask])

    assert 0.35 < stderr_ratio < 0.65
    assert np.isclose(stderr_ratio, expected_ratio, atol=0.15)


@pytest.mark.physics
def test_mcsolve_serial_and_threaded_requests_are_deterministic() -> None:
    import openquantumsim as oqs
    from openquantumsim._julia_bridge import backend_available

    if not backend_available():
        pytest.skip("Julia backend is not available.")

    gamma = 0.35
    atom = oqs.SpinSpace(0.5, label="atom")
    H = 0.0 * oqs.sigmaz(atom)
    collapse = np.sqrt(gamma) * oqs.sigmam(atom)
    excited = oqs.basis(atom, "up")
    excited_projector = oqs.Operator(oqs.ket2dm(excited), atom, "P_excited")
    times = np.linspace(0.0, 2.0, 21)

    serial = oqs.mcsolve(
        H,
        excited,
        times,
        c_ops=[collapse],
        e_ops=[excited_projector],
        n_traj=250,
        options=oqs.Options(seed=77, max_step=0.01, n_jobs=1),
    )
    threaded = oqs.mcsolve(
        H,
        excited,
        times,
        c_ops=[collapse],
        e_ops=[excited_projector],
        n_traj=250,
        options=oqs.Options(seed=77, max_step=0.01, n_jobs=-1),
    )

    assert threaded.solver_stats["n_jobs_requested"] == -1
    assert threaded.solver_stats["n_workers"] >= 1
    assert np.allclose(serial.expect[0], threaded.expect[0])
    assert np.allclose(serial.expect_std[0], threaded.expect_std[0])
    assert np.allclose(serial.expect_stderr[0], threaded.expect_stderr[0])


@pytest.mark.physics
def test_mcsolve_checkpoint_resume_matches_clean_run(tmp_path: Path) -> None:
    import openquantumsim as oqs
    from openquantumsim._julia_bridge import backend_available

    if not backend_available():
        pytest.skip("Julia backend is not available.")

    gamma = 0.35
    atom = oqs.SpinSpace(0.5, label="atom")
    H = 0.0 * oqs.sigmaz(atom)
    collapse = np.sqrt(gamma) * oqs.sigmam(atom)
    excited = oqs.basis(atom, "up")
    excited_projector = oqs.Operator(oqs.ket2dm(excited), atom, "P_excited")
    times = np.linspace(0.0, 2.0, 21)
    checkpoint_file = tmp_path / "mcsolve_checkpoint.h5"

    oqs.mcsolve(
        H,
        excited,
        times,
        c_ops=[collapse],
        e_ops=[excited_projector],
        n_traj=30,
        options=oqs.Options(
            seed=505,
            max_step=0.01,
            n_jobs=1,
            checkpoint_file=str(checkpoint_file),
            checkpoint_every=10,
        ),
    )
    resumed = oqs.mcsolve(
        H,
        excited,
        times,
        c_ops=[collapse],
        e_ops=[excited_projector],
        n_traj=80,
        options=oqs.Options(
            seed=505,
            max_step=0.01,
            n_jobs=1,
            checkpoint_file=str(checkpoint_file),
            checkpoint_every=10,
        ),
    )
    clean = oqs.mcsolve(
        H,
        excited,
        times,
        c_ops=[collapse],
        e_ops=[excited_projector],
        n_traj=80,
        options=oqs.Options(seed=505, max_step=0.01, n_jobs=1),
    )

    assert checkpoint_file.exists()
    assert resumed.solver_stats["resumed"] is True
    assert resumed.solver_stats["checkpoint_start_completed"] == 30
    assert resumed.solver_stats["checkpoint_completed"] == 80
    assert resumed.solver_stats["checkpoint_previous_target_n_traj"] == 30
    assert np.allclose(resumed.expect[0], clean.expect[0])
    assert np.allclose(resumed.expect_std[0], clean.expect_std[0])
    assert np.allclose(resumed.expect_stderr[0], clean.expect_stderr[0])


@pytest.mark.physics
def test_mcsolve_progress_flag_does_not_change_results() -> None:
    import openquantumsim as oqs
    from openquantumsim._julia_bridge import backend_available

    if not backend_available():
        pytest.skip("Julia backend is not available.")

    gamma = 0.35
    atom = oqs.SpinSpace(0.5, label="atom")
    H = 0.0 * oqs.sigmaz(atom)
    collapse = np.sqrt(gamma) * oqs.sigmam(atom)
    excited = oqs.basis(atom, "up")
    excited_projector = oqs.Operator(oqs.ket2dm(excited), atom, "P_excited")
    times = np.linspace(0.0, 1.0, 11)

    quiet = oqs.mcsolve(
        H,
        excited,
        times,
        c_ops=[collapse],
        e_ops=[excited_projector],
        n_traj=25,
        options=oqs.Options(
            seed=606,
            max_step=0.01,
            n_jobs=1,
            checkpoint_every=10,
            progress=False,
        ),
    )
    noisy = oqs.mcsolve(
        H,
        excited,
        times,
        c_ops=[collapse],
        e_ops=[excited_projector],
        n_traj=25,
        options=oqs.Options(
            seed=606,
            max_step=0.01,
            n_jobs=1,
            checkpoint_every=10,
            progress=True,
        ),
    )

    assert quiet.solver_stats["progress"] is False
    assert noisy.solver_stats["progress"] is True
    assert np.allclose(quiet.expect[0], noisy.expect[0])
    assert np.allclose(quiet.expect_std[0], noisy.expect_std[0])
    assert np.allclose(quiet.expect_stderr[0], noisy.expect_stderr[0])
