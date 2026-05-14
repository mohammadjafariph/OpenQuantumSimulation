from pathlib import Path

import numpy as np
import pytest

import openquantumsim as oqs


def test_result_hdf5_roundtrip(tmp_path: Path) -> None:
    times = np.linspace(0.0, 1.0, 4)
    states = [
        np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128),
        np.array([[0.8, 0.0], [0.0, 0.2]], dtype=np.complex128),
    ]
    result = oqs.Result(
        times=times,
        states=states,
        expect=[np.array([1.0, 0.8, 0.6, 0.4], dtype=np.complex128)],
        expect_std=[np.array([0.0, 0.1, 0.2, 0.3], dtype=np.float64)],
        expect_stderr=[np.array([0.0, 0.01, 0.02, 0.03], dtype=np.float64)],
        state_observables={
            "purity": np.array([1.0, 0.68, 0.52, 0.52], dtype=np.complex128),
        },
        state_observables_std={
            "purity": np.array([0.0, 0.05, 0.08, 0.07], dtype=np.float64),
        },
        state_observables_stderr={
            "purity": np.array([0.0, 0.005, 0.008, 0.007], dtype=np.float64),
        },
        entropy=np.array([0.0, 0.1, 0.2, 0.3], dtype=np.float64),
        solver_stats={
            "retcode": "Success",
            "n_traj": 500,
            "wall_time": 0.25,
            "threaded": True,
            "checkpoint": None,
            "notes": ["serial-reference", "threaded"],
        },
    )

    path = tmp_path / "result.h5"
    result.save_hdf5(path)

    loaded = oqs.load_result(path)

    assert np.allclose(loaded.times, result.times)
    assert loaded.states is not None
    assert len(loaded.states) == len(states)
    assert np.allclose(loaded.states[0], states[0])
    assert np.allclose(loaded.expect[0], result.expect[0])
    assert np.allclose(loaded.expect_std[0], result.expect_std[0])
    assert np.allclose(loaded.expect_stderr[0], result.expect_stderr[0])
    assert np.allclose(
        loaded.state_observables["purity"],
        result.state_observables["purity"],
    )
    assert np.allclose(
        loaded.state_observables_std["purity"],
        result.state_observables_std["purity"],
    )
    assert np.allclose(
        loaded.state_observables_stderr["purity"],
        result.state_observables_stderr["purity"],
    )
    assert loaded.entropy is not None
    assert result.entropy is not None
    assert np.allclose(loaded.entropy, result.entropy)
    assert loaded.solver_stats["retcode"] == "Success"
    assert loaded.solver_stats["n_traj"] == 500
    assert loaded.solver_stats["wall_time"] == 0.25
    assert loaded.solver_stats["threaded"] is True
    assert loaded.solver_stats["checkpoint"] is None
    assert loaded.solver_stats["notes"] == ["serial-reference", "threaded"]


def test_result_hdf5_roundtrip_without_optional_arrays(tmp_path: Path) -> None:
    result = oqs.Result(times=np.array([0.0, 1.0], dtype=np.float64))

    path = tmp_path / "empty_result.h5"
    result.save_hdf5(path)
    loaded = oqs.Result.load_hdf5(path)

    assert np.allclose(loaded.times, result.times)
    assert loaded.states is None
    assert loaded.expect == []
    assert loaded.expect_std == []
    assert loaded.expect_stderr == []
    assert loaded.state_observables == {}
    assert loaded.state_observables_std == {}
    assert loaded.state_observables_stderr == {}
    assert loaded.entropy is None
    assert loaded.solver_stats == {}


def test_result_hdf5_roundtrip_with_ket_states(tmp_path: Path) -> None:
    result = oqs.Result(
        times=np.array([0.0, 1.0], dtype=np.float64),
        states=[
            np.array([1.0, 0.0], dtype=np.complex128),
            np.array([0.0, 1.0], dtype=np.complex128),
        ],
    )

    path = tmp_path / "ket_result.h5"
    result.save_hdf5(path)
    loaded = oqs.load_result(path)

    assert loaded.states is not None
    assert len(loaded.states) == 2
    assert loaded.states[0].shape == (2,)
    assert np.allclose(loaded.states[1], [0.0, 1.0])


def test_result_can_evaluate_state_observables() -> None:
    result = oqs.Result(
        times=np.array([0.0, 1.0], dtype=np.float64),
        states=[
            np.array([1.0, 0.0], dtype=np.complex128),
            np.array([0.0, 1.0], dtype=np.complex128),
        ],
    )

    values = result.evaluate_state_observables(
        {"ground_population": lambda ket: abs(ket[1]) ** 2},
    )

    assert np.allclose(values["ground_population"], [0.0, 1.0])
    assert np.allclose(result.state_observables["ground_population"], [0.0, 1.0])


def test_result_observable_names_are_valid(tmp_path: Path) -> None:
    result = oqs.Result(
        times=np.array([0.0], dtype=np.float64),
        state_observables={
            "bad/name": np.array([0.0], dtype=np.complex128),
        },
    )

    with pytest.raises(ValueError, match="must not contain"):
        result.save_hdf5(tmp_path / "bad_result.h5")


def test_result_hdf5_refuses_overwrite(tmp_path: Path) -> None:
    result = oqs.Result(times=np.array([0.0], dtype=np.float64))
    path = tmp_path / "result.h5"

    result.save_hdf5(path)

    with pytest.raises(FileExistsError):
        result.save_hdf5(path, overwrite=False)
