"""Input checks of Options and of Result persistence."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py  # type: ignore[import-untyped]
import numpy as np
import pytest

import openquantumsim as oqs


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_step": 0.0},
        {"n_traj": 0},
        {"n_jobs": 0},
        {"n_jobs": -2},
        {"checkpoint_every": 0},
    ],
)
def test_options_reject_invalid_monte_carlo_settings(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        oqs.Options(**kwargs)


def test_options_accept_every_valid_combination() -> None:
    oqs.Options(method="tsit5", ode_solver="tsit5")
    oqs.Options(method="ode", ode_solver="vern9", seed=0, max_step=0.1, n_jobs=1)


def _result(**overrides: Any) -> oqs.Result:
    settings: dict[str, Any] = {"times": np.array([0.0, 1.0])}
    settings.update(overrides)
    return oqs.Result(**settings)


def test_result_needs_saved_states_to_evaluate_observables() -> None:
    with pytest.raises(ValueError, match="no saved states"):
        _result().evaluate_state_observables({"purity": oqs.purity})


def test_result_can_evaluate_without_storing() -> None:
    result = _result(states=[np.eye(2) / 2, np.eye(2) / 2])

    values = result.evaluate_state_observables({"purity": oqs.purity}, inplace=False)

    np.testing.assert_allclose(values["purity"], [0.5, 0.5])
    assert result.state_observables == {}


@pytest.mark.parametrize(
    ("overrides", "error", "message"),
    [
        ({"entropy": np.zeros(3)}, ValueError, "entropy must have the same shape"),
        ({"expect": [np.zeros(3)]}, ValueError, "match times"),
        ({"state_observables": {"p": np.zeros(3)}}, ValueError, "match times"),
        ({"state_observables": {1: np.zeros(2)}}, TypeError, "must be strings"),
        ({"state_observables": {"": np.zeros(2)}}, ValueError, "must not be empty"),
        ({"states": [np.zeros(2), np.zeros(3)]}, ValueError, "same shape"),
        ({"states": [np.zeros((2, 2, 2))] * 2}, ValueError, "ket vectors or density"),
    ],
)
def test_save_hdf5_rejects_inconsistent_results(
    tmp_path: Path,
    overrides: dict[str, Any],
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        _result(**overrides).save_hdf5(tmp_path / "result.h5")


def test_save_hdf5_accepts_an_empty_state_list(tmp_path: Path) -> None:
    path = tmp_path / "result.h5"
    _result(states=[]).save_hdf5(path)

    with h5py.File(path, "r") as handle:
        assert handle["states"].shape == (0, 0)


def test_load_hdf5_refuses_files_from_elsewhere(tmp_path: Path) -> None:
    path = tmp_path / "other.h5"
    with h5py.File(path, "w") as handle:
        handle.attrs["format"] = "something.else"

    with pytest.raises(ValueError, match="not an OpenQuantumSim result"):
        oqs.load_result(path)


def test_solver_stats_round_trip_numpy_scalars_and_bytes(tmp_path: Path) -> None:
    path = tmp_path / "result.h5"
    _result(
        solver_stats={
            "nsteps": np.int64(12),
            "wall_time": np.float64(0.5),
            "retcode": "Success",
            "notes": {"seed": None},
        },
    ).save_hdf5(path)
    with h5py.File(path, "a") as handle:
        # Older writers stored strings as bytes and had no encoding marker.
        handle["solver_stats"].attrs["legacy"] = np.bytes_("yes")

    stats = oqs.load_result(path).solver_stats

    assert stats["nsteps"] == 12
    assert isinstance(stats["nsteps"], int)
    assert stats["wall_time"] == 0.5
    assert stats["notes"] == {"seed": None}
    assert stats["legacy"] == "yes"
