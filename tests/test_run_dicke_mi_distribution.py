from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import h5py
import numpy as np
import pytest


def _load_runner() -> ModuleType:
    path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "dicke"
        / "run_mi_distribution.py"
    )
    spec = importlib.util.spec_from_file_location(
        "examples.dicke.run_mi_distribution",
        path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parse_values_and_labels() -> None:
    runner = _load_runner()

    assert runner._parse_int_values("2, 4,6", "N") == [2, 4, 6]
    assert runner._parse_float_values("0.02, 0.1", "kappa") == [0.02, 0.1]
    assert runner._safe_float_label(0.1) == "0p1"
    assert runner._effective_batch_size(0, 10) == 10
    assert runner._trajectory_batches(1, 6, 2) == [(1, 3), (3, 5), (5, 6)]


def test_prepare_output_and_steady_datasets(tmp_path: Path) -> None:
    runner = _load_runner()
    output = tmp_path / "mi.h5"
    args = argparse.Namespace(
        n_values=[2],
        kappa_values=[0.1],
        n_traj=2,
        time_points=4,
        t_final=0.3,
        max_step=0.05,
        checkpoint_every=1,
        batch_size=0,
        n_jobs=1,
        steady_start_frac=0.5,
        seed=2026,
    )
    times = np.linspace(0.0, args.t_final, args.time_points)

    with h5py.File(output, "w") as handle:
        runner._prepare_output(handle, args, times)
        group = runner._require_point_group(handle, args, 2, 0.1, times)
        group["MI_time_A"][:, :] = np.array(
            [[0.0, 1.0, 2.0, 4.0], [0.0, 2.0, 4.0, 8.0]],
            dtype=np.float64,
        )
        group["MI_time_B"][:, :] = np.array(
            [[0.0, 3.0, 6.0, 9.0], [0.0, 4.0, 8.0, 12.0]],
            dtype=np.float64,
        )
        group.attrs["completed"] = 2
        runner._write_steady_datasets(group, 2)

    with h5py.File(output, "r") as handle:
        assert handle.attrs["format"] == "openquantumsim.dicke_mi_distribution"
        group = handle["kappa_0p1"]["N_2"]
        assert np.allclose(group["MI_steady_A"][:], [3.0, 6.0])
        assert np.allclose(group["MI_steady_B"][:], [7.5, 10.0])
        assert group.attrs["completed"] == 2


def test_run_point_writes_serial_batches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()

    def fake_run_batch(request: Any) -> Any:
        rows = [
            np.full(len(request.times), float(traj_idx), dtype=np.float64)
            for traj_idx in range(request.start, request.stop)
        ]
        mi_a = np.vstack(rows)
        return runner.BatchResult(
            start=request.start,
            stop=request.stop,
            mi_a=mi_a,
            mi_b=mi_a + 10.0,
        )

    monkeypatch.setattr(runner, "_run_trajectory_batch", fake_run_batch)

    args = argparse.Namespace(
        n_values=[2],
        kappa_values=[0.1],
        n_traj=3,
        time_points=4,
        t_final=0.3,
        max_step=0.05,
        checkpoint_every=2,
        batch_size=2,
        n_jobs=1,
        steady_start_frac=0.25,
        seed=2026,
        no_progress=True,
    )
    times = np.linspace(0.0, args.t_final, args.time_points)

    with h5py.File(tmp_path / "mi.h5", "w") as handle:
        runner._prepare_output(handle, args, times)
        group = runner._require_point_group(handle, args, 2, 0.1, times)
        runner._run_point(group, args, 2, 0.1, times, 0)

        assert group.attrs["completed"] == 3
        assert np.allclose(group["MI_time_A"][:, 0], [0.0, 1.0, 2.0])
        assert np.allclose(group["MI_time_B"][:, 0], [10.0, 11.0, 12.0])
        assert np.allclose(group["MI_steady_A"][:], [0.0, 1.0, 2.0])
