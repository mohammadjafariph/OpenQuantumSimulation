from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import h5py
import numpy as np

import openquantumsim as oqs


def _load_run_sweep() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_sweep.py"
    spec = importlib.util.spec_from_file_location("run_sweep", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parse_float_values() -> None:
    run_sweep = _load_run_sweep()

    assert run_sweep._parse_float_values("0.02, 0.05,0.1", "kappa") == [
        0.02,
        0.05,
        0.1,
    ]


def test_sweep_summary_from_result_files(tmp_path: Path) -> None:
    run_sweep = _load_run_sweep()
    result_path = tmp_path / "results" / "point_0000_kappa_0p1.h5"
    checkpoint_path = tmp_path / "checkpoints" / "point_0000_kappa_0p1_checkpoint.h5"
    result = oqs.Result(
        times=np.array([0.0, 1.0], dtype=np.float64),
        expect=[np.array([1.0, 0.8], dtype=np.complex128)],
        expect_std=[np.array([0.0, 0.2], dtype=np.float64)],
        expect_stderr=[np.array([0.0, 0.02], dtype=np.float64)],
        solver_stats={"retcode": "Success"},
    )
    result.save_hdf5(result_path)
    points = [
        {
            "id": "point_0000_kappa_0p1",
            "kappa": 0.1,
            "seed": 2026,
            "status": "done",
            "result_file": str(result_path),
            "checkpoint_file": str(checkpoint_path),
        },
    ]

    run_sweep._write_summary(tmp_path, points)

    csv_text = (tmp_path / "summary.csv").read_text(encoding="utf-8")
    assert "point_0000_kappa_0p1" in csv_text
    assert "0.8" in csv_text
    with h5py.File(tmp_path / "summary.h5", "r") as handle:
        assert handle.attrs["format"] == "openquantumsim.sweep.summary"
        assert np.allclose(handle["kappa"][:], [0.1])
        assert np.allclose(handle["final_expect"][:], [0.8])
        assert np.allclose(handle["final_stderr"][:], [0.02])
