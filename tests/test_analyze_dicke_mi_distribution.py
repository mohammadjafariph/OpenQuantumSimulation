from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import h5py
import numpy as np


def _load_analyzer() -> ModuleType:
    path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "dicke"
        / "analyze_mi_distribution.py"
    )
    spec = importlib.util.spec_from_file_location(
        "examples.dicke.analyze_mi_distribution",
        path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_input(path: Path) -> None:
    with h5py.File(path, "w") as handle:
        handle.attrs["format"] = "openquantumsim.dicke_mi_distribution"
        handle.attrs["format_version"] = "1"
        handle.create_dataset("N_values", data=np.array([2], dtype=np.int64))
        handle.create_dataset("kappa_values", data=np.array([0.1], dtype=np.float64))
        handle.create_dataset("t_axis", data=np.linspace(0.0, 0.3, 4))

        group = handle.create_group("kappa_0p1").create_group("N_2")
        group.attrs["N"] = 2
        group.attrs["kappa"] = 0.1
        group.attrs["completed"] = 3
        group.attrs["n_transient"] = 2
        group.create_dataset(
            "MI_steady_A",
            data=np.array([1.0, 2.0, 3.0], dtype=np.float64),
        )
        group.create_dataset(
            "MI_steady_B",
            data=np.array([2.0, 4.0, 6.0], dtype=np.float64),
        )
        group.create_dataset("MI_time_A", data=np.zeros((3, 4), dtype=np.float64))
        group.create_dataset("MI_time_B", data=np.zeros((3, 4), dtype=np.float64))


def test_analyze_writes_summary_files(tmp_path: Path) -> None:
    analyzer = _load_analyzer()
    input_path = tmp_path / "dicke_mi.h5"
    output_dir = tmp_path / "analysis"
    _write_input(input_path)

    rows = analyzer.analyze(input_path, output_dir, make_plots=False)

    assert len(rows) == 1
    row = rows[0]
    assert row.N == 2
    assert row.kappa == 0.1
    assert row.completed == 3
    assert np.isclose(row.MI_mean_A, 2.0)
    assert np.isclose(row.MI_mean_B, 4.0)
    assert np.isclose(row.MI_mean_AB, 3.0)

    csv_text = (output_dir / "summary.csv").read_text(encoding="utf-8")
    assert "MI_mean_AB" in csv_text
    assert "3.0" in csv_text

    with h5py.File(output_dir / "summary.h5", "r") as handle:
        assert handle.attrs["format"] == "openquantumsim.dicke_mi_analysis"
        assert np.allclose(handle["N"][:], [2])
        assert np.allclose(handle["MI_mean_AB"][:], [3.0])


def test_summary_falls_back_to_time_series(tmp_path: Path) -> None:
    analyzer = _load_analyzer()
    input_path = tmp_path / "dicke_mi_missing_steady.h5"
    _write_input(input_path)
    with h5py.File(input_path, "a") as handle:
        group = handle["kappa_0p1"]["N_2"]
        del group["MI_steady_A"]
        del group["MI_steady_B"]
        group["MI_time_A"][:, :] = np.array(
            [[0.0, 1.0, 2.0, 4.0], [0.0, 2.0, 4.0, 8.0], [0.0, 3.0, 6.0, 12.0]],
            dtype=np.float64,
        )
        group["MI_time_B"][:, :] = 2.0 * group["MI_time_A"][:, :]

    rows = analyzer.analyze(input_path, tmp_path / "analysis", make_plots=False)

    assert len(rows) == 1
    assert np.isclose(rows[0].MI_mean_A, 6.0)
    assert np.isclose(rows[0].MI_mean_B, 12.0)
