from pathlib import Path

import h5py
import numpy as np
import pytest

import openquantumsim as oqs


def _toy_result(value: float) -> oqs.Result:
    return oqs.Result(
        times=np.array([0.0, 1.0], dtype=np.float64),
        expect=[np.array([0.0, value], dtype=np.complex128)],
        expect_stderr=[np.array([0.0, 0.01 * value], dtype=np.float64)],
        solver_stats={"retcode": "Success"},
    )


def test_parameter_sweep_runs_grid_and_writes_outputs(tmp_path: Path) -> None:
    calls: list[str] = []
    sweep = oqs.ParameterSweep(
        base_system={"model": "toy"},
        params={"kappa": [0.1, 0.2], "drive": [0.0, 1.0]},
    )

    def runner(point: oqs.SweepPoint) -> oqs.Result:
        calls.append(point.id)
        assert point.context["model"] == "toy"
        value = float(point.params["kappa"]) + float(point.params["drive"])
        return _toy_result(value)

    result = sweep.run(runner, output_dir=tmp_path)

    assert len(calls) == 4
    assert len(result.points) == 4
    assert len(result.summary) == 4
    assert result.manifest_path == tmp_path / "manifest.json"
    assert result.summary_csv_path == tmp_path / "summary.csv"
    assert result.summary_h5_path == tmp_path / "summary.h5"
    assert all(point["status"] == "done" for point in result.points)
    assert all(Path(str(point["result_file"])).exists() for point in result.points)
    assert result.summary[0]["final_expect"] == pytest.approx(0.1)
    assert result.summary[-1]["final_expect"] == pytest.approx(1.2)

    csv_text = (tmp_path / "summary.csv").read_text(encoding="utf-8")
    assert "final_expect" in csv_text
    assert "point_0000_kappa_0p1_drive_0" in csv_text
    with h5py.File(tmp_path / "summary.h5", "r") as handle:
        assert handle.attrs["format"] == "openquantumsim.sweep.summary"
        assert np.allclose(handle["kappa"][:], [0.1, 0.1, 0.2, 0.2])
        assert np.allclose(handle["final_expect"][:], [0.1, 1.1, 0.2, 1.2])


def test_parameter_sweep_skips_completed_points(tmp_path: Path) -> None:
    sweep = oqs.ParameterSweep({}, {"gamma": [0.1, 0.2]})
    first = sweep.run(
        lambda point: _toy_result(float(point.params["gamma"])),
        output_dir=tmp_path,
    )

    def should_not_run(_point: oqs.SweepPoint) -> oqs.Result:
        msg = "completed points should have been skipped"
        raise AssertionError(msg)

    second = sweep.run(should_not_run, output_dir=tmp_path)

    assert len(first.summary) == len(second.summary) == 2
    assert [row["final_expect"] for row in second.summary] == [0.1, 0.2]


def test_parameter_sweep_keep_going_records_failures(tmp_path: Path) -> None:
    sweep = oqs.ParameterSweep({}, {"gamma": [0.1, 0.2]})

    def runner(point: oqs.SweepPoint) -> dict[str, float]:
        gamma = float(point.params["gamma"])
        if gamma > 0.1:
            msg = "intentional failure"
            raise RuntimeError(msg)
        return {"final_value": gamma}

    result = sweep.run(runner, output_dir=tmp_path, keep_going=True)

    assert [point["status"] for point in result.points] == ["done", "failed"]
    assert len(result.summary) == 1
    assert result.summary[0]["final_value"] == pytest.approx(0.1)
    assert "RuntimeError" in str(result.points[1]["error"])


def test_parameter_sweep_custom_summary_without_files() -> None:
    sweep = oqs.ParameterSweep({"scale": 2.0}, {"x": np.array([1.0, 3.0])})

    def runner(point: oqs.SweepPoint) -> float:
        return float(point.context["scale"]) * float(point.params["x"])

    result = sweep.run(
        runner,
        summarize=lambda point, value: {
            "scaled": float(value),
            "label": f"x={point.params['x']}",
        },
    )

    assert result.output_dir is None
    assert [row["scaled"] for row in result.summary] == [2.0, 6.0]
    assert [row["label"] for row in result.summary] == ["x=1.0", "x=3.0"]
