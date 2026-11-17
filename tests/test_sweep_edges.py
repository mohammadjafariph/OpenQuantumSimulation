"""Restart, failure and summary behaviour of ParameterSweep."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py  # type: ignore[import-untyped]
import numpy as np
import pytest

import openquantumsim as oqs
from openquantumsim import sweep as sweep_module


def _result(value: complex) -> oqs.Result:
    return oqs.Result(
        times=np.array([0.0, 1.0]),
        expect=[np.array([0.0, value], dtype=np.complex128)],
        expect_std=[np.array([0.0, 0.2])],
        expect_stderr=[np.array([0.0, 0.02])],
        entropy=np.array([0.0, 0.5]),
        solver_stats={"retcode": "Success"},
    )


def test_grid_values_treat_strings_mappings_and_scalars_as_single_values() -> None:
    grid = sweep_module._expand_grid(
        {
            "label": "abc",
            "config": {"a": 1},
            "scalar": 2.5,
            "array": np.array([1, 2]),
        },
    )

    assert [point["array"] for point in grid] == [1, 2]
    assert all(point["label"] == "abc" for point in grid)
    assert all(point["config"] == {"a": 1} for point in grid)
    assert sweep_module._expand_grid({}) == [{}]
    with pytest.raises(ValueError, match="has no values"):
        sweep_module._expand_grid({"empty": []})


def test_point_labels_are_file_safe_and_bounded() -> None:
    label = sweep_module._point_label(3, {"kappa": -0.25, "name": "a b/c"})
    long_label = sweep_module._point_label(0, {"x" * 200: 1})

    assert label == "point_0003_kappa_m0p25_name_a_b_c"
    assert len(long_label) <= len("point_0000_") + 96
    assert sweep_module._point_label(7, {}) == "point_0007"


def test_rerun_with_a_different_grid_is_refused(tmp_path: Path) -> None:
    oqs.ParameterSweep({"model": "toy"}, {"k": [1]}).run(
        lambda _point: {"value": 1.0},
        output_dir=tmp_path,
    )

    with pytest.raises(ValueError, match="does not match"):
        oqs.ParameterSweep({"model": "toy"}, {"k": [2]}).run(
            lambda _point: {"value": 1.0},
            output_dir=tmp_path,
        )


def test_manifest_from_elsewhere_is_refused(tmp_path: Path) -> None:
    (tmp_path / "manifest.json").write_text('{"format": "other"}', encoding="utf-8")

    with pytest.raises(ValueError, match="Not an OpenQuantumSim sweep manifest"):
        oqs.ParameterSweep({}, {"k": [1]}).run(
            lambda _point: {"v": 1},
            output_dir=tmp_path,
        )


def test_restart_reruns_every_point_and_deletes_old_results(tmp_path: Path) -> None:
    calls: list[str] = []
    sweep = oqs.ParameterSweep({}, {"k": [1, 2]})

    def runner(point: oqs.SweepPoint) -> oqs.Result:
        calls.append(point.id)
        return _result(float(point.params["k"]))

    sweep.run(runner, output_dir=tmp_path)
    sweep.run(runner, output_dir=tmp_path)
    assert len(calls) == 2

    sweep.run(runner, output_dir=tmp_path, restart=True)
    assert len(calls) == 4

    sweep.run(runner, output_dir=tmp_path, force=True)
    assert len(calls) == 6


def test_failure_stops_the_sweep_unless_asked_to_keep_going(tmp_path: Path) -> None:
    def runner(point: oqs.SweepPoint) -> dict[str, float]:
        if point.params["k"] == 2:
            raise RuntimeError("diverged")
        return {"value": 1.0}

    with pytest.raises(RuntimeError, match="diverged"):
        oqs.ParameterSweep({}, {"k": [1, 2, 3]}).run(runner, output_dir=tmp_path)

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    statuses = [point["status"] for point in manifest["points"]]
    assert statuses == ["done", "failed", "pending"]
    assert "diverged" in manifest["points"][1]["error"]


def test_skipped_point_with_unreadable_result_keeps_its_stored_summary(
    tmp_path: Path,
) -> None:
    sweep = oqs.ParameterSweep({}, {"k": [1]})
    first = sweep.run(lambda _point: _result(1.0), output_dir=tmp_path)
    result_file = Path(str(first.points[0]["result_file"]))
    result_file.write_bytes(b"not an hdf5 file")

    second = sweep.run(lambda _point: _result(9.0), output_dir=tmp_path)

    assert second.outputs == [None]
    assert second.summary[0]["final_expect"] == 1.0


def test_skipped_point_without_stored_summary_is_summarised_from_its_file(
    tmp_path: Path,
) -> None:
    sweep = oqs.ParameterSweep({}, {"k": [1]})
    sweep.run(lambda _point: _result(1.0), output_dir=tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["points"][0]["summary"] = None
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    rerun = sweep.run(lambda _point: _result(9.0), output_dir=tmp_path)

    assert rerun.summary[0]["final_expect"] == 1.0
    assert rerun.summary[0]["final_entropy"] == 0.5


def test_summaries_of_complex_results_and_custom_outputs() -> None:
    complex_row = sweep_module._result_summary(_result(1.0 + 2.0j))
    assert complex_row["final_expect_0_real"] == 1.0
    assert complex_row["final_expect_0_imag"] == 2.0
    assert complex_row["final_std"] == 0.2
    assert complex_row["final_stderr"] == 0.02

    class Custom:
        pass

    run = oqs.ParameterSweep({}, {"k": [1]}).run(lambda _point: Custom())
    assert run.summary[0]["output_type"] == "Custom"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (np.float64(1.5), 1.5),
        (1 + 0j, 1.0),
        (1 + 2j, "(1+2j)"),
        (Path("a/b"), str(Path("a/b"))),
        (np.array([1, 2]), "[1, 2]"),
        ({"b": 1, "a": 2}, '{"a": 2, "b": 1}'),
        (object, str(object)),
    ],
)
def test_summary_values_are_flattened_to_scalars(value: Any, expected: Any) -> None:
    assert sweep_module._summary_value(value) == expected


def test_jsonable_handles_numpy_complex_and_unknown_values() -> None:
    converted = sweep_module._jsonable(
        {"z": 1 + 2j, "p": Path("x"), "a": np.array([1.0]), "t": (1, 2), "o": object},
    )

    assert converted["z"] == {"real": 1.0, "imag": 2.0}
    assert converted["p"] == "x"
    assert converted["a"] == [1.0]
    assert converted["t"] == [1, 2]
    assert converted["o"] == repr(object)


def test_summary_files_keep_column_types(tmp_path: Path) -> None:
    rows = [
        {"id": "a", "index": 0, "status": "done", "flag": True, "n": 1, "x": 0.5},
        {"id": "b", "index": 1, "status": "done", "flag": False, "n": 2, "x": None},
    ]

    sweep_module._write_summary_files(tmp_path, rows)

    with h5py.File(tmp_path / "summary.h5", "r") as handle:
        assert handle["flag"].dtype == np.bool_
        assert handle["n"].dtype == np.int64
        # A numeric value one point did not report is stored as NaN.
        assert handle["x"].dtype == np.float64
        assert handle["x"][0] == 0.5
        assert np.isnan(handle["x"][1])
    csv_text = (tmp_path / "summary.csv").read_text(encoding="utf-8")
    assert csv_text.splitlines()[0] == "id,index,status,flag,n,x"


def test_sweep_whose_points_report_different_keys_still_writes_summaries(
    tmp_path: Path,
) -> None:
    # This used to crash while writing summary.h5: a column with a number for
    # one point and text or nothing for another could not be stored.
    def runner(point: oqs.SweepPoint) -> dict[str, Any]:
        if point.params["k"] == 1:
            return {"energy": 1.5, "note": "converged"}
        return {"energy": "diverged"}

    run = oqs.ParameterSweep({}, {"k": [1, 2]}).run(runner, output_dir=tmp_path)

    assert run.summary_h5_path is not None
    with h5py.File(run.summary_h5_path, "r") as handle:
        energies = [value.decode() for value in handle["energy"][:]]
        notes = [value.decode() for value in handle["note"][:]]
    assert energies == ["1.5", "diverged"]
    assert notes == ["converged", ""]
