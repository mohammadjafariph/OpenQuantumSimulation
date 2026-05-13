"""Parameter sweep utilities."""

from __future__ import annotations

import csv
import json
import re
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Any, TypeAlias, cast

import h5py  # type: ignore[import-untyped]
import numpy as np

from .result import Result, load_result

Runner: TypeAlias = Callable[["SweepPoint"], object]
Summarizer: TypeAlias = Callable[["SweepPoint", object], Mapping[str, Any]]


@dataclass(frozen=True)
class SweepPoint:
    """One concrete parameter point in a sweep."""

    index: int
    id: str
    params: dict[str, Any]
    base_system: Mapping[str, Any]
    result_file: Path | None = None
    output_dir: Path | None = None

    @property
    def context(self) -> dict[str, Any]:
        """Return ``base_system`` merged with this point's parameters."""
        values = dict(self.base_system)
        values.update(self.params)
        return values


@dataclass(frozen=True)
class SweepRunResult:
    """Result metadata returned by :meth:`ParameterSweep.run`."""

    points: list[dict[str, Any]]
    summary: list[dict[str, Any]]
    outputs: list[object | None] = field(default_factory=list)
    output_dir: Path | None = None
    manifest_path: Path | None = None
    summary_csv_path: Path | None = None
    summary_h5_path: Path | None = None


@dataclass(frozen=True)
class ParameterSweep:
    """Specification for a parameter-grid simulation sweep.

    ``base_system`` stores fixed configuration shared by all points. ``params``
    maps each swept parameter name to a scalar or sequence of values. The run
    callback receives a :class:`SweepPoint` and may return a solver
    :class:`~openquantumsim.result.Result`, a scalar mapping, or any custom
    object paired with a custom ``summarize`` callback.
    """

    base_system: Mapping[str, Any]
    params: Mapping[str, Any]

    def run(
        self,
        runner: Runner,
        *,
        output_dir: str | Path | None = None,
        summarize: Summarizer | None = None,
        force: bool = False,
        restart: bool = False,
        keep_going: bool = False,
    ) -> SweepRunResult:
        """Run the parameter grid.

        When ``output_dir`` is provided, the sweep writes a restartable
        ``manifest.json``, one HDF5 result per point when the callback returns
        a :class:`Result`, plus ``summary.csv`` and ``summary.h5``.
        Completed points are skipped on later calls unless ``force`` or
        ``restart`` is set.
        """
        output_path = Path(output_dir) if output_dir is not None else None
        records = _build_records(_expand_grid(self.params), output_path)
        config = {
            "base_system": _jsonable(dict(self.base_system)),
            "params": _jsonable(dict(self.params)),
        }

        manifest_path: Path | None = None
        created_at = time.time()
        if output_path is not None:
            output_path.mkdir(parents=True, exist_ok=True)
            (output_path / "results").mkdir(parents=True, exist_ok=True)
            manifest_path = output_path / "manifest.json"
            stored = None if restart else _load_manifest(manifest_path)
            if stored is not None:
                if stored.get("config") != config:
                    msg = "Existing sweep manifest config does not match this run."
                    raise ValueError(msg)
                created_at = float(stored.get("created_at", created_at))
                _merge_stored_records(records, stored)
            if restart:
                _delete_point_outputs(records)
            _write_manifest(manifest_path, config, records, created_at)

        summary_rows: list[dict[str, Any]] = []
        outputs: list[object | None] = []

        for record in records:
            if _should_skip(record, force):
                skipped_output = _load_saved_output(record)
                outputs.append(skipped_output)
                row = _stored_or_loaded_summary(record, skipped_output)
                if row is not None:
                    summary_rows.append(row)
                continue

            point = _point_from_record(record, self.base_system, output_path)
            record["status"] = "running"
            record["started_at"] = time.time()
            record["finished_at"] = None
            record["error"] = None
            _write_manifest_if_enabled(manifest_path, config, records, created_at)

            try:
                run_output = runner(point)
                _save_output(record, run_output)
                row = _summarize_output(point, run_output, summarize)
                record["status"] = "done"
                record["finished_at"] = time.time()
                record["summary"] = _jsonable(row)
                summary_rows.append(row)
                outputs.append(run_output)
                _write_manifest_if_enabled(manifest_path, config, records, created_at)
            except Exception as exc:
                record["status"] = "failed"
                record["finished_at"] = time.time()
                record["error"] = repr(exc)
                outputs.append(None)
                _write_manifest_if_enabled(manifest_path, config, records, created_at)
                if not keep_going:
                    raise

        summary_csv_path = output_path / "summary.csv" if output_path else None
        summary_h5_path = output_path / "summary.h5" if output_path else None
        if output_path is not None:
            _write_summary_files(output_path, summary_rows)

        return SweepRunResult(
            points=[_public_record(record) for record in records],
            summary=summary_rows,
            outputs=outputs,
            output_dir=output_path,
            manifest_path=manifest_path,
            summary_csv_path=summary_csv_path,
            summary_h5_path=summary_h5_path,
        )


def _expand_grid(params: Mapping[str, Any]) -> list[dict[str, Any]]:
    names = list(params)
    values = [_grid_values(params[name], name) for name in names]
    if not names:
        return [{}]
    return [dict(zip(names, combo, strict=True)) for combo in product(*values)]


def _grid_values(value: Any, name: str) -> list[Any]:
    if isinstance(value, str | bytes) or isinstance(value, Mapping):
        values = [value]
    elif isinstance(value, np.ndarray):
        values = cast(list[Any], value.tolist())
    elif isinstance(value, Iterable):
        values = list(value)
    else:
        values = [value]
    if not values:
        msg = f"sweep parameter {name!r} has no values."
        raise ValueError(msg)
    return values


def _build_records(
    grid: list[dict[str, Any]],
    output_dir: Path | None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, params in enumerate(grid):
        point_id = _point_label(index, params)
        result_file = None
        if output_dir is not None:
            result_file = str(output_dir / "results" / f"{point_id}.h5")
        records.append(
            {
                "id": point_id,
                "index": index,
                "params": params,
                "status": "pending",
                "result_file": result_file,
                "started_at": None,
                "finished_at": None,
                "error": None,
                "summary": None,
            },
        )
    return records


def _point_label(index: int, params: Mapping[str, Any]) -> str:
    parts = [_slug(f"{name}_{_format_value(value)}") for name, value in params.items()]
    suffix = "_".join(part for part in parts if part)
    if len(suffix) > 96:
        suffix = suffix[:96].rstrip("_")
    return f"point_{index:04d}" if not suffix else f"point_{index:04d}_{suffix}"


def _format_value(value: Any) -> str:
    native = _native_scalar(value)
    if isinstance(native, float):
        return f"{native:.12g}"
    return str(native)


def _slug(value: str) -> str:
    text = value.replace("-", "m").replace(".", "p")
    text = re.sub(r"[^0-9A-Za-z_]+", "_", text)
    return text.strip("_")


def _load_manifest(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("format") != "openquantumsim.sweep":
        msg = f"Not an OpenQuantumSim sweep manifest: {path}"
        raise ValueError(msg)
    return cast(dict[str, Any], manifest)


def _merge_stored_records(
    records: list[dict[str, Any]],
    stored: Mapping[str, Any],
) -> None:
    stored_points = stored.get("points", [])
    if not isinstance(stored_points, list):
        return
    stored_by_id = {
        str(point.get("id")): point
        for point in stored_points
        if isinstance(point, Mapping)
    }
    for record in records:
        stored_record = stored_by_id.get(str(record["id"]))
        if stored_record is None:
            continue
        for key in ("status", "started_at", "finished_at", "error", "summary"):
            record[key] = stored_record.get(key)


def _delete_point_outputs(records: list[dict[str, Any]]) -> None:
    for record in records:
        result_file = record.get("result_file")
        if result_file:
            Path(str(result_file)).unlink(missing_ok=True)


def _write_manifest_if_enabled(
    path: Path | None,
    config: Mapping[str, Any],
    records: list[dict[str, Any]],
    created_at: float,
) -> None:
    if path is not None:
        _write_manifest(path, config, records, created_at)


def _write_manifest(
    path: Path,
    config: Mapping[str, Any],
    records: list[dict[str, Any]],
    created_at: float,
) -> None:
    manifest = {
        "format": "openquantumsim.sweep",
        "format_version": 1,
        "created_at": created_at,
        "updated_at": time.time(),
        "config": _jsonable(dict(config)),
        "points": [_manifest_record(record) for record in records],
    }
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def _manifest_record(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "id": record["id"],
        "index": record["index"],
        "params": _jsonable(record["params"]),
        "status": record["status"],
        "result_file": record["result_file"],
        "started_at": record["started_at"],
        "finished_at": record["finished_at"],
        "error": record["error"],
        "summary": _jsonable(record["summary"]),
    }


def _public_record(record: Mapping[str, Any]) -> dict[str, Any]:
    public = _manifest_record(record)
    public["params"] = dict(cast(Mapping[str, Any], record["params"]))
    return public


def _should_skip(record: Mapping[str, Any], force: bool) -> bool:
    if force or record.get("status") != "done":
        return False
    result_file = record.get("result_file")
    return bool(result_file and Path(str(result_file)).exists())


def _load_saved_output(record: Mapping[str, Any]) -> Result | None:
    result_file = record.get("result_file")
    if not result_file:
        return None
    try:
        return load_result(str(result_file))
    except Exception:
        return None


def _stored_or_loaded_summary(
    record: Mapping[str, Any],
    output: object | None,
) -> dict[str, Any] | None:
    stored = record.get("summary")
    if isinstance(stored, Mapping):
        return dict(stored)
    if output is None:
        return None
    point = _point_from_record(record, {}, None)
    return _summarize_output(point, output, None)


def _point_from_record(
    record: Mapping[str, Any],
    base_system: Mapping[str, Any],
    output_dir: Path | None,
) -> SweepPoint:
    result_file_raw = record.get("result_file")
    result_file = Path(str(result_file_raw)) if result_file_raw else None
    return SweepPoint(
        index=int(record["index"]),
        id=str(record["id"]),
        params=dict(cast(Mapping[str, Any], record["params"])),
        base_system=base_system,
        result_file=result_file,
        output_dir=output_dir,
    )


def _save_output(record: Mapping[str, Any], output: object) -> None:
    result_file = record.get("result_file")
    if isinstance(output, Result) and result_file:
        output.save_hdf5(str(result_file), overwrite=True)


def _summarize_output(
    point: SweepPoint,
    output: object,
    summarize: Summarizer | None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "id": point.id,
        "index": point.index,
        "status": "done",
        **point.params,
    }
    if point.result_file is not None:
        row["result_file"] = str(point.result_file)

    if summarize is not None:
        extra = dict(summarize(point, output))
    elif isinstance(output, Result):
        extra = _result_summary(output)
    elif isinstance(output, Mapping):
        extra = {
            str(key): _summary_value(value)
            for key, value in output.items()
            if _is_summary_compatible(value)
        }
    else:
        extra = {"output_type": type(output).__name__}

    row.update(extra)
    return {key: _summary_value(value) for key, value in row.items()}


def _result_summary(result: Result) -> dict[str, Any]:
    row: dict[str, Any] = {"n_times": int(len(result.times))}
    if len(result.times) > 0:
        row["t_final"] = float(result.times[-1])
    retcode = result.solver_stats.get("retcode")
    if retcode is not None:
        row["retcode"] = retcode

    for idx, expect_values in enumerate(result.expect):
        if len(expect_values) == 0:
            continue
        _add_complex_value(row, f"final_expect_{idx}", complex(expect_values[-1]))
        if idx == 0 and "final_expect_0" in row:
            row["final_expect"] = row["final_expect_0"]
    for idx, std_values in enumerate(result.expect_std):
        if len(std_values) > 0:
            row[f"final_std_{idx}"] = float(std_values[-1])
            if idx == 0:
                row["final_std"] = float(std_values[-1])
    for idx, stderr_values in enumerate(result.expect_stderr):
        if len(stderr_values) > 0:
            row[f"final_stderr_{idx}"] = float(stderr_values[-1])
            if idx == 0:
                row["final_stderr"] = float(stderr_values[-1])
    if result.entropy is not None and len(result.entropy) > 0:
        row["final_entropy"] = float(result.entropy[-1])
    return row


def _add_complex_value(row: dict[str, Any], name: str, value: complex) -> None:
    if abs(value.imag) < 1e-14:
        row[name] = float(value.real)
    else:
        row[f"{name}_real"] = float(value.real)
        row[f"{name}_imag"] = float(value.imag)


def _summary_value(value: Any) -> Any:
    native = _native_scalar(value)
    if native is None or isinstance(native, str | bool | int | float):
        return native
    if isinstance(native, complex):
        if abs(native.imag) < 1e-14:
            return float(native.real)
        return str(native)
    if isinstance(native, Path):
        return str(native)
    if isinstance(native, np.ndarray):
        return json.dumps(_jsonable(native.tolist()), sort_keys=True)
    if isinstance(native, Mapping | list | tuple):
        return json.dumps(_jsonable(native), sort_keys=True)
    return str(native)


def _is_summary_compatible(value: Any) -> bool:
    native = _native_scalar(value)
    return isinstance(
        native,
        str | bool | int | float | complex | Path | Mapping | list | tuple | np.ndarray,
    ) or native is None


def _write_summary_files(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    _write_summary_csv(output_dir / "summary.csv", rows)
    _write_summary_hdf5(output_dir / "summary.h5", rows)


def _write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = _fieldnames(rows)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})


def _write_summary_hdf5(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = _fieldnames(rows)
    with h5py.File(path, "w") as handle:
        handle.attrs["format"] = "openquantumsim.sweep.summary"
        handle.attrs["format_version"] = "1"
        for key in fieldnames:
            values = [row.get(key) for row in rows]
            _write_hdf5_column(handle, key, values)


def _fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    preferred = ["id", "index", "status"]
    seen = set(preferred)
    fieldnames = preferred[:]
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    return fieldnames


def _write_hdf5_column(group: h5py.Group, key: str, values: list[Any]) -> None:
    if all(isinstance(value, bool | np.bool_) for value in values):
        group.create_dataset(key, data=np.asarray(values, dtype=np.bool_))
    elif all(_is_int_value(value) for value in values):
        group.create_dataset(key, data=np.asarray(values, dtype=np.int64))
    elif all(_is_float_value(value) for value in values):
        group.create_dataset(key, data=np.asarray(values, dtype=np.float64))
    else:
        dtype = h5py.string_dtype(encoding="utf-8")
        group.create_dataset(
            key,
            data=np.asarray([_csv_value(value) for value in values], dtype=dtype),
        )


def _is_int_value(value: Any) -> bool:
    return isinstance(value, int | np.integer) and not isinstance(
        value,
        bool | np.bool_,
    )


def _is_float_value(value: Any) -> bool:
    return isinstance(value, int | float | np.integer | np.floating) and not isinstance(
        value,
        bool | np.bool_,
    )


def _csv_value(value: Any) -> str | int | float | bool:
    native = _native_scalar(value)
    if native is None:
        return ""
    if isinstance(native, str | bool | int | float):
        return native
    return str(_summary_value(native))


def _jsonable(value: Any) -> Any:
    native = _native_scalar(value)
    if native is None or isinstance(native, str | bool | int | float):
        return native
    if isinstance(native, complex):
        return {"real": native.real, "imag": native.imag}
    if isinstance(native, Path):
        return str(native)
    if isinstance(native, np.ndarray):
        return _jsonable(native.tolist())
    if isinstance(native, Mapping):
        return {str(key): _jsonable(item) for key, item in native.items()}
    if isinstance(native, tuple | list):
        return [_jsonable(item) for item in native]
    return repr(native)


def _native_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value
