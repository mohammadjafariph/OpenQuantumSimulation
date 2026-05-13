"""Run a restartable MCWF parameter sweep with structured outputs."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np

import openquantumsim as oqs


def main() -> None:
    args = _parse_args()
    output_dir = args.output_dir
    results_dir = output_dir / "results"
    checkpoints_dir = output_dir / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "manifest.json"
    points = _build_points(args, results_dir, checkpoints_dir)
    manifest = _load_or_create_manifest(manifest_path, args, points)
    _write_manifest(manifest_path, manifest)

    for point in manifest["points"]:
        result_path = Path(str(point["result_file"]))
        checkpoint_path = Path(str(point["checkpoint_file"]))
        if point["status"] == "done" and result_path.exists() and not args.force:
            print(f"Skipping done point {_point_description(point)}")
            continue

        if args.restart:
            result_path.unlink(missing_ok=True)
            checkpoint_path.unlink(missing_ok=True)

        print(f"Running point {_point_description(point)}")
        point["status"] = "running"
        point["started_at"] = time.time()
        point["error"] = None
        _write_manifest(manifest_path, manifest)

        try:
            result = _run_point(point, args)
            result.save_hdf5(result_path, overwrite=True)
            point["status"] = "done"
            point["finished_at"] = time.time()
            point["checkpoint_completed"] = result.solver_stats.get(
                "checkpoint_completed",
            )
            point["final_expect"] = float(result.expect[0].real[-1])
            point["final_stderr"] = float(result.expect_stderr[0][-1])
            _write_manifest(manifest_path, manifest)
        except Exception as exc:
            point["status"] = "failed"
            point["finished_at"] = time.time()
            point["error"] = repr(exc)
            _write_manifest(manifest_path, manifest)
            if not args.keep_going:
                raise

    _write_summary(output_dir, manifest["points"])
    print(f"Wrote manifest: {manifest_path}")
    print(f"Wrote summary: {output_dir / 'summary.csv'}")
    print(f"Wrote summary: {output_dir / 'summary.h5'}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a restartable OpenQuantumSim MCWF parameter sweep.",
    )
    parser.add_argument(
        "--system",
        choices=["qubit_decay"],
        default="qubit_decay",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("runs/sweep"))
    parser.add_argument(
        "--kappa-values",
        type=str,
        default="0.1",
        help="Comma-separated coupling/decay values, e.g. 0.02,0.05,0.1.",
    )
    parser.add_argument("--n-traj", type=int, default=2000)
    parser.add_argument("--time-points", type=int, default=41)
    parser.add_argument("--t-final", type=float, default=4.0)
    parser.add_argument("--max-step", type=float, default=0.01)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Delete existing point outputs/checkpoints before running.",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue to later points if one point fails.",
    )
    args = parser.parse_args()

    args.kappa_values = _parse_float_values(args.kappa_values, "kappa")
    if args.n_traj <= 0:
        parser.error("--n-traj must be positive")
    if args.time_points <= 1:
        parser.error("--time-points must be greater than 1")
    if args.t_final <= 0:
        parser.error("--t-final must be positive")
    if args.max_step <= 0:
        parser.error("--max-step must be positive")
    if args.checkpoint_every <= 0:
        parser.error("--checkpoint-every must be positive")
    if args.n_jobs != -1 and args.n_jobs <= 0:
        parser.error("--n-jobs must be -1 or positive")
    return args


def _parse_float_values(raw: str, name: str) -> list[float]:
    values = [value.strip() for value in raw.split(",") if value.strip()]
    if not values:
        msg = f"{name} values must not be empty"
        raise ValueError(msg)
    parsed = [float(value) for value in values]
    if any(value < 0 for value in parsed):
        msg = f"{name} values must be non-negative"
        raise ValueError(msg)
    return parsed


def _build_points(
    args: argparse.Namespace,
    results_dir: Path,
    checkpoints_dir: Path,
) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for kappa in args.kappa_values:
        index = len(points)
        label = _point_label(index, kappa)
        points.append(
            {
                "id": label,
                "index": index,
                "kappa": kappa,
                "seed": args.seed + index,
                "status": "pending",
                "result_file": str(results_dir / f"{label}.h5"),
                "checkpoint_file": str(checkpoints_dir / f"{label}_checkpoint.h5"),
                "error": None,
            },
        )
    return points


def _point_label(index: int, kappa: float) -> str:
    value = f"{kappa:.12g}".replace("-", "m").replace(".", "p")
    return f"point_{index:04d}_kappa_{value}"


def _load_or_create_manifest(
    manifest_path: Path,
    args: argparse.Namespace,
    points: list[dict[str, Any]],
) -> dict[str, Any]:
    config = {
        "system": args.system,
        "kappa_values": args.kappa_values,
        "n_traj": args.n_traj,
        "time_points": args.time_points,
        "t_final": args.t_final,
        "max_step": args.max_step,
        "checkpoint_every": args.checkpoint_every,
        "seed": args.seed,
        "n_jobs": args.n_jobs,
    }
    if not manifest_path.exists() or args.restart:
        return {
            "format": "openquantumsim.sweep",
            "format_version": 1,
            "created_at": time.time(),
            "updated_at": time.time(),
            "config": config,
            "points": points,
        }

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != "openquantumsim.sweep":
        msg = f"Not an OpenQuantumSim sweep manifest: {manifest_path}"
        raise ValueError(msg)
    if manifest.get("config") != config:
        msg = "Existing sweep manifest config does not match this run."
        raise ValueError(msg)
    return manifest


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    manifest["updated_at"] = time.time()
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def _point_description(point: dict[str, Any]) -> str:
    return f"{point['id']} kappa={point['kappa']}"


def _run_point(point: dict[str, Any], args: argparse.Namespace) -> oqs.Result:
    if args.system == "qubit_decay":
        return _run_qubit_decay_point(point, args)
    msg = f"Unknown sweep system: {args.system}"
    raise ValueError(msg)


def _run_qubit_decay_point(
    point: dict[str, Any],
    args: argparse.Namespace,
) -> oqs.Result:
    atom = oqs.SpinSpace(0.5, label="atom")
    hamiltonian = 0.0 * oqs.sigmaz(atom)
    collapse = np.sqrt(float(point["kappa"])) * oqs.sigmam(atom)
    excited = oqs.basis(atom, "up")
    excited_projector = oqs.Operator(oqs.ket2dm(excited), atom, "P_excited")
    times = np.linspace(0.0, args.t_final, args.time_points)
    return oqs.mcsolve(
        hamiltonian,
        excited,
        times,
        c_ops=[collapse],
        e_ops=[excited_projector],
        n_traj=args.n_traj,
        options=oqs.Options(
            seed=int(point["seed"]),
            max_step=args.max_step,
            n_jobs=args.n_jobs,
            checkpoint_file=str(point["checkpoint_file"]),
            checkpoint_every=args.checkpoint_every,
            progress=not args.no_progress,
        ),
    )


def _write_summary(output_dir: Path, points: list[dict[str, Any]]) -> None:
    rows = [_summary_row(point) for point in points if point["status"] == "done"]
    csv_path = output_dir / "summary.csv"
    h5_path = output_dir / "summary.h5"

    fieldnames = [
        "id",
        "kappa",
        "seed",
        "status",
        "final_expect",
        "final_stderr",
        "result_file",
        "checkpoint_file",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    kappa = np.array([row["kappa"] for row in rows], dtype=np.float64)
    final_expect = np.array([row["final_expect"] for row in rows], dtype=np.float64)
    final_stderr = np.array([row["final_stderr"] for row in rows], dtype=np.float64)
    with h5py.File(h5_path, "w") as handle:
        handle.attrs["format"] = "openquantumsim.sweep.summary"
        handle.attrs["format_version"] = "1"
        handle.create_dataset("kappa", data=kappa)
        handle.create_dataset("final_expect", data=final_expect)
        handle.create_dataset("final_stderr", data=final_stderr)


def _summary_row(point: dict[str, Any]) -> dict[str, Any]:
    result = oqs.load_result(str(point["result_file"]))
    return {
        "id": point["id"],
        "kappa": float(point["kappa"]),
        "seed": int(point["seed"]),
        "status": point["status"],
        "final_expect": float(result.expect[0].real[-1]),
        "final_stderr": float(result.expect_stderr[0][-1]),
        "result_file": point["result_file"],
        "checkpoint_file": point["checkpoint_file"],
    }


if __name__ == "__main__":
    main()
