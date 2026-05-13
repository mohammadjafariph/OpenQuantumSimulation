"""Run restartable two-ensemble Dicke mutual-information trajectories."""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import openquantumsim as oqs  # noqa: E402
from examples.dicke.observables import trajectory_dicke_mutual_information  # noqa: E402
from examples.dicke.system import two_ensemble_dicke_system  # noqa: E402

FloatArray = np.ndarray[Any, np.dtype[np.float64]]


@dataclass(frozen=True)
class BatchRequest:
    """Serializable trajectory-batch request for worker processes."""

    n_spins: int
    kappa: float
    times: FloatArray
    max_step: float
    seed: int
    point_index: int
    start: int
    stop: int


@dataclass(frozen=True)
class BatchResult:
    """Mutual-information arrays returned by one trajectory batch."""

    start: int
    stop: int
    mi_a: FloatArray
    mi_b: FloatArray


def main() -> None:
    args = _parse_args()
    output = args.output

    if output.exists() and (args.force or args.restart):
        output.unlink()
    output.parent.mkdir(parents=True, exist_ok=True)

    times = np.linspace(0.0, args.t_final, args.time_points)
    with h5py.File(output, "a") as handle:
        _prepare_output(handle, args, times)
        point_index = 0
        for kappa in args.kappa_values:
            for n_spins in args.n_values:
                group = _require_point_group(handle, args, n_spins, kappa, times)
                completed = int(group.attrs["completed"])
                if completed >= args.n_traj:
                    _write_steady_datasets(group, int(group.attrs["n_transient"]))
                    if not args.no_progress:
                        print(f"Skipping complete point N={n_spins} kappa={kappa}")
                    point_index += 1
                    continue

                if not args.no_progress:
                    print(
                        f"Running point N={n_spins} kappa={kappa} "
                        f"from trajectory {completed + 1}/{args.n_traj}",
                    )
                _run_point(group, args, n_spins, kappa, times, point_index)
                point_index += 1

    print(f"Saved Dicke MI distribution: {output}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run restartable Dicke mutual-information trajectory batches.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/dicke_mi_distribution.h5"),
    )
    parser.add_argument(
        "--n-values",
        type=str,
        default="2",
        help="Comma-separated Dicke ensemble sizes, e.g. 6,8,12.",
    )
    parser.add_argument(
        "--kappa-values",
        type=str,
        default="0.1",
        help="Comma-separated collective decay couplings, e.g. 0.02,0.05,0.1.",
    )
    parser.add_argument("--n-traj", type=int, default=100)
    parser.add_argument("--time-points", type=int, default=201)
    parser.add_argument("--t-final", type=float, default=20.0)
    parser.add_argument("--max-step", type=float, default=0.01)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="Trajectories per worker task. Defaults to --checkpoint-every.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Worker processes for trajectory batches. Use -1 for all CPUs.",
    )
    parser.add_argument("--steady-start-frac", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the output file before running.",
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Alias for --force, kept for consistency with sweep runners.",
    )
    args = parser.parse_args()

    args.n_values = _parse_int_values(args.n_values, "N")
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
    if args.batch_size < 0:
        parser.error("--batch-size must be non-negative")
    if args.n_jobs != -1 and args.n_jobs <= 0:
        parser.error("--n-jobs must be -1 or positive")
    if args.steady_start_frac < 0 or args.steady_start_frac >= 1:
        parser.error("--steady-start-frac must satisfy 0 <= value < 1")
    return args


def _parse_int_values(raw: str, name: str) -> list[int]:
    values = [value.strip() for value in raw.split(",") if value.strip()]
    if not values:
        msg = f"{name} values must not be empty"
        raise ValueError(msg)
    parsed = [int(value) for value in values]
    if any(value <= 0 for value in parsed):
        msg = f"{name} values must be positive"
        raise ValueError(msg)
    return parsed


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


def _prepare_output(
    handle: h5py.File,
    args: argparse.Namespace,
    times: np.ndarray[Any, np.dtype[np.float64]],
) -> None:
    config = _config(args)
    if "format" not in handle.attrs:
        handle.attrs["format"] = "openquantumsim.dicke_mi_distribution"
        handle.attrs["format_version"] = "1"
        handle.attrs["config_json"] = json.dumps(config, sort_keys=True)
        handle.create_dataset("N_values", data=np.array(args.n_values, dtype=np.int64))
        handle.create_dataset(
            "kappa_values",
            data=np.array(args.kappa_values, dtype=np.float64),
        )
        handle.create_dataset("t_axis", data=times)
        return

    if handle.attrs.get("format") != "openquantumsim.dicke_mi_distribution":
        msg = "Output file is not an OpenQuantumSim Dicke MI distribution file."
        raise ValueError(msg)
    if json.loads(str(handle.attrs["config_json"])) != config:
        msg = "Existing output config does not match this run. Use --restart."
        raise ValueError(msg)
    if not np.allclose(np.asarray(handle["t_axis"]), times):
        msg = "Existing output time axis does not match this run."
        raise ValueError(msg)


def _config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "N_values": args.n_values,
        "kappa_values": args.kappa_values,
        "n_traj": args.n_traj,
        "time_points": args.time_points,
        "t_final": args.t_final,
        "max_step": args.max_step,
        "checkpoint_every": args.checkpoint_every,
        "steady_start_frac": args.steady_start_frac,
        "seed": args.seed,
    }


def _require_point_group(
    handle: h5py.File,
    args: argparse.Namespace,
    n_spins: int,
    kappa: float,
    times: np.ndarray[Any, np.dtype[np.float64]],
) -> h5py.Group:
    kappa_group = handle.require_group(f"kappa_{_safe_float_label(kappa)}")
    group = kappa_group.require_group(f"N_{n_spins}")
    shape = (args.n_traj, len(times))
    _require_dataset(group, "MI_time_A", shape)
    _require_dataset(group, "MI_time_B", shape)

    if "t_axis" not in group:
        group.create_dataset("t_axis", data=times)
    elif not np.allclose(np.asarray(group["t_axis"]), times):
        msg = f"Time axis mismatch for N={n_spins}, kappa={kappa}."
        raise ValueError(msg)

    n_transient = _steady_start_index(len(times), args.steady_start_frac)
    group.attrs["N"] = n_spins
    group.attrs["kappa"] = kappa
    group.attrs["n_traj"] = args.n_traj
    group.attrs["time_points"] = len(times)
    group.attrs["t_final"] = float(times[-1])
    group.attrs["max_step"] = args.max_step
    group.attrs["steady_start_frac"] = args.steady_start_frac
    group.attrs["n_transient"] = n_transient
    if "completed" not in group.attrs:
        group.attrs["completed"] = 0
    return group


def _require_dataset(
    group: h5py.Group,
    name: str,
    shape: tuple[int, int],
) -> h5py.Dataset:
    if name in group:
        dataset = group[name]
        if dataset.shape != shape:
            msg = f"Dataset {dataset.name} has shape {dataset.shape}, expected {shape}."
            raise ValueError(msg)
        return dataset
    return group.create_dataset(name, shape=shape, dtype=np.float64)


def _run_point(
    group: h5py.Group,
    args: argparse.Namespace,
    n_spins: int,
    kappa: float,
    times: np.ndarray[Any, np.dtype[np.float64]],
    point_index: int,
) -> None:
    mi_time_a = group["MI_time_A"]
    mi_time_b = group["MI_time_B"]
    completed = int(group.attrs["completed"])
    batch_size = _effective_batch_size(args.batch_size, args.checkpoint_every)
    batches = _trajectory_batches(completed, args.n_traj, batch_size)
    if not batches:
        _write_steady_datasets(group, int(group.attrs["n_transient"]))
        group.file.flush()
        return

    requests = [
        BatchRequest(
            n_spins=n_spins,
            kappa=kappa,
            times=times,
            max_step=args.max_step,
            seed=args.seed,
            point_index=point_index,
            start=start,
            stop=stop,
        )
        for start, stop in batches
    ]
    n_jobs = min(_effective_n_jobs(args.n_jobs), len(requests))

    if n_jobs == 1:
        for request in requests:
            result = _run_trajectory_batch(request)
            _write_batch_result(group, mi_time_a, mi_time_b, result, args)
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            for result in executor.map(_run_trajectory_batch, requests):
                _write_batch_result(group, mi_time_a, mi_time_b, result, args)

    _write_steady_datasets(group, int(group.attrs["n_transient"]))
    group.file.flush()


def _run_trajectory_batch(request: BatchRequest) -> BatchResult:
    system = two_ensemble_dicke_system(request.n_spins, request.kappa)
    batch_count = request.stop - request.start
    mi_a_batch = np.empty((batch_count, len(request.times)), dtype=np.float64)
    mi_b_batch = np.empty((batch_count, len(request.times)), dtype=np.float64)

    for local_idx, traj_idx in enumerate(range(request.start, request.stop)):
        seed = request.seed + request.point_index * 1_000_000 + traj_idx
        result = oqs.single_trajectory(
            system.H,
            system.psi0,
            request.times,
            c_ops=system.c_ops,
            e_ops=system.e_ops,
            options=oqs.Options(
                seed=seed,
                max_step=request.max_step,
                save_states=True,
            ),
        )
        if result.states is None:
            msg = "single_trajectory did not return saved states."
            raise RuntimeError(msg)

        mi_a, mi_b = trajectory_dicke_mutual_information(
            result.states,
            request.n_spins,
        )
        mi_a_batch[local_idx, :] = mi_a
        mi_b_batch[local_idx, :] = mi_b

    return BatchResult(
        start=request.start,
        stop=request.stop,
        mi_a=mi_a_batch,
        mi_b=mi_b_batch,
    )


def _write_batch_result(
    group: h5py.Group,
    mi_time_a: h5py.Dataset,
    mi_time_b: h5py.Dataset,
    result: BatchResult,
    args: argparse.Namespace,
) -> None:
    mi_time_a[result.start : result.stop, :] = result.mi_a
    mi_time_b[result.start : result.stop, :] = result.mi_b
    group.attrs["completed"] = result.stop
    group.file.flush()
    if not args.no_progress:
        print(
            f"  checkpoint N={group.attrs['N']} kappa={group.attrs['kappa']}: "
            f"{result.stop}/{group.attrs['n_traj']}",
        )


def _write_steady_datasets(group: h5py.Group, steady_start_index: int) -> None:
    completed = int(group.attrs["completed"])
    if completed <= 0:
        return

    mi_time_a = np.asarray(group["MI_time_A"][:completed, :], dtype=np.float64)
    mi_time_b = np.asarray(group["MI_time_B"][:completed, :], dtype=np.float64)
    if steady_start_index < 0 or steady_start_index >= mi_time_a.shape[1]:
        msg = "steady_start_index must point inside the time axis."
        raise ValueError(msg)

    _replace_dataset(
        group,
        "MI_steady_A",
        mi_time_a[:, steady_start_index:].mean(axis=1),
    )
    _replace_dataset(
        group,
        "MI_steady_B",
        mi_time_b[:, steady_start_index:].mean(axis=1),
    )


def _replace_dataset(
    group: h5py.Group,
    name: str,
    data: np.ndarray[Any, np.dtype[np.float64]],
) -> None:
    if name in group:
        del group[name]
    group.create_dataset(name, data=data)


def _steady_start_index(time_points: int, steady_start_frac: float) -> int:
    return min(time_points - 1, int(np.ceil(steady_start_frac * time_points)))


def _effective_batch_size(batch_size: int, checkpoint_every: int) -> int:
    return checkpoint_every if batch_size == 0 else batch_size


def _effective_n_jobs(n_jobs: int) -> int:
    if n_jobs == -1:
        return max(os.cpu_count() or 1, 1)
    return n_jobs


def _trajectory_batches(
    completed: int,
    n_traj: int,
    batch_size: int,
) -> list[tuple[int, int]]:
    return [
        (start, min(start + batch_size, n_traj))
        for start in range(completed, n_traj, batch_size)
    ]


def _safe_float_label(value: float) -> str:
    return f"{value:.12g}".replace("-", "m").replace(".", "p")


if __name__ == "__main__":
    main()
