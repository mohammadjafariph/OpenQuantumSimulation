"""Benchmark Dicke mutual-information trajectory batches."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from examples.dicke import run_mi_distribution as mi_runner  # noqa: E402


def main() -> None:
    args = _parse_args()
    times = np.linspace(0.0, args.t_final, args.time_points)

    print("OpenQuantumSim Dicke MI benchmark")
    print(f"cpu_count={os.cpu_count() or 1}")
    print(
        "config: "
        f"N={args.N}, kappa={args.kappa}, n_traj={args.n_traj}, "
        f"time_points={args.time_points}, t_final={args.t_final}, "
        f"max_step={args.max_step}, batch_size={args.batch_size}, "
        f"repeats={args.repeats}",
    )

    _warm_up(args, times)
    rows: list[dict[str, Any]] = []
    serial_elapsed: float | None = None
    reference_signature: float | None = None
    for n_jobs in args.n_jobs:
        row, signature = _benchmark_mode(args, times, n_jobs, reference_signature)
        if n_jobs == 1:
            serial_elapsed = float(row["elapsed_s"])
            reference_signature = signature
            row["speedup_vs_serial"] = 1.0
        elif serial_elapsed is not None:
            row["speedup_vs_serial"] = serial_elapsed / float(row["elapsed_s"])
        rows.append(row)

    _print_rows(rows)
    if args.target_n_traj is not None:
        _print_runtime_estimates(rows, args.target_n_traj)

    if args.json:
        payload = {
            "config": {
                "N": args.N,
                "batch_size": args.batch_size,
                "kappa": args.kappa,
                "max_step": args.max_step,
                "n_traj": args.n_traj,
                "repeats": args.repeats,
                "target_n_traj": args.target_n_traj,
                "t_final": args.t_final,
                "time_points": args.time_points,
            },
            "rows": rows,
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nWrote {args.json}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark OpenQuantumSim Dicke MI trajectory batches.",
    )
    parser.add_argument("--N", type=int, default=6)
    parser.add_argument("--kappa", type=float, default=0.1)
    parser.add_argument("--n-traj", type=int, default=20)
    parser.add_argument("--time-points", type=int, default=101)
    parser.add_argument("--t-final", type=float, default=1.0)
    parser.add_argument("--max-step", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--warmup-trajectories", type=int, default=1)
    parser.add_argument(
        "--n-jobs",
        type=int,
        nargs="+",
        default=[1, 2],
        help="Process counts to benchmark. Use -1 for all CPUs.",
    )
    parser.add_argument(
        "--target-n-traj",
        type=int,
        default=None,
        help="Optional production trajectory count for runtime estimates.",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Optional path for a JSON benchmark report.",
    )
    args = parser.parse_args()

    if args.N <= 0:
        parser.error("--N must be positive")
    if args.kappa < 0:
        parser.error("--kappa must be non-negative")
    if args.n_traj <= 0:
        parser.error("--n-traj must be positive")
    if args.time_points <= 1:
        parser.error("--time-points must be greater than 1")
    if args.t_final <= 0:
        parser.error("--t-final must be positive")
    if args.max_step <= 0:
        parser.error("--max-step must be positive")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.repeats <= 0:
        parser.error("--repeats must be positive")
    if args.warmup_trajectories <= 0:
        parser.error("--warmup-trajectories must be positive")
    if args.target_n_traj is not None and args.target_n_traj <= 0:
        parser.error("--target-n-traj must be positive")
    invalid_jobs = [value for value in args.n_jobs if value != -1 and value <= 0]
    if invalid_jobs:
        parser.error("--n-jobs values must be -1 or positive")
    return args


def _warm_up(
    args: argparse.Namespace,
    times: np.ndarray[Any, np.dtype[np.float64]],
) -> None:
    request = mi_runner.BatchRequest(
        n_spins=args.N,
        kappa=args.kappa,
        times=times[: min(len(times), 3)],
        max_step=args.max_step,
        seed=args.seed,
        point_index=0,
        start=0,
        stop=args.warmup_trajectories,
    )
    mi_runner._run_trajectory_batch(request)


def _benchmark_mode(
    args: argparse.Namespace,
    times: np.ndarray[Any, np.dtype[np.float64]],
    n_jobs: int,
    reference_signature: float | None,
) -> tuple[dict[str, Any], float]:
    elapsed_values: list[float] = []
    signature_delta = 0.0
    latest_signature: float | None = None
    effective_jobs = _effective_n_jobs(n_jobs)
    batch_ranges = mi_runner._trajectory_batches(0, args.n_traj, args.batch_size)
    worker_count = min(effective_jobs, len(batch_ranges))

    for repeat in range(args.repeats):
        requests = [
            mi_runner.BatchRequest(
                n_spins=args.N,
                kappa=args.kappa,
                times=times,
                max_step=args.max_step,
                seed=args.seed + repeat * 1_000_000,
                point_index=0,
                start=start,
                stop=stop,
            )
            for start, stop in batch_ranges
        ]
        started = time.perf_counter()
        results = _run_requests(requests, worker_count)
        elapsed = time.perf_counter() - started
        elapsed_values.append(elapsed)

        latest_signature = _signature(results)
        if reference_signature is not None:
            signature_delta = max(
                signature_delta,
                abs(latest_signature - reference_signature),
            )

    if latest_signature is None:
        msg = "benchmark did not run any repeats"
        raise RuntimeError(msg)

    elapsed_s = statistics.median(elapsed_values)
    trajectories_per_second = args.n_traj / elapsed_s
    row = {
        "n_jobs": n_jobs,
        "workers": worker_count,
        "elapsed_s": elapsed_s,
        "trajectories_per_second": trajectories_per_second,
        "seconds_per_trajectory": elapsed_s / args.n_traj,
        "speedup_vs_serial": None,
        "signature_delta": signature_delta,
    }
    return row, latest_signature


def _run_requests(
    requests: list[mi_runner.BatchRequest],
    worker_count: int,
) -> list[mi_runner.BatchResult]:
    if worker_count == 1:
        return [mi_runner._run_trajectory_batch(request) for request in requests]
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        return list(executor.map(mi_runner._run_trajectory_batch, requests))


def _signature(results: list[mi_runner.BatchResult]) -> float:
    total = 0.0
    for result in results:
        total += float(np.sum(result.mi_a))
        total += float(np.sum(result.mi_b))
    return total


def _effective_n_jobs(n_jobs: int) -> int:
    if n_jobs == -1:
        return max(os.cpu_count() or 1, 1)
    return n_jobs


def _print_rows(rows: list[dict[str, Any]]) -> None:
    print()
    print(
        f"{'n_jobs':>8} {'workers':>8} {'elapsed_s':>10} "
        f"{'traj/s':>10} {'s/traj':>10} {'speedup':>8} {'sig_delta':>11}",
    )
    print("-" * 80)
    for row in rows:
        speedup = row["speedup_vs_serial"]
        speedup_text = "-" if speedup is None else f"{float(speedup):.2f}x"
        print(
            f"{row['n_jobs']:>8} "
            f"{row['workers']:>8} "
            f"{float(row['elapsed_s']):>10.4f} "
            f"{float(row['trajectories_per_second']):>10.3f} "
            f"{float(row['seconds_per_trajectory']):>10.4f} "
            f"{speedup_text:>8} "
            f"{float(row['signature_delta']):>11.3e}",
        )


def _print_runtime_estimates(rows: list[dict[str, Any]], target_n_traj: int) -> None:
    print()
    print(f"Estimated wall times for {target_n_traj} trajectories:")
    for row in rows:
        seconds = target_n_traj / float(row["trajectories_per_second"])
        print(
            f"  n_jobs={row['n_jobs']}: "
            f"{_format_duration(seconds)} at "
            f"{float(row['trajectories_per_second']):.3f} traj/s",
        )


def _format_duration(seconds: float) -> str:
    seconds = max(seconds, 0.0)
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        minutes = int(seconds // 60)
        return f"{minutes}m {seconds - 60 * minutes:.1f}s"
    hours = int(seconds // 3600)
    minutes = int((seconds - 3600 * hours) // 60)
    return f"{hours}h {minutes}m"


if __name__ == "__main__":
    main()
