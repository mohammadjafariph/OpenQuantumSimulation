"""Benchmark serial vs threaded Monte Carlo wave-function trajectories."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np

import openquantumsim as oqs


def main() -> None:
    args = _parse_args()
    system = _build_qubit_decay(args.gamma, args.time_points, args.t_final)

    print("OpenQuantumSim mcsolve benchmark")
    print(f"JULIA_NUM_THREADS={os.environ.get('JULIA_NUM_THREADS', 'unset')}")
    print(
        "config: "
        f"n_traj={args.n_traj}, time_points={args.time_points}, "
        f"t_final={args.t_final}, max_step={args.max_step}, repeats={args.repeats}"
    )

    _warm_up(system, args)

    rows: list[dict[str, Any]] = []
    reference_expect: np.ndarray | None = None
    serial_elapsed: float | None = None
    for n_jobs in args.n_jobs:
        row, reference_expect = _benchmark_mode(
            system,
            args,
            n_jobs,
            reference_expect,
        )
        if n_jobs == 1:
            serial_elapsed = float(row["elapsed_s"])
            row["speedup_vs_serial"] = 1.0
        elif serial_elapsed is not None:
            row["speedup_vs_serial"] = serial_elapsed / float(row["elapsed_s"])
        rows.append(row)

    _print_rows(rows)
    if any(row["n_jobs"] != 1 and row["n_workers"] == 1 for row in rows):
        print()
        print(
            "Note: threaded mode used one worker. Start a fresh process with "
            "JULIA_NUM_THREADS=auto or JULIA_NUM_THREADS=<N> for scaling numbers."
        )

    if args.json:
        payload = {
            "config": {
                "gamma": args.gamma,
                "max_step": args.max_step,
                "n_traj": args.n_traj,
                "repeats": args.repeats,
                "t_final": args.t_final,
                "time_points": args.time_points,
                "julia_num_threads": os.environ.get("JULIA_NUM_THREADS"),
            },
            "rows": rows,
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nWrote {args.json}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark OpenQuantumSim mcsolve serial vs threaded execution.",
    )
    parser.add_argument("--n-traj", type=int, default=1000)
    parser.add_argument("--time-points", type=int, default=41)
    parser.add_argument("--t-final", type=float, default=4.0)
    parser.add_argument("--gamma", type=float, default=0.35)
    parser.add_argument("--max-step", type=float, default=0.01)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--warmup-trajectories", type=int, default=20)
    parser.add_argument(
        "--n-jobs",
        type=int,
        nargs="+",
        default=[1, -1],
        help="Job modes to benchmark. Use 1 for serial and -1 for all Julia threads.",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Optional path for a JSON benchmark report.",
    )
    args = parser.parse_args()

    if args.n_traj <= 0:
        parser.error("--n-traj must be positive")
    if args.time_points <= 1:
        parser.error("--time-points must be greater than 1")
    if args.t_final <= 0:
        parser.error("--t-final must be positive")
    if args.gamma < 0:
        parser.error("--gamma must be non-negative")
    if args.max_step <= 0:
        parser.error("--max-step must be positive")
    if args.repeats <= 0:
        parser.error("--repeats must be positive")
    if args.warmup_trajectories <= 0:
        parser.error("--warmup-trajectories must be positive")
    invalid_jobs = [value for value in args.n_jobs if value != -1 and value <= 0]
    if invalid_jobs:
        parser.error("--n-jobs values must be -1 or positive")
    return args


def _build_qubit_decay(
    gamma: float,
    time_points: int,
    t_final: float,
) -> tuple[oqs.Operator, np.ndarray, np.ndarray, oqs.Operator, oqs.Operator]:
    atom = oqs.SpinSpace(0.5, label="atom")
    hamiltonian = 0.0 * oqs.sigmaz(atom)
    collapse = np.sqrt(gamma) * oqs.sigmam(atom)
    excited = oqs.basis(atom, "up")
    excited_projector = oqs.Operator(oqs.ket2dm(excited), atom, "P_excited")
    times = np.linspace(0.0, t_final, time_points)
    return hamiltonian, excited, times, collapse, excited_projector


def _warm_up(
    system: tuple[oqs.Operator, np.ndarray, np.ndarray, oqs.Operator, oqs.Operator],
    args: argparse.Namespace,
) -> None:
    hamiltonian, psi0, times, collapse, excited_projector = system
    warmup_times = times[: min(len(times), 3)]
    for n_jobs in sorted(set(args.n_jobs)):
        oqs.mcsolve(
            hamiltonian,
            psi0,
            warmup_times,
            c_ops=[collapse],
            e_ops=[excited_projector],
            n_traj=args.warmup_trajectories,
            options=oqs.Options(
                seed=args.seed,
                max_step=args.max_step,
                n_jobs=n_jobs,
                progress=False,
            ),
        )


def _benchmark_mode(
    system: tuple[oqs.Operator, np.ndarray, np.ndarray, oqs.Operator, oqs.Operator],
    args: argparse.Namespace,
    n_jobs: int,
    reference_expect: np.ndarray | None,
) -> tuple[dict[str, Any], np.ndarray]:
    hamiltonian, psi0, times, collapse, excited_projector = system
    elapsed_values: list[float] = []
    backend_values: list[float] = []
    workers_values: list[int] = []
    threaded_values: list[bool] = []
    max_delta = 0.0
    latest_expect: np.ndarray | None = None

    for repeat in range(args.repeats):
        started = time.perf_counter()
        result = oqs.mcsolve(
            hamiltonian,
            psi0,
            times,
            c_ops=[collapse],
            e_ops=[excited_projector],
            n_traj=args.n_traj,
            options=oqs.Options(
                seed=args.seed + repeat,
                max_step=args.max_step,
                n_jobs=n_jobs,
                progress=False,
            ),
        )
        elapsed_values.append(time.perf_counter() - started)
        backend_values.append(float(result.solver_stats.get("wall_time", 0.0)))
        workers_values.append(int(result.solver_stats.get("n_workers", 1)))
        threaded_values.append(bool(result.solver_stats.get("threaded", False)))
        latest_expect = result.expect[0].real.copy()

        if reference_expect is None and n_jobs == 1 and repeat == 0:
            reference_expect = latest_expect.copy()
        if reference_expect is not None:
            max_delta = max(
                max_delta,
                float(np.max(np.abs(latest_expect - reference_expect))),
            )

    if latest_expect is None:
        msg = "benchmark did not run any repeats"
        raise RuntimeError(msg)
    if reference_expect is None:
        reference_expect = latest_expect.copy()

    row = {
        "n_jobs": n_jobs,
        "n_workers": max(workers_values),
        "threaded": any(threaded_values),
        "elapsed_s": statistics.median(elapsed_values),
        "backend_wall_s": statistics.median(backend_values),
        "speedup_vs_serial": None,
        "max_abs_delta": max_delta,
    }
    return row, reference_expect


def _print_rows(rows: list[dict[str, Any]]) -> None:
    print()
    print(
        f"{'n_jobs':>8} {'workers':>8} {'threaded':>9} "
        f"{'elapsed_s':>10} {'backend_s':>10} {'speedup':>8} {'max_delta':>11}"
    )
    print("-" * 76)
    for row in rows:
        speedup = row["speedup_vs_serial"]
        speedup_text = "-" if speedup is None else f"{float(speedup):.2f}x"
        print(
            f"{row['n_jobs']:>8} "
            f"{row['n_workers']:>8} "
            f"{str(row['threaded']):>9} "
            f"{float(row['elapsed_s']):>10.4f} "
            f"{float(row['backend_wall_s']):>10.4f} "
            f"{speedup_text:>8} "
            f"{float(row['max_abs_delta']):>11.3e}"
        )


if __name__ == "__main__":
    main()
