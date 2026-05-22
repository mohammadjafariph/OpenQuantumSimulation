"""Benchmark OpenQuantumSim MCWF trajectories against QuTiP."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import qutip as qt  # type: ignore[import-untyped]

import openquantumsim as oqs


def main() -> None:
    """Run the benchmark and optionally write a JSON report."""
    args = _parse_args()
    qutip_system = _qutip_qubit_decay(args.gamma, args.time_points, args.t_final)
    oqs_system = _oqs_qubit_decay(args.gamma, args.time_points, args.t_final)

    print("OpenQuantumSim vs QuTiP MCWF benchmark")
    print(f"JULIA_NUM_THREADS={os.environ.get('JULIA_NUM_THREADS', 'unset')}")
    print(
        "config: "
        f"n_traj={args.n_traj}, repeats={args.repeats}, "
        f"time_points={args.time_points}, t_final={args.t_final}, "
        f"max_step={args.max_step}"
    )

    _warm_up(qutip_system, oqs_system, args)

    rows = [
        _benchmark_count(count, qutip_system, oqs_system, args)
        for count in args.n_traj
    ]
    _print_rows(rows)

    if args.json:
        payload = {
            "config": {
                "gamma": args.gamma,
                "julia_num_threads": os.environ.get("JULIA_NUM_THREADS"),
                "max_step": args.max_step,
                "repeats": args.repeats,
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
        description="Benchmark qubit-decay MCWF trajectories against QuTiP.",
    )
    parser.add_argument("--n-traj", type=int, nargs="+", default=[50, 200, 1000])
    parser.add_argument("--time-points", type=int, default=31)
    parser.add_argument("--t-final", type=float, default=2.0)
    parser.add_argument("--gamma", type=float, default=0.35)
    parser.add_argument("--max-step", type=float, default=0.02)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--warmup-trajectories", type=int, default=5)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    if any(count <= 0 for count in args.n_traj):
        parser.error("--n-traj values must be positive")
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
    return args


def _qutip_qubit_decay(
    gamma: float,
    time_points: int,
    t_final: float,
) -> dict[str, Any]:
    excited = qt.basis(2, 0)
    return {
        "H": 0 * qt.sigmaz(),
        "psi0": excited,
        "times": np.linspace(0.0, t_final, time_points),
        "c_ops": [np.sqrt(gamma) * qt.sigmam()],
        "e_ops": [excited * excited.dag()],
    }


def _oqs_qubit_decay(
    gamma: float,
    time_points: int,
    t_final: float,
) -> tuple[oqs.Operator, np.ndarray, np.ndarray, oqs.Operator, oqs.Operator]:
    atom = oqs.SpinSpace(0.5, label="atom")
    excited = oqs.basis(atom, "up")
    return (
        0.0 * oqs.sigmaz(atom),
        excited,
        np.linspace(0.0, t_final, time_points),
        np.sqrt(gamma) * oqs.sigmam(atom),
        oqs.Operator(oqs.ket2dm(excited), atom, "P_excited"),
    )


def _warm_up(
    qutip_system: dict[str, Any],
    oqs_system: tuple[oqs.Operator, np.ndarray, np.ndarray, oqs.Operator, oqs.Operator],
    args: argparse.Namespace,
) -> None:
    warmup_times = qutip_system["times"][: min(len(qutip_system["times"]), 3)]
    qt.mcsolve(
        qutip_system["H"],
        qutip_system["psi0"],
        warmup_times,
        qutip_system["c_ops"],
        e_ops=qutip_system["e_ops"],
        ntraj=args.warmup_trajectories,
        seeds=args.seed,
        options={"progress_bar": ""},
    )

    hamiltonian, psi0, times, collapse, excited_projector = oqs_system
    oqs.mcsolve(
        hamiltonian,
        psi0,
        times[: min(len(times), 3)],
        c_ops=[collapse],
        e_ops=[excited_projector],
        n_traj=args.warmup_trajectories,
        options=oqs.Options(
            seed=args.seed,
            max_step=args.max_step,
            n_jobs=-1,
            progress=False,
        ),
    )


def _benchmark_count(
    n_traj: int,
    qutip_system: dict[str, Any],
    oqs_system: tuple[oqs.Operator, np.ndarray, np.ndarray, oqs.Operator, oqs.Operator],
    args: argparse.Namespace,
) -> dict[str, Any]:
    qutip_elapsed: list[float] = []
    oqs_elapsed: list[float] = []
    oqs_backend: list[float] = []
    oqs_workers: list[int] = []

    for repeat in range(args.repeats):
        started = time.perf_counter()
        qt.mcsolve(
            qutip_system["H"],
            qutip_system["psi0"],
            qutip_system["times"],
            qutip_system["c_ops"],
            e_ops=qutip_system["e_ops"],
            ntraj=n_traj,
            seeds=args.seed + repeat,
            options={"progress_bar": ""},
        )
        qutip_elapsed.append(time.perf_counter() - started)

        hamiltonian, psi0, times, collapse, excited_projector = oqs_system
        started = time.perf_counter()
        result = oqs.mcsolve(
            hamiltonian,
            psi0,
            times,
            c_ops=[collapse],
            e_ops=[excited_projector],
            n_traj=n_traj,
            options=oqs.Options(
                seed=args.seed + repeat,
                max_step=args.max_step,
                n_jobs=-1,
                progress=False,
            ),
        )
        oqs_elapsed.append(time.perf_counter() - started)
        oqs_backend.append(float(result.solver_stats.get("wall_time", 0.0)))
        oqs_workers.append(int(result.solver_stats.get("n_workers", 1)))

    qutip_median = statistics.median(qutip_elapsed)
    oqs_median = statistics.median(oqs_elapsed)
    backend_median = statistics.median(oqs_backend)
    return {
        "n_traj": n_traj,
        "qutip_elapsed_s": qutip_median,
        "oqs_elapsed_s": oqs_median,
        "oqs_backend_wall_s": backend_median,
        "oqs_workers": max(oqs_workers),
        "oqs_speedup_vs_qutip": qutip_median / oqs_median,
        "oqs_backend_speedup_vs_qutip": qutip_median / backend_median,
    }


def _print_rows(rows: list[dict[str, Any]]) -> None:
    print()
    print(
        f"{'n_traj':>8} {'QuTiP_s':>10} {'OQS_s':>10} {'OQS_backend_s':>14} "
        f"{'workers':>8} {'speedup':>9} {'backend':>9}"
    )
    print("-" * 82)
    for row in rows:
        print(
            f"{int(row['n_traj']):>8} "
            f"{float(row['qutip_elapsed_s']):>10.5f} "
            f"{float(row['oqs_elapsed_s']):>10.5f} "
            f"{float(row['oqs_backend_wall_s']):>14.5f} "
            f"{int(row['oqs_workers']):>8} "
            f"{float(row['oqs_speedup_vs_qutip']):>8.2f}x "
            f"{float(row['oqs_backend_speedup_vs_qutip']):>8.2f}x"
        )


if __name__ == "__main__":
    main()
