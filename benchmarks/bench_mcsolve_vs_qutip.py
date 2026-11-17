"""Benchmark OpenQuantumSim MCWF trajectories against QuTiP.

Both libraries solve the same qubit-decay problem with the same integrator
tolerances, and both results are checked against the exact excited-state
population ``exp(-gamma t)`` so that a speed number is never reported for a
wrong answer.

QuTiP runs serially (its default). OpenQuantumSim is timed twice: with one
Julia thread (``n_jobs=1``), which is the like-for-like comparison, and with
every available Julia thread (``n_jobs=-1``). Set ``JULIA_NUM_THREADS``
before starting Python to choose how many threads exist.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import qutip as qt  # type: ignore[import-untyped]

import openquantumsim as oqs

# Default tolerances are QuTiP 5's own defaults, applied to both libraries.
DEFAULT_RTOL = 1e-6
DEFAULT_ATOL = 1e-8


def main() -> None:
    """Run the benchmark and optionally write a JSON report."""
    args = _parse_args()
    qutip_system = _qutip_qubit_decay(args.gamma, args.time_points, args.t_final)
    oqs_system = _oqs_qubit_decay(args.gamma, args.time_points, args.t_final)
    environment = _environment()

    print("OpenQuantumSim vs QuTiP MCWF benchmark")
    for key, value in environment.items():
        print(f"{key}: {value}")
    print(
        "config: "
        f"n_traj={args.n_traj}, repeats={args.repeats}, "
        f"time_points={args.time_points}, t_final={args.t_final}, "
        f"rtol={args.rtol}, atol={args.atol}, max_step={args.max_step}"
    )

    _warm_up(qutip_system, oqs_system, args)

    rows = [
        _benchmark_count(count, qutip_system, oqs_system, args)
        for count in args.n_traj
    ]
    _print_rows(rows)

    if args.json:
        payload = {
            "environment": environment,
            "config": {
                "gamma": args.gamma,
                "max_step": args.max_step,
                "repeats": args.repeats,
                "rtol": args.rtol,
                "atol": args.atol,
                "seed": args.seed,
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
    parser.add_argument("--n-traj", type=int, nargs="+", default=[200, 1000, 5000])
    parser.add_argument("--time-points", type=int, default=31)
    parser.add_argument("--t-final", type=float, default=2.0)
    parser.add_argument("--gamma", type=float, default=0.35)
    parser.add_argument(
        "--rtol",
        type=float,
        default=DEFAULT_RTOL,
        help="Relative tolerance for both libraries (QuTiP's default).",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=DEFAULT_ATOL,
        help="Absolute tolerance for both libraries (QuTiP's default).",
    )
    parser.add_argument(
        "--max-step",
        type=float,
        default=None,
        help="Optional cap on the integrator step for both libraries.",
    )
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
    if args.rtol <= 0 or args.atol <= 0:
        parser.error("--rtol and --atol must be positive")
    if args.max_step is not None and args.max_step <= 0:
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


def _qutip_options(args: argparse.Namespace) -> dict[str, Any]:
    options: dict[str, Any] = {
        "progress_bar": "",
        "rtol": args.rtol,
        "atol": args.atol,
    }
    if args.max_step is not None:
        options["max_step"] = args.max_step
    return options


def _oqs_options(args: argparse.Namespace, seed: int, n_jobs: int) -> oqs.Options:
    return oqs.Options(
        seed=seed,
        rtol=args.rtol,
        atol=args.atol,
        max_step=args.max_step,
        n_jobs=n_jobs,
        progress=False,
    )


def _run_qutip(
    qutip_system: dict[str, Any],
    times: np.ndarray,
    n_traj: int,
    seed: int,
    args: argparse.Namespace,
) -> np.ndarray:
    result = qt.mcsolve(
        qutip_system["H"],
        qutip_system["psi0"],
        times,
        qutip_system["c_ops"],
        e_ops=qutip_system["e_ops"],
        ntraj=n_traj,
        seeds=seed,
        options=_qutip_options(args),
    )
    return np.real(np.asarray(result.expect[0]))


def _run_oqs(
    oqs_system: tuple[oqs.Operator, np.ndarray, np.ndarray, oqs.Operator, oqs.Operator],
    times: np.ndarray,
    n_traj: int,
    seed: int,
    n_jobs: int,
    args: argparse.Namespace,
) -> oqs.Result:
    hamiltonian, psi0, _, collapse, excited_projector = oqs_system
    return oqs.mcsolve(
        hamiltonian,
        psi0,
        times,
        c_ops=[collapse],
        e_ops=[excited_projector],
        n_traj=n_traj,
        options=_oqs_options(args, seed, n_jobs),
    )


def _warm_up(
    qutip_system: dict[str, Any],
    oqs_system: tuple[oqs.Operator, np.ndarray, np.ndarray, oqs.Operator, oqs.Operator],
    args: argparse.Namespace,
) -> None:
    times = qutip_system["times"][: min(len(qutip_system["times"]), 3)]
    _run_qutip(qutip_system, times, args.warmup_trajectories, args.seed, args)
    for n_jobs in (1, -1):
        _run_oqs(oqs_system, times, args.warmup_trajectories, args.seed, n_jobs, args)


def _benchmark_count(
    n_traj: int,
    qutip_system: dict[str, Any],
    oqs_system: tuple[oqs.Operator, np.ndarray, np.ndarray, oqs.Operator, oqs.Operator],
    args: argparse.Namespace,
) -> dict[str, Any]:
    times = qutip_system["times"]
    exact = np.exp(-args.gamma * times)
    qutip_elapsed: list[float] = []
    serial_elapsed: list[float] = []
    threaded_elapsed: list[float] = []
    qutip_errors: list[float] = []
    oqs_errors: list[float] = []
    workers = 1

    for repeat in range(args.repeats):
        seed = args.seed + repeat

        started = time.perf_counter()
        qutip_mean = _run_qutip(qutip_system, times, n_traj, seed, args)
        qutip_elapsed.append(time.perf_counter() - started)
        qutip_errors.append(float(np.max(np.abs(qutip_mean - exact))))

        started = time.perf_counter()
        serial = _run_oqs(oqs_system, times, n_traj, seed, 1, args)
        serial_elapsed.append(time.perf_counter() - started)
        oqs_errors.append(float(np.max(np.abs(serial.expect[0].real - exact))))

        started = time.perf_counter()
        threaded = _run_oqs(oqs_system, times, n_traj, seed, -1, args)
        threaded_elapsed.append(time.perf_counter() - started)
        workers = max(workers, int(threaded.solver_stats.get("n_workers", 1)))
        if not np.array_equal(serial.expect[0], threaded.expect[0]):
            msg = "OpenQuantumSim results changed with the thread count."
            raise RuntimeError(msg)

    qutip_median = statistics.median(qutip_elapsed)
    serial_median = statistics.median(serial_elapsed)
    threaded_median = statistics.median(threaded_elapsed)
    return {
        "n_traj": n_traj,
        # Statistical error of a mean of n_traj Bernoulli samples is at most
        # 0.5 / sqrt(n_traj); both errors should be a few times that.
        "statistical_scale": 0.5 / np.sqrt(n_traj),
        "qutip_elapsed_s": qutip_median,
        "oqs_serial_elapsed_s": serial_median,
        "oqs_threaded_elapsed_s": threaded_median,
        "oqs_workers": workers,
        "oqs_serial_speedup_vs_qutip": qutip_median / serial_median,
        "oqs_threaded_speedup_vs_qutip": qutip_median / threaded_median,
        "qutip_max_error": max(qutip_errors),
        "oqs_max_error": max(oqs_errors),
    }


def _print_rows(rows: list[dict[str, Any]]) -> None:
    print()
    print(
        f"{'n_traj':>7} {'QuTiP_s':>9} {'OQS_1t_s':>9} {'OQS_Nt_s':>9} "
        f"{'threads':>7} {'1t_vs_QuTiP':>11} {'Nt_vs_QuTiP':>11} "
        f"{'QuTiP_err':>9} {'OQS_err':>9} {'0.5/sqrt(n)':>11}"
    )
    print("-" * 104)
    for row in rows:
        print(
            f"{int(row['n_traj']):>7} "
            f"{float(row['qutip_elapsed_s']):>9.4f} "
            f"{float(row['oqs_serial_elapsed_s']):>9.4f} "
            f"{float(row['oqs_threaded_elapsed_s']):>9.4f} "
            f"{int(row['oqs_workers']):>7} "
            f"{float(row['oqs_serial_speedup_vs_qutip']):>10.2f}x "
            f"{float(row['oqs_threaded_speedup_vs_qutip']):>10.2f}x "
            f"{float(row['qutip_max_error']):>9.4f} "
            f"{float(row['oqs_max_error']):>9.4f} "
            f"{float(row['statistical_scale']):>11.4f}"
        )


def _environment() -> dict[str, str]:
    from openquantumsim._julia_bridge import get_julia, load_backend

    load_backend()
    julia = get_julia()
    return {
        "OpenQuantumSim": oqs.__version__,
        "QuTiP": str(qt.__version__),
        "Python": platform.python_version(),
        "Julia": str(julia.seval("string(VERSION)")),
        "Julia threads": str(julia.seval("Threads.nthreads()")),
        "Platform": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "Processor": platform.processor() or "unknown",
        "CPU count": str(os.cpu_count()),
    }


if __name__ == "__main__":
    main()
