"""Compare mesolve integrator choices for speed and accuracy.

Solves damped Jaynes-Cummings models with every propagation choice
``Options`` offers (``ode`` with Tsit5, Vern7 or Vern9, and ``krylov``) at the
same requested tolerance, and reports the median wall time next to the
largest expectation-value error against a high-precision reference solve
(Vern9 at ``rtol=1e-12``). It needs no QuTiP; use ``bench_vs_qutip.py`` for the
cross-library comparison.

    python benchmarks/bench_mesolve.py --cases jc5 jc10 jc20 --json runs/mesolve.json
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np

import openquantumsim as oqs

CHOICES: dict[str, dict[str, str]] = {
    "tsit5": {"method": "ode", "ode_solver": "tsit5"},
    "vern7": {"method": "ode", "ode_solver": "vern7"},
    "vern9": {"method": "ode", "ode_solver": "vern9"},
    "krylov": {"method": "krylov"},
}
REFERENCE = oqs.Options(method="ode", ode_solver="vern9", rtol=1e-12, atol=1e-14)


def main() -> None:
    """Run the benchmark and optionally write a JSON report."""
    args = _parse_args()
    times = np.linspace(0.0, args.t_final, args.time_points)
    environment = _environment()

    print("OpenQuantumSim mesolve integrator benchmark")
    for key, value in environment.items():
        print(f"{key}: {value}")
    print(
        f"config: repeats={args.repeats}, time_points={args.time_points}, "
        f"t_final={args.t_final}, rtol={args.rtol}, atol={args.atol}"
    )

    rows: list[dict[str, Any]] = []
    for case in args.cases:
        system = _jaynes_cummings(int(case.removeprefix("jc")))
        reference = _solve(system, times, REFERENCE)
        for choice in args.choices:
            options = oqs.Options(rtol=args.rtol, atol=args.atol, **CHOICES[choice])
            _solve(system, times, options)  # warm-up and compilation
            elapsed: list[float] = []
            result = None
            for _ in range(args.repeats):
                started = time.perf_counter()
                result = _solve(system, times, options)
                elapsed.append(time.perf_counter() - started)
            assert result is not None
            rows.append(
                {
                    "case": case,
                    "dim": system.H.dim,
                    "choice": choice,
                    "median_s": statistics.median(elapsed),
                    "min_s": min(elapsed),
                    "max_error": _max_error(result, reference),
                    "nsteps": int(result.solver_stats.get("nsteps", 0)),
                    "trace_drift": float(result.solver_stats.get("trace_drift", 0.0)),
                },
            )
            _print_row(rows[-1])

    if args.json:
        payload = {
            "environment": environment,
            "config": {
                "cases": args.cases,
                "choices": args.choices,
                "repeats": args.repeats,
                "rtol": args.rtol,
                "atol": args.atol,
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
        description="Compare mesolve integrator choices for speed and accuracy.",
    )
    parser.add_argument("--cases", nargs="+", default=["jc5", "jc10", "jc20"])
    parser.add_argument(
        "--choices",
        nargs="+",
        default=list(CHOICES),
        choices=list(CHOICES),
    )
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--time-points", type=int, default=81)
    parser.add_argument("--t-final", type=float, default=6.0)
    parser.add_argument("--rtol", type=float, default=1e-8)
    parser.add_argument("--atol", type=float, default=1e-10)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    for case in args.cases:
        if not case.startswith("jc") or not case[2:].isdigit() or int(case[2:]) < 2:
            parser.error(f"invalid case {case!r}; use jc<N> with N >= 2.")
    if args.repeats <= 0:
        parser.error("--repeats must be positive")
    if args.time_points <= 1:
        parser.error("--time-points must be greater than 1")
    if args.t_final <= 0:
        parser.error("--t-final must be positive")
    if args.rtol <= 0 or args.atol <= 0:
        parser.error("--rtol and --atol must be positive")
    return args


def _jaynes_cummings(cavity_dim: int) -> oqs.JaynesCummingsSystem:
    # The same model as bench_vs_qutip.py, so the timings can be compared.
    return oqs.jaynes_cummings_system(
        cavity_dim,
        cavity_frequency=1.0,
        atom_frequency=1.0,
        coupling=0.08,
        cavity_decay=0.015,
        atom_decay=0.025,
    )


def _solve(
    system: oqs.JaynesCummingsSystem,
    times: np.ndarray,
    options: oqs.Options,
) -> oqs.Result:
    return oqs.mesolve(
        system.H,
        system.rho0,
        times,
        c_ops=system.c_ops,
        e_ops=system.e_ops,
        options=options,
    )


def _max_error(result: oqs.Result, reference: oqs.Result) -> float:
    return max(
        float(np.max(np.abs(np.asarray(got) - np.asarray(want))))
        for got, want in zip(result.expect, reference.expect, strict=True)
    )


def _print_row(row: dict[str, Any]) -> None:
    print(
        f"{row['case']:<6} dim={row['dim']:<4} {row['choice']:<7} "
        f"median={1e3 * row['median_s']:8.2f} ms  "
        f"max_error={row['max_error']:.2e}  steps={row['nsteps']:<5} "
        f"trace_drift={row['trace_drift']:.1e}"
    )


def _environment() -> dict[str, str]:
    from openquantumsim._julia_bridge import get_julia, load_backend

    load_backend()
    julia = get_julia()
    return {
        "OpenQuantumSim": oqs.__version__,
        "Python": platform.python_version(),
        "Julia": str(julia.seval("string(VERSION)")),
        "Platform": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "Processor": platform.processor() or "unknown",
    }


if __name__ == "__main__":
    main()
