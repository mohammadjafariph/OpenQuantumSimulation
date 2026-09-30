"""Benchmark sparse Lindblad scaling at large Hilbert dimensions.

Spin chains with nearest-neighbour exchange and local amplitude damping.
OpenQuantumSim timing is always measured; QuTiP timing and the cross-engine
expectation deltas are included only when QuTiP is installed.

Run locally (after `python -m pip install -e ".[dev]"`):

    python benchmarks/bench_sparse_scaling.py --cases spin9 spin10
        python benchmarks/bench_sparse_scaling.py --cases spin11 --repeats 1 \
            --json runs/sparse_scaling.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

import openquantumsim as oqs

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class SparseCase:
    """A spin-chain case built once for both engines (QuTiP side optional)."""

    name: str
    dim: int
    n_spins: int
    times: FloatArray
    oqs_H: oqs.Operator
    oqs_rho0: np.ndarray
    oqs_c_ops: list[oqs.Operator]
    oqs_e_ops: list[oqs.Operator]
    qutip_H: Any
    qutip_rho0: Any
    qutip_c_ops: list[Any]
    qutip_e_ops: list[Any]


def build_spin_chain(
    n_spins: int,
    times: FloatArray,
    qutip: Any | None,
) -> SparseCase:
    """Build a nearest-neighbour exchange chain with local damping."""
    exchange = 1.0
    coupling = 0.15
    gamma = 0.05

    spins = [oqs.SpinSpace(0.5, label=f"s{i}") for i in range(n_spins)]

    def op(i: int, name: str) -> oqs.Operator:
        factors = [
            oqs.identity(s) if j != i else getattr(oqs, name)(s)
            for j, s in enumerate(spins)
        ]
        return oqs.tensor(*factors)

    oqs_H = 0.0 * op(0, "sigmaz")
    for i in range(n_spins - 1):
        for a, b in (("sigmax", "sigmax"), ("sigmay", "sigmay"), ("sigmaz", "sigmaz")):
            oqs_H = oqs_H + exchange * op(i, a) * op(i + 1, b)
    for i in range(n_spins):
        oqs_H = oqs_H + coupling * op(i, "sigmax")

    oqs_c_ops = [np.sqrt(gamma) * op(i, "sigmam") for i in range(n_spins)]
    dim = 2**n_spins
    full_state = np.zeros(dim, dtype=np.complex128)
    full_state[0] = 1.0
    oqs_rho0 = oqs.ket2dm(full_state)
    def site_projector(i: int) -> oqs.Operator:
        factors = [
            oqs.identity(s)
            if j != i
            else oqs.Operator(oqs.ket2dm(oqs.basis(s, 0)), s)
            for j, s in enumerate(spins)
        ]
        return oqs.tensor(*factors)

    oqs_e_ops = [site_projector(0), site_projector(n_spins - 1)]

    qutip_H = None
    qutip_rho0 = None
    qutip_c_ops: list[Any] = []
    qutip_e_ops: list[Any] = []
    if qutip is not None:
        identity2 = qutip.qeye(2)
        def site(op_i: int, single: Any) -> Any:
            return qutip.tensor(
                *[identity2 if j != op_i else single for j in range(n_spins)],
            )

        sx = [site(i, qutip.sigmax()) for i in range(n_spins)]
        sy = [site(i, qutip.sigmay()) for i in range(n_spins)]
        sz = [site(i, qutip.sigmaz()) for i in range(n_spins)]
        sm = [site(i, qutip.sigmam()) for i in range(n_spins)]

        qutip_H = 0 * sz[0]
        for i in range(n_spins - 1):
            qutip_H = qutip_H + exchange * (
                sx[i] * sx[i + 1] + sy[i] * sy[i + 1] + sz[i] * sz[i + 1]
            )
        for i in range(n_spins):
            qutip_H = qutip_H + coupling * sx[i]
        qutip_c_ops = [np.sqrt(gamma) * sm[i] for i in range(n_spins)]
        qutip_rho0 = qutip.tensor(*[qutip.basis(2, 0) for _ in range(n_spins)])
        qutip_rho0 = qutip_rho0 * qutip_rho0.dag()
        qutip_e_ops = [sm[0].dag() * sm[0], sm[-1].dag() * sm[-1]]

    return SparseCase(
        name=f"spin{n_spins}",
        dim=dim,
        n_spins=n_spins,
        times=times,
        oqs_H=oqs_H,
        oqs_rho0=oqs_rho0,
        oqs_c_ops=oqs_c_ops,
        oqs_e_ops=oqs_e_ops,
        qutip_H=qutip_H,
        qutip_rho0=qutip_rho0,
        qutip_c_ops=qutip_c_ops,
        qutip_e_ops=qutip_e_ops,
    )


def main() -> None:
    args = _parse_args()
    try:
        import qutip  # type: ignore[import-untyped]
    except ImportError:
        qutip = None
        print("QuTiP not installed: measuring OpenQuantumSim timings only.")

    times = np.linspace(0.0, args.t_final, args.time_points)
    rows: list[dict[str, Any]] = []

    print("OpenQuantumSim sparse scaling benchmark")
    print(f"OpenQuantumSim: {oqs.__version__}")
    if qutip is not None:
        print(f"QuTiP: {qutip.__version__}")
    print(
        "config: "
        f"repeats={args.repeats}, time_points={args.time_points}, "
        f"t_final={args.t_final}, rtol={args.rtol:g}, atol={args.atol:g}"
    )
    print()

    for case_name in args.cases:
        n_spins = int(case_name.removeprefix("spin"))
        print(f"building {case_name} (dim={2**n_spins})")
        case = build_spin_chain(n_spins, times, qutip)

        oqs_result, oqs_times = _time_oqs(case, args)
        oqs_median = statistics.median(oqs_times)
        row: dict[str, Any] = {
            "case": case.name,
            "dim": case.dim,
            "n_spins": case.n_spins,
            "time_points": len(case.times),
            "engine": "openquantumsim",
            "method": "mesolve",
            "median_s": oqs_median,
            "min_s": min(oqs_times),
        }

        if qutip is not None:
            _run_qutip(case, args, qutip)
            qutip_times: list[float] = []
            for _ in range(args.repeats):
                start = time.perf_counter()
                qutip_result = _run_qutip(case, args, qutip)
                qutip_times.append(time.perf_counter() - start)
            qutip_median = statistics.median(qutip_times)
            row["engine"] = "both"
            row["qutip_median_s"] = qutip_median
            row["speedup_vs_qutip"] = qutip_median / oqs_median
            row["max_abs_delta_vs_qutip"] = _max_expect_delta(
                oqs_result.expect,
                [np.asarray(e).ravel() for e in qutip_result.expect],
            )

        rows.append(row)
        _print_row(row)

    if args.json is not None:
        payload = {
            "config": {
                "repeats": args.repeats,
                "time_points": args.time_points,
                "t_final": args.t_final,
                "rtol": args.rtol,
                "atol": args.atol,
                "cases": args.cases,
            },
            "versions": {
                "openquantumsim": oqs.__version__,
                "qutip": getattr(qutip, "__version__", None),
            },
            "rows": rows,
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nWrote {args.json}")


def _time_oqs(
    case: SparseCase,
    args: argparse.Namespace,
) -> tuple[oqs.Result, list[float]]:
    timings: list[float] = []
    result: oqs.Result | None = None
    for _ in range(args.repeats):
        start = time.perf_counter()
        result = oqs.mesolve(
            case.oqs_H,
            case.oqs_rho0,
            case.times,
            c_ops=case.oqs_c_ops,
            e_ops=case.oqs_e_ops,
            options=oqs.Options(rtol=args.rtol, atol=args.atol),
        )
        timings.append(time.perf_counter() - start)
    assert result is not None
    return result, timings


def _run_qutip(
    case: SparseCase,
    args: argparse.Namespace,
    qutip: Any,
) -> Any:
    return qutip.mesolve(
        case.qutip_H,
        case.qutip_rho0,
        case.times,
        c_ops=case.qutip_c_ops,
        e_ops=case.qutip_e_ops,
        options={"rtol": args.rtol, "atol": args.atol},
    )


def _max_expect_delta(
    oqs_expect: list[NDArray[np.complex128]],
    qutip_expect: list[NDArray[np.float64]],
) -> float:
    delta = 0.0
    for left, right in zip(oqs_expect, qutip_expect, strict=False):
        delta = max(
            delta,
            float(
                np.max(
                    np.abs(
                        np.asarray(left).ravel().real
                        - np.asarray(right).ravel().real,
                    ),
                ),
            ),
        )
    return delta


def _print_row(row: dict[str, Any]) -> None:
    text = (
        f"{row['case']:<10} dim={row['dim']:<6} {row['engine']:<6} "
        f"oqs_median={row['median_s']:.3f}s"
    )
    if "qutip_median_s" in row:
        text += (
            f"  qutip_median={row['qutip_median_s']:.3f}s"
            f"  speedup={row['speedup_vs_qutip']:.2f}x"
            f"  max|delta|={row['max_abs_delta_vs_qutip']:.2e}"
        )
    print(text)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark sparse Lindblad scaling at large Hilbert dimensions.",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--time-points", type=int, default=51)
    parser.add_argument("--t-final", type=float, default=10.0)
    parser.add_argument("--rtol", type=float, default=1e-6)
    parser.add_argument("--atol", type=float, default=1e-8)
    parser.add_argument(
        "--cases",
        nargs="+",
        default=["spin9", "spin10", "spin11"],
        help="Cases to run: spin<N> such as spin9 (dim 512) or spin11 (dim 2048).",
    )
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    if args.repeats <= 0:
        parser.error("--repeats must be positive")
    if args.time_points <= 1:
        parser.error("--time-points must be greater than 1")
    for case_name in args.cases:
        if not case_name.startswith("spin") or not case_name[4:].isdigit():
            parser.error(f"invalid case {case_name!r}; use spin<N>.")
        if not 2 <= int(case_name[4:]) <= 14:
            parser.error("spin<N> requires 2 <= N <= 14.")
    return args


if __name__ == "__main__":
    main()