"""Benchmark OpenQuantumSim mesolve against QuTiP mesolve."""

from __future__ import annotations

import argparse
import gc
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
class BenchmarkModel:
    """A model represented in both OpenQuantumSim and QuTiP."""

    name: str
    dim: int
    times: FloatArray
    oqs_H: oqs.Operator
    oqs_rho0: np.ndarray
    oqs_c_ops: list[oqs.Operator]
    oqs_e_ops: list[oqs.Operator]
    qutip_H: Any
    qutip_rho0: Any
    qutip_c_ops: list[Any]
    qutip_e_ops: list[Any]


def main() -> None:
    args = _parse_args()
    try:
        import qutip  # type: ignore[import-untyped]
    except ImportError as exc:
        msg = "QuTiP is required for this benchmark: python -m pip install qutip"
        raise SystemExit(msg) from exc

    models = _build_models(args, qutip)
    rows: list[dict[str, Any]] = []

    print("OpenQuantumSim vs QuTiP mesolve benchmark")
    print(f"OpenQuantumSim: {oqs.__version__}")
    print(f"QuTiP: {qutip.__version__}")
    print(
        "config: "
        f"repeats={args.repeats}, time_points={args.time_points}, "
        f"t_final={args.t_final}, rtol={args.rtol:g}, atol={args.atol:g}"
    )
    print()

    for model in models:
        print(f"warming {model.name} (dim={model.dim})")
        _warm_model(model, qutip, args)
        qutip_result, qutip_times = _time_qutip(model, qutip, args)
        qutip_median = statistics.median(qutip_times)
        rows.append(
            {
                "case": model.name,
                "dim": model.dim,
                "time_points": len(model.times),
                "engine": "qutip",
                "method": "mesolve",
                "median_s": qutip_median,
                "min_s": min(qutip_times),
                "speedup_vs_qutip": 1.0,
                "max_abs_delta_vs_qutip": 0.0,
            },
        )

        for method in args.oqs_methods:
            oqs_result, oqs_times = _time_oqs(model, args, method)
            median = statistics.median(oqs_times)
            rows.append(
                {
                    "case": model.name,
                    "dim": model.dim,
                    "time_points": len(model.times),
                    "engine": "openquantumsim",
                    "method": method,
                    "median_s": median,
                    "min_s": min(oqs_times),
                    "speedup_vs_qutip": qutip_median / median,
                    "max_abs_delta_vs_qutip": _max_expect_delta(
                        oqs_result.expect,
                        qutip_result.expect,
                    ),
                },
            )

    _print_table(rows)

    if args.json is not None:
        payload = {
            "config": {
                "repeats": args.repeats,
                "time_points": args.time_points,
                "t_final": args.t_final,
                "rtol": args.rtol,
                "atol": args.atol,
                "oqs_methods": args.oqs_methods,
            },
            "versions": {
                "openquantumsim": oqs.__version__,
                "qutip": qutip.__version__,
            },
            "rows": rows,
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nWrote {args.json}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark OpenQuantumSim mesolve against QuTiP mesolve.",
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--time-points", type=int, default=101)
    parser.add_argument("--t-final", type=float, default=8.0)
    parser.add_argument("--rtol", type=float, default=1e-8)
    parser.add_argument("--atol", type=float, default=1e-10)
    parser.add_argument("--krylov-dim", type=int, default=30)
    parser.add_argument(
        "--cases",
        nargs="+",
        default=["qubit", "jc5", "jc10", "jc20"],
        help="Cases to run: qubit, jc<N> such as jc10 or jc20.",
    )
    parser.add_argument(
        "--oqs-methods",
        nargs="+",
        default=["auto", "krylov", "ode"],
        choices=["auto", "krylov", "ode"],
    )
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    if args.repeats <= 0:
        parser.error("--repeats must be positive")
    if args.time_points <= 1:
        parser.error("--time-points must be greater than 1")
    if args.t_final <= 0:
        parser.error("--t-final must be positive")
    if args.rtol <= 0 or args.atol <= 0:
        parser.error("--rtol and --atol must be positive")
    if args.krylov_dim <= 0:
        parser.error("--krylov-dim must be positive")
    return args


def _build_models(args: argparse.Namespace, qutip: Any) -> list[BenchmarkModel]:
    models: list[BenchmarkModel] = []
    for case in args.cases:
        if case == "qubit":
            models.append(_build_qubit(args, qutip))
        elif case.startswith("jc"):
            try:
                cavity_dim = int(case[2:])
            except ValueError as exc:
                msg = f"Invalid Jaynes-Cummings case {case!r}; use jc<N>."
                raise ValueError(msg) from exc
            models.append(_build_jaynes_cummings(args, qutip, cavity_dim))
        else:
            msg = f"Unknown benchmark case {case!r}."
            raise ValueError(msg)
    return models


def _build_qubit(args: argparse.Namespace, qutip: Any) -> BenchmarkModel:
    gamma = 0.35
    times = np.linspace(0.0, args.t_final, args.time_points)

    atom = oqs.SpinSpace(0.5, label="atom")
    oqs_H = 0.0 * oqs.sigmaz(atom)
    excited = oqs.basis(atom, "up")
    oqs_rho0 = oqs.ket2dm(excited)
    oqs_collapse = np.sqrt(gamma) * oqs.sigmam(atom)
    oqs_projector = oqs.Operator(oqs.ket2dm(excited), atom, "P_excited")

    sm = qutip.sigmam()
    qutip_H = 0.0 * qutip.sigmaz()
    qutip_rho0 = qutip.basis(2, 0) * qutip.basis(2, 0).dag()
    qutip_collapse = np.sqrt(gamma) * sm
    qutip_projector = qutip.basis(2, 0) * qutip.basis(2, 0).dag()

    return BenchmarkModel(
        name="qubit_decay",
        dim=2,
        times=times,
        oqs_H=oqs_H,
        oqs_rho0=oqs_rho0,
        oqs_c_ops=[oqs_collapse],
        oqs_e_ops=[oqs_projector],
        qutip_H=qutip_H,
        qutip_rho0=qutip_rho0,
        qutip_c_ops=[qutip_collapse],
        qutip_e_ops=[qutip_projector],
    )


def _build_jaynes_cummings(
    args: argparse.Namespace,
    qutip: Any,
    cavity_dim: int,
) -> BenchmarkModel:
    cavity_frequency = 1.0
    atom_frequency = 1.0
    coupling = 0.08
    cavity_decay = 0.015
    atom_decay = 0.025
    times = np.linspace(0.0, args.t_final, args.time_points)
    system = oqs.jaynes_cummings_system(
        cavity_dim,
        cavity_frequency=cavity_frequency,
        atom_frequency=atom_frequency,
        coupling=coupling,
        cavity_decay=cavity_decay,
        atom_decay=atom_decay,
    )

    cavity_eye = qutip.qeye(cavity_dim)
    atom_eye = qutip.qeye(2)
    a = qutip.tensor(qutip.destroy(cavity_dim), atom_eye)
    sm = qutip.tensor(cavity_eye, qutip.sigmam())
    sz = qutip.tensor(cavity_eye, qutip.sigmaz())
    n_cavity = a.dag() * a
    excited_projector = qutip.tensor(
        cavity_eye,
        qutip.basis(2, 0) * qutip.basis(2, 0).dag(),
    )
    hamiltonian = (
        cavity_frequency * n_cavity
        + 0.5 * atom_frequency * sz
        + coupling * (a.dag() * sm + a * sm.dag())
    )
    psi0 = qutip.tensor(qutip.basis(cavity_dim, 0), qutip.basis(2, 0))
    collapse_ops = [np.sqrt(cavity_decay) * a, np.sqrt(atom_decay) * sm]

    return BenchmarkModel(
        name=f"jaynes_cummings_{cavity_dim}",
        dim=2 * cavity_dim,
        times=times,
        oqs_H=system.H,
        oqs_rho0=system.rho0,
        oqs_c_ops=system.c_ops,
        oqs_e_ops=system.e_ops,
        qutip_H=hamiltonian,
        qutip_rho0=psi0 * psi0.dag(),
        qutip_c_ops=collapse_ops,
        qutip_e_ops=[n_cavity, excited_projector],
    )


def _warm_model(model: BenchmarkModel, qutip: Any, args: argparse.Namespace) -> None:
    warm_times = model.times[: min(5, len(model.times))]
    for method in args.oqs_methods:
        oqs.mesolve(
            model.oqs_H,
            model.oqs_rho0,
            warm_times.tolist(),
            c_ops=model.oqs_c_ops,
            e_ops=model.oqs_e_ops,
            options=oqs.Options(
                method=method,
                rtol=args.rtol,
                atol=args.atol,
                krylov_dim=args.krylov_dim,
            ),
        )
    qutip.mesolve(
        model.qutip_H,
        model.qutip_rho0,
        warm_times,
        c_ops=model.qutip_c_ops,
        e_ops=model.qutip_e_ops,
        options={"rtol": args.rtol, "atol": args.atol, "store_states": False},
    )


def _time_oqs(
    model: BenchmarkModel,
    args: argparse.Namespace,
    method: str,
) -> tuple[oqs.Result, list[float]]:
    elapsed: list[float] = []
    latest: oqs.Result | None = None
    for _ in range(args.repeats):
        gc.collect()
        started = time.perf_counter()
        latest = oqs.mesolve(
            model.oqs_H,
            model.oqs_rho0,
            model.times.tolist(),
            c_ops=model.oqs_c_ops,
            e_ops=model.oqs_e_ops,
            options=oqs.Options(
                method=method,
                rtol=args.rtol,
                atol=args.atol,
                krylov_dim=args.krylov_dim,
            ),
        )
        elapsed.append(time.perf_counter() - started)
    if latest is None:
        msg = "OpenQuantumSim benchmark produced no result."
        raise RuntimeError(msg)
    return latest, elapsed


def _time_qutip(
    model: BenchmarkModel,
    qutip: Any,
    args: argparse.Namespace,
) -> tuple[Any, list[float]]:
    elapsed: list[float] = []
    latest: Any | None = None
    for _ in range(args.repeats):
        gc.collect()
        started = time.perf_counter()
        latest = qutip.mesolve(
            model.qutip_H,
            model.qutip_rho0,
            model.times,
            c_ops=model.qutip_c_ops,
            e_ops=model.qutip_e_ops,
            options={"rtol": args.rtol, "atol": args.atol, "store_states": False},
        )
        elapsed.append(time.perf_counter() - started)
    if latest is None:
        msg = "QuTiP benchmark produced no result."
        raise RuntimeError(msg)
    return latest, elapsed


def _max_expect_delta(oqs_expect: list[np.ndarray], qutip_expect: list[Any]) -> float:
    delta = 0.0
    for oqs_values, qutip_values in zip(oqs_expect, qutip_expect, strict=True):
        qutip_array = np.asarray(qutip_values, dtype=np.complex128)
        delta = max(delta, float(np.max(np.abs(oqs_values - qutip_array))))
    return delta


def _print_table(rows: list[dict[str, Any]]) -> None:
    print()
    print(
        f"{'case':<22} {'dim':>5} {'engine':<15} {'method':<8} "
        f"{'median_s':>10} {'min_s':>10} {'vs_qutip':>10} {'max_delta':>11}"
    )
    print("-" * 100)
    for row in rows:
        print(
            f"{row['case']:<22} "
            f"{int(row['dim']):>5} "
            f"{row['engine']:<15} "
            f"{row['method']:<8} "
            f"{float(row['median_s']):>10.5f} "
            f"{float(row['min_s']):>10.5f} "
            f"{float(row['speedup_vs_qutip']):>9.2f}x "
            f"{float(row['max_abs_delta_vs_qutip']):>11.3e}"
        )


if __name__ == "__main__":
    main()
