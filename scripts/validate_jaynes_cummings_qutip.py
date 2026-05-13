"""Validate OpenQuantumSim Jaynes-Cummings dynamics against QuTiP."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

import openquantumsim as oqs


def main() -> int:
    args = _parse_args()
    try:
        import qutip  # type: ignore[import-not-found]
    except ImportError:
        print("QuTiP is not installed. Install with: python -m pip install qutip")
        return 2

    times = np.linspace(0.0, args.t_final, args.time_points)
    system = oqs.jaynes_cummings_system(
        args.cavity_dim,
        cavity_frequency=args.cavity_frequency,
        atom_frequency=args.atom_frequency,
        coupling=args.coupling,
        cavity_decay=args.cavity_decay,
        atom_decay=args.atom_decay,
    )
    oqs_result = oqs.mesolve(
        system.H,
        system.rho0,
        times,
        c_ops=system.c_ops,
        e_ops=system.e_ops,
        options=oqs.Options(method=args.method, rtol=args.rtol, atol=args.atol),
    )
    qutip_result = _run_qutip(args, qutip, times)

    oqs_photons = oqs_result.expect[0].real
    oqs_excited = oqs_result.expect[1].real
    qutip_photons = np.asarray(qutip_result.expect[0], dtype=np.float64)
    qutip_excited = np.asarray(qutip_result.expect[1], dtype=np.float64)
    photon_delta = np.max(np.abs(oqs_photons - qutip_photons))
    excited_delta = np.max(np.abs(oqs_excited - qutip_excited))
    max_delta = float(max(photon_delta, excited_delta))

    if args.output_csv is not None:
        _write_csv(
            args.output_csv,
            times,
            oqs_photons,
            qutip_photons,
            oqs_excited,
            qutip_excited,
        )

    print(f"max photon-number delta: {photon_delta:.3e}")
    print(f"max excited-population delta: {excited_delta:.3e}")
    print(f"tolerance: {args.tolerance:.3e}")
    if max_delta > args.tolerance:
        print("Jaynes-Cummings QuTiP validation: FAILED")
        return 1
    print("Jaynes-Cummings QuTiP validation: passed")
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare OpenQuantumSim Jaynes-Cummings mesolve against QuTiP.",
    )
    parser.add_argument("--cavity-dim", type=int, default=5)
    parser.add_argument("--cavity-frequency", type=float, default=1.0)
    parser.add_argument("--atom-frequency", type=float, default=1.0)
    parser.add_argument("--coupling", type=float, default=0.08)
    parser.add_argument("--cavity-decay", type=float, default=0.015)
    parser.add_argument("--atom-decay", type=float, default=0.025)
    parser.add_argument("--time-points", type=int, default=81)
    parser.add_argument("--t-final", type=float, default=8.0)
    parser.add_argument("--method", choices=["ode", "krylov"], default="ode")
    parser.add_argument("--rtol", type=float, default=1e-9)
    parser.add_argument("--atol", type=float, default=1e-11)
    parser.add_argument("--tolerance", type=float, default=5e-7)
    parser.add_argument("--output-csv", type=Path)
    args = parser.parse_args()

    if args.cavity_dim <= 0:
        parser.error("--cavity-dim must be positive")
    if args.time_points <= 1:
        parser.error("--time-points must be greater than 1")
    if args.t_final <= 0:
        parser.error("--t-final must be positive")
    if args.cavity_decay < 0 or args.atom_decay < 0:
        parser.error("decay rates must be non-negative")
    if args.rtol <= 0 or args.atol <= 0:
        parser.error("--rtol and --atol must be positive")
    return args


def _run_qutip(args: argparse.Namespace, qutip: object, times: np.ndarray) -> object:
    cavity_eye = qutip.qeye(args.cavity_dim)
    atom_eye = qutip.qeye(2)
    a = qutip.tensor(qutip.destroy(args.cavity_dim), atom_eye)
    sm = qutip.tensor(cavity_eye, qutip.sigmam())
    sz = qutip.tensor(cavity_eye, qutip.sigmaz())
    n_cavity = a.dag() * a
    excited_projector = qutip.tensor(
        cavity_eye,
        qutip.basis(2, 0) * qutip.basis(2, 0).dag(),
    )
    hamiltonian = (
        args.cavity_frequency * n_cavity
        + 0.5 * args.atom_frequency * sz
        + args.coupling * (a.dag() * sm + a * sm.dag())
    )
    psi0 = qutip.tensor(qutip.basis(args.cavity_dim, 0), qutip.basis(2, 0))
    collapse_ops = []
    if args.cavity_decay > 0:
        collapse_ops.append(np.sqrt(args.cavity_decay) * a)
    if args.atom_decay > 0:
        collapse_ops.append(np.sqrt(args.atom_decay) * sm)
    return qutip.mesolve(
        hamiltonian,
        psi0 * psi0.dag(),
        times,
        c_ops=collapse_ops,
        e_ops=[n_cavity, excited_projector],
        options={"rtol": 1e-10, "atol": 1e-12, "store_states": False},
    )


def _write_csv(
    path: Path,
    times: np.ndarray,
    oqs_photons: np.ndarray,
    qutip_photons: np.ndarray,
    oqs_excited: np.ndarray,
    qutip_excited: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "time",
                "oqs_photons",
                "qutip_photons",
                "oqs_excited",
                "qutip_excited",
            ],
        )
        for row in zip(
            times,
            oqs_photons,
            qutip_photons,
            oqs_excited,
            qutip_excited,
            strict=True,
        ):
            writer.writerow(row)


if __name__ == "__main__":
    sys.exit(main())
