"""Run a checkpointed MCWF qubit-decay job and save the result."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import openquantumsim as oqs


def main() -> None:
    args = _parse_args()
    output = args.output
    checkpoint = args.checkpoint_file

    if output.exists() and not args.force:
        msg = f"Output already exists: {output}. Pass --force to overwrite."
        raise SystemExit(msg)

    atom = oqs.SpinSpace(0.5, label="atom")
    hamiltonian = 0.0 * oqs.sigmaz(atom)
    collapse = np.sqrt(args.gamma) * oqs.sigmam(atom)
    excited = oqs.basis(atom, "up")
    excited_projector = oqs.Operator(oqs.ket2dm(excited), atom, "P_excited")
    times = np.linspace(0.0, args.t_final, args.time_points)

    result = oqs.mcsolve(
        hamiltonian,
        excited,
        times,
        c_ops=[collapse],
        e_ops=[excited_projector],
        n_traj=args.n_traj,
        options=oqs.Options(
            seed=args.seed,
            max_step=args.max_step,
            n_jobs=args.n_jobs,
            checkpoint_file=str(checkpoint),
            checkpoint_every=args.checkpoint_every,
            progress=not args.no_progress,
        ),
    )
    result.save_hdf5(output, overwrite=True)

    final_population = float(result.expect[0].real[-1])
    stderr = float(result.expect_stderr[0][-1])
    print(f"Saved result: {output}")
    print(f"Checkpoint: {checkpoint}")
    print(f"Final excited population: {final_population:.6g} +/- {stderr:.3g}")
    print(f"Solver stats: {result.solver_stats}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run checkpointed OpenQuantumSim MCWF qubit decay.",
    )
    parser.add_argument("--output", type=Path, default=Path("runs/qubit_decay.h5"))
    parser.add_argument(
        "--checkpoint-file",
        type=Path,
        default=Path("runs/qubit_decay_checkpoint.h5"),
    )
    parser.add_argument("--n-traj", type=int, default=2000)
    parser.add_argument("--time-points", type=int, default=41)
    parser.add_argument("--t-final", type=float, default=4.0)
    parser.add_argument("--gamma", type=float, default=0.35)
    parser.add_argument("--max-step", type=float, default=0.01)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--force", action="store_true")
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
    if args.checkpoint_every <= 0:
        parser.error("--checkpoint-every must be positive")
    if args.n_jobs != -1 and args.n_jobs <= 0:
        parser.error("--n-jobs must be -1 or positive")
    return args


if __name__ == "__main__":
    main()
