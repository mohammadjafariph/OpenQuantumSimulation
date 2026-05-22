"""Deterministic spontaneous emission of a two-level system."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import openquantumsim as oqs

try:
    from ._common import (
        Summary,
        add_common_arguments,
        configure_matplotlib,
        emit_summary,
        example_output_dir,
    )
except ImportError:  # pragma: no cover - used when executed as a script
    from _common import (  # type: ignore[no-redef]
        Summary,
        add_common_arguments,
        configure_matplotlib,
        emit_summary,
        example_output_dir,
    )


def run_example(
    *,
    output_dir: Path | None = None,
    fast: bool = False,
    save_plots: bool = True,
) -> Summary:
    """Run spontaneous emission and compare against the analytic curve."""
    atom = oqs.SpinSpace(0.5, label="atom")
    excited = oqs.basis(atom, "up")

    gamma = 0.25
    t_final = 0.4 if fast else 8.0
    time_points = 5 if fast else 201
    times = np.linspace(0.0, t_final, time_points)

    H = 0.0 * oqs.sigmaz(atom)
    rho0 = oqs.ket2dm(excited)
    collapse = np.sqrt(gamma) * oqs.sigmam(atom)
    projector = oqs.Operator(oqs.ket2dm(excited), atom, "P_excited")

    result = oqs.mesolve(
        H,
        rho0,
        times,
        c_ops=[collapse],
        e_ops=[projector],
        options=oqs.Options(rtol=1e-8, atol=1e-10),
    )

    expected = np.exp(-gamma * times)
    population = result.expect[0].real
    max_error = float(np.max(np.abs(population - expected)))

    target = example_output_dir("qubit_decay", output_dir)
    result.save_hdf5(target / "qubit_decay.h5")
    if save_plots:
        plt = configure_matplotlib()
        fig, ax = plt.subplots(figsize=(6.0, 3.6))
        ax.plot(times, population, label="OpenQuantumSim")
        ax.plot(times, expected, "--", label="analytic")
        ax.set_xlabel("time")
        ax.set_ylabel("excited population")
        ax.legend()
        fig.tight_layout()
        fig.savefig(target / "qubit_decay.png", dpi=160)
        plt.close(fig)

    return {
        "example": "qubit_decay",
        "time_points": len(times),
        "max_abs_error": max_error,
        "final_population": float(population[-1]),
        "output_dir": str(target),
    }


def main() -> None:
    """Run the example from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    args = parser.parse_args()
    summary = run_example(
        output_dir=args.output_dir,
        fast=args.fast,
        save_plots=not args.no_plots,
    )
    emit_summary(summary, as_json=args.json)


if __name__ == "__main__":
    main()

