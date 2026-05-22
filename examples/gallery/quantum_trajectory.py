"""Monte Carlo wave-function simulation of finite trajectory noise."""

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
    """Estimate spontaneous emission from a finite trajectory ensemble."""
    atom = oqs.SpinSpace(0.5, label="atom")
    excited = oqs.basis(atom, "up")

    gamma = 0.3
    t_final = 0.4 if fast else 6.0
    time_points = 5 if fast else 151
    n_traj = 12 if fast else 500
    times = np.linspace(0.0, t_final, time_points)

    H = 0.0 * oqs.sigmaz(atom)
    collapse = np.sqrt(gamma) * oqs.sigmam(atom)
    projector = oqs.Operator(oqs.ket2dm(excited), atom, "P_excited")
    result = oqs.mcsolve(
        H,
        excited,
        times,
        c_ops=[collapse],
        e_ops=[projector],
        n_traj=n_traj,
        options=oqs.Options(seed=2026, max_step=0.02, n_jobs=1),
    )

    population = result.expect[0].real
    stderr = result.expect_stderr[0]
    expected = np.exp(-gamma * times)
    mean_abs_error = float(np.mean(np.abs(population - expected)))

    target = example_output_dir("quantum_trajectory", output_dir)
    result.save_hdf5(target / "quantum_trajectory.h5")
    if save_plots:
        plt = configure_matplotlib()
        fig, ax = plt.subplots(figsize=(6.0, 3.6))
        ax.plot(times, population, label=f"{n_traj} trajectories")
        ax.fill_between(
            times,
            population - 2.0 * stderr,
            population + 2.0 * stderr,
            alpha=0.25,
            label="2 standard errors",
        )
        ax.plot(times, expected, "--", label="analytic")
        ax.set_xlabel("time")
        ax.set_ylabel("excited population")
        ax.legend()
        fig.tight_layout()
        fig.savefig(target / "quantum_trajectory.png", dpi=160)
        plt.close(fig)

    return {
        "example": "quantum_trajectory",
        "time_points": len(times),
        "n_traj": n_traj,
        "mean_abs_error": mean_abs_error,
        "final_standard_error": float(stderr[-1]),
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

