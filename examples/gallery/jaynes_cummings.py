"""Damped Jaynes-Cummings dynamics in a truncated cavity."""

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
    """Run a small damped Jaynes-Cummings model and plot energy exchange."""
    cavity_dim = 3 if fast else 8
    t_final = 0.8 if fast else 30.0
    time_points = 7 if fast else 301
    times = np.linspace(0.0, t_final, time_points)

    system = oqs.jaynes_cummings_system(
        cavity_dim,
        cavity_frequency=1.0,
        atom_frequency=1.0,
        coupling=0.08,
        cavity_decay=0.02,
        atom_decay=0.01,
        initial_photon=0,
        atom_state="up",
    )
    result = oqs.mesolve(
        system.H,
        system.rho0,
        times,
        c_ops=system.c_ops,
        e_ops=system.e_ops,
        options=oqs.Options(rtol=1e-8, atol=1e-10),
    )

    target = example_output_dir("jaynes_cummings", output_dir)
    result.save_hdf5(target / "jaynes_cummings.h5")
    if save_plots:
        plt = configure_matplotlib()
        fig, ax = plt.subplots(figsize=(6.0, 3.6))
        ax.plot(times, result.expect[0].real, label="<n_cavity>")
        ax.plot(times, result.expect[1].real, label="P(atom excited)")
        ax.set_xlabel("time")
        ax.set_ylabel("expectation")
        ax.legend()
        fig.tight_layout()
        fig.savefig(target / "jaynes_cummings.png", dpi=160)
        plt.close(fig)

    return {
        "example": "jaynes_cummings",
        "time_points": len(times),
        "cavity_dim": cavity_dim,
        "max_photon_number": float(np.max(result.expect[0].real)),
        "final_atom_excited_population": float(result.expect[1].real[-1]),
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

