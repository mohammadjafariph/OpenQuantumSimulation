"""Time-dependent resonant drive of a dissipative two-level system."""

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
    """Run a qubit with a smooth pulsed drive and weak spontaneous emission."""
    atom = oqs.SpinSpace(0.5, label="atom")
    ground = oqs.basis(atom, "down")
    excited = oqs.basis(atom, "up")

    gamma = 0.04
    omega = 1.2
    t_final = 1.0 if fast else 10.0
    time_points = 7 if fast else 251
    times = np.linspace(0.0, t_final, time_points)

    envelope = oqs.InterpolatedCoefficient(
        [0.0, 0.25 * t_final, 0.75 * t_final, t_final],
        [0.0, 1.0, 1.0, 0.0],
    )
    H = oqs.time_dependent_hamiltonian(
        0.0 * oqs.sigmaz(atom),
        [(0.5 * omega * oqs.sigmax(atom), envelope)],
    )
    rho0 = oqs.ket2dm(ground)
    collapse = np.sqrt(gamma) * oqs.sigmam(atom)
    excited_projector = oqs.Operator(oqs.ket2dm(excited), atom, "P_excited")

    result = oqs.mesolve(
        H,
        rho0,
        times,
        c_ops=[collapse],
        e_ops=[excited_projector, oqs.sigmax(atom), oqs.sigmaz(atom)],
        options=oqs.Options(rtol=1e-8, atol=1e-10),
    )

    target = example_output_dir("driven_qubit", output_dir)
    result.save_hdf5(target / "driven_qubit.h5")
    if save_plots:
        plt = configure_matplotlib()
        fig, ax = plt.subplots(figsize=(6.0, 3.6))
        ax.plot(times, result.expect[0].real, label="P(excited)")
        ax.plot(times, result.expect[1].real, label="<sigma_x>")
        ax.plot(times, result.expect[2].real, label="<sigma_z>")
        ax.set_xlabel("time")
        ax.set_ylabel("expectation")
        ax.legend()
        fig.tight_layout()
        fig.savefig(target / "driven_qubit.png", dpi=160)
        plt.close(fig)

    return {
        "example": "driven_qubit",
        "time_points": len(times),
        "final_excited_population": float(result.expect[0].real[-1]),
        "peak_excited_population": float(np.max(result.expect[0].real)),
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

