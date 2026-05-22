"""Wigner and Husimi-Q functions for a finite Fock-space state."""

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
    """Evaluate Wigner and Q functions for a coherent-state superposition."""
    space = oqs.FockSpace(12 if fast else 30, label="cavity")
    alpha = 1.2 + 0.4j
    cat = oqs.coherent(space, alpha) + oqs.coherent(space, -alpha)
    cat = cat / np.linalg.norm(cat)
    rho = oqs.ket2dm(cat)

    grid_points = 31 if fast else 151
    x, p = oqs.phase_space_grid(xlim=(-4.0, 4.0), points=grid_points)
    wigner = oqs.wigner(rho, x, p)
    q_values = oqs.q_function(rho, x, p)
    negativity = float(np.sum(np.abs(wigner) - wigner) * (x[1] - x[0]) * (p[1] - p[0]))

    target = example_output_dir("phase_space", output_dir)
    np.savez(target / "phase_space.npz", x=x, p=p, wigner=wigner, q_function=q_values)
    if save_plots:
        plt = configure_matplotlib()
        fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.6), constrained_layout=True)
        vmax = float(np.max(np.abs(wigner)))
        axes[0].imshow(
            wigner,
            origin="lower",
            extent=(float(x[0]), float(x[-1]), float(p[0]), float(p[-1])),
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
            aspect="auto",
        )
        axes[0].set_title("Wigner")
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("p")
        axes[1].imshow(
            q_values,
            origin="lower",
            extent=(float(x[0]), float(x[-1]), float(p[0]), float(p[-1])),
            cmap="viridis",
            aspect="auto",
        )
        axes[1].set_title("Husimi Q")
        axes[1].set_xlabel("x")
        fig.savefig(target / "phase_space.png", dpi=160)
        plt.close(fig)

    return {
        "example": "phase_space",
        "fock_dim": space.dim,
        "grid_points": grid_points,
        "wigner_min": float(np.min(wigner)),
        "q_max": float(np.max(q_values)),
        "wigner_negativity": negativity,
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

