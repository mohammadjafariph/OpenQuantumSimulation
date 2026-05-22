"""Restartable sweep over a qubit decay rate."""

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
    """Sweep spontaneous-emission rates and write restartable outputs."""
    target = example_output_dir("parameter_sweep", output_dir)
    times = np.linspace(0.0, 0.4 if fast else 4.0, 5 if fast else 101)
    gamma_values = [0.1, 0.3] if fast else [0.05, 0.1, 0.2, 0.4]

    sweep = oqs.ParameterSweep(
        base_system={"model": "qubit_decay", "time_points": len(times)},
        params={"gamma": gamma_values},
    )
    run = sweep.run(
        lambda point: _run_decay_point(point, times),
        output_dir=target,
        summarize=_summarize_decay_point,
        restart=True,
    )

    final_values = [float(row["final_population"]) for row in run.summary]
    if save_plots:
        plt = configure_matplotlib()
        fig, ax = plt.subplots(figsize=(5.4, 3.4))
        ax.plot(gamma_values, final_values, marker="o")
        ax.set_xlabel("decay rate gamma")
        ax.set_ylabel("final excited population")
        fig.tight_layout()
        fig.savefig(target / "parameter_sweep.png", dpi=160)
        plt.close(fig)

    return {
        "example": "parameter_sweep",
        "points": len(run.summary),
        "min_final_population": float(min(final_values)),
        "max_final_population": float(max(final_values)),
        "manifest": str(run.manifest_path),
        "summary_csv": str(run.summary_csv_path),
        "output_dir": str(target),
    }


def _run_decay_point(point: oqs.SweepPoint, times: np.ndarray) -> oqs.Result:
    gamma = float(point.params["gamma"])
    atom = oqs.SpinSpace(0.5, label="atom")
    excited = oqs.basis(atom, "up")
    H = 0.0 * oqs.sigmaz(atom)
    collapse = np.sqrt(gamma) * oqs.sigmam(atom)
    projector = oqs.Operator(oqs.ket2dm(excited), atom, "P_excited")
    return oqs.mesolve(
        H,
        oqs.ket2dm(excited),
        times,
        c_ops=[collapse],
        e_ops=[projector],
        options=oqs.Options(rtol=1e-8, atol=1e-10),
    )


def _summarize_decay_point(
    point: oqs.SweepPoint,
    result: object,
) -> dict[str, float | int | str]:
    if not isinstance(result, oqs.Result):
        msg = "parameter-sweep example expected an OpenQuantumSim Result."
        raise TypeError(msg)
    population = result.expect[0].real
    expected = np.exp(-float(point.params["gamma"]) * result.times)
    return {
        "id": point.id,
        "gamma": float(point.params["gamma"]),
        "final_population": float(population[-1]),
        "max_abs_error": float(np.max(np.abs(population - expected))),
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

