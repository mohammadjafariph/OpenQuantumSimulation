"""Entanglement dynamics and emission spectrum of two coupled qubits."""

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
    """Run coupled-qubit entanglement dynamics and an emission spectrum."""
    qubit_a = oqs.SpinSpace(0.5, label="a")
    qubit_b = oqs.SpinSpace(0.5, label="b")
    qubits = qubit_a * qubit_b
    rho0 = oqs.ket2dm(oqs.basis(qubits, 1))  # |up, down>: one excitation

    exchange = 1.5
    # Swap excitation between the qubits: J/2 (|up,down><down,up| + h.c.)
    swap = oqs.tensor(oqs.sigmam(qubit_a), oqs.sigmap(qubit_b)) + oqs.tensor(
        oqs.sigmap(qubit_a), oqs.sigmam(qubit_b)
    )
    H = (exchange / 2.0) * swap
    dephasing_rate = 0.15
    c_ops = [
        np.sqrt(dephasing_rate) * oqs.tensor(oqs.sigmaz(qubit_a), oqs.eye(qubit_b)),
        np.sqrt(dephasing_rate) * oqs.tensor(oqs.eye(qubit_a), oqs.sigmaz(qubit_b)),
    ]

    if fast:
        t_final, time_points = 0.4, 5
    else:
        t_final, time_points = 2.0, 81
    times = np.linspace(0.0, t_final, time_points)
    # The spectrum window stays long enough to resolve the exchange
    # splitting even in fast mode (the 4x4 problem is cheap).
    taus = np.linspace(0.0, 20.0, 401)

    negativity_obs = oqs.negativity_observable((2, 2), 0, 1)
    result = oqs.mesolve(
        H,
        rho0,
        times,
        c_ops=c_ops,
        state_observables=negativity_obs,
        options=oqs.Options(rtol=1e-8, atol=1e-10, save_states=True),
    )
    negativity_values = result.state_observables["negativity"].real

    # Emission spectrum of qubit "a": <sigma+(tau) sigma-(0)>.
    a_op = oqs.tensor(oqs.sigmap(qubit_a), oqs.eye(qubit_b))
    wlist, spectrum = oqs.spectrum_2op_1t(
        H,
        rho0,
        taus,
        a_op,
        oqs.tensor(oqs.sigmam(qubit_a), oqs.eye(qubit_b)),
        c_ops=c_ops,
        options=oqs.Options(rtol=1e-8, atol=1e-10),
    )
    peak_frequency = float(wlist[np.argmax(np.abs(spectrum))])

    target = example_output_dir("entanglement_spectra", output_dir)
    if save_plots:
        plt = configure_matplotlib()
        fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
        axes[0].plot(times, negativity_values)
        axes[0].set_xlabel("time")
        axes[0].set_ylabel("negativity")
        axes[1].plot(wlist, np.abs(spectrum))
        axes[1].set_xlabel("angular frequency")
        axes[1].set_ylabel("|S(w)|")
        fig.tight_layout()
        fig.savefig(target / "entanglement_spectra.png", dpi=160)
        plt.close(fig)

    return {
        "example": "entanglement_spectra",
        "max_negativity": float(np.max(negativity_values)),
        "peak_frequency": peak_frequency,
        "exchange_splitting": exchange,
        "time_points": len(times),
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