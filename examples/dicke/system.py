"""Two-ensemble Dicke model used by the roadmap example."""

from __future__ import annotations

import numpy as np

import openquantumsim as oqs


def two_ensemble_dicke_system(
    N: int,
    kappa: float,
    *,
    wx: tuple[float, float] = (1.2, 0.9),
    local_decay: float = 2.0,
) -> oqs.MCWFSystem:
    """Build the two symmetric Dicke-ensemble MCWF example model."""
    if N <= 0:
        msg = "N must be positive."
        raise ValueError(msg)
    if kappa < 0:
        msg = "kappa must be non-negative."
        raise ValueError(msg)
    if local_decay < 0:
        msg = "local_decay must be non-negative."
        raise ValueError(msg)

    space_a = oqs.DickeSpace(N, label="A")
    space_b = oqs.DickeSpace(N, label="B")
    hilbert = space_a * space_b
    id_a = oqs.eye(space_a)
    id_b = oqs.eye(space_b)
    jm_a = oqs.collective_lowering(space_a)
    jm_b = oqs.collective_lowering(space_b)
    jx_a = oqs.collective_x(space_a)
    jx_b = oqs.collective_x(space_b)

    jm_a_full = oqs.tensor(jm_a, id_b)
    jm_b_full = oqs.tensor(id_a, jm_b)
    hamiltonian = wx[0] * oqs.tensor(jx_a, id_b) + wx[1] * oqs.tensor(id_a, jx_b)
    collective_lowering = jm_a_full + jm_b_full
    c_ops = [
        np.sqrt(kappa / N) * collective_lowering,
        np.sqrt(local_decay / N) * jm_a_full,
        np.sqrt(local_decay / N) * jm_b_full,
    ]

    psi0 = np.kron(
        oqs.dicke_state(space_a, N),
        oqs.dicke_state(space_b, N),
    ).astype(np.complex128)
    excitation_a = oqs.collective_excitation(space_a)
    excitation_b = oqs.collective_excitation(space_b)
    excitation_fraction = (
        oqs.tensor(excitation_a, id_b) + oqs.tensor(id_a, excitation_b)
    ) * (1 / (2 * N))
    excitation_fraction = oqs.Operator(
        excitation_fraction.data,
        hilbert,
        "excitation_fraction",
    )

    return oqs.MCWFSystem(
        H=hamiltonian,
        psi0=psi0,
        c_ops=c_ops,
        e_ops=[excitation_fraction],
        hilbert=hilbert,
        metadata={
            "system": "two_ensemble_dicke",
            "N": N,
            "kappa": kappa,
            "wx": wx,
            "local_decay": local_decay,
        },
    )
