"""Reusable open-system containers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .hilbert import CompositeSpace, FockSpace, SpinSpace
from .operators import (
    Operator,
    basis,
    create,
    destroy,
    eye,
    ket2dm,
    num,
    sigmam,
    sigmap,
    sigmaz,
    tensor,
)

Array = NDArray[np.complex128]


@dataclass(frozen=True)
class MCWFSystem:
    """Container for a Monte Carlo wave-function system."""

    H: Operator
    psi0: Array
    c_ops: list[Operator]
    e_ops: list[Operator]
    hilbert: CompositeSpace
    metadata: dict[str, Any]


@dataclass(frozen=True)
class JaynesCummingsSystem:
    """Container for a truncated Jaynes-Cummings open quantum system."""

    H: Operator
    psi0: Array
    rho0: Array
    c_ops: list[Operator]
    e_ops: list[Operator]
    hilbert: CompositeSpace
    metadata: dict[str, Any]


def jaynes_cummings_system(
    cavity_dim: int,
    *,
    cavity_frequency: float = 1.0,
    atom_frequency: float = 1.0,
    coupling: float = 0.05,
    cavity_decay: float = 0.0,
    atom_decay: float = 0.0,
    initial_photon: int = 0,
    atom_state: str = "up",
) -> JaynesCummingsSystem:
    """Build a damped Jaynes-Cummings model in ``cavity ⊗ atom`` order.

    The Hamiltonian is
    ``ωc a†a + 0.5 ωa σz + g(a† σ- + a σ+)`` with optional cavity and
    atomic decay collapse operators.
    """
    if cavity_dim <= 0:
        msg = "cavity_dim must be positive."
        raise ValueError(msg)
    if initial_photon < 0 or initial_photon >= cavity_dim:
        msg = "initial_photon must satisfy 0 <= initial_photon < cavity_dim."
        raise ValueError(msg)
    if cavity_decay < 0 or atom_decay < 0:
        msg = "decay rates must be non-negative."
        raise ValueError(msg)

    cavity = FockSpace(cavity_dim, label="cavity")
    atom = SpinSpace(0.5, label="atom")
    hilbert = cavity * atom
    id_cavity = eye(cavity)
    id_atom = eye(atom)

    a = tensor(destroy(cavity), id_atom)
    adag = tensor(create(cavity), id_atom)
    n_cavity = tensor(num(cavity), id_atom)
    sz_atom = tensor(id_cavity, sigmaz(atom))
    sm_atom = tensor(id_cavity, sigmam(atom))
    sp_atom = tensor(id_cavity, sigmap(atom))

    hamiltonian = (
        cavity_frequency * n_cavity
        + 0.5 * atom_frequency * sz_atom
        + coupling * (adag * sm_atom + a * sp_atom)
    )

    psi0 = np.kron(
        basis(cavity, initial_photon),
        basis(atom, atom_state),
    ).astype(np.complex128)
    rho0 = ket2dm(psi0)

    collapse_ops: list[Operator] = []
    if cavity_decay > 0:
        collapse_ops.append(np.sqrt(cavity_decay) * a)
    if atom_decay > 0:
        collapse_ops.append(np.sqrt(atom_decay) * sm_atom)

    excited = basis(atom, "up")
    excited_projector = tensor(
        id_cavity,
        Operator(ket2dm(excited), atom, "P_excited"),
    )
    observables = [n_cavity, excited_projector]

    return JaynesCummingsSystem(
        H=hamiltonian,
        psi0=psi0,
        rho0=rho0,
        c_ops=collapse_ops,
        e_ops=observables,
        hilbert=hilbert,
        metadata={
            "cavity_dim": cavity_dim,
            "cavity_frequency": cavity_frequency,
            "atom_frequency": atom_frequency,
            "coupling": coupling,
            "cavity_decay": cavity_decay,
            "atom_decay": atom_decay,
            "initial_photon": initial_photon,
            "atom_state": atom_state,
        },
    )
