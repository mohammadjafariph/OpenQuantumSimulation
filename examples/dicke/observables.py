"""Dicke-example observable helpers."""

from __future__ import annotations

from collections.abc import Sequence
from math import isfinite, lgamma
from typing import cast

import numpy as np
from numpy.typing import NDArray

import openquantumsim as oqs

Array = NDArray[np.complex128]
FloatArray = NDArray[np.float64]


def precompute_dicke_reduction(n_spins: int, subsystem_size: int) -> FloatArray:
    """Combinatorial reduction matrix for symmetric Dicke ``k``-RDMs."""
    if n_spins < 0:
        msg = "n_spins must be non-negative."
        raise ValueError(msg)
    if subsystem_size < 0 or subsystem_size > n_spins:
        msg = "subsystem_size must satisfy 0 <= subsystem_size <= n_spins."
        raise ValueError(msg)

    factors = np.zeros((n_spins + 1, subsystem_size + 1), dtype=np.float64)
    log_denom = _log_binomial(n_spins, subsystem_size)
    for r in range(n_spins + 1):
        for p in range(subsystem_size + 1):
            log_value = (
                _log_binomial(r, p)
                + _log_binomial(n_spins - r, subsystem_size - p)
                - log_denom
            )
            factors[r, p] = np.exp(0.5 * log_value) if isfinite(log_value) else 0.0
    return factors


def dicke_k_rdm(rho: Array, reduction: FloatArray) -> Array:
    """Reduced density matrix inside the symmetric Dicke manifold."""
    matrix = np.asarray(rho, dtype=np.complex128)
    factors = np.asarray(reduction, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        msg = "rho must be a square density matrix."
        raise ValueError(msg)
    if factors.ndim != 2 or factors.shape[0] != matrix.shape[0]:
        msg = "reduction matrix row count must match rho dimension."
        raise ValueError(msg)

    reduced = factors.T.astype(np.complex128) @ matrix @ factors.astype(np.complex128)
    reduced = 0.5 * (reduced + reduced.conj().T)
    trace = float(np.real(np.trace(reduced)))
    return cast(Array, reduced / trace if trace > 0 else reduced)


def dicke_mutual_information(
    rho: Array,
    n_spins: int,
    *,
    subsystem_size: int | None = None,
) -> float:
    """Mutual information between ``k`` and ``N-k`` spins in a Dicke ensemble."""
    if n_spins < 1:
        msg = "n_spins must be positive."
        raise ValueError(msg)
    matrix = np.asarray(rho, dtype=np.complex128)
    if matrix.shape != (n_spins + 1, n_spins + 1):
        msg = "rho must have shape (n_spins + 1, n_spins + 1)."
        raise ValueError(msg)

    k = max(1, n_spins // 2) if subsystem_size is None else int(subsystem_size)
    if k < 0 or k > n_spins:
        msg = "subsystem_size must satisfy 0 <= subsystem_size <= n_spins."
        raise ValueError(msg)

    rho_k = dicke_k_rdm(matrix, precompute_dicke_reduction(n_spins, k))
    rho_complement = dicke_k_rdm(
        matrix,
        precompute_dicke_reduction(n_spins, n_spins - k),
    )
    return (
        oqs.von_neumann_entropy(rho_k)
        + oqs.von_neumann_entropy(rho_complement)
        - oqs.von_neumann_entropy(matrix)
    )


def two_ensemble_dicke_mutual_information(
    ket: Array,
    n_spins: int,
    *,
    subsystem_size: int | None = None,
) -> tuple[float, float]:
    """Dicke ``k``/``N-k`` mutual information for both ensembles in a pure ket."""
    dim = n_spins + 1
    rho_a, rho_b = oqs.partial_traces(ket, dim, dim)
    return (
        dicke_mutual_information(rho_a, n_spins, subsystem_size=subsystem_size),
        dicke_mutual_information(rho_b, n_spins, subsystem_size=subsystem_size),
    )


def trajectory_dicke_mutual_information(
    states: Sequence[Array],
    n_spins: int,
    *,
    subsystem_size: int | None = None,
) -> tuple[FloatArray, FloatArray]:
    """Compute two-ensemble Dicke mutual-information series from saved kets."""
    mi_a = np.empty(len(states), dtype=np.float64)
    mi_b = np.empty(len(states), dtype=np.float64)
    for idx, state in enumerate(states):
        mi_a[idx], mi_b[idx] = two_ensemble_dicke_mutual_information(
            state,
            n_spins,
            subsystem_size=subsystem_size,
        )
    return mi_a, mi_b


def _log_binomial(n: int, k: int) -> float:
    if k < 0 or k > n:
        return -float("inf")
    return lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1)
