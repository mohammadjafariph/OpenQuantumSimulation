"""Observable utilities for states and density matrices."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from .operators import Operator

Array = NDArray[np.complex128]
StateObservable = Callable[[Array], Any]


def expect(operator: Operator, state: Array) -> complex:
    """Expectation value of an operator for a ket or density matrix."""
    array = np.asarray(state, dtype=np.complex128)
    if array.ndim == 1:
        return complex(np.vdot(array, operator.data @ array))
    if array.ndim == 2:
        return complex(np.trace(operator.data @ array))
    msg = "state must be a ket vector or density matrix."
    raise ValueError(msg)


def von_neumann_entropy(rho: Array, *, base: float = 2.0) -> float:
    """Von Neumann entropy `-Tr(rho log rho)`."""
    evals = _density_eigenvalues(rho, normalize=True)
    nz = evals[evals > 1e-15]
    if len(nz) == 0:
        return 0.0
    return float(-np.sum(nz * np.log(nz)) / np.log(base))


def renyi_entropy(state: Array, alpha: float = 2.0, *, base: float = 2.0) -> float:
    """Renyi entropy ``S_alpha = log(Tr(rho^alpha))/(1-alpha)``."""
    if alpha < 0:
        msg = "alpha must be non-negative."
        raise ValueError(msg)
    if np.isclose(alpha, 1.0):
        return von_neumann_entropy(state, base=base)
    evals = _density_eigenvalues(state, normalize=True)
    positive = evals[evals > 1e-15]
    if np.isclose(alpha, 0.0):
        rank = max(len(positive), 1)
        return float(np.log(rank) / np.log(base))
    if np.isinf(alpha):
        return float(-np.log(np.max(evals)) / np.log(base))
    moment = np.sum(positive**alpha)
    return float(np.log(moment) / ((1.0 - alpha) * np.log(base)))


def purity(state: Array, *, normalize: bool = False) -> float:
    """Return ``Tr(rho^2)`` for a ket or density matrix."""
    matrix = _density_matrix(state)
    if normalize:
        matrix = _normalize_density_matrix(matrix)
    return float(np.real(np.trace(matrix @ matrix)))


def linear_entropy(state: Array, *, normalized: bool = False) -> float:
    """Linear entropy ``1 - Tr(rho^2)``.

    Set ``normalized=True`` to scale by ``d/(d-1)`` for a ``d``-dimensional
    state, giving 1 for a maximally mixed state.
    """
    matrix = _normalize_density_matrix(_density_matrix(state))
    value = max(0.0, 1.0 - purity(matrix))
    if normalized:
        dim = matrix.shape[0]
        if dim <= 1:
            return 0.0
        value *= dim / (dim - 1)
    return float(value)


def participation_ratio(state: Array) -> float:
    """Effective number of populated eigenstates, ``1 / Tr(rho^2)``."""
    value = purity(state, normalize=True)
    return float(np.inf if value <= 0 else 1.0 / value)


def fidelity(state_a: Array, state_b: Array, *, squared: bool = True) -> float:
    """Uhlmann state fidelity between two kets or density matrices.

    By default this returns the squared convention
    ``F = (Tr sqrt(sqrt(rho) sigma sqrt(rho)))^2``. For pure states this is
    ``|<psi|phi>|^2``; for ``rho`` and ``|psi>`` it is ``<psi|rho|psi>``.
    Set ``squared=False`` to return the root fidelity.
    """
    root = _root_fidelity(state_a, state_b)
    return float(root * root if squared else root)


def infidelity(state_a: Array, state_b: Array) -> float:
    """Return ``1 - fidelity(state_a, state_b)``."""
    return float(max(0.0, 1.0 - fidelity(state_a, state_b)))


def trace_norm(matrix: Array, *, hermitian: bool | None = None) -> float:
    """Trace norm ``Tr(sqrt(A†A))``."""
    array = np.asarray(matrix, dtype=np.complex128)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        msg = "matrix must be square."
        raise ValueError(msg)
    use_hermitian = is_hermitian(array) if hermitian is None else hermitian
    if use_hermitian:
        return float(np.sum(np.abs(np.linalg.eigvalsh(_hermitian_part(array)))))
    return float(np.sum(np.linalg.svd(array, compute_uv=False)))


def trace_distance(state_a: Array, state_b: Array) -> float:
    """Trace distance ``0.5 * ||rho - sigma||_1``."""
    rho = _normalize_density_matrix(_density_matrix(state_a))
    sigma = _normalize_density_matrix(_density_matrix(state_b))
    _check_same_shape(rho, sigma)
    return float(0.5 * trace_norm(rho - sigma, hermitian=True))


def hilbert_schmidt_distance(state_a: Array, state_b: Array) -> float:
    """Hilbert-Schmidt distance ``sqrt(Tr((rho-sigma)^2))``."""
    rho = _normalize_density_matrix(_density_matrix(state_a))
    sigma = _normalize_density_matrix(_density_matrix(state_b))
    _check_same_shape(rho, sigma)
    diff = rho - sigma
    return float(np.sqrt(max(0.0, np.real(np.trace(diff.conj().T @ diff)))))


def bures_angle(state_a: Array, state_b: Array) -> float:
    """Bures angle ``acos(sqrt(F))`` in radians."""
    root = np.clip(fidelity(state_a, state_b, squared=False), 0.0, 1.0)
    return float(np.arccos(root))


def bures_distance(state_a: Array, state_b: Array) -> float:
    """Bures distance ``sqrt(2 * (1 - sqrt(F)))``."""
    root = np.clip(fidelity(state_a, state_b, squared=False), 0.0, 1.0)
    return float(np.sqrt(2.0 * (1.0 - root)))


def is_hermitian(matrix: Array, *, atol: float = 1e-10) -> bool:
    """Return whether ``matrix`` is Hermitian within ``atol``."""
    array = np.asarray(matrix, dtype=np.complex128)
    return bool(
        array.ndim == 2
        and array.shape[0] == array.shape[1]
        and np.allclose(array, array.conj().T, atol=atol)
    )


def is_density_matrix(matrix: Array, *, atol: float = 1e-10) -> bool:
    """Return whether ``matrix`` is positive, Hermitian, and trace one."""
    array = np.asarray(matrix, dtype=np.complex128)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        return False
    if not is_hermitian(array, atol=atol):
        return False
    if not np.isclose(np.trace(array), 1.0, atol=atol):
        return False
    evals = np.linalg.eigvalsh(_hermitian_part(array))
    return bool(np.min(evals) >= -atol)


def normalize_state(state: Array) -> Array:
    """Normalize a ket by norm or a density matrix by trace."""
    array = np.asarray(state, dtype=np.complex128)
    if array.ndim == 1:
        norm = np.linalg.norm(array)
        if norm <= 0:
            msg = "state vector has zero norm."
            raise ValueError(msg)
        return cast(Array, array / norm)
    if array.ndim == 2 and array.shape[0] == array.shape[1]:
        return _normalize_density_matrix(array)
    msg = "state must be a ket vector or square density matrix."
    raise ValueError(msg)


def populations(state: Array) -> NDArray[np.float64]:
    """Return basis populations from a ket or density matrix."""
    matrix = _normalize_density_matrix(_density_matrix(state))
    return np.maximum(np.real(np.diag(matrix)), 0.0)


def l1_coherence(state: Array) -> float:
    """Basis-dependent l1 coherence, sum of off-diagonal magnitudes."""
    matrix = _normalize_density_matrix(_density_matrix(state))
    return float(np.sum(np.abs(matrix)) - np.sum(np.abs(np.diag(matrix))))


def bloch_vector(state: Array) -> NDArray[np.float64]:
    """Return the Bloch vector ``(<sigma_x>, <sigma_y>, <sigma_z>)``."""
    rho = _normalize_density_matrix(_density_matrix(state))
    if rho.shape != (2, 2):
        msg = "bloch_vector is defined only for a two-level state."
        raise ValueError(msg)
    sx = 2.0 * np.real(rho[0, 1])
    sy = -2.0 * np.imag(rho[0, 1])
    sz = np.real(rho[0, 0] - rho[1, 1])
    return np.array([sx, sy, sz], dtype=np.float64)


def evaluate_state_observables(
    states: Sequence[Array],
    observables: Mapping[str, object],
) -> dict[str, Array]:
    """Evaluate named scalar callbacks on each saved ket or density matrix."""
    state_arrays = [np.asarray(state, dtype=np.complex128) for state in states]
    values_by_name: dict[str, Array] = {}
    for name, callback in observables.items():
        _validate_observable_name(name)
        if not callable(callback):
            msg = f"state observable {name!r} must be callable."
            raise TypeError(msg)
        observable = cast(StateObservable, callback)
        values = np.empty(len(state_arrays), dtype=np.complex128)
        for idx, state in enumerate(state_arrays):
            value = np.asarray(observable(state))
            if value.shape != ():
                msg = f"state observable {name!r} must return a scalar."
                raise ValueError(msg)
            values[idx] = complex(value.item())
        values_by_name[name] = values
    return values_by_name


def state_metrics(
    *,
    purity: bool = False,
    entropy: bool = False,
    linear_entropy: bool = False,
    participation_ratio: bool = False,
    population_indices: Sequence[int] | None = None,
    l1_coherence: bool = False,
    bloch_vector: bool = False,
    fidelity_to: Array | None = None,
    trace_distance_to: Array | None = None,
) -> dict[str, StateObservable]:
    """Build common state-observable callbacks for solver runs.

    The returned mapping can be passed directly to ``mesolve`` or
    ``single_trajectory`` via ``state_observables=...``.
    """
    metrics: dict[str, StateObservable] = {}
    if purity:
        metrics.update(purity_observable())
    if entropy:
        metrics.update(entropy_observable())
    if linear_entropy:
        metrics.update(linear_entropy_observable())
    if participation_ratio:
        metrics.update(participation_ratio_observable())
    if population_indices is not None:
        metrics.update(population_observables(population_indices))
    if l1_coherence:
        metrics.update(l1_coherence_observable())
    if bloch_vector:
        metrics.update(bloch_observables())
    if fidelity_to is not None:
        metrics.update(fidelity_observable(fidelity_to))
    if trace_distance_to is not None:
        metrics.update(trace_distance_observable(trace_distance_to))
    return metrics


def fidelity_observable(
    reference: Array,
    *,
    name: str = "fidelity",
) -> dict[str, StateObservable]:
    """Return a named fidelity callback mapping for solver runs."""
    reference_array = np.asarray(reference, dtype=np.complex128)
    return {name: _scalar_metric(lambda state: fidelity(state, reference_array))}


def purity_observable(*, name: str = "purity") -> dict[str, StateObservable]:
    """Return a named purity callback mapping for solver runs."""
    return {name: _scalar_metric(globals()["purity"])}


def entropy_observable(*, name: str = "entropy") -> dict[str, StateObservable]:
    """Return a named von Neumann entropy callback mapping for solver runs."""
    return {name: _scalar_metric(von_neumann_entropy)}


def linear_entropy_observable(
    *,
    normalized: bool = False,
    name: str = "linear_entropy",
) -> dict[str, StateObservable]:
    """Return a named linear-entropy callback mapping for solver runs."""
    return {
        name: _scalar_metric(
            lambda state: linear_entropy(state, normalized=normalized),
        ),
    }


def participation_ratio_observable(
    *,
    name: str = "participation_ratio",
) -> dict[str, StateObservable]:
    """Return a named participation-ratio callback mapping for solver runs."""
    return {name: _scalar_metric(participation_ratio)}


def l1_coherence_observable(
    *,
    name: str = "l1_coherence",
) -> dict[str, StateObservable]:
    """Return a named l1-coherence callback mapping for solver runs."""
    return {name: _scalar_metric(l1_coherence)}


def trace_distance_observable(
    reference: Array,
    *,
    name: str = "trace_distance",
) -> dict[str, StateObservable]:
    """Return a named trace-distance callback mapping for solver runs."""
    reference_array = np.asarray(reference, dtype=np.complex128)
    return {name: _scalar_metric(lambda state: trace_distance(state, reference_array))}


def population_observable(
    index: int,
    *,
    name: str | None = None,
) -> dict[str, StateObservable]:
    """Return a named callback for one basis population."""
    index = int(index)
    if index < 0:
        msg = "population index must be non-negative."
        raise ValueError(msg)
    metric_name = name or f"population_{index}"
    _validate_observable_name(metric_name)
    return {metric_name: _population_metric(index)}


def population_observables(
    indices: Sequence[int],
    *,
    prefix: str = "population",
) -> dict[str, StateObservable]:
    """Return callbacks for selected basis populations."""
    _validate_observable_name(prefix)
    callbacks: dict[str, StateObservable] = {}
    for raw_index in indices:
        index = int(raw_index)
        if index < 0:
            msg = "population indices must be non-negative."
            raise ValueError(msg)
        callbacks[f"{prefix}_{index}"] = _population_metric(index)
    return callbacks


def bloch_observables(prefix: str = "bloch") -> dict[str, StateObservable]:
    """Return callbacks for Bloch-vector components."""
    _validate_observable_name(prefix)
    labels = ("x", "y", "z")
    return {
        f"{prefix}_{label}": _bloch_metric(idx)
        for idx, label in enumerate(labels)
    }


def partial_trace(
    state: Array,
    dims: Sequence[int],
    keep: int | Sequence[int],
) -> Array:
    """Trace out all subsystems except ``keep`` for a ket or density matrix."""
    dims_tuple = _validate_dims(dims)
    keep_tuple = _normalize_keep(keep, len(dims_tuple))
    matrix = _density_matrix(state)
    total_dim = int(np.prod(dims_tuple))
    if matrix.shape != (total_dim, total_dim):
        msg = "state dimension does not match subsystem dimensions."
        raise ValueError(msg)

    tensor = matrix.reshape((*dims_tuple, *dims_tuple))
    active_subsystems = len(dims_tuple)
    keep_set = set(keep_tuple)
    trace_out = [idx for idx in range(len(dims_tuple)) if idx not in keep_set]
    for axis in sorted(trace_out, reverse=True):
        tensor = np.trace(tensor, axis1=axis, axis2=axis + active_subsystems)
        active_subsystems -= 1

    kept_dims = tuple(dims_tuple[idx] for idx in keep_tuple)
    reduced_dim = int(np.prod(kept_dims)) if kept_dims else 1
    return cast(Array, tensor.reshape((reduced_dim, reduced_dim)))


def mutual_information(
    state: Array,
    dims: Sequence[int],
    subsystem_a: int | Sequence[int],
    subsystem_b: int | Sequence[int],
    *,
    base: float = 2.0,
) -> float:
    """Quantum mutual information ``I(A:B)`` for arbitrary subsystem groups."""
    dims_tuple = _validate_dims(dims)
    keep_a = _normalize_keep(subsystem_a, len(dims_tuple))
    keep_b = _normalize_keep(subsystem_b, len(dims_tuple))
    if set(keep_a) & set(keep_b):
        msg = "subsystem_a and subsystem_b must be disjoint."
        raise ValueError(msg)

    keep_ab = tuple(sorted((*keep_a, *keep_b)))
    rho_ab = partial_trace(state, dims_tuple, keep_ab)
    rho_a = partial_trace(state, dims_tuple, keep_a)
    rho_b = partial_trace(state, dims_tuple, keep_b)
    return (
        von_neumann_entropy(rho_a, base=base)
        + von_neumann_entropy(rho_b, base=base)
        - von_neumann_entropy(rho_ab, base=base)
    )


def bipartite_mutual_information(
    state: Array,
    dim_a: int,
    dim_b: int,
    *,
    base: float = 2.0,
) -> float:
    """Quantum mutual information for a two-part system ``A ⊗ B``."""
    return mutual_information(state, (dim_a, dim_b), 0, 1, base=base)


def partial_traces(state: Array, dim_a: int, dim_b: int) -> tuple[Array, Array]:
    """Bipartite reduced density matrices for a ket or density matrix."""
    return (
        partial_trace(state, (dim_a, dim_b), 0),
        partial_trace(state, (dim_a, dim_b), 1),
    )


def _scalar_metric(callback: Callable[[Array], object]) -> StateObservable:
    def _wrapped(state: Array) -> complex:
        value = np.asarray(callback(state))
        if value.shape != ():
            msg = "state metric callback must return a scalar."
            raise ValueError(msg)
        return complex(value.item())

    return _wrapped


def _population_metric(index: int) -> StateObservable:
    def _wrapped(state: Array) -> complex:
        return complex(float(populations(state)[index]))

    return _wrapped


def _bloch_metric(index: int) -> StateObservable:
    def _wrapped(state: Array) -> complex:
        return complex(float(bloch_vector(state)[index]))

    return _wrapped


def _density_matrix(state: Array) -> Array:
    array = np.asarray(state, dtype=np.complex128)
    if array.ndim == 1:
        vector = array.reshape(-1)
        return cast(Array, np.outer(vector, vector.conj()))
    if array.ndim == 2 and array.shape[0] == array.shape[1]:
        return array
    msg = "state must be a ket vector or square density matrix."
    raise ValueError(msg)


def _normalize_density_matrix(matrix: Array) -> Array:
    hermitian = _hermitian_part(matrix)
    total = np.trace(hermitian)
    if abs(total) <= 1e-15:
        msg = "density matrix has zero trace."
        raise ValueError(msg)
    return cast(Array, hermitian / total)


def _density_eigenvalues(state: Array, *, normalize: bool) -> NDArray[np.float64]:
    matrix = _density_matrix(state)
    if normalize:
        matrix = _normalize_density_matrix(matrix)
    evals = np.linalg.eigvalsh(_hermitian_part(matrix))
    evals = np.maximum(evals.real, 0.0)
    if normalize:
        total = evals.sum()
        if total > 0:
            evals = evals / total
    return cast(NDArray[np.float64], evals)


def _root_fidelity(state_a: Array, state_b: Array) -> float:
    array_a = np.asarray(state_a, dtype=np.complex128)
    array_b = np.asarray(state_b, dtype=np.complex128)

    if array_a.ndim == 1 and array_b.ndim == 1:
        ket_a = normalize_state(array_a)
        ket_b = normalize_state(array_b)
        _check_vector_lengths(ket_a, ket_b)
        return float(np.clip(abs(np.vdot(ket_a, ket_b)), 0.0, 1.0))

    if array_a.ndim == 1:
        ket = normalize_state(array_a)
        rho = _normalize_density_matrix(_density_matrix(array_b))
        _check_vector_matrix_shape(ket, rho)
        value = np.real(np.vdot(ket, rho @ ket))
        return float(np.sqrt(np.clip(value, 0.0, 1.0)))

    if array_b.ndim == 1:
        ket = normalize_state(array_b)
        rho = _normalize_density_matrix(_density_matrix(array_a))
        _check_vector_matrix_shape(ket, rho)
        value = np.real(np.vdot(ket, rho @ ket))
        return float(np.sqrt(np.clip(value, 0.0, 1.0)))

    rho = _normalize_density_matrix(_density_matrix(array_a))
    sigma = _normalize_density_matrix(_density_matrix(array_b))
    _check_same_shape(rho, sigma)
    sqrt_rho = _sqrt_psd(rho)
    middle = _hermitian_part(sqrt_rho @ sigma @ sqrt_rho)
    evals = np.maximum(np.linalg.eigvalsh(middle).real, 0.0)
    return float(np.clip(np.sum(np.sqrt(evals)), 0.0, 1.0))


def _sqrt_psd(matrix: Array) -> Array:
    evals, vectors = np.linalg.eigh(_hermitian_part(matrix))
    evals = np.maximum(evals.real, 0.0)
    return cast(Array, (vectors * np.sqrt(evals)) @ vectors.conj().T)


def _hermitian_part(matrix: Array) -> Array:
    array = np.asarray(matrix, dtype=np.complex128)
    return cast(Array, 0.5 * (array + array.conj().T))


def _check_same_shape(left: Array, right: Array) -> None:
    if left.shape != right.shape:
        msg = f"states have incompatible shapes: {left.shape} != {right.shape}."
        raise ValueError(msg)


def _check_vector_lengths(left: Array, right: Array) -> None:
    if left.shape != right.shape:
        msg = f"kets have incompatible lengths: {left.shape} != {right.shape}."
        raise ValueError(msg)


def _check_vector_matrix_shape(vector: Array, matrix: Array) -> None:
    if vector.shape != (matrix.shape[0],):
        msg = "ket length does not match density matrix dimension."
        raise ValueError(msg)


def _validate_dims(dims: Sequence[int]) -> tuple[int, ...]:
    dims_tuple = tuple(int(dim) for dim in dims)
    if not dims_tuple:
        msg = "dims must contain at least one subsystem dimension."
        raise ValueError(msg)
    if any(dim <= 0 for dim in dims_tuple):
        msg = "subsystem dimensions must be positive."
        raise ValueError(msg)
    return dims_tuple


def _normalize_keep(keep: int | Sequence[int], n_subsystems: int) -> tuple[int, ...]:
    raw: tuple[int, ...]
    if isinstance(keep, int):
        raw = (keep,)
    else:
        raw = tuple(int(idx) for idx in keep)
    if len(raw) != len(set(raw)):
        msg = "keep contains duplicate subsystem indices."
        raise ValueError(msg)
    if any(idx < 0 or idx >= n_subsystems for idx in raw):
        msg = "subsystem index outside dims."
        raise ValueError(msg)
    return tuple(sorted(raw))


def _validate_observable_name(name: object) -> None:
    if not isinstance(name, str):
        msg = "state observable names must be strings."
        raise TypeError(msg)
    if not name:
        msg = "state observable names must not be empty."
        raise ValueError(msg)
    if "/" in name:
        msg = "state observable names must not contain '/'."
        raise ValueError(msg)
