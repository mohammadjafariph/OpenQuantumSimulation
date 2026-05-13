"""Python solver entry points."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Number
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
from scipy import sparse as sp  # type: ignore[import-untyped]

from ._julia_bridge import JuliaBridgeUnavailable, load_backend
from .observables import StateObservable, evaluate_state_observables
from .operators import Operator
from .result import Options, Result
from .timedep import InterpolatedCoefficient, TimeDependentHamiltonian

Array = NDArray[np.complex128]
FloatArray = NDArray[np.float64]


def mesolve(
    H: Operator | TimeDependentHamiltonian,
    rho0: Array,
    tlist: Sequence[float],
    *,
    c_ops: Sequence[Operator] | None = None,
    e_ops: Sequence[Operator] | None = None,
    state_observables: Mapping[str, StateObservable] | None = None,
    options: Options | None = None,
) -> Result:
    """Solve a Lindblad master equation.

    Supports time-independent Hamiltonians and ``TimeDependentHamiltonian``
    objects with scalar coefficients.
    """
    opts = options or Options()
    times = np.asarray(tlist, dtype=np.float64)
    rho0_array = np.asarray(rho0, dtype=np.complex128)
    H_data = H.base.data if isinstance(H, TimeDependentHamiltonian) else H.data
    c_arrays = [op.data for op in c_ops or []]
    e_arrays = [op.data for op in e_ops or []]
    needs_states = bool(opts.save_states or state_observables)

    _validate_mesolve_inputs(H, rho0_array, times, c_arrays, e_arrays)

    try:
        backend = load_backend()
    except JuliaBridgeUnavailable as exc:
        msg = "mesolve requires the Julia backend; run setup_julia.py first."
        raise NotImplementedError(msg) from exc

    if isinstance(H, TimeDependentHamiltonian):
        raw = backend.mesolve_time_dependent(
            _matrix_payload(backend, H_data),
            [
                _matrix_payload(
                    backend,
                    term.operator.data,
                )
                for term in H.terms
            ],
            [_coefficient_payload(backend, term.coefficient) for term in H.terms],
            rho0_array,
            times,
            [_matrix_payload(backend, array) for array in c_arrays],
            [_matrix_payload(backend, array) for array in e_arrays],
            float(opts.rtol),
            float(opts.atol),
            needs_states,
            bool(opts.compute_entropy),
        )
    else:
        raw = backend.mesolve(
            _matrix_payload(backend, H_data),
            rho0_array,
            times,
            [_matrix_payload(backend, array) for array in c_arrays],
            [_matrix_payload(backend, array) for array in e_arrays],
            float(opts.rtol),
            float(opts.atol),
            needs_states,
            str(opts.method),
            int(opts.krylov_dim),
            bool(opts.compute_entropy),
        )

    raw_times = np.asarray(_field(raw, "times"), dtype=np.float64)
    raw_expect = np.asarray(_field(raw, "expect"), dtype=np.complex128)
    raw_entropy = (
        np.asarray(_field(raw, "entropy"), dtype=np.float64)
        if opts.compute_entropy
        else None
    )
    raw_states = np.asarray(_field(raw, "states"), dtype=np.complex128)
    stats = _to_python_dict(_field(raw, "solver_stats"))

    expects = [raw_expect[idx, :].copy() for idx in range(raw_expect.shape[0])]
    state_series = _density_states(raw_states, len(raw_times)) if needs_states else []
    states = state_series if opts.save_states else None
    observed = (
        evaluate_state_observables(state_series, state_observables)
        if state_observables
        else {}
    )
    if observed:
        stats["state_observables"] = list(observed)

    return Result(
        times=raw_times,
        states=states,
        expect=expects,
        expect_std=[],
        expect_stderr=[],
        state_observables=observed,
        entropy=raw_entropy,
        solver_stats=stats,
    )


def mcsolve(
    H: Operator,
    psi0: Array,
    tlist: Sequence[float],
    *,
    c_ops: Sequence[Operator] | None = None,
    e_ops: Sequence[Operator] | None = None,
    n_traj: int | None = None,
    state_observables: Mapping[str, StateObservable] | None = None,
    options: Options | None = None,
) -> Result:
    """Run Monte Carlo wave-function trajectories."""
    if state_observables:
        msg = (
            "state_observables are currently supported by mesolve and "
            "single_trajectory; use e_ops for mcsolve ensemble expectations."
        )
        raise NotImplementedError(msg)

    opts = options or Options()
    times = np.asarray(tlist, dtype=np.float64)
    psi0_array = np.asarray(psi0, dtype=np.complex128).reshape(-1)
    c_arrays = [op.data for op in c_ops or []]
    e_arrays = [op.data for op in e_ops or []]
    trajectory_count = int(n_traj if n_traj is not None else opts.n_traj)
    seed = 0 if opts.seed is None else int(opts.seed)

    _validate_mcsolve_inputs(
        H,
        psi0_array,
        times,
        c_arrays,
        e_arrays,
        trajectory_count,
        opts.max_step,
        opts.n_jobs,
        opts.checkpoint_every,
    )

    try:
        backend = load_backend()
    except JuliaBridgeUnavailable as exc:
        msg = "mcsolve requires the Julia backend; run setup_julia.py first."
        raise NotImplementedError(msg) from exc

    raw = backend.mcsolve(
        _matrix_payload(backend, H.data),
        psi0_array,
        times,
        [_matrix_payload(backend, array) for array in c_arrays],
        [_matrix_payload(backend, array) for array in e_arrays],
        trajectory_count,
        seed,
        float(opts.max_step),
        int(opts.n_jobs),
        "" if opts.checkpoint_file is None else str(opts.checkpoint_file),
        int(opts.checkpoint_every),
        bool(opts.progress),
    )

    raw_times = np.asarray(_field(raw, "times"), dtype=np.float64)
    raw_expect = np.asarray(_field(raw, "expect"), dtype=np.complex128)
    raw_expect_std = np.asarray(_field(raw, "expect_std"), dtype=np.float64)
    raw_expect_stderr = np.asarray(_field(raw, "expect_stderr"), dtype=np.float64)
    raw_entropy = np.asarray(_field(raw, "entropy"), dtype=np.float64)
    stats = _to_python_dict(_field(raw, "solver_stats"))
    expects = [raw_expect[idx, :].copy() for idx in range(raw_expect.shape[0])]
    expect_std = [
        raw_expect_std[idx, :].copy() for idx in range(raw_expect_std.shape[0])
    ]
    expect_stderr = [
        raw_expect_stderr[idx, :].copy() for idx in range(raw_expect_stderr.shape[0])
    ]

    return Result(
        times=raw_times,
        states=None,
        expect=expects,
        expect_std=expect_std,
        expect_stderr=expect_stderr,
        entropy=raw_entropy,
        solver_stats=stats,
    )


def single_trajectory(
    H: Operator,
    psi0: Array,
    tlist: Sequence[float],
    *,
    c_ops: Sequence[Operator] | None = None,
    e_ops: Sequence[Operator] | None = None,
    state_observables: Mapping[str, StateObservable] | None = None,
    options: Options | None = None,
) -> Result:
    """Run one Monte Carlo wave-function trajectory.

    Set ``Options(save_states=True)`` to return the ket at each requested time.
    """
    opts = options or Options()
    times = np.asarray(tlist, dtype=np.float64)
    psi0_array = np.asarray(psi0, dtype=np.complex128).reshape(-1)
    c_arrays = [op.data for op in c_ops or []]
    e_arrays = [op.data for op in e_ops or []]
    seed = 0 if opts.seed is None else int(opts.seed)
    needs_states = bool(opts.save_states or state_observables)

    _validate_mcsolve_inputs(
        H,
        psi0_array,
        times,
        c_arrays,
        e_arrays,
        1,
        opts.max_step,
        1,
        1,
    )

    try:
        backend = load_backend()
    except JuliaBridgeUnavailable as exc:
        msg = "single_trajectory requires the Julia backend; run setup_julia.py first."
        raise NotImplementedError(msg) from exc

    raw = backend.single_trajectory(
        _matrix_payload(backend, H.data),
        psi0_array,
        times,
        [_matrix_payload(backend, array) for array in c_arrays],
        [_matrix_payload(backend, array) for array in e_arrays],
        seed,
        float(opts.max_step),
        needs_states,
    )

    raw_times = np.asarray(_field(raw, "times"), dtype=np.float64)
    raw_expect = np.asarray(_field(raw, "expect"), dtype=np.complex128)
    raw_states = np.asarray(_field(raw, "states"), dtype=np.complex128)

    expects = [raw_expect[idx, :].copy() for idx in range(raw_expect.shape[0])]
    state_series = _ket_states(raw_states, len(raw_times)) if needs_states else []
    states = state_series if opts.save_states else None
    observed = (
        evaluate_state_observables(state_series, state_observables)
        if state_observables
        else {}
    )

    return Result(
        times=raw_times,
        states=states,
        expect=expects,
        expect_std=[],
        expect_stderr=[],
        state_observables=observed,
        entropy=np.zeros_like(raw_times),
        solver_stats={
            "nsteps": max(len(raw_times) - 1, 0),
            "retcode": "Success",
            "n_traj": 1,
            "max_step": float(opts.max_step),
            "seed": seed,
            "save_states": bool(opts.save_states),
            "state_observables": list(observed),
        },
    )


def steadystate(
    H: Operator,
    c_ops: Sequence[Operator],
    *,
    method: str = "iterative-gmres",
    options: Options | None = None,
) -> Array:
    """Compute a Lindblad steady state."""
    opts = options or Options()
    c_arrays = [op.data for op in c_ops]
    if H.shape[0] != H.shape[1]:
        msg = "H must be square."
        raise ValueError(msg)
    for op in c_arrays:
        if op.shape != H.shape:
            msg = "collapse operators must have the same shape as H."
            raise ValueError(msg)
    try:
        backend = load_backend()
    except JuliaBridgeUnavailable as exc:
        msg = "steadystate requires the Julia backend; run setup_julia.py first."
        raise NotImplementedError(msg) from exc
    raw = backend.steadystate(
        _matrix_payload(backend, H.data),
        [_matrix_payload(backend, array) for array in c_arrays],
        method=str(method),
        rtol=float(opts.rtol),
        krylov_dim=int(opts.krylov_dim),
    )
    return np.asarray(raw, dtype=np.complex128)


def _validate_mesolve_inputs(
    H: Operator | TimeDependentHamiltonian,
    rho0: Array,
    times: FloatArray,
    c_ops: Sequence[Any],
    e_ops: Sequence[Any],
) -> None:
    if H.shape[0] != H.shape[1]:
        msg = "H must be square."
        raise ValueError(msg)
    if rho0.shape != H.shape:
        msg = "rho0 must have the same shape as H."
        raise ValueError(msg)
    if times.ndim != 1 or len(times) == 0:
        msg = "tlist must be a non-empty one-dimensional sequence."
        raise ValueError(msg)
    if np.any(np.diff(times) < 0):
        msg = "tlist must be sorted in ascending order."
        raise ValueError(msg)
    for op in [*c_ops, *e_ops]:
        if op.shape != H.shape:
            msg = "collapse and expectation operators must have the same shape as H."
            raise ValueError(msg)


def _coefficient_payload(backend: Any, coefficient: object) -> object:
    if isinstance(coefficient, InterpolatedCoefficient):
        return backend.InterpolatedCoefficient(coefficient.times, coefficient.values)
    if isinstance(coefficient, Number):
        return complex(cast(Any, coefficient))
    if callable(coefficient):
        return coefficient
    msg = f"unsupported coefficient type: {type(coefficient).__name__}"
    raise TypeError(msg)


def _matrix_payload(backend: Any, matrix_like: object) -> object:
    """Return a Julia-friendly dense or sparse matrix payload."""
    if sp.issparse(matrix_like):
        matrix = sp.csc_matrix(matrix_like, dtype=np.complex128)
        matrix.eliminate_zeros()
        return _sparse_matrix_payload(backend, matrix)
    array = np.asarray(matrix_like, dtype=np.complex128)
    if _should_send_sparse(array):
        matrix = sp.csc_matrix(array)
        matrix.eliminate_zeros()
        return _sparse_matrix_payload(backend, matrix)
    return array


def _sparse_matrix_payload(backend: Any, matrix: Any) -> object:
    colptr = np.asarray(matrix.indptr, dtype=np.int64) + 1
    rowval = np.asarray(matrix.indices, dtype=np.int64) + 1
    nzval = np.asarray(matrix.data, dtype=np.complex128)
    return backend.sparse_from_csc(
        int(matrix.shape[0]),
        int(matrix.shape[1]),
        colptr,
        rowval,
        nzval,
    )


def _should_send_sparse(array: Array) -> bool:
    if array.ndim != 2 or array.size < 64:
        return False
    nnz = int(np.count_nonzero(array))
    return 4 * nnz <= int(array.size)


def _validate_mcsolve_inputs(
    H: Operator,
    psi0: Array,
    times: FloatArray,
    c_ops: Sequence[Any],
    e_ops: Sequence[Any],
    n_traj: int,
    max_step: float,
    n_jobs: int,
    checkpoint_every: int,
) -> None:
    if H.shape[0] != H.shape[1]:
        msg = "H must be square."
        raise ValueError(msg)
    if psi0.shape != (H.shape[0],):
        msg = "psi0 must be a ket vector with length matching H."
        raise ValueError(msg)
    if times.ndim != 1 or len(times) == 0:
        msg = "tlist must be a non-empty one-dimensional sequence."
        raise ValueError(msg)
    if np.any(np.diff(times) < 0):
        msg = "tlist must be sorted in ascending order."
        raise ValueError(msg)
    if n_traj <= 0:
        msg = "n_traj must be positive."
        raise ValueError(msg)
    if max_step <= 0:
        msg = "max_step must be positive."
        raise ValueError(msg)
    if n_jobs != -1 and n_jobs <= 0:
        msg = "n_jobs must be -1 or positive."
        raise ValueError(msg)
    if checkpoint_every <= 0:
        msg = "checkpoint_every must be positive."
        raise ValueError(msg)
    for op in [*c_ops, *e_ops]:
        if op.shape != H.shape:
            msg = "collapse and expectation operators must have the same shape as H."
            raise ValueError(msg)


def _field(raw: object, name: str) -> object:
    try:
        return getattr(raw, name)
    except AttributeError:
        return raw[name]  # type: ignore[index]


def _density_states(raw_states: Array, n_times: int) -> list[Array]:
    if raw_states.ndim != 3 or raw_states.shape[2] != n_times:
        msg = "backend did not return density states with shape (d, d, n_times)."
        raise RuntimeError(msg)
    return [raw_states[:, :, idx].copy() for idx in range(raw_states.shape[2])]


def _ket_states(raw_states: Array, n_times: int) -> list[Array]:
    if raw_states.ndim != 2 or raw_states.shape[1] != n_times:
        msg = "backend did not return ket states with shape (d, n_times)."
        raise RuntimeError(msg)
    return [raw_states[:, idx].copy() for idx in range(raw_states.shape[1])]


def _optional_field(raw: object, name: str) -> object | None:
    try:
        return cast(object, getattr(raw, name))
    except Exception:
        try:
            return cast(object, raw[name])  # type: ignore[index]
        except Exception:
            return None


def _to_python_dict(raw: object) -> dict[str, object]:
    names = (
        "nsteps",
        "nfev",
        "wall_time",
        "retcode",
        "n_traj",
        "max_step",
        "n_jobs_requested",
        "n_workers",
        "threaded",
        "checkpoint_file",
        "checkpoint_every",
        "checkpoint_completed",
        "checkpoint_start_completed",
        "checkpoint_previous_target_n_traj",
        "progress",
        "resumed",
        "method",
        "requested_method",
        "krylov_dim",
        "compute_entropy",
        "time_dependent",
    )
    values: dict[str, object] = {}
    for name in names:
        value = _optional_field(raw, name)
        if value is not None:
            values[name] = value
    return values
