"""Common result and option dataclasses."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import h5py  # type: ignore[import-untyped]
import numpy as np
from numpy.typing import NDArray

from .hilbert import HilbertSpace
from .operators import Operator

Array = NDArray[np.complex128]
FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class QuantumSystem:
    """Container for a Hamiltonian, collapse operators, and Hilbert space."""

    H: Operator
    c_ops: list[Operator]
    hilbert: HilbertSpace


@dataclass(frozen=True)
class Options:
    """Solver options shared by Python wrappers."""

    method: str = "auto"
    ode_solver: str = "auto"
    rtol: float = 1e-8
    atol: float = 1e-10
    krylov_dim: int = 30
    n_traj: int = 500
    n_jobs: int = -1
    backend: str = "cpu"
    progress: bool = False
    save_states: bool = False
    compute_entropy: bool = False
    seed: int | None = None
    max_step: float = 1e-2
    checkpoint_file: str | None = None
    checkpoint_every: int = 100


@dataclass
class Result:
    """Simulation result returned by solver wrappers."""

    times: FloatArray
    states: list[Array] | None = None
    expect: list[Array] = field(default_factory=list)
    expect_std: list[FloatArray] = field(default_factory=list)
    expect_stderr: list[FloatArray] = field(default_factory=list)
    state_observables: dict[str, Array] = field(default_factory=dict)
    entropy: FloatArray | None = None
    solver_stats: dict[str, Any] = field(default_factory=dict)

    def evaluate_state_observables(
        self,
        observables: Mapping[str, Any],
        *,
        inplace: bool = True,
    ) -> dict[str, Array]:
        """Evaluate scalar callbacks on saved states and store the time series."""
        if self.states is None:
            msg = "Result has no saved states to evaluate."
            raise ValueError(msg)
        from .observables import evaluate_state_observables

        values = evaluate_state_observables(self.states, observables)
        if inplace:
            self.state_observables.update(values)
        return values

    def save_hdf5(self, path: str | Path, *, overwrite: bool = True) -> None:
        """Write this result to an HDF5 file."""
        target = Path(path)
        if target.exists() and not overwrite:
            msg = f"Refusing to overwrite existing file: {target}"
            raise FileExistsError(msg)
        target.parent.mkdir(parents=True, exist_ok=True)

        times = np.asarray(self.times, dtype=np.float64)
        with h5py.File(target, "w") as handle:
            handle.attrs["format"] = "openquantumsim.result"
            handle.attrs["format_version"] = "1"
            handle.create_dataset("times", data=times)
            handle.create_dataset(
                "expect",
                data=_stack_series(self.expect, np.complex128, len(times)),
            )
            handle.create_dataset(
                "expect_std",
                data=_stack_series(self.expect_std, np.float64, len(times)),
            )
            handle.create_dataset(
                "expect_stderr",
                data=_stack_series(self.expect_stderr, np.float64, len(times)),
            )
            if self.state_observables:
                observables_group = handle.create_group("state_observables")
                for name, values in self.state_observables.items():
                    observables_group.create_dataset(
                        _validate_series_name(name),
                        data=_series_array(values, np.complex128, len(times)),
                    )

            if self.entropy is not None:
                entropy = np.asarray(self.entropy, dtype=np.float64)
                if entropy.shape != times.shape:
                    msg = "entropy must have the same shape as times."
                    raise ValueError(msg)
                handle.create_dataset("entropy", data=entropy)

            if self.states is not None:
                handle.create_dataset("states", data=_stack_states(self.states))

            stats_group = handle.create_group("solver_stats")
            for key, value in self.solver_stats.items():
                _write_stat(stats_group, key, value)

    @classmethod
    def load_hdf5(cls, path: str | Path) -> Result:
        """Load a solver result from an HDF5 file."""
        with h5py.File(Path(path), "r") as handle:
            if handle.attrs.get("format") != "openquantumsim.result":
                msg = "HDF5 file is not an OpenQuantumSim result."
                raise ValueError(msg)

            times = np.asarray(handle["times"], dtype=np.float64)
            states = None
            if "states" in handle:
                raw_states = np.asarray(handle["states"], dtype=np.complex128)
                states = [raw_states[idx].copy() for idx in range(raw_states.shape[0])]

            entropy = None
            if "entropy" in handle:
                entropy = np.asarray(handle["entropy"], dtype=np.float64)

            state_observables: dict[str, Array] = {}
            if "state_observables" in handle:
                group = handle["state_observables"]
                state_observables = {
                    name: np.asarray(dataset, dtype=np.complex128)
                    for name, dataset in group.items()
                    if isinstance(dataset, h5py.Dataset)
                }

            stats: dict[str, Any] = {}
            if "solver_stats" in handle:
                stats = _read_stats(handle["solver_stats"])

            return cls(
                times=times,
                states=states,
                expect=_unstack_series(handle["expect"], np.complex128),
                expect_std=_unstack_series(handle["expect_std"], np.float64),
                expect_stderr=_unstack_series(handle["expect_stderr"], np.float64),
                state_observables=state_observables,
                entropy=entropy,
                solver_stats=stats,
            )


def load_result(path: str | Path) -> Result:
    """Load a solver result saved by :meth:`Result.save_hdf5`."""
    return Result.load_hdf5(path)


def _stack_series(
    series: list[NDArray[Any]],
    dtype: type[np.complex128] | type[np.float64],
    width: int,
) -> NDArray[Any]:
    if len(series) == 0:
        return np.empty((0, width), dtype=dtype)
    arrays = [np.asarray(values, dtype=dtype) for values in series]
    for values in arrays:
        if values.shape != (width,):
            msg = "result series entries must be one-dimensional and match times."
            raise ValueError(msg)
    return np.vstack(arrays)


def _unstack_series(
    dataset: h5py.Dataset,
    dtype: type[np.complex128] | type[np.float64],
) -> list[NDArray[Any]]:
    values = np.asarray(dataset, dtype=dtype)
    return [values[idx, :].copy() for idx in range(values.shape[0])]


def _series_array(
    values: NDArray[Any],
    dtype: type[np.complex128] | type[np.float64],
    width: int,
) -> NDArray[Any]:
    array = np.asarray(values, dtype=dtype)
    if array.shape != (width,):
        msg = "result time series entries must be one-dimensional and match times."
        raise ValueError(msg)
    return array


def _validate_series_name(name: object) -> str:
    if not isinstance(name, str):
        msg = "result time series names must be strings."
        raise TypeError(msg)
    if not name:
        msg = "result time series names must not be empty."
        raise ValueError(msg)
    if "/" in name:
        msg = "result time series names must not contain '/'."
        raise ValueError(msg)
    return name


def _stack_states(states: list[Array]) -> Array:
    if len(states) == 0:
        return np.empty((0, 0), dtype=np.complex128)
    arrays = [np.asarray(state, dtype=np.complex128) for state in states]
    shape = arrays[0].shape
    if len(shape) not in (1, 2):
        msg = "states must be ket vectors or density matrices."
        raise ValueError(msg)
    for state in arrays:
        if state.shape != shape:
            msg = "all saved states must have the same shape."
            raise ValueError(msg)
    return np.stack(arrays, axis=0)


def _write_stat(group: h5py.Group, key: str, value: Any) -> None:
    native_value = _native_scalar(value)
    encoding_key = f"{key}__encoding"
    if native_value is None:
        group.attrs[key] = ""
        group.attrs[encoding_key] = "none"
    elif isinstance(native_value, bool | int | float | str):
        group.attrs[key] = native_value
        group.attrs[encoding_key] = "native"
    else:
        group.attrs[key] = json.dumps(native_value)
        group.attrs[encoding_key] = "json"


def _read_stats(group: h5py.Group) -> dict[str, Any]:
    stats: dict[str, Any] = {}
    for key in group.attrs:
        if key.endswith("__encoding"):
            continue
        encoding = _decode_attr(group.attrs.get(f"{key}__encoding", "native"))
        value = _decode_attr(group.attrs[key])
        if encoding == "none":
            stats[key] = None
        elif encoding == "json":
            stats[key] = json.loads(str(value))
        else:
            stats[key] = value
    return stats


def _native_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _decode_attr(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, np.generic):
        return value.item()
    return value
