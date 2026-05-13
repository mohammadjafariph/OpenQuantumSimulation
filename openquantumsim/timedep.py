"""Time-dependent Hamiltonian helpers."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from numbers import Number
from typing import Any, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from .hilbert import HilbertSpace
from .operators import Operator

Array = NDArray[np.complex128]
FloatArray = NDArray[np.float64]


@dataclass(frozen=True, init=False)
class InterpolatedCoefficient:
    """Linearly interpolated scalar coefficient for time-dependent operators."""

    times: FloatArray
    values: Array

    def __init__(
        self,
        times: Sequence[float],
        values: Sequence[complex | float],
    ) -> None:
        time_array = np.asarray(times, dtype=np.float64)
        value_array = np.asarray(values, dtype=np.complex128)
        if time_array.ndim != 1 or value_array.ndim != 1:
            msg = "times and values must be one-dimensional."
            raise ValueError(msg)
        if len(time_array) == 0:
            msg = "times and values must not be empty."
            raise ValueError(msg)
        if len(time_array) != len(value_array):
            msg = "times and values must have the same length."
            raise ValueError(msg)
        if len(time_array) > 1 and np.any(np.diff(time_array) <= 0):
            msg = "times must be sorted in strictly ascending order."
            raise ValueError(msg)
        object.__setattr__(self, "times", time_array)
        object.__setattr__(self, "values", value_array)

    def __call__(self, t: float) -> complex:
        """Evaluate the coefficient at time ``t`` by linear interpolation."""
        if len(self.times) == 1:
            return complex(self.values[0])
        real = np.interp(t, self.times, self.values.real)
        imag = np.interp(t, self.times, self.values.imag)
        return complex(real, imag)


CoefficientLike: TypeAlias = (
    Number | Callable[[float], object] | InterpolatedCoefficient
)


@dataclass(frozen=True)
class HamiltonianTerm:
    """One term ``coefficient(t) * operator`` in a time-dependent Hamiltonian."""

    operator: Operator
    coefficient: CoefficientLike


TermLike: TypeAlias = HamiltonianTerm | tuple[Operator, CoefficientLike]


@dataclass(frozen=True, init=False)
class TimeDependentHamiltonian:
    """Hamiltonian of the form ``H0 + sum_i f_i(t) H_i``."""

    base: Operator
    terms: tuple[HamiltonianTerm, ...]

    def __init__(self, base: Operator, terms: Sequence[TermLike]) -> None:
        normalized = tuple(_normalize_term(term) for term in terms)
        for term in normalized:
            if term.operator.shape != base.shape:
                msg = "time-dependent Hamiltonian terms must match base shape."
                raise ValueError(msg)
        object.__setattr__(self, "base", base)
        object.__setattr__(self, "terms", normalized)

    @property
    def shape(self) -> tuple[int, int]:
        """Matrix shape."""
        return self.base.shape

    @property
    def dim(self) -> int:
        """Matrix dimension."""
        return self.base.dim

    @property
    def space(self) -> HilbertSpace | None:
        """Hilbert-space descriptor from the base operator."""
        return self.base.space

    def to_numpy(self, t: float) -> Array:
        """Evaluate ``H(t)`` as a dense NumPy matrix."""
        data = self.base.to_numpy().copy()
        for term in self.terms:
            coefficient = _evaluate_coefficient(term.coefficient, t)
            data += coefficient * term.operator.to_numpy()
        return cast(Array, data)

    def at(self, t: float) -> Operator:
        """Evaluate ``H(t)`` and return it as an :class:`Operator`."""
        return Operator(self.to_numpy(t), self.base.space, label="H(t)")

    def __call__(self, t: float) -> Operator:
        """Evaluate ``H(t)`` and return it as an :class:`Operator`."""
        return self.at(t)


def time_dependent_hamiltonian(
    base: Operator,
    terms: Sequence[TermLike],
) -> TimeDependentHamiltonian:
    """Build ``H(t) = base + sum_i coefficient_i(t) * operator_i``."""
    return TimeDependentHamiltonian(base, terms)


def _normalize_term(term: TermLike) -> HamiltonianTerm:
    if isinstance(term, HamiltonianTerm):
        _validate_coefficient(term.coefficient)
        return term
    operator, coefficient = term
    _validate_coefficient(coefficient)
    return HamiltonianTerm(operator, coefficient)


def _evaluate_coefficient(coefficient: CoefficientLike, t: float) -> complex:
    if isinstance(coefficient, InterpolatedCoefficient):
        return coefficient(t)
    if isinstance(coefficient, Number):
        return complex(cast(Any, coefficient))
    value = coefficient(t)
    return complex(cast(Any, value))


def _validate_coefficient(coefficient: object) -> None:
    if isinstance(coefficient, InterpolatedCoefficient):
        return
    if isinstance(coefficient, Number):
        return
    if callable(coefficient):
        value = coefficient(0.0)
        try:
            complex(value)
        except (TypeError, ValueError) as exc:
            msg = "coefficient callables must return a scalar number."
            raise TypeError(msg) from exc
        return
    msg = f"unsupported coefficient type: {type(coefficient).__name__}"
    raise TypeError(msg)
