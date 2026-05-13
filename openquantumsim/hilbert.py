"""Hilbert-space descriptors for OpenQuantumSim."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from functools import reduce
from math import isclose
from operator import mul


class HilbertSpace:
    """Base Hilbert-space descriptor."""

    @property
    def dim(self) -> int:
        """Return the finite Hilbert-space dimension."""
        raise NotImplementedError

    def __mul__(self, other: HilbertSpace) -> CompositeSpace:
        """Tensor-product composition using `space_a * space_b`."""
        if isinstance(self, CompositeSpace):
            left = self.spaces
        else:
            left = (self,)
        if isinstance(other, CompositeSpace):
            right = other.spaces
        else:
            right = (other,)
        return CompositeSpace(left + right)


@dataclass(frozen=True)
class FockSpace(HilbertSpace):
    """Truncated bosonic Fock space with states `|0>` through `|N-1>`."""

    N: int = 2
    label: str | None = None

    def __post_init__(self) -> None:
        if self.N <= 0:
            msg = "FockSpace dimension N must be positive."
            raise ValueError(msg)

    @property
    def dim(self) -> int:
        """Return the truncation dimension."""
        return self.N


@dataclass(frozen=True)
class SpinSpace(HilbertSpace):
    """Spin-S irreducible representation with dimension `2S + 1`."""

    S: float = 0.5
    label: str | None = None

    def __post_init__(self) -> None:
        dim = 2 * self.S + 1
        if self.S < 0 or not isclose(dim, round(dim), rel_tol=0.0, abs_tol=1e-12):
            msg = "SpinSpace requires S >= 0 and 2S + 1 to be an integer."
            raise ValueError(msg)

    @property
    def dim(self) -> int:
        """Return `2S + 1`."""
        return int(round(2 * self.S + 1))


@dataclass(frozen=True)
class DickeSpace(HilbertSpace):
    """Permutation-symmetric Dicke manifold for ``n_spins`` two-level systems."""

    n_spins: int
    label: str | None = None

    def __post_init__(self) -> None:
        if self.n_spins < 0:
            msg = "DickeSpace requires n_spins >= 0."
            raise ValueError(msg)

    @property
    def dim(self) -> int:
        """Return the symmetric-manifold dimension ``n_spins + 1``."""
        return self.n_spins + 1

    @property
    def total_spin(self) -> float:
        """Return the collective spin ``S = n_spins / 2``."""
        return self.n_spins / 2


@dataclass(frozen=True)
class CompositeSpace(HilbertSpace):
    """Tensor product of multiple Hilbert spaces."""

    spaces: tuple[HilbertSpace, ...]
    label: str | None = None

    def __post_init__(self) -> None:
        if len(self.spaces) == 0:
            msg = "CompositeSpace requires at least one subsystem."
            raise ValueError(msg)

    @property
    def dim(self) -> int:
        """Return the product dimension."""
        return reduce(mul, (space.dim for space in self.spaces), 1)

    def __iter__(self) -> Iterator[HilbertSpace]:
        return iter(self.spaces)

    def __len__(self) -> int:
        return len(self.spaces)
