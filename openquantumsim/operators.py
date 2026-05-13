"""Python operator and state primitives.

Generated operators use sparse storage internally where that preserves the
structure of common open-system models. The public ``to_numpy`` and
``__array__`` interfaces still expose dense arrays for convenient inspection
and interoperability.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import factorial, sqrt
from typing import Any, ClassVar, cast

import numpy as np
from numpy.typing import NDArray
from scipy import sparse as sp  # type: ignore[import-untyped]

from .hilbert import CompositeSpace, DickeSpace, FockSpace, HilbertSpace, SpinSpace

Array = NDArray[np.complex128]
MatrixData = Any


@dataclass(frozen=True)
class Operator:
    """Linear operator with an optional Hilbert-space descriptor."""

    __array_priority__: ClassVar[float] = 10000.0

    data: MatrixData
    space: HilbertSpace | None = None
    label: str | None = None

    def __post_init__(self) -> None:
        data = _canonical_matrix_data(self.data)
        if data.shape[0] != data.shape[1]:
            msg = "Operator data must be a square matrix."
            raise ValueError(msg)
        object.__setattr__(self, "data", data)

    @property
    def shape(self) -> tuple[int, int]:
        """Matrix shape."""
        return (int(self.data.shape[0]), int(self.data.shape[1]))

    @property
    def dim(self) -> int:
        """Matrix dimension."""
        return int(self.data.shape[0])

    def dag(self) -> Operator:
        """Return the Hermitian adjoint."""
        label = f"{self.label or 'O'}^dag"
        return Operator(self.data.conj().T, self.space, label=label)

    def to_numpy(self) -> Array:
        """Return a dense NumPy array representation."""
        return _dense_matrix(self.data)

    def __array__(self, dtype: Any | None = None) -> NDArray[Any]:
        return np.asarray(self.to_numpy(), dtype=dtype)

    def __add__(self, other: Operator) -> Operator:
        self._check_same_shape(other)
        return Operator(_add_data(self.data, other.data), self.space)

    def __sub__(self, other: Operator) -> Operator:
        self._check_same_shape(other)
        return Operator(_add_data(self.data, -other.data), self.space)

    def __neg__(self) -> Operator:
        return Operator(-self.data, self.space)

    def __mul__(self, other: complex | float | int | Operator) -> Operator:
        if isinstance(other, Operator):
            if self.shape[1] != other.shape[0]:
                msg = "Operator dimensions are incompatible for multiplication."
                raise ValueError(msg)
            return Operator(_matmul_data(self.data, other.data), self.space)
        return Operator(self.data * complex(other), self.space)

    def __rmul__(self, other: complex | float | int) -> Operator:
        return Operator(complex(other) * self.data, self.space)

    def __matmul__(self, other: Operator) -> Operator:
        """Tensor product using the `@` operator."""
        return tensor(self, other)

    def _check_same_shape(self, other: Operator) -> None:
        if self.shape != other.shape:
            msg = f"Operator shapes differ: {self.shape} != {other.shape}."
            raise ValueError(msg)


def _dimension(space_or_dim: HilbertSpace | int) -> int:
    if isinstance(space_or_dim, HilbertSpace):
        return space_or_dim.dim
    return int(space_or_dim)


def _operator_space(space_or_dim: HilbertSpace | int) -> HilbertSpace | None:
    return space_or_dim if isinstance(space_or_dim, HilbertSpace) else None


def eye(space_or_dim: HilbertSpace | int) -> Operator:
    """Identity operator."""
    dim = _dimension(space_or_dim)
    return Operator(
        sp.identity(dim, dtype=np.complex128, format="csc"),
        _operator_space(space_or_dim),
        "I",
    )


def destroy(space_or_dim: HilbertSpace | int) -> Operator:
    """Bosonic annihilation operator for a finite-dimensional truncation."""
    dim = _dimension(space_or_dim)
    data = sp.diags(
        np.sqrt(np.arange(1, dim, dtype=np.float64)).astype(np.complex128),
        offsets=1,
        shape=(dim, dim),
        format="csc",
    )
    return Operator(data, _operator_space(space_or_dim), "a")


def create(space_or_dim: HilbertSpace | int) -> Operator:
    """Bosonic creation operator."""
    return destroy(space_or_dim).dag()


def num(space_or_dim: HilbertSpace | int) -> Operator:
    """Bosonic number operator."""
    dim = _dimension(space_or_dim)
    return Operator(
        sp.diags(
            np.arange(dim, dtype=np.complex128),
            offsets=0,
            shape=(dim, dim),
            format="csc",
        ),
        _operator_space(space_or_dim),
        "n",
    )


def sigmax(space_or_dim: HilbertSpace | int | None = None) -> Operator:
    """Pauli sigma-x operator."""
    space = _spin_half_space(space_or_dim)
    return Operator(
        sp.csc_matrix(np.array([[0, 1], [1, 0]], dtype=np.complex128)),
        space,
        "sigmax",
    )


def sigmay(space_or_dim: HilbertSpace | int | None = None) -> Operator:
    """Pauli sigma-y operator."""
    space = _spin_half_space(space_or_dim)
    return Operator(
        sp.csc_matrix(np.array([[0, -1j], [1j, 0]], dtype=np.complex128)),
        space,
        "sigmay",
    )


def sigmaz(space_or_dim: HilbertSpace | int | None = None) -> Operator:
    """Pauli sigma-z operator."""
    space = _spin_half_space(space_or_dim)
    return Operator(
        sp.csc_matrix(np.array([[1, 0], [0, -1]], dtype=np.complex128)),
        space,
        "sigmaz",
    )


def sigmam(space_or_dim: HilbertSpace | int | None = None) -> Operator:
    """Two-level lowering operator `|down><up|`."""
    space = _spin_half_space(space_or_dim)
    return Operator(
        sp.csc_matrix(np.array([[0, 0], [1, 0]], dtype=np.complex128)),
        space,
        "sigmam",
    )


def sigmap(space_or_dim: HilbertSpace | int | None = None) -> Operator:
    """Two-level raising operator `|up><down|`."""
    return sigmam(space_or_dim).dag()


def spin_jm(space_or_dim: SpinSpace | DickeSpace | int) -> Operator:
    """Spin-S lowering operator in the Dicke basis ordered by descending m."""
    dim = _dimension(space_or_dim)
    data = _collective_lowering_data(dim)
    return Operator(data, _operator_space(space_or_dim), "Jm")


def spin_jp(space_or_dim: SpinSpace | DickeSpace | int) -> Operator:
    """Spin-S raising operator."""
    return spin_jm(space_or_dim).dag()


def spin_jx(space_or_dim: SpinSpace | DickeSpace | int) -> Operator:
    """Spin-S x operator."""
    jm = spin_jm(space_or_dim)
    return Operator((jm.data + jm.data.conj().T) / 2, jm.space, "Jx")


def spin_jz(space_or_dim: SpinSpace | DickeSpace | int) -> Operator:
    """Spin-S z operator in the Dicke basis ordered by descending m."""
    dim = _dimension(space_or_dim)
    values = _collective_m_values(dim)
    return Operator(
        sp.diags(
            values.astype(np.complex128),
            offsets=0,
            shape=(dim, dim),
            format="csc",
        ),
        _operator_space(space_or_dim),
        "Jz",
    )


def dicke_jm(space_or_dim: DickeSpace | int) -> Operator:
    """Collective Dicke lowering operator ``J_-``."""
    return spin_jm(space_or_dim)


def dicke_jp(space_or_dim: DickeSpace | int) -> Operator:
    """Collective Dicke raising operator ``J_+``."""
    return spin_jp(space_or_dim)


def dicke_jx(space_or_dim: DickeSpace | int) -> Operator:
    """Collective Dicke drive operator ``J_x``."""
    return spin_jx(space_or_dim)


def dicke_jz(space_or_dim: DickeSpace | int) -> Operator:
    """Collective Dicke inversion operator ``J_z``."""
    return spin_jz(space_or_dim)


def dicke_excitation(space_or_dim: DickeSpace | int) -> Operator:
    """Collective excitation-number operator in the symmetric Dicke manifold."""
    dim = _dimension(space_or_dim)
    values = np.arange(dim - 1, -1, -1, dtype=np.float64)
    return Operator(
        sp.diags(
            values.astype(np.complex128),
            offsets=0,
            shape=(dim, dim),
            format="csc",
        ),
        _operator_space(space_or_dim),
        "Nexc",
    )


def collective_lowering(space_or_dim: DickeSpace | int) -> Operator:
    """Alias for the Dicke collective lowering operator ``J_-``."""
    return dicke_jm(space_or_dim)


def collective_raising(space_or_dim: DickeSpace | int) -> Operator:
    """Alias for the Dicke collective raising operator ``J_+``."""
    return dicke_jp(space_or_dim)


def collective_x(space_or_dim: DickeSpace | int) -> Operator:
    """Alias for the Dicke collective drive operator ``J_x``."""
    return dicke_jx(space_or_dim)


def collective_z(space_or_dim: DickeSpace | int) -> Operator:
    """Alias for the Dicke collective inversion operator ``J_z``."""
    return dicke_jz(space_or_dim)


def collective_excitation(space_or_dim: DickeSpace | int) -> Operator:
    """Alias for the Dicke collective excitation-number operator."""
    return dicke_excitation(space_or_dim)


def tensor(*items: Operator) -> Operator:
    """Kronecker product of operators."""
    if len(items) == 0:
        msg = "tensor requires at least one operator."
        raise ValueError(msg)
    data = items[0].data
    spaces: list[HilbertSpace] = []
    if items[0].space is not None:
        spaces.append(items[0].space)
    for item in items[1:]:
        data = _kron_data(data, item.data)
        if item.space is not None:
            spaces.append(item.space)
    space = CompositeSpace(tuple(spaces)) if len(spaces) == len(items) else None
    return Operator(data, space)


def basis(space_or_dim: HilbertSpace | int, index: int | str) -> Array:
    """Basis ket as a column vector."""
    dim = _dimension(space_or_dim)
    idx = _basis_index(space_or_dim, index)
    ket = np.zeros(dim, dtype=np.complex128)
    ket[idx] = 1.0
    return ket


def fock(space_or_dim: HilbertSpace | int, n: int) -> Array:
    """Fock basis state `|n>`."""
    return basis(space_or_dim, n)


def dicke_state(space: DickeSpace, excitations: int) -> Array:
    """Symmetric Dicke state with a fixed number of excited spins."""
    if excitations < 0 or excitations > space.n_spins:
        msg = "excitations must satisfy 0 <= excitations <= n_spins."
        raise ValueError(msg)
    return basis(space, space.n_spins - excitations)


def ket2dm(ket: Array) -> Array:
    """Convert a ket to a density matrix."""
    vector = np.asarray(ket, dtype=np.complex128).reshape(-1)
    return np.asarray(np.outer(vector, vector.conj()), dtype=np.complex128)


def coherent(space_or_dim: FockSpace | int, alpha: complex) -> Array:
    """Truncated coherent state."""
    dim = _dimension(space_or_dim)
    values = np.array(
        [alpha**n / sqrt(factorial(n)) for n in range(dim)],
        dtype=np.complex128,
    )
    values *= np.exp(-0.5 * abs(alpha) ** 2)
    norm = np.linalg.norm(values)
    return values / norm if norm > 0 else values


def thermal_dm(space_or_dim: FockSpace | int, nbar: float) -> Array:
    """Truncated thermal density matrix for a bosonic mode."""
    if nbar < 0:
        msg = "nbar must be non-negative."
        raise ValueError(msg)
    dim = _dimension(space_or_dim)
    probs = np.array([(nbar**n) / ((nbar + 1) ** (n + 1)) for n in range(dim)])
    probs = probs / probs.sum()
    return np.diag(probs.astype(np.complex128))


def _spin_half_space(space_or_dim: HilbertSpace | int | None) -> HilbertSpace | None:
    if space_or_dim is None:
        return None
    dim = _dimension(space_or_dim)
    if dim != 2:
        msg = "Pauli operators require a two-dimensional spin/qubit space."
        raise ValueError(msg)
    return _operator_space(space_or_dim)


def _basis_index(space_or_dim: HilbertSpace | int, index: int | str) -> int:
    dim = _dimension(space_or_dim)
    if isinstance(index, str):
        if isinstance(space_or_dim, DickeSpace):
            labels = {
                "all_up": 0,
                "all_excited": 0,
                "top": 0,
                "all_down": dim - 1,
                "ground": dim - 1,
                "bottom": dim - 1,
            }
            try:
                return labels[index.lower()]
            except KeyError as exc:
                msg = f"Unknown Dicke basis label {index!r}."
                raise ValueError(msg) from exc
        if not isinstance(space_or_dim, SpinSpace) or dim != 2:
            msg = "String basis labels are currently supported only for SpinSpace(0.5)."
            raise ValueError(msg)
        labels = {"up": 0, "down": 1, "e": 0, "g": 1}
        try:
            return labels[index.lower()]
        except KeyError as exc:
            msg = f"Unknown basis label {index!r}."
            raise ValueError(msg) from exc
    if index < 0 or index >= dim:
        msg = f"Basis index {index} outside dimension {dim}."
        raise ValueError(msg)
    return index


def _collective_m_values(dim: int) -> NDArray[np.float64]:
    spin = (dim - 1) / 2
    return spin - np.arange(dim, dtype=np.float64)


def _collective_lowering_data(dim: int) -> MatrixData:
    if dim <= 0:
        msg = "dimension must be positive."
        raise ValueError(msg)
    spin = (dim - 1) / 2
    m_values = spin - np.arange(dim - 1, dtype=np.float64)
    coeffs = np.sqrt(spin * (spin + 1) - m_values * (m_values - 1))
    return sp.diags(
        coeffs.astype(np.complex128),
        offsets=-1,
        shape=(dim, dim),
        format="csc",
    )


def _canonical_matrix_data(data: object) -> MatrixData:
    if sp.issparse(data):
        matrix = sp.csc_matrix(data, dtype=np.complex128)
        matrix.eliminate_zeros()
        if len(matrix.shape) != 2:
            msg = "Operator data must be two-dimensional."
            raise ValueError(msg)
        return matrix
    array = np.asarray(data, dtype=np.complex128)
    if array.ndim != 2:
        msg = "Operator data must be two-dimensional."
        raise ValueError(msg)
    return array


def _dense_matrix(data: MatrixData) -> Array:
    if sp.issparse(data):
        return cast(Array, np.asarray(data.toarray(), dtype=np.complex128))
    return cast(Array, np.asarray(data, dtype=np.complex128))


def _add_data(left: MatrixData, right: MatrixData) -> MatrixData:
    if sp.issparse(left) and sp.issparse(right):
        return _canonical_matrix_data(left + right)
    if sp.issparse(left) or sp.issparse(right):
        return _dense_matrix(left) + _dense_matrix(right)
    return _dense_matrix(left) + _dense_matrix(right)


def _matmul_data(left: MatrixData, right: MatrixData) -> MatrixData:
    if sp.issparse(left) or sp.issparse(right):
        return _canonical_matrix_data(left @ right)
    return _dense_matrix(left) @ _dense_matrix(right)


def _kron_data(left: MatrixData, right: MatrixData) -> MatrixData:
    if sp.issparse(left) or sp.issparse(right):
        return _canonical_matrix_data(
            sp.kron(
                sp.csc_matrix(left, dtype=np.complex128),
                sp.csc_matrix(right, dtype=np.complex128),
                format="csc",
            ),
        )
    return cast(Array, np.asarray(np.kron(left, right), dtype=np.complex128))
