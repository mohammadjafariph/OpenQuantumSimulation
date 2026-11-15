"""Exception and warning types raised by OpenQuantumSim.

The hierarchy separates three situations a user may need to tell apart:

* the Julia backend could not be started (:class:`JuliaBridgeUnavailable`),
* the backend raised an error while running (:class:`BackendError`), and
* a solver ran but did not produce a trustworthy answer
  (:class:`SolverError`, :class:`ConvergenceError`, and the
  :class:`SolverHealthWarning` warning).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class OpenQuantumSimError(Exception):
    """Base class for errors raised by OpenQuantumSim."""


class BackendError(OpenQuantumSimError, RuntimeError):
    """The Julia backend raised an error.

    ``julia_type`` holds the name of the Julia exception type when it is
    known, for example ``"DimensionMismatch"``.
    """

    def __init__(self, message: str, *, julia_type: str | None = None) -> None:
        super().__init__(message)
        self.julia_type = julia_type


class JuliaBridgeUnavailable(BackendError, NotImplementedError):
    """The Julia backend could not be loaded.

    This also derives from :class:`NotImplementedError` because solver entry
    points raised that type before a dedicated exception existed.
    """


class SolverError(OpenQuantumSimError, RuntimeError):
    """A solver finished without a trustworthy result.

    ``solver_stats`` carries the statistics reported by the backend, including
    the return code that triggered the error.
    """

    def __init__(
        self,
        message: str,
        *,
        solver_stats: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.solver_stats: dict[str, Any] = dict(solver_stats or {})


class ConvergenceError(SolverError):
    """An iterative method (Krylov exponentiation or GMRES) did not converge."""


class SolverHealthWarning(UserWarning):
    """A solver result may be inaccurate.

    Emitted when conserved quantities drift beyond the expected tolerance, or
    in place of :class:`SolverError` when ``Options(strict=False)`` is used.
    """
