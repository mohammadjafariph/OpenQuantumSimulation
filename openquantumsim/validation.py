"""Validation suite: textbook problems with known answers.

Every case solves a small open-quantum-system problem whose answer is known
independently of this library (a closed formula, an exactly solvable rate
equation, a higher-accuracy reference solve, or QuTiP) and reports one number:
how far the solver is from that answer, next to the tolerance it must meet.

Run the suite and write a report with::

    oqs validate --output-dir validation-report

or from Python::

    import openquantumsim.validation as validation
    report = validation.run_validation()
    print(report.to_markdown())

The same cases back ``tests/physics/test_validation_suite.py``, so a release
cannot pass its tests without also passing its validation report.
"""

from __future__ import annotations

import importlib
import json
import os
import platform
import sys
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import expm  # type: ignore[import-untyped]

from ._version import __version__
from .correlations import correlation_2op_1t
from .hilbert import DickeSpace, FockSpace, SpinSpace
from .operators import (
    Operator,
    basis,
    coherent,
    create,
    destroy,
    dicke_jm,
    dicke_jz,
    ket2dm,
    num,
    sigmam,
    sigmap,
    sigmax,
    sigmay,
    sigmaz,
)
from .result import Options
from .solvers import brmesolve, mcsolve, mesolve, steadystate
from .systems import jaynes_cummings_system
from .timedep import time_dependent_hamiltonian

FloatArray = NDArray[np.float64]
CaseResult = tuple[float, str]

__all__ = [
    "ValidationCase",
    "ValidationOutcome",
    "ValidationReport",
    "ValidationSkipped",
    "run_validation",
    "validation_cases",
]

# Deterministic solvers are run at this accuracy unless a case says otherwise.
_TIGHT = Options(rtol=1e-9, atol=1e-11)
# Trajectories per Monte Carlo case, and the fixed seed that makes the
# statistical checks reproducible.
_MC_TRAJECTORIES = 4000
_MC_SEED = 20261115


class ValidationSkipped(Exception):
    """A case could not run because an optional dependency is missing."""


@dataclass(frozen=True)
class ValidationCase:
    """One problem with a known answer."""

    name: str
    title: str
    category: str
    solver: str
    reference: str
    metric: str
    tolerance: float
    run: Callable[[], CaseResult]


@dataclass(frozen=True)
class ValidationOutcome:
    """Result of running one :class:`ValidationCase`."""

    case: ValidationCase
    status: str
    measured: float | None
    wall_time: float
    detail: str = ""

    @property
    def passed(self) -> bool:
        """Whether the case met its tolerance."""
        return self.status == "passed"


@dataclass(frozen=True)
class ValidationReport:
    """Outcomes of a validation run together with the environment it ran in."""

    outcomes: tuple[ValidationOutcome, ...]
    environment: dict[str, str]

    @property
    def passed(self) -> bool:
        """Whether no case failed or errored. Skipped cases do not fail."""
        return all(outcome.status in ("passed", "skipped") for outcome in self.outcomes)

    def counts(self) -> dict[str, int]:
        """Number of cases per status."""
        totals = {"passed": 0, "failed": 0, "error": 0, "skipped": 0}
        for outcome in self.outcomes:
            totals[outcome.status] = totals.get(outcome.status, 0) + 1
        return totals

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable description of the report."""
        return {
            "format": "openquantumsim.validation",
            "format_version": "1",
            "passed": self.passed,
            "counts": self.counts(),
            "environment": dict(self.environment),
            "cases": [
                {
                    "name": outcome.case.name,
                    "title": outcome.case.title,
                    "category": outcome.case.category,
                    "solver": outcome.case.solver,
                    "reference": outcome.case.reference,
                    "metric": outcome.case.metric,
                    "tolerance": outcome.case.tolerance,
                    "measured": outcome.measured,
                    "status": outcome.status,
                    "wall_time": outcome.wall_time,
                    "detail": outcome.detail,
                }
                for outcome in self.outcomes
            ],
        }

    def to_markdown(self) -> str:
        """Render the report as a Markdown document."""
        counts = self.counts()
        verdict = "PASSED" if self.passed else "FAILED"
        lines = [
            "# OpenQuantumSim validation report",
            "",
            f"**Result: {verdict}** - {counts['passed']} passed, "
            f"{counts['failed']} failed, {counts['error']} errored, "
            f"{counts['skipped']} skipped.",
            "",
            "Each case compares a solver with an answer known independently of "
            "this library. *Measured* must not exceed *Tolerance*.",
            "",
            "## Environment",
            "",
            "| | |",
            "|---|---|",
        ]
        lines += [f"| {key} | {value} |" for key, value in self.environment.items()]
        for category in _ordered_categories(self.outcomes):
            lines += [
                "",
                f"## {_CATEGORY_TITLES.get(category, category)}",
                "",
                "| Case | Solver | Known answer | Metric | Measured | Tolerance "
                "| Status | Time |",
                "|---|---|---|---|---|---|---|---|",
            ]
            for outcome in self.outcomes:
                if outcome.case.category != category:
                    continue
                case = outcome.case
                measured = (
                    "-" if outcome.measured is None else f"{outcome.measured:.2e}"
                )
                lines.append(
                    f"| {case.title} | `{case.solver}` | {case.reference} "
                    f"| {case.metric} | {measured} | {case.tolerance:.1e} "
                    f"| {outcome.status} | {outcome.wall_time:.2f} s |"
                )
        notes = [outcome for outcome in self.outcomes if outcome.detail]
        if notes:
            lines += ["", "## Details", ""]
            lines += [
                f"- **{outcome.case.title}**: {outcome.detail}" for outcome in notes
            ]
        return "\n".join(lines) + "\n"

    def write(self, output_dir: str | Path) -> tuple[Path, Path]:
        """Write ``validation_report.md`` and ``.json``; return both paths."""
        target = Path(output_dir)
        target.mkdir(parents=True, exist_ok=True)
        markdown_path = target / "validation_report.md"
        json_path = target / "validation_report.json"
        markdown_path.write_text(self.to_markdown(), encoding="utf-8")
        json_path.write_text(
            json.dumps(self.to_dict(), indent=2) + "\n",
            encoding="utf-8",
        )
        return markdown_path, json_path


_CATEGORY_TITLES = {
    "dynamics": "Dynamics with a closed-form answer",
    "steady-state": "Steady states",
    "monte-carlo": "Monte Carlo against the master equation",
    "convergence": "Convergence with tolerance and trajectory count",
    "reference": "Cross-checks against QuTiP",
}


def _ordered_categories(outcomes: Iterable[ValidationOutcome]) -> list[str]:
    seen: list[str] = []
    for outcome in outcomes:
        if outcome.case.category not in seen:
            seen.append(outcome.case.category)
    return seen


def run_validation(
    names: Sequence[str] | None = None,
    *,
    progress: Callable[[ValidationOutcome], None] | None = None,
) -> ValidationReport:
    """Run the validation cases and collect a report.

    ``names`` restricts the run to the given case names. ``progress`` is
    called with each outcome as soon as it is available.
    """
    cases = validation_cases()
    if names is not None:
        known = {case.name for case in cases}
        unknown = sorted(set(names) - known)
        if unknown:
            msg = f"unknown validation case(s): {', '.join(unknown)}"
            raise ValueError(msg)
        cases = [case for case in cases if case.name in set(names)]

    outcomes = []
    for case in cases:
        outcome = run_case(case)
        outcomes.append(outcome)
        if progress is not None:
            progress(outcome)
    return ValidationReport(outcomes=tuple(outcomes), environment=_environment())


def run_case(case: ValidationCase) -> ValidationOutcome:
    """Run one case, turning exceptions into an ``error`` outcome."""
    start = time.perf_counter()
    try:
        measured, detail = case.run()
    except ValidationSkipped as exc:
        return ValidationOutcome(case, "skipped", None, 0.0, str(exc))
    except Exception as exc:
        elapsed = time.perf_counter() - start
        detail = f"{type(exc).__name__}: {exc}"
        return ValidationOutcome(case, "error", None, elapsed, detail)
    elapsed = time.perf_counter() - start
    passed = bool(np.isfinite(measured)) and measured <= case.tolerance
    status = "passed" if passed else "failed"
    return ValidationOutcome(case, status, float(measured), elapsed, detail)


def validation_cases() -> list[ValidationCase]:
    """Return every validation case, in report order."""
    return [
        ValidationCase(
            "rabi_oscillation",
            "Rabi oscillation of a driven qubit",
            "dynamics",
            "mesolve",
            "P_e(t) = cos^2(Omega t / 2)",
            "max abs error",
            1e-7,
            _rabi_oscillation,
        ),
        ValidationCase(
            "spontaneous_emission",
            "Spontaneous emission from a superposition",
            "dynamics",
            "mesolve",
            "P_e = exp(-gamma t) / 2, coherence = exp(-gamma t / 2) / 2",
            "max abs error",
            1e-7,
            _spontaneous_emission,
        ),
        ValidationCase(
            "pure_dephasing",
            "Pure dephasing with free precession",
            "dynamics",
            "mesolve",
            "coherence = exp(-i omega t - 2 gamma t) / 2, populations fixed",
            "max abs error",
            1e-7,
            _pure_dephasing,
        ),
        ValidationCase(
            "damped_oscillator",
            "Damped harmonic oscillator, coherent state",
            "dynamics",
            "mesolve",
            "<a> = alpha exp(-i omega t - kappa t / 2), <n> = |alpha|^2 exp(-kappa t)",
            "max abs error",
            1e-7,
            _damped_oscillator,
        ),
        ValidationCase(
            "thermal_relaxation",
            "Oscillator relaxing to a thermal bath",
            "dynamics",
            "mesolve",
            "<n>(t) = nbar (1 - exp(-kappa t))",
            "max abs error",
            1e-7,
            _thermal_relaxation,
        ),
        ValidationCase(
            "thermal_qubit",
            "Qubit relaxing to thermal equilibrium",
            "dynamics",
            "mesolve",
            "P_e(t) = P_ss + (1 - P_ss) exp(-gamma (2 nbar + 1) t)",
            "max abs error",
            1e-7,
            _thermal_qubit,
        ),
        ValidationCase(
            "superradiant_decay",
            "Superradiant decay of 8 spins (Dicke cascade)",
            "dynamics",
            "mesolve",
            "level populations from the exact rate-equation cascade",
            "max abs error",
            1e-7,
            _superradiant_decay,
        ),
        ValidationCase(
            "vacuum_rabi",
            "Jaynes-Cummings vacuum Rabi oscillation",
            "dynamics",
            "mesolve",
            "P_e = cos^2(g t), <n> = sin^2(g t)",
            "max abs error",
            1e-7,
            _vacuum_rabi,
        ),
        ValidationCase(
            "rotating_drive",
            "Detuned rotating drive (time-dependent Hamiltonian)",
            "dynamics",
            "mesolve (time-dependent)",
            "generalised Rabi formula, flip amplitude Omega^2 / (Omega^2 + delta^2)",
            "max abs error",
            1e-7,
            _rotating_drive,
        ),
        ValidationCase(
            "krylov_propagation",
            "Damped Jaynes-Cummings with Krylov propagation",
            "dynamics",
            "mesolve (krylov)",
            "Vern9 solve at rtol = 1e-13",
            "max abs error",
            1e-8,
            _krylov_propagation,
        ),
        ValidationCase(
            "bloch_redfield_thermal",
            "Bloch-Redfield qubit in a thermal bath",
            "dynamics",
            "brmesolve",
            "P_e(t) = P_ss + (1 - P_ss) exp(-gamma (2 nbar + 1) t)",
            "max abs error",
            1e-7,
            _bloch_redfield_thermal,
        ),
        ValidationCase(
            "emission_correlation",
            "Two-time correlation of a decaying qubit",
            "dynamics",
            "correlation_2op_1t",
            "<sigma+(tau) sigma-(0)> = exp(i omega tau - gamma tau / 2)",
            "max abs error",
            1e-7,
            _emission_correlation,
        ),
        ValidationCase(
            "thermal_oscillator_steady_state",
            "Thermal steady state of an oscillator",
            "steady-state",
            "steadystate",
            "Boltzmann populations p_n ~ (nbar / (nbar + 1))^n",
            "max abs error",
            1e-10,
            _thermal_oscillator_steady_state,
        ),
        ValidationCase(
            "thermal_qubit_steady_state",
            "Thermal steady state of a qubit",
            "steady-state",
            "steadystate",
            "P_e = nbar / (2 nbar + 1)",
            "max abs error",
            1e-10,
            _thermal_qubit_steady_state,
        ),
        ValidationCase(
            "resonance_fluorescence",
            "Resonance fluorescence steady state",
            "steady-state",
            "steadystate",
            "P_e = Omega^2 / (gamma^2 + 2 Omega^2), "
            "|coherence| = Omega gamma / (gamma^2 + 2 Omega^2)",
            "max abs error",
            1e-10,
            _resonance_fluorescence,
        ),
        ValidationCase(
            "mcsolve_driven_qubit",
            "Driven, decaying qubit",
            "monte-carlo",
            "mcsolve",
            "mesolve at rtol = 1e-9",
            "max |error| / standard error",
            5.0,
            _mcsolve_driven_qubit,
        ),
        ValidationCase(
            "mcsolve_fast_decay",
            "Decay much faster than the output spacing",
            "monte-carlo",
            "mcsolve",
            "P_e(t) = exp(-gamma t), gamma = 20",
            "max |error| / standard error",
            5.0,
            _mcsolve_fast_decay,
        ),
        ValidationCase(
            "mcsolve_superradiance",
            "Superradiant decay of 6 spins",
            "monte-carlo",
            "mcsolve",
            "level populations from the exact rate-equation cascade",
            "max |error| / standard error",
            5.0,
            _mcsolve_superradiance,
        ),
        ValidationCase(
            "mcsolve_jaynes_cummings",
            "Damped Jaynes-Cummings model",
            "monte-carlo",
            "mcsolve",
            "mesolve at rtol = 1e-9",
            "max |error| / standard error",
            5.0,
            _mcsolve_jaynes_cummings,
        ),
        ValidationCase(
            "tolerance_convergence",
            "Error follows the requested tolerance",
            "convergence",
            "mesolve",
            "Vern9 solve at rtol = 1e-13",
            "max error / rtol over rtol = 1e-4 ... 1e-10",
            30.0,
            _tolerance_convergence,
        ),
        ValidationCase(
            "integrator_agreement",
            "Tsit5, Vern7 and Vern9 agree",
            "convergence",
            "mesolve",
            "Vern9 solve at rtol = 1e-13",
            "max abs error at rtol = 1e-10",
            1e-8,
            _integrator_agreement,
        ),
        ValidationCase(
            "trajectory_convergence",
            "Standard error shrinks as 1 / sqrt(n_traj)",
            "convergence",
            "mcsolve",
            "stderr(4 n) / stderr(n) = 1 / 2",
            "|ratio - 0.5|",
            0.05,
            _trajectory_convergence,
        ),
        ValidationCase(
            "qutip_jaynes_cummings",
            "Damped Jaynes-Cummings model",
            "reference",
            "mesolve",
            "qutip.mesolve at rtol = 1e-10",
            "max abs error",
            5e-7,
            _qutip_jaynes_cummings,
        ),
        ValidationCase(
            "qutip_steady_state",
            "Driven, damped cavity steady state",
            "reference",
            "steadystate",
            "qutip.steadystate",
            "max abs error",
            1e-8,
            _qutip_steady_state,
        ),
        ValidationCase(
            "qutip_correlation",
            "Cavity two-time correlation",
            "reference",
            "correlation_2op_1t",
            "qutip.correlation_2op_1t",
            "max abs error",
            1e-6,
            _qutip_correlation,
        ),
    ]


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _max_abs(values: Any) -> float:
    return float(np.max(np.abs(np.asarray(values))))


def _qubit() -> tuple[SpinSpace, Any, Operator]:
    atom = SpinSpace(0.5, label="atom")
    excited = basis(atom, "up")
    return atom, excited, Operator(ket2dm(excited), atom, "P_excited")


def _level_projectors(space: DickeSpace | FockSpace) -> list[Operator]:
    dim = space.dim
    projectors = []
    for index in range(dim):
        data = np.zeros((dim, dim), dtype=np.complex128)
        data[index, index] = 1.0
        projectors.append(Operator(data, space, f"P_{index}"))
    return projectors


def _superradiant_populations(n_spins: int, gamma: float, times: Any) -> FloatArray:
    """Exact level populations for collective decay from the top Dicke state.

    Level ``k`` (``k`` photons emitted) decays to ``k + 1`` at the rate
    ``gamma (N - k)(k + 1)``; this is ``gamma (J + M)(J - M + 1)`` with
    ``M = N/2 - k``. The populations follow from exponentiating that
    pure-death rate matrix, independently of any quantum solver.
    """
    rates = gamma * np.array([(n_spins - k) * (k + 1) for k in range(n_spins)])
    generator = np.zeros((n_spins + 1, n_spins + 1))
    for k, rate in enumerate(rates):
        generator[k, k] -= rate
        generator[k + 1, k] += rate
    initial = np.zeros(n_spins + 1)
    initial[0] = 1.0
    return np.array([expm(generator * float(t)) @ initial for t in times])


def _z_score(estimate: Any, stderr: Any, exact: Any, n_traj: int) -> float:
    """Largest deviation in units of the reported standard error.

    The standard error is floored at ``1 / n_traj`` so that points where no
    trajectory happened to differ (zero sample variance) are still compared
    on a meaningful scale.
    """
    deviation = np.abs(np.asarray(estimate).real - np.asarray(exact))
    scale = np.maximum(np.asarray(stderr), 1.0 / n_traj)
    return float(np.max(deviation / scale))


def _mc_options(**overrides: Any) -> Options:
    settings: dict[str, Any] = {"seed": _MC_SEED, "n_jobs": -1}
    settings.update(overrides)
    return Options(**settings)


def _import_qutip() -> Any:
    try:
        return importlib.import_module("qutip")
    except ImportError as exc:
        if os.environ.get("OQS_REQUIRE_QUTIP", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }:
            msg = "QuTiP is required (OQS_REQUIRE_QUTIP) but not installed."
            raise RuntimeError(msg) from exc
        msg = "QuTiP is not installed; run `pip install openquantumsim[validation]`."
        raise ValidationSkipped(msg) from exc


# --------------------------------------------------------------------------
# Dynamics with a closed-form answer
# --------------------------------------------------------------------------


def _rabi_oscillation() -> CaseResult:
    omega = 0.8
    atom, excited, projector = _qubit()
    times = np.linspace(0.0, 20.0, 201)
    result = mesolve(
        0.5 * omega * sigmax(atom),
        ket2dm(excited),
        times,
        e_ops=[projector],
        options=_TIGHT,
    )
    error = _max_abs(result.expect[0].real - np.cos(0.5 * omega * times) ** 2)
    return error, ""


def _spontaneous_emission() -> CaseResult:
    gamma = 0.35
    atom, excited, projector = _qubit()
    plus = (excited + basis(atom, "down")) / np.sqrt(2.0)
    times = np.linspace(0.0, 10.0, 101)
    result = mesolve(
        0.0 * sigmaz(atom),
        ket2dm(plus),
        times,
        c_ops=[np.sqrt(gamma) * sigmam(atom)],
        e_ops=[projector, sigmam(atom)],
        options=_TIGHT,
    )
    population_error = _max_abs(result.expect[0].real - 0.5 * np.exp(-gamma * times))
    coherence_error = _max_abs(result.expect[1] - 0.5 * np.exp(-0.5 * gamma * times))
    return max(population_error, coherence_error), ""


def _pure_dephasing() -> CaseResult:
    gamma, omega = 0.2, 1.3
    atom, excited, projector = _qubit()
    plus = (excited + basis(atom, "down")) / np.sqrt(2.0)
    times = np.linspace(0.0, 8.0, 161)
    result = mesolve(
        0.5 * omega * sigmaz(atom),
        ket2dm(plus),
        times,
        c_ops=[np.sqrt(gamma) * sigmaz(atom)],
        e_ops=[projector, sigmam(atom)],
        options=_TIGHT,
    )
    population_error = _max_abs(result.expect[0].real - 0.5)
    coherence = 0.5 * np.exp((-1j * omega - 2.0 * gamma) * times)
    return max(population_error, _max_abs(result.expect[1] - coherence)), ""


def _damped_oscillator() -> CaseResult:
    omega, kappa, alpha = 1.0, 0.3, 1.5
    cavity = FockSpace(40, label="cavity")
    a = destroy(cavity)
    times = np.linspace(0.0, 10.0, 101)
    result = mesolve(
        omega * num(cavity),
        ket2dm(coherent(cavity, alpha)),
        times,
        c_ops=[np.sqrt(kappa) * a],
        e_ops=[a, num(cavity)],
        options=_TIGHT,
    )
    amplitude = alpha * np.exp((-1j * omega - 0.5 * kappa) * times)
    photons = abs(alpha) ** 2 * np.exp(-kappa * times)
    error = max(
        _max_abs(result.expect[0] - amplitude),
        _max_abs(result.expect[1].real - photons),
    )
    return error, "Fock space truncated at 40 levels."


def _thermal_relaxation() -> CaseResult:
    kappa, nbar = 0.5, 0.5
    cavity = FockSpace(40, label="cavity")
    a = destroy(cavity)
    times = np.linspace(0.0, 12.0, 121)
    result = mesolve(
        1.0 * num(cavity),
        ket2dm(basis(cavity, 0)),
        times,
        c_ops=[
            np.sqrt(kappa * (nbar + 1.0)) * a,
            np.sqrt(kappa * nbar) * create(cavity),
        ],
        e_ops=[num(cavity)],
        options=_TIGHT,
    )
    expected = nbar * (1.0 - np.exp(-kappa * times))
    return _max_abs(result.expect[0].real - expected), (
        "Fock space truncated at 40 levels."
    )


def _thermal_qubit() -> CaseResult:
    gamma, nbar = 0.4, 0.7
    atom, excited, projector = _qubit()
    times = np.linspace(0.0, 10.0, 101)
    result = mesolve(
        0.5 * sigmaz(atom),
        ket2dm(excited),
        times,
        c_ops=[
            np.sqrt(gamma * (nbar + 1.0)) * sigmam(atom),
            np.sqrt(gamma * nbar) * sigmap(atom),
        ],
        e_ops=[projector],
        options=_TIGHT,
    )
    steady = nbar / (2.0 * nbar + 1.0)
    expected = steady + (1.0 - steady) * np.exp(-gamma * (2.0 * nbar + 1.0) * times)
    return _max_abs(result.expect[0].real - expected), ""


def _superradiant_decay() -> CaseResult:
    n_spins, gamma = 8, 1.0
    ensemble = DickeSpace(n_spins, label="ensemble")
    times = np.linspace(0.0, 2.0, 81)
    result = mesolve(
        0.0 * dicke_jz(ensemble),
        ket2dm(basis(ensemble, "all_up")),
        times,
        c_ops=[np.sqrt(gamma) * dicke_jm(ensemble)],
        e_ops=_level_projectors(ensemble),
        options=_TIGHT,
    )
    expected = _superradiant_populations(n_spins, gamma, times)
    measured = np.array([series.real for series in result.expect]).T
    return _max_abs(measured - expected), ""


def _vacuum_rabi() -> CaseResult:
    coupling = 0.25
    system = jaynes_cummings_system(4, coupling=coupling)
    times = np.linspace(0.0, 30.0, 301)
    result = mesolve(
        system.H,
        system.rho0,
        times,
        e_ops=system.e_ops,
        options=_TIGHT,
    )
    error = max(
        _max_abs(result.expect[0].real - np.sin(coupling * times) ** 2),
        _max_abs(result.expect[1].real - np.cos(coupling * times) ** 2),
    )
    return error, ""


def _rotating_drive() -> CaseResult:
    transition, drive, rabi = 2.0, 1.7, 0.5
    atom, excited, projector = _qubit()
    hamiltonian = time_dependent_hamiltonian(
        0.5 * transition * sigmaz(atom),
        [
            (0.5 * rabi * sigmax(atom), lambda t: float(np.cos(drive * t))),
            (0.5 * rabi * sigmay(atom), lambda t: float(np.sin(drive * t))),
        ],
    )
    times = np.linspace(0.0, 15.0, 151)
    result = mesolve(
        hamiltonian,
        ket2dm(excited),
        times,
        e_ops=[projector],
        options=_TIGHT,
    )
    detuning = transition - drive
    generalised = np.hypot(rabi, detuning)
    flipped = (rabi / generalised) ** 2 * np.sin(0.5 * generalised * times) ** 2
    return _max_abs(result.expect[0].real - (1.0 - flipped)), ""


def _damped_jaynes_cummings() -> Any:
    return jaynes_cummings_system(
        8,
        coupling=0.3,
        cavity_decay=0.05,
        atom_decay=0.03,
    )


def _reference_solution(system: Any, times: Any) -> list[Any]:
    reference = mesolve(
        system.H,
        system.rho0,
        times,
        c_ops=system.c_ops,
        e_ops=system.e_ops,
        options=Options(ode_solver="vern9", rtol=1e-13, atol=1e-15),
    )
    return reference.expect


def _expectation_error(result: Any, reference: list[Any]) -> float:
    return max(
        _max_abs(series - expected)
        for series, expected in zip(result.expect, reference, strict=True)
    )


def _krylov_propagation() -> CaseResult:
    system = _damped_jaynes_cummings()
    times = np.linspace(0.0, 20.0, 41)
    reference = _reference_solution(system, times)
    result = mesolve(
        system.H,
        system.rho0,
        times,
        c_ops=system.c_ops,
        e_ops=system.e_ops,
        options=Options(method="krylov", rtol=1e-10, atol=1e-10, krylov_dim=30),
    )
    residual = result.solver_stats["krylov_max_residual"]
    return _expectation_error(result, reference), (
        f"Largest Krylov residual estimate {residual:.1e}."
    )


def _bloch_redfield_thermal() -> CaseResult:
    gamma, nbar, transition = 0.4, 0.7, 1.0
    atom, excited, projector = _qubit()
    times = np.linspace(0.0, 10.0, 101)

    def spectrum(omega: float) -> float:
        # Emission sits at negative Bohr frequencies, absorption at positive.
        return gamma * (nbar + 1.0) if omega < 0 else gamma * nbar

    result = brmesolve(
        0.5 * transition * sigmaz(atom),
        ket2dm(excited),
        times,
        a_ops=[(sigmax(atom), spectrum)],
        e_ops=[projector],
        options=_TIGHT,
    )
    steady = nbar / (2.0 * nbar + 1.0)
    expected = steady + (1.0 - steady) * np.exp(-gamma * (2.0 * nbar + 1.0) * times)
    return _max_abs(result.expect[0].real - expected), ""


def _emission_correlation() -> CaseResult:
    gamma, omega = 0.4, 1.5
    atom, excited, _projector = _qubit()
    taus = np.linspace(0.0, 10.0, 101)
    correlation = correlation_2op_1t(
        0.5 * omega * sigmaz(atom),
        ket2dm(excited),
        taus,
        sigmap(atom),
        sigmam(atom),
        c_ops=[np.sqrt(gamma) * sigmam(atom)],
        options=Options(rtol=1e-10, atol=1e-10),
    )
    expected = np.exp((1j * omega - 0.5 * gamma) * taus)
    return _max_abs(correlation - expected), ""


# --------------------------------------------------------------------------
# Steady states
# --------------------------------------------------------------------------


def _thermal_oscillator_steady_state() -> CaseResult:
    kappa, nbar, dim = 0.5, 0.8, 30
    cavity = FockSpace(dim, label="cavity")
    rho = steadystate(
        1.0 * num(cavity),
        [
            np.sqrt(kappa * (nbar + 1.0)) * destroy(cavity),
            np.sqrt(kappa * nbar) * create(cavity),
        ],
    )
    weights = (nbar / (nbar + 1.0)) ** np.arange(dim)
    expected = np.diag(weights / weights.sum())
    return _max_abs(np.asarray(rho) - expected), (
        "Detailed balance holds exactly in the truncated space, so the "
        "truncated Boltzmann distribution is the exact answer."
    )


def _thermal_qubit_steady_state() -> CaseResult:
    gamma, nbar = 0.4, 0.7
    atom, _excited, _projector = _qubit()
    c_ops = [
        np.sqrt(gamma * (nbar + 1.0)) * sigmam(atom),
        np.sqrt(gamma * nbar) * sigmap(atom),
    ]
    steady = nbar / (2.0 * nbar + 1.0)
    expected = np.diag([steady, 1.0 - steady])
    errors = [
        _max_abs(
            np.asarray(steadystate(0.5 * sigmaz(atom), c_ops, method=method)) - expected
        )
        for method in ("direct", "iterative-gmres")
    ]
    return max(errors), "Checked with both the direct and the GMRES method."


def _resonance_fluorescence() -> CaseResult:
    rabi, gamma = 0.9, 0.5
    atom, _excited, _projector = _qubit()
    rho = np.asarray(
        steadystate(0.5 * rabi * sigmax(atom), [np.sqrt(gamma) * sigmam(atom)])
    )
    denominator = gamma**2 + 2.0 * rabi**2
    error = max(
        abs(rho[0, 0].real - rabi**2 / denominator),
        abs(abs(rho[0, 1]) - rabi * gamma / denominator),
    )
    return float(error), ""


# --------------------------------------------------------------------------
# Monte Carlo against the master equation
# --------------------------------------------------------------------------


def _mcsolve_driven_qubit() -> CaseResult:
    rabi, gamma = 1.0, 0.4
    atom, excited, projector = _qubit()
    hamiltonian = 0.5 * rabi * sigmax(atom)
    c_ops = [np.sqrt(gamma) * sigmam(atom)]
    e_ops = [projector, sigmay(atom)]
    times = np.linspace(0.0, 10.0, 41)
    exact = mesolve(
        hamiltonian,
        ket2dm(excited),
        times,
        c_ops=c_ops,
        e_ops=e_ops,
        options=_TIGHT,
    )
    sampled = mcsolve(
        hamiltonian,
        excited,
        times,
        c_ops=c_ops,
        e_ops=e_ops,
        n_traj=_MC_TRAJECTORIES,
        options=_mc_options(),
    )
    score = max(
        _z_score(
            sampled.expect[idx],
            sampled.expect_stderr[idx],
            exact.expect[idx].real,
            _MC_TRAJECTORIES,
        )
        for idx in range(len(e_ops))
    )
    return score, f"{_MC_TRAJECTORIES} trajectories, seed {_MC_SEED}."


def _mcsolve_fast_decay() -> CaseResult:
    gamma = 20.0
    atom, excited, projector = _qubit()
    times = np.linspace(0.0, 0.5, 26)
    sampled = mcsolve(
        0.0 * sigmaz(atom),
        excited,
        times,
        c_ops=[np.sqrt(gamma) * sigmam(atom)],
        e_ops=[projector],
        n_traj=_MC_TRAJECTORIES,
        options=_mc_options(),
    )
    score = _z_score(
        sampled.expect[0],
        sampled.expect_stderr[0],
        np.exp(-gamma * times),
        _MC_TRAJECTORIES,
    )
    return score, (
        f"{_MC_TRAJECTORIES} trajectories, seed {_MC_SEED}, default solver "
        "options. A fixed-step jump rule is biased here."
    )


def _mcsolve_superradiance() -> CaseResult:
    n_spins, gamma = 6, 2.0
    ensemble = DickeSpace(n_spins, label="ensemble")
    times = np.linspace(0.0, 1.0, 41)
    projectors = _level_projectors(ensemble)
    sampled = mcsolve(
        0.0 * dicke_jz(ensemble),
        basis(ensemble, "all_up"),
        times,
        c_ops=[np.sqrt(gamma) * dicke_jm(ensemble)],
        e_ops=projectors,
        n_traj=_MC_TRAJECTORIES,
        options=_mc_options(),
    )
    expected = _superradiant_populations(n_spins, gamma, times)
    score = max(
        _z_score(
            sampled.expect[level],
            sampled.expect_stderr[level],
            expected[:, level],
            _MC_TRAJECTORIES,
        )
        for level in range(n_spins + 1)
    )
    return score, f"{_MC_TRAJECTORIES} trajectories, seed {_MC_SEED}."


def _mcsolve_jaynes_cummings() -> CaseResult:
    system = jaynes_cummings_system(
        6,
        coupling=0.5,
        cavity_decay=0.3,
        atom_decay=0.1,
    )
    times = np.linspace(0.0, 12.0, 49)
    exact = mesolve(
        system.H,
        system.rho0,
        times,
        c_ops=system.c_ops,
        e_ops=system.e_ops,
        options=_TIGHT,
    )
    sampled = mcsolve(
        system.H,
        system.psi0,
        times,
        c_ops=system.c_ops,
        e_ops=system.e_ops,
        n_traj=_MC_TRAJECTORIES,
        options=_mc_options(),
    )
    score = max(
        _z_score(
            sampled.expect[idx],
            sampled.expect_stderr[idx],
            exact.expect[idx].real,
            _MC_TRAJECTORIES,
        )
        for idx in range(len(system.e_ops))
    )
    return score, f"{_MC_TRAJECTORIES} trajectories, seed {_MC_SEED}."


# --------------------------------------------------------------------------
# Convergence
# --------------------------------------------------------------------------


def _tolerance_convergence() -> CaseResult:
    system = _damped_jaynes_cummings()
    times = np.linspace(0.0, 20.0, 41)
    reference = _reference_solution(system, times)
    tolerances = [1e-4, 1e-6, 1e-8, 1e-10]
    errors = []
    for rtol in tolerances:
        result = mesolve(
            system.H,
            system.rho0,
            times,
            c_ops=system.c_ops,
            e_ops=system.e_ops,
            options=Options(rtol=rtol, atol=1e-2 * rtol),
        )
        errors.append(_expectation_error(result, reference))
    detail = "; ".join(
        f"rtol {rtol:.0e}: error {error:.1e}"
        for rtol, error in zip(tolerances, errors, strict=True)
    )
    pairs = zip(errors, errors[1:], strict=False)
    if any(later >= earlier for earlier, later in pairs):
        return float("inf"), f"Error did not shrink monotonically. {detail}"
    worst = max(error / rtol for rtol, error in zip(tolerances, errors, strict=True))
    return worst, detail


def _integrator_agreement() -> CaseResult:
    system = _damped_jaynes_cummings()
    times = np.linspace(0.0, 20.0, 41)
    reference = _reference_solution(system, times)
    errors = {}
    for solver in ("tsit5", "vern7", "vern9"):
        result = mesolve(
            system.H,
            system.rho0,
            times,
            c_ops=system.c_ops,
            e_ops=system.e_ops,
            options=Options(ode_solver=solver, rtol=1e-10, atol=1e-12),
        )
        errors[solver] = _expectation_error(result, reference)
    detail = "; ".join(f"{solver}: {error:.1e}" for solver, error in errors.items())
    return max(errors.values()), detail


def _trajectory_convergence() -> CaseResult:
    gamma = 0.35
    atom, excited, projector = _qubit()
    # The initial state is exact at times[0], so statistics start at times[1].
    times = np.linspace(0.0, 4.0, 17)
    stderrs = []
    counts = (2000, 8000)
    for n_traj in counts:
        sampled = mcsolve(
            0.0 * sigmaz(atom),
            excited,
            times,
            c_ops=[np.sqrt(gamma) * sigmam(atom)],
            e_ops=[projector],
            n_traj=n_traj,
            options=_mc_options(),
        )
        stderrs.append(np.asarray(sampled.expect_stderr[0])[1:])
        exact = np.exp(-gamma * times[1:])
        binomial = np.sqrt(exact * (1.0 - exact) / n_traj)
        if not np.allclose(stderrs[-1], binomial, rtol=0.1):
            return float("inf"), (
                "Reported standard error differs from the binomial value "
                f"sqrt(p (1 - p) / n) at n_traj = {n_traj}."
            )
    ratio = float(np.median(stderrs[1] / stderrs[0]))
    return abs(ratio - 0.5), (
        f"stderr ratio {ratio:.3f} between {counts[1]} and {counts[0]} "
        "trajectories; stderr also matches sqrt(p (1 - p) / n) within 10%."
    )


# --------------------------------------------------------------------------
# Cross-checks against QuTiP
# --------------------------------------------------------------------------


def _qutip_jaynes_cummings() -> CaseResult:
    qutip = _import_qutip()
    dim, coupling, cavity_decay, atom_decay = 5, 0.08, 0.015, 0.025
    times = np.linspace(0.0, 8.0, 81)
    system = jaynes_cummings_system(
        dim,
        coupling=coupling,
        cavity_decay=cavity_decay,
        atom_decay=atom_decay,
    )
    result = mesolve(
        system.H,
        system.rho0,
        times,
        c_ops=system.c_ops,
        e_ops=system.e_ops,
        options=_TIGHT,
    )

    a = qutip.tensor(qutip.destroy(dim), qutip.qeye(2))
    sm = qutip.tensor(qutip.qeye(dim), qutip.sigmam())
    sz = qutip.tensor(qutip.qeye(dim), qutip.sigmaz())
    excited = qutip.tensor(qutip.qeye(dim), qutip.basis(2, 0).proj())
    hamiltonian = a.dag() * a + 0.5 * sz + coupling * (a.dag() * sm + a * sm.dag())
    psi0 = qutip.tensor(qutip.basis(dim, 0), qutip.basis(2, 0))
    reference = qutip.mesolve(
        hamiltonian,
        psi0 * psi0.dag(),
        times,
        c_ops=[np.sqrt(cavity_decay) * a, np.sqrt(atom_decay) * sm],
        e_ops=[a.dag() * a, excited],
        options={"rtol": 1e-10, "atol": 1e-12, "store_states": False},
    )
    error = max(
        _max_abs(result.expect[idx].real - np.asarray(reference.expect[idx]))
        for idx in range(2)
    )
    return error, f"QuTiP {qutip.__version__}."


def _qutip_steady_state() -> CaseResult:
    qutip = _import_qutip()
    dim, drive, detuning, kappa = 25, 2.0, 1.0, 0.5
    cavity = FockSpace(dim, label="cavity")
    a = destroy(cavity)
    rho = steadystate(
        drive * (a + create(cavity)) + detuning * num(cavity),
        [np.sqrt(kappa) * a],
    )
    qa = qutip.destroy(dim)
    reference = qutip.steadystate(
        drive * (qa + qa.dag()) + detuning * qa.dag() * qa,
        [np.sqrt(kappa) * qa],
    )
    return _max_abs(np.asarray(rho) - reference.full()), (
        f"QuTiP {qutip.__version__}; 25-level cavity."
    )


def _qutip_correlation() -> CaseResult:
    qutip = _import_qutip()
    dim, drive, detuning, kappa = 12, 0.6, 1.0, 0.5
    cavity = FockSpace(dim, label="cavity")
    a = destroy(cavity)
    hamiltonian = drive * (a + create(cavity)) + detuning * num(cavity)
    rho0 = ket2dm(coherent(cavity, 0.8))
    taus = np.linspace(0.0, 8.0, 41)
    correlation = correlation_2op_1t(
        hamiltonian,
        rho0,
        taus,
        create(cavity),
        a,
        c_ops=[np.sqrt(kappa) * a],
        options=Options(rtol=1e-10, atol=1e-10),
    )
    qa = qutip.destroy(dim)
    reference = qutip.correlation_2op_1t(
        drive * (qa + qa.dag()) + detuning * qa.dag() * qa,
        qutip.Qobj(rho0),
        taus,
        [np.sqrt(kappa) * qa],
        qa.dag(),
        qa,
        options={"rtol": 1e-10, "atol": 1e-12},
    )
    return _max_abs(correlation - np.asarray(reference)), (
        f"QuTiP {qutip.__version__}; <a^dag(tau) a(0)> from a coherent state."
    )


# --------------------------------------------------------------------------
# Environment
# --------------------------------------------------------------------------


def _environment() -> dict[str, str]:
    environment = {
        "OpenQuantumSim": __version__,
        "Date (UTC)": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
        "Python": platform.python_version(),
        "Platform": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "Processor": platform.processor() or "unknown",
        "NumPy": np.__version__,
        "SciPy": _module_version("scipy"),
        "QuTiP": _module_version("qutip"),
    }
    environment.update(_julia_environment())
    return environment


def _module_version(name: str) -> str:
    try:
        module = importlib.import_module(name)
    except ImportError:
        return "not installed"
    return str(getattr(module, "__version__", "unknown"))


def _julia_environment() -> dict[str, str]:
    try:
        from ._julia_bridge import get_julia, load_backend

        load_backend()
        julia = get_julia()
        return {
            "Julia": str(julia.seval("string(VERSION)")),
            "Julia threads": str(julia.seval("Threads.nthreads()")),
            "Monte Carlo generator": str(
                julia.seval("OpenQuantumSimJL.MCSOLVE_RNG_NAME")
            ),
        }
    except Exception:
        return {"Julia": "unavailable"}


def _print_progress(outcome: ValidationOutcome) -> None:
    measured = "-" if outcome.measured is None else f"{outcome.measured:.2e}"
    print(
        f"[{outcome.status:>7}] {outcome.case.name:<32} "
        f"measured {measured:>9}  tolerance {outcome.case.tolerance:.1e}  "
        f"({outcome.wall_time:.1f} s)",
        flush=True,
    )
    if outcome.status in ("failed", "error") and outcome.detail:
        print(f"          {outcome.detail}", file=sys.stderr, flush=True)
