# Changelog

All notable changes to OpenQuantumSim will be documented here.

The project follows semantic versioning once the public API reaches `0.1.0`.
Until then, entries are grouped under alpha releases.

## Unreleased

- Added entanglement measures to `openquantumsim.observables`: `negativity`,
  `logarithmic_negativity`, and `concurrence`, plus a `negativity_observable`
  callback builder for solver runs.
- Added `spectrum_correlation_fft` and `spectrum_2op_1t` to
  `openquantumsim.correlations` for frequency-domain spectra of sampled
  correlation functions via FFT.
- Added `position`, `momentum`, `identity`, and `coherent_dm` operators.
- Added `propagator` for time-independent unitary and Lindblad propagator
  superoperators, plus `lindblad_superoperator` and `apply_superoperator`
  for working with column-stacked superoperator matrices.
- `Operator` now supports scalar division (`op / x`); previously dividing an
  operator returned a plain NumPy array instead of an `Operator`.
- Fixed Windows path handling in the Julia bridge and `oqs` CLI: backend and
  sysimage paths are normalized to posix form before reaching Julia.

## 0.1.0a5 - 2026-05-27

- Made `oqs build-sysimage` validate the generated sysimage with a fresh
  JuliaCall subprocess before registering it for automatic reuse.
- Removed the hard Julia-backend dependency on `PythonCall` so local sysimages
  do not bake PythonCall into the OpenQuantumSim backend image.
- Kept Python callable coefficients working by converting callback return
  values through JuliaCall's already-loaded PythonCall module at runtime.
- Added a startup/sysimage benchmark harness with partial JSON reports on
  failed runs.

## 0.1.0a4 - 2026-05-26

- Added the `oqs` command-line entry point with `oqs setup-julia` for explicit
  one-time backend precompilation.
- Added `oqs build-sysimage`, which builds and registers a local Julia sysimage
  for faster repeated solver startup.
- Added automatic cached-sysimage discovery before JuliaCall is imported, with
  `OPENQUANTUMSIM_USE_SYSIMAGE=0` and `OPENQUANTUMSIM_JULIA_SYSIMAGE` overrides.

## 0.1.0a3 - 2026-05-22

- Reworked the README opening into a scientist-facing package pitch with CI,
  Python-version, license, docs, and PyPI badges.
- Added a benchmark summary figure to the README and performance docs using
  the documented QuTiP comparison and a new MCWF trajectory benchmark against
  QuTiP.
- Added `benchmarks/bench_mcsolve_vs_qutip.py` for reproducible trajectory
  scaling checks against QuTiP.
- Added Colab links for public tutorial notebooks.
- Added a Dicke synchronization notebook demonstrating collective-spin
  dynamics in the symmetric Dicke manifold.

## 0.1.0a2 - 2026-05-22

- Published a cleaner public documentation set through GitHub Pages, including
  API references, tutorials, validation notes, performance notes, and the HDF5
  result schema.
- Replaced older README/release text with user-facing package documentation
  suitable for PyPI.
- Added a five-minute qubit-decay tutorial as the first public getting-started
  path.
- Reduced Python wrapper overhead in solver-stat conversion; in the local
  benchmark, 100 warm qubit-decay `mesolve` calls dropped from 0.671 s to
  0.050 s.
- Reduced normal Julia backend startup overhead by skipping package
  instantiation on routine solver loads and keeping forced instantiation in
  `setup_julia.py`.
- Replaced the broad Julia `DifferentialEquations` dependency with
  `OrdinaryDiffEq` for the backend code path used by the solvers.
- Added larger Jaynes-Cummings performance spot checks against QuTiP.

## 0.1.0a1 - 2026-05-15

- Removed an invalid PyPI trove classifier from the package metadata so the
  alpha can be published through TestPyPI/PyPI.
- Documented TestPyPI and PyPI trusted-publisher verification commands.

## 0.1.0a0 - 2026-05-14

- Added a Python frontend and Julia backend package scaffold.
- Added dense Hilbert-space, state, and operator helpers.
- Added `mesolve`, `mcsolve`, and `single_trajectory` Python entry points.
- Added HDF5 result persistence and checkpointed MCWF runs.
- Added model-agnostic partial trace, entropy, purity, and mutual information
  utilities.
- Added named scalar `state_observables` for `mesolve` and
  `single_trajectory`.
- Moved the two-ensemble Dicke workflow into `examples/dicke` so the public
  package remains model agnostic.
- Added release-hygiene files for citation, contribution, changelog, and
  publish-readiness tracking.
