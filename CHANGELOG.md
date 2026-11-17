# Changelog

All notable changes to OpenQuantumSim will be documented here.

The project follows semantic versioning once the public API reaches `0.1.0`.
Until then, entries are grouped under alpha releases.

## Unreleased

Phase 1 of the roadmap, "trust the numbers": every solver result is either
correct to the requested tolerance or arrives with an error or a warning.

Changed behaviour (read before upgrading):

- `mcsolve` and `single_trajectory` now use the waiting-time quantum-jump
  algorithm on an adaptive integrator, with jump times located by
  root-finding. Accuracy is set by `Options.rtol`/`atol` instead of a fixed
  time step. `Options.max_step` now defaults to `None` and only caps the
  adaptive step; it no longer has to be chosen for accuracy.
- Trajectories draw from Julia's Xoshiro256++ generator, one independent
  stream per trajectory derived from `Options.seed`. Results for a given seed
  differ from earlier releases and no longer depend on `n_jobs` or the thread
  count. `n_jobs` now limits the number of threads instead of meaning "one or
  all". With `seed=None` the seed actually used is reported as
  `solver_stats["base_seed"]`. Every explicit seed is reproducible:
  `Options(seed=0)` used to draw a fresh random seed and now means seed 0.
- Monte Carlo checkpoints use format version 2 and are tied to the tolerances
  and integrator as well as the system. Checkpoints written by earlier
  releases are refused with an explanatory error instead of being resumed.
- Solvers no longer return a result silently when something went wrong. A
  failed integration, an unconverged Krylov step or an unconverged GMRES
  solve raises `SolverError` or `ConvergenceError`; use
  `Options(strict=False)` to get a `SolverHealthWarning` and the partial
  result (values after the failure are NaN) instead.
- `steadystate` defaults to `method="auto"` (direct sparse solve up to
  Hilbert dimension 128, GMRES above) instead of always using GMRES.
- A missing Julia backend raises `JuliaBridgeUnavailable`. It still derives
  from `NotImplementedError`, so existing `except` clauses keep working.
- `Options` validates its fields on construction. `Options.ode_solver` is now
  honoured (`"tsit5"`, `"vern7"`, `"vern9"`); `Options(backend="gpu")` raises
  `NotImplementedError` instead of being ignored.

Added:

- Exception and warning types in `openquantumsim.errors`:
  `OpenQuantumSimError`, `BackendError` (wraps Julia exceptions and keeps the
  Julia type name), `JuliaBridgeUnavailable`, `SolverError`,
  `ConvergenceError`, and `SolverHealthWarning`.
- Health statistics in `Result.solver_stats`: integrator return code, step
  and rejection counts, Krylov residuals, the drift of the trace and of
  Hermiticity, and `min_eigenvalue`, the most negative eigenvalue of the
  density matrix. Drift beyond the requested accuracy emits a
  `SolverHealthWarning`. Trace and Hermiticity are checked at every output
  time. Positivity is spot-checked at up to 8 evenly spaced times (4 above
  Hilbert dimension 64), always including the last, with a Cholesky test;
  the eigenvalue is only computed for a state that fails it, so
  `min_eigenvalue` is 0 when every checked state is positive to within
  `1e-9` of its trace. The checks cost under 10% of a small `mesolve`.
- `steadystate(..., return_stats=True)` returns the convergence flag and the
  residual `|L rho| / |L|` next to the state.
- A validation suite of textbook problems with known answers in
  `openquantumsim.validation`, run with `oqs validate` (writes a Markdown and
  JSON report) and by `tests/physics/test_validation_suite.py`.
- `scripts/sync_julia_backend.py` to copy the development Julia backend into
  the packaged mirror, with `--check` for CI.
- Time grids are typed to accept NumPy arrays as well as sequences.

Performance:

- Lindblad generators are assembled once from (row, column, value)
  triplets instead of by repeated sparse additions, and applied row by row
  instead of column by column, split across Julia threads for large
  systems. A dimension-512 spin chain `mesolve` dropped from about 10.7 s to
  3.7 s on one thread (2.6 s on four); building its generator dropped from
  2.2 s to 0.4 s. On the sparse spin-chain benchmark OpenQuantumSim went
  from 0.7x QuTiP's speed to 1.7x-1.9x.
- `mcsolve` and `single_trajectory` keep the Hamiltonian, collapse and
  expectation operators sparse when the Hilbert space has dimension 32 or
  more and the operators are mostly zero: 6x faster at dimension 256 and 7x
  at 512 on spin chains.
- Time-dependent `mesolve` integrates a sparse superoperator sum for the
  same kind of problem instead of multiplying dense matrices: 6x faster at
  dimension 40 and 10x at 80.
- `brmesolve` groups transitions by sorting their Bohr frequencies, builds
  the secular generator sparsely in the energy eigenbasis and propagates
  there, rotating only the outputs. At dimension 20 it dropped from 1.8 s to
  3 ms, and at dimension 40 from 54 s to 12 ms; dimension 80 now takes
  about 70 ms. `br_liouvillian` still returns the computational-basis
  matrix.
- Correlation functions with `method="auto"` integrate with the ODE solver
  instead of Krylov propagation, 2.5-3.6x faster at the same accuracy.
  Krylov propagation is still available with `method="krylov"`.

Fixed:

- `ParameterSweep` crashed while writing `summary.h5` when a summary column
  had a value for some points and not others, or mixed numbers and text.
  Missing numbers are now written as NaN and mixed columns as text.
- The README, quickstart, theory page and `brmesolve` docstring said a
  constant Bloch-Redfield spectrum `g` is exactly equivalent to a Lindblad
  operator `sqrt(g) * A`. That holds only when `A` connects a single Bohr
  frequency; for `sigmax` on a qubit the secular approximation drops cross
  terms and the two differ (by about 0.08 in `<sigmax>` for `g = 0.1`).

Scripts and benchmarks:

- `--max-step` is optional in every example, benchmark and helper script and
  defaults to no cap.
- `benchmarks/bench_mcsolve.py` and `examples/dicke/bench_mi.py` reported a
  nonzero "delta" between job modes because they compared runs with
  different seeds. They now compare runs that share a seed, which is exactly
  zero for every thread count.
- `benchmarks/bench_mesolve.py` was a placeholder; it now compares the
  `mesolve` integrator choices for speed and accuracy.
- `scripts/run_benchmarks.sh` re-runs every benchmark on the performance
  page, and `benchmarks/plot_benchmark_summary.py` redraws the summary figure
  from their JSON reports.
- `examples/dicke/run_mi_distribution.py` records the trajectory algorithm in
  its output and refuses to resume a file written by the old fixed-step
  stepper, so samples from the two algorithms are never mixed.
- `benchmarks/bench_mcsolve_vs_qutip.py` runs both libraries at the same
  tolerances, times OpenQuantumSim on one thread and on all threads, checks
  both against the exact answer, and records the environment.
- The README and the performance page report new measurements for the
  current code on an AMD Ryzen 5 5600X; the Apple M1 numbers are kept as a
  historical snapshot.

Continuous integration:

- The Julia backend job runs with two threads, so the threaded code paths
  are tested.
- The physics job installs QuTiP and fails, instead of skipping, when QuTiP
  or the Julia backend is missing, so the reference cross-checks always run.
- Test coverage is measured and the validation report is attached to each
  run. Python 3.13 and Julia 1.10, 1.11 and latest are tested.

## 0.1.0a6 - 2026-09-30

- Added `brmesolve`, a secular Bloch-Redfield master-equation solver backed
  by the Julia runtime, with `a_ops` coupling operators and user-supplied
  spectrum callables; transitions sharing a Bohr frequency are grouped into
  single secular channels.
- Added a Bloch-Redfield testset to the Julia backend suite and physics
  tests cross-validating zero-temperature damping against `mesolve`.

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
- Added a coupled-qubit entanglement and emission-spectrum gallery example.
- Added `benchmarks/bench_sparse_scaling.py`, a sparse Hilbert-space scaling
  benchmark (spin chains at Hilbert dimensions ~512-2048) against QuTiP when
  it is installed.
- Added macOS to the CI matrix for Python lint/tests and Python-Julia physics
  tests, so the tested platforms match the benchmarked platforms.
- Fixed CI typing against numpy stubs and switched CI to the
  juliapkg-managed Julia executable.
- Documented the authorship policy in the README and polished the Sphinx
  documentation set.

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
