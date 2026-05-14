# Changelog

All notable changes to OpenQuantumSim will be documented here.

The project follows semantic versioning once the public API reaches `0.1.0`.
Until then, entries are grouped under alpha releases.

## Unreleased

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
