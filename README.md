# OpenQuantumSim

OpenQuantumSim is a Python package for simulating open quantum systems with a
Julia backend for numerical propagation. It provides a Python-first interface
for constructing Hilbert spaces, states, operators, Lindblad models,
Monte Carlo wave-function trajectories, observables, parameter sweeps, and
common state diagnostics.

The package is currently released as an alpha. The API is usable, tested, and
published on PyPI, but minor interface changes may still occur before a stable
`0.1` release.

## Installation

```bash
python -m pip install openquantumsim
```

OpenQuantumSim uses JuliaCall to load the packaged Julia backend. The first
solver call on a new machine may spend a few minutes resolving and precompiling
Julia packages.

For development from source:

```bash
git clone https://github.com/mohammadjafariph/OpenQuantumSimulation.git
cd OpenQuantumSimulation
python -m pip install -e ".[dev]"
python setup_julia.py
```

## Features

- Hilbert-space helpers for finite Fock spaces, spin spaces, tensor-product
  systems, and symmetric Dicke manifolds.
- State and operator constructors for common open-system models.
- Lindblad master-equation propagation with dense and sparse backends.
- Monte Carlo wave-function trajectories with backend-side aggregation for
  selected diagnostics.
- Time-dependent Hamiltonians with callable or interpolated coefficients.
- Steady-state solves, two-time correlations, and parameter sweeps.
- State metrics including purity, entropy, fidelity, trace distance,
  populations, coherences, and Bloch-vector components.
- Wigner and Husimi-Q phase-space distributions for finite Fock spaces.
- HDF5 result persistence for solver outputs and sweep summaries.
- Validation scripts comparing analytic limits and QuTiP reference models.

## Quick Example

The example below solves spontaneous emission for a two-level system and
compares the excited-state population with the analytic result
`exp(-gamma * t)`.

```python
import numpy as np
import openquantumsim as oqs

atom = oqs.SpinSpace(0.5, label="atom")
excited = oqs.basis(atom, "up")

gamma = 0.2
H = 0.0 * oqs.sigmaz(atom)
rho0 = oqs.ket2dm(excited)
collapse = np.sqrt(gamma) * oqs.sigmam(atom)
projector = oqs.Operator(oqs.ket2dm(excited), atom, "P_excited")
times = np.linspace(0.0, 0.2, 3)

result = oqs.mesolve(
    H,
    rho0,
    times,
    c_ops=[collapse],
    e_ops=[projector],
    options=oqs.Options(rtol=1e-8, atol=1e-10),
)

expected = np.exp(-gamma * times)
assert np.allclose(result.expect[0].real, expected, atol=2e-7)
print(result.expect[0].real)
```

## Time-Dependent Hamiltonians

Time-dependent systems can be written as
`H(t) = H0 + sum_i f_i(t) H_i`.

```python
drive = oqs.InterpolatedCoefficient([0.0, 5.0], [0.0, 0.2])
H_t = oqs.time_dependent_hamiltonian(
    0.5 * oqs.sigmaz(atom),
    [(oqs.sigmax(atom), drive)],
)

result = oqs.mesolve(H_t, rho0, np.linspace(0.0, 5.0, 101), c_ops=[collapse])
```

## Monte Carlo Trajectories

```python
result = oqs.mcsolve(
    H,
    excited,
    times,
    c_ops=[collapse],
    e_ops=[projector],
    n_traj=1000,
    options=oqs.Options(seed=2026, max_step=0.01, progress=True),
)

population_mean = result.expect[0].real
population_stderr = result.expect_stderr[0].real
```

Long trajectory runs can checkpoint partial sums and resume from the same
operators, seed, time grid, and solver options:

```python
result = oqs.mcsolve(
    H,
    excited,
    times,
    c_ops=[collapse],
    e_ops=[projector],
    n_traj=20_000,
    options=oqs.Options(
        seed=2026,
        max_step=0.01,
        checkpoint_file="runs/mcsolve_checkpoint.h5",
        checkpoint_every=100,
    ),
)
```

## State Diagnostics

State diagnostics can be evaluated during deterministic solves, single
trajectories, and supported trajectory aggregations.

```python
metrics = oqs.state_metrics(
    purity=True,
    fidelity_to=excited,
    population_indices=[0, 1],
)

trajectory = oqs.single_trajectory(
    H,
    excited,
    times,
    c_ops=[collapse],
    state_observables=metrics,
    options=oqs.Options(seed=2026, max_step=0.01),
)

purity = trajectory.state_observables["purity"].real
```

## Parameter Sweeps

`ParameterSweep` expands a parameter grid, skips completed points on rerun,
writes a restartable manifest, and saves aggregate summaries.

```python
sweep = oqs.ParameterSweep(
    base_system={"model": "qubit_decay"},
    params={"gamma": [0.05, 0.1, 0.2]},
)

run = sweep.run(run_one_point, output_dir="runs/gamma_sweep")
print(run.summary)
```

## Phase-Space Utilities

Finite Fock-space states can be inspected with Wigner and Husimi-Q
distributions.

```python
space = oqs.FockSpace(30)
rho = oqs.ket2dm(oqs.coherent(space, 1.0 + 0.5j))
x, p = oqs.phase_space_grid(xlim=(-5.0, 5.0), points=201)

W = oqs.wigner(rho, x, p)
Q = oqs.q_function(rho, x, p)
ax = oqs.plot_wigner(rho, x, p)
```

## Result Persistence

Solver results can be saved and loaded in HDF5 format.

```python
result.save_hdf5("runs/qubit_decay.h5")
loaded = oqs.load_result("runs/qubit_decay.h5")
```

The schema stores time points, expectation series, optional saved states,
state-observable series, Monte Carlo uncertainty estimates, entropy, and solver
statistics.

## Validation

The validation suite includes analytic qubit decay and a damped
Jaynes-Cummings comparison against QuTiP.

```bash
python -m pip install -e ".[validation]"
python scripts/validate_jaynes_cummings_qutip.py
```

Performance comparison scripts are available under `benchmarks/`. Benchmark
results depend strongly on problem size, backend warmup, hardware, and thread
configuration; record those settings with any reported timings.

## Documentation

Build the local documentation with:

```bash
python -m pip install -e ".[docs]"
sphinx-build -b html docs docs/_build/html
```

The documentation includes API pages, tutorials, validation examples,
benchmark notes, and the HDF5 result schema.

## Development

Run the Python test suite:

```bash
python -m pytest
```

Run the Julia backend tests:

```bash
julia --project=src/OpenQuantumSimJL -e 'using Pkg; Pkg.test()'
```

Repository layout:

```text
openquantumsim/             Python frontend
src/OpenQuantumSimJL/       Julia backend package
tests/                      Python test suite
docs/                       Sphinx documentation
benchmarks/                 Benchmark scripts
scripts/                    Development and validation helpers
examples/                   Domain examples built on the public API
```
