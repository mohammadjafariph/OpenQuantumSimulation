# OpenQuantumSim

Python-accessible, Julia-powered tools for simulating open quantum systems.

OpenQuantumSim is starting as a research-grade package with a Python frontend
and a Julia backend. The first development target is a reliable MVP for
Lindblad master-equation propagation, Monte Carlo wave-function trajectories,
basic Hilbert-space construction, observables, and validation against canonical
open-system examples.

## Current Status

This repository is pre-alpha. It is usable for development and local research
runs, but not ready for a public release until the gate in
`docs/release_checklist.md` is green. The current scaffold includes:

- A Python package namespace: `openquantumsim`
- A Julia backend package: `OpenQuantumSimJL`
- Basic Hilbert-space types and dense Python operator primitives
- Sparse Julia operator and observable primitives
- Symmetric Dicke manifolds for collective spin ensembles
- General subsystem utilities such as `partial_trace` and mutual information
- Solver entry points for `mesolve`, time-dependent `mesolve`, `mcsolve`, and
  `steadystate`
- Starter tests for Python and Julia
- Release metadata and package-data wiring for the Julia backend

## Roadmap Focus

Phase 1 and Phase 2 follow the R&D roadmap:

1. Implement `FockSpace`, `SpinSpace`, basic states, and operators.
2. Build a `juliacall` bridge that loads `OpenQuantumSimJL`.
3. Implement `mesolve` for time-independent Lindblad dynamics.
4. Implement `mcsolve` for MCWF trajectories with thread-local RNG.
5. Add sparse/Krylov propagation and composite-space observables.
6. Add time-dependent Hamiltonians and parameter-sweep execution.
7. Validate against qubit decay and Jaynes-Cummings reference results.

## Quick Start

```bash
python -m pip install -e ".[dev]"
python setup_julia.py
python -m pytest
```

For the Julia backend:

```bash
julia --project=src/OpenQuantumSimJL -e 'using Pkg; Pkg.test()'
```

Build the local API docs and tutorial notebooks with:

```bash
python -m pip install -e ".[docs]"
sphinx-build -b html docs docs/_build/html
```

## Benchmarks

The first benchmark harness measures Monte Carlo trajectory scaling for the
qubit decay validation problem:

```bash
PYTHON_JULIACALL_HANDLE_SIGNALS=yes JULIA_NUM_THREADS=auto \
    python benchmarks/bench_mcsolve.py --n-traj 1000 --repeats 3
```

Use `scripts/run_benchmarks.sh` to run the default benchmark entry point. The
report includes Python elapsed time, Julia backend wall time, worker count, and
the maximum expectation-value delta from the serial reference.

The Dicke mutual-information research example has a separate batch benchmark:

```bash
python examples/dicke/bench_mi.py \
    --N 6 \
    --n-traj 20 \
    --time-points 101 \
    --batch-size 5 \
    --n-jobs 1 4 \
    --target-n-traj 1000
```

## Production Runner

A checkpointed MCWF runner is available for the qubit-decay validation problem:

```bash
python scripts/run_mcsolve_qubit_decay.py \
    --n-traj 2000 \
    --checkpoint-file runs/qubit_decay_checkpoint.h5 \
    --output runs/qubit_decay.h5 \
    --force
```

The script resumes from an existing matching checkpoint, prints progress by
default, and saves the final solver result as HDF5.

## Sweep Runner

The public `ParameterSweep` API expands parameter grids, skips completed
points on rerun, writes a restartable manifest, saves returned `Result`
objects, and produces aggregate `summary.csv` / `summary.h5` files:

```python
sweep = oqs.ParameterSweep(
    base_system={"model": "qubit_decay"},
    params={"kappa": [0.02, 0.05, 0.1]},
)

run = sweep.run(run_one_point, output_dir="runs/kappa_sweep")
print(run.summary)
```

The CLI wrapper uses the same output layout:

```bash
python scripts/run_sweep.py \
    --kappa-values 0.02,0.05,0.1 \
    --n-traj 2000 \
    --output-dir runs/kappa_sweep
```

Re-running the same command skips completed points and resumes unfinished
checkpointed points.

Research examples live on top of the general solver API. The roadmap
two-ensemble Dicke study has its model builder and analysis scripts in
`examples/dicke/`.

Single trajectories can save kets for trajectory-level diagnostics:

```python
import numpy as np
import openquantumsim as oqs
from examples.dicke.observables import trajectory_dicke_mutual_information
from examples.dicke.system import two_ensemble_dicke_system

system = two_ensemble_dicke_system(N=6, kappa=0.1)
times = np.linspace(0.0, 1.0, 101)

result = oqs.single_trajectory(
    system.H,
    system.psi0,
    times,
    c_ops=system.c_ops,
    e_ops=system.e_ops,
    options=oqs.Options(seed=2026, max_step=0.01, save_states=True),
)

mi_a, mi_b = trajectory_dicke_mutual_information(result.states or [], 6)
```

For batched MI distributions, use the restartable trajectory runner:

```bash
python examples/dicke/run_mi_distribution.py \
    --n-values 6,8,12 \
    --kappa-values 0.1 \
    --n-traj 1000 \
    --time-points 2001 \
    --t-final 200 \
    --n-jobs 4 \
    --batch-size 10 \
    --output runs/dicke_mi_distribution.h5
```

The HDF5 output stores `MI_time_A`, `MI_time_B`, `MI_steady_A`, and
`MI_steady_B` under `kappa_<value>/N_<N>` groups and can resume incomplete
points from the stored trajectory count. Worker processes compute trajectory
batches; the parent process owns all HDF5 writes.

Summarize and plot completed MI distribution runs with:

```bash
python examples/dicke/analyze_mi_distribution.py \
    --input runs/dicke_mi_distribution.h5 \
    --output-dir runs/dicke_mi_analysis
```

The analyzer writes `summary.csv`, `summary.h5`, `steady_mi_mean.png`, and
`steady_mi_boxplot.png`.

The first recommended pilot settings are captured in `examples/dicke/README.md`.

## Validation

Canonical validation cases currently cover analytic qubit decay and a
Jaynes-Cummings comparison against QuTiP. Install the optional validation extra
or QuTiP directly, then run:

```bash
python scripts/validate_jaynes_cummings_qutip.py
```

The script compares cavity photon number and atomic excited-state population
between OpenQuantumSim and QuTiP, and exits nonzero if the maximum deviation
exceeds the requested tolerance.

Two-time correlations use the quantum regression theorem for time-independent
Lindblad systems:

```python
taus = np.linspace(0.0, 4.0, 101)
corr = oqs.correlation_2op_1t(
    H,
    rho0,
    taus,
    oqs.sigmap(qubit),
    oqs.sigmam(qubit),
    c_ops=[collapse],
)
```

Finite Fock-space states can also be inspected in phase space:

```python
x, p = oqs.phase_space_grid(xlim=(-5.0, 5.0), points=201)
rho = oqs.ket2dm(oqs.coherent(oqs.FockSpace(30), 1.0 + 0.5j))

W = oqs.wigner(rho, x, p)
Q = oqs.q_function(rho, x, p)
ax = oqs.plot_wigner(rho, x, p)
```

## Minimal Python Example

```python
import numpy as np
import openquantumsim as oqs

qubit = oqs.SpinSpace(0.5, label="atom")
sm = oqs.sigmam(qubit)
H = 0.5 * oqs.sigmaz(qubit)
rho0 = oqs.ket2dm(oqs.basis(qubit, "up"))

print(H.shape)
print(np.trace(rho0))
```

Subsystem observables are model agnostic:

```python
bell = np.array([1, 0, 0, 1], dtype=np.complex128) / np.sqrt(2)
rho_a = oqs.partial_trace(bell, dims=(2, 2), keep=0)
mi_ab = oqs.mutual_information(bell, dims=(2, 2), subsystem_a=0, subsystem_b=1)
```

Time-dependent Hamiltonians can be written as `H0 + sum_i f_i(t) H_i`:

```python
drive = oqs.InterpolatedCoefficient([0.0, 5.0], [0.0, 0.2])
H_t = oqs.time_dependent_hamiltonian(
    0.5 * oqs.sigmaz(qubit),
    [(oqs.sigmax(qubit), drive)],
)

result = oqs.mesolve(H_t, rho0, np.linspace(0.0, 5.0, 101))
```

Collective open-system models can use a symmetric Dicke manifold instead of
the full `2**N` spin Hilbert space:

```python
ensemble = oqs.DickeSpace(20, label="atoms")
Jm = oqs.collective_lowering(ensemble)
Jx = oqs.collective_x(ensemble)
Nexc = oqs.collective_excitation(ensemble)

H = 0.5 * Jx
c_ops = [np.sqrt(0.1 / ensemble.n_spins) * Jm]
psi0 = oqs.dicke_state(ensemble, excitations=ensemble.n_spins)
```

Arbitrary scalar diagnostics can be evaluated from saved kets or density
matrices and persisted with the solver result:

For `mcsolve`, built-in diagnostics from `state_metrics` that are linear
expectations or pure-trajectory constants are aggregated in the Julia backend
with mean, standard deviation, and standard error.

```python
metrics = oqs.state_metrics(
    purity=True,
    fidelity_to=psi0,
    population_indices=[0, 1],
)
metrics["left_entropy"] = lambda ket: oqs.von_neumann_entropy(
    oqs.partial_trace(ket, dims=(2, 2), keep=0)
)

result = oqs.single_trajectory(
    H,
    psi0,
    times,
    c_ops=c_ops,
    state_observables=metrics,
    options=oqs.Options(seed=2026, max_step=0.01),
)

left_entropy = result.state_observables["left_entropy"].real
return_fidelity = result.state_observables["fidelity"].real
```

## Result Persistence

Solver results can be saved in a portable HDF5 format:

```python
result.save_hdf5("runs/qubit_decay.h5")
loaded = oqs.load_result("runs/qubit_decay.h5")
```

The file stores time points, expectation series, optional saved states,
state-observable series, Monte Carlo uncertainty estimates, entropy, and solver
statistics. The full schema is documented in
`docs/result_hdf5_schema.rst`.

Long Monte Carlo trajectory runs can also checkpoint partial trajectory sums:

```python
result = oqs.mcsolve(
    H,
    psi0,
    times,
    c_ops=[collapse],
    e_ops=[observable],
    n_traj=20_000,
    options=oqs.Options(
        seed=2026,
        checkpoint_file="runs/mcsolve_checkpoint.h5",
        checkpoint_every=100,
        progress=True,
    ),
)
```

Calling `mcsolve` again with the same checkpoint file, seed, operators, time
grid, and `max_step` resumes from the stored trajectory count. Set
`progress=False` for silent batch runs.

## Repository Layout

```text
openquantumsim/             Python frontend
src/OpenQuantumSimJL/       Julia backend package
tests/                      Python test suite
docs/                       Sphinx documentation skeleton
benchmarks/                 Benchmark scripts
scripts/                    Development helpers
```
