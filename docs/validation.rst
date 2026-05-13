Validation Gallery
==================

OpenQuantumSim should earn trust by matching simple analytic limits and
established reference implementations. The current validation gallery covers a
closed-form qubit decay problem, a damped Jaynes-Cummings comparison against
QuTiP, and the command used for direct QuTiP speed comparisons.

Analytic Qubit Decay
--------------------

A two-level atom with spontaneous emission has excited-state population

.. math::

   P_e(t) = \exp(-\gamma t).

The corresponding OpenQuantumSim model is intentionally small enough to inspect
line by line:

.. code-block:: python

   import numpy as np
   import openquantumsim as oqs

   gamma = 0.35
   atom = oqs.SpinSpace(0.5, label="atom")
   H = 0.0 * oqs.sigmaz(atom)
   collapse = np.sqrt(gamma) * oqs.sigmam(atom)
   excited = oqs.basis(atom, "up")
   rho0 = oqs.ket2dm(excited)
   P_excited = oqs.Operator(oqs.ket2dm(excited), atom, "P_excited")
   times = np.linspace(0.0, 6.0, 61)

   result = oqs.mesolve(
       H,
       rho0,
       times,
       c_ops=[collapse],
       e_ops=[P_excited],
       options=oqs.Options(rtol=1e-9, atol=1e-11),
   )

   expected = np.exp(-gamma * times)
   max_error = np.max(np.abs(result.expect[0].real - expected))

This case is covered by ``tests/physics/test_qubit_decay.py``.

Jaynes-Cummings Against QuTiP
-----------------------------

The Jaynes-Cummings validation uses the same cavity-atom Hamiltonian and
collapse operators in OpenQuantumSim and QuTiP, then compares cavity photon
number and atomic excited-state population.

Install the optional validation dependency, then run:

.. code-block:: bash

   python -m pip install -e ".[validation]"
   python scripts/validate_jaynes_cummings_qutip.py \
       --cavity-dim 5 \
       --time-points 81 \
       --t-final 8.0 \
       --tolerance 5e-7 \
       --output-csv runs/jaynes_cummings_qutip_validation.csv

The script exits nonzero if either observable exceeds the requested tolerance.
A typical successful run prints:

.. code-block:: text

   max photon-number delta: ...
   max excited-population delta: ...
   tolerance: 5.000e-07
   Jaynes-Cummings QuTiP validation: passed

This case is covered by ``tests/physics/test_jaynes_cummings_qutip.py`` when
QuTiP is installed.

QuTiP Speed Benchmark
---------------------

Use the benchmark harness to compare OpenQuantumSim and QuTiP on the same
models and time grids:

.. code-block:: bash

   python benchmarks/bench_vs_qutip.py \
       --case qubit \
       --case jc5 \
       --time-points 101 \
       --t-final 6.0 \
       --repeats 3 \
       --output-json runs/bench_vs_qutip.json

The report includes median wall time, minimum wall time, speedup relative to
QuTiP, and the maximum expectation-value deviation from the QuTiP reference.
For publishable benchmark numbers, record the CPU model, Julia thread count,
Python version, Julia version, OpenQuantumSim commit, and QuTiP version next to
the generated JSON.
