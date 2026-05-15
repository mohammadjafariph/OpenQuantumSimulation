Performance Benchmarks
======================

This page records a reproducible OpenQuantumSim performance baseline. The
numbers below are not a universal leaderboard; they document one hardware and
software configuration so future benchmark results can be compared against the
same reference point.

Benchmark Environment
---------------------

.. list-table::
   :header-rows: 1

   * - Item
     - Value
   * - Date
     - 2026-05-14
   * - OpenQuantumSim commit
     - ``e27d402``
   * - CPU
     - Apple M1
   * - Logical CPU count
     - 8
   * - Platform
     - macOS 26.4.1 arm64
   * - Python
     - 3.14.3
   * - Julia backend runtime
     - 1.11.9 through JuliaCall
   * - ``julia --version``
     - 1.12.5
   * - OpenQuantumSim
     - 0.1.0a0
   * - QuTiP
     - 5.2.3
   * - NumPy / SciPy / h5py
     - 2.4.4 / 1.17.1 / 3.16.0

Deterministic Solver: OpenQuantumSim vs QuTiP
---------------------------------------------

Command:

.. code-block:: bash

   MPLCONFIGDIR=/private/tmp/oqs-mpl \
   python benchmarks/bench_vs_qutip.py \
       --repeats 3 \
       --time-points 81 \
       --t-final 6.0 \
       --cases qubit jc5 jc10 \
       --oqs-methods auto krylov ode \
       --json runs/benchmarks/bench_vs_qutip_m1_2026-05-14.json

Settings: ``rtol=1e-8``, ``atol=1e-10``. OpenQuantumSim used the default
single-threaded backend process for this deterministic benchmark.

.. list-table::
   :header-rows: 1

   * - Case
     - Dimension
     - QuTiP median
     - OQS auto median
     - Best OQS median
     - OQS auto vs QuTiP
     - Max expectation delta
   * - Qubit decay
     - 2
     - 1.108 ms
     - 5.931 ms
     - 5.880 ms (``ode``)
     - 0.19x
     - 7.49e-09
   * - Jaynes-Cummings 5
     - 10
     - 1.344 ms
     - 5.693 ms
     - 5.622 ms (``ode``)
     - 0.24x
     - 1.21e-09
   * - Jaynes-Cummings 10
     - 20
     - 1.853 ms
     - 6.390 ms
     - 6.068 ms (``ode``)
     - 0.29x
     - 7.23e-09

Interpretation: QuTiP is faster for these small deterministic systems. In this
regime, OpenQuantumSim wall time is dominated by Python-to-Julia overhead and
solver setup, while expectation values agree with QuTiP at about ``1e-9`` to
``1e-8``.

Monte Carlo Wave Function Scaling
---------------------------------

Command:

.. code-block:: bash

   JULIA_NUM_THREADS=4 MPLCONFIGDIR=/private/tmp/oqs-mpl \
   python benchmarks/bench_mcsolve.py \
       --n-traj 200 \
       --time-points 31 \
       --t-final 2.0 \
       --max-step 0.02 \
       --repeats 3 \
       --warmup-trajectories 10 \
       --n-jobs 1 -1 \
       --json runs/benchmarks/bench_mcsolve_m1_2026-05-14.json

.. list-table::
   :header-rows: 1

   * - ``n_jobs``
     - Workers
     - Threaded
     - Median elapsed
     - Backend wall time
     - Speedup vs serial
     - Max expectation delta
   * - 1
     - 1
     - False
     - 6.595 ms
     - 3.868 ms
     - 1.00x
     - 1.00e-02
   * - -1
     - 4
     - True
     - 4.092 ms
     - 1.542 ms
     - 1.61x
     - 1.00e-02

Interpretation: backend-side trajectory aggregation and threading work. The
small benchmark shows useful scaling, though the wall time is still heavily
affected by Python-call overhead at this size. Larger trajectory counts should
give a clearer measure of Julia-side scaling.

Dicke Mutual-Information Batch Runner
-------------------------------------

Command:

.. code-block:: bash

   JULIA_NUM_THREADS=1 MPLCONFIGDIR=/private/tmp/oqs-mpl \
   python examples/dicke/bench_mi.py \
       --N 4 \
       --kappa 0.1 \
       --n-traj 12 \
       --time-points 21 \
       --t-final 0.2 \
       --max-step 0.02 \
       --batch-size 2 \
       --repeats 2 \
       --warmup-trajectories 1 \
       --n-jobs 1 2 \
       --target-n-traj 1000 \
       --json runs/benchmarks/bench_dicke_mi_m1_2026-05-14.json

.. list-table::
   :header-rows: 1

   * - ``n_jobs``
     - Workers
     - Median elapsed
     - Trajectories / s
     - Seconds / trajectory
     - Speedup vs serial
   * - 1
     - 1
     - 0.0836 s
     - 143.46
     - 0.0070
     - 1.00x
   * - 2
     - 2
     - 19.7627 s
     - 0.607
     - 1.6469
     - 0.004x

Interpretation: this small Dicke MI benchmark exposes process-startup overhead.
Each short-lived worker initializes its own Julia backend, so process
parallelism is slower for small batches. Larger batches better amortize startup
costs.

Reproducing Results
-------------------

The raw JSON outputs are generated under ``runs/benchmarks/`` and are ignored by
Git. Re-run the commands above to regenerate the local benchmark artifacts.
