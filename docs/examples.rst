Example Gallery
===============

The scripts in ``examples/gallery`` are complete, runnable starting points for
common open-system workflows. Each script accepts ``--fast`` for a small smoke
run, ``--output-dir`` for generated figures/data, and ``--json`` for a compact
machine-readable summary.

Run an example from a source checkout with:

.. code-block:: bash

   python examples/gallery/qubit_decay.py --fast

The default output location is ``runs/example_gallery/<example-name>``.

Deterministic Qubit Decay
-------------------------

Solves spontaneous emission with ``mesolve``, saves the result as HDF5, plots
the excited-state population, and compares against ``exp(-gamma * t)``.

.. literalinclude:: ../examples/gallery/qubit_decay.py
   :language: python
   :caption: examples/gallery/qubit_decay.py

Driven Qubit
------------

Builds a time-dependent Hamiltonian from an interpolated drive envelope and
tracks the excited population and Bloch components of a dissipative qubit.

.. literalinclude:: ../examples/gallery/driven_qubit.py
   :language: python
   :caption: examples/gallery/driven_qubit.py

Jaynes-Cummings Dynamics
------------------------

Uses the built-in Jaynes-Cummings system helper to simulate energy exchange
between a truncated cavity and a two-level atom.

.. literalinclude:: ../examples/gallery/jaynes_cummings.py
   :language: python
   :caption: examples/gallery/jaynes_cummings.py

Quantum Trajectories
--------------------

Runs a finite Monte Carlo wave-function ensemble and plots the sample mean with
standard-error bands.

.. literalinclude:: ../examples/gallery/quantum_trajectory.py
   :language: python
   :caption: examples/gallery/quantum_trajectory.py

Phase-Space Distributions
-------------------------

Computes Wigner and Husimi-Q functions for a finite Fock-space cat state. This
example is pure Python and does not require loading the Julia backend.

.. literalinclude:: ../examples/gallery/phase_space.py
   :language: python
   :caption: examples/gallery/phase_space.py

Parameter Sweep
---------------

Runs a restartable sweep over a qubit decay rate, writing a manifest, one HDF5
result per point, ``summary.csv``, and ``summary.h5``.

.. literalinclude:: ../examples/gallery/parameter_sweep.py
   :language: python
   :caption: examples/gallery/parameter_sweep.py

