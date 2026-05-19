Qubit Decay in Five Minutes
===========================

This tutorial solves the simplest useful Lindblad problem: a two-level atom
spontaneously emitting into its environment. It is a good first check that the
Python frontend, Julia backend, observables, and result object are all working.

The model is

.. math::

   \dot{\rho} =
   -i[H,\rho] + \gamma\left(
       \sigma_- \rho \sigma_+ -
       \frac{1}{2}\{\sigma_+\sigma_-, \rho\}
   \right),

with initial state ``|up>``. The excited-state population has the analytic
solution ``exp(-gamma * t)``.

Full Script
-----------

Save this as ``qubit_decay.py`` and run it with Python. The first execution on
a new machine may spend a few minutes precompiling Julia dependencies.

.. code-block:: python

   from pathlib import Path

   import matplotlib.pyplot as plt
   import numpy as np
   import openquantumsim as oqs


   atom = oqs.SpinSpace(0.5, label="atom")
   excited = oqs.basis(atom, "up")

   gamma = 0.2
   times = np.linspace(0.0, 10.0, 201)

   H = 0.0 * oqs.sigmaz(atom)
   rho0 = oqs.ket2dm(excited)
   collapse = np.sqrt(gamma) * oqs.sigmam(atom)
   projector = oqs.Operator(oqs.ket2dm(excited), atom, "P_excited")

   result = oqs.mesolve(
       H,
       rho0,
       times,
       c_ops=[collapse],
       e_ops=[projector],
       options=oqs.Options(rtol=1e-8, atol=1e-10),
   )

   population = result.expect[0].real
   analytic = np.exp(-gamma * times)
   max_error = np.max(np.abs(population - analytic))

   print(f"Maximum population error: {max_error:.3e}")
   assert max_error < 2e-6

   Path("runs").mkdir(exist_ok=True)
   result.save_hdf5("runs/qubit_decay.h5")

   fig, ax = plt.subplots(figsize=(6.0, 3.8))
   ax.plot(times, population, label="OpenQuantumSim")
   ax.plot(times, analytic, "--", label="analytic")
   ax.set_xlabel("time")
   ax.set_ylabel("excited-state population")
   ax.legend()
   fig.tight_layout()
   fig.savefig("runs/qubit_decay.png", dpi=160)

Step by Step
------------

``SpinSpace(0.5)`` creates a two-dimensional Hilbert space. The basis state
``"up"`` is used as the initially excited state, and ``sigmam`` is the lowering
operator that appears in the collapse channel.

.. code-block:: python

   atom = oqs.SpinSpace(0.5, label="atom")
   excited = oqs.basis(atom, "up")
   collapse = np.sqrt(gamma) * oqs.sigmam(atom)

Expectation values are requested through ``e_ops``. Here the observable is the
projector ``|up><up|``, so ``result.expect[0]`` is the excited-state population
sampled at the requested time points.

.. code-block:: python

   projector = oqs.Operator(oqs.ket2dm(excited), atom, "P_excited")
   result = oqs.mesolve(H, rho0, times, c_ops=[collapse], e_ops=[projector])
   population = result.expect[0].real

The solver returns a :class:`openquantumsim.Result`. Results can be saved to
HDF5 for later analysis or to attach solver outputs to a parameter sweep.

.. code-block:: python

   result.save_hdf5("runs/qubit_decay.h5")
   loaded = oqs.load_result("runs/qubit_decay.h5")

Common Variations
-----------------

To add coherent precession, use a nonzero Hamiltonian:

.. code-block:: python

   omega = 1.0
   H = 0.5 * omega * oqs.sigmaz(atom)

To estimate trajectory-level sampling error, solve the same model with Monte
Carlo wave-function trajectories:

.. code-block:: python

   mc = oqs.mcsolve(
       H,
       excited,
       times,
       c_ops=[collapse],
       e_ops=[projector],
       n_traj=1000,
       options=oqs.Options(seed=2026, max_step=0.01),
   )

   mean_population = mc.expect[0].real
   standard_error = mc.expect_stderr[0]

The deterministic solution is the best starting point for small systems. Use
``mcsolve`` when the Hilbert space is too large for density-matrix propagation
or when individual quantum trajectories are part of the analysis.
