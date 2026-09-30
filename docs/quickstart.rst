Quick Start
===========

Install OpenQuantumSim from PyPI:

.. code-block:: bash

   python -m pip install openquantumsim

OpenQuantumSim uses JuliaCall to load the packaged Julia backend. The first
solver call on a new machine may spend a few minutes resolving and precompiling
Julia packages.

Backend Setup and Startup Speed
-------------------------------

Run the backend setup once after installing OpenQuantumSim:

.. code-block:: bash

   oqs setup-julia

This moves Julia package resolution and precompilation out of the first solver
call. For repeated simulation sessions, build a local Julia sysimage:

.. code-block:: bash

   oqs build-sysimage

The sysimage build can take several minutes, but future solver calls
automatically reuse it through JuliaCall. Rebuild it after upgrading Julia or
OpenQuantumSim. Set ``OPENQUANTUMSIM_USE_SYSIMAGE=0`` to disable automatic
sysimage use for a process, or set ``OPENQUANTUMSIM_JULIA_SYSIMAGE`` to point
at a custom sysimage.

Spontaneous Emission
--------------------

This example solves spontaneous emission for a two-level system and compares
the excited-state population with the analytic result.

.. code-block:: python

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

Beyond the Basics
-----------------

Weak system-bath coupling with the secular Bloch-Redfield equation, where each
bath coupling pairs an operator with a spectrum callable:

.. code-block:: python

   kappa = 0.4
   zero_temperature = lambda omega: kappa if omega < 0 else 0.0

   br = oqs.brmesolve(
       H,
       rho0,
       times,
       a_ops=[(oqs.sigmam(atom), zero_temperature)],
       e_ops=[projector],
   )

A constant spectrum ``gamma = g`` is equivalent to a Lindblad collapse operator
``sqrt(g) * A``, which makes BR results easy to check against ``mesolve``.

Entanglement and correlation diagnostics work on plain NumPy states, so they
can be applied to any saved density matrix:

.. code-block:: python

   conc = oqs.concurrence(rho)                      # two-qubit states
   neg = oqs.negativity(rho, (2, 2), 0, 1)          # any bipartite cut
   taus = np.linspace(0.0, 10.0, 201)
   wlist, spectrum = oqs.spectrum_2op_1t(
       H, rho0, taus, oqs.sigmap(atom), oqs.sigmam(atom), c_ops=[collapse],
   )

Development Install
-------------------

For local development from source:

.. code-block:: bash

   git clone https://github.com/mohammadjafariph/OpenQuantumSimulation.git
   cd OpenQuantumSimulation
   python -m pip install -e ".[dev]"
   oqs setup-julia

Run the Python and Julia tests with:

.. code-block:: bash

   python -m pytest
   julia --project=src/OpenQuantumSimJL -e 'using Pkg; Pkg.test()'
