Quick Start
===========

Install OpenQuantumSim from PyPI:

.. code-block:: bash

   python -m pip install openquantumsim

OpenQuantumSim uses JuliaCall to load the packaged Julia backend. The first
solver call on a new machine may spend a few minutes resolving and precompiling
Julia packages.

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

Development Install
-------------------

For local development from source:

.. code-block:: bash

   git clone https://github.com/mohammadjafariph/OpenQuantumSimulation.git
   cd OpenQuantumSimulation
   python -m pip install -e ".[dev]"
   python setup_julia.py

Run the Python and Julia tests with:

.. code-block:: bash

   python -m pytest
   julia --project=src/OpenQuantumSimJL -e 'using Pkg; Pkg.test()'
