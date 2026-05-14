Fresh-Clone Quickstart Validation
=================================

This page records the first clean quickstart test against the public GitHub
repository. It is a release-readiness check: a new user should be able to clone
the repository, create a virtual environment, install OpenQuantumSim, initialize
the Julia backend, and run a minimal physics example.

Validation Environment
----------------------

.. list-table::
   :header-rows: 1

   * - Item
     - Value
   * - Date
     - 2026-05-14
   * - Repository
     - ``https://github.com/mohammadjafariph/OpenQuantumSimulation.git``
   * - Commit tested
     - ``03cfa64``
   * - Temporary clone
     - ``/private/tmp/oqs-quickstart-20260514c``
   * - CPU
     - Apple M1
   * - Python
     - 3.14.3
   * - Julia runtime selected by JuliaCall
     - 1.11.9

Commands
--------

.. code-block:: bash

   git clone --depth 1 \
       https://github.com/mohammadjafariph/OpenQuantumSimulation.git \
       /private/tmp/oqs-quickstart-20260514c

   cd /private/tmp/oqs-quickstart-20260514c
   python3 -m venv .venv
   .venv/bin/python -m pip install -e .
   .venv/bin/python setup_julia.py

The first backend setup precompiled the Julia SciML stack. On the Apple M1 test
machine this took about two minutes and ended with:

.. code-block:: text

   Julia backend ready: /private/tmp/oqs-quickstart-20260514c/src/OpenQuantumSimJL

Smoke Test
----------

The smoke test solved spontaneous emission for a two-level system and compared
the excited-state population against ``exp(-gamma * t)``:

.. code-block:: python

   import numpy as np
   import openquantumsim as oqs

   atom = oqs.SpinSpace(0.5, label="atom")
   H = 0.0 * oqs.sigmaz(atom)
   excited = oqs.basis(atom, "up")
   rho0 = oqs.ket2dm(excited)

   gamma = 0.2
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

Observed output:

.. code-block:: text

   [1.0, 0.9801986733067711, 0.9607894391523246]
   [1.0, 0.9801986733067553, 0.9607894391523232]

Result
------

The fresh-clone quickstart passed. The only noteworthy first-run cost is Julia
package precompilation, which is expected and should be mentioned to early
users.
