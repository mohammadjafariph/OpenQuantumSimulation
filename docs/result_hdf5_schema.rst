Result HDF5 Schema
==================

OpenQuantumSim writes solver results with :meth:`openquantumsim.Result.save_hdf5`.
The on-disk format is intended to be stable across patch releases and readable
from Python, Julia, HDF5 command-line tools, and analysis notebooks.

Solver Result Files
-------------------

Every result file has these root attributes:

.. list-table::
   :header-rows: 1

   * - Attribute
     - Value
   * - ``format``
     - ``openquantumsim.result``
   * - ``format_version``
     - ``1``

The root datasets are:

.. list-table::
   :header-rows: 1

   * - Path
     - Shape
     - Type
     - Meaning
   * - ``/times``
     - ``(n_times,)``
     - ``float64``
     - Time grid requested by the solver.
   * - ``/expect``
     - ``(n_e_ops, n_times)``
     - ``complex128``
     - Mean expectation values for each ``e_ops`` operator.
   * - ``/expect_std``
     - ``(n_e_ops, n_times)``
     - ``float64``
     - Per-time ensemble standard deviation for Monte Carlo expectation values.
       Empty for deterministic solvers.
   * - ``/expect_stderr``
     - ``(n_e_ops, n_times)``
     - ``float64``
     - Standard error of the Monte Carlo mean. Empty for deterministic solvers.
   * - ``/entropy``
     - ``(n_times,)``
     - ``float64``
     - Optional entropy series when a solver computes it.
   * - ``/states``
     - ``(n_times, d)`` or ``(n_times, d, d)``
     - ``complex128``
     - Optional saved kets or density matrices.

When no expectation operators are supplied, the expectation datasets are
present with shape ``(0, n_times)``.

State Observable Groups
-----------------------

State observables are named scalar diagnostics produced by
``state_observables=...``. Dataset names are the observable names, so names must
be non-empty strings and cannot contain ``/``.

.. list-table::
   :header-rows: 1

   * - Path
     - Shape
     - Type
     - Meaning
   * - ``/state_observables/<name>``
     - ``(n_times,)``
     - ``complex128``
     - Diagnostic mean or deterministic value.
   * - ``/state_observables_std/<name>``
     - ``(n_times,)``
     - ``float64``
     - Monte Carlo standard deviation for backend-aggregated diagnostics.
   * - ``/state_observables_stderr/<name>``
     - ``(n_times,)``
     - ``float64``
     - Monte Carlo standard error for backend-aggregated diagnostics.

For ``mesolve`` and ``single_trajectory`` callback diagnostics, only
``/state_observables`` is normally present. For ``mcsolve`` built-in diagnostics,
all three groups are written when the values are available.

Solver Statistics
-----------------

``/solver_stats`` is a group whose attributes store solver metadata such as
``retcode``, ``n_traj``, ``max_step``, ``wall_time``, checkpoint information, and
backend options. Each key may have a paired ``<key>__encoding`` attribute with
one of these values:

.. list-table::
   :header-rows: 1

   * - Encoding
     - Meaning
   * - ``native``
     - Attribute is a native HDF5 scalar or string.
   * - ``json``
     - Attribute is JSON text and should be decoded with ``json.loads``.
   * - ``none``
     - Attribute represents Python ``None``.

Loading Results
---------------

Use :func:`openquantumsim.load_result` for normal reads:

.. code-block:: python

   import openquantumsim as oqs

   result = oqs.load_result("runs/qubit_decay.h5")
   population = result.expect[0].real

Direct HDF5 access is also straightforward:

.. code-block:: python

   import h5py

   with h5py.File("runs/qubit_decay.h5", "r") as handle:
       times = handle["times"][:]
       first_expectation = handle["expect"][0, :]

Parameter Sweep Output
----------------------

When :meth:`openquantumsim.ParameterSweep.run` receives ``output_dir``, it
writes a restartable directory:

.. code-block:: text

   output_dir/
     manifest.json
     results/
       point_0000_....h5
       point_0001_....h5
     summary.csv
     summary.h5

``manifest.json`` has ``format = "openquantumsim.sweep"``,
``format_version = "1"``, the sweep ``config``, ``created_at``, and a ``points``
list. Each point records its ``id``, ``index``, parameter values, status,
timestamps, optional error text, result file path, and summary row.

``summary.h5`` has root attributes:

.. list-table::
   :header-rows: 1

   * - Attribute
     - Value
   * - ``format``
     - ``openquantumsim.sweep.summary``
   * - ``format_version``
     - ``1``

Each summary column is a root dataset. Numeric and boolean columns are stored as
native HDF5 arrays; mixed or structured values are stored as UTF-8 strings using
the same values that appear in ``summary.csv``.

Compatibility Notes
-------------------

Version ``1`` result files written before backend-side ``mcsolve`` diagnostics
may not contain ``/state_observables_std`` or ``/state_observables_stderr``.
``openquantumsim.load_result`` treats missing diagnostic uncertainty groups as
empty dictionaries.
