Publishing
==========

OpenQuantumSim releases are built by ``.github/workflows/release.yml``. The
workflow always builds the source distribution and wheel, runs ``twine check``,
and uploads the artifacts to the matching GitHub Release. Publishing to package
indexes is manually gated through PyPI trusted publishing.

Current Release State
---------------------

``v0.1.0a0`` is tagged and the GitHub Release contains both distribution
artifacts. TestPyPI trusted publishing was configured successfully after an
initial setup miss:

.. code-block:: text

   invalid-publisher: valid token, but no corresponding publisher

That meant GitHub generated a valid OIDC token, but TestPyPI did not yet have
a trusted publisher matching this repository and workflow. The matching
TestPyPI publisher is now configured.

A later TestPyPI attempt reached the upload step but PyPI rejected the
``v0.1.0a0`` metadata because ``Programming Language :: Julia`` is not a valid
trove classifier. ``v0.1.0a1`` removes that classifier and is the first package
index candidate.

Trusted Publisher Settings
--------------------------

Configure these on TestPyPI first. If the project does not exist yet, create a
pending publisher for the project name ``openquantumsim``.

.. list-table::
   :header-rows: 1

   * - Field
     - Value
   * - Project name
     - ``openquantumsim``
   * - Owner
     - ``mohammadjafariph``
   * - Repository
     - ``OpenQuantumSimulation``
   * - Workflow
     - ``release.yml``
   * - Environment
     - ``testpypi``

After TestPyPI succeeds, configure the same trusted publisher on PyPI with the
environment set to ``pypi``.

TestPyPI Publish
----------------

Run the release workflow manually for the existing tag:

.. code-block:: bash

   gh workflow run release.yml \
       -f tag=v0.1.0a1 \
       -f publish_target=testpypi \
       --repo mohammadjafariph/OpenQuantumSimulation

Then verify installation from TestPyPI:

.. code-block:: bash

   python scripts/check_index_install.py \
       --index testpypi \
       --version 0.1.0a1

The script installs the published package into a fresh virtual environment,
loads the packaged Julia backend from ``site-packages``, and runs a tiny
spontaneous-emission ``mesolve`` smoke test.

PyPI Publish
------------

Only publish to PyPI after TestPyPI installation passes:

.. code-block:: bash

   gh workflow run release.yml \
       -f tag=v0.1.0a1 \
       -f publish_target=pypi \
       --repo mohammadjafariph/OpenQuantumSimulation

Then verify installation from PyPI:

.. code-block:: bash

   python scripts/check_index_install.py \
       --index pypi \
       --version 0.1.0a1
