Errors and Warnings
===================

Solvers report a result they cannot vouch for instead of returning it
silently. A failed solve raises; a suspicious one warns. Pass
``Options(strict=False)`` to turn a failure into a warning and keep the
partial result.

.. automodule:: openquantumsim.errors
   :members:
   :show-inheritance:
