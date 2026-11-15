"""Every validation case must meet its tolerance.

The cases are the ones behind ``oqs validate``, so the test suite and the
published validation report can never disagree.
"""

from __future__ import annotations

import pytest

import openquantumsim.validation as validation

CASES = validation.validation_cases()


@pytest.mark.physics
@pytest.mark.parametrize("case", CASES, ids=[case.name for case in CASES])
def test_validation_case_meets_tolerance(case: validation.ValidationCase) -> None:
    outcome = validation.run_case(case)

    if outcome.status == "skipped":
        pytest.skip(outcome.detail)
    assert outcome.status == "passed", (
        f"{case.title} ({case.solver}): {outcome.status}; measured "
        f"{outcome.measured} against tolerance {case.tolerance}. {outcome.detail}"
    )
