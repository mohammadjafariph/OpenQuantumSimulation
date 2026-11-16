"""Backend-free tests of the validation report machinery."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import openquantumsim.cli as cli
import openquantumsim.validation as validation


def _case(name: str, run: Any, *, tolerance: float = 1e-6) -> validation.ValidationCase:
    return validation.ValidationCase(
        name=name,
        title=f"Title of {name}",
        category="dynamics",
        solver="mesolve",
        reference="closed formula",
        metric="max abs error",
        tolerance=tolerance,
        run=run,
    )


def _raise(exc: Exception) -> Any:
    def run() -> tuple[float, str]:
        raise exc

    return run


FAKE_CASES = [
    _case("good", lambda: (1e-9, "")),
    _case("bad", lambda: (1e-3, "too far")),
    _case("not_finite", lambda: (float("nan"), "")),
    _case("broken", _raise(RuntimeError("boom"))),
    _case("optional", _raise(validation.ValidationSkipped("QuTiP is not installed"))),
]


@pytest.fixture
def fake_suite(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(validation, "validation_cases", lambda: list(FAKE_CASES))
    monkeypatch.setattr(validation, "_environment", lambda: {"Julia": "test"})


def test_case_names_are_unique_and_tolerances_positive() -> None:
    cases = validation.validation_cases()
    names = [case.name for case in cases]

    assert len(cases) >= 10
    assert len(names) == len(set(names))
    assert all(case.tolerance > 0 for case in cases)
    assert {case.category for case in cases} <= set(validation._CATEGORY_TITLES)


def test_run_case_classifies_every_outcome() -> None:
    statuses = {case.name: validation.run_case(case) for case in FAKE_CASES}

    assert statuses["good"].status == "passed"
    assert statuses["good"].passed
    assert statuses["bad"].status == "failed"
    assert statuses["bad"].measured == 1e-3
    # A NaN measurement must never count as meeting a tolerance.
    assert statuses["not_finite"].status == "failed"
    assert statuses["broken"].status == "error"
    assert statuses["broken"].detail == "RuntimeError: boom"
    assert statuses["optional"].status == "skipped"
    assert statuses["optional"].measured is None


def test_report_fails_on_failure_or_error_but_not_on_skip(fake_suite: None) -> None:
    full = validation.run_validation()
    assert not full.passed
    assert full.counts() == {"passed": 1, "failed": 2, "error": 1, "skipped": 1}

    partial = validation.run_validation(["good", "optional"])
    assert partial.passed
    assert [outcome.case.name for outcome in partial.outcomes] == ["good", "optional"]


def test_run_validation_rejects_unknown_names(fake_suite: None) -> None:
    with pytest.raises(ValueError, match="unknown validation case.*missing"):
        validation.run_validation(["good", "missing"])


def test_run_validation_reports_progress(fake_suite: None) -> None:
    seen: list[str] = []

    validation.run_validation(
        ["good", "bad"],
        progress=lambda outcome: seen.append(outcome.case.name),
    )

    assert seen == ["good", "bad"]


def test_report_is_written_as_markdown_and_json(
    fake_suite: None,
    tmp_path: Path,
) -> None:
    report = validation.run_validation()

    markdown_path, json_path = report.write(tmp_path / "report")

    markdown = markdown_path.read_text(encoding="utf-8")
    assert "**Result: FAILED**" in markdown
    assert "Title of good" in markdown
    assert "too far" in markdown
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data["format"] == "openquantumsim.validation"
    assert data["passed"] is False
    assert data["environment"] == {"Julia": "test"}
    assert [case["status"] for case in data["cases"]] == [
        "passed",
        "failed",
        "failed",
        "error",
        "skipped",
    ]


def test_cli_validate_list_prints_every_case(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert cli.main(["validate", "--list"]) == 0

    output = capsys.readouterr().out
    for case in validation.validation_cases():
        assert case.name in output


def test_cli_validate_exit_code_follows_the_report(
    fake_suite: None,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    passing = cli.main(["validate", "--case", "good", "--output-dir", str(tmp_path)])
    assert passing == 0
    assert (tmp_path / "validation_report.md").is_file()
    assert "Validation PASSED" in capsys.readouterr().out

    assert cli.main(["validate", "--case", "bad"]) == 1
    assert cli.main(["validate", "--case", "missing"]) == 2
