import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_docs_api_pages_exist() -> None:
    api_pages = [
        "correlations",
        "hilbert",
        "observables",
        "operators",
        "phase_space",
        "plotting",
        "results",
        "solvers",
        "sweeps",
        "systems",
        "timedep",
    ]

    for page in api_pages:
        path = ROOT / "docs" / "api" / f"{page}.rst"
        text = path.read_text(encoding="utf-8")
        assert ".. automodule:: openquantumsim." in text


def test_tutorial_notebooks_are_valid_and_clear() -> None:
    notebooks = sorted((ROOT / "docs" / "tutorials").glob("*.ipynb"))

    assert [path.name for path in notebooks] == [
        "01_qubit_decay_mesolve.ipynb",
        "02_parameter_sweep.ipynb",
        "03_phase_space_plotting.ipynb",
        "04_state_metrics_observables.ipynb",
    ]
    for path in notebooks:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        assert notebook["nbformat"] == 4
        assert notebook["cells"]
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                assert cell["execution_count"] is None
                assert cell["outputs"] == []


def test_validation_gallery_documents_reference_commands() -> None:
    text = (ROOT / "docs" / "validation.rst").read_text(encoding="utf-8")

    assert "Analytic Qubit Decay" in text
    assert "Jaynes-Cummings Against QuTiP" in text
    assert "benchmarks/bench_vs_qutip.py" in text
    assert "scripts/validate_jaynes_cummings_qutip.py" in text


def test_result_hdf5_schema_documents_current_groups() -> None:
    text = (ROOT / "docs" / "result_hdf5_schema.rst").read_text(encoding="utf-8")

    assert "openquantumsim.result" in text
    assert "/state_observables_std/<name>" in text
    assert "openquantumsim.sweep.summary" in text
