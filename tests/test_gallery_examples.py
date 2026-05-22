from __future__ import annotations

from pathlib import Path

import pytest

from examples.gallery import phase_space


def test_phase_space_gallery_example_runs_without_backend(tmp_path: Path) -> None:
    summary = phase_space.run_example(
        output_dir=tmp_path / "phase_space",
        fast=True,
        save_plots=True,
    )

    assert summary["example"] == "phase_space"
    assert summary["grid_points"] == 31
    assert (tmp_path / "phase_space" / "phase_space.npz").is_file()
    assert (tmp_path / "phase_space" / "phase_space.png").is_file()


@pytest.mark.physics
def test_backend_gallery_examples_run_in_fast_mode(tmp_path: Path) -> None:
    import openquantumsim as oqs
    from examples.gallery import (
        driven_qubit,
        jaynes_cummings,
        parameter_sweep,
        quantum_trajectory,
        qubit_decay,
    )
    from openquantumsim._julia_bridge import backend_available

    if not backend_available():
        pytest.skip("Julia backend is not available.")

    modules = [
        qubit_decay,
        driven_qubit,
        jaynes_cummings,
        quantum_trajectory,
        parameter_sweep,
    ]
    for module in modules:
        summary = module.run_example(
            output_dir=tmp_path / str(module.__name__).split(".")[-1],
            fast=True,
            save_plots=True,
        )
        assert isinstance(summary["example"], str)
        assert Path(str(summary["output_dir"])).is_dir()

    assert oqs.__version__

