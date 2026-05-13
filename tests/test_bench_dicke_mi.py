from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest


def _load_benchmark() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "examples" / "dicke" / "bench_mi.py"
    spec = importlib.util.spec_from_file_location("examples.dicke.bench_mi", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_signature_and_duration_format() -> None:
    bench = _load_benchmark()
    result = bench.mi_runner.BatchResult(
        start=0,
        stop=2,
        mi_a=np.ones((2, 3), dtype=np.float64),
        mi_b=2.0 * np.ones((2, 3), dtype=np.float64),
    )

    assert bench._signature([result]) == 18.0
    assert bench._format_duration(5.25) == "5.2s"
    assert bench._format_duration(65.0) == "1m 5.0s"
    assert bench._format_duration(3700.0) == "1h 1m"


def test_benchmark_mode_uses_batches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bench = _load_benchmark()
    calls: list[tuple[int, int]] = []

    def fake_run_requests(requests: list[Any], worker_count: int) -> list[Any]:
        calls.append((len(requests), worker_count))
        results = []
        for request in requests:
            count = request.stop - request.start
            values = np.full((count, len(request.times)), request.start + 1.0)
            results.append(
                bench.mi_runner.BatchResult(
                    start=request.start,
                    stop=request.stop,
                    mi_a=values,
                    mi_b=2.0 * values,
                ),
            )
        return results

    monkeypatch.setattr(bench, "_run_requests", fake_run_requests)
    args = argparse.Namespace(
        N=2,
        batch_size=2,
        kappa=0.1,
        max_step=0.05,
        n_traj=5,
        repeats=2,
        seed=2026,
    )
    times = np.linspace(0.0, 0.2, 3)

    row, signature = bench._benchmark_mode(args, times, 2, None)

    assert calls == [(3, 2), (3, 2)]
    assert row["n_jobs"] == 2
    assert row["workers"] == 2
    assert row["trajectories_per_second"] > 0
    assert row["seconds_per_trajectory"] > 0
    assert signature > 0
