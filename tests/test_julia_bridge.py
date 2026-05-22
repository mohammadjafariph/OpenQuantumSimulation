from __future__ import annotations

from pathlib import Path
from typing import Any

import openquantumsim._julia_bridge as bridge


class _FakePkg:
    def __init__(self) -> None:
        self.activated: list[str] = []
        self.instantiate_calls = 0
        self.resolve_calls = 0

    def activate(self, path: str, **_kwargs: object) -> None:
        self.activated.append(path)

    def instantiate(self, **_kwargs: object) -> None:
        self.instantiate_calls += 1

    def resolve(self, **_kwargs: object) -> None:
        self.resolve_calls += 1


class _FakeJulia:
    def __init__(self, *, fail_backend_loads: int = 0) -> None:
        self.Pkg = _FakePkg()
        self.OpenQuantumSimJL = object()
        self.devnull = object()
        self.fail_backend_loads = fail_backend_loads
        self.seval_calls: list[str] = []

    def seval(self, expression: str) -> None:
        self.seval_calls.append(expression)
        if expression == "using OpenQuantumSimJL" and self.fail_backend_loads > 0:
            self.fail_backend_loads -= 1
            raise RuntimeError("backend unavailable")


def _install_fake_bridge(monkeypatch: Any, fake: _FakeJulia) -> None:
    monkeypatch.setattr(bridge, "_BACKEND", None)
    monkeypatch.setattr(bridge, "get_julia", lambda: fake)
    monkeypatch.setattr(bridge, "backend_path", lambda: Path("/fake/backend"))


def test_load_backend_skips_instantiate_on_available_backend(monkeypatch: Any) -> None:
    fake = _FakeJulia()
    _install_fake_bridge(monkeypatch, fake)

    loaded = bridge.load_backend()

    assert loaded is fake.OpenQuantumSimJL
    assert fake.Pkg.activated == ["/fake/backend"]
    assert fake.Pkg.instantiate_calls == 0
    assert fake.Pkg.resolve_calls == 0
    assert fake.seval_calls == ["using Pkg", "using OpenQuantumSimJL"]


def test_load_backend_instantiates_after_failed_fast_load(monkeypatch: Any) -> None:
    fake = _FakeJulia(fail_backend_loads=1)
    _install_fake_bridge(monkeypatch, fake)

    loaded = bridge.load_backend()

    assert loaded is fake.OpenQuantumSimJL
    assert fake.Pkg.instantiate_calls == 1
    assert fake.Pkg.resolve_calls == 0
    assert fake.seval_calls == [
        "using Pkg",
        "using OpenQuantumSimJL",
        "using OpenQuantumSimJL",
    ]


def test_load_backend_can_force_instantiate(monkeypatch: Any) -> None:
    fake = _FakeJulia()
    _install_fake_bridge(monkeypatch, fake)

    loaded = bridge.load_backend(instantiate=True)

    assert loaded is fake.OpenQuantumSimJL
    assert fake.Pkg.instantiate_calls == 1
    assert fake.seval_calls == ["using Pkg", "using OpenQuantumSimJL"]
