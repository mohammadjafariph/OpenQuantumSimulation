from __future__ import annotations

import os
import platform
import sys
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


def test_configure_sysimage_from_explicit_env(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    sysimage = tmp_path / "OpenQuantumSimJL_sysimage.so"
    sysimage.write_bytes(b"sysimage")
    monkeypatch.delenv(bridge.JULIACALL_SYSIMAGE_ENV, raising=False)
    monkeypatch.setenv(bridge.SYSIMAGE_ENV, str(sysimage))

    bridge._maybe_configure_sysimage()

    assert os.environ[bridge.JULIACALL_SYSIMAGE_ENV] == str(sysimage)


def test_configure_sysimage_from_matching_metadata(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    sysimage = tmp_path / "OpenQuantumSimJL_sysimage.so"
    sysimage.write_bytes(b"sysimage")
    monkeypatch.delenv(bridge.JULIACALL_SYSIMAGE_ENV, raising=False)
    monkeypatch.delenv(bridge.SYSIMAGE_ENV, raising=False)
    monkeypatch.setenv(bridge.CACHE_DIR_ENV, str(tmp_path))
    monkeypatch.setattr(bridge, "_juliapkg_meta_matches", lambda _metadata: True)

    metadata = _metadata(sysimage, bridge.backend_fingerprint())
    bridge.write_sysimage_metadata(metadata)

    bridge._maybe_configure_sysimage()

    assert os.environ[bridge.JULIACALL_SYSIMAGE_ENV] == str(sysimage)


def test_configure_sysimage_ignores_stale_metadata(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    sysimage = tmp_path / "OpenQuantumSimJL_sysimage.so"
    sysimage.write_bytes(b"sysimage")
    monkeypatch.delenv(bridge.JULIACALL_SYSIMAGE_ENV, raising=False)
    monkeypatch.delenv(bridge.SYSIMAGE_ENV, raising=False)
    monkeypatch.setenv(bridge.CACHE_DIR_ENV, str(tmp_path))
    monkeypatch.setattr(bridge, "_juliapkg_meta_matches", lambda _metadata: True)

    bridge.write_sysimage_metadata(_metadata(sysimage, "stale"))

    bridge._maybe_configure_sysimage()

    assert bridge.JULIACALL_SYSIMAGE_ENV not in os.environ


def _metadata(sysimage: Path, fingerprint: str) -> bridge.SysimageMetadata:
    return {
        "schema_version": 1,
        "sysimage_path": str(sysimage),
        "backend_path": str(bridge.backend_path()),
        "backend_fingerprint": fingerprint,
        "julia_executable": "",
        "julia_version": "",
        "platform": sys.platform,
        "machine": platform.machine(),
        "created_at": "2026-05-26T00:00:00+00:00",
    }
