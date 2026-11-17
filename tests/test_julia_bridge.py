from __future__ import annotations

import json
import os
import platform
import sys
from pathlib import Path
from typing import Any

import pytest

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


def test_sysimage_metadata_is_ignored_when_unreadable_or_incomplete(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv(bridge.CACHE_DIR_ENV, str(tmp_path))
    path = bridge.sysimage_metadata_path()

    assert bridge.read_sysimage_metadata() is None
    path.write_text("{not json", encoding="utf-8")
    assert bridge.read_sysimage_metadata() is None
    path.write_text('["a list"]', encoding="utf-8")
    assert bridge.read_sysimage_metadata() is None
    path.write_text('{"schema_version": 1}', encoding="utf-8")
    assert bridge.read_sysimage_metadata() is None


def test_cache_dir_defaults_to_the_home_directory(monkeypatch: Any) -> None:
    monkeypatch.delenv(bridge.CACHE_DIR_ENV, raising=False)

    assert bridge.cache_dir() == Path.home() / ".cache" / "openquantumsim"


def test_explicit_sysimage_must_exist(monkeypatch: Any, tmp_path: Path) -> None:
    monkeypatch.setenv(bridge.SYSIMAGE_ENV, str(tmp_path / "missing.so"))
    assert bridge.configured_sysimage_path() is None

    present = tmp_path / "present.so"
    present.write_bytes(b"image")
    monkeypatch.setenv(bridge.SYSIMAGE_ENV, str(present))
    assert bridge.configured_sysimage_path() == present


def test_sysimage_can_be_switched_off(monkeypatch: Any, tmp_path: Path) -> None:
    monkeypatch.delenv(bridge.SYSIMAGE_ENV, raising=False)
    monkeypatch.setenv(bridge.USE_SYSIMAGE_ENV, "off")

    assert bridge.configured_sysimage_path() is None


def test_matching_metadata_with_a_deleted_sysimage_is_ignored(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv(bridge.SYSIMAGE_ENV, raising=False)
    monkeypatch.delenv(bridge.USE_SYSIMAGE_ENV, raising=False)
    monkeypatch.setenv(bridge.CACHE_DIR_ENV, str(tmp_path))
    monkeypatch.setattr(bridge, "_juliapkg_meta_matches", lambda _metadata: True)
    bridge.write_sysimage_metadata(
        _metadata(tmp_path / "deleted.so", bridge.backend_fingerprint()),
    )

    assert bridge.configured_sysimage_path() is None


def test_existing_juliacall_sysimage_setting_is_left_alone(monkeypatch: Any) -> None:
    monkeypatch.setenv(bridge.JULIACALL_SYSIMAGE_ENV, "/chosen/by/user.so")
    monkeypatch.setattr(
        bridge,
        "configured_sysimage_path",
        lambda: (_ for _ in ()).throw(AssertionError("must not be consulted")),
    )

    bridge._maybe_configure_sysimage()

    assert os.environ[bridge.JULIACALL_SYSIMAGE_ENV] == "/chosen/by/user.so"
    assert bridge.active_sysimage_path() == Path("/chosen/by/user.so")


def test_juliapkg_metadata_comparison(monkeypatch: Any, tmp_path: Path) -> None:
    juliapkg_state = pytest.importorskip("juliapkg.state")
    meta = tmp_path / "meta.json"
    monkeypatch.setitem(juliapkg_state.STATE, "meta", str(meta))
    recorded = _metadata(tmp_path / "image.so", "fingerprint")
    recorded["julia_executable"] = "/opt/julia/bin/julia"
    recorded["julia_version"] = "1.11.9"

    # Without juliapkg metadata there is nothing to contradict the record.
    assert bridge._juliapkg_meta_matches(recorded)
    meta.write_text("{broken", encoding="utf-8")
    assert bridge._juliapkg_meta_matches(recorded)
    meta.write_text(
        json.dumps({"executable": "/opt/julia/bin/julia", "version": "1.11.9"}),
        encoding="utf-8",
    )
    assert bridge._juliapkg_meta_matches(recorded)
    meta.write_text(
        json.dumps({"executable": "/opt/julia/bin/julia", "version": "1.12.0"}),
        encoding="utf-8",
    )
    assert not bridge._juliapkg_meta_matches(recorded)


def test_require_backend_names_the_caller(monkeypatch: Any) -> None:
    def unavailable(**_kwargs: object) -> None:
        raise bridge.JuliaBridgeUnavailable("no julia")

    monkeypatch.setattr(bridge, "load_backend", unavailable)

    with pytest.raises(bridge.JuliaBridgeUnavailable, match="mesolve requires"):
        bridge.require_backend("mesolve")
    # Existing code that catches NotImplementedError keeps working.
    with pytest.raises(NotImplementedError):
        bridge.require_backend("mesolve")


def test_backend_available_reports_load_failures(monkeypatch: Any) -> None:
    def unavailable(**_kwargs: object) -> None:
        raise bridge.JuliaBridgeUnavailable("no julia")

    monkeypatch.setattr(bridge, "load_backend", unavailable)

    assert bridge.backend_available() is False


def test_failed_backend_load_is_reported_with_its_path(monkeypatch: Any) -> None:
    # Fails the fast load, the load after instantiate, and the load after
    # resolve, so every recovery path has been tried.
    fake = _FakeJulia(fail_backend_loads=3)
    _install_fake_bridge(monkeypatch, fake)

    with pytest.raises(bridge.JuliaBridgeUnavailable, match="/fake/backend"):
        bridge.load_backend()


def test_julia_error_description_uses_the_julia_exception(monkeypatch: Any) -> None:
    class JuliaError(Exception):
        def __init__(self, message: str) -> None:
            super().__init__(message)
            self.exception = "inner"

    class Describer:
        def seval(self, expression: str) -> Any:
            if "nameof" in expression:
                return lambda _inner: "DimensionMismatch"
            return lambda _inner: "sizes differ"

    monkeypatch.setattr(bridge, "_JL", Describer())

    julia_type, message = bridge._describe_julia_error(JuliaError("ignored"))

    assert julia_type == "DimensionMismatch"
    assert message == "Julia backend error (DimensionMismatch): sizes differ"


def test_julia_error_description_falls_back_to_the_text(monkeypatch: Any) -> None:
    class JuliaError(Exception):
        exception = "inner"

    class Broken:
        def seval(self, _expression: str) -> Any:
            raise RuntimeError("julia is gone")

    monkeypatch.setattr(bridge, "_JL", Broken())

    julia_type, message = bridge._describe_julia_error(
        JuliaError("ArgumentError: bad\nStacktrace:\n [1] f"),
    )

    assert julia_type is None
    assert message == "Julia backend error: ArgumentError: bad"
