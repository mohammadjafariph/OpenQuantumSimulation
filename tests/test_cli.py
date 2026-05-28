from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import openquantumsim.cli as cli


def test_setup_julia_command_loads_backend(monkeypatch: Any, capsys: Any) -> None:
    calls: list[bool] = []
    monkeypatch.setattr(cli, "backend_path", lambda: Path("/fake/backend"))
    monkeypatch.setattr(
        cli,
        "load_backend",
        lambda *, instantiate=False: calls.append(instantiate),
    )

    assert cli.main(["setup-julia"]) == 0

    assert calls == [True]
    assert "Julia backend ready: /fake/backend" in capsys.readouterr().out


def test_build_sysimage_refuses_existing_output(
    monkeypatch: Any,
    tmp_path: Path,
    capsys: Any,
) -> None:
    sysimage = tmp_path / "OpenQuantumSimJL_sysimage.so"
    sysimage.write_bytes(b"existing")
    monkeypatch.setattr(cli, "_juliapkg_executable", lambda: "/usr/bin/julia")
    monkeypatch.setattr(
        cli,
        "_julia_query",
        lambda _julia, expression: "so" if "Libdl.dlext" in expression else "1.11.9",
    )

    status = cli.main(["build-sysimage", "--output", str(sysimage)])

    assert status == 2
    assert "Refusing to overwrite existing sysimage" in capsys.readouterr().err


def test_packagecompiler_script_includes_backend_and_output(tmp_path: Path) -> None:
    script = cli._packagecompiler_script(
        backend=Path("/backend"),
        build_project=tmp_path / "project",
        sysimage=tmp_path / "sysimage.so",
        cpu_target="generic",
        precompile_file=tmp_path / "precompile.jl",
    )

    assert 'Pkg.develop(Pkg.PackageSpec(path="/backend"))' in script
    assert f'sysimage_path="{tmp_path / "sysimage.so"}"' in script
    assert f'precompile_execution_file="{tmp_path / "precompile.jl"}' in script
    assert 'cpu_target="generic"' in script


def test_validate_sysimage_uses_juliacall_subprocess(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    calls: list[dict[str, Any]] = []
    sysimage = tmp_path / "sysimage.so"

    def fake_run(
        command: list[str],
        *,
        check: bool,
        env: dict[str, str],
    ) -> None:
        calls.append({"command": command, "check": check, "env": env})

    monkeypatch.delenv(cli.JULIACALL_SYSIMAGE_ENV, raising=False)
    monkeypatch.delenv(cli.USE_SYSIMAGE_ENV, raising=False)
    monkeypatch.delenv("JULIA_PYTHONCALL_EXE", raising=False)
    monkeypatch.setattr(cli.subprocess, "run", fake_run)

    cli._validate_sysimage(sysimage)

    assert len(calls) == 1
    assert calls[0]["command"][:2] == [sys.executable, "-c"]
    assert calls[0]["check"] is True
    env = calls[0]["env"]
    assert env[cli.JULIACALL_SYSIMAGE_ENV] == str(sysimage)
    assert env[cli.USE_SYSIMAGE_ENV] == "0"
    assert env["JULIA_PYTHONCALL_EXE"] == sys.executable
    assert env["PYTHON_JULIACALL_HANDLE_SIGNALS"] == "yes"
