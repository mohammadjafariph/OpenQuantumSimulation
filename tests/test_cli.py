from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import openquantumsim._julia_bridge as bridge
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
    assert f'sysimage_path="{(tmp_path / "sysimage.so").as_posix()}"' in script
    precompile_path = (tmp_path / "precompile.jl").as_posix()
    assert f'precompile_execution_file="{precompile_path}' in script
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


def _fake_julia_toolchain(monkeypatch: Any, calls: dict[str, Any]) -> None:
    monkeypatch.setattr(cli, "_juliapkg_executable", lambda: "/usr/bin/julia")
    monkeypatch.setattr(
        cli,
        "_julia_query",
        lambda _julia, expression: "so" if "Libdl.dlext" in expression else "1.11.9",
    )

    def fake_run_julia(julia: str, script: str) -> None:
        calls["julia"] = julia
        calls["script"] = script

    monkeypatch.setattr(cli, "_run_julia", fake_run_julia)
    monkeypatch.setattr(
        cli,
        "_validate_sysimage",
        lambda sysimage: calls.setdefault("validated", sysimage),
    )


def test_build_sysimage_builds_validates_and_registers(
    monkeypatch: Any,
    tmp_path: Path,
    capsys: Any,
) -> None:
    calls: dict[str, Any] = {}
    _fake_julia_toolchain(monkeypatch, calls)
    monkeypatch.setenv(bridge.CACHE_DIR_ENV, str(tmp_path / "cache"))
    sysimage = tmp_path / "out" / "image.so"

    status = cli.main(
        ["build-sysimage", "--output", str(sysimage), "--cpu-target", "generic"],
    )

    assert status == 0
    assert calls["julia"] == "/usr/bin/julia"
    assert 'cpu_target="generic"' in calls["script"]
    assert "precompile_execution_file" in calls["script"]
    assert calls["validated"] == sysimage.resolve()
    output = capsys.readouterr().out
    assert "Future solver calls will use this sysimage automatically." in output
    metadata = bridge.read_sysimage_metadata()
    assert metadata is not None
    assert metadata["julia_version"] == "1.11.9"
    assert metadata["sysimage_path"] == str(sysimage.resolve())


def test_build_sysimage_without_workload_can_keep_build_dir(
    monkeypatch: Any,
    tmp_path: Path,
    capsys: Any,
) -> None:
    calls: dict[str, Any] = {}
    _fake_julia_toolchain(monkeypatch, calls)
    monkeypatch.setenv(bridge.CACHE_DIR_ENV, str(tmp_path / "cache"))

    status = cli.main(
        [
            "build-sysimage",
            "--output",
            str(tmp_path / "image.so"),
            "--no-precompile-workload",
            "--keep-build-dir",
        ],
    )

    assert status == 0
    assert "precompile_execution_file" not in calls["script"]
    assert "Kept build directory" in capsys.readouterr().out


def test_main_without_a_command_prints_help(capsys: Any) -> None:
    assert cli.main([]) == 1
    assert "usage: oqs" in capsys.readouterr().out


def test_julia_helpers_run_subprocesses(monkeypatch: Any) -> None:
    seen: list[tuple[list[str], dict[str, Any]]] = []

    class Completed:
        stdout = " 1.11.9\n"

    def fake_run(command: list[str], **kwargs: Any) -> Completed:
        seen.append((command, kwargs))
        return Completed()

    monkeypatch.setattr(cli.subprocess, "run", fake_run)

    assert cli._julia_query("/usr/bin/julia", "print(VERSION)") == "1.11.9"
    cli._run_julia("/usr/bin/julia", "1 + 1")

    assert seen[0][0] == ["/usr/bin/julia", "--startup-file=no", "-e", "print(VERSION)"]
    assert seen[1][1]["env"]["JULIA_PYTHONCALL_EXE"]


def test_julia_string_escapes_windows_paths_and_quotes() -> None:
    assert cli._julia_string(Path("C:/a b/c")) == '"C:/a b/c"'
    # Quotes and backslashes must be escaped inside a Julia string literal.
    assert cli._julia_string(r'say "hi" \n') == r'"say \"hi\" \\n"'
