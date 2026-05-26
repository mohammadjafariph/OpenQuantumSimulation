"""Command-line utilities for OpenQuantumSim."""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

from ._julia_bridge import (
    backend_fingerprint,
    backend_path,
    cache_dir,
    load_backend,
    sysimage_metadata_path,
    write_sysimage_metadata,
)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the ``oqs`` command-line interface."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    command = args.command
    if command == "setup-julia":
        return setup_julia_command()
    if command == "build-sysimage":
        return build_sysimage_command(
            output=args.output,
            force=args.force,
            cpu_target=args.cpu_target,
            precompile_workload=not args.no_precompile_workload,
            keep_build_dir=args.keep_build_dir,
        )
    parser.print_help()
    return 1


def setup_julia_command() -> int:
    """Instantiate and precompile the packaged Julia backend."""
    backend = backend_path()
    load_backend(instantiate=True)
    print(f"Julia backend ready: {backend}")
    return 0


def build_sysimage_command(
    *,
    output: Path | None = None,
    force: bool = False,
    cpu_target: str = "native",
    precompile_workload: bool = True,
    keep_build_dir: bool = False,
) -> int:
    """Build and register a local Julia sysimage for faster backend startup."""
    julia = _juliapkg_executable()
    julia_version = _julia_query(julia, "print(VERSION)")
    dlext = _julia_query(julia, "using Libdl; print(Libdl.dlext)")
    sysimage = output or (cache_dir() / f"OpenQuantumSimJL_sysimage.{dlext}")
    sysimage = sysimage.expanduser().resolve()

    if sysimage.exists() and not force:
        print(
            f"Refusing to overwrite existing sysimage: {sysimage}\n"
            "Pass --force to rebuild it.",
            file=sys.stderr,
        )
        return 2

    sysimage.parent.mkdir(parents=True, exist_ok=True)
    build_dir = Path(tempfile.mkdtemp(prefix="oqs-sysimage-"))
    try:
        precompile_file = build_dir / "precompile_workload.jl"
        if precompile_workload:
            precompile_file.write_text(_precompile_workload(), encoding="utf-8")
        script = _packagecompiler_script(
            backend=backend_path(),
            build_project=build_dir / "julia_project",
            sysimage=sysimage,
            cpu_target=cpu_target,
            precompile_file=precompile_file if precompile_workload else None,
        )

        print("Building OpenQuantumSim Julia sysimage.")
        print(f"Julia: {julia} ({julia_version})")
        print(f"Output: {sysimage}")
        _run_julia(julia, script)

        write_sysimage_metadata(
            {
                "schema_version": 1,
                "sysimage_path": str(sysimage),
                "backend_path": str(backend_path()),
                "backend_fingerprint": backend_fingerprint(),
                "julia_executable": julia,
                "julia_version": julia_version,
                "platform": sys.platform,
                "machine": platform.machine(),
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        print(f"Sysimage metadata: {sysimage_metadata_path()}")
        print("Future solver calls will use this sysimage automatically.")
        return 0
    finally:
        if keep_build_dir:
            print(f"Kept build directory: {build_dir}")
        else:
            shutil.rmtree(build_dir, ignore_errors=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="oqs",
        description="OpenQuantumSim command-line utilities.",
    )
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser(
        "setup-julia",
        help="Instantiate and precompile the Julia backend once.",
    )

    sysimage = subparsers.add_parser(
        "build-sysimage",
        help="Build a local Julia sysimage for faster repeated solver startup.",
    )
    sysimage.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Sysimage output path. Defaults to the OpenQuantumSim cache.",
    )
    sysimage.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing sysimage.",
    )
    sysimage.add_argument(
        "--cpu-target",
        default="native",
        help="PackageCompiler CPU target. Use 'generic' for a portable image.",
    )
    sysimage.add_argument(
        "--no-precompile-workload",
        action="store_true",
        help="Skip the small solver workload used to precompile common methods.",
    )
    sysimage.add_argument(
        "--keep-build-dir",
        action="store_true",
        help="Keep the temporary build project for debugging.",
    )
    return parser


def _juliapkg_executable() -> str:
    try:
        import juliapkg  # type: ignore[import-untyped]
    except Exception as exc:  # pragma: no cover - installation dependent
        msg = "juliapkg is required to build an OpenQuantumSim sysimage."
        raise RuntimeError(msg) from exc
    return str(juliapkg.executable())


def _julia_query(julia: str, expression: str) -> str:
    result = subprocess.run(
        [julia, "--startup-file=no", "-e", expression],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _run_julia(julia: str, script: str) -> None:
    subprocess.run(
        [julia, "--startup-file=no", "-e", script],
        check=True,
        env=_julia_subprocess_env(),
    )


def _julia_subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("JULIA_PYTHONCALL_EXE", sys.executable)
    return env


def _packagecompiler_script(
    *,
    backend: Path,
    build_project: Path,
    sysimage: Path,
    cpu_target: str,
    precompile_file: Path | None,
) -> str:
    kwargs = [
        f"project={_julia_string(build_project)}",
        f"sysimage_path={_julia_string(sysimage)}",
        f"cpu_target={_julia_string(cpu_target)}",
    ]
    if precompile_file is not None:
        kwargs.append(f"precompile_execution_file={_julia_string(precompile_file)}")
    keyword_args = ",\n    ".join(kwargs)
    return f"""
import Pkg
Pkg.activate({_julia_string(build_project)})
Pkg.add(Pkg.PackageSpec(name="PackageCompiler"))
Pkg.develop(Pkg.PackageSpec(path={_julia_string(backend)}))
Pkg.instantiate()
Pkg.precompile()
using PackageCompiler
create_sysimage(
    [:OpenQuantumSimJL];
    {keyword_args},
)
"""


def _precompile_workload() -> str:
    return """
using OpenQuantumSimJL
using LinearAlgebra

H = ComplexF64[0 0; 0 0]
rho0 = ComplexF64[1 0; 0 0]
psi0 = ComplexF64[1, 0]
collapse = [sqrt(0.2) * Matrix{ComplexF64}(sigmam())]
projector = [ComplexF64[1 0; 0 0]]
times = [0.0, 0.1]

mesolve(H, rho0, times, collapse, projector, 1e-6, 1e-8, false, "ode", 10, false)
mesolve(H, rho0, times, collapse, projector, 1e-6, 1e-8, false, "krylov", 10, false)
single_trajectory(H, psi0, times, collapse, projector, 2026, 0.05, false)
mcsolve(H, psi0, times, collapse, projector, 2, 2026, 0.05, 1, "", 100, false)
steadystate(H, collapse; method="direct")
"""


def _julia_string(value: str | Path) -> str:
    text = str(value)
    escaped = text.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


if __name__ == "__main__":
    raise SystemExit(main())
