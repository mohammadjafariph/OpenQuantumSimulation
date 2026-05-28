"""Benchmark OpenQuantumSim backend startup with and without a Julia sysimage."""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
from collections.abc import Sequence
from pathlib import Path
from statistics import median
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    tomllib = None  # type: ignore[assignment]

PROFILE_PREFIX = "OQS_STARTUP_PROFILE "

PROFILE_CODE = r"""
import json
import time

started = time.perf_counter()
import openquantumsim as oqs
import_seconds = time.perf_counter() - started

started = time.perf_counter()
import numpy as np
numpy_import_seconds = time.perf_counter() - started

from openquantumsim._julia_bridge import (
    active_sysimage_path,
    configured_sysimage_path,
    get_julia,
    load_backend,
)

configured = configured_sysimage_path()
started = time.perf_counter()
get_julia()
get_julia_seconds = time.perf_counter() - started
active = active_sysimage_path()

started = time.perf_counter()
load_backend()
load_backend_seconds = time.perf_counter() - started

space = oqs.SpinSpace(0.5, label="atom")
excited = oqs.basis(space, "up")
rho0 = oqs.ket2dm(excited)
hamiltonian = 0.0 * oqs.sigmaz(space)
collapse = np.sqrt(0.2) * oqs.sigmam(space)
projector = oqs.Operator(oqs.ket2dm(excited), space, "P_excited")
times = np.linspace(0.0, 0.2, 3)

started = time.perf_counter()
result = oqs.mesolve(
    hamiltonian,
    rho0,
    times,
    c_ops=[collapse],
    e_ops=[projector],
    options=oqs.Options(rtol=1e-8, atol=1e-10),
)
mesolve_seconds = time.perf_counter() - started

expected = np.exp(-0.2 * times)
if not np.allclose(result.expect[0].real, expected, atol=2e-7):
    raise RuntimeError(f"unexpected mesolve output: {result.expect[0].real!r}")

payload = {
    "import_seconds": import_seconds,
    "numpy_import_seconds": numpy_import_seconds,
    "get_julia_seconds": get_julia_seconds,
    "load_backend_seconds": load_backend_seconds,
    "first_mesolve_seconds": mesolve_seconds,
    "total_seconds": (
        import_seconds
        + numpy_import_seconds
        + get_julia_seconds
        + load_backend_seconds
        + mesolve_seconds
    ),
    "configured_sysimage": str(configured) if configured else None,
    "active_sysimage": str(active) if active else None,
    "expectation": result.expect[0].real.tolist(),
}
print("OQS_STARTUP_PROFILE " + json.dumps(payload, sort_keys=True))
"""


def main(argv: Sequence[str] | None = None) -> int:
    """Run the startup benchmark."""
    args = _parse_args(argv)
    version = args.version or _project_version()
    work_dir = args.work_dir or Path(tempfile.mkdtemp(prefix="oqs-startup-bench-"))
    work_dir = work_dir.expanduser().resolve()
    if work_dir.exists() and any(work_dir.iterdir()) and not args.reuse_work_dir:
        raise SystemExit(
            f"Refusing to use non-empty work directory: {work_dir}\n"
            "Pass --reuse-work-dir to reuse it.",
        )

    if not work_dir.exists():
        work_dir.mkdir(parents=True)

    report: dict[str, Any] = {
        "version": version,
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "install_source": str(args.install_local.resolve())
        if args.install_local
        else "pypi",
        "work_dir": str(work_dir),
        "profiles": [],
    }

    try:
        venv = work_dir / "venv"
        cache = work_dir / "openquantumsim-cache"
        python = _create_venv(venv)
        oqs = _script_path(venv, "oqs")
        env = _benchmark_env(cache)

        _run([str(python), "-m", "pip", "install", "--upgrade", "pip"])
        _run(
            _install_command(
                python,
                version,
                args.index_url,
                args.extra_index_url,
                args.install_local,
            ),
        )

        print("\n== Cold first solver without explicit setup ==")
        cold = _run_profile(python, env, label="cold_no_setup")
        report["profiles"].append(cold)

        print("\n== Explicit backend setup ==")
        report["setup_julia_seconds"] = _time_command([str(oqs), "setup-julia"], env)

        print("\n== Warm startup without sysimage ==")
        warm_env = env | {"OPENQUANTUMSIM_USE_SYSIMAGE": "0"}
        for repeat in range(1, args.repeats + 1):
            report["profiles"].append(
                _run_profile(python, warm_env, label=f"warm_no_sysimage_{repeat}"),
            )

        if not args.skip_sysimage:
            print("\n== Build sysimage ==")
            command = [str(oqs), "build-sysimage", "--force"]
            if args.cpu_target:
                command.extend(["--cpu-target", args.cpu_target])
            report["build_sysimage_seconds"] = _time_command(command, env)

            print("\n== Warm startup with sysimage ==")
            for repeat in range(1, args.repeats + 1):
                report["profiles"].append(
                    _run_profile(python, env, label=f"warm_sysimage_{repeat}"),
                )

        _add_summaries(report)
        _write_report(report, args.json)
        _print_summary(report)
    except Exception as exc:
        report["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
        _add_summaries(report)
        _write_report(report, args.json)
        _print_summary(report)
        raise
    finally:
        if args.keep_work_dir:
            print(f"Kept work directory: {work_dir}")
        elif args.work_dir is None:
            shutil.rmtree(work_dir, ignore_errors=True)

    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Install OpenQuantumSim into a fresh virtual environment and measure "
            "first solver startup before and after sysimage construction."
        ),
    )
    parser.add_argument(
        "--version",
        default=None,
        help="OpenQuantumSim version to install. Defaults to pyproject.toml.",
    )
    parser.add_argument(
        "--index-url",
        default=None,
        help="Optional pip package index URL.",
    )
    parser.add_argument(
        "--extra-index-url",
        default=None,
        help="Optional secondary pip package index URL.",
    )
    parser.add_argument(
        "--install-local",
        type=Path,
        default=None,
        help="Install OpenQuantumSim from a local source tree instead of PyPI.",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Working directory for the throwaway venv and OQS cache.",
    )
    parser.add_argument(
        "--reuse-work-dir",
        action="store_true",
        help="Allow using a non-empty --work-dir.",
    )
    parser.add_argument(
        "--keep-work-dir",
        action="store_true",
        help="Keep the temporary venv/cache after the benchmark.",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Optional JSON report path.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=2,
        help="Fresh-process warm profile repeats for each mode.",
    )
    parser.add_argument(
        "--cpu-target",
        default="native",
        help="PackageCompiler CPU target passed to oqs build-sysimage.",
    )
    parser.add_argument(
        "--skip-sysimage",
        action="store_true",
        help="Skip PackageCompiler sysimage construction.",
    )
    return parser.parse_args(argv)


def _project_version() -> str:
    pyproject = Path("pyproject.toml")
    if tomllib is not None:
        with pyproject.open("rb") as file:
            data = tomllib.load(file)
        return str(data["project"]["version"])

    match = re.search(
        r'^version\s*=\s*"([^"]+)"',
        pyproject.read_text(encoding="utf-8"),
        flags=re.MULTILINE,
    )
    if match is None:
        raise RuntimeError("Could not read project version from pyproject.toml")
    return match.group(1)


def _create_venv(venv: Path) -> Path:
    _run([sys.executable, "-m", "venv", str(venv)])
    return _script_path(venv, "python")


def _script_path(venv: Path, name: str) -> Path:
    scripts = "Scripts" if os.name == "nt" else "bin"
    suffix = ".exe" if os.name == "nt" else ""
    return venv / scripts / f"{name}{suffix}"


def _benchmark_env(cache: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["OPENQUANTUMSIM_CACHE_DIR"] = str(cache)
    env.setdefault("PYTHON_JULIACALL_HANDLE_SIGNALS", "yes")
    env.setdefault("JULIA_NUM_THREADS", "1")
    return env


def _install_command(
    python: Path,
    version: str,
    index_url: str | None,
    extra_index_url: str | None,
    local_root: Path | None,
) -> list[str]:
    command = [str(python), "-m", "pip", "install"]
    if index_url:
        command.extend(["--index-url", index_url])
    if extra_index_url:
        command.extend(["--extra-index-url", extra_index_url])
    if local_root is not None:
        command.append(str(local_root.expanduser().resolve()))
    else:
        command.append(f"openquantumsim=={version}")
    return command


def _run(command: Sequence[str], env: dict[str, str] | None = None) -> None:
    print("+ " + " ".join(command))
    subprocess.run(command, check=True, env=env)


def _time_command(command: Sequence[str], env: dict[str, str]) -> float:
    started = time.perf_counter()
    _run(command, env)
    elapsed = time.perf_counter() - started
    print(f"elapsed: {elapsed:.3f} s")
    return elapsed


def _run_profile(python: Path, env: dict[str, str], *, label: str) -> dict[str, Any]:
    print(f"profile: {label}")
    result = subprocess.run(
        [str(python), "-c", textwrap.dedent(PROFILE_CODE)],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    if result.returncode != 0:
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        raise subprocess.CalledProcessError(
            result.returncode,
            result.args,
            output=result.stdout,
            stderr=result.stderr,
        )
    payload = _parse_profile_output(result.stdout)
    payload["label"] = label
    print(
        f"  import={payload['import_seconds']:.3f}s "
        f"get_julia={payload['get_julia_seconds']:.3f}s "
        f"load_backend={payload['load_backend_seconds']:.3f}s "
        f"first_mesolve={payload['first_mesolve_seconds']:.3f}s "
        f"total={payload['total_seconds']:.3f}s "
        f"sysimage={bool(payload['active_sysimage'])}",
    )
    return payload


def _parse_profile_output(output: str) -> dict[str, Any]:
    for line in reversed(output.splitlines()):
        if line.startswith(PROFILE_PREFIX):
            return json.loads(line.removeprefix(PROFILE_PREFIX))
    raise RuntimeError(f"profile output did not contain {PROFILE_PREFIX!r}:\n{output}")


def _add_summaries(report: dict[str, Any]) -> None:
    profiles = report["profiles"]
    report["summary"] = {
        "cold_no_setup": _summary_for(profiles, "cold_no_setup"),
        "warm_no_sysimage": _summary_for(profiles, "warm_no_sysimage_"),
        "warm_sysimage": _summary_for(profiles, "warm_sysimage_"),
    }


def _summary_for(profiles: list[dict[str, Any]], label_prefix: str) -> dict[str, Any]:
    matching = [
        profile
        for profile in profiles
        if str(profile["label"]).startswith(label_prefix)
    ]
    if not matching:
        return {}
    keys = [
        "import_seconds",
        "numpy_import_seconds",
        "get_julia_seconds",
        "load_backend_seconds",
        "first_mesolve_seconds",
        "total_seconds",
    ]
    return {
        key: median(float(profile[key]) for profile in matching)
        for key in keys
    } | {"n": len(matching)}


def _write_report(report: dict[str, Any], path: Path | None) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {path}")


def _print_summary(report: dict[str, Any]) -> None:
    print("\nSummary")
    for label, values in report["summary"].items():
        if not values:
            continue
        print(
            f"{label}: total={values['total_seconds']:.3f}s, "
            f"get_julia={values['get_julia_seconds']:.3f}s, "
            f"load_backend={values['load_backend_seconds']:.3f}s, "
            f"first_mesolve={values['first_mesolve_seconds']:.3f}s",
        )
    if "setup_julia_seconds" in report:
        print(f"setup_julia: {report['setup_julia_seconds']:.3f}s")
    if "build_sysimage_seconds" in report:
        print(f"build_sysimage: {report['build_sysimage_seconds']:.3f}s")


if __name__ == "__main__":
    raise SystemExit(main())
