"""Install OpenQuantumSim from PyPI/TestPyPI and run a backend smoke test."""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Literal

from check_wheel_install import _smoke_code, _venv_python

ROOT = Path(__file__).resolve().parents[1]
IndexName = Literal["testpypi", "pypi"]


def main() -> int:
    """Return 0 when an index install and Julia-backend smoke test passes."""
    args = _parse_args()
    version = args.version or _project_version()
    tmpdir = Path(tempfile.mkdtemp(prefix=f"oqs-{args.index}-install-"))
    try:
        venv = tmpdir / "venv"
        _run([sys.executable, "-m", "venv", str(venv)], cwd=tmpdir)
        python = _venv_python(venv)
        _run(
            [
                str(python),
                "-m",
                "pip",
                "install",
                "--upgrade",
                "pip",
            ],
            cwd=tmpdir,
        )
        _run(_install_command(python, args.index, version), cwd=tmpdir)
        _run([str(python), "-c", _smoke_code()], cwd=tmpdir)
        print(
            f"Index install smoke test passed: "
            f"{args.index} openquantumsim=={version}",
        )
        return 0
    finally:
        if args.keep_temp:
            print(f"Kept temporary directory: {tmpdir}")
        else:
            shutil.rmtree(tmpdir, ignore_errors=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Install OpenQuantumSim from PyPI or TestPyPI in a fresh virtualenv, "
            "load the packaged Julia backend, and run a tiny mesolve smoke test."
        ),
    )
    parser.add_argument(
        "--index",
        choices=["testpypi", "pypi"],
        default="testpypi",
        help="Package index to install from.",
    )
    parser.add_argument(
        "--version",
        help="Package version to install. Defaults to the version in pyproject.toml.",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep the temporary virtualenv directory for debugging.",
    )
    return parser.parse_args()


def _install_command(python: Path, index: IndexName, version: str) -> list[str]:
    package = f"openquantumsim=={version}"
    command = [str(python), "-m", "pip", "install"]
    if index == "testpypi":
        command.extend(
            [
                "--index-url",
                "https://test.pypi.org/simple/",
                "--extra-index-url",
                "https://pypi.org/simple/",
            ],
        )
    command.append(package)
    return command


def _project_version() -> str:
    pyproject = ROOT / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")
    in_project = False
    version_pattern = re.compile(r'^version\s*=\s*"([^"]+)"\s*$')
    for line in text.splitlines():
        stripped = line.strip()
        if stripped == "[project]":
            in_project = True
            continue
        if in_project and stripped.startswith("["):
            break
        match = version_pattern.match(stripped)
        if in_project and match:
            return match.group(1)
    msg = f"could not read project.version from {pyproject}"
    raise RuntimeError(msg)


def _run(command: list[str], *, cwd: Path) -> None:
    print("+", " ".join(command))
    subprocess.run(command, cwd=cwd, check=True)


if __name__ == "__main__":
    sys.exit(main())
