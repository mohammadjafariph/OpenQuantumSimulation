"""Build and smoke-test an OpenQuantumSim wheel in a fresh virtualenv."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    args = _parse_args()
    tmpdir = Path(tempfile.mkdtemp(prefix="oqs-wheel-check-"))
    try:
        wheel = _build_wheel(tmpdir)
        venv = tmpdir / "venv"
        _run([sys.executable, "-m", "venv", str(venv)], cwd=tmpdir)
        python = _venv_python(venv)
        _run([str(python), "-m", "pip", "install", str(wheel)], cwd=tmpdir)
        _run([str(python), "-c", _smoke_code()], cwd=tmpdir)
        print("Installed-wheel smoke test passed.")
        return 0
    finally:
        if args.keep_temp:
            print(f"Kept temporary directory: {tmpdir}")
        else:
            shutil.rmtree(tmpdir, ignore_errors=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build an OpenQuantumSim wheel, install it in a fresh virtualenv, "
            "load the packaged Julia backend, and run a tiny mesolve smoke test."
        ),
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep the temporary build and virtualenv directory for debugging.",
    )
    return parser.parse_args()


def _build_wheel(tmpdir: Path) -> Path:
    wheel_dir = tmpdir / "dist"
    _run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            str(ROOT),
            "--no-deps",
            "-w",
            str(wheel_dir),
        ],
        cwd=ROOT,
    )
    wheels = sorted(wheel_dir.glob("openquantumsim-*.whl"))
    if len(wheels) != 1:
        msg = f"expected exactly one wheel in {wheel_dir}, found {len(wheels)}"
        raise RuntimeError(msg)
    return wheels[0]


def _venv_python(venv: Path) -> Path:
    if sys.platform == "win32":
        return venv / "Scripts" / "python.exe"
    return venv / "bin" / "python"


def _run(command: list[str], *, cwd: Path) -> None:
    print("+", " ".join(command))
    subprocess.run(command, cwd=cwd, check=True)


def _smoke_code() -> str:
    return (
        "import numpy as np, openquantumsim as oqs; "
        "from openquantumsim._julia_bridge import backend_path, load_backend; "
        "path = backend_path(); "
        "assert 'site-packages' in str(path), path; "
        "assert (path / 'Project.toml').is_file(), path; "
        "load_backend(); "
        "atom = oqs.SpinSpace(0.5, label='atom'); "
        "H = 0.0 * oqs.sigmaz(atom); "
        "excited = oqs.basis(atom, 'up'); "
        "rho0 = oqs.ket2dm(excited); "
        "collapse = np.sqrt(0.2) * oqs.sigmam(atom); "
        "projector = oqs.Operator(oqs.ket2dm(excited), atom, 'P_excited'); "
        "times = np.linspace(0.0, 0.2, 3); "
        "result = oqs.mesolve("
        "H, rho0, times, c_ops=[collapse], e_ops=[projector], "
        "options=oqs.Options(rtol=1e-8, atol=1e-10)"
        "); "
        "expected = np.exp(-0.2 * times); "
        "assert np.allclose(result.expect[0].real, expected, atol=2e-7), "
        "result.expect[0].real; "
        "print(path); "
        "print(result.expect[0].real.tolist())"
    )


if __name__ == "__main__":
    sys.exit(main())
