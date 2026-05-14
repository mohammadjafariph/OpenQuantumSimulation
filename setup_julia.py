"""Project-local Julia backend setup helper."""

from __future__ import annotations

import subprocess

from openquantumsim._julia_bridge import backend_path


def main() -> None:
    """Instantiate the Julia backend environment."""
    backend = backend_path()
    cmd = [
        "julia",
        f"--project={backend}",
        "-e",
        (
            "using Pkg; "
            "try "
            "Pkg.instantiate(); Pkg.precompile(); "
            "catch err "
            "@warn \"Julia backend setup failed; resolving manifest and retrying\" "
            "exception=(err, catch_backtrace()); "
            "Pkg.resolve(); Pkg.instantiate(); Pkg.precompile(); "
            "end"
        ),
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
