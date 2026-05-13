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
        "using Pkg; Pkg.instantiate(); Pkg.precompile()",
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
