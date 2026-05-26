"""Project-local Julia backend setup helper."""

from __future__ import annotations

from openquantumsim.cli import setup_julia_command


def main() -> None:
    """Instantiate the Julia backend through the same runtime used by JuliaCall."""
    raise SystemExit(setup_julia_command())


if __name__ == "__main__":
    main()
