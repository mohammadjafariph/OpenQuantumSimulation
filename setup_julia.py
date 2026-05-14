"""Project-local Julia backend setup helper."""

from __future__ import annotations

from openquantumsim._julia_bridge import backend_path, load_backend


def main() -> None:
    """Instantiate the Julia backend through the same runtime used by JuliaCall."""
    backend = backend_path()
    load_backend()
    print(f"Julia backend ready: {backend}")


if __name__ == "__main__":
    main()
