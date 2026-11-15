"""Check basic publish-readiness invariants for OpenQuantumSim."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_BACKEND = ROOT / "src" / "OpenQuantumSimJL"
PACKAGED_BACKEND = ROOT / "openquantumsim" / "julia" / "OpenQuantumSimJL"

REQUIRED_FILES = [
    "README.md",
    "LICENSE",
    "CITATION.cff",
    "CHANGELOG.md",
    "CONTRIBUTING.md",
    "MANIFEST.in",
    "pyproject.toml",
    ".github/workflows/release.yml",
    "openquantumsim/py.typed",
    "docs/publishing.rst",
    "docs/release_checklist.md",
    "scripts/check_index_install.py",
    "scripts/check_wheel_install.py",
]

MIRRORED_SUFFIXES = {".jl", ".toml"}


def backend_files(root: Path) -> set[str]:
    """Return every mirrored backend file under ``root`` as a posix path."""
    return {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.suffix in MIRRORED_SUFFIXES
    }


def main() -> int:
    """Return 0 when the lightweight publish-readiness gate passes."""
    failures: list[str] = []
    for relative in REQUIRED_FILES:
        path = ROOT / relative
        if not path.is_file():
            failures.append(f"missing required file: {relative}")

    source_files = backend_files(SOURCE_BACKEND)
    packaged_files = backend_files(PACKAGED_BACKEND)
    if not source_files:
        failures.append("missing source backend: src/OpenQuantumSimJL")
    for relative in sorted(source_files - packaged_files):
        failures.append(f"missing packaged backend file: {relative}")
    for relative in sorted(packaged_files - source_files):
        failures.append(f"packaged backend file has no source: {relative}")
    for relative in sorted(source_files & packaged_files):
        source = SOURCE_BACKEND / relative
        packaged = PACKAGED_BACKEND / relative
        if source.read_bytes() != packaged.read_bytes():
            failures.append(f"packaged backend is out of sync: {relative}")
    if any("backend" in failure for failure in failures):
        failures.append("run `python scripts/sync_julia_backend.py` to fix the mirror")

    if failures:
        print("Publish readiness: NOT READY")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("Publish readiness: basic file/package-data gate passed.")
    print("Remaining release gates are tracked in docs/release_checklist.md.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
