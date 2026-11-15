"""Copy the development Julia backend into the packaged mirror.

The wheel ships ``openquantumsim/julia/OpenQuantumSimJL`` as package data. It
must stay byte-identical to ``src/OpenQuantumSimJL``. Run this after editing
the backend:

    python scripts/sync_julia_backend.py          # copy source -> mirror
    python scripts/sync_julia_backend.py --check  # exit 1 if they differ
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_BACKEND = ROOT / "src" / "OpenQuantumSimJL"
PACKAGED_BACKEND = ROOT / "openquantumsim" / "julia" / "OpenQuantumSimJL"
MIRRORED_SUFFIXES = {".jl", ".toml"}


def backend_files(root: Path) -> list[str]:
    """Return the mirrored backend files under ``root`` as posix paths."""
    return sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.suffix in MIRRORED_SUFFIXES
    )


def mirror_differences() -> list[str]:
    """Describe every difference between the source backend and its mirror."""
    source_files = set(backend_files(SOURCE_BACKEND))
    packaged_files = set(backend_files(PACKAGED_BACKEND))
    differences = [
        f"missing packaged backend file: {relative}"
        for relative in sorted(source_files - packaged_files)
    ]
    differences += [
        f"packaged backend file has no source: {relative}"
        for relative in sorted(packaged_files - source_files)
    ]
    for relative in sorted(source_files & packaged_files):
        source = (SOURCE_BACKEND / relative).read_bytes()
        packaged = (PACKAGED_BACKEND / relative).read_bytes()
        if source != packaged:
            differences.append(f"packaged backend is out of sync: {relative}")
    return differences


def sync() -> list[str]:
    """Make the packaged mirror identical to the source backend."""
    changed: list[str] = []
    source_files = backend_files(SOURCE_BACKEND)
    for relative in source_files:
        source = SOURCE_BACKEND / relative
        packaged = PACKAGED_BACKEND / relative
        if packaged.is_file() and packaged.read_bytes() == source.read_bytes():
            continue
        packaged.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, packaged)
        changed.append(f"copied {relative}")
    for relative in sorted(set(backend_files(PACKAGED_BACKEND)) - set(source_files)):
        (PACKAGED_BACKEND / relative).unlink()
        changed.append(f"removed {relative}")
    return changed


def main() -> int:
    """Synchronise the mirror, or only report differences with ``--check``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Report differences without changing any file.",
    )
    args = parser.parse_args()

    if args.check:
        differences = mirror_differences()
        for difference in differences:
            print(difference)
        if differences:
            return 1
        print("Packaged Julia backend matches src/OpenQuantumSimJL.")
        return 0

    changed = sync()
    for line in changed:
        print(line)
    print(f"Packaged Julia backend synchronised ({len(changed)} file(s) changed).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
