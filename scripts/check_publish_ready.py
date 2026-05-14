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

BACKEND_MIRROR_FILES = [
    "Project.toml",
    "Manifest.toml",
    "src/HilbertSpace.jl",
    "src/Lindblad.jl",
    "src/Observables.jl",
    "src/OpenQuantumSimJL.jl",
    "src/Correlations.jl",
    "src/Operators.jl",
    "src/Parallel.jl",
    "src/Propagators.jl",
    "src/SteadyState.jl",
    "src/TimeDep.jl",
    "src/Trajectories.jl",
    "src/Utils.jl",
    "test/runtests.jl",
]


def main() -> int:
    """Return 0 when the lightweight publish-readiness gate passes."""
    failures: list[str] = []
    for relative in REQUIRED_FILES:
        path = ROOT / relative
        if not path.is_file():
            failures.append(f"missing required file: {relative}")

    for relative in BACKEND_MIRROR_FILES:
        source = SOURCE_BACKEND / relative
        packaged = PACKAGED_BACKEND / relative
        if not source.is_file():
            failures.append(f"missing source backend file: {source.relative_to(ROOT)}")
            continue
        if not packaged.is_file():
            failures.append(
                f"missing packaged backend file: {packaged.relative_to(ROOT)}"
            )
            continue
        if source.read_bytes() != packaged.read_bytes():
            failures.append(f"packaged backend is out of sync: {relative}")

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
