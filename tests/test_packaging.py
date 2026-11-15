from pathlib import Path

from openquantumsim._julia_bridge import backend_path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_BACKEND = ROOT / "src" / "OpenQuantumSimJL"
PACKAGED_BACKEND = ROOT / "openquantumsim" / "julia" / "OpenQuantumSimJL"

REQUIRED_BACKEND_FILES = {
    "Project.toml",
    "Manifest.toml",
    "src/OpenQuantumSimJL.jl",
    "src/Health.jl",
    "src/Lindblad.jl",
    "src/BlochRedfield.jl",
    "src/Trajectories.jl",
    "test/runtests.jl",
}


def _backend_files(root: Path) -> set[str]:
    return {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.suffix in {".jl", ".toml"}
    }


def test_backend_path_has_julia_project() -> None:
    path = backend_path()

    assert (path / "Project.toml").is_file()
    assert (path / "src" / "OpenQuantumSimJL.jl").is_file()


def test_packaged_julia_backend_mirror_matches_source() -> None:
    source_files = _backend_files(SOURCE_BACKEND)
    packaged_files = _backend_files(PACKAGED_BACKEND)

    assert REQUIRED_BACKEND_FILES <= source_files
    assert packaged_files == source_files
    for relative in sorted(source_files):
        source = SOURCE_BACKEND / relative
        packaged = PACKAGED_BACKEND / relative
        assert packaged.read_bytes() == source.read_bytes(), (
            f"{relative} differs; run `python scripts/sync_julia_backend.py`"
        )
