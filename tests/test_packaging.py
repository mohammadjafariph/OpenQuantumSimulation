from pathlib import Path

from openquantumsim._julia_bridge import backend_path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_BACKEND = ROOT / "src" / "OpenQuantumSimJL"
PACKAGED_BACKEND = ROOT / "openquantumsim" / "julia" / "OpenQuantumSimJL"

BACKEND_FILES = [
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


def test_backend_path_has_julia_project() -> None:
    path = backend_path()

    assert (path / "Project.toml").is_file()
    assert (path / "src" / "OpenQuantumSimJL.jl").is_file()


def test_packaged_julia_backend_mirror_matches_source() -> None:
    for relative in BACKEND_FILES:
        source = SOURCE_BACKEND / relative
        packaged = PACKAGED_BACKEND / relative

        assert source.is_file()
        assert packaged.is_file()
        assert packaged.read_bytes() == source.read_bytes()
