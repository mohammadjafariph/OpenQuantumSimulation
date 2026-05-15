# Contributing

OpenQuantumSim is an alpha-stage package, so the most useful contributions are
small, well-tested changes that make the public API clearer, more reliable, or
easier to install.

## Development Setup

```bash
python -m pip install -e ".[dev]"
python setup_julia.py
python -m pytest
```

Run the Julia backend tests directly with:

```bash
julia --project=src/OpenQuantumSimJL -e 'using Pkg; Pkg.test()'
```

## Checks Before A Pull Request

```bash
python -m ruff check openquantumsim examples tests scripts benchmarks README.md
python -m mypy openquantumsim
python -m pytest
python scripts/check_publish_ready.py
```

## Contribution Guidelines

- Keep the `openquantumsim` package model agnostic. Specific physics studies
  belong in `examples/` unless they are genuinely reusable library primitives.
- Add tests for public behavior changes.
- Keep Python and Julia backend changes synchronized. The packaged Julia
  backend mirror under `openquantumsim/julia/OpenQuantumSimJL` must match the
  development backend under `src/OpenQuantumSimJL`.
- Document new public API in the README or Sphinx docs.
- Avoid committing generated simulation outputs.
