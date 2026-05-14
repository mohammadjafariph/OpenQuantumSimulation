# Publish Readiness Checklist

OpenQuantumSim is not ready for a public release until every alpha gate below is
green. This file is the standing answer to: "Is it ready to publish yet?"

## Current Status

Status: public alpha candidate; not ready for public beta.

The public alpha gate is green. Before a broader beta release, the package
still needs a fresh-machine quickstart test and follow-up optimization work.

## Public Alpha Gate

- [x] MIT license present.
- [x] Changelog, citation metadata, and contribution guide present.
- [x] Julia backend included as Python package data.
- [x] General public API separated from research-specific examples.
- [x] Python lint, typing, and tests pass locally.
- [x] Continuous integration configured for Python lint/tests.
- [x] Continuous integration configured for Julia backend tests.
- [x] Wheel install tested in a clean virtual environment.
- [x] Julia backend instantiation tested from an installed wheel.
- [x] Continuous integration passing on the public repository.
- [x] At least two canonical validation examples documented.
- [x] API reference generated from docstrings.
- [x] Version, repository URL, and citation metadata confirmed against the
      actual public repository.

## Public Beta Gate

- [x] `mcsolve` supports backend-side aggregation for built-in trajectory
      diagnostics that reduce to linear expectations or pure-trajectory
      constants.
- [x] QuTiP comparison benchmark documented.
- [x] Performance benchmarks published with hardware and thread settings.
- [x] Result HDF5 schema documented.
- [ ] First external-user quickstart tested from a fresh machine.

## Release Command Checklist

Run these before tagging a release:

```bash
python scripts/check_publish_ready.py
python scripts/check_wheel_install.py
python -m ruff check openquantumsim examples tests scripts benchmarks README.md
python -m mypy openquantumsim
python -m pytest
julia --project=src/OpenQuantumSimJL -e 'using Pkg; Pkg.test()'
python -m pip wheel . --no-deps -w dist
```
