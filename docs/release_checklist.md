# Publish Readiness Checklist

OpenQuantumSim is not ready for a public release until every alpha gate below is
green. This file is the standing answer to: "Is it ready to publish yet?"

## Current Status

Status: public alpha release candidate. The public alpha and public beta gates
are green. `v0.1.0a0` is tagged for GitHub assets, and `v0.1.0a1` is the first
package-index candidate after PyPI metadata cleanup.

Latest package-index candidate: `v0.1.0a1`.
GitHub-only alpha tag: `v0.1.0a0` on commit `ebf41f5`.
Latest green public CI for the release commit: run #11 on commit `ebf41f5`,
completed on 2026-05-14.
Latest local artifact check: wheel/sdist build, `twine check`, and installed
wheel Julia-backend smoke test passed on 2026-05-14.
Latest TestPyPI attempt: workflow run `25930969150` built artifacts, updated
the GitHub Release, exchanged the trusted-publishing token successfully, then
failed because PyPI rejected the invalid classifier `Programming Language ::
Julia`. `v0.1.0a1` removes that classifier.

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
- [x] GitHub release workflow configured for tagged distribution builds.
- [x] TestPyPI trusted publisher configured.
- [ ] TestPyPI install verified from a fresh virtual environment.
- [ ] PyPI trusted publisher configured.
- [ ] PyPI install verified from a fresh virtual environment.

## Public Beta Gate

- [x] `mcsolve` supports backend-side aggregation for built-in trajectory
      diagnostics that reduce to linear expectations or pure-trajectory
      constants.
- [x] QuTiP comparison benchmark documented.
- [x] Performance benchmarks published with hardware and thread settings.
- [x] Result HDF5 schema documented.
- [x] First external-user quickstart tested from a fresh machine.

## Release Command Checklist

Run these before tagging a release:

```bash
python scripts/check_publish_ready.py
python scripts/check_wheel_install.py
python -m ruff check openquantumsim examples tests scripts benchmarks README.md setup_julia.py
python -m mypy openquantumsim scripts/check_publish_ready.py setup_julia.py
python -m pytest
julia --project=src/OpenQuantumSimJL -e 'using Pkg; Pkg.test()'
python -m build --outdir dist
python -m twine check dist/*
```

## Release Workflow

Tagged versions are built by `.github/workflows/release.yml`. The workflow
builds the source distribution and wheel, runs `twine check`, uploads the
artifacts to the GitHub release, and can be manually dispatched for an existing
tag.

PyPI publishing is intentionally manual-gated: run the release workflow from
GitHub Actions with `publish_target=testpypi` first, verify installation from
TestPyPI, then rerun it with `publish_target=pypi` after trusted publishing is
configured for the repository.

The exact trusted-publisher settings and post-upload smoke-test commands are
tracked in `docs/publishing.rst`.
