"""Shared test configuration.

Physics tests need the Julia backend, and reference cross-checks need QuTiP.
On a developer machine a missing dependency skips the test. In CI the
environment variables below turn a missing dependency into a failure, so a
broken setup can never pass by silently skipping the tests that matter.

``OQS_REQUIRE_BACKEND=1``  fail (not skip) when the Julia backend is missing.
``OQS_REQUIRE_QUTIP=1``    fail (not skip) when QuTiP is missing.
"""

from __future__ import annotations

import importlib
import os
from typing import Any

import pytest

_TRUTHY = {"1", "true", "yes", "on"}


def _required(variable: str) -> bool:
    return os.environ.get(variable, "").strip().lower() in _TRUTHY


@pytest.fixture(autouse=True)
def _backend_for_physics_tests(request: pytest.FixtureRequest) -> None:
    """Skip or fail tests marked ``physics``/``integration`` without a backend."""
    markers = {mark.name for mark in request.node.iter_markers()}
    if not markers & {"physics", "integration"}:
        return
    from openquantumsim._julia_bridge import backend_available

    if backend_available():
        return
    if _required("OQS_REQUIRE_BACKEND"):
        pytest.fail("Julia backend is required (OQS_REQUIRE_BACKEND) but unavailable.")
    pytest.skip("Julia backend is not available.")


@pytest.fixture
def qutip() -> Any:
    """Return the QuTiP module, skipping or failing when it is missing."""
    try:
        return importlib.import_module("qutip")
    except ImportError:
        if _required("OQS_REQUIRE_QUTIP"):
            pytest.fail("QuTiP is required (OQS_REQUIRE_QUTIP) but not installed.")
        pytest.skip("QuTiP is not installed.")
