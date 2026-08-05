"""Shared pytest fixtures and helpers for the test suite."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from multiscale_optics_agent.registry.loader import Registry

ROOT = Path(__file__).resolve().parents[1]
KNOWLEDGE_ROOT = ROOT / "knowledge" / "solvers"


@pytest.fixture(scope="session")
def registry() -> Registry:
    return Registry.from_package()


def load_probe_expected(solver: str, probe: str) -> dict:
    """Load the recorded ground truth from knowledge/solvers/<solver>/expected/<probe>.json.

    This is evidence captured by running knowledge/solvers/<solver>/probes/<probe>.py
    against the real solver; adapter tests compare against it instead of
    re-deriving an oracle.
    """
    path = KNOWLEDGE_ROOT / solver / "expected" / f"{probe}.json"
    return json.loads(path.read_text())
