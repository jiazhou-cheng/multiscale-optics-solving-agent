"""Unit checks on the M1 scientific-fingerprint projection.

Solver-free and instant, unlike `test_m1_reproducibility.py`, which needs four
full branch runs (~7 minutes) to notice the same class of defect.

Background: the M1.8 review found the wave branch failing to reproduce its
scientific fingerprint. The cause was not the solver -- the scaling field
hashes were bitwise identical across runs -- but the projection hashing the
accuracy section wholesale, including each case's `solver.runtime_seconds`.
The published `volatile_exclusions` policy already claimed wall-clock values
were excluded; the implementation did not honour it. These tests pin the
implementation to the policy.
"""

from __future__ import annotations

import pytest

from multiscale_optics_agent.evaluation.m1_bundle import VOLATILE_KEYS, _strip_volatile

pytestmark = pytest.mark.benchmark


def test_strips_volatile_keys_at_any_nesting_depth() -> None:
    payload = {
        "metrics": {
            "cases": [
                {"name": "a", "solver": {"runtime_seconds": 1.23, "power_out": 4.0}},
                {"name": "b", "solver": {"runtime_seconds": 9.99, "power_out": 5.0}},
            ]
        },
        "pass": True,
    }
    stripped = _strip_volatile(payload)

    assert stripped["pass"] is True
    for case in stripped["metrics"]["cases"]:
        assert "runtime_seconds" not in case["solver"]
        assert "power_out" in case["solver"], "scientific values must survive stripping"
    assert [case["name"] for case in stripped["metrics"]["cases"]] == ["a", "b"]


def test_two_runs_differing_only_in_wall_clock_project_identically() -> None:
    """The exact failure mode M1.8 found: same physics, different machine load."""

    def section(runtime: float) -> dict:
        return {
            "metrics": {
                "cases": [
                    {
                        "errors": {"phase_error_rad": 6.49e-06},
                        "solver": {"runtime_seconds": runtime},
                    }
                ]
            },
            "pass": True,
        }

    assert _strip_volatile(section(2.34)) == _strip_volatile(section(0.014))


def test_a_real_scientific_difference_still_survives() -> None:
    """Stripping must not be so aggressive that it hides a physics change."""

    def section(error: float) -> dict:
        return {
            "metrics": {
                "cases": [
                    {"errors": {"phase_error_rad": error}, "solver": {"runtime_seconds": 1.0}}
                ]
            }
        }

    assert _strip_volatile(section(6.49e-06)) != _strip_volatile(section(6.49e-03))


def test_declared_volatile_keys_cover_wall_clock_and_run_identity() -> None:
    assert "runtime_seconds" in VOLATILE_KEYS
    assert "timestamp_utc" in VOLATILE_KEYS
    assert "run_id" in VOLATILE_KEYS
    assert "output_directory" in VOLATILE_KEYS


def test_scalars_and_empty_containers_pass_through_unchanged() -> None:
    assert _strip_volatile(3.5) == 3.5
    assert _strip_volatile("text") == "text"
    assert _strip_volatile(None) is None
    assert _strip_volatile([]) == []
    assert _strip_volatile({}) == {}
