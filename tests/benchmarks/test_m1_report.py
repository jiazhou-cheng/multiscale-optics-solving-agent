"""Structural checks on the M1 exit report (CHE-19).

These are cheap and solver-free on purpose: they run without Optiland or
Chromatix installed, so the report cannot quietly lose a required section or
drift away from the manifest while the expensive reproduction tests are
skipped.

They deliberately do **not** re-assert the numeric results. Those live in
`outputs/M1/*/result.json`, are hashed into each bundle manifest, and are
checked by `test_m1_reproducibility.py`. Duplicating them here would create a
second place to update and a second thing to go stale.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "benchmarks" / "M1_BASELINE_REPORT.md"


@pytest.fixture(scope="module")
def report_text() -> str:
    assert REPORT.exists(), f"M1 exit report is missing: {REPORT}"
    return REPORT.read_text()


@pytest.mark.parametrize(
    "section",
    [
        "Exact commands",
        "Environment",
        "Ray branch",
        "Wave branch",
        "Independence evidence",
        "Claim audit",
        "Reproducibility policy",
        "Risks and known limitations",
        "What M2 should carry forward",
    ],
)
def test_report_contains_every_required_section(report_text: str, section: str) -> None:
    assert section in report_text, f"M1 report is missing the {section!r} section"


def test_report_separates_accuracy_from_performance_for_both_branches(
    report_text: str,
) -> None:
    """CHE-19 requires accuracy and performance evidence per branch, not merged."""
    assert report_text.count("## Accuracy") >= 2
    assert report_text.count("## Performance") >= 2
    # Accuracy must be stated as gating performance, not the other way round.
    assert "before" in report_text and "performance number is" in report_text


def test_report_records_both_branch_commands_from_the_manifest(report_text: str) -> None:
    manifest = yaml.safe_load((ROOT / "benchmarks" / "manifest.yaml").read_text())
    tasks = {task["id"]: task for task in manifest["levels"][1]["tasks"]}
    for benchmark_id in ("L1-RAY-01", "L1-WAVE-01"):
        task = tasks[benchmark_id]
        assert "bundle_command" in task, f"{benchmark_id} must link its bundle command"
        assert "evaluator_version" in task, f"{benchmark_id} must pin an evaluator version"
        assert f"benchmarks/level1/{benchmark_id}/run_all.py" in report_text
    assert "benchmarks/verify_m1_independence.py" in report_text


def test_report_states_pinned_engine_identity_for_both_branches(report_text: str) -> None:
    assert "optiland 0.6.0" in report_text
    assert "chromatix 0.6.0" in report_text
    # The wave engine is a VCS install, so the commit is part of its identity.
    assert "d24bdf0022835bb8ce1cdcc6aeafbc7fcb39daee" in report_text


def test_report_keeps_unverified_capabilities_unverified(report_text: str) -> None:
    """M1 must not let a capability leak into the record it never established."""
    for claim in ("gradient", "GPU", "vector field", "coupling"):
        assert claim.lower() in report_text.lower()
    assert "still unverified" in report_text


def test_report_records_the_blocked_wave_case_without_hiding_it(report_text: str) -> None:
    assert "BLOCKED" in report_text
    assert "high_na_ff_lens" in report_text
    # A blocked case must be justified as blocked rather than quietly failed,
    # and must not have been resolved by moving a tolerance.
    assert "blocked, not failed" in report_text
    assert "No tolerance was changed" in report_text


def test_report_declares_the_reproducibility_exclusions(report_text: str) -> None:
    for excluded in ("timestamps", "run identifiers", "process IDs"):
        assert excluded in report_text
    assert "scientific fingerprint" in report_text.lower()


def test_report_discloses_the_dirty_worktree_caveat(report_text: str) -> None:
    """A clean-checkout claim must not be made from an uncommitted tree."""
    assert "dirty worktree" in report_text.lower()
