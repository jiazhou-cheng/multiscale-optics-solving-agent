"""CHE-57 (PB6): the repo-owned Optiland tutorial reproductions, as a regression gate.

Each test executes one reproduction from
`tests_tutorial/cases/optiland/` against the pinned `optiland==0.6.0`
install and asserts two things:

1. every validation check the reproduction declares still passes, and
2. the recorded evidence in `tests_tutorial/cases/optiland/expected/`
   still describes what the solver does -- numeric metrics are compared with a
   tolerance, and the *set* of checks is compared exactly, so a silently
   dropped check fails the test rather than passing vacuously.

Refresh the recorded evidence with

    ./run.sh python tests_tutorial/cases/optiland/run_all.py --write-expected

A reproduction that is genuinely stochastic declares its own ``metric_rtol`` in
its ``TutorialMeta`` (with the reason in its docstring); everything else replays
at the deterministic budget.

This file lives in ``tests_tutorial/`` rather than ``tests/`` (CHE-67): the
Optiland reproductions cost 983 s, so they are an *on-demand* suite that
``pytest``'s ``testpaths = ["tests"]`` never collects. Run them explicitly:

    ./run.sh pytest -q tests_tutorial          # or: make test-tutorial

They are on-demand, not archived -- they stay maintained, and they need no
unarchive step. Reproductions marked ``slow`` in their ``TutorialMeta`` keep the
``slow`` marker so the historical cost split stays readable, but that no longer
decides anything here: nothing in this file runs in the default suite.
Reproductions marked ``needs_torch`` additionally carry the ``torch`` marker, and
the whole module carries ``tutorial`` so ``-m tutorial`` names exactly this suite.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

# The reproductions live beside this harness now. CHE-92 moved them out of
# `knowledge/`, which had left this file doing a `sys.path.insert` back into a
# directory marked retrieval-only -- a split brain where the test tree and the
# agent-facing context were the same files.
TUTORIAL_DIR = Path(__file__).resolve().parent / "cases" / "optiland"
sys.path.insert(0, str(TUTORIAL_DIR))

pytest.importorskip("optiland")

from _optiland_harness import (  # noqa: E402
    expected_path,
    load_tutorial_module,
    tutorial_module_paths,
)

pytestmark = [
    pytest.mark.tutorial,
    pytest.mark.optiland,
    pytest.mark.integration,
]

# Relative tolerance for replaying a recorded numeric metric. Optiland's numpy
# path is deterministic, so this is a float64-reassociation budget, not a
# physics tolerance -- except for the torch/Adam reproduction, which accumulates
# 100 optimizer steps and is given a looser budget below.
METRIC_RTOL = 1e-9
LOOSE_METRIC_RTOL = 1e-4


def _params():
    out = []
    for path in tutorial_module_paths():
        module = load_tutorial_module(path)
        meta = module.TUTORIAL
        # Every reproduction is a pinned-dependency regression gate (CHE-64): it
        # answers "has the pinned solver changed?", not "is our physics right?".
        # `tutorial` now comes from the module-level `pytestmark` -- the whole file
        # is the on-demand suite (CHE-67) -- so only the per-reproduction facts are
        # tagged here. `slow` is kept because the recorded cost split reads off it.
        marks = []
        if meta.slow:
            marks.append(pytest.mark.slow)
        if meta.needs_torch:
            marks.append(pytest.mark.torch)
        out.append(pytest.param(path, id=meta.slug, marks=marks))
    return out


def _flatten(value, prefix=""):
    """Flatten recorded metrics to {path: leaf} so numbers can be compared pairwise."""
    if isinstance(value, dict):
        for key, sub in value.items():
            yield from _flatten(sub, f"{prefix}.{key}" if prefix else str(key))
    elif isinstance(value, list):
        for i, sub in enumerate(value):
            yield from _flatten(sub, f"{prefix}[{i}]")
    else:
        yield prefix, value


@pytest.mark.parametrize("path", _params())
def test_tutorial_reproduction(path: Path) -> None:
    module = load_tutorial_module(path)
    meta = module.TUTORIAL
    result = module.run()

    failures = [f"{c.name}: {c.detail}" for c in result.failures]
    assert not failures, f"{meta.slug} validation failed:\n  " + "\n  ".join(failures)

    # A reproduction whose checks all pass but that checks nothing is worthless.
    assert result.checks, f"{meta.slug} declared no validation checks"
    kinds = {c.kind for c in result.checks}
    assert kinds - {"qualitative"}, (
        f"{meta.slug} has only qualitative checks; every reproduction must assert at "
        "least one reference, analytic or invariant property"
    )

    recorded = json.loads(expected_path(meta.slug).read_text())
    assert recorded["url"] == meta.url
    assert recorded["level"] == meta.level

    # The set of checks must not shrink or drift silently.
    assert {c["name"] for c in recorded["checks"]} == {c.name for c in result.checks}
    assert all(c["passed"] for c in recorded["checks"])

    rtol = meta.metric_rtol
    if rtol is None:
        rtol = LOOSE_METRIC_RTOL if meta.needs_torch else METRIC_RTOL
    observed = dict(_flatten(result.metrics))
    for key, want in _flatten(recorded["metrics"]):
        assert key in observed, f"{meta.slug}: recorded metric {key} is no longer produced"
        got = observed[key]
        if isinstance(want, bool) or isinstance(got, bool) or not isinstance(want, (int, float)):
            assert got == want, f"{meta.slug}: metric {key} changed: {got!r} != {want!r}"
        elif math.isinf(want) or math.isnan(want):
            assert (math.isinf(got) and math.isinf(want) and (got > 0) == (want > 0)) or (
                math.isnan(got) and math.isnan(want)
            ), f"{meta.slug}: metric {key} changed: {got!r} != {want!r}"
        else:
            assert math.isclose(
                float(got), float(want), rel_tol=rtol, abs_tol=1e-12
            ), f"{meta.slug}: metric {key} drifted: {got!r} vs recorded {want!r}"


def test_every_expected_file_has_a_reproduction() -> None:
    """No orphaned evidence: a deleted reproduction must not leave a stale record."""
    slugs = {p.stem for p in tutorial_module_paths()}
    recorded = {p.stem for p in (TUTORIAL_DIR / "expected").glob("*.json")}
    assert recorded <= slugs, f"orphaned expected/ files: {sorted(recorded - slugs)}"
