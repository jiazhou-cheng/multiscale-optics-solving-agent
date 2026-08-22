"""CHE-57 (PB6): the repo-owned Chromatix example reproductions, as a regression gate.

Each test executes one reproduction from
`knowledge/solvers/chromatix/tutorials/` against the pinned `chromatix==0.6.0`
install (commit `d24bdf0`) and asserts two things:

1. every validation check the reproduction declares still passes, and
2. the recorded evidence in `knowledge/solvers/optiland/tutorials/expected/`
   still describes what the solver does -- numeric metrics are compared with a
   tolerance, and the *set* of checks is compared exactly, so a silently
   dropped check fails the test rather than passing vacuously.

Refresh the recorded evidence with

    ./run.sh python knowledge/solvers/chromatix/tutorials/run_all.py --write-expected

A reproduction that is genuinely stochastic declares its own ``metric_rtol`` in
its ``TutorialMeta`` (with the reason in its docstring); everything else replays
at the deterministic budget.

This file lives in ``tests_tutorial/`` rather than ``tests/`` (CHE-67): the
Chromatix reproductions cost 1020 s, so they are an *on-demand* suite that
``pytest``'s ``testpaths = ["tests"]`` never collects. Run them explicitly:

    ./run.sh pytest -q tests_tutorial          # or: make test-tutorial

They are on-demand, not archived -- they stay maintained, and they need no
unarchive step. Reproductions marked ``slow`` in their ``TutorialMeta`` keep the
``slow`` marker so the historical cost split stays readable, but that no longer
decides anything here: nothing in this file runs in the default suite. All of
these reproductions exercise the real Chromatix engine through JAX, so every test
carries the ``chromatix`` and ``jax`` markers on top of ``integration``, and the
whole module carries ``tutorial`` so ``-m tutorial`` names exactly this suite.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

TUTORIAL_DIR = (
    Path(__file__).resolve().parents[1] / "knowledge" / "solvers" / "chromatix" / "tutorials"
)
sys.path.insert(0, str(TUTORIAL_DIR))

pytest.importorskip("chromatix")
pytest.importorskip("jax")

from _chromatix_harness import (  # noqa: E402
    expected_path,
    load_tutorial_module,
    tutorial_module_paths,
)

pytestmark = [
    pytest.mark.tutorial,
    pytest.mark.chromatix,
    pytest.mark.jax,
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

    rtol = meta.metric_rtol if meta.metric_rtol is not None else METRIC_RTOL
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
            # abs_tol is scaled to the recorded magnitude so a metric whose true
            # value is ~0 (a residual, a symmetry error) is not compared at 1e-12.
            assert math.isclose(
                float(got), float(want), rel_tol=rtol, abs_tol=max(1e-9, 1e-6 * abs(want))
            ), f"{meta.slug}: metric {key} drifted: {got!r} vs recorded {want!r}"


def test_jax_x64_is_pinned_off() -> None:
    """The recorded numbers assume complex64.

    x64 is process-global; something else may set it as an import side effect and the
    adapter registry imports every adapter eagerly, so collection order could
    otherwise flip this process-wide (see `conventions.md`, "Numerical dtype").
    The harness pins it at import; this asserts the pin held.
    """
    import jax
    from _chromatix_harness import pin_jax_precision

    pin_jax_precision()
    assert jax.config.jax_enable_x64 is False


def test_every_expected_file_has_a_reproduction() -> None:
    """No orphaned evidence: a deleted reproduction must not leave a stale record."""
    slugs = {p.stem for p in tutorial_module_paths()}
    recorded = {p.stem for p in (TUTORIAL_DIR / "expected").glob("*.json")}
    assert recorded <= slugs, f"orphaned expected/ files: {sorted(recorded - slugs)}"
