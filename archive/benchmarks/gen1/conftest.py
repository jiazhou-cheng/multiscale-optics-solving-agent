"""Make this archived generation unrunnable, not merely unselected.

Mirrors ``archive/tests/gen1/conftest.py`` and exists for the same reason: the
two mechanisms that normally keep an archive out of a run -- ``testpaths`` and
``norecursedirs`` in ``pyproject.toml`` -- are configuration a future edit can
silently widen, and naming a file explicitly walks past both.

There is a second reason here that does not apply to the test archive. These are
*benchmark runners*, so the plausible accident is not `pytest archive/...` but
copying a command out of a milestone report -- `M1_BASELINE_REPORT.md` and the
protocol documents quote `./run.sh python benchmarks/level1/L1-RAY-01/...` paths
that no longer exist. Those reports are the milestone record and were not
rewritten, so the paths in them stay stale on purpose; the command simply fails
with "no such file", which is the right outcome.

This hook covers the remaining case: a test file under this tree being collected.

Unarchiving is an explicit ``git mv`` back, justified in a Linear issue. Deleting
this hook is not unarchiving -- it only removes the guard.
"""

from __future__ import annotations

import pathlib

import pytest

_GENERATION = pathlib.Path(__file__).resolve().parent
_REPO_ROOT = _GENERATION.parents[2]


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    archived = sorted(
        {
            str(path.relative_to(_REPO_ROOT))
            for item in items
            if _GENERATION in (path := pathlib.Path(str(item.fspath)).resolve()).parents
        }
    )
    if not archived:
        return
    raise pytest.UsageError(
        f"refusing to run {len(archived)} archived benchmark file(s) from "
        f"{_GENERATION.relative_to(_REPO_ROOT)}: " + ", ".join(archived) + ". These were "
        "archived by CHE-88 (superseded gen1 benchmark suites and the modules only they "
        "consumed) and are kept for history only. To run one again, unarchive it "
        "explicitly: `git mv` it back. See archive/benchmarks/gen1/README.md."
    )
