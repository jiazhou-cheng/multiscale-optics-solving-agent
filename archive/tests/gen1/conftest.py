"""CHE-67: make the archived tests in this generation unrunnable, not merely unselected.

Three mechanisms keep archived tests out of a normal run, in increasing order of
how deliberate the mistake has to be:

1. ``testpaths = ["tests"]`` in ``pyproject.toml`` -- a bare ``pytest`` never
   looks here.
2. ``norecursedirs`` includes ``archive`` -- ``pytest .`` and ``pytest archive``
   do not recurse into this tree either.
3. this hook -- naming an archived *file* explicitly, which is the only way past
   (1) and (2), aborts the whole session with a usage error.

(3) exists because (1) and (2) are configuration a future edit can silently
widen, and because a shell-completed path or a copy-pasted node id from
``docs/testing/test_inventory.md`` (written before this archival, so its paths
all still say ``tests/...``) is exactly the accident this archive is supposed to
be safe against. Measured: with only (1) and (2) in place,
``pytest archive/tests/gen1/tests/test_m1_protocol.py`` collected and ran the
archived tests.

The abort is deliberately session-wide rather than a per-item deselect: a mixed
selection like ``pytest tests archive/tests/gen1/...`` fails loudly instead of
quietly running the half that is still active, so the operator finds out that
what they asked for no longer exists.

Unarchiving is an explicit change, per CHE-67: ``git mv`` the file back under
``tests/``. Deleting or weakening this hook is not unarchiving -- it just removes
the guard, and ``tests/test_suite_layout.py`` fails if this file stops carrying
the abort.
"""

from __future__ import annotations

import pathlib

import pytest

_GENERATION = pathlib.Path(__file__).resolve().parent


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    archived = sorted(
        {
            str(path.relative_to(_GENERATION.parents[2]))
            for item in items
            if _GENERATION in (path := pathlib.Path(str(item.fspath)).resolve()).parents
        }
    )
    if not archived:
        return
    raise pytest.UsageError(
        f"refusing to run {len(archived)} archived test file(s) from "
        f"{_GENERATION.relative_to(_GENERATION.parents[2])}: "
        + ", ".join(archived)
        + ". These tests were archived by CHE-67 (outdated milestone evidence, the "
        "superseded benchmark suite, or an out-of-scope solver adapter) and are kept "
        "for history only. To run one again, unarchive it explicitly: `git mv` it back "
        "under `tests/`. See archive/tests/gen1/README.md."
    )
