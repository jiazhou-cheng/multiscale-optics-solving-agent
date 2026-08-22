"""Where the repository root is, found rather than counted.

Three modules needed "the repository root" and each computed it as
``Path(__file__).resolve().parents[N]``. A parent count is a silent dependency
on a file's directory depth: nothing about it fails at import, and a wrong value
just points somewhere plausible. CHE-89 had to change two of them by hand when
``src/`` lost a level, and CHE-90 broke a third by moving
``metalens_controller.py`` one level deeper -- the resulting ``parents[2]``
pointed at ``src/`` and no test noticed, because the call site was a subprocess
``cwd`` on a GPU path that skips by default.

So the root is located by looking for a marker instead of by counting. That
cannot go silently wrong: either the marker is found or the lookup raises.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

__all__ = ["repository_root"]

#: Files that only exist at the root of a source checkout. ``pyproject.toml``
#: alone is not enough -- a nested build directory can carry one -- so both must
#: be present.
_MARKERS = ("pyproject.toml", "AGENTS.md")


@lru_cache(maxsize=1)
def repository_root() -> Path:
    """The root of this source checkout.

    Raises ``RuntimeError`` when there is no checkout -- an installed wheel, for
    instance. That is the correct outcome rather than a fallback: every caller
    wants a *repository* path (a prompt file, a git command's working
    directory), and none of those exist in a wheel. Returning a plausible
    directory instead would turn a missing checkout into a missing file much
    further downstream.
    """
    here = Path(__file__).resolve()
    for candidate in (here, *here.parents):
        if all((candidate / marker).is_file() for marker in _MARKERS):
            return candidate
    raise RuntimeError(
        f"no repository checkout above {here}: none of its parents contains all of "
        f"{', '.join(_MARKERS)}. This helper locates a *source tree*; code that must "
        "also work from an installed distribution should not call it."
    )
