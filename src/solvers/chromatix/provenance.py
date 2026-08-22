"""What was installed, what ran, and a hash of what it produced.

`_installed_chromatix_provenance` is the interesting one: it reports the
distribution actually resolved, because the pinned commit is what makes this a
verified integration rather than a package name that happened to import.

The array hash mixes dtype and shape into the digest, so a complex64 result is
distinguishable from a complex128 one carrying the same values.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from solvers.chromatix.constants import (
    _PINNED_COMMIT,
    _PINNED_VERSION,
)

if TYPE_CHECKING:
    pass



def _installed_chromatix_provenance() -> tuple[str | None, str | None, list[str]]:
    """Return the *actually installed* chromatix (version, commit) plus warnings.

    The commit is read from the installed distribution's ``direct_url.json``
    (PEP 610), which pip writes for a VCS install. This is the real installed
    revision, not the value copied into ``solver_card.yaml``; a mismatch
    between the two is surfaced as a warning rather than silently ignored.
    """
    warnings: list[str] = []
    try:
        distribution = importlib.metadata.distribution("chromatix")
    except importlib.metadata.PackageNotFoundError:
        return None, None, ["chromatix distribution metadata is not installed."]

    version = distribution.version
    commit: str | None = None
    for entry in distribution.files or []:
        if entry.name == "direct_url.json":
            try:
                commit = json.loads(entry.read_text())["vcs_info"]["commit_id"]
            except (OSError, ValueError, KeyError, TypeError):
                commit = None
            break

    if version != _PINNED_VERSION:
        warnings.append(
            f"installed chromatix version {version!r} differs from the pinned "
            f"{_PINNED_VERSION!r} in knowledge/solvers/chromatix/solver_card.yaml."
        )
    if commit is None:
        warnings.append(
            "installed chromatix commit could not be read from direct_url.json; "
            "the pinned-commit claim cannot be verified from this environment."
        )
    elif commit != _PINNED_COMMIT:
        warnings.append(
            f"installed chromatix commit {commit!r} differs from the pinned "
            f"{_PINNED_COMMIT!r} in knowledge/solvers/chromatix/solver_card.yaml."
        )
    return version, commit, warnings


def _cpu_device_name() -> str:
    """Return an observable CPU description without claiming core isolation."""
    model = platform.processor().strip()
    if not model:
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.lower().startswith("model name"):
                    model = line.split(":", 1)[1].strip()
                    break
        except OSError:
            pass
    return model or platform.machine() or "cpu"


def _scientific_array_hash(arrays: Mapping[str, Any]) -> str:
    """Hash names, dtype, shape, and contiguous bytes independent of file metadata."""
    import numpy as np

    digest = hashlib.sha256()
    for name in sorted(arrays):
        array = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(json.dumps(list(array.shape)).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()
