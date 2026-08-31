"""Importing `numerics/` pulls no backend, checked against `sys.modules`.

CHE-173 (R02.1), acceptance criterion 1. Checked by running a fresh interpreter
rather than by inspecting the source, because the failure this guards against is
transitive: a module that looks backend-free can import one three levels down,
and reading the imports of the file in front of you will not show it.

The same subprocess covers criterion 1's second half -- *capability listing*
pulls no backend -- because `capability_rows()` is the operation most likely to
grow a backend import later: the honest way to check what a package can compute
in is to ask it, and that is exactly the shortcut this criterion forbids. The
table is a record of what a probe measured, not a live query.
"""

from __future__ import annotations

import json
import subprocess
import sys

#: Not "every third-party package". NumPy is deliberately absent: it is the array
#: vocabulary the other three are described in, `numerics/arrays.py` imports it at
#: module scope, and the criterion is about solver and accelerator frameworks.
BACKENDS = ("jax", "jaxlib", "torch", "optiland", "chromatix")


def _modules_after(statement: str) -> set[str]:
    """Top-level module names loaded by a fresh interpreter running `statement`."""
    source = (
        f"{statement}\n"
        "import sys, json\n"
        "print(json.dumps(sorted({m.split('.')[0] for m in sys.modules})))\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", source], capture_output=True, text=True, check=True
    )
    return set(json.loads(completed.stdout))


def test_importing_numerics_pulls_no_backend() -> None:
    loaded = _modules_after("import numerics")
    assert not loaded & set(BACKENDS), (
        f"importing numerics loaded {sorted(loaded & set(BACKENDS))}. The package is the "
        "bottom of the dependency graph; a backend import here puts that backend at the "
        "bottom too."
    )


def test_listing_capabilities_pulls_no_backend() -> None:
    loaded = _modules_after(
        "from numerics.precision import capability_rows, capabilities_for\n"
        "rows = capability_rows()\n"
        "assert rows, 'the capability table is empty'\n"
        "capabilities_for('M_WAVE_CHROMATIX')\n"
    )
    assert not loaded & set(BACKENDS), (
        f"listing capabilities loaded {sorted(loaded & set(BACKENDS))}. A capability is a "
        "recorded measurement, not a live query -- asking the package would make the "
        "declaration agree with itself by construction."
    )


def test_the_check_would_notice_a_backend() -> None:
    """The detection half: prove the subprocess actually sees an import."""
    loaded = _modules_after("import numerics\nimport jax")
    assert "jax" in loaded
