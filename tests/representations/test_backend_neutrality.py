"""Importing `representations/` pulls no backend, checked against `sys.modules`.

CHE-174 (R02.2), acceptance criterion 4, and the R02 parent's criterion 1. The
same subprocess shape as `tests/numerics/test_no_backend_import.py` and for the
same reason: the failure is transitive, so reading the imports of the file in
front of you does not settle it.

`representations/` is where this matters most. A representation that knows which
package produced it has stopped being neutral ground, and the reference
implementation's solver/coupler import cycles both started with a convenience
import at this layer.
"""

from __future__ import annotations

import json
import subprocess
import sys

#: NumPy is deliberately absent, as in the numerics sibling: it is the array
#: vocabulary the others are described in, not a solver or an accelerator. This
#: module happens not to import it either -- geometry is Python floats -- but the
#: criterion is about backends, and R02.3/R02.4 will bring arrays.
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


def test_importing_representations_pulls_no_backend() -> None:
    loaded = _modules_after("import representations")
    assert not loaded & set(BACKENDS), (
        f"importing representations loaded {sorted(loaded & set(BACKENDS))}. A "
        "representation is backend-neutral ground; importing a solver's framework here "
        "makes every consumer of the boundary depend on it."
    )


def test_declaring_a_boundary_pulls_no_backend() -> None:
    """Constructing the types, not merely importing the package."""
    loaded = _modules_after(
        "from representations import Frame, ReferenceSurface\n"
        "Frame()\n"
        "ReferenceSurface(name='exit_pupil', z_m=-3.2e-3, medium_index=1.0)\n"
    )
    assert not loaded & set(BACKENDS), (
        f"declaring a boundary loaded {sorted(loaded & set(BACKENDS))}"
    )


def test_the_check_would_notice_a_backend() -> None:
    """The detection half: prove the subprocess actually sees an import."""
    loaded = _modules_after("import representations\nimport jax")
    assert "jax" in loaded
