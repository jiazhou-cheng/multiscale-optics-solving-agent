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
#: vocabulary the others are described in, not a solver or an accelerator. The
#: package imports it at module scope from R02.3 onward and still pulls no solver
#: and no accelerator framework, which is the criterion.
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
    """Constructing every type, not merely importing the package.

    R02's parent criterion 1: the pure representation tests pass while importing
    neither Optiland, Chromatix, JAX nor Torch. Both representations are built
    here, with arrays, because `RayBundle.__post_init__` runs a norm through the
    namespace dispatch and `ScalarField.coordinates()` builds a grid -- either
    could reach for a framework without the import being visible at module level.
    """
    loaded = _modules_after(
        "import numpy as np\n"
        "from representations import Frame, RayBundle, ReferenceSurface, ScalarField\n"
        "surface = ReferenceSurface(name='exit_pupil', z_m=-3.2e-3, medium_index=1.0)\n"
        "Frame()\n"
        "bundle = RayBundle(\n"
        "    positions_m=np.zeros((4, 3)),\n"
        "    directions=np.tile([0.0, 0.0, 1.0], (4, 1)),\n"
        "    wavelength_m=550e-9,\n"
        "    reference_surface=surface,\n"
        ")\n"
        "field = ScalarField(\n"
        "    u=np.ones((4, 6), dtype=np.complex64),\n"
        "    sample_pitch_m=(1e-6, 1e-6),\n"
        "    wavelength_m=550e-9,\n"
        "    reference_surface=surface,\n"
        ")\n"
        "field.coordinates()\n"
        "field.discrete_power()\n"
        "assert bundle.count == 4\n"
    )
    assert not loaded & set(BACKENDS), (
        f"declaring a boundary loaded {sorted(loaded & set(BACKENDS))}"
    )


def test_the_check_would_notice_a_backend() -> None:
    """The detection half: prove the subprocess actually sees an import."""
    loaded = _modules_after("import representations\nimport jax")
    assert "jax" in loaded
