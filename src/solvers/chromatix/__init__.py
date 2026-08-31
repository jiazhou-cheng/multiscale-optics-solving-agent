"""Chromatix scalar-wave propagation, behind an anti-corruption boundary.

CHE-183 / CHE-184 (R06.1 / R06.2). The public surface is one function:

```python
solvers.chromatix.propagate(field, *, distance_m, model) -> ScalarField
```

A `representations.ScalarField` goes in and a `ScalarField` comes out. No
chromatix type and no JAX buffer crosses this line: a NumPy caller gets a NumPy
field back, and `tests/solvers/test_chromatix_boundary.py` asserts that with an
AST walk over every module outside this package plus a `sys.modules` check in a
fresh interpreter.

Two modules, in dependency order:

* `fields` -- CHE-183. The translation: capability negotiation (a `complex128`
  request is refused, not downcast), padding and crop state, the pitch check, the
  padded-shape memory estimate, and `edge_energy_fraction` as the wraparound
  diagnostic.
* `solver` -- CHE-184. The propagation itself, and the carrier-phase answer: a
  returned field states in its typed `validity` whether its phase is absolute or
  carrier-removed, because the two differ by a constant that `|U|^2` cannot see.

Importing this package imports **no backend**: `chromatix` and `jax` are imported
inside `fields.import_backend`, which is called from inside the functions that
need it.

`propagate` is the API. `fields` is exported for the boundary and precision tests
that hold this package to its capability row; everything else in it is
native-facing by construction.
"""

from solvers.chromatix.fields import (
    CAPABILITIES,
    EDGE_ENERGY_REPORTING_THRESHOLD,
    edge_energy_fraction,
    padded_field_bytes,
    padded_shape,
)
from solvers.chromatix.solver import (
    DERIVATIVE,
    MODELS,
    carrier_phase_rad,
    propagate,
)

__all__ = [
    "CAPABILITIES",
    "DERIVATIVE",
    "EDGE_ENERGY_REPORTING_THRESHOLD",
    "MODELS",
    "carrier_phase_rad",
    "edge_energy_fraction",
    "padded_field_bytes",
    "padded_shape",
    "propagate",
]
