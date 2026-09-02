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
* `focal_plane` -- CHE-209. The ideal lens as the transformation between its two
  focal planes: the one operation here that legitimately *changes* the sample
  pitch, which it declares in float64 and the boundary then checks the backend
  against.

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
    fourier_plane_pitch_m,
    padded_field_bytes,
    padded_shape,
)
from solvers.chromatix.focal_plane import DIRECTIONS, focal_plane_transform
from solvers.chromatix.solver import (
    DERIVATIVE,
    MODELS,
    carrier_phase_rad,
    propagate,
)

#: The public callables in this package that are **semantic operations**, as
#: strings, one per `operations.catalog` record. CHE-221 (R03.4).
#:
#: Strings, and *this package does not import* `operations`: the dependency
#: allowlist gives no implementation package an edge to `operations/`, and an edge
#: would end the one property that package exists to provide -- listing what the
#: project can do would have loaded what it does it with.
#: `tests/operations/test_catalog.py` walks this tuple against the catalog in both
#: directions.
#:
#: Hand-maintained, and deliberately not derived from `__all__`:
#: `carrier_phase_rad`, `edge_energy_fraction`, `fourier_plane_pitch_m`, `padded_shape` and
#: `padded_field_bytes` are sizing and diagnostic helpers over declarations, not operations over
#: a representation. Note that `propagate` carries TWO catalog records -- `S_WAVE_CHROMATIX` the
#: backend and `O_ASM_PROPAGATE` the physical operation -- which the gate allows by keying
#: uniqueness on (implementation, kind) rather than on implementation alone.
#:
#: The residual failure this cannot catch is someone landing a public operation and
#: not adding it here. That is the honest limit of a mechanical gate -- the two
#: directions checked are catalog-against-this-tuple, not this-tuple-against
#: reality -- and it is the reason the tuple is one line of strings rather than
#: something cleverer.
OPERATIONS: tuple[str, ...] = ("focal_plane_transform", "propagate")

__all__ = [
    "CAPABILITIES",
    "DERIVATIVE",
    "DIRECTIONS",
    "EDGE_ENERGY_REPORTING_THRESHOLD",
    "MODELS",
    "OPERATIONS",
    "carrier_phase_rad",
    "edge_energy_fraction",
    "focal_plane_transform",
    "fourier_plane_pitch_m",
    "padded_field_bytes",
    "padded_shape",
    "propagate",
]
