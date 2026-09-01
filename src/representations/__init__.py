"""Physical state at a declared boundary.

A representation describes what the light *is* at some place, with every
convention it depends on made explicit and testable: units, axes, frame,
handedness, wavelength, phasor sign, polarization, coherence, normalization,
sampling, reference plane.

`representations/` may import `numerics/` and nothing else in this project. It
must not import a solver, a coupler, or a backend: the moment a representation
knows which package produced it, it has stopped being neutral ground, and the
reference implementation's two solver/coupler import cycles both started that
way.

Target content (R02): exactly one public ray representation and one scalar-field
representation. PSF is **not** here -- an observable derived from state is a
measurement, not a representation. Coherence is a stronger contract on the ray
representation, not a subtype of it.

Four modules, and the order is the dependency order:

* `contracts` -- CHE-175 (R02.3). `ContractError` and `CONTRACT_CODES`, the one
  catchable failure type a coupler branches on, plus the array-intake rules every
  representation applies at construction.
* `geometry` -- CHE-174 (R02.2). `Frame` and `ReferenceSurface`, the declarations
  both representations embed, and the frozen boundary conventions.
* `rays` -- CHE-175 (R02.3). `RayBundle`. Coherence is a contract on it
  (`require_coherent()`), not a subtype of it, and the sampling measure is
  declared separately from the amplitude.
* `scalar` -- CHE-176 (R02.4). `ScalarField`, with typed `validity` in place of a
  provenance string.

That is the whole public physical data model: one ray type, one field type. There
is no PSF type, no second ray carrier, and no base class under either.
"""

from representations.contracts import CONTRACT_CODES, ContractError, require_finite
from representations.geometry import (
    AXIS_ORDER,
    HANDEDNESS,
    ORIGIN_RULE,
    PHASOR,
    PROPAGATION_AXIS,
    SPATIAL_FACTOR,
    Frame,
    ReferenceSurface,
)
from representations.rays import (
    MEASURE_KINDS,
    UNVERIFIED,
    MeasureKind,
    RayBundle,
    direction_norm_tolerance,
)
from representations.scalar import (
    VALIDITY_FLAGS,
    VALIDITY_NOTES,
    ScalarField,
    ValidityFlag,
)

__all__ = [
    "AXIS_ORDER",
    "CONTRACT_CODES",
    "HANDEDNESS",
    "MEASURE_KINDS",
    "ORIGIN_RULE",
    "PHASOR",
    "PROPAGATION_AXIS",
    "SPATIAL_FACTOR",
    "UNVERIFIED",
    "VALIDITY_FLAGS",
    "VALIDITY_NOTES",
    "ContractError",
    "Frame",
    "MeasureKind",
    "RayBundle",
    "ReferenceSurface",
    "ScalarField",
    "ValidityFlag",
    "direction_norm_tolerance",
    "require_finite",
]
