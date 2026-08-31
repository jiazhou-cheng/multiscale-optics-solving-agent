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

Landed so far:

* `geometry` -- CHE-174 (R02.2). `Frame` and `ReferenceSurface`, the two
  declarations both of those representations embed, plus the convention constants
  they are validated against. It imports nothing at all, `numerics` included: a
  frame is not a numeric policy.

The ray and field types themselves (R02.3, R02.4) are not here yet, and the
package holds no placeholder standing in for them.
"""

from representations.geometry import (
    AXIS_ORDER,
    HANDEDNESS,
    ORIGIN_RULE,
    PROPAGATION_AXIS,
    Frame,
    ReferenceSurface,
)

__all__ = [
    "AXIS_ORDER",
    "HANDEDNESS",
    "ORIGIN_RULE",
    "PROPAGATION_AXIS",
    "Frame",
    "ReferenceSurface",
]
