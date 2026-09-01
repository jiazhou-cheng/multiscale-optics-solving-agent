"""Physical operators: operations that change the physical state of a field.

An operator consumes a representation and produces one at a *different* physical
state. That is the line `docs/architecture_principles.md` section 2 draws against
a coupler, which changes the *representation* of the same state -- and heavy
numerics do not move an operation across it.

`operators/` may import `representations/`, `couplers/` and `numerics/`, and must
not import a solver or a backend. The one operator here needs neither: an
elementwise multiply happens in whatever array namespace the field already
carries.

One module, landed by CHE-211 (R06.6):

* `transmission` -- `complex_transmission`, the single thin element
  `U * A * exp(i phi)`, plus the mask builders that feed it
  (`circular_aperture_amplitude`) and the pure arithmetic that turns a system NA
  into a stop radius in the Fourier plane's own coordinates
  (`numerical_aperture_radius_m`).

Not here, deliberately: `phase_mask`, `amplitude_mask`, `pupil` and `grating` as
separate operations. Each is `complex_transmission` with one factor at its
identity, and a project that ships them separately has to keep them consistent
forever.

Also not here: the ideal lens. `solvers.chromatix.focal_plane_transform` is a
`physical_operator` too, and it lives with its backend because backend ownership
beats taxonomy -- a forwarding wrapper in this package would put a second name on
one implementation without adding a boundary.
"""

from operators.transmission import (
    EDGES,
    circular_aperture_amplitude,
    complex_transmission,
    numerical_aperture_radius_m,
)

__all__ = [
    "EDGES",
    "circular_aperture_amplitude",
    "complex_transmission",
    "numerical_aperture_radius_m",
]
