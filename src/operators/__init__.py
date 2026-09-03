"""Physical operators: operations that change the physical state of a field.

An operator consumes a representation and produces one at a *different* physical
state. That is the line `docs/architecture_principles.md` section 2 draws against
a coupler, which changes the *representation* of the same state -- and heavy
numerics do not move an operation across it.

`operators/` may import `representations/`, `couplers/` and `numerics/`, and must
not import a solver or a backend. Neither operator here needs one: an elementwise
multiply and an advance along a direction both happen in whatever array namespace
the artifact already carries.

Three modules:

* `diffractive_surface` (CHE-194, R10.2) -- `diffractive_surface`, the diffractive
  surface as a physical operation: `RayBundle -> RayBundle`, at the surface's own
  reference surface. Its interior converts representation twice and its identity is
  still a physical state change, because an optical surface changes the state. The
  two couplers inside it are R07's and R08's own functions, and the thin element is
  R06.6's -- this module writes no numerics of its own, which is why the
  composition is hard-coded rather than generalized.
* `ray_propagation` (CHE-192, R09.2) -- `propagate_rays`, `RayBundle(S1) ->
  RayBundle(S2)`: physical evolution through one declared medium, with the optical
  path growing by `n * s`. It is CHE-50's declared remedy for the wavelet sum's
  missing curvature term, which is what makes it independently selectable rather
  than a convenience. The first use of the `operators -> couplers` edge: the floor
  on `|d_z|` is `couplers.grazing_floor_for_phase_budget`, because a near-grazing
  ray's arc length is the same unrepresentable optical path R07.4 ported that bound
  for.
* `transmission` (CHE-211, R06.6) -- `complex_transmission`, the single thin element
  `U * A * exp(i phi)`, plus the mask builders that feed it
  (`circular_aperture_amplitude`) and the pure arithmetic that turns a system NA
  into a stop radius in the Fourier plane's own coordinates
  (`numerical_aperture_radius_m`).

**No composite-operator framework**, and the condition is stated rather than
assumed: `docs/architecture_principles.md` permits one only if at least two
production compositions immediately need it, and there is one. `diffractive_surface`
is three named calls in sequence. The way to tell when that changes is that two
functions here want to share a step, not that one of them looks parameterizable.

Not here, deliberately: `phase_mask`, `amplitude_mask`, `pupil` and `grating` as
separate operations. Each is `complex_transmission` with one factor at its
identity, and a project that ships them separately has to keep them consistent
forever.

Wave propagation is also not here, and that is R09.1's decision rather than an
omission: Chromatix owns those numerics and `backends.chromatix.propagate` exposes
them as a `physical_operator`. A forwarding wrapper in this package would do no
numerical work, and relocating semantic ownership is not a reason for a function to
exist.

Also not here: the ideal lens. `backends.chromatix.focal_plane_transform` is a
`physical_operator` too, and it lives with its backend because backend ownership
beats taxonomy -- a forwarding wrapper in this package would put a second name on
one implementation without adding a boundary.
"""

from operators.diffractive_surface import (
    DIFFRACTIVE_MODELS,
    DiffractiveModel,
    DiffractiveSurface,
    diffractive_surface,
)
from operators.ray_propagation import propagate_rays
from operators.transmission import (
    EDGES,
    circular_aperture_amplitude,
    complex_transmission,
    numerical_aperture_radius_m,
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
#: `circular_aperture_amplitude` and `numerical_aperture_radius_m` are mask builders -- they
#: produce an array to hand to `complex_transmission`, and consume and produce no representation.
#:
#: The residual failure this cannot catch is someone landing a public operation and
#: not adding it here. That is the honest limit of a mechanical gate -- the two
#: directions checked are catalog-against-this-tuple, not this-tuple-against
#: reality -- and it is the reason the tuple is one line of strings rather than
#: something cleverer.
OPERATIONS: tuple[str, ...] = (
    "complex_transmission",
    "diffractive_surface",
    "propagate_rays",
)

__all__ = [
    "DIFFRACTIVE_MODELS",
    "EDGES",
    "OPERATIONS",
    "DiffractiveModel",
    "DiffractiveSurface",
    "circular_aperture_amplitude",
    "complex_transmission",
    "diffractive_surface",
    "numerical_aperture_radius_m",
    "propagate_rays",
]
