"""The physically meaningful initialization of a representation.

`docs/architecture_principles.md` section 2 defines the term. In short: a
representation defines the *structure and conventions* of physical state at a
declared boundary, and a source defines how that state is *initialized* from
physical source parameters. A source therefore **does not consume an existing
physical representation -- it creates the initial state of one**, which makes it
the only operation in the graph with no input representation.

`sources/` is a **new package**, landed by CHE-210 (R06.5) as a deliberate
architecture change with the owner's decision. Its operations register as
`solver`-kind, because mapping a problem statement into a representation is what
that document calls a solver and there is no fifth operation kind; what separates
this package from `solvers/<backend>/` is that a source has no external backend,
so per-backend organization has nothing to organize. Section 3 records why each
alternative home was worse.

The allowlist row is `sources/ -> problems, representations, numerics`. Only the
last two are exercised today: a source *declaration* belongs in `problems/` when
something needs one, and the edge is declared so that landing it is not a second
architecture change. There is deliberately no edge to `solvers/`, `couplers/`,
`operators/` or `measurements/`: a source is upstream of everything that consumes
state.

Representation-independent as a package, representation-explicit per operation
---------------------------------------------------------------------------------
A source operation may initialize a `ScalarField`, a `RayBundle`, or any other
landed representation, and **which one it returns must be unambiguous in the
public API** -- in the signature's return type and in the descriptor's `output`.
So this package is not partitioned by representation and must not grow a
subpackage per representation, and no constructor here may return one of two
representations depending on its arguments.

Both landed representations are initialized here as of CHE-215 (R06.10):
`collimated_bundle` returns a `RayBundle` and the other three return a
`ScalarField`, side by side in one flat package rather than under a
`sources/wave/` and `sources/ray/` split. R06.5 predicted the collimated bundle by
name and blessed it under this same rule, so item 1 of that ticket needed no scope
change.

This package imports **no backend**. Every source here is arithmetic on the
project's own grid or on the caller's own points. `chromatix.functional.plane_wave`
and `chromatix.functional.gaussian_beam` are cross-checks, and both carry a
`power=` amplitude renormalization this tree does not want.

The four modules
----------------
* `collimated_bundle` -- `collimated_bundle`, one angular mode launched from `(N,
  3)` explicit `(x, y, z)` points (normal incidence is `direction = (0, 0, 1)`,
  not a second function), and `direction_from_angle`, the pure converter from
  `(theta, phi)` to a direction cosine.
* `plane_wave` -- `plane_wave`, a uniform illumination on a grid (normal incidence
  is `k_t = (0, 0)`), and `transverse_wavevector_from_angle`, the pure converter
  from `(theta, phi)` to `k_t` in rad/m.
* `gaussian_beam` -- `gaussian_beam`, a Gaussian **at its waist plane**, where the
  field is a real envelope times the same carrier ramp and is therefore exact.
* `spherical_wave` -- `spherical_wave`, an analytic diverging or converging point
  emitter with the `1/R` amplitude carried and its reference distance declared.

`_grid` is private and holds what the three grid-shaped sources share: the shape
and pitch declarations, the `Frame.origin_index` coordinate axes, and the two
refusals `|k_t| <= n k0` and `|k_t| <= pi/d`. Those refusals are written once on
purpose -- two sources with independently written copies diverge the first time one
is edited, and the symptom is one source accepting a geometry the other refuses,
with nothing in the suite comparing them.

The layer this package is, and the three it is not
---------------------------------------------------
Stated because "make me a beam through a lens" has four plausible homes and only
one right one:

* a **source** *initializes* a representation from physical source parameters. It
  consumes no representation, which is what makes it the only operation in the
  graph with no input.
* an **operator** *modifies* physical state through an element -- the thin-element
  complex transmission and the ideal-lens focal-plane transform are operators.
* **propagation** *moves* state between planes. It is an operator too, and a
  distinct one: nothing here takes a `z` to propagate to.
* a **solver/problem** resolves the *system-dependent* launch conditions -- the
  stop, the entrance pupil position and diameter, the system NA, the pupil map, a
  finite-conjugate aim. None of that is knowable from source parameters alone and
  none of it is inferred here.

Prefer composition over a new constructor
------------------------------------------
A new source is for a field that **cannot** be composed, not for a common
composition. "A plane wave through a lens" is `plane_wave(...)` into the landed
ideal-lens operator. "A truncated spherical wave" is `spherical_wave(...)` into the
landed thin-element operator, which is strictly more expressive than an
`aperture=` argument would be -- any complex mask, not just a hard disc. Adding a
constructor for either would fork the physics between a source and an operator
that already models it.

Minimal examples
----------------
```python
from numpy import column_stack, meshgrid, arange, full
from representations import ReferenceSurface
from sources import (collimated_bundle, direction_from_angle, gaussian_beam,
                     plane_wave, spherical_wave, transverse_wavevector_from_angle)

pupil = ReferenceSurface(name="entrance_pupil", z_m=0.0, medium_index=1.0)

# A collimated bundle at 5 degrees, on a 32 x 48 grid of launch points. Positions
# are (x, y, z) columns while the grid is (y, x): stack x first.
y = (arange(32) - 16) * 2e-6
x = (arange(48) - 24) * 2e-6
grid_y, grid_x = meshgrid(y, x, indexing="ij")
rays = collimated_bundle(
    column_stack([grid_x.ravel(), grid_y.ravel(), full(grid_x.size, 0.0)]),
    direction=direction_from_angle(0.0873, 0.0),
    wavelength_m=0.532e-6,
    reference_surface=pupil,
    # The measure is undeclared by default; state it when the sampling is known.
    measure_weight=full(grid_x.size, 2e-6 * 2e-6),
    measure_kind="quadrature_area_m2",
)

illumination = plane_wave(
    (256, 256), sample_pitch_m=(0.2e-6, 0.2e-6), wavelength_m=0.532e-6,
    reference_surface=pupil,
    transverse_wavevector_rad_per_m=transverse_wavevector_from_angle(
        0.0873, 0.0, wavelength_m=0.532e-6, medium_index=1.0),
)

# w0 is the 1/e *amplitude* radius, hence the 1/e^2 intensity radius.
beam = gaussian_beam(
    (256, 256), sample_pitch_m=(0.2e-6, 0.2e-6), wavelength_m=0.532e-6,
    reference_surface=pupil, waist_radius_m=8e-6,
)

# amplitude is the field at R = 1 m, not a peak; the source is upstream because
# converging=False means exp(+i n k0 R).
point = spherical_wave(
    (256, 256), sample_pitch_m=(0.2e-6, 0.2e-6), wavelength_m=0.532e-6,
    reference_surface=pupil, source_position_m=(0.0, 0.0, -1e-3),
)
```

What is in scope, and what is still not
----------------------------------------
This docstring used to exclude "point sources, Gaussian beams as a source
primitive" outright. **That exclusion was lifted for CHE-215 on the owner's
decision**, deliberately and recorded here rather than left as a docstring that
contradicts `__all__`: a Gaussian at its waist and an analytic spherical wave are
both closed-form fields exact at a declared surface, which is exactly what the
sentence below says a source here is. Note what was *not* lifted -- the reason the
line named point sources at all was the delta-on-a-pixel reading, and that is still
refused; a wave-optics point emitter is `spherical_wave`, with explicit geometry
and checked sampling.

Still not here:

* **spectra and chromatic fields** -- one wavelength per representation; a spectrum
  is several artifacts.
* **polarization** -- every representation here is scalar.
* **partially coherent illumination** -- fully coherent, and not implied by the
  word "source".
* **any physical model of an illumination unit** -- no LED array geometry, no lamp,
  no etendue.
* **delta-function emitters** -- a single nonzero pixel on a `ScalarField` is a
  delta whose spectrum is flat to the Nyquist limit and therefore aliased by
  construction.
* **pupil-aware or finite-conjugate launches** -- aiming needs the stop, the pupil
  and the system NA, which is the solver/problem layer (CHE-207, R05.5).
* **automatic aperture or NA inference** -- no source here inspects a downstream
  element. Truncation composes with the thin-element operator.
* **arbitrary-`z` Gaussian beams** -- off-waist is a *paraxial* solution and
  `ValidityFlag` has no token for it, so it owns a follow-up ticket that extends
  that vocabulary rather than mis-declaring `frozenset()`.

A source here is an analytic field at a declared surface, nothing more.

No source registers an `OperationDescriptor` today, and CHE-215 deliberately did
not invent registration for four functions: per `AGENTS.md` each descriptor's
`output` representation must be unambiguous, and this package now spans two
representations, so when sources enter the registry they go in together. Owed to
R03/R12, which also owns the fact that `sources/` may not import `operations/` and
`operations/` may not import `sources/`, so there is no production registration
site yet.
"""

from sources.collimated_bundle import collimated_bundle, direction_from_angle
from sources.gaussian_beam import gaussian_beam
from sources.plane_wave import plane_wave, transverse_wavevector_from_angle
from sources.spherical_wave import spherical_wave

__all__ = [
    "collimated_bundle",
    "direction_from_angle",
    "gaussian_beam",
    "plane_wave",
    "spherical_wave",
    "transverse_wavevector_from_angle",
]
