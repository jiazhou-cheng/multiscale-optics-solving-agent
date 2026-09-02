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
this package from `backends/<backend>/` is that a source has no external backend,
so per-backend organization has nothing to organize. Section 3 records why each
alternative home was worse.

The allowlist row is `sources/ -> problems, representations, numerics`. Only the
last two are exercised today: a source *declaration* belongs in `problems/` when
something needs one, and the edge is declared so that landing it is not a second
architecture change. There is deliberately no edge to `backends/`, `couplers/`,
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

A source may be described without a system. A ray launch may not
-----------------------------------------------------------------
**CHE-219 (R05.8) removed ray initialization from this package**, and the decision
is recorded here rather than left as an absence. `collimated_bundle` used to live
here and return a `RayBundle` built from caller-supplied points and a shared
direction. That operation creates a ray representation without knowing whether
those points correspond to the entrance pupil, the stop, the first traced surface,
a valid finite-conjugate aim, or the constructed system at all -- and the actual
launch positions and directions of a source *into a system* depend on the stop,
the entrance pupil's location and diameter, every surface preceding the stop, the
object distance, the field, the backend's pupil map and the ray aimer's
convergence behaviour. None of that is knowable from source parameters alone.

So there is exactly one rule and no middle state:

* a **declarative source description** is representation-independent and may live
  here (or, when it is a problem statement rather than a constructor, in
  `problems/` -- `problems.SourceSpec` is one);
* **wave-source construction** that does not describe an optical-system ray launch
  stays here: an analytic field at a declared surface is not aimed at anything;
* a **system-launch `RayBundle`** is produced by the solver that owns the aiming --
  `backends.optiland.launch` -- and by nothing in this package.

`direction_from_angle` left with it, having been audited under the same rule: its
production purpose was to turn a source field into a launched ray direction, so
that responsibility moved with the launch. `transverse_wavevector_from_angle`
stays, because `k_t` on a `ScalarField` grid is not a ray aim. Both functions now
live at `tests/fixtures/ray_bundles.py`, as the test helper they always were in
practice -- nothing in `src/` ever called either.

`sources/` therefore initializes exactly one representation today, the
`ScalarField`, in one flat package. That is a narrowing of R06.10's position and
not a return to R06.5's: the *rule* is unchanged -- one flat package, the return
representation explicit per operation, no subpackage per representation -- and a
future source of a representation that is genuinely initializable without a
system needs no architecture change to land here.

This package imports **no backend**. Every source here is arithmetic on the
project's own grid or on the caller's own points. `chromatix.functional.plane_wave`
and `chromatix.functional.gaussian_beam` are cross-checks, and both carry a
`power=` amplitude renormalization this tree does not want.

The three modules
-----------------
* `plane_wave` -- `plane_wave`, a uniform illumination on a grid (normal incidence
  is `k_t = (0, 0)`), and `transverse_wavevector_from_angle`, the pure converter
  from `(theta, phi)` to `k_t` in rad/m.
* `gaussian_beam` -- `gaussian_beam`, a Gaussian **at its waist plane**, where the
  field is a real envelope times the same carrier ramp and is therefore exact.
* `spherical_wave` -- `spherical_wave`, an analytic diverging or converging point
  emitter with the `1/R` amplitude carried and its reference distance declared.

`_grid` is private and holds what the three sources share: the shape
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
  finite-conjugate aim, the ray aimer and its mode. None of that is knowable from
  source parameters alone, none of it is inferred here, and as of CHE-219 (R05.8)
  none of it is *approximated* here either: the operation that materializes a
  launch is `backends.optiland.launch`, which takes the constructed system as a
  required argument.

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
from representations import ReferenceSurface
from sources import (gaussian_beam, plane_wave, spherical_wave,
                     transverse_wavevector_from_angle)

pupil = ReferenceSurface(name="entrance_pupil", z_m=0.0, medium_index=1.0)

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
  and the system NA, which is the solver/problem layer (CHE-207, R05.5), and as of
  CHE-219 (R05.8) is `backends.optiland.launch`.
* **ray initialization of any kind** -- see the rule above. Not "not yet": a
  system-launch `RayBundle` is not a thing this package can correctly produce.
* **automatic aperture or NA inference** -- no source here inspects a downstream
  element. Truncation composes with the thin-element operator.
* **arbitrary-`z` Gaussian beams** -- off-waist is a *paraxial* solution and
  `ValidityFlag` has no token for it, so it owns a follow-up ticket that extends
  that vocabulary rather than mis-declaring `frozenset()`.

A source here is an analytic field at a declared surface, nothing more.

All three sources are in the production operation catalog, as of CHE-221 (R03.4):
`S_SOURCE_PLANE_WAVE`, `S_SOURCE_GAUSSIAN_BEAM` and `S_SOURCE_SPHERICAL_WAVE`. They
went in together, which is what CHE-215 said the condition was -- per `AGENTS.md`
each descriptor's returned representation must be unambiguous, and all three
declare `ScalarField` as `returns=("scalar_field",)`.

**And each declares `inputs=()`**, which is CHE-222 (R03.5): a source consumes no
upstream representation, and until that ticket the descriptor had no way to say so.
`S_SOURCE_PLANE_WAVE` carried `input="scalar_field"` -- the representation it
*produces*, named on both sides -- which contradicted this package's own docstring,
`docs/architecture_principles.md` §2 and the signature. The schema now refuses a
non-`solver` kind with no input, so "produces without consuming" is a checked
declaration rather than a convention.

**This package still does not import `operations/`, and does not need to.** The
catalog lives inside `operations/` and names each implementation as a
`"module.path:attribute"` string, so the dependency runs in neither direction and
the allowlist is unchanged. What this package declares is `OPERATIONS`, a tuple of
strings naming the three; `tests/operations/test_catalog.py` walks that against the
catalog in both directions, so a fourth source added to `OPERATIONS` cannot land
without a record, and a record naming a fourth cannot land without the tuple.
`OPERATIONS` is itself hand-maintained, which is the honest limit stated beside the
tuple below: a source added to neither is invisible to a gate that compares the two
against each other.
"""

from sources.gaussian_beam import gaussian_beam
from sources.plane_wave import plane_wave, transverse_wavevector_from_angle
from sources.spherical_wave import spherical_wave

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
#: `transverse_wavevector_from_angle` is a pure unit converter, degrees to rad/m, with no
#: representation on either side.
#:
#: The residual failure this cannot catch is someone landing a public operation and
#: not adding it here. That is the honest limit of a mechanical gate -- the two
#: directions checked are catalog-against-this-tuple, not this-tuple-against
#: reality -- and it is the reason the tuple is one line of strings rather than
#: something cleverer.
OPERATIONS: tuple[str, ...] = ("gaussian_beam", "plane_wave", "spherical_wave")

__all__ = [
    "OPERATIONS",
    "gaussian_beam",
    "plane_wave",
    "spherical_wave",
    "transverse_wavevector_from_angle",
]
