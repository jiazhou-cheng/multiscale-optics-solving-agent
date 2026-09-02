"""Native Optiland rays to a neutral `RayBundle`: the OPL, the pupil, the measure.

CHE-180 (R05.2). The translation half of the anti-corruption layer, and the one
place in the tree where Optiland's native ray state is read. Everything the
package knows about `RealRays` -- that `.i` is a weight and not an amplitude,
that `.opd` is an accumulator and not a declared optical path, that lengths are
millimetres -- stops here.

Nothing below is new physics. Four measured results are being *applied*, and each
is cited where it is used:

`CHE-30` -- the `RealRays.opd` convention
    Absolute accumulated **optical** (index-weighted) path in the lens geometry
    unit, seeded to zero at the ray launch state. For an object at infinity
    `optiland/fields/field_types/angle.py` aims that launch plane at
    `positions[1] - (EPD - min(positions[1:-1]))`, so **the zero moves when the
    aperture changes**. Every part -- sign, unit, physical meaning, reference --
    was established against manufactured geometries with closed-form answers,
    each with the competing hypothesis it rules out, every case exact to float64
    round-off. `tests/physics/test_optiland_opl_convention.py` re-establishes it.

`CHE-32` -- the exit-pupil handoff
    The exported positions at the exit pupil are each ray's image-space
    *asymptote* at that plane, while the accumulator still ends at the final
    traced image surface. Position and phase would otherwise describe different
    planes. Image space is homogeneous, so the correction is exact:
    `n_image * (z_image - z_plane) / N` per ray, with `n_image` read from the
    prescription rather than assumed to be 1.

`CHE-40` -- the required conditioning
    Phase is formed from `OPL_i - OPL_ref`, never from the absolute value. On
    M3-SINGLET-REF the absolute path is 1.0e4 waves and the piston removed here
    is 1219 waves, against 11.7 waves of actual wavefront. A second, less obvious
    benefit: because the reference is a ray from the same trace, the
    aperture-dependent launch-plane zero is common to every ray and cancels
    *exactly* -- which makes the declared OPL invariant under the one thing CHE-30
    warned makes the absolute value meaningless.

`CHE-207` -- the finite conjugate
    A **point source** launches every ray of one field from a single point, which
    is a degenerate spherical wavefront centred on it, so the CHE-41 term is
    exactly zero rather than merely small. Measured directly on a finite-conjugate
    singlet rather than inferred from the infinite-object result, which is what
    the canceled CHE-46 required before the old refusal could be lifted: the
    launch origin spread is exactly 0.0 in all three coordinates, the launch `opd`
    is exactly 0.0, the launch *directions* spread (a diverging bundle, the mirror
    image of the collimated case), and the captured launch state reproduces
    `Optic.trace` bit-identically. An origin spread that is not zero is refused
    rather than approximated: an extended finite source is a different physical
    problem.

`CHE-41` -- which *surface* the path is measured from
    That cancellation is exact for the launch plane's **location** and says
    nothing about its **orientation**. `opd` is seeded on a plane perpendicular to
    z, which is a wavefront only for a bundle travelling along z. For an off-axis
    collimated bundle the plane and the wavefront differ by
    `n_object * (d0 . r_launch)` -- linear in the launch coordinate, so a piston
    on axis and a tilt off it. Omitting it leaves a pupil OPL that is a clean
    converging sphere aimed at the *axis* at every field angle: on
    M3-REVERSE-TELEPHOTO at `Hy = 0.2` it retained 0.13% of the required tilt and
    put the focus 209 um from the traced chief-ray intersection, with a
    0.072-wave-P-V residual that looked perfectly healthy. That is why it survived
    three tickets of on-axis verification, and why the term is measured here and
    *refused* rather than defaulted when it cannot be.

Three quantities, kept apart
----------------------------
`amplitude` is `sqrt(intensity)`. `RealRays.i` is a real, non-negative,
power-like per-ray weight and carries **no phase**; every radian of a
reconstructed field comes from the optical path. Reading `i` as a complex
amplitude would be a category error, and this module states the mapping rather
than leaving it to be inferred.

`measure_weight` is the absolute per-ray pupil area element, in square metres,
and is **never** folded into the amplitude. The reference implementation did fold
it in (CHE-47, `amplitude = sqrt(w) * dA`); `representations.RayBundle` separates
the two, so the area element travels as the declared measure and the coupler that
discretizes the aperture integral applies it. That is the same physics with the
declaration in the place that can refuse it.

`optical_path_m` is the *declared* path from `declare_optical_path_m`, never the
native accumulator. `to_ray_bundle` refuses an optical path whose reference does
not carry `OPL_REFERENCE_VERSION`, which is what makes "pass `opd_native` as an
OPL" a structured refusal rather than a piston-and-sign gamble.

Why the quadrature lives here
-----------------------------
The hexapolar area weight has exactly one producer -- this solver -- so it stays
inside the package rather than becoming a generic quadrature framework nothing
else uses yet. The *contract* is elsewhere and unchanged: `measure_kind` on
`RayBundle` must be declared or the consumer refuses.

Two translations, and the reason they are not one
-------------------------------------------------
CHE-217 (R05.6) added a second, narrow path: a `RayBundle` the project already
holds, traced *through* a system. `to_ray_bundle` translates rays the solver
generated; `to_traced_ray_bundle` carries a supplied bundle's own amplitude and
quadrature across the trace. Reusing the first for the second is not a shortcut,
it is the defect the second exists to avoid, and both halves of it are silent:

* `amplitude = sqrt(intensity)` is correct on the generated path and *only*
  there, because the solver seeds `intensity = ones_like` without apodization and
  clips by zeroing rather than removing, so `i in {0, 1}` is a survival flag and
  `sqrt(1) = 1`. Applied to a caller's `a_i` it returns `|a_i|` and drops every
  radian of phase. See `AMPLITUDE_SIDECAR_RULE`.
* the generated path's `measure_weight` is an entrance-pupil area element for a
  hexapolar fan. Applied to a supplied bundle it would substitute a pupil
  quadrature for the caller's importance weights -- and R05 deliberately moved the
  quadrature weight off the amplitude so that R07's kernel applies
  `measure_weight` itself, which makes a substituted measure a rescaling of every
  reconstruction downstream that nothing can observe.

So on the supplied-bundle path the amplitude and the measure are the caller's,
the trace may filter the ensemble but may not restate either, and the optical
path is a *composition* rather than a fresh absolute declaration
(`COMPOSED_OPL_REFERENCE_VERSION`).

What CHE-219 (R05.8) took out of this module, and what it left in
-----------------------------------------------------------------
This module translates. It no longer *decides where rays start*.

Three things left for `launch.py`, and all three were reconstructions of launch
state that the trace had already thrown away: the post-hoc pupil measure (which
regenerated the hexapolar fan, re-read the entrance pupil, and recovered a ring
index from traced output), the object-space optical-path reference (which
regenerated the entire launch state through the backend's ray generator and
relied on the second invocation matching the first), and the field normalization
that turns a source's field angle into the backend's pupil-field coordinate.
`to_ray_bundle` now *consumes* the launch declaration those produce.

What stayed is everything that is a translation rather than an initialization:
`declare_optical_path_m` (which applies the object-space term rather than
measuring it), `exit_pupil`, the hexapolar arithmetic itself
(`hexapolar_ring_index` and `hexapolar_area_weight_m2` -- shared, with launch as
their only caller, so there is one implementation of the quadrature), and the
whole supplied-bundle path: `to_native_rays`, `require_launch_surface`,
`to_traced_ray_bundle` and `compose_optical_path_m`. A supplied bundle is already
initialized; the solver does not aim it, resample it, or assign it a pupil
measure.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any

import numpy as np

from numerics import ArrayNamespace, DevicePlacement, Precision, dtype_of, to_namespace
from representations import (
    UNVERIFIED,
    ContractError,
    Frame,
    MeasureKind,
    RayBundle,
    ReferenceSurface,
    direction_norm_tolerance,
)

__all__ = [
    "ADMISSIBLE_OPL_REFERENCES",
    "AMPLITUDE_MAPPING",
    "AMPLITUDE_SIDECAR_RULE",
    "COMPOSED_OPL_REFERENCE_VERSION",
    "LAUNCH_PLANE_WAVEFRONT",
    "LAUNCH_POINT_SOURCE",
    "LAUNCH_SURFACE_TOLERANCE_M",
    "MONOCHROMATIC_WAVELENGTH_RULE",
    "NATIVE_LENGTH_M",
    "NATIVE_WAVELENGTH_M",
    "OPL_REFERENCE_VERSION",
    "OPTILAND_INTENSITY_RULE",
    "REFERENCE_SURFACES",
    "SKIP_OBJECT_SURFACE",
    "SUPPLIED_MEASURE_RULE",
    "SUPPLIED_RAY_SURVIVAL_RULE",
    "compose_optical_path_m",
    "declare_optical_path_m",
    "exit_pupil",
    "hexapolar_area_weight_m2",
    "hexapolar_ray_count",
    "hexapolar_ring_index",
    "require_declared_optical_path",
    "require_launch_surface",
    "surface_positions_m",
    "to_native_rays",
    "to_ray_bundle",
    "to_traced_ray_bundle",
]

#: The lens geometry unit, in metres. CHE-30 part 4: `RealRays.opd` and every
#: coordinate are in the prescription's geometry unit, millimetres for every
#: problem this project states, and `opd` scales exactly with the prescription
#: (measured: a 10x geometry gives a 10x `opd`, ratio error 0.0).
NATIVE_LENGTH_M = 1.0e-3

#: `RealRays.w` and every Optiland wavelength are micrometres.
NATIVE_WAVELENGTH_M = 1.0e-6

#: The surfaces a bundle may be declared on, and what each one means.
#:
#: `image_surface`
#:     The final traced surface. Positions are the traced intersections,
#:     untouched, which is what keeps the frozen ray fingerprints reproducible.
#: `exit_pupil`
#:     The image of the stop in image space, located by `Paraxial.XPL()`, which is
#:     **signed and measured from the image surface** rather than from the global
#:     origin. Positions there are each ray's image-space *asymptote*, not a
#:     physical intersection: the pupil is frequently virtual (on both fixture
#:     systems it is), so the extended line passes back through glass the ray
#:     never travelled in that state. That is the construction the exit pupil is
#:     defined by and what a wavefront-over-the-pupil calculation wants, and it is
#:     not "where the ray is".
REFERENCE_SURFACES: tuple[str, ...] = ("image_surface", "exit_pupil")

#: The version of the OPL reference this module declares, and the prefix
#: `to_ray_bundle` requires before it will carry an optical path.
#:
#: The prefix is load-bearing rather than decorative, and it is the reference
#: implementation's own device: a declaration that only a specific producer can
#: write is what stops the raw accumulator being handed over under a plausible
#: label. Versioned rather than edited in place because a handoff convention is
#: part of the boundary contract -- v1 (CHE-33) named the reference a plane and
#: used it as a wavefront, which was exactly right on axis and dropped the whole
#: convergence tilt off it.
OPL_REFERENCE_VERSION = "optiland-declared-opl/v2"

#: The version of the OPL reference the *supplied-bundle* path declares, and the
#: second prefix `require_declared_optical_path` admits.
#:
#: A separate version rather than a reuse of the one above, because it names a
#: different quantity. `OPL_REFERENCE_VERSION` is an *absolute* declaration this
#: solver constructed from end to end: it fixed the zero at the traced chief ray
#: and named the object-space wavefront the path is measured from. This one is a
#: **composition**: the zero and the reference surface are the incoming bundle's,
#: whatever they were, and all this trace contributed is the increment through the
#: system. Labelling the two the same would tell a consumer that a chief-ray-zeroed
#: absolute path had been handed over when what it holds is somebody else's
#: reference plus an increment, and `require_coherent()` cannot tell them apart.
#:
#: CHE-217 (R05.6) extends the admissible vocabulary here deliberately. It does
#: **not** loosen the check: this prefix is written by exactly one function,
#: `compose_optical_path_m`, and the native accumulator still carries no prefix at
#: all.
COMPOSED_OPL_REFERENCE_VERSION = "optiland-composed-opl/v1"

#: Every OPL reference prefix this package will carry on a bundle, and the whole
#: of what `require_declared_optical_path` admits. Enumerated rather than matched
#: by pattern: a `startswith("optiland-")` test would admit the next label anybody
#: invents, which is exactly the "plausible label" the versioning exists to stop.
ADMISSIBLE_OPL_REFERENCES: tuple[str, ...] = (
    OPL_REFERENCE_VERSION,
    COMPOSED_OPL_REFERENCE_VERSION,
)

#: How `RealRays.i` becomes an amplitude. Stated because it is a modelling
#: decision, and this is the producer that makes it.
#:
#: **This mapping is the solver-generated path's and only its.** See
#: `AMPLITUDE_SIDECAR_RULE` for why it must not be reused for a supplied bundle.
AMPLITUDE_MAPPING = (
    "amplitude = sqrt(intensity); intensity is Optiland RealRays.i, a real "
    "non-negative power-like per-ray weight. It carries no phase, so every radian "
    "of a reconstructed field comes from optical_path_m. No per-ray area factor is "
    "multiplied in and no 1/N is applied: the area element travels separately as "
    "measure_weight, and a traced ray set is a physical ensemble rather than a "
    "Monte-Carlo sample of one."
)

#: What happens to a **supplied** bundle's complex amplitude, and the rule that
#: makes reusing `AMPLITUDE_MAPPING` on this path a defect rather than a shortcut.
#:
#: Reused verbatim in substance from
#: `pre-rewrite-2026-08-30:src/core/coherent_batch.py:89` (`AMPLITUDE_SIDECAR_RULE`).
#: `sqrt(i)` is right on the solver-generated path and *only* there, because
#: `RayGenerator.generate_rays` seeds `intensity = ones_like` when there is no
#: apodization and Optiland clips by zeroing the row rather than removing it, so
#: `i in {0, 1}` is a pure survival flag and `sqrt(1) = 1`. Hand the same output
#: path a caller's `a_i` and it is silently replaced -- by `|a_i|` if the modulus
#: was bridged in as an intensity, dropping every radian of phase, or by `1` if it
#: was not. No error, no diagnostic, and no downstream intensity check can see it.
AMPLITUDE_SIDECAR_RULE = (
    "the complex amplitude is a SIDECAR on this path: it is never passed to the "
    "solver, never reconstructed from the native intensity, and is re-associated "
    "after the trace row for row. sqrt(intensity) is the solver-generated path's "
    "mapping and is wrong here -- it would return |a| and drop the phase, which no "
    "intensity check downstream can detect."
)

#: Why a real intensity crosses into the solver at all, and the one thing that is
#: read back off it. `pre-rewrite-2026-08-30:src/core/coherent_batch.py:94`.
OPTILAND_INTENSITY_RULE = (
    "the native intensity is seeded from |a|^2, formed on this side of the "
    "boundary so the only amplitude-derived quantity that reaches the solver is "
    "unambiguously an intensity, and is read back for exactly one purpose: "
    "deciding which rays the trace clipped. It is never read as an amplitude."
)

#: Why the solver is handed one wavelength rather than one per ray.
#:
#: `RayBundle.wavelength_m` is a scalar by contract, so a per-ray array was always
#: a broadcast of a single number. optiland 0.6.0 memoizes `BaseMaterial.n`/`.k` on
#: the *contents* of whatever it is handed -- `_create_cache_key` evaluates
#: `tuple(ravel(to_numpy(wavelength)))` -- so a per-ray broadcast copies the array
#: to the host, builds an N-element Python tuple and hashes it, four times per
#: surface. CHE-118 measured that at 97% of the trace stage. A size-1 array takes
#: the solver's own documented scalar path and broadcasts against the N-ray
#: geometry, and the equivalence is exact *because* the bundle is monochromatic:
#: with one wavelength there is nothing a per-ray array can say that a scalar
#: cannot.
MONOCHROMATIC_WAVELENGTH_RULE = (
    "one solver invocation is a single-wavelength solve, so the solver is handed a "
    "size-1 wavelength array rather than one entry per ray. The pinned release keys "
    "its refractive-index cache on the array's contents, which made the redundant "
    "broadcast cost O(rays) of host-side tuple construction per surface (CHE-118)."
)

#: What a trace does and does not do to a supplied bundle's membership, and the
#: convention this package settled on. CHE-217 (R05.6) chose it explicitly.
#:
#: `to_ray_bundle` **drops** clipped rows, which is right when the solver also
#: generated the rays: nothing outside holds a per-ray array to stay aligned with.
#: A supplied bundle is the opposite case -- the caller still holds the ensemble it
#: handed over, and dropping rows silently breaks that correspondence. So the row
#: is kept and the amplitude is zeroed, which is the reference implementation's
#: convention (`coherent_trace.py`, `zeroed_amplitude = where(valid, amplitude, 0)`)
#: and keeps the output count equal to the input count.
#:
#: Survival is `i > 0` and finite geometry, and nothing else. A clipped ray's
#: state is frozen at the clip and can be non-finite, which `RayBundle` forbids
#: outright, so its geometry is replaced by the same harmless placeholder
#: `to_ray_bundle` uses. That substitution is safe *because* the amplitude is zero:
#: the row contributes nothing to any reconstruction.
SUPPLIED_RAY_SURVIVAL_RULE = (
    "a trace may filter a supplied bundle but never redefines it: the row count is "
    "preserved so the output aligns with the caller's own arrays row for row, and a "
    "ray the trace did not survive is MARKED by a zeroed amplitude rather than "
    "removed. Survival is intensity > 0 and finite geometry, and nothing else. A "
    "non-surviving row's geometry is a placeholder (position 0, direction +z, path "
    "increment 0) because the frozen native state can be non-finite; it is "
    "unreadable by construction, the zeroed amplitude being what makes it so. One "
    "consequence, stated rather than hidden: a supplied amplitude whose |a|^2 "
    "underflows the compute precision reads as a clipped ray, because a zero "
    "intensity is how the solver reports one. A ray carrying no representable power "
    "is marked, not traced."
)

#: What a trace does to a supplied bundle's sampling measure, which is nothing.
#:
#: The reason is `operators.propagate_rays`'s, verbatim in substance: the
#: quadrature that fixed each plane wavelet's coefficient was taken at the surface
#: where the rays were declared, and a plane wavelet is an infinite plane wave
#: whose coefficient is fixed once. Passing through a surface changes where the
#: wavelet is stated to cross a plane; it does not change what the wavelet is.
#:
#: Declared as a constant rather than written inline because it is the *absence*
#: of a computation, and an absence has nowhere else to be stated. It is also
#: what the launch quadrature must not be reached from on this path: substituting
#: this solver's own entrance-pupil area element for a caller's importance
#: weights rescales every reconstruction downstream by a factor nothing can
#: observe, R05 having deliberately moved the quadrature weight off the amplitude
#: so that R07's kernel applies `measure_weight` itself.
SUPPLIED_MEASURE_RULE = (
    "passed through unchanged, weights and kind both. The quadrature that fixed "
    "each plane wavelet's coefficient was taken at the surface where the rays were "
    "declared, and passing through a surface does not restate it. No pupil area "
    "element was assigned and no ring index was recovered."
)

#: How far the first traced surface may sit from the surface a supplied bundle
#: declares itself on, in metres, before the pairing is refused.
#:
#: A nanometre, reused from the reference implementation's
#: `coherent_trace._PLANE_TOLERANCE_M`. The two are the *same* surface by
#: construction, so any real difference is a setup error rather than round-off --
#: and a float32 representation of a 60 um coordinate is good to about 4e-12 m, so
#: the tolerance is loose enough not to refuse an honest float32 bundle.
LAUNCH_SURFACE_TOLERANCE_M = 1.0e-9

#: How many surfaces a supplied bundle's trace skips: the object surface, and
#: only it.
#:
#: A built `Optic` always carries one, and for an infinite conjugate it sits at
#: `z = -inf`, so rays supplied at a real surface must not be traced through it.
#: One rather than a caller-chosen count on purpose: a skip is not a physical
#: choice, and exposing it would let a caller quietly trace a *sub-system* while
#: every artifact still named the whole prescription. `require_launch_surface`
#: checks the surface this lands on rather than trusting the number.
SKIP_OBJECT_SURFACE = 1

#: Absolute tolerance, in units of ring spacing, for a normalized pupil radius to
#: count as landing on a hexapolar ring. `optiland.distribution` places ring `j`
#: at exactly `rho = j / num_rings` in float64
#: (`r = linspace(0, 1, num_rings + 1)`), so an un-vignetted fan agrees to
#: float64 round-off. A bundle that does not is refused rather than mis-binned.
_RING_TOLERANCE = 1.0e-6

#: The two launch geometries the pinned solver produces, named so the declared
#: optical path can say which one its reference is, rather than leaving a reader
#: to infer it from the problem. They are *declared* here and *produced* by
#: `launch._object_space_reference`, which is the CHE-219 direction: this module
#: reads a launch geometry off the declaration it is handed and never decides one.
#:
#: They are mirror images, and that is the whole reason they need separate
#: arithmetic: at infinity the launch **directions** are common and the origins
#: spread over a plane, so the reference is a plane wavefront and the offset is
#: `n * (d0 . r_launch)`. For a point source the **origin** is common and the
#: directions spread, so the reference is a sphere about that point and the offset
#: is exactly zero.
LAUNCH_PLANE_WAVEFRONT = "plane wavefront of a collimated bundle, launched on a plane normal to z"
LAUNCH_POINT_SOURCE = "spherical wavefront diverging from a single object point"

#: What Optiland reports for a ray it clipped: the intensity is zeroed and the
#: row is kept with its state frozen at the clip. So a positive intensity is the
#: aliveness test, and non-finite geometry is treated the same way -- a NaN
#: position is not a position.
_ALIVE_RULE = (
    "intensity > 0 and every geometric component finite. Optiland clips by zeroing "
    "RealRays.i rather than by removing the row, so a zeroed ray still carries a "
    "position frozen at the clip; it is dropped here rather than travelling inside an "
    "artifact whose contract forbids reading it."
)


def hexapolar_ray_count(num_rings: int) -> int:
    """How many rays an un-vignetted `num_rings` hexapolar fan produces.

    `1 + 3n(n + 1)`: the central ray plus `6j` rays on each of `n` rings. Exposed
    because it is the row count a caller has to match to get a declared measure,
    and computing it at the call site is how a fan gets requested by density and
    checked against a guess.
    """
    if num_rings < 1:
        raise ValueError(f"num_rings must be >= 1, got {num_rings!r}")
    return 1 + 3 * num_rings * (num_rings + 1)


def hexapolar_ring_index(
    pupil_x: np.ndarray[Any, Any], pupil_y: np.ndarray[Any, Any], num_rings: int
) -> np.ndarray[Any, Any]:
    """Recover each ray's hexapolar ring index from its normalized pupil coordinate.

    `pupil_x` / `pupil_y` are the normalized entrance-pupil coordinates the fan was
    sampled on -- the unit disk, ring `j` at radius `j / num_rings` for
    `j = 0 .. num_rings`, with `j = 0` the single central ray.

    Raises:
        ContractError: `MEASURE_UNDECLARED` if any ray's normalized radius misses
            every ring by more than `_RING_TOLERANCE`. That means the fan was
            vignetted -- a dropped ray shifts which rows correspond to which ring
            -- or was never hexapolar, and in either case no area element can be
            assigned rather than one being guessed.
    """
    if num_rings < 1:
        raise ValueError(f"num_rings must be >= 1, got {num_rings!r}")
    x = np.asarray(pupil_x, dtype=np.float64)
    y = np.asarray(pupil_y, dtype=np.float64)
    if x.shape != y.shape:
        raise ContractError(
            "SHAPE_MISMATCH",
            f"pupil_x {x.shape} must match pupil_y {y.shape}",
            declaration="pupil_y",
        )

    ring_float = np.hypot(x, y) * num_rings
    ring_index = np.round(ring_float).astype(np.int64)
    residual = np.abs(ring_float - ring_index)
    if bool(np.any(residual > _RING_TOLERANCE)) or bool(
        np.any((ring_index < 0) | (ring_index > num_rings))
    ):
        worst = int(np.argmax(residual))
        raise ContractError(
            "MEASURE_UNDECLARED",
            (
                f"{int(np.sum(residual > _RING_TOLERANCE))} of {x.size} rays do not land "
                f"on a ring of a {num_rings}-ring hexapolar fan; the worst has normalized "
                f"radius {float(np.hypot(x, y)[worst]):.9f} against the nearest ring at "
                f"{float(ring_index[worst]) / num_rings:.9f}. No pupil area element can be "
                "assigned to these rays, and assigning a uniform one would invent a "
                "quadrature."
            ),
            declaration="measure_weight",
            remedy=(
                "Declare a quadrature measure only for an un-vignetted hexapolar fan, "
                f"where the traced count is 1 + 3n(n + 1) = {hexapolar_ray_count(num_rings)}. "
                "Otherwise leave measure_kind 'undeclared' so the consumer refuses."
            ),
        )
    return ring_index


def hexapolar_area_weight_m2(
    ring_index: np.ndarray[Any, Any], num_rings: int, aperture_radius_m: float
) -> np.ndarray[Any, Any]:
    """The absolute per-ray pupil area element, in square metres.

    Ring `j`'s nominal cell is `pi a^2 / (3 n^2)` -- the aperture area over the
    `3n^2` rays a hexapolar fan asymptotically approaches. Every interior ring's
    `6j` points share that cell exactly, which is the right quadrature there. Two
    measured boundary corrections:

    * the central ray (`j = 0`) gets **3/4** of the nominal cell: it represents a
      disk of radius `a / (2n)`, i.e. `pi a^2 / (4 n^2)`.
    * the outermost ring (`j = n`) gets **1/2**: it sits exactly on `rho = a` and
      represents only the inner half of its annulus, there being no ray beyond the
      rim to average with.

    With these, `sum_i dA_i = pi a^2 (1 + 1/(4 n^2))` exactly, so the total
    converges on the aperture area as the ring count grows. That convergence is
    the point: without an *absolute* area element the reconstructed discrete power
    grew as `(ray count)^2.0024` instead of converging, and treating every ray as
    an equal-weight sample left a sensor-plane residual of 3.84e-3 that the
    corrected measure collapses to 4.07e-4.
    """
    if num_rings < 1:
        raise ValueError(f"num_rings must be >= 1, got {num_rings!r}")
    if not math.isfinite(aperture_radius_m) or aperture_radius_m <= 0.0:
        raise ContractError(
            "UNIT_NOT_SI",
            f"aperture_radius_m must be positive and finite, got {aperture_radius_m!r}; it "
            "is the physical radius the relative cell areas are scaled to",
            declaration="measure_weight",
        )
    index = np.asarray(ring_index)
    nominal_m2 = math.pi * aperture_radius_m**2 / (3.0 * num_rings**2)
    weight = np.full(index.shape, nominal_m2, dtype=np.float64)
    weight[index == 0] = 0.75 * nominal_m2
    weight[index == num_rings] = 0.5 * nominal_m2
    return weight


def _host(value: Any) -> np.ndarray[Any, Any]:
    """One solver array on the host, with its precision preserved.

    Deliberately not `np.asarray(..., dtype=np.float64)`. Forcing float64 was the
    single line that made a float32 or GPU trace indistinguishable from a float64
    host one downstream; for the default host/float64 path this is a no-op, so the
    frozen fingerprints are unchanged.
    """
    import optiland.backend.utils as be_utils

    return np.asarray(be_utils.to_numpy(value))


def _scalar(thunk: Any) -> float | None:
    """A finite scalar read from the system, or `None` when it cannot be read.

    Everything including the attribute lookup happens inside the guard: an
    absence has to degrade to "not available" so a caller can refuse on it,
    rather than turning into an unrelated crash.
    """
    try:
        value = float(np.asarray(_host(thunk())).ravel()[0])
    except Exception:
        return None
    return value if math.isfinite(value) else None


def exit_pupil(lens: Any, *, image_plane_z_mm: float) -> dict[str, Any]:
    """Read the exit pupil from the system, and say what the reading means.

    `Paraxial.XPL()` is signed and measured **from the image surface**, so the
    plane is `image_z + XPL`. Getting that wrong yields a plausible plane rather
    than an error, which is why it is stated here once.

    Raises:
        ContractError: `MISSING_DECLARATION` when the paraxial solver cannot
            supply `XPL()`/`XPD()`, or supplies a non-finite one. A telecentric or
            degenerate configuration has no finite exit pupil and this module will
            not substitute one -- an unresolved plane must not be mistaken for a
            plane at z = 0.
    """
    location_from_image_mm = _scalar(lens.paraxial.XPL)
    diameter_mm = _scalar(lens.paraxial.XPD)
    if location_from_image_mm is None or diameter_mm is None:
        raise ContractError(
            "MISSING_DECLARATION",
            "an exit-pupil reference surface was requested, but the system's paraxial "
            f"solver did not supply a finite XPL()/XPD() (XPL={location_from_image_mm!r}, "
            f"XPD={diameter_mm!r}). The plane is read from the system, never guessed, so "
            "this refuses rather than exporting rays against an invented reference.",
            declaration="reference_surface",
            remedy="Declare 'image_surface', or use a system with a finite exit pupil.",
        )

    pupil_z_mm = image_plane_z_mm + location_from_image_mm
    beyond = [
        z
        for z in (
            _scalar(lambda surface=surface: surface.geometry.cs.z)
            for surface in lens.surfaces.surfaces[:-1]
        )
        if z is not None and z > pupil_z_mm
    ]
    return {
        "z_mm": pupil_z_mm,
        "location_from_image_mm": location_from_image_mm,
        "diameter_mm": diameter_mm,
        # A virtual pupil is not a wrong plane, but it changes what a position at
        # that plane is, so the fact is reported rather than left to be assumed.
        "is_virtual": bool(beyond),
        "refracting_surfaces_beyond_pupil_z_mm": beyond,
    }


def declare_optical_path_m(
    opd_native: np.ndarray[Any, Any],
    *,
    direction_z: np.ndarray[Any, Any],
    pupil_radius_m: np.ndarray[Any, Any],
    image_z_mm: float,
    plane_z_mm: float,
    image_space_index: float,
    object_space: dict[str, Any],
    on_axis: bool,
) -> tuple[np.ndarray[Any, Any], str]:
    """Turn the native accumulator into a declared optical path, or refuse.

    **This is the only function that produces an admissible optical path**, and
    the reference string it returns is the only one `to_ray_bundle` accepts. Four
    steps, in this order, each the application of a measured result:

    1. native unit -> metres (CHE-30 part 4);
    2. transfer from the traced image surface to the declared plane (CHE-32).
       Exact rather than approximate, because image space is homogeneous;
    3. move the reference from Optiland's launch *state* onto the incoming
       *wavefront*, when and only when the term varies across the bundle. A
       constant term is a piston that step 4 removes exactly, so adding it would
       only spend float precision on something that cannot survive -- and not
       adding it is what keeps every on-axis number bit-identical rather than one
       rounding away. Which wavefront depends on the source, and the declaration
       names it:

       * an object at **infinity** (CHE-41): the launch plane is not a wavefront of
         a tilted bundle, and the difference `n * (d0 . r_launch)` is the whole
         convergence tilt off axis;
       * a **point source** (CHE-207): the launch state is one point, so it already
         *is* a wavefront and the term is exactly zero;
    4. remove the chief-ray piston (CHE-40), sign 'ray minus chief', so a larger
       value means a longer optical path.

    Computed in float64 regardless of the trace's precision, for the reason step 3
    gives: the removed piston is of order 1e4 waves and the signal is of order 10.

    Raises:
        ContractError: `MISSING_DECLARATION` when the object-space term is
            unavailable and the field is *not* on axis. That case cannot be
            repaired downstream -- the missing quantity is a function of the launch
            coordinate and no object-space coordinate survives into the exported
            pupil arrays -- and declaring the path without it produces a clean
            converging sphere aimed at the axis, 209 um from the traced chief-ray
            intersection with a 0.072-wave residual that looks healthy.
    """
    optical_path_m = np.asarray(opd_native, dtype=np.float64) * NATIVE_LENGTH_M

    # 2. Image surface -> declared plane. A no-op when they are the same plane, so
    #    there is one code path rather than a branch that can disagree with itself.
    if bool(np.any(np.asarray(direction_z) == 0.0)):
        raise ContractError(
            "NON_UNIT_DIRECTION",
            "a ray has N = 0 and never reaches the traced image surface, so its optical "
            "path cannot be referred to the declared plane",
            declaration="directions",
        )
    step_m = ((image_z_mm - plane_z_mm) * NATIVE_LENGTH_M) / np.asarray(
        direction_z, dtype=np.float64
    )
    optical_path_m = optical_path_m - image_space_index * step_m

    # 3. Launch plane -> incoming wavefront.
    if not object_space["available"]:
        if not on_axis:
            raise ContractError(
                "MISSING_DECLARATION",
                (
                    "this trace is off axis and the object-space reference term "
                    "n_object * (d0 . r_launch) is unavailable, so the accumulated path is "
                    "measured from a plane perpendicular to z rather than from a wavefront "
                    f"of the incoming bundle. The solver declined it because: "
                    f"{object_space['reason']}"
                ),
                declaration="optical_path_reference",
                remedy=(
                    "Trace on axis, or make the term available. It cannot be repaired "
                    "downstream: it is linear in the LAUNCH coordinate and the exported "
                    "pupil arrays carry no object-space coordinate to reconstruct it from."
                ),
            )
        object_space_note = (
            "unavailable, and the traced field is on axis, so the incoming bundle travels "
            "along z, Optiland's launch plane IS a wavefront of it, and the term would be "
            f"a constant that step 4 removes exactly. Accepted for this field only. Reason: "
            f"{object_space['reason']}"
        )
    elif object_space["launch_geometry"] == LAUNCH_POINT_SOURCE:
        # A point source. The term is exactly zero for every ray, because the
        # launch point IS the wavefront -- see `_point_source_reference`. Nothing
        # to add, and the reference surface named in the declaration below is the
        # sphere rather than the plane.
        object_space_note = (
            "exactly zero for every ray: the source is a single point, so the launch state "
            "is a degenerate spherical wavefront centred on it and there is no optical path "
            "from that wavefront to the launch point. Measured, not assumed -- the launch "
            f"origin spread is 0.0 in all three coordinates at "
            f"{object_space['launch_point_native']!r} in native units"
        )
    elif object_space["span_native"] == 0.0:
        object_space_note = (
            "present and constant across the bundle: a pure piston, not applied, because "
            "step 4 removes it exactly and adding it would move an on-axis number that "
            "nothing about this reference has any business moving"
        )
    else:
        optical_path_m = optical_path_m + object_space["offset_native"] * NATIVE_LENGTH_M
        object_space_note = (
            "applied: the term varies across the bundle by "
            f"{object_space['span_native'] * NATIVE_LENGTH_M:.6e} m, so the omission is a "
            "tilt rather than a piston, and off axis this term IS the convergence tilt"
        )

    # 4. Remove the chief-ray piston.
    radius = np.asarray(pupil_radius_m, dtype=np.float64)
    chief = int(np.argmin(radius))
    reference_m = float(optical_path_m[chief])
    relative_m = optical_path_m - reference_m

    # Which surface the path is measured FROM, named rather than assumed. Getting
    # this wrong in the declaration would be worse than getting it wrong in the
    # arithmetic: a consumer would be told the reference is a plane wavefront when
    # it is a diverging sphere, and no downstream check reads the object distance.
    reference_wavefront = object_space.get("launch_geometry") or (
        f"{LAUNCH_PLANE_WAVEFRONT} (assumed: the term was unavailable and the field is "
        "on axis, where the two references agree)"
    )
    declaration = (
        f"{OPL_REFERENCE_VERSION}: zero at the traced chief ray (smallest pupil radius, "
        f"row {chief}, rho = {float(radius[chief]):.6e} m) evaluated at the declared plane "
        f"z = {plane_z_mm * NATIVE_LENGTH_M!r} m; sign 'ray minus chief', so a larger value "
        f"is a longer optical path. The SURFACE the path is measured from is the "
        f"{reference_wavefront}, not the solver's launch state. Image-space index "
        f"{image_space_index!r}, read from the prescription. Object-space reference term: "
        f"{object_space_note}. Removed piston {reference_m:.9e} m."
    )
    return relative_m, declaration


def to_ray_bundle(
    lens: Any,
    traced: Any,
    *,
    launch: dict[str, Any],
    reference_surface: str,
) -> tuple[RayBundle, dict[str, Any]]:
    """Translate one native trace into a neutral `RayBundle` plus diagnostics.

    `launch` is the declaration `solvers.optiland.launch.launch` returned for the
    rays this trace was run on. **CHE-219 (R05.8) made that an input rather than
    something reconstructed here.** This function used to receive
    `field + wavelength_um + num_rings` and rebuild, from the traced output, two
    things that existed only at launch: the hexapolar ring identity each ray's
    pupil-area measure comes from, and the whole launch state the CHE-41
    object-space reference term is measured on. Both are now read off `launch`,
    and neither the distribution nor the entrance pupil is consulted here at all.

    The physics is unchanged -- CHE-30, CHE-41 and CHE-207 all still apply through
    `declare_optical_path_m`, from the same term, with the same sign and the same
    reference version. What changed is that the term describes the rays that were
    launched instead of a second, hopefully identical, launch.

    Returns the bundle and a diagnostics mapping. Nothing in either is native: no
    `RealRays`, no intensity, no accumulator, no millimetre.

    Raises:
        ContractError: the launch declaration does not describe this trace row for
            row, the exit pupil could not be resolved, the image-space index could
            not be read, the optical path could not be declared, or every traced
            ray was clipped.
    """
    wavelength_um = float(launch["wavelength_um"])
    if reference_surface not in REFERENCE_SURFACES:
        raise ContractError(
            "MISSING_DECLARATION",
            f"reference_surface must be one of {list(REFERENCE_SURFACES)}, got "
            f"{reference_surface!r}. There is no default: the difference between the two "
            "is the whole pupil-to-focus distance.",
            declaration="reference_surface",
        )

    native = {
        name: _host(value)
        for name, value in (
            ("x", traced.x),
            ("y", traced.y),
            ("z", traced.z),
            ("L", traced.L),
            ("M", traced.M),
            ("N", traced.N),
            ("intensity", traced.i),
            ("wavelength", traced.w),
            ("accumulator", traced.opd),
        )
    }
    shapes = {name: array.shape for name, array in native.items()}
    if native["x"].ndim != 1 or len(set(shapes.values())) != 1:
        raise ContractError(
            "SHAPE_MISMATCH",
            f"the solver returned ray columns that are not equal-length 1-D arrays: {shapes!r}",
            declaration="positions_m",
        )
    traced_count = int(native["x"].size)
    if traced_count == 0:
        raise ContractError(
            "EMPTY_ENSEMBLE",
            "the solver returned no rays; there is nothing to declare",
            declaration="positions_m",
        )
    # The launch declaration is the authority on what these rays *are*, so it has
    # to describe them row for row before any of it is applied. It is not a
    # degradation: the measure, the amplitude and the object-space term are all
    # per-ray arrays taken at launch, and aligning them against a different number
    # of rows would silently attribute one ray's pupil cell to another.
    launch_count = int(launch["ray_count"])
    if launch_count != traced_count:
        raise ContractError(
            "SHAPE_MISMATCH",
            f"the launch declaration describes {launch_count} rays but the trace exported "
            f"{traced_count}, so the two cannot be matched row for row. Optiland keeps a "
            "clipped ray's row and zeroes its intensity rather than removing it, so a "
            "count that differs means this declaration came from a different launch.",
            declaration="positions_m",
        )

    alive = (
        (native["intensity"] > 0)
        & np.isfinite(native["x"])
        & np.isfinite(native["y"])
        & np.isfinite(native["z"])
        & np.isfinite(native["L"])
        & np.isfinite(native["M"])
        & np.isfinite(native["N"])
        & np.isfinite(native["accumulator"])
    )
    alive_count = int(np.count_nonzero(alive))
    if alive_count == 0:
        raise ContractError(
            "EMPTY_ENSEMBLE",
            f"all {traced_count} traced rays were clipped or returned non-finite state, so "
            "the bundle would be empty. " + _ALIVE_RULE,
            declaration="positions_m",
        )
    if alive_count != traced_count:
        # A clipped ray's state is frozen at the clip, and it is dropped at the end
        # of this function -- but every array below is still full length so the
        # regenerated pupil and launch states can be matched row for row. Harmless
        # placeholders keep the intermediate arithmetic finite (a frozen ray can
        # legitimately have N = 0, which would otherwise divide by zero in a
        # quantity that is then discarded). Substituting is safe precisely because
        # nothing computed from a dead row survives into the bundle.
        native = {name: array.copy() for name, array in native.items()}
        for name, placeholder in (
            ("x", 0.0),
            ("y", 0.0),
            ("z", 0.0),
            ("L", 0.0),
            ("M", 0.0),
            ("N", 1.0),
            ("accumulator", 0.0),
        ):
            native[name][~alive] = placeholder

    wavelengths = np.unique(native["wavelength"])
    if wavelengths.size != 1:
        raise ContractError(
            "MISSING_DECLARATION",
            f"the trace carries {wavelengths.size} distinct wavelengths; a RayBundle is "
            "monochromatic per evaluation and a spectrum is several bundles",
            declaration="wavelength_m",
        )
    wavelength_m = float(wavelengths[0]) * NATIVE_WAVELENGTH_M
    # The second thing the declaration has to agree with the trace about. The row
    # count above establishes that the two describe the same *number* of rays; this
    # establishes that they describe the same *light*, which the row count cannot.
    # It matters because `wavelength_um` from the declaration is what both material
    # index lookups below are evaluated at, so a declaration from another
    # wavelength would refer the optical path through indices the trace never used.
    #
    # Compared in the precision the *trace* ran in, not in float64. The solver
    # stamps the requested wavelength onto the rays through its own array
    # constructor, so a float32 trace of 0.55 um comes back as 0.550000011920929
    # and an exact float64 comparison would refuse every float32 trace -- measured,
    # and caught by `test_float32_traces_in_float32_and_says_so`. Casting the
    # declaration into the traced dtype is exact in both precisions and stays a
    # test for "a different wavelength" rather than becoming a tolerance.
    declared_native = float(np.asarray(wavelength_um, dtype=wavelengths.dtype))
    if float(wavelengths[0]) != declared_native:
        raise ContractError(
            "MISSING_DECLARATION",
            f"the launch declaration is for wavelength {wavelength_um!r} but the trace "
            f"carries {float(wavelengths[0])!r} (native units), so this declaration did "
            "not come from the launch that was traced. Both refractive indices below are "
            "read at the declared wavelength, so the mismatch would refer the optical "
            "path through a medium the trace did not travel in.",
            declaration="wavelength_m",
        )

    image_z_mm = _scalar(lambda: lens.surfaces.surfaces[-1].geometry.cs.z)
    if image_z_mm is None:
        raise ContractError(
            "MISSING_DECLARATION",
            "the final surface's axial position could not be read from the system, so no "
            "reference surface can be located",
            declaration="reference_surface",
        )

    if reference_surface == "exit_pupil":
        pupil = exit_pupil(lens, image_plane_z_mm=image_z_mm)
        plane_z_mm = pupil["z_mm"]
        # Each ray's image-space asymptote at the pupil plane. A reparameterization
        # along the ray, so directions are the traced values and no optical path is
        # added or removed here -- the OPL transfer is step 2 of the declaration.
        step_mm = (plane_z_mm - native["z"]) / native["N"]
        position_mm = (
            native["x"] + native["L"] * step_mm,
            native["y"] + native["M"] * step_mm,
            np.full_like(native["x"], plane_z_mm),
        )
    else:
        pupil = None
        plane_z_mm = image_z_mm
        # The traced coordinates, untouched, which is what keeps the frozen ray
        # fingerprints reproducible.
        position_mm = (native["x"], native["y"], native["z"])

    image_space_index = _scalar(
        lambda: lens.surfaces.surfaces[-1].material_pre.n(wavelength_um)
    )
    if image_space_index is None or image_space_index <= 0.0:
        raise ContractError(
            "MISSING_DECLARATION",
            "the image-space refractive index could not be read from the prescription, and "
            "the optical path between the traced image surface and the declared plane is "
            "index-weighted. 'It is air' is a property of two particular systems, not of "
            "lenses.",
            declaration="reference_surface",
        )

    object_space = launch["object_space"]
    # The chief ray is the smallest pupil radius among the rays that survived: a
    # clipped ray's frozen position could otherwise win the argmin and make the
    # removed piston the path of a ray nobody is allowed to read.
    pupil_radius_m = np.where(
        alive, np.hypot(position_mm[0], position_mm[1]) * NATIVE_LENGTH_M, np.inf
    )
    optical_path_m, optical_path_reference = declare_optical_path_m(
        native["accumulator"],
        direction_z=native["N"],
        pupil_radius_m=pupil_radius_m,
        image_z_mm=image_z_mm,
        plane_z_mm=plane_z_mm,
        image_space_index=image_space_index,
        object_space=object_space,
        on_axis=tuple(launch["field"]) == (0.0, 0.0),
    )

    # Declared at launch, by the layer that chose the sampling. Nothing here
    # regenerates a distribution, recovers a ring index, or reads a row count to
    # guess what the pupil quadrature was.
    measure_weight = launch["measure_weight"]
    measure_kind: MeasureKind = launch["measure_kind"]
    measure_note = launch["measure_note"]

    positions = np.stack([column[alive] for column in position_mm], axis=1) * NATIVE_LENGTH_M
    directions = np.stack([native[name][alive] for name in ("L", "M", "N")], axis=1)
    bundle = RayBundle(
        positions_m=positions,
        directions=directions,
        wavelength_m=wavelength_m,
        reference_surface=ReferenceSurface(
            name=reference_surface,
            z_m=plane_z_mm * NATIVE_LENGTH_M,
            medium_index=image_space_index,
        ),
        frame=Frame(),
        # sqrt of a real, non-negative power-like weight: a phase-free amplitude.
        # `adopt_array(widen_real=True)` widens it to the complex dtype of the
        # SAME precision, so a float32 trace does not gain ten digits here.
        amplitude=np.sqrt(native["intensity"][alive]),
        optical_path_m=optical_path_m[alive],
        optical_path_reference=optical_path_reference,
        measure_weight=None if measure_weight is None else measure_weight[alive],
        measure_kind=measure_kind,
    )
    require_declared_optical_path(bundle)

    diagnostics: dict[str, Any] = {
        "traced_ray_count": traced_count,
        "alive_ray_count": alive_count,
        "clipped_ray_count": traced_count - alive_count,
        "alive_rule": _ALIVE_RULE,
        "reference_surface": reference_surface,
        "reference_surface_z_m": plane_z_mm * NATIVE_LENGTH_M,
        "traced_image_surface_z_m": image_z_mm * NATIVE_LENGTH_M,
        "image_space_refractive_index": image_space_index,
        "amplitude_mapping": AMPLITUDE_MAPPING,
        "measure": measure_note,
        "optical_path_reference": optical_path_reference,
        # Where the rays came from, carried through rather than re-derived --
        # CHE-219. `aiming` in particular cannot be recovered from traced output at
        # all, which is why it travels as a declaration.
        "launch_ray_count": launch_count,
        "launch_geometry": launch["launch_geometry"],
        "launch_surface_z_m": launch["launch_surface"].z_m,
        "launch_aiming": launch["aiming"],
        "launch_field": tuple(launch["field"]),
        "object_space_reference_available": object_space["available"],
        "object_space_reference_reason": object_space["reason"],
        "object_space_reference_span_m": (
            None
            if object_space["span_native"] is None
            else object_space["span_native"] * NATIVE_LENGTH_M
        ),
        "observed_dtype": str(dtype_of(positions)),
        "direction_norm_tolerance": direction_norm_tolerance(dtype_of(directions)),
        "polarization": "missing; RealRays carries no polarization state, and none is fabricated",
        "coherence": (
            "the bundle declares an amplitude and a referenced optical path, so "
            "RayBundle.require_coherent() accepts it. Sequential rays are not themselves a "
            "coherent complex field; the declaration is what makes one derivable."
        ),
        "exit_pupil": (
            None
            if pupil is None
            else {
                "source": "optic.paraxial.XPL() and XPD(), read not constructed",
                "z_m": pupil["z_mm"] * NATIVE_LENGTH_M,
                "location_from_image_m": pupil["location_from_image_mm"] * NATIVE_LENGTH_M,
                "diameter_m": pupil["diameter_mm"] * NATIVE_LENGTH_M,
                "is_virtual": pupil["is_virtual"],
                "position_semantics": (
                    "each ray's image-space ASYMPTOTE at the pupil plane, not a physical "
                    "intersection; the pupil is virtual on both fixture systems"
                ),
            }
        ),
    }
    return bundle, diagnostics


# ---------------------------------------------------------------------------
# The supplied-bundle path. CHE-217 (R05.6).
#
# Everything above translates rays the solver generated. Everything below carries
# a `RayBundle` the project already holds *through* a system, and the difference
# that matters is one sentence: on this path the amplitude and the quadrature are
# the caller's, and the trace may filter the ensemble but may not restate either.
# ---------------------------------------------------------------------------


def surface_positions_m(lens: Any) -> list[float]:
    """Every surface's axial position, in metres, read from the built system.

    The object surface is included and is `-inf` for an infinite conjugate, which
    is the whole reason `require_launch_surface` exists: the position is read
    rather than assumed, so "the first surface a supplied bundle is traced
    through" is a fact about the constructed lens and not about the skip count.
    """
    positions = _host(lens.surfaces.positions).ravel()
    return [float(value) * NATIVE_LENGTH_M for value in positions]


def require_launch_surface(
    lens: Any, surface: ReferenceSurface, *, skip: int, wavelength_um: float
) -> float:
    """Check that `surface` is where the trace actually starts, and in what medium.

    Returns the object-space refractive index the trace will weight its first
    transfer by, read from the prescription.

    A built `Optic` always carries an object surface, at `z = -inf` for an
    infinite conjugate, so rays supplied at a real surface must not be traced
    through it and `skip` exists. But a skip count that is merely *asserted*
    substitutes one silent failure for another: getting it wrong is not a crash,
    it is a trace through a different optical system that returns plausible
    numbers. So the first non-skipped surface is located and compared.

    The **medium** is checked alongside the coordinate, and for the same reason
    the coordinate is. The composed optical path this path declares is
    `opl_in + n-weighted accumulator`, and the two halves have to be measured in
    one medium: a caller declaring `n = 1` at a surface the prescription puts in
    glass gets a composed path that is wrong by the index ratio over the first
    transfer, with no error and nothing downstream that can attribute it. "It is
    air" is a property of two particular systems, not of lenses.

    Raises:
        ContractError: `SHAPE_MISMATCH` if `skip` leaves no surface to trace;
            `FRAME_MISMATCH` if the first traced surface is not where the bundle
            says it is; `MISSING_DECLARATION` if the object-space index cannot be
            read, or disagrees with the medium the bundle declares.
    """
    positions_m = surface_positions_m(lens)
    if skip >= len(positions_m):
        raise ContractError(
            "SHAPE_MISMATCH",
            f"skip={skip} leaves no surface to trace in a {len(positions_m)}-surface "
            "system, so there is nothing for the supplied rays to pass through",
            declaration="skip",
        )
    first_z_m = positions_m[skip]
    if not math.isfinite(first_z_m) or abs(first_z_m - surface.z_m) > LAUNCH_SURFACE_TOLERANCE_M:
        raise ContractError(
            "FRAME_MISMATCH",
            (
                f"the first traced surface sits at z = {first_z_m!r} m but the bundle "
                f"declares itself on {surface.name!r} at z = {surface.z_m!r} m, a "
                f"difference of more than {LAUNCH_SURFACE_TOLERANCE_M!r} m. Tracing would "
                "propagate the rays through a different optical system than the one they "
                "were declared against, and would succeed while doing it."
            ),
            declaration="reference_surface",
            remedy=(
                "Build the problem so that the surface following the object surface is the "
                "one the rays are declared on, or declare the rays on the surface the "
                "trace starts at."
            ),
        )

    index = _scalar(lambda: lens.surfaces.surfaces[skip - 1].material_post.n(wavelength_um))
    if index is None or index <= 0.0:
        raise ContractError(
            "MISSING_DECLARATION",
            "the object-space refractive index could not be read from the prescription, "
            "and the optical path this trace accumulates from the launch surface is "
            "index-weighted, so it cannot be composed with the path the bundle already "
            "carries.",
            declaration="reference_surface",
        )
    # Relative, because an index is dimensionless: the length tolerance above has
    # nothing to say about it, and a catalog index is a computed float rather than
    # a transcribed one.
    if not math.isclose(index, surface.medium_index, rel_tol=1.0e-9, abs_tol=0.0):
        raise ContractError(
            "MISSING_DECLARATION",
            (
                f"the bundle declares its surface in a medium of index "
                f"{surface.medium_index!r}, but the prescription puts the space before the "
                f"first traced surface in one of {index!r}. The incoming optical path was "
                "measured in the first medium and the accumulator will be weighted by the "
                "second, so the composed path would be wrong by that ratio over the first "
                "transfer -- and nothing downstream can attribute it back here."
            ),
            declaration="reference_surface",
            remedy=(
                "Declare the medium the rays actually travel in, or trace them through a "
                "prescription whose object space is that medium."
            ),
        )
    return index


def to_native_rays(
    rays: RayBundle,
    *,
    namespace: ArrayNamespace,
    device: DevicePlacement,
    precision: Precision,
) -> Any:
    """Build the native ray object a supplied `RayBundle` describes.

    The inbound half of the anti-corruption layer for this path, and the *only*
    place a project representation becomes native ray state. Three conversions
    happen here and each is stated rather than implied:

    * positions metres -> the geometry unit (`NATIVE_LENGTH_M`);
    * the vacuum wavelength metres -> micrometres, as **one** scalar in a size-1
      array -- see `MONOCHROMATIC_WAVELENGTH_RULE`;
    * the complex amplitude -> `|a|^2`, formed on *this* side of the boundary so
      that the only amplitude-derived quantity the solver ever sees is
      unambiguously an intensity. See `OPTILAND_INTENSITY_RULE` and
      `AMPLITUDE_SIDECAR_RULE`: the complex amplitude itself does not cross, and
      the pinned solver has no complex ray field to put it in.

    Directions are dimensionless and cross unconverted. Nothing else crosses:
    `measure_weight` is a property of how the caller sampled its pupil and the
    solver has no concept of it, so passing it in would be meaningless and
    reading one back would be an invention.
    """
    from optiland.rays import RealRays

    amplitude, _ = rays.require_coherent()
    xp = rays.xp
    # |a|^2 in the bundle's own namespace, so no complex buffer is handed across.
    intensity = xp.abs(amplitude) ** 2

    def native(value: Any) -> Any:
        return to_namespace(
            value,
            namespace=namespace,
            device=None if namespace is ArrayNamespace.NUMPY else device,
            dtype=precision.real_dtype,
        )

    positions = native(rays.positions_m / NATIVE_LENGTH_M)
    directions = native(rays.directions)
    return RealRays(
        positions[:, 0],
        positions[:, 1],
        positions[:, 2],
        directions[:, 0],
        directions[:, 1],
        directions[:, 2],
        native(intensity),
        # ONE wavelength, not N copies of it.
        native(xp.asarray([rays.wavelength_m / NATIVE_WAVELENGTH_M])),
    )


def compose_optical_path_m(
    rays: RayBundle,
    accumulated_m: Any,
    *,
    launch_surface_z_m: float,
    image_surface_z_m: float,
    object_space_index: float,
) -> tuple[Any, str]:
    """Add this trace's accumulated path to the one the bundle already carries.

    The decision this path had to make explicitly, and the three things it is
    **not**:

    * it is not the native accumulator promoted to an optical path. A freshly
      constructed native ray seeds its accumulator to zero at the launch state, so
      the accumulator alone is the path *through the system only* and says nothing
      about where the light was before it -- which for a coupler's output is the
      whole optical path so far;
    * it is not a re-declaration. No chief-ray piston is removed and no reference
      surface is renamed. The incoming bundle already fixed both, and re-zeroing
      would silently discard a reference a consumer may be matching against;
    * it is not `declare_optical_path_m`'s quantity, so it does not carry that
      function's version prefix. See `COMPOSED_OPL_REFERENCE_VERSION`.

    The sum is therefore measured from exactly the surface the incoming reference
    names, with exactly the incoming sign convention, extended through the traced
    surfaces. `accumulated_m` is the native accumulator already scaled to metres
    and already in the bundle's own array state, so the addition neither converts
    a unit nor moves a buffer.

    Precision note, stated because it is inherited rather than chosen: the sum is
    formed in the bundle's dtype. Unlike `declare_optical_path_m` this path cannot
    remove the absolute piston -- the incoming reference is what fixes the zero --
    so a float32 bundle carries the *absolute* composed path at float32 relative
    accuracy. That is the caller's declared precision and it is reported rather
    than silently upgraded.
    """
    reference = rays.optical_path_reference or ""
    declaration = (
        f"{COMPOSED_OPL_REFERENCE_VERSION}: the optical path the SUPPLIED bundle already "
        f"carried, whose own reference is {reference!r}, PLUS the index-weighted optical "
        f"path this trace accumulated from the launch surface "
        f"{rays.reference_surface.name!r} at z = {launch_surface_z_m!r} m through to the "
        f"traced image surface at z = {image_surface_z_m!r} m. The zero, the reference "
        f"surface and the sign convention are the incoming bundle's and are unchanged: no "
        f"chief-ray piston was removed and no reference was renamed, so a consumer "
        f"matching against the incoming declaration still matches. Object-space index "
        f"{object_space_index!r}, read from the prescription and checked against the "
        f"medium the bundle declares. {SUPPLIED_RAY_SURVIVAL_RULE}"
    )
    return rays.optical_path_m + accumulated_m, declaration


def to_traced_ray_bundle(
    lens: Any,
    traced: Any,
    rays: RayBundle,
    *,
    launch_surface_z_m: float,
    object_space_index: float,
    wavelength_um: float,
) -> tuple[RayBundle, dict[str, Any]]:
    """Translate a trace of a *supplied* bundle back into that bundle, evolved.

    The sibling of `to_ray_bundle`, and the difference is the whole ticket: this
    one takes the amplitude and the measure from `rays` rather than deriving them
    from the trace. There is no `sqrt(intensity)` here, no regenerated hexapolar
    fan, no entrance-pupil area element and no ring index -- the caller's
    quadrature is what the caller declared, and substituting a pupil quadrature
    for it would rescale every reconstruction downstream by a factor nothing can
    observe.

    Built with `dataclasses.replace`, which is not a stylistic choice: the fields
    this operation does not touch -- `wavelength_m`, `frame`, `measure_weight`,
    `measure_kind`, `phasor` -- are then not merely re-assigned to the same value,
    they are never named at all, and a future edit cannot restate one by accident.

    Returns the bundle and a diagnostics mapping, on `to_ray_bundle`'s convention.
    Nothing in either is native.

    Raises:
        ContractError: the image-space index could not be read, or every supplied
            ray was clipped.
    """
    native = {
        name: _host(value)
        for name, value in (
            ("x", traced.x),
            ("y", traced.y),
            ("z", traced.z),
            ("L", traced.L),
            ("M", traced.M),
            ("N", traced.N),
            ("intensity", traced.i),
            ("accumulator", traced.opd),
        )
    }
    shapes = {name: array.shape for name, array in native.items()}
    if native["x"].ndim != 1 or len(set(shapes.values())) != 1:
        raise ContractError(
            "SHAPE_MISMATCH",
            f"the solver returned ray columns that are not equal-length 1-D arrays: {shapes!r}",
            declaration="positions_m",
        )
    if int(native["x"].size) != rays.count:
        raise ContractError(
            "SHAPE_MISMATCH",
            f"the trace returned {int(native['x'].size)} rays for the {rays.count} that "
            "were supplied, so the two cannot be matched row for row and the caller's "
            "amplitude cannot be re-associated",
            declaration="positions_m",
        )

    # `_ALIVE_RULE`, applied to a supplied ensemble. Intensity and finite geometry,
    # and nothing else -- no aperture model of this package's own, which is R05.9's.
    alive = (
        (native["intensity"] > 0)
        & np.isfinite(native["x"])
        & np.isfinite(native["y"])
        & np.isfinite(native["z"])
        & np.isfinite(native["L"])
        & np.isfinite(native["M"])
        & np.isfinite(native["N"])
        & np.isfinite(native["accumulator"])
    )
    alive_count = int(np.count_nonzero(alive))
    if alive_count == 0:
        raise ContractError(
            "EMPTY_ENSEMBLE",
            f"none of the {rays.count} supplied rays survived the trace, so every "
            "amplitude would be zero and the bundle would carry no light. " + _ALIVE_RULE,
            declaration="positions_m",
        )
    if alive_count != rays.count:
        # See `SUPPLIED_RAY_SURVIVAL_RULE`: the row stays, the geometry becomes a
        # placeholder because the frozen native state can be non-finite and
        # `RayBundle` forbids that outright, and the amplitude is zeroed below,
        # which is what makes the placeholder unreadable rather than merely wrong.
        native = {name: array.copy() for name, array in native.items()}
        for name, placeholder in (
            ("x", 0.0),
            ("y", 0.0),
            ("z", 0.0),
            ("L", 0.0),
            ("M", 0.0),
            ("N", 1.0),
            ("accumulator", 0.0),
        ):
            native[name][~alive] = placeholder

    image_z_m = _scalar(lambda: lens.surfaces.surfaces[-1].geometry.cs.z)
    if image_z_m is None:
        raise ContractError(
            "MISSING_DECLARATION",
            "the final surface's axial position could not be read from the system, so the "
            "surface the traced bundle is declared on cannot be located",
            declaration="reference_surface",
        )
    image_z_m *= NATIVE_LENGTH_M
    image_space_index = _scalar(
        lambda: lens.surfaces.surfaces[-1].material_pre.n(wavelength_um)
    )
    if image_space_index is None or image_space_index <= 0.0:
        raise ContractError(
            "MISSING_DECLARATION",
            "the image-space refractive index could not be read from the prescription, and "
            "the traced bundle's surface declares the medium it sits in. 'It is air' is a "
            "property of two particular systems, not of lenses.",
            declaration="reference_surface",
        )

    # Back into the state the caller handed over, so the untouched amplitude and
    # measure and the evolved geometry are one artifact rather than two halves in
    # two ecosystems -- which `RayBundle` refuses anyway.
    def home(value: Any) -> Any:
        return to_namespace(
            value,
            namespace=rays.state.namespace,
            device=None if rays.state.namespace is ArrayNamespace.NUMPY else rays.state.device,
            dtype=rays.state.dtype,
        )

    xp = rays.xp
    survived = home(alive.astype(np.float64)) > 0.0
    optical_path_m, optical_path_reference = compose_optical_path_m(
        rays,
        home(native["accumulator"] * NATIVE_LENGTH_M),
        launch_surface_z_m=launch_surface_z_m,
        image_surface_z_m=image_z_m,
        object_space_index=object_space_index,
    )
    bundle = dataclasses.replace(
        rays,
        positions_m=home(
            np.stack([native[name] for name in ("x", "y", "z")], axis=1) * NATIVE_LENGTH_M
        ),
        directions=home(np.stack([native[name] for name in ("L", "M", "N")], axis=1)),
        reference_surface=ReferenceSurface(
            name="image_surface",
            z_m=image_z_m,
            medium_index=image_space_index,
        ),
        # The caller's own coefficients, re-associated row for row and zeroed
        # exactly where the trace did not survive. Never `sqrt(intensity)`.
        amplitude=xp.where(survived, rays.amplitude, xp.zeros_like(rays.amplitude)),
        optical_path_m=optical_path_m,
        optical_path_reference=optical_path_reference,
    )
    require_declared_optical_path(bundle)

    diagnostics: dict[str, Any] = {
        "supplied_ray_count": rays.count,
        "surviving_ray_count": alive_count,
        "marked_ray_count": rays.count - alive_count,
        "alive_rule": _ALIVE_RULE,
        "survival_rule": SUPPLIED_RAY_SURVIVAL_RULE,
        "amplitude_handling": AMPLITUDE_SIDECAR_RULE,
        "intensity_handling": OPTILAND_INTENSITY_RULE,
        "wavelength_handling": MONOCHROMATIC_WAVELENGTH_RULE,
        "launch_surface_z_m": launch_surface_z_m,
        "launch_surface_tolerance_m": LAUNCH_SURFACE_TOLERANCE_M,
        "object_space_refractive_index": object_space_index,
        "reference_surface": "image_surface",
        "reference_surface_z_m": image_z_m,
        "image_space_refractive_index": image_space_index,
        "optical_path_reference": optical_path_reference,
        "measure_kind": rays.measure_kind,
        "measure": SUPPLIED_MEASURE_RULE,
        "observed_dtype": str(dtype_of(bundle.positions_m)),
        "direction_norm_tolerance": direction_norm_tolerance(dtype_of(bundle.directions)),
        "polarization": "missing; the native ray carries none on this path, and none is fabricated",
    }
    return bundle, diagnostics


def require_declared_optical_path(bundle: RayBundle) -> None:
    """Refuse a bundle whose optical path is the native accumulator in disguise.

    Two jobs, and both are the same check:

    * a **post-condition** on `to_ray_bundle`, which calls it on every bundle it
      emits. Not redundant with the call it follows: it is what makes "the emitted
      path came from `declare_optical_path_m`" a checked property of the artifact
      rather than a property of the current control flow.
    * a **callable refusal** for any consumer holding a bundle that claims to have
      come from this solver. Scaling the native accumulator into metres yields a
      quantity with plausible magnitude, plausible units and a plausible-looking
      spread. What it does not have is a declared reference -- and a wrong
      reference is a harmless piston while a wrong *sign* conjugates the
      wavefront, so a converging beam reconstructs as a diverging one and no
      intensity check can tell the difference.

    So the only admissible optical path is one whose reference carries a prefix
    from `ADMISSIBLE_OPL_REFERENCES`, and each of those two prefixes is written by
    exactly one function: `declare_optical_path_m` for a path this solver
    constructed absolutely, and `compose_optical_path_m` for one extended through
    a system from a path a caller already held. CHE-217 (R05.6) added the second
    **by enumerating it**, which is not the same as loosening the check: the
    native accumulator carries no prefix at all and is refused exactly as before.

    A bundle with no optical path at all passes: carrying no phase is honest, and
    `RayBundle.require_coherent()` is what refuses to read one that is not there.
    """
    if bundle.optical_path_m is None:
        return
    reference = bundle.optical_path_reference or ""
    if not reference.startswith(ADMISSIBLE_OPL_REFERENCES):
        raise ContractError(
            "OPL_REFERENCE_UNVERIFIED",
            (
                f"the optical path is declared {reference!r}, which is not a reference this "
                f"solver produced. Optiland's `opd` is a native accumulator, not a declared "
                f"physical quantity: it is absolute, its zero moves with the aperture, and "
                f"its orientation is a plane rather than a wavefront. Passing it through as "
                f"an optical path length is refused."
            ),
            declaration="optical_path_reference",
            remedy=(
                f"Obtain the path from declare_optical_path_m or compose_optical_path_m, "
                f"whose references start with one of {list(ADMISSIBLE_OPL_REFERENCES)}, or "
                f"carry it with the reference declared {UNVERIFIED!r} so require_coherent() "
                f"refuses to read it as a phase."
            ),
        )
