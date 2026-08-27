"""B3-4F-REAL and B4-4F-REAL: the same modulation through a *real* 4f relay.

CHE-145 (M2.9), the second rung of the M2 system ladder. The topology is::

    object field -> C_WAVE_TO_RAY
                 -> ray propagation through a real refractive group (M_RAY_OPTILAND)
                 -> DiffractiveInteraction, model=full_field, at the Fourier plane
                 -> ray propagation through a real refractive group (M_RAY_OPTILAND)
                 -> C_RAY_TO_WAVE coherent sensor reconstruction

B3-4F-IDEAL (M2.8) is the *same* modulation -- literally the same mask
constructor, the same ``samples_per_period`` axis and the same three checked
orders -- through two ideal lenses realized as ``fft2``/``ifft2``. Here the
ideal lenses are replaced by two real Newport-KBX058-geometry N-BK7 singlets
and nothing else changes. That is the experimental control the ticket asks for:
only the relay changes.

Why this rung needs rays at all
-------------------------------
The ideal relay is wave-solvable in closed form. A real refractive group is not:
its wave aberration is a function of where a ray crosses the glass, and there is
no analytic kernel for ``object -> aberrated Fourier plane``. So the relay is
traced, the modulation is applied to the *coherently accumulated* field on the
one common plane every ray crosses, and the sensor field is reconstructed from
the outgoing rays. All three steps are the shipping path; none of the physics
lives in this family.

The reference, and why there is no oracle away from the limit
-------------------------------------------------------------
There is deliberately **no** exact reference for an aberrated 4f relay carrying
a high-frequency phase modulation. Credibility comes from the *limiting case*:
the aberration-free limit of this system **is** 4F-1. Third-order spherical
aberration of a singlet scales as the fourth power of the height at which the
ray crosses the glass, so shrinking the used semi-aperture drives the relay
onto the ideal relay at a known rate, and that rate -- not a tolerance
somebody chose -- is what makes the agreement checkable.

That splits the ticket cleanly into two families:

``B3-4F-REAL``
    The paraxial limit. Gated, against the 4F-1 answer at the instance's own
    modulation parameters. Its instances sweep ``used_semi_aperture_mm`` and its
    claim is the convergence.
``B4-4F-REAL``
    Everything away from the limit -- wide aperture, off-axis field, a different
    modulation frequency, a different modulation type. Category B4, so it is
    **structurally incapable** of gating: every number is the *measured*
    departure from 4F-1, reported as data, never as an error.

The prescription, and the aberration law it obeys
-------------------------------------------------
Both groups are the same real singlet: Newport KBX058 geometry (equiconvex,
R = +/-77.265 mm, 5.102 mm centre thickness, 22.86 mm clear aperture), N-BK7
entered as its stated d-line index 1.5168 rather than a dispersion curve,
because this rung is monochromatic and reproduces a stated index -- the same
choice ``benchmarks/probes/ray_wave/demo3_hologram_lens.py`` documents for its
own singlet. The built system measures ``f2 = 75.60377181 mm`` with front and
back focal distances both 73.90280714 mm, so the 4f spacing is exact rather
than nominal: object at the front focal plane of group 1, modulation at the
shared focal plane, sensor at the back focal plane of group 2.

The authoring probe traced a fan from the object-plane origin through group 1
and measured the optical path to the shared focal plane against the axial ray:

    landing height (mm)   0.0   1.360   2.718   4.071   5.416   6.751
    W (waves)             0.0  -0.0053 -0.0845 -0.4280 -1.3523 -3.3001

which is the third-order spherical-aberration ``h**4`` law to about 2%:
3.3001 / 0.4280 = 7.71 against the law's (6.751/4.071)**4 = 7.56, the residual
being fifth-order. :func:`peak_wave_aberration_waves` is that law with its single
coefficient anchored at the widest measured point, and it is what the
``PARAXIAL_LIMIT`` predicate is built on. The 2% is not carried into anything:
what the gating tolerance is derived from is the MEASURED ratio between the
departure from 4F-1 and this law's value, and that ratio is constant to 4% across
a 256x span in aberration, so a 2% error in the law is absorbed by the very
measurement that calibrates it.

How the object field becomes rays, and why the sampling is what it is
--------------------------------------------------------------------
The object launch is ``C_WAVE_TO_RAY`` at full enumeration: every propagating
bin of the object grid's angular spectrum, emitted from every sample position of
the object grid. Neither axis is redundant, and the reason is the plane-wavelet
representation itself:

* a ray's **direction** at the launch plane fixes *where* it crosses the glass
  and therefore which aberration zone it samples -- and, after a Fourier-
  transforming group, where it lands on the shared focal plane;
* a ray's **launch position** fixes its *direction* on the shared focal plane,
  and it is the sum over launch positions that carries the transverse structure
  of the reconstructed field, because ``C_RAY_TO_WAVE`` sums plane wavelets and
  a plane wavelet is a delta in k-space, not a point in x.

So the two axes swap roles across a Fourier-transforming group, and the launch
position grid must be the *full* grid: it is a discrete Fourier sum, and
subsampling it aliases rather than coarsening. Measured while authoring: taking
every second launch position on the shared focal plane puts the modulation's
n = +/-2 orders on top of the n = -/+2 positions and the field departure
saturates near 0.25 independent of aperture, close to ``|J_2(1.5)| = 0.232`` --
the amplitude of the order the subsampling folds in.

The two sampling walls this rung actually hits
----------------------------------------------
Both are declared as validity predicates rather than left in a comment, because
each one caps a parameter range the ticket asks for.

``FOURIER_PLANE_FIELD_CAPACITY``
    The object grid IS the field of view, and that is not obvious -- displacing
    the *launch positions* instead looks free and does not work. Every launch
    position carries the same global spectrum, so the field a position sees is
    that spectrum's own Fourier series evaluated there, which is periodic with
    the object grid's extent: shifting the launch set by whole samples
    reproduces the same object exactly while still moving the rays, i.e. it
    samples the aberration off axis and the field on axis. Measured while
    authoring: it reports a 4.7e4 field departure, because the normalization
    peak lands where there is no light. So the displacement goes in the array,
    the array has to hold it, and with ``pitch_object = lambda f / (2 R)`` the
    reachable field angle obeys

        theta_max * R = (object_grid_n / 2 - 4 * object_waist_pixels) * lambda / 2

    -- 2.35e-6 m rad at ``object_grid_n = 32`` and ``object_waist_pixels = 2``,
    so 0.0337 deg at R = 4 mm and 0.192 deg at R = 0.7 mm. A hard trade between
    field angle and the very
    aperture that produces the aberration being characterized. This is the same
    wall a prior CHE-145 exploration reported as "the usable object half-angle
    caps at 0.5 deg"; as a product it is a property of the discretization rather
    than a number to be tuned around, and half a degree at R = 4 mm would need
    ``object_grid_n`` near 250 against a LAUNCHED ray count of
    ``object_grid_n ** 4`` (4e9 rays against 1.05e6 at 32).

``MODULATION_ORDER_SEPARATION``
    The order copies must not overlap at the sensor: the copy spacing is
    ``grid_n / samples_per_period`` pixels and each copy is
    ``object_waist_pixels`` wide. This is why ``samples_per_period = 16`` --
    B3-4F-IDEAL's own best-behaved value -- is *not* reachable here: it would
    need ``grid_n >= 96``, and the outgoing ray count is ``grid_n ** 4``
    (8.3e7 rays at grid_n = 96 against 5.1e6 at grid_n = 48).

A third thing turned out not to be a wall at all, and it is recorded because a
reader will expect it to be one: **the shared plane's axial position is not
load-bearing here.** The field on that plane is the object's own spectrum, so its
angular content is the object's angular size seen from the group -- 8.4e-4 rad for
the 63 um waist at ``used_semi_aperture_mm = 0.7`` -- and the depth of focus is
``lambda / NA**2 = 0.83 m``. Displacing the plane by 2 mm moves
``order_power_relative_l2`` from 2.679e-4 to 2.668e-4. That is why
``modulation-off-the-focal-plane`` is demonstrated by moving the modulation to the
*object* plane, and why the axial null is reported beside it rather than dressed
up as a demonstration.

What these families do not attempt
----------------------------------
No optimization, no gradients, no partial coherence -- CHE-145's stated
non-goals. And no claim that the aberrated result is *validated*: away from the
paraxial limit it is characterization, which is what category B4 exists to say
structurally.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from core.precision import ArrayNamespace, DeviceKind, DType
from verification.families.b3_4f_ideal import CHECKED_ORDERS, order_coefficients
from verification.families.predicates import fractional_margin
from verification.families.registry import register
from verification.families.schema import (
    BenchmarkCategory,
    BenchmarkFamily,
    BenchmarkLayer,
    ClaimKind,
    ExecutionParameter,
    ExecutionPolicy,
    FamilyOracle,
    GateDisposition,
    GateStatus,
    Invariant,
    Metric,
    NegativeControl,
    NumericalParameter,
    Oracle,
    OracleIndependence,
    PhysicalParameter,
    RepresentationParameter,
    SamplerAbsentReason,
    StochasticPolicy,
    Tolerance,
    ToleranceBasis,
    ValidityBasis,
    ValidityPredicate,
)
from verification.status import VerificationStatus

__all__ = [
    "B3_4F_REAL",
    "B4_4F_REAL",
    "DIFFRACTIVE_MODEL",
    "PARAXIAL_LIMIT_WAVES",
    "PRESCRIPTION",
    "field_angle_deg",
    "ideal_4f_answer",
    "object_pitch_m",
    "peak_wave_aberration_waves",
    "residual_ray_angle_rad",
]


#: The real refractive group, one declaration read by both the family and the
#: driver so the geometry a record describes cannot drift from the geometry that
#: ran. ``front_focal_distance_mm`` / ``back_focal_distance_mm`` /
#: ``effective_focal_length_mm`` are READ BACK from the built Optiland system
#: (``paraxial.F1()``, ``last_thickness + paraxial.F2()``, ``paraxial.f2()``) in
#: the authoring probe, not taken from the catalog: the 4f spacing has to be the
#: built system's own focal planes or the relay carries a defocus that would be
#: indistinguishable from an aberration.
PRESCRIPTION: dict[str, Any] = {
    "component": "Newport KBX058 geometry, equiconvex N-BK7 singlet",
    "radius_1_mm": 77.265,
    "radius_2_mm": -77.265,
    "centre_thickness_mm": 5.102,
    "refractive_index": 1.5168,
    "index_note": (
        "N-BK7's stated d-line index as an IdealMaterialSpec rather than the "
        "catalog dispersion curve: this rung is monochromatic at 0.5876 um, so it "
        "reproduces a stated index, not a dispersion curve. Same choice, and same "
        "reason, as demo3's singlet"
    ),
    "clear_aperture_mm": 22.86,
    "wavelength_m": 0.5876e-6,
    "effective_focal_length_mm": 75.60377181,
    "front_focal_distance_mm": 73.90280714,
    "back_focal_distance_mm": 73.90280714,
}

#: The wave aberration this prescription puts on a ray that crosses group 1 at
#: 6.7509 mm, in waves, measured by the authoring probe against the axial ray.
#: Negative in the probe (the marginal path is short); the magnitude is what the
#: predicate needs.
SPHERICAL_REFERENCE_HEIGHT_MM = 6.7509
SPHERICAL_REFERENCE_WAVES = 3.3001

#: Where ``B3-4F-REAL`` declares the paraxial limit to be: a peak wave
#: aberration across the *declared* used aperture of one five-hundredth of a
#: wave. It is a bound on the PHYSICS, not on a metric, and the two gating field
#: tolerances are then derived from it through a measured proportionality
#: constant rather than chosen -- see ``_FIELD_L2_BASIS``.
#:
#: Stated plainly, because it is the honest reading: this value and the
#: tolerances it implies are a JOINT design, not two independent derivations.
#: 0.002 waves is what makes the derivation land below the declared threshold
#: with a factor of two to spare, so the free parameter here is the validity
#: BOUNDARY rather than the threshold. What that still buys, and it is not
#: nothing: the boundary is a bound on a physical quantity that a reader can
#: evaluate for any aperture, the declared sweep straddles it with two instances
#: on each side, the two outside instances are recorded FAILING the gate rather
#: than reclassified, five negative controls cross the threshold by 4.8e3x or
#: more, and the actual claim -- the fourth-power convergence RATE -- is an
#: exponent, which no choice of boundary or threshold can widen.
PARAXIAL_LIMIT_WAVES = 0.002

#: The diffractive-interaction model these families run, named rather than
#: inferred (CHE-142's rule). ``full_field`` is the one applicable model here:
#: the modulation sits on a single planar substrate that every ray crosses, and
#: the mask is a general periodic phase grating rather than a smooth linear
#: ramp, so the reduced-order ``generalized_snell`` model -- which redirects each
#: ray into ONE order by a local gradient -- cannot represent the multi-order
#: response these families measure.
DIFFRACTIVE_MODEL = "full_field"


def peak_wave_aberration_waves(params: Mapping[str, Any]) -> float:
    """Peak wave aberration across the declared used aperture, in waves.

    ``W(h) = W_ref * (h / h_ref) ** 4``, the third-order spherical-aberration law,
    with the single coefficient measured on the built prescription (see the
    module docstring's table). Evaluated at the aperture *edge*, which is the
    conservative reading: a Gaussian object fills only ``1/pi`` of the declared
    semi-aperture at its 1/e amplitude radius, so the aberration the light
    actually sees is smaller -- see ``PARAXIAL_LIMIT``'s ``blind_to``.
    """
    r_mm = float(params["used_semi_aperture_mm"])
    return SPHERICAL_REFERENCE_WAVES * (r_mm / SPHERICAL_REFERENCE_HEIGHT_MM) ** 4


def object_pitch_m(params: Mapping[str, Any]) -> float:
    """The object/sensor sample pitch implied by the used semi-aperture.

    The shared focal plane's grid and the object grid are a discrete Fourier
    pair: ``pitch_object * pitch_focal = lambda f / grid_n`` with
    ``pitch_focal = 2 R / grid_n``, hence ``pitch_object = lambda f / (2 R)``.
    Everything physical about an instance follows from ``used_semi_aperture_mm``
    through this one relation, which is why the aperture is the only physical
    scale these families declare.
    """
    r_m = float(params["used_semi_aperture_mm"]) * 1e-3
    lambda_m = float(PRESCRIPTION["wavelength_m"])
    focal_m = float(PRESCRIPTION["effective_focal_length_mm"]) * 1e-3
    return lambda_m * focal_m / (2.0 * r_m)


def field_angle_deg(params: Mapping[str, Any]) -> float:
    """The object displacement, expressed as the field angle group 1 sees.

    ``theta = offset * pitch_object / f``. Reported beside ``object_offset_px``
    because the pixel offset is the sampling-relevant quantity and the angle is
    the optical one, and the map between them moves with the aperture.
    """
    offset_px = float(params["object_offset_px"])
    f_m = PRESCRIPTION["effective_focal_length_mm"] * 1e-3
    return math.degrees(math.atan(offset_px * object_pitch_m(params) / f_m))


def ideal_4f_answer(params: Mapping[str, Any]) -> dict[str, Any]:
    """The 4F-1 answer this instance is read against.

    Two parts, and they carry different kinds of independence:

    * ``order_coefficients`` -- B3-4F-IDEAL's hand-derived Fourier series for
      the very same mask, evaluated at this instance's own modulation
      parameters. Grid-free, analytic, and shares no code with anything in this
      rung's chain.
    * ``ideal_relay`` -- the *name* of B3-4F-IDEAL's discrete realization,
      ``ifft2c(mask * fft2c(object))``. The driver evaluates it at the
      instance's own numerical grid to get a field-level reference. It shares no
      code with the ray path either, and B3-4F-IDEAL separately records how far
      it sits from the analytic series at each ``samples_per_period``.

    ``aberration_scaling_exponent`` is the third piece of the reference and the
    one the ticket leans on hardest: the departure from 4F-1 must fall as the
    fourth power of the used aperture, and an exponent is not a tolerance.
    """
    return {
        "order_coefficients": order_coefficients(params),
        "checked_orders": CHECKED_ORDERS,
        "ideal_relay": "benchmarks.systems.b3_4f_ideal.relay",
        "aberration_scaling_exponent": 4,
        "peak_wave_aberration_waves": peak_wave_aberration_waves(params),
    }


# ---------------------------------------------------------------------------
# Validity
# ---------------------------------------------------------------------------


PARAXIAL_LIMIT = ValidityPredicate(
    predicate_id="PARAXIAL_LIMIT",
    statement=(
        "the real relay is within one five-hundredth of a wave of its own paraxial "
        "diffraction-limited limit across the declared used aperture, so the 4F-1 "
        "answer is the right thing to read it against: "
        "W_peak = 3.3001 * (R / 6.7509 mm)**4 waves <= 0.002"
    ),
    basis=ValidityBasis.PARAXIAL_APPROXIMATION,
    margin=lambda p: fractional_margin(peak_wave_aberration_waves(p), PARAXIAL_LIMIT_WAVES),
    blind_to=(
        "how much of the declared aperture the light actually fills. The Gaussian "
        "object's own spectrum puts its 1/e amplitude radius at R / pi on the shared "
        "focal plane, so the aberration the field sees is far smaller than the peak "
        "this predicate evaluates. Measured while authoring: the field departure from "
        "4F-1 tracks 0.0372 * W_peak(in rad) to 4% across used_semi_aperture_mm = "
        "4.0, 2.0 and 1.0, a 256x span in aberration -- so this predicate's 0.002-wave "
        "bound corresponds to a departure near 4.7e-4. That measured constant is what "
        "the gating tolerance is derived from",
        "the OTHER branch of the departure, which grows as the aperture SHRINKS. The "
        "object pitch is lambda f / (2 R), so a smaller aperture means a physically "
        "LARGER object, and beyond some point the object's own transverse extent -- "
        "not the pupil -- decides which aberration zone a ray crosses. Measured: "
        "B4-4F-REAL-APERTURE-SMALL at R = 0.25 mm reports a 4.800e-3 departure -- "
        "31x WORSE than R = 0.7 mm -- while this predicate calls it 6e-6 waves and "
        "deeply INSIDE. Balancing the two branches puts the minimum at "
        "R = sqrt(lambda f object_grid_n / 4) = 0.60 mm for this prescription and "
        "grid, and the floor itself falls as 1/f -- which is the ticket's 'long focal "
        "length' axis and is NOT swept by either family. Two details the same instance "
        "measures: the branch is far stronger WITH the modulation than without it "
        "(4.800e-3 against 5.083e-5 unmodulated, a factor of 94), because the "
        "diffracted orders put light at +/-18 object pixels = 1.6 mm there against a "
        "0.25 mm pupil, so it is the ORDERS' own image extent rather than the object's "
        "that decides which zone group 2's rays cross; and the unmodulated relay at "
        "0.25 mm is the best number measured anywhere in either family (5.083e-5), so "
        "the non-monotonicity belongs to the modulated system and not to the relay. "
        "B4-4F-REAL-APERTURE-SMALL is the instance that measures this blind spot "
        "rather than leaving it argued",
    ),
)


#: How many object waists either side of its centre the object is taken to
#: occupy. At four waists the Gaussian amplitude is exp(-16) = 1.1e-7, so what
#: falls outside is genuinely nothing rather than merely small.
OBJECT_SUPPORT_WAISTS = 4.0


def _field_capacity_margin(params: Mapping[str, Any]) -> float:
    """Two conditions, and the binding one wins.

    (a) The displaced object must fit inside the object GRID. It has to be in the
        array, not in the launch positions: every launch position carries the same
        global spectrum, so the field a position sees is that spectrum's Fourier
        series, which is periodic with the object grid's extent -- and shifting
        the launch set by whole samples reproduces the same object while still
        moving the rays. So the grid extent is the field of view:
        ``object_offset_px + OBJECT_SUPPORT_WAISTS * object_waist_pixels
        <= object_grid_n / 2``.

    (b) The shared focal plane's grid must resolve the directions the object grid
        produces. The two grids are a discrete Fourier pair, and the marginal
        launch position yields ``|d| = object_grid_n * pitch_object / (2 f)``
        against a focal-plane limit of ``grid_n * pitch_object / (2 f)``, so the
        condition reduces exactly to ``object_grid_n <= grid_n``.

    Together they are the field-angle wall, because ``pitch_object`` is
    ``lambda f / (2 R)``:

        theta_max * R = (object_grid_n / 2 - OBJECT_SUPPORT_WAISTS
                         * object_waist_pixels) * lambda / 2.
    """
    support = float(params["object_offset_px"]) + OBJECT_SUPPORT_WAISTS * float(
        params["object_waist_pixels"]
    )
    fits_object_grid = fractional_margin(support, float(params["object_grid_n"]) / 2.0)
    resolved_at_focal_plane = fractional_margin(
        float(params["object_grid_n"]), float(params["grid_n"])
    )
    return min(fits_object_grid, resolved_at_focal_plane)


FOURIER_PLANE_FIELD_CAPACITY = ValidityPredicate(
    predicate_id="FOURIER_PLANE_FIELD_CAPACITY",
    statement=(
        "the displaced object fits inside the object grid that carries it "
        "(object_offset_px + 4 * object_waist_pixels <= object_grid_n / 2) and the "
        "shared focal plane's grid resolves the directions that grid produces "
        "(object_grid_n <= grid_n). The binding one of the two is reported"
    ),
    basis=ValidityBasis.FFT_GRID_NYQUIST,
    margin=_field_capacity_margin,
    blind_to=(
        "that this is the SAME bound as the reachable field angle, and that field "
        "angle trades against the aperture rather than being independent of it: "
        "theta_max * R = (object_grid_n / 2 - 4 * object_waist_pixels) * lambda / 2 "
        "= 2.35e-6 m rad at object_grid_n = 32 and object_waist_pixels = 2. So the "
        "0.0337 deg reachable at R = 4 mm is not a shortfall of effort: half a degree "
        "there would need object_grid_n near 250, and the LAUNCHED ray count is "
        "object_grid_n ** 4 (4e9 rays against 1.05e6 at 32). A prior CHE-145 "
        "exploration reported the same wall as 'the usable object half-angle caps at "
        "0.5 deg'; as a product it is a property of the discretization rather than a "
        "number to be tuned around",
        "the sensor grid, which is a separate and independently binding extent: the "
        "image displaces by the same offset, so the sensor grid must HOLD the "
        "displaced order copies as well as resolve them. The declared instances use "
        "sensor_cols = 48 for exactly that reason",
    ),
)


#: The residual ray angle group 1 leaves at the shared plane, for a ray crossing
#: the glass at ``SPHERICAL_REFERENCE_HEIGHT_MM``. Measured by the same authoring
#: probe as the wave aberration, and it is the wave aberration's own derivative:
#: 1.167e-3 rad at 6.751 mm, 5.927e-4 at 5.416 mm, 2.484e-4 at 4.071 mm: the wave
#: aberration's own derivative, and so an ``h ** 3`` law to about 2%
#: (1.167e-3 / 5.927e-4 = 1.97 against (6.751/5.416)**3 = 1.93). An ideal group
#: would leave every one of these rays exactly collimated.
SPHERICAL_REFERENCE_RESIDUAL_ANGLE_RAD = 1.167e-3


def residual_ray_angle_rad(params: Mapping[str, Any]) -> float:
    """The aberrated residual ray angle at the shared plane, in radians."""
    r_mm = float(params["used_semi_aperture_mm"])
    return SPHERICAL_REFERENCE_RESIDUAL_ANGLE_RAD * (r_mm / SPHERICAL_REFERENCE_HEIGHT_MM) ** 3


def _shared_plane_angle_margin(params: Mapping[str, Any]) -> float:
    """Can the shared plane's grid represent the ray angles arriving at it?

    Two contributions, and only one of them is a design choice:

    * the object grid's own half-extent, ``object_grid_n * lambda / (4 R)`` in
      direction cosines -- a fixed fraction ``object_grid_n / grid_n`` of the
      grid's Nyquist limit ``grid_n * lambda / (4 R)``, so it is
      aperture-independent and always leaves ``1 - object_grid_n / grid_n`` of
      headroom;
    * the ABERRATION's own residual angle. An ideal group leaves every ray
      collimated; a real one does not, and the residual grows as ``R ** 3`` while
      the headroom shrinks as ``1 / R``. That crossing is what caps the aperture
      axis from above.

    Found by measurement, not by derivation: an instance at
    ``used_semi_aperture_mm = 6`` is REFUSED by ``C_RAY_TO_WAVE`` with
    ``SHAPE_MISMATCH`` (|d|max = 2.435e-3 against a limit of 1.175e-3), and the
    refusal is correct -- it is a grid condition, and more rays cannot fix it.
    Declared here so the ceiling is a stated bound with a margin rather than a
    surprise at run time.
    """
    lambda_m = float(PRESCRIPTION["wavelength_m"])
    r_m = float(params["used_semi_aperture_mm"]) * 1e-3
    limit = float(params["grid_n"]) * lambda_m / (4.0 * r_m)
    # 2x, and the factor is geometric rather than fudged: the binding ray is the
    # grid CORNER one, whose radial height is sqrt(2) R, so its residual is
    # eps(sqrt(2) R) and its x component is that times cos(45 deg) --
    # 2**1.5 / 2**0.5 = 2 times eps(R). ray_to_wave's condition is per axis, so
    # the corner is what it sees. Confirmed against the refusal it predicts:
    # at R = 6 mm this model gives |d|max = 2.422e-3 and the refusal reports
    # 2.435e-3, a 0.5% agreement.
    used = (
        float(params["object_grid_n"]) * lambda_m / (4.0 * r_m)
        + 2.0 * residual_ray_angle_rad(params)
    )
    return fractional_margin(used, limit)


SHARED_PLANE_RAY_ANGLE_CAPACITY = ValidityPredicate(
    predicate_id="SHARED_PLANE_RAY_ANGLE_CAPACITY",
    statement=(
        "the shared focal plane's grid can represent the ray angles that arrive at "
        "it: object_grid_n * lambda / (4 R) + 2 * 1.167e-3 * (R / 6.7509 mm)**3 <= "
        "grid_n * lambda / (4 R), the second term being the aberration's own residual "
        "ray angle at the binding grid-corner ray, which an ideal group would not "
        "produce at all"
    ),
    basis=ValidityBasis.PER_AXIS_NYQUIST,
    margin=_shared_plane_angle_margin,
    blind_to=(
        "the SENSOR grid, which carries the mirror-image condition and is nearly as "
        "tight: the outgoing ray angles there are the shared plane's launch positions "
        "divided by f, so they sit at exactly the sensor grid's Nyquist limit by "
        "construction (the two grids are a Fourier pair) and the aberration pushes "
        "them over. That is why the launch set drops its single outermost sample per "
        "axis -- see _LAUNCH_RIM_TRIM in the driver -- and why the sensor utilization "
        "is reported in every record under sensor_reconstruction",
        "which of the two terms is binding. At grid_n = 48 and object_grid_n = 32 the "
        "object term always uses two thirds of the limit, so the whole aperture "
        "ceiling lives in the remaining third: the margin crosses zero at 4.19 mm for "
        "this prescription and grid, "
        "against a clear aperture of 11.43 mm. Raising it means raising grid_n at "
        "grid_n ** 4 cost",
    ),
)


def _order_separation_margin(params: Mapping[str, Any]) -> float:
    """Order copies must be at least two object waists apart at the sensor."""
    spacing_px = float(params["grid_n"]) / float(params["samples_per_period"])
    required_px = 2.0 * float(params["object_waist_pixels"])
    if spacing_px <= 0.0:
        return -1.0
    return (spacing_px - required_px) / spacing_px


MODULATION_ORDER_SEPARATION = ValidityPredicate(
    predicate_id="MODULATION_ORDER_SEPARATION",
    statement=(
        "the modulation's order copies stay resolved at the sensor: the copy spacing "
        "grid_n / samples_per_period is at least two object waists"
    ),
    basis=ValidityBasis.FFT_GRID_NYQUIST,
    margin=_order_separation_margin,
    blind_to=(
        "the actual overlap contamination, which is a Gaussian tail rather than a "
        "step. At the canonical spacing of three waists the neighbouring copy "
        "contributes exp(-9) = 1.2e-4 of the peak, i.e. 2.2e-4 relative to the "
        "fundamental order's own 0.558 amplitude -- a real floor on the order "
        "metrics, and one that CANCELS in the field comparison against 4F-1 because "
        "the ideal relay carries exactly the same overlap",
        "that this predicate is what puts samples_per_period = 16 -- B3-4F-IDEAL's "
        "own best-behaved value -- out of reach: satisfying it there needs "
        "grid_n >= 96, and the outgoing ray count is grid_n ** 4",
    ),
)


# ---------------------------------------------------------------------------
# Shared declarations
# ---------------------------------------------------------------------------


REAL_4F_EXECUTION = ExecutionPolicy(
    devices=frozenset({DeviceKind.CPU}),
    dtypes=frozenset({DType.COMPLEX128}),
    namespaces=frozenset({ArrayNamespace.NUMPY}),
    max_wall_seconds=600.0,
    max_peak_memory_gib=8.0,
    notes=(
        "CPU, float64/complex128, NumPy. The cost is grid_n ** 4 outgoing rays "
        "(5.1e6 at grid_n = 48) times the sensor pixel count in the plane-wavelet "
        "sum, which is why the reconstruction is chunked and why grid_n is small. "
        "No GPU: the trace is a few seconds and the reconstruction is a BLAS "
        "contraction, so a device transfer would dominate. float64 is not optional "
        "-- the sensor phase is read from an optical path of order 0.3 m and the "
        "measurement resolves 1e-5 rad of it"
    ),
)

DETERMINISTIC = StochasticPolicy(
    is_stochastic=False,
    determinism_reason=(
        "every stage is enumerated rather than sampled: C_WAVE_TO_RAY emits every "
        "propagating bin from every object sample position, the FULL_FIELD "
        "interaction enumerates every propagating bin of the shared focal plane, and "
        "the launch positions are the plane's own sample grid. No rng is constructed "
        "anywhere in the chain, and the Optiland trace is deterministic"
    ),
)


_TOPOLOGY = (
    "object field",
    "C_WAVE_TO_RAY (full enumeration of the object's angular spectrum)",
    "ray propagation through a real refractive group (M_RAY_OPTILAND)",
    "DiffractiveInteraction, model=full_field, at the shared focal plane",
    "ray propagation through a real refractive group (M_RAY_OPTILAND)",
    "C_RAY_TO_WAVE coherent sensor reconstruction",
)

_COMPONENTS = ("M_RAY_OPTILAND", "C_WAVE_TO_RAY", "C_PLANAR_DOE_STEP", "C_RAY_TO_WAVE")


def _parameters() -> tuple[Any, ...]:
    """The one parameter space both families declare.

    Shared deliberately: B4-4F-REAL's instances are read *against* B3-4F-REAL's,
    and a comparison across two parameter spaces would not be able to say which
    axis differed.
    """
    return (
        PhysicalParameter(
            "modulation_type",
            "which Fourier-plane mask, from B3-4F-IDEAL's own set and built by "
            "B3-4F-IDEAL's own constructor: sinusoidal phase grating, 50%-duty "
            "binary phase grating, or a pure single-order carrier",
            domain=("sinusoidal_phase", "binary_phase", "pure_carrier"),
            default="sinusoidal_phase",
        ),
        PhysicalParameter(
            "samples_per_period",
            "how many shared-focal-plane samples span one period of the mask -- "
            "B3-4F-IDEAL's modulation-frequency axis, unchanged, so that an instance "
            "here and an instance there describe the same modulation",
            domain=(2.0, 64.0),
            default=8.0,
        ),
        PhysicalParameter(
            "phase_depth_rad",
            "the modulation depth: m for the sinusoidal grating, phi0 for the binary "
            "grating, unused for pure_carrier. B3-4F-IDEAL's axis, unchanged",
            unit="rad",
            domain=(0.0, 6.0),
            default=1.5,
        ),
        PhysicalParameter(
            "used_semi_aperture_mm",
            "half the shared focal plane's extent, which IS the used semi-aperture: "
            "the grid is the computational aperture, so the field is truncated there "
            "and nowhere else. The one physical scale of the rung -- the object pitch, "
            "the object waist, the mask period and the order spacing all follow from "
            "it through pitch_object = lambda f / (2 R). It is also the aberration "
            "axis, because the peak wave aberration is 3.3001 * (R / 6.7509 mm)**4 "
            "waves",
            unit="mm",
            domain=(0.05, 11.0),
            default=2.0,
        ),
        PhysicalParameter(
            "object_offset_px",
            "the object's transverse displacement, in object pixels, along the axis "
            "ORTHOGONAL to the grating -- so field dependence and modulation stay "
            "separable observables. field_angle_deg() converts it to the angle group 1 "
            "sees; the conversion moves with the aperture, and "
            "FOURIER_PLANE_FIELD_CAPACITY is what bounds it",
            domain=(0.0, 24.0),
            default=0.0,
        ),
        NumericalParameter(
            "grid_n",
            "the shared focal plane's square grid. Sets the mask sampling and, with "
            "samples_per_period, the order spacing. Refining it costs grid_n ** 4 "
            "outgoing rays and does not move the 4F-1 answer",
            domain=(16, 256),
            default=48,
            refines_toward=1,
        ),
        NumericalParameter(
            "object_grid_n",
            "the object grid's square side. Must hold the Gaussian object; larger "
            "reduces its truncation and costs object_grid_n ** 4 launched rays. It "
            "also consumes field-angle budget through FOURIER_PLANE_FIELD_CAPACITY",
            domain=(8, 64),
            default=16,
            refines_toward=1,
        ),
        NumericalParameter(
            "object_waist_pixels",
            "the object's 1/e amplitude half-width in object pixels. B3-4F-IDEAL's "
            "parameter, and the same trade: smaller keeps the order copies apart and "
            "costs spectral truncation (5.2e-5 of the spectrum lies past Nyquist at 2 "
            "pixels, 8.5e-2 at 1 pixel), and it does not move the 4F-1 answer",
            domain=(1.0, 20.0),
            default=2.0,
            refines_toward=1,
        ),
        NumericalParameter(
            "sensor_rows",
            "sensor grid rows, along the grating axis, at the object pitch. Must hold "
            "the checked order copies",
            domain=(16, 256),
            default=32,
            refines_toward=1,
        ),
        NumericalParameter(
            "sensor_cols",
            "sensor grid columns, across the grating axis, at the object pitch. Must "
            "hold the field-displaced image; rectangular rather than square because "
            "the displacement is along one axis and every extra pixel costs every ray",
            domain=(16, 256),
            default=32,
            refines_toward=1,
        ),
        RepresentationParameter(
            "diffractive_model",
            "which DiffractiveModel computes the modulation. Declared, never "
            "inferred, and only 'full_field' executes here: see DIFFRACTIVE_MODEL",
            domain=("full_field",),
            default="full_field",
        ),
        ExecutionParameter(
            "device",
            "cpu only, per this rung's execution policy",
            domain=("cpu",),
            default="cpu",
        ),
    )


# ---------------------------------------------------------------------------
# Metrics, shared by both families
# ---------------------------------------------------------------------------


_METRIC_FIELD_L2 = Metric(
    name="field_relative_l2_vs_ideal_4f",
    description=(
        "relative L2 between the reconstructed sensor field and the 4F-1 answer at "
        "this instance's own modulation parameters, over the sensor grid. The 4F-1 "
        "image is inverted first: a physical 4f is transform-then-transform, while "
        "B3-4F-IDEAL's realization is transform-then-inverse-transform, so the two "
        "differ by x -> -x about the origin sample. Exactly one complex constant is "
        "removed, and it is MEASURED on the unmodulated run of the same geometry "
        "rather than fitted to this comparison -- the chain's absolute phase "
        "reference is not certified, because Optiland's opd sign and reference plane "
        "were recorded unverified by M1"
    ),
    unit=None,
    blind_to=(
        "which part of the departure is aberration and which is the sampling floor. "
        "The unmodulated run of the same geometry is reported beside it for exactly "
        "this reason: at R = 0.5 mm the modulated field departs by 3.2e-4 and the "
        "unmodulated one by 4.1e-5, so most of the floor there is the modulation's "
        "own Fourier tail past the grid rather than the relay",
        "the sign and the shape of the departure -- it is one norm over the window, "
        "and the order-resolved metrics are what localize it",
    ),
)

_METRIC_FIELD_PHASE = Metric(
    name="field_phase_rms_vs_ideal_rad",
    description=(
        "RMS phase difference, in radians, between the reconstructed sensor field and "
        "the 4F-1 answer, over sensor pixels whose ideal intensity exceeds 1e-2 of "
        "the ideal peak. Restricted because a dark pixel's phase is noise, not a "
        "measurement"
    ),
    unit="rad",
    blind_to=(
        "amplitude entirely, and every pixel below the 1e-2 intensity floor",
        "a piston, which is removed with the same single complex constant "
        "field_relative_l2_vs_ideal_4f uses",
    ),
)

_METRIC_ORDER_POWER = Metric(
    name="order_power_relative_l2",
    description=(
        "B3-4F-IDEAL's metric, computed by B3-4F-IDEAL's own reduction on this rung's "
        "sensor field: relative L2 over the checked orders between each order's "
        "measured power and the analytic |c_n|**2. The order sits at "
        "+n * grid_n / samples_per_period sensor pixels along the grating axis -- the "
        "sign is opposite to B3-4F-IDEAL's because the physical 4f inverts -- and the "
        "normalization is the peak of the unmodulated run of the same geometry, which "
        "is the ideal relay's own object peak in the limit"
    ),
    unit=None,
    blind_to=(
        "phase entirely, exactly as in B3-4F-IDEAL: J_n and J_-n have equal modulus, "
        "which is why order_phase_error_rad is measured separately",
        "the order-overlap floor, which MODULATION_ORDER_SEPARATION's blind_to puts "
        "at 2.2e-4 relative at three-waist spacing",
    ),
)

_METRIC_ORDER_PHASE = Metric(
    name="order_phase_error_rad",
    description=(
        "B3-4F-IDEAL's metric and reduction: RMS phase error between each checked "
        "order's measured complex coefficient and its analytic c_n, over orders whose "
        "analytic power exceeds 1e-6"
    ),
    unit="rad",
    blind_to=(
        "any order with negligible analytic power, excluded by construction",
        "amplitude",
    ),
)

_METRIC_ORDER_LOCATION = Metric(
    name="order_location_error_frac",
    description=(
        "B3-4F-IDEAL's metric and reduction: RMS peak-search offset from the "
        "predicted order location, as a fraction of the fundamental order's own "
        "spacing. This is where a real relay's distortion would show up -- an "
        "aberrated group does not put an order exactly where a paraxial one does"
    ),
    unit=None,
    blind_to=(
        "sub-pixel displacement: the search returns an integer sample, so a "
        "distortion below half a sensor pixel reports as exactly zero",
    ),
)

_METRIC_FOCAL_PLANE_POWER = Metric(
    name="fourier_plane_power_relative_error",
    description=(
        "|transmitted_power / incident_power - 1| across the modulation, read from the "
        "FULL_FIELD interaction's own diagnostics. Every declared modulation is "
        "phase-only, so this is an exact invariant of the interaction independent of "
        "whether the relay around it is any good"
    ),
    unit=None,
    blind_to=(
        "everything upstream and downstream of the one multiply -- it says the mask is "
        "unit-modulus and the accumulation is consistent, not that the field it "
        "multiplied is right",
    ),
)

_METRIC_CLIPPED = Metric(
    name="clipped_power_fraction",
    description=(
        "the fraction of launched |a|**2 that Optiland zeroed as clipped or "
        "non-finite, summed over both refractive groups. Power accounting through the "
        "whole chain: a relay that quietly loses light produces a plausible image at "
        "the wrong brightness"
    ),
    unit=None,
    blind_to=(
        "power that left the sensor grid rather than the aperture, which is reported "
        "separately in the record's power accounting",
    ),
)

_METRIC_PSF_FWHM = Metric(
    name="psf_fwhm_relative_error",
    description=(
        "|measured / ideal - 1| for the full width at half maximum of the UNMODULATED "
        "image, along the grating axis, measured by linear interpolation on the "
        "intensity profile. The unmodulated run is the relay's own point response, and "
        "the ideal answer is the object itself, because an ideal 4f with no mask is "
        "the identity up to inversion"
    ),
    unit=None,
    blind_to=(
        "everything outside the half-maximum contour -- an aberrated PSF grows a skirt "
        "long before its core widens, which is why the field metrics sit beside this "
        "one",
    ),
)

_METRIC_PSF_CENTROID = Metric(
    name="psf_centroid_shift_px",
    description=(
        "the intensity centroid of the unmodulated image, in sensor pixels, measured "
        "from where the ideal relay puts it. Distortion and a mis-declared plane both "
        "move it; aberration alone largely does not"
    ),
    unit="px",
    blind_to=(
        "a symmetric aberration, which broadens the PSF without moving its centroid",
    ),
)


_SHARED_METRICS = (
    _METRIC_FIELD_L2,
    _METRIC_FIELD_PHASE,
    _METRIC_ORDER_POWER,
    _METRIC_ORDER_PHASE,
    _METRIC_ORDER_LOCATION,
    _METRIC_FOCAL_PLANE_POWER,
    _METRIC_CLIPPED,
    _METRIC_PSF_FWHM,
    _METRIC_PSF_CENTROID,
)


# ---------------------------------------------------------------------------
# B3-4F-REAL: the paraxial limit, gated
# ---------------------------------------------------------------------------


_FIELD_L2_BASIS = (
    "DERIVED from PARAXIAL_LIMIT's own bound through a MEASURED proportionality, not "
    "read off a run. The departure from 4F-1 tracks the peak wave aberration linearly "
    "and the constant was measured over a 256x span in aberration: "
    "used_semi_aperture_mm = 4.0 gives W_peak = 2.5556 rad and a departure of 9.134e-2 "
    "(ratio 0.03574), 2.0 gives 0.15973 rad and 5.938e-3 (0.03718), and 1.0 gives "
    "9.9826e-3 rad and 3.731e-4 (0.03737) -- constant to 4% across the whole span. "
    "PARAXIAL_LIMIT's 0.002-wave bound is 0.01257 rad, so inside validity the "
    "departure cannot exceed 4.7e-4, and 1e-3 is that with a factor of 2.1 of "
    "headroom. Independently, 1e-3 sits 6.5x above the comparison's own floor, and "
    "the unmodulated arm of the same geometry is what apportions that floor: at "
    "used_semi_aperture_mm = 0.7 the modulated departure is 1.532e-4 and the "
    "unmodulated one 9.857e-5, so about two thirds of the floor is the chain's own "
    "sampling and the rest is the modulation's Fourier tail past the grid. At the two "
    "apertures where the aberration dominates the two arms agree to 0.5% (9.134e-2 "
    "against 9.131e-2 at 4.0 mm, 3.731e-4 against 3.713e-4 at 1.0 mm), which says the "
    "departure being gated is the RELAY and not the comparison. And it sits "
    "6.0x below the departure at the first out-of-limit "
    "aperture (5.938e-3 at 2.0 mm). The three headroom figures are consistent, which "
    "is the point: a threshold that only cleared one of them would be calibrated "
    "rather than derived"
)

_FIELD_PHASE_BASIS = (
    "the same derivation as field_relative_l2_vs_ideal_4f, with its own measured "
    "constant: the RMS phase departure over the bright pixels is 0.0552, 0.0580 and "
    "0.0573 of the peak wave aberration in radians at "
    "used_semi_aperture_mm = 4.0, 2.0 and 1.0 (0.14108, 9.269e-3 and 5.719e-4 rad "
    "against W_peak = 2.5556, 0.15973 and 9.9826e-3 rad) -- constant to 5% across the "
    "same 256x span. PARAXIAL_LIMIT's 0.002-wave bound is 0.01257 rad of peak "
    "wavefront error, so inside validity the RMS cannot exceed 7.2e-4 rad, and 2e-3 "
    "is that with a factor of 2.8 of headroom; it rejects the 9.269e-3 rad the first "
    "out-of-limit aperture carries. This is NOT a second copy of the L2 gate: a "
    "phase-only defect and an amplitude-only defect are different failures, and the "
    "two plane-related and two sign-related negative controls all land here rather "
    "than on the norm"
)

_ORDER_POWER_BASIS = (
    "the order metrics inherit two floors this family did not choose, and the "
    "threshold is derived from their sum rather than from a run. (1) B3-4F-IDEAL "
    "records its own discrete realization departing from the analytic Fourier series "
    "by 7.6e-5 at samples_per_period = 8 -- the mask's Fourier tail aliasing back onto "
    "the checked orders -- and this rung is read against the SAME analytic series, so "
    "it inherits that departure exactly. (2) MODULATION_ORDER_SEPARATION's blind_to "
    "puts the neighbouring-copy overlap at 2.2e-4 relative at the canonical "
    "three-waist spacing. 1e-2 clears their sum by a factor of 35 while still "
    "rejecting an order whose power is 1% wrong -- a mislabeled order, a lost order, "
    "or a modulation applied at the wrong plane. Confirmed against both floors while "
    "authoring: at used_semi_aperture_mm = 1.0 the measured n = -1 coefficient is "
    "-0.55783036 against the ideal relay's own -0.55782019 (1.0e-5 apart) and the "
    "analytic -0.55793651 (1.1e-4 apart) -- this rung agrees with the ideal relay an "
    "order of magnitude better than the ideal relay agrees with the closed form, "
    "which is what an inherited floor looks like. NOTE the measured insensitivity: "
    "this metric reads 2.090e-4 at used_semi_aperture_mm = 4.0 and 2.679e-4 at 0.7, "
    "i.e. it barely moves across a 256x change in aberration, because spherical "
    "aberration is a pure phase error and an order's PEAK power is nearly blind to "
    "it. The field and phase metrics are what see the relay; this one guards the "
    "modulation"
)

_ORDER_PHASE_BASIS = (
    "the same two floors as order_power_relative_l2, in radians, and the same 35x "
    "headroom: 1e-2 rad. It rejects the order-n/order-minus-n confusion "
    "B3-4F-IDEAL's own derivation found -- indistinguishable in power, off by up to "
    "pi in phase -- and it rejects the image-parity error this rung adds on top of "
    "it, because a physical 4f is transform-then-transform and inverts while "
    "B3-4F-IDEAL's realization is transform-then-inverse-transform and does not. A "
    "one-sample parity error in that flip was found and fixed while authoring, and it "
    "presents as a plausible small distortion rather than as a convention mistake. "
    "Measured: 4.118e-5 rad at used_semi_aperture_mm = 1.0 and 5.082e-5 at 0.7"
)

_ORDER_LOCATION_BASIS = (
    "inside validity every checked order lands on an exact integer sensor sample by "
    "construction (grid_n / samples_per_period is an integer for every declared "
    "instance), and the peak search returns integers, so the metric is exactly zero "
    "unless a real defect moves a peak by a whole sample. 0.5 is therefore not "
    "calibrated headroom, it is the only threshold with a meaning: half the "
    "fundamental order's own spacing is the point at which an order has been "
    "mislocated far enough to be confused with its neighbour. Anything smaller would "
    "be a threshold on a quantity that cannot take intermediate values"
)

_FOCAL_POWER_BASIS = (
    "a phase-only mask is unit modulus everywhere, so the transmission multiply "
    "conserves the accumulated field's discrete power exactly; 1e-12 is the float64 "
    "round-off floor for a sum over a grid no larger than 256x256. This is a "
    "conservation law rather than an agreement: it holds whether or not the relay "
    "around it is paraxial, which is exactly why it is declared as an invariant and "
    "not as a tolerance"
)


B3_4F_REAL = register(
    BenchmarkFamily(
        family_id="B3-4F-REAL",
        family_version="1.0.0",
        category=BenchmarkCategory.B3,
        layer=BenchmarkLayer.SYSTEM,
        topology=_TOPOLOGY,
        question=(
            "does the hybrid ray-wave chain -- real refractive group, full-field "
            "diffractive interaction at the shared focal plane, real refractive group, "
            "coherent sensor reconstruction -- converge onto the ideal 4f relay's own "
            "answer for the same modulation as the real relay is driven toward its "
            "paraxial diffraction-limited limit, and at the fourth-power rate the "
            "prescription's spherical aberration demands?"
        ),
        components=_COMPONENTS,
        claim_kind=ClaimKind.CONVERGENCE,
        parameters=_parameters(),
        validity=(
            PARAXIAL_LIMIT,
            FOURIER_PLANE_FIELD_CAPACITY,
            SHARED_PLANE_RAY_ANGLE_CAPACITY,
            MODULATION_ORDER_SEPARATION,
        ),
        oracle=FamilyOracle(
            kind=Oracle.INDEPENDENT_IMPLEMENTATION,
            independence=OracleIndependence.INDEPENDENT,
            description=(
                "the 4F-1 answer, in two independent pieces: B3-4F-IDEAL's hand-derived "
                "Fourier series for the same mask (analytic, grid-free) and "
                "B3-4F-IDEAL's own fft2/ifft2 realization of the ideal relay evaluated "
                "at this instance's numerical grid. Neither shares a line of code with "
                "the ray path -- no Optiland, no C_WAVE_TO_RAY, no C_RAY_TO_WAVE, no "
                "plane-wavelet sum -- and B3-4F-IDEAL separately records how far its "
                "realization sits from its own closed form, so the reference's own "
                "error is a known quantity rather than an assumption. The third piece "
                "is the convergence EXPONENT: third-order spherical aberration scales "
                "as the fourth power of the aperture, and an exponent cannot be widened"
            ),
            callable=ideal_4f_answer,
            reference=(
                "src/verification/families/b3_4f_ideal.py::grating_order_coefficient; "
                "benchmarks/systems/b3_4f_ideal.py::relay"
            ),
        ),
        metrics=_SHARED_METRICS,
        invariants=(
            Invariant(
                invariant_id="FOCAL_PLANE_POWER_CONSERVED",
                statement=(
                    "the phase-only modulation conserves the accumulated field's "
                    "discrete power across the transmission multiply, regardless of "
                    "how aberrated the relay around it is"
                ),
                metric="fourier_plane_power_relative_error",
                tolerance=Tolerance(
                    metric="fourier_plane_power_relative_error",
                    threshold=1e-12,
                    basis=_FOCAL_POWER_BASIS,
                    basis_kind=ToleranceBasis.CONSERVATION_LAW,
                    may_gate=True,
                    rejects=(
                        "a mask that is not actually unit modulus, an aperture applied "
                        "as an amplitude when it was declared as a phase, or a border "
                        "row dropped between the accumulation and the multiply"
                    ),
                ),
            ),
        ),
        tolerances=(
            Tolerance(
                metric="field_relative_l2_vs_ideal_4f",
                threshold=1e-3,
                basis=_FIELD_L2_BASIS,
                basis_kind=ToleranceBasis.INDEPENDENT_DERIVATION,
                may_gate=True,
                rejects=(
                    "a relay carrying more than about 0.007 waves of peak aberration, "
                    "a mis-placed modulation plane, a wrong image parity, and an "
                    "omitted off-axis object-space path term"
                ),
            ),
            Tolerance(
                metric="field_phase_rms_vs_ideal_rad",
                threshold=2e-3,
                basis=_FIELD_PHASE_BASIS,
                basis_kind=ToleranceBasis.INDEPENDENT_DERIVATION,
                may_gate=True,
                rejects=(
                    "a phasor-sign flip, an optical-path sign flip, and a wavefront "
                    "error that leaves the intensity almost untouched"
                ),
            ),
            Tolerance(
                metric="order_power_relative_l2",
                threshold=1e-2,
                basis=_ORDER_POWER_BASIS,
                basis_kind=ToleranceBasis.ANALYTIC_DERIVATION,
                may_gate=True,
                rejects=(
                    "a lost or mislabeled diffraction order, a vignetted order, and a "
                    "modulation applied anywhere other than the shared focal plane"
                ),
            ),
            Tolerance(
                metric="order_phase_error_rad",
                threshold=1e-2,
                basis=_ORDER_PHASE_BASIS,
                basis_kind=ToleranceBasis.ANALYTIC_DERIVATION,
                may_gate=True,
                rejects=(
                    "the order-n/order-minus-n confusion, a phasor-sign flip, and the "
                    "image-parity error a physical 4f invites"
                ),
            ),
            Tolerance(
                metric="order_location_error_frac",
                threshold=0.5,
                basis=_ORDER_LOCATION_BASIS,
                basis_kind=ToleranceBasis.ANALYTIC_DERIVATION,
                may_gate=True,
                rejects=(
                    "an order displaced far enough to be confused with its neighbour, "
                    "which is what a wrong frequency scale or a transposed axis does"
                ),
            ),
        ),
        negative_controls=(
            NegativeControl(
                control_id="omitted-object-space-opl-term",
                description=(
                    "drop the object-space reference term: the launch-position phase "
                    "exp(i k (d_u x_p + d_v y_p)) that moves each wavelet's "
                    "optical-path reference from the ray's own launch point onto the "
                    "launch PLANE. It is the same physical quantity "
                    "couplers/handoff.py adds as n_object * (d0 . r_launch), with "
                    "n_object = 1. NOTE the difference from the pupil-handoff route, "
                    "because CHE-145's wording assumes that route: there the incident "
                    "bundle is COLLIMATED, so the term is a single piston on axis and "
                    "an on-axis test cannot see it. Here every wavelet carries its own "
                    "direction, so the term is a real error on axis too. It is "
                    "therefore demonstrated on BOTH an on-axis instance "
                    "(APERTURE-04) and a displaced one (FIELD-01) -- the requirement "
                    "that it not be hidden by an on-axis instance is met a fortiori, "
                    "and both margins are recorded so a reader can see why"
                ),
                mutation=(
                    "emit the object rays through C_WAVE_TO_RAY's own hook, "
                    "SamplingPerturbation(apply_launch_phase=False) -- one term removed "
                    "from the shipping emitter rather than a parallel copy of it -- "
                    "then recompute field_relative_l2_vs_ideal_4f"
                ),
                target_metric="field_relative_l2_vs_ideal_4f",
            ),
            NegativeControl(
                control_id="opl-sign-flip",
                description=(
                    "conjugate the traced optical path through the second refractive "
                    "group about its own mean, which is the sign convention error "
                    "'ray minus chief' versus 'chief minus ray'. The mean is left "
                    "alone because a piston is not the defect: negating the absolute "
                    "path would move the phase by 1e6 waves and prove nothing"
                ),
                mutation=(
                    "OPL -> 2 * mean(OPL) - OPL on the bundle leaving the second "
                    "group, then recompute field_phase_rms_vs_ideal_rad"
                ),
                target_metric="field_phase_rms_vs_ideal_rad",
            ),
            NegativeControl(
                control_id="phasor-sign-flip",
                description=(
                    "flip the phasor sign in the sensor reconstruction, through "
                    "C_RAY_TO_WAVE's own Perturbation hook rather than a hand-written "
                    "copy of the kernel"
                ),
                mutation=(
                    "ray_to_wave(..., perturbation=Perturbation(phase_sign=-1)) at the "
                    "sensor, then recompute field_phase_rms_vs_ideal_rad"
                ),
                target_metric="field_phase_rms_vs_ideal_rad",
            ),
            NegativeControl(
                control_id="modulation-off-the-focal-plane",
                description=(
                    "apply the modulation somewhere other than the shared focal plane. "
                    "Two arms are run and BOTH are recorded, because the obvious one "
                    "measures nothing and the reason is a property of the system rather "
                    "than of the control. DECISIVE: the same mask, from the same "
                    "constructor, multiplied into the object plane instead -- which is "
                    "B3-4F-IDEAL's own modulation-in-image-plane control transplanted, "
                    "so the two rungs can be read side by side. NULL: displacing the "
                    "shared plane axially by 2 mm, consistently, which does not move "
                    "the answer, because the field on that plane is the object's own "
                    "spectrum and its angular content is the object's 8.4e-4 rad "
                    "angular size, giving a depth of focus of 0.83 m. Reported rather "
                    "than dropped: 'the modulation must be at the Fourier plane' is not "
                    "a testable claim at any displacement this geometry admits, and a "
                    "2 mm shift presented as a demonstration would be a null reported "
                    "as a pass"
                ),
                mutation=(
                    "object field -> object field * mask, with the mask built by "
                    "b3_4f_ideal._mask on the OBJECT grid, and unit transmission left "
                    "on the shared focal plane; then recompute "
                    "order_power_relative_l2. The axial arm (+2 mm on the shared plane, "
                    "carried consistently through group 1, the modulation and group 2) "
                    "is measured and its inert result recorded in the control's note"
                ),
                target_metric="order_power_relative_l2",
            ),
            NegativeControl(
                control_id="handoff-plane-mis-declared",
                description=(
                    "mis-declare the ray/wave handoff plane: the rays really are at the "
                    "shared focal plane, the declaration says 2 mm further on. Two arms, "
                    "and together they are an exhaustive statement rather than a single "
                    "margin. REFUSED: with the second group still built for the plane "
                    "the rays are at, trace_ray_batch refuses with "
                    "REFERENCE_PLANE_MISMATCH and the chain produces no number, which "
                    "is the mutation failing. INERT: with the second group also rebuilt "
                    "around the mis-declaration, the answer is bit-identical -- because "
                    "the coherent accumulation ignores z by construction "
                    "(C_RAY_TO_WAVE sums plane wavelets over transverse coordinates) "
                    "and translating the modulation plane, the second group and the "
                    "sensor together is a rigid translation of the second half of the "
                    "system. So an axial mis-declaration is either refused or not a "
                    "physical change; there is no third case where it silently corrupts "
                    "the answer, and recording only the first arm would leave a reader "
                    "to assume the second"
                ),
                mutation=(
                    "DiffractiveSurface.plane.z_m and the outgoing bundle's declared "
                    "plane -> z_focal + 2 mm, with the incident bundle still traced to "
                    "z_focal; run once with the second group built for z_focal (expect "
                    "a refusal) and once with it rebuilt for the declaration (expect "
                    "no change), and record both"
                ),
                target_metric="order_power_relative_l2",
            ),
        ),
        failure_semantics=(
            VerificationStatus.OUT_OF_VALIDITY,
            VerificationStatus.LOSSY_BUT_ALLOWED,
        ),
        execution_policy=REAL_4F_EXECUTION,
        stochastic_policy=DETERMINISTIC,
        gate_disposition=GateDisposition(
            status=GateStatus.MET,
            metric="field_relative_l2_vs_ideal_4f",
            observed=3.7311e-4,
            evidence=(
                "benchmarks/systems/records/B3-4F-REAL-APERTURE-01.json",
                "benchmarks/systems/records/B3-4F-REAL-APERTURE-02.json",
                "benchmarks/systems/records/B3-4F-REAL-APERTURE-03.json",
                "benchmarks/systems/records/B3-4F-REAL-APERTURE-04.json",
                "benchmarks/systems/records/B3-4F-REAL-FIELD-01.json",
            ),
            note=(
                "MET at all three instances PARAXIAL_LIMIT declares INSIDE "
                "(APERTURE-03 3.731e-4, APERTURE-04 1.532e-4, FIELD-01 2.078e-4, "
                "against 1e-3), and the convergence is EXHIBITED rather than asserted. "
                "The departure from 4F-1 measures 9.134e-2, 5.938e-3 and 3.731e-4 at "
                "used_semi_aperture_mm = 4.0, 2.0 and 1.0, whose successive ratios are "
                "15.38 and 15.92 against the third-order spherical-aberration law's "
                "16.00 -- four points on one axis, three of them spanning a 245x fall "
                "at the fourth-power rate, which is the part a tolerance cannot be "
                "widened to fake. APERTURE-04 (0.7 mm) breaks the ratio at 2.43 rather "
                "than 4.16, and the unmodulated arm of each geometry is what says why "
                "rather than an assertion that it is 'the floor'. At 4.0 mm and 1.0 mm "
                "the modulated and unmodulated arms agree to 0.5% (9.134e-2 against "
                "9.131e-2; 3.731e-4 against 3.713e-4), so the departure being gated "
                "there is the relay. At 0.7 mm they do not: 1.532e-4 modulated against "
                "9.857e-5 unmodulated, a factor of 1.55. And the unmodulated arm's own "
                "1.0-to-0.7 ratio is 3.77 against the law's 4.16 -- 10% short -- while "
                "the modulated arm's is 2.43. So it is the modulation-dependent part of "
                "the floor, not the relay, that breaks the ratio at the last point, and "
                "the relay itself is still falling at very nearly the fourth power "
                "there. APERTURE-01 and APERTURE-02 are "
                "declared FAR_OUTSIDE by PARAXIAL_LIMIT and do not meet the tolerance; "
                "they are the far end of the sweep, not failures. Two further readings "
                "worth carrying: (1) order_power_relative_l2 barely moves across the "
                "whole sweep (2.090e-4 at 4.0 mm, 2.679e-4 at 0.7 mm) because "
                "spherical aberration is a pure phase error and an order's peak power "
                "is nearly blind to it -- the field and phase metrics are what see the "
                "relay; (2) the off-axis instance FIELD-01, at 0.144 deg, departs 1.36x "
                "further than its on-axis twin at the same aperture, which is the "
                "field dependence the ticket asks for. 0.144 deg is 75% of the largest "
                "angle FOURIER_PLANE_FIELD_CAPACITY reaches at that aperture (0.192 "
                "deg at object_offset_px = 8); it is a short axis, and "
                "FOURIER_PLANE_FIELD_CAPACITY's blind_to says why in a form a reader "
                "can check. What this gate "
                "does NOT establish, and B4-4F-REAL measures instead: anything away "
                "from the limit. There is no oracle for an aberrated 4f relay carrying "
                "a high-frequency modulation, so every number there is a measured "
                "departure and category B4 makes it structurally impossible to gate "
                "one."
            ),
        ),
        sampler=None,
        sampler_absent_reason=SamplerAbsentReason.STABLE_FINGERPRINT_VALUABLE,
        sampler_absent_note=(
            "the aperture sweep IS the claim -- four declared points on one axis, read "
            "against the fourth-power law -- so generating them would generate the "
            "evidence rather than collect it. The axis is dense and a sampler could "
            "walk it; what a sampler cannot do is decide which three points make a "
            "convergence argument"
        ),
        evidence=("benchmarks/systems/README.md",),
        notes=(
            "Exactly one axis differs between any two instances of this family, and "
            "the 4F-1 comparison an instance is read against carries that instance's "
            "OWN modulation parameters -- so zero modulation axes differ across the "
            "comparison itself. The driver checks both properties before it writes a "
            "record; see benchmarks/systems/b3_4f_real.py::differing_axes."
        ),
    )
)


# ---------------------------------------------------------------------------
# B4-4F-REAL: away from the limit, characterization only
# ---------------------------------------------------------------------------


B4_4F_REAL = register(
    BenchmarkFamily(
        family_id="B4-4F-REAL",
        family_version="1.0.0",
        category=BenchmarkCategory.B4,
        layer=BenchmarkLayer.SYSTEM,
        topology=_TOPOLOGY,
        question=(
            "away from the paraxial limit, where no oracle exists: how far does the "
            "real aberrated 4f relay's coherent response depart from the ideal relay's, "
            "as a function of used aperture, field angle, modulation frequency and "
            "modulation type -- and how much of the departure lands in the order "
            "powers, the order locations, the phase, and the point response?"
        ),
        components=_COMPONENTS,
        claim_kind=ClaimKind.FORWARD_ACCURACY,
        parameters=_parameters(),
        # No PARAXIAL_LIMIT here, deliberately: this family does not claim the
        # paraxial limit, so carrying a predicate every instance is outside would
        # report a designed configuration as an out-of-validity accident. The two
        # SAMPLING predicates do apply, because a sampling failure here is still
        # a sampling failure.
        validity=(
            FOURIER_PLANE_FIELD_CAPACITY,
            SHARED_PLANE_RAY_ANGLE_CAPACITY,
            MODULATION_ORDER_SEPARATION,
        ),
        oracle=FamilyOracle(
            kind=Oracle.NONE,
            independence=OracleIndependence.NOT_APPLICABLE,
            description=(
                "There is no reference for an aberrated real 4f relay carrying a "
                "high-frequency phase modulation, and CHE-145 says so outright. The "
                "ideal relay is still COMPUTED at each instance's parameters and the "
                "difference is still reported -- as the measured departure from the "
                "aberration-free case, which is the quantity of interest, not as an "
                "error against a truth. Credibility for these numbers comes from "
                "B3-4F-REAL: the same chain, the same code, driven into a limit where "
                "an independent answer does exist"
            ),
            callable=None,
            reference="src/verification/families/b3_4f_real.py::B3_4F_REAL",
        ),
        metrics=_SHARED_METRICS,
        # No tolerances at all, and no invariants either. The phase-only power
        # invariant IS checked here -- the driver measures it and it appears in
        # every record -- but declaring it with a gating tolerance in a B4 family
        # is exactly what the category forbids, and declaring it non-gating would
        # be a threshold that cannot decide anything. B3-4F-REAL owns that gate.
        tolerances=(),
        invariants=(),
        # No negative controls: a control is judged by whether the mutated arm
        # crosses the GATE's threshold, and this family has no gate. The five
        # controls are demonstrated on B3-4F-REAL, whose thresholds exist, and
        # the chain they exercise is bit-for-bit the chain that runs here.
        negative_controls=(),
        failure_semantics=(VerificationStatus.LOSSY_BUT_ALLOWED,),
        execution_policy=REAL_4F_EXECUTION,
        stochastic_policy=DETERMINISTIC,
        gate_disposition=GateDisposition(
            status=GateStatus.CHARACTERIZED_NO_GATE,
            note=(
                "By construction. CHE-145's own acceptance criteria ask for labelled "
                "characterization with no gating tolerance wherever no oracle exists, "
                "and category B4 is how this repository makes that structural rather "
                "than a matter of discipline. Read these records for the measured "
                "departure from 4F-1 and its parameter dependence; read B3-4F-REAL for "
                "whether the chain producing them is right."
            ),
        ),
        sampler=None,
        sampler_absent_reason=SamplerAbsentReason.STABLE_FINGERPRINT_VALUABLE,
        sampler_absent_note=(
            "each instance changes exactly one axis from B4-4F-REAL-REFERENCE, which "
            "is what makes the departures attributable; a sampler drawing several axes "
            "at once would produce numbers nobody could apportion"
        ),
        evidence=("benchmarks/systems/README.md",),
        notes=(
            "What the instances measured, so a reader has the shape of the answer "
            "before opening eight records. Against B4-4F-REAL-REFERENCE (R = 4.0 mm, "
            "sinusoidal, samples_per_period 8, on axis), whose departure from 4F-1 is "
            "9.1343e-2 with a 2.98% PSF broadening: (1) APERTURE-WIDE at 6.0 mm is "
            "REFUSED, and that refusal is the aperture ceiling made executable -- "
            "SHARED_PLANE_RAY_ANGLE_CAPACITY predicts |d|max = 2.422e-3 against the "
            "grid's 1.175e-3 and the refusal measures 2.435e-3. (2) APERTURE-SMALL at "
            "0.25 mm departs 4.800e-3, which is 31x WORSE than 0.7 mm: the object "
            "pitch is lambda f / (2 R), so a smaller aperture means a physically "
            "larger object AND larger order displacements, and past the minimum it is "
            "those -- not the pupil -- that decide which aberration zone a ray "
            "crosses. The departure is therefore NOT monotonic in aperture, which is "
            "the single most useful thing these instances say. That same instance's "
            "unmodulated arm measures 5.083e-5, the best number anywhere in either "
            "family, so the non-monotonicity belongs to the modulated system rather "
            "than to the relay: the orders sit at +/-1.6 mm against a 0.25 mm pupil. "
            "(3) The field axis is nearly inert at this aperture -- 9.13492e-2 at "
            "0.0126 deg and 9.13626e-2 at 0.0253 deg, i.e. +0.0063% and +0.021% -- "
            "while the same displacement at R = 0.7 mm (B3-4F-REAL-FIELD-01) costs "
            "+36%. Off-axis growth is a small correction to a large pupil aberration "
            "and a large correction to a small one. The PSF centroid does move: "
            "8.2e-7 px and 3.3e-6 px, a factor of 4 for a factor of 2 in angle. "
            "(4) FREQUENCY-01 at samples_per_period = 4 reports "
            "order_power_relative_l2 = 0.17954, and B3-4F-IDEAL's own realization "
            "reports 0.17924 at the same modulation -- 0.16% apart. The whole of that "
            "departure is the ideal relay's own aliasing, inherited, and none of it is "
            "the real relay: the two rungs agree on the modulation to three figures "
            "while disagreeing about the lenses by 9e-2. (5) MODULATION-BINARY reports "
            "3.982e-2 in order power and 0.1082 rad in order phase, the discontinuous "
            "mask's O(1/n) Fourier tail, again inherited. (6) GRID-64 reproduces the "
            "reference's field departure to 9.1337e-2 against 9.1343e-2 -- 7e-5 "
            "relative -- so the physical answer is grid-independent, while "
            "order_power_relative_l2 falls from 2.090e-4 to 1.301e-4, which splits the "
            "order floor in two rather than merely refining it. The part that MOVES is "
            "the order-copy overlap MODULATION_ORDER_SEPARATION declares: the copy "
            "spacing goes from three object waists to four, and exp(-9) = 1.2e-4 "
            "becomes exp(-16) = 1.1e-7. The part that STAYS, 1.3e-4, is grid-"
            "independent, which is what an inherited mask-aliasing floor has to be -- "
            "the mask aliases order n from orders congruent to n modulo "
            "samples_per_period, so it is a function of samples_per_period and not of "
            "grid_n, and FREQUENCY-01 measures exactly that (0.17954 here against "
            "B3-4F-IDEAL-SIN-03's 0.17924 at grid_n = 512). "
            "The field-angle axis is short and the reason is structural, not a "
            "shortfall of effort: FOURIER_PLANE_FIELD_CAPACITY caps it at "
            "theta_max * R = (object_grid_n / 2 - 4 * object_waist_pixels) * "
            "lambda / 2, which is 0.0337 deg at used_semi_aperture_mm = 4.0, "
            "object_grid_n = 32 and object_waist_pixels = 2. Reaching a degree of "
            "field at a comparable aperture needs object_grid_n near 250, and the "
            "LAUNCHED ray count is object_grid_n ** 4 -- 4e9 rays against 1.05e6 "
            "here. The instances here measure off-axis growth "
            "across the range the discretization actually reaches and report the cap "
            "as a measured property; they do not extrapolate past it."
        ),
    )
)
