"""B3-DOE-INLINE and B4-DOE-INLINE: a DOE *inside* a refractive train (CHE-148).

M2.12, the third rung of the M2 system ladder, and the only one where the
diffractive interaction's **ray output** is load-bearing::

    ray -> DiffractiveInteraction (model=generalized_snell)
        -> ray -> real refractive optics (M_RAY_OPTILAND)
        -> image plane -> C_RAY_TO_WAVE coherent reconstruction

Two topologies execute, declared as one parameter rather than two families,
because every claim below is the same claim on both:

``grating_then_lens``
    A collimated bundle strikes a linear phase ramp 2 mm in front of a real
    singlet, and the order lands at the singlet's own back focal plane. This is
    the one system in the whole rung with a *conventional* reference: the
    textbook grating-plus-lens result ``x = f tan(arcsin(m lambda / Lambda))``.
``lens_then_grating_then_lens``
    A collimated bundle is focused by one real singlet, the ramp sits 65 mm
    before the intermediate focus -- so the incident bundle is **converging**,
    with 9.66 waves of optical-path sag across it and a per-ray incident
    direction -- and a second real singlet relays the intermediate image at
    -1.914x onto a sensor. This is the topology in which the OPL rebasing and the
    incident direction term are not trivially zero, which is why it exists.

What is genuinely new here, against CHE-145
-------------------------------------------
``B3-4F-REAL`` (CHE-145) already runs ``ray -> FULL_FIELD interaction -> ray ->
real refractive group -> sensor``, so "a DOE between refractive optics" is not
by itself new, and this family does **not** re-do it. What is new is the
``GENERALIZED_SNELL`` route, where the interaction is one ray in and one ray
out, and therefore where:

* the outgoing bundle is a bundle a downstream refractive trace can be handed
  directly, with no intervening reconstruction to hide a convention error;
* the OPL rebasing is a per-ray *additive* convention -- ``opl_out = opl_in +
  phi(x, y) / k0``, with ``phi`` read at the ray's nearest sample of the
  declared surface -- rather than a re-referencing of a field;
* the amplitude semantics are ``|a_out| = |a_in| |t(x, y)|`` per ray, so a
  phase-only surface conserves ray power *exactly* rather than up to a discrete
  field sum;
* and a linear phase ramp makes the whole thing analytically solvable, because
  the model's gradient estimator is exact to round-off on a ramp at any pitch
  (``couplers/generalized_snell.py``: ``angle(t[+1] conj(t[-1]))`` of a
  unit-modulus product).

Consequently the ticket's "convergence in secondary-ray count" observable is
**not applicable to this rung**: ``GENERALIZED_SNELL`` emits no secondary rays
at all. The axis that replaces it is the *incident* ray count, swept by
``B4-DOE-INLINE-RAYS-256``, and the secondary-ray axis is where it already
lives -- ``B3-4F-REAL``'s ``FULL_FIELD`` enumeration.

The oracle, in four independent pieces
--------------------------------------
1. **The grating equation, analytically.** The outgoing transverse wavevector
   must be ``k_t^in + m (2 pi / Lambda) x_hat`` exactly, so the outgoing
   direction cosine is ``d_x^in + m lambda / Lambda`` to round-off. Measured
   3.5e-16 -- this is not an approximation being checked, it is a closed form.
2. **The paraxial order position.** The built system's ``A`` element from the
   DOE plane to the sensor is 9.0e-17, i.e. the sensor *is* the paraxial back
   focal plane, and the ``B`` element there is ``75.60377181 mm`` -- the EFL
   itself, which is why the reference reduces to the textbook
   ``f tan(arcsin(m lambda / Lambda))`` with no ABCD product left in it.
3. **The Strehl of the phase-sampling sawtooth.** Because the interaction reads
   ``phi`` at the ray's *nearest sample*, a ramp of period ``Lambda`` sampled at
   pitch ``p`` puts a sawtooth phase error of half-width ``pi p / Lambda`` on
   the pupil -- and nothing else, since the *gradient* of a ramp is exact. The
   coherent average of a uniform sawtooth is ``sin(a) / a`` with
   ``a = pi p / Lambda``, so the Strehl-like peak ratio against the same system
   with no DOE must be ``(sin(a) / a)**2``. Measured against that closed form
   over ``p / Lambda`` from 0.01 to 0.24 -- a 0.99967-to-0.82430 span in the
   predicted value -- the worst departure is 3.5e-4. The competing Marechal form
   ``exp(-a**2 / 3)`` is indistinguishable below ``p / Lambda = 0.05`` and wrong
   by 3.9e-3 at 0.24, so the sweep also *identifies* which law it is rather than
   merely agreeing with one.
4. **The independently constructed admissible bundle.** The driver rebuilds the
   outgoing bundle from the closed form -- grating equation for the directions,
   ``phi_snap / k0`` for the optical path, ``|a_in| |t_snap|`` for the amplitude
   -- with no line of ``couplers/generalized_snell.py`` in it, traces *that*
   through the same downstream group, and compares. On ``grating_then_lens`` a
   second, even more independent arm is available and is run: a plain
   ``collimated_bundle`` tilted to the analytic order angle, which contains no
   diffractive anything at all. Measured agreement in sensor position, worst over
   the declared set: 1.07e-16 m against the closed form and 1.07e-16 m against
   the tilted bundle.

The exactness limit, which is the strongest control on this topology
-------------------------------------------------------------------
CHE-148 asks for the zero-phase and zero-gradient limits to reproduce the plain
refractive system "exactly (to round-off)". On ``grating_then_lens`` with
``phi == 0`` the answer is stronger than round-off: **every array is bitwise
identical** -- outgoing directions, positions, amplitudes and optical paths all
differ by exactly ``0.0``, the downstream trace therefore reproduces the plain
trace bitwise, and the Strehl-like ratio is ``1.0000000000000000``. With
``phi == 1 rad`` (zero *gradient*, non-zero phase) the geometry is still exact
to 1.1e-18 in direction cosine and the optical path reproduces the plain path
plus the declared piston to 4.4e-12 waves. On the relay topology, where the
incident bundle carries real positions and paths, the same limits close to
9.3e-18 m in position and 8.8e-10 waves in optical path, and the Strehl-like
ratio is 0.99999999999840.

That 1e-9-wave figure is the float64 round-off floor of the composition, not a
physics departure, and the arithmetic says so: the sensor-side optical path is
of order 0.5 m, ``eps * 0.5 m`` is 1.1e-16 m, and 1.1e-16 m is 1.9e-10 waves at
0.5876 um. The declared threshold, 1e-8 waves, is ten times that floor.

The two sampling walls, and one place a shipping predicate is blind
------------------------------------------------------------------
``RAMP_GRADIENT_ESTIMATOR_UNALIASED`` is declared analytically -- ``4 p /
Lambda <= 1`` -- rather than read from the model's own margin, and the reason is
a measured blind spot in the model's margin that this family found and records:

    p / Lambda   measured margin   analytic 1 - 4 p / Lambda   d_x measured / predicted
    0.20             +0.2000                 +0.2000                  1 - 5.6e-16
    0.24             +0.0400                 +0.0400                  1 - 1.1e-16
    0.30             +0.2000                 -0.2000                 -0.667  (WRONG SIGN)
    0.40             +0.6000                 -0.6000                 -0.250  (WRONG SIGN)

The estimator's raw step across the 5-tap stencil is ``4 pi p / Lambda``, which
wraps into ``(-pi, pi]`` above ``p / Lambda = 0.25``; the model measures the
*wrapped* step, so its margin comes back POSITIVE and increasing exactly where
the gradient has aliased and the order is being emitted on the wrong side of the
axis. The third GENERALIZED_SNELL predicate does catch it --
``single_order_dominance`` falls from 0.813 to 0.0002 -- so the model is not
unguarded, but the guard that fires is not the one whose name suggests it.
``B4-DOE-INLINE-PITCH-ALIASED`` is the instance that measures this rather than
leaving it argued.

The second wall is ``SENSOR_GRID_DIRECTION_CAPACITY``: the reconstruction window
must resolve ``R / f + |m lambda / Lambda| + sin(theta_in)`` in direction
cosine, against ``lambda / (2 * window_pitch)``.

What this rung found in the shipping code
----------------------------------------
``B3-DOE-INLINE-ORDER-MINUS1`` -- the same ramp, order ``m = -1`` -- reported a
``strehl_quantization_relative_error = 0.999994`` (a Strehl-like ratio of 6e-6
against a predicted 0.99967) and ``psf_fwhm_relative_error = 0.4672``, on a
system whose ray-side order position was correct to 9.3e-5. The
cause was an order/OPL inconsistency in ``couplers/generalized_snell.py``
(CHE-143): the momentum equation carried ``m grad(phi)`` while the optical path
carried ``phi`` rather than ``m phi``, so for ``|m| != 1`` the rays were
deflected as if the phase were one thing and given the optical path of another.
Two code-independent arguments identify the correct form, and both are now
measured as exact identities:

* ``exp(i (-1) phi)`` and ``exp(i (+1) (-phi))`` are the same complex factor, so
  ``(order=-1, phi)`` and ``(order=+1, conj(t))`` must return the same bundle.
  They already agreed in direction; with ``phi`` alone they returned *opposite*
  optical paths. With ``m phi`` they agree bitwise -- 0.0 in both direction and
  optical path.
* ``order=0`` is the undiffracted transmission and picks up no ramp at all. With
  ``phi`` alone it was given the whole ramp phase on an undeflected ray; with
  ``m phi`` its optical path is the incident one exactly.

``float(1) * phase`` is IEEE-exact, so the fix is bitwise the previous behaviour
at ``order=1`` -- every shipped test and every record written before CHE-148 --
and changes only ``|m| != 1``. After it, ORDER-MINUS1 reproduces PERIOD-100's
scalar observables to round-off -- identically where the quantity is a ratio
(``strehl_quantization_relative_error`` 3.243739e-06 and
``psf_fwhm_relative_error`` 1.293916e-05 in both) and to a part in 500 where it
is itself a round-off residual (``order_position_vs_admissible_residual_m``
5.3505e-17 against 5.3614e-17) -- which is the mirror symmetry the geometry has.

Two caveats, because the argument is stated as an identity. The ``exp(i m phi)``
equality survives wrapping exactly, since ``m wrap(phi)`` and ``m phi`` differ by
``2 pi`` times an integer for integer ``m``. The ``(order=-1, t)`` /
``(order=+1, conj(t))`` equality is *bitwise* only away from the branch cut, where
``angle(conj(t)) == -angle(t)`` fails: at ``phi = pi`` both give ``+pi``. That is
a whole-wave offset on one sample and physically inert, but the identity is not
universal, and the test that pins it uses inputs away from the cut.

A second, unrelated observation about the same model, recorded because a reader
of these records will meet it: the model's own ``single_order_dominance``
diagnostic reads 0.8129 for a well-sampled ramp at ``m = +1``, **0.1483 at
m = -1 of the same ramp**, 0.4085 with a 0.5 deg incident tilt, and 1.68e-4 once
the gradient has aliased. The 0.1483 is not a defect: a blazed ramp genuinely
has almost no ``-1`` order, and the diagnostic is saying so. What it exposes is
that ``GENERALIZED_SNELL`` hands the outgoing ray its *full* amplitude
regardless -- so ``interaction_power_ratio_error = 0`` is a conservation law of
the model, not of the physics, and this family says that in the metric's own
``blind_to`` rather than letting a zero read as an efficiency claim. The 0.4085
off-axis figure is a target-direction artifact (the diagnostic compares the
surface's local spectrum against a direction that already contains the incident
tilt) and is reported, not acted on: it is ``C_GENERALIZED_SNELL``'s diagnostic
and B1-GSL-VALIDITY's subject, not this rung's.

What could not be established, with numbers
-------------------------------------------
* **A pupil-side wavefront metric.** The obvious observable for the phasor
  sign -- the reconstructed field's phase on a plane 2 mm before focus, read
  against an analytic sphere converging on the analytic order position -- was
  built and rejected. It has a floor of 0.00971 in ``1 - |<exp(i delta)>|``
  (0.140 rad RMS) that is present *with the DOE removed entirely*, and it does
  not resolve the sawtooth at all: the analytic prediction at
  ``p / Lambda = 0.1`` is 0.01637 and the measured value moves by 1.7e-4. The
  reason is structural rather than fixable: ``C_RAY_TO_WAVE`` sums every ray
  into every pixel, so a per-ray sawtooth appears as a reduction in ``|u|`` --
  which is precisely the Strehl the family does gate -- and not as a phase map.
  The metric is not carried, and the phasor-sign control targets
  ``admissible_bundle_field_residual`` instead.
* **A nonparaxial regime, in the strict sense.** ``B4-DOE-INLINE-APERTURE-500``
  reaches ``NA = 0.066`` with 0.99 waves of peak spherical aberration, which is
  46x the diffraction limit and produces the interference structure the ticket
  asks for -- but it is an *aberrated* regime, not a high-NA one. The ceiling is
  the prescription: an 11.43 mm semi clear aperture at ``f = 75.6 mm`` is
  ``NA = 0.15`` even fully filled, and the DOE grid at ``p = 2 um`` is already
  ``5252 x 5252`` complex128 (441 MB) at ``R = 5 mm``. Stated rather than
  implied, because "nonparaxial" in the ticket and ``NA = 0.066`` in the record
  are not the same claim.
* **A PSF window that holds every declared response.** The window is sized on
  the *diffraction* width and centred on the *analytic* order position, both by
  declaration, and two B4 instances fall outside what that can measure:
  ``APERTURE-500``'s unmodulated core is 2.4x the Airy width (10.98 um against
  4.57 um) because it carries 0.99 waves of spherical aberration, so its window
  half-extent of 6.86 um does not hold it -- and its Strehl-like ratio comes out
  at **1.156**, above one, which is the tell rather than a result. ``PITCH-
  ALIASED``'s order lands 740 um from a window of half-extent 34 um, so its
  peak, FWHM and Strehl are computed on a patch of field that contains no order
  at all (they read 0.7377, 7.9e-4 and 1.1e-3, all meaningless). Both are
  detected and recorded rather than argued: every record carries
  ``psf_window_holds_the_order`` and
  ``psf_window_holds_the_point_response``, and they are ``False`` exactly
  there. The fix would be a window sized on the *measured* response, which
  would make the readout grid a function of the run.
* **The ray-density diagnostic.** ``C_RAY_TO_WAVE`` reports
  ``ray_density_status = 'not_computed_above_scan_limit'`` and
  ``max_adjacent_ray_phase_rad = None`` for every instance here, because the
  scan is capped below 12644 rays. So the adjacent-ray phase condition is
  *undeclared*, not satisfied, and no predicate in this family pretends
  otherwise.

What these families do not attempt
----------------------------------
No optimization, no gradients, no vector diffraction, and no multi-order
simultaneous emission: ``GENERALIZED_SNELL`` declares one order per call and
that is what runs, so a real DOE's order-dependent efficiency is not
represented anywhere here -- see ``interaction_power_ratio_error``'s
``blind_to``, which measures the model's own dominance diagnostic at 0.1483 for
the ``m = -1`` order of a blazed ramp while the model still hands that order the
ray's full amplitude.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from fractions import Fraction
from typing import Any

from core.precision import ArrayNamespace, DeviceKind, DType

# The real refractive component, imported by identity from CHE-145's rung rather
# than re-entered, so "the same singlet as B3-4F-REAL" is true by construction
# rather than by inspection. Its EFL and back focal distance are read back from
# the built Optiland system there; this family's own hand-derived paraxial ABCD
# reproduces both to 2.3e-11 relative, which tests/test_b3_doe_inline.py asserts
# rather than assumes.
from verification.families.b3_4f_real import PRESCRIPTION
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
    "B3_DOE_INLINE",
    "B4_DOE_INLINE",
    "DIFFRACTIVE_MODEL",
    "PARAXIAL_ORDER_POSITION_LIMIT",
    "PRESCRIPTION",
    "TOPOLOGIES",
    "airy_fwhm_m",
    "analytic_order_direction",
    "clear_aperture_margin",
    "grating_lens_answer",
    "paraxial_image_side_na",
    "paraxial_order_position_departure",
    "psf_window_pitch_m",
    "sawtooth_residue_count",
    "strehl_quantization",
]


#: The diffractive-interaction model these families run, named rather than
#: inferred (CHE-142's rule). ``generalized_snell`` is the *only* applicable
#: model on this topology and the reason is the topology itself: a
#: representation-changing model returns a reconstructed field or a spectral
#: ray population, and this rung's whole question is whether a bundle handed
#: straight to a refractive trace carries the right conventions. ``full_field``
#: on a linear ramp would also throw away the thing being measured -- it emits
#: every propagating bin, so the single-order response the analytic reference
#: describes would arrive mixed with its neighbours.
DIFFRACTIVE_MODEL = "generalized_snell"

#: The two declared topologies, and the axial layout each one means, in mm from
#: the DOE plane (``grating_then_lens``) or from the input plane
#: (``lens_then_grating_then_lens``). Read by both the family and the driver so
#: a record cannot describe a geometry other than the one that ran.
TOPOLOGIES: dict[str, dict[str, Any]] = {
    "grating_then_lens": {
        "statement": (
            "collimated bundle at the DOE plane -> linear phase ramp -> 2 mm gap -> "
            "the singlet -> its own paraxial back focal plane"
        ),
        "doe_to_first_vertex_mm": 2.0,
        # The departure of the traced order position from
        # f tan(arcsin(m lambda / Lambda)), anchored at R = 1 mm. Third order in
        # the used semi-aperture, i.e. R**2 in a CENTROID. Measured while
        # authoring over R = 0.25, 0.5, 1.0, 2.0 and 3.0 mm: 2.979e-6, 2.093e-5,
        # 9.278e-5, 3.806e-4 and 8.621e-4, successive ratios 7.03, 4.43, 4.10 and
        # 2.27 against the law's 4.00 -- so the law holds to about 10% over
        # 0.5-2 mm and the R**2 form OVER-predicts (i.e. is conservative) at both
        # ends. The three of those points that are declared instances record
        # 2.0915e-5, 9.2709e-5 and 3.8035e-4, which the law predicts to 11%, 0.1%
        # and 2.4%. The departure is attributed, not assumed: the
        # tilted-collimated arm, which contains no DOE, departs by the SAME
        # fraction, with a double ratio of 0.0 to 4.9e-15 at every declared
        # instance where the gradient estimator is unaliased, so all of it is the
        # singlet and none of it is the interaction.
        "departure_coefficient": 9.278e-5,
        "departure_exponent": 2.0,
    },
    "lens_then_grating_then_lens": {
        "statement": (
            "collimated bundle -> singlet 1 -> converging bundle at the DOE plane, "
            "65 mm before the intermediate focus -> linear phase ramp -> 178.406 mm -> "
            "singlet 2 -> the conjugate image plane, magnification -1.9139"
        ),
        "doe_before_intermediate_focus_mm": 65.0,
        "second_group_object_distance_over_efl": 1.5,
        # Measured while authoring on the same axis: 1.366e-3 at R = 0.5 mm and
        # 2.804e-3 at R = 1.0 mm, ratio 2.05 -- LINEAR in the aperture rather than
        # quadratic. Only R = 1.0 mm is a declared instance and it records
        # 2.8023e-3, which the law predicts to 0.06%. The exponent differs because
        # the order is relayed at -1.914x through a second real group and traverses
        # it off axis. So the paraxial order position is not a usable
        # oracle on this topology at any aperture this geometry admits, which is
        # what PARAXIAL_ORDER_POSITION says by declaring it far outside rather
        # than by a footnote.
        "departure_coefficient": 2.804e-3,
        "departure_exponent": 1.0,
    },
}

#: Where the paraxial order-position oracle is declared to hold: a departure of
#: one part in 1e4. Chosen so the declared aperture sweep straddles it --
#: R = 0.5 mm inside (margin 0.768), R = 1.0 mm inside (margin 0.0722), R = 2.0 mm
#: FAR_OUTSIDE -- with the gating threshold three times above it, and so that the
#: relay topology is unambiguously outside rather than marginal. The authoring
#: sweep also ran R = 0.25 and 3.0 mm; neither is a declared instance.
PARAXIAL_ORDER_POSITION_LIMIT = 1e-4

#: Airy full width at half maximum for a circular pupil, as a multiple of
#: ``lambda / (2 NA)``. Not a fitted constant: it is the first solution of
#: ``(2 J_1(v) / v)**2 = 1/2``, ``v = 1.61633``, giving
#: ``FWHM = 2 v lambda / (2 pi NA) / 2``. Reported beside the measured width in
#: every record and NOT gated -- see ``_PSF_FWHM_BASIS`` for why the gated form
#: is the ratio to the same system's own unmodulated response instead.
AIRY_FWHM_OVER_LAMBDA_2NA = 1.028987


def airy_fwhm_m(numerical_aperture: float, wavelength_m: float) -> float:
    """The diffraction-limited FWHM of a circular pupil, in metres."""
    return AIRY_FWHM_OVER_LAMBDA_2NA * wavelength_m / (2.0 * float(numerical_aperture))


#: How many diffraction FWHM either side of the analytic order position the
#: reconstruction window spans. 1.5 holds the core and the first ring's inner
#: flank, which is what a peak and a FWHM need; anything integral would need much
#: more, which is why no power integral is measured on this window.
PSF_WINDOW_HALF_EXTENT_FWHMS = 1.5


def paraxial_image_side_na(params: Mapping[str, Any]) -> float:
    """The image-side numerical aperture, paraxially, from declared parameters only.

    Declared rather than measured because it is what sizes the reconstruction
    window, and a window whose size came out of the run would put the validity
    predicate and the driver on two different numbers.

    ``grating_then_lens`` is ``R / f``: the bundle fills the aperture and the
    sensor is the back focal plane. On the relay the bundle converges to the
    intermediate focus at ``BFD`` past group 1, so at the second group -- a
    further ``u`` on -- its semi-radius is ``R u / BFD``, and the image-side cone
    is that over the second group's own image distance ``v``. The measured
    ``|d|max`` of the traced unmodulated arm is recorded beside this in every
    record; the two agree to 3.5% on the relay, which is the paraxial/real
    difference and is irrelevant to a readout grid.
    """
    r_mm = float(params["used_semi_aperture_mm"])
    f_mm = float(PRESCRIPTION["effective_focal_length_mm"])
    if str(params["system_topology"]) == "grating_then_lens":
        return r_mm / f_mm
    declared = TOPOLOGIES["lens_then_grating_then_lens"]
    bfd_mm = float(PRESCRIPTION["back_focal_distance_mm"])
    object_mm = float(declared["second_group_object_distance_over_efl"]) * f_mm
    # The conjugate image distance for an object `object_mm` in front of a thin
    # equivalent of the same singlet: 1/v = 1/f - 1/u.
    image_mm = 1.0 / (1.0 / f_mm - 1.0 / object_mm)
    return (r_mm * object_mm / bfd_mm) / image_mm


def psf_window_pitch_m(params: Mapping[str, Any]) -> float:
    """The reconstruction window's sample pitch, derived from the declaration.

    ``2 * 1.5 * FWHM_Airy / (psf_window_px - 1)``, so the window always spans
    +/- 1.5 diffraction widths whatever the aperture and the topology do. Derived
    rather than declared as a length because the declared apertures span a factor
    of 10 in ``NA`` and therefore a factor of 10 in the width being resolved: one
    fixed pitch would either alias the narrowest response or truncate the widest.
    """
    fwhm = airy_fwhm_m(paraxial_image_side_na(params), float(PRESCRIPTION["wavelength_m"]))
    return 2.0 * PSF_WINDOW_HALF_EXTENT_FWHMS * fwhm / (int(params["psf_window_px"]) - 1)


def analytic_order_direction(params: Mapping[str, Any]) -> tuple[float, float, float]:
    """The outgoing direction cosines the grating equation demands.

    ``k_t^out = k_t^in + m grad_t(phi)``, divided by ``n_t k0``, with
    ``n_i = n_t = 1`` and ``grad_x(phi) = 2 pi / Lambda`` for the declared ramp.
    Closed form, no grid: this is the reference the outgoing bundle is read
    against, and it is exact rather than paraxial.
    """
    lambda_m = float(PRESCRIPTION["wavelength_m"])
    tilt_rad = math.radians(float(params["incident_tilt_deg"]))
    d_y = math.sin(tilt_rad)
    if str(params["doe_phase_kind"]) != "linear_ramp":
        d_x = 0.0
    else:
        period_m = float(params["grating_period_um"]) * 1e-6
        d_x = float(params["order"]) * lambda_m / period_m
    d_z = math.sqrt(max(0.0, 1.0 - d_x**2 - d_y**2))
    return (d_x, d_y, d_z)


def strehl_quantization(params: Mapping[str, Any]) -> float:
    """``(sin a / a)**2`` with ``a = pi p / Lambda`` -- the sawtooth Strehl.

    Derived, not fitted. The interaction reads the surface phase at the ray's
    *nearest sample*, so a ray at ``x`` is given ``phi(p round(x / p))`` instead
    of ``phi(x)``. For a ramp that is a pure phase error ``2 pi (x_snap - x) /
    Lambda``, uniform on ``+/- pi p / Lambda`` for ray positions incommensurate
    with the grid, and the ray *directions* are untouched because the gradient of
    a ramp is exact at any pitch. The coherent average of a uniform phase error
    of half-width ``a`` is ``sin(a) / a``, so the peak intensity relative to the
    same system with no DOE is its square. A flat surface has ``a = 0`` and
    predicts exactly 1.
    """
    if str(params["doe_phase_kind"]) != "linear_ramp":
        return 1.0
    a = math.pi * float(params["doe_pitch_um"]) / float(params["grating_period_um"])
    if a == 0.0:
        return 1.0
    return (math.sin(a) / a) ** 2


#: How many distinct sawtooth residues the Strehl law needs before its discrete
#: form is close enough to the continuous one to be gated. The correction for
#: ``N`` equidistributed residues is ``sin(a) / (N sin(a / N))`` in amplitude,
#: i.e. about ``a**2 / (3 N**2)`` in intensity; at the widest declared
#: ``a = pi * 0.20 = 0.628`` that is 5e-4 -- a tenth of the declared threshold --
#: at ``N = 20``. The declared instances reach 127, 128 or -- for
#: ``B4-DOE-INLINE-RAYS-256``, where the spacing happens to be 200/51 of the
#: pitch -- 51, all of which put the correction below 4e-5.
MINIMUM_SAWTOOTH_RESIDUES = 20.0


def sawtooth_residue_count(params: Mapping[str, Any]) -> float:
    """How many distinct values ``x_snap - x`` takes over the launch grid.

    The Strehl law's premise is that the nearest-sample phase error is a
    *uniform* sawtooth, and that is a property of the launch grid against the
    surface pitch rather than something a run can be trusted to supply. The
    launch positions are ``linspace(-R, R, N)``, so the residues are
    ``j * q mod 1`` with ``q = 2 R / ((N - 1) p)``, and the number of distinct
    ones is the denominator of ``q`` in lowest terms, capped at ``N``.

    It matters, and the failure is a false NEGATIVE rather than a false positive.
    Every declared collimated instance lands on 127 or 128, except
    ``B4-DOE-INLINE-RAYS-256`` which lands on 51 -- the count is NOT monotone in
    the ray count, because it is a denominator rather than a size. But a
    commensurate choice inside the declared domain -- ``rays_per_axis = 101`` at
    ``R = 1 mm`` and ``p = 20 um``, giving a ray spacing of exactly 20 um -- makes
    every ray snap with the SAME offset, so the sawtooth vanishes, the measured
    Strehl goes to 1, and the gate fails against a prediction of 0.875 with the
    code entirely correct. ``rays_per_axis = 201`` halves the spacing and gives two
    residues, which is no better. This
    predicate is what turns that into a declared out-of-validity configuration
    instead of a mystery, and ``tests/test_b3_doe_inline.py`` runs the
    counterexample.
    """
    if str(params["doe_phase_kind"]) != "linear_ramp":
        return math.inf
    per_axis = int(params["rays_per_axis"])
    if per_axis < 2:
        return 1.0
    if str(params["system_topology"]) != "grating_then_lens":
        # On the relay the positions at the DOE are a linspace grid imaged
        # through a REAL singlet: a scale factor plus a nonlinear aberration
        # term, so no two spacings are exactly equal and commensurability with
        # the surface pitch cannot arise. Reported as the full count with the
        # reason stated rather than computed from R and N, which would be the
        # wrong grid entirely.
        return float(per_axis)
    spacing_um = 2.0 * float(params["used_semi_aperture_mm"]) * 1e3 / (per_axis - 1)
    ratio = Fraction(spacing_um / float(params["doe_pitch_um"])).limit_denominator(10**9)
    return float(min(ratio.denominator, per_axis))


def paraxial_order_position_departure(params: Mapping[str, Any]) -> float:
    """The declared law for how far the traced order sits from the paraxial one.

    ``c_topology * (R / 1 mm) ** e_topology``, both constants measured on the
    built prescription and recorded in :data:`TOPOLOGIES` with the sweep they
    came from. This is a bound on the PHYSICS -- how far a real singlet's order
    position is from the textbook formula -- and the gating tolerance is derived
    from it rather than from a run.
    """
    topology = TOPOLOGIES[str(params["system_topology"])]
    r_mm = float(params["used_semi_aperture_mm"])
    exponent = float(topology["departure_exponent"])
    return float(topology["departure_coefficient"]) * float(r_mm**exponent)


def clear_aperture_margin(params: Mapping[str, Any]) -> float:
    """Does the diffracted beam still fit through the singlet's clear aperture?

    ``R + L |sin theta_out| <= CA / 2``, with ``L`` the DOE-to-vertex distance on
    ``grating_then_lens`` and the DOE-to-second-group distance on the relay. A
    vignetted order loses power silently on the ray side -- Optiland zeroes the
    intensity and the reconstruction simply has fewer wavelets -- so this is
    declared rather than left to ``downstream_clipped_power_fraction`` to
    discover after the fact.
    """
    topology = TOPOLOGIES[str(params["system_topology"])]
    lever_mm = float(
        topology.get("doe_to_first_vertex_mm")
        or topology["doe_before_intermediate_focus_mm"]
        + float(topology["second_group_object_distance_over_efl"])
        * float(PRESCRIPTION["effective_focal_length_mm"])
    )
    d_x, d_y, _ = analytic_order_direction(params)
    used_mm = float(params["used_semi_aperture_mm"]) + lever_mm * math.hypot(d_x, d_y)
    return fractional_margin(used_mm, float(PRESCRIPTION["clear_aperture_mm"]) / 2.0)


def grating_lens_answer(params: Mapping[str, Any]) -> dict[str, Any]:
    """The four independent references this instance is read against.

    Every entry is either a closed form evaluated here, grid-free, or the name
    of a construction the driver performs without touching
    ``couplers/generalized_snell.py``.
    """
    lambda_m = float(PRESCRIPTION["wavelength_m"])
    f_mm = float(PRESCRIPTION["effective_focal_length_mm"])
    d_x, d_y, d_z = analytic_order_direction(params)
    return {
        "grating_equation_direction": [d_x, d_y, d_z],
        # The paraxial order position, and it is the textbook form rather than an
        # ABCD product: the built system's A element from the DOE plane to the
        # sensor is 9.0e-17 and its B element is the EFL itself, so
        # x = B tan(theta) collapses to f tan(theta). Only meaningful on
        # grating_then_lens; the relay carries its own B and its own magnification
        # and the driver computes both from the same hand ABCD.
        "paraxial_order_position_mm": (
            [f_mm * d_x / d_z, f_mm * d_y / d_z]
            if str(params["system_topology"]) == "grating_then_lens"
            else None
        ),
        "strehl_quantization": strehl_quantization(params),
        "airy_fwhm_over_lambda_2na": AIRY_FWHM_OVER_LAMBDA_2NA,
        "admissible_bundle": (
            "benchmarks.systems.b3_doe_inline.closed_form_outgoing -- the outgoing "
            "bundle rebuilt from the grating equation, phi_snap / k0 and "
            "|a_in| |t_snap|, sharing no code with couplers/generalized_snell.py"
        ),
        "tilt_equivalent_bundle": (
            "couplers.ray_to_wave.collimated_bundle at the analytic order angle, "
            "traced through the same singlet with no diffractive element at all "
            "(grating_then_lens only)"
        ),
        "departure_law": {
            "coefficient": TOPOLOGIES[str(params["system_topology"])][
                "departure_coefficient"
            ],
            "exponent": TOPOLOGIES[str(params["system_topology"])]["departure_exponent"],
        },
        "wavelength_m": lambda_m,
    }


# ---------------------------------------------------------------------------
# Validity
# ---------------------------------------------------------------------------


def _unaliased_margin(params: Mapping[str, Any]) -> float:
    if str(params["doe_phase_kind"]) != "linear_ramp":
        return math.inf
    ratio = float(params["doe_pitch_um"]) / float(params["grating_period_um"])
    return fractional_margin(4.0 * ratio, 1.0)


RAMP_GRADIENT_ESTIMATOR_UNALIASED = ValidityPredicate(
    predicate_id="RAMP_GRADIENT_ESTIMATOR_UNALIASED",
    statement=(
        "the 5-tap gradient estimator's raw phase step across the declared ramp stays "
        "below the wrap boundary: the step is 4 pi p / Lambda, so 4 p / Lambda <= 1. "
        "Stated analytically from the declared pitch and period, NOT read from the "
        "model's own measured margin -- see blind_to"
    ),
    basis=ValidityBasis.GENERALIZED_SNELL_GRADIENT_SMOOTHNESS,
    margin=_unaliased_margin,
    blind_to=(
        "nothing about the ramp -- for a linear ramp this predicate is exact, because "
        "the curvature sub-check of the model's own predicate is identically zero and "
        "the raw step is the whole condition. What it is NOT blind to, and the model's "
        "own margin IS, is the wrap: measured while authoring, at p / Lambda = 0.30 the "
        "model reports worst_local_gradient_smoothness_margin = +0.2000 -- POSITIVE, "
        "and larger still (+0.6000) at 0.40 -- while the emitted direction cosine is "
        "-0.019587 against the correct +0.029380, i.e. the order comes out on the wrong "
        "side of the axis. The estimator measures angle(t[+1] conj(t[-1])), which wraps "
        "into (-pi, pi], so above p / Lambda = 0.25 the measured step SHRINKS as the "
        "aliasing worsens. This predicate is the unwrapped condition and has no such "
        "branch. The model's third predicate does catch the case -- "
        "single_order_dominance falls from 0.813 to 0.0002 -- and "
        "B4-DOE-INLINE-PITCH-ALIASED is the instance that records all of it",
        "a surface that is not a ramp. Every declared instance here is a linear ramp or "
        "a constant, which is what makes the analytic form available; a general DOE "
        "needs the model's own curvature term and this predicate would understate the "
        "requirement",
    ),
)


def _propagating_order_margin(params: Mapping[str, Any]) -> float:
    d_x, d_y, _ = analytic_order_direction(params)
    return fractional_margin(math.hypot(d_x, d_y), 1.0)


PROPAGATING_ORDER_EXISTS = ValidityPredicate(
    predicate_id="PROPAGATING_ORDER_EXISTS",
    statement=(
        "the requested order propagates: |k_t^out| < n_t k0, i.e. "
        "|sin(theta_in) y_hat + m lambda / Lambda x_hat| < 1 for n_t = 1"
    ),
    basis=ValidityBasis.GENERALIZED_SNELL_PROPAGATING_ORDER,
    margin=_propagating_order_margin,
    blind_to=(
        "how much of the outgoing cone the downstream refractive train can actually "
        "accept. An order can propagate and still miss the lens entirely, which is what "
        "ORDER_WITHIN_CLEAR_APERTURE bounds separately -- at Lambda = 50 um the order "
        "leaves at 0.674 deg and clears the 11.43 mm semi-aperture by a factor of 11, "
        "but a 2 um period would leave at 17 deg and land 24 mm off axis, outside the "
        "glass",
    ),
)


SAWTOOTH_EQUIDISTRIBUTION = ValidityPredicate(
    predicate_id="SAWTOOTH_EQUIDISTRIBUTION",
    statement=(
        "the nearest-sample phase error the Strehl law predicts is actually a sawtooth: "
        "the launch grid must produce at least 20 distinct residues of the ray position "
        "modulo the surface pitch. The count is the denominator of "
        "2 R / ((N - 1) p) in lowest terms, capped at N -- see "
        "sawtooth_residue_count(), which computes it from the declared parameters and "
        "not from the run"
    ),
    basis=ValidityBasis.PER_AXIS_NYQUIST,
    margin=lambda p: fractional_margin(
        MINIMUM_SAWTOOTH_RESIDUES, sawtooth_residue_count(p)
    ),
    blind_to=(
        "how EVENLY the residues are spread, only how many there are. A count is a "
        "necessary condition and the discrete correction sin(a) / (N sin(a / N)) assumes "
        "the ideal spread; a grid producing 20 residues clustered in a tenth of a period "
        "would satisfy this and not the law. Every declared instance reaches at least 51 "
        "on a linspace grid, where the spread is exact by construction, so the "
        "distinction has no consequence here and the predicate is the cheap necessary "
        "form rather than the full one",
        "the aperture's own contribution to the same average. The sawtooth is averaged "
        "over the PUPIL, and the pupil is a disc while this count is taken along one "
        "axis; the two agree because the ramp varies along x only, so every y row sees "
        "the same residue set. A two-dimensional surface phase would need the joint "
        "count and this predicate would understate it",
    ),
)


PARAXIAL_ORDER_POSITION = ValidityPredicate(
    predicate_id="PARAXIAL_ORDER_POSITION",
    statement=(
        "the real train's order position is within one part in 1e4 of the paraxial "
        "closed form the oracle is derived under: c * (R / 1 mm)**e <= 1e-4, with "
        "(c, e) = (9.278e-5, 2) on grating_then_lens and (2.804e-3, 1) on "
        "lens_then_grating_then_lens, both measured on the built prescription"
    ),
    basis=ValidityBasis.PARAXIAL_APPROXIMATION,
    margin=lambda p: fractional_margin(
        paraxial_order_position_departure(p), PARAXIAL_ORDER_POSITION_LIMIT
    ),
    blind_to=(
        "that it bounds the LENS and not the interaction, which is the whole reason the "
        "sweep is worth running. Measured while authoring: the tilted-collimated arm -- "
        "a plain collimated_bundle at the analytic order angle, with no diffractive "
        "element anywhere in it -- departs from the paraxial position by the SAME "
        "fraction -- double ratio 0.0 to 4.9e-15 -- at every declared instance where "
        "the gradient estimator is unaliased. So this "
        "predicate is a statement about a real singlet's third-order behaviour, the "
        "interaction contributes nothing measurable to it, and "
        "order_position_vs_admissible_residual_m is the metric that says so, at 1.07e-16 m "
        "worst over the declared set",
        "the relay topology in any useful way. Its measured departure is LINEAR in the "
        "aperture (1.366e-3 at 0.5 mm and 2.804e-3 at 1.0 mm, both measured while "
        "authoring; the declared instance at 1.0 mm records 2.8023e-3) rather than "
        "quadratic, "
        "because the order is relayed at -1.914x through a second real group and "
        "crosses it off axis -- so it is far outside this bound at every aperture the "
        "geometry admits, and B3-DOE-INLINE-RELAY-01 is recorded FAILING "
        "order_position_relative_error rather than being reclassified. Its convention "
        "and admissibility metrics are unaffected and are the reason it is here",
        "the Strehl comparison's own normalization. The peak ratio is taken against the "
        "same geometry with NO DOE, which is an ON-AXIS response, so the order arm also "
        "carries whatever off-axis aberration the displacement costs. Measured: 3.5e-4 "
        "worst on grating_then_lens where the order sits 0.89 mm off axis in a 75.6 mm "
        "system, and 1.5142e-3 on the relay where it sits 0.73 mm off axis behind a "
        "-1.914x group. That difference is what SIZES "
        "strehl_quantization_relative_error's threshold, and _STREHL_BASIS states the "
        "14.4x collimated headroom it costs rather than hiding it",
    ),
)


def _sensor_direction_margin(params: Mapping[str, Any]) -> float:
    lambda_m = float(PRESCRIPTION["wavelength_m"])
    d_x, d_y, _ = analytic_order_direction(params)
    # Conservative on purpose: the image-side cone and the order's own chief
    # direction are added rather than combined, and on the relay the order's
    # direction is demagnified in reality. A Nyquist bound may overstate.
    used = paraxial_image_side_na(params) + math.hypot(d_x, d_y)
    limit = lambda_m / (2.0 * psf_window_pitch_m(params))
    return fractional_margin(used, limit)


SENSOR_GRID_DIRECTION_CAPACITY = ValidityPredicate(
    predicate_id="SENSOR_GRID_DIRECTION_CAPACITY",
    statement=(
        "the reconstruction window resolves the ray directions arriving at it: "
        "NA_image + |m lambda / Lambda| + sin(theta_in) <= lambda / (2 window_pitch), "
        "with both the image-side NA and the window pitch derived from the declared "
        "parameters (paraxial_image_side_na, psf_window_pitch_m) so that the predicate "
        "and the driver read one number"
    ),
    basis=ValidityBasis.PER_AXIS_NYQUIST,
    margin=_sensor_direction_margin,
    blind_to=(
        "the ray DENSITY condition, which C_RAY_TO_WAVE declines to evaluate at these "
        "ray counts: every instance here reports ray_density_status = "
        "'not_computed_above_scan_limit' and max_adjacent_ray_phase_rad = None, because "
        "the scan is capped below 12644 rays. The adjacent-ray phase condition is "
        "therefore UNDECLARED here rather than satisfied, and the only executable "
        "evidence that the ray population is dense enough is the incident-ray-count "
        "axis B4-DOE-INLINE-RAYS-256 sweeps (7080 -> 12644 -> 28600 -> 51040 rays moves "
        "sweeps: 12644 -> 51040 rays moves the Strehl departure from 6.687e-6 to "
        "9.660e-6, the coherent/geometric L2 from 0.11557 to 0.11405 (1.3% relative) and "
        "the order position from 9.2709e-5 to 9.2831e-5 (1.3e-4 relative), so the "
        "population is dense enough that nothing physical moves -- which is evidence "
        "about this configuration and not a substitute for the condition C_RAY_TO_WAVE "
        "declined to evaluate)",
        "the window EXTENT, which is a separate and looser condition: the window is "
        "sized at 1.5 diffraction FWHM about the analytic order position, so it holds "
        "the core and truncates the skirt. That is deliberate for a peak and a FWHM and "
        "wrong for anything integral, which is why no power-integral metric is measured "
        "on it",
    ),
)


ORDER_WITHIN_CLEAR_APERTURE = ValidityPredicate(
    predicate_id="ORDER_WITHIN_CLEAR_APERTURE",
    statement=(
        "the diffracted beam still passes the glass: R + L |sin theta_out| <= CA / 2 "
        "= 11.43 mm, with L the DOE-to-glass lever of the declared topology"
    ),
    basis=ValidityBasis.DECLARED_PLANARITY,
    margin=clear_aperture_margin,
    blind_to=(
        "vignetting anywhere other than the first surface after the DOE, and the real "
        "aperture of the second group on the relay topology. It is a necessary "
        "condition, not a sufficient one, and downstream_clipped_power_fraction is the "
        "measured companion -- 0.0 exactly on every declared instance",
    ),
)


# ---------------------------------------------------------------------------
# Shared declarations
# ---------------------------------------------------------------------------


DOE_INLINE_EXECUTION = ExecutionPolicy(
    devices=frozenset({DeviceKind.CPU}),
    dtypes=frozenset({DType.COMPLEX128}),
    namespaces=frozenset({ArrayNamespace.NUMPY}),
    max_wall_seconds=600.0,
    max_peak_memory_gib=8.0,
    notes=(
        "CPU, float64/complex128, NumPy. The cost is dominated by the DOE grid, which "
        "is complex128 and must span the ray footprint: grid_n = 2 ceil(1.05 R / p), so "
        "2102x2102 (70 MB) at R = 1 mm and p = 1 um and 5252x5252 (441 MB) at R = 5 mm "
        "and p = 2 um. Six traces per instance -- the order arm, the unmodulated arm, "
        "the closed-form admissible arm, the two flat-phase exactness arms and (on "
        "grating_then_lens) the tilted-collimated arm -- at 7e3 to 5e4 rays each, which "
        "is seconds. No GPU: nothing here is large enough for a transfer to pay, and "
        "float64 is not optional because the exactness limits are measured at 1e-18 in "
        "direction cosine and 1e-12 waves in optical path"
    ),
)

DETERMINISTIC = StochasticPolicy(
    is_stochastic=False,
    determinism_reason=(
        "GENERALIZED_SNELL samples nothing: it is one ray in, one ray out, with the "
        "phase and its gradient read from a fixed stencil at the ray's own nearest "
        "sample. No rng is constructed anywhere in the chain, the launch positions are "
        "a declared Cartesian grid masked to a circle, the Optiland trace is "
        "deterministic, and C_RAY_TO_WAVE is a deterministic sum. Two runs of the same "
        "instance are bitwise identical"
    ),
)


_TOPOLOGY = (
    "collimated (or tilted) coherent ray bundle, or a converging bundle from a real "
    "refractive group",
    "DiffractiveInteraction, model=generalized_snell, at a planar linear-ramp surface",
    "outgoing ray bundle, handed DIRECTLY to a refractive trace",
    "ray propagation through real refractive optics (M_RAY_OPTILAND)",
    "C_RAY_TO_WAVE coherent reconstruction at the image plane",
)

_COMPONENTS = ("M_RAY_OPTILAND", "C_GENERALIZED_SNELL", "C_RAY_TO_WAVE")


def _parameters() -> tuple[Any, ...]:
    """The one parameter space both families declare.

    Shared deliberately, for CHE-145's reason: B4-DOE-INLINE's instances are read
    against B3-DOE-INLINE's, and a comparison across two parameter spaces could
    not say which axis differed.
    """
    return (
        RepresentationParameter(
            "system_topology",
            "which of the two declared axial layouts runs. See TOPOLOGIES for the "
            "geometry each one means; the difference that matters is whether the "
            "bundle incident on the DOE is collimated (so its optical path is a "
            "constant and its direction is one vector) or converging out of a real "
            "refractive group (so both are per-ray)",
            domain=tuple(TOPOLOGIES),
            default="grating_then_lens",
        ),
        PhysicalParameter(
            "doe_phase_kind",
            "the surface's phase: a linear ramp, a constant zero, or a constant "
            "1 radian. The two constants are the zero-phase and zero-gradient limits "
            "CHE-148 requires, and they are a declared parameter rather than a special "
            "case in the driver so that an exactness instance is an instance",
            domain=("linear_ramp", "flat_zero", "flat_piston"),
            default="linear_ramp",
        ),
        PhysicalParameter(
            "grating_period_um",
            "the ramp period. With the pitch it sets the sawtooth Strehl "
            "(sin a / a)**2, a = pi p / Lambda, and with the order it sets the outgoing "
            "angle lambda / Lambda. Ignored when doe_phase_kind is flat, and declared "
            "anyway so a flat instance differs from the ramp in exactly one axis",
            unit="um",
            domain=(4.0, 2000.0),
            default=100.0,
        ),
        PhysicalParameter(
            "order",
            "the diffraction order m, declared and never inferred (CHE-143's rule). "
            "One order per call: GENERALIZED_SNELL emits one, which is why "
            "multi-order simultaneous emission is a stated non-goal",
            domain=(-3.0, 3.0),
            default=1.0,
        ),
        PhysicalParameter(
            "used_semi_aperture_mm",
            "half the launched bundle's extent, masked to a circle. The aberration "
            "axis: the peak wave aberration is 3.3001 (R / 6.7509 mm)**4 waves on this "
            "prescription (B3-4F-REAL's measured law), and the paraxial order-position "
            "departure follows c (R / 1 mm)**e -- see TOPOLOGIES",
            unit="mm",
            domain=(0.1, 11.0),
            default=1.0,
        ),
        PhysicalParameter(
            "incident_tilt_deg",
            "the incident bundle's tilt about the x axis, i.e. ORTHOGONAL to the "
            "grating vector, so the incident direction term and the diffraction stay "
            "separable observables. It is also the only axis on which the incident "
            "optical path is non-constant on grating_then_lens, which is what makes "
            "the OPL double-count control detectable there at all",
            unit="deg",
            domain=(0.0, 2.0),
            default=0.0,
        ),
        NumericalParameter(
            "doe_pitch_um",
            "the surface's sample pitch. The one numerical parameter that MOVES the "
            "answer, and analytically: it is the sawtooth half-width pi p / Lambda in "
            "the Strehl law, so refining it drives the Strehl to 1. It also sets the "
            "grid, at grid_n = 2 ceil(1.05 R / p) complex128 samples per axis",
            unit="um",
            domain=(0.25, 40.0),
            default=1.0,
            refines_toward=-1,
        ),
        NumericalParameter(
            "rays_per_axis",
            "the launch grid's side before the circular mask, so the ray count is "
            "about pi/4 of its square. Should not move the answer: measured 1e-5 in "
            "the Strehl and 7e-5 in the coherent/geometric departure from 7080 to "
            "51040 rays",
            domain=(48, 512),
            default=128,
            refines_toward=1,
        ),
        NumericalParameter(
            "psf_window_px",
            "the square reconstruction window used for the peak, the FWHM and the field "
            "comparison. Its PITCH is not a separate parameter: it is derived as "
            "2 * 1.5 * FWHM_Airy / (psf_window_px - 1) by psf_window_pitch_m(), so the "
            "window always spans +/- 1.5 diffraction widths whatever the aperture does "
            "-- the declared apertures span a factor of 10 in NA and one fixed pitch "
            "would alias the narrowest response or truncate the widest. Odd, so the "
            "analytic order position lands on a sample and the parabolic peak fit is "
            "centred",
            domain=(17, 129),
            default=65,
            refines_toward=1,
        ),
        NumericalParameter(
            "structure_window_px",
            "the square window on which the coherent field is compared to the blurred "
            "geometric ray density, at structure_window_pitch_um. Larger than the PSF "
            "window on purpose: the interference structure the B4 family measures lives "
            "in the skirt, not in the core",
            domain=(64, 512),
            default=256,
            refines_toward=1,
        ),
        NumericalParameter(
            "structure_window_pitch_um",
            "the structure window's sample pitch. Fixed across the B4 aperture sweep "
            "rather than sized per instance, so that an aperture comparison is one axis; "
            "the cost is that the structure is sampled at 22.9 samples per diffraction "
            "FWHM at R = 1 mm and only 4.6 at R = 5 mm, which is a stated resolution "
            "limit on the widest instance and not a hidden one",
            unit="um",
            domain=(0.2, 20.0),
            default=1.0,
            refines_toward=-1,
        ),
        RepresentationParameter(
            "diffractive_model",
            "which DiffractiveModel computes the interaction. Declared, never "
            "inferred, and only 'generalized_snell' executes here: see "
            "DIFFRACTIVE_MODEL",
            domain=("generalized_snell",),
            default="generalized_snell",
        ),
        NumericalParameter(
            "patch_px",
            "GeneralizedSnellParameters.patch_px: the transverse scale the smoothness "
            "predicate is checked against and the window single_order_dominance "
            "transforms. Odd. It does NOT enter the physics on a ramp -- the gradient "
            "is exact at any pitch -- so it moves only the two diagnostics, which is "
            "why it is declared rather than hidden",
            domain=(5, 257),
            default=65,
            refines_toward=1,
        ),
        ExecutionParameter(
            "device",
            "cpu only, per this rung's execution policy",
            domain=("cpu",),
            default="cpu",
        ),
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


_METRIC_GRATING_EQUATION = Metric(
    name="grating_equation_direction_residual",
    description=(
        "max over rays of the largest component of |d_out - d_analytic|, where "
        "d_analytic is the grating equation's own answer: k_t^in + m (2 pi / Lambda) "
        "x_hat, normalized by n_t k0. Dimensionless direction cosine, absolute rather "
        "than relative, so a flat-phase instance is measurable on the same scale"
    ),
    unit=None,
    blind_to=(
        "where the ray IS -- it reads only the direction. A surface whose phase gradient "
        "were evaluated at the wrong transverse position would still satisfy this for a "
        "linear ramp, because a ramp's gradient is position-independent. That is exactly "
        "why outgoing_opl_rebase_residual_waves is measured beside it: the ramp's PHASE "
        "is position-dependent, so the optical path is what pins the position at which "
        "the surface was read",
        "the order's power. GENERALIZED_SNELL puts all of a ray's amplitude into the one "
        "declared order by construction, so 'the right direction' and 'the right "
        "fraction of the light' are not the same question here, and single_order_"
        "dominance -- reported, 0.813 for a ramp at patch_px = 65 -- is the one that "
        "asks the second",
    ),
)

_METRIC_OPL_REBASE = Metric(
    name="outgoing_opl_rebase_residual_waves",
    description=(
        "max over rays of |(opl_out - opl_in) - phi_snap / k0| / lambda, with phi_snap "
        "the declared surface's phase at the ray's nearest sample, computed in the "
        "driver from the surface array rather than read back from the interaction. This "
        "is CHE-148's 'OPL rebasing asserted rather than assumed', literally: the "
        "convention opl_out = opl_in + phi / k0 is reproduced from the declaration"
    ),
    unit=None,
    blind_to=(
        "the sign convention DOWNSTREAM of the interaction -- whether the driver then "
        "composes the traced optical path with opl_out once, twice or not at all. That "
        "is the defect the opl-not-rebased control injects and "
        "admissible_bundle_field_residual is where it lands",
        "an OPL that is right modulo 2 pi and wrong in absolute terms. It is measured as "
        "a length in waves without wrapping, so a whole-wave error is visible -- which "
        "is the point, since a plane-wavelet sum uses the absolute path",
    ),
)

_METRIC_AMPLITUDE = Metric(
    name="outgoing_amplitude_residual",
    description=(
        "max over rays of ||a_out| - |a_in| |t_snap||, with |t_snap| the declared "
        "surface's modulus at the ray's nearest sample. CHE-148's 'amplitude semantics "
        "asserted rather than assumed': the model carries the modulus of the "
        "transmission and nothing else -- no obliquity factor, no order-splitting "
        "weight, no energy renormalization"
    ),
    unit=None,
    blind_to=(
        "the PHASE of the transmission, which by design does not reach the amplitude at "
        "all on this model -- it goes into the optical path. A surface whose phase and "
        "modulus were swapped would satisfy this and fail outgoing_opl_rebase_residual_"
        "waves",
        "whether carrying only |t| is the right physics for a real etched DOE, which it "
        "is not in general: a real surface has an order-dependent efficiency this model "
        "does not represent. That is a declared limitation of GENERALIZED_SNELL, not a "
        "measurement this family can make",
    ),
)

_METRIC_ADMISSIBLE_POSITION = Metric(
    name="order_position_vs_admissible_residual_m",
    description=(
        "max over rays of |position_sensor(interaction arm) - position_sensor(closed-form "
        "arm)| in metres, where the closed-form arm is the outgoing bundle rebuilt in the "
        "driver from the grating equation and phi_snap / k0 and traced through the SAME "
        "downstream group. This is CHE-148's admissibility criterion in its strongest "
        "form: the interaction's outgoing bundle is not merely accepted by a refractive "
        "trace, it is the bundle an independent construction would have handed it"
    ),
    unit="m",
    blind_to=(
        "the downstream trace itself, which both arms share. This metric certifies the "
        "INTERACTION's ray output, not Optiland: if the trace were wrong, both arms "
        "would be wrong together and this would still read round-off. order_position_"
        "relative_error against the paraxial closed form is what looks at the trace",
        "the optical path, which does not move a ray's landing point. "
        "admissible_bundle_field_residual is the same comparison with the path included",
    ),
)

_METRIC_ADMISSIBLE_FIELD = Metric(
    name="admissible_bundle_field_residual",
    description=(
        "relative L2 between the reconstructed sensor field of the interaction arm and "
        "of the closed-form admissible arm, over the PSF window. The complete form of "
        "the admissibility assertion -- positions, directions, amplitudes AND optical "
        "paths -- and the metric every optical-path and phasor negative control is "
        "judged on, because it is the only declared metric that reads the absolute "
        "phase of the reconstruction"
    ),
    unit=None,
    blind_to=(
        "a GLOBAL phasor-convention error in C_RAY_TO_WAVE, which would flip both arms "
        "together and cancel. That is C_RAY_TO_WAVE's own qualification (B1/B2), not "
        "this system's, and the control this family runs is the system-level question "
        "instead: does the driver compose the optical path with the sign the "
        "reconstruction expects. Stated rather than left for a reader to notice, because "
        "a control that mutates one arm of a two-arm comparison is easy to over-read",
        "any absolute correctness. Both arms are reconstructed by the same summer, so "
        "this is an identity check on the bundle, not an accuracy claim about the field. "
        "The accuracy claims are strehl_quantization_relative_error and "
        "order_position_relative_error",
        "the JOINT sign of the surface-phase convention and the optical-path-to-phasor "
        "mapping, and this is the one gap in the rung worth stating in one place rather "
        "than assembling from three. The direction metrics and the optical-path metric "
        "are locked to each other by that one convention, this metric compares two arms "
        "that share it, and the only field-side ACCURACY gate -- "
        "strehl_quantization_relative_error -- reads |u| and is provably blind to a "
        "phasor flip (measured: bit-identical under it). So nothing in this rung "
        "independently validates the absolute phase sign of the ray/wave handoff. The "
        "observable that would is the pupil-side wavefront metric the module docstring "
        "records as rejected for having a floor that does not resolve the sawtooth",
    ),
)

_METRIC_ORDER_POSITION = Metric(
    name="order_position_relative_error",
    description=(
        "|x_centroid - x_analytic| / max(|x_analytic|, lambda), with x_analytic the "
        "paraxial closed form -- f tan(arcsin(m lambda / Lambda)) on grating_then_lens, "
        "B tan(theta) from the hand ABCD on the relay -- and x_centroid the "
        "amplitude-weighted mean of the traced ray positions at the image plane. The "
        "lambda floor in the denominator is what lets a flat-phase instance, whose "
        "analytic position is exactly zero, be measured on the same metric"
    ),
    unit=None,
    blind_to=(
        "which of the two systems is responsible. On grating_then_lens the answer is "
        "measured rather than argued: the tilted-collimated arm, with no diffractive "
        "element in it, departs by the same fraction with a double ratio of 0.0 to "
        "4.9e-15, so all of the 9.271e-5 "
        "at R = 1 mm is the singlet's third-order behaviour",
        "the spot's shape. It is a centroid, and at R = 3 mm the centroid offset (0.83 "
        "um) is a quarter of the RMS spot radius (3.80 um) -- so beyond the paraxial "
        "bound this metric is reading the aberrated spot's own asymmetry, which is why "
        "PARAXIAL_ORDER_POSITION exists and why psf_fwhm_relative_error sits beside it",
    ),
)

_METRIC_ORDER_POSITION_FIELD = Metric(
    name="order_position_field_relative_error",
    description=(
        "the same comparison against the same analytic position, but measured on the "
        "WAVE side: the parabolically refined peak of the reconstructed intensity on the "
        "PSF window. The ray centroid and the field peak are two different estimators of "
        "one physical quantity and they are reported separately on purpose"
    ),
    unit=None,
    blind_to=(
        "an asymmetric skirt, which moves a centroid and not a peak -- the reason both "
        "are declared. Measured at R = 1 mm, Lambda = 200 um: the ray centroid sits "
        "21.1 nm from the analytic position and the field peak 41.7 nm, a factor of 2 "
        "that is a real property of the aberrated response and not a disagreement",
        "sub-sample bias in the parabolic fit, which is second order in the offset and "
        "is bounded by the window pitch rather than measured",
    ),
)

_METRIC_STREHL = Metric(
    name="strehl_quantization_relative_error",
    description=(
        "|S_measured / (sin a / a)**2 - 1| with a = pi p / Lambda, where S_measured is "
        "the parabolically refined peak intensity of the order arm divided by that of "
        "the same geometry with NO diffractive surface. The prediction is the coherent "
        "average of the nearest-sample phase sawtooth the interaction's OPL rebasing "
        "produces, so this metric is a closed-form test of that rebasing's physical "
        "consequence rather than of its arithmetic"
    ),
    unit=None,
    blind_to=(
        "the off-axis aberration the order arm acquires and the on-axis normalization "
        "arm does not. Measured: 3.5e-4 worst over p / Lambda in [0.01, 0.24] on "
        "grating_then_lens, where the order sits at most 0.89 mm off axis in a 75.6 mm "
        "system, against 1.5142e-3 on the relay, where it sits 0.73 mm off axis behind "
        "a -1.914x second group. The threshold accommodates the second, and the basis "
        "states the 14.4x collimated headroom that costs",
        "the shape of the sawtooth's effect beyond the peak. A uniform pupil phase error "
        "also puts light in a skirt, which this metric never looks at; the B4 family's "
        "coherent/geometric comparison is what does",
    ),
)

_METRIC_PSF_FWHM = Metric(
    name="psf_fwhm_relative_error",
    description=(
        "|FWHM(order arm) / FWHM(unmodulated arm) - 1| along the grating axis, both "
        "measured by linear interpolation on the same reconstruction window. The claim "
        "is that a linear phase ramp is a pure TILT, so to the extent the downstream "
        "train is aplanatic over the order displacement it must not change the point "
        "response's width at all"
    ),
    unit=None,
    blind_to=(
        "the absolute width, which is the more obvious thing to check and the weaker "
        "one. The Airy comparison IS computed and reported in every record: the "
        "unmodulated FWHM is 22.8815 um against 1.028987 lambda / (2 NA) = 22.8564 um "
        "at R = 1 mm (1.1e-3), and 43.7121 um against 44.59 um on the relay (2.0e-2), "
        "where the image-side pupil is neither a clean circle nor a declared quantity. "
        "One of those two numbers would be a gate and the other a fudge, so neither is",
        "everything outside the half-maximum contour. An aberrated response grows a "
        "skirt long before its core widens, which is what the B4 family measures",
    ),
)

_METRIC_REFRACTIVE_LIMIT = Metric(
    name="refractive_limit_residual_waves",
    description=(
        "the exactness limit, as one number: the worst, over the zero-phase and "
        "zero-gradient arms and over all rays, of |delta position| / lambda and "
        "|delta optical path - declared piston| / lambda between the flat-surface arm "
        "and the arm with no diffractive surface at all. Measured on EVERY instance, "
        "including the ramp ones, because it is a property of the interaction and not of "
        "the instance's own surface"
    ),
    unit=None,
    blind_to=(
        "nothing on grating_then_lens with phi == 0, where it is not a tolerance at all: "
        "every array is bitwise identical and the number is exactly 0.0. Where it is a "
        "tolerance is the relay, whose incident bundle has real positions and paths, and "
        "there it reads 8.8e-10 waves against a float64 composition floor of 1.9e-10",
        "a defect that cancels between the two arms. Both are traced by the same code "
        "through the same lens, so this says the interaction is transparent in the "
        "limit, not that the trace is right",
    ),
)

_METRIC_INTERACTION_POWER = Metric(
    name="interaction_power_ratio_error",
    description=(
        "|sum |a_out|**2 / sum |a_in|**2 - 1| across the interaction. Every declared "
        "surface is phase-only, so this is an exact conservation law on the ray side -- "
        "and note it is EXACT here in a way it cannot be on the field-forming models: "
        "GENERALIZED_SNELL multiplies each ray's amplitude by |t| at one sample, so a "
        "unit-modulus surface leaves the array untouched rather than conserving a "
        "discrete sum"
    ),
    unit=None,
    blind_to=(
        "power that the ORDER does not carry. This model puts all of a ray's amplitude "
        "into the one declared order by construction, so ray power being conserved says "
        "nothing about diffraction efficiency -- a real surface would send most of it "
        "elsewhere. single_order_dominance, reported at 0.813, is the nearest thing this "
        "rung measures, and it is a property of the local spectrum rather than an "
        "efficiency",
    ),
)

_METRIC_CLIPPED = Metric(
    name="downstream_clipped_power_fraction",
    description=(
        "the fraction of |a|**2 entering the downstream trace that Optiland zeroed as "
        "clipped or non-finite, summed over every refractive group after the "
        "interaction. Power accounting for the half of the chain the interaction hands "
        "its rays to: a vignetted order produces a plausible image at the wrong "
        "brightness and an order-position centroid biased toward the surviving side"
    ),
    unit=None,
    blind_to=(
        "power that left the reconstruction WINDOW rather than the aperture, which is a "
        "different loss and is not an error at all -- the window is sized at 1.5 "
        "diffraction FWHM and deliberately truncates the skirt",
    ),
)


_SHARED_METRICS = (
    _METRIC_GRATING_EQUATION,
    _METRIC_OPL_REBASE,
    _METRIC_AMPLITUDE,
    _METRIC_ADMISSIBLE_POSITION,
    _METRIC_ADMISSIBLE_FIELD,
    _METRIC_ORDER_POSITION,
    _METRIC_ORDER_POSITION_FIELD,
    _METRIC_STREHL,
    _METRIC_PSF_FWHM,
    _METRIC_REFRACTIVE_LIMIT,
    _METRIC_INTERACTION_POWER,
    _METRIC_CLIPPED,
)


# The four extra observables the characterization family adds. They exist
# because the ticket asks for interference-structure fidelity "as a residual
# spatial spectrum rather than a single scalar", and because the claim that a
# ray-only model cannot produce the structure at all is a claim about a
# ray-only model, which has to be computed to be compared.
_METRIC_COHERENT_VS_GEOMETRIC = Metric(
    name="coherent_vs_geometric_relative_l2",
    description=(
        "relative L2 between the coherently reconstructed intensity and the traced ray "
        "density blurred to the diffraction limit, both normalized to unit sum on the "
        "structure window. The ray-only model is the ray density itself: a Gaussian blur "
        "of FWHM lambda / (2 NA) is the most generous thing a ray model can be given, "
        "since it grants it the diffraction limit it cannot derive"
    ),
    unit=None,
    blind_to=(
        "where the departure is. It is one norm over one window, which is exactly why "
        "the two fringe-contrast metrics and the spectral-band fractions are declared "
        "beside it -- CHE-148 asks for a residual spectrum rather than a scalar and this "
        "is the scalar",
        "the blur kernel's shape, which is a choice. A Gaussian of the Airy FWHM is not "
        "an Airy pattern, and a ray model that convolved with an Airy would look better "
        "here without becoming any more able to produce a fringe",
    ),
)

_METRIC_FRINGE_COHERENT = Metric(
    name="fringe_contrast_coherent",
    description=(
        "the deepest interior minimum of the coherent intensity's radial profile, "
        "relative to the smaller of its two flanking maxima: (env - min) / (env + min). "
        "A fringe visibility, bounded in [0, 1], measured on the profile rather than "
        "asserted from a picture"
    ),
    unit=None,
    blind_to=(
        "the number of fringes and their spacing, which the spectral bands carry. It "
        "reports the single deepest one",
        "azimuthal structure, being a radial average. Every declared instance is "
        "rotationally symmetric apart from the order displacement, so the average is "
        "about the order's own position",
    ),
)

_METRIC_FRINGE_GEOMETRIC = Metric(
    name="fringe_contrast_geometric",
    description=(
        "the same measurement on the blurred ray density. Declared as a metric rather "
        "than a footnote because it is the whole content of CHE-148's 'a ray-only model "
        "cannot produce this at all': it is 0.00000 -- no interior minimum above the "
        "1e-3 floor exists -- at every aperture and every ray count measured, while the "
        "coherent value is 0.98968 at R = 1 mm, 0.69297 at 3 mm and 0.27117 at 5 mm"
    ),
    unit=None,
    blind_to=(
        "a ray model with more physics in it. This is a ray DENSITY, which is what a "
        "ray tracer produces; a ray model carrying optical path and summing amplitudes "
        "is not a ray-only model, it is the coherent arm",
    ),
)

_METRIC_BAND_COHERENT = Metric(
    name="high_band_spectrum_fraction_coherent",
    description=(
        "the fraction of the coherent intensity's power spectrum in the second of four "
        "equal radial bands of the structure window's frequency grid. The residual "
        "spatial spectrum CHE-148 asks for, reduced to the one band where the two "
        "models actually differ; all four bands for both arms are in every record"
    ),
    unit=None,
    blind_to=(
        "the sign of the difference, which is the interesting part and is not monotone. "
        "Measured: at R = 1 mm the coherent arm has 1.667e-9 against the geometric arm's "
        "6.298e-26 -- sixteen orders of magnitude MORE high-frequency content, which is "
        "the fringes; at R = 3 mm 2.148e-4 against 2.005e-4, a factor of only 1.07, so "
        "the bands barely separate there; and at R = 5 mm 8.786e-3 against 1.629e-2, "
        "i.e. LESS, because the aberrated geometric caustic has sharp edges that "
        "diffraction smooths. A single-signed reading of this metric would be wrong at "
        "the widest of those three apertures",
    ),
)

_METRIC_BAND_GEOMETRIC = Metric(
    name="high_band_spectrum_fraction_geometric",
    description="the same band fraction on the blurred ray density.",
    unit=None,
    blind_to=(
        "the same things as its coherent twin. It is here so the comparison is a pair "
        "of measurements rather than a measurement and a claim",
    ),
)

_CHARACTERIZATION_METRICS = (
    *_SHARED_METRICS,
    _METRIC_COHERENT_VS_GEOMETRIC,
    _METRIC_FRINGE_COHERENT,
    _METRIC_FRINGE_GEOMETRIC,
    _METRIC_BAND_COHERENT,
    _METRIC_BAND_GEOMETRIC,
)


# ---------------------------------------------------------------------------
# Tolerance bases
# ---------------------------------------------------------------------------


_GRATING_EQUATION_BASIS = (
    "the float64 round-off floor, and this is one of the rare metrics where that is the "
    "correct basis rather than an admission. The gradient of a linear ramp is exact to "
    "round-off at ANY sample pitch on this estimator -- angle(t[+1] conj(t[-1])) of a "
    "unit-modulus product, so no unwrapping and no differencing error -- and the grating "
    "equation is then two multiplications and a square root. Measured across the whole "
    "declared set: exactly 0.0 at zero phase, 3.30e-17 at p / Lambda = 0.20, 7.02e-16 at "
    "the reference, and 1.40e-15 at Lambda = 50 um, which is the worst. 1e-14 is 7.2x "
    "that worst value and still 1e8 below anything the controls produce. Measured on "
    "PERIOD-100, it rejects the order-sign flip (1.1752e-2, which is 2 lambda / Lambda, "
    "1.2e12x), the surface-phasor conjugation (1.1752e-2 -- the same magnitude for a "
    "different reason, which is why both arms are run), and an un-renormalized outgoing "
    "direction that a downstream solver silently normalizes (1.0144e-7, 1.0e7x, and it "
    "is the analytic dx**3 / 2 = 1.0144e-7 at dx = 5.876e-3). The un-renormalized bundle "
    "does not even reach this metric on the shipping path: RayBundle refuses it with "
    "NON_UNIT_DIRECTION at a worst unit-norm deviation of 1.72635e-5 against a 1e-9 "
    "allowance"
)

_OPL_REBASE_BASIS = (
    "a convention reproduced from a declaration, so the only error available is "
    "arithmetic -- but the arithmetic is a SUBTRACTION of two nearly equal absolute "
    "paths, and that is what sets the floor. phi_snap / k0 is computed in the driver from "
    "the declared period, pitch, order and ray position; the interaction computes it from "
    "the transmission array; the metric is the difference of (opl_out - opl_in) and that, "
    "and opl_in is 0 on a collimated on-axis bundle, 8.6e-6 m off axis and 0.088 m on the "
    "relay. eps on 0.088 m is 2.0e-17 m, which is 3.3e-11 waves, so the collimated and "
    "off-axis arms can be exact while the relay cannot. Measured: exactly 0.0 at every "
    "collimated instance, 1.4274e-15 waves off axis, 2.9479e-12 waves on the relay. 1e-9 "
    "waves is 30x the derived relay floor and 339x the worst measured. It rejects a "
    "rebasing evaluated at the ray's true position rather than its nearest sample "
    "(up to pi p / Lambda radians = 0.5 waves at p / Lambda = 0.24), a rebasing with the "
    "wrong sign, a rebasing that dropped the incident path, and -- the case that made "
    "this metric worth declaring -- a rebasing that omitted the order factor m, which is "
    "0.318 waves for a 1 rad piston at m = -1 and up to 1.0 wave for a wrapped ramp"
)

_AMPLITUDE_BASIS = (
    "the same kind of statement for the amplitude, and the same floor: |a_out| is "
    "|a_in| times one array lookup, so 1e-14 is round-off with headroom on unit "
    "amplitudes. Measured 2.2e-16. It rejects an obliquity factor applied where the "
    "model declares none (cos(theta) - 1 = 1.7e-5 at Lambda = 50 um), an energy "
    "renormalization, and an amplitude that picked up the transmission's PHASE as well "
    "as its modulus (which for a unit-modulus ramp would be a factor of exp(i phi) and "
    "so a modulus error of zero -- caught instead by outgoing_opl_rebase_residual_waves, "
    "and the pair is what makes the two assertions independent)"
)

_ADMISSIBLE_POSITION_BASIS = (
    "float64 round-off on a sensor position of order 1e-4 to 1e-3 m carried through a "
    "trace with mm/m conversions, i.e. a few times 1e-17 m. Measured across the declared "
    "set: exactly 0.0 at zero phase, 3.09e-18 m at p / Lambda = 0.20, 5.35e-17 m at the "
    "reference, 9.60e-17 m on the relay, and 1.07e-16 m at Lambda = 50 um, which is the "
    "worst -- the residual grows with the deflection, because a longer lever amplifies a "
    "1e-16 disagreement in direction. 1e-15 m is a femtometre: 9.4x that worst value, and "
    "4e7 below the 41 nm at which the real singlet's own third-order behaviour starts to "
    "matter. Where the fully independent tilted-collimated arm exists it agrees to the "
    "same order (2.80e-17 m at Lambda = 200 um, 1.07e-16 m at Lambda = 50 um). What it "
    "rejects is any "
    "difference at all between the interaction's outgoing bundle and an independently "
    "constructed admissible one that a refractive trace can see: a wrong order, a wrong "
    "position at which the surface was read, a mis-declared reference plane. It is also "
    "the metric that fires hardest on the aliased instance, at 7.407e-4 m"
)

_ADMISSIBLE_FIELD_BASIS = (
    "the round-off floor of an ABSOLUTE-PHASE comparison in this chain, which is coarser "
    "than the bundles themselves and the derivation has to use the coarser one. The two "
    "arms' bundles agree to 1e-16 m in position and exactly 0.0 in optical path, but the "
    "reconstruction evaluates exp(i k0 opl) on an absolute optical path of 0.081 m "
    "(1.4e5 waves) on grating_then_lens and 0.49 m on the relay: the phase argument is "
    "8.6e5 rad there and 5.2e6 rad on the relay, float64 eps on the latter is 1.2e-9 rad, "
    "and a 1e-16 m disagreement in the path is itself 1e-9 rad. So about 1e-9 relative is "
    "the floor of the comparison rather than a chosen number. Measured: exactly 0.0 at "
    "zero phase, 3.19e-13 to 7.67e-13 across most of the declared set, and 1.44e-10 at "
    "Lambda = 50 um and 1.11e-10 on the relay, which are the two worst and are both an "
    "order of magnitude BELOW the derived bound -- the two arms' paths cancel better than "
    "the derivation requires. 1e-8 is 69x the worst measured and 10x the derived floor. "
    "This is the metric every optical-path and phasor control is judged on, and their "
    "margins are 1e8 or more because a mutated arm bears no phase relation to the "
    "reference at all: measured on PERIOD-100 the dropped-path arm is 1.00234 and the "
    "reconstruction phasor flip 1.14877, while the conjugated-path arm -- the weakest of "
    "the four -- is 0.02054, still 2e6x the threshold"
)

_ORDER_POSITION_BASIS = (
    "DERIVED from PARAXIAL_ORDER_POSITION's bound, not read off a run. That predicate "
    "declares the real train within one part in 1e4 of the paraxial closed form, and "
    "3e-4 is that with a factor of three of headroom -- enough that the measured law's "
    "own 10% scatter about c (R / 1 mm)**e cannot flip a verdict, and tight enough that "
    "R = 2 mm (3.8035e-4, declared outside) is rejected. The headroom is checked three "
    "ways and they agree: the three declared grating periods at R = 1 mm measure "
    "9.4959e-5, 9.2709e-5 and 8.3706e-5 -- a 3.16x margin, nearly independent of the "
    "order angle, which is what a pure aperture effect looks like; the departure is "
    "entirely ATTRIBUTED, since the no-DOE tilted-collimated arm departs by the same "
    "fraction with a double ratio of 0.0 to 4.9e-15 at every instance where that arm "
    "exists and the estimator is unaliased -- the one exception is "
    "B4-DOE-INLINE-PITCH-ALIASED at 1.6667, where the two arms genuinely disagree "
    "because the aliased order is not where the tilt puts it; and the order-sign control "
    "lands at 2.00009, which is 6.7e3x the threshold. "
    "On the relay topology the same threshold rejects 2.8023e-3, and that instance is "
    "recorded FAILING with PARAXIAL_ORDER_POSITION declaring it far outside beforehand "
    "-- it is the far end of the topology axis, not a failure"
)

_ORDER_POSITION_FIELD_BASIS = (
    "NOT GATE-DECIDING, and the reason is a floor nobody could derive. Across the "
    "declared collimated instances the value is 1.333e-4 at R = 0.5 mm, 1.858e-4 at "
    "1.0 mm, 1.768e-4 at Lambda = 50 um, 2.175e-4 at p = 5 um and 1.622e-5 at "
    "p = 20 um -- i.e. it moves by a factor of 13 across a set on which the RAY-side "
    "departure it is supposed to track is the SAME 9.271e-5 at four of those five "
    "points. A quantity that moves 13x while the thing it estimates does not move at "
    "all is not estimating it. Two candidate sources were checked and neither accounts "
    "for the pattern: the parabolic sub-sample bias is not constant in samples (0.028 of "
    "a window pixel at R = 0.5 mm against 0.077 at 1.0 mm) and the aberrated skirt's "
    "pull on a peak is not constant in metres either (59 nm, 82 nm, 168 nm at R = 0.5, "
    "1.0 and 2.0 mm). With no derivation the honest threshold is none: 1e-3 is recorded "
    "as the band every declared collimated instance sits inside -- by 2.6x at the "
    "widest, APERTURE-200's 3.775e-4, and by 62x at the narrowest -- may_gate is False, "
    "and CHE-148's own rule (characterization labelling and no gating tolerance where no "
    "oracle exists) is what this is. The metric is still worth reporting, and for the "
    "reason it was declared: it is the only observable that reads where the LIGHT is "
    "rather than where the rays are, so a defect that moved one without the other would "
    "show here first"
)

_STREHL_BASIS = (
    "ANALYTIC, from the sawtooth the OPL rebasing produces, with the headroom set by one "
    "identified effect the law does not contain -- and the cost of that headroom stated "
    "rather than absorbed. The law is (sin a / a)**2, a = pi p / Lambda, and the "
    "agreement measured WHILE AUTHORING over p / Lambda = 0.01, 0.05, 0.10, 0.15, 0.20 "
    "and 0.24 -- a span from 0.99967 to 0.82430 in the predicted value -- was 3.3e-6, "
    "3.2e-5, 3.5e-5, 3.0e-4, 3.5e-4 and 2.1e-4; three of those six points (0.10, 0.15, "
    "0.24) have no committed record and are labelled as authoring measurements for that "
    "reason. The DECLARED instances reach p / Lambda = 0.20, which is what makes the "
    "sweep an IDENTIFICATION rather than an agreement: B3-DOE-INLINE-PITCH-20 measures a "
    "departure of 3.4647e-4 from the sinc law while the competing Marechal form "
    "exp(-a**2 / 3) predicts 0.876696 against the sinc law's 0.875140, a 1.56e-3 "
    "separation -- so the committed evidence rules the Marechal form out by a factor of "
    "4.5. Across every collimated instance the family declares INSIDE the worst departure "
    "is that same 3.4647e-4, so 5e-3 is 14.4x it. That factor is NOT the derivation's "
    "headroom, it is the price of one instance: the peak ratio is normalized against the "
    "ON-AXIS unmodulated arm, so the order arm also carries whatever off-axis aberration "
    "its own displacement costs, and on the relay -- where the order sits 0.73 mm off "
    "axis behind a -1.914x second group -- that pushes the departure to 1.5142e-3, only "
    "3.3x inside. A threshold tight enough to gate the collimated topology at its own "
    "floor would be a threshold on the second singlet's field aberration wearing this "
    "metric's name, so the collimated claim is gated at 14.4x rather than at 3x and the "
    "loss of sensitivity is on the record here. What 5e-3 still rejects, with margin: an "
    "unrebased optical path (measured on PERIOD-100 the Strehl-like ratio falls from "
    "0.99967 to 6e-6, a departure of 0.99998, i.e. 200x the threshold), a doubled "
    "incident path off axis, and a pitch/period pair reported against the wrong sawtooth "
    "width -- the law's own value moves 0.092 for a halving of p at the widest declared "
    "point (0.87514 at p / Lambda = 0.20 against 0.96753 at 0.10)"
)

_PSF_FWHM_BASIS = (
    "ANALYTIC in its claim -- a linear phase ramp is a pure tilt, so an aplanatic train's "
    "point response cannot change width at all -- with the headroom set by one measured "
    "effect the claim's premise excludes: the real train is not aplanatic over the "
    "order's displacement. Measured across the declared collimated instances the "
    "residual runs from 1.8e-15 (zero gradient, where the claim is exact) through "
    "3.3e-6 at Lambda = 200 um and 1.29e-5 at the reference to 2.047e-4 at p = 20 um and "
    "3.872e-4 at R = 2 mm, and on the relay it is 1.3196e-3, where the order sits 0.73 mm "
    "off axis behind a -1.914x second group. The relay figure was checked for being a "
    "measurement artifact and is not one: it reads 1.320e-3, 1.214e-3 and 1.497e-3 at "
    "psf_window_px = 65, 129 and 257, i.e. stable across a 4x window refinement, so it is "
    "the second group's real off-axis aberration. 1e-2 is 7.6x that and 26x the worst "
    "collimated value; as with strehl_quantization_relative_error the factor is the price "
    "of keeping the relay in the gated family rather than the derivation's own headroom, "
    "and it is stated rather than implied. What it rejects is a broadening of one percent "
    "-- far below the 47% an unrebased optical path produced while authoring, and below "
    "anything a mis-declared pitch, a lost half-aperture or a wrong reconstruction plane "
    "would give. The absolute Airy comparison is deliberately NOT gated and is reported "
    "instead -- 9.3e-4 at the reference and 2.1e-2 on the relay, where the image-side "
    "pupil is neither a clean circle nor a declared quantity"
)

_REFRACTIVE_LIMIT_BASIS = (
    "the float64 round-off floor of the composition, derived from the quantities involved "
    "rather than from the result. The sensor-side optical path is of order 0.49 m on the "
    "relay, eps * 0.49 m is 1.1e-16 m, and 1.1e-16 m is 1.9e-10 waves at 0.5876 um; the "
    "trace adds mm/m conversions and Optiland's own arithmetic, so a few times that is "
    "the floor. 1e-8 waves is ten times it and 10.3x the worst measured. Measured: "
    "exactly 0.0 -- bitwise, every array -- for zero phase on grating_then_lens, "
    "4.3818e-12 waves for the zero-gradient piston there, 1.9236e-11 waves off axis, and "
    "9.7271e-10 waves on the relay, where the incident bundle carries real positions and "
    "9.661 waves of path sag. This is CHE-148's strongest available control and it is a "
    "conservation law, not an agreement: it holds whatever the ramp or the aberration "
    "does, which is why it is declared as an invariant"
)

_INTERACTION_POWER_BASIS = (
    "a conservation law that is EXACT on this model rather than discrete: |a_out| = "
    "|a_in| |t| per ray, and a phase-only surface has |t| = 1 at every sample, so the "
    "amplitude array is returned untouched and the sum is bitwise equal. 1e-14 is "
    "round-off with headroom on a sum over 5e4 unit terms; measured 0.0. It rejects a "
    "surface that is not actually unit modulus, an energy renormalization applied where "
    "none is declared, and an obliquity or order-splitting weight folded into the "
    "amplitude"
)

_CLIPPED_BASIS = (
    "zero is the only defensible threshold here, because ORDER_WITHIN_CLEAR_APERTURE is "
    "declared and every instance satisfies it with a factor of 11 to spare -- so any "
    "clipped power at all is either a vignetting the predicate failed to foresee or a "
    "non-finite ray, and both are defects rather than degrees. 1e-15 is exact zero with "
    "a float64 allowance on the ratio; measured 0.0 with 0 invalid rays on every "
    "declared instance. It rejects a diffracted order that misses the glass, which "
    "biases the order centroid toward the surviving side and would otherwise present as "
    "a plausible distortion"
)


# ---------------------------------------------------------------------------
# B3-DOE-INLINE: the gated family
# ---------------------------------------------------------------------------


# ORDER MATTERS: `verification/projection.py` names the FIRST gating tolerance as
# the metric of the projected claim row while taking the observed value from
# `GateDisposition`, so the two have to agree or the coverage matrix publishes one
# metric's threshold against another's number. `order_position_relative_error` is
# the disposition's metric and therefore has to lead. Same rule b3_4f_real.py
# follows, for the same reason.
_TOLERANCES = (
    Tolerance(
        metric="order_position_relative_error",
        threshold=3e-4,
        basis=_ORDER_POSITION_BASIS,
        basis_kind=ToleranceBasis.ANALYTIC_DERIVATION,
        may_gate=True,
        rejects=(
            "an order at the wrong angle or on the wrong side, a wrong frequency scale, "
            "a train carrying more than one part in 1e4 of departure from its own "
            "paraxial closed form"
        ),
    ),
    Tolerance(
        metric="grating_equation_direction_residual",
        threshold=1e-14,
        basis=_GRATING_EQUATION_BASIS,
        basis_kind=ToleranceBasis.NUMERICAL_PRECISION_FLOOR,
        may_gate=True,
        rejects=(
            "an order-sign flip, a conjugated surface transmission, a gradient taken "
            "along the wrong axis, and an outgoing direction that was not renormalized "
            "before a downstream solver silently normalized it"
        ),
    ),
    Tolerance(
        metric="order_position_vs_admissible_residual_m",
        threshold=1e-15,
        basis=_ADMISSIBLE_POSITION_BASIS,
        basis_kind=ToleranceBasis.NUMERICAL_PRECISION_FLOOR,
        may_gate=True,
        rejects=(
            "any difference between the interaction's outgoing bundle and an "
            "independently constructed admissible one that a refractive trace can see: "
            "a wrong order, a wrong position at which the surface was read, a "
            "mis-declared reference plane"
        ),
    ),
    Tolerance(
        metric="admissible_bundle_field_residual",
        threshold=1e-8,
        basis=_ADMISSIBLE_FIELD_BASIS,
        basis_kind=ToleranceBasis.NUMERICAL_PRECISION_FLOOR,
        may_gate=True,
        rejects=(
            "an optical path composed without the interaction's rebasing, an incident "
            "path counted twice, an optical path conjugated about its mean, and a "
            "phasor sign flipped on one side of the handoff"
        ),
    ),
    Tolerance(
        metric="order_position_field_relative_error",
        threshold=1e-3,
        basis=_ORDER_POSITION_FIELD_BASIS,
        basis_kind=ToleranceBasis.RECORDED_MEASUREMENT,
        may_gate=False,
        rejects=(
            "nothing, by declaration. It is the band the declared collimated instances "
            "occupy, recorded so a drift is visible, and it decides nothing"
        ),
    ),
    Tolerance(
        metric="strehl_quantization_relative_error",
        threshold=5e-3,
        basis=_STREHL_BASIS,
        basis_kind=ToleranceBasis.ANALYTIC_DERIVATION,
        may_gate=True,
        rejects=(
            "an unrebased or double-counted optical path, a sawtooth of the wrong "
            "width, and a surface read at the ray's true position instead of its "
            "nearest sample"
        ),
    ),
    Tolerance(
        metric="psf_fwhm_relative_error",
        threshold=1e-2,
        basis=_PSF_FWHM_BASIS,
        basis_kind=ToleranceBasis.ANALYTIC_DERIVATION,
        may_gate=True,
        rejects=(
            "a point response broadened by one percent, which is what a mis-declared "
            "pitch, a lost half-aperture or a wrong reconstruction plane produces from a "
            "surface that is only a tilt -- an unrebased optical path gives 47%"
        ),
    ),
)


_INVARIANTS = (
    Invariant(
        invariant_id="OPL_REBASE_CONVENTION",
        statement=(
            "the outgoing optical path is the incident optical path plus the declared "
            "surface's phase at the ray's own nearest sample, divided by k0 -- "
            "reproduced from the declaration rather than read back from the interaction"
        ),
        metric="outgoing_opl_rebase_residual_waves",
        tolerance=Tolerance(
            metric="outgoing_opl_rebase_residual_waves",
            threshold=1e-9,
            basis=_OPL_REBASE_BASIS,
            basis_kind=ToleranceBasis.NUMERICAL_PRECISION_FLOOR,
            may_gate=True,
            rejects=(
                "a rebasing evaluated at the ray's true position rather than its "
                "nearest sample, a sign error in the rebasing, and a dropped incident "
                "path"
            ),
        ),
    ),
    Invariant(
        invariant_id="AMPLITUDE_SEMANTICS",
        statement=(
            "the outgoing amplitude carries the modulus of the transmission at the "
            "ray's nearest sample and nothing else: no obliquity factor, no "
            "order-splitting weight, no renormalization"
        ),
        metric="outgoing_amplitude_residual",
        tolerance=Tolerance(
            metric="outgoing_amplitude_residual",
            threshold=1e-14,
            basis=_AMPLITUDE_BASIS,
            basis_kind=ToleranceBasis.NUMERICAL_PRECISION_FLOOR,
            may_gate=True,
            rejects=(
                "an obliquity factor applied where none is declared, an energy "
                "renormalization, and an amplitude that absorbed the transmission's "
                "phase"
            ),
        ),
    ),
    Invariant(
        invariant_id="REFRACTIVE_LIMIT_EXACT",
        statement=(
            "with the surface's phase flat, the whole chain reproduces the plain "
            "refractive train to round-off -- bitwise, where the incident bundle is "
            "collimated -- and with the phase flat but non-zero it reproduces it plus "
            "exactly the declared piston"
        ),
        metric="refractive_limit_residual_waves",
        tolerance=Tolerance(
            metric="refractive_limit_residual_waves",
            threshold=1e-8,
            basis=_REFRACTIVE_LIMIT_BASIS,
            basis_kind=ToleranceBasis.NUMERICAL_PRECISION_FLOOR,
            may_gate=True,
            rejects=(
                "a spurious deflection at a flat surface, a piston applied twice or "
                "with the wrong sign, and a reference plane silently moved by the "
                "interaction"
            ),
        ),
    ),
    Invariant(
        invariant_id="PHASE_ONLY_POWER_CONSERVED",
        statement=(
            "a phase-only surface conserves the incident rays' summed |a|**2 exactly "
            "across the interaction"
        ),
        metric="interaction_power_ratio_error",
        tolerance=Tolerance(
            metric="interaction_power_ratio_error",
            threshold=1e-14,
            basis=_INTERACTION_POWER_BASIS,
            basis_kind=ToleranceBasis.CONSERVATION_LAW,
            may_gate=True,
            rejects=(
                "a transmission that is not unit modulus, an energy renormalization, "
                "and a weight folded into the amplitude"
            ),
        ),
    ),
    Invariant(
        invariant_id="DOWNSTREAM_POWER_ACCOUNTED",
        statement=(
            "no power is lost in the refractive trace the interaction's rays are handed "
            "to: every declared instance satisfies ORDER_WITHIN_CLEAR_APERTURE with a "
            "factor of 11 to spare, so any clipping is a defect rather than a degree"
        ),
        metric="downstream_clipped_power_fraction",
        tolerance=Tolerance(
            metric="downstream_clipped_power_fraction",
            threshold=1e-15,
            basis=_CLIPPED_BASIS,
            basis_kind=ToleranceBasis.CONSERVATION_LAW,
            may_gate=True,
            rejects=(
                "a diffracted order that misses the glass, which biases the order "
                "centroid toward the surviving side and reads as a plausible distortion"
            ),
        ),
    ),
)


_NEGATIVE_CONTROLS = (
    NegativeControl(
        control_id="opl-not-rebased",
        description=(
            "compose the downstream traced optical path WITHOUT the interaction's "
            "rebasing, two ways, because the two are different defects and one of them "
            "is inert on axis. DROPPED: add zero instead of the outgoing bundle's own "
            "optical path, so the ramp never reaches the pupil at all. DOUBLE-COUNTED: "
            "add the outgoing path AND the incident path again, which is the mistake "
            "the convention invites, since opl_out already contains opl_in. NOTE the "
            "on-axis null and why it is reported rather than hidden: a collimated "
            "bundle launched at its own reference plane has opl_in identically 0, so "
            "double-counting adds nothing and the arm is bit-identical to the baseline. "
            "It is therefore demonstrated on the instances where the incident path is "
            "real -- OFFAXIS-01, whose 0.5 deg tilt puts 29.235 waves of sag across the "
            "bundle, and RELAY-01, whose converging incident bundle carries 9.658 waves "
            "-- with the on-axis inert arm measured and recorded beside them"
        ),
        mutation=(
            "opl_total -> traced_opl + 0 * opl_out (dropped) and opl_total -> "
            "traced_opl + opl_out + opl_in (double-counted), then recompute "
            "admissible_bundle_field_residual and strehl_quantization_relative_error"
        ),
        target_metric="admissible_bundle_field_residual",
    ),
    NegativeControl(
        control_id="phasor-sign-flip",
        description=(
            "flip the phasor sign on one side of the ray/wave handoff, two ways. "
            "RECONSTRUCTION: C_RAY_TO_WAVE's own Perturbation(phase_sign=-1) hook -- "
            "the shipping kernel with one sign changed, not a hand-written copy -- "
            "applied to the interaction arm while the independently constructed "
            "admissible arm keeps the correct convention. OPTICAL PATH: the composed "
            "path conjugated about its own mean, 2 mean(opl) - opl, which is the "
            "'ray minus chief' versus 'chief minus ray' error; the mean is left alone "
            "because negating an absolute path of 0.1 m would move the phase by 1e5 "
            "waves and prove nothing. NOTE what this control does NOT establish, "
            "because it is easy to over-read: a phasor flip applied to BOTH arms "
            "cancels, since both are reconstructed by the same summer, and a real "
            "intensity observable is blind to it outright -- the Strehl-like peak is "
            "bit-identical under the flip, because conjugating a field with real "
            "amplitudes leaves its modulus untouched, which is measured and recorded. "
            "The system-level question this control does answer is whether the driver "
            "composes the optical path with the sign the reconstruction expects"
        ),
        mutation=(
            "ray_to_wave(..., perturbation=Perturbation(phase_sign=-1)) on the "
            "interaction arm only, and separately opl -> 2 mean(opl) - opl on the "
            "composed bundle; recompute admissible_bundle_field_residual for both"
        ),
        target_metric="admissible_bundle_field_residual",
    ),
    NegativeControl(
        control_id="order-sign-flip",
        description=(
            "emit the opposite diffraction order, two ways, and they are the same "
            "magnitude for different reasons. ORDER: GeneralizedSnellParameters("
            "order=-m), so the model's own declared order is negated. SURFACE: the "
            "transmission conjugated, t -> conj(t), which is the phasor error "
            "DiffractiveSurface.from_phase exists to prevent -- a caller writing "
            "exp(-i phi) gets a real DOE that diffracts to the wrong side and looks "
            "entirely plausible. Both arms are run because a family that ran only the "
            "first would not show that the surface's own phasor convention is load "
            "bearing"
        ),
        mutation=(
            "order -> -order, and separately transmission -> conj(transmission); "
            "recompute grating_equation_direction_residual and "
            "order_position_relative_error for both"
        ),
        target_metric="grating_equation_direction_residual",
    ),
    NegativeControl(
        control_id="secondary-directions-not-renormalized",
        description=(
            "add the grating's transverse kick to the outgoing direction VECTOR without "
            "re-solving for its normal component -- d_out = d_in + m grad(phi) / (n_t "
            "k0) with d_z left alone -- which is the natural wrong implementation and "
            "leaves |d| = sqrt(1 + (m lambda / Lambda)**2). Two arms, and together they "
            "are exhaustive. REFUSED: the RayBundle contract rejects it outright with "
            "NON_UNIT_DIRECTION before any trace, at a worst unit-norm deviation of "
            "1.72635e-5 against a 1e-9 allowance -- the analytic dx**2 / 2 at "
            "dx = 5.876e-3 -- so the mutation cannot reach a number. "
            "MEASURED: the same vector normalized after the fact, which is what a "
            "downstream solver that silently normalizes would do -- the direction "
            "cosine becomes dx / sqrt(1 + dx**2) and the residual is the analytic "
            "dx**3 / 2 = 1.0144e-7, i.e. 1.0e7x the 1e-14 gate. The second arm exists "
            "because the first one only "
            "proves OUR contract catches it, and the physically interesting question is "
            "how large the error would have been if it had not"
        ),
        mutation=(
            "construct the outgoing bundle with d = (d_x + m lambda / Lambda, d_y, "
            "d_z^in); record the RayBundle refusal, then normalize the same vector and "
            "recompute grating_equation_direction_residual"
        ),
        target_metric="grating_equation_direction_residual",
    ),
)


B3_DOE_INLINE = register(
    BenchmarkFamily(
        family_id="B3-DOE-INLINE",
        family_version="1.0.0",
        category=BenchmarkCategory.B3,
        layer=BenchmarkLayer.SYSTEM,
        topology=_TOPOLOGY,
        question=(
            "when a diffractive interaction sits INSIDE a refractive train, so that its "
            "outgoing ray bundle has to be a legitimate input to a downstream "
            "refractive trace, does that bundle carry the conventions it declares -- "
            "the grating equation for its directions, opl_in + phi / k0 for its optical "
            "path, |a_in| |t| for its amplitude -- and does the resulting order land "
            "where the analytic grating-plus-lens answer puts it, with the Strehl the "
            "surface's own phase sampling predicts?"
        ),
        components=_COMPONENTS,
        # CONVENTION rather than FORWARD_ACCURACY, and the split from B4-DOE-INLINE
        # is where it shows: six of this family's seven gated tolerances and all
        # five of its invariants measure whether the outgoing bundle CARRIES the
        # conventions it declares -- the grating equation, opl_in + m phi / k0,
        # |a_in| |t|, the reference plane, ray power -- which is CHE-148's own
        # statement of why this rung is the hard one. The order position is a
        # forward-accuracy claim and it is here too, but it is the supporting
        # measurement rather than the thing that is new. B4-DOE-INLINE carries the
        # FORWARD_ACCURACY cell, on the same components and a superset of the
        # metrics, which is also what keeps the two out of one cell of the ledger.
        claim_kind=ClaimKind.CONVENTION,
        parameters=_parameters(),
        validity=(
            RAMP_GRADIENT_ESTIMATOR_UNALIASED,
            PROPAGATING_ORDER_EXISTS,
            SAWTOOTH_EQUIDISTRIBUTION,
            PARAXIAL_ORDER_POSITION,
            SENSOR_GRID_DIRECTION_CAPACITY,
            ORDER_WITHIN_CLEAR_APERTURE,
        ),
        oracle=FamilyOracle(
            kind=Oracle.ANALYTIC,
            independence=OracleIndependence.INDEPENDENT,
            description=(
                "four closed forms and one independent construction, none of which "
                "touches couplers/generalized_snell.py: (1) the grating equation for "
                "the outgoing direction, exact rather than paraxial; (2) the textbook "
                "grating-plus-lens order position f tan(arcsin(m lambda / Lambda)) -- "
                "which is the right form and not an approximation of an ABCD product, "
                "because the built system's A element from the DOE plane to the sensor "
                "is 9.0e-17 and its B element IS the EFL; (3) the Strehl of the "
                "nearest-sample phase sawtooth, (sin a / a)**2 with a = pi p / Lambda, "
                "verified over a 0.99967-to-0.82430 span in the predicted value and "
                "distinguished from the competing Marechal form; (4) the Airy FWHM of a "
                "circular pupil, reported. The construction is the outgoing bundle "
                "itself, rebuilt in the driver from (1) and from phi_snap / k0, and -- "
                "on grating_then_lens -- a plain collimated_bundle tilted to the "
                "analytic order angle, which contains no diffractive element at all"
            ),
            callable=grating_lens_answer,
            reference=(
                "src/verification/families/b3_doe_inline.py::grating_lens_answer; "
                "benchmarks/systems/b3_doe_inline.py::closed_form_outgoing"
            ),
        ),
        metrics=_SHARED_METRICS,
        invariants=_INVARIANTS,
        tolerances=_TOLERANCES,
        negative_controls=_NEGATIVE_CONTROLS,
        failure_semantics=(
            VerificationStatus.OUT_OF_VALIDITY,
            VerificationStatus.LOSSY_BUT_ALLOWED,
        ),
        execution_policy=DOE_INLINE_EXECUTION,
        stochastic_policy=DETERMINISTIC,
        gate_disposition=GateDisposition(
            status=GateStatus.MET,
            metric="order_position_relative_error",
            observed=9.4959e-5,
            evidence=(
                "benchmarks/systems/records/B3-DOE-INLINE-PERIOD-200.json",
                "benchmarks/systems/records/B3-DOE-INLINE-PERIOD-100.json",
                "benchmarks/systems/records/B3-DOE-INLINE-PERIOD-050.json",
                "benchmarks/systems/records/B3-DOE-INLINE-ORDER-MINUS1.json",
                "benchmarks/systems/records/B3-DOE-INLINE-ZEROPHASE.json",
                "benchmarks/systems/records/B3-DOE-INLINE-ZEROGRAD.json",
                "benchmarks/systems/records/B3-DOE-INLINE-PITCH-5.json",
                "benchmarks/systems/records/B3-DOE-INLINE-PITCH-20.json",
                "benchmarks/systems/records/B3-DOE-INLINE-APERTURE-050.json",
                "benchmarks/systems/records/B3-DOE-INLINE-APERTURE-200.json",
                "benchmarks/systems/records/B3-DOE-INLINE-OFFAXIS-01.json",
                "benchmarks/systems/records/B3-DOE-INLINE-RELAY-01.json",
            ),
            note=(
                "MET at all ten instances PARAXIAL_ORDER_POSITION declares INSIDE, and "
                "CHE-148's three-frequency criterion is met at one aperture with a 3.2x "
                "margin: Lambda = 200, 100 and 50 um report 9.4959e-5, 9.2709e-5 and "
                "8.3706e-5 against 3e-4. That the three barely differ is the informative "
                "part rather than a coincidence -- the departure is an APERTURE effect "
                "and not an angle effect, and the aperture sweep says so: 2.0915e-5 at "
                "R = 0.5 mm, 9.2709e-5 at 1.0 mm and 3.8035e-4 at 2.0 mm, successive "
                "ratios 4.43 and 4.10 against the third-order law's 4.00. "
                "It is also fully ATTRIBUTED, which is what makes the number a statement "
                "about the singlet and not about the interaction: the tilted-collimated "
                "arm -- a plain collimated_bundle at the analytic order angle with no "
                "diffractive element in it -- departs from the paraxial position by the "
                "SAME fraction at every instance where the estimator is unaliased, with "
                "a double ratio of 0.0 to 4.9e-15, and its sensor positions agree with "
                "the diffracted arm's to 1.07e-16 m at worst. "
                "The exactness limit is stronger than 'to round-off': with phi == 0 on "
                "the collimated topology every array is BITWISE identical -- outgoing "
                "directions, positions, amplitudes and optical paths all 0.0, the "
                "downstream trace bitwise the plain trace, and the Strehl-like ratio "
                "exactly 1.0000000000000000. With phi == 1 rad the geometry holds to "
                "1.095e-18 in direction cosine and the optical path reproduces the plain "
                "path plus the declared piston to 4.382e-12 waves; on the relay, whose "
                "incident bundle carries real positions and 9.661 waves of path sag, the "
                "same limits close to 9.326e-18 m and 9.727e-10 waves. "
                "The Strehl law is a measurement rather than a formality: the predicted "
                "value moves from 0.99992 (Lambda = 200 um) through 0.99868 (50 um) to "
                "0.99180 (p = 5 um) and the departures are 7.59e-7, 1.65e-5 and 3.19e-5. "
                "All four negative controls fire, with detection margins on PERIOD-100 "
                "of 1.487e12 (opl-not-rebased, dropped arm 1.00234), 1.704e12 "
                "(phasor-sign-flip, reconstruction arm 1.14877), 1.675e13 "
                "(order-sign-flip, 1.1752e-2) and a structural NON_UNIT_DIRECTION "
                "refusal plus a measured 1.0144e-7 against a 1e-14 gate "
                "(secondary-directions-not-renormalized). The double-counted "
                "optical-path arm is inert on axis by construction and fires at 1.00178 "
                "on OFFAXIS-01 (1.306e12x, incident path sag 29.235 waves) and 1.02487 "
                "on RELAY-01 (9.199e9x, sag 9.661 waves). "
                "APERTURE-200 and RELAY-01 are declared FAR_OUTSIDE by "
                "PARAXIAL_ORDER_POSITION beforehand and do not meet "
                "order_position_relative_error (3.8035e-4 against a predicted 3.7112e-4, "
                "and 2.8023e-3 against a predicted 2.8040e-3 -- the declared law "
                "predicts both to 2.5% and 0.06%). They are the far ends of the aperture "
                "and topology axes, not failures, and every convention and admissibility "
                "metric is met on both -- which is the point of keeping RELAY-01 in the "
                "gated family: it is the only instance where the incident optical path "
                "and the incident direction are per-ray, so nothing about the rebasing "
                "is trivially zero there. "
                "What this gate does NOT establish, and B4-DOE-INLINE measures instead: "
                "anything in the aberrated regime. There is no reference for the "
                "coherent image of a diffracted order through an aberrated singlet, so "
                "every number there is a measured departure and category B4 makes it "
                "structurally impossible to gate one."
            ),
        ),
        sampler=None,
        sampler_absent_reason=SamplerAbsentReason.SETUP_IS_THE_CERTIFICATE,
        sampler_absent_note=(
            "the declared set IS the argument, and a sampler cannot make it: three "
            "grating periods at one aperture are what CHE-148 asks for, two flat-phase "
            "instances are the exactness limit, an aperture sweep straddling "
            "PARAXIAL_ORDER_POSITION is what makes its bound checkable, and one "
            "off-axis and one relay instance are the only places the incident optical "
            "path and the incident direction are non-trivial. Each of those is a "
            "different question; drawing points on the joint domain would answer none "
            "of them"
        ),
        evidence=("benchmarks/systems/README.md",),
        notes=(
            "Exactly one declared axis differs between any instance and the instance it "
            "is read against, checked by the driver before it writes a record; see "
            "benchmarks/systems/b3_doe_inline.py::differing_axes."
        ),
    )
)


# ---------------------------------------------------------------------------
# B4-DOE-INLINE: the interference-structure characterization
# ---------------------------------------------------------------------------


B4_DOE_INLINE = register(
    BenchmarkFamily(
        family_id="B4-DOE-INLINE",
        family_version="1.0.0",
        category=BenchmarkCategory.B4,
        layer=BenchmarkLayer.SYSTEM,
        topology=_TOPOLOGY,
        question=(
            "away from the paraxial limit, where downstream refractive aberration puts "
            "real interference structure in the diffracted order's image: how much "
            "structure is there, how far is it from what a ray-only model can produce, "
            "and where in the spatial spectrum does the difference live -- as a function "
            "of used aperture, topology, incident ray count and a surface pitch past the "
            "gradient estimator's own aliasing bound?"
        ),
        components=_COMPONENTS,
        claim_kind=ClaimKind.FORWARD_ACCURACY,
        parameters=_parameters(),
        # PARAXIAL_ORDER_POSITION is dropped here, deliberately and for CHE-145's
        # reason: this family does not claim the paraxial limit, so carrying a
        # predicate every instance is outside would report a designed
        # configuration as an out-of-validity accident. The four SAMPLING and
        # GEOMETRY predicates stay, because a sampling failure here is still a
        # sampling failure -- and PITCH-ALIASED is deliberately outside one of
        # them, which is that instance's entire content.
        validity=(
            RAMP_GRADIENT_ESTIMATOR_UNALIASED,
            PROPAGATING_ORDER_EXISTS,
            SAWTOOTH_EQUIDISTRIBUTION,
            SENSOR_GRID_DIRECTION_CAPACITY,
            ORDER_WITHIN_CLEAR_APERTURE,
        ),
        oracle=FamilyOracle(
            kind=Oracle.NONE,
            independence=OracleIndependence.NOT_APPLICABLE,
            description=(
                "There is no reference for the coherent image of a diffracted order "
                "through an aberrated real singlet, and CHE-148 says so: it asks for "
                "characterization labelling and no gating tolerance wherever no oracle "
                "exists. The ray-only comparison arm IS computed at every instance and "
                "the difference is reported -- as the measured departure of the coherent "
                "field from what a ray density can express, which is the quantity of "
                "interest, not as an error against a truth. Credibility for these "
                "numbers comes from B3-DOE-INLINE: the same chain, the same code, driven "
                "into a limit where four closed forms do exist. The convention metrics "
                "are still measured here and are still round-off; they are simply not "
                "gated, because a B4 family may not gate"
            ),
            callable=None,
            reference="src/verification/families/b3_doe_inline.py::B3_DOE_INLINE",
        ),
        metrics=_CHARACTERIZATION_METRICS,
        # No tolerances and no invariants. The five invariants ARE measured here
        # -- the driver computes them and they appear in every record -- but
        # declaring one with a gating tolerance in a B4 family is exactly what the
        # category forbids, and declaring it non-gating would be a threshold that
        # cannot decide anything. B3-DOE-INLINE owns those gates.
        tolerances=(),
        invariants=(),
        # No negative controls: a control is judged by whether the mutated arm
        # crosses a GATE's threshold, and this family has no gate. All four are
        # demonstrated on B3-DOE-INLINE, whose thresholds exist, over a chain that
        # is bit-for-bit the chain that runs here.
        negative_controls=(),
        failure_semantics=(VerificationStatus.LOSSY_BUT_ALLOWED,),
        execution_policy=DOE_INLINE_EXECUTION,
        stochastic_policy=DETERMINISTIC,
        gate_disposition=GateDisposition(
            status=GateStatus.CHARACTERIZED_NO_GATE,
            note=(
                "By construction. CHE-148 requires characterization labelling with no "
                "gating tolerance wherever no oracle exists, and category B4 is how this "
                "repository makes that structural rather than a matter of discipline. "
                "Read these records for the measured interference structure and its "
                "parameter dependence; read B3-DOE-INLINE for whether the chain "
                "producing them is right."
            ),
        ),
        sampler=None,
        sampler_absent_reason=SamplerAbsentReason.STABLE_FINGERPRINT_VALUABLE,
        sampler_absent_note=(
            "each instance changes exactly one axis from B4-DOE-INLINE-REFERENCE, which "
            "is what makes the departures attributable; a sampler drawing several axes "
            "at once would produce numbers nobody could apportion"
        ),
        evidence=("benchmarks/systems/README.md",),
        notes=(
            "What the instances measured, so a reader has the shape of the answer before "
            "opening six records. Against B4-DOE-INLINE-REFERENCE (R = 1 mm, Lambda = "
            "100 um, p = 2 um, 12644 rays), whose coherent/geometric departure is 0.1156 "
            "with a coherent fringe contrast of 0.9897:\n"
            "(1) THE HEADLINE, and it is the same at every instance: "
            "fringe_contrast_geometric is EXACTLY 0.00000 everywhere -- the "
            "diffraction-blurred ray density has no interior minimum above the 1e-3 "
            "floor at any aperture, any topology or any ray count -- while the coherent "
            "field's is 0.9897 at R = 1 mm, 0.6930 at 3 mm and 0.2712 at 5 mm. That is "
            "CHE-148's 'a ray-only model cannot produce this at all', as a pair of "
            "measurements rather than a claim, and the ray model was handed the "
            "diffraction limit it cannot derive (a Gaussian blur of the Airy FWHM) to "
            "make the comparison as unfavourable as possible.\n"
            "(2) The spatial spectrum discriminates by sixteen orders of magnitude where "
            "the aberration is small and REVERSES where it is large. In the second of "
            "four radial bands: 1.667e-9 coherent against 6.298e-26 geometric at "
            "R = 1 mm; 2.148e-4 against 2.005e-4 at 3 mm (a factor of 1.07 -- the bands "
            "barely separate there); and 8.786e-3 against 1.629e-2 at 5 mm, i.e. the "
            "coherent field has LESS high-frequency content than the ray density, "
            "because the aberrated geometric caustic has sharp edges that diffraction "
            "smooths. A single-signed reading of this metric would be wrong at one of "
            "those three apertures.\n"
            "(3) The coherent/geometric L2 is NOT monotone in aperture: 0.1156, 0.3777, "
            "0.2984 at R = 1, 3, 5 mm. The fringe contrast is monotone and the L2 is "
            "not, which is why both are declared.\n"
            "(4) Ray-count convergence, the axis that replaces the ticket's "
            "secondary-ray count (GENERALIZED_SNELL emits no secondary rays): 12644 -> "
            "51040 incident rays moves the L2 from 0.1155668 to 0.1140479 (1.3% "
            "relative), the fringe contrast from 0.98968 to 0.99113 (0.15%), the Strehl "
            "departure from 6.687e-6 to 9.660e-6 and the order position from 9.2709e-5 "
            "to "
            "9.2831e-5 (1.3e-4 relative). Converged to about a percent on the structure "
            "metrics and to 1e-4 on everything physical.\n"
            "(5) The Strehl law breaks in the aberrated regime, as it must: the sawtooth "
            "prediction holds to 6.7e-6 at R = 1 mm, 4.1e-3 at 3 mm and 0.1578 at 5 mm. "
            "APERTURE-500's ratio comes out ABOVE ONE (1.1562), and that is the "
            "diagnostic rather than the result -- see the flags in (7).\n"
            "(6) PITCH-ALIASED, at p / Lambda = 0.30, is the instance that makes the "
            "gradient estimator's blind spot executable. The model's OWN smoothness "
            "margin reads +0.2000 -- positive, i.e. 'fine' -- against the analytic "
            "1 - 4 p / Lambda = -0.2000, and the emitted direction cosine is -3.917e-3 "
            "against the correct +5.876e-3: the order comes out on the WRONG SIDE of the "
            "axis. Every metric that looks at where the order is says so loudly "
            "(order_position_relative_error 1.6667, admissible position residual "
            "7.407e-4 m, admissible field residual 1.0585) and the model's third "
            "predicate is what catches it -- single_order_dominance falls from 0.8127 to "
            "1.68e-4. RAMP_GRADIENT_ESTIMATOR_UNALIASED declares the instance OUTSIDE "
            "before it runs.\n"
            "(7) Two instances are outside what the PSF window can measure and both say "
            "so in their own record: APERTURE-500 has "
            "psf_window_holds_the_point_response = False (its core is 10.98 um against a "
            "6.86 um window half-extent, because 0.99 waves of spherical aberration make "
            "it 2.4x the Airy width) and PITCH-ALIASED has psf_window_holds_the_order = "
            "False (its order is 740 um from a 34 um window). Their peak, FWHM and "
            "Strehl numbers are recorded and are not comparable to the others."
        ),
    )
)
