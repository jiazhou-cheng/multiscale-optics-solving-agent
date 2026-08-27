"""B3-DOE-INLINE / B4-DOE-INLINE: the declarations that carry the physics (CHE-148).

What is worth a test here is not "the numbers came out" -- the committed records
under ``benchmarks/systems/records/`` are that -- but the handful of statements
the families make that would be silently wrong if they drifted:

* the singlet is B3-4F-REAL's, by identity, and this rung's own hand-derived
  paraxial ABCD reproduces its read-back focal distances -- because that ABCD is
  one of the four oracle pieces and must not share code with what it judges;
* the reason the order-position reference is the *textbook* form rather than an
  ABCD product: the built system's A element from the DOE plane to the sensor is
  zero and its B element IS the effective focal length;
* the Strehl law is the sinc of the phase-sampling sawtooth and is
  *distinguishable* from the Marechal form the sweep was designed to rule out;
* the two sampling predicates bracket the declared instances, so the aperture
  sweep and the pitch sweep are convergence arguments rather than repetitions;
* the reconstruction window's pitch has exactly one definition, shared by the
  validity predicate and the driver;
* every declared instance differs from the instance it is read against in
  exactly one axis, which is CHE-148's own criterion;
* the reference bundle really is independent -- the driver does not import the
  model it is the reference for;
* and the chain composes end to end through the shipping path, with the
  diffractive model named.

The last one runs the real chain at a deliberately tiny configuration rather
than being skipped as expensive. A composition test that never composes is the
failure mode this file exists to avoid.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

# The drivers live under `benchmarks/`, which is not an installed package -- only
# `src/` is on the path. Same local addition `tests/test_b3_4f_real.py` makes,
# and for the same reason.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from benchmarks.systems import b3_doe_inline as driver  # noqa: E402
from couplers.interaction import DiffractiveModel  # noqa: E402
from verification.families.b3_doe_inline import (  # noqa: E402
    AIRY_FWHM_OVER_LAMBDA_2NA,
    B3_DOE_INLINE,
    B4_DOE_INLINE,
    DIFFRACTIVE_MODEL,
    PARAXIAL_ORDER_POSITION_LIMIT,
    PRESCRIPTION,
    TOPOLOGIES,
    analytic_order_direction,
    paraxial_image_side_na,
    paraxial_order_position_departure,
    psf_window_pitch_m,
    sawtooth_residue_count,
    strehl_quantization,
)
from verification.families.schema import BenchmarkCategory, ValidityState  # noqa: E402

pytestmark = [pytest.mark.coupler]


# ---------------------------------------------------------------------------
# The prescription, and the hand ABCD that is one of the oracle pieces
# ---------------------------------------------------------------------------


def test_the_singlet_is_b3_4f_reals_own_prescription_by_identity() -> None:
    """Asserted as object identity, not as agreement.

    A copied prescription could drift, and then two rungs would be describing two
    singlets while every record claimed one.
    """
    from verification.families import b3_4f_real

    assert PRESCRIPTION is b3_4f_real.PRESCRIPTION


def test_the_hand_abcd_reproduces_the_built_systems_focal_distances() -> None:
    """The paraxial ABCD in the driver is hand-derived so it shares no code with
    Optiland -- which is what makes it usable as a reference. It therefore has to
    be checked against Optiland's own read-back values, or the two could disagree
    and only the order position would ever notice.
    """
    assert PRESCRIPTION["effective_focal_length_mm"] == pytest.approx(
        driver.EFL_MM, rel=1e-9
    )
    assert PRESCRIPTION["back_focal_distance_mm"] == pytest.approx(driver.BFD_MM, rel=1e-9)


def test_the_sensor_is_the_paraxial_back_focal_plane_so_b_is_the_efl() -> None:
    """Why the reference is ``f tan(arcsin(m lambda / Lambda))`` and not an ABCD product.

    ``A = 0`` from the DOE plane to the sensor means the landing height does not
    depend on where a ray crossed the DOE, and ``B = f`` means the whole transfer
    is the textbook grating-plus-lens formula. If the sensor drifted off the focal
    plane both statements would fail together, and the analytic order position
    would quietly become the wrong closed form.
    """
    geom = driver.geometry(driver.ALL_PARAMETERS["B3-DOE-INLINE-PERIOD-100"])
    assert abs(geom["transfer_a"]) < 1e-12
    assert geom["transfer_b_mm"] == pytest.approx(driver.EFL_MM, rel=1e-12)
    # And the analytic order position is then f tan(theta), which the family
    # computes without any matrix at all.
    d_x, _d_y, d_z = geom["analytic_direction"]
    assert geom["analytic_order_position_m"][0] == pytest.approx(
        driver.EFL_MM * 1e-3 * d_x / d_z, rel=1e-12
    )


def test_the_relay_conjugate_really_is_a_conjugate() -> None:
    """The relay's sensor is solved for ``B = 0`` object-to-image, so the
    intermediate focus is imaged sharply. A residual there would be a defocus
    indistinguishable from an aberration in every metric this family measures.
    """
    geom = driver.geometry(driver.ALL_PARAMETERS["B3-DOE-INLINE-RELAY-01"])
    assert abs(geom["conjugate_residual_mm"]) < 1e-9
    assert geom["magnification"] == pytest.approx(-1.9138814, rel=1e-6)


# ---------------------------------------------------------------------------
# The Strehl law, and the fact that the sweep identifies it
# ---------------------------------------------------------------------------


def test_the_strehl_law_is_the_sawtooth_sinc_squared() -> None:
    for pitch_um, period_um in ((1.0, 100.0), (5.0, 100.0), (24.0, 100.0)):
        params = {
            "doe_phase_kind": "linear_ramp",
            "doe_pitch_um": pitch_um,
            "grating_period_um": period_um,
        }
        a = math.pi * pitch_um / period_um
        assert strehl_quantization(params) == pytest.approx(
            (math.sin(a) / a) ** 2, rel=1e-12
        )
    # A flat surface has no sawtooth at all and predicts exactly 1, which is what
    # lets an exactness instance be measured on the same metric as a ramp.
    for kind in ("flat_zero", "flat_piston"):
        assert (
            strehl_quantization(
                {"doe_phase_kind": kind, "doe_pitch_um": 5.0, "grating_period_um": 100.0}
            )
            == 1.0
        )


def test_the_declared_pitch_sweep_distinguishes_sinc_from_marechal() -> None:
    """The sweep has to be able to rule the competing law OUT, or agreeing with
    one of them says nothing.

    ``(sin a / a)**2`` and ``exp(-a**2 / 3)`` share their first two terms, so at
    the reference pitch they are indistinguishable at the measurement's own
    precision. The declared sweep must reach a pitch where they are not: the
    measured departures from the sinc law are 3.2e-6 and 3.2e-5 at
    ``p / Lambda`` = 0.01 and 0.05, so the two laws have to separate by more than
    that somewhere the family actually runs.
    """
    def gap(ratio: float) -> float:
        a = math.pi * ratio
        return abs((math.sin(a) / a) ** 2 - math.exp(-(a**2) / 3.0))

    # At the reference instance the two laws agree far better than the
    # measurement resolves, so that instance alone cannot identify the law.
    assert gap(0.01) < 1e-7
    # The authoring sweep reached p / Lambda = 0.24, where they differ by 3.7e-3
    # against a measured departure of 2.1e-4 -- a factor of 18.
    assert gap(0.24) > 3e-3
    # And a DECLARED instance has to reach that far, or the committed evidence
    # agrees with a law without identifying it. The widest declared ratio must
    # separate the two laws by more than the measured departure from the sinc law
    # at that ratio -- 3.5e-4, the worst measured anywhere in the authoring sweep.
    widest = max(
        params["doe_pitch_um"] / params["grating_period_um"]
        for params in driver.GATED_PARAMETERS.values()
        if params["doe_phase_kind"] == "linear_ramp"
    )
    assert gap(widest) > 3.0 * 3.5e-4, widest


def test_the_sawtooth_premise_is_declared_and_the_counterexample_is_refused() -> None:
    """The Strehl law needs the nearest-sample error to be a *sawtooth*, and that is
    a property of the launch grid against the surface pitch rather than something a
    run supplies.

    A commensurate choice inside the declared domain destroys the premise while
    leaving the code correct: at ``R = 1 mm`` and ``p = 20 um``,
    ``rays_per_axis = 101`` gives a ray spacing of exactly 20 um, so every ray snaps
    with the SAME offset, the sawtooth vanishes and the measured Strehl would go to 1
    against a prediction of 0.875. ``rays_per_axis = 201`` halves the spacing and
    gives two residues, which is no better. Without ``SAWTOOTH_EQUIDISTRIBUTION``
    either reads as a physics failure; with it, both are declared out-of-validity
    configurations. Pinned here because the hole was found by review, not by a run.
    """
    for per_axis, expected_residues in ((101, 1.0), (201, 2.0)):
        counterexample = driver._params(rays_per_axis=per_axis, doe_pitch_um=20.0)
        assert sawtooth_residue_count(counterexample) == expected_residues, per_axis
        state, margins = B3_DOE_INLINE.evaluate_validity(counterexample)
        assert margins["SAWTOOTH_EQUIDISTRIBUTION"] < 0.0, per_axis
        assert state is ValidityState.FAR_OUTSIDE, per_axis

    # Every declared instance is inside, and the count is NOT monotone in the ray
    # count -- it is a denominator, so RAYS-256 lands on 51 where 128 rays land on
    # 127. Both clear the floor; asserting the count itself would pin arithmetic
    # nobody depends on.
    for instance_id, params in driver.ALL_PARAMETERS.items():
        margin = B4_DOE_INLINE.evaluate_validity(params)[1]["SAWTOOTH_EQUIDISTRIBUTION"]
        assert margin > 0.0, instance_id
    assert sawtooth_residue_count(driver.ALL_PARAMETERS["B4-DOE-INLINE-RAYS-256"]) == 51.0
    assert sawtooth_residue_count(driver.ALL_PARAMETERS["B3-DOE-INLINE-PERIOD-100"]) == 127.0
    # A flat surface has no sawtooth to equidistribute, so the predicate has no
    # content there rather than a large margin.
    assert math.isinf(sawtooth_residue_count(driver.ALL_PARAMETERS["B3-DOE-INLINE-ZEROPHASE"]))


def test_the_airy_constant_is_the_bessel_root_and_not_a_round_number() -> None:
    """``FWHM = 1.028987 lambda / (2 NA)`` is the first solution of
    ``(2 J_1(v) / v)**2 = 1/2``. Pinned so that a "tidied" 1.0 or 1.03 would fail
    rather than shift every recorded window pitch by a percent.
    """
    from scipy.special import j1

    v = AIRY_FWHM_OVER_LAMBDA_2NA * math.pi / 2.0
    assert (2.0 * j1(v) / v) ** 2 == pytest.approx(0.5, rel=1e-4)


# ---------------------------------------------------------------------------
# The validity predicates bracket the declared instances
# ---------------------------------------------------------------------------


def test_the_aliasing_predicate_is_the_unwrapped_condition() -> None:
    """``1 - 4 p / Lambda``, exactly, and NOT the model's own measured margin.

    The model measures the WRAPPED estimator step, so above ``p / Lambda = 0.25``
    its margin comes back positive while the gradient has aliased -- measured
    +0.2000 at 0.30 with the order emitted on the wrong side of the axis. This
    predicate has no such branch, and a drift toward the measured form would
    reintroduce the blind spot silently.
    """
    _state, margins = B4_DOE_INLINE.evaluate_validity(
        driver.ALL_PARAMETERS["B4-DOE-INLINE-PITCH-ALIASED"]
    )
    aliased = driver.ALL_PARAMETERS["B4-DOE-INLINE-PITCH-ALIASED"]
    ratio = aliased["doe_pitch_um"] / aliased["grating_period_um"]
    assert ratio > 0.25
    assert margins["RAMP_GRADIENT_ESTIMATOR_UNALIASED"] == pytest.approx(
        1.0 - 4.0 * ratio, rel=1e-12
    )
    assert margins["RAMP_GRADIENT_ESTIMATOR_UNALIASED"] < 0.0
    # And the widest declared GATED pitch is inside, so the pitch axis straddles
    # the bound rather than sitting on one side of it.
    _state, inside = B3_DOE_INLINE.evaluate_validity(
        driver.ALL_PARAMETERS["B3-DOE-INLINE-PITCH-5"]
    )
    assert inside["RAMP_GRADIENT_ESTIMATOR_UNALIASED"] > 0.0


def test_the_paraxial_predicate_brackets_the_aperture_and_topology_axes() -> None:
    """The sweep has to straddle the boundary or it is not a convergence sweep,
    and the relay has to be outside or the family would be gating a closed form
    that does not describe it.
    """
    states = {
        instance_id: B3_DOE_INLINE.evaluate_validity(params)[0]
        for instance_id, params in driver.GATED_PARAMETERS.items()
    }
    assert states["B3-DOE-INLINE-APERTURE-050"] is ValidityState.INSIDE
    assert states["B3-DOE-INLINE-PERIOD-100"] is ValidityState.INSIDE
    assert states["B3-DOE-INLINE-APERTURE-200"] is ValidityState.FAR_OUTSIDE
    assert states["B3-DOE-INLINE-RELAY-01"] is ValidityState.FAR_OUTSIDE


def test_the_departure_law_is_the_measured_power_law_per_topology() -> None:
    """Anchored at ``R = 1 mm`` on each topology's own measured coefficient, with
    its own exponent -- quadratic on the collimated system and LINEAR on the
    relay. Collapsing the two into one law is the drift this pins against: it
    would make the relay read as marginally outside rather than as 30x outside.
    """
    for topology, expected_ratio in (
        ("grating_then_lens", 4.0),
        ("lens_then_grating_then_lens", 2.0),
    ):
        at_one = paraxial_order_position_departure(
            {"system_topology": topology, "used_semi_aperture_mm": 1.0}
        )
        at_two = paraxial_order_position_departure(
            {"system_topology": topology, "used_semi_aperture_mm": 2.0}
        )
        assert at_one == pytest.approx(TOPOLOGIES[topology]["departure_coefficient"], rel=1e-12)
        assert at_two / at_one == pytest.approx(expected_ratio, rel=1e-12)
    # The gating threshold has to sit above the validity bound or an instance the
    # family calls INSIDE could fail the gate it is judged by.
    threshold = B3_DOE_INLINE.tolerance_for("order_position_relative_error").threshold
    assert threshold > PARAXIAL_ORDER_POSITION_LIMIT
    assert threshold / PARAXIAL_ORDER_POSITION_LIMIT == pytest.approx(3.0, rel=1e-9)


def test_the_clear_aperture_predicate_is_satisfied_with_the_stated_margin() -> None:
    """Every declared instance clears the glass, which is what makes
    ``downstream_clipped_power_fraction``'s zero threshold defensible rather than
    lucky.
    """
    for instance_id, params in driver.ALL_PARAMETERS.items():
        margin = B4_DOE_INLINE.evaluate_validity(params)[1]["ORDER_WITHIN_CLEAR_APERTURE"]
        assert margin > 0.5, instance_id


# ---------------------------------------------------------------------------
# One definition of the reconstruction window
# ---------------------------------------------------------------------------


def test_the_window_pitch_has_one_definition_shared_by_predicate_and_driver() -> None:
    """A validity predicate bounding one pitch while the driver used another would
    be a bound on nothing. Checked by reading the pitch back out of a real run.
    """
    params = driver.ALL_PARAMETERS["B3-DOE-INLINE-APERTURE-050"]
    run = driver.run_chain(params)
    _measurements, diagnostics = driver.measure(run, params, with_structure=False)
    assert diagnostics["reconstruction"]["psf_window_pitch_m"] == pytest.approx(
        psf_window_pitch_m(params), rel=0.0, abs=0.0
    )
    # And on the collimated topology the declared image-side NA is R / f exactly,
    # which is the relation the window size follows from.
    assert paraxial_image_side_na(params) == pytest.approx(
        params["used_semi_aperture_mm"] / PRESCRIPTION["effective_focal_length_mm"],
        rel=1e-12,
    )


# ---------------------------------------------------------------------------
# CHE-148's own acceptance criteria, as executable checks
# ---------------------------------------------------------------------------


def test_every_instance_differs_from_its_reference_in_exactly_one_axis() -> None:
    for instance_id, params in driver.ALL_PARAMETERS.items():
        reference_id = driver._REFERENCE_INSTANCE[instance_id]
        if instance_id == reference_id:
            continue
        differing = driver.differing_axes(params, driver.ALL_PARAMETERS[reference_id])
        assert len(differing) == 1, (instance_id, differing)


def test_at_least_three_grating_frequencies_are_declared_at_one_aperture() -> None:
    """CHE-148 asks for the analytic order position over at least three grating
    frequencies. Checked as a property of the declared set rather than trusted to
    a docstring, and checked at ONE aperture so the three are a frequency axis
    rather than three different systems.
    """
    reference = driver.ALL_PARAMETERS["B3-DOE-INLINE-PERIOD-100"]
    periods = {
        params["grating_period_um"]
        for params in driver.GATED_PARAMETERS.values()
        if params["doe_phase_kind"] == "linear_ramp"
        and params["used_semi_aperture_mm"] == reference["used_semi_aperture_mm"]
        and params["system_topology"] == reference["system_topology"]
        and params["doe_pitch_um"] == reference["doe_pitch_um"]
    }
    assert len(periods) >= 3, periods


def test_both_exactness_limits_are_declared_as_instances() -> None:
    """The zero-PHASE and zero-GRADIENT limits are different controls -- one
    checks that a flat surface does nothing, the other that a non-zero constant
    does exactly one piston and nothing else -- so both are declared.
    """
    kinds = {params["doe_phase_kind"] for params in driver.GATED_PARAMETERS.values()}
    assert {"flat_zero", "flat_piston", "linear_ramp"} <= kinds


def test_the_characterization_family_cannot_gate() -> None:
    """CHE-148: no gating tolerance where there is no oracle.

    Category B4 already forbids it structurally, so this checks the thing that
    could still go wrong -- a tolerance added to the shared metric tuple in the
    expectation that the category would catch it in the OTHER family.
    """
    assert B4_DOE_INLINE.category is BenchmarkCategory.B4
    assert B4_DOE_INLINE.tolerances == ()
    assert B4_DOE_INLINE.invariants == ()
    assert B4_DOE_INLINE.negative_controls == ()
    assert B3_DOE_INLINE.category is BenchmarkCategory.B3
    assert any(t.may_gate for t in B3_DOE_INLINE.tolerances)
    # The field-side order position is declared and deliberately does NOT gate,
    # because its floor could not be derived. A drift to may_gate=True would turn
    # a recorded band into a verdict.
    assert not B3_DOE_INLINE.tolerance_for("order_position_field_relative_error").may_gate


def test_all_four_declared_controls_are_assigned_to_an_instance() -> None:
    """A declared control nobody runs reads exactly like a control that passed."""
    declared = {control.control_id for control in B3_DOE_INLINE.negative_controls}
    assigned = {cid for ids in driver._CONTROLS_ON.values() for cid in ids}
    assert declared == assigned
    assert declared == {
        "opl-not-rebased",
        "phasor-sign-flip",
        "order-sign-flip",
        "secondary-directions-not-renormalized",
    }


def test_the_opl_double_count_control_runs_where_the_incident_path_is_not_zero() -> None:
    """The double-counted arm is identically inert on a collimated on-axis bundle,
    whose incident optical path is exactly zero. So the control has to be
    demonstrated somewhere the incident path is real, or it would be a null
    reported as a pass.
    """
    carriers = [
        instance_id
        for instance_id, controls in driver._CONTROLS_ON.items()
        if "opl-not-rebased" in controls
    ]
    non_trivial = [
        instance_id
        for instance_id in carriers
        if driver.ALL_PARAMETERS[instance_id]["incident_tilt_deg"] != 0.0
        or driver.ALL_PARAMETERS[instance_id]["system_topology"] != "grating_then_lens"
    ]
    assert len(non_trivial) >= 2, carriers


def test_the_diffractive_model_is_named_and_is_generalized_snell() -> None:
    assert DiffractiveModel.GENERALIZED_SNELL.value == DIFFRACTIVE_MODEL
    for params in driver.ALL_PARAMETERS.values():
        assert params["diffractive_model"] == DIFFRACTIVE_MODEL


def test_the_reference_bundle_does_not_import_the_model_it_judges() -> None:
    """``closed_form_outgoing`` is the oracle for the interaction's ray output, so
    its independence is a property of the source and not of a comment. Checked at
    the module level: the driver reaches the interaction only through
    ``couplers.interaction``, never through ``couplers.generalized_snell``.
    """
    source = Path(driver.__file__).read_text(encoding="utf-8")
    assert "couplers.generalized_snell" not in source
    assert "generalized_snell_step" not in source


def test_the_closed_form_reference_reproduces_the_grating_equation() -> None:
    """The reference's directions are the closed form the family declares, checked
    against ``analytic_order_direction`` -- which is a different expression in a
    different module.
    """
    params = driver.ALL_PARAMETERS["B3-DOE-INLINE-PERIOD-100"]
    geom = driver.geometry(params)
    positions = driver._launch_positions(geom["used_semi_aperture_m"], 8)
    from couplers.ray_to_wave import collimated_bundle

    incident = collimated_bundle(
        positions_xy_m=positions,
        direction=(0.0, 0.0, 1.0),
        wavelength_m=PRESCRIPTION["wavelength_m"],
        plane_z_m=0.0,
        plane_name="doe",
    )
    reference, phase = driver.closed_form_outgoing(incident, params, geom)
    expected = np.asarray(analytic_order_direction(params))
    assert np.allclose(np.asarray(reference.directions), expected, rtol=0.0, atol=1e-15)
    # And the reference's optical path is the WRAPPED ramp phase over k0, because
    # that is what the declared surface array can hold.
    assert np.abs(phase).max() <= math.pi + 1e-12


# ---------------------------------------------------------------------------
# The composition, actually composed
# ---------------------------------------------------------------------------


@pytest.mark.optiland
@pytest.mark.integration
def test_the_chain_composes_end_to_end_at_a_tiny_configuration() -> None:
    """One real pass through the shipping path at a deliberately small size.

    Not a numerical claim -- the committed records are that. What is checked is
    that the composition still holds together and that the three things the chain
    must not do quietly are still not quiet: the model is named in the result, the
    zero-phase limit is BITWISE the plain refractive train, and a phase-only
    surface conserves ray power exactly rather than approximately.
    """
    params = driver._params(
        used_semi_aperture_mm=0.25,
        rays_per_axis=32,
        doe_pitch_um=5.0,
        psf_window_px=33,
        structure_window_px=64,
        structure_window_pitch_um=4.0,
    )
    assert B3_DOE_INLINE.evaluate_validity(params)[0] is ValidityState.INSIDE

    run = driver.run_chain(params)
    assert run["interaction_diagnostics"]["model"] == DiffractiveModel.GENERALIZED_SNELL.value
    assert run["ray_count"] > 0
    assert run["arms"]["order"][2]["invalid_rays"] == 0

    # The zero-phase limit, bitwise -- the strongest control this topology has.
    assert run["exactness"]["flat_zero"]["bitwise_identical"] is True
    assert run["exactness"]["flat_zero"]["downstream_position_residual_m"] == 0.0
    assert run["exactness"]["flat_zero"]["downstream_opl_residual_m"] == 0.0

    measurements, diagnostics = driver.measure(run, params)
    assert set(measurements) >= {metric.name for metric in B3_DOE_INLINE.metrics}
    assert set(measurements) >= {metric.name for metric in B4_DOE_INLINE.metrics}
    assert all(math.isfinite(m.value) for m in measurements.values())
    assert diagnostics["diffractive_model"] == DiffractiveModel.GENERALIZED_SNELL.value

    # The three conventions CHE-148 asks to be asserted rather than assumed.
    assert measurements["outgoing_opl_rebase_residual_waves"].value < 1e-12
    assert measurements["outgoing_amplitude_residual"].value == 0.0
    assert measurements["interaction_power_ratio_error"].value == 0.0
    # And the outgoing bundle is admissible: it traced without refusal and lands
    # where an independently constructed bundle does.
    assert measurements["order_position_vs_admissible_residual_m"].value < 1e-15
    assert measurements["downstream_clipped_power_fraction"].value == 0.0
    # A ray-only model cannot produce the interference structure, at any size.
    assert measurements["fringe_contrast_geometric"].value == 0.0
    assert measurements["fringe_contrast_coherent"].value > 0.5
