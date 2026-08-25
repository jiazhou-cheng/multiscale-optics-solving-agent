"""The B1 primitive families, and the physics their declarations have to get right.

CHE-106 / CHE-107 (M1.1, M1.2). These are declarations, not runs: nothing here
calls Optiland or Chromatix, and the families themselves report ``NOT_MEASURED``
or ``MEASURED_OFF_GATE`` accordingly. What *is* testable without a solver is
whether each family says something true and something checkable -- that its
oracle evaluates to the right number, that its validity predicate changes state
where the physics says it should, and that its negative controls name a defect
the metric could actually see.

The two families the rest exist around get more attention than the others:

* ``B1-RAY-OFFAXIS-OPL``. The omitted term is linear in the launch coordinate,
  so it is a *constant* on axis and cancels in the chief-ray subtraction. The
  test below shows the required tilt going to zero with the field angle, which
  is the reason three separate characterizations looked at this system and
  reported nothing wrong.
* ``B1-WAVE-ASM-VALIDITY``. Its whole subject is behaviour *near* a boundary, so
  the test walks an instance across ``z = N pitch^2 / lambda`` and checks the
  aggregated validity state changes on the way.
"""

from __future__ import annotations

import math

import pytest

from verification.families import (
    FAMILIES,
    BenchmarkCategory,
    BenchmarkFamily,
    ParameterKind,
    ValidityState,
    families_for_category,
)
from verification.families.b1_ray import (
    B1_RAY_EFL,
    B1_RAY_LAGRANGE,
    B1_RAY_OFFAXIS_OPL,
    B1_RAY_PLATE,
    B1_RAY_SNELL,
)
from verification.families.b1_wave import (
    B1_WAVE_AIRY,
    B1_WAVE_ASM_VALIDITY,
    B1_WAVE_FWDBWD,
    B1_WAVE_GAUSS,
    B1_WAVE_PLANEPHASE,
    B1_WAVE_TALBOT,
    B1_WAVE_TILT,
)
from verification.families.schema import GateStatus, Oracle, OracleIndependence

B1_RAY = (B1_RAY_EFL, B1_RAY_PLATE, B1_RAY_SNELL, B1_RAY_LAGRANGE, B1_RAY_OFFAXIS_OPL)
B1_WAVE = (
    B1_WAVE_GAUSS,
    B1_WAVE_AIRY,
    B1_WAVE_TILT,
    B1_WAVE_PLANEPHASE,
    B1_WAVE_FWDBWD,
    B1_WAVE_TALBOT,
    B1_WAVE_ASM_VALIDITY,
)
ALL_B1 = B1_RAY + B1_WAVE


# --------------------------------------------------------------------------- #
# What every B1 family owes
# --------------------------------------------------------------------------- #


def test_the_families_the_milestones_promised_are_registered() -> None:
    """Named explicitly rather than derived from the registry, so that dropping
    a family fails here instead of shrinking the expectation."""
    promised = {
        "B1-RAY-EFL",
        "B1-RAY-PLATE",
        "B1-RAY-SNELL",
        "B1-RAY-LAGRANGE",
        "B1-RAY-OFFAXIS-OPL",
        "B1-WAVE-GAUSS",
        "B1-WAVE-AIRY",
        "B1-WAVE-TILT",
        "B1-WAVE-PLANEPHASE",
        "B1-WAVE-FWDBWD",
        "B1-WAVE-TALBOT",
        "B1-WAVE-ASM-VALIDITY",
    }
    registered = {f.family_id for f in families_for_category(BenchmarkCategory.B1)}
    assert promised <= registered, f"missing: {sorted(promised - registered)}"


@pytest.mark.parametrize("family", ALL_B1, ids=lambda f: f.family_id)
def test_no_b1_gate_rests_on_an_oracle_that_shares_code(family: BenchmarkFamily) -> None:
    """PB7/CHE-58 finding F2 as a per-family assertion.

    The schema already refuses a ``CROSS_ROUTE`` oracle outside B4, so this
    would be redundant if it only checked the kind. What it adds is the
    positive statement: every B1 gate is decided by a closed form, a
    conservation law, or an independent implementation -- never by the solver
    under test answering a second time.
    """
    assert family.oracle.independence is OracleIndependence.INDEPENDENT
    assert family.oracle.kind in (
        Oracle.ANALYTIC,
        Oracle.CONSERVATION_LAW,
        Oracle.INDEPENDENT_IMPLEMENTATION,
    ), f"{family.family_id}: {family.oracle.kind.value} is not an independent decider"


@pytest.mark.parametrize("family", ALL_B1, ids=lambda f: f.family_id)
def test_every_b1_family_declares_a_physical_parameter_and_a_refinement(
    family: BenchmarkFamily,
) -> None:
    """A family with no physical axis is a fixed case wearing a family's clothes.

    The refinement rule is conditional, and the condition is the interesting
    part: a family that declares a NUMERICAL parameter has said its answer
    depends on a discretization, and it then owes a direction along which that
    dependence should vanish. B1-RAY-SNELL and B1-RAY-LAGRANGE declare none,
    because an exact algebraic relation has no discretization to refine -- which
    is a statement about the physics, not an omission.
    """
    assert family.parameters_of_kind(ParameterKind.PHYSICAL)
    numerical = family.parameters_of_kind(ParameterKind.NUMERICAL)
    if numerical:
        assert family.refinement_dimensions, (
            f"{family.family_id}: declares NUMERICAL parameters {[p.name for p in numerical]} "
            "and none of them says which way it refines"
        )


@pytest.mark.parametrize("family", ALL_B1, ids=lambda f: f.family_id)
def test_every_b1_family_declares_a_negative_control_on_a_metric_it_measures(
    family: BenchmarkFamily,
) -> None:
    assert family.negative_controls, f"{family.family_id}: no deliberately broken twin"
    metrics = {m.name for m in family.metrics}
    for control in family.negative_controls:
        assert control.target_metric in metrics
        assert control.mutation.strip(), f"{control.control_id}: no mutation to reimplement"


@pytest.mark.parametrize("family", ALL_B1, ids=lambda f: f.family_id)
def test_every_b1_gate_states_what_it_rejects(family: BenchmarkFamily) -> None:
    """A threshold that rejects nothing nameable is one nobody can defend."""
    for tolerance in family.gating_tolerances:
        assert tolerance.rejects.strip(), (
            f"{family.family_id}/{tolerance.metric}: gating tolerance names no wrong "
            "answer it rejects"
        )


@pytest.mark.parametrize("family", ALL_B1, ids=lambda f: f.family_id)
def test_an_unrun_family_says_so_rather_than_implying_a_pass(
    family: BenchmarkFamily,
) -> None:
    """MEASURED_OFF_GATE and NOT_MEASURED are the only honest states here.

    None of these families has been executed through the substrate. A family
    reporting MET would be claiming a measurement that does not exist.
    """
    assert family.gate_disposition is not None
    assert family.gate_disposition.status in (
        GateStatus.MEASURED_OFF_GATE,
        GateStatus.NOT_MEASURED,
    ), (
        f"{family.family_id} reports {family.gate_disposition.status.value}. Nothing "
        "has run these through the executor yet; CHE-113 and CHE-115 are what change it."
    )
    assert family.gate_disposition.note.strip()


# --------------------------------------------------------------------------- #
# The ray oracles evaluate to the right numbers
# --------------------------------------------------------------------------- #


def test_the_efl_oracle_reproduces_the_reference_prescription() -> None:
    params = {
        "radius_mm": 25.0,
        "index": 1.5168,
        "thickness_mm": 4.0,
        "wavelength_um": 0.5876,
        "marginal_ray_angle_rad": 0.01,
        "pupil_rings": 32,
    }
    assert B1_RAY_EFL.oracle.callable is not None
    assert B1_RAY_EFL.oracle.callable(params) == pytest.approx(25.0 / 0.5168)


def test_the_plate_oracle_keeps_its_sign() -> None:
    assert B1_RAY_PLATE.oracle.callable is not None
    shift = B1_RAY_PLATE.oracle.callable({"thickness_mm": 10.0, "index": 1.6})
    assert shift == pytest.approx(3.75)
    assert shift > 0.0, "away from the plate; -3.75 is the wrong answer this gate rejects"


def test_snell_is_exact_and_reversible() -> None:
    oracle = B1_RAY_SNELL.oracle.callable
    assert oracle is not None
    forward = oracle(
        {"index_incident": 1.0, "index_transmitted": 1.5, "incidence_angle_rad": 0.6}
    )
    back = oracle(
        {"index_incident": 1.5, "index_transmitted": 1.0, "incidence_angle_rad": forward}
    )
    assert back == pytest.approx(0.6, abs=1e-15)


def test_snell_past_the_critical_angle_is_out_of_validity_not_an_exception_class() -> None:
    """TIR is a physical regime. The predicate reports it; the oracle refuses to
    invent a refracted angle for it. Those are different statements and the
    substrate keeps them apart."""
    params = {"index_incident": 1.5, "index_transmitted": 1.0, "incidence_angle_rad": 0.9}
    critical = math.asin(1.0 / 1.5)
    assert params["incidence_angle_rad"] > critical

    status, margins = B1_RAY_SNELL.evaluate_validity(params)
    assert status is ValidityState.OUTSIDE
    assert margins["TIR_CRITICAL_ANGLE"] < 0.0

    with pytest.raises(ValueError, match="past the critical angle"):
        B1_RAY_SNELL.oracle.callable(params)  # type: ignore[misc]


def test_the_tir_margin_crosses_zero_at_the_critical_angle() -> None:
    critical = math.asin(1.0 / 1.5)
    base = {"index_incident": 1.5, "index_transmitted": 1.0}
    (predicate,) = B1_RAY_SNELL.validity
    assert predicate.margin({**base, "incidence_angle_rad": critical}) == pytest.approx(0.0)
    assert predicate.margin({**base, "incidence_angle_rad": 0.0}) == pytest.approx(1.0)
    assert predicate.state({**base, "incidence_angle_rad": critical * 0.5}) is (
        ValidityState.INSIDE
    )


def test_no_critical_angle_going_into_the_denser_medium() -> None:
    """+inf rather than a large finite number: an unbounded direction is not a
    comfortable one, and a sampler must not be handed a boundary to chase."""
    (predicate,) = B1_RAY_SNELL.validity
    margin = predicate.margin(
        {"index_incident": 1.0, "index_transmitted": 1.5, "incidence_angle_rad": 1.5}
    )
    assert math.isinf(margin)


def test_the_lagrange_invariant_is_the_conserved_quantity_it_claims() -> None:
    """Refraction at a surface preserves ``n u``; the invariant must not move."""
    oracle = B1_RAY_LAGRANGE.oracle.callable
    assert oracle is not None
    before = {
        "index_object_space": 1.0,
        "marginal_ray_angle_rad": 0.02,
        "marginal_ray_height_mm": 0.0,
        "chief_ray_angle_rad": 0.01,
        "chief_ray_height_mm": 5.0,
    }
    # Into n = 1.5: paraxial refraction scales the angles by 1/n at a flat
    # surface, and the n prefactor restores the product.
    after = {
        **before,
        "index_object_space": 1.5,
        "marginal_ray_angle_rad": 0.02 / 1.5,
        "chief_ray_angle_rad": 0.01 / 1.5,
    }
    assert oracle(after) == pytest.approx(oracle(before), rel=1e-15)


# --------------------------------------------------------------------------- #
# B1-RAY-OFFAXIS-OPL: the defect that is invisible on axis
# --------------------------------------------------------------------------- #


def test_the_required_launch_tilt_vanishes_on_axis() -> None:
    """Why three characterizations found nothing.

    The omitted term is ``n_object * (d0 . r_launch)``, linear in the launch
    coordinate. At zero field it is a constant across the pupil and cancels in
    the chief-ray subtraction, so an on-axis suite measures a required tilt of
    zero and cannot tell a correct implementation from one that omits it.
    """
    oracle = B1_RAY_OFFAXIS_OPL.oracle.callable
    assert oracle is not None
    base = {
        "pupil_diameter_m": 0.02,
        "wavelength_m": 5.5e-7,
        "index_object_space": 1.0,
    }
    assert oracle({**base, "field_angle_rad": 0.0}) == pytest.approx(0.0)

    off_axis = oracle({**base, "field_angle_rad": 0.2})
    assert off_axis > 1000.0, (
        "at Hy = 0.2 over a 20 mm pupil the omitted term is thousands of waves of "
        "tilt -- which is why recovering 0.13% of it still converged cleanly"
    )


def test_the_required_tilt_is_linear_in_the_launch_coordinate() -> None:
    """Doubling the pupil doubles the term; that linearity is exactly what makes
    it a constant, and therefore invisible, at zero field."""
    oracle = B1_RAY_OFFAXIS_OPL.oracle.callable
    assert oracle is not None
    base = {"field_angle_rad": 0.2, "wavelength_m": 5.5e-7, "index_object_space": 1.0}
    small = oracle({**base, "pupil_diameter_m": 0.01})
    large = oracle({**base, "pupil_diameter_m": 0.02})
    assert large == pytest.approx(2.0 * small)


def test_the_off_axis_family_refuses_an_on_axis_instance_as_out_of_validity() -> None:
    """The instance that cannot detect the defect is reported as outside the
    domain rather than as a passing case."""
    params = {
        "field_angle_rad": 0.0,
        "pupil_diameter_m": 0.02,
        "wavelength_m": 5.5e-7,
        "index_object_space": 1.0,
        "pupil_rings": 32,
        "prescription": "M3-REVERSE-TELEPHOTO",
    }
    on_axis = B1_RAY_OFFAXIS_OPL.instantiate("on-axis", params)
    off_axis = B1_RAY_OFFAXIS_OPL.instantiate("off-axis", {**params, "field_angle_rad": 0.2})
    assert on_axis.validity_status is ValidityState.FAR_OUTSIDE
    assert off_axis.validity_status is ValidityState.INSIDE


def test_the_off_axis_family_declares_the_che41_omission_as_a_control() -> None:
    ids = {c.control_id for c in B1_RAY_OFFAXIS_OPL.negative_controls}
    assert "omit-object-space-term" in ids
    assert "on-axis-cannot-detect-it" in ids, (
        "the control on the control: an on-axis run of the same omission must be shown "
        "not to fire, or the family has not established why the off-axis instance is "
        "the one doing the work"
    )


def test_the_che41_shortfall_is_three_orders_outside_the_gate() -> None:
    """0.0013 recovered against a 1e-3 tolerance on |1 - recovered|."""
    tolerance = B1_RAY_OFFAXIS_OPL.tolerance_for("launch_tilt_fraction_recovered")
    assert tolerance is not None
    shortfall = abs(1.0 - 0.0013)
    assert shortfall / tolerance.threshold > 900.0


# --------------------------------------------------------------------------- #
# The wave oracles
# --------------------------------------------------------------------------- #


def test_the_plane_wave_phase_oracle_pins_kz_off_axis() -> None:
    """On axis ``k_z = k`` and a frequency-grid scale error is invisible; off
    axis it is not. The same blind spot as the ray family's on-axis case."""
    oracle = B1_WAVE_PLANEPHASE.oracle.callable
    assert oracle is not None
    base = {"wavelength_um": 0.5, "distance_um": 100.0, "medium_index": 1.0}

    on_axis = oracle({**base, "transverse_frequency_per_um": 0.0})
    assert on_axis == pytest.approx(2.0 * math.pi * 100.0 / 0.5)

    off_axis = oracle({**base, "transverse_frequency_per_um": 1.0})
    assert off_axis < on_axis, "k_z shrinks as transverse frequency grows"


def test_a_plane_wave_past_the_light_cone_is_out_of_validity() -> None:
    params = {
        "wavelength_um": 0.5,
        "distance_um": 100.0,
        "medium_index": 1.0,
        "transverse_frequency_per_um": 3.0,
        "grid_n": 256,
        "sample_pitch_um": 0.25,
    }
    status, margins = B1_WAVE_PLANEPHASE.evaluate_validity(params)
    assert margins["PROPAGATING_BAND"] < 0.0
    assert not status.is_inside
    with pytest.raises(ValueError, match="evanescent"):
        B1_WAVE_PLANEPHASE.oracle.callable(params)  # type: ignore[misc]


def test_the_talbot_distance_goes_as_the_square_of_the_period() -> None:
    oracle = B1_WAVE_TALBOT.oracle.callable
    assert oracle is not None
    single = oracle({"period_um": 10.0, "wavelength_um": 0.5})
    double = oracle({"period_um": 20.0, "wavelength_um": 0.5})
    assert single == pytest.approx(2.0 * 100.0 / 0.5)
    assert double == pytest.approx(4.0 * single)


def test_the_half_talbot_control_targets_a_field_that_looks_just_as_periodic() -> None:
    control = next(c for c in B1_WAVE_TALBOT.negative_controls if c.control_id == "half-talbot")
    assert "half a period" in control.description


def test_the_round_trip_family_says_what_it_cannot_see() -> None:
    """A metric's blind spot is part of its definition, and this one has a large
    one: any error the backward pass undoes is invisible by construction."""
    metric = B1_WAVE_FWDBWD.metric("round_trip_relative_l2")
    assert any("its own inverse" in b for b in metric.blind_to)


# --------------------------------------------------------------------------- #
# B1-WAVE-ASM-VALIDITY: the family that sweeps a boundary
# --------------------------------------------------------------------------- #


def test_the_asm_validity_family_changes_state_across_its_own_boundary() -> None:
    """Walk an instance across ``z = N pitch^2 / lambda`` and watch the
    aggregated state move INSIDE -> NEAR_BOUNDARY -> OUTSIDE -> FAR_OUTSIDE.

    This is what a boolean validity flag could not express, and it is the reason
    the margin is signed and normalized.
    """
    grid_n, pitch_um, lam_um = 512, 0.25, 0.532
    limit_um = grid_n * pitch_um * pitch_um / lam_um

    def state_at(distance_um: float) -> ValidityState:
        params = {
            "waist_um": 5.0,
            "distance_um": distance_um,
            "wavelength_um": lam_um,
            "grid_n": grid_n,
            "sample_pitch_um": pitch_um,
        }
        status, _ = B1_WAVE_ASM_VALIDITY.evaluate_validity(params)
        return status

    assert state_at(0.2 * limit_um) is ValidityState.INSIDE
    assert state_at(0.99 * limit_um) is ValidityState.NEAR_BOUNDARY
    assert state_at(1.2 * limit_um) is ValidityState.OUTSIDE
    assert state_at(3.0 * limit_um) is ValidityState.FAR_OUTSIDE


def test_the_asm_validity_family_gates_only_inside_its_domain() -> None:
    """The tolerance is the same 2e-2 B1-WAVE-GAUSS uses, and its basis says so.

    The claim is not "the ASM is accurate to 2e-2"; it is "the ASM meets the
    same threshold on one side of the boundary and not on the other".
    """
    tolerance = B1_WAVE_ASM_VALIDITY.tolerance_for(
        "asm_radius_relative_error_vs_closed_form"
    )
    gauss = B1_WAVE_GAUSS.tolerance_for("gaussian_radius_relative_error")
    assert tolerance is not None and gauss is not None
    assert tolerance.threshold == gauss.threshold
    assert "INSIDE the declared validity domain only" in tolerance.basis


def test_the_asm_validity_family_declares_that_the_failure_is_silent() -> None:
    """A benchmark that only checked for exceptions would call the aliased
    regime fine, because nothing raises."""
    control = next(
        c for c in B1_WAVE_ASM_VALIDITY.negative_controls if c.control_id == "silent-wrap"
    )
    assert "not an exception" in control.description


def test_the_asm_validity_family_explains_why_it_has_no_sampler() -> None:
    """`sampler = None` is a declaration. Here the reason is specific: a uniform
    draw would spend its budget where the family has nothing to say."""
    assert B1_WAVE_ASM_VALIDITY.sampler is None
    assert B1_WAVE_ASM_VALIDITY.sampler_absent_reason is not None
    assert "boundary sampling" in B1_WAVE_ASM_VALIDITY.sampler_absent_note.lower()


# --------------------------------------------------------------------------- #
# The representation parameter that should not change the answer
# --------------------------------------------------------------------------- #


def test_the_tilt_family_makes_the_encoding_a_representation_parameter() -> None:
    """`tilt_encoding` should not change the answer, and the measured hazard says
    it does by 2*pi and a sign. That is the shape the four-way split exists to
    make expressible: a family whose RepresentationParameter moves the oracle
    value has found a defect, not a preference.
    """
    encoding = next(p for p in B1_WAVE_TILT.parameters if p.name == "tilt_encoding")
    assert encoding.kind is ParameterKind.REPRESENTATION
    assert set(encoding.domain or ()) == {"explicit_phase_ramp", "kykx_argument"}

    controls = {c.control_id for c in B1_WAVE_TILT.negative_controls}
    assert {"kykx-two-pi", "kykx-sign"} <= controls


def test_two_instances_of_one_family_differing_only_in_representation_differ_in_fingerprint() -> (
    None
):
    """They compute different things and must be told apart, even though the
    correct answer is the same."""
    base = {
        "tilt_rad": math.radians(5.0),
        "distance_um": 200.0,
        "wavelength_um": 0.532,
        "grid_n": 512,
        "sample_pitch_um": 0.5,
        "tilt_encoding": "explicit_phase_ramp",
    }
    a = B1_WAVE_TILT.instantiate("ramp", base)
    b = B1_WAVE_TILT.instantiate("kykx", {**base, "tilt_encoding": "kykx_argument"})
    assert a.fingerprint != b.fingerprint


# --------------------------------------------------------------------------- #
# Projection into the coverage view
# --------------------------------------------------------------------------- #


def test_the_b1_families_show_up_in_the_ledger_as_the_states_they_declare() -> None:
    from verification.claim_ledger import all_claims

    by_metric = {c.metric: c for c in all_claims() if c.metric}
    assert by_metric["efl_relative_error"].gate_status is GateStatus.MEASURED_OFF_GATE
    assert by_metric["launch_tilt_fraction_recovered"].gate_status is GateStatus.NOT_MEASURED
    assert by_metric["talbot_revival_relative_l2"].gate_status is GateStatus.NOT_MEASURED


def test_an_unmeasured_family_does_not_project_a_tolerance_it_has_not_met() -> None:
    """A declared threshold with nothing measured against it is a claim with no
    content, so the coverage view carries the threshold as a caveat instead."""
    from verification.families.projection import claims_from_family

    (claim,) = claims_from_family(B1_RAY_OFFAXIS_OPL)
    assert claim.tolerance is None
    assert claim.observed is None
    assert any("nothing measured against it yet" in c for c in claim.caveats)
    assert claim.metric == "launch_tilt_fraction_recovered", (
        "the metric NAME is known even when the number is not"
    )


def test_b0_has_not_landed_without_its_evidence() -> None:
    """A guard on scope creep, narrowed as milestones land.

    M1.1/M1.2 authored B1, M2 authored B2, M4.1 and M4.3 authored B3 and B4. B0
    -- contract and recovery, including the two measured traps where the
    contract reads ``ok`` and the physics is wrong -- is still owed by M1.3, and
    a B0 family appearing before it would mean that milestone landed without the
    measurements this file is the pattern for.
    """
    categories = {f.category for f in FAMILIES}
    assert BenchmarkCategory.B1 in categories
    assert BenchmarkCategory.B0 not in categories, (
        "M1.3 owes B0-UNITS-01 and B0-UNITS-02, whose numbers are preserved in "
        "src/verification/hazards.py and covered by tests/test_preserved_evidence.py"
    )
