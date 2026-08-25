"""The nine canonical B1 wave instances, run through the Chromatix graph node.

CHE-107 (M1.2). Three of these carried closed forms verified against the pinned
solver in a retired task set and four had never been executed at all. Every one
now runs through ``GraphExecutor``, with the input field supplied as a declared
graph source, so the precision bridge, the artifact boundary's pad-state and
normalization declarations, the input/output pitch distinction and the device
observation are all in the loop rather than skipped by a direct library call.

Three families were re-parameterized to make their own claims true, and each
change is a finding rather than a tuning
----------------------------------------------------------------------------
* **GAUSS, TILT and FWDBWD sat outside their own validity predicate.** The
  A1-verified Gaussian configuration -- w0 = 5 um, z = 100 um -- on a 512 grid at
  0.25 um pitch has ``N pitch^2 / lambda = 60.15 um`` against a 100 um
  propagation, so it met its gate to 1.8e-4 while being declared OUT_OF_VALIDITY.
  A number that good inside a domain the family says it has left is not a pass;
  it is a measurement whose validity claim contradicts itself. The grids moved,
  the physics did not.

* **TALBOT could not revive.** A binary grating carries every odd order, and the
  m-th order's phase after ``z_T`` departs from the paraxial ``2 pi m^2`` by
  ``(pi/2) m^4 (lambda/d)^2`` -- so the residual is set by the highest
  propagating order the grid admits. At d = 8 um with 32 samples per period that
  is m = 16 and 455 rad of dephasing, and the measured residual was 2.4e-1 with a
  5e-3 gate. Measured across eleven configurations; d = 32 um at 8 samples per
  period gives m = 4, 0.111 rad, and 2.2e-4. The tolerance did not move.

* **ASM-VALIDITY could not demonstrate its own failure mode.** At a waist of
  8 um the beam's bandwidth is 1/(pi w0) = 0.04 cycles/um against a grid Nyquist
  of 2, so the kernel's aliasing never touches it: four times past the sampling
  limit the closed form was still reproduced to 1.6e-5. At 0.6 um it wraps and is
  2.4 samples across the waist, so the estimator is 4% biased and the INSIDE
  instances fail too. 2.5 um is the resolution.
"""

from __future__ import annotations

import importlib.util
import math
import sys

import pytest

from core.execution import RunStatus
from core.paths import repository_root
from verification.result import NegativeControlOutcome
from verification.status import VerificationStatus

pytestmark = [pytest.mark.integration, pytest.mark.chromatix]


def _driver():
    name = "b1_wave_driver"
    if name in sys.modules:
        return sys.modules[name]
    path = repository_root() / "benchmarks" / "instances" / "b1_wave.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def runs():
    return _driver().run_all()


def _metric(run, name):
    return next(m for m in run.result.physics_accuracy if m.metric == name)


# --------------------------------------------------------------------------- #
# Coverage
# --------------------------------------------------------------------------- #


def test_every_declared_instance_was_executed(runs) -> None:
    declared = set(_driver().declared_instance_ids())
    assert set(runs) == declared
    assert len(declared) == 9, sorted(declared)


def test_all_seven_families_are_covered(runs) -> None:
    assert {run.family.family_id for run in runs.values()} == {
        "B1-WAVE-GAUSS",
        "B1-WAVE-AIRY",
        "B1-WAVE-TILT",
        "B1-WAVE-PLANEPHASE",
        "B1-WAVE-FWDBWD",
        "B1-WAVE-TALBOT",
        "B1-WAVE-ASM-VALIDITY",
    }


def test_every_run_went_through_the_graph_executor(runs) -> None:
    """Not a direct library call. The node, the adapter, the boundary.

    A ``chromatix.functional`` call would measure Chromatix's arithmetic and skip
    the adapter's conventions, which is a narrower claim than any of these
    families state.
    """
    for instance_id, run in runs.items():
        assert run.record.nodes, instance_id
        assert [node.component for node in run.record.nodes] == ["M_WAVE_CHROMATIX"]
        assert run.record.provenance.get("executor_version"), instance_id


def test_no_gate_is_decided_by_another_chromatix_call(runs) -> None:
    from verification.families.schema import ToleranceBasis

    admissible = {
        ToleranceBasis.ANALYTIC_DERIVATION,
        ToleranceBasis.CONSERVATION_LAW,
        ToleranceBasis.NUMERICAL_PRECISION_FLOOR,
        ToleranceBasis.INDEPENDENT_DERIVATION,
    }
    for run in runs.values():
        for tolerance in run.family.tolerances:
            if tolerance.may_gate:
                assert tolerance.basis_kind in admissible
                assert tolerance.rejects.strip()


# --------------------------------------------------------------------------- #
# Every instance is inside the domain it claims, or says it is not
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "instance_id",
    [
        "B1-WAVE-GAUSS-01",
        "B1-WAVE-AIRY-01",
        "B1-WAVE-TILT-01",
        "B1-WAVE-PLANEPHASE-01",
        "B1-WAVE-FWDBWD-01",
        "B1-WAVE-TALBOT-01",
        "B1-WAVE-ASM-VALIDITY-01",
        "B1-WAVE-ASM-VALIDITY-02",
    ],
)
def test_a_met_gate_is_inside_its_own_validity_domain(runs, instance_id: str) -> None:
    """The contradiction CHE-107 found and closed.

    A family that reports a metric inside its tolerance AND a status of
    out_of_validity is reporting a well-resolved answer to a question it has
    declared itself unable to answer. Three of these instances were in exactly
    that state -- GAUSS, TILT and FWDBWD -- because their grids did not satisfy
    the ``z <= N pitch^2 / lambda`` predicate the families declare. The grids
    moved; the physics did not.
    """
    run = runs[instance_id]
    assert run.result.status is VerificationStatus.OK, (
        f"{instance_id}: {run.result.status.value}. Margins: "
        + "; ".join(f"{m.predicate_id}={m.margin:+.4f}" for m in run.result.validity.margins)
    )
    assert run.result.validity.observed.is_inside
    assert all(metric.met for metric in run.result.physics_accuracy)


# --------------------------------------------------------------------------- #
# The closed forms
# --------------------------------------------------------------------------- #


def test_the_gaussian_reproduces_its_inherited_number(runs) -> None:
    """1.8e-4, which is what the retired A1 task measured.

    The point of re-running an inherited figure is that it becomes reproducible
    from code in the tree rather than from a report. Agreement to the third
    significant figure is the useful outcome: it means the historical number was
    right and the configuration is understood.
    """
    metric = _metric(runs["B1-WAVE-GAUSS-01"], "gaussian_radius_relative_error")
    assert metric.met is True
    assert metric.measured.value == pytest.approx(1.8e-4, rel=0.15)
    # The estimator's own bias on the analytic field, as the error bar.
    assert metric.measured.uncertainty is not None
    assert metric.measured.uncertainty < metric.measured.value


def test_the_airy_null_is_bias_corrected_rather_than_trusted(runs) -> None:
    """A first-null estimator is badly biased at coarse sampling.

    So the same estimator is run over the analytic Airy pattern on the same grid
    and the cancellation is reported. Comparing a measured null straight to
    ``0.61 lambda/NA`` at coarse pitch measures the estimator, not the solver.
    """
    run = runs["B1-WAVE-AIRY-01"]
    metric = _metric(run, "airy_first_null_relative_error")
    assert metric.met is True
    codes = {d["code"] for d in run.record.diagnostics}
    assert "ESTIMATOR_BIAS_CANCELLED_RATIO" in codes
    assert "FOCUS_IS_INSIDE_THE_ASM_SAMPLING_LIMIT" in codes
    # The factor-of-two confusion is rejected.
    control = next(
        c for c in run.result.negative_control_results if c.control_id == "diameter-for-radius"
    )
    assert control.outcome is NegativeControlOutcome.FIRED


def test_the_tilt_walkoff_carries_its_sign(runs) -> None:
    run = runs["B1-WAVE-TILT-01"]
    metric = _metric(run, "tilt_centroid_signed_relative_error")
    assert metric.met is True
    controls = {c.control_id: c for c in run.result.negative_control_results}
    assert controls["kykx-sign"].outcome is NegativeControlOutcome.FIRED
    assert controls["kykx-two-pi"].outcome is NegativeControlOutcome.FIRED
    # A sign error on a signed metric is a 2x relative error, so the control's
    # separation from the baseline is enormous rather than marginal.
    assert controls["kykx-sign"].mutated.value > 1.9


def test_the_kykx_unit_hazard_is_measured_where_it_belongs(runs) -> None:
    """Not in this family, and the split is deliberate.

    ``kykx`` means cycles per length on ``asm_propagate`` and radians per length
    on ``plane_wave``, and the displacement runs opposite in sign to the
    parameter on the propagator. That is a CONVENTION trap, so B0-UNITS-02 owns
    it; this family owns the walk-off. Measuring both here would conflate "the
    physics is right" with "the caller read the right unit".
    """
    detail = next(
        d["detail"]
        for d in runs["B1-WAVE-TILT-01"].record.diagnostics
        if d["code"] == "THE_KYKX_ENCODING_IS_MEASURED_ELSEWHERE"
    )
    assert "B0-UNITS-02" in detail


def test_the_plane_wave_phase_is_gated_off_axis(runs) -> None:
    """On axis ``k_z = k`` and a frequency-grid scale error is exactly invisible.

    So the instance declares a nonzero transverse frequency, and the measured
    quantity is the advance RELATIVE to an on-axis plane wave -- which is
    ``(k_z - k) z``, where the absolute advance is thousands of radians and wraps.
    """
    run = runs["B1-WAVE-PLANEPHASE-01"]
    assert float(run.instance.parameters["transverse_frequency_per_um"]) > 0.0
    metric = _metric(run, "plane_wave_phase_residual_rad")
    assert metric.met is True
    codes = {d["code"] for d in run.record.diagnostics}
    assert "OFF_AXIS_IS_WHERE_THIS_IS_MEASURABLE" in codes

    controls = {c.control_id: c for c in run.result.negative_control_results}
    assert controls["phasor-sign-flip"].outcome is NegativeControlOutcome.FIRED
    assert controls["frequency-grid-two-pi"].outcome is NegativeControlOutcome.FIRED


def test_the_round_trip_returns_the_input_and_can_be_made_to_fail(runs) -> None:
    """A round trip that cannot be made to fail proves nothing.

    Both controls are asymmetries the round trip cannot undo: one sample of
    lateral shift between the legs, and an aperture wide enough that energy
    leaves the window. Neither is a factor-scale change -- a control that moved
    the field by an order of magnitude would say nothing about a 1e-5 gate.
    """
    run = runs["B1-WAVE-FWDBWD-01"]
    metric = _metric(run, "round_trip_relative_l2")
    assert metric.met is True
    assert metric.measured.value < metric.measured.uncertainty, (
        "the round trip should land BELOW the single-pass complex64 floor, because "
        "the two legs' phase errors are correlated"
    )
    controls = {c.control_id: c for c in run.result.negative_control_results}
    assert controls["asymmetric-fftshift"].outcome is NegativeControlOutcome.FIRED
    assert controls["wrapped-aperture"].outcome is NegativeControlOutcome.FIRED


def test_the_round_trip_declares_what_it_cannot_see(runs) -> None:
    """A convention error shared by both legs cancels exactly.

    Which is why this family is not sufficient on its own, and why
    B1-WAVE-PLANEPHASE gates the one-way phase separately.
    """
    codes = {d["code"] for d in runs["B1-WAVE-FWDBWD-01"].record.diagnostics}
    assert "WHAT_A_ROUND_TRIP_CANNOT_SEE" in codes


# --------------------------------------------------------------------------- #
# Talbot
# --------------------------------------------------------------------------- #


def test_the_grating_revives_at_the_talbot_distance(runs) -> None:
    run = runs["B1-WAVE-TALBOT-01"]
    metric = _metric(run, "talbot_revival_relative_l2")
    assert metric.met is True, f"{metric.measured.value:.3e}"


def test_the_half_talbot_control_is_a_shifted_grating_not_a_scrambled_one(runs) -> None:
    """Which is what makes it the right control for a revival claim.

    At ``z_T/2`` the pattern revives SHIFTED by half a period: a field that looks
    exactly as much like a grating as the revival does, displaced. A metric that
    only checked "does this look periodic" would pass it, and this one does not.
    """
    control = next(
        c
        for c in runs["B1-WAVE-TALBOT-01"].result.negative_control_results
        if c.control_id == "half-talbot"
    )
    assert control.outcome is NegativeControlOutcome.FIRED
    assert control.mutated.value > 0.1, (
        "an anticorrelated pattern should be a large residual, not a marginal one"
    )


def test_the_talbot_configuration_keeps_its_orders_paraxial(runs) -> None:
    """The reason the period is 32 um and not 8.

    A binary grating carries every odd order, and the m-th order's phase after
    ``z_T`` departs from the paraxial ``2 pi m^2`` by ``(pi/2) m^4
    (lambda/d)^2``. The residual is therefore set by the highest propagating
    order the grid admits, ``m_max = d / (2 pitch)``, and it is a computable
    number rather than an empirical one.
    """
    p = runs["B1-WAVE-TALBOT-01"].instance.parameters
    period = float(p["period_um"])
    wavelength = float(p["wavelength_um"])
    pitch = period / int(p["samples_per_period"])
    m_max = int(period / (2.0 * pitch))
    dephasing = (math.pi / 2.0) * m_max**4 * (wavelength / period) ** 2
    assert m_max == 4
    assert dephasing < 0.2, f"highest-order dephasing {dephasing:.4f} rad"
    # And the measured residual is the same order as that dephasing, which is the
    # check that the model of the residual is the right model.
    measured = _metric(runs["B1-WAVE-TALBOT-01"], "talbot_revival_relative_l2").measured.value
    assert measured < dephasing


def test_the_grating_is_exactly_periodic_on_the_grid(runs) -> None:
    """Otherwise the residual is the wrap discontinuity, not the propagator."""
    p = runs["B1-WAVE-TALBOT-01"].instance.parameters
    assert int(p["samples_per_period"]) * int(p["periods_across_grid"]) > 0
    codes = {d["code"] for d in runs["B1-WAVE-TALBOT-01"].record.diagnostics}
    assert "EXACTLY_PERIODIC_ON_THE_GRID" in codes
    assert "REVIVAL_IS_INSIDE_THE_SAMPLING_LIMIT" in codes


# --------------------------------------------------------------------------- #
# The validity sweep: the load-bearing family
# --------------------------------------------------------------------------- #


def test_the_sweep_straddles_the_declared_boundary(runs) -> None:
    """Three instances, three sides, and the margins say which is which."""
    margins = {}
    for suffix, side in ((1, "inside"), (2, "near_boundary"), (3, "outside")):
        run = runs[f"B1-WAVE-ASM-VALIDITY-{suffix:02d}"]
        assert run.instance.expected["side"] == side
        margin = next(
            m.margin for m in run.result.validity.margins if m.predicate_id == "ASM_TF_SAMPLING"
        )
        margins[side] = margin
    assert margins["inside"] > margins["near_boundary"] > 0.0 > margins["outside"]
    # NEAR the boundary means near, not merely inside: a signed normalized margin
    # is what makes that expressible at all, and a boolean predicate could not.
    assert margins["near_boundary"] < 0.15


def test_the_inside_and_boundary_instances_meet_their_gates(runs) -> None:
    for suffix in (1, 2):
        run = runs[f"B1-WAVE-ASM-VALIDITY-{suffix:02d}"]
        assert run.result.status is VerificationStatus.OK
        for metric in run.result.physics_accuracy:
            assert metric.met is True, f"{run.instance.instance_id}/{metric.metric}"


def test_crossing_the_boundary_is_silent_and_wrong(runs) -> None:
    """The whole content of the family, in one assertion.

    The far-side run SUCCEEDS. Nothing raises, nothing refuses, and a field comes
    back that looks like a Gaussian. Its radius is 9% wrong and 0.8% of its power
    sits within one pitch of the window edge, where a correctly propagated beam
    of that size puts essentially none. A benchmark that only checked "did it
    run" would report this as fine.
    """
    run = runs["B1-WAVE-ASM-VALIDITY-03"]
    assert run.record.status is RunStatus.SUCCEEDED
    assert run.record.refusal is None
    assert run.result.status is VerificationStatus.OUT_OF_VALIDITY

    radius = _metric(run, "asm_radius_relative_error_vs_closed_form")
    wrapped = _metric(run, "wrapped_power_fraction")
    assert radius.met is False, (
        "the gate must NOT be met past the boundary; a pass here would mean the "
        "metric cannot see aliasing"
    )
    assert wrapped.met is False
    assert radius.measured.value > 5e-2

    codes = {d["code"] for d in run.record.diagnostics}
    assert "IT_RAN" in codes


def test_the_two_boundary_controls_are_cross_instance_and_fire(runs) -> None:
    """The mutation is the propagation DISTANCE and nothing else.

    Same code, same grid, same oracle, same estimator -- one parameter moved.
    That is what makes these controls rather than two unrelated runs, and it is
    why their baselines are the inside instance's numbers.
    """
    run = runs["B1-WAVE-ASM-VALIDITY-03"]
    controls = {c.control_id: c for c in run.result.negative_control_results}
    for name in ("past-the-boundary-must-not-pass", "silent-wrap"):
        control = controls[name]
        assert control.outcome is NegativeControlOutcome.FIRED, name
        assert control.baseline is not None and control.mutated is not None
        assert control.baseline.value < control.mutated.value


def test_the_wrapped_power_metric_separates_a_big_beam_from_a_folded_one(runs) -> None:
    """The distinction the first version of this metric conflated.

    At 500 um the beam is genuinely 34 um across in a 64 um window, so a
    correctly propagated field also has power near the edge. Measuring the outer
    quarter would therefore read 0.35 for the right answer and 0.35 for the wrong
    one. The metric is the edge band and the ANALYTIC field's own edge fraction
    is carried as the error bar, so the two are separable.
    """
    for suffix in (1, 2, 3):
        run = runs[f"B1-WAVE-ASM-VALIDITY-{suffix:02d}"]
        wrapped = _metric(run, "wrapped_power_fraction")
        assert wrapped.measured.uncertainty is not None
    inside = _metric(runs["B1-WAVE-ASM-VALIDITY-01"], "wrapped_power_fraction")
    outside = _metric(runs["B1-WAVE-ASM-VALIDITY-03"], "wrapped_power_fraction")
    assert inside.measured.value < 1e-9
    assert outside.measured.value > 1e-3
    # And the analytic field of the same size does NOT put that much at the edge,
    # which is what makes the measured value evidence of folding.
    assert outside.measured.uncertainty < outside.measured.value


def test_the_waist_choice_is_recorded_as_a_measured_tension(runs) -> None:
    """Because the obvious choices both fail, in opposite directions."""
    detail = next(
        d["detail"]
        for d in runs["B1-WAVE-ASM-VALIDITY-03"].record.diagnostics
        if d["code"] == "WHY_THE_WAIST_IS_2_5_UM"
    )
    assert "1.6e-5" in detail and "4%" in detail


# --------------------------------------------------------------------------- #
# WAVE-2 / WAVE-3: precision and device
# --------------------------------------------------------------------------- #


def test_the_complex64_cost_is_measured_against_the_independent_fp64_oracle() -> None:
    """CHE-107's WAVE-2, measured once rather than twice.

    B0-DTYPE-01 propagates the same field through the shipping Chromatix path at
    complex64 and through ``verification/asm_oracle.angular_spectrum_float64`` --
    an independent float64 implementation that shares no code with Chromatix, so
    what it measures is the cost of the REPRESENTATION rather than of the
    implementation. Measuring it with a second Chromatix call would put the
    truncation on both sides of the comparison.

    This test asserts that the measurement exists and is re-run, rather than
    duplicating it: the number lives on B0-DTYPE's gate disposition, and
    ``tests/test_b0_instances.py`` is what re-checks it.
    """
    from verification.families.b0_contract import B0_DTYPE
    from verification.claim_ledger import GateStatus

    disposition = B0_DTYPE.gate_disposition
    assert disposition.status is GateStatus.MET
    assert disposition.metric == "measured_precision_loss"
    assert disposition.observed is not None
    assert disposition.observed < 1e-4
    assert "asm_oracle" in disposition.note
    assert "eps32" in disposition.note
    assert any("test_b0_instances" in reference for reference in disposition.evidence)


def test_a_cuda_request_is_refused_rather_than_run_on_the_host() -> None:
    """The registry's note on this is emphatic and correct.

    A process-global JAX platform pin produces a successful host run for a caller
    who asked for CUDA, with no error raised -- so a requested device must never
    be reported as an actual one. Here the request is refused before anything
    runs, and the CPU row's placement is read off the output array.
    """
    observation = _driver().device_observation()
    rows = {row["requested_device"]: row for row in observation["rows"]}

    cpu = rows["cpu"]
    assert cpu["outcome"] == "executed"
    assert cpu["observed"]["device"] == "cpu"
    assert cpu["observed"]["dtype"] == "complex64", (
        "Chromatix is complex64-only and this is read off the array, not the request"
    )
    assert cpu["honoured"]

    cuda = rows["cuda"]
    assert cuda["outcome"] == "refused"
    assert "no silent fallback" in cuda["detail"]
    assert "gpu" in cuda["detail"] or "cuda" in cuda["detail"]


def test_no_run_claims_a_device_it_did_not_use(runs) -> None:
    """Mechanically, across every instance."""
    for instance_id, run in runs.items():
        placement = next(
            d["detail"] for d in run.record.diagnostics if d["code"] == "OBSERVED_PLACEMENT"
        )
        assert "'device': 'cpu'" in placement, f"{instance_id}: {placement}"


def test_the_output_pitch_is_read_and_not_assumed(runs) -> None:
    """A PSF's or a radius's axes come from the propagated field's OUTPUT pitch.

    Angular-spectrum propagation happens to preserve it, which makes the check
    cheap rather than vacuous: it is the same assertion for a method that does
    not, and the failure it guards is silent -- every distance the measurement
    reports would be rescaled by the ratio while the intensity map looked
    entirely ordinary.
    """
    for instance_id, run in runs.items():
        detail = next(
            d["detail"]
            for d in run.record.diagnostics
            if d["code"] == "OUTPUT_PITCH_IS_THE_INPUT_PITCH"
        )
        assert "relative drift" in detail, instance_id


# --------------------------------------------------------------------------- #
# Provenance and fingerprints
# --------------------------------------------------------------------------- #


def test_every_record_is_keyed_to_its_instance(runs) -> None:
    for instance_id, run in runs.items():
        assert run.record.instance_id == instance_id
        assert run.record.instance_fingerprint == run.instance.fingerprint
        assert run.result.provenance.fingerprint_matched


def test_every_reported_number_carries_an_uncertainty_and_a_basis(runs) -> None:
    from verification.result import UncertaintyBasis

    for instance_id, run in runs.items():
        for metric in run.result.physics_accuracy:
            assert metric.measured.uncertainty is not None, f"{instance_id}/{metric.metric}"
            assert metric.measured.uncertainty_basis is not UncertaintyBasis.NOT_ESTIMATED


@pytest.mark.parametrize("instance_id", ["B1-WAVE-GAUSS-01", "B1-WAVE-TALBOT-01"])
def test_the_scientific_fingerprint_reproduces_across_two_runs(instance_id: str) -> None:
    """Two independent executions of the same configuration, same fingerprint."""
    from verification.evidence import result_fingerprint

    first = _driver().run_instance(instance_id)
    second = _driver().run_instance(instance_id)
    assert first.record.run_id != second.record.run_id
    assert result_fingerprint(first.result) == result_fingerprint(second.result), (
        f"{instance_id} does not reproduce. Diagnose the nondeterminism rather than "
        "re-recording the new value."
    )


def test_the_family_gates_agree_with_what_was_measured(runs) -> None:
    from verification.claim_ledger import GateStatus

    for run in runs.values():
        disposition = run.family.gate_disposition
        if disposition.status not in (GateStatus.MET, GateStatus.NOT_MET):
            continue
        # The ASM-validity family's disposition is about its INSIDE instance;
        # the outside one is declared to fail and is the family's control.
        if run.instance.expected.get("side") == "outside":
            continue
        metric = _metric(run, disposition.metric)
        assert metric.met is (disposition.status is GateStatus.MET), (
            f"{run.family.family_id} claims {disposition.status.value} and "
            f"{metric.metric} measured {metric.measured.value:.3e} against "
            f"{metric.tolerance:.3e}"
        )
