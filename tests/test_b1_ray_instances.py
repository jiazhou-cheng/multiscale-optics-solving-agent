"""The eight canonical B1 ray instances, run against Optiland. M1.1's exit gate.

CHE-106 (M1.1). Before this, the ray families were declarations: three carried
numbers inherited from a retired task set and two carried none. This file runs
all eight canonical instances through the shipping adapter and asserts what each
gate is decided by, not merely that a number came out.

What the run establishes, and the three conventions it had to pin first
----------------------------------------------------------------------
Each of the three was found by measurement, and each one had a wrong reading that
produced a plausible number:

* a collimated on-axis ray does not move transversely before the first surface,
  so ``sin(i) = rho/R`` is exact from the exported launch coordinate -- which is
  what makes the Snell gate a machine-precision one rather than a paraxial one;
* the image plane refracts, so a system whose last surface carries glass applies
  a second refraction there. Reading the exported angle as an in-glass angle gave
  22.3 degrees where the geometry requires 12.9;
* a traced focal length is not a paraxial focal length. The reference singlet's
  innermost ring reads 1.1e-6 relative at 64 rings and 7.2e-5 at 8 -- a clean
  factor of four -- so the paraxial value is the ``h -> 0`` limit of a ladder and
  a single rung is an aberrated number.

One family's gate does not close, and that is reported rather than accommodated.
``B1-RAY-LAGRANGE`` measures 1.3e-7 against a 1e-10 tolerance whose basis is a
conservation law that holds paraxially, and the measurement is of real rays. See
``test_the_lagrange_gate_is_unmet_and_says_why``.
"""

from __future__ import annotations

import importlib.util
import math
import sys

import numpy as np
import pytest

from core.execution import RunStatus
from core.paths import repository_root
from verification.result import NegativeControlOutcome

pytestmark = [pytest.mark.integration, pytest.mark.optiland]


def _driver():
    name = "b1_ray_driver"
    if name in sys.modules:
        return sys.modules[name]
    path = repository_root() / "benchmarks" / "instances" / "b1_ray.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def runs():
    return _driver().run_all()


@pytest.fixture(scope="module")
def matrix():
    return _driver().device_precision_matrix()


# --------------------------------------------------------------------------- #
# Coverage
# --------------------------------------------------------------------------- #


def test_every_declared_instance_was_executed(runs) -> None:
    declared = set(_driver().declared_instance_ids())
    assert set(runs) == declared
    assert len(declared) == 8, sorted(declared)


def test_all_five_families_are_covered(runs) -> None:
    """Four incidence angles on Snell, one instance each elsewhere."""
    families = {run.family.family_id for run in runs.values()}
    assert families == {
        "B1-RAY-EFL",
        "B1-RAY-PLATE",
        "B1-RAY-SNELL",
        "B1-RAY-LAGRANGE",
        "B1-RAY-OFFAXIS-OPL",
    }


def test_no_gate_is_decided_by_another_optiland_output(runs) -> None:
    """PB7/CHE-58 finding F2, as a standing assertion.

    ``FFTPSF`` and ``HuygensPSF`` share one ``Wavefront``/OPD front end and are
    not two oracles, so Optiland's own outputs cannot decide Optiland's
    correctness. The schema refuses a ``CROSS_ROUTE`` oracle outside B4 anyway;
    what this checks is the weaker and more easily violated property, that every
    gating tolerance's basis is analytic or a conservation law.
    """
    from verification.families.schema import ToleranceBasis

    admissible = {
        ToleranceBasis.ANALYTIC_DERIVATION,
        ToleranceBasis.CONSERVATION_LAW,
        ToleranceBasis.NUMERICAL_PRECISION_FLOOR,
        ToleranceBasis.INDEPENDENT_DERIVATION,
    }
    for run in runs.values():
        for tolerance in run.family.tolerances:
            if not tolerance.may_gate:
                continue
            assert tolerance.basis_kind in admissible, (
                f"{run.family.family_id}/{tolerance.metric}: gating on "
                f"{tolerance.basis_kind}"
            )
            assert tolerance.rejects.strip(), (
                f"{run.family.family_id}/{tolerance.metric} names no wrong answer"
            )


# --------------------------------------------------------------------------- #
# The closed forms
# --------------------------------------------------------------------------- #


def _metric(run, name):
    return next(m for m in run.result.physics_accuracy if m.metric == name)


def test_the_thick_singlet_reproduces_both_closed_forms(runs) -> None:
    """R/(n-1) and EFL - t/n, from a trace, in the paraxial limit."""
    run = runs["B1-RAY-EFL-01"]
    assert run.record.status is RunStatus.SUCCEEDED
    efl = _metric(run, "efl_relative_error")
    bfl = _metric(run, "bfl_relative_error")
    for metric in (efl, bfl):
        assert metric.tolerance_may_gate
        assert metric.met is True, (
            f"{metric.metric} = {metric.measured.value:.3e} against "
            f"{metric.tolerance:.3e}"
        )
        assert metric.measured.uncertainty is not None


def test_the_two_focal_lengths_are_graded_separately(runs) -> None:
    """They differ by 2.64 mm on this prescription.

    An implementation that reports the EFL twice passes the EFL check and fails
    the BFL one, which is the whole reason the family declares two metrics
    instead of one.
    """
    run = runs["B1-RAY-EFL-01"]
    expected = run.instance.expected
    assert abs(expected["efl_mm"] - expected["bfl_mm"]) == pytest.approx(2.6371, abs=1e-3)

    control = next(
        c for c in run.result.negative_control_results if c.control_id == "thin-lens-bfl"
    )
    assert control.outcome is NegativeControlOutcome.FIRED
    assert control.mutated is not None
    assert control.mutated.value > 1e-6


def test_the_plate_shift_carries_its_sign(runs) -> None:
    """t(1 - 1/n), positive AWAY from the plate.

    The sign is half the claim: a magnitude-only comparison passes a system whose
    focus moved the wrong way, and the family's own tolerance basis names that as
    the wrong answer it rejects.
    """
    run = runs["B1-RAY-PLATE-01"]
    metric = _metric(run, "plate_focal_shift_signed_relative_error")
    assert metric.met is True, f"{metric.measured.value:.3e}"

    controls = {c.control_id: c for c in run.result.negative_control_results}
    assert controls["sign-flip"].outcome is NegativeControlOutcome.FIRED
    assert controls["t-over-n"].outcome is NegativeControlOutcome.FIRED
    # t/n is 6.25 mm where the answer is 3.75: a 67% error, not a rounding one.
    assert controls["t-over-n"].mutated.value > 0.5


def test_the_plate_measurement_is_a_difference_of_two_traces(runs) -> None:
    """So every common-mode property of the trace cancels.

    The lens, the sampling and the axial-crossing extraction are identical in
    both arms, which is what makes the residual the plate's and not the
    measurement's.
    """
    run = runs["B1-RAY-PLATE-01"]
    codes = {d["code"] for d in run.record.diagnostics}
    assert "PAIRED_CONTROL_ARM" in codes


@pytest.mark.parametrize(
    "instance_id",
    ["B1-RAY-SNELL-01", "B1-RAY-SNELL-02", "B1-RAY-SNELL-03", "B1-RAY-SNELL-04"],
)
def test_snell_holds_to_the_floating_point_floor(runs, instance_id: str) -> None:
    """Four incidence angles, exact on both sides.

    There is no approximation and no fitted constant anywhere in this
    comparison: the incidence angle comes from the exported launch coordinate,
    the refraction from Snell, and the exit face from Snell again. So the only
    admissible residual is float64 round-off, which is why the tolerance is
    1e-12 and why meeting it by four orders is the expected outcome rather than
    a surprise.
    """
    run = runs[instance_id]
    metric = _metric(run, "refraction_angle_absolute_error_rad")
    assert metric.met is True, f"{metric.measured.value:.3e}"
    assert metric.measured.value < 1e-14, (
        "an exact comparison that lands near its tolerance rather than far below "
        "it means something in the chain is not exact"
    )


def test_snell_is_measured_across_a_range_of_incidence_angles(runs) -> None:
    """A single angle cannot establish Snell's law.

    Small-angle substitution agrees with Snell to first order, so a paraxial-only
    suite would pass an implementation that had replaced sin with its argument.
    The instances span 0.075 to 0.467 rad for that reason.
    """
    angles = sorted(
        float(runs[f"B1-RAY-SNELL-{i:02d}"].instance.parameters["incidence_angle_rad"])
        for i in (1, 2, 3, 4)
    )
    assert angles[0] < 0.1 < angles[-1]
    assert angles[-1] > 0.45
    # And the substitution control fires at the steep end, which is the
    # demonstration that the range is what does the work.
    steep = runs["B1-RAY-SNELL-04"]
    control = next(
        c
        for c in steep.result.negative_control_results
        if c.control_id == "small-angle-substitution"
    )
    assert control.outcome is NegativeControlOutcome.FIRED
    assert control.mutated.value > 1e-3


def test_the_direction_cosines_stay_unit_norm(runs) -> None:
    """The declared invariant, measured on real traced directions."""
    run = runs["B1-RAY-SNELL-04"]
    invariant = next(
        i for i in run.result.invariant_results if i.invariant_id == "DIRECTION_COSINES_UNIT_NORM"
    )
    assert invariant.met
    assert invariant.measured.value < 1e-14


# --------------------------------------------------------------------------- #
# Convergence
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("instance_id", ["B1-RAY-EFL-01", "B1-RAY-PLATE-01"])
def test_convergence_is_a_fitted_exponent_over_four_rungs(runs, instance_id: str) -> None:
    """Not "the finest point looks small".

    Four rungs, a fitted exponent, and a standard error on the exponent. The
    expected value is a prediction rather than a fit target: spherical
    aberration is quadratic in the aperture and the innermost ring's height
    halves with each doubling of the ring count, so -2 is what the physics
    requires and a different exponent would be a finding.
    """
    convergence = runs[instance_id].result.convergence
    assert len(convergence.ladder) >= 4
    assert convergence.expected_exponent == -2.0
    exponent = convergence.fitted_exponent
    assert exponent is not None
    assert exponent.value == pytest.approx(-2.0, abs=0.05), (
        f"{instance_id}: fitted {exponent.value:.4f}"
    )
    assert convergence.converged is True
    # Every rung carries its own measured value, so the fit can be re-checked.
    assert len(convergence.values) == len(convergence.ladder)
    assert all(value.value > 0.0 for value in convergence.values)


def test_the_gated_value_is_the_limit_and_not_the_finest_rung(runs) -> None:
    """The extrapolation is what makes the 1e-6 gate meetable AND meaningful.

    The finest rung is 1.1e-6 -- inside the gate by a hair, and only because the
    ladder is deep. The Richardson limit is 2.8e-12, four orders better, and the
    difference between the two is the aberration the closed form does not
    describe. Gating the rung would have made the tolerance a statement about the
    ray count.
    """
    run = runs["B1-RAY-EFL-01"]
    gated = _metric(run, "efl_relative_error").measured.value
    finest_rung = run.result.convergence.values[-1].value
    assert gated < finest_rung / 100.0, (
        f"gated {gated:.3e} against the finest rung {finest_rung:.3e}: the "
        "extrapolation is not buying anything, which means the ladder is not "
        "converging as the fit claims"
    )


# --------------------------------------------------------------------------- #
# The off-axis reference term: M1's highest-value measurement
# --------------------------------------------------------------------------- #


def test_the_off_axis_tilt_is_recovered_and_gated(runs) -> None:
    run = runs["B1-RAY-OFFAXIS-OPL-01"]
    metric = _metric(run, "launch_tilt_fraction_recovered")
    assert metric.met is True, f"{metric.measured.value:.3e}"
    assert run.instance.parameters["field_angle_rad"] > 0.0


def test_omitting_the_object_space_term_fails_through_the_shipping_adapter(runs) -> None:
    """One term removed from the production path, not a parallel copy.

    ``HandoffPerturbation(reference_incoming_wavefront=False)`` is the adapter's
    own switch for exactly this, so the broken arm runs the same code with
    ``n_object * (d0 . r_launch)`` left out and nothing else changed.
    """
    run = runs["B1-RAY-OFFAXIS-OPL-01"]
    control = next(
        c
        for c in run.result.negative_control_results
        if c.control_id == "omit-object-space-term"
    )
    assert control.outcome is NegativeControlOutcome.FIRED
    assert control.baseline is not None and control.mutated is not None
    # The unperturbed arm passes and the perturbed one does not, which is what a
    # control has to show. CHE-41 measured 0.13% of the tilt surviving; the
    # separation here is three orders.
    assert control.baseline.value < 1e-3 < control.mutated.value
    assert control.mutated.value > 0.99, (
        "the omission should lose essentially all of the tilt, not some of it"
    )


def test_the_omission_is_invisible_on_axis_which_is_why_it_survived(runs) -> None:
    """The blind spot, demonstrated rather than described.

    On axis the omitted term is a constant across the pupil and the chief-ray
    subtraction removes it exactly, so the same omission changes nothing. That is
    why the defect survived CHE-30, CHE-32 and CHE-33 -- every one of them looked
    on axis -- and it is why the family declares this as a blindness of the
    metric rather than as a negative control. A control is a deliberately WRONG
    twin; on axis the omission is not wrong, it is unobservable.
    """
    run = runs["B1-RAY-OFFAXIS-OPL-01"]
    detail = next(
        d["detail"] for d in run.record.diagnostics if d["code"] == "ON_AXIS_IS_BLIND_TO_IT"
    )
    assert "term span 0.000000000" in detail

    metric_spec = next(
        m for m in run.family.metrics if m.name == "launch_tilt_fraction_recovered"
    )
    assert any("ON-AXIS" in blind for blind in metric_spec.blind_to)
    # And it is NOT declared as a negative control, which the previous version of
    # the family did with expectation=MUST_FAIL against a description saying it
    # must not fire.
    assert "on-axis-cannot-detect-it" not in {
        c.control_id for c in run.family.negative_controls
    }


def test_the_launch_and_exit_pupils_are_not_interchangeable(runs) -> None:
    """The trap this measurement fell into first.

    The omitted term is a function of the LAUNCH coordinate, so its
    peak-to-valley span is over the launch extent -- 0.300 mm on this system --
    while the declared OPL is expressed over the exit pupil, 0.462 mm. Regressing
    one against the other reads 0.6556 and looks like a 34% shortfall in a term
    that is exact.
    """
    run = runs["B1-RAY-OFFAXIS-OPL-01"]
    detail = next(
        d["detail"]
        for d in run.record.diagnostics
        if d["code"] == "DECLARED_VERSUS_REALIZED_PUPIL"
    )
    assert "0.6556" in detail
    realized = float(run.record.observed_parameters["pupil_diameter_m"])
    declared = float(run.instance.parameters["pupil_diameter_m"])
    assert realized == pytest.approx(declared, rel=1e-3)


# --------------------------------------------------------------------------- #
# The gate that does not close
# --------------------------------------------------------------------------- #


def test_the_lagrange_gate_is_unmet_and_says_why(runs) -> None:
    """A NOT_MET gate with a reason, which is a result and not an absence.

    The Lagrange invariant's two-ray bilinear form is preserved by a LINEAR
    symplectic map. Ray refraction at a curved surface is symplectic and not
    linear, so only the differential form is exactly conserved and any
    finite-real-ray evaluation carries an aberration residual. Measured
    directly while authoring this: the differential ratio between two rays of one
    fan converges to 1 + 7.1e-3 at a 5-degree field and does NOT approach 1 as
    the separation shrinks, which is the signature of a finite-form residual
    rather than a numerical one.

    So the tolerance is not widened. What is recorded is the measured drift, the
    ladder it sits on, and the statement that the tolerance's basis needs
    re-deriving against the aberration it cannot see.
    """
    run = runs["B1-RAY-LAGRANGE-01"]
    metric = _metric(run, "lagrange_invariant_relative_drift")
    assert metric.tolerance == 1e-10, "the tolerance must not have moved"
    assert metric.met is False
    assert metric.measured.value < 1e-5, (
        "the drift should still be small: this is an aberration residual, not a "
        "broken invariant"
    )
    detail = next(
        d["detail"] for d in run.record.diagnostics if d["code"] == "WHY_THE_GATE_CANNOT_CLOSE"
    )
    assert "LINEAR symplectic map" in detail
    assert "left where it is" in detail


def test_the_lagrange_drift_vanishes_with_the_field(runs) -> None:
    """Which is the conservation statement the family can actually support.

    The invariant IS conserved paraxially, and the ladder shows it: five halvings
    of the field angle and the drift falls monotonically. No expected exponent is
    declared because the measured one is near 2.5 rather than a clean integer,
    and asserting 2 would be asserting the wrong model.
    """
    convergence = runs["B1-RAY-LAGRANGE-01"].result.convergence
    assert len(convergence.ladder) >= 5
    assert convergence.expected_exponent is None
    assert convergence.converged is None
    exponent = convergence.fitted_exponent
    assert exponent is not None
    assert exponent.value > 1.5, f"the drift is not vanishing: exponent {exponent.value}"
    assert exponent.uncertainty is not None


def test_the_lagrange_control_still_fires(runs) -> None:
    """An unmet gate is not an excuse for an unexercised control.

    The control removes an index from the GLASS rather than from the arithmetic,
    so the mutation goes through the shipping trace, and the invariant follows
    the system it was given.
    """
    control = next(
        c
        for c in runs["B1-RAY-LAGRANGE-01"].result.negative_control_results
        if c.control_id == "omit-index-at-refraction"
    )
    assert control.outcome is NegativeControlOutcome.FIRED


# --------------------------------------------------------------------------- #
# RAY-4: device and precision, read off the arrays
# --------------------------------------------------------------------------- #


def test_the_matrix_covers_every_declared_combination(matrix) -> None:
    requested = {
        (row["requested"]["backend"], row["requested"]["device"], row["requested"]["dtype"])
        for row in matrix["rows"]
    }
    assert ("numpy", "cpu", "float64") in requested
    assert ("torch", "cpu", "float32") in requested
    assert ("torch", "cuda", "float64") in requested
    assert ("numpy", "cuda", "float64") in requested


def test_every_executed_row_reports_what_the_arrays_actually_are(matrix) -> None:
    """Observed, never requested. The one rule this whole section exists for."""
    assert matrix["executed"], "no combination executed at all"
    for row in matrix["executed"]:
        observed = row["observed"]
        assert observed["namespace"] in {"numpy", "jax", "torch"}
        assert observed["device"].startswith("cpu") or observed["device"].startswith("cuda")
        assert observed["dtype"]
        assert row["honoured_device"], row
        assert row["honoured_dtype"], row


def test_float32_and_float64_agree_to_the_precision_available(matrix) -> None:
    """The measured cost of the lower precision, not an assumption that there is none."""
    by_request = {
        (r["requested"]["backend"], r["requested"]["dtype"]): r for r in matrix["executed"]
    }
    fp64 = by_request[("torch", "float64")]["innermost_efl_relative_error"]
    fp32 = by_request[("torch", "float32")]["innermost_efl_relative_error"]
    # Both are dominated by the SAME aberration at this ring count, so the
    # difference between them is the precision cost and it is small.
    assert abs(fp32 - fp64) < 1e-5, (fp32, fp64)
    assert fp32 > 0.0 and fp64 > 0.0


def test_a_cuda_request_this_container_cannot_honour_is_refused_not_downgraded(matrix) -> None:
    """The assertion the ticket asks for in as many words.

    A requested CUDA run that executes on the CPU must not be reported as CUDA.
    Here it is not reported at all: the adapter refuses before importing a
    solver, and the message says outright that there is deliberately no silent
    fallback.
    """
    cuda_rows = [r for r in matrix["rows"] if r["requested"]["device"] == "cuda"]
    assert cuda_rows
    for row in cuda_rows:
        assert row["outcome"] == "refused", row
        assert "no silent fallback" in row["detail"] or "cannot drive cuda" in row["detail"]
    # And nothing claimed to run on CUDA.
    assert not [
        r for r in matrix["executed"] if r["observed"]["device"].startswith("cuda")
    ], "this is a CPU-only session; a CUDA observation here would be fabricated"


def test_the_numpy_float32_over_claim_is_now_a_named_refusal(matrix) -> None:
    """A capability the table declared and the implementation could not deliver.

    Found by running the matrix: ``set_precision('float32')`` changes what
    optiland's own array constructor builds, and ``Optic.trace`` on the numpy
    backend still returns float64 -- so the trace carried float32-scale error
    (9.23e-11 on the reference singlet's direction cosines) inside float64
    arrays and died on the artifact boundary's unit-norm check with a bare
    ``ValueError``.

    Two defects, and the fix goes to the request rather than to the boundary: a
    float64 artifact whose norms are off by 9e-11 IS malformed, so the check is
    right and the request is what should not have been accepted. Refused eagerly
    now, with the torch backend named as the path that honours float32 end to
    end.
    """
    row = next(
        r
        for r in matrix["rows"]
        if r["requested"] == {"backend": "numpy", "device": "cpu", "dtype": "float32"}
    )
    assert row["outcome"] == "refused"
    assert "float64" in row["detail"]
    assert "torch" in row["detail"]


def test_no_declared_combination_fails_unstructured(matrix) -> None:
    """The class this section closed.

    A combination is either executed, or refused with a reason. A third state --
    declared, attempted, and dead on an internal check -- is the one an agent
    cannot recover from, and there are none left.
    """
    assert matrix["declared_but_failed"] == (), [
        (r["requested"], r["error_type"], r["error_message"])
        for r in matrix["declared_but_failed"]
    ]


# --------------------------------------------------------------------------- #
# RAY-5: a structured refusal, with no fabricated numbers
# --------------------------------------------------------------------------- #


def test_an_unsupported_configuration_refuses_with_a_reason_and_no_numbers() -> None:
    outcome = _driver().unsupported_configuration()
    assert outcome["refused"]
    assert "PRESCRIPTION_NAME_UNKNOWN" in outcome["detail"]
    # The supported set, so a caller can choose again without guessing.
    assert "ReverseTelephoto" in outcome["detail"]
    assert "M3SingletRef" in outcome["detail"]
    # And nothing came back.
    assert outcome["outputs"] == {}


# --------------------------------------------------------------------------- #
# RAY-6: the scientific fingerprint, across two independent runs
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("instance_id", ["B1-RAY-EFL-01", "B1-RAY-SNELL-02"])
def test_the_scientific_fingerprint_reproduces_across_two_runs(instance_id: str) -> None:
    """Two independent executions of the same configuration, same fingerprint.

    Independent means separately constructed: two calls to ``run_instance``, each
    building its own prescription, tracing its own rays into its own temporary
    directory, and producing its own run id. The projection strips run ids and
    timings and keeps every measured value and every error bar, so this is a
    statement about the physics and not about the bookkeeping.
    """
    from verification.evidence import result_fingerprint

    first = _driver().run_instance(instance_id)
    second = _driver().run_instance(instance_id)
    assert first.record.run_id != second.record.run_id, "not two independent runs"
    assert result_fingerprint(first.result) == result_fingerprint(second.result), (
        f"{instance_id} does not reproduce. Diagnose the nondeterminism rather "
        "than re-recording the new value."
    )
    assert first.instance.fingerprint == second.instance.fingerprint


def test_a_changed_measurement_changes_the_fingerprint(runs) -> None:
    """The other half, so the fingerprint is not trivially constant.

    Stripping too much is the silent failure: a projection that ignored the
    measured values would call two different computations identical and report
    reproducibility that means nothing.
    """
    from verification.evidence import result_fingerprint

    run = runs["B1-RAY-EFL-01"]
    baseline = result_fingerprint(run.result)
    perturbed = run.result.model_copy(
        update={
            "physics_accuracy": [
                metric.model_copy(
                    update={
                        "measured": metric.measured.model_copy(
                            update={"value": metric.measured.value * 1.01}
                        )
                    }
                )
                for metric in run.result.physics_accuracy
            ]
        }
    )
    assert result_fingerprint(perturbed) != baseline


# --------------------------------------------------------------------------- #
# Provenance
# --------------------------------------------------------------------------- #


def test_every_record_is_keyed_to_its_instance(runs) -> None:
    for instance_id, run in runs.items():
        assert run.record.instance_id == instance_id
        assert run.record.instance_fingerprint == run.instance.fingerprint
        assert run.result.provenance.fingerprint_matched


def test_every_reported_number_carries_an_uncertainty_and_a_basis(runs) -> None:
    """A value with no error bar is a schema violation, not a pass."""
    from verification.result import UncertaintyBasis

    for instance_id, run in runs.items():
        for metric in run.result.physics_accuracy:
            assert metric.measured.uncertainty_basis is not UncertaintyBasis.NOT_ESTIMATED, (
                f"{instance_id}/{metric.metric} reports no uncertainty basis"
            )
            assert metric.measured.uncertainty is not None
        exponent = run.result.convergence.fitted_exponent
        if exponent is not None:
            assert exponent.uncertainty_basis is UncertaintyBasis.FIT_STANDARD_ERROR


def test_the_realized_parameters_are_recorded_not_assumed(runs) -> None:
    """The verifier re-evaluates validity against what ran, so it needs what ran."""
    for instance_id, run in runs.items():
        assert run.record.observed_parameters, instance_id
        assert run.result.validity.observed is not None


def test_the_family_gates_agree_with_what_was_measured(runs) -> None:
    """The dispositions on the families and the numbers in these runs are one thing.

    A gate recorded as MET whose instance measures worse than its tolerance is
    the failure mode a hand-maintained disposition always eventually has.
    """
    from verification.claim_ledger import GateStatus

    for run in runs.values():
        disposition = run.family.gate_disposition
        if disposition.status is GateStatus.MET:
            metric = _metric(run, disposition.metric)
            assert metric.met is True, (
                f"{run.family.family_id} claims MET and {metric.metric} measures "
                f"{metric.measured.value:.3e} against {metric.tolerance:.3e}"
            )
        elif disposition.status is GateStatus.NOT_MET:
            metric = _metric(run, disposition.metric)
            assert metric.met is False, (
                f"{run.family.family_id} claims NOT_MET and the gate now passes; "
                "update the disposition rather than leaving it stale"
            )


def test_the_measurement_is_not_secretly_a_paraxial_solver_call(runs) -> None:
    """Every number here comes from traced rays.

    Optiland exposes ``paraxial.f2()``, and using it would have made the EFL
    family a test of Optiland's paraxial module against a closed form rather than
    a test of its trace. The distinction is the whole difference between coverage
    and a correctness gate, so it is asserted structurally.
    """
    source = (
        repository_root() / "benchmarks" / "instances" / "b1_ray.py"
    ).read_text(encoding="utf-8")
    assert "paraxial" not in source.replace("paraxial approximation", "").replace(
        "paraxial", "", 0
    ) or "paraxial.f2" not in source
    assert "paraxial.f2" not in source
    assert "paraxial.EPD" not in source
    assert "launch_x_m" in source and "launch_y_m" in source


def test_the_ladder_is_not_silently_truncated() -> None:
    """Four rungs, stated, so a shortened ladder is a visible change."""
    assert _driver().RINGS_LADDER == (8, 16, 32, 64)
    assert len(_driver().LAGRANGE_FIELDS_DEG) == 5
    assert all(
        b == pytest.approx(a / 2.0)
        for a, b in zip(
            _driver().LAGRANGE_FIELDS_DEG, _driver().LAGRANGE_FIELDS_DEG[1:], strict=False
        )
    )
    assert not math.isnan(np.float64(0.0))
