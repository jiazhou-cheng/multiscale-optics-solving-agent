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

One family's gate used to not close, and correcting it was a finding about the
oracle rather than about the solver. ``B1-RAY-LAGRANGE`` measured 1.3e-7 against a
1e-10 tolerance whose basis was a conservation law that holds paraxially, while
the measurement was of finite real rays -- a basis that cannot bound its metric on
either side of the threshold. It now gates on the DIFFERENTIAL symplectic
invariant at 1e-13, which is three decades tighter, and keeps the finite-ray
number at its original value and threshold as a non-gating characterization. See
``test_the_symplectic_invariant_closes_to_roundoff`` and
``test_the_finite_ray_drift_is_reported_and_does_not_gate``.

The device and precision half of RAY-4 is split by what the host can do. What a
CPU-only session can establish is here -- a CUDA request is refused and never
downgraded, and no comparison involving CUDA is reported as agreeing. The measured
CPU-vs-CUDA agreement is in ``tests/test_b1_ray_gpu.py`` behind the ``gpu``
marker, because a refusal path cannot close an agreement criterion.

What it costs the default gate
------------------------------
Measured, not estimated: this file plus ``tests/test_b1_families.py`` is 140
tests in 6.9 s on the CPU image, against a 372 s default suite. Every instance is
read from its committed record rather than re-traced, which is what keeps it
there; the traces themselves are the drivers in ``benchmarks/instances/`` and the
GPU file above, both on demand.
"""

from __future__ import annotations

import importlib.util
import itertools
import json
import math
import re
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
# The gate that closes once the invariant is the right one
# --------------------------------------------------------------------------- #


def _diagnostic(run, code: str) -> str:
    return next(d["detail"] for d in run.record.diagnostics if d["code"] == code)


def _number_after(text: str, marker: str) -> float:
    """The first number following ``marker`` in a diagnostic string.

    The diagnostics are prose with numbers in them, and a test that slices them
    by index is a test that breaks when somebody rewords a sentence. One regex,
    anchored on the phrase that carries the meaning.
    """
    match = re.search(
        re.escape(marker) + r"\s*(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)", text
    )
    assert match is not None, f"no number after {marker!r} in: {text}"
    return float(match.group(1))


def test_the_symplectic_invariant_closes_to_roundoff(runs) -> None:
    """The corrected oracle, and the number that says the solver was never wrong.

    With ``q`` the transverse position on a plane of constant ``z`` and
    ``p = n (L, M)`` the index-weighted direction cosines -- *not* the ray slopes
    ``M / N`` -- the ray map is the flow of a Hamiltonian system in ``z``, so its
    TANGENT map is symplectic at every point however nonlinear the map is, and
    ``omega(v_a, v_b) = sum_k (dp_k^a dq_k^b - dp_k^b dq_k^a)`` is conserved
    exactly for real rays at any aperture and any field.

    That is a strictly stronger claim than the paraxial invariant this family used
    to gate on, and the tolerance it is held to is three decades TIGHTER: 1e-13,
    derived from float64 round-off in the extrapolation weights, against the 1e-10
    that came before. Nothing was widened to get here.
    """
    run = runs["B1-RAY-LAGRANGE-01"]
    metric = _metric(run, "symplectic_invariant_relative_residual")
    assert metric.tolerance == 1e-13
    assert metric.tolerance_may_gate is True
    assert metric.met is True
    assert metric.measured.value < 1e-14, metric.measured.value
    # And it is not zero by construction: a metric that cannot be nonzero is not
    # measuring anything. It sits at a few float64 epsilons.
    assert metric.measured.value > 0.0
    assert metric.measured.value > np.finfo(np.float64).eps, (
        "a residual below one epsilon would mean the two omegas are bit-identical, "
        "which would suggest the image-plane state is not being read independently"
    )
    assert run.family.gate_disposition.metric == "symplectic_invariant_relative_residual"


def test_the_symplectic_residual_is_second_order_in_the_separation(runs) -> None:
    """Which is what proves the residual is removable rather than physical.

    Two symmetric secants approximate two tangent vectors to ``O(eps^2)``, so a
    residual that is finite-difference truncation error MUST go as ``eps^2`` and
    one that is a real conservation violation has no reason to. The exponent is
    therefore a prediction with a declared tolerance, not a fitted description --
    which is the difference between this and the previous formulation's field
    ladder, whose exponent came out near 2.5 and could only be reported.

    Six rungs, because four is the minimum for an exponent claim and Richardson to
    the ``eps^4`` column consumes two more.
    """
    convergence = runs["B1-RAY-LAGRANGE-01"].result.convergence
    assert convergence.dimension == "pupil_half_separation_m"
    assert len(convergence.ladder) >= 6, len(convergence.ladder)
    assert convergence.expected_exponent == 2.0
    assert convergence.converged is True
    exponent = convergence.fitted_exponent
    assert exponent is not None
    assert exponent.value == pytest.approx(2.0, abs=0.01), exponent.value
    assert exponent.uncertainty is not None and exponent.uncertainty < 1e-3
    assert float(exponent.note.split("r^2=")[1]) >= 0.9999, exponent.note
    # The fit is recorded to full precision as its own diagnostic, because an r^2
    # rounded to six places inside a note is not a number anybody can re-check.
    fit = _diagnostic(runs["B1-RAY-LAGRANGE-01"], "CONVERGENCE_FIT")
    assert _number_after(fit, "r^2 =") >= 0.9999, fit
    assert _number_after(fit, "secant separation") == pytest.approx(2.0, abs=0.01), fit
    # The ladder is monotone: every halving of the separation quarters the
    # residual. A non-monotone ladder would mean round-off had already been
    # reached and the fit was through noise.
    values = [m.value for m in convergence.values]
    assert values == sorted(values, reverse=True), values
    for coarse, fine in itertools.pairwise(values):
        assert fine / coarse == pytest.approx(0.25, rel=0.02), (coarse, fine)


def test_the_richardson_columns_land_on_the_float64_floor(runs) -> None:
    """The extrapolation's own evidence, not just its answer.

    A single extrapolated number proves nothing about whether the extrapolation
    was legitimate. The tableau does: the raw column falls as ``eps^2``, the next
    falls by 16x per halving because what is left is the ``O(eps^4)`` remainder,
    and the third stops falling because it has reached round-off. That progression
    is what makes ``1.2e-15`` an extrapolated limit rather than a coincidence.
    """
    run = runs["B1-RAY-LAGRANGE-01"]
    detail = _diagnostic(run, "RICHARDSON_TABLEAU")
    columns = [
        [float(value) for value in part.split(":", 1)[1].split()]
        for part in detail.split(" | ")
    ]
    assert len(columns) == 3, detail
    raw, second, third = columns
    assert len(raw) == 6 and len(second) == 5 and len(third) == 4
    # The eps^2 term is removed, so the remainder is smaller by orders and falls
    # by ~2^4 per halving.
    assert abs(second[-1]) < abs(raw[-1]) / 100
    for coarse, fine in itertools.pairwise(second):
        assert abs(fine) / abs(coarse) == pytest.approx(1 / 16, rel=0.15), (coarse, fine)
    # And the third column has stopped moving: it is at the floor.
    assert abs(third[-1]) < 1e-14, third
    assert abs(third[-1]) < abs(second[-1])


def test_both_non_symplectic_controls_fire(runs) -> None:
    """A gate a known-wrong twin can pass is not a gate.

    ``omega`` is bilinear, so a metric sensitive to one factor is not
    automatically sensitive to the other. Both halves are therefore broken
    separately -- the image-plane momenta scaled by ``1 + 1e-6`` with the
    positions untouched, and the reverse -- and each must read the scale factor
    back. What the metric IS blind to is the conjugate pair, and that is declared
    on the metric rather than tested away.
    """
    controls = {
        c.control_id: c for c in runs["B1-RAY-LAGRANGE-01"].result.negative_control_results
    }
    for control_id in ("non-symplectic-momentum-scale", "non-symplectic-position-scale"):
        control = controls[control_id]
        assert control.outcome is NegativeControlOutcome.FIRED, control
        assert control.target_metric == "symplectic_invariant_relative_residual"
        # The perturbation is 1e-6 and omega is linear in each factor, so the
        # residual reads the perturbation back rather than merely exceeding a
        # threshold. That is a calibration of the metric, not just a detection.
        assert control.mutated.value == pytest.approx(1e-6, rel=1e-3), control.mutated.value


def test_a_degenerate_tangent_pair_carries_no_invariant(runs) -> None:
    """The construction error that produced the original wrong conclusion.

    CHE-106 recorded that "the differential ratio between two rays of one fan
    converges to 1 + 7.1e-3 and does NOT approach 1 as their separation shrinks"
    and read that as proof no finite-ray evaluation could recover the invariant.
    Two rays of one pupil fan differ only in pupil coordinate, so both secants
    approach the same tangent direction; and on the object side of a collimated
    bundle every ray shares one launch direction, so ``dp`` is zero for both and
    ``omega_object`` is IDENTICALLY zero. The ratio was a 0/0.

    It is a declared control now, and it reads infinite rather than large --
    because "the reference is exactly zero" is a different fact from "the
    reference is small", and only the first says the pair spans nothing.
    """
    control = next(
        c
        for c in runs["B1-RAY-LAGRANGE-01"].result.negative_control_results
        if c.control_id == "degenerate-tangent-pair"
    )
    assert control.outcome is NegativeControlOutcome.FIRED
    assert math.isinf(control.mutated.value)
    assert "IDENTICALLY zero" in control.mutated.note


def test_the_finite_ray_drift_is_reported_and_does_not_gate(runs) -> None:
    """The retained metric, at its original value and its original threshold.

    This is the number the family used to fail on: 1.328629e-07 against 1e-10.
    Three things are asserted about it and they are the whole of the correction.

    It is *unchanged*: the same finite-real-ray paraxial form, the same
    measurement, and the threshold is still 1e-10. Nothing was widened.

    It no longer *gates*, and the reason is a derivation rather than a
    convenience: its declared basis was a conservation law, and the residual it
    measures is aberration. The paraxial Lagrange invariant is preserved by a
    LINEAR symplectic map; real refraction at a curved surface is symplectic and
    not linear, so this quantity is not conserved and no threshold derived from
    float64 round-off could ever have bounded it -- on either side of the number.

    And the conservation claim it was standing in for is not dropped: it has its
    own gate now, on the quantity that actually has the property.
    """
    run = runs["B1-RAY-LAGRANGE-01"]
    metric = _metric(run, "lagrange_invariant_relative_drift")
    assert metric.tolerance == 1e-10, "the tolerance must not have moved"
    assert metric.tolerance_may_gate is False
    assert metric.measured.value == pytest.approx(1.328629e-07, rel=1e-4)
    assert metric.met is False, (
        "the number is still outside the threshold and is still reported as such; "
        "what changed is that the threshold is no longer allowed to decide"
    )
    assert "aberration" in (metric.tolerance_basis or "")
    # The gate is decided by the other metric, and the result says so.
    assert run.result.status.value == "ok"
    detail = _diagnostic(run, "WHICH_INVARIANT_THE_GATE_IS_ABOUT")
    assert "LINEAR symplectic map" in detail
    assert "degenerate-tangent-pair" in detail


def test_the_paraxial_field_ladder_is_still_measured(runs) -> None:
    """The characterization the old formulation could support, kept.

    The paraxial invariant IS conserved paraxially, and the drift vanishing over
    five halvings of the field angle is the statement of that. It is no longer the
    family's convergence report -- that slot belongs to the separation ladder the
    gate is extrapolated over -- so it lives in a diagnostic, and it is still
    measured rather than deleted along with the gate it used to justify.
    """
    detail = _diagnostic(runs["B1-RAY-LAGRANGE-01"], "PARAXIAL_FIELD_LADDER")
    drifts = [
        float(part.split("drift=")[1]) for part in detail.split("; ") if "drift=" in part
    ]
    assert len(drifts) == 5, detail
    assert drifts == sorted(drifts, reverse=True), drifts
    # Vanishing faster than linearly in the field, which is the paraxial statement.
    assert drifts[-1] < drifts[0] / 100


def test_the_lagrange_control_still_fires(runs) -> None:
    """An unmet gate is not an excuse for an unexercised control, and neither is
    a met one.

    The control removes an index from the GLASS rather than from the arithmetic,
    so the mutation goes through the shipping trace. It stays pointed at the
    finite-ray metric on purpose: the symplectic gate is blind to the prescription
    by construction -- every valid trace of every valid system is symplectic --
    and that blind spot is measured rather than asserted.
    """
    run = runs["B1-RAY-LAGRANGE-01"]
    control = next(
        c
        for c in run.result.negative_control_results
        if c.control_id == "omit-index-at-refraction"
    )
    assert control.outcome is NegativeControlOutcome.FIRED
    assert control.target_metric == "lagrange_invariant_relative_drift"

    inert = _diagnostic(run, "PRESCRIPTION_MUTATION_IS_INERT")
    residual = _number_after(inert, "residual at")
    assert residual < 1e-13, (
        "the same mutation must leave the symplectic gate inside its tolerance: "
        "that is the declared blind spot, and if the mutation DID fire there the "
        "metric would be measuring the prescription and its blind_to would be wrong"
    )


def test_the_slope_momentum_substitution_is_recorded_as_inert(runs) -> None:
    """The old formulation's own error, measured instead of assumed to be the cause.

    Substituting the ray slope ``u = M / N`` for the canonical momentum
    ``p = n M`` is what the previous formulation did, and it is natural to expect
    that to be the defect. It is not, and the measurement says why: at an axial
    base ray the two agree to first order, so they give the same tangent map and
    the extrapolated residual is at round-off either way. The substitution shows
    up only at finite separation.

    So it is a diagnostic and not a declared control. A control whose verdict at
    the gated value is decided by which of two round-off numbers happens to be
    larger is a coin flip, and calling that coverage is worse than declaring
    nothing.
    """
    detail = _diagnostic(runs["B1-RAY-LAGRANGE-01"], "SLOPE_MOMENTUM_IS_INERT_IN_THE_LIMIT")
    assert "INERT at the" in detail
    finest = _number_after(detail, "reads")
    extrapolated = _number_after(detail, "extrapolates to")
    assert finest > 1e-9, "at finite separation the substitution must be visible"
    assert extrapolated < 1e-13, "and in the limit it must not be"


def test_the_reference_planes_are_planes(runs) -> None:
    """The precondition the whole construction rests on, checked rather than assumed.

    ``q`` is a transverse coordinate and ``z`` is the evolution parameter only if
    both ends of the map are planes of constant ``z``. The image plane's measured
    ``z`` spread across the fan is zero, and the launch plane's flatness is
    checked by the adapter itself -- ``_resolve_object_space_reference`` refuses to
    offer a launch direction at all unless the regenerated launch state is
    collimated, planar, finite and row-matched to the trace.
    """
    detail = _diagnostic(runs["B1-RAY-LAGRANGE-01"], "REFERENCE_PLANES")
    assert _number_after(detail, "z spread of") == 0.0, detail
    assert "constant z" in detail


# --------------------------------------------------------------------------- #
# RAY-4: device and precision, read off the arrays
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def agreement(matrix):
    return _driver().device_precision_agreement(matrix)


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
        # And the adapter's own comparison of resolved against actual agrees, so
        # two independent readings of the same fact are consistent.
        assert row["adapter_reported_mismatches"] == [], row
        assert row["adapter_observed_actual"]["device"] == observed["device"], row


def test_placement_is_read_from_the_live_arrays_and_not_the_persisted_copy(matrix) -> None:
    """The defect that hid a real CUDA run, as a standing assertion.

    ``np.savez`` needs host bytes, so the ``.npz`` beside a record is always numpy
    on the CPU whatever executed -- the adapter says as much itself under
    ``metadata['serialization']``. Reading placement from it reported a genuine
    ``cuda:0`` float64 trace as ``honoured_device: false`` on the GPU runner: a
    real CUDA execution recorded as a downgrade.

    Two things are asserted, and the first holds on every host. The dtype must
    survive persistence, because the adapter deliberately does not force float64
    on the way out and a float32 trace that arrived as float64 would mean the
    precision claim was fabricated somewhere. The namespace of the persisted copy
    must be numpy, always -- which is exactly why it cannot be the source of the
    device answer.
    """
    for row in matrix["executed"]:
        observed, persisted = row["observed"], row["persisted"]
        assert observed["dtype"] == persisted["dtype"], row
        assert persisted["namespace"] == "numpy", (
            "the persisted copy is host bytes by construction, so if it ever reads "
            "as torch or jax then the serialization boundary has moved and this "
            "section's reasoning needs redoing"
        )
        assert persisted["device"] == "cpu", row
        if observed["device"].startswith("cuda"):
            # The contrast, on a GPU runner: the two disagree, and the live one is
            # the true one.
            assert observed["namespace"] == "torch", row
            assert observed != persisted, row


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


def test_a_cuda_request_is_either_honoured_or_refused_and_never_downgraded(matrix) -> None:
    """The assertion the ticket asks for, in both environments.

    A requested CUDA run that executes on the CPU must not be reported as CUDA.
    There are exactly two admissible outcomes and this checks whichever one the
    host produced:

    * CPU-only session -- the adapter refuses before importing a solver, and the
      message says outright that there is deliberately no silent fallback. No row
      claims to have run on CUDA.
    * GPU session -- the row executed AND the arrays came back on a CUDA device,
      read off the live tensor. An executed CUDA row whose arrays are on the host
      is the failure this forbids, and it is forbidden in both directions: it
      cannot pass by refusing when a device is present either.

    The environment is decided by what the arrays say, not by a flag: ``matrix
    ['cuda_executed']`` is derived from the observed placement of the rows.
    """
    cuda_rows = [r for r in matrix["rows"] if r["requested"]["device"] == "cuda"]
    assert cuda_rows
    for row in cuda_rows:
        assert row["outcome"] in {"executed", "refused"}, row
        if row["outcome"] == "refused":
            assert "no silent fallback" in row["detail"] or "cannot drive cuda" in row["detail"]
            continue
        # Executed: it must actually be on the device, from the arrays.
        assert row["observed"]["device"].startswith("cuda"), (
            "a CUDA request that executed on the host is exactly the silent "
            f"downgrade this forbids: {row}"
        )
        assert row["honoured_device"] and row["honoured_dtype"], row
        assert row["applied_to_optiland"]["set_device"].startswith("cuda"), row
        assert row["record_device"] == "gpu", row

    executed_on_cuda = [
        r for r in matrix["executed"] if r["observed"]["device"].startswith("cuda")
    ]
    if not matrix["cuda_executed"]:
        assert not executed_on_cuda, (
            "no CUDA execution was observed, so a CUDA observation here would be "
            "fabricated"
        )
        # numpy/cuda is declared unsupported and must refuse whatever the host is.
        assert all(r["outcome"] == "refused" for r in cuda_rows)
    else:
        assert executed_on_cuda, "cuda_executed is set and no row observed a CUDA device"


def test_no_cuda_agreement_is_reported_without_a_cuda_execution(agreement, matrix) -> None:
    """A comparison is measured or it is unavailable. There is no third state.

    In a CPU-only session the three CUDA comparison classes report ``unavailable``
    and name the arm that did not execute together with the refusal code that
    stopped it. They are never reported as agreeing, and they are never silently
    dropped -- a missing row reads as coverage, which is the failure mode this is
    designed against.
    """
    cuda_classes = {
        "cpu_fp64_vs_cuda_fp64",
        "cpu_fp32_vs_cuda_fp32",
        "cuda_fp32_vs_cuda_fp64",
    }
    by_id = {c["comparison_id"]: c for c in agreement["comparisons"]}
    assert cuda_classes <= set(by_id), sorted(by_id)
    for comparison_id in cuda_classes:
        comparison = by_id[comparison_id]
        if matrix["cuda_executed"]:
            assert comparison["status"] == "measured", comparison
            assert "cuda" in (
                comparison["actual_reference"]["device"]
                + comparison["actual_compared"]["device"]
            ), comparison
        else:
            assert comparison["status"] == "unavailable", comparison
            assert comparison["unavailable_because"], comparison
            assert all(
                arm["outcome"] in {"refused", "declared_but_failed"}
                for arm in comparison["unavailable_because"]
            ), comparison
    assert set(agreement["cuda_comparisons_measured"]) <= cuda_classes
    if not matrix["cuda_executed"]:
        assert agreement["cuda_comparisons_measured"] == ()


def test_every_measured_comparison_meets_its_derived_tolerance(agreement) -> None:
    """The agreement thresholds, tested from the measured outputs.

    Every comparison that ran carries both arms' actual placement, the reference
    and compared value of each quantity at the element where they differ most, the
    absolute and normalized error, and a tolerance derived from the coarser of the
    two precisions -- ``64 * eps(dtype)``, the same multiple the adapter derives
    for its own direction-norm check. Nothing here is compared against a recorded
    constant: the verdict is recomputed from the arrays on every run.
    """
    assert agreement["measured"], "no comparison was measured at all"
    for comparison in agreement["measured"]:
        tolerance = comparison["tolerance"]["threshold"]
        assert tolerance > 0.0
        assert comparison["tolerance"]["basis"].strip()
        for name, quantity in comparison["quantities"].items():
            label = f"{comparison['comparison_id']}/{name}"
            assert quantity["normalized_error"] <= tolerance, (
                label,
                quantity["normalized_error"],
                tolerance,
            )
            # Both values, so the number can be re-derived rather than trusted.
            assert abs(quantity["reference_value"] - quantity["compared_value"]) == (
                pytest.approx(quantity["absolute_error"], rel=1e-12, abs=1e-300)
            ), label
        assert comparison["met"], comparison
    assert agreement["all_measured_met"]


def test_the_cross_dtype_tolerance_is_the_coarser_precisions_floor(agreement) -> None:
    """A float32 arm cannot be held to a float64 floor, and is not.

    The distinction is the whole reason the tolerance is derived per comparison
    rather than shared: a same-dtype cross-device comparison differs only in the
    order of operations and belongs at float64 round-off, while a cross-dtype one
    is bounded by the float32 arm's own floor. Collapsing the two would either
    report a precision cost as a correctness failure or let a real cross-device
    disagreement through.
    """
    eps32 = float(np.finfo(np.float32).eps)
    eps64 = float(np.finfo(np.float64).eps)
    for comparison in agreement["measured"]:
        dtypes = {
            comparison["actual_reference"]["dtype"],
            comparison["actual_compared"]["dtype"],
        }
        expected = 64.0 * (eps32 if any("float32" in d for d in dtypes) else eps64)
        assert comparison["tolerance"]["threshold"] == pytest.approx(expected), comparison
        if comparison["class"] == "same_dtype_cross_device":
            # Same arithmetic on two devices: the agreement should be far better
            # than the bound, and on float64 it is essentially exact.
            assert comparison["worst_normalized_error"] < expected / 100, comparison


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


def test_the_persisted_device_record_is_the_gpu_measurement(matrix) -> None:
    """The committed record, and what it is allowed to claim.

    The acceptance criterion asks for the agreement to be persisted rather than
    printed, so it lives at ``benchmarks/probes/records/optiland/
    b1_ray_device_precision.json`` and is stamped through
    ``core.provenance.record_provenance`` -- whose code half
    ``tests/test_provenance_fingerprint.py`` re-verifies against the tree on every
    default-gate run.

    What is asserted here is that the record cannot claim a CUDA measurement
    without the evidence for one *inside itself*: every comparison it reports as
    measured with a CUDA arm must carry that arm's actual device, read off the
    arrays. This is checked on the CPU host too, and that is the point -- a
    committed GPU record is exactly the artifact nobody re-runs, so the CPU gate
    is where its internal consistency has to be enforced.
    """
    path = (
        repository_root()
        / "benchmarks"
        / "probes"
        / "records"
        / "optiland"
        / "b1_ray_device_precision.json"
    )
    assert path.is_file(), f"{path} is missing; regenerate with the GPU command in its docstring"
    record = json.loads(path.read_text())
    assert "record_provenance" in record
    assert record["environment"]["cuda_executed"] is True, (
        "the committed record must be the GPU measurement. A CPU-only record here "
        "would leave the CPU-vs-CUDA criterion open, and saying so is the point of "
        "this assertion. Regenerate on a GPU runner: MOA_GPUS=device=6 ./run.sh "
        "--gpu python benchmarks/instances/b1_ray.py --device-matrix"
    )
    assert record["environment"]["torch"]["cuda_is_available"] is True
    assert record["environment"]["torch"]["device_name"]
    assert record["environment"]["optiland_version"]

    measured = [c for c in record["agreement"]["comparisons"] if c["status"] == "measured"]
    cuda_measured = [
        c
        for c in measured
        if "cuda" in c["actual_reference"]["device"] or "cuda" in c["actual_compared"]["device"]
    ]
    assert {c["comparison_id"] for c in cuda_measured} == {
        "cpu_fp64_vs_cuda_fp64",
        "cpu_fp32_vs_cuda_fp32",
        "cuda_fp32_vs_cuda_fp64",
    }
    for comparison in measured:
        assert comparison["met"], comparison["comparison_id"]
        for name, quantity in comparison["quantities"].items():
            assert quantity["normalized_error"] <= comparison["tolerance"]["threshold"], (
                comparison["comparison_id"],
                name,
            )
            assert "reference_value" in quantity and "compared_value" in quantity
    # Every executed row states what the arrays were, and the persisted copy is
    # recorded separately so the two cannot be confused by a later reader.
    for row in record["rows"]:
        if row["outcome"] != "executed":
            continue
        assert row["observed"]["dtype"] == row["persisted"]["dtype"]
        assert row["persisted"]["device"] == "cpu"
    cuda_rows = [
        r
        for r in record["rows"]
        if r["outcome"] == "executed" and r["observed"]["device"].startswith("cuda")
    ]
    assert cuda_rows, "cuda_executed is true and no row observed a CUDA device"
    for row in cuda_rows:
        assert row["observed"]["namespace"] == "torch"
        assert row["persisted"]["namespace"] == "numpy"
        assert row["applied_to_optiland"]["get_device"].startswith("cuda")


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
