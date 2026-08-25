"""The patch/global equivalence family, executed. M2.3's exit gate.

CHE-111. The composed patch route already had unit tests; what it did not have
was a *benchmark* -- an independent oracle with its padding named, a tolerance
with a basis, a convergence statement, and negative controls that fire through
the shipping switches. This asserts those, and asserts which instance decides
which claim.

The structural finding this family exists to record
--------------------------------------------------
The equivalence is **exact where the patch mode grid and the oracle mode grid
coincide, and only there**. At full aperture with ``pad_factor=1`` they coincide,
and the anchor reads 1.4e-12. On a sub-aperture decomposition each patch carries
its own pad-21 grid, which is not commensurate with the 15-px reconstruction
grid, so the ray sum is the NON-periodic propagated field while the ASM is the
periodic one -- and at z = 1.26 mm they differ by 0.84 with neither
implementation being wrong. Comparing at the DOE plane, where that distinction
does not arise, the enumerated sum reproduces the field to 1.7e-15.

That is the CHE-96 oracle-padding lesson in a third coordinate, and it is why
every score in this file names the pad it was measured against.

Which instance decides what, and why it has to be split that way
---------------------------------------------------------------
Two of the three negative controls are **exactly inert at full aperture**: the
coverage correction is 1 when one patch covers everything, and the launch phase
is 0 for a patch centred on the origin. That is how a real ``A_patch/A_draw``
inversion survived in this repository. So the anchor gates exactness and the
sub-aperture instances gate the controls, and the anchor reports the inert
mutations' measured values so the blindness is a number rather than a claim.

The drawn ladder is UNMET against the exactness gate on purpose: a Monte Carlo
estimate over drawn centres converges at about P^-1/2 and 225 draws is 0.19, not
1e-12. The gate belongs to the exact instances, and the tolerance basis says so.
"""

from __future__ import annotations

import importlib.util
import sys

import pytest

from core.paths import repository_root
from verification.result import NegativeControlOutcome, UncertaintyBasis
from verification.status import VerificationStatus

pytestmark = [pytest.mark.integration, pytest.mark.coupler]

#: 15 px of aperture plus the pad-5 patch's half-width on each side, squared.
ENUMERATED_POSITIONS = 361


def _driver():
    name = "b2_equiv_driver"
    if name in sys.modules:
        return sys.modules[name]
    path = repository_root() / "benchmarks" / "instances" / "b2_equiv.py"
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
def refusals():
    return _driver().declared_refusals()


def _metric(run, name):
    return next(m for m in run.result.physics_accuracy if m.metric == name)


def _control(run, name):
    return next(c for c in run.result.negative_control_results if c.control_id == name)


def _diagnostic(run, code):
    return next(d["detail"] for d in run.record.diagnostics if d["code"] == code)


def _enumerated(runs):
    return runs["B2-EQUIV-SUB-ENUMERATED"]


# --------------------------------------------------------------------------- #
# Coverage
# --------------------------------------------------------------------------- #


def test_every_declared_instance_was_executed(runs) -> None:
    assert set(runs) == set(_driver().declared_instance_ids())
    assert "B2-EQUIV-FULL-01" in runs
    assert "B2-EQUIV-SUB-ENUMERATED" in runs
    assert len(runs) == 6, sorted(runs)


def test_all_instances_belong_to_the_equivalence_family(runs) -> None:
    assert {run.family.family_id for run in runs.values()} == {"B2-EQUIV"}


def test_every_instance_ran_rather_than_refusing(runs) -> None:
    for instance_id, run in runs.items():
        assert run.result.status is VerificationStatus.OK, instance_id
        assert run.record.refusal is None, instance_id


# --------------------------------------------------------------------------- #
# The exactness anchor
# --------------------------------------------------------------------------- #


def test_the_full_aperture_limit_is_exact(runs) -> None:
    """One patch over the whole aperture IS the window, so the two routes agree
    to round-off -- and this is the family's exactness limit.

    Every propagating mode is enumerated, so there is no sampling error in it,
    and the comparison is against ``verification/asm_oracle`` in float64 rather
    than against another patch route. The error bar is the energy residual,
    which is what catches a transfer function that has stopped being
    unit-modulus.
    """
    run = runs["B2-EQUIV-FULL-01"]
    metric = _metric(run, "patch_vs_global_relative_l2")
    assert metric.met is True, f"{metric.measured.value:.3e}"
    assert metric.measured.value < 1e-11
    assert metric.measured.uncertainty is not None
    assert metric.measured.uncertainty_basis is UncertaintyBasis.FLOATING_POINT_FLOOR
    assert run.record.observed_parameters["patch_count"] == 1


def test_the_anchor_names_the_oracle_and_its_padding(runs) -> None:
    """A score is not defined until the oracle's grid is.

    The same route against a pad-200 oracle reads 8.8e-3 and against a pad-101
    oracle 0.33, and neither is an error in either implementation -- both are
    wraparound between two periods. So the pad is part of the measurement.
    """
    detail = _diagnostic(runs["B2-EQUIV-FULL-01"], "THE_ORACLE_AND_ITS_PAD")
    assert "angular_spectrum_float64" in detail
    assert "pad 0" in detail
    assert "pad-200" in detail and "pad-101" in detail


def test_the_clearance_exemption_is_recorded_as_a_property_not_a_relaxation(runs) -> None:
    """Padding the full-aperture patch moves its modes off the oracle's grid.

    So ``pad_factor=1`` here is not a loosened setting that happens to score
    better -- it is the condition under which the two mode sets are the same set.
    'Fixing' the exemption by padding would change the operator being compared.
    """
    detail = _diagnostic(runs["B2-EQUIV-FULL-01"], "THE_CLEARANCE_EXEMPTION_IS_PRESERVED")
    assert "pad_factor 1" in detail
    assert "not a relaxation" in detail
    assert "would change the mode grid" in detail


def test_the_anchor_measures_the_two_defects_it_cannot_see(runs) -> None:
    """Both mutations run at full aperture and both come back at the anchor's own
    value. A control that is inert reports green and proves nothing, and this is
    how the real coverage inversion survived -- so the blindness is measured
    here and gated on the sub-aperture instances."""
    run = runs["B2-EQUIV-FULL-01"]
    anchor = _metric(run, "patch_vs_global_relative_l2").measured.value
    detail = _diagnostic(run, "WHAT_THE_ANCHOR_CANNOT_SEE")
    assert "exactly 1" in detail

    for control_id in ("omit-coverage-correction", "launch-phase-per-patch"):
        control = _control(run, control_id)
        assert control.outcome is NegativeControlOutcome.NOT_RUN, control_id
        assert control.mutated is not None, control_id
        # Inert means the mutated arm lands on the anchor, not merely near it.
        assert control.mutated.value == pytest.approx(anchor, rel=1e-6), control_id
        assert "SUB-APERTURE" in control.note or "sub-aperture" in control.note


def test_coverage_is_exactly_one_at_full_aperture(runs) -> None:
    """Which is the mechanism behind the blindness above, stated as a number."""
    metric = _metric(runs["B2-EQUIV-FULL-01"], "coverage_corrected_power_ratio")
    assert metric.measured.value == pytest.approx(0.0, abs=1e-12)
    assert metric.measured.uncertainty_basis is UncertaintyBasis.EXACT
    assert "BY CONSTRUCTION" in metric.measured.note


# --------------------------------------------------------------------------- #
# The enumerated sub-aperture instance
# --------------------------------------------------------------------------- #


def test_the_enumerated_sub_aperture_estimator_is_unbiased(runs) -> None:
    """Enumerating every draw position, the sub-aperture sum IS the field.

    This is what separates 'the estimator is wrong' from 'the estimator has
    sampling error': the drawn ladder's 0.19 at 225 patches is Monte Carlo
    residual, and here -- with all 361 positions summed and no sampling left --
    the same coupler reproduces the reference to round-off. Without this
    instance the family could not tell those two apart, and the gate would be
    unmeetable by construction.
    """
    run = _enumerated(runs)
    metric = _metric(run, "patch_vs_global_relative_l2")
    assert metric.met is True, f"{metric.measured.value:.3e}"
    assert metric.measured.value < 1e-13
    assert run.record.observed_parameters["patch_count"] == ENUMERATED_POSITIONS


def test_coverage_is_not_one_on_a_sub_aperture_instance(runs) -> None:
    """Which is what makes the correction gateable here and not on the anchor."""
    run = _enumerated(runs)
    detail = _diagnostic(run, "COVERAGE_IS_NOT_ONE_HERE")
    assert "cannot see the correction; this instance can" in detail
    coverage = float(detail.split("coverage ")[1].split(" at")[0])
    assert coverage != pytest.approx(1.0, abs=1e-6), (
        "if coverage were 1 here the controls below would be inert again"
    )


def test_the_ray_count_is_the_declared_budget(runs) -> None:
    """patches x enumerated modes, an exact integer identity.

    A route that silently dropped an evanescent mode or double-emitted one would
    break this before it broke any field comparison.
    """
    run = _enumerated(runs)
    invariant = next(
        i for i in run.result.invariant_results if i.invariant_id == "OUTGOING_COUNT_IS_THE_BUDGET"
    )
    assert invariant.met
    assert invariant.measured.value == pytest.approx(0.0, abs=1e-15)
    assert invariant.measured.uncertainty_basis is UncertaintyBasis.EXACT


# --------------------------------------------------------------------------- #
# The controls, where they are not inert
# --------------------------------------------------------------------------- #


def test_all_three_controls_fire_on_the_enumerated_instance(runs) -> None:
    """The instance whose own sampling error is zero, so the separation each
    control shows is the mutation's and nothing else. On a drawn instance a
    control competes with the Monte Carlo residual and a fired verdict would be
    ambiguous."""
    run = _enumerated(runs)
    baseline = _metric(run, "patch_vs_global_relative_l2").measured.value
    for control_id in (
        "omit-coverage-correction",
        "launch-phase-per-patch",
        "grid-snapping-is-not-free",
    ):
        control = _control(run, control_id)
        assert control.outcome is NegativeControlOutcome.FIRED, control_id
        assert control.mutated.value > control.baseline.value, control_id
        assert control.mutated.value > 1e4 * baseline, (
            f"{control_id} separates by only {control.mutated.value / baseline:.1f}x"
        )


def test_the_controls_are_gated_against_the_correct_arm_not_an_absolute(runs) -> None:
    """Because the sub-aperture residual is itself a function of granularity, so
    an absolute threshold would be measuring the decomposition rather than the
    mutation."""
    control = _control(_enumerated(runs), "omit-coverage-correction")
    assert "against the CORRECT arm at the same patch count" in control.note


def test_the_controls_do_not_run_on_the_coarse_rungs(runs) -> None:
    """Stated rather than silently skipped: a control at 4 patches is dominated
    by the decomposition's own 1.6 residual, not by the mutation."""
    for instance_id, run in runs.items():
        if not instance_id.startswith("B2-EQUIV-SUB-"):
            continue
        if run.record.observed_parameters["patch_count"] == ENUMERATED_POSITIONS:
            continue
        for control_id in (
            "omit-coverage-correction",
            "launch-phase-per-patch",
            "grid-snapping-is-not-free",
        ):
            control = _control(run, control_id)
            assert control.outcome is NegativeControlOutcome.NOT_RUN, f"{instance_id}/{control_id}"
            assert "finest patch count" in control.note


def test_grid_snapping_is_paid_for_rather_than_free(runs) -> None:
    """Continuous centres inject a sub-sample linear phase no patch corrects, so
    the sweep plateaus instead of converging. The coupler snaps, and this is the
    measurement of what snapping buys."""
    control = _control(_enumerated(runs), "grid-snapping-is-not-free")
    assert control.outcome is NegativeControlOutcome.FIRED
    assert _enumerated(runs).record.observed_parameters["grid_snapping"] == "snapped"


# --------------------------------------------------------------------------- #
# Convergence
# --------------------------------------------------------------------------- #


def test_the_sub_aperture_sweep_converges(runs) -> None:
    """The residual FALLS with the number of drawn centres, over four rungs.

    No expected exponent is declared and none is gated: the coherent patch sum's
    rate in the number of DRAWN centres is a Monte Carlo rate over a finite
    population, and asserting one would be asserting a model this family has not
    established. That it falls monotonically is the convergence statement, and
    the fitted slope is reported with its standard error alongside.
    """
    convergence = _enumerated(runs).result.convergence
    rungs = sorted(zip(convergence.ladder, convergence.values, strict=True))
    assert len(rungs) >= 4, rungs
    values = [measurement.value for _, measurement in rungs]
    assert values == sorted(values, reverse=True), f"not monotone: {rungs}"
    assert values[0] / values[-1] > 5.0

    assert convergence.expected_exponent is None, (
        "declaring one here would be asserting a rate the family has not established"
    )
    fitted = convergence.fitted_exponent
    assert fitted is not None
    assert fitted.uncertainty is not None
    assert fitted.uncertainty_basis is UncertaintyBasis.FIT_STANDARD_ERROR
    # Consistent with a Monte Carlo rate, reported and not gated.
    assert -0.9 < fitted.value < -0.2, f"{fitted.value:+.4f}"


def test_the_convergence_ladder_records_the_oracle_pad(runs) -> None:
    detail = _diagnostic(_enumerated(runs), "CONVERGENCE_LADDER_WITH_ITS_ORACLE_PAD")
    assert "angular_spectrum_float64 at pad 0" in detail
    assert detail.count("patches ->") >= 4


def test_apodization_breaks_the_convergence(runs) -> None:
    """A taper below 1 removes field no other patch replaces, so the
    partition-of-unity argument behind the equivalence is exactly what it breaks
    -- which is why the shipping step carries no apodization and the registry
    warns about it."""
    detail = _diagnostic(_enumerated(runs), "APODIZATION_BREAKS_THE_CONVERGENCE")
    assert "raised-cosine taper" in detail
    assert "partition-of-unity" in detail
    apodized = [
        float(part.split("-> ")[1].split(".")[0] + "." + part.split("-> ")[1].split(".")[1][:12])
        for part in detail.split("; ")
        if "-> " in part
    ]
    correct = [m.value for m in _enumerated(runs).result.convergence.values]
    assert len(apodized) == len(correct) >= 4, f"{apodized} vs {correct}"

    # The claim is that the taper breaks the CONVERGENCE, so a worse floor is not
    # enough evidence for it -- a uniformly worse-but-still-converging ladder
    # would satisfy that and would not be this finding. What is asserted is that
    # the apodized ladder stops improving: its total improvement across the whole
    # sweep is a small fraction of the correct ladder's, so adding patches buys
    # essentially nothing.
    correct_gain = correct[0] / correct[-1]
    apodized_gain = apodized[0] / apodized[-1]
    assert correct_gain > 5.0, f"the correct ladder should converge: {correct}"
    assert apodized_gain < correct_gain / 3.0, (
        f"the apodized ladder improves by {apodized_gain:.2f}x against the correct "
        f"ladder's {correct_gain:.2f}x, which is not a broken convergence: "
        f"{apodized} vs {correct}"
    )
    # And it does not reach the correct ladder's floor at any rung.
    assert min(apodized) > min(correct), f"apodized {apodized} vs correct {correct}"


def test_the_drawn_ladder_reports_unmet_rather_than_being_exempted(runs) -> None:
    """The exactness gate belongs to the exact instances, and the drawn rungs are
    the family's own evidence for the sampling cost.

    Recording them MET would require widening 1e-12 to 0.2, which would make the
    gate unable to see anything. Recording them as exempt would hide four
    measured numbers. UNMET with the basis naming the split is what they are.
    """
    drawn = {
        instance_id: run
        for instance_id, run in runs.items()
        if instance_id.startswith("B2-EQUIV-SUB-")
        and run.record.observed_parameters["patch_count"] != ENUMERATED_POSITIONS
    }
    assert len(drawn) == 4, sorted(drawn)
    for instance_id, run in drawn.items():
        metric = _metric(run, "patch_vs_global_relative_l2")
        assert metric.met is False, f"{instance_id}: {metric.measured.value:.3e}"
        assert metric.measured.value > 1e-3

    tolerance = _enumerated(runs).family.tolerance_for("patch_vs_global_relative_l2")
    assert "exact instances" in tolerance.basis
    assert "P^-1/2" in tolerance.basis or "Monte Carlo" in tolerance.basis


# --------------------------------------------------------------------------- #
# Where the equivalence stops
# --------------------------------------------------------------------------- #


def test_where_the_equivalence_holds_and_where_it_stops_is_recorded(runs) -> None:
    """The most useful finding in this family, and it is a finding rather than a
    fix: 1.7e-15 at the DOE plane and 0.84 at z = 1.26 mm, with neither route
    wrong. A sub-aperture patch's pad-21 mode grid is not commensurate with the
    15-px reconstruction grid, so the ray sum is the non-periodic propagated
    field and the ASM is the periodic one."""
    detail = _diagnostic(_enumerated(runs), "WHERE_THE_EQUIVALENCE_HOLDS_AND_WHERE_IT_STOPS")
    assert "AT THE DOE PLANE" in detail
    assert "neither number is a defect in either implementation" in detail
    assert "not commensurate" in detail
    assert "CHE-96" in detail


def test_the_noise_limited_relation_is_characterization_and_says_so() -> None:
    """NCC(A,B) ~= sqrt(NCC(A,A') NCC(B,B')) has no oracle in it.

    It says what agreement is ACHIEVABLE given two noise levels, so matching it
    demonstrates that a disagreement is noise rather than a defect -- a weaker
    statement than being right, and one that must not be used as a gate.
    """
    relation = _driver().noise_limited_relation(0.0147, 0.0121, 0.0138)
    assert relation["predicted"] == pytest.approx((0.0121 * 0.0138) ** 0.5, rel=1e-12)
    assert 0.5 < relation["ratio"] < 2.0
    assert "CHARACTERIZATION and not a gate" in _driver().noise_limited_relation.__doc__


# --------------------------------------------------------------------------- #
# The refusals M2.3 asks to be gated
# --------------------------------------------------------------------------- #


def test_an_even_patch_size_is_refused_rather_than_rounded(refusals) -> None:
    """The paper's own sizes are 40, 50 and 100 -- all even -- so an even request
    is the LIKELY one, and rounding it would silently hand back a different
    operator than the caller asked for."""
    even = refusals["even_patch_px"]
    assert even["refused"] is True
    assert even["code"] == "SHAPE_MISMATCH"
    # A refusal that does not say what to do instead is a crash with a nicer name.
    assert "5" in even["detail"]


def test_a_conformal_substrate_is_refused_rather_than_approximated(refusals) -> None:
    conformal = refusals["conformal_substrate"]
    assert conformal["refused"] is True
    assert conformal["code"] == "MISSING_DECLARATION"


def test_the_pad_is_derived_and_reported_rather_than_taken(refusals) -> None:
    """``pad_factor`` is a preference: the step raises the pad until clearance,
    centring and oddness all hold, then reports what it actually used. A caller
    that assumed its factor was honoured would be reasoning about a grid that
    does not exist."""
    derived = refusals["derived_pad"]
    assert set(derived) == {"pad_factor=1", "pad_factor=2", "pad_factor=3"}
    for label, pad in derived.items():
        assert pad % 2 == 1, f"{label} gave an even pad {pad}"
        assert pad >= 5, f"{label} gave {pad}"


# --------------------------------------------------------------------------- #
# Cross-cutting
# --------------------------------------------------------------------------- #


def test_every_reported_number_carries_an_uncertainty_and_a_basis(runs) -> None:
    for instance_id, run in runs.items():
        for metric in run.result.physics_accuracy:
            basis = metric.measured.uncertainty_basis
            if basis is UncertaintyBasis.NOT_ESTIMATED:
                assert metric.measured.uncertainty is None, f"{instance_id}/{metric.metric}"
            else:
                assert metric.measured.uncertainty is not None, f"{instance_id}/{metric.metric}"


def test_every_record_is_keyed_to_its_instance(runs) -> None:
    for instance_id, run in runs.items():
        assert run.record.instance_id == instance_id
        assert run.record.instance_fingerprint == run.instance.fingerprint
        assert run.result.provenance.fingerprint_matched


def test_the_family_gate_agrees_with_the_instance_it_is_about(runs) -> None:
    """The anchor, not an average over the ladder: the drawn rungs are declared
    to exceed the gate and are the family's evidence for the sampling cost."""
    from verification.claim_ledger import GateStatus

    disposition = runs["B2-EQUIV-FULL-01"].family.gate_disposition
    assert disposition.status is GateStatus.MET
    metric = _metric(runs["B2-EQUIV-FULL-01"], disposition.metric)
    assert metric.met is True
    assert [ref for ref in disposition.evidence if "::" in ref]
    assert any("test_the_full_aperture_limit_is_exact" in ref for ref in disposition.evidence)


def test_the_ladder_is_not_silently_truncated(runs) -> None:
    """Five sub-aperture instances, four drawn plus the enumerated one, stated so
    a shortened ladder is a visible change rather than a quieter result."""
    counts = sorted(
        run.record.observed_parameters["patch_count"]
        for instance_id, run in runs.items()
        if instance_id.startswith("B2-EQUIV-SUB-")
    )
    assert counts == [4, 16, 64, 225, ENUMERATED_POSITIONS], counts


# --------------------------------------------------------------------------- #
# Route agreement on the paper's own systems
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def agreement():
    return _driver().route_agreement()


def test_the_noise_limited_relation_reproduces_the_demo3_ratio(agreement) -> None:
    """The instrument, run on the record's own inputs, must land on the record's
    own output. That is what makes it a reusable instrument rather than a stored
    number: if ``noise_limited_relation`` were changed, this diverges.

    demo3 has no independent oracle at all -- the paper states no conventional
    reference exists for the system -- so this relation is the entire instrument
    and there is nothing here that could be promoted to a gate.
    """
    recomputed = agreement["demo3"]["recomputed"]
    recorded = agreement["demo3"]["as_recorded"]
    assert recomputed["predicted"] == pytest.approx(recorded["predicted"], rel=1e-12)
    assert recomputed["measured"] == pytest.approx(recorded["measured"], rel=1e-12)
    assert recomputed["ratio"] == pytest.approx(recorded["ratio"], rel=1e-12)
    assert recomputed["ratio"] == pytest.approx(1.1367, abs=1e-3)


def test_the_demo3_relation_is_evaluated_rather_than_re_derived(agreement) -> None:
    """A 60M-ray CUDA run is not re-executed in a gate, and the record says so
    rather than letting the number read as freshly measured."""
    provenance = agreement["provenance"]
    assert "noise_limited_relation, freshly" in provenance["what_executed_here"]
    assert "NOT re-executed" in provenance["what_executed_here"]
    for key in ("demo2_record", "demo3_record"):
        assert (repository_root() / provenance[key]).is_file(), provenance[key]


def test_the_demo2_route_agreement_is_not_gated_and_says_why(agreement) -> None:
    """CHE-111 asks for it to be gated. It cannot be, and the reason is the
    substrate's own rule rather than a convenience: RW-F against RW-P is a
    CROSS_ROUTE oracle, which forces B4, and a B4 family may not carry a gating
    tolerance. Two of our own routes agreeing is not evidence either is right --
    if they share a convention error they agree perfectly.
    """
    demo2 = agreement["demo2"]
    assert "CROSS_ROUTE" in demo2["why_it_cannot_gate"]
    assert "may not carry a gating tolerance" in demo2["why_it_cannot_gate"]
    assert demo2["cross_route_ncc"] == pytest.approx(0.99941808, abs=1e-7)


def test_the_independent_oracle_number_is_what_carries_the_demo2_claim(agreement) -> None:
    """And the coincidence is the finding.

    The sub-aperture route against the *independent* float64 ASM reads 0.99941823
    -- numerically indistinguishable from the 0.99941808 cross-route number. So
    the two routes agree at exactly the level at which each independently matches
    an oracle, which is the evidence that their agreement is not hiding a shared
    convention error. That statement is only available because the independent
    comparison sits beside the cross-route one.
    """
    demo2 = agreement["demo2"]
    independent = demo2["sub_aperture_vs_independent_oracle_ncc"]
    cross = demo2["cross_route_ncc"]
    assert independent == pytest.approx(0.99941823, abs=1e-7)
    assert abs(independent - cross) < 1e-6, (
        f"independent {independent!r} vs cross-route {cross!r}: the whole point is "
        "that these coincide"
    )
    assert "angular_spectrum_float64" in demo2["oracle"]
    # And the full-aperture limit on the paper's own system is still round-off.
    assert demo2["full_aperture_vs_independent_oracle_relative_l2"] < 1e-11
