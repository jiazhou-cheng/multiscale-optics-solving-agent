"""Composed families, and the line between deciding and reporting.

CHE-116 (M4.1) and CHE-134 (M4.3). The two milestones are tested together
because the interesting property is the boundary between them: a composed case
with an oracle and a composed case without one are not two points on one scale,
and the schema is what stops the second borrowing the first's authority.

Three things get the most attention here:

* ``B3-PSF-SINGLET`` carries ``NOT_MET`` with the measured 2.21e-3 against a
  frozen 1.0e-3, and every tolerance basis migrated verbatim from
  ``benchmarks/physics/L2-PSF-01/tolerances.yaml``. Those strings are the
  hardest thing in the repository to reconstruct; a test compares them
  character for character against the file.
* ``B4-DEMO3`` cannot be given a gating tolerance, and the test proves the
  schema refuses one rather than trusting the label.
* ``B3-DUALROUTE`` could not be authored as M4.1 described it. Every PSF route
  on the Cooke triplet is our own code, so there is no independent leg; the
  family gates on an intermediate invariant instead, and the route comparison
  moved to a B4 family that cannot gate. The tests below pin both halves of
  that split so it cannot quietly be re-merged.

Two things CHE-116 found while closing M4.1, both pinned below
--------------------------------------------------------------
* **The energy-accounting intermediate all three B3 families gate on is not
  measurable from the shipping surface.** Not one family's oversight -- one
  property of the ray-to-wave boundary. The gates stay declared, unmeasured and
  un-widened, and ``benchmarks/probes/b3_energy_accounting.py`` measures the
  numbers that say so.
* **``B3-DEMO2``'s oracle was mis-declared.** It named the paper's published
  figure as the decider; every committed number is against our own float64 ASM.
  The value did not move, the attribution did, and the test that used to assert
  the wrong thing now asserts the right one and quotes what it used to say.
"""

from __future__ import annotations

import json

import pytest
import yaml

from core.paths import repository_root
from verification.families import (
    BenchmarkCategory,
    BenchmarkFamily,
    ParameterKind,
    Tolerance,
    ToleranceBasis,
    families_for_category,
)
from verification.families.b3_composed import B3_DEMO2, B3_DUALROUTE, B3_PSF_SINGLET
from verification.families.b4_characterization import (
    B4_COST,
    B4_DEMO3,
    B4_DUALROUTE_AGREEMENT,
    DEMO3_WHY_NOT,
)
from verification.families.schema import GateStatus, Oracle, OracleIndependence

ROOT = repository_root()
B3 = (B3_PSF_SINGLET, B3_DEMO2, B3_DUALROUTE)
B4 = (B4_DEMO3, B4_COST, B4_DUALROUTE_AGREEMENT)


# --------------------------------------------------------------------------- #
# B3: something independent decides
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("family", B3, ids=lambda f: f.family_id)
def test_every_b3_family_names_a_decider_that_can_decide(family: BenchmarkFamily) -> None:
    """Admissible: a closed form, a genuinely independent route, a justified
    equivalence, or an intermediate invariant. Not admissible: our own code
    answering a second time."""
    assert family.oracle.may_decide_correctness, (
        f"{family.family_id}: oracle {family.oracle.kind.value} / "
        f"{family.oracle.independence.value} cannot decide anything, so this is a B4 "
        "family wearing a B3 label"
    )
    assert family.gating_tolerances or family.invariants, (
        f"{family.family_id}: nothing here can come out either way"
    )


@pytest.mark.parametrize("family", B3, ids=lambda f: f.family_id)
def test_every_b3_family_checks_an_intermediate_not_only_the_end(
    family: BenchmarkFamily,
) -> None:
    """A correct final image can hide an incorrect intermediate convention."""
    has_intermediate = bool(family.invariants) or any(
        "power" in m.name or "handoff" in m.name for m in family.metrics
    )
    assert has_intermediate, (
        f"{family.family_id}: only the final result is checked, so a compensating "
        "pair of intermediate errors would pass"
    )


def test_the_singlet_gate_is_carried_unmet_and_unwidened() -> None:
    """The frozen 1.0e-3, against the measured 2.21e-3. No tolerance is widened.

    The new schema can express an unmet gate as a first-class state, which is
    precisely why carrying it forward honestly is now possible instead of being
    a footnote in a prose field.
    """
    tolerance = B3_PSF_SINGLET.tolerance_for("fft_oracle_intensity_relative_l2")
    assert tolerance is not None
    assert tolerance.threshold == 1.0e-3
    assert tolerance.may_gate

    disposition = B3_PSF_SINGLET.gate_disposition
    assert disposition is not None
    assert disposition.status is GateStatus.NOT_MET
    assert disposition.observed == pytest.approx(2.2072391812867093e-3)
    assert disposition.observed > tolerance.threshold


def test_the_singlet_tolerance_bases_migrated_verbatim() -> None:
    """Character for character against the file they came from.

    ``*_basis`` strings are the hardest thing in the repository to reconstruct.
    A migration that "tidied" one would lose the reasoning that makes the
    threshold defensible, and this is the only check that would notice.
    """
    committed = yaml.safe_load(
        (ROOT / "benchmarks/physics/L2-PSF-01/tolerances.yaml").read_text(encoding="utf-8")
    )

    def normalized(text: str) -> str:
        return " ".join(text.split())

    tolerance = B3_PSF_SINGLET.tolerance_for("fft_oracle_intensity_relative_l2")
    assert tolerance is not None
    assert normalized(tolerance.basis) == normalized(
        committed["fft_oracle_intensity_relative_l2_basis"]
    )

    controls = {c.control_id: c for c in B3_PSF_SINGLET.negative_controls}
    assert normalized(controls["opl-sign-flip"].caveat) == normalized(
        committed["opl_sign_flip_basis"]
    )
    assert normalized(controls["inverted-quadrature-weight"].caveat) == normalized(
        committed["quadrature_weight_min_improvement_factor_basis"]
    )


def test_o2_is_reported_and_cannot_gate() -> None:
    """Our own float64 ASM/RS propagator, kept as characterization evidence.

    L2-PSF-01 set a negative-control floor from an O2 comparison once and had to
    retire it as circular. The schema now refuses the promotion rather than
    relying on nobody trying it again.
    """
    o2 = B3_PSF_SINGLET.tolerance_for("o2_asm_intensity_relative_l2")
    assert o2 is not None
    assert not o2.may_gate
    assert o2.basis_kind is ToleranceBasis.CROSS_ROUTE_AGREEMENT
    assert "circular validation" in o2.basis

    with pytest.raises(ValueError, match="may_gate must be False"):
        Tolerance(
            metric="o2_asm_intensity_relative_l2",
            threshold=1e-3,
            basis=o2.basis,
            basis_kind=ToleranceBasis.CROSS_ROUTE_AGREEMENT,
            may_gate=True,
        )


def test_the_singlet_carries_the_control_that_fires_backwards() -> None:
    """0.42 against a 1.2 floor: the uniform configuration is CLOSER to the
    oracle than the weighted one. Reported, not hidden and not widened."""
    from verification.families.schema import NegativeControlExpectation

    control = next(
        c for c in B3_PSF_SINGLET.negative_controls if c.control_id == "inverted-quadrature-weight"
    )
    assert control.expectation is NegativeControlExpectation.KNOWN_FIRES_BACKWARDS
    assert "0.42" in control.caveat


def test_the_singlet_declares_the_controls_it_has_not_run() -> None:
    """An omitted control reads exactly like a control that passed.

    Both of the unrun ones are unrun for the same interesting reason: they are
    identity operations at the frozen on-axis instance, so declaring them is
    also a statement about which instance would be needed to exercise them.
    """
    from verification.families.schema import NegativeControlExpectation

    unrun = {
        c.control_id: c
        for c in B3_PSF_SINGLET.negative_controls
        if c.expectation is NegativeControlExpectation.NOT_IMPLEMENTED
    }
    assert set(unrun) == {"axis-transpose", "launch-phase-error"}
    for control in unrun.values():
        assert "off axis" in control.caveat or "off-axis" in control.caveat


def test_demo2_names_the_oracle_that_actually_decided_its_number() -> None:
    """CHE-116: this test used to assert the opposite, and the assertion was wrong.

    It was ``test_demo2_is_graded_against_something_outside_this_repository`` and
    it passed by checking that the word "published" appeared in the oracle
    description. It did -- the description said the decider was "the published
    figure from the paper's own implementation ... a different group's code" --
    and no committed number was ever measured against it. Every figure this
    family carries, including the ``0.999418`` in its gate disposition, is
    ``routes.rw_p.vs_oracle.ncc_intensity`` from
    ``benchmarks/probes/records/ray_wave/demo2_paper_jax.json``, measured against
    ``verification/asm_oracle.angular_spectrum_float64`` -- our own float64
    propagator. The paper publishes SI Table S2 summary numbers against its OWN
    oracle plus a printed figure, so there is no field array here to compare to;
    the probe record says as much in its own
    ``paper_numbers_are_context_not_thresholds`` field.

    ``INDEPENDENT`` is still the right label and is asserted unchanged: the
    oracle shares no kernel and no traced rays with ``C_PATCH_WFT`` or
    ``C_RAY_TO_WAVE``, which is what the enum means. What changed is the claim
    about where it lives, because "independent of the coupler" and "external to
    this repository" are different guarantees and only the first one is true.

    So this test now pins the corrected attribution, and pins the *observed
    value* unchanged -- the correction moved an attribution, not a number.
    """
    assert B3_DEMO2.oracle.kind is Oracle.INDEPENDENT_IMPLEMENTATION
    assert B3_DEMO2.oracle.independence is OracleIndependence.INDEPENDENT

    description = B3_DEMO2.oracle.description
    assert "verification/asm_oracle.angular_spectrum_float64" in description
    assert "THE PUBLISHED FIGURE IS NOT AN ORACLE HERE" in description
    assert "src/verification/asm_oracle.py" in B3_DEMO2.oracle.reference

    # The metric descriptions were the other half of the mis-statement.
    for name in ("demo2_ncc", "demo2_relative_l2"):
        metric = next(m for m in B3_DEMO2.metrics if m.name == name)
        assert "NOT against the published intensity" in metric.description

    assert B3_DEMO2.gate_disposition is not None
    assert B3_DEMO2.gate_disposition.observed == pytest.approx(0.999418)
    assert "vs_oracle" in B3_DEMO2.gate_disposition.note
    assert "asm_oracle" in B3_DEMO2.gate_disposition.note

    # No threshold moved. This is the line that makes the correction a
    # re-attribution rather than a re-scoping.
    assert B3_DEMO2.tolerance_for("demo2_ncc").threshold == 0.999
    assert B3_DEMO2.tolerance_for("demo2_relative_l2").threshold == 5e-2


def test_the_demo2_asm_oracle_is_reconciled_against_the_singlets_o2_rule() -> None:
    """The objection independent review raised, pinned so it stays answered.

    Correcting the oracle attribution created a visible tension in one file:
    ``B3-PSF-SINGLET`` declares an ASM oracle with ``may_gate=False`` on the
    grounds that "using our own numerical code as the answer key for our own
    numerical code is circular validation", and three hundred lines later
    ``B3-DEMO2`` now truthfully names an in-repository ASM propagator as its
    decider with ``may_gate=True``. Before the correction the tension was hidden
    by a false claim of externality; after it, it has to be argued or escalated.

    It is argued, on the input rather than the algorithm -- the singlet's O2 is
    built from a fit to the coupler's own traced pupil and so shares traced data,
    while demo2's ASM starts from the physical input mask -- **and** the
    ``may_gate`` decision is explicitly flagged as the owner's rather than
    resolved. Both halves are asserted, because the argument without the escalation
    would be this issue deciding a policy question in its own favour.
    """
    description = B3_DEMO2.oracle.description
    assert "B3-PSF-SINGLET'S O2 RULE" in description
    assert "circular validation" in description
    assert "_traced_pupil_wavefront" in description
    assert "the INPUT," in description
    assert "OWNER'S CALL" in description

    # Flagged, not silently resolved in either direction.
    assert "may_gate IS STILL True" in description
    for name in ("demo2_ncc", "demo2_relative_l2"):
        assert B3_DEMO2.tolerance_for(name).may_gate is True

    # And the singlet's own rule is untouched, which is what makes the
    # comparison meaningful rather than a rewrite of both sides.
    o2 = B3_PSF_SINGLET.tolerance_for("o2_asm_intensity_relative_l2")
    assert o2.may_gate is False
    assert "circular validation" in o2.basis


def test_the_dualroute_note_does_not_claim_the_arm_ruled_out_a_constant() -> None:
    """The withdrawn inference, pinned withdrawn.

    A claim that was published in a gating tolerance's own reason field and then
    retracted is exactly the kind of thing that creeps back when someone tidies
    the prose. The retraction is asserted, not just the absence of the claim.
    """
    note = B3_DUALROUTE.gate_disposition.note
    assert "UNTESTED" in note
    assert "withdrawn" in note
    assert "independent review showed it" in note
    assert "floor-dominated" in note
    # The sound half must still be there -- withdrawing the inference must not
    # take the measurement with it.
    assert "sum(|amplitude|^2) falls by nearly the same factor" in note


def test_the_demo2_observed_value_is_stored_at_full_precision() -> None:
    """Six figures cannot distinguish the oracle number from the route agreement.

    ``vs_oracle.ncc_intensity`` is 0.9994182326189224 and
    ``route_agreement.ncc_intensity`` is 0.9994180762008337. Both round to
    0.999418, they are different quantities, and the family carries the first.
    Storing the truncated value made the attribution an argument; storing the
    full one makes it a lookup.
    """
    observed = B3_DEMO2.gate_disposition.observed
    assert observed == 0.9994182326189224
    assert observed != 0.9994180762008337, "that is the route agreement, not the oracle"

    record = json.loads(
        (
            ROOT / "benchmarks" / "probes" / "records" / "ray_wave" / "demo2_paper_jax.json"
        ).read_text()
    )
    assert record["routes"]["rw_p"]["vs_oracle"]["ncc_intensity"] == observed


def test_the_demo2_promotion_argument_was_corrected_not_dropped() -> None:
    """A family whose stated reason for existing turns out to be false needs a
    true one or it needs retiring. This records which happened."""
    notes = B3_DEMO2.notes
    assert "EXTERNAL published" in notes, "the original claim is quoted, not erased"
    assert "does not hold" in notes
    assert "independent OF THE COUPLERS UNDER TEST" in notes
    assert "regression anchor" in notes


def test_demo2_is_non_generative_because_generation_destroys_its_value() -> None:
    """There is no published budget for a drawn point.

    CHE-116 left ``sampler_absent_note`` alone deliberately, and it still says
    "external independence" -- a phrase the same issue found unsupported one
    field above. It is not a contradiction here: the reason this case must not be
    generated is that its configuration and ray budget are the paper's, and a
    drawn parameter point has neither. That argument survives the oracle
    correction intact, so the note was not rewritten to look consistent. The
    phrase is asserted as-is rather than quietly softened, and the reader is sent
    to the oracle description for what "external" does and does not cover here.
    """
    from verification.families.schema import SamplerAbsentReason

    assert B3_DEMO2.sampler is None
    assert B3_DEMO2.sampler_absent_reason is SamplerAbsentReason.SETUP_IS_THE_CERTIFICATE
    assert "external independence" in B3_DEMO2.sampler_absent_note
    assert "specific budget" in B3_DEMO2.sampler_absent_note


# --------------------------------------------------------------------------- #
# The dualroute split
# --------------------------------------------------------------------------- #


def test_the_dualroute_case_split_because_no_independent_leg_exists() -> None:
    """M4.1 asked for one B3 family; the evidence supports two families.

    Every PSF route on the Cooke triplet is our own code -- FFTPSF and
    HuygensPSF share one Wavefront/OPD front end, and ray->wave shares the
    trace. What can still decide something is an intermediate invariant.
    """
    assert B3_DUALROUTE.category is BenchmarkCategory.B3
    assert B3_DUALROUTE.oracle.kind is Oracle.CONSERVATION_LAW
    assert B3_DUALROUTE.oracle.may_decide_correctness

    assert B4_DUALROUTE_AGREEMENT.category is BenchmarkCategory.B4
    assert B4_DUALROUTE_AGREEMENT.oracle.kind is Oracle.CROSS_ROUTE
    assert not B4_DUALROUTE_AGREEMENT.oracle.may_decide_correctness
    assert not B4_DUALROUTE_AGREEMENT.gating_tolerances


def test_the_route_comparison_could_not_have_been_a_b3_family() -> None:
    """Not a preference: the schema refuses it at construction."""
    with pytest.raises(ValueError, match="must be category B4"):
        BenchmarkFamily(
            **{
                **{
                    field: getattr(B4_DUALROUTE_AGREEMENT, field)
                    for field in (
                        "family_version",
                        "layer",
                        "question",
                        "components",
                        "claim_kind",
                        "parameters",
                        "oracle",
                        "metrics",
                        "execution_policy",
                        "stochastic_policy",
                        "sampler_absent_reason",
                        "evidence",
                    )
                },
                "family_id": "B3-DUALROUTE-AGREEMENT-PROBE",
                "category": BenchmarkCategory.B3,
            }
        )


def test_the_off_axis_route_discrepancy_is_attributed_rather_than_open() -> None:
    """0.313 relative L2 against 0.0138 for the other pair, and the cause is
    identified: FFTPSF sets its pixel scale from a single scalar working F/#
    against an anisotropic image-space pupil. An attributed cross-route
    discrepancy is worth more than an unattributed agreement.
    """
    note = B4_DUALROUTE_AGREEMENT.gate_disposition.note  # type: ignore[union-attr]
    assert "5.284" in note and "6.030" in note, "the measured F/# anisotropy"
    assert "1.0955" in note, "the measured scale fit, beside the predicted 1.100"


# --------------------------------------------------------------------------- #
# B4: measured, and unable to decide
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("family", B4, ids=lambda f: f.family_id)
def test_no_b4_family_carries_a_gating_tolerance(family: BenchmarkFamily) -> None:
    assert not family.gating_tolerances
    assert family.gate_disposition is not None
    assert family.gate_disposition.status is GateStatus.CHARACTERIZED_NO_GATE


@pytest.mark.parametrize("family", B4, ids=lambda f: f.family_id)
def test_the_schema_refuses_to_let_a_b4_family_gate(family: BenchmarkFamily) -> None:
    """Proved by attempting the promotion, not by trusting the label."""
    import dataclasses

    gating = Tolerance(
        metric=family.metrics[0].name,
        threshold=1.0,
        basis="an analytic derivation, so the tolerance itself would be allowed to gate",
        basis_kind=ToleranceBasis.ANALYTIC_DERIVATION,
        may_gate=True,
        rejects="nothing; this tolerance exists only to be refused",
    )
    with pytest.raises(ValueError, match="B4 family cannot carry a gating tolerance"):
        dataclasses.replace(family, tolerances=(*family.tolerances, gating))


def test_the_demo3_why_not_migrated_unreworded() -> None:
    """From ``manifest.yaml``'s ``characterizations:`` block, word for word.

    It is the most valuable sentence in that file, and rewriting it to sound
    better would be the first step toward rewriting it to sound passable.
    """
    manifest = yaml.safe_load(
        (ROOT / "benchmarks/manifest.yaml").read_text(encoding="utf-8")
    )
    demo3 = next(c for c in manifest["characterizations"] if c["id"] == "DEMO3-RW-P")
    assert " ".join(DEMO3_WHY_NOT.split()) == " ".join(demo3["why_not"].split())
    assert DEMO3_WHY_NOT in (B4_DEMO3.gate_disposition.note or "")  # type: ignore[union-attr]


def test_demo3_has_no_oracle_and_says_so() -> None:
    """This issue does not invent one. The missing thing is a reference, not a
    budget, and no amount of compute produces it."""
    assert B4_DEMO3.oracle.kind is Oracle.NONE
    assert B4_DEMO3.oracle.independence is OracleIndependence.NOT_APPLICABLE
    assert B4_DEMO3.oracle.callable is None


def test_demo3_requires_an_ensemble_rather_than_a_realization() -> None:
    """Seed-to-seed NCC from one seed is undefined, and a family that produced a
    number anyway would be fabricating the ensemble."""
    policy = B4_DEMO3.stochastic_policy
    assert policy.is_stochastic
    assert policy.minimum_seeds >= 4
    assert len(policy.required_evidence) == 4, "all four evidence kinds, they fail separately"

    seed = next(p for p in B4_DEMO3.parameters if p.name == "seed")
    assert seed.kind is ParameterKind.EXECUTION

    controls = {c.control_id for c in B4_DEMO3.negative_controls}
    assert "single-seed-is-not-a-result" in controls
    assert "slope-from-two-rungs" in controls


def test_demo3_reports_no_stochastic_evidence_it_has_not_established() -> None:
    """The family declares which kinds it OWES; the ledger records which are
    ESTABLISHED. Filling the second from the first would turn an obligation into
    a citation, and the ledger would report evidence that does not exist."""
    from verification.families.projection import claims_from_family

    claims = claims_from_family(B4_DEMO3)
    for claim in claims:
        assert claim.stochastic is not None
        assert len(claim.stochastic.missing) == 4, (
            "nothing has been measured through this family yet, and the ledger must "
            "say so rather than citing the requirement as though it were the evidence"
        )


def test_the_demo3_seed_to_seed_metric_says_why_it_can_never_gate() -> None:
    """Two realizations of a biased estimator correlate at 1.0."""
    metric = B4_DEMO3.metric("seed_to_seed_ncc")
    assert any("bias" in blind for blind in metric.blind_to)


def test_the_cost_family_opened_a_new_coverage_axis() -> None:
    """Cost is not device parity: the same device can be fast or slow and both
    are right. It gets its own claim kind rather than being filed under one that
    means something else."""
    from verification.claim_ledger import ClaimKind

    assert B4_COST.claim_kind is ClaimKind.COST
    assert B4_COST.oracle.kind is Oracle.NONE


def test_every_b4_metric_states_a_blind_spot() -> None:
    """A characterization number with no stated blind spot is the easiest kind
    to over-read, because nothing about it looks like a limit."""
    for family in B4:
        for metric in family.metrics:
            assert metric.blind_to
            assert all(blind.strip() for blind in metric.blind_to)


# --------------------------------------------------------------------------- #
# Registry shape
# --------------------------------------------------------------------------- #


def test_the_composed_families_the_milestones_promised_are_registered() -> None:
    b3 = {f.family_id for f in families_for_category(BenchmarkCategory.B3)}
    b4 = {f.family_id for f in families_for_category(BenchmarkCategory.B4)}
    assert {"B3-PSF-SINGLET", "B3-DEMO2", "B3-DUALROUTE"} <= b3
    assert {"B4-DEMO3", "B4-COST"} <= b4


def test_no_b4_family_is_in_a_required_gate_collection() -> None:
    """These are the expensive families -- one is 0.14 h a run on a GPU. A suite
    that ran them on every change would stop being run."""
    for family in B4:
        for instance in family.canonical_instances:
            assert instance.split_tag != "required", (
                f"{family.family_id}/{instance.instance_id} is tagged required, and a "
                "B4 family cannot decide anything a required gate would be checking"
            )
        assert family.execution_policy.max_wall_seconds is not None, (
            "an expensive family with no declared envelope is one nothing can budget for"
        )


# --------------------------------------------------------------------------- #
# CHE-116: the energy-accounting intermediate, and the entry point that went
# --------------------------------------------------------------------------- #

ENERGY_RECORD = ROOT / "benchmarks" / "probes" / "records" / "b3_energy_accounting.json"


def _energy_record() -> dict:
    assert ENERGY_RECORD.is_file(), (
        f"{ENERGY_RECORD.relative_to(ROOT)} is missing. Regenerate it:\n"
        "    ./run.sh python benchmarks/probes/b3_energy_accounting.py --write"
    )
    return json.loads(ENERGY_RECORD.read_text())


def test_the_b3_energy_accounting_intermediate_is_measured_and_does_not_close() -> None:
    """AC 4 asked for the intermediate to be CHECKED, not only declared.

    It is now checked, and it fails to close -- by five to eight orders of
    magnitude, on all three families, from one convention. That is a far more
    useful state than the previous one, where three families each declared a
    gating conservation invariant that nothing had ever evaluated.

    The test asserts the finding rather than a pass, because the finding is the
    deliverable. If a later change makes one of these read 1.0, this test fails
    and whoever made it read 1.0 owes the derivation.
    """
    record = _energy_record()
    ratios = record["finding"]["ratios_by_family"]
    assert set(ratios) == {"B3-PSF-SINGLET", "B3-DEMO2", "B3-DUALROUTE"}
    for family_id, ratio in ratios.items():
        assert ratio is not None, f"{family_id} has no measured quotient"
        assert ratio < 1e-3, (
            f"{family_id} now reads {ratio!r}. If the ray-to-wave absolute-power "
            "normalization has been settled, this test is the one to rewrite -- and "
            "the derivation and its oracle belong in the same change."
        )
    assert record["finding"]["all_far_from_unity"] is True


def test_the_scaling_arm_measures_the_dimensional_defect_directly() -> None:
    """What the arm is for, after independent review cut down what it claimed.

    The first version of this test asserted ``abs(ratio_change_factor - 1) >
    0.01`` and was named ``..._is_not_a_calibration_constant``: the argument was
    that a quotient which MOVES under refinement cannot be fixed by one constant.
    Review showed the drift does not support that. On this configuration the
    quotient sits within 0.5% of the plain grid area at 32 rings and the
    reconstruction is floor-dominated -- every ray splats a full-grid ramp, and
    the measured border energy is near what a flat field gives -- so the 10.9%
    drift is the focal spot's share rising above a floor falling as 1/N, not
    evidence about closure. Two points on such a grid decide nothing about a
    constant.

    What the arm DOES measure survives and is stronger, because it is the
    dimensional defect itself rather than an inference from it: ``sum(|a|)`` is
    invariant under refinement while ``sum(|a|^2)`` falls by roughly the ray-count
    factor. That is the squared area element in the denominator, seen directly.
    """
    arm = _energy_record()["ray_count_scaling_arm"]
    rungs = arm["rungs"]
    assert len(rungs) == 2
    assert rungs[0]["hexapolar_rings"] != rungs[1]["hexapolar_rings"]
    assert rungs[0]["ray_count"] < rungs[1]["ray_count"]

    # sum(|a|) converged; sum(|a|^2) tracking 1/N. The pair IS the finding.
    assert abs(arm["abs_sum_change_factor"] - 1.0) < 1e-3, (
        "sum(|amplitude|) is no longer invariant under refinement, which would mean "
        "the bundle's own amplitude bookkeeping moved and this arm is measuring "
        "something else now"
    )
    rays = arm["ray_count_change_factor"]
    assert arm["squared_sum_change_factor"] * rays == pytest.approx(1.0, rel=0.05), (
        "sum(|amplitude|^2) no longer scales as 1/N, which is the observation the "
        "dimensional finding rests on"
    )

    # And the withdrawn claim must stay withdrawn.
    verdict = arm["verdict"]
    assert "DOES NOT SHOW" in verdict
    assert "withdrawn" in verdict
    assert "UNTESTED" in _energy_record()["finding"]["statement"]


def test_the_arm_records_that_its_configuration_is_floor_dominated() -> None:
    """The context that stops the next reader over-reading the arm.

    A quotient that equals the grid area to 0.5% is not a subtle physics result,
    it is a near-uniform field integrated over the whole window. Recording that
    is what keeps someone from citing the drift as a convergence measurement.
    """
    for rung in _energy_record()["ray_count_scaling_arm"]["rungs"]:
        assert rung["grid_area_m2"] > 0.0
        assert rung["ratio_over_grid_area"] == pytest.approx(1.0, rel=0.2)
        flat = rung["border_energy_fraction_if_field_were_flat"]
        assert rung["border_energy_fraction"] == pytest.approx(flat, rel=0.2)


def test_no_energy_tolerance_was_widened_or_had_its_gate_removed() -> None:
    """The rule that makes an unmet gate worth carrying.

    An invariant nobody can evaluate is a strong temptation to set
    ``may_gate=False`` and make the diagnostics go quiet. CHE-116 did not, on all
    three families, and this is the test that keeps it that way.
    """
    expected = {
        "B3-PSF-SINGLET": ("HANDOFF_ENERGY_CLOSES", 1e-3),
        "B3-DEMO2": ("PATCH_ENERGY_CLOSES", 1e-3),
    }
    for family in (B3_PSF_SINGLET, B3_DEMO2):
        invariant_id, threshold = expected[family.family_id]
        invariant = next(i for i in family.invariants if i.invariant_id == invariant_id)
        assert invariant.tolerance.threshold == threshold
        assert invariant.tolerance.may_gate is True
        assert invariant.tolerance.basis_kind is ToleranceBasis.CONSERVATION_LAW

    route = B3_DUALROUTE.tolerance_for("route_power_ratio")
    assert route.threshold == 1e-2
    assert route.may_gate is True


def test_the_dualroute_gate_says_why_it_is_unmeasured_not_just_that_it_is() -> None:
    """``NOT_MEASURED`` with no reason is indistinguishable from nobody looking."""
    disposition = B3_DUALROUTE.gate_disposition
    assert disposition.status is GateStatus.NOT_MEASURED
    note = disposition.note
    assert "NOT FORMABLE FROM" in note.upper()
    assert "incident_amplitude_power_sum" in note
    assert "b3_energy_accounting.json" in note
    assert "NOTHING WAS WIDENED" in note
    # The other two families' identical invariant is named, so a reader cannot
    # come away thinking this is one family's local problem.
    assert "HANDOFF_ENERGY_CLOSES" in note
    assert "PATCH_ENERGY_CLOSES" in note


def test_every_b3_family_has_a_recorded_runtime_and_memory_envelope() -> None:
    """AC 7, and 'declared' is not 'recorded'.

    Each family declares ``max_wall_seconds`` and ``max_peak_memory_gib`` in its
    ``ExecutionPolicy``. Before CHE-116 nothing had measured against them. The
    record now carries an observed wall time and an observed peak for each, with
    the verdict computed rather than asserted.
    """
    cases = {case["family_id"]: case for case in _energy_record()["cases"]}
    assert set(cases) == {"B3-PSF-SINGLET", "B3-DEMO2", "B3-DUALROUTE"}
    for family in B3:
        envelope = cases[family.family_id]["envelope"]
        policy = family.execution_policy
        assert envelope["declared_max_wall_seconds"] == policy.max_wall_seconds
        assert envelope["declared_max_peak_memory_gib"] == policy.max_peak_memory_gib
        assert envelope["observed_wall_seconds"] > 0.0
        assert envelope["wall_inside_declared_envelope"] is True

        # `None` is an accepted verdict, and the reason is a real defect this
        # issue's first draft had: the demo2 envelope was computed from the perf
        # harness's own RSS (0.027 GiB), which never samples the subprocess the
        # harness times, and it produced a passing verdict against a 40 GiB
        # envelope out of a number the record itself declared irrelevant. A
        # verdict from an absent measurement is worse than no verdict, so `_fits`
        # returns None and this test accepts it -- but only WITH a stated source,
        # so "unmeasured" cannot be silent.
        assert envelope["memory_inside_declared_envelope"] in (True, None)
        if envelope["memory_inside_declared_envelope"] is None:
            assert "UNMEASURED" in envelope["observed_peak_rss_source"]
        else:
            assert envelope["observed_peak_rss_gib"] > 0.0

        # Computed from what each case actually ran on, not hardcoded.
        assert envelope["devices_used"] == envelope["gpu_count_used"] <= 1
        assert envelope["fits_one_gpu"] is True

        # The shared-host rule that matters more than either number.
        if "peak_cgroup_swap_bytes" in envelope:
            assert envelope["peak_cgroup_swap_bytes"] in (0, None)


def test_the_demo2_envelope_uses_the_subprocess_peak_not_the_harness_process() -> None:
    """The specific mis-reading, pinned so it cannot come back.

    ``benchmarks/perf/records/demo2_paper_rw_p_ramp_sum_cuda.json`` carries two
    peaks: ``memory_report.peak_rss_bytes`` (29.4 MB, the harness) and
    ``subprocess.peak_child_rss_bytes`` (2.80 GB, the run). They differ by a
    factor of 95 and only one of them is this case's cost.
    """
    envelope = next(
        case for case in _energy_record()["cases"] if case["family_id"] == "B3-DEMO2"
    )["envelope"]
    assert "subprocess.peak_child_rss_bytes" in envelope["observed_peak_rss_source"]
    assert envelope["observed_peak_rss_gib"] == pytest.approx(2.6065, abs=1e-3)
    assert envelope["harness_process_peak_rss_gib"] < 0.1, (
        "the harness figure is kept, under a name that says what it is, so the "
        "factor-of-95 gap stays visible in the record"
    )
    # Host RSS is not device memory, and the record must not let that pass.
    assert "UNMEASURED" in envelope["peak_rss_caveat"]


def test_the_last_bespoke_entry_point_for_the_singlet_is_gone() -> None:
    """The acceptance criterion is a deletion, so the test is an absence.

    ``run_benchmark.py`` was 600 lines and the only way to run this workload
    until CHE-115 reproduced its frozen gate number bit-identically through the
    executor. ``evaluate.py`` went with it: its only input was the bundle that
    runner wrote. ``tolerances.yaml`` stays, because the family's tolerance bases
    are migrated from it verbatim and a test compares them character for
    character.
    """
    bundle = ROOT / "benchmarks" / "physics" / "L2-PSF-01"
    assert not (bundle / "run_benchmark.py").exists()
    assert not (bundle / "evaluate.py").exists()
    assert (bundle / "tolerances.yaml").is_file(), "the evidence file must NOT be deleted"

    replacement = ROOT / "benchmarks" / "instances" / "b3_psf_singlet.py"
    graph = ROOT / "examples" / "graphs" / "psf_singlet_sensor.yaml"
    record = ROOT / "benchmarks" / "instances" / "records" / "B3-PSF-SINGLET-01.json"
    for path in (replacement, graph, record):
        assert path.is_file(), f"{path.relative_to(ROOT)} is the replacement and is missing"

    # A deletion that leaves a caller pointing at the corpse is not a deletion.
    harness = (ROOT / "benchmarks" / "perf" / "run_baselines.py").read_text()
    assert '"L2-PSF-01" / "run_benchmark.py"' not in harness
    assert '"L2-PSF-01", "run_benchmark.py"' not in harness
    assert "b3_psf_singlet.py" in harness
