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
"""

from __future__ import annotations

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


def test_demo2_is_graded_against_something_outside_this_repository() -> None:
    """The one composed reproduction with an EXTERNAL reference, which is what
    makes it the regression anchor for the whole ray-wave path."""
    assert B3_DEMO2.oracle.kind is Oracle.INDEPENDENT_IMPLEMENTATION
    assert B3_DEMO2.oracle.independence is OracleIndependence.INDEPENDENT
    assert "published" in B3_DEMO2.oracle.description
    assert B3_DEMO2.gate_disposition is not None
    assert B3_DEMO2.gate_disposition.observed == pytest.approx(0.999418)


def test_demo2_is_non_generative_because_generation_destroys_its_value() -> None:
    """There is no published figure for a drawn point."""
    from verification.families.schema import SamplerAbsentReason

    assert B3_DEMO2.sampler is None
    assert B3_DEMO2.sampler_absent_reason is SamplerAbsentReason.SETUP_IS_THE_CERTIFICATE
    assert "external independence" in B3_DEMO2.sampler_absent_note


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
