"""B0: the five negative outcomes, and the two that are not negative at all.

CHE-108 (M1.3). What happens when a component is asked for something it cannot
do is a correctness property in its own right, and it is the one that decides
whether an agent can recover. The two failure modes it must never see are a
fabricated number and an unstructured traceback.

Two things get tested hardest.

**The five statuses must not collapse.** Each canonical instance names the
outcome it expects, and a test asserts the four contract statuses are produced
by four *different* instances -- because a conformance suite that maps "no
executable precision", "malformed request" and "the approximation does not
apply" to one failure code teaches an agent nothing.

**The interesting failures are silent.** ``B0-UNITS``'s two instances run
perfectly, every boundary check passes, the contract status is ``ok``, and the
physics is wrong. A suite built on "invalid input raises" would miss the entire
class, and the family declares that as a negative control rather than leaving it
implied.
"""

from __future__ import annotations

import math

import pytest

from core.boundary import ContractCode
from verification.families import (
    BenchmarkCategory,
    BenchmarkFamily,
    ValidityState,
    families_for_category,
)
from verification.families.b0_contract import B0_CONTRACT, B0_DTYPE, B0_UNITS, B0_VALIDITY
from verification.families.schema import GateStatus
from verification.hazards import KYKX_TWO_PI_AND_SIGN, UNITS_MICROMETRE_NANOMETRE
from verification.refusals import REFUSAL_CATALOGUE, RefusalEntry, refusal_for, statuses_covered
from verification.status import VerificationStatus

B0 = (B0_CONTRACT, B0_DTYPE, B0_VALIDITY, B0_UNITS)


def test_the_b0_families_and_instances_the_milestone_promised_exist() -> None:
    registered = {f.family_id for f in families_for_category(BenchmarkCategory.B0)}
    assert {"B0-CONTRACT", "B0-DTYPE", "B0-VALIDITY", "B0-UNITS"} <= registered

    instances = {i.instance_id for f in B0 for i in f.canonical_instances}
    assert {
        "B0-DTYPE-01",
        "B0-CAPINT-01",
        "B0-DEVICE-01",
        "B0-DEVICE-02",
        "B0-META-01",
        "B0-HANDOFF-01",
        "B0-PATCH-01",
        "B0-VALIDITY-01",
        "B0-UNITS-01",
        "B0-UNITS-02",
    } == instances


# --------------------------------------------------------------------------- #
# The five must not collapse
# --------------------------------------------------------------------------- #


def _expected_statuses(family: BenchmarkFamily) -> dict[str, str]:
    return {
        i.instance_id: (i.expected or {})["status"]
        for i in family.canonical_instances
        if "status" in (i.expected or {})
    }


def test_the_contract_family_produces_four_distinct_negative_outcomes() -> None:
    """An agent that cannot tell them apart cannot recover from any of them."""
    expected = _expected_statuses(B0_CONTRACT)
    assert set(expected.values()) == {
        VerificationStatus.UNSUPPORTED.value,
        VerificationStatus.INVALID_CONFIGURATION.value,
        VerificationStatus.OUT_OF_VALIDITY.value,
        VerificationStatus.BLOCKED.value,
    }


def test_the_fifth_outcome_comes_from_the_dtype_family() -> None:
    """``lossy_but_allowed`` is not a refusal: the run succeeds and something is
    lost, so it belongs with the family that measures the loss."""
    assert _expected_statuses(B0_DTYPE) == {
        "B0-DTYPE-01": VerificationStatus.LOSSY_BUT_ALLOWED.value
    }


def test_all_five_negative_outcomes_are_covered_by_a_canonical_instance() -> None:
    covered = {
        status for family in B0 for status in _expected_statuses(family).values()
    }
    assert covered == {
        VerificationStatus.UNSUPPORTED.value,
        VerificationStatus.INVALID_CONFIGURATION.value,
        VerificationStatus.OUT_OF_VALIDITY.value,
        VerificationStatus.LOSSY_BUT_ALLOWED.value,
        VerificationStatus.BLOCKED.value,
    }


def test_the_empty_capability_intersection_is_unsupported_not_invalid() -> None:
    """Project risk R5, as an instance.

    Chromatix accepts only complex64; C_PATCH_WFT computes only in complex128 and
    only on CPU. A composed route through both has no precision at which it can
    execute. An agent WILL propose such routes, and whether that is refused at
    planning time or discovered three nodes in is the difference between a
    recoverable failure and a dead end.
    """
    instance = next(i for i in B0_CONTRACT.canonical_instances if i.instance_id == "B0-CAPINT-01")
    assert (instance.expected or {})["status"] == VerificationStatus.UNSUPPORTED.value
    assert instance.validity_status is ValidityState.FAR_OUTSIDE
    assert "no precision at which it can execute" in (instance.expected or {})["why"].lower() or (
        "NO precision at which it can execute" in (instance.expected or {})["why"]
    )


def test_a_blocked_refusal_is_one_the_component_could_have_proceeded_past() -> None:
    """OPL_REFERENCE_UNVERIFIED is blocked rather than invalid: nothing about the
    request is malformed, and the missing thing is a declaration the coupler
    refuses to default."""
    entry = refusal_for(ContractCode.OPL_REFERENCE_UNVERIFIED.value)
    assert entry.status is VerificationStatus.BLOCKED
    assert entry.could_have_proceeded

    instance = next(i for i in B0_CONTRACT.canonical_instances if i.instance_id == "B0-HANDOFF-01")
    assert (instance.expected or {})["code"] == ContractCode.OPL_REFERENCE_UNVERIFIED.value


def test_out_of_validity_means_it_would_run_and_be_wrong() -> None:
    """NON_HEXAPOLAR_SAMPLING is the archetype: the quadrature weight a ray
    carries is derived from its ring, so applying it to a rectangular bundle
    assigns weights that mean nothing. The code runs."""
    entry = refusal_for(ContractCode.NON_HEXAPOLAR_SAMPLING.value)
    assert entry.status is VerificationStatus.OUT_OF_VALIDITY
    assert entry.could_have_proceeded

    instance = next(i for i in B0_CONTRACT.canonical_instances if i.instance_id == "B0-PATCH-01")
    assert instance.validity_status is ValidityState.FAR_OUTSIDE


def test_the_curvature_crossing_is_out_of_validity_rather_than_unsupported() -> None:
    """eps_curv = 0.2 against arcsin(1/20) = 0.05004. The tangent-plane picture
    still computes; it computes the wrong thing."""
    instance = next(
        i for i in B0_VALIDITY.canonical_instances if i.instance_id == "B0-VALIDITY-01"
    )
    assert (instance.expected or {})["status"] == VerificationStatus.OUT_OF_VALIDITY.value
    assert (instance.expected or {})["bound_rad"] == pytest.approx(math.asin(0.05))
    assert instance.validity_status is ValidityState.FAR_OUTSIDE
    assert instance.validity_margins["SI_S3_CURVATURE"] < 0.0


# --------------------------------------------------------------------------- #
# The refusal catalogue
# --------------------------------------------------------------------------- #


def test_every_contract_code_has_a_catalogue_entry() -> None:
    """A refusal a caller cannot look up is a traceback with extra steps."""
    declared = {code.value for code in ContractCode}
    catalogued = set(REFUSAL_CATALOGUE)
    assert declared == catalogued, (
        f"missing entries: {sorted(declared - catalogued)}; "
        f"entries for codes that no longer exist: {sorted(catalogued - declared)}"
    )


@pytest.mark.parametrize(
    "entry", sorted(REFUSAL_CATALOGUE.values(), key=lambda e: e.code), ids=lambda e: e.code
)
def test_every_refusal_names_a_trigger_a_remedy_and_an_outcome(entry: RefusalEntry) -> None:
    assert entry.trigger.strip()
    assert entry.remedy.strip()
    assert entry.status is not VerificationStatus.OK


def test_the_catalogue_reports_which_outcomes_it_cannot_cover() -> None:
    """``unsupported`` and ``lossy_but_allowed`` come from CapabilityError and
    the precision bridge rather than from a ContractCode, so a catalogue built
    only from contract codes legitimately does not cover them.

    Saying which are missing is more useful than claiming coverage it does not
    have -- and asserting the gap keeps it from being closed by accident with a
    mis-classified entry.
    """
    covered = statuses_covered()
    assert VerificationStatus.UNSUPPORTED not in covered
    assert VerificationStatus.LOSSY_BUT_ALLOWED not in covered
    assert {
        VerificationStatus.INVALID_CONFIGURATION,
        VerificationStatus.OUT_OF_VALIDITY,
        VerificationStatus.BLOCKED,
    } <= covered


def test_an_uncatalogued_code_says_so_rather_than_returning_nothing() -> None:
    with pytest.raises(KeyError, match="traceback with extra steps"):
        refusal_for("NOT_A_CODE")


# --------------------------------------------------------------------------- #
# The silent ones
# --------------------------------------------------------------------------- #


def test_both_measured_traps_expect_an_ok_contract() -> None:
    """That is the finding, not a defect in the boundary layer. There is nothing
    for it to complain about, which is why a conformance suite that stops at
    contract conformance calls them fine."""
    for instance in B0_UNITS.canonical_instances:
        assert (instance.expected or {})["contract_status"] == "ok"
        assert instance.validity_status is ValidityState.INSIDE


def test_the_units_family_keeps_the_measured_wrong_numbers() -> None:
    coating = next(i for i in B0_UNITS.canonical_instances if i.instance_id == "B0-UNITS-01")
    assert (coating.expected or {})["wrong_value"] == UNITS_MICROMETRE_NANOMETRE.wrong_value
    assert (coating.expected or {})["bare_glass"] == UNITS_MICROMETRE_NANOMETRE.right_value

    kykx = next(i for i in B0_UNITS.canonical_instances if i.instance_id == "B0-UNITS-02")
    assert (kykx.expected or {})["wrong_value"] == KYKX_TWO_PI_AND_SIGN.wrong_value
    assert (kykx.expected or {})["correct_value"] == KYKX_TWO_PI_AND_SIGN.right_value


def test_the_units_family_declares_that_a_contract_only_suite_misses_it() -> None:
    control = next(
        c for c in B0_UNITS.negative_controls if c.control_id == "contract-only-suite"
    )
    assert "misses this" in control.description


def test_the_units_family_is_non_generative_because_the_wrong_number_is_the_artifact() -> None:
    from verification.families.schema import SamplerAbsentReason

    assert B0_UNITS.sampler is None
    assert B0_UNITS.sampler_absent_reason is SamplerAbsentReason.HISTORICAL_REGRESSION
    assert "measured wrong number IS the artifact" in B0_UNITS.sampler_absent_note


# --------------------------------------------------------------------------- #
# Precision honesty
# --------------------------------------------------------------------------- #


def test_the_precision_loss_is_a_number_and_not_a_warning() -> None:
    """2.5e-5 at z = 40 um, measured. A warning cannot be compared against the
    closed form; a number can."""
    instance = next(i for i in B0_DTYPE.canonical_instances if i.instance_id == "B0-DTYPE-01")
    assert (instance.expected or {})["measured_precision_loss"] == 2.5e-5

    tolerance = B0_DTYPE.tolerance_for("measured_precision_loss")
    assert tolerance is not None and tolerance.may_gate


def test_the_eps32_per_radian_bound_brackets_the_measured_loss() -> None:
    """The oracle is a closed form, so the recorded number can be CHECKED rather
    than merely present -- and this test caught the first draft getting it wrong.

    That draft used sqrt(eps64) = 1.49e-8 for float32's epsilon, which is a
    different number from 2**-23 = 1.19e-7, and produced a "bound" of 6.8e-6
    that the MEASURED 2.5e-5 sat above. A bound below its own measurement is not
    a bound. With the right epsilon it reads 5.4e-5 at z = 40 um, and the
    measurement sits comfortably under it.
    """
    oracle = B0_DTYPE.oracle.callable
    assert oracle is not None
    bound = oracle({"propagation_distance_m": 4.0e-5, "wavelength_m": 5.5e-7})
    assert bound == pytest.approx(5.45e-5, rel=0.02)
    assert bound > 2.5e-5, "the measured loss must sit UNDER the one-eps32-per-radian bound"

    # And it grows linearly with distance, which is why the M3 reference singlet
    # was scaled to a tenth: 6.3e-2 was measured at z = 47 mm.
    far = oracle({"propagation_distance_m": 4.7e-2, "wavelength_m": 5.5e-7})
    assert far / bound == pytest.approx(4.7e-2 / 4.0e-5, rel=1e-9)
    assert far > 6.3e-2, "the 47 mm measurement also sits under the bound"


def test_a_silent_truncation_is_a_declared_negative_control() -> None:
    """Field.build handed a complex128 array returns complex64 even under
    jax_enable_x64=True, and nothing reports it. Keeping complex128 out of
    accepted_input_dtypes is what moves the loss to where something measures it."""
    controls = {c.control_id for c in B0_DTYPE.negative_controls}
    assert {"silent-truncation", "requested-reported-as-actual"} <= controls


# --------------------------------------------------------------------------- #
# No fabricated output
# --------------------------------------------------------------------------- #


def test_the_no_fabricated_output_rule_is_a_gating_tolerance() -> None:
    """An AGENTS.md non-negotiable, mechanically checkable: failed or unsupported
    solvers return structured diagnostics and never invent fields, metrics,
    convergence or provenance."""
    tolerance = B0_CONTRACT.tolerance_for("fabricated_output_count")
    assert tolerance is not None
    assert tolerance.may_gate
    assert tolerance.threshold < 1.0, "the rule is zero, expressed as a count threshold"


@pytest.mark.parametrize("family", B0, ids=lambda f: f.family_id)
def test_no_b0_family_claims_a_gate_it_has_not_measured(family: BenchmarkFamily) -> None:
    """A decided gate must cite something that RE-RUNS it.

    Until CHE-108 executed the canonical instances, every B0 family was
    NOT_MEASURED or MEASURED_OFF_GATE and this test enforced that. All four are
    now MET, so the question changes rather than disappearing: what stops a
    future MET from being a claim nobody re-checks?

    The answer, and what is asserted here: the cited evidence has to include a
    pytest node id in the default suite. A gate whose only evidence is a source
    file is MEASURED_OFF_GATE by definition -- something ran once and nothing
    re-checks it -- and the schema already refuses a decided gate with no metric,
    no observed value, or no evidence at all.
    """
    disposition = family.gate_disposition
    assert disposition is not None
    if disposition.status not in (GateStatus.MET, GateStatus.NOT_MET):
        assert disposition.status in (
            GateStatus.MEASURED_OFF_GATE,
            GateStatus.NOT_MEASURED,
        )
        return

    assert disposition.metric is not None
    assert disposition.observed is not None
    executable = [reference for reference in disposition.evidence if "::" in reference]
    assert executable, (
        f"{family.family_id} claims {disposition.status.value} and cites no test that "
        f"re-runs it: {list(disposition.evidence)}. A decided gate whose evidence is "
        "only a source file is MEASURED_OFF_GATE."
    )


# --------------------------------------------------------------------------- #
# The footgun this authoring found
# --------------------------------------------------------------------------- #


def test_attaching_instances_after_registering_is_refused() -> None:
    """A copy the registry does not hold is a family that silently has no
    canonical instances.

    ``with_instances`` returns a new object, so calling it on an already
    registered family and rebinding the module-level name leaves the REGISTRY
    holding the instance-free original. Found while authoring these four, and
    now a constructor-level refusal rather than something to remember.
    """
    with pytest.raises(ValueError, match="already registered"):
        B0_UNITS.with_instances()
