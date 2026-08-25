"""The representation transitions, and the rule that gives them meaning.

CHE-109 through CHE-112 (M2.1-M2.4). Every coupler here is declared
``lossy: true``, so the interesting assertions are not "is it correct" but
"what is the declared budget, what is provably discarded, and does the control
that would catch a shared convention error actually fire".

The rule this file defends hardest is M2.4's: **a round trip that cannot be made
to fail proves nothing.** A shared convention error cancels between the two
directions, so ``B2-ROUNDTRIP`` makes "the broken twin ran" a validity predicate
rather than a habit -- an instance whose twin was not executed reports
``FAR_OUTSIDE`` its own validity domain, and the verifier will not present its
round-trip number as evidence.
"""

from __future__ import annotations

import pytest

from core.paths import repository_root

from verification.families import (
    BenchmarkCategory,
    BenchmarkFamily,
    ParameterKind,
    ValidityState,
    families_for_category,
)
from verification.families.b2_transitions import (
    B2_EQUIV,
    B2_R2W_EXACT,
    B2_R2W_ROUTE,
    B2_ROUNDTRIP,
    B2_W2R_STOCH,
    WHAT_DOES_NOT_SURVIVE,
)
from verification.families.schema import (
    GateStatus,
    Oracle,
    OracleIndependence,
    ToleranceBasis,
)
from verification.metrics import METRICS

B2 = (B2_R2W_EXACT, B2_R2W_ROUTE, B2_W2R_STOCH, B2_EQUIV, B2_ROUNDTRIP)


def test_the_transitions_the_milestone_promised_are_registered() -> None:
    registered = {f.family_id for f in families_for_category(BenchmarkCategory.B2)}
    assert {
        "B2-R2W-EXACT",
        "B2-R2W-ROUTE",
        "B2-W2R-STOCH",
        "B2-EQUIV",
        "B2-ROUNDTRIP",
    } <= registered


@pytest.mark.parametrize("family", B2, ids=lambda f: f.family_id)
def test_every_b2_metric_resolves_to_a_central_definition_or_says_why_not(
    family: BenchmarkFamily,
) -> None:
    """M2.4's centralization, checked where it matters most.

    These families are the ones whose numbers get compared across benchmarks, so
    a locally-defined "relative L2" here is exactly the drift the metrics module
    exists to stop.
    """
    for metric in family.metrics:
        if metric.definition is not None:
            assert metric.definition in METRICS, (
                f"{family.family_id}/{metric.name} names {metric.definition}, which is "
                "not a central definition"
            )


# --------------------------------------------------------------------------- #
# The exact/fast split
# --------------------------------------------------------------------------- #


def test_the_exact_and_fast_routes_are_separate_families() -> None:
    """One family covering both would let the fast route inherit the exact one's
    evidence, and they answer different questions: a limit and a budget."""
    # The exactness family's oracle is ANALYTIC, not a deterministic limit. It was
    # declared DETERMINISTIC_LIMIT before it ran, and executing it showed the
    # instance actually compares against the closed-form plane wave -- which is
    # the stronger oracle and the one the gate needs, because an enumerated
    # reference shares the kernel and can only establish the sampling.
    assert B2_R2W_EXACT.oracle.kind is Oracle.ANALYTIC
    assert B2_R2W_EXACT.oracle.independence is OracleIndependence.INDEPENDENT
    assert B2_R2W_ROUTE.oracle.kind is Oracle.DETERMINISTIC_LIMIT
    assert B2_R2W_ROUTE.oracle.reference == "B2-R2W-EXACT", (
        "the fast route is measured AGAINST the exact one, not beside it"
    )

    exact = B2_R2W_EXACT.tolerance_for("exactness_relative_l2_field")
    budget = B2_R2W_ROUTE.tolerance_for("route_field_relative_l2")
    assert exact is not None and budget is not None
    assert exact.threshold < budget.threshold / 1e8, (
        "an exactness limit and an error budget must not be the same order of number, "
        "or the distinction between them is decorative"
    )
    assert exact.basis_kind is ToleranceBasis.NUMERICAL_PRECISION_FLOOR
    assert budget.basis_kind is ToleranceBasis.INDEPENDENT_DERIVATION


def test_the_exactness_family_says_what_its_oracle_does_not_establish() -> None:
    """The scope statement, updated to the oracle the instance actually uses.

    This test used to assert "independent of the SAMPLING and not of the kernel",
    which was the correct scope for the enumerated-reference framing the family
    carried before it was executed. The executed instance compares against a
    closed-form plane wave, so the kernel IS pinned and asserting otherwise would
    understate the gate.

    What still has to be said is what an analytic comparison on THIS bundle
    cannot say: a collimated on-axis bundle has projection factor exactly 1 and
    no oblique ramp, so two of the four conventions this gate is credited with are
    pinned in the configuration run rather than in general. B2-W2R-STOCH measures
    those blind spots as numbers.
    """
    metric = B2_R2W_EXACT.metric("exactness_relative_l2_field")
    assert any("inert on THIS bundle" in blind for blind in metric.blind_to), metric.blind_to
    description = B2_R2W_EXACT.oracle.description
    assert "CLOSED-FORM" in description
    assert "shares no code with the coupler" in description
    # And it must still say why the weaker framing was not enough, so the change
    # reads as a correction rather than as a promotion.
    assert "can only establish the sampling" in description


def test_the_exactness_oracle_names_a_callable_that_exists() -> None:
    """An oracle whose callable is a string nobody resolves is a citation.

    The family declares `benchmarks/instances/b2_transitions.py::_plane_wave_on_grid`,
    and this checks both halves: the file is there and the symbol is in it.
    """
    reference = B2_R2W_EXACT.oracle.callable
    assert reference, "an ANALYTIC oracle owes its closed form"
    path, _, symbol = reference.partition("::")
    source = repository_root() / path
    assert source.is_file(), reference
    assert f"def {symbol}(" in source.read_text(), f"{reference} names a missing symbol"


def test_on_node_and_off_node_are_a_declared_representation_choice() -> None:
    """Conflating them is how a route gets credited with an exactness it only
    has on grid nodes."""
    alignment = next(p for p in B2_R2W_EXACT.parameters if p.name == "sample_alignment")
    assert alignment.kind is ParameterKind.REPRESENTATION
    controls = {c.control_id for c in B2_R2W_EXACT.negative_controls}
    assert "off-node-is-not-exact" in controls


def test_the_route_family_records_the_asymmetry_rather_than_an_average() -> None:
    """Exact on an on-node system and O(1e-1) on an off-node one, same route.

    Averaging those would produce a number describing neither system, which is
    why ``system`` is a PhysicalParameter and the note carries both. The
    assertion used to pin the inherited 7.1e-13 / 1.7% pair from a 60M-ray demo3
    probe; those are still cited, but as a B4 characterization rather than as
    this family's measurement, and the note now has to say which is which.
    """
    system = next(p for p in B2_R2W_ROUTE.parameters if p.name == "system")
    assert system.kind is ParameterKind.PHYSICAL
    note = B2_R2W_ROUTE.gate_disposition.note  # type: ignore[union-attr]
    assert "ON-NODE:" in note and "OFF-NODE:" in note
    assert "B4-DEMO3" in note, (
        "the paper-scale numbers are a probe and must be attributed to one"
    )
    assert "NOT the paper-scale demo3 numbers" in note


def test_ncc_alone_would_have_certified_the_lossy_route() -> None:
    """The clearest case in this repository for why one metric is not enough.

    The k-space route reads NCC 0.999868 while 1.7% of the energy is gone. The
    family declares that as a negative control, so the reason both metrics are
    reported is recorded rather than remembered.
    """
    control = next(
        c for c in B2_R2W_ROUTE.negative_controls
        if c.control_id == "ncc-alone-would-have-passed-it"
    )
    assert "0.999868" in control.description
    ncc_metric = B2_R2W_ROUTE.metric("route_ncc")
    assert any("power loss" in blind for blind in ncc_metric.blind_to)


# --------------------------------------------------------------------------- #
# The stochastic family
# --------------------------------------------------------------------------- #


def test_the_four_stochastic_evidence_kinds_stay_apart() -> None:
    """They fail independently: exact in the limit and still biased; unbiased
    and converging at the wrong rate; converging correctly at a variance the
    required ray count cannot reach."""
    policy = B2_W2R_STOCH.stochastic_policy
    assert policy.is_stochastic
    assert len(policy.required_evidence) == 4
    assert policy.minimum_seeds >= 8

    names = {m.name for m in B2_W2R_STOCH.metrics}
    assert {
        "enumeration_limit_relative_l2",
        "ensemble_mean_bias",
        "fitted_convergence_exponent",
        "variance_at_sampling_density",
    } <= names, "one metric per evidence kind, so they cannot be collapsed into a verdict"


def test_the_bias_tolerance_is_stated_in_sigma_not_in_field_units() -> None:
    """The resolvable bias depends on the ensemble size, so a fixed field-unit
    threshold would be tighter or looser depending on how many seeds were run --
    which makes it a statement about the budget rather than about the estimator."""
    tolerance = B2_W2R_STOCH.tolerance_for("ensemble_mean_bias")
    assert tolerance is not None
    assert B2_W2R_STOCH.metric("ensemble_mean_bias").unit == "sigma"
    assert tolerance.threshold == 3.0


def test_evanescent_power_is_accounted_rather_than_dropped() -> None:
    """It does not survive the transition, and the amount is part of the result."""
    invariants = {i.invariant_id for i in B2_W2R_STOCH.invariants}
    assert "EVANESCENT_POWER_ACCOUNTED" in invariants
    assert any(
        "evanescent" in item.lower() for item in WHAT_DOES_NOT_SURVIVE["C_WAVE_TO_RAY"]
    )


def test_no_gradient_is_claimed_for_the_stochastic_coupler() -> None:
    """derivative.verified is false in the registry and the family says so.

    A surrogate gradient's bias is characterized, not validated, and nothing
    here promotes it.
    """
    from registry.loader import Registry

    coupler = Registry.from_package().couplers["C_WAVE_TO_RAY"]
    assert coupler.derivative.verified is False
    note = B2_W2R_STOCH.gate_disposition.note  # type: ignore[union-attr]
    assert "NO GRADIENT IS CLAIMED" in note


def test_the_wave_to_ray_coupler_is_not_a_composable_graph_edge() -> None:
    """M2.2's open question, answered and made unmissable.

    It is declared in the registry -- which per AGENTS.md is a statement that a
    graph may address it -- and it has no executable graph node. Rather than
    leave that as an absence, the executor refuses an edge naming it before
    anything runs.
    """
    from runtime.executor import _COUPLER_MODULES

    assert "C_WAVE_TO_RAY" not in _COUPLER_MODULES
    note = B2_W2R_STOCH.notes
    assert "NO executable graph node" in note
    assert "test_a_coupler_with_no_graph_node_is_refused_before_anything_runs" in note


# --------------------------------------------------------------------------- #
# Equivalence
# --------------------------------------------------------------------------- #


def test_patch_granularity_is_a_representation_parameter() -> None:
    """One patch is the global case, and the answer must not depend on the
    number. That is only expressible because the kind says the parameter should
    not change the oracle value."""
    patch_count = next(p for p in B2_EQUIV.parameters if p.name == "patch_count")
    assert patch_count.kind is ParameterKind.REPRESENTATION
    assert patch_count.domain == (1, 4096)


def test_the_cascade_composability_invariant_is_declared_here() -> None:
    """N DOEs in series give a bounded ray count, not a product.

    Two in series give 256 then 256, not 256 x 64. Without it a multi-element
    diffractive system is combinatorially unrunnable, so it is a COMPOSABILITY
    invariant and belongs with the transitions rather than in a per-coupler
    bundle.
    """
    invariants = {i.invariant_id for i in B2_EQUIV.invariants}
    assert "OUTGOING_COUNT_IS_THE_BUDGET" in invariants
    statement = next(
        i.statement for i in B2_EQUIV.invariants if i.invariant_id == "OUTGOING_COUNT_IS_THE_BUDGET"
    )
    assert "256 x 64" in statement


def test_grid_snapping_is_declared_to_have_a_cost() -> None:
    """A representation choice with no measurable cost would mean the metric
    cannot see it, which is a finding about the metric."""
    snapping = next(p for p in B2_EQUIV.parameters if p.name == "grid_snapping")
    assert snapping.kind is ParameterKind.REPRESENTATION
    controls = {c.control_id for c in B2_EQUIV.negative_controls}
    assert "grid-snapping-is-not-free" in controls


def test_the_curvature_bound_is_a_validity_predicate_on_the_equivalence_family() -> None:
    """The SI S3 bound, executable rather than assumed.

    A declared-planarity predicate was here too and was removed while authoring
    FIXED-V1: it read a ``surface_sag_m`` and a ``planarity_tolerance_m`` this
    family does not declare, so it could not be evaluated on the family's own
    parameters. A bound nothing can check is not a bound, and the curvature
    predicate covers the same physics with parameters that exist -- an infinite
    substrate radius IS the planar case, and it fingerprints.
    """
    predicates = {p.predicate_id for p in B2_EQUIV.validity}
    assert "SI_S3_CURVATURE" in predicates

    import math

    params = {
        "aperture_width_m": 1e-3,
        "substrate_radius_m": math.inf,
        "wavelength_m": 5.32e-7,
        "patch_count": 1,
        "grid_snapping": "exact",
        "pad_width": 566,
        "patch_width_m": 1e-3,
        "tangent_plane_error_rad": 0.0,
    }
    status, margins = B2_EQUIV.evaluate_validity(params)
    assert margins["SI_S3_CURVATURE"] == 1.0, "planar admits exactly zero tangent error"
    assert status is ValidityState.INSIDE


# --------------------------------------------------------------------------- #
# The round-trip rule
# --------------------------------------------------------------------------- #


def test_a_round_trip_without_its_broken_twin_is_outside_its_own_validity() -> None:
    """M2.4's rule, as a schema-enforced condition rather than a habit.

    A round trip that cannot be made to fail proves nothing, because a shared
    convention error cancels between the two directions. Making "the twin ran"
    a validity predicate means the verifier reports OUT_OF_VALIDITY for an
    instance without one, instead of presenting its number as evidence.
    """
    params = {
        "wavelength_m": 5.32e-7,
        "numerical_aperture": 0.3,
        "grid_n": 64,
        "sample_count": 1_000_000,
        "direction": "wave_ray_wave",
        "arm": "enumerated",
        "seed": 0,
    }
    without_twin = B2_ROUNDTRIP.instantiate("no-twin", params, seed=0)
    with_twin = B2_ROUNDTRIP.instantiate(
        "with-twin", {**params, "broken_twin_ran": True}, seed=0
    )
    assert without_twin.validity_status is ValidityState.FAR_OUTSIDE
    assert with_twin.validity_status is ValidityState.INSIDE


def test_the_measured_pairing_is_recorded_with_both_numbers() -> None:
    """5.31e-16 correct against 1.414 broken. The pairing is the point, and
    either number alone says nothing.

    Both numbers moved when the round trip was actually executed: the correct arm
    from an inherited 1.32e-15 to a freshly measured 5.31e-16 (a different probe
    field and the SI eq S5 1/N the reconstruction was owed), and the twin from
    1.40 to 1.414 -- which is sqrt(2), the exact distance between a field and its
    conjugate for a probe whose spectrum is not Hermitian-symmetric. The first
    probe was a centred real Gaussian, for which both twins were no-ops.
    """
    disposition = B2_ROUNDTRIP.gate_disposition
    assert disposition is not None
    assert disposition.observed == pytest.approx(5.31117e-16, rel=1e-3)
    assert "1.414" in disposition.note
    assert "cancels between the two directions" in disposition.note


def test_the_detection_margin_is_a_number_and_cannot_be_gated_by_this_schema() -> None:
    """A control that fires by 1.1x and one that fires by 1e15 are different
    evidence, and a bare boolean cannot tell them apart -- so the margin is
    reported. What it cannot be is *gated*, and that is a schema limitation
    named here rather than worked around.

    ``MetricResult.met`` is ``measured <= tolerance`` everywhere in the verifier.
    For a quantity where LARGER IS BETTER that comparison asserts the opposite of
    the claim: gating a detection margin at ``<= 1e3`` would mark a control that
    separates by 1e15 as failing and a barely-firing one as passing. Inverting
    the number to fit the schema would hide the limitation inside a metric name,
    so ``may_gate`` is false and the under-powered-control finding is carried by
    the negative-control outcomes, where the direction is the right way round.

    The threshold is kept as the documented expectation, and a two-sided
    tolerance kind is recorded as schema follow-up rather than improvised here.
    """
    tolerance = B2_ROUNDTRIP.tolerance_for("detection_margin")
    assert tolerance is not None
    assert tolerance.may_gate is False
    assert tolerance.threshold == 1e3
    assert "LARGER IS BETTER" in tolerance.basis
    assert "undermines_the_gate" in tolerance.basis
    assert tolerance.rejects


def test_the_round_trip_metric_declares_the_error_it_cannot_see() -> None:
    """Any error that is its own inverse. The backward pass undoes whatever the
    forward pass did, so a wrong kernel round-trips perfectly -- which is the
    whole reason the twin is required."""
    metric = B2_ROUNDTRIP.metric("round_trip_relative_rms")
    assert any("OWN INVERSE" in blind for blind in metric.blind_to)
    assert METRICS["relative_rms"].blind_to


def test_the_off_axis_blindness_audit_is_a_declared_control() -> None:
    """CHE-44's concern was never audited. It is a control now: a control battery
    whose metrics are all centred cannot see an off-axis error."""
    controls = {c.control_id for c in B2_ROUNDTRIP.negative_controls}
    assert "off-axis-blindness-audit" in controls


def test_the_roundtrip_sampler_is_absent_for_a_design_reason_not_a_budget_one() -> None:
    """Generating instances is cheap here -- the reference is the input. What
    makes it non-generative is that every generated instance needs its twin
    generated too, and a sampler that emitted only the correct arm would quietly
    weaken every instance it drew."""
    assert B2_ROUNDTRIP.sampler is None
    assert "broken twin generated too" in B2_ROUNDTRIP.sampler_absent_note


# --------------------------------------------------------------------------- #
# What does not survive
# --------------------------------------------------------------------------- #


def test_every_coupler_states_what_it_discards() -> None:
    """Stated as a deliverable rather than as a caveat: an agent reasoning about
    a chain of representations needs to know which quantities it may still ask
    about downstream."""
    from registry.loader import Registry

    registry_couplers = set(Registry.from_package().couplers)
    assert registry_couplers <= set(WHAT_DOES_NOT_SURVIVE), (
        f"no discard statement for {sorted(registry_couplers - set(WHAT_DOES_NOT_SURVIVE))}"
    )
    for coupler, discarded in WHAT_DOES_NOT_SURVIVE.items():
        assert discarded, f"{coupler}: every coupler here is lossy, so something is gone"
        assert all(item.strip() for item in discarded)


def test_per_ray_correspondence_is_named_as_lost() -> None:
    """The outgoing amplitude is a spectral amplitude U~[m]/p[m], not a
    transformed incident weight, so "which incident ray became this one" has no
    answer."""
    doe = WHAT_DOES_NOT_SURVIVE["C_PLANAR_DOE_STEP"]
    assert any("U~[m]/p[m]" in item for item in doe)
    assert any("REBASED" in item for item in doe)


@pytest.mark.parametrize("family", B2, ids=lambda f: f.family_id)
def test_no_b2_family_claims_a_gate_it_has_not_measured(family: BenchmarkFamily) -> None:
    """A gate is decided by a run, and the run has to be re-runnable.

    This assertion used to read "the status must be MEASURED_OFF_GATE or
    NOT_MEASURED", which was correct while nothing had executed and became the
    wrong assertion the moment the instances did. What it was protecting is the
    rule underneath: a declared benchmark is not a measured benchmark, so a MET
    or NOT_MET verdict owes an observed number and an executable reference that
    reproduces it -- and a NOT_MEASURED one must not quietly carry a number.
    """
    disposition = family.gate_disposition
    assert disposition is not None

    if disposition.status is GateStatus.NOT_MEASURED:
        assert disposition.observed is None, (
            f"{family.family_id} says NOT_MEASURED and reports {disposition.observed}"
        )
        assert disposition.note, "an unmeasured gate owes the reason it is unmeasured"
        return

    assert disposition.observed is not None, family.family_id
    assert disposition.metric in {metric.name for metric in family.metrics}, (
        f"{family.family_id} gates on {disposition.metric!r}, which it does not declare"
    )
    node_ids = [ref for ref in disposition.evidence if "::" in ref]
    assert node_ids, (
        f"{family.family_id} claims {disposition.status.value} with no test that re-runs it: "
        f"{disposition.evidence}"
    )
    for ref in node_ids:
        path, _, node = ref.partition("::")
        assert (repository_root() / path).exists(), f"{family.family_id} cites a missing {path}"
        assert node.startswith("test_"), ref


@pytest.mark.parametrize("family", B2, ids=lambda f: f.family_id)
def test_a_gated_tolerance_exists_for_every_decided_b2_gate(family: BenchmarkFamily) -> None:
    """And it is one the family is allowed to gate on.

    A CROSS_ROUTE or SHARES_CODE oracle forces B4 and a B4 family cannot gate;
    conversely a B2 family that has decided MET must be pointing at a tolerance
    whose basis permits gating, or the verdict is resting on a number that was
    only ever meant to be reported.
    """
    disposition = family.gate_disposition
    assert disposition is not None
    if disposition.status is GateStatus.NOT_MEASURED:
        pytest.skip("nothing decided")
    tolerance = family.tolerance_for(disposition.metric)
    assert tolerance is not None, f"{family.family_id} gates on an undeclared tolerance"
    assert tolerance.basis.strip(), f"{family.family_id}: a tolerance with no basis is a number"
    if disposition.status in (GateStatus.MET, GateStatus.NOT_MET):
        assert tolerance.may_gate, (
            f"{family.family_id} decided {disposition.status.value} on "
            f"{disposition.metric!r}, whose basis says it may not gate"
        )
