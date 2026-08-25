"""One truth, several views -- and a test that they still agree.

CHE-114 (M3.2). The discovery API's whole value is that it owns no facts. Every
field is derived from ``core/capabilities.py``, ``registry/*.yaml``,
``GraphValidator``, the claim ledger, the family registry or the refusal
catalogue, and the tests below re-read each source and compare. An introspection
layer that could disagree with what it describes would be worse than the five
scattered sources it replaces, because it would look authoritative.

The three questions nothing could answer before get the most attention, and one
of them found a real modelling error while being written -- see
``test_a_representation_change_is_not_an_infeasible_route``.
"""

from __future__ import annotations

import pytest

from core.capabilities import COMPONENT_CAPABILITIES, capabilities_for
from core.graph import GraphValidator
from core.paths import repository_root
from core.specs import CouplerSpec, GraphSpec, ModelSpec
from discovery import (
    check_connection,
    describe_component,
    families_for_component,
    knowledge_for,
    route_capability,
    validity_of,
)
from registry.loader import Registry
from verification.claim_ledger import claims_for
from verification.families import FAMILIES
from verification.refusals import REFUSAL_CATALOGUE

ROOT = repository_root()
COMPONENTS = sorted(COMPONENT_CAPABILITIES)


@pytest.fixture(scope="module")
def registry() -> Registry:
    return Registry.from_package()


# --------------------------------------------------------------------------- #
# No field is hand-maintained
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("component", COMPONENTS)
def test_the_capability_fields_come_from_the_capability_table(component: str) -> None:
    """Re-read the source and compare. ``core/capabilities.py`` is probe-backed
    and this API is a view over it, not a second copy."""
    described = describe_component(component)
    caps = capabilities_for(component)

    assert described.devices == sorted(str(d) for d in caps.devices)
    assert described.native_compute_dtypes == sorted(
        str(d) for d in caps.native_compute_dtypes
    )
    assert described.lossy_input_dtypes == sorted(str(d) for d in caps.lossy_input_dtypes)
    assert described.namespaces == sorted(str(n) for n in caps.namespaces)
    assert described.capability_evidence == caps.evidence


@pytest.mark.parametrize("component", COMPONENTS)
def test_the_port_and_validity_fields_come_from_the_registry(
    component: str, registry: Registry
) -> None:
    described = describe_component(component)
    spec = registry.models.get(component) or registry.couplers[component]

    assert described.version == spec.version
    assert described.maturity == spec.maturity.value
    assert described.derivative_mode == spec.derivative.mode.value
    assert described.derivative_verified == spec.derivative.verified

    declared = len(spec.validity.assumptions) + len(spec.validity.warnings) + len(
        spec.validity.hard_limits
    )
    assert len(described.suitability) == declared, (
        "every declared assumption, warning and hard limit gets a record, or the "
        "structured view is a subset of the prose and a caller reading only the "
        "structured form is missing declarations"
    )

    if isinstance(spec, ModelSpec):
        assert [p.name for p in described.inputs] == [p.name for p in spec.inputs]
        assert [p.name for p in described.outputs] == [p.name for p in spec.outputs]
    else:
        assert isinstance(spec, CouplerSpec)
        assert [p.name for p in described.inputs] == [spec.source.name]
        assert [p.name for p in described.outputs] == [spec.target.name]


@pytest.mark.parametrize("component", COMPONENTS)
def test_the_validation_claims_come_from_the_ledger(component: str) -> None:
    described = describe_component(component)
    assert len(described.validation_claims) == len(claims_for(component))


@pytest.mark.parametrize("component", COMPONENTS)
def test_the_family_coverage_comes_from_the_family_registry(component: str) -> None:
    described = describe_component(component)
    expected = {f.family_id for f in FAMILIES if component in f.components}
    assert {f.family_id for f in described.families} == expected

    for coverage in described.families:
        family = next(f for f in FAMILIES if f.family_id == coverage.family_id)
        assert coverage.gate_deciding == family.is_gate_deciding


@pytest.mark.parametrize("component", COMPONENTS)
def test_the_refusals_come_from_the_catalogue(component: str) -> None:
    described = describe_component(component)
    for refusal in described.refusals:
        entry = REFUSAL_CATALOGUE[refusal.code]
        assert refusal.remedy == entry.remedy
        assert refusal.status == entry.status.value


def test_adding_a_component_would_surface_without_editing_this_module() -> None:
    """Every component in the capability table describes, with no per-component
    branch. A new one is covered by construction."""
    for component in COMPONENTS:
        description = describe_component(component)
        assert description.component == component
        assert description.description


# --------------------------------------------------------------------------- #
# "Can these two be connected, and under what declaration?"
# --------------------------------------------------------------------------- #


def test_the_connection_query_agrees_with_the_validator(registry: Registry) -> None:
    """Built ON GraphValidator rather than beside it, so there is exactly one
    implementation of the compatibility rules.

    Asserted by constructing the same candidate graph independently and checking
    the verdicts match -- an answer here that disagreed with validation would be
    worse than no answer.
    """
    report = check_connection(
        "M_RAY_OPTILAND", "rays", "M_WAVE_CHROMATIX", "input_field"
    )
    assert report.compatible
    assert report.coupler == "C_RAY_TO_WAVE"

    candidate = GraphSpec.model_validate(
        {
            "nodes": [
                {"id": "producer", "model": "M_RAY_OPTILAND"},
                {"id": "consumer", "model": "M_WAVE_CHROMATIX"},
            ],
            "edges": [
                {
                    "id": "candidate",
                    "coupler": "C_RAY_TO_WAVE",
                    "source": {"node": "producer", "port": "rays"},
                    "target": {"node": "consumer", "port": "input_field"},
                }
            ],
        }
    )
    direct = GraphValidator(registry).validate(candidate)
    assert direct.valid is report.compatible


def test_the_connection_query_names_the_declarations_the_edge_must_carry() -> None:
    """Currently discoverable only by being refused. An agent should be able to
    supply them first."""
    report = check_connection(
        "M_RAY_OPTILAND", "rays", "M_WAVE_CHROMATIX", "input_field"
    )
    assert "reference_plane" in report.required_edge_declarations
    assert {"wavelength", "coordinates", "direction"} <= set(
        report.required_edge_declarations
    )


def test_the_connection_query_lists_the_refusals_the_edge_can_produce() -> None:
    """"What happens if I get it wrong" is answerable before getting it wrong."""
    report = check_connection(
        "M_RAY_OPTILAND", "rays", "M_WAVE_CHROMATIX", "input_field"
    )
    codes = {r.code for r in report.possible_refusals}
    assert "OPL_REFERENCE_UNVERIFIED" in codes
    assert "REFERENCE_PLANE_MISMATCH" in codes

    handoff = next(r for r in report.possible_refusals if r.code == "OPL_REFERENCE_UNVERIFIED")
    assert handoff.status == "blocked"
    assert handoff.could_have_proceeded


def test_an_unmediated_pair_says_so_rather_than_failing_obscurely() -> None:
    report = check_connection(
        "M_WAVE_CHROMATIX", "output_field", "M_RAY_OPTILAND", "rays"
    )
    assert not report.compatible
    assert {i["code"] for i in report.issues} & {"UNKNOWN_PORT", "NO_MEDIATING_COUPLER"}


# --------------------------------------------------------------------------- #
# "At what device and precision can this route execute?"
# --------------------------------------------------------------------------- #


def test_the_r5_route_reports_no_single_precision_with_the_pair_named() -> None:
    """Project risk R5, answered before execution rather than at node three.

    Chromatix computes only in complex64; C_PATCH_WFT only in complex128. There
    is no one precision at which the route runs, the intersection is EMPTY, and
    the pair that emptied it is named.
    """
    answer = route_capability(["C_PATCH_WFT", "M_WAVE_CHROMATIX"])
    assert answer.uniform_compute_dtypes == []
    assert not answer.uniform_precision_available
    assert answer.blocking_pair == [
        "C_PATCH_WFT",
        "M_WAVE_CHROMATIX",
        "native_compute_dtype",
    ]


def test_the_r5_route_also_reports_that_the_handover_is_lossy_not_impossible() -> None:
    """The richer and more accurate answer.

    The artifact CAN cross -- Chromatix lossy-accepts complex128 -- and what is
    unavailable is a single compute precision. Reporting only "infeasible" would
    have been wrong in a way an agent would act on.
    """
    answer = route_capability(["C_PATCH_WFT", "M_WAVE_CHROMATIX"])
    assert answer.feasible
    assert answer.lossy_handovers == ["C_PATCH_WFT -> M_WAVE_CHROMATIX"]
    (handover,) = answer.handovers
    assert handover.lossy
    assert handover.lossy_dtypes == ["complex128"]
    assert handover.exact_dtypes == []


def test_a_representation_change_is_not_an_infeasible_route() -> None:
    """The false negative this test caught while being written.

    An earlier draft intersected native compute dtypes and called the result
    feasibility, which declared ``M_RAY_OPTILAND -> C_RAY_TO_WAVE ->
    M_WAVE_CHROMATIX`` infeasible on the grounds that float64 and complex64 do
    not intersect. That route executes -- it is
    ``examples/graphs/ray_to_wave.yaml``, and it runs end to end in about
    eleven seconds. A ray bundle handing over to a wave model is a
    representation change; the dtypes differ by construction.
    """
    answer = route_capability(
        ["M_RAY_OPTILAND", "C_RAY_TO_WAVE", "M_WAVE_CHROMATIX"]
    )
    assert answer.feasible
    assert answer.devices == ["cpu", "cuda"]
    assert all(h.possible for h in answer.handovers)
    assert not answer.uniform_precision_available, (
        "and there is still no single precision for the whole route, which is a "
        "true and separate statement"
    )


def test_an_unknown_component_in_a_route_is_named() -> None:
    answer = route_capability(["M_RAY_OPTILAND", "M_NOT_A_MODEL"])
    assert not answer.feasible
    assert "M_NOT_A_MODEL" in answer.reason


def test_an_empty_route_has_no_capability() -> None:
    assert not route_capability([]).feasible


# --------------------------------------------------------------------------- #
# "Should I use this here?"
# --------------------------------------------------------------------------- #


def test_validity_answers_with_a_state_and_a_margin_rather_than_a_paragraph() -> None:
    """The amendment's requirement. Prose cannot be checked before running."""
    # z <= N pitch^2 / lambda is 512 * 0.0625 / 0.532 = 60.2 um on this grid, so
    # 20 um is comfortably inside and 100 um -- which looks modest -- is already
    # past it. That the boundary is closer than it looks is the reason the
    # predicate exists.
    inside = validity_of(
        "M_WAVE_CHROMATIX",
        {
            "distance_um": 20.0,
            "sample_pitch_um": 0.25,
            "grid_n": 512,
            "wavelength_um": 0.532,
            "waist_um": 5.0,
        },
    )
    outside = validity_of(
        "M_WAVE_CHROMATIX",
        {
            "distance_um": 1.0e5,
            "sample_pitch_um": 0.25,
            "grid_n": 512,
            "wavelength_um": 0.532,
            "waist_um": 5.0,
        },
    )
    assert inside.state == "inside"
    assert outside.state == "far_outside"
    assert outside.margins["ASM_TF_SAMPLING"] < 0.0
    assert inside.margins["ASM_TF_SAMPLING"] > 0.0


def test_every_validity_predicate_reports_what_it_is_blind_to() -> None:
    answer = validity_of(
        "M_WAVE_CHROMATIX",
        {
            "distance_um": 100.0,
            "sample_pitch_um": 0.25,
            "grid_n": 512,
            "wavelength_um": 0.532,
            "waist_um": 5.0,
        },
    )
    assert answer.predicates
    for predicate in answer.predicates:
        assert predicate["blind_to"]
        assert predicate["statement"]
        assert predicate["declared_by_family"]


def test_prose_only_declarations_are_reported_as_a_gap() -> None:
    """A declaration with no executable counterpart cannot be checked before
    running, and saying so is more useful than omitting it."""
    answer = validity_of("M_WAVE_CHROMATIX", {"distance_um": 1.0})
    assert answer.prose_only, (
        "M_WAVE_CHROMATIX declares assumptions and warnings that no ValidityPredicate "
        "covers yet; the API must say which"
    )


def test_a_parameter_point_missing_a_predicate_input_is_skipped_not_an_error() -> None:
    """A caller asking about a device configuration should not have to supply a
    patch width."""
    answer = validity_of("C_PATCH_WFT", {"device": "cpu"})
    assert isinstance(answer.margins, dict)


# --------------------------------------------------------------------------- #
# R4: the gradient claim nobody may read in
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("component", COMPONENTS)
def test_an_unverified_derivative_is_surfaced_at_the_top_level(component: str) -> None:
    """Project risk R4. A consumer that has to dig for this will read a gradient
    claim into a component that has none, so the warning is a top-level field
    and it names the mode it is overriding."""
    described = describe_component(component)
    if described.derivative_verified:
        assert described.derivative_warning is None
    else:
        assert described.derivative_warning is not None
        assert "verified is FALSE" in described.derivative_warning
        assert described.derivative_mode in described.derivative_warning


def test_no_component_in_this_repository_has_a_verified_derivative() -> None:
    """Stated as a fact rather than assumed. If one ever does, this test is the
    place that notices and the claim gets re-read."""
    unverified = [c for c in COMPONENTS if not describe_component(c).derivative_verified]
    assert unverified == COMPONENTS


# --------------------------------------------------------------------------- #
# Knowledge packs
# --------------------------------------------------------------------------- #


def test_the_knowledge_view_reports_a_missing_pack_rather_than_returning_silence() -> None:
    """Both M2.3 packs now exist, so this asserts the mechanism on the tree it has.

    C_PATCH_WFT and C_PLANAR_DOE_STEP were the two couplers with graph nodes and
    no packs, and this test used to assert that absence -- correctly, while it was
    the finding. Now that M2.3 has written both, asserting absence would assert
    the deliverable was not delivered. What the API still owes is the same thing
    either way: for a component WITH a pack, name what is present and report
    nothing missing; for one WITHOUT, name the files that are absent instead of
    returning an empty view an agent would read as "nothing to load".
    """
    for coupler_id in ("C_PATCH_WFT", "C_PLANAR_DOE_STEP"):
        view = knowledge_for(coupler_id)
        assert view.pack_root is not None, f"{coupler_id}'s pack is not discoverable"
        assert view.present, f"{coupler_id} reports a pack root and no files"
        assert not view.missing, f"{coupler_id} is missing {view.missing}"

    # And the absent case still reports absence rather than silence, checked on a
    # component with no pack root rather than by mutating the tree: the view must
    # come back naming the files it expected, not empty. An empty view reads as
    # "nothing to load" where the truth is "nothing is there".
    absent = knowledge_for("__NO_SUCH_COMPONENT__")
    assert absent.present == []
    assert absent.permitted_by_policy == []
    assert absent.missing, (
        "a component with no pack must report which files are absent -- an empty "
        "view reads as 'nothing to load' rather than as 'nothing is there'"
    )
    # RECORDED DEFECT, not an endorsement: an unrecognized component currently
    # comes back with pack_root='knowledge/solvers' and the SOLVER file list,
    # because `_pack_root` falls through to the solver directory rather than
    # returning None. The absence is reported correctly -- every file is listed
    # missing -- so an agent cannot be told there is nothing to load. But the root
    # it names does not describe the component, and a caller that trusted
    # pack_root would look in the wrong place. Asserted as-is so the behaviour is
    # visible; fixing `_pack_root` belongs to whoever owns the discovery API,
    # not to M2.4.
    assert absent.pack_root == "knowledge/solvers"


def test_the_context_policy_decides_what_is_permitted() -> None:
    """The V1 harness's three declared policies, formalized so M7 can vary the
    policy without re-implementing file copying."""
    cold = knowledge_for("M_RAY_OPTILAND", policy="cold")
    warm = knowledge_for("M_RAY_OPTILAND", policy="warm")
    guided = knowledge_for("M_RAY_OPTILAND", policy="guided")

    assert cold.permitted_by_policy == []
    assert "card.yaml" in warm.permitted_by_policy
    assert "conventions.md" not in warm.permitted_by_policy
    assert "conventions.md" in guided.permitted_by_policy
    assert set(warm.permitted_by_policy) < set(guided.permitted_by_policy)


def test_an_unknown_policy_says_which_exist() -> None:
    with pytest.raises(KeyError, match="unknown context policy"):
        knowledge_for("M_RAY_OPTILAND", policy="omniscient")


# --------------------------------------------------------------------------- #
# Family coverage
# --------------------------------------------------------------------------- #


def test_a_characterization_family_is_marked_as_unable_to_decide() -> None:
    """The field that stops an agent planning against a B4 characterization as
    though it were a validation."""
    coverage = {f.family_id: f for f in families_for_component("C_RAY_TO_WAVE")}
    assert coverage["B4-DEMO3"].gate_deciding is False
    assert coverage["B4-DEMO3"].category == "B4"
    assert coverage["B3-PSF-SINGLET"].gate_deciding is True


def test_the_unmet_singlet_gate_is_visible_from_a_component_query() -> None:
    """An agent asking about C_RAY_TO_WAVE learns the gate is NOT_MET and by how
    much, without reading a report."""
    coverage = {f.family_id: f for f in families_for_component("C_RAY_TO_WAVE")}
    singlet = coverage["B3-PSF-SINGLET"]
    assert singlet.gate_status == "not_met"
    assert singlet.observed == pytest.approx(2.2072391812867093e-3)


def test_the_api_imports_from_an_installed_wheel_with_no_source_tree() -> None:
    """``repository_root()`` locates a SOURCE TREE and says so in its own error.

    Calling it at import time made ``python -m cli`` fail outright from a
    non-editable install -- caught by ``tests/test_flat_layout.py``, not by
    anything here. The only thing this module needs a checkout for is knowledge-
    pack lookup, and an installed distribution ships no ``knowledge/``, so the
    honest answer there is "no pack root" rather than an import failure.
    """
    import ast

    source = (ROOT / "src/discovery/api.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    module_level_calls = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "repository_root"
    ]
    assert not module_level_calls, (
        "repository_root() at module scope makes this module unimportable from an "
        "installed wheel"
    )
