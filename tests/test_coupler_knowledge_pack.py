"""CHE-22 — the coupler knowledge packs must stay honest about their evidence.

The M0 docs audit recorded that no document in this repository stated the
coupler's mathematics. These packs fix that. The risk they introduce is the
opposite one: a well-written pack reads like evidence. These tests keep the
distinction between "documented from the paper" and "verified here" visible.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from core.specs import DerivativeMode, Maturity
from registry.loader import Registry

pytestmark = pytest.mark.coupler

ROOT = Path(__file__).resolve().parents[1]
COUPLERS = ROOT / "knowledge/couplers"
DIRECTIONS = ("ray_to_wave", "wave_to_ray")
PACK_FILES = (
    "card.yaml",
    "theory.md",
    "conventions.md",
    "failure_guide.md",
    "source_manifest.yaml",
)
DOI = "10.1021/acsphotonics.6c00818"


def _card(direction: str) -> dict:
    return yaml.safe_load((COUPLERS / direction / "card.yaml").read_text())


def _manifest(direction: str) -> dict:
    return yaml.safe_load((COUPLERS / direction / "source_manifest.yaml").read_text())


@pytest.mark.parametrize("direction", DIRECTIONS)
@pytest.mark.parametrize("filename", PACK_FILES)
def test_pack_file_exists_and_is_non_trivial(direction: str, filename: str) -> None:
    path = COUPLERS / direction / filename
    assert path.is_file(), path
    assert len(path.read_text().strip()) > 500, path


@pytest.mark.parametrize("direction", DIRECTIONS)
def test_card_pins_the_doi_and_names_the_paper_equations(direction: str) -> None:
    card = _card(direction)
    source = card["scientific_source"]

    assert source["doi"] == DOI
    assert source["equations"], "a coupler card must name the equations it implements"


@pytest.mark.parametrize("direction", DIRECTIONS)
def test_reference_implementation_is_recorded_as_read_but_never_executed(direction: str) -> None:
    """The paper's code is public, and what this repository did with it must be exact.

    Three states are distinct and the manifest must keep them distinct:
    **vendored** (copied in), **executed** (run, so its output could become a
    number here), and **read**. It was never the first two and, since CHE-95,
    it has been the third.

    This test used to assert the entry said "NOT used", full stop. That is no
    longer true and asserting it would make the manifest lie. CHE-95 read the
    upstream implementation for its *option inventory* -- that four knobs exist
    and are worth having -- and implemented each from its physical description.
    The equations still come from the paper, which is what makes the two
    implementations agreeing on a convention corroboration rather than copying.

    So what is asserted is the pair that actually matters: not vendored, not
    executed, and the reading recorded with a date and a stated scope. A read
    with no scope is indistinguishable from a copy.
    """
    card = _card(direction)
    status = card["scientific_source"]["reference_implementation_status"]
    assert "NOT vendored" in status
    assert "NOT executed" in status

    entries = [
        entry
        for entry in _manifest(direction)["sources"]
        if entry.get("source_type") == "reference_implementation"
    ]
    assert len(entries) == 1
    entry = entries[0]
    assert entry["vendored"] is False
    assert entry["pinned"] is False
    assert entry["executed_by_this_repository"] is False
    assert entry.get("read_at"), (
        "the upstream implementation was read by CHE-95; the manifest must say "
        "when, or a future reader cannot tell a read from a copy."
    )
    assert entry.get("read_scope"), "a read with no recorded scope is not a record of anything"
    note = entry["usage_note"]
    assert "never executed" in note and "never vendored" in note
    assert "come from the paper" in note, (
        "the manifest must still say that the equations came from the published "
        "text, which is the claim the whole entry exists to protect."
    )


def test_the_patch_step_card_section_names_things_that_exist() -> None:
    """A card section describing an operator has to be bound to that operator.

    CHE-96 added `patch_step` to the wave_to_ray card and a pointer to it from
    the ray_to_wave card. The failure mode this guards is the ordinary one for
    prose evidence: the section keeps reading correctly after the module it
    describes has been renamed, the node it names has been deleted, or the
    coupler id has drifted from the registry's.

    What is checked is the load-bearing part -- the paths resolve, the coupler
    id is registered, and the three refusals the section claims are actually
    the three the implementation makes. The measured numbers are not re-derived
    here; they belong to `tests/test_patch_wft.py` and to the probe records,
    and duplicating them would create a second place to update.
    """
    card = _card("wave_to_ray")
    section = card["patch_step"]
    assert section["coupler_id"] == "C_PATCH_WFT"
    assert section["coupler_id"] in Registry.from_package().couplers

    for key in ("implementation", "graph_node"):
        path = section[key].split("::")[0]
        assert (ROOT / path).is_file(), f"{key} names {path}, which does not exist"

    assert set(section["three_refusals"]) == {
        "even_patch_px",
        "derived_pad",
        "coverage_basis",
    }

    pointer = _card("ray_to_wave")["patch_step_reference"]
    assert pointer == "knowledge/couplers/wave_to_ray/card.yaml::patch_step"
    referenced_file, _, key = pointer.partition("::")
    assert (ROOT / referenced_file).is_file()
    assert key in card, "the pointer names a section the target card does not have"


@pytest.mark.parametrize("direction", DIRECTIONS)
def test_vendored_reference_data_is_a_separate_entry_that_names_real_files(
    direction: str,
) -> None:
    """Vendored data and read-but-not-vendored code are different claims (CHE-96).

    CHE-96 committed two SLM phase masks from the paper's reference
    implementation. They are inputs to the Fig 5b/5c reproductions and cannot be
    regenerated here, so vendoring them is right -- but recording them under the
    existing `reference_implementation` entry would flip its `vendored` flag to
    true and destroy the claim that entry exists to make. Hence a second entry
    with its own `source_type`, and hence this test, which asserts the two stay
    distinct rather than merely that both exist.

    The files are checked to be present because a manifest entry naming a path
    that is not there records nothing. Pinning is checked because a vendored
    input that changes silently makes every number measured from it
    unreproducible.
    """
    entries = [
        entry
        for entry in _manifest(direction)["sources"]
        if entry.get("source_type") == "reference_data"
    ]
    assert len(entries) == 1
    entry = entries[0]
    assert entry["vendored"] is True, "these arrays ARE copied in; the entry must say so"
    assert entry["executed_by_this_repository"] is False, "data is not executed"
    assert entry["pinned"] is True and entry.get("pinned_revision")
    files = entry["files"]
    assert files, "an entry that names no file records nothing"
    for relative in files:
        assert (ROOT / relative).is_file(), f"{relative} is recorded but absent"

    implementation = [
        entry
        for entry in _manifest(direction)["sources"]
        if entry.get("source_type") == "reference_implementation"
    ]
    assert implementation[0]["vendored"] is False, (
        "vendoring the data must not have been recorded as vendoring the code"
    )


@pytest.mark.parametrize("direction", DIRECTIONS)
def test_validation_status_matches_the_probes_the_card_actually_lists(direction: str) -> None:
    """The validation-status ladder only means anything if a card cannot claim a
    rung it has no evidence for, and cannot sit on a lower rung than its own
    evidence while quietly being used as if validated."""
    card = _card(direction)
    status = card["validation_status"]
    probes = card["validated_probe_ids"]

    if status == "unvalidated":
        assert probes == [], "an unvalidated card must not list passing probes"
        assert card["devices_tested"] == []
        assert card["implementation_location"] == "not_yet_implemented"
    else:
        assert probes, f"{direction} claims {status!r} with no probe evidence"
        assert card["devices_tested"], "a validated card must name the devices it ran on"
        assert (ROOT / card["implementation_location"]).is_file()
        assert (ROOT / card["test_location"]).is_file()

    # Regardless of rung: unprobed regimes stay listed, and no gradient is
    # claimed anywhere in M2 without the evidence coupler_protocol.yaml names.
    assert card["not_yet_probed"]
    assert card["derivative"]["verified"] is False


@pytest.mark.parametrize("direction", DIRECTIONS)
def test_coupler_core_forbids_importing_either_engine(direction: str) -> None:
    card = _card(direction)
    forbidden = set()
    for entry in card["frameworks_deliberately_avoided"]:
        forbidden.update(entry["forbidden_imports"])
    assert forbidden == {"optiland", "chromatix"}


def test_ray_to_wave_refuses_the_two_inputs_m1_marked_unverified() -> None:
    """M1 recorded that Optiland's opd sign/reference are unverified and that
    its `intensity` is a weight rather than a complex amplitude. A coupler that
    silently accepted either would launder an unverified quantity into a phase
    or an amplitude."""
    refused = {entry["name"]: entry["reason"] for entry in _card("ray_to_wave")["deliberately_refused"]}

    assert set(refused) == {"opd_native", "ray_intensity_as_amplitude"}
    assert "sign" in refused["opd_native"]
    assert "intensity_is_not_amplitude" in refused["ray_intensity_as_amplitude"]

    conventions = (COUPLERS / "ray_to_wave/conventions.md").read_text()
    for hazard in ("H1", "H2", "H3"):
        assert hazard in conventions


def test_wave_to_ray_records_the_estimator_as_knowingly_biased() -> None:
    """SI S7.2 says plainly that the estimator neglects direction gradients.
    The card must carry that, and must keep the surrogate derivative and the
    true derivative as two separate quantities."""
    derivative = _card("wave_to_ray")["derivative"]

    assert derivative["verified"] is False
    assert derivative["known_biased"] is True
    assert derivative["omitted_terms"]
    assert len(derivative["two_quantities_not_to_conflate"]) == 2

    rejected = derivative["rejected_alternative"]
    assert rejected["id"] == "gumbel_softmax"
    assert rejected["m2_status"] == "documented, not implemented"
    # The Gumbel-max identity leaves forward sampling statistics unchanged; only
    # the backward pass is a surrogate. Conflating the two would misstate S7.3.
    assert rejected["forward_bias"].startswith("none")
    assert "biased surrogate" in rejected["backward"]


def test_wave_to_ray_registration_is_not_a_capability_claim() -> None:
    """The M1 independence check asserted C_WAVE_TO_RAY did *not* exist.

    CHE-23 registered the coupler, so that check could no longer hold. It was
    replaced rather than deleted, because the property it protected still
    matters: the registry must not claim a wave->ray capability that has not
    been established.

    CHE-88 archived `verify_m1_independence.py` with the rest of the gen1
    suite. This test used to assert that the four replacement checks appeared
    *as strings* in that script, which was already the weaker form of the
    question -- and archiving the script would have silently retired the audit a
    second time, in exactly the way the card warns about.

    So the four claims are asserted here, against the packaged registry, where
    they are properties rather than source text. That survives the archival and
    is what the card's `registry_claims_now_audited` list now points at.
    """
    card = _card("wave_to_ray")
    note = card["registry_note"]
    assert "wave_to_ray_not_claimed" in note
    assert "REPLACED" in note

    spec = Registry.from_package().couplers["C_WAVE_TO_RAY"]

    # `maturity` was `experimental` when the replacement checks were written and
    # is `characterized` since CHE-87, which set the field from what is actually
    # measured. The claim the check protects is the ceiling, not the exact
    # rung: a Monte Carlo estimator with no certified gradient is not validated.
    assert spec.maturity != Maturity.VALIDATED

    assert spec.derivative.verified is False
    assert spec.derivative.mode == DerivativeMode.SURROGATE
    assert spec.lossy is True

    audited = set(card["registry_claims_now_audited"])
    assert audited == {
        "wave_to_ray_not_validated",
        "wave_to_ray_gradient_unverified",
        "wave_to_ray_gradient_mode_is_surrogate",
        "wave_to_ray_declared_lossy",
    }, (
        "the card lists claims this test does not audit, or vice versa. Both "
        "must move together -- a claim listed as audited and checked by nothing "
        "is worse than an unlisted one."
    )


def test_curvature_bound_is_recorded_with_its_assumptions() -> None:
    conditions = {
        entry["id"]: entry for entry in _card("wave_to_ray")["validity_conditions"]
    }
    curvature = conditions["curvature_bound"]

    assert "arcsin" in curvature["statement"]
    assert curvature["independent_of"] == "the DOE phase profile"
    # A bound stated without its assumptions is not a bound.
    assert len(curvature["assumptions"]) == 3


def test_papers_policy_exception_is_documented_rather_than_contradicted() -> None:
    """The rule and its exception must both be written down, and agree.

    `knowledge/README.md` forbids storing copyrighted full papers. The owner is
    this paper's first author, which is the recorded exception -- so storing
    them *was* permitted, and this test used to assert the two PDFs were
    present.

    CHE-92 removed them anyway, on cost rather than permission: 11.5 MB of
    binary in a `retrieval_only` directory is paid for by every clone, for
    something no agent reads. The exception still stands and is still recorded,
    because a future contributor should not have to re-derive it.

    So what is asserted flipped, and the flip is the point: the policy, the
    exception, and the DOI must all be findable, and the bytes must **not** be
    here. A rule that says "no PDFs" with a PDF beside it is worse than either.
    """
    readme = (ROOT / "knowledge/README.md").read_text()
    assert "do not store copyrighted full papers" in readme.lower()
    assert "exception" in readme.lower()
    assert "raywave_tracing" in readme

    paper_readme = ROOT / "knowledge/papers/raywave_tracing/README.md"
    assert paper_readme.is_file()
    paper_text = paper_readme.read_text()
    assert DOI in paper_text
    # The exception is recorded, not silently dropped along with the files.
    assert "first author" in paper_text

    stored = sorted(p.name for p in (ROOT / "knowledge").rglob("*.pdf"))
    assert not stored, (
        f"knowledge/ holds {stored}. The directory is disclosed to an agent and "
        "marked retrieval_only; a PDF there is cost without a reader. The paper "
        "is retrievable from the DOI recorded in papers/raywave_tracing/README.md, "
        "and every claim taken from it is stated in the coupler packs with its "
        "section cited, so no claim depends on the file being present."
    )



def test_ray_to_wave_card_carries_che50s_decision_and_names_who_it_affects() -> None:
    """CHE-68 (R5). A closed decision that lives only in a Linear comment is not
    a documented limitation. The card must say what is missing, who it breaks,
    what the correct remedy is today, and -- because PB7 looked and did not see
    it -- why PB7's silence is not evidence of absence."""
    limitation = _card("ray_to_wave")["known_limitations"]["no_wavefront_curvature_term"]

    assert limitation["issue"] == "CHE-50"
    # Tracked, not promoted to a defect and not quietly dropped.
    assert "tracked known limitation" in limitation["status"]
    assert "no kernel change" in limitation["decided"]

    # The statement has to name the term and the scope of validity, or a reader
    # cannot tell whether their own use is affected.
    assert "exp(i k r^2 / 2R)" in limitation["statement"]
    assert "further" in limitation["statement"]

    # A limitation with no remedy and no revisit condition is a shrug.
    assert "advance the RAY STATE" in limitation["correct_remedy_today"]
    assert limitation["revisit_when"]

    # PB7 is the trap: three PSF routes agreed and none of them could have seen
    # this. Recording that keeps the next reader from citing PB7 as clearance.
    assert "PB7" in limitation["who_is_not_a_witness"]

    # Every surface named must exist, or the card is pointing at nothing.
    for surface in limitation["surfaced_in"]:
        path = surface.split(" ")[0]
        if path != "this":
            assert (ROOT / path).is_file(), surface


def test_ray_to_wave_card_does_not_present_the_missing_record_as_evidence() -> None:
    """CHE-63's record has never been generated. The card may cite it as the
    intended evidence path, but not without saying it is absent -- otherwise a
    reader takes a filename for a measurement."""
    text = (COUPLERS / "ray_to_wave/card.yaml").read_text()
    record = "benchmarks/probes/records/m3r_sensor_handoff.json"

    assert not (ROOT / record).exists(), (
        "the record now exists; drop this test and the NOT COMMITTED notes on the card"
    )
    assert record in text
    assert "NOT COMMITTED" in text
    assert "CHE-63" in text


def test_the_near_grazing_cancellation_hazard_is_recorded_with_its_numbers() -> None:
    """CHE-70's measured defect, on the card and in the conventions, or nowhere.

    The hazard is *open*: the kernel loses the phase of near-grazing modes and does
    not warn, and CHE-70 handled it at the caller with a declared band limit rather
    than changing verified numerics. That disposition is only honest if a consumer
    can read it, so the card must carry the measurement (not a summary of it), the
    conventions must carry the hazard under a heading, and the derivation must sit
    in the module that applies it.
    """
    limitation = _card("ray_to_wave")["known_limitations"][
        "near_grazing_phase_cancellation"
    ]
    assert limitation["issue"] == "CHE-70"
    assert limitation["status"].startswith("open")
    measured = limitation["measured_impact"]
    assert measured["offending_bins"].startswith("8")
    assert measured["offending_optical_path_m"] == pytest.approx(4745.3, rel=1e-3)
    assert measured["offending_power_fraction"] == pytest.approx(2.25e-7, rel=1e-2)
    # Both sides of the measurement, so the card cannot state the good number alone.
    assert "2.8e-09" in measured["exactness_limit_without_band_limit"]
    assert "8.9e-14" in measured["exactness_limit_with_band_limit"]
    assert "not a kernel change" in limitation["current_handling"].lower()

    conventions = (COUPLERS / "ray_to_wave" / "conventions.md").read_text()
    assert "### H4" in conventions
    assert "4745 m" in conventions
    assert "Four hazards" in conventions, (
        "the hazard count in the heading must match the hazards below it"
    )

    streaming = (
        Path(__file__).resolve().parents[1]
        / "src/couplers/streaming.py"
    ).read_text()
    assert "grazing_floor_for_phase_budget" in streaming
    assert "eps * k * Z / d_n" in streaming, (
        "the derivation belongs next to the code that applies it, not only in a report"
    )
