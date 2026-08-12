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

ROOT = Path(__file__).resolve().parents[1]
COUPLERS = ROOT / "knowledge/couplers"
DIRECTIONS = ("ray_to_wave", "wave_to_ray")
PACK_FILES = (
    "coupler_card.yaml",
    "theory.md",
    "conventions.md",
    "failure_guide.md",
    "source_manifest.yaml",
)
DOI = "10.1021/acsphotonics.6c00818"


def _card(direction: str) -> dict:
    return yaml.safe_load((COUPLERS / direction / "coupler_card.yaml").read_text())


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
def test_reference_implementation_is_recorded_as_unused(direction: str) -> None:
    """The paper's code exists and is public. Nothing here may cite it as
    evidence, because this repository neither pins nor executes it."""
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


def test_wave_to_ray_records_the_independence_check_it_replaced() -> None:
    """verify_m1_independence.py used to assert C_WAVE_TO_RAY did NOT exist.
    Registering it retired that check, so the replacements must actually be
    present -- otherwise a claim audit silently got weaker."""
    card = _card("wave_to_ray")
    note = card["registry_note"]
    assert "wave_to_ray_not_claimed" in note
    assert "REPLACED" in note

    verifier = (ROOT / "benchmarks/verify_m1_independence.py").read_text()
    assert '"wave_to_ray_not_claimed": "C_WAVE_TO_RAY" not in by_id' not in verifier
    for claim in card["registry_claims_now_audited"]:
        assert f'"{claim}"' in verifier, claim


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
    """knowledge/README.md forbids storing copyrighted full papers. Two full
    PDFs now live under knowledge/papers/. Either the rule or the exception has
    to be written down."""
    readme = (ROOT / "knowledge/README.md").read_text()
    assert "do not store copyrighted full papers" in readme.lower()
    assert "exception" in readme.lower()
    assert "raywave_tracing" in readme

    paper_readme = ROOT / "knowledge/papers/raywave_tracing/README.md"
    assert paper_readme.is_file()
    assert DOI in paper_readme.read_text()

    for name in ("paper.pdf", "supporting_information.pdf"):
        assert (ROOT / "knowledge/papers/raywave_tracing" / name).is_file()
