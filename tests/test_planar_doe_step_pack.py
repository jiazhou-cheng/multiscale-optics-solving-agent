"""The composed coupler's knowledge pack must stay bound to what it describes.

CHE-138 (M2.3). ``C_PLANAR_DOE_STEP`` had a graph node, a capability
declaration and a registry entry, and no agent-facing pack -- so an agent
routing through it had to read the implementation. This pack closes that.

The risk a pack introduces is the one ``tests/test_coupler_knowledge_pack.py``
already names: a well-written document reads like evidence. These checks keep
the pack honest in the two ways that matter for a *composed* coupler.

**It must not restate its halves.** The whole justification for a thin pack is
that ``C_RAY_TO_WAVE`` and ``C_WAVE_TO_RAY`` own their own conventions. A pack
that copied them would create a second place to update and a second place to
drift.

**It must not out-claim the registry.** The capability is an intersection, the
maturity is ``characterized``, the derivative is an unverified surrogate, and
three of the four kinds of stochastic evidence are absent. A card is exactly the
artifact where those would quietly become "supported".
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from core.capabilities import C_PLANAR_DOE_STEP_CAPABILITIES
from core.specs import DerivativeMode, Maturity
from registry.loader import Registry

pytestmark = pytest.mark.coupler

ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT / "knowledge/couplers/planar_doe_step"
PACK_FILES = (
    "card.yaml",
    "theory.md",
    "conventions.md",
    "failure_guide.md",
    "source_manifest.yaml",
)


@pytest.fixture(scope="module")
def card() -> dict[str, Any]:
    loaded: dict[str, Any] = yaml.safe_load((PACK / "card.yaml").read_text())
    return loaded


@pytest.fixture(scope="module")
def manifest() -> dict[str, Any]:
    loaded: dict[str, Any] = yaml.safe_load((PACK / "source_manifest.yaml").read_text())
    return loaded


@pytest.mark.parametrize("filename", PACK_FILES)
def test_pack_file_exists_and_is_non_trivial(filename: str) -> None:
    path = PACK / filename
    assert path.is_file(), path
    assert len(path.read_text().strip()) > 500, path


def test_the_pack_is_where_the_discovery_api_looks_for_it() -> None:
    """A pack the routing API cannot find is a file, not a pack."""
    from discovery.api import knowledge_for

    view = knowledge_for("C_PLANAR_DOE_STEP")
    assert view.pack_root == "knowledge/couplers/planar_doe_step"
    assert view.missing == [], view.missing


# --------------------------------------------------------------------------- #
# It must not restate its halves
# --------------------------------------------------------------------------- #


def test_the_card_names_both_halves_and_points_at_their_packs(card: dict[str, Any]) -> None:
    composed = {entry["component"]: entry for entry in card["composed_from"]}
    assert set(composed) == {"C_RAY_TO_WAVE", "C_WAVE_TO_RAY"}
    for entry in composed.values():
        assert (ROOT / entry["pack"]).is_dir(), entry["pack"]
        assert entry["supplies"]


def test_the_prose_files_send_the_reader_to_the_halves_rather_than_repeating_them() -> None:
    """Each prose file must defer explicitly, at the top, not merely mention them.

    The failure this guards is a pack that grows a full copy of the accumulation
    conventions and then disagrees with the original six months later.
    """
    for filename in ("theory.md", "conventions.md"):
        head = (PACK / filename).read_text()[:800]
        assert "knowledge/couplers/ray_to_wave" in head, filename
        assert "knowledge/couplers/wave_to_ray" in head, filename


# --------------------------------------------------------------------------- #
# It must not out-claim the registry, the capability table, or the ledger
# --------------------------------------------------------------------------- #


def test_the_card_does_not_widen_the_capability(card: dict[str, Any]) -> None:
    """The capability is an intersection and the card says so, pointing at the
    module that owns it rather than restating a device or dtype list it could
    then contradict."""
    rule = card["capability_rule"]
    assert "INTERSECTION" in rule
    assert "core/capabilities.py" in rule
    # The card must not carry its own device/dtype lists at all: a second copy
    # is a second thing to drift.
    assert "devices" not in card
    assert "dtypes" not in card
    assert C_PLANAR_DOE_STEP_CAPABILITIES.component == "C_PLANAR_DOE_STEP"


def test_the_card_matches_the_registry_on_every_claim_it_repeats(card: dict[str, Any]) -> None:
    spec = Registry.from_package().couplers["C_PLANAR_DOE_STEP"]
    assert card["role"] in Registry.from_package().couplers

    assert Maturity(card["maturity"]) == spec.maturity
    assert spec.maturity == Maturity.CHARACTERIZED, (
        "a composed step with no analytic oracle of its own must not be validated"
    )
    assert DerivativeMode(card["derivative"]["mode"]) == spec.derivative.mode
    assert card["derivative"]["verified"] is spec.derivative.verified is False
    assert card["derivative"]["known_biased"] is True
    assert set(card["invariants"]) == set(spec.invariants)
    assert card["cost_model"]["scaling"] == spec.cost_model.scaling


def test_the_card_keeps_the_three_stochastic_gaps_visible(card: dict[str, Any]) -> None:
    """Only the exactness limit is established. The ledger says so and the card
    must not round it off -- an agent reading a sampled result needs to know it
    has no stated error."""
    gaps = " ".join(card["not_yet_probed"]).lower()
    for missing in ("unbiasedness", "convergence exponent", "variance"):
        assert missing in gaps, missing
    assert "cuda" in gaps


def test_every_path_and_test_the_card_names_exists(card: dict[str, Any]) -> None:
    """Prose evidence rots by pointing at things that moved."""
    for key in ("implementation_location", "graph_node", "test_location"):
        assert (ROOT / card[key]).is_file(), f"{key} -> {card[key]}"

    for entry in card["validated_probe_ids"]:
        path, _, test_name = entry.partition("::")
        assert (ROOT / path).is_file(), path
        if test_name:
            assert f"def {test_name}(" in (ROOT / path).read_text(), entry


def test_the_validity_predicate_the_card_names_is_a_real_predicate(card: dict[str, Any]) -> None:
    conditions = {entry["id"]: entry for entry in card["validity_conditions"]}
    planarity = conditions["planarity"]
    module, _, name = planarity["predicate"].partition("::")
    assert (ROOT / module).is_file()
    assert f"def {name}(" in (ROOT / module).read_text()
    # Undefined off a plane, not merely less accurate. The distinction is the
    # reason the condition is a predicate rather than a warning.
    assert "undefined" in planarity["why_it_is_not_a_soft_limit"]


def test_the_coupler_core_still_forbids_importing_either_engine(card: dict[str, Any]) -> None:
    forbidden: set[str] = set()
    for entry in card["frameworks_deliberately_avoided"]:
        forbidden.update(entry["forbidden_imports"])
    assert forbidden == {"optiland", "chromatix"}


# --------------------------------------------------------------------------- #
# The three conventions that are invisible in an intensity
# --------------------------------------------------------------------------- #


def test_the_card_carries_all_three_invisible_conventions(card: dict[str, Any]) -> None:
    """OPL rebase, spectral amplitude, and energy policy. The node's docstring
    calls these the interesting part of the declaration precisely because none
    shows up in an intensity, and a pack that dropped one would be worse than no
    pack: it would read as complete."""
    changes = card["what_changes_across_the_step"]
    assert set(changes) == {
        "optical_path_length_is_reset_to_zero",
        "amplitude_becomes_a_spectral_amplitude",
        "power_is_not_conserved_by_default",
    }
    assert changes["power_is_not_conserved_by_default"]["statement"].strip().endswith(
        "and should stay false."
    )


def test_the_failure_guide_separates_diagnosed_refusals_from_undiagnosed_ones() -> None:
    """``diagnose`` runs before execution; the core raises during it. A request
    that fails the second way is one ``validate_request`` accepted, which is a
    gap in ``diagnose`` and is flagged as ``undiagnosed_refusal``."""
    text = (PACK / "failure_guide.md").read_text()
    assert "undiagnosed_refusal" in text
    assert "`SHAPE_MISMATCH`" in text
    assert "`OPL_REFERENCE_UNVERIFIED`" in text
    # The flag must be described as marking a gap, not merely as a field name.
    assert "gap in" in text


def test_the_failure_guide_lists_the_silent_traps_the_refusals_cannot_catch() -> None:
    """The refusal table is the cheap half. The reason to load this pack is the
    list of things that run clean and are still wrong."""
    text = (PACK / "failure_guide.md").read_text()
    for trap in ("S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8"):
        assert f"### {trap} " in text, trap
    assert "single realization is not evidence" in text


# --------------------------------------------------------------------------- #
# Provenance
# --------------------------------------------------------------------------- #


def test_the_reference_implementation_is_read_never_run_and_never_copied(
    manifest: dict[str, Any],
) -> None:
    entries = [
        entry
        for entry in manifest["sources"]
        if entry.get("source_type") == "reference_implementation"
    ]
    assert len(entries) == 1
    entry = entries[0]
    assert entry["vendored"] is False
    assert entry["executed_by_this_repository"] is False
    assert entry["pinned"] is False
    assert entry.get("read_at") and entry.get("read_scope")
    assert "never executed" in entry["usage_note"]
    assert "never vendored" in entry["usage_note"]
    assert "come from the paper" in entry["usage_note"]


def test_the_manifest_stores_no_pdf_and_names_only_files_that_are_here(
    manifest: dict[str, Any],
) -> None:
    """CHE-92 removed the vendored PDFs on cost. A manifest that still declared a
    `local_copy` would point at nothing."""
    for entry in manifest["sources"]:
        assert "local_copy" not in entry, entry.get("source_type")
    for relative in manifest["internal_evidence"]:
        assert (ROOT / relative).exists(), relative


def test_the_pack_defers_to_the_registry_on_the_sampling_asymmetry(card: dict[str, Any]) -> None:
    """The uniform-primary / magnitude-secondary asymmetry is under active
    revision. A card that froze a copy of its justification would be stale the
    moment that lands, so it must point at the registry instead of restating it.
    """
    committed = card["sampling"]["primary_positions"]["committed_position"]
    assert "src/registry/couplers.yaml" in committed
    assert "read the registry" in committed.lower()
