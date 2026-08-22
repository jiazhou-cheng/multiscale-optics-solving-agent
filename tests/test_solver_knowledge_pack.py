"""The solver knowledge packs must stay honest about their evidence.

`tests/test_coupler_knowledge_pack.py` has guarded the coupler packs since
CHE-22. There was no solver equivalent, and the solver cards were the ones that
drifted — three disagreements between the flat and deep card tiers, every one of
them *understating* what had been verified. Understatement is the harder failure
to notice, because nothing goes wrong; work is just redone.

CHE-92 removed the second card tier, which removes the place those three
disagreements lived. This file keeps the remaining ways a card can lie:

* by restating a device or dtype table that `core/capabilities.py` owns, which
  is a third copy that can only drift;
* by inventing a `validation_status` outside the three-value ladder, which makes
  the field unreadable by anything;
* by leaving a cleared item on `not_yet_probed`, which `knowledge/README.md`
  makes a gate on unattended execution — so a stale entry either blocks
  validated work or teaches the reader to discount the list;
* by pointing at evidence that is not there.

Static and cheap: YAML and the file tree, no solver import.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SOLVERS = ROOT / "knowledge" / "solvers"
NAMES = ("optiland", "chromatix")

#: The three values `knowledge/README.md` defines, and no others.
LADDER = ("unvalidated", "environment_verified", "scientifically_validated")

#: Field names that used to hold a restated device/dtype table. Each is now
#: owned by `core/capabilities.py`; a card that reintroduces one has created a
#: copy nothing checks.
FORBIDDEN_RESTATEMENTS = (
    "devices_tested",
    "devices_reported_by_package",
    "dtypes_validated_for_m1",
    "dtypes_tested",
    "precision_verdict_che61",
    "supported_dtypes",
    "supported_devices",
)


def _card(name: str) -> dict[str, Any]:
    return yaml.safe_load((SOLVERS / name / "card.yaml").read_text(encoding="utf-8"))


@pytest.mark.parametrize("name", NAMES)
def test_there_is_exactly_one_card(name: str) -> None:
    """One card per component, and no second tier to disagree with it."""
    pack = SOLVERS / name
    cards = sorted(p.name for p in pack.glob("*card*.yaml"))
    assert cards == ["card.yaml"], (
        f"{name} has {cards}. One card per component: CHE-92 merged the flat "
        "routing card into this one because two cards with no consistency check "
        "had already drifted three ways."
    )
    assert not (ROOT / "knowledge" / "solver_cards").exists(), (
        "knowledge/solver_cards/ is back. That directory was the second card "
        "tier; reintroducing it reintroduces the drift."
    )


@pytest.mark.parametrize("name", NAMES)
def test_the_card_does_not_restate_a_capability_table(name: str) -> None:
    card = _card(name)
    restated = sorted(field for field in FORBIDDEN_RESTATEMENTS if field in card)
    assert not restated, (
        f"{name}/card.yaml restates {restated}, which core/capabilities.py owns "
        "and tests/test_registry_matches_capabilities.py holds the registry to. "
        "A third copy in prose can only drift. Point at the declaration with a "
        "`capabilities:` field and keep the *consequences* the table cannot "
        "express -- that CUDA is torch-backend-only, that the torch backend "
        "defaults to float32 -- which are not the same thing as restating it."
    )
    assert "capabilities" in card, (
        f"{name}/card.yaml has no `capabilities:` pointer at its authoritative "
        "declaration in core/capabilities.py."
    )
    assert card["capabilities"].startswith("core/capabilities.py"), card["capabilities"]


@pytest.mark.parametrize("name", NAMES)
def test_validation_status_is_on_the_ladder(name: str) -> None:
    """A free string cannot be read by anything, and two cards had proved it.

    `analytically_validated_scalar_asm_cpu` fused the rung with its qualifier.
    Both halves matter and neither could be checked while they were one string.
    """
    card = _card(name)
    assert card["validation_status"] in LADDER, (
        f"{name}/card.yaml has validation_status={card['validation_status']!r}, "
        f"which is not one of {LADDER}. If the value carries a qualifier "
        "(\"scalar ASM, CPU\"), that belongs in validation_scope -- keep it, do "
        "not fuse it back into the rung."
    )
    assert card.get("validation_scope"), (
        f"{name}/card.yaml has no validation_scope. The ladder value alone does "
        "not tell a reader whether their task is inside what was validated, "
        "which is the question knowledge/README.md tells them to ask."
    )


@pytest.mark.parametrize("name", NAMES)
def test_not_yet_probed_is_a_gate_and_is_current(name: str) -> None:
    """A cleared item must be removed, not annotated.

    `knowledge/README.md` makes this list a precondition for unattended
    execution. An entry reading "outstanding (but actually done)" is worse than
    no list: it either blocks validated work or trains the reader to skim past
    the entries that are real.
    """
    card = _card(name)
    entries = card.get("not_yet_probed", [])
    assert entries, f"{name}/card.yaml has no not_yet_probed list"
    cleared = [
        entry
        for entry in entries
        if isinstance(entry, str)
        and ("gpu-marked suite" in entry or "GPU device placement" in entry)
    ]
    assert not cleared, (
        f"{name}/card.yaml still lists {cleared} as unprobed. The dedicated GPU "
        "suite passes -- 48 tests, 69 s on one RTX A6000. Remove a cleared entry "
        "rather than annotating it."
    )


@pytest.mark.parametrize("name", NAMES)
def test_the_pack_points_at_evidence_that_exists(name: str) -> None:
    """A card claim whose evidence moved is worse than no card.

    CHE-92 relocated every probe and record out of `knowledge/`. The point of
    relocating rather than deleting was that the claims keep their backing, so
    the pointers have to resolve.
    """
    card = _card(name)
    pack = card.get("knowledge_pack", {})
    assert isinstance(pack, dict), (
        f"{name}/card.yaml's knowledge_pack is a bare path. It should name where "
        "each kind of material went: prose, probes, records, tutorials."
    )
    for kind, location in pack.items():
        assert (ROOT / str(location)).exists(), (
            f"{name}/card.yaml points {kind} at {location}, which does not exist."
        )


@pytest.mark.parametrize("name", NAMES)
def test_the_pack_holds_prose_and_nothing_executable(name: str) -> None:
    """`knowledge/` is disclosed to an agent, so it is prose and declarations."""
    pack = SOLVERS / name
    executable = sorted(p.name for p in pack.rglob("*") if p.suffix in {".py", ".npz", ".pdf"})
    assert not executable, (
        f"{name}'s pack contains {executable}. Probes belong in "
        "benchmarks/probes/, their records in benchmarks/probes/records/, and "
        "tutorial reproductions in tests_tutorial/cases/ -- knowledge/ is what "
        "is disclosed to an agent, not where things run."
    )


def test_usage_notes_replaced_capability_notes() -> None:
    """One word, one meaning.

    "Capability" named both `core/capabilities.py` -- executable and
    authoritative -- and a prose file of advice. The rename is the whole fix, so
    it is worth a line that fails if it is undone.
    """
    for name in NAMES:
        assert (SOLVERS / name / "usage_notes.md").is_file(), name
        assert not (SOLVERS / name / "capability_notes.md").exists(), (
            f"{name}/capability_notes.md is back. `capabilities` is the name of "
            "the executable declaration in core/; advisory prose is usage_notes."
        )
