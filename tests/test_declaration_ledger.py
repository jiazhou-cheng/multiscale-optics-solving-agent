"""The registry declares 52 things. This is where a 53rd with no evidence fails.

CHE-108 (M1.3) B0.1 and CHE-112 (M2.4) PRESERVE-1, which asked for the same
mechanism in different words. One ledger, asserted complete in both directions,
with every evidence reference resolved against the tree rather than trusted as a
string -- because an evidence reference nobody resolves is the same thing as no
evidence, with more characters.

What each test here is guarding against, concretely:

* a declaration added to ``registry/*.yaml`` with no coverage entry -- the gap
  the ledger exists to close;
* a coverage entry left behind after its declaration was edited or deleted, so
  the ledger reports coverage of a sentence that is no longer there;
* an entry whose evidence names a test or a file that does not exist, which is
  the failure mode a hand-maintained list always eventually has;
* an invariant claimed without a tolerance basis, which M2.4 names as its own
  acceptance criterion.
"""

from __future__ import annotations

import re

import pytest

from core.paths import repository_root
from verification.declaration_ledger import (
    COVERAGE_LEDGER,
    Coverage,
    CoverageKind,
    DeclarationKind,
    coverage_for,
    registry_declarations,
    resolve_ledger,
)
from verification.families import FAMILIES

ROOT = repository_root()


@pytest.fixture(scope="module")
def report():
    return resolve_ledger()


# ---------------------------------------------------------------------------
# Completeness, in both directions
# ---------------------------------------------------------------------------


def test_every_registry_declaration_has_a_coverage_entry(report) -> None:
    """The acceptance criterion, mechanically. No silent omissions."""
    assert report.gaps == (), "declarations with no coverage entry: " + "; ".join(
        f"{d.component}/{d.kind.value}: {d.text[:90]}" for d in report.gaps
    )


def test_no_coverage_entry_survives_its_declaration(report) -> None:
    """An entry anchoring nothing is a claim about a sentence that moved."""
    assert report.orphaned == (), "entries anchoring no declaration: " + "; ".join(
        f"{e.component}/{e.kind.value}: {e.anchor[:60]!r}" for e in report.orphaned
    )


def test_no_anchor_is_ambiguous(report) -> None:
    """Two declarations behind one anchor means one of them is uncovered."""
    assert report.ambiguous == (), "; ".join(report.ambiguous)


def test_the_ledger_covers_every_declaration_kind() -> None:
    """A kind with no entries would mean the enumeration missed a registry field."""
    covered_kinds = {entry.kind for entry in COVERAGE_LEDGER}
    assert covered_kinds == set(DeclarationKind)


def test_the_declaration_count_is_reported_rather_than_frozen(report) -> None:
    """No magic total. What is asserted is that the ledger and the registry agree."""
    assert len(report.covered) == len(registry_declarations())


# ---------------------------------------------------------------------------
# Every reference resolves against the tree
# ---------------------------------------------------------------------------


def _resolve(reference: str) -> str | None:
    """``None`` if the reference resolves; otherwise why it does not."""
    if "::" in reference:
        path_part, _, test_name = reference.partition("::")
        path = ROOT / path_part
        if not path.exists():
            return f"file {path_part} does not exist"
        text = path.read_text(encoding="utf-8")
        if not re.search(rf"\bdef {re.escape(test_name)}\b", text):
            return f"{path_part} has no test named {test_name}"
        return None
    if reference in FAMILIES:
        return None
    if re.fullmatch(r"B[0-4]-[A-Z0-9-]+", reference):
        # A family-id shape that is not a family. It may still be a canonical
        # instance, and an instance id does not have to be its family's id plus a
        # suffix -- B0-HANDOFF-01 belongs to B0-CONTRACT -- so this searches
        # rather than derives the owner.
        for family in FAMILIES.values():
            if any(i.instance_id == reference for i in family.canonical_instances):
                return None
        return f"{reference} is not a registered family or a canonical instance of one"
    path = ROOT / reference
    if not path.exists():
        return f"path {reference} does not exist"
    return None


@pytest.mark.parametrize(
    "entry",
    [e for e in COVERAGE_LEDGER if e.evidence],
    ids=lambda e: f"{e.component}:{e.kind.value}:{e.anchor[:28]}",
)
def test_every_evidence_reference_resolves(entry: Coverage) -> None:
    """A reference is a claim that something exists. Check it.

    This is the test that keeps the ledger honest over time: renaming a test or
    moving a probe breaks the coverage claim that pointed at it, rather than
    leaving a ledger that reads complete and points nowhere.
    """
    failures = [
        f"{reference}: {problem}"
        for reference in entry.evidence
        if (problem := _resolve(reference)) is not None
    ]
    assert not failures, "; ".join(failures)


# ---------------------------------------------------------------------------
# What the classifications are allowed to mean
# ---------------------------------------------------------------------------


def test_a_blank_reason_cannot_buy_a_non_executable_classification() -> None:
    """The placeholder guard, exercised rather than trusted."""
    for reason in ("", "   ", "n/a", "TODO", "not tested"):
        with pytest.raises(ValueError, match="real reason"):
            Coverage(
                component="C_RAY_TO_WAVE",
                kind=DeclarationKind.WARNING,
                anchor="anything",
                coverage_kind=CoverageKind.EXPLICIT_NON_EXECUTABLE,
                reason=reason,
            )


def test_an_executable_claim_without_a_reference_is_refused() -> None:
    with pytest.raises(ValueError, match="names none"):
        Coverage(
            component="C_RAY_TO_WAVE",
            kind=DeclarationKind.WARNING,
            anchor="anything",
            coverage_kind=CoverageKind.EXECUTABLE_TEST,
            reason="a reason long enough to be a reason, which is not the missing part here",
        )


def test_an_unresolvable_reference_shape_is_refused() -> None:
    """``see the tests`` is not a reference."""
    with pytest.raises(ValueError, match="not resolvable"):
        Coverage(
            component="C_RAY_TO_WAVE",
            kind=DeclarationKind.WARNING,
            anchor="anything",
            coverage_kind=CoverageKind.EXECUTABLE_TEST,
            evidence=("see the tests",),
        )


@pytest.mark.parametrize(
    "entry",
    [e for e in COVERAGE_LEDGER if e.kind is DeclarationKind.INVARIANT],
    ids=lambda e: f"{e.component}:{e.anchor}",
)
def test_every_invariant_states_a_tolerance_basis(entry: Coverage) -> None:
    """M2.4's criterion: an assertion, a tolerance, and a basis for the tolerance."""
    assert entry.coverage_kind.is_executable
    assert entry.tolerance_basis is not None
    # The basis has to name what kind of basis it is, not merely be prose. The
    # five admissible kinds are the schema's ToleranceBasis values that may gate.
    assert any(
        kind in entry.tolerance_basis
        for kind in (
            "analytic_derivation",
            "conservation_law",
            "numerical_precision_floor",
            "independent_derivation",
        )
    ), f"{entry.anchor}: the basis does not name a tolerance-basis kind"


def test_an_invariant_cannot_be_declared_untestable() -> None:
    """M2.4 is explicit: remove it from the registry instead."""
    with pytest.raises(ValueError, match="explicit_non_executable"):
        Coverage(
            component="C_RAY_TO_WAVE",
            kind=DeclarationKind.INVARIANT,
            anchor="pupil_power_consistency",
            coverage_kind=CoverageKind.EXPLICIT_NON_EXECUTABLE,
            reason="a long enough reason that the reason guard is not what rejects this one",
            tolerance_basis="conservation_law: whatever",
        )


def test_an_invariant_without_a_tolerance_basis_is_refused() -> None:
    with pytest.raises(ValueError, match="tolerance basis"):
        Coverage(
            component="C_RAY_TO_WAVE",
            kind=DeclarationKind.INVARIANT,
            anchor="pupil_power_consistency",
            coverage_kind=CoverageKind.EXECUTABLE_TEST,
            evidence=("tests/test_declaration_ledger.py",),
        )


# ---------------------------------------------------------------------------
# The non-executable classifications, individually
# ---------------------------------------------------------------------------


def test_the_non_executable_entries_are_few_and_each_says_why() -> None:
    """Reported rather than capped: the interesting number is which ones.

    Six of fifty-two, and every one of them is a statement about something
    outside this repository's forward path -- upstream thread safety, a version
    scope note, a cost asymptote, a vectorial comparison there is no second
    model for, and an interpolation stage this implementation does not have.
    None of them is a physics claim about a shipping computation.
    """
    non_executable = [e for e in COVERAGE_LEDGER if not e.coverage_kind.is_executable]
    assert len(non_executable) == 6, [e.anchor[:50] for e in non_executable]
    for entry in non_executable:
        assert len(" ".join(entry.reason.split())) >= 60
        # Each one says what *is* checked instead, so the classification is a
        # boundary rather than an absence.
        assert entry.evidence, f"{entry.anchor[:40]}: name what is checked instead"


def test_coverage_for_names_the_gap_rather_than_returning_nothing() -> None:
    with pytest.raises(KeyError, match="no coverage entry"):
        coverage_for("C_RAY_TO_WAVE:assumption:a-declaration-nobody-made")


def test_a_known_declaration_resolves_by_id() -> None:
    entry = coverage_for("C_WAVE_TO_RAY:hard_limit:grazing_bin")
    assert entry.coverage_kind is CoverageKind.EXECUTABLE_TEST
