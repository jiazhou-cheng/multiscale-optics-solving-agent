"""The claim ledger must stay true, and its citations must resolve.

CHE-104 (M0.3). A ledger is only worth having if it cannot quietly become
fiction, and there are exactly three ways it can:

1. **A component grows and the ledger does not.** Then the coverage matrix shows
   full coverage of a set that no longer matches the code.
2. **A claim cites evidence that stops existing.** A renamed test or a moved
   record leaves a citation that reads as backing and is not.
3. **A gate-deciding claim rests on an oracle that shares code with what it
   judges.** This is the circular-validation rule, and L2-PSF-01 already
   violated it once -- its negative-control floor was originally set from an O2
   comparison, our own ASM/RS propagator judging our own coupler, and had to be
   retired.

Each gets a test below. The evidence-resolution test is the expensive one and
the one that earns its keep: it collects the whole suite once and checks every
cited node ID against what pytest can actually find.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from core.capabilities import COMPONENT_CAPABILITIES
from verification.claim_ledger import (
    CLAIMS,
    GAPS,
    KNOWLEDGE_PACK_REQUIRED_FILES,
    LEDGER_COMPONENTS,
    NO_VALIDATED_CLAIM,
    GateStatus,
    OracleIndependence,
    claims_for,
    open_gates,
)

ROOT = Path(__file__).resolve().parents[1]
VALIDATION_DIR = ROOT / "benchmarks" / "validation"


# ---------------------------------------------------------------------------
# 1. Coverage: the ledger's component set is the code's component set
# ---------------------------------------------------------------------------


def test_the_ledger_covers_every_component_the_capability_table_declares() -> None:
    """`core.capabilities` is the executable truth about what exists."""
    declared = set(COMPONENT_CAPABILITIES)
    ledger = set(LEDGER_COMPONENTS)

    assert declared == ledger, (
        f"capability table and ledger disagree about which components exist. "
        f"Only in capabilities: {sorted(declared - ledger)}. "
        f"Only in the ledger: {sorted(ledger - declared)}."
    )


def test_the_ledger_covers_every_registry_entry() -> None:
    models = yaml.safe_load((ROOT / "src" / "registry" / "models.yaml").read_text())["models"]
    couplers = yaml.safe_load((ROOT / "src" / "registry" / "couplers.yaml").read_text())["couplers"]
    registered = {entry["id"] for entry in [*models, *couplers]}

    missing = registered - set(LEDGER_COMPONENTS)
    assert not missing, (
        f"these are registered as supported and have no ledger entry: {sorted(missing)}. "
        "A component the repository advertises must say what has been validated about "
        "it, even if the answer is NO_VALIDATED_CLAIM."
    )


@pytest.mark.parametrize("component", LEDGER_COMPONENTS)
def test_every_component_has_a_claim_or_an_explicit_absence(component: str) -> None:
    """Silence is not an answer. Either there is a claim or there is a reason."""
    entries = claims_for(component)
    if entries:
        return
    assert component in NO_VALIDATED_CLAIM, (
        f"{component} has no ledger claim and no NO_VALIDATED_CLAIM record. "
        "An empty row in the coverage matrix must be a deliberate statement."
    )
    assert NO_VALIDATED_CLAIM[component].strip(), f"{component}'s absence has no reason"


# ---------------------------------------------------------------------------
# 2. Evidence resolves
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def collected_node_ids() -> frozenset[str]:
    """Every test node pytest can actually collect, as `path::name`.

    Collected in a subprocess rather than imported, because the question is what
    *pytest* finds -- a node ID that only resolves under a particular import
    order is not a citation anyone can follow.
    """
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "--no-header", "tests"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    ids = {
        line.split("[")[0].strip()
        for line in result.stdout.splitlines()
        if "::" in line and not line.startswith(("ERROR", "E "))
    }
    if not ids:  # pragma: no cover - collection failing is its own loud problem
        pytest.fail(
            f"could not collect any test IDs:\n{result.stdout[-2000:]}\n{result.stderr[-2000:]}"
        )
    return frozenset(ids)


def _evidence_kind(reference: str) -> str:
    if "::" in reference:
        return "test"
    return "path"


def _all_cited_references() -> set[str]:
    """Every citation in the ledger, including the ones inside StochasticEvidence.

    Review caught that the four `StochasticEvidence` node IDs were not in this
    set: they are the block where a rename would rot silently, since nothing
    else reads them. AC2 says EVERY entry names evidence that resolves.
    """
    found: set[str] = set()
    for claim in CLAIMS:
        found.update(claim.evidence)
        if claim.stochastic is None:
            continue
        for name in (
            "exactness_limit",
            "unbiasedness",
            "convergence_exponent",
            "variance_characterization",
        ):
            reference = getattr(claim.stochastic, name)
            if reference:
                found.add(reference)
    return found


@pytest.mark.parametrize(
    "reference",
    sorted({e for e in _all_cited_references() if "::" in e}),
)
def test_every_cited_test_node_is_collectible(
    reference: str, collected_node_ids: frozenset[str]
) -> None:
    """A renamed test must break the citation, not silently orphan it."""
    assert reference in collected_node_ids, (
        f"the ledger cites {reference!r}, which pytest cannot collect. Either the "
        "test was renamed or removed -- update the ledger to name the evidence that "
        "actually exists, rather than leaving a citation that reads as backing."
    )


@pytest.mark.parametrize(
    "reference",
    sorted({e for e in _all_cited_references() if "::" not in e}),
)
def test_every_cited_path_exists(reference: str) -> None:
    """Record paths, registry files and report anchors."""
    path, _, anchor = reference.partition("#")
    target = ROOT / path
    assert target.is_file(), f"the ledger cites {path!r}, which does not exist"
    if anchor:
        assert anchor.lower() in target.read_text().lower(), (
            f"{path} exists but carries no anchor matching {anchor!r}"
        )


def test_no_claim_cites_nothing() -> None:
    empty = [f"{c.component}/{c.kind.value}" for c in CLAIMS if not c.evidence]
    assert not empty, f"claims with no evidence at all: {empty}"


# ---------------------------------------------------------------------------
# 3. Our own code never decides correctness for our own code
# ---------------------------------------------------------------------------


def test_no_gate_deciding_claim_rests_on_an_oracle_that_shares_code() -> None:
    """The standing rule, as an executable constraint.

    O2 -- the repository's own float64 ASM/RS propagator -- is real evidence and
    is kept. What it may not do is decide a pass/fail, because a bug shared
    between an implementation and its own self-test cancels exactly.
    """
    offenders = [
        f"{c.component}/{c.kind.value}: {c.claim[:70]}"
        for c in CLAIMS
        if c.gate_deciding and c.oracle_independence is OracleIndependence.SHARES_CODE
    ]
    assert not offenders, (
        "these claims are marked gate_deciding while their oracle shares code with "
        f"the thing under test: {offenders}. Either find an independent oracle or "
        "demote the claim to characterization."
    )


def test_every_claim_declares_its_oracle_independence() -> None:
    """There is no default. Not declaring is the thing being prevented."""
    for claim in CLAIMS:
        assert isinstance(claim.oracle_independence, OracleIndependence), (
            f"{claim.component}/{claim.kind.value} does not declare oracle_independence"
        )


def test_a_gate_with_a_tolerance_reports_what_was_observed() -> None:
    """A declared threshold and no measurement is a claim with no content."""
    incomplete = [
        f"{c.component}/{c.kind.value}"
        for c in CLAIMS
        if c.tolerance is not None and c.observed is None
    ]
    assert not incomplete, f"tolerance declared with nothing measured against it: {incomplete}"


def test_every_tolerance_states_its_basis() -> None:
    """Why that number. A tolerance with no basis is a number someone can widen."""
    unjustified = [
        f"{c.component}/{c.kind.value}"
        for c in CLAIMS
        if c.tolerance is not None and not (c.tolerance_basis or "").strip()
    ]
    assert not unjustified, f"tolerance with no stated basis: {unjustified}"


# ---------------------------------------------------------------------------
# 4. The open-gate register reproduces the state the repository already knows
# ---------------------------------------------------------------------------


def test_the_open_gate_register_reproduces_l2_psf_01() -> None:
    """Cross-check against the committed prose, which is the independent source.

    If these disagree, one of the two is wrong and the discrepancy is the finding
    -- this test exists so that it surfaces rather than being smoothed over by
    whichever file someone read last.
    """
    tolerances = yaml.safe_load(
        (ROOT / "benchmarks" / "physics" / "L2-PSF-01" / "tolerances.yaml").read_text()
    )
    # Cross-checked against the RECORD, not against a number retyped into this
    # test. A hardcoded expectation here would make the ledger and the test agree
    # with each other while both drifted from the measurement.
    finest = json.loads(
        (ROOT / "benchmarks" / "probes" / "records" / "m3_quadrature_weight.json").read_text()
    )["finest_configuration"]
    gates = {c.metric: c for c in open_gates()}

    primary = next(c for m, c in gates.items() if "fft_oracle_intensity_relative_l2" in m)
    assert primary.tolerance == tolerances["fft_oracle_intensity_relative_l2"] == 1.0e-3
    assert primary.observed == pytest.approx(finest["weighted_vs_o1"]), (
        "the ledger's observed value must be the one in the record, to the record's "
        "precision"
    )
    assert primary.observed == pytest.approx(2.21e-3, abs=5e-6), "the value CHE-104 names"
    assert primary.gate_status is GateStatus.NOT_MET

    control = next(c for m, c in gates.items() if "quadrature_weight_min_improvement" in m)
    assert control.tolerance == tolerances["quadrature_weight_min_improvement_factor"] == 1.2
    assert control.observed == pytest.approx(finest["improvement_factor_vs_o1"])
    assert control.observed < 1.0, (
        "this is what makes L2-PSF-01's negative_controls_pass false: the control "
        "fires backwards, the uniform configuration being closer to the analytic "
        "oracle than the weighted one"
    )
    assert control.gate_status is GateStatus.NOT_MET

    # And the record agrees the gate is unmet, rather than the ledger asserting it.
    assert finest["gate_met"] is False


def test_the_register_is_not_empty_while_the_manifest_says_a_gate_is_open() -> None:
    """The manifest's prose and the register must agree that something is open."""
    manifest = (ROOT / "benchmarks" / "manifest.yaml").read_text()
    assert "gate_disposition" in manifest
    assert open_gates(), (
        "benchmarks/manifest.yaml still carries a gate_disposition describing an "
        "unmet gate, but the ledger reports no open gates. One of them is wrong."
    )


# ---------------------------------------------------------------------------
# 5. The gap list is a scope, not a wish
# ---------------------------------------------------------------------------


def test_every_gap_is_ranked_and_owned() -> None:
    valid = {"critical", "high", "medium", "low"}
    for gap in GAPS:
        assert gap.severity in valid, f"{gap.component}: bad severity {gap.severity!r}"
        assert gap.owner.strip(), f"{gap.component}/{gap.kind.value} has no owner"
        assert gap.rationale.strip(), (
            f"{gap.component}/{gap.kind.value} is ranked {gap.severity} with no reason. "
            "The ranking criterion is about what a gap would let through, so the "
            "reason is the ranking."
        )
        assert gap.blocks, (
            f"{gap.component}/{gap.kind.value} blocks nothing -- then why is it a gap?"
        )


def test_every_gap_names_a_component_the_ledger_knows() -> None:
    unknown = {g.component for g in GAPS} - set(LEDGER_COMPONENTS)
    assert not unknown, f"gaps against unknown components: {sorted(unknown)}"


def test_every_open_gate_has_a_gap_that_owns_it() -> None:
    """An unmet gate with nobody assigned is how one stays unmet."""
    owed = {g.component for g in GAPS}
    orphans = sorted({c.component for c in open_gates()} - owed)
    assert not orphans, (
        f"these components have an unmet gate and no entry in the gap list: {orphans}"
    )


# ---------------------------------------------------------------------------
# 6. Knowledge-pack completeness
# ---------------------------------------------------------------------------

#: Which pack directory each component's knowledge lives in. `None` means the
#: pack does not exist, which is the audit's actual finding rather than an
#: omission -- both couplers below have graph nodes and no agent-facing docs.
PACK_PATHS = {
    "M_RAY_OPTILAND": ("solver", "knowledge/solvers/optiland"),
    "M_WAVE_CHROMATIX": ("solver", "knowledge/solvers/chromatix"),
    "C_RAY_TO_WAVE": ("coupler", "knowledge/couplers/ray_to_wave"),
    "C_WAVE_TO_RAY": ("coupler", "knowledge/couplers/wave_to_ray"),
    "C_PLANAR_DOE_STEP": ("coupler", None),
    "C_PATCH_WFT": ("coupler", None),
}


def test_the_pack_audit_covers_every_component() -> None:
    assert set(PACK_PATHS) == set(LEDGER_COMPONENTS)


@pytest.mark.parametrize(
    "component", [c for c, (_, path) in PACK_PATHS.items() if path is not None]
)
def test_an_existing_knowledge_pack_is_complete(component: str) -> None:
    kind, relative = PACK_PATHS[component]
    directory = ROOT / relative
    missing = [
        name for name in KNOWLEDGE_PACK_REQUIRED_FILES[kind] if not (directory / name).is_file()
    ]
    assert not missing, f"{relative} is missing {missing}"


@pytest.mark.parametrize("component", [c for c, (_, path) in PACK_PATHS.items() if path is None])
def test_a_missing_knowledge_pack_is_filed_as_a_gap(component: str) -> None:
    """The audit's finding, pinned so it cannot be forgotten rather than fixed.

    `C_PLANAR_DOE_STEP` and `C_PATCH_WFT` are the two couplers that HAVE graph
    nodes, and they are the two with no card, conventions, failure guide or
    theory. Implementation is M2.3 (CHE-111); this test fails when the pack
    appears, which is the prompt to move it into the completeness check above.
    """
    _, relative = PACK_PATHS[component]
    assert relative is None

    assert any(g.component == component and "knowledge pack" in g.gap.lower() for g in GAPS), (
        f"{component} has no knowledge pack and no gap entry saying so"
    )

    for candidate in ("planar_doe_step", "patch_wft"):
        directory = ROOT / "knowledge" / "couplers" / candidate
        assert not directory.exists(), (
            f"{directory.relative_to(ROOT)} now exists. Move {component} into "
            "PACK_PATHS with its real path so the completeness check applies to it, "
            "and close the gap-list entry."
        )


# ---------------------------------------------------------------------------
# 7. The generated artifacts still equal what generates them
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def generators() -> dict:
    sys.path.insert(0, str(ROOT / "scripts"))
    from generate_validation_coverage import ARTIFACTS

    return ARTIFACTS


@pytest.mark.parametrize("name", ["coverage_matrix.md", "open_gates.md", "gap_list.md"])
def test_the_generated_document_matches_the_ledger(name: str, generators: dict) -> None:
    """Regenerated in memory and compared, never written.

    Same reasoning as `tests/test_generated_artifacts.py`: a test that fixes the
    drift it was meant to report is not a test.
    """
    committed = (VALIDATION_DIR / name).read_text()
    assert committed == generators[name](), (
        f"benchmarks/validation/{name} is stale. Regenerate with: "
        "./run.sh python scripts/generate_validation_coverage.py"
    )
