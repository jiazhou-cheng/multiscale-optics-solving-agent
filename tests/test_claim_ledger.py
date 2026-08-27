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
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "--no-header",
            # CHE-140: this subprocess must not inherit the default *selection*.
            # `addopts` now carries `-m "not slow"`, and with it inherited the
            # question this fixture asks silently changed from "can pytest find
            # this node" to "is this node in the default run" -- so eleven
            # perfectly valid ledger citations, all of them to tests that carry
            # `slow`, were reported as uncollectible. An empty `-m` clears the
            # expression; `-n 0` turns the sharding off, since a collect-only run
            # has nothing to parallelize and xdist changes how the ids print.
            "-m",
            "",
            "-n",
            "0",
            "tests",
        ],
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
        "the ledger's observed value must be the one in the record, to the record's precision"
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


def test_the_register_is_not_empty_while_the_frozen_tolerance_is_unreached() -> None:
    """The tolerance file and the register must agree that something is open.

    This used to cross-check against ``benchmarks/manifest.yaml``'s
    ``gate_disposition`` prose. CHE-133 deleted that prose along with the
    ``levels:`` block it lived in -- it was the exact failure mode the register
    exists to fix, an unmet gate discoverable only by reading a paragraph. The
    independent source is now the frozen tolerance beside the measurement it was
    never reached by.
    """
    tolerances = yaml.safe_load(
        (ROOT / "benchmarks" / "physics" / "L2-PSF-01" / "tolerances.yaml").read_text()
    )
    finest = json.loads(
        (ROOT / "benchmarks" / "probes" / "records" / "m3_quadrature_weight.json").read_text()
    )["finest_configuration"]
    assert finest["weighted_vs_o1"] > tolerances["fft_oracle_intensity_relative_l2"], (
        "the record no longer exceeds the frozen tolerance. If the gate has genuinely "
        "been met, that is a finding to attribute -- not a reason to delete this test."
    )
    assert open_gates(), (
        "the frozen 1.0e-3 tolerance is still unreached by the recorded measurement, "
        "and the ledger reports no open gates. One of them is wrong."
    )


def test_the_retired_levels_block_did_not_take_its_disposition_with_it() -> None:
    """CHE-133 preserve check: the PB7 rule survived the manifest edit.

    The gate must not be closed against another Optiland PSF route, because
    FFTPSF and HuygensPSF share one Wavefront/OPD front end and are one oracle,
    not two. That was prose in ``manifest.yaml``; it is now a caveat on the claim
    AND a construction-time rule in ``BenchmarkFamily``.
    """
    primary = next(
        c for c in open_gates() if "fft_oracle_intensity_relative_l2" in (c.metric or "")
    )
    assert any("FFTPSF" in caveat and "HuygensPSF" in caveat for caveat in primary.caveats), (
        "the oracle-independence rule that guarded this gate is no longer recorded "
        "against the claim it guards"
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

#: Which pack directory each component's knowledge lives in. `None` used to mean
#: the pack did not exist, which was the audit's actual finding rather than an
#: omission -- the two couplers that HAVE graph nodes had no agent-facing docs.
#: Both are now written, so no entry is `None` and the completeness check applies
#: to every component. The `None` branch is kept because the next component
#: registered without a pack should land in it rather than in a passing test.
PACK_PATHS = {
    "M_RAY_OPTILAND": ("solver", "knowledge/solvers/optiland"),
    "M_WAVE_CHROMATIX": ("solver", "knowledge/solvers/chromatix"),
    "C_RAY_TO_WAVE": ("coupler", "knowledge/couplers/ray_to_wave"),
    "C_WAVE_TO_RAY": ("coupler", "knowledge/couplers/wave_to_ray"),
    "C_PLANAR_DOE_STEP": ("coupler", "knowledge/couplers/planar_doe_step"),
    "C_PATCH_WFT": ("coupler", "knowledge/couplers/patch_wft"),
    "C_GENERALIZED_SNELL": ("coupler", "knowledge/couplers/generalized_snell"),
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

    `C_PLANAR_DOE_STEP` and `C_PATCH_WFT` were the two couplers that HAVE graph
    nodes and had no card, conventions, failure guide or theory. M2.3 wrote the
    first, and deliberately deferred the second: that coupler's estimator
    contract -- the centre density, the unbiasedness weights, the measured
    variance reduction -- was being rewritten by CHE-120, so a pack written
    before it landed would have been wrong on landing.

    CHE-120 has landed and the pack is written, so this parameterization is now
    EMPTY and the completeness check above covers every component. It is kept
    rather than deleted because it is where the next component registered without
    a pack should land -- deleting it would make that arrive as a passing test.
    """
    _, relative = PACK_PATHS[component]
    assert relative is None

    assert any(g.component == component and "knowledge pack" in g.gap.lower() for g in GAPS), (
        f"{component} has no knowledge pack and no gap entry saying so"
    )


def test_every_coupler_with_a_graph_node_now_has_a_pack() -> None:
    """The closed finding, asserted forwards.

    The audit's point was never "two directories are absent" -- it was that the
    two couplers an agent has to CONFIGURE were the two with nothing to read, and
    their configuration is where a coupler gets got wrong. So the standing
    assertion is over graph nodes, not over a list of names: registering a new
    graph node without a pack fails here.
    """
    from registry.loader import Registry

    registry = Registry.from_package()
    for coupler_id, coupler in registry.couplers.items():
        if not getattr(coupler, "graph_node", None):
            continue
        entry = PACK_PATHS.get(coupler_id)
        assert entry is not None, f"{coupler_id} has a graph node and no PACK_PATHS entry"
        _, relative = entry
        assert relative is not None, (
            f"{coupler_id} has a graph node and no knowledge pack. An agent asked to "
            "build a graph through this node has nothing to read about its "
            "conventions, and the conventions are what a coupler gets wrong."
        )
        assert (ROOT / relative).is_dir(), relative


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


# ---------------------------------------------------------------------------
# The ledger as a projection (CHE-131, M0.5.2)
# ---------------------------------------------------------------------------


def test_the_ledger_is_the_family_registry_plus_what_is_not_migrated_yet() -> None:
    """``CLAIMS`` is derived, not typed twice.

    The direction is: the family registry is authoritative wherever it has
    content, and this table is its projection plus the rows M1/M2/M4 have not
    absorbed yet. Asserting the arithmetic here is what stops somebody adding a
    hand-written claim *beside* a family and having both survive.
    """
    from verification.claim_ledger import _LEGACY_CLAIMS, all_claims
    from verification.families.projection import claims_from_families

    projected = claims_from_families()
    assert all_claims() == projected + _LEGACY_CLAIMS


def test_a_family_and_a_legacy_row_cannot_describe_the_same_cell() -> None:
    """The anti-drift mechanism.

    When M1/M2/M4 land a family, the hand-written row it replaces must be
    deleted -- leaving it produces a collision here rather than a silent
    duplicate that slowly disagrees with the family it shadows.
    """
    from verification.claim_ledger import _LEGACY_CLAIMS
    from verification.families.projection import claims_from_families

    # Keyed on the metric as well as the cell: a component can legitimately have
    # several forward-accuracy claims about different quantities, and C_RAY_TO_WAVE
    # already does. What must not happen is two rows about the SAME measurement.
    legacy_cells = {(c.component, c.kind, c.metric) for c in _LEGACY_CLAIMS}
    collisions = sorted(
        f"{c.component}/{c.kind.value}/{c.metric}"
        for c in claims_from_families()
        if (c.component, c.kind, c.metric) in legacy_cells
    )
    assert not collisions, (
        "a registered family and a hand-written claim occupy the same cell:\n  "
        + "\n  ".join(collisions)
        + "\nDelete the legacy row in src/verification/claim_ledger.py::_LEGACY_CLAIMS; "
        "the family is now the source of truth for it."
    )


def test_no_metric_is_claimed_twice_within_the_projection() -> None:
    """Two families measuring the same quantity on the same component is two
    families that disagree with each other eventually.

    Keyed on the metric rather than the cell, because several B1 families
    legitimately make forward-accuracy claims about one solver -- they just
    measure different things.
    """
    from verification.families.projection import claims_from_families

    seen: dict[tuple[str, object, object], str] = {}
    for claim in claims_from_families():
        key = (claim.component, claim.kind, claim.metric)
        assert key not in seen, (
            f"{claim.component}/{claim.kind.value}/{claim.metric} is claimed by two "
            f"families: {seen[key]!r} and {claim.claim!r}"
        )
        seen[key] = claim.claim
