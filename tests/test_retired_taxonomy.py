"""The retired task taxonomy cannot creep back.

CHE-133 (M0.5.4). ``A1-*`` and the planned ``L1-*``/``L2-*``/``L3-*`` tasks were
the old evaluation design: an id, a level number, a protocol id, and the
interesting scientific content as prose in fields nothing could query. Deleting
them is easy; keeping them deleted is what needs a test, because the cheapest
way to add a benchmark is to copy the shape of the last one.

Why this has an allowlist, and why that is not a loophole
---------------------------------------------------------
Three kinds of reference legitimately survive, and each one is a different
argument:

* **The live singlet workload.** ``benchmarks/physics/L2-PSF-01/`` is still the
  only way to execute that case. CHE-133 explicitly does *not* delete it; the
  executor and family runner replace it in M4 (CHE-115/CHE-116), and it becomes
  ``B3-PSF-SINGLET-01`` then. Deleting the directory now to satisfy a grep would
  be deleting the only runnable path for a workload before its replacement
  exists.
* **The historical record.** Reports under ``benchmarks/reports/`` and the
  frozen protocol contracts describe runs that happened, under ids that were
  real at the time. Rewriting them to remove an identifier would falsify what
  was run, which is a worse outcome than a grep hit.
* **The bookkeeping of the deletion itself.** The inventory, this file, and the
  places that explain where each retired piece went all have to name what they
  retired.

Every allowlisted path carries its reason below and the list is asserted to be
minimal: a path that no longer matches is removed, so the allowlist cannot
quietly outlive its justification.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from core.paths import repository_root

ROOT = repository_root()

#: The task-identifier shapes of the retired design.
TAXONOMY = re.compile(r"\b(?:A1-[A-Z]{2,4}-\d{2}|L[123]-[A-Z]+-\d{2})\b")

#: Directories that are outside the live surface entirely.
EXCLUDED_TREES = (".git", "archive", "build", "__pycache__", ".pytest_cache", "outputs", "runs")

#: File suffixes worth scanning. A binary record cannot creep a taxonomy back.
SCANNED_SUFFIXES = {".py", ".md", ".yaml", ".yml", ".json", ".toml", ".sh", ".cfg"}

#: path prefix -> why a reference there is legitimate. Checked for minimality.
ALLOWED: dict[str, str] = {
    "benchmarks/physics/L2-PSF-01": (
        "the only runnable path for the singlet workload until the executor and "
        "family runner replace it; CHE-133 explicitly keeps it, M4 renames it to "
        "B3-PSF-SINGLET-01"
    ),
    "benchmarks/reports": (
        "the milestone record. Rewriting a historical report to remove an "
        "identifier would falsify what was run"
    ),
    "benchmarks/protocols": (
        "frozen contracts holding tolerance derivations that exist nowhere else; "
        "each goes once the families express its content executably, which is what "
        "CHE-106 did to the M1 baseline protocol. M2 and M3 are still here"
    ),
    "benchmarks/perf/records/l2_psf_01_cpu.json": (
        "a committed cost baseline keyed by the workload it measured"
    ),
    "benchmarks/perf/run_baselines.py": (
        "the cost harness names the singlet workload as one of its baseline "
        "configurations; it is a workload key, not a task definition"
    ),
    "benchmarks/schemas/provenance.schema.json": (
        "CHE-133 FINDING, deletion deferred: this file carries the SAME baked-in "
        "benchmark_id enum [L1-RAY-01, L1-WAVE-01, L2-COUPLER-01] that "
        "result.schema.json was deleted for, and the ticket did not name it. Its "
        "only readers are archived gen1 tests, which cannot run. Reclassified C in "
        "benchmarks/inventory.yaml with the removal owned by CHE-115"
    ),
    "benchmarks/inventory.yaml": "the triage has to name what it classified",
    "benchmarks/instances/b3_psf_singlet.py": (
        "the substrate proof, whose docstring explains why its number is not the "
        "singlet runner's and what reproducing that would take"
    ),
    "benchmarks/INVENTORY.md": "generated from the triage, so it names what was classified",
    "benchmarks/README.md": "points at the live runner and says why it is still there",
    "benchmarks/manifest.yaml": "records where the deleted levels: block's content went",
    "benchmarks/validation": (
        "generated views of the claim ledger, whose evidence paths point into the "
        "live singlet workload"
    ),
    "benchmarks/probes": (
        "probe sources and records that cite the singlet workload as the "
        "configuration they measured"
    ),
    "docs": (
        "architecture and milestone documents describing what was built and when; "
        "historical, and not a place a new benchmark gets defined"
    ),
    "README.md": "repository overview; describes the singlet workload",
    "examples/graphs/ray_to_wave.yaml": "an example graph carrying the singlet task_id",
    "knowledge": (
        "solver packs citing the singlet workload as the case a convention was "
        "measured on"
    ),
    "src/couplers/handoff.py": "docstring citing the singlet workload's reference plane",
    "src/registry/prescriptions.py": "the M3-SINGLET-REF prescription's own module",
    "src/solvers": "adapter docstrings citing the singlet workload",
    "src/verification": (
        "the claim ledger's open-gate entries and the schema docstrings explaining "
        "which incident each rule came from"
    ),
    "scripts/generate_benchmark_inventory.py": (
        "renders the triage, and its docstring names the example the triage turns on"
    ),
    "tests/test_retired_taxonomy.py": (
        "this file, which has to spell the identifiers it is banning"
    ),
    "tests/test_claim_ledger.py": "cross-checks the singlet workload's open gate",
    "tests/test_benchmark_inventory.py": "documents the L2-PSF-01 split-row case",
    "tests/test_b3_b4_families.py": (
        "reads the singlet workload's tolerances.yaml to check the migrated bases "
        "verbatim, which is the one thing that would notice a reworded one"
    ),
    "tests/test_family_schema.py": "docstrings naming the incident each rule came from",
    "tests/test_verifier.py": "docstrings naming the incident each rule came from",
    "tests/test_optiland_adapter.py": "the singlet prescription's adapter tests",
    "tests/test_psf_verification.py": "the singlet workload's verification tests",
    "tests/test_ray_to_wave.py": "the singlet workload's coupler tests",
    "tests/test_performance_harness.py": "the singlet workload's cost baseline",
    "tests_tutorial": "upstream tutorial reproductions, opt-in and not the live surface",
}


def _scanned_files() -> list[Path]:
    files: list[Path] = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix not in SCANNED_SUFFIXES:
            continue
        rel = path.relative_to(ROOT)
        if set(rel.parts) & set(EXCLUDED_TREES):
            continue
        files.append(rel)
    return sorted(files)


def _hits() -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    for rel in _scanned_files():
        try:
            text = (ROOT / rel).read_text(encoding="utf-8")
        except UnicodeDecodeError:  # pragma: no cover - binary with a scanned suffix
            continue
        matches = sorted(set(TAXONOMY.findall(text)))
        if matches:
            found[str(rel)] = matches
    return found


def _is_allowed(rel: str) -> bool:
    return any(rel == prefix or rel.startswith(prefix + "/") for prefix in ALLOWED)


def test_no_new_place_uses_a_retired_task_identifier() -> None:
    offenders = {rel: hits for rel, hits in _hits().items() if not _is_allowed(rel)}
    assert not offenders, (
        "these files name a task identifier from the retired A1/L1/L2/L3 taxonomy and "
        "are not on the allowlist in this file:\n  "
        + "\n  ".join(f"{rel}: {', '.join(hits)}" for rel, hits in offenders.items())
        + "\n\nThe replacement is a BenchmarkFamily id (B0-*, B1-*, B2-*, B3-*, B4-*). "
        "If a reference is genuinely legitimate -- historical record, or the live "
        "singlet runner that M4 replaces -- add it to ALLOWED with the reason."
    )


def test_the_a1_task_set_is_gone_from_the_live_surface() -> None:
    """The six retired tasks specifically, wherever they are named as live.

    A prompt file or a recorded expectation is a task definition; the allowlist
    above admits *prose about* the retirement, not a task that still exists.
    """
    assert not list((ROOT / "benchmarks/agents").glob("prompts/A1-*.md"))
    assert not list((ROOT / "benchmarks/agents").glob("expected/A1-*.json"))
    assert not (ROOT / "benchmarks/agents/test_agent_suite.py").exists()

    from agent.benchmark_suite import SUITES, registered_tasks

    assert SUITES == {}
    assert registered_tasks() == ()


def test_the_result_schema_with_the_baked_in_taxonomy_is_gone() -> None:
    """Its ``benchmark_id`` enum was ``[L1-RAY-01, L1-WAVE-01, L2-COUPLER-01]``.

    A schema cannot describe a result outside a task set that no longer exists,
    so this one had to go rather than be extended.
    """
    assert not (ROOT / "benchmarks/schemas/result.schema.json").exists()
    replacement = ROOT / "schemas/verification_result.schema.json"
    assert replacement.is_file(), "the replacement schema must exist before the old one goes"
    assert not TAXONOMY.search(replacement.read_text(encoding="utf-8"))


def test_the_manifest_no_longer_carries_a_levels_block() -> None:
    import yaml

    manifest = yaml.safe_load((ROOT / "benchmarks/manifest.yaml").read_text(encoding="utf-8"))
    assert "levels" not in manifest
    assert "characterizations" in manifest, (
        "the characterizations block is bucket A and must survive the deletion"
    )


def test_the_allowlist_stays_minimal() -> None:
    """A prefix that no longer matches anything is removed.

    Otherwise the allowlist outlives its justification and quietly readmits a
    directory whose reason has expired -- which is how the taxonomy would come
    back.
    """
    hits = _hits()
    unused = sorted(
        prefix
        for prefix in ALLOWED
        if not any(rel == prefix or rel.startswith(prefix + "/") for rel in hits)
    )
    assert not unused, (
        "these allowlist entries no longer match any file. Delete them:\n  "
        + "\n  ".join(unused)
    )


@pytest.mark.parametrize("prefix", sorted(ALLOWED))
def test_every_allowlist_entry_states_a_reason(prefix: str) -> None:
    assert len(ALLOWED[prefix]) > 30, f"{prefix}: the reason is too short to be one"


#: The retired M1 baseline protocol id, in both its versions.
M1_PROTOCOL = re.compile(r"\bM1-BASELINE-CPU-V[12]\b")

#: path prefix -> why naming the retired protocol id there is legitimate.
M1_PROTOCOL_ALLOWED: dict[str, str] = {
    "benchmarks/reports": (
        "the milestone record. Those runs really executed under that protocol id, "
        "and rewriting the report would falsify what was run"
    ),
    "benchmarks/inventory.yaml": "the deleted: block has to name what it deleted",
    "benchmarks/INVENTORY.md": "generated from the triage, so it names the same rows",
    "benchmarks/README.md": "says which protocol went and on what ground",
    "benchmarks/protocols/m2_coupler_protocol.md": (
        "records that M2 used to extend this protocol and that the rules are now "
        "declared rather than inherited -- the M2 contract's own history"
    ),
    "benchmarks/protocols/coupler_protocol.yaml": (
        "same, in the machine-readable half: the comment where `extends:` used to be"
    ),
    "benchmarks/schemas/provenance.schema.json": (
        "the retired schema whose protocol_id enum is a CHE-133 finding with its "
        "removal owned by CHE-115; it describes records that were produced under "
        "these ids, and it is read only by archived tests"
    ),
    "src/solvers/optiland/baseline.py": (
        "docstring citing the contract the standalone runner was built to. Deferred, "
        "not kept: benchmarks/inventory.yaml carries it as a pending edit, because "
        "this file is inside every instance record's code_fingerprint and a "
        "docstring edit would stale all of them"
    ),
    "src/solvers/chromatix/baseline.py": "the same deferral, for the wave runner",
    "tests/test_retired_taxonomy.py": "this file, which has to spell what it bans",
}


def test_the_m1_baseline_protocol_is_gone() -> None:
    """CHE-106 (M1.1) deleted it once the B1 families expressed its content.

    The condition was written into the artifact's own inventory row -- "moves once
    M1 expresses its content executably" -- so the deletion is the row being
    honoured rather than a cleanup. What must not come back is a benchmark
    declaring a ``protocol_id`` that no file defines: an execution contract that
    exists only as a string in a result is exactly the unfalsifiable claim the
    family schema replaced.
    """
    assert not (ROOT / "benchmarks/protocols/protocol.yaml").exists()
    assert not (ROOT / "benchmarks/protocols/m1_baseline_protocol.md").exists()

    offenders = {}
    for rel in _scanned_files():
        text = (ROOT / rel).read_text(encoding="utf-8", errors="ignore")
        if not M1_PROTOCOL.search(text):
            continue
        if any(
            str(rel) == prefix or str(rel).startswith(prefix + "/")
            for prefix in M1_PROTOCOL_ALLOWED
        ):
            continue
        offenders[str(rel)] = sorted(set(M1_PROTOCOL.findall(text)))
    assert not offenders, (
        "these files name the retired M1 baseline protocol and are not on the "
        f"allowlist in this file: {offenders}\n"
        "A B0-B4 family declares its own execution policy, tolerances and validity; "
        "there is no protocol id to inherit."
    )


def test_what_the_m1_protocol_carried_still_exists_somewhere() -> None:
    """The deletion is only honest if the content it held survives.

    Named here rather than left to the inventory prose, so that moving one of
    these out from under the deletion fails a test instead of quietly making the
    ``deleted:`` row wrong.
    """
    assert (ROOT / "benchmarks/probes/engine_independence.py").is_file(), (
        "the engine-independence rule the protocol froze"
    )
    assert (ROOT / "benchmarks/probes/records/che12_engine_report.json").is_file(), (
        "and the record that shows it passed"
    )
    for pack in ("optiland", "chromatix"):
        assert (ROOT / f"knowledge/solvers/{pack}/conventions.md").is_file(), (
            "the boundary conventions, measured rather than declared"
        )
    for module in ("b1_ray", "b1_wave"):
        source = (ROOT / f"src/verification/families/{module}.py").read_text(encoding="utf-8")
        assert "ExecutionPolicy(" in source, (
            f"{module}: the device/dtype half of the protocol is a family policy now"
        )
