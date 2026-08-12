#!/usr/bin/env python3
"""M0.2: classify every documentation file with an explicit, reproducible rule.

Classification vocabulary (CHE-6):
    canonical            single source of truth for something the project needs now
    duplicate-of-X       content that overlaps X and is not the source of truth
    off-scope reference  correct and useful, but outside the v0.1 Optiland /
                         Chromatix / ray-wave-coupling scope; retrieval-only
    stale                contradicted by the current repository state

Scope note: v0.1 active scope is Optiland, Chromatix, and the ray-wave coupling
layer (BOTH directions: C_RAY_TO_WAVE and C_WAVE_TO_RAY). The initial executable
forward pipeline exercises only C_RAY_TO_WAVE; wave-to-ray material is still in
scope and must not be archived for being unexercised.

Rules are applied in order: explicit override, then directory rule, then a
default. Every row carries the rationale that produced it.

Usage (container only):
    ./run.sh python docs/audit/probes/classify_docs.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("/workspace") if Path("/workspace").is_dir() else Path.cwd()

IN_SCOPE_SOLVERS = ("optiland", "chromatix")
OFF_SCOPE_SOLVERS = ("fmmax", "fdtdx", "jax_fem", "sax")

# path -> (classification, rationale)
OVERRIDES: dict[str, tuple[str, str]] = {
    "AGENTS.md": ("canonical", "declared canonical_static_context in CONTEXT_MANIFEST.yaml"),
    "CLAUDE.md": ("canonical", "one-line pointer to AGENTS.md; enforced by check_context_sync"),
    "CONTEXT_MANIFEST.yaml": ("canonical", "machine-readable loading policy"),
    "README.md": (
        "canonical",
        "repository entry doc; rewritten under CHE-7 to carry the container-only rule",
    ),
    "MIGRATION_PLAN.md": (
        "stale",
        "describes a context migration whose steps 1-3 are already done and lists eight "
        "documents to archive (PROJECT_PLAN.md, PAPER_INTRODUCTION.md, ...) that no longer "
        "exist in this repository; its 'Target Structure' remains the only written source "
        "for docs/archive/, so extract that before archiving",
    ),
    "VALIDATION_REPORT.md": (
        "stale",
        "dated 2026-07-29; claims 8 tests, solvers not installed, ruff/mypy unavailable. "
        "Current evidence: 52-test baseline, all 19 solver probes MATCH, ruff runs",
    ),
    "knowledge/README.md": (
        "canonical",
        "explains the knowledge/ layout; still accurate for solver_cards/ and papers/",
    ),
    "knowledge/source_manifest.yaml": ("canonical", "authoritative upstream source links"),
    "docs/context/CURRENT_SCOPE.md": (
        "canonical",
        "current-scope doc for the ray/wave slice; names the canonical graph",
    ),
    "docs/context/RAY_WAVE_VERTICAL_SLICE.md": (
        "canonical",
        "execution plan for the active slice",
    ),
    "docs/context/MODULE_GRANULARITY.md": (
        "canonical",
        "granularity rules referenced by AGENTS.md",
    ),
    "linear/ISSUE_TEMPLATE.md": (
        "duplicate-of Linear project description",
        "the Linear project mandates a different eight-section issue format "
        "(Goal/Context/Acceptance Criteria/Out of Scope/Dependencies/Likely Files Affected/"
        "Verification Commands/Required Deliverables); this file's template differs",
    ),
    "linear/PROJECT_SETUP.md": (
        "stale",
        "describes a project named 'Ray-Wave Vertical Slice' with workflow states and labels "
        "that do not match the live Linear project or team states (Backlog/Todo/In Progress/"
        "In Review/Done/Canceled/Duplicate, no labels defined)",
    ),
    "linear/BACKLOG_RAY_WAVE.md": (
        "duplicate-of Linear project description",
        "a proposed backlog superseded by the live project's M0-M4 issue list",
    ),
    "benchmarks/README.md": ("off-scope reference", "benchmark suite is M3, not M0-M2"),
    "benchmarks/manifest.yaml": ("off-scope reference", "benchmark suite is M3"),
    "benchmarks/level1/README.md": ("off-scope reference", "benchmark suite is M3"),
    "benchmarks/level2/README.md": ("off-scope reference", "benchmark suite is M3"),
    "benchmarks/level3/README.md": ("off-scope reference", "benchmark suite is M3"),
    "examples/graphs/ray_to_wave.yaml": (
        "canonical",
        "the only graph exercising C_RAY_TO_WAVE; loaded by tests/test_graph_validation.py",
    ),
}


def classify(rel: str) -> tuple[str, str]:
    if rel in OVERRIDES:
        return OVERRIDES[rel]

    parts = rel.split("/")

    if rel.startswith("docs/audit/"):
        return ("canonical", "M0 audit output produced by CHE-5/CHE-6")

    if rel.startswith("knowledge/solvers/"):
        solver = parts[2]
        if solver in IN_SCOPE_SOLVERS:
            return ("canonical", f"in-scope solver knowledge pack ({solver})")
        if solver in OFF_SCOPE_SOLVERS:
            return ("off-scope reference", f"{solver} is out of scope for v0.1; retrieval-only")
        return ("unclassified", "unknown solver directory")

    if rel.startswith("knowledge/solver_cards/"):
        solver = Path(rel).stem
        scope = "in-scope" if solver in IN_SCOPE_SOLVERS else "off-scope"
        return (
            f"duplicate-of knowledge/solvers/{solver}/solver_card.yaml",
            f"routing-card subset of the nested validation card ({scope}); "
            "verified divergent by machine diff, different key sets",
        )

    if rel.startswith("examples/graphs/"):
        return ("off-scope reference", "smoke graph for a solver outside the v0.1 slice")

    if rel.startswith("schemas/"):
        return (
            "canonical",
            "generated by scripts/export_schemas.py; regeneration produced no diff",
        )

    return ("unclassified", "no rule matched")


def main() -> None:
    rows = []
    targets: list[Path] = []
    for d in ("knowledge", "docs", "linear", "benchmarks", "examples", "schemas"):
        base = ROOT / d
        if base.is_dir():
            targets.extend(
                p
                for p in base.rglob("*")
                if p.is_file() and "__pycache__" not in p.parts and not p.name.startswith(".")
            )
    targets.extend(ROOT.glob("*.md"))
    targets.append(ROOT / "CONTEXT_MANIFEST.yaml")

    for p in sorted({t for t in targets if t.is_file()}):
        rel = str(p.relative_to(ROOT))
        cls, why = classify(rel)
        rows.append({"path": rel, "classification": cls, "rationale": why})

    counts: dict[str, int] = {}
    for r in rows:
        key = r["classification"].split(" ")[0]
        counts[key] = counts.get(key, 0) + 1

    print("### CLASSIFICATION TABLE")
    print("| path | classification | rationale |")
    print("| --- | --- | --- |")
    for r in rows:
        print(f"| `{r['path']}` | {r['classification']} | {r['rationale']} |")

    print()
    print("### SUMMARY")
    print(json.dumps({"files": len(rows), "by_classification": counts}, indent=2))

    print()
    print("### JSON")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
