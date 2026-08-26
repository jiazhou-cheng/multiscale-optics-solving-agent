"""Render ``benchmarks/INVENTORY.md`` from ``benchmarks/inventory.yaml``.

CHE-130 (M0.5.1). The YAML is the source of truth because the coverage test has
to read it; the Markdown exists so the triage is reviewable in a diff. Keeping
one generated from the other is the only way the two cannot disagree, and
``tests/test_benchmark_inventory.py`` re-renders in memory and compares rather
than trusting that somebody remembered to run this.

    ./run.sh python scripts/generate_benchmark_inventory.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from core.paths import repository_root
from verification.families import FAMILIES, BenchmarkLayer

__all__ = ["main", "render"]

BUCKET_TITLE = {
    "A": "A — reusable scientific infrastructure or evidence",
    "B": "B — candidate canonical case",
    "C": "C — obsolete task layer",
}

#: CHE-141 (M2.5). The layer axis is orthogonal to the B0-B4 categories, so the
#: inventory needs its own grouping by it: a reader asking "what supports this
#: system claim" should not have to open six family modules to find out.
LAYER_TITLE = {
    BenchmarkLayer.QUALIFICATION: "A — qualification",
    BenchmarkLayer.NUMERICAL: "B — numerical realization and validity",
    BenchmarkLayer.SYSTEM: "C — system",
}

LAYER_QUESTION = {
    BenchmarkLayer.QUALIFICATION: (
        "is this operator the thing it claims to be? Conventions, invariants, "
        "estimator exactness. Every family here owes a negative control."
    ),
    BenchmarkLayer.NUMERICAL: (
        "does a choice that should not move the answer stay inside its budget, "
        "and where does it stop? Every family here owes a refinement dimension."
    ),
    BenchmarkLayer.SYSTEM: (
        "is a physically meaningful end-to-end optical system modelled "
        "correctly? Every family here owes a topology and two observables."
    ),
}


def _cell(text: str | None) -> str:
    """One table cell: newlines flattened, pipes escaped."""
    if not text:
        return "—"
    return " ".join(str(text).split()).replace("|", "\\|")


def _artifact(entry: dict) -> str:
    path = f"`{entry['path']}`"
    part = entry.get("part")
    return f"{path}<br>*{_cell(part)}*" if part else path


def _layer_view() -> list[str]:
    """The families grouped by layer, read from the registry and not from YAML.

    Deliberately sourced from ``src/verification/families/`` rather than from
    ``inventory.yaml``: ``layer`` is a required field with no default on
    ``BenchmarkFamily``, so a table generated from the families themselves
    cannot drift from them, whereas a hand-maintained copy in the YAML would be
    a second place for one fact and therefore one place for them to disagree.
    """
    families = sorted(FAMILIES.values(), key=lambda f: f.family_id)
    lines = [
        "## Families by layer — CHE-141 (M2.5)",
        "",
        "Generated from `src/verification/families/`, not from "
        "`benchmarks/inventory.yaml`. The B0-B4 category says what may *decide* "
        "a family; the layer says what is being *claimed*, and the two are "
        "independent — `B3-PSF-SINGLET` and `B3-DUALROUTE` share a category, "
        "and one is a statement about an optical system while the other "
        "compares two numerical realizations of it.",
        "",
        "Layer-C artifacts authored from M2.7 onward live in "
        "`benchmarks/systems/`. Existing layer-C evidence is **re-homed by "
        "classification, not moved on disk** — see "
        "`benchmarks/systems/README.md` for why, and "
        "`docs/benchmark_design.md` for the axis and its three consistency "
        "rules.",
        "",
        "The instance counts below are the instances declared **on the family "
        "object**. B3 and B4 families construct theirs in their "
        "`benchmarks/instances/` drivers instead, so a zero there is a fact "
        "about where the instances are built and not an absence of evidence — "
        "`benchmarks/instances/records/` is the committed record set.",
        "",
    ]
    for layer in (
        BenchmarkLayer.QUALIFICATION,
        BenchmarkLayer.NUMERICAL,
        BenchmarkLayer.SYSTEM,
    ):
        rows = [f for f in families if f.layer is layer]
        instances = sum(len(f.canonical_instances) for f in rows)
        lines += [
            f"### Layer {LAYER_TITLE[layer]}",
            "",
            f"{LAYER_QUESTION[layer]} {len(rows)} families, "
            f"{instances} canonical instances declared on the family.",
            "",
            "| family | category | components | what it claims |",
            "| -- | -- | -- | -- |",
        ]
        for fam in rows:
            claim = fam.question
            if layer is BenchmarkLayer.SYSTEM:
                claim = (
                    "**topology:** "
                    + " → ".join(fam.topology)
                    + "<br>**observables:** "
                    + ", ".join(sorted({m.name for m in fam.metrics}))
                )
            lines.append(
                f"| `{fam.family_id}` | {fam.category.value} | "
                + ", ".join(f"`{c}`" for c in fam.components)
                + f" | {_cell(claim)} |"
            )
        lines.append("")
    return lines


def render(inventory: dict) -> str:
    entries = inventory["entries"]
    counts = {b: sum(1 for e in entries if e["bucket"] == b) for b in "ABC"}

    lines: list[str] = [
        "<!-- GENERATED by scripts/generate_benchmark_inventory.py from "
        "benchmarks/inventory.yaml. Do not edit. -->",
        "",
        "# Benchmark artifact inventory — CHE-130 (M0.5.1)",
        "",
        "Every artifact under "
        + ", ".join(f"`{s}`" for s in inventory["scope"])
        + " classified into exactly one bucket, **artifact by artifact rather than "
        "directory by directory**. Files carrying more than one verdict appear once "
        "per part.",
        "",
        "| bucket | test | action | rows |",
        "| -- | -- | -- | -- |",
        "| **A** | would this still be true and useful if no agent and no benchmark "
        "task existed? | preserve; name the destination | "
        f"{counts['A']} |",
        "| **B** | is this a specific physical setup whose result is worth freezing? "
        f"| a positive justification is required | {counts['B']} |",
        "| **C** | does this exist only to serve the old evaluation design? | delete "
        f"| {counts['C']} |",
        "",
        "Nothing is deleted by this inventory. Deletion is CHE-133 (M0.5.4), and the "
        "`L2-PSF-01` runner goes later still — it is the only way to run that case "
        "until the executor and family runner exist.",
        "",
        "Coverage is enforced by `tests/test_benchmark_inventory.py`: it enumerates "
        "the scope directories and fails if a file is present in the tree and absent "
        "from this table, or if a row matches nothing. An inventory that silently "
        "omits a file is how evidence gets deleted.",
        "",
    ]

    for bucket in "ABC":
        rows = [e for e in entries if e["bucket"] == bucket]
        lines += [f"## Bucket {BUCKET_TITLE[bucket]}", ""]
        if bucket == "C":
            lines += ["| artifact | why it goes |", "| -- | -- |"]
            lines += [
                f"| {_artifact(e)} | {_cell(e['justification'])}"
                + (f" **{_cell(e['note'])}**" if e.get("note") else "")
                + " |"
                for e in rows
            ]
        else:
            lines += [
                "| artifact | justification | destination in the new architecture |",
                "| -- | -- | -- |",
            ]
            lines += [
                f"| {_artifact(e)} | {_cell(e['justification'])}"
                + (f" **{_cell(e['note'])}**" if e.get("note") else "")
                + f" | {_cell(e.get('destination'))} |"
                for e in rows
            ]
        lines.append("")

    deleted = inventory.get("deleted", ())
    if deleted:
        lines += [
            "## Removed, and where the content went",
            "",
            "Kept here rather than as table rows: the coverage test refuses a row that "
            "matches no file, because a classification of something that is not there "
            "reads as coverage and is not.",
            "",
            "| artifact | n | removed by | survives as |",
            "| -- | -- | -- | -- |",
        ]
        lines += [
            f"| `{d['path']}` | {d.get('count', 1)} | {_cell(d['by'])} | "
            f"{_cell(d['survives_as'])} |"
            for d in deleted
        ]
        lines.append("")

    lines += _layer_view()

    lines += ["## Edits this triage requires but does not make", ""]
    lines += [
        "| file | section | required change | owner | why not here |",
        "| -- | -- | -- | -- | -- |",
    ]
    for edit in inventory.get("pending_edits", ()):
        lines.append(
            f"| `{edit['path']}` | {_cell(edit['section'])} | "
            f"{_cell(edit['required_change'])} | {_cell(edit['owner'])} | "
            f"{_cell(edit['why_not_here'])} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    root = repository_root()
    inventory = yaml.safe_load((root / "benchmarks/inventory.yaml").read_text("utf-8"))
    out = root / "benchmarks/INVENTORY.md"
    out.write_text(render(inventory), encoding="utf-8")
    print(f"wrote {out.relative_to(root)} ({len(inventory['entries'])} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
