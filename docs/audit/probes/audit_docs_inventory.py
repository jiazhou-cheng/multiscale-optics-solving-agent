#!/usr/bin/env python3
"""M0.2 audit probe: inventory docs/knowledge/linear material and coupler terms.

Read-only. Prints Markdown tables and a JSON blob to stdout; writes nothing.

Covers:
  * every file under knowledge/, docs/, linear/, benchmarks/, examples/, schemas/
    plus root *.md and CONTEXT_MANIFEST.yaml,
  * a term scan separating ray->wave from wave->ray coupler evidence,
  * CONTEXT_MANIFEST.yaml declared-path existence,
  * knowledge/solver_cards/<s>.yaml vs knowledge/solvers/<s>/solver_card.yaml.

Usage (container only):
    ./run.sh python docs/audit/probes/audit_docs_inventory.py
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

ROOT = Path("/workspace") if Path("/workspace").is_dir() else Path.cwd()
DOC_DIRS = ("knowledge", "docs", "linear", "benchmarks", "examples", "schemas")

RAY_TO_WAVE = re.compile(r"c_ray_to_wave|ray[_\s-]*to[_\s-]*wave|ray2wave|ray->wave", re.I)
WAVE_TO_RAY = re.compile(r"c_wave_to_ray|wave[_\s-]*to[_\s-]*ray|wave2ray|wave->ray", re.I)
LEGACY = re.compile(r"ray_wave|ray_ewave|raywave|ray-wave", re.I)


def doc_files() -> list[Path]:
    out: list[Path] = []
    for d in DOC_DIRS:
        base = ROOT / d
        if not base.is_dir():
            continue
        out.extend(
            p
            for p in base.rglob("*")
            if p.is_file() and "__pycache__" not in p.parts and not p.name.startswith(".")
        )
    out.extend(ROOT.glob("*.md"))
    out.append(ROOT / "CONTEXT_MANIFEST.yaml")
    return sorted({p for p in out if p.is_file()})


def first_heading(text: str) -> str:
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("#"):
            return s.lstrip("#").strip()
        if s.startswith(("id:", "name:", "solver:", "title:")):
            return s
    for line in text.splitlines():
        if line.strip():
            return line.strip()[:120]
    return ""


def main() -> None:
    files = doc_files()
    by_hash: dict[str, list[str]] = {}
    rows = []
    for p in files:
        rel = str(p.relative_to(ROOT))
        try:
            text = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = ""
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12] if text else "empty"
        by_hash.setdefault(digest, []).append(rel)
        rows.append(
            {
                "path": rel,
                "bytes": p.stat().st_size,
                "lines": len(text.splitlines()),
                "sha256_12": digest,
                "heading": first_heading(text),
                "n_ray_to_wave": len(RAY_TO_WAVE.findall(text)),
                "n_wave_to_ray": len(WAVE_TO_RAY.findall(text)),
                "n_legacy_raywave": len(LEGACY.findall(text)),
            }
        )

    exact_dupes = {h: paths for h, paths in by_hash.items() if len(paths) > 1 and h != "empty"}

    pairs = []
    cards_dir = ROOT / "knowledge" / "solver_cards"
    if cards_dir.is_dir():
        for card in sorted(cards_dir.glob("*.yaml")):
            solver = card.stem
            other = ROOT / "knowledge" / "solvers" / solver / "solver_card.yaml"
            a = card.read_text(encoding="utf-8")
            b = other.read_text(encoding="utf-8") if other.is_file() else None
            top = lambda t: sorted(  # noqa: E731
                line.split(":")[0]
                for line in t.splitlines()
                if line and not line.startswith((" ", "#", "-")) and ":" in line
            )
            pairs.append(
                {
                    "solver": solver,
                    "flat_card": str(card.relative_to(ROOT)),
                    "nested_card": str(other.relative_to(ROOT)) if other.is_file() else None,
                    "flat_lines": len(a.splitlines()),
                    "nested_lines": len(b.splitlines()) if b is not None else None,
                    "identical": (b is not None and a == b),
                    "flat_top_keys": top(a),
                    "nested_top_keys": top(b) if b is not None else None,
                }
            )

    manifest_paths = []
    mtext = (ROOT / "CONTEXT_MANIFEST.yaml").read_text(encoding="utf-8")
    for token in re.findall(r"[A-Za-z0-9_./-]+\.(?:md|yaml|yml|json)|[a-z_]+/", mtext):
        token = token.strip()
        if token in {"", "/"}:
            continue
        manifest_paths.append({"declared": token, "exists": (ROOT / token).exists()})
    seen: set[str] = set()
    manifest_paths = [
        m for m in manifest_paths if not (m["declared"] in seen or seen.add(m["declared"]))
    ]

    top_dirs = sorted(
        p.name + "/" for p in ROOT.iterdir() if p.is_dir() and not p.name.startswith(".")
    )
    manifest_missing_dirs = [d for d in top_dirs if d not in mtext]

    summary = {
        "doc_files": len(rows),
        "exact_duplicate_groups": len(exact_dupes),
        "files_mentioning_ray_to_wave": sum(1 for r in rows if r["n_ray_to_wave"]),
        "files_mentioning_wave_to_ray": sum(1 for r in rows if r["n_wave_to_ray"]),
        "files_mentioning_legacy_raywave": sum(1 for r in rows if r["n_legacy_raywave"]),
        "manifest_declared_paths": len(manifest_paths),
        "manifest_missing_paths": [m["declared"] for m in manifest_paths if not m["exists"]],
        "top_dirs_absent_from_manifest": manifest_missing_dirs,
    }

    print("### JSON")
    print(
        json.dumps(
            {
                "summary": summary,
                "files": rows,
                "exact_duplicates": exact_dupes,
                "solver_card_pairs": pairs,
                "manifest_paths": manifest_paths,
            },
            indent=2,
        )
    )

    print()
    print("### DOC TABLE")
    print("| path | lines | r2w | w2r | legacy | heading |")
    print("| --- | --- | --- | --- | --- | --- |")
    for r in rows:
        print(
            f"| `{r['path']}` | {r['lines']} | {r['n_ray_to_wave']} | {r['n_wave_to_ray']} | "
            f"{r['n_legacy_raywave']} | {r['heading'][:90] or '—'} |"
        )

    print()
    print("### SOLVER CARD PAIRS")
    for p in pairs:
        print(json.dumps(p))

    print()
    print("### MANIFEST PATHS")
    for m in manifest_paths:
        print(f"{'OK  ' if m['exists'] else 'MISS'} {m['declared']}")

    print()
    print("### SUMMARY")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
