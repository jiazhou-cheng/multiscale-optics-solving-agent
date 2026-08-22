#!/usr/bin/env python3
"""M0.1 audit probe: build the internal import graph and file inventory.

Read-only. Prints a machine-readable JSON blob plus Markdown tables to stdout;
writes nothing to disk, so it can run as root inside the container without
leaving root-owned files in the mounted repository.

Usage (container only):
    ./run.sh python docs/audit/probes/audit_import_graph.py
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

ROOT = Path("/workspace") if Path("/workspace").is_dir() else Path.cwd()
SCAN_DIRS = ("src", "tests", "scripts", "examples", "benchmarks", "schemas", "tmp_probes")
PKG = "multiscale_optics_agent"


def py_files() -> list[Path]:
    out: list[Path] = []
    for d in SCAN_DIRS:
        base = ROOT / d
        if not base.is_dir():
            continue
        out.extend(p for p in base.rglob("*.py") if "__pycache__" not in p.parts)
    return sorted(out)


def data_files() -> list[Path]:
    out: list[Path] = []
    for d in SCAN_DIRS:
        base = ROOT / d
        if not base.is_dir():
            continue
        for pattern in ("*.yaml", "*.yml", "*.json"):
            out.extend(p for p in base.rglob(pattern) if "__pycache__" not in p.parts)
    return sorted(out)


def module_name(path: Path) -> str:
    rel = path.relative_to(ROOT)
    parts = list(rel.parts)
    if parts[0] == "src":
        parts = parts[1:]
    parts[-1] = parts[-1][: -len(".py")]
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def docstring_line(tree: ast.Module) -> str:
    doc = ast.get_docstring(tree)
    if not doc:
        return ""
    return doc.strip().splitlines()[0].strip()


def main() -> None:
    files = py_files()
    mod_to_path = {module_name(p): p for p in files}
    known_pkg_mods = {m for m in mod_to_path if m.startswith(PKG)}

    records: dict[str, dict] = {}
    imports_of: dict[str, set[str]] = {}
    string_literals: dict[str, set[str]] = {}
    external: dict[str, set[str]] = {}

    for path in files:
        mod = module_name(path)
        src = path.read_text(encoding="utf-8")
        tree = ast.parse(src, filename=str(path))
        targets: set[str] = set()
        lits: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    targets.add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.level:  # relative import
                    base = mod.rsplit(".", node.level)[0] if "." in mod else mod
                    targets.add(f"{base}.{node.module}" if node.module else base)
                elif node.module:
                    targets.add(node.module)
                    for alias in node.names:
                        targets.add(f"{node.module}.{alias.name}")
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                v = node.value
                if any(v.endswith(ext) for ext in (".yaml", ".yml", ".json", ".md", ".py")):
                    lits.add(v)
        string_literals[mod] = lits

        internal: set[str] = set()
        for t in targets:
            cands = [m for m in known_pkg_mods if t == m or t.startswith(m + ".")]
            if cands:
                internal.add(max(cands, key=len))
        internal.discard(mod)
        imports_of[mod] = internal
        external[mod] = {
            t.split(".")[0]
            for t in targets
            if not t.startswith(PKG) and not t.startswith("tests")
        }

        records[mod] = {
            "path": str(path.relative_to(ROOT)),
            "purpose": docstring_line(tree),
            "loc": len(src.splitlines()),
            "imports_internal": sorted(internal),
            "imports_external_top": sorted(external[mod]),
        }

    imported_by: dict[str, set[str]] = {m: set() for m in mod_to_path}
    for mod, targets in imports_of.items():
        for t in targets:
            imported_by.setdefault(t, set()).add(mod)

    for mod, rec in records.items():
        by = sorted(imported_by.get(mod, set()))
        rec["imported_by"] = by
        rel = rec["path"]
        if rel.startswith("tests/"):
            status = "test"
        elif rel.startswith(("scripts/", "tmp_probes/", "docs/")):
            status = "standalone-script"
        elif not by:
            status = "unreferenced-by-python"
        elif all(b.startswith("tests") for b in by):
            status = "test-only"
        else:
            status = "active"
        rec["status"] = status

    assets = []
    for p in data_files():
        rel = str(p.relative_to(ROOT))
        name = p.name
        referenced = sorted(
            m for m, lits in string_literals.items() if any(name in lit for lit in lits)
        )
        assets.append(
            {
                "path": rel,
                "bytes": p.stat().st_size,
                "referenced_by_python": referenced,
            }
        )

    summary = {
        "root": str(ROOT),
        "python_files": len(files),
        "by_status": {
            s: sum(1 for r in records.values() if r["status"] == s)
            for s in sorted({r["status"] for r in records.values()})
        },
        "asset_files": len(assets),
        "unreferenced_assets": sum(1 for a in assets if not a["referenced_by_python"]),
        "external_packages_imported": sorted(
            {e for mod in external for e in external[mod]}
        ),
    }

    print("### JSON")
    print(json.dumps({"summary": summary, "modules": records, "assets": assets}, indent=2))

    print()
    print("### MODULE TABLE")
    print("| module | path | loc | status | imported_by | purpose |")
    print("| --- | --- | --- | --- | --- | --- |")
    for mod in sorted(records):
        r = records[mod]
        by = ", ".join(b.replace(PKG + ".", "") for b in r["imported_by"]) or "—"
        print(
            f"| `{mod}` | `{r['path']}` | {r['loc']} | {r['status']} | {by} | "
            f"{r['purpose'] or '—'} |"
        )

    print()
    print("### ASSET TABLE")
    print("| asset | bytes | referenced_by_python |")
    print("| --- | --- | --- |")
    for a in assets:
        ref = ", ".join(a["referenced_by_python"]) or "—"
        print(f"| `{a['path']}` | {a['bytes']} | {ref} |")

    print()
    print("### SUMMARY")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
