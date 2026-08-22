#!/usr/bin/env python3
"""CHE-64: merge resource profiles into one per-test inventory and report.

Reads the JSON files written by ``scripts/pytest_resource_profile.py`` (one per
profiling chunk), joins them against an AST pass over ``tests/`` for each test's
declared purpose, and emits:

* ``docs/testing/test_inventory.json`` -- the machine-readable inventory
* ``docs/testing/test_inventory.md``   -- the human review table

The copies CHE-64 published at those paths were archived to
``docs/archive/2026-08-testing/`` once CHE-67 moved the tests they index, so a
run of this script now writes a *current* inventory rather than overwriting a
historical record. Re-running it needs the profiler's JSON first, which is
2642 s of test execution.

Run:
    ./run.sh python scripts/build_test_inventory.py

Purpose text is taken from the test's docstring where it has one and derived from
its name where it does not; every row records which, because "derived from the
name" is a weaker claim than "the author wrote this down" and the reviewer
deciding whether to delete a test should be able to tell them apart.
"""

from __future__ import annotations

import ast
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PROFILE_DIR = ROOT / "outputs" / "CHE-64"
OUT_JSON = ROOT / "docs" / "testing" / "test_inventory.json"
OUT_MD = ROOT / "docs" / "testing" / "test_inventory.md"

#: Tier A's selection expression, mirrored from AGENTS.md. A test is Tier A iff
#: it carries none of these markers.
#:
#: CHE-67 note: `benchmark`, `fmmax` and `fdtdx` no longer select anything
#: under `tests/` -- every test carrying them was archived to
#: `archive/tests/gen1/`, which is outside pytest collection. They are kept in this
#: set so re-running this script over a *pre-CHE-67* profile still reproduces the
#: tier labels that `docs/archive/2026-08-testing/test_inventory.md` was published
#: with. `tutorial`
#: is deliberately absent: the tutorials are now separated by directory
#: (`tests_tutorial/`), not by marker, so a profile of that suite should show its
#: real cost rather than a tier label.
# `sax` is deliberately absent: CHE-72 deleted the marker with the integration.
# The historical Tier A expression in docs/archive/2026-08-testing/ still says
# `not sax`, which
# keeps working because an unknown name evaluates false in a `-m` expression.
TIER_A_EXCLUDING_MARKERS = {"slow", "benchmark", "fmmax", "fdtdx"}
#: Quarantined to its own session (CHE-60): needs `./run.sh --gpu ... -m gpu`.
GPU_MARKER = "gpu"

#: Flakiness measured OUT OF BAND, by repeated runs, not by this profiler. The
#: profiler can only call a test flaky if it happened to observe two different
#: outcomes, which needs the test to be profiled twice -- so a genuine flake seen
#: once would otherwise be filed as a plain failure and lost. Keyed by nodeid.
KNOWN_FLAKY: dict[str, dict[str, str]] = {
    "tests/test_optiland_tutorials.py::test_tutorial_reproduction"
    "[t21_surface_roughness_scattering]": {
        "measured_rate": "~2 failures in ~22 runs (about 9%)",
        "evidence": "CHE-64: repeated single-test runs; passes in isolation, so "
        "this is nondeterminism, not test-order dependence",
        "cause": "hard threshold `centroid_offset_over_rms < 0.1` on a random "
        "quantity (line 122/135), and the resulting bool is compared EXACTLY by "
        "the harness, bypassing the reproduction's declared metric_rtol=0.35. "
        "Cannot be seeded: optiland.scatter is numba-compiled and its RNG is "
        "unreachable from numpy.",
        "disposition": "not fixed in CHE-64 -- widening the bound is a tolerance "
        "decision on a physical claim. See docs/archive/2026-08-testing/test_runtime_audit.md F1.",
    },
}


def _tier(markers: set[str]) -> str:
    if GPU_MARKER in markers:
        return "GPU (own session)"
    hit = sorted(markers & TIER_A_EXCLUDING_MARKERS)
    if not hit:
        return "A"
    if "benchmark" in hit:
        return "C (benchmark)"
    if {"fmmax", "fdtdx"} & set(hit):
        return "B (out-of-scope solver)"
    return "B (slow)"


def _humanize(name: str) -> str:
    """Turn `test_the_rim_does_not_sharpen` into `The rim does not sharpen`."""
    body = re.sub(r"^test_?", "", name)
    body = body.replace("_", " ").strip()
    return body[:1].upper() + body[1:] if body else name


def _first_sentence(text: str) -> str:
    line = " ".join(text.strip().split())
    match = re.match(r"(.+?[.!?])(?:\s|$)", line)
    return (match.group(1) if match else line).strip()


def collect_purposes() -> dict[tuple[str, str], dict[str, str]]:
    """Map (file, test function name) -> purpose text and its provenance."""
    out: dict[tuple[str, str], dict[str, str]] = {}
    # Both active test roots: the default suite and the on-demand tutorial suite
    # (CHE-67). Missing the latter would silently downgrade every tutorial test's
    # purpose to "derived from the name".
    paths = sorted(ROOT.glob("tests/**/*.py")) + sorted(ROOT.glob("tests_tutorial/**/*.py"))
    for path in paths:
        rel = str(path.relative_to(ROOT))
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:
            continue

        def visit(node: ast.AST, file_rel: str, prefix: str = "") -> None:
            for child in ast.iter_child_nodes(node):
                if isinstance(child, ast.ClassDef):
                    visit(child, file_rel, f"{child.name}::")
                elif isinstance(
                    child, (ast.FunctionDef, ast.AsyncFunctionDef)
                ) and child.name.startswith("test"):
                    doc = ast.get_docstring(child)
                    out[(file_rel, prefix + child.name)] = (
                        {"purpose": _first_sentence(doc), "purpose_source": "docstring"}
                        if doc
                        else {
                            "purpose": _humanize(child.name),
                            "purpose_source": "derived-from-name",
                        }
                    )

        visit(tree, rel)
    return out


def load_profiles() -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Merge every profile chunk. Later chunks win; partials are recorded."""
    merged: dict[str, dict[str, Any]] = {}
    chunks: list[dict[str, Any]] = []
    for path in sorted(PROFILE_DIR.glob("*.json")):
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if "tests" not in payload:
            continue
        chunks.append(
            {
                "file": path.name,
                "complete": payload.get("complete"),
                "session_seconds": payload.get("session_seconds"),
                "test_count": payload.get("test_count"),
                "markexpr": payload.get("selection", {}).get("markexpr", ""),
                "swap_breach": payload.get("swap_guard", {}).get("breach"),
            }
        )
        for record in payload["tests"]:
            nodeid = record["nodeid"]
            prior = merged.get(nodeid)
            # Keep the SLOWEST observation when a test was profiled more than
            # once. An audit that under-reports cost is worse than one that
            # over-reports it, and a test seen fast once and slow once is a test
            # whose cost is not yet understood.
            if prior is None or record["duration_s"] > prior["duration_s"]:
                merged[nodeid] = record
            # Never let a later PASS erase an earlier FAIL: a test that failed in
            # any observed run is flaky, and that is the single most important
            # thing this inventory can surface.
            if prior is not None and prior["outcome"] == "failed":
                merged[nodeid]["outcome"] = "failed"
                merged[nodeid]["flaky"] = True
            if prior is not None and record["outcome"] != prior["outcome"]:
                merged[nodeid]["flaky"] = True
                merged[nodeid]["observed_outcomes"] = sorted(
                    {prior["outcome"], record["outcome"]}
                )
    return merged, chunks


def _key(nodeid: str) -> tuple[str, str]:
    """nodeid -> (file, test name without parametrization)."""
    file_part, _, rest = nodeid.partition("::")
    parts = [p for p in rest.split("::") if p]
    if parts:
        parts[-1] = parts[-1].split("[")[0]
    if len(parts) >= 2:
        return file_part, f"{parts[-2]}::{parts[-1]}"
    return file_part, parts[-1] if parts else ""


def build() -> dict[str, Any]:
    purposes = collect_purposes()
    profiles, chunks = load_profiles()

    rows: list[dict[str, Any]] = []
    for nodeid, record in sorted(profiles.items()):
        file_part, name = _key(nodeid)
        info = purposes.get((file_part, name)) or purposes.get(
            (file_part, name.split("::")[-1])
        )
        markers = set(record.get("markers") or [])
        rows.append(
            {
                "nodeid": nodeid,
                "file": file_part,
                "test": name,
                "markers": sorted(markers),
                "current_tier": _tier(markers),
                "purpose": (info or {}).get("purpose", "(no docstring; name not resolved)"),
                "purpose_source": (info or {}).get("purpose_source", "unresolved"),
                "duration_s": round(record["duration_s"], 4),
                "peak_rss_mib": round(record["peak_rss_kib"] / 1024, 1),
                "rss_growth_kib": record.get("rss_growth_kib"),
                "outcome": record["outcome"],
                "flaky": bool(record.get("flaky")) or nodeid in KNOWN_FLAKY,
                "known_flaky": KNOWN_FLAKY.get(nodeid),
                "observed_outcomes": record.get("observed_outcomes"),
                "skip_reason": record.get("skip_reason"),
            }
        )

    by_file: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"tests": 0, "duration_s": 0.0, "peak_rss_mib": 0.0}
    )
    for row in rows:
        agg = by_file[row["file"]]
        agg["tests"] += 1
        agg["duration_s"] += row["duration_s"]
        agg["peak_rss_mib"] = max(agg["peak_rss_mib"], row["peak_rss_mib"])

    total = sum(r["duration_s"] for r in rows)
    return {
        "issue": "CHE-64",
        "generated_by": "scripts/build_test_inventory.py",
        "profile_chunks": chunks,
        "totals": {
            "tests_profiled": len(rows),
            "total_measured_seconds": round(total, 1),
            "purpose_from_docstring": sum(
                1 for r in rows if r["purpose_source"] == "docstring"
            ),
            "purpose_derived_from_name": sum(
                1 for r in rows if r["purpose_source"] == "derived-from-name"
            ),
            "flaky": sum(1 for r in rows if r["flaky"]),
            "skipped": sum(1 for r in rows if r["outcome"] == "skipped"),
        },
        "by_file": {
            k: {**v, "duration_s": round(v["duration_s"], 2)}
            for k, v in sorted(by_file.items(), key=lambda kv: -kv[1]["duration_s"])
        },
        "tests": rows,
    }


def render_markdown(data: dict[str, Any]) -> str:
    rows = data["tests"]
    total = data["totals"]["total_measured_seconds"]
    slow = sorted(rows, key=lambda r: -r["duration_s"])
    heavy = sorted(rows, key=lambda r: -r["peak_rss_mib"])

    out: list[str] = []
    w = out.append
    w("# Per-test inventory — runtime, memory, purpose, tier (CHE-64)")
    w("")
    w(
        "Generated by `scripts/build_test_inventory.py` from the profile chunks "
        "written by `scripts/pytest_resource_profile.py`. Regenerate with:"
    )
    w("")
    w("```")
    w("./run.sh python scripts/build_test_inventory.py")
    w("```")
    w("")
    t = data["totals"]
    w(f"- **{t['tests_profiled']} tests profiled**, {total:.0f} s of measured wall time")
    w(
        f"- Purpose from a docstring: {t['purpose_from_docstring']}; "
        f"derived from the test name: {t['purpose_derived_from_name']}"
    )
    w(f"- Flaky (differing outcomes across runs): **{t['flaky']}**")
    w(f"- Skipped: {t['skipped']}")
    w("")
    w("## Largest contributors to suite runtime")
    w("")
    w("| s | % of measured | peak MiB | tier | test |")
    w("|---:|---:|---:|---|---|")
    for r in slow[:30]:
        w(
            f"| {r['duration_s']:.1f} | {100 * r['duration_s'] / total:.1f}% | "
            f"{r['peak_rss_mib']:.0f} | {r['current_tier']} | `{r['nodeid']}` |"
        )
    cum = sum(r["duration_s"] for r in slow[:10])
    w("")
    w(
        f"The 10 slowest tests are **{cum:.0f} s = {100 * cum / total:.0f}%** of all "
        f"measured wall time, out of {t['tests_profiled']} tests."
    )
    w("")
    w("## Most memory-heavy tests (peak RSS)")
    w("")
    w("| peak MiB | s | tier | test |")
    w("|---:|---:|---|---|")
    for r in heavy[:15]:
        w(
            f"| {r['peak_rss_mib']:.0f} | {r['duration_s']:.1f} | "
            f"{r['current_tier']} | `{r['nodeid']}` |"
        )
    w("")
    w("## Cost by file")
    w("")
    w("| s | tests | peak MiB | file |")
    w("|---:|---:|---:|---|")
    for name, agg in data["by_file"].items():
        w(f"| {agg['duration_s']:.1f} | {agg['tests']} | {agg['peak_rss_mib']:.0f} | `{name}` |")
    w("")
    w("## Full inventory")
    w("")
    w("Sorted by cost. `purpose` marked *(name)* was derived from the test name, ")
    w("not written by the author.")
    w("")
    w("| s | peak MiB | tier | status | test | purpose |")
    w("|---:|---:|---|---|---|---|")
    for r in slow:
        purpose = r["purpose"].replace("|", "\\|")
        if r["purpose_source"] != "docstring":
            purpose += " *(name)*"
        status = r["outcome"] + (" **FLAKY**" if r["flaky"] else "")
        w(
            f"| {r['duration_s']:.2f} | {r['peak_rss_mib']:.0f} | {r['current_tier']} | "
            f"{status} | `{r['test']}`<br/><sub>{r['file']}</sub> | {purpose} |"
        )
    w("")
    return "\n".join(out)


def main() -> None:
    data = build()
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(data, indent=2) + "\n")
    OUT_MD.write_text(render_markdown(data))
    t = data["totals"]
    print(f"wrote {OUT_JSON.relative_to(ROOT)}")
    print(f"wrote {OUT_MD.relative_to(ROOT)}")
    print(
        f"{t['tests_profiled']} tests, {t['total_measured_seconds']} s measured, "
        f"{t['flaky']} flaky, {t['skipped']} skipped"
    )
    for chunk in data["profile_chunks"]:
        if chunk["complete"] is False:
            print(f"  PARTIAL chunk (outer timeout): {chunk['file']}")
        if chunk["swap_breach"]:
            print(f"  SWAP BREACH recorded in {chunk['file']}")


if __name__ == "__main__":
    main()
