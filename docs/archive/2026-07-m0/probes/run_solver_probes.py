#!/usr/bin/env python3
"""M0.2 audit probe runner: replay every knowledge/solvers/*/probes/*.py.

For each probe, run it in a subprocess, parse stdout as JSON, and compare it to
the recorded knowledge/solvers/<solver>/expected/<probe>.json. Reports one of:

    MATCH        stdout JSON equals the expected JSON exactly
    DIFF         probe ran, JSON parsed, but values differ (keys listed)
    NON_JSON     probe ran but stdout is not a single JSON document
    NO_EXPECTED  probe ran but no expected/<probe>.json exists
    FAIL         probe exited non-zero (stderr tail recorded)
    TIMEOUT      probe exceeded the per-probe timeout

Read-only with respect to the repository: writes nothing to disk.

Usage (container only):
    ./run.sh python docs/audit/probes/run_solver_probes.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path("/workspace") if Path("/workspace").is_dir() else Path.cwd()
TIMEOUT_S = 900


def diff_keys(expected, actual, prefix=""):
    """Return a list of 'path: expected -> actual' strings for differing leaves."""
    out = []
    if isinstance(expected, dict) and isinstance(actual, dict):
        for k in sorted(set(expected) | set(actual)):
            p = f"{prefix}.{k}" if prefix else k
            if k not in expected:
                out.append(f"{p}: <missing in expected> -> {actual[k]!r}")
            elif k not in actual:
                out.append(f"{p}: {expected[k]!r} -> <missing in actual>")
            else:
                out.extend(diff_keys(expected[k], actual[k], p))
    elif isinstance(expected, list) and isinstance(actual, list):
        if expected != actual:
            out.append(f"{prefix}: list differs (len {len(expected)} -> {len(actual)})")
    elif expected != actual:
        out.append(f"{prefix}: {expected!r} -> {actual!r}")
    return out


def parse_json_payload(stdout: str):
    """Parse the JSON document in ``stdout``, tolerating a non-JSON preamble.

    Some solvers (jax-fem) print an ASCII banner on import before the probe's
    own output, so a strict json.loads of the whole stream fails even though
    the probe emitted a well-formed report.
    """
    try:
        return json.loads(stdout)
    except json.JSONDecodeError:
        pass
    start = stdout.find("{")
    while start != -1:
        try:
            return json.loads(stdout[start:])
        except json.JSONDecodeError:
            start = stdout.find("{", start + 1)
    return None


def main() -> None:
    results = []
    probes = sorted((ROOT / "knowledge" / "solvers").glob("*/probes/*.py"))
    for probe in probes:
        solver = probe.parents[1].name
        name = probe.stem
        expected_path = probe.parents[1] / "expected" / f"{name}.json"
        rel = str(probe.relative_to(ROOT))
        print(f"--- running {rel}", flush=True)
        try:
            proc = subprocess.run(
                [sys.executable, str(probe)],
                capture_output=True,
                text=True,
                timeout=TIMEOUT_S,
                cwd=str(ROOT),
            )
        except subprocess.TimeoutExpired:
            results.append(
                {
                    "solver": solver,
                    "probe": name,
                    "path": rel,
                    "status": "TIMEOUT",
                    "detail": f"exceeded {TIMEOUT_S}s",
                }
            )
            continue

        if proc.returncode != 0:
            tail = (proc.stderr or "").strip().splitlines()[-3:]
            results.append(
                {
                    "solver": solver,
                    "probe": name,
                    "path": rel,
                    "status": "FAIL",
                    "returncode": proc.returncode,
                    "detail": " | ".join(tail),
                }
            )
            continue

        actual = parse_json_payload(proc.stdout)
        if actual is None:
            results.append(
                {
                    "solver": solver,
                    "probe": name,
                    "path": rel,
                    "status": "NON_JSON",
                    "detail": (proc.stdout or "").strip()[:200],
                }
            )
            continue

        if not expected_path.is_file():
            results.append(
                {
                    "solver": solver,
                    "probe": name,
                    "path": rel,
                    "status": "NO_EXPECTED",
                    "detail": f"missing {expected_path.relative_to(ROOT)}",
                }
            )
            continue

        expected = json.loads(expected_path.read_text(encoding="utf-8"))
        diffs = diff_keys(expected, actual)
        results.append(
            {
                "solver": solver,
                "probe": name,
                "path": rel,
                "status": "MATCH" if not diffs else "DIFF",
                "n_diffs": len(diffs),
                "detail": "; ".join(diffs[:8]),
            }
        )

    print()
    print("### PROBE STATUS TABLE")
    print("| solver | probe | status | detail |")
    print("| --- | --- | --- | --- |")
    for r in results:
        detail = (r.get("detail") or "").replace("|", "\\|")[:220] or "—"
        print(f"| {r['solver']} | `{r['probe']}` | {r['status']} | {detail} |")

    print()
    print("### JSON")
    print(json.dumps(results, indent=2))

    print()
    print("### SUMMARY")
    counts: dict[str, int] = {}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    print(json.dumps({"probes": len(results), "by_status": counts}, indent=2))


if __name__ == "__main__":
    main()
