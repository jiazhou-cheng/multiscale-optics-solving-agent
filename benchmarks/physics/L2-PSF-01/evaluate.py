#!/usr/bin/env python3
"""L2-PSF-01 evaluator — re-check a bundle against its own recorded hashes.

Exit codes:
    0  bundle is internally consistent
    2  a hashed artifact does not match the hash recorded in provenance.json
    3  the bundle is malformed or incomplete

Mirrors L2-COUPLER-01's evaluator (CHE-29): a benchmark that cannot notice a
corrupted artifact cannot certify one, so this is run by the benchmark itself
against a deliberately mutated copy, and the observed exit code is what gets
recorded — never a hardcoded expectation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REQUIRED = (
    "result.json",
    "provenance.json",
    "arrays.npz",
    "tolerances.yaml",
    "README.md",
    "convergence.json",
    "plot.png",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle", type=Path)
    args = parser.parse_args()
    bundle = args.bundle

    for name in REQUIRED:
        if not (bundle / name).is_file():
            print(json.dumps({"status": "malformed", "missing": name}))
            return 3

    try:
        provenance = json.loads((bundle / "provenance.json").read_text())
        recorded = provenance["artifact_hashes"]
    except (OSError, ValueError, KeyError) as exc:
        print(json.dumps({"status": "malformed", "error": str(exc)}))
        return 3

    mismatches = []
    for name, expected in recorded.items():
        path = bundle / name
        if not path.is_file():
            mismatches.append({"artifact": name, "reason": "missing"})
            continue
        observed = hashlib.sha256(path.read_bytes()).hexdigest()
        if observed != expected:
            mismatches.append({"artifact": name, "expected": expected, "observed": observed})

    if mismatches:
        print(json.dumps({"status": "hash_mismatch", "mismatches": mismatches}, indent=2))
        return 2

    print(json.dumps({"status": "consistent", "artifacts": len(recorded)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
