"""Validate L1-RAY-01 accuracy or CHE-16 scaling evidence bundles."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate_scaling(output_dir: Path) -> list[str]:
    errors: list[str] = []
    required = {
        "result.json",
        "provenance.json",
        "raw_timing_samples.json",
        "tolerances.yaml",
        "scaling.png",
        "README.md",
    }
    missing = sorted(name for name in required if not (output_dir / name).is_file())
    if missing:
        return [f"missing required artifacts: {missing}"]
    result = json.loads((output_dir / "result.json").read_text())
    provenance = json.loads((output_dir / "provenance.json").read_text())
    if result.get("status") != "complete":
        errors.append(f"result status is {result.get('status')!r}, expected 'complete'")
        return errors
    scaling = result.get("scaling", {})
    if scaling.get("requested_sampling_sequence") != [8, 16, 32, 64]:
        errors.append("canonical requested sampling sequence must be [8, 16, 32, 64]")
    cases = scaling.get("cases", [])
    if len(cases) != 4:
        errors.append(f"expected four scaling cases, got {len(cases)}")
    for case in cases:
        counts = case.get("counts", {})
        if counts.get("traced", 0) <= 0 or counts.get("surviving", 0) <= 0:
            errors.append(
                f"non-positive actual count for sampling {case.get('requested_sampling')}"
            )
        if counts.get("generated") != counts.get("traced"):
            errors.append(
                f"generated/traced count mismatch for sampling {case.get('requested_sampling')}"
            )
        if not case.get("determinism", {}).get("pass"):
            errors.append(f"determinism failed for sampling {case.get('requested_sampling')}")
        if case.get("timing", {}).get("measured_repeats", 0) < 5:
            errors.append(f"fewer than five repeats for sampling {case.get('requested_sampling')}")
        artifact = output_dir / case.get("scientific_artifact", "")
        if not artifact.is_file():
            errors.append(f"missing scientific artifact {artifact}")
        elif _sha256(artifact) != case.get("artifact_file_sha256"):
            errors.append(f"scientific artifact hash mismatch: {artifact}")
    if not result.get("accuracy", {}).get("pass"):
        errors.append("CHE-17 accuracy gate failed")
    if not scaling.get("smallest_case_under_10_seconds"):
        errors.append("smallest canonical case exceeded 10 seconds")
    if not scaling.get("pass"):
        errors.append("scaling aggregate gate failed")
    if provenance.get("forbidden_modules_loaded"):
        errors.append(f"forbidden modules loaded: {provenance['forbidden_modules_loaded']}")
    for filename, expected in provenance.get("artifact_hashes", {}).items():
        path = output_dir / filename
        if not path.is_file() or _sha256(path) != expected:
            errors.append(f"provenance artifact hash mismatch: {filename}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--section", choices=("scaling",), default="scaling")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/L1-RAY-01/scaling"))
    args = parser.parse_args()
    errors = evaluate_scaling(args.output_dir.resolve())
    print(json.dumps({"section": args.section, "pass": not errors, "errors": errors}, indent=2))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
