"""Machine-readable evaluator for the CHE-15 L1-WAVE-01 scaling section."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate_scaling_bundle(output_dir: Path) -> list[str]:
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
        return [f"result status is {result.get('status')!r}, expected 'complete'"]
    scaling = result.get("scaling", {})
    if scaling.get("grid_sequence") != [64, 128, 256]:
        errors.append("canonical grid sequence must be [64, 128, 256]")
    cases = scaling.get("cases", [])
    if len(cases) != 3:
        errors.append(f"expected three scaling cases, got {len(cases)}")
    padded_cells: list[int] = []
    for case in cases:
        grid = case.get("grid", {})
        padding = case.get("padding", {})
        timing = case.get("timing", {})
        padded_cells.append(int(padding.get("propagated_cells", 0)))
        if padding.get("policy") != "auto_transfer" or padding.get("pad_width", 0) <= 0:
            errors.append(f"automatic padding missing for grid {grid.get('input_n')}")
        if timing.get("steady_measured_repeats", 0) < 5:
            errors.append(f"fewer than five steady repeats for grid {grid.get('input_n')}")
        if timing.get("compile_plus_execute_seconds", -1) < 0:
            errors.append(f"missing first-call timing for grid {grid.get('input_n')}")
        if not case.get("determinism", {}).get("pass"):
            errors.append(f"determinism failed for grid {grid.get('input_n')}")
        if not case.get("gaussian_accuracy", {}).get("pass"):
            errors.append(f"Gaussian accuracy failed for grid {grid.get('input_n')}")
        artifact = output_dir / case.get("field_artifact", "")
        if not artifact.is_file():
            errors.append(f"missing complex field artifact for grid {grid.get('input_n')}")
        elif _sha256(artifact) != case.get("field_artifact_sha256"):
            errors.append(f"complex field artifact hash mismatch for grid {grid.get('input_n')}")
    if padded_cells != sorted(padded_cells):
        errors.append("padded propagated-cell counts must increase with grid sequence")
    if not result.get("accuracy", {}).get("pass"):
        errors.append("current CHE-18 accuracy bundle gate failed")
    if not scaling.get("smallest_case_compile_inclusive_under_10_seconds"):
        errors.append("smallest compile-inclusive case exceeded 10 seconds")
    if not scaling.get("pass"):
        errors.append("scaling aggregate gate failed")
    if provenance.get("forbidden_modules_loaded"):
        errors.append(f"forbidden modules loaded: {provenance['forbidden_modules_loaded']}")
    for filename, expected_hash in provenance.get("artifact_hashes", {}).items():
        path = output_dir / filename
        if not path.is_file() or _sha256(path) != expected_hash:
            errors.append(f"provenance artifact hash mismatch: {filename}")
    return errors
