"""Prove M1 ray/wave entry points are separate and load no coupler implementation."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]

BRANCH_SOURCES = {
    "ray": [
        "benchmarks/level1/L1-RAY-01/run_all.py",
        "benchmarks/level1/L1-RAY-01/run_benchmark.py",
        "benchmarks/level1/L1-RAY-01/run_scaling.py",
        "knowledge/solvers/optiland/probes/standalone_baseline.py",
        "src/multiscale_optics_agent/adapters/optiland_adapter.py",
        "src/multiscale_optics_agent/adapters/optiland_benchmark_adapter.py",
    ],
    "wave": [
        "benchmarks/level1/L1-WAVE-01/run_all.py",
        "benchmarks/level1/L1-WAVE-01/evaluate.py",
        "benchmarks/level1/L1-WAVE-01/run_scaling.py",
        "knowledge/solvers/chromatix/probes/standalone_baseline.py",
        "src/multiscale_optics_agent/adapters/chromatix_adapter.py",
        "src/multiscale_optics_agent/adapters/chromatix_benchmark_adapter.py",
        "src/multiscale_optics_agent/adapters/chromatix_scaling_adapter.py",
    ],
}


def _imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    return sorted(set(imported))


def _source_check(branch: str) -> dict[str, Any]:
    forbidden_engine = "chromatix" if branch == "ray" else "optiland"
    details = {}
    violations = []
    coupler_ids = []
    for relative in BRANCH_SOURCES[branch]:
        path = ROOT / relative
        imports = _imports(path)
        source = path.read_text()
        bad_imports = [
            name
            for name in imports
            if name == forbidden_engine
            or name.startswith(f"{forbidden_engine}.")
            or name.startswith("multiscale_optics_agent.couplers")
        ]
        ids = [
            identifier
            for identifier in ("C_RAY_TO_WAVE", "C_WAVE_TO_RAY")
            if identifier in source
        ]
        details[relative] = {
            "imports": imports,
            "forbidden_imports": bad_imports,
            "coupler_ids": ids,
        }
        violations.extend(f"{relative}: {name}" for name in bad_imports)
        coupler_ids.extend(f"{relative}: {identifier}" for identifier in ids)
    return {
        "files": details,
        "forbidden_import_violations": violations,
        "coupler_id_references": coupler_ids,
        "pass": not violations and not coupler_ids,
    }


def _bundle_check(path: Path) -> dict[str, Any]:
    result = json.loads((path / "result.json").read_text())
    provenance = json.loads((path / "provenance.json").read_text())
    nested = [
        json.loads((path / "scaling" / "provenance.json").read_text()),
        json.loads((path / "scaling" / "accuracy_gate" / "provenance.json").read_text()),
    ]
    forbidden = provenance.get("forbidden_modules_loaded", []) + [
        name for item in nested for name in item.get("forbidden_modules_loaded", [])
    ]
    return {
        "status": result.get("status"),
        "scientific_fingerprint": result.get("reproducibility", {}).get(
            "scientific_fingerprint"
        ),
        "forbidden_modules_loaded": forbidden,
        "corruption_detected": result.get("reproducibility", {})
        .get("corrupted_fixture_rejection", {})
        .get("detected"),
        "pass": result.get("status") == "complete"
        and result.get("reproducibility", {}).get("pass") is True
        and not forbidden,
    }


def _validated_device_claims() -> dict[str, list[str]]:
    """Devices each component has actually executed on, from the capability model.

    Imported rather than hard-coded so this audit cannot disagree with the thing
    the tests validate. Importing ``core.capabilities`` is not an engine import
    and does not touch the independence property this script exists to check --
    that property is about the ray and wave *adapter sources* not importing each
    other's engine or a coupler, and it is checked separately by ``_source_check``.
    """
    from multiscale_optics_agent.core.capabilities import COMPONENT_CAPABILITIES

    return {
        name: sorted(kind.to_spec_device_name() for kind in capability.devices)
        for name, capability in COMPONENT_CAPABILITIES.items()
    }


def _validated_dtype_claims() -> dict[str, list[str]]:
    """Input dtypes each component accepts natively -- not merely ingests.

    ``lossy_input_dtypes`` is excluded on purpose: Chromatix swallows a
    complex128 array and returns complex64, and counting that as an accepted
    dtype would turn a measured precision loss into a support claim.
    """
    from multiscale_optics_agent.core.capabilities import COMPONENT_CAPABILITIES

    return {
        name: sorted(str(dtype) for dtype in capability.accepted_input_dtypes)
        for name, capability in COMPONENT_CAPABILITIES.items()
    }


def _claim_audit() -> dict[str, Any]:
    registry = yaml.safe_load(
        (ROOT / "src/multiscale_optics_agent/registry/models.yaml").read_text()
    )
    models = {item["id"]: item for item in registry["models"]}
    couplers = yaml.safe_load(
        (ROOT / "src/multiscale_optics_agent/registry/couplers.yaml").read_text()
    )["couplers"]
    by_id = {item["id"]: item for item in couplers}
    ray = models["M_RAY_OPTILAND"]
    wave = models["M_WAVE_CHROMATIX"]
    # CHE-61 (PB4b) rewrote three of these checks. What they exist to catch is
    # unchanged -- a registry that claims more than has been executed -- but they
    # were written as literal locks on the M1 snapshot (`devices == ["cpu"]`,
    # `dtypes == ["float64"]`), and PB4b validated GPU and float32 execution for
    # the ray model and GPU for the wave model. A literal lock cannot tell
    # "someone widened this without evidence" from "someone widened this and
    # produced the evidence", so the check now compares the registry against the
    # executable capability model instead of against a frozen list. That is the
    # same assertion, sourced from measurement rather than from a snapshot, and
    # the capability model is in turn gated by
    # tests/test_precision_execution_matrix.py (host) and
    # tests/test_precision_gpu_pipeline.py (device).
    #
    # `wave_complex64_only` is deliberately still a literal, because it is a
    # statement about the package and not about this project: Chromatix's
    # ScalarField.__init__ casts unconditionally to complex64, so complex128 will
    # never appear here for a reason no amount of project work can change.
    validated_devices = _validated_device_claims()
    validated_dtypes = _validated_dtype_claims()
    checks = {
        "ray_device_claim_is_validated": sorted(ray["devices"])
        == validated_devices["M_RAY_OPTILAND"],
        "ray_dtype_claim_is_validated": sorted(ray["dtypes"])
        == validated_dtypes["M_RAY_OPTILAND"],
        "ray_gradient_unverified": ray["derivative"]["verified"] is False,
        "wave_device_claim_is_validated": sorted(wave["devices"])
        == validated_devices["M_WAVE_CHROMATIX"],
        "wave_complex64_only": wave["dtypes"] == ["complex64"],
        "wave_scalar_only": wave["approximation"] == "scalar_wave",
        "wave_gradient_unverified": wave["derivative"]["verified"] is False,
        "ray_to_wave_experimental": by_id["C_RAY_TO_WAVE"]["maturity"] == "experimental",
        "ray_to_wave_gradient_unverified": by_id["C_RAY_TO_WAVE"]["derivative"][
            "verified"
        ]
        is False,
        # CHE-23 registered C_WAVE_TO_RAY, so the original check --
        # "C_WAVE_TO_RAY" not in by_id -- can no longer hold. It is replaced
        # rather than deleted, because the thing it protected still matters:
        # the registry must not claim a wave->ray capability M2 has not
        # established. Registration is not a capability claim; maturity,
        # verified-gradient, and lossiness are.
        "wave_to_ray_registered_experimental": (
            "C_WAVE_TO_RAY" in by_id and by_id["C_WAVE_TO_RAY"]["maturity"] == "experimental"
        ),
        "wave_to_ray_gradient_unverified": (
            by_id["C_WAVE_TO_RAY"]["derivative"]["verified"] is False
            if "C_WAVE_TO_RAY" in by_id
            else True
        ),
        # SI S7.2's estimator differentiates a fixed-direction surrogate of the
        # objective, not the objective itself. Declaring native_autodiff here
        # would overstate what the paper's method computes.
        "wave_to_ray_gradient_mode_is_surrogate": (
            by_id["C_WAVE_TO_RAY"]["derivative"]["mode"] == "surrogate"
            if "C_WAVE_TO_RAY" in by_id
            else True
        ),
        "wave_to_ray_declared_lossy": (
            by_id["C_WAVE_TO_RAY"]["lossy"] is True if "C_WAVE_TO_RAY" in by_id else True
        ),
    }
    return {"checks": checks, "pass": all(checks.values())}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ray-bundle", type=Path, default=Path("outputs/M1/ray"))
    parser.add_argument("--wave-bundle", type=Path, default=Path("outputs/M1/wave"))
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/M1/independence_report.json")
    )
    args = parser.parse_args()
    checks = {
        "ray_source": _source_check("ray"),
        "wave_source": _source_check("wave"),
        "ray_bundle": _bundle_check(args.ray_bundle.resolve()),
        "wave_bundle": _bundle_check(args.wave_bundle.resolve()),
        "claim_audit": _claim_audit(),
    }
    report = {
        "schema_version": 1,
        "status": "passed" if all(item["pass"] for item in checks.values()) else "failed",
        "checks": checks,
        "conclusion": (
            "The M1 ray and wave commands execute in separate processes and neither imports "
            "the other engine, C_RAY_TO_WAVE, C_WAVE_TO_RAY, or a coupler implementation."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
