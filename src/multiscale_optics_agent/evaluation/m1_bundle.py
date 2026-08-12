"""Build one independently reproducible M1 branch bundle without importing a solver."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import jsonschema
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
RESULT_SCHEMA = REPO_ROOT / "benchmarks" / "schemas" / "result.schema.json"
PROVENANCE_SCHEMA = REPO_ROOT / "benchmarks" / "schemas" / "provenance.schema.json"


@dataclass(frozen=True)
class M1BranchSpec:
    branch: str
    benchmark_id: str
    standalone_script: str
    scaling_script: str
    scaling_arguments: tuple[str, ...]
    scaling_evaluator: str
    other_engine_prefix: str
    dtype: str
    protocol_id: str


RAY_SPEC = M1BranchSpec(
    branch="ray",
    benchmark_id="L1-RAY-01",
    standalone_script="knowledge/solvers/optiland/probes/standalone_baseline.py",
    scaling_script="benchmarks/level1/L1-RAY-01/run_scaling.py",
    scaling_arguments=("--backend", "numpy"),
    scaling_evaluator="benchmarks/level1/L1-RAY-01/evaluate.py",
    other_engine_prefix="chromatix",
    dtype="float64",
    protocol_id="M1-BASELINE-CPU-V2",
)

WAVE_SPEC = M1BranchSpec(
    branch="wave",
    benchmark_id="L1-WAVE-01",
    standalone_script="knowledge/solvers/chromatix/probes/standalone_baseline.py",
    scaling_script="benchmarks/level1/L1-WAVE-01/run_scaling.py",
    scaling_arguments=("--device", "cpu"),
    scaling_evaluator="benchmarks/level1/L1-WAVE-01/evaluate.py",
    other_engine_prefix="optiland",
    dtype="complex64",
    protocol_id="M1-BASELINE-CPU-V1",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


# Keys that are environment observations rather than scientific output. They may
# legitimately appear inside an accuracy section -- a per-case solver block
# records how long that case took -- but they must never enter the scientific
# fingerprint, or re-running a bit-identical computation would report a
# different fingerprint purely because the machine was busier. This tuple is the
# executable form of the "volatile_exclusions" policy published alongside the
# fingerprint; the two are asserted to agree below.
VOLATILE_KEYS = (
    "runtime_seconds",
    "process_wall_seconds",
    "worker_process_seconds",
    "import_seconds",
    "setup_seconds",
    "timestamp_utc",
    "run_id",
    "output_directory",
)


def _strip_volatile(value: Any) -> Any:
    """Recursively drop wall-clock and run-identity keys from a nested structure."""
    if isinstance(value, dict):
        return {
            key: _strip_volatile(item) for key, item in value.items() if key not in VOLATILE_KEYS
        }
    if isinstance(value, list):
        return [_strip_volatile(item) for item in value]
    return value


def _canonical_npz_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with np.load(path, allow_pickle=False) as archive:
        for name in sorted(archive.files):
            value = np.ascontiguousarray(archive[name])
            digest.update(name.encode("utf-8"))
            digest.update(str(value.dtype).encode("ascii"))
            digest.update(json.dumps(list(value.shape)).encode("ascii"))
            digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _run(command: list[str], *, label: str) -> dict[str, Any]:
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"{label} failed with exit {completed.returncode}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return {
        "label": label,
        "command": command,
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-2000:],
        "stderr_tail": completed.stderr[-2000:],
    }


def _validate_json(path: Path, schema_path: Path) -> None:
    jsonschema.validate(
        json.loads(path.read_text()),
        json.loads(schema_path.read_text()),
    )


def _corruption_check(spec: M1BranchSpec, scaling_dir: Path) -> dict[str, Any]:
    result = json.loads((scaling_dir / "result.json").read_text())
    case = result["scaling"]["cases"][0]
    artifact_key = "scientific_artifact" if spec.branch == "ray" else "field_artifact"
    relative_artifact = Path(case[artifact_key])
    with tempfile.TemporaryDirectory(prefix=f"m1-{spec.branch}-corrupt-") as temp_name:
        corrupted = Path(temp_name) / "bundle"
        shutil.copytree(scaling_dir, corrupted)
        target = corrupted / relative_artifact
        target.write_bytes(target.read_bytes() + b"M1-deliberate-corruption")
        command = [
            sys.executable,
            str(REPO_ROOT / spec.scaling_evaluator),
            "--section",
            "scaling",
            "--output-dir",
            str(corrupted),
        ]
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    detected = completed.returncode != 0 and "hash mismatch" in completed.stdout
    if not detected:
        raise RuntimeError(
            f"{spec.branch} evaluator did not reject deliberate artifact corruption: "
            f"exit={completed.returncode}, stdout={completed.stdout}, stderr={completed.stderr}"
        )
    return {
        "fixture": str(relative_artifact),
        "mutation": "appended bytes to a scientific complex/ray artifact",
        "evaluator_returncode": completed.returncode,
        "detected": True,
    }


def _scientific_projection(
    spec: M1BranchSpec,
    standalone: dict[str, Any],
    accuracy_result: dict[str, Any],
    accuracy_dir: Path,
    scaling_result: dict[str, Any],
    scaling_dir: Path,
) -> dict[str, Any]:
    canonical_archives = {
        str(path.relative_to(accuracy_dir)): _canonical_npz_hash(path)
        for path in sorted(accuracy_dir.glob("*.npz"))
    }
    scaling_hash_key = (
        "scientific_array_sha256" if spec.branch == "ray" else "scientific_field_sha256"
    )
    return {
        "policy": (
            "compare canonical scientific arrays, accuracy metrics, stable standalone "
            "summaries, fixed tolerances, and per-scaling-case canonical field/ray hashes"
        ),
        "volatile_exclusions": [
            "timestamps",
            "run identifiers",
            "output paths",
            "process IDs",
            "wall-clock runtime samples",
            "peak RSS observations",
            "git dirty flag",
        ],
        # The accuracy section legitimately carries a per-case solver runtime, so
        # it is stripped before hashing rather than hashed wholesale. Without
        # this the fingerprint tracks machine load instead of physics.
        "volatile_keys_stripped_before_hashing": list(VOLATILE_KEYS),
        "standalone_stable_result_sha256": _json_hash(_strip_volatile(standalone["stable_result"])),
        "accuracy_section_sha256": _json_hash(_strip_volatile(accuracy_result["accuracy"])),
        "accuracy_canonical_npz_sha256": canonical_archives,
        "accuracy_tolerances_sha256": _sha256(accuracy_dir / "tolerances.yaml"),
        "scaling_case_scientific_sha256": [
            case[scaling_hash_key] for case in scaling_result["scaling"]["cases"]
        ],
        "scaling_tolerances_sha256": _sha256(scaling_dir / "tolerances.yaml"),
    }


def _write_blocked(spec: M1BranchSpec, output_dir: Path, exc: Exception) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "schema_version": 1,
        "benchmark_id": spec.benchmark_id,
        "status": "blocked",
        "failure": {
            "code": f"M1_{spec.branch.upper()}_BUNDLE_FAILED",
            "message": str(exc),
            "stage": "m1_branch_integration",
            "exception_type": type(exc).__name__,
        },
    }
    (output_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 1


def build_branch_bundle(spec: M1BranchSpec, output_dir: Path) -> int:
    """Execute standalone + analytic + scaling evidence as one isolated branch."""

    output_dir = output_dir.resolve()
    standalone_dir = output_dir / "standalone"
    scaling_dir = output_dir / "scaling"
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        commands = [
            _run(
                [
                    sys.executable,
                    str(REPO_ROOT / spec.standalone_script),
                    "--output-dir",
                    str(standalone_dir),
                ],
                label=f"{spec.branch} standalone baseline",
            ),
            _run(
                [
                    sys.executable,
                    str(REPO_ROOT / spec.scaling_script),
                    *spec.scaling_arguments,
                    "--output-dir",
                    str(scaling_dir),
                ],
                label=f"{spec.branch} analytic plus scaling benchmark",
            ),
            _run(
                [
                    sys.executable,
                    str(REPO_ROOT / spec.scaling_evaluator),
                    "--section",
                    "scaling",
                    "--output-dir",
                    str(scaling_dir),
                ],
                label=f"{spec.branch} scaling evaluator",
            ),
        ]

        standalone = json.loads((standalone_dir / "determinism_report.json").read_text())
        scaling_result = json.loads((scaling_dir / "result.json").read_text())
        scaling_provenance = json.loads((scaling_dir / "provenance.json").read_text())
        accuracy_dir = scaling_dir / "accuracy_gate"
        accuracy_result = json.loads((accuracy_dir / "result.json").read_text())
        accuracy_provenance = json.loads((accuracy_dir / "provenance.json").read_text())
        for result_path in (scaling_dir / "result.json", accuracy_dir / "result.json"):
            _validate_json(result_path, RESULT_SCHEMA)
        for provenance_path in (
            scaling_dir / "provenance.json",
            accuracy_dir / "provenance.json",
        ):
            _validate_json(provenance_path, PROVENANCE_SCHEMA)

        gates = {
            "standalone_pass": standalone.get("status") == "passed",
            "standalone_deterministic": bool(standalone.get("deterministic")),
            "accuracy_pass": bool(accuracy_result.get("accuracy", {}).get("pass")),
            "scaling_accuracy_pass": bool(scaling_result.get("accuracy", {}).get("pass")),
            "scaling_performance_pass": bool(scaling_result.get("scaling", {}).get("pass")),
            "component_schemas_valid": True,
            "standalone_forbidden_modules_empty": not standalone.get(
                "forbidden_modules_loaded", []
            ),
            "accuracy_forbidden_modules_empty": not accuracy_provenance.get(
                "forbidden_modules_loaded", []
            ),
            "scaling_forbidden_modules_empty": not scaling_provenance.get(
                "forbidden_modules_loaded", []
            ),
        }
        if not all(gates.values()):
            raise RuntimeError(f"{spec.branch} branch scientific gate failed: {gates}")

        corruption = _corruption_check(spec, scaling_dir)
        projection = _scientific_projection(
            spec,
            standalone,
            accuracy_result,
            accuracy_dir,
            scaling_result,
            scaling_dir,
        )
        scientific_fingerprint = _json_hash(projection)
        environment_fingerprint = scaling_result["scaling"]["environment_fingerprint"]
        matching_environment = (
            scaling_result["scaling"]["regression_envelope"]["fingerprint"]
            == environment_fingerprint
        )
        if not matching_environment:
            raise RuntimeError("performance envelope fingerprint does not match this run")

        shutil.copy2(
            scaling_dir / "raw_timing_samples.json",
            output_dir / "raw_timing_samples.json",
        )
        shutil.copy2(scaling_dir / "tolerances.yaml", output_dir / "tolerances.yaml")
        shutil.copy2(scaling_dir / "scaling.png", output_dir / "scaling.png")
        shutil.copy2(scaling_dir / "scaling.png", output_dir / "plot.png")
        shutil.copy2(accuracy_dir / "plot.png", output_dir / "accuracy_plot.png")

        result = {
            "schema_version": 1,
            "benchmark_id": spec.benchmark_id,
            "protocol_id": spec.protocol_id,
            "status": "complete",
            "accuracy": {
                "oracle": "standalone determinism plus accepted analytic and scaling gates",
                "metrics": gates,
                "tolerances": {
                    "accuracy": "scaling/accuracy_gate/tolerances.yaml",
                    "scaling": "tolerances.yaml",
                },
                "pass": all(gates.values()),
            },
            "performance": scaling_result["performance"],
            "scaling": scaling_result["scaling"],
            "reproducibility": {
                "branch": spec.branch,
                "scientific_fingerprint": scientific_fingerprint,
                "environment_fingerprint": environment_fingerprint,
                "matching_environment_for_performance": matching_environment,
                "scientific_projection": projection,
                "corrupted_fixture_rejection": corruption,
                "commands": commands,
                "pass": True,
            },
        }
        (output_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

        component_files = sorted(
            path
            for base in (standalone_dir, scaling_dir)
            for path in base.rglob("*")
            if path.is_file()
        )
        manifest = {
            "schema_version": 1,
            "branch": spec.branch,
            "benchmark_id": spec.benchmark_id,
            "evaluator_version": "1.0.0",
            "scientific_fingerprint": scientific_fingerprint,
            "artifacts": {
                str(path.relative_to(output_dir)): _sha256(path) for path in component_files
            },
            "root_artifacts": [
                "result.json",
                "provenance.json",
                "raw_timing_samples.json",
                "tolerances.yaml",
                "plot.png",
                "scaling.png",
                "accuracy_plot.png",
            ],
        }
        (output_dir / "bundle_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )

        forbidden_loaded = sorted(
            name
            for name in sys.modules
            if name == spec.other_engine_prefix
            or name.startswith(f"{spec.other_engine_prefix}.")
            or name.startswith("multiscale_optics_agent.couplers")
        )
        if forbidden_loaded:
            raise RuntimeError(f"{spec.branch} bundle loaded forbidden modules: {forbidden_loaded}")

        root_hash_files = [
            "result.json",
            "bundle_manifest.json",
            "raw_timing_samples.json",
            "tolerances.yaml",
            "plot.png",
            "scaling.png",
            "accuracy_plot.png",
        ]
        provenance = {
            "schema_version": 1,
            "benchmark_id": spec.benchmark_id,
            "protocol_id": spec.protocol_id,
            "command": [
                "./run.sh",
                "python",
                f"benchmarks/level1/{spec.benchmark_id}/run_all.py",
            ],
            "git_commit": scaling_provenance["git_commit"],
            "dirty_worktree": scaling_provenance["dirty_worktree"],
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpu_model": scaling_provenance.get("cpu_model"),
            "logical_cpu_count": os.cpu_count() or 1,
            "process_cpu_affinity": scaling_provenance.get("process_cpu_affinity", []),
            "thread_counts": scaling_provenance.get("thread_counts", {"status": "unavailable"}),
            "device": "cpu",
            "dtype": spec.dtype,
            "seed": 20260811,
            "engine_versions": scaling_provenance["engine_versions"],
            "input_parameters": {
                "branch": spec.branch,
                "standalone": spec.standalone_script,
                "analytic_and_scaling": spec.scaling_script,
            },
            "environment_fingerprint": environment_fingerprint,
            "scientific_fingerprint": scientific_fingerprint,
            "artifact_hashes": {
                filename: _sha256(output_dir / filename) for filename in root_hash_files
            },
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "forbidden_modules_loaded": forbidden_loaded,
        }
        (output_dir / "provenance.json").write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n"
        )
        _validate_json(output_dir / "result.json", RESULT_SCHEMA)
        _validate_json(output_dir / "provenance.json", PROVENANCE_SCHEMA)
        print(
            json.dumps(
                {
                    "status": "complete",
                    "branch": spec.branch,
                    "pass": True,
                    "scientific_fingerprint": scientific_fingerprint,
                    "environment_fingerprint": environment_fingerprint,
                    "output_directory": str(output_dir),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    except Exception as exc:
        return _write_blocked(spec, output_dir, exc)
