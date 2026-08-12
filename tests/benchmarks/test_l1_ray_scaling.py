from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("optiland")
pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = ROOT / "benchmarks" / "level1" / "L1-RAY-01"


@pytest.fixture(scope="module")
def ray_scaling_bundle(tmp_path_factory: pytest.TempPathFactory) -> Path:
    output = tmp_path_factory.mktemp("l1-ray-scaling")
    completed = subprocess.run(
        [
            sys.executable,
            str(BENCHMARK / "run_scaling.py"),
            "--backend",
            "numpy",
            "--output-dir",
            str(output),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return output


def test_scaling_bundle_records_actual_counts_and_separate_timings(
    ray_scaling_bundle: Path,
) -> None:
    result = json.loads((ray_scaling_bundle / "result.json").read_text())
    scaling = result["scaling"]
    assert scaling["requested_sampling_sequence"] == [8, 16, 32, 64]
    actual_counts = []
    for case in scaling["cases"]:
        counts = case["counts"]
        actual_counts.append(counts["traced"])
        assert counts["generated"] == counts["traced"]
        assert counts["surviving"] > 0
        assert counts["traced"] != counts["requested_sampling"]
        assert case["surface_count"] == 15
        assert case["timing"]["import_seconds"] >= 0
        assert case["timing"]["setup_seconds"] >= 0
        assert len(case["timing"]["trace_samples_seconds"]) == 7
        assert case["throughput"]["traced_rays_per_second"] > 0
        assert "actual traced/surviving" in case["throughput"]["denominator_note"]
    assert actual_counts == sorted(actual_counts)


def test_determinism_memory_accuracy_and_environment_gates_pass(ray_scaling_bundle: Path) -> None:
    result = json.loads((ray_scaling_bundle / "result.json").read_text())
    assert result["accuracy"]["pass"]
    assert result["scaling"]["pass"]
    assert result["scaling"]["determinism_pass"]
    assert result["scaling"]["smallest_case_under_10_seconds"]
    assert len(result["scaling"]["environment_fingerprint"]) == 64
    for case in result["scaling"]["cases"]:
        assert case["determinism"]["array_hashes_identical"]
        assert case["determinism"]["physical_summaries_identical"]
        assert case["peak_memory"]["status"] in {"measured", "unsupported"}


def test_machine_readable_artifacts_and_evaluator(ray_scaling_bundle: Path) -> None:
    required = {
        "result.json",
        "provenance.json",
        "raw_timing_samples.json",
        "tolerances.yaml",
        "scaling.png",
        "README.md",
    }
    assert required <= {path.name for path in ray_scaling_bundle.iterdir()}
    assert (ray_scaling_bundle / "scaling.png").stat().st_size > 10_000
    completed = subprocess.run(
        [
            sys.executable,
            str(BENCHMARK / "evaluate.py"),
            "--section",
            "scaling",
            "--output-dir",
            str(ray_scaling_bundle),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_evaluator_rejects_corrupted_scientific_artifact(
    ray_scaling_bundle: Path, tmp_path: Path
) -> None:
    copied = tmp_path / "corrupted"
    import shutil

    shutil.copytree(ray_scaling_bundle, copied)
    artifact = next((copied / "scientific_artifacts").glob("*.npz"))
    artifact.write_bytes(artifact.read_bytes() + b"corruption")
    spec = importlib.util.spec_from_file_location(
        "l1_ray_scaling_evaluate", BENCHMARK / "evaluate.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    errors = module.evaluate_scaling(copied)
    assert any("hash mismatch" in error for error in errors)
