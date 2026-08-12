from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("optiland")
pytest.importorskip("chromatix")
pytestmark = [pytest.mark.integration, pytest.mark.jax]

ROOT = Path(__file__).resolve().parents[2]


def _run_branch(script: str, output: Path) -> dict:
    completed = subprocess.run(
        [sys.executable, script, "--output-dir", str(output)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return json.loads((output / "result.json").read_text())


@pytest.fixture(scope="module")
def reproduced_branches(tmp_path_factory: pytest.TempPathFactory):
    root = tmp_path_factory.mktemp("m1-reproduction")
    ray_a = _run_branch("benchmarks/level1/L1-RAY-01/run_all.py", root / "ray_a")
    ray_b = _run_branch("benchmarks/level1/L1-RAY-01/run_all.py", root / "ray_b")
    wave_a = _run_branch("benchmarks/level1/L1-WAVE-01/run_all.py", root / "wave_a")
    wave_b = _run_branch("benchmarks/level1/L1-WAVE-01/run_all.py", root / "wave_b")
    return root, {"ray": (ray_a, ray_b), "wave": (wave_a, wave_b)}


def test_each_branch_reproduces_its_scientific_fingerprint(reproduced_branches) -> None:
    _, branches = reproduced_branches
    for first, second in branches.values():
        assert first["status"] == second["status"] == "complete"
        assert first["accuracy"]["pass"] and second["accuracy"]["pass"]
        assert (
            first["reproducibility"]["scientific_fingerprint"]
            == second["reproducibility"]["scientific_fingerprint"]
        )
        assert (
            first["reproducibility"]["environment_fingerprint"]
            == second["reproducibility"]["environment_fingerprint"]
        )
        assert (
            "timestamps" in first["reproducibility"]["scientific_projection"]["volatile_exclusions"]
        )


def test_each_branch_rejects_corruption_and_emits_complete_bundle(
    reproduced_branches,
) -> None:
    root, branches = reproduced_branches
    for branch, (result, _) in branches.items():
        assert result["reproducibility"]["corrupted_fixture_rejection"]["detected"]
        bundle = root / f"{branch}_a"
        required = {
            "result.json",
            "provenance.json",
            "bundle_manifest.json",
            "raw_timing_samples.json",
            "tolerances.yaml",
            "plot.png",
            "scaling.png",
            "accuracy_plot.png",
        }
        assert required <= {path.name for path in bundle.iterdir()}


def test_executable_independence_and_claim_audit(reproduced_branches) -> None:
    root, _ = reproduced_branches
    completed = subprocess.run(
        [
            sys.executable,
            "benchmarks/verify_m1_independence.py",
            "--ray-bundle",
            str(root / "ray_a"),
            "--wave-bundle",
            str(root / "wave_a"),
            "--output",
            str(root / "independence.json"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    report = json.loads((root / "independence.json").read_text())
    assert report["status"] == "passed"
    assert all(item["pass"] for item in report["checks"].values())
