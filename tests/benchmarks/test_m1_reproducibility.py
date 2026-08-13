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

# Published by M1 in benchmarks/M1_BASELINE_REPORT.md ("Ray branch -- L1-RAY-01")
# and cited by M2 in benchmarks/M2_COUPLER_REPORT.md. Not re-recorded here: the
# assertion below checks that this literal is still the one the report states,
# so the lock cannot drift away from the document it comes from.
L1_RAY_SCIENTIFIC_FINGERPRINT = "43dab1eedf5ca8fcd6a2674bcc6fb58020933aec8ad8618ad49583826cfc7236"


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


def test_ray_branch_still_carries_its_published_m1_fingerprint(reproduced_branches) -> None:
    """The default Optiland export path must stay where M1 left it.

    CHE-32 (M3.3) added ``config['handoff_plane']`` to the ray adapter, with
    ``exit_pupil`` as a new option and ``image_surface`` as the unchanged
    default. The test above only proves the branch agrees with *itself*, which a
    moved-but-consistent export would also satisfy. This one pins it to the
    value M1 published, so adding a plane cannot silently move the existing one.
    """
    assert (
        L1_RAY_SCIENTIFIC_FINGERPRINT in (ROOT / "benchmarks" / "M1_BASELINE_REPORT.md").read_text()
    )

    _, branches = reproduced_branches
    for result in branches["ray"]:
        assert result["reproducibility"]["scientific_fingerprint"] == L1_RAY_SCIENTIFIC_FINGERPRINT


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
