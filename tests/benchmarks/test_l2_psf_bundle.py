"""CHE-39 (M3.10) — the L2-PSF-01 bundle must be internally honest.

Mirrors ``test_l2_coupler_bundle.py`` (CHE-29): these tests do not re-derive
the M3 physics (CHE-38/CHE-47 own that), they check the properties that make
a *bundle* trustworthy -- that its fingerprint is reproducible and covers
physics rather than machine load, that its corruption evidence was measured
rather than asserted, that its negative controls actually fired, and that it
does not claim the physical-correctness gate is met when CHE-47 measured that
it is not.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = ROOT / "benchmarks/level2/L2-PSF-01"
BUNDLE = ROOT / "outputs/M3/L2-PSF-01"

pytestmark = pytest.mark.skipif(
    not (BUNDLE / "result.json").is_file(),
    reason=(
        "no L2-PSF-01 bundle present; run "
        "./run.sh python benchmarks/level2/L2-PSF-01/run_benchmark.py "
        "--output-dir outputs/M3/L2-PSF-01"
    ),
)


@pytest.fixture(scope="module")
def result() -> dict:
    return json.loads((BUNDLE / "result.json").read_text())


@pytest.fixture(scope="module")
def provenance_record() -> dict:
    return json.loads((BUNDLE / "provenance.json").read_text())


def test_bundle_declares_the_frozen_protocol(result: dict) -> None:
    assert result["benchmark_id"] == "L2-PSF-01"
    assert result["protocol_id"] == "M3-SLICE-CPU-V1"
    assert result["status"] == "complete"
    assert result["graph"] == ["M_RAY_OPTILAND", "C_RAY_TO_WAVE", "M_WAVE_CHROMATIX"]
    assert result["terminal_measurement"] == "psf"


def test_the_manifest_and_the_bundle_agree() -> None:
    manifest = yaml.safe_load((ROOT / "benchmarks/manifest.yaml").read_text())
    tasks = {task["id"]: task for task in manifest["levels"][2]["tasks"]}
    psf = tasks["L2-PSF-01"]
    assert psf["implemented"] is True
    assert (ROOT / psf["entry_point"]) == BENCHMARK_DIR / "run_benchmark.py"


def test_the_gate_outcome_is_reported_honestly_not_hidden(result: dict) -> None:
    """CHE-47 measured the frozen gate is NOT met on the real traced system.

    A bundle that silently passed, or that widened the gate to pass, would be
    exactly the failure mode AGENTS.md forbids: a runnable script standing in
    for a verified scientific result.
    """
    accuracy = result["accuracy"]
    assert accuracy["gate"] == pytest.approx(1.0e-3)
    assert accuracy["no_gate_was_widened"] is True
    verdict = accuracy["verdict"]
    assert verdict["PHYSICALLY_CORRECT"] is False
    assert verdict["physical_correctness"] == "characterized_gate_not_met"
    assert verdict["DISCRETIZATION_CONVERGED"] is True
    assert verdict["HANDOFF_WITHIN_DECLARED_VALIDITY_REGION"] is True


def test_two_of_three_negative_controls_fired(result: dict) -> None:
    """opl_sign_flip and exit_pupil_hard_support_reconstruction must fire.

    quadrature_weight_regression is decided against O1 (analytic Airy), the
    sole gate-deciding oracle (never O2, our own custom ASM/RS propagator --
    see benchmarks/level2/L2-PSF-01/tolerances.yaml). Against O1, the
    production quadrature weight measurably makes the sensor residual worse
    on the real aberrated system, so this control honestly reads
    detected=False. That is the correct, non-circular answer, not a bug --
    see test_m3_quadrature_weight.py::
    test_the_weighted_result_does_not_improve_on_uniform_weights_vs_o1.
    """
    controls = result["accuracy"]["negative_controls"]
    assert set(controls) == {
        "opl_sign_flip",
        "quadrature_weight_regression",
        "exit_pupil_hard_support_reconstruction",
    }
    assert controls["opl_sign_flip"]["detected"] is True
    assert controls["exit_pupil_hard_support_reconstruction"]["detected"] is True
    assert controls["quadrature_weight_regression"]["detected"] is False
    assert result["accuracy"]["negative_controls_pass"] is False
    assert result["accuracy"]["pass"] is False


def test_the_full_graph_demonstration_actually_exercised_chromatix(result: dict) -> None:
    """Both source probes' primary configuration has zero post-handoff
    propagation, so this is the one place M_WAVE_CHROMATIX does real work."""
    demo = result["accuracy"]["full_graph_demonstration"]
    assert demo["status"] == "succeeded"
    assert demo["propagation_m"] > 0.0
    assert demo["engine"].startswith("chromatix_adapter")
    assert demo["relative_l2_vs_o1_analytic_airy_gate_disc"] > 0.0
    assert demo["relative_l2_vs_o2_asm_gate_disc_diagnostic_only"] > 0.0


def test_differentiability_is_characterized_not_promoted(result: dict) -> None:
    differentiability = result["differentiability"]
    assert differentiability["derivative_verified"] is False
    assert differentiability["derivative_mode"] == "finite_difference"
    assert "NOT promoted" in differentiability["claim"]


def test_corruption_evidence_was_measured_not_asserted(result: dict) -> None:
    corruption = result["reproducibility"]["corrupted_fixture_rejection"]
    assert corruption["detected"] is True
    assert corruption["evaluator_returncode"] == 2
    assert corruption["clean_bundle_returncode"] == 0
    assert result["reproducibility"]["pass"] is True


def test_the_evaluator_agrees_live_on_a_clean_and_a_corrupted_copy(tmp_path: Path) -> None:
    import shutil

    clean = subprocess.run(
        [sys.executable, str(BENCHMARK_DIR / "evaluate.py"), str(BUNDLE)],
        capture_output=True, text=True, check=False,
    )
    assert clean.returncode == 0, clean.stdout

    mutated = tmp_path / "bundle"
    shutil.copytree(BUNDLE, mutated)
    with (mutated / "arrays.npz").open("ab") as handle:
        handle.write(b"corrupted")
    corrupted = subprocess.run(
        [sys.executable, str(BENCHMARK_DIR / "evaluate.py"), str(mutated)],
        capture_output=True, text=True, check=False,
    )
    assert corrupted.returncode == 2
    assert "hash_mismatch" in corrupted.stdout


def test_the_scientific_fingerprint_excludes_volatile_keys(
    result: dict, provenance_record: dict
) -> None:
    from multiscale_optics_agent.evaluation.m1_bundle import VOLATILE_KEYS

    reproducibility = result["reproducibility"]
    assert len(reproducibility["scientific_fingerprint"]) == 64
    stripped = reproducibility["scientific_projection"]["volatile_keys_stripped"]
    assert set(stripped) == set(VOLATILE_KEYS)
    assert provenance_record["dirty_worktree"] in (True, False)


def test_the_registry_still_has_no_c_field_to_psf_entry() -> None:
    """CHE-36 (M3.7) retired it as an architectural primitive, not a gap to
    reconcile. CHE-39's claim audit re-asserts the absence, it does not
    reintroduce a coupler for a trivial |U|^2 observable."""
    registry_text = (
        ROOT / "src/multiscale_optics_agent/registry/couplers.yaml"
    ).read_text()
    declarations = [
        line for line in registry_text.splitlines() if line.strip() == "- id: C_FIELD_TO_PSF"
    ]
    assert not declarations, declarations
