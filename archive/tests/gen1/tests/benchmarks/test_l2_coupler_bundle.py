"""CHE-29 — the L2-COUPLER-01 bundle must be internally honest.

These tests do not re-derive the physics; the CHE-24..CHE-28 suites do that.
They check the properties that make a *bundle* trustworthy: that its gates
actually gated, that its fingerprint covers physics and not machine load, that
its corruption evidence was measured rather than asserted, and that nothing in
it promotes a claim the milestone did not establish.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = ROOT / "benchmarks/level2/L2-COUPLER-01"
BUNDLE = ROOT / "outputs/M2/coupler"

pytestmark = [
    pytest.mark.skipif(
        not (BUNDLE / "result.json").is_file(),
        reason=(
            "no L2-COUPLER-01 bundle present; run "
            "./run.sh python benchmarks/level2/L2-COUPLER-01/run_benchmark.py "
            "--output-dir outputs/M2/coupler"
        ),
    ),
    pytest.mark.coupler,
    pytest.mark.benchmark,
]


@pytest.fixture(scope="module")
def result() -> dict:
    return json.loads((BUNDLE / "result.json").read_text())


@pytest.fixture(scope="module")
def provenance_record() -> dict:
    return json.loads((BUNDLE / "provenance.json").read_text())


def test_bundle_declares_the_frozen_protocol(result: dict) -> None:
    assert result["benchmark_id"] == "L2-COUPLER-01"
    assert result["protocol_id"] == "M2-COUPLER-CPU-V1"
    assert result["status"] == "complete"


def test_both_json_files_validate_against_the_repository_schemas(result, provenance_record) -> None:
    jsonschema = pytest.importorskip("jsonschema")

    for document, schema_name in (
        (result, "result.schema.json"),
        (provenance_record, "provenance.schema.json"),
    ):
        schema = json.loads((ROOT / "benchmarks/schemas" / schema_name).read_text())
        jsonschema.validate(document, schema)


def test_every_required_artifact_is_present_and_hashed(provenance_record) -> None:
    required = {
        "result.json",
        "provenance.json",
        "arrays.npz",
        "plot.png",
        "tolerances.yaml",
        "README.md",
        "convergence.json",
        "ensemble_statistics.json",
    }
    for name in required:
        assert (BUNDLE / name).is_file(), name

    hashed = set(provenance_record["artifact_hashes"])
    # provenance.json cannot contain its own hash; everything else must be covered.
    assert hashed == required - {"provenance.json"}


def test_accuracy_gates_were_evaluated_and_all_passed(result: dict) -> None:
    accuracy = result["accuracy"]
    assert accuracy["gates"], "a section with no gates cannot have gated anything"
    for gate in accuracy["gates"]:
        assert accuracy["metrics"][gate] is True, gate
    assert accuracy["pass"] is True


def test_performance_is_present_only_because_accuracy_passed(result: dict) -> None:
    """The protocol's ordering rule. A fast wrong answer is not a result, so a
    bundle that reports timing must also report passing gates."""
    assert result["accuracy"]["pass"] is True
    assert result["stochastic"]["pass"] is True
    assert result["performance"]["measured_repeats"] >= 5
    assert result["performance"]["warmup_runs"] >= 1


def test_all_five_negative_controls_were_detected(result: dict) -> None:
    metrics = result["accuracy"]["metrics"]
    controls = metrics["negative_controls"]

    assert set(controls) == {
        "phase_sign",
        "oblique_ramp",
        "axis_transpose",
        "importance_weight",
        "launch_phase",
    }
    assert all(controls.values()), controls
    assert metrics["negative_controls_detected"] == metrics["negative_controls_total"] == 5


def test_the_round_trip_can_actually_fail(result: dict) -> None:
    """The single most important number in the accuracy section. If a
    deliberately mismatched phase-sign pairing did not break the round trip,
    the exactness result would prove nothing."""
    metrics = result["accuracy"]["metrics"]
    tolerances = result["accuracy"]["tolerances"]

    assert metrics["round_trip_relative_rms"] <= tolerances["round_trip_relative_rms"]
    assert (
        metrics["round_trip_mismatched_phase_relative_rms"]
        >= tolerances["mismatched_pairing_min_relative_rms"]
    )


def test_the_curvature_bound_bounds_every_measured_case(result: dict) -> None:
    cases = result["accuracy"]["metrics"]["curvature_cases"]
    assert len(cases) == 12
    for case in cases:
        assert case["holds"] is True, case
        assert case["measured_rad"] <= case["bound_rad"], case


def test_stochastic_tolerances_are_derived_rather_than_chosen(result: dict) -> None:
    stochastic = result["stochastic"]

    # The exactness limit's tolerance comes from dtype round-off.
    assert stochastic["exactness_limit"]["tolerance_basis"] == "dtype_roundoff_derived"
    assert stochastic["exactness_limit"]["enumerated_all_propagating_bins"] is True

    # The unbiasedness tolerance IS the measured standard error.
    unbiasedness = stochastic["unbiasedness"]
    assert abs(unbiasedness["mean_error"]) <= (
        unbiasedness["k_sigma"] * unbiasedness["standard_error"]
    )
    assert stochastic["realizations"] >= 32

    # Convergence is gated on a fitted exponent over a sweep, not one point.
    convergence = stochastic["convergence"]
    assert len(convergence["sweep"]) >= 5
    assert abs(convergence["fitted_exponent"] + 0.5) <= convergence["exponent_tolerance"]


def test_variance_is_reported_for_both_densities_on_both_spectra(result: dict) -> None:
    variance = result["stochastic"]["variance_by_density"]
    assert set(variance) == {"multilobed", "concentrated"}
    for entry in variance.values():
        assert {"p_uni", "p_mag"} <= set(entry)
    # p_mag exploits spectral concentration; that is the characterized property.
    assert (
        variance["concentrated"]["magnitude_advantage"]
        > variance["multilobed"]["magnitude_advantage"]
    )


def test_corruption_evidence_was_measured_not_asserted(result: dict) -> None:
    """A hardcoded returncode here would be fabricated evidence for the one
    property that makes every other hash in the bundle mean anything."""
    corruption = result["reproducibility"]["corrupted_fixture_rejection"]

    assert corruption["detected"] is True
    assert corruption["evaluator_returncode"] == 2
    # Both halves: the mutated bundle rejected AND the clean bundle accepted.
    assert corruption["clean_bundle_returncode"] == 0
    assert "hash_mismatch" in corruption["evaluator_stdout_head"]


def test_the_evaluator_really_does_reject_a_corrupted_bundle(tmp_path) -> None:
    """Independent re-run of the corruption check, rather than trusting the
    bundle's own account of it."""
    import shutil

    mutated = tmp_path / "bundle"
    shutil.copytree(BUNDLE, mutated)

    clean = subprocess.run(
        [sys.executable, str(BENCHMARK_DIR / "evaluate.py"), str(mutated)],
        capture_output=True, text=True, check=False,
    )
    assert clean.returncode == 0, clean.stdout

    with (mutated / "arrays.npz").open("ab") as handle:
        handle.write(b"x")
    corrupted = subprocess.run(
        [sys.executable, str(BENCHMARK_DIR / "evaluate.py"), str(mutated)],
        capture_output=True, text=True, check=False,
    )
    assert corrupted.returncode == 2
    assert "hash_mismatch" in corrupted.stdout


def test_the_fingerprint_covers_physics_and_not_machine_load(result: dict) -> None:
    """The M1.8 lesson. Per-case wall-clock had leaked into the wave branch's
    fingerprint and made it track how busy the machine was."""
    from multiscale_optics_agent.evaluation.m1_bundle import VOLATILE_KEYS

    reproducibility = result["reproducibility"]
    assert reproducibility["branch"] == "coupler"
    assert len(reproducibility["scientific_fingerprint"]) == 64

    stripped = reproducibility["scientific_projection"]["volatile_keys_stripped"]
    assert set(stripped) == set(VOLATILE_KEYS)
    for key in ("runtime_seconds", "timestamp_utc", "run_id"):
        assert key in stripped


def test_no_solver_engine_was_loaded_by_the_coupler_benchmark(provenance_record) -> None:
    assert provenance_record["forbidden_modules_loaded"] == []
    assert provenance_record["coupler_direction"] == "bidirectional"
    assert provenance_record["device"] == "cpu"
    assert provenance_record["dtype"] == "complex128"
    assert provenance_record["rng_generator"].startswith("numpy.random.Generator")


def test_differentiability_is_characterized_and_not_promoted(result: dict) -> None:
    differentiability = result["differentiability"]

    assert differentiability["claim"] != "verified"
    assert differentiability["omitted_terms"]
    assert differentiability["finite_difference_comparison"]["table"]
    assert "NOT promoted" in differentiability["promotion"]
    # The density-live control must be the one that IS biased, or the comparison
    # is not showing what it claims to show.
    assert differentiability["control_density_live"]["claim"] == "characterized_biased"


def test_the_registry_still_claims_no_verified_coupler_gradient() -> None:
    couplers = yaml.safe_load(
        (ROOT / "src/multiscale_optics_agent/registry/couplers.yaml").read_text()
    )["couplers"]
    by_id = {entry["id"]: entry for entry in couplers}

    for coupler_id in ("C_RAY_TO_WAVE", "C_WAVE_TO_RAY"):
        assert by_id[coupler_id]["derivative"]["verified"] is False
        assert by_id[coupler_id]["maturity"] == "experimental"


def test_the_manifest_records_which_l2_benchmarks_are_actually_implemented() -> None:
    """L2-PSF-01 is the end-to-end graph. CHE-39 (M3.10) implemented it -- the
    manifest must say so, but "implemented" must not be read as "gate met": the
    note is required to name the still-open physical-correctness gap rather
    than let a bare ``implemented: true`` imply the graph is fully verified."""
    manifest = yaml.safe_load((ROOT / "benchmarks/manifest.yaml").read_text())
    tasks = {task["id"]: task for task in manifest["levels"][2]["tasks"]}

    coupler = tasks["L2-COUPLER-01"]
    assert coupler["implemented"] is True
    assert coupler["protocol_id"] == "M2-COUPLER-CPU-V1"
    assert (ROOT / coupler["entry_point"]).is_file()

    psf = tasks["L2-PSF-01"]
    assert psf["implemented"] is True
    assert "blocked_by" not in psf
    assert psf["protocol_id"] == "M3-SLICE-CPU-V1"
    assert (ROOT / psf["entry_point"]).is_file()
    assert "1.0e-3" in psf["note"] or "1.0e-03" in psf["note"]
