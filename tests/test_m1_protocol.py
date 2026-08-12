from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_engine_probe_uses_independent_compatible_processes() -> None:
    completed = subprocess.run(
        [sys.executable, "benchmarks/probes/verify_m1_engines.py"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(completed.stdout)

    assert report["status"] == "compatible"
    assert report["execution"] == "independent_subprocesses"
    ray = report["reports"]["ray"]
    wave = report["reports"]["wave"]
    assert ray["environment"]["process_id"] != wave["environment"]["process_id"]
    assert ray["engines"]["ray"]["forbidden_modules_loaded"] == []
    assert wave["engines"]["wave"]["forbidden_modules_loaded"] == []
    assert ray["engines"]["ray"]["known_distance"]["status"] == "verified"
    assert ray["engines"]["ray"]["known_distance"]["native_geometry_length_unit"] == "mm"
    assert all(ray["engines"]["ray"]["pin_checks"].values())
    assert all(wave["engines"]["wave"]["pin_checks"].values())


def test_protocol_freezes_required_measurements_and_artifacts() -> None:
    protocol = yaml.safe_load((ROOT / "benchmarks/protocol.yaml").read_text())

    assert protocol["protocol_id"] == "M1-BASELINE-CPU-V1"
    assert protocol["execution"]["warmup_runs"] >= 1
    assert protocol["execution"]["measured_repeats"] >= 5
    assert protocol["execution"]["timing"]["primary_statistic"] == "median"
    assert protocol["execution"]["device"] == "cpu"
    assert protocol["result_sections"]["accuracy"]["separate_from_performance"]
    assert protocol["result_sections"]["performance"]["separate_from_accuracy"]
    assert set(protocol["artifacts"]["required"]) == {
        "result.json",
        "provenance.json",
        "arrays.npz",
        "plot.png",
        "tolerances.yaml",
        "README.md",
    }


def test_protocol_json_schemas_are_valid_json_and_match_protocol() -> None:
    result_schema = json.loads((ROOT / "benchmarks/schemas/result.schema.json").read_text())
    provenance_schema = json.loads((ROOT / "benchmarks/schemas/provenance.schema.json").read_text())

    assert result_schema["properties"]["status"]["enum"] == ["complete", "blocked"]
    assert (
        result_schema["properties"]["performance"]["properties"]["measured_repeats"]["minimum"] == 5
    )
    assert provenance_schema["properties"]["protocol_id"]["enum"] == [
        "M1-BASELINE-CPU-V1",
        "M1-BASELINE-CPU-V2",
    ]
    assert (
        provenance_schema["properties"]["artifact_hashes"]["additionalProperties"]["pattern"]
        == "^[0-9a-f]{64}$"
    )


def test_cards_and_container_pins_match_verified_stack() -> None:
    requirements = (ROOT / "docker/requirements.txt").read_text()
    dockerfile = (ROOT / "docker/Dockerfile").read_text()
    optiland = yaml.safe_load((ROOT / "knowledge/solver_cards/optiland.yaml").read_text())
    chromatix = yaml.safe_load((ROOT / "knowledge/solver_cards/chromatix.yaml").read_text())
    registry = yaml.safe_load(
        (ROOT / "src/multiscale_optics_agent/registry/models.yaml").read_text()
    )
    models = {model["id"]: model for model in registry["models"]}

    assert "optiland==0.6.0" in requirements
    assert "jax==0.6.2" in requirements
    assert "numpy==2.2.6" in requirements
    assert "d24bdf0022835bb8ce1cdcc6aeafbc7fcb39daee" in requirements
    assert "torch==2.13.0" in dockerfile
    assert optiland["validation_status"] == "environment_verified"
    assert chromatix["validation_status"] == "environment_verified"
    assert "optiland==0.6.0" in optiland["install"]["source"]
    assert "d24bdf0022835bb8ce1cdcc6aeafbc7fcb39daee" in chromatix["install"]["source"]
    assert models["M_RAY_OPTILAND"]["source"]["pinned_version"] == "0.6.0"
    assert models["M_WAVE_CHROMATIX"]["source"]["pinned_version"] == "0.6.0"
    assert models["M_WAVE_CHROMATIX"]["source"]["pinned_commit"] == (
        "d24bdf0022835bb8ce1cdcc6aeafbc7fcb39daee"
    )
