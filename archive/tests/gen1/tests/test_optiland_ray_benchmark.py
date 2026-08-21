from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

pytest.importorskip("optiland")

pytestmark = [pytest.mark.integration, pytest.mark.optiland, pytest.mark.benchmark]

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "benchmarks" / "level1" / "L1-RAY-01"


@pytest.fixture(scope="module")
def ray_benchmark_bundle(tmp_path_factory):
    output = tmp_path_factory.mktemp("l1-ray-01")
    completed = subprocess.run(
        [sys.executable, str(BENCHMARK / "run_benchmark.py"), "--output-dir", str(output)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return output


def test_bundle_contains_required_protocol_and_scientific_artifacts(ray_benchmark_bundle) -> None:
    required = {
        "result.json",
        "provenance.json",
        "arrays.npz",
        "plot.png",
        "free_space_diagnostics.png",
        "paraxial_diagnostics.png",
        "catalog_diagnostics.png",
        "tolerances.yaml",
        "README.md",
        "input_config.yaml",
        "prescription.yaml",
        "ray_inputs.npz",
        "expected.npz",
        "error_attribution.json",
    }
    assert required <= {path.name for path in ray_benchmark_bundle.iterdir()}
    for plot_name in (
        "plot.png",
        "free_space_diagnostics.png",
        "paraxial_diagnostics.png",
        "catalog_diagnostics.png",
    ):
        assert (ray_benchmark_bundle / plot_name).stat().st_size > 10_000

    provenance = json.loads((ray_benchmark_bundle / "provenance.json").read_text())
    assert provenance["benchmark_id"] == "L1-RAY-01"
    assert provenance["protocol_id"] == "M1-BASELINE-CPU-V2"
    assert "surface-shape" in provenance["protocol_amendment"]
    assert provenance["device"] == "cpu"
    assert provenance["dtype"] == "float64"
    assert provenance["engine_versions"]["optiland"] == "0.6.0"
    assert provenance["forbidden_modules_loaded"] == []
    for filename, expected_hash in provenance["artifact_hashes"].items():
        actual = hashlib.sha256((ray_benchmark_bundle / filename).read_bytes()).hexdigest()
        assert actual == expected_hash


def test_all_three_accuracy_cases_and_negative_convention_test_pass(
    ray_benchmark_bundle,
) -> None:
    result = json.loads((ray_benchmark_bundle / "result.json").read_text())
    assert result["status"] == "complete"
    assert result["accuracy"]["pass"]
    metrics = result["accuracy"]["metrics"]
    assert metrics["free_space"]["pass"]
    assert metrics["paraxial_thin_lens"]["pass"]
    assert metrics["catalog_lens"]["pass"]
    assert metrics["convention_negative_test"]["detected"]

    tolerances = result["accuracy"]["tolerances"]
    assert (
        metrics["free_space"]["max_position_error_m"] <= (tolerances["free_space_position_scaled"])
    )
    assert (
        metrics["free_space"]["max_geometric_path_or_opl_error_m"]
        <= (tolerances["free_space_path_scaled"])
    )
    assert (
        metrics["paraxial_thin_lens"]["focal_intercept_relative_error"]
        <= (tolerances["paraxial_focal_intercept_relative_to_focal_length"])
    )
    for case in ("free_space", "paraxial_thin_lens", "catalog_lens"):
        assert metrics[case]["max_direction_norm_error"] <= 1e-12


def test_catalog_lens_reference_regression_and_conventions(ray_benchmark_bundle) -> None:
    prescription = yaml.safe_load((ray_benchmark_bundle / "prescription.yaml").read_text())
    assert prescription["manufacturer"] == "Edmund Optics"
    assert prescription["part_number"] == 'TECHSPEC stock "#45-362"'
    assert prescription["surface_1_radius_mm"] == pytest.approx(25.84)
    assert prescription["center_thickness_mm"] == pytest.approx(3.23)
    assert prescription["clear_aperture_mm"] == pytest.approx(19.0)
    assert prescription["catalog_efl_mm"] == pytest.approx(50.0)
    assert prescription["catalog_bfl_mm"] == pytest.approx(47.87)

    result = json.loads((ray_benchmark_bundle / "result.json").read_text())
    metrics = result["accuracy"]["metrics"]["catalog_lens"]
    assert metrics["launch_slopes_rad"] == [-0.01, 0.0, 0.01]
    assert metrics["ray_count"] == 27
    assert metrics["invalid_ray_count"] == 0
    assert metrics["vignetted_ray_count"] == 0
    assert metrics["refractive_index_absolute_error"] <= 5e-6
    assert metrics["efl_relative_error"] <= 0.01
    assert metrics["bfl_relative_error"] <= 0.01
    assert metrics["axial_geometric_path_error_m"] <= 1e-10
    assert metrics["axial_opl_error_m"] <= 1e-10
    assert metrics["on_axis_centroid_absolute_m"] <= 1e-12
    assert metrics["field_antisymmetry_error_m"] <= 1e-12
    assert metrics["front_surface_max_sag_error_m"] <= 1e-12
    assert metrics["rear_surface_max_plane_error_m"] <= 1e-12
    assert metrics["front_surface_sample_count"] == 27
    assert metrics["surface_shape_pass"]
    assert metrics["clear_aperture_m"] == pytest.approx(0.019)
    assert metrics["inside_aperture_transmitted_count"] == 2
    assert metrics["inside_aperture_vignetted_count"] == 0
    assert metrics["outside_aperture_transmitted_count"] == 0
    assert metrics["outside_aperture_vignetted_count"] == 2
    assert metrics["aperture_classification_pass"]
    assert "ray minus chief" in metrics["opd_sign"]

    arrays = np.load(ray_benchmark_bundle / "arrays.npz")
    assert arrays["catalog_x_m"].shape == (27,)
    assert arrays["catalog_history_x_m"].shape == (4, 27)
    assert arrays["catalog_opd_ray_minus_chief_m"].shape == (27,)
    assert np.all(np.isfinite(arrays["catalog_opd_ray_minus_chief_m"]))
    assert arrays["catalog_aperture_x_m"].shape == (4,)
    assert arrays["catalog_front_surface_height_m"].shape == (401,)
    assert arrays["catalog_front_surface_z_m"].shape == (401,)
    assert arrays["catalog_rear_surface_height_m"].shape == (401,)
    assert np.allclose(arrays["catalog_rear_surface_z_m"], 0.00323)
    assert arrays["catalog_clear_aperture_m"] == pytest.approx([0.019])
    assert arrays["catalog_image_plane_z_m"] == pytest.approx([0.05110])
    expected = np.load(ray_benchmark_bundle / "expected.npz")
    assert "catalog_opd_ray_minus_chief_m" not in expected.files
    assert "catalog_analytic_efl_m" in expected.files


def test_accuracy_and_performance_are_separate(ray_benchmark_bundle) -> None:
    result = json.loads((ray_benchmark_bundle / "result.json").read_text())
    performance = result["performance"]
    assert performance["warmup_runs"] == 2
    assert performance["measured_repeats"] == 7
    assert len(performance["samples_seconds"]) == 7
    assert performance["statistics"]["median_seconds"] >= 0
    assert performance["peak_memory"]["status"] == "measured"


def test_geometric_path_native_mm_conversion_regression() -> None:
    module_path = BENCHMARK / "run_benchmark.py"
    spec = importlib.util.spec_from_file_location("l1_ray_01_run_benchmark", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module._absolute_mm_error_to_m(2.0, 1.0) == pytest.approx(1e-3)
