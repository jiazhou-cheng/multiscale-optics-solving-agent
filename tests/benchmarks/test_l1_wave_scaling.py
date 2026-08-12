from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("chromatix")
pytestmark = [pytest.mark.jax, pytest.mark.integration]

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = ROOT / "benchmarks" / "level1" / "L1-WAVE-01"


@pytest.fixture(scope="module")
def wave_scaling_bundle(tmp_path_factory: pytest.TempPathFactory) -> Path:
    output = tmp_path_factory.mktemp("l1-wave-scaling")
    completed = subprocess.run(
        [
            sys.executable,
            str(BENCHMARK / "run_scaling.py"),
            "--device",
            "cpu",
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


def test_grid_padding_and_timing_contract(wave_scaling_bundle: Path) -> None:
    result = json.loads((wave_scaling_bundle / "result.json").read_text())
    scaling = result["scaling"]
    assert scaling["grid_sequence"] == [64, 128, 256]
    pads = []
    cells = []
    for case in scaling["cases"]:
        n = case["grid"]["input_n"]
        assert case["grid"]["input_shape"] == [n, n]
        assert case["grid"]["physical_window_m"] == pytest.approx(64e-6)
        assert case["grid"]["input_spacing_m"] == pytest.approx([64e-6 / n] * 2)
        assert case["padding"]["policy"] == "auto_transfer"
        pads.append(case["padding"]["pad_width"])
        cells.append(case["padding"]["propagated_cells"])
        assert case["timing"]["compile_plus_execute_seconds"] >= 0
        assert len(case["timing"]["steady_samples_seconds"]) == 7
        assert "never mixed" in case["timing"]["cache_policy"]
        assert case["throughput"]["propagated_output_cells_per_second"] > 0
    assert len(set(pads)) > 1
    assert cells == sorted(cells)


def test_gaussian_power_determinism_and_environment_gates(wave_scaling_bundle: Path) -> None:
    result = json.loads((wave_scaling_bundle / "result.json").read_text())
    assert result["accuracy"]["pass"]
    assert result["scaling"]["pass"]
    assert result["scaling"]["gaussian_accuracy_and_power_pass"]
    assert result["scaling"]["determinism_pass"]
    assert result["scaling"]["smallest_case_compile_inclusive_under_10_seconds"]
    assert len(result["scaling"]["environment_fingerprint"]) == 64
    for case in result["scaling"]["cases"]:
        accuracy = case["gaussian_accuracy"]
        assert accuracy["radius_relative_error"] <= 0.01
        assert accuracy["centroid_input_pixels"] <= 0.1
        assert accuracy["power_conservation_relative_error"] <= 1e-3
        assert case["determinism"]["field_hashes_identical"]
        assert case["determinism"]["physical_summaries_identical"]
        assert case["runtime_environment"]["jax_backend"] == "cpu"
        assert case["runtime_environment"]["jax_enable_x64"] is False
        assert case["runtime_environment"]["dtype"] == "complex64"


def test_complex_field_artifacts_and_scaling_evaluator(wave_scaling_bundle: Path) -> None:
    result = json.loads((wave_scaling_bundle / "result.json").read_text())
    for case in result["scaling"]["cases"]:
        field = np.load(wave_scaling_bundle / case["field_artifact"])
        assert field.dtype == np.complex64
        assert field.shape == tuple(case["grid"]["output_shape"])
        assert np.iscomplexobj(field) and np.all(np.isfinite(field))
    completed = subprocess.run(
        [
            sys.executable,
            str(BENCHMARK / "evaluate.py"),
            "--section",
            "scaling",
            "--output-dir",
            str(wave_scaling_bundle),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_scaling_evaluator_rejects_corrupted_field(
    wave_scaling_bundle: Path, tmp_path: Path
) -> None:
    copied = tmp_path / "corrupted"
    shutil.copytree(wave_scaling_bundle, copied)
    field = next((copied / "complex_fields").glob("*.npy"))
    field.write_bytes(field.read_bytes() + b"corruption")
    spec = importlib.util.spec_from_file_location(
        "l1_wave_scaling_evaluate", BENCHMARK / "scaling_evaluate.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    errors = module.evaluate_scaling_bundle(copied)
    assert any("hash mismatch" in error for error in errors)
