"""Acceptance tests for the L1-WAVE-01 Chromatix wave benchmark (CHE-18).

The bundle is generated once per module by running the real evaluator through
the real pinned Chromatix/JAX install, then asserted against. Nothing here
recomputes physics: if a claim is not already in the emitted artifacts, it is
not a claim this benchmark makes.

Case 3 is expected to be **blocked**. Chromatix 0.6.0's ``high_na_ff_lens``
does not produce a sampling-independent focal field, so there is no converged
quantity to compare against the Richards-Wolf oracle. The tests below assert
that the benchmark detects and documents that, rather than asserting the
solver is correct.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

pytest.importorskip("chromatix")

pytestmark = [
    pytest.mark.jax,
    pytest.mark.integration,
    pytest.mark.chromatix,
    pytest.mark.benchmark,
]

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = ROOT / "benchmarks" / "level1" / "L1-WAVE-01"


@pytest.fixture(scope="module")
def wave_benchmark_bundle(tmp_path_factory):
    output = tmp_path_factory.mktemp("l1-wave-01")
    completed = subprocess.run(
        [
            sys.executable,
            str(BENCHMARK / "evaluate.py"),
            "--case",
            "gaussian",
            "--output-dir",
            str(output),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return output


@pytest.fixture(scope="module")
def wave_result(wave_benchmark_bundle):
    return json.loads((wave_benchmark_bundle / "result.json").read_text())


def _case1(result):
    return [
        c for c in result["accuracy"]["metrics"]["case1_exact_primitive"] if c["role"] == "accuracy"
    ]


def _case2(result):
    return [
        c for c in result["accuracy"]["metrics"]["case2_paraxial_focus"] if c["role"] == "accuracy"
    ]


def _case3(result):
    return result["accuracy"]["metrics"]["case3_high_na_vector"]


# ---------------------------------------------------------------------------
# Bundle and provenance
# ---------------------------------------------------------------------------
def test_bundle_contains_required_protocol_and_scientific_artifacts(
    wave_benchmark_bundle,
) -> None:
    required = {
        "result.json",
        "provenance.json",
        "arrays.npz",
        "plot.png",
        "tolerances.yaml",
        "README.md",
        "reference.json",
        "reference_fields.npz",
        "input_config.yaml",
        "error_attribution.json",
        "solver_summaries.json",
    }
    assert required <= {path.name for path in wave_benchmark_bundle.iterdir()}
    assert (wave_benchmark_bundle / "plot.png").stat().st_size > 10_000

    provenance = json.loads((wave_benchmark_bundle / "provenance.json").read_text())
    assert provenance["benchmark_id"] == "L1-WAVE-01"
    assert provenance["protocol_id"] == "M1-BASELINE-CPU-V1"
    assert provenance["device"] == "cpu"
    assert provenance["dtype"] == "complex64"
    assert provenance["engine_versions"]["chromatix"] == "0.6.0"
    assert (
        provenance["engine_versions"]["chromatix_commit"]
        == "d24bdf0022835bb8ce1cdcc6aeafbc7fcb39daee"
    )
    assert provenance["jax_backend"] == "cpu"
    assert provenance["jax_enable_x64"] is False
    # A wave-only baseline: no ray model and no coupler may be loaded.
    assert provenance["forbidden_modules_loaded"] == []
    for filename, expected_hash in provenance["artifact_hashes"].items():
        actual = hashlib.sha256((wave_benchmark_bundle / filename).read_bytes()).hexdigest()
        assert actual == expected_hash


def test_gating_excludes_the_blocked_case_and_says_so(wave_result) -> None:
    accuracy = wave_result["accuracy"]
    assert wave_result["status"] == "complete"
    assert accuracy["pass"]
    assert accuracy["gated_cases"] == ["case1", "case2"]
    assert accuracy["blocked_cases"] == ["case3"]
    assert "does not gate" in accuracy["blocked_case_note"]


# ---------------------------------------------------------------------------
# Case 1 -- exact homogeneous primitive
# ---------------------------------------------------------------------------
def test_case1_matches_the_exact_dispersion_relation_within_derived_round_off(
    wave_result,
) -> None:
    cases = _case1(wave_result)
    assert len(cases) >= 6

    for case in cases:
        errors = case["errors"]
        derived = case["derived_tolerances"]
        # The tolerance is derived from float32 phase round-off, not chosen,
        # and must track the accumulated phase rather than being a constant.
        assert derived["phase_error_rad"] >= derived["float32_phase_round_off_rad"]
        assert errors["phase_error_rad"] <= derived["phase_error_rad"]
        assert errors["complex_normalized_rms"] <= derived["complex_normalized_rms"]
        # An exact eigenmode: amplitude and discrete power are untouched.
        assert errors["amplitude_relative"] <= 1e-5
        assert errors["power_conservation_relative"] <= 1e-5
        round_trip_bound = max(1e-5, 2 * derived["complex_normalized_rms"])
        assert errors["round_trip_normalized_rms"] <= round_trip_bound
        assert case["pass"]

    # Longer propagation must cost more phase precision; a constant tolerance
    # would have hidden this.
    by_phase = sorted(cases, key=lambda c: abs(c["closed_form"]["accumulated_phase_rad"]))
    assert (
        by_phase[-1]["derived_tolerances"]["phase_error_rad"]
        > by_phase[0]["derived_tolerances"]["phase_error_rad"]
    )


def test_case1_runs_unpadded_because_the_mode_is_periodic(wave_result) -> None:
    for case in _case1(wave_result):
        assert case["grid"]["pad_width"] == 0
        assert case["solver"]["pad_width"] == 0
        assert case["solver"]["padded"] is False
        assert case["solver"]["cropped"] is False
        assert case["solver"]["output_shape"] == [case["grid"]["n"], case["grid"]["n"]]
        assert case["gates"]["no_padding_applied"]


def test_case1_includes_an_axis_asymmetric_and_a_large_angle_mode(wave_result) -> None:
    """A symmetric on-axis mode alone could not detect an axis swap."""
    modes = [(c["mode"]["mode_y"], c["mode"]["mode_x"]) for c in _case1(wave_result)]
    assert any(my != mx for my, mx in modes)
    assert any(my < 0 for my, _ in modes)
    assert max(c["mode"]["sin_theta"] for c in _case1(wave_result)) > 0.3


# ---------------------------------------------------------------------------
# Case 2 -- ideal signed paraxial focusing
# ---------------------------------------------------------------------------
def test_case2_signed_focal_position_and_diffraction_widths(wave_result) -> None:
    cases = _case2(wave_result)
    tolerances = wave_result["accuracy"]["tolerances"]
    assert len(cases) >= 3

    tilts = sorted(c["pupil"]["tilt_x_rad"] for c in cases)
    assert tilts[0] < 0 < tilts[-1], "signed focusing needs a negative and a positive tilt"
    assert any(c["pupil"]["tilt_x_rad"] != c["pupil"]["tilt_y_rad"] for c in cases), (
        "an asymmetric tilt is what makes a y/x swap visible in the focal position"
    )

    for case in cases:
        errors = case["errors"]
        closed = case["closed_form"]
        # Focus sits at +f*theta, with the sign.
        assert closed["centroid_x_m"] == pytest.approx(
            case["pupil"]["focal_length_m"] * case["pupil"]["tilt_x_rad"]
        )
        assert errors["centroid_x_input_pixels"] <= tolerances["case2_centroid_input_pixels"]
        assert errors["centroid_y_input_pixels"] <= tolerances["case2_centroid_input_pixels"]
        assert errors["fwhm_x_relative"] <= tolerances["case2_fwhm_relative"]
        assert errors["fwhm_y_relative"] <= tolerances["case2_fwhm_relative"]
        assert errors["first_sidelobe_relative"] <= tolerances["case2_sidelobe_relative"]
        assert errors["overlap_fresnel"] >= tolerances["case2_overlap_fresnel_minimum"]
        assert (
            errors["power_conservation_relative"] <= tolerances["case2_power_conservation_relative"]
        )
        assert case["pass"]


def test_case2_rectangular_aperture_widths_differ_between_axes(wave_result) -> None:
    """The two axes must be distinguishable, or an axis swap is undetectable."""
    for case in _case2(wave_result):
        pupil = case["pupil"]
        assert pupil["aperture_samples_x"] != pupil["aperture_samples_y"]
        assert pupil["aperture_samples_x"] % 2 == 1, "odd counts keep the aperture centred"
        assert pupil["aperture_samples_y"] % 2 == 1
        closed = case["closed_form"]
        # Narrower aperture -> wider focal lobe.
        assert closed["fwhm_y_m"] > closed["fwhm_x_m"]
        assert closed["first_sidelobe_ratio"] == pytest.approx(0.04718036, rel=1e-6)


def test_case2_error_attribution_puts_the_residual_in_the_model_not_the_solver(
    wave_result,
) -> None:
    for case in _case2(wave_result):
        attribution = case["error_attribution"]
        assert set(attribution["per_axis"]) == {"x", "y"}
        # Chromatix agrees with an independent float64 angular spectrum far more
        # closely than the paraxial oracle agrees with either of them.
        assert attribution["solver_implementation"] < attribution["paraxial_model"]
        assert attribution["convention"] < 1e-4
        assert attribution["normalization"] < 1e-3
        assert case["errors"]["overlap_independent_asm"] >= 0.9999


# ---------------------------------------------------------------------------
# Case 3 -- high-NA vectorial focusing (expected: blocked)
# ---------------------------------------------------------------------------
def test_case3_oracle_is_shown_converged_before_it_judges_anything(wave_result) -> None:
    case3 = _case3(wave_result)
    convergence = case3["oracle_quadrature_convergence"]
    assert len(convergence["quadrature_points"]) >= 4
    assert convergence["quadrature_points"] == sorted(convergence["quadrature_points"])
    assert convergence["tail_relative_spread"] <= 1e-6
    assert case3["gates"]["oracle_quadrature_converged"]
    # Known aplanatic value for NA = 0.9 in air.
    assert case3["closed_form"]["iz_over_ix"] == pytest.approx(0.150087, rel=1e-4)


def test_case3_is_blocked_by_a_solver_sampling_non_convergence(wave_result) -> None:
    case3 = _case3(wave_result)
    assert case3["status"] == "blocked"
    assert case3["pass"] is False

    qualification = case3["sampling_qualification"]
    assert qualification["scale_stable"] is False
    # Refining only the pupil sampling moves the focal scale by far more than
    # the tolerance -- a physical field cannot do that.
    assert qualification["ez_peak_radius_relative_spread"] > 10 * qualification["scale_tolerance"]

    sweep = case3["pupil_sampling_sweep"]
    assert len(sweep) >= 4
    assert [s["pupil_samples"] for s in sweep] == sorted(s["pupil_samples"] for s in sweep)
    radii = [s["ez_peak_radius_m"] for s in sweep]
    assert max(radii) > 2 * min(radii)

    reason = case3["blocked_reason"]
    assert reason and "high_na_ff_lens" in reason
    assert "f_grid" in reason, "the blocked reason must name the root cause, not just the symptom"


def test_case3_blocked_status_is_attributable_to_the_solver_not_the_oracle(
    wave_result,
) -> None:
    """The oracle converges to 1e-14 while the solver moves by >100%."""
    case3 = _case3(wave_result)
    oracle_spread = case3["oracle_quadrature_convergence"]["tail_relative_spread"]
    solver_spread = case3["sampling_qualification"]["ez_peak_radius_relative_spread"]
    assert oracle_spread < 1e-6 < solver_spread


# ---------------------------------------------------------------------------
# Negative perturbations
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "perturbation",
    [
        "case1_paraxial_dispersion",
        "case2_lens_sign_flip",
        "case2_axis_transpose",
        "case2_si_scale",
    ],
)
def test_deliberate_perturbations_are_detected_while_controls_pass(
    wave_result, perturbation: str
) -> None:
    report = wave_result["accuracy"]["metrics"]["perturbations"]
    assert report["pass"]

    entry = report["perturbations"][perturbation]
    assert entry["detected"] is True
    assert entry["observed"] > entry["detection_threshold"]
    # Every control must pass, so a detection cannot come from an evaluator
    # that simply rejects everything.
    assert all(report["controls_pass"].values())


def test_paraxial_dispersion_perturbation_proves_case1_resolves_the_exact_sqrt(
    wave_result,
) -> None:
    """Case 1 must distinguish sqrt(1-(lambda f)^2) from its Taylor expansion."""
    entry = wave_result["accuracy"]["metrics"]["perturbations"]["perturbations"][
        "case1_paraxial_dispersion"
    ]
    assert entry["observed"] > 1.0, "a paraxial k_z must produce radians of phase error"


# ---------------------------------------------------------------------------
# Conventions and artifacts
# ---------------------------------------------------------------------------
def test_saved_fields_are_complex_amplitudes(wave_benchmark_bundle, wave_result) -> None:
    arrays = np.load(wave_benchmark_bundle / "arrays.npz")
    reference = np.load(wave_benchmark_bundle / "reference_fields.npz")

    case = _case2(wave_result)[0]
    output_field = arrays[f"{case['name']}_output_field"]
    assert np.iscomplexobj(output_field)
    assert output_field.shape == (case["grid"]["padded_n"], case["grid"]["padded_n"])
    assert np.any(output_field.imag != 0.0)
    assert np.all(np.isfinite(output_field))

    # Intensity is derived, never substituted for the amplitude.
    intensity = np.abs(output_field) ** 2
    assert not np.allclose(intensity, np.abs(output_field))

    # The vector field carries all three Cartesian components.
    vector = arrays["case3_output_field_xyz"]
    assert vector.ndim == 3 and vector.shape[-1] == 3
    assert np.iscomplexobj(vector)
    assert "case3_expected_field_xyz" in reference.files


def test_declared_conventions_are_recorded_and_match_the_config(
    wave_benchmark_bundle, wave_result
) -> None:
    conventions = wave_result["accuracy"]["metrics"]["conventions"]
    config = yaml.safe_load((wave_benchmark_bundle / "input_config.yaml").read_text())
    assert conventions == config["conventions"]
    assert conventions["axis_order"] == "(y, x)"
    assert conventions["phasor"] == "exp(-i omega t)"
    assert "E_x, E_y, E_z" in conventions["vector_component_order"]
    assert "E_z, E_y, E_x" in conventions["vector_component_order"], (
        "chromatix's opposite component order must be recorded, not silently converted"
    )

    summaries = json.loads((wave_benchmark_bundle / "solver_summaries.json").read_text())
    metadata = summaries[_case2(wave_result)[0]["name"]]["field_metadata"]
    assert metadata["axis_order"] == "(y, x)"
    assert metadata["package_commit"] == "d24bdf0022835bb8ce1cdcc6aeafbc7fcb39daee"


def test_oracle_module_never_imports_the_solver(wave_benchmark_bundle) -> None:
    """The analytic reference must be buildable without Chromatix installed."""
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                f"import sys; sys.path.insert(0, {str(BENCHMARK)!r}); "
                "import oracles, measurement; "
                "leaked = [m for m in sys.modules "
                "if m.startswith(('chromatix', 'jax'))]; "
                "assert not leaked, leaked"
            ),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_accuracy_and_performance_are_separate(wave_result) -> None:
    performance = wave_result["performance"]
    assert performance["warmup_runs"] == 2
    assert performance["measured_repeats"] == 7
    assert len(performance["samples_seconds"]) == 7
    assert performance["peak_memory"]["status"] == "measured"
    assert "accuracy" not in performance
    assert "performance" not in wave_result["accuracy"]
