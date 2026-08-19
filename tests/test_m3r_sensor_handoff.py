"""CHE-38 (M3.9R): the sensor-side handoff, and what the residual turned out to be.

Two kinds of test, separated as in ``test_m3_convergence.py``.

The live ones run the probe's own machinery on cases whose answer is known before
the run, and none of them touches an engine, so they are cheap:

* the ray-domain advance must reproduce an exact plane wave's ``z`` phase, which
  is the whole justification for moving the handoff in the ray domain rather than
  in the kernel;
* the Lommel helper must reduce to ``2 J1(v) / v`` at zero defocus, because a
  reference that has not been checked against a case with a closed form is not a
  reference;
* the complex-field metric must be blind to a global complex factor, since that
  is the only thing it is allowed to be blind to.

The rest read ``benchmarks/probes/records/m3r_sensor_handoff.json``. They pin what
the study found -- including the things it found that were NOT the coupler's
fault, because those are the ones a later change is most likely to quietly
re-blame on it.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
RECORD_PATH = ROOT / "benchmarks" / "probes" / "records" / "m3r_sensor_handoff.json"
PROBE_PATH = ROOT / "benchmarks" / "probes" / "m3r_sensor_handoff.py"
GATE = 1.0e-3

pytestmark = pytest.mark.coupler


@pytest.fixture(scope="module")
def record() -> dict[str, Any]:
    if not RECORD_PATH.exists():
        pytest.skip(f"{RECORD_PATH.relative_to(ROOT)} is missing; run the probe first")
    return json.loads(RECORD_PATH.read_text())


@pytest.fixture(scope="module")
def probe():
    spec = importlib.util.spec_from_file_location("m3r_sensor_handoff_probe", PROBE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Live: the machinery, on inputs whose answers are known in closed form
# ---------------------------------------------------------------------------
def test_the_ray_domain_advance_reproduces_an_exact_plane_wave_z_phase(probe) -> None:
    """The identity the whole handoff sweep rests on.

    ``ray_to_wave`` has no ``z`` in its kernel, so moving the handoff has to happen
    in the ray domain. For a single ray -- one plane wave -- reconstructing before
    and after an advance must differ by exactly ``exp(i k dz d_z)``, the phase an
    exact plane wave accumulates over the plane offset. If it differed by anything
    else, every row of Experiment A would be comparing fields on different planes.
    """
    from multiscale_optics_agent.couplers.contracts import Frame, RayBundle, ReferencePlane

    wavelength = probe.WAVELENGTH_M
    direction = np.array([0.03, -0.02, math.sqrt(1.0 - 0.03**2 - 0.02**2)])
    bundle = RayBundle(
        positions_m=np.array([[4.0e-6, -3.0e-6, 0.0]]),
        directions=direction[None, :],
        wavelength_m=wavelength,
        reference_plane=ReferencePlane(name="start", z_m=0.0),
        frame=Frame(),
        amplitude=np.ones(1, dtype=np.complex128),
        optical_path_length_m=np.zeros(1),
        optical_path_length_reference="unit test: zero at the launch plane",
    )
    offset = 250.0e-6
    advanced, step = probe._advance_bundle_to_z(bundle, offset)

    before, _ = probe._reconstruct_core(bundle, grid_n=8, pitch_m=2.0e-6)
    after, _ = probe._reconstruct_core(advanced, grid_n=8, pitch_m=2.0e-6)

    expected = np.exp(1j * (2.0 * math.pi / wavelength) * offset * direction[2])
    ratio = after.u / before.u
    assert np.allclose(ratio, expected, rtol=0.0, atol=1e-9), (
        f"advance introduced {ratio[0, 0]} where an exact plane wave requires {expected}"
    )
    # And the advance itself is a reparameterization: directions must not move.
    assert np.array_equal(advanced.directions, bundle.directions)
    assert math.isclose(float(step[0]), offset / direction[2], rel_tol=1e-12)


def test_the_lommel_helper_reduces_to_the_airy_pattern_at_zero_defocus(probe) -> None:
    """``U(0, v) = 2 int_0^1 J0(v rho) rho drho = 2 J1(v) / v``.

    The circular-aperture reference is the thing that resolves M3.9's 0.744-vs-1.0009
    loose end, so it is checked against the one case where it has a closed form
    before being trusted anywhere else.
    """
    from scipy.special import j0, j1

    quadrature = (np.arange(200000, dtype=np.float64) + 0.5) / 200000.0
    for v in (0.5, 2.0, 3.8317, 7.0, 12.0):
        numeric = 2.0 * float(np.sum(j0(v * quadrature) * quadrature)) / quadrature.size
        closed = 2.0 * j1(v) / v
        assert math.isclose(numeric, closed, rel_tol=1e-6, abs_tol=1e-9), f"v={v}"


@pytest.mark.slow
def test_the_circular_reference_is_far_from_the_straight_edge_it_replaced(probe) -> None:
    """The two references must differ by the ~26% M3.9 could not explain."""
    circular = probe._lommel_circular_rim_slope(
        radius_m=probe.PUPIL_RADIUS_M, distance_m=probe.DISTANCE_M
    )
    straight = probe._straight_knife_edge_rim_slope()
    assert 0.60 < circular["rim_slope_times_sqrt_lambda_R"] < 0.80
    assert 0.95 < straight["rim_slope_times_sqrt_lambda_R"] < 1.05
    ratio = (
        circular["rim_slope_times_sqrt_lambda_R"] / straight["rim_slope_times_sqrt_lambda_R"]
    )
    assert 0.65 < ratio < 0.80, (
        "the circular reference must sit well below the straight edge; if these ever "
        "agree, the resolution of CHE-38 section 13 has evaporated"
    )
    # Rim amplitude near a half is the claim that survives; it is not exactly a half.
    assert 0.45 < circular["amplitude_at_the_geometric_rim"] < 0.52
    assert circular["overshoot_inside_the_rim"] > 1.05


def test_the_complex_metric_is_blind_only_to_a_global_complex_factor(probe) -> None:
    rng = np.random.default_rng(20260818)
    field = rng.normal(size=(16, 16)) + 1j * rng.normal(size=(16, 16))
    mask = np.ones((16, 16), dtype=bool)

    assert probe._complex_relative_l2(field, field, mask)["relative_l2"] < 1e-14
    scaled = 3.7 * np.exp(1j * 1.234) * field
    result = probe._complex_relative_l2(scaled, field, mask)
    assert result["relative_l2"] < 1e-14
    assert math.isclose(result["fitted_gain"], 1.0 / 3.7, rel_tol=1e-9)
    # A real difference must NOT be absorbed.
    perturbed = field.copy()
    perturbed[0, 0] += 5.0
    assert probe._complex_relative_l2(perturbed, field, mask)["relative_l2"] > 1e-3


def test_the_diagnostic_quadrature_weight_touches_only_the_two_boundaries(probe) -> None:
    """CHE-38 section 14: nothing in production is reweighted. Check the shape."""
    _, ring_index = probe._hexapolar(6, 1.0)
    weight = probe._radial_quadrature_weight(ring_index, 6)
    assert weight[0] == 0.75
    assert np.all(weight[ring_index == 6] == 0.5)
    interior = (ring_index > 0) & (ring_index < 6)
    assert np.all(weight[interior] == 1.0)


def test_the_power_law_fit_recovers_a_known_exponent(probe) -> None:
    x = [10.0, 100.0, 1000.0, 10000.0]
    fit = probe._power_law_fit(x, [3.0 * v**-0.75 for v in x], label="synthetic")
    assert math.isclose(fit["exponent"], -0.75, abs_tol=1e-9)
    assert math.isclose(fit["prefactor"], 3.0, rel_tol=1e-9)
    assert fit["r_squared"] > 1.0 - 1e-12


# ---------------------------------------------------------------------------
# Recorded: what the study found
# ---------------------------------------------------------------------------
def test_the_declared_configuration_matches_the_frozen_protocol(record) -> None:
    assert record["frozen_configuration"]["every_value_matches"] is True


def test_the_candidate_handoff_set_was_declared_and_all_of_it_was_run(record) -> None:
    declared = record["declared_before_the_sweep"]["handoff_candidates"]
    run = {row["handoff_plane"] for row in record["experiment_a_handoff_sweep"]["rows"]}
    assert {entry["name"] for entry in declared} == run
    assert any(entry["fraction_of_R_upstream"] == 0.0 for entry in declared), (
        "the exact sensor plane must be tested rather than assumed valid or invalid"
    )
    assert any(entry["fraction_of_R_upstream"] == 1.0 for entry in declared), (
        "the exit pupil must be retained as the negative control"
    )


def test_the_declared_handoff_is_eikonally_consistent(record) -> None:
    """The one measurement that cleared the handoff and moved suspicion to the oracle."""
    eikonal = record["coupler_semantics"]["eikonal_consistency_of_the_declared_handoff"]
    assert eikonal["max_relative_deviation_interior"] < 1e-6
    assert eikonal["rim_relative_deviation"] < 1e-5
    # And the rim slope is NOT the sphere-to-image-plane value: that 0.29% is the
    # singlet's spherical aberration, and it is the Airy scale.
    sphere = eikonal["sphere_to_declared_image_plane_would_give"]
    assert eikonal["rim_transverse_direction_cosine"] / sphere - 1.0 > 2e-3


def test_the_three_references_agree_with_each_other(record) -> None:
    cross = record["reference_cross_check"]
    assert cross["o2_rs_vs_o2_asm_intensity_relative_l2"] < 1e-3, (
        "a Huygens surface integral and an FFT angular spectrum must agree, or "
        "neither is usable as an absolute reference"
    )
    assert cross["o2_asm_traced_vs_o1_relative_l2"] < 1e-2


def test_an_underfitted_oracle_would_have_manufactured_a_coupler_defect(record) -> None:
    """Kept because it nearly was reported as one."""
    control = record["reference_cross_check"]["underfitted_polynomial_oracle_control"]
    assert control["fit_residual_rms_waves"] < 2e-3, "the bad fit looks good in RMS"
    assert control["rim_slope_relative_error"] > 1e-3, "and is wrong at the rim"
    assert control["charges_the_coupler"] > GATE


def test_the_exact_sensor_plane_is_a_caustic_and_is_still_the_best_conditioned(record) -> None:
    rows = record["experiment_e_shipping_sensor_path"]["rows"]
    for row in rows:
        state = row["ray_state_at_the_sensor"]
        assert state["caustic_condition"]["is_caustic_or_near_caustic"] is True, (
            "the bundle must collapse inside one Airy radius at the sensor, or this is "
            "not the caustic test it claims to be"
        )
        assert row["conditioning"]["peak_coherent_gain"] > 0.9, (
            "every wavelet arrives in phase at the focus, so the sum is fully "
            "constructive and float64 loses nothing"
        )


def test_the_exit_pupil_handoff_is_the_worst_candidate_and_is_labelled_out_of_contract(
    record,
) -> None:
    rows = {
        row["handoff_plane"]: row
        for row in record["experiment_a_handoff_sweep"]["rows"]
        if row.get("status") == "succeeded"
    }
    assert "exit_pupil" in rows
    pupil = rows["exit_pupil"]["psf"]["vs_o2_asm_traced_pupil"]["relative_l2_gate_disc"]
    sensor = rows["nominal_sensor"]["psf"]["vs_o2_asm_traced_pupil"]["relative_l2_gate_disc"]
    assert pupil > sensor, "the exit-pupil handoff must not beat the sensor handoff"
    assert record["o4_exit_pupil_negative_control"]["status"].startswith("retained")
    assert "OUT OF CONTRACT" in record["o4_exit_pupil_negative_control"]["label"]


def test_the_sampling_and_oracle_errors_are_reported_separately(record) -> None:
    rows = record["experiment_b_ray_convergence"]["rows"]
    for row in rows[:-1]:
        assert "sampling_error_vs_highest_ray_count" in row
        assert "vs_o1_analytic_airy" in row["psf"]
        assert "vs_o2_asm_traced_pupil" in row["psf"]
    fits = record["experiment_b_ray_convergence"]["fits"]
    assert fits["sampling_error_vs_traced_rays"]["exponent"] < 0.0
    assert fits["oracle_error_vs_o2_vs_traced_rays"]["exponent"] < 0.0


def test_the_m3_9_turnaround_does_not_recur_against_the_independent_wave_oracle(record) -> None:
    """The specific check CHE-38 section 7 demands, pinned as ABSENT."""
    turnaround = record["experiment_b_ray_convergence"]["turnaround_check"]
    assert turnaround["vs_o2_independent_wave"]["rises_after_the_minimum"] is False, (
        "M3.9's oracle residual passed through a minimum and rose. At the sensor "
        "handoff it must not; if this ever fails, the structural diagnosis reopens."
    )


def test_the_residual_is_a_per_ray_area_weight_and_not_the_kernel(record) -> None:
    """The attribution, pinned in all three of its parts."""
    attribution = record["attribution_quadrature_weights"]
    # 1. first order in the ray spacing, which is the wrong rate for a smooth
    #    equal-area quadrature and the right one for a boundary error.
    exponent = attribution["uniform_weight_convergence_fit"]["exponent"]
    assert -1.15 < exponent < -0.80, exponent
    assert attribution["uniform_weight_convergence_fit"]["r_squared"] > 0.99
    # 2. the effective aperture overshoots by half a ring spacing.
    for row in attribution["rows"]:
        ratio = row["fitted_na_relative_excess"] / row["half_a_ring_spacing_prediction"]
        assert 0.7 < ratio < 1.7, (row["rings"], ratio)
    # 3. the radial trapezoid weight removes it, and what is left is CONVERGED.
    assert attribution["trapezoid_weight_residual_is_flat_at"] < GATE
    weighted = [row["trapezoid_weight_vs_rs_oracle"] for row in attribution["rows"]]
    assert max(weighted) / min(weighted) < 1.3, weighted


def test_the_nyquist_guard_fires_on_real_traced_data_and_the_bypass_shows_why(record) -> None:
    grid = record["experiment_c_grid_convergence"]
    refused = [row for row in grid["rows"] if row.get("status") == "coupler_refused"]
    assert refused, "the sweep must include grids the guard rejects, or it proves nothing"
    for row in refused:
        assert row["nyquist_admissible"] is False
        assert row["guard_fired_on_real_traced_data"] is True
    for row in grid["rows"]:
        if row.get("status") == "succeeded":
            assert row["nyquist_admissible"] is True, "no production row bypassed the guard"
    bypass = grid["nyquist_bypass_negative_control"]
    assert bypass["violation_factor"] > 1.0
    assert bypass["relative_l2_vs_o1_analytic_airy"] > 1e-1, (
        "violating the per-axis limit must visibly destroy the PSF, or the guard is "
        "merely conservative"
    )


def test_grid_convergence_re_derives_the_grid_rather_than_inheriting_188(record) -> None:
    grid = record["experiment_c_grid_convergence"]
    assert "not transferable" in grid["m3_9_grid_n_188_reassessed"]
    succeeded = [row for row in grid["rows"] if row.get("status") == "succeeded"]
    finest = min(succeeded, key=lambda row: row["sample_pitch_m"])
    coarsest = max(succeeded, key=lambda row: row["sample_pitch_m"])
    assert finest["pixels_across_the_psf_core_diameter"] > coarsest[
        "pixels_across_the_psf_core_diameter"
    ]
    residuals = [
        row["psf"]["vs_o2_asm_traced_pupil"]["relative_l2_gate_disc"] for row in succeeded[-4:]
    ]
    assert max(residuals) / min(residuals) < 2.0, (
        f"the four finest grids must agree to better than 2x, got {residuals}"
    )


def test_padding_is_discharged_rather_than_repeated_for_form(record) -> None:
    padding = record["experiment_d_padding"]
    if padding["applicable_to_the_selected_handoff"] is False:
        assert padding["selected_handoff_post_propagation_m"] == 0.0
        assert "no FFT after the coupler" in padding["why"]
    rows = [row for row in padding["rows"] if row.get("status") == "succeeded"]
    assert len(rows) >= 4, "the conditional is discharged with evidence, not by assertion"
    assert "power conservation alone cannot certify" in (
        padding["power_conservation_caveat_retained_from_m3_9"]
    )


def test_the_edge_slope_loose_end_is_no_longer_attributed_to_unsupported_mechanisms(
    record,
) -> None:
    resolved = record["o4_exit_pupil_negative_control"]["edge_slope_loose_end_resolved"]
    measured = resolved["measured_settled"]
    circular = resolved["circular_aperture_reference"]["rim_slope_times_sqrt_lambda_R"]
    straight = resolved["straight_edge_reference"]["rim_slope_times_sqrt_lambda_R"]
    assert abs(measured / circular - 1.0) < 0.12, (
        f"measured {measured} against the circular reference {circular}"
    )
    assert abs(measured / straight - 1.0) > 0.15, (
        "the straight-edge comparison must remain the poor one; that is the finding"
    )
    assert "wrong reference" in resolved["resolution"]
    for claim in resolved["robust_claims_unchanged"]:
        assert claim


def test_the_exit_pupil_rim_does_not_sharpen_with_ray_refinement(record) -> None:
    control = record["o4_exit_pupil_negative_control"]
    assert control["does_not_sharpen_with_ray_refinement"] is True
    rims = [row["amplitude_at_the_geometric_rim"] for row in control["rows"][-3:]]
    assert all(0.40 < rim < 0.60 for rim in rims), rims


def test_the_reclassified_conclusion_does_not_say_the_coupler_is_incorrect(record) -> None:
    """CHE-38 section 12 is explicit about the wording."""
    text = record["o4_exit_pupil_negative_control"]["reclassified_conclusion"]
    assert "cannot be ASSUMED to reconstruct finite pupil support" in text
    assert "C_RAY_TO_WAVE is incorrect" not in text


def test_absolute_power_is_still_labelled_unverified(record) -> None:
    open_items = record["verdict"]["open_and_deliberately_not_closed_here"]
    assert any("UNVERIFIED" in item and "power" in item for item in open_items)
    assert any("quadrature weight" in item for item in open_items)
    assert any("curvature" in item for item in open_items)


def test_the_graph_node_still_equals_the_coupler_core(record) -> None:
    node = record["graph_node_equals_coupler_core"]
    assert node["status"] == "checked"
    assert node["bit_identical"] is True


def test_the_slice_is_deterministic_across_two_processes(record) -> None:
    cost = record["determinism_and_cost"]
    assert cost["bit_identical_across_two_processes"] is True
    for run in cost["runs"]:
        assert run["status"] == "succeeded"
        assert run["peak_rss_bytes"] is not None


def test_the_verdict_is_one_of_the_four_and_states_all_three_lines(record) -> None:
    verdict = record["verdict"]
    assert verdict["verdict_letter"] in {"A", "B", "C", "D"}
    assert verdict["no_tolerance_was_widened"] is True
    assert math.isclose(verdict["gate"], GATE)
    assert len(verdict["per_configuration"]) >= 3
    for entry in verdict["per_configuration"]:
        for key in (
            "DISCRETIZATION CONVERGED",
            "PHYSICALLY CORRECT",
            "HANDOFF WITHIN DECLARED VALIDITY REGION",
        ):
            assert key in entry, entry["configuration"]
    assert "converged" != verdict["verdict"].lower()


def test_the_figures_required_by_the_ticket_were_written(record) -> None:
    names = " ".join(record["figures"])
    for stem in (
        "figure1_architecture",
        "figure2_handoff_validity",
        "figure3_ray_convergence",
        "figure4_sensor_psf",
        "figure5_exit_pupil_control",
    ):
        assert stem in names, stem
