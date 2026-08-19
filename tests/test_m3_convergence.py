"""CHE-38 (M3.9): the convergence study, and the aperture defect it found.

Two kinds of test, deliberately separated, as in ``test_m3_psf_verification.py``.

The live ones run the probe's own machinery on cases whose answers are known
before the run: a fitted exponent must come back off a synthetic power law, and
the reconstruction of a synthetic converging bundle must put **half** the
interior amplitude at the geometric rim of its aperture, which is the mechanism
this ticket found. None of them touch an engine, so they are cheap.

The rest read ``benchmarks/probes/records/m3_convergence.json``. Re-running twelve
ray counts, ten grids and seven paddings inside the suite would cost ten minutes;
the probe is the evidence and these assertions pin what it found -- including the
two gate failures, which stay pinned as failures, and the non-monotone ray trend,
which is pinned as non-monotone so that a later change cannot quietly restore
M3.8's reading of it.
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
import yaml

ROOT = Path(__file__).resolve().parents[1]
RECORD_PATH = ROOT / "benchmarks" / "probes" / "records" / "m3_convergence.json"
PROBE_PATH = ROOT / "benchmarks" / "probes" / "m3_convergence.py"
PROTOCOL_PATH = ROOT / "benchmarks" / "slice_protocol.yaml"

pytestmark = pytest.mark.coupler

WAVELENGTH_M = 0.55e-6
R_M = 4.837461300309598e-3


@pytest.fixture(scope="module")
def record() -> dict[str, Any]:
    if not RECORD_PATH.exists():
        pytest.skip(f"{RECORD_PATH.relative_to(ROOT)} is missing; run the probe first")
    return json.loads(RECORD_PATH.read_text())


@pytest.fixture(scope="module")
def protocol() -> dict[str, Any]:
    return yaml.safe_load(PROTOCOL_PATH.read_text())


@pytest.fixture(scope="module")
def probe():
    spec = importlib.util.spec_from_file_location("m3_convergence_probe", PROBE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Live: the fitting and windowing machinery, on inputs with known answers
# ---------------------------------------------------------------------------


def test_the_power_law_fit_recovers_a_known_exponent(probe) -> None:
    """An exponent quoted without its range is not evidence; check the machinery first."""
    x = [10.0, 100.0, 1000.0, 10000.0]
    y = [3.0 * value**-0.5 for value in x]
    fit = probe._power_law_fit(x, y, label="synthetic")

    assert fit["exponent"] == pytest.approx(-0.5, abs=1e-12)
    assert fit["prefactor"] == pytest.approx(3.0, rel=1e-12)
    assert fit["r_squared"] == pytest.approx(1.0, abs=1e-12)
    assert fit["x_range"] == [pytest.approx(10.0), pytest.approx(10000.0)]
    # Two points can be fitted by anything, so the fit refuses to try.
    assert probe._power_law_fit([1.0, 2.0], [1.0, 2.0], label="two")["status"] == (
        "too_few_points_to_fit"
    )


def test_the_central_crop_keeps_the_pinned_origin(probe) -> None:
    """A half-pixel slip in the crop would show up as a fake padding residual."""
    array = np.zeros((1320, 1320))
    array[660, 662] = 1.0  # index n // 2 on axis 0, two pixels out on axis 1
    cropped = probe._central_crop(array, 188)

    assert cropped.shape == (188, 188)
    assert np.unravel_index(int(cropped.argmax()), cropped.shape) == (94, 96)


def test_the_synthetic_bundle_carries_the_exact_path_to_its_focus(probe) -> None:
    """It is the control for the reconstruction operator, so it must be perfect."""
    from multiscale_optics_agent.evaluation.psf_oracles import pupil_aberration

    bundle = probe._synthetic_converging_bundle(16, radius_m=125.0e-6, distance_m=R_M)
    aberration = pupil_aberration(
        bundle,
        plane_z_m=probe.SINGLET["pupil_z_m"],
        observation_point_m=(0.0, 0.0, probe.SINGLET["pupil_z_m"] + R_M),
        fit_sphere=False,
    )
    assert aberration.rms_waves == pytest.approx(0.0, abs=1e-12)
    assert aberration.pupil_radius_m == pytest.approx(125.0e-6, rel=1e-12)


def test_the_wavelet_sum_puts_half_the_amplitude_at_the_geometric_rim(probe) -> None:
    """The mechanism CHE-38 found, as a live test rather than a recorded number.

    Each ray is an infinite plane wave, so the sum is a stationary-phase estimate
    of the pupil field and resolves it only over a Fresnel zone of the converging
    curvature. A hard aperture therefore comes back as a knife-edge diffraction
    profile: amplitude 1/2 at the geometric rim, with an overshoot fringe inside
    it. Both are asserted, because either alone could be a coincidence.

    Deliberately synthetic and small: no engine, a 180 um aperture inside a 500 um
    window so the soft edge is not clipped, and 7057 rays so the sampling pedestal
    is gone. The aperture cannot be the frozen 249.8 um here, because on the frozen
    188^2 window the rim sits exactly at the window edge and the profile is cut off
    at 0.52 -- which is the same clipping that made the pupil-edge diagnostic
    impossible on the frozen grid.
    """
    from multiscale_optics_agent.couplers.ray_to_wave import ray_to_wave

    radius = 180.0e-6
    bundle = probe._synthetic_converging_bundle(48, radius_m=radius, distance_m=R_M)
    field, _ = ray_to_wave(
        bundle,
        grid_shape=(188, 188),
        sample_pitch_m=(probe.SINGLET["pitch_m"], probe.SINGLET["pitch_m"]),
    )
    profile = probe._edge_profile(
        np.asarray(field.u), pitch_m=probe.SINGLET["pitch_m"], radius_m=radius
    )

    assert profile["amplitude_at_the_geometric_rim"] == pytest.approx(0.5, abs=0.06)
    assert profile["overshoot_inside_the_rim"] > 1.05
    # And it is a Fresnel scale, not a sampling scale: the transition is an order of
    # magnitude wider than the 2.6 um ray spacing this bundle has, and its slope is
    # within 30% of what the Fresnel integrals predict for a knife edge.
    assert profile["rim_slope_per_m"] is not None
    assert 1.0 / profile["rim_slope_per_m"] > 10.0 * (radius / 48)
    predicted = probe._knife_edge_rim_slope()["predicted_rim_slope_times_sqrt_lambda_R"]
    assert profile["rim_slope_times_edge_scale"] / predicted == pytest.approx(0.75, abs=0.25)


def test_the_reconstruction_is_scale_free_after_peak_normalization(probe) -> None:
    """Why a convergence study is possible at all despite CHE-33's missing weight."""
    from multiscale_optics_agent.couplers.contracts import RayBundle
    from multiscale_optics_agent.couplers.ray_to_wave import ray_to_wave

    bundle = probe._synthetic_converging_bundle(16, radius_m=125.0e-6, distance_m=R_M)
    scaled = RayBundle(
        positions_m=bundle.positions_m,
        directions=bundle.directions,
        wavelength_m=bundle.wavelength_m,
        reference_plane=bundle.reference_plane,
        frame=bundle.frame,
        amplitude=np.asarray(bundle.amplitude) * 4.0,
        optical_path_length_m=bundle.optical_path_length_m,
        optical_path_length_reference=bundle.optical_path_length_reference,
    )
    shape, pitch = (94, 94), (probe.SINGLET["pitch_m"], probe.SINGLET["pitch_m"])
    plain, _ = ray_to_wave(bundle, grid_shape=shape, sample_pitch_m=pitch)
    rescaled, _ = ray_to_wave(scaled, grid_shape=shape, sample_pitch_m=pitch)

    a = np.abs(np.asarray(plain.u)) ** 2
    b = np.abs(np.asarray(rescaled.u)) ** 2
    assert np.array_equal(a / a.max(), b / b.max())
    # The absolute scale, on the other hand, moves by exactly the square.
    assert b.max() / a.max() == pytest.approx(16.0, rel=1e-12)


def test_the_nyquist_precondition_brackets_its_own_limit(probe) -> None:
    """The grid condition is exact, so it can be bracketed rather than sampled."""
    from multiscale_optics_agent.couplers.contracts import ContractCode, ContractError
    from multiscale_optics_agent.couplers.ray_to_wave import (
        grid_nyquist_direction_limit,
        ray_to_wave,
    )

    bundle = probe._synthetic_converging_bundle(6, radius_m=125.0e-6, distance_m=R_M)
    largest = float(np.max(np.abs(np.asarray(bundle.directions)[:, :2])))
    critical = WAVELENGTH_M / (2.0 * largest)
    assert grid_nyquist_direction_limit(WAVELENGTH_M, critical) == pytest.approx(largest, rel=1e-12)

    field, diagnostics = ray_to_wave(
        bundle, grid_shape=(32, 32), sample_pitch_m=(critical, critical)
    )
    assert diagnostics.grid_nyquist_satisfied is True
    assert field.u.shape == (32, 32)

    with pytest.raises(ContractError) as refused:
        ray_to_wave(
            bundle,
            grid_shape=(32, 32),
            sample_pitch_m=(critical * 1.0001, critical * 1.0001),
        )
    assert refused.value.code is ContractCode.SHAPE_MISMATCH
    assert "more rays will not help" in str(refused.value).lower()


# ---------------------------------------------------------------------------
# The recorded evidence
# ---------------------------------------------------------------------------


def test_the_probe_ran_the_frozen_on_axis_configuration(record: dict) -> None:
    """Every geometric number is read back from the protocol, not restated."""
    frozen = record["frozen_configuration"]
    assert frozen["every_value_matches"] is True
    assert "CHE-41" in frozen["on_axis_only"]

    # The frozen pupil_extent_m is the MEASURED pupil diameter, and the window is
    # that rounded up to a whole pixel -- 0.055% larger, not equal. Asserting
    # equality was this probe's own first mistake, so the relation is pinned as the
    # ceiling relation it actually is.
    window = frozen["window_versus_pupil"]
    assert window["grid_n_is_the_ceiling"] is True
    assert window["window_over_pupil"] == pytest.approx(1.00055, abs=1e-4)
    assert window["pupil_in_pixels"] == pytest.approx(187.897, abs=0.01)


def test_the_ray_trend_is_not_monotone_and_rises_after_its_minimum(record: dict) -> None:
    """The finding that changes M3.8's conclusion. Pinned so it cannot be undone.

    M3.8 read its six-point sweep as flattening at 6.7e-3 and attributed the whole
    excess to ray sampling. Extended to 787969 rays the residual turns around, so
    the attribution was to a term that stops helping.
    """
    sweep = record["ray_count_convergence"]
    rows = [row for row in sweep["rows"] if row.get("relative_l2_vs_fft_oracle")]

    assert sweep["monotonically_falling_against_the_oracle"] is False
    assert rows[0]["traced_rays"] < rows[-1]["traced_rays"]
    assert len(rows) >= 10

    residuals = [row["relative_l2_vs_fft_oracle"] for row in rows]
    minimum_index = int(np.argmin(residuals))
    assert 0 < minimum_index < len(residuals) - 1, "the minimum must be interior"
    assert sweep["rises_after_the_minimum_by"] > 1.15
    # And it never reaches the gate anywhere in the sweep.
    assert min(residuals) > 1.0e-3

    # The second oracle turns around too, and it is the textbook answer for a
    # diffraction-limited circular pupil -- so refinement moves the slice AWAY from
    # the accepted answer. This is what rules out "the oracle is the wrong one".
    airy = [row["relative_l2_vs_analytic_airy"] for row in rows]
    airy_minimum = int(np.argmin(airy))
    assert 0 < airy_minimum < len(airy) - 1
    assert airy[-1] > 1.3 * airy[airy_minimum]


def test_the_ray_sampling_term_is_measured_against_a_ray_refined_reference(
    record: dict,
) -> None:
    """The term the protocol left null, separated from the term no rays can fix."""
    sweep = record["ray_count_convergence"]
    fit = sweep["fits"]["ray_sampling_error_vs_traced_rays"]

    assert fit["points"] >= 5
    assert fit["exponent"] < 0.0, "ray refinement must reduce the ray-sampling term"
    assert fit["r_squared"] > 0.9
    assert fit["x_range"][1] > 1.0e5
    # It is measured against the refined PSF, not against the oracle, which is
    # what makes it separable from the aperture term.
    assert "787969" in fit["label"] or "PSF" in fit["label"]

    ladder = [
        row["relative_l2_vs_ray_refined_psf"]
        for row in sweep["rows"]
        if row.get("relative_l2_vs_ray_refined_psf") is not None
    ]
    assert ladder == sorted(ladder, reverse=True), "must fall monotonically"


def test_the_frozen_ray_count_criterion_is_refuted(record: dict, protocol: dict) -> None:
    """Confirming or refuting it is a result either way. It is refuted, twice."""
    criterion = record["ray_count_convergence"]["protocol_ray_count_criterion"]

    assert criterion["frozen_starting_value"] == 4096
    assert criterion["rule_evaluated_at_the_frozen_grid"] == pytest.approx(8836.0, rel=1e-9)
    assert criterion["rule_and_starting_value_disagree"] is True
    assert "REFUTED" in criterion["verdict"]
    # The protocol now records the same result beside the frozen numbers.
    frozen = protocol["sampling"]["ray_count_criterion"]
    assert frozen["starting_value"] == 4096, "the frozen value is recorded, not edited"
    assert "CHE-38" in json.dumps(frozen)


def test_the_aperture_edge_hypothesis_is_confirmed_and_is_not_a_ray_spacing(
    record: dict,
) -> None:
    """M3.8's leading hypothesis: right about the cause, wrong about the scale."""
    study = record["aperture_edge_hypothesis"]
    edges = study["edge_vs_ray_count"]
    settled = [row for row in edges if row["traced_rays"] >= 12000]

    assert "CONFIRMED" in study["verdict"]
    assert len(settled) >= 2
    for row in settled:
        # The knife-edge value. This is the fingerprint, not a fitted parameter.
        assert row["amplitude_at_the_geometric_rim"] == pytest.approx(0.5, abs=0.03)
        assert row["overshoot_inside_the_rim"] > 1.05
        # And it is tens of ray spacings wide, not one -- and grows with refinement,
        # because the transition is fixed and the spacing is what shrinks.
        assert row["rim_transition_in_ray_spacings"] > 10.0
    spacings = [row["rim_transition_in_ray_spacings"] for row in settled]
    assert spacings == sorted(spacings)
    assert study["rim_slope_is_ray_count_independent_to_10_percent"] is True

    # The scale is a Fresnel scale: the rim slope tracks 1 / sqrt(lambda R) as R
    # varies at fixed aperture, fixed grid and fixed ray count.
    fit = study["rim_slope_vs_sqrt_lambda_R_fit"]
    assert fit["points"] >= 5
    assert fit["exponent"] == pytest.approx(-1.0, abs=0.15)
    assert fit["r_squared"] > 0.98
    assert study["sqrt_lambda_R_in_pixels"] == pytest.approx(19.4, abs=0.1)
    # The scaling law holds; the 1-D idealization's CONSTANT does not, and the gap
    # is reported rather than absorbed.
    ratio = study["measured_over_predicted_rim_slope"]
    assert ratio["mean"] == pytest.approx(0.74, abs=0.05)
    assert ratio["spread"] < 1.10
    assert "reported rather than absorbed" in study["where_the_idealization_stops"]


def test_the_error_decomposes_at_the_pupil_plane(record: dict) -> None:
    """Neither the propagation nor the oracle is implicated. Checked, not asserted."""
    decomposition = record["error_decomposition"]
    floor = decomposition["non_ray_floor"]["value"]

    # With the ray leg removed entirely, the same ASM, padding, window,
    # measurement and comparison agree with the oracle 5x better.
    assert floor < 0.2 * decomposition["shipping_vs_oracle"]
    # ... and yet the floor alone is already outside the gate.
    assert floor > 1.0e-3

    # Comparing the shipping PSF against the analytic hard-mask pupil reproduces
    # its disagreement with the oracle, so the discrepancy is created upstream of
    # the propagation.
    assert decomposition["shipping_vs_continuous_hard_mask"] == pytest.approx(
        decomposition["shipping_vs_oracle"], rel=0.25
    )

    # The synthetic perfect bundle reproduces it too, which exonerates Optiland,
    # the OPL declaration and the residual aberration.
    synthetic = decomposition["synthetic_perfect_bundle_vs_continuous_hard_mask"]
    assert synthetic, "the synthetic isolation must have run"
    for entry in synthetic.values():
        assert entry["vs_continuous_hard_mask"] > 3.0e-3

    # The second oracle, which shares nothing with the first, points the same way.
    airy = decomposition["relative_l2_vs_the_analytic_airy_oracle"]
    assert airy["continuous_hard_mask"] < airy["shipping"]
    assert airy["continuous_hard_mask"] < airy["continuous_fresnel_edge"]

    # A control that failed, kept as a bound on the claim.
    assert "did NOT work" in decomposition["amplitude_only_fresnel_control_failed"]


def test_the_grid_precondition_fires_exactly_where_nyquist_binds(record: dict) -> None:
    """M3.5's refusal on real traced data, at the pitch the protocol predicts."""
    grid = record["grid_convergence"]

    assert grid["measured_agrees_with_frozen"] is True
    assert grid["max_per_axis_direction_cosine"] == pytest.approx(0.05171631827291936, rel=1e-12)
    assert grid["predicted_smallest_admissible_grid_n"] == 94
    assert grid["refusal_contract_codes"] == ["SHAPE_MISMATCH"]
    assert grid["precondition_fires_exactly_on_the_inadmissible_grids"] is True
    assert grid["largest_refused_grid_n"] == 93
    assert grid["smallest_admitted_grid_n"] == 94
    # The refusal is a FAILED result with a remedy, not an exception and not a
    # silently coarsened field.
    refused = [row for row in grid["rows"] if row["status"] == "coupler_refused"]
    assert refused
    for row in refused:
        assert "more rays will not help" in row["error"].lower()


def test_the_nyquist_guard_is_guarding_something_real(record: dict) -> None:
    """A precondition that fires proves something about the precondition.

    Whether it is worth having is a separate question, and the answer is measured:
    the same rays reconstructed with ``enforce_grid_nyquist=False`` at an
    inadmissible pitch alias, and the PSF is wrong by several times what the same
    configuration one pixel above the limit gives.
    """
    bypass = record["grid_convergence"]["bypassing_the_precondition"]

    assert bypass["still_reports_the_violation"] is True
    assert bypass["worst_violating_over_admissible"] > 3.0
    assert bypass["airy_peak_deficit_worst_over_admissible"] > 10.0
    assert "not monotone" in bypass["verdict"].lower()
    # The graph node has no such switch, which is the right default.
    assert "cannot be bypassed" in bypass["how"]

    violating = [row for row in bypass["rows"] if row.get("nyquist_admissible") is False]
    assert len(violating) >= 3
    for row in violating:
        assert row["coupler_reported_grid_nyquist_satisfied"] is False


def test_padding_shows_wraparound_in_the_wings_first(record: dict) -> None:
    """Where it shows up first is the point; the core stays plausible."""
    padding = record["padding_convergence"]
    rows = {row["pad_width"]: row for row in padding["rows"]}

    unpadded = rows[0]
    assert (
        unpadded["wing_relative_l2_3_to_5_airy_radii"]
        > 10.0 * unpadded["core_relative_l2_within_1_airy_radius"]
    )
    assert (
        unpadded["far_wing_relative_l2_5_to_30_airy_radii"]
        > 2.0 * unpadded["wing_relative_l2_3_to_5_airy_radii"]
    )
    # And the adapter's power ratio reads ~1 there, which is CHE-35's trap: an
    # unpadded run recirculates the light that should have left the window.
    assert unpadded["propagated_window_power_ratio"] == pytest.approx(1.0, abs=1e-5)
    # Monotone improvement in the wings with padding.
    wings = [
        rows[pad]["wing_relative_l2_3_to_5_airy_radii"]
        for pad in sorted(rows)
        if rows[pad].get("wing_relative_l2_3_to_5_airy_radii") is not None
    ]
    assert wings == sorted(wings, reverse=True)
    # CHE-35's demotion of edge_energy_fraction holds on real reconstructed data,
    # and more sharply: from pad 0 to pad 47 the indicator improves by more than the
    # error does, so it claims progress the PSF has not made.
    edge_ratio = unpadded["output_edge_energy_fraction"] / rows[47]["output_edge_energy_fraction"]
    error_ratio = (
        unpadded["wing_relative_l2_3_to_5_airy_radii"]
        / rows[47]["wing_relative_l2_3_to_5_airy_radii"]
    )
    assert edge_ratio > 2.0 > error_ratio
    assert "must not be used to certify padding" in padding["edge_energy_indicator_retested"]
    # The frozen pad_width is not relaxed even though a smaller one meets this
    # study's own threshold.
    assert "566 is retained" in padding["frozen_pad_width_verdict"]


def test_the_stated_configuration_is_converged_and_says_it_is_not_correct(
    record: dict,
) -> None:
    """Both halves matter. A converged configuration alone would mislead."""
    converged = record["converged_configuration"]

    assert converged["refinement_threshold"] == 1.0e-3
    assert "tightest gate" in converged["threshold_provenance"]
    assert converged["stated_configuration"]["grid_n"] == 188
    assert converged["stated_configuration"]["pad_width"] == 566
    assert converged["stated_configuration"]["traced_rays"] >= 100000
    assert converged["converged_ray_count"]["threshold_met"] is True
    ladder = [
        row["relative_l2_vs_ray_refined_psf"] for row in converged["converged_ray_count"]["ladder"]
    ]
    assert ladder == sorted(ladder, reverse=True)
    assert "converged is not correct" in converged["the_warning_that_goes_with_it"]
    assert converged["residual_at_the_stated_configuration"] > 1.0e-3


def test_the_configuration_is_bit_identical_across_two_processes(record: dict) -> None:
    """No RNG anywhere in the slice, so this is pass/fail and not a tolerance."""
    determinism = record["determinism_and_cost"]
    runs = determinism["runs"]

    assert determinism["bit_identical_across_two_processes"] is True
    assert len(runs) == 2
    assert runs[0]["psf_sha256"] == runs[1]["psf_sha256"]
    # The retained raw scale too, not only the normalized array: peak
    # normalization would hide a constant factor, which is exactly M3.7's point.
    assert runs[0]["raw_peak_intensity"] == runs[1]["raw_peak_intensity"]
    assert runs[0]["raw_window_energy"] == runs[1]["raw_window_energy"]
    for run in runs:
        assert run["status"] == "succeeded"
        assert run["peak_rss_bytes"] > 0
        assert run["seconds_total"] > 0.0
        assert run["handshake"] == "exec_complete"
        # Peak RSS is sampled from OUTSIDE the child, because ru_maxrss inside it is
        # the parent's high-water mark. The child's self-reported figure is kept so
        # the difference is visible rather than trusted.
        assert "VmRSS" in run["peak_rss_method"]
        assert run["peak_rss_bytes"] < run["self_reported_ru_maxrss_bytes"]

    check = determinism["rusage_inheritance_check"]
    assert check["child_inherits_the_parent_high_water_mark"] is True
    assert check["check_is_non_vacuous"] is True


def test_both_intensity_gates_stay_pinned_as_failures(record: dict, protocol: dict) -> None:
    """A gate failure is a finding. It must not be able to go green quietly.

    And the gates themselves are read back from the protocol, so widening one to
    make this pass fails here instead.
    """
    gates = record["gates"]
    frozen = protocol["tolerance_budget"]["gates"]

    assert gates["no_tolerance_was_widened"] is True
    assert protocol["tolerance_budget"]["no_gate_may_be_satisfied_by_widening"] is True

    fft = gates["fft_oracle_intensity_relative_l2"]
    assert fft["gate"] == float(frozen["fft_oracle_intensity_relative_l2"]["value"]) == 1.0e-3
    assert fft["verdict"] == "FAIL"
    assert fft["closes_with_more_rays"] is False
    assert fft["at_the_highest_ray_count"] > fft["gate"]
    assert "cannot close by ray refinement" in fft["diagnosis"]

    airy = gates["airy_peak_intensity_relative"]
    assert airy["gate"] == float(frozen["airy_peak_intensity_relative"]["value"]) == 2.0e-3
    assert airy["verdict"] == "FAIL"
    assert abs(airy["at_the_highest_ray_count"]) > airy["gate"]


def test_the_regression_envelope_is_declared_for_numbers_and_not_for_seconds(
    record: dict,
) -> None:
    """Either fix M2's L6 for the slice or repeat the disclaimer deliberately."""
    envelope = record["regression_envelope"]

    assert envelope["accuracy_envelope"]["declared"] is True
    assert envelope["accuracy_envelope"]["enforced_by"] == "tests/test_m3_convergence.py"
    bounds = envelope["accuracy_envelope"]["bounds"]
    assert bounds["ray_sweep_is_non_monotone"] is True
    assert bounds["psf_is_bit_identical_across_processes"] is True
    assert bounds["amplitude_at_the_geometric_rim"]["value"] == 0.5

    assert envelope["timing_envelope"]["declared"] is False
    assert "shared" in envelope["timing_envelope"]["why_not"]
    assert len(envelope["timing_envelope"]["observed_seconds"]) == 2


def test_the_declared_accuracy_envelope_still_holds(record: dict) -> None:
    """The envelope, applied to this record. This is the regression test itself."""
    bounds = record["regression_envelope"]["accuracy_envelope"]["bounds"]
    frozen_row = next(
        row for row in record["ray_count_convergence"]["rows"] if row.get("rings") == 32
    )

    assert frozen_row["relative_l2_vs_fft_oracle"] == pytest.approx(
        bounds["relative_l2_vs_fft_oracle_at_the_frozen_configuration"]["value"], rel=0.02
    )
    assert record["error_decomposition"]["non_ray_floor"]["value"] == pytest.approx(
        bounds["non_ray_floor"]["value"], rel=0.05
    )
    # M3.8 measured 1.509e-2 at this configuration with its own probe and its own
    # oracle settings. Reproducing it here is a cross-check between two probes.
    assert frozen_row["relative_l2_vs_fft_oracle"] == pytest.approx(1.509e-2, rel=0.02)


def test_the_window_power_fraction_che35_left_untrended_is_now_trended(
    record: dict,
) -> None:
    """CHE-35's 0.63 was a ray-sampling artefact, and the trend says so."""
    rows = [
        row
        for row in record["ray_count_convergence"]["rows"]
        if row.get("power_fraction_in_the_frozen_observation_window") is not None
    ]
    fractions = [row["power_fraction_in_the_frozen_observation_window"] for row in rows]

    assert len(fractions) >= 10
    assert fractions[0] < 0.65, "at the lowest ray counts much of the power is pedestal"
    assert min(fractions) < 0.55
    assert fractions[-1] > 0.99, "it recovers under ray refinement"
    assert fractions[-1] > fractions[0]

    within = [row["fraction_of_window_intensity_within_3_airy_radii"] for row in rows]
    assert within[-1] > 10.0 * within[0]


def test_the_oversampling_factor_is_re_justified_with_the_radius_counted_right(
    record: dict, protocol: dict
) -> None:
    """The sampling half of CHE-33's diameter/radius defect, which this ticket owns."""
    review = record["oversampling_factor_review"]

    assert review["frozen_factor"] == 2
    assert review["pixels_per_airy_radius_at_the_critical_pitch"] == pytest.approx(1.22, abs=0.01)
    assert review["pixels_per_airy_radius_at_the_frozen_grid"] == pytest.approx(2.44, abs=0.01)
    assert "DIAMETER counted as a radius" in review["the_defect"]

    item = next(
        entry
        for entry in protocol["open_structural_items"]
        if entry["id"] == "airy_radius_entry_is_a_diameter"
    )
    assert item["sampling_half_status"].startswith("resolved")
    assert "CHE-38" in item["sampling_half_resolved_by"]


def test_the_protocol_records_the_converged_configuration_and_its_warning(
    protocol: dict,
) -> None:
    """The write-up has to carry both halves, and no placeholder may survive it."""
    block = protocol["m3_9_convergence"]
    converged = block["converged_configuration"]

    assert converged["traced_rays"] == 394219
    assert converged["grid_n"] == 188
    assert converged["pad_width"] == 566
    assert converged["residual_against_the_fft_oracle_there"] > converged["gate"]
    assert "converged is not correct" in converged["the_warning_that_goes_with_it"]
    assert converged["cost"], "runtime and peak RSS are an acceptance criterion"
    assert "VmRSS" in block["peak_rss_measurement_note"]["consequence_here"]
    assert "not amended by an M3 ticket" in block["peak_rss_measurement_note"]["not_changed_here"]

    assert block["grid"]["smallest_admissible_grid_n"] == 94
    assert "SHAPE_MISMATCH" in block["grid"]["refused"]
    assert block["determinism"]["result"] == "bit-identical"
    assert block["regression_envelope"]["accuracy"] == "declared"
    assert block["regression_envelope"]["timing"].startswith("NOT declared")
    assert "FAIL" in block["gates"]["fft_oracle_intensity_relative_l2"]
    assert "FAIL" in block["gates"]["airy_peak_intensity_relative"]

    # A placeholder left in the protocol would be a fabricated number waiting to be
    # read as a measured one.
    assert "TO_BE_FILLED" not in PROTOCOL_PATH.read_text()


def test_the_protocol_records_the_measured_budget_terms(protocol: dict) -> None:
    """``ray_sampling_error`` was null with owner CHE-38. It is measured now.

    And the term that actually dominates is entered beside it, because a budget
    that omits its largest named error source is not a budget.
    """
    terms = {term["name"]: term for term in protocol["tolerance_budget"]["terms"]}

    ray = terms["ray_sampling_error"]
    assert ray["value"] == 1.0e-3
    assert ray["status"] == "measured"
    assert ray["measured_by"] == "CHE-38 (M3.9)"
    # It is measured against the ray-refined PSF, not against an oracle, which is
    # what makes it separable from the aperture term.
    assert "ray-refined" in json.dumps(ray).lower()
    assert len(ray["sweep"]) >= 10

    aperture = terms["wavelet_aperture_edge_error"]
    assert aperture["value"] > float(
        protocol["tolerance_budget"]["gates"]["fft_oracle_intensity_relative_l2"]["value"]
    )
    assert aperture["status"] == "measured"
    assert aperture["refines_away_with_ray_count"] is False
    assert aperture["refines_away_with_grid"] is False
    assert aperture["refines_away_with_padding"] is False
    # Entering a measured error source larger than the gate is not widening the
    # gate, and the term has to say so where a reader will see it.
    assert "no gate moved" in aperture["this_is_not_a_tolerance_widening"]


def test_the_superseded_m3_8_reading_is_recorded_as_superseded(protocol: dict) -> None:
    """M3.8's conclusion changed, so the change is in the amendment list."""
    amendments = {entry["id"]: entry for entry in protocol["amendments"]}
    amendment = amendments["A3-ray-refinement-does-not-close-the-oracle-gate"]

    assert "CHE-38" in amendment["issue"]
    assert "flatten" in amendment["supersedes"].lower()
    assert amendment["probe"] == "benchmarks/probes/m3_convergence.py"
    assert amendment["evidence"] == "benchmarks/probes/records/m3_convergence.json"

    item = next(
        entry
        for entry in protocol["open_structural_items"]
        if entry["id"] == "fft_oracle_gate_fails_with_an_unattributed_floor"
    )
    assert item["status"] == "attributed by CHE-38 (M3.9); the gate still fails"


def test_the_new_aperture_item_is_handed_on_with_its_scale_law(protocol: dict) -> None:
    """The follow-up this ticket must not do itself."""
    item = next(
        entry
        for entry in protocol["open_structural_items"]
        if entry["id"] == "wavelet_sum_cannot_represent_a_hard_aperture_edge"
    )
    assert item["found_by"] == "CHE-38 (M3.9)"
    assert item["severity"].startswith("high")
    assert "sqrt(lambda R)" in json.dumps(item)
    assert "fresnel_number" in json.dumps(item).lower()
    assert item["not_fixed_here_because"]


def test_the_probe_declares_the_ticket_non_goals(record: dict) -> None:
    """No wavelength sweep, no GPU, no optimization loop."""
    joined = " ".join(record["non_goals"]).lower()
    assert "wavelength sweep" in joined
    assert "gpu" in joined
    assert "optimization loop" in joined
    assert record["device"] == "cpu"
    assert math.isclose(record["wavelength_m"], WAVELENGTH_M)
