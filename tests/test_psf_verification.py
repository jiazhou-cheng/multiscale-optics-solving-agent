"""CHE-37 (M3.8): the slice verified against independent oracles.

Two kinds of test here, deliberately separated.

The fast ones exercise the oracle code live on synthetic inputs whose answers are
known in closed form -- a perfect converging pupil must produce an Airy pattern, a
known reference sphere must be recovered, a known injected tilt must move the PSF
a predicted distance. An oracle that has not been checked against something it
cannot get wrong is not an oracle.

The rest read ``benchmarks/probes/records/m3_psf_verification.json``, recorded by
``benchmarks/probes/m3_psf_verification.py`` against the real engines. Re-running
the full slice at six ray counts inside the test suite would cost minutes; the
probe is the evidence and these assertions pin what it found -- including the two
gate failures, which are pinned as failures so that a later change cannot quietly
turn them green.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from multiscale_optics_agent.couplers.contracts import (
    ContractCode,
    ContractError,
    Frame,
    RayBundle,
    ReferencePlane,
)
from multiscale_optics_agent.evaluation.psf_oracles import (
    AIRY_FIRST_NULL_COEFFICIENT_EXACT,
    AIRY_FIRST_NULL_COEFFICIENT_ROUNDED,
    AIRY_J1_FIRST_ZERO,
    airy_first_null_radius_m,
    airy_psf_on_grid,
    azimuthal_profile,
    first_null_comparison,
    fit_reference_sphere,
    fraunhofer_psf,
    measure_first_null_radius_m,
    pupil_aberration,
    radial_profile,
    resample_to_grid,
)

pytestmark = pytest.mark.coupler

ROOT = Path(__file__).resolve().parents[1]
RECORD_PATH = ROOT / "benchmarks" / "probes" / "records" / "m3_psf_verification.json"

WAVELENGTH_M = 0.55e-6
R_M = 4.837461300309598e-3
PUPIL_RADIUS_M = 0.4987073505473812e-3 / 2
PITCH_M = 2.6587352810843895e-06
GRID_N = 188
NA_PARAXIAL = PUPIL_RADIUS_M / R_M


@pytest.fixture(scope="module")
def record() -> dict:
    if not RECORD_PATH.exists():
        pytest.skip(f"{RECORD_PATH.relative_to(ROOT)} is missing; run the probe first")
    return json.loads(RECORD_PATH.read_text())


def _perfect_bundle(n: int = 6000) -> RayBundle:
    """Rays with the exact optical path to a focus: zero wavefront error."""
    rng = np.random.default_rng(7)
    rho = PUPIL_RADIUS_M * np.sqrt(rng.uniform(0.0, 1.0, n))
    theta = rng.uniform(0.0, 2 * np.pi, n)
    x, y = rho * np.cos(theta), rho * np.sin(theta)
    path = np.sqrt(x**2 + y**2 + R_M**2)
    return RayBundle(
        positions_m=np.stack([x, y, np.zeros(n)], axis=1),
        directions=np.stack([-x / path, -y / path, np.full(n, R_M) / path], axis=1),
        wavelength_m=WAVELENGTH_M,
        reference_plane=ReferencePlane(name="exit_pupil", z_m=0.0),
        frame=Frame(),
        amplitude=np.ones(n),
        optical_path_length_m=-path,
        optical_path_length_reference="synthetic: exact path to the nominal focus",
    )


# ---------------------------------------------------------------------------
# The Airy radius, and the protocol defect
# ---------------------------------------------------------------------------


def test_the_airy_coefficient_is_a_radius_not_a_diameter() -> None:
    """0.60983, from the first zero of J1. Not 1.22, which is the diameter."""
    assert pytest.approx(3.8317059702075125, rel=1e-15) == AIRY_J1_FIRST_ZERO
    assert pytest.approx(
        AIRY_J1_FIRST_ZERO / (2.0 * math.pi), rel=1e-15
    ) == AIRY_FIRST_NULL_COEFFICIENT_EXACT
    assert pytest.approx(0.6098349456, rel=1e-9) == AIRY_FIRST_NULL_COEFFICIENT_EXACT
    # The rounded value the protocol quotes is 0.03% high, which is recorded rather
    # than absorbed: the oracle uses the exact one.
    assert (
        pytest.approx(1.00027, rel=1e-3)
    ) == AIRY_FIRST_NULL_COEFFICIENT_ROUNDED / AIRY_FIRST_NULL_COEFFICIENT_EXACT


def test_the_frozen_protocol_airy_radius_is_actually_the_diameter() -> None:
    """The defect this ticket owns, asserted against the protocol file itself.

    ``open_structural_items.airy_radius_entry_is_a_diameter`` records it; this makes
    it executable, so the day someone corrects the field this test tells them the
    oracle no longer needs to route around it.
    """
    import yaml

    protocol = yaml.safe_load((ROOT / "benchmarks" / "slice_protocol.yaml").read_text())
    singlet = next(s for s in protocol["systems"] if s["id"] == "M3-SINGLET-REF")
    frozen_um = float(singlet["airy_radius_um"])
    na = float(singlet["derived"]["numerical_aperture"])

    true_radius_um = airy_first_null_radius_m(WAVELENGTH_M, na) * 1e6
    assert frozen_um / true_radius_um == pytest.approx(2.0, rel=1e-3)
    assert true_radius_um == pytest.approx(6.4856, rel=1e-3)

    frozen_pixels = float(protocol["sampling"]["grids"]["M3-SINGLET-REF"]["airy_radius_in_pixels"])
    true_pixels = airy_first_null_radius_m(WAVELENGTH_M, na) / PITCH_M
    assert frozen_pixels / true_pixels == pytest.approx(2.0, rel=1e-2)
    assert true_pixels == pytest.approx(2.44, abs=0.01)


# ---------------------------------------------------------------------------
# The oracles, checked against cases they cannot get wrong
# ---------------------------------------------------------------------------


def test_the_fft_oracle_reproduces_the_analytic_airy_pattern() -> None:
    """W = 0 on a circular aperture has a closed-form answer, so this is checkable."""
    aberration = pupil_aberration(
        _perfect_bundle(), plane_z_m=0.0, observation_point_m=(0.0, 0.0, R_M)
    )
    assert aberration.rms_waves == pytest.approx(0.0, abs=1e-12)

    oracle = fraunhofer_psf(
        aberration,
        pupil_pitch_m=PITCH_M,
        pupil_grid_n=GRID_N,
        fft_grid_n=8 * GRID_N,
        distance_m=R_M,
    )
    psf = oracle.intensity / oracle.intensity.max()
    analytic = airy_psf_on_grid(
        shape=psf.shape,
        sample_pitch_m=oracle.sample_pitch_m,
        wavelength_m=WAVELENGTH_M,
        numerical_aperture=NA_PARAXIAL,
    )
    predicted = airy_first_null_radius_m(WAVELENGTH_M, NA_PARAXIAL)
    yy = (np.arange(psf.shape[0]) - psf.shape[0] // 2) * oracle.sample_pitch_m[0]
    xx = (np.arange(psf.shape[1]) - psf.shape[1] // 2) * oracle.sample_pitch_m[1]
    core = np.hypot(yy[:, None], xx[None, :]) <= 5.0 * predicted

    relative = np.linalg.norm((psf - analytic)[core]) / np.linalg.norm(analytic[core])
    # ~5.9e-4, set by the pixelated aperture edge on a 93.8-pixel-radius pupil.
    assert relative < 2.0e-3, relative
    assert np.unravel_index(int(psf.argmax()), psf.shape) == (
        psf.shape[0] // 2,
        psf.shape[1] // 2,
    )


def test_the_first_null_estimator_is_biased_at_frozen_sampling() -> None:
    """The bias is the reason the comparison is done in ratio, not absolutely.

    At the frozen 2.66 um pitch there are 2.44 pixels per Airy radius and the
    estimator reads an EXACTLY KNOWN Airy pattern ~11% high. A test that compared a
    measured null against 0.61 lambda/NA at this sampling would be measuring the
    estimator.
    """
    analytic = airy_psf_on_grid(
        shape=(512, 512),
        sample_pitch_m=(PITCH_M, PITCH_M),
        wavelength_m=WAVELENGTH_M,
        numerical_aperture=NA_PARAXIAL,
    )
    comparison = first_null_comparison(
        analytic,
        sample_pitch_m=(PITCH_M, PITCH_M),
        wavelength_m=WAVELENGTH_M,
        numerical_aperture=NA_PARAXIAL,
    )
    assert comparison["pixels_per_airy_radius"] == pytest.approx(2.44, abs=0.01)
    assert comparison["analytic_estimator_bias"] > 0.08
    # And the bias cancels in the ratio, which is the point.
    assert comparison["ratio_measured_over_analytic"] == pytest.approx(1.0, abs=1e-9)


def test_binned_radial_profile_is_worse_than_interpolated_for_a_null() -> None:
    """Why ``azimuthal_profile`` exists: binning at the pixel pitch smears the null."""
    analytic = airy_psf_on_grid(
        shape=(512, 512),
        sample_pitch_m=(PITCH_M / 4, PITCH_M / 4),
        wavelength_m=WAVELENGTH_M,
        numerical_aperture=NA_PARAXIAL,
    )
    predicted = airy_first_null_radius_m(WAVELENGTH_M, NA_PARAXIAL)

    binned_r, binned_p, _ = radial_profile(analytic, sample_pitch_m=(PITCH_M / 4, PITCH_M / 4))
    binned = measure_first_null_radius_m(binned_r, binned_p)
    fine_r, fine_p = azimuthal_profile(
        analytic,
        sample_pitch_m=(PITCH_M / 4, PITCH_M / 4),
        max_radius_m=3.0 * predicted,
        radial_samples=1200,
    )
    interpolated = measure_first_null_radius_m(fine_r, fine_p)

    assert binned is not None and interpolated is not None
    assert abs(interpolated / predicted - 1.0) < abs(binned / predicted - 1.0)


def test_the_reference_sphere_fit_recovers_a_known_centre() -> None:
    """Solve for the centre, do not subtract a ramp.

    The bundle is built to converge at a deliberately off-axis, off-plane point. A
    linear ramp removal cannot find it; the Gauss-Newton fit must.
    """
    n = 3000
    rng = np.random.default_rng(11)
    rho = PUPIL_RADIUS_M * np.sqrt(rng.uniform(0.0, 1.0, n))
    theta = rng.uniform(0.0, 2 * np.pi, n)
    x, y = rho * np.cos(theta), rho * np.sin(theta)
    truth = (12.0e-6, -37.0e-6, R_M + 55.0e-6)
    path = np.sqrt((truth[0] - x) ** 2 + (truth[1] - y) ** 2 + truth[2] ** 2)

    sphere = fit_reference_sphere(
        positions_m=np.stack([x, y, np.zeros(n)], axis=1),
        plane_z_m=0.0,
        optical_path_length_m=-path,
        wavelength_m=WAVELENGTH_M,
        initial_center_m=(0.0, 0.0, R_M),
    )
    assert sphere.center_m[0] == pytest.approx(truth[0], abs=1e-9)
    assert sphere.center_m[1] == pytest.approx(truth[1], abs=1e-9)
    assert sphere.center_m[2] == pytest.approx(truth[2], abs=1e-8)
    assert sphere.residual_rms_waves < 1e-6
    assert sphere.residual_rms_waves < sphere.initial_residual_rms_waves


def test_the_oracle_refuses_an_amplitude_that_carries_phase() -> None:
    """Taking the modulus would silently model a different pupil."""
    bundle = _perfect_bundle(n=500)
    with_phase = RayBundle(
        positions_m=bundle.positions_m,
        directions=bundle.directions,
        wavelength_m=bundle.wavelength_m,
        reference_plane=bundle.reference_plane,
        frame=bundle.frame,
        amplitude=np.full(500, 1.0 + 0.3j),
        optical_path_length_m=bundle.optical_path_length_m,
        optical_path_length_reference=bundle.optical_path_length_reference,
    )
    with pytest.raises(ContractError) as refused:
        pupil_aberration(with_phase, plane_z_m=0.0, observation_point_m=(0.0, 0.0, R_M))
    assert refused.value.code is ContractCode.PHASOR_MISMATCH


def test_resampling_preserves_a_known_pattern() -> None:
    """The comparison machinery must not itself distort what it compares."""
    fine = airy_psf_on_grid(
        shape=(1024, 1024),
        sample_pitch_m=(PITCH_M / 8, PITCH_M / 8),
        wavelength_m=WAVELENGTH_M,
        numerical_aperture=NA_PARAXIAL,
    )
    coarse_pitch = (PITCH_M, PITCH_M)
    resampled = resample_to_grid(
        fine, from_pitch_m=(PITCH_M / 8, PITCH_M / 8), to_pitch_m=coarse_pitch, to_shape=(64, 64)
    )
    direct = airy_psf_on_grid(
        shape=(64, 64),
        sample_pitch_m=coarse_pitch,
        wavelength_m=WAVELENGTH_M,
        numerical_aperture=NA_PARAXIAL,
    )
    assert np.max(np.abs(resampled - direct)) < 5e-3


# ---------------------------------------------------------------------------
# The recorded evidence
# ---------------------------------------------------------------------------


def test_the_diffraction_limited_case_matches_the_analytic_airy(record: dict) -> None:
    """Profile residual and first-null radius, on the frozen configuration."""
    airy = record["diffraction_limited"]["vs_analytic_airy"]

    assert airy["relative_l2_profile_residual"] < 1.0e-2
    assert airy["max_abs_profile_residual"] < 1.0e-2

    null = airy["first_null"]
    # The estimator bias is ~11% at this sampling, so the meaningful comparison is
    # against the same estimator on the analytic pattern.
    assert null["analytic_estimator_bias"] > 0.08
    assert null["ratio_measured_over_analytic"] == pytest.approx(1.0, abs=0.01)


def test_the_wavefront_reproduces_the_frozen_peak_to_valley(record: dict) -> None:
    """An independent route to a number three other routes already agree on.

    The protocol froze 0.016996 waves P-V for M3-SINGLET-REF and Optiland's own
    ``Wavefront`` reports 0.016999. This ticket's reference-sphere machinery is a
    third implementation, so agreement is evidence about the machinery.
    """
    measured = record["diffraction_limited"]["pupil_wavefront_at_observation_plane"][
        "peak_to_valley_waves"
    ]
    assert measured == pytest.approx(0.016999, rel=2e-3)


def test_the_defocused_case_is_genuinely_aberrated_and_marechal_still_applies(
    record: dict,
) -> None:
    """Past the Rayleigh limit, inside Maréchal's regime. Both matter."""
    wavefront = record["defocused"]["pupil_wavefront_at_observation_plane"]

    assert wavefront["peak_to_valley_waves"] > 0.25, "not aberrated past the Rayleigh limit"
    assert wavefront["rayleigh_quarter_wave_limited"] is False
    assert wavefront["rms_waves"] <= 0.1, "outside the Maréchal validity regime"
    assert wavefront["marechal_valid_regime"] is True

    # And it must actually disagree with Airy, or the non-Airy path is untested.
    airy_residual = record["defocused"]["vs_analytic_airy_should_disagree"][
        "relative_l2_profile_residual"
    ]
    diffraction_limited = record["diffraction_limited"]["vs_analytic_airy"][
        "relative_l2_profile_residual"
    ]
    assert airy_residual > 10.0 * diffraction_limited


def test_the_strehl_cross_check_agrees_with_marechal(record: dict) -> None:
    """A sanity cross-check, not a gate. A few percent is what it can deliver."""
    strehl = record["defocused"]["strehl"]

    assert strehl["measured_strehl"] < 0.8, "an unaberrated case cannot test Strehl"
    assert strehl["ratio_measured_over_marechal"] == pytest.approx(1.0, abs=0.15)
    assert strehl["marechal_valid_regime"] is True
    assert "not a tolerance gate" in strehl["validity"].lower().replace("NOT", "not")


def test_the_energy_ledger_closes_where_it_must(record: dict) -> None:
    """The measurement step must be exactly unity; the rest are attributed."""
    for configuration in ("diffraction_limited", "defocused"):
        ledger = record[configuration]["energy_ledger"]
        ratios = ledger["ratios"]

        assert ratios["psf_integral_over_propagated_out"] == pytest.approx(1.0, abs=1e-5)
        assert ratios["propagated_out_over_in"] == pytest.approx(1.0, abs=1e-4)
        # Every named step carries an attribution, and the ray-weight conversion is
        # explicitly NOT expected to be unity.
        for key in ratios:
            assert key in ledger["attribution"]
        assert "NOT a conservation law" in ledger["attribution"]["pupil_over_traced"]
        assert "wraparound" in ledger["attribution"]["propagated_out_over_in"]


def test_the_negative_controls_report_margins_and_two_of_them_are_blind(
    record: dict,
) -> None:
    """Detected, undetected, and *why* -- following M2's rule about weak controls."""
    controls = {c["control"]: c for c in record["negative_controls"]["controls"]}

    # The unperturbed control reproduces the graph node bit for bit.
    assert record["negative_controls"]["unperturbed_control"]["relative_l2_vs_airy"] == (
        record["negative_controls"]["graph_node_residual_for_reference"]
    )

    for name in ("opl_sign_flip", "reconstruction_phase_sign_flip", "oblique_ramp_omitted"):
        assert controls[name]["detection_margin"] > 10.0, name
        assert controls[name]["detected"] is True, name

    # A sign flip on the distance is nearly degenerate with a sign flip on the
    # phase: conjugating the field and reversing the propagation are the same
    # operation. Two of the four requested controls test one degree of freedom.
    assert controls["propagation_distance_sign"]["detection_margin"] > 10.0
    assert controls["propagation_distance_sign"]["relative_l2_vs_airy"] == pytest.approx(
        controls["reconstruction_phase_sign_flip"]["relative_l2_vs_airy"], rel=1e-6
    )

    # And the two that cannot fire here are recorded as not firing.
    assert controls["axis_transpose"]["detection_margin"] == pytest.approx(1.0, abs=1e-4)
    assert controls["axis_transpose"]["detected"] is False
    assert controls["amplitude_weight_omitted"]["detection_margin"] == pytest.approx(
        1.0, abs=1e-4
    )
    assert controls["amplitude_weight_omitted"]["detected"] is False


def test_the_amplitude_degree_of_freedom_is_not_exercised_at_all(record: dict) -> None:
    """Not "weak" -- an exact no-op, which is a stronger and worse statement."""
    amplitude = record["amplitude_degree_of_freedom"]

    assert amplitude["ray_intensity_spread"] == 0.0
    assert amplitude["relative_spread"] == 0.0
    assert "NOT exercised" in amplitude["finding"]


def test_the_orientation_control_can_be_made_to_fail(record: dict) -> None:
    """The blind spot the frozen configurations leave, covered by a known tilt."""
    control = record["orientation_control"]

    assert control["status"] == "ran"
    # The PSF moves the analytically predicted distance, along x only.
    assert control["shift_ratio_measured_over_predicted"] == pytest.approx(1.0, abs=0.05)
    assert control["tilt_moves_the_psf_along_x_only"] is True
    # And a transpose swaps which axis it moves along, so it is detectable.
    assert control["transpose_swaps_the_displacement_axis"] is True
    assert control["transpose_is_detectable_here"] is True


def test_the_off_axis_handoff_puts_the_psf_where_the_rays_go(record: dict) -> None:
    """Four narrower positive claims, replacing one assertion of a defect.

    M3.8 pinned the absence of the pupil tilt so it could not be forgotten. CHE-41
    supplied it, so an assertion that it is missing would now be false. Following
    M2's precedent -- whose ``wave_to_ray_not_claimed`` absence-check was replaced
    by four narrower claims rather than deleted -- each thing the old test
    established is replaced by the positive statement that took its place, and the
    superseded numbers are asserted to still be *recorded* so the history cannot be
    quietly dropped.

    The oracle proper (sub-pixel PSF position, the fitted sphere centre, and a
    reference-sphere-free geometric spot) lives in
    tests/test_m3_off_axis_handoff.py against CHE-41's own record. This test only
    asks that M3.8's own path agrees.
    """
    handoff = record["off_axis_handoff"]

    # 1. The tilt is present. The least-squares slope reads ~0.19% above y/R
    #    because a sphere's slope is not constant across the pupil, so this is
    #    bounded loosely on purpose: the sphere fit is the oracle, not the slope.
    assert handoff["slope_present_as_fraction_of_required"] == pytest.approx(1.0, abs=0.01)
    # 2. The reconstructed wave converges where the rays go, not on the axis.
    assert handoff["fitted_centre_distance_from_geometric_point_m"] < 2.0e-6
    assert handoff["fitted_centre_distance_from_axis_m"] > 1.0e-4
    # 3. The wavefront against the geometric image point is diffraction limited,
    #    which is what a converging sphere aimed at the right point looks like.
    assert handoff["wavefront_pv_waves_against_the_geometric_point"] < 0.25
    # 4. The shipping PSF lands there too, within one pixel of the prediction.
    assert abs(handoff["geometric_image_height_pixels"]) > 100
    assert (
        abs(
            handoff["measured_psf_peak_offset_pixels"]
            - handoff["geometric_image_height_pixels"]
        )
        <= 1.0
    )

    # The defect is retained as measured, not erased.
    superseded = handoff["superseded_finding"]
    assert superseded["slope_present_as_fraction_of_required"] < 0.01
    assert superseded["wavefront_pv_waves_against_the_geometric_point"] > 50.0
    assert abs(superseded["measured_psf_peak_offset_pixels"]) <= 2
    assert "CHE-41" in handoff["fixed_by"]


def test_the_two_failing_gates_are_pinned_as_failing_with_their_diagnosis(
    record: dict,
) -> None:
    """A gate failure is a finding. It must not be able to go green quietly.

    If a later change makes these pass, this test fails and forces someone to say
    what changed -- which is the point. No tolerance was widened to reach them.
    """
    gates = record["gates"]

    assert gates["energy_accounting_unexplained_residual"]["verdict"] == "pass"

    fft = gates["fft_oracle_intensity_relative_l2"]
    assert fft["verdict"] == "FAIL"
    assert fft["ratio_measured_over_gate"] > 10.0
    assert "ray_sampling_error" in fft["diagnosis"]
    assert "flattens" in fft["unresolved"].lower()

    airy = gates["airy_peak_intensity_relative"]
    assert airy["verdict"] == "FAIL"
    assert airy["best_over_ray_count_sweep"] < airy["gate"], (
        "the metric must reach the gate at higher ray count, or the attribution to "
        "ray sampling is not supported"
    )


def test_the_gate_failures_are_attributed_by_a_measured_ray_count_trend(
    record: dict,
) -> None:
    """Attribution by measurement: vary only the rays, watch the residual fall."""
    convergence = record["ray_count_convergence"]

    assert convergence["monotonically_falling"] is True
    assert convergence["first_over_last_ratio"] > 3.0

    rows = convergence["rows"]
    assert len(rows) >= 5
    assert rows[0]["traced_rays"] < rows[-1]["traced_rays"]
    # The peak-intensity metric collapses over the same sweep, which is what ties
    # both gate failures to the one unmeasured term.
    assert abs(rows[0]["airy_peak_deficit_full_window"]) > 0.5
    assert abs(rows[-1]["airy_peak_deficit_full_window"]) < 0.05
