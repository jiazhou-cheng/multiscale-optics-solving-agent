"""CHE-37 (M3.8): the slice verified against independent oracles.

Two kinds of test here, deliberately separated.

The fast ones exercise the oracle code live on synthetic inputs whose answers are
known in closed form -- a perfect converging pupil must produce an Airy pattern, a
known reference sphere must be recovered, a known injected tilt must move the PSF
a predicted distance. An oracle that has not been checked against something it
cannot get wrong is not an oracle.

The rest read ``benchmarks/probes/records/m3_psf_verification.json``, recorded by
``benchmarks/probes/psf_oracle_verification.py`` against the real engines. Re-running
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

from core.boundary import (
    ContractCode,
    ContractError,
    Frame,
    RayBundle,
    ReferencePlane,
)
from verification.psf_oracles import (
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
#: CHE-103. The sampling study that establishes which term the frozen
#: configuration's first-null deviation belongs to, kept in its own record
#: because it is a separate scientific question with its own sweeps.
FIRST_NULL_CONVERGENCE_PATH = (
    ROOT / "benchmarks" / "probes" / "records" / "m3_first_null_grid_convergence.json"
)

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


@pytest.fixture(scope="module")
def first_null_convergence() -> dict:
    if not FIRST_NULL_CONVERGENCE_PATH.exists():
        pytest.skip(
            f"{FIRST_NULL_CONVERGENCE_PATH.relative_to(ROOT)} is missing; run "
            "benchmarks/probes/first_null_grid_convergence.py first"
        )
    return json.loads(FIRST_NULL_CONVERGENCE_PATH.read_text())


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
    assert (
        pytest.approx(AIRY_J1_FIRST_ZERO / (2.0 * math.pi), rel=1e-15)
        == AIRY_FIRST_NULL_COEFFICIENT_EXACT
    )
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

    protocol = yaml.safe_load(
        (ROOT / "benchmarks" / "protocols" / "slice_protocol.yaml").read_text()
    )
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
    """The profile agrees. The first-null radius is not a converged measurement.

    Both halves are the current truth and they point opposite ways, which is why
    they are asserted separately rather than behind one tolerance.
    """
    airy = record["diffraction_limited"]["vs_analytic_airy"]

    assert airy["relative_l2_profile_residual"] < 1.0e-2
    assert airy["max_abs_profile_residual"] < 1.0e-2

    null = airy["first_null"]
    # The estimator bias is ~11% at this sampling, so the meaningful comparison is
    # against the same estimator on the analytic pattern.
    assert null["analytic_estimator_bias"] > 0.08


def test_the_first_null_radius_is_pinned_as_an_unconverged_measurement(
    record: dict, first_null_convergence: dict
) -> None:
    """CHE-103: this metric used to be asserted within 1% of unity. It is not.

    The number moved from 0.9963 to 1.0345 when the records were regenerated
    against the current tree, and the tempting readings were both wrong: widen
    the tolerance, or blame CHE-47's quadrature weight for apodizing the rim.
    ``benchmarks/probes/first_null_grid_convergence.py`` measured it instead. The
    frozen configuration puts 2.44 pixels across the Airy radius and the disputed
    shift is 0.09 pixels; refining the grid collapses the difference between the
    two amplitude conventions by 4x, while refining the ray count by 9x does not
    move it at all.

    So the metric is pinned as UNCONVERGED rather than as passing or failing. The
    assertion below is deliberately not a physics tolerance -- it exists so that
    a change which alters this number cannot pass quietly, and so that anyone who
    wants to tighten it has to fix the sampling first. Absolute first-null
    accuracy belongs to M2.1 (CHE-109)'s error budget.
    """
    null = record["diffraction_limited"]["vs_analytic_airy"]["first_null"]
    verdict = first_null_convergence["verdict"]

    assert verdict["frozen_configuration_is_first_null_converged"] is False
    assert verdict["deviation_is_a_grid_artifact"] is True
    assert verdict["pixels_per_airy_radius_frozen"] < 4.0, (
        "under 4 pixels per Airy radius is what makes this measurement unconverged; "
        "if the frozen grid is ever refined past that, re-derive the metric rather "
        "than re-pinning it"
    )

    # Pinned at the measured value, not at a physically motivated bound.
    assert null["ratio_measured_over_analytic"] == pytest.approx(1.0345, abs=5e-4)


def test_the_quadrature_weight_is_exonerated_by_grid_refinement(
    first_null_convergence: dict,
) -> None:
    """The controlled comparison that stopped CHE-47 being blamed for the shift.

    Same rays, same grid, one bit changed: whether the per-ray quadrature area
    weight is folded into the launch amplitude. At the frozen sampling the two
    conventions differ by 3.6% on the first-null radius; once the grid resolves
    the pattern they agree to well under 1%. A real apodization would not care
    how finely it was sampled.

    The collapse is quoted against the WORST refined point (4.1x), not the best
    (11x). The refined arm is non-monotone at the ~0.5% level -- the estimator's
    own floor on this grid -- and quoting the best point would be picking the
    number that flatters the conclusion.
    """
    grid = first_null_convergence["grid_refinement_holding_the_rays"]
    ring = first_null_convergence["ray_refinement_holding_the_grid"]

    assert abs(grid["frozen_weighted_minus_uniform"]) > 0.03
    assert grid["resolved_weighted_minus_uniform_max"] < 0.01
    assert abs(grid["frozen_weighted_minus_uniform"]) > (
        3.0 * grid["resolved_weighted_minus_uniform_max"]
    ), "the probe's own criterion for calling the deviation a grid artifact"

    # And the other arm: refining the rays alone does nothing, so this is not a
    # ray-sampling term and not an under-resolved rim quadrature cell.
    assert ring["weighted_spread_over_32_to_96_rings"] < 2.0e-3


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
    assert (
        record["negative_controls"]["unperturbed_control"]["relative_l2_vs_airy"]
        == (record["negative_controls"]["graph_node_residual_for_reference"])
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

    # The transpose still cannot fire here: a circular pupil and an on-axis PSF
    # are symmetric under it, and this SCORE azimuthally averages about the grid
    # centre. Recorded as not firing rather than quietly omitted.
    assert controls["axis_transpose"]["detection_margin"] == pytest.approx(1.0, abs=1e-4)
    assert controls["axis_transpose"]["detected"] is False

    # amplitude_weight_omitted is a WEAKER statement than this record used to
    # make, and the weakening is the CHE-103 finding rather than a regression.
    # Before CHE-47 the launched amplitude was exactly 1 for every ray, so
    # replacing it with 1 was an exact no-op and the difference was 0.0 -- which
    # the old test asserted, calling it "a stronger and worse statement". Since
    # CHE-47 the control removes a real rim/centre trapezoid correction, so it
    # perturbs the normalized profile by a measured amount. It is still below the
    # detection threshold, so the control still does not fire; what changed is
    # that it is now a genuine perturbation that the Airy-residual score is too
    # blunt to see, rather than nothing at all.
    omitted = controls["amplitude_weight_omitted"]
    assert omitted["max_abs_difference_vs_unperturbed"] > 0.0, (
        "an exact 0.0 here would mean the quadrature weight had silently stopped "
        "being applied -- the pre-CHE-47 configuration, and the state that made "
        "the committed records stop reproducing"
    )
    assert omitted["max_abs_difference_vs_unperturbed"] == pytest.approx(0.00928, abs=5e-5)
    assert 1.0 < omitted["detection_margin"] < 1.2
    assert omitted["detected"] is False

    # The raw scale is what the frozen peak normalization cannot see, and it is
    # ~2.6e20: the control reverts to unit amplitude while the shipping path
    # launches an amplitude carrying the per-ray area element in m^2, so the
    # intensities differ by the square of that element. This is the same factor
    # that made the committed records look like they had lost 20 orders of
    # magnitude of power, and it is the reason a peak-normalized score is blind
    # to it.
    assert omitted["raw_peak_ratio_vs_control"] > 1e19


def test_the_amplitude_degree_of_freedom_is_exercised_but_only_by_the_quadrature(
    record: dict,
) -> None:
    """CHE-103: this test used to assert the opposite, and was measuring one factor.

    It read Optiland's per-ray intensity weight, found it flat, and concluded the
    amplitude path was not exercised. That inference stopped holding at CHE-47,
    which made the launched amplitude ``sqrt(intensity) * quadrature_weight_m2``.
    The hexapolar area weight takes three values by construction -- centre ray
    3/4 of the nominal cell, rim ring 1/2, interior 1 -- so the product is not
    flat even though one factor is. The old assertion could not see this because
    it never looked at the amplitude the coupler actually receives.

    The distinction that survives is narrower and worth keeping: the amplitude is
    exercised by the *sampling*, not by the *physics*. A quadrature weight is
    fixed by how the pupil was diced, so it cannot stand in for apodization,
    vignetting or a polarized Fresnel transmission -- none of which M3 has, and
    all of which remain untested.
    """
    amplitude = record["amplitude_degree_of_freedom"]

    # One factor is still flat, and that part of the old finding was correct.
    assert amplitude["ray_intensity_spread"] == 0.0
    assert amplitude["relative_spread"] == 0.0

    # The product is not.
    assert amplitude["amplitude_is_exercised"] is True
    assert amplitude["declared_amplitude_distinct_values"] == 3
    assert amplitude["declared_amplitude_relative_spread"] > 0.5
    assert amplitude["declared_amplitude_min"] == pytest.approx(
        0.5 * amplitude["declared_amplitude_max"], rel=1e-9
    ), "the rim ring carries exactly half the nominal cell area"
    assert "NON-QUADRATURE" in amplitude["finding"]


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
        abs(handoff["measured_psf_peak_offset_pixels"] - handoff["geometric_image_height_pixels"])
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
    # CHE-103: this assertion used to require best_over_ray_count_sweep < gate,
    # on the reasoning that "the metric must reach the gate at higher ray count,
    # or the attribution to ray sampling is not supported". Against the current
    # tree it does not: 5.40e-3 against a 2.0e-3 gate, where the old record had
    # 2.93e-4. Refining the rays no longer brings this metric inside its gate.
    #
    # That is a finding, and the honest form of it is to say the attribution is
    # no longer supported rather than to drop the check or widen the gate. It is
    # NOT re-diagnosed here: M4.2 (CHE-117) owns settling L2-PSF-01's unmet gate,
    # and CHE-103's scope is reconciling the evidence with the code, not
    # rebuilding the coupler's error budget. What this test does is stop the
    # changed state from being read as the old one.
    assert airy["best_over_ray_count_sweep"] > airy["gate"], (
        "if ray refinement has started reaching this gate again, the CHE-103 "
        "finding has been resolved -- say so and re-derive the attribution, do "
        "not restore the old assertion silently"
    )
    assert airy["best_over_ray_count_sweep"] == pytest.approx(5.40e-3, abs=5e-5)


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


# ---------------------------------------------------------------------------
# CHE-103: two findings that had no test, which is how they drifted unnoticed
# ---------------------------------------------------------------------------


def test_the_off_axis_reconstruction_got_worse_and_that_is_recorded(record: dict) -> None:
    """The off-axis residual moved 7.5x and nothing was asserting on it.

    Regenerating against the current tree took
    ``off_axis_negative_controls.unperturbed_control.relative_l2_vs_airy`` from
    ``1.48e-3`` to ``1.11e-2``, and dropped every off-axis detection margin by
    the same factor -- not because the controls got worse but because they are
    quoted against that denominator. On axis the same metric barely moved
    (``5.87e-3 -> 5.51e-3``), so this is specific to the off-axis field.

    The ``amplitude_weight_omitted`` control is the informative one: its margin
    is now ``0.134``, i.e. **removing** CHE-47's quadrature weight makes the
    off-axis PSF agree with the analytic Airy 7.5x better. A rim taper against a
    uniform-pupil oracle is the obvious candidate and is not established here.
    M2.1 (CHE-109) owns the ray->wave error budget this belongs to.

    This test exists so the number cannot move again without someone noticing.
    """
    off = record["off_axis_negative_controls"]
    controls = {c["control"]: c for c in off["controls"]}

    assert off["unperturbed_control"]["relative_l2_vs_airy"] == pytest.approx(1.109e-2, rel=1e-2)

    # Removing the weight IMPROVES off-axis agreement. Asserted as < 1 rather
    # than at a value, because the direction is the finding.
    omitted = controls["amplitude_weight_omitted"]
    assert omitted["detection_margin"] < 1.0, (
        "the off-axis finding is that omitting the quadrature weight makes the "
        "residual smaller; a margin >= 1 means that reversed and the CHE-103 "
        "finding needs re-deriving"
    )
    assert omitted["detection_margin"] == pytest.approx(0.1336, abs=5e-4)

    # The genuinely detectable controls still fire, by a wide margin.
    for name in ("opl_sign_flip", "reconstruction_phase_sign_flip", "oblique_ramp_omitted"):
        assert controls[name]["detection_margin"] > 10.0, name


def test_the_record_states_the_precision_it_was_measured_in(record: dict) -> None:
    """The second, independent cause of the CHE-100 record drift.

    The 20-order-of-magnitude power change was CHE-47's quadrature weight. It was
    not the only delta: the old records' PSF numbers carry float64 precision
    (``9697164.659904482`` is not float32-representable) and the current ones are
    float32-exact (``9697163.0``). Same amplitude convention on both sides of
    that comparison, so it is a separate change.

    The mechanism is ``jax_enable_x64``. Chromatix's ``ScalarField`` casts to
    ``complex64``, but under x64 the FFTs behind ``kernel_propagate`` promote,
    and ``pin_wave_engine_precision`` -- which turns the flag off at
    Chromatix-import time -- "cannot retroactively downcast a field that was
    already constructed under x64", as its own docstring says. That call site is
    byte-identical between the commit that wrote the old record and this one, so
    this was never a code change: it was import order deciding the precision of
    committed scientific evidence.

    Recording the dtype is the durable half of the fix. The other half -- check
    the flag at the boundary that depends on it rather than setting it earlier
    and hoping -- is CHE-102's rule applied to the wave solver, and is not this
    ticket's scope.
    """
    precision = record["measurement_precision"]

    assert precision["propagated_field_dtype"] == "complex64"
    assert precision["psf_intensity_dtype"] == "float32"
    # The pupil is reconstructed on the host in double and only loses precision
    # at the Chromatix boundary, which is where the protocol's 3.5e-4 term sits.
    assert precision["pupil_field_dtype"] == "complex128"
    assert "jax_enable_x64" in precision["note"]
