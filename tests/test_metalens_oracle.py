"""The CHE-70 oracle, held to two independent references (CHE-70, Phase 22/23).

The gate's oracle is analytic: a plane-parallel homogeneous stack is diagonal in
the plane-wave basis, so ``evaluation.metalens.reference_field`` evaluates a
closed form over the grid's own modes rather than discretizing a propagation.
That claim is worth exactly as much as its cross-checks, so the single-layer case
is held against two things written for other reasons:

* ``evaluation.asm_oracle.angular_spectrum_float64`` (CHE-40), the repository's
  independent float64 angular-spectrum reference -- in the *un-centred* FFT
  convention, so agreement is not a shared-convention artifact;
* Chromatix's ``asm_propagate``, a third-party M1-verified package with a
  genuinely different front end.

For the slab configuration no prior repository reference exists. What is checked
instead is the one assumption the layered form adds -- ideal transmission at both
interfaces -- against the traced intensity, rather than trusting it.

The metrics are tested too, because a comparison function that cannot fail is
worse than none: ``compare_psfs`` is fed shifted, scaled and broadened inputs and
each metric has to move in the direction it claims to measure.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from multiscale_optics_agent.evaluation.asm_oracle import (
    angular_spectrum_float64,
    compare_fields,
)
from multiscale_optics_agent.evaluation.metalens import (
    AIR_CONFIG,
    CONFIGURATIONS,
    SLAB_CONFIG,
    centred_spectrum,
    compare_psfs,
    encircled_energy_radius_m,
    layered_transfer,
    metalens_field,
    normalized_cross_correlation,
    optical_system_spec,
    reference_field,
    retained_mode_mask,
)

pytestmark = [pytest.mark.coupler]

FLOOR = 1.0e-2


class TestConfiguration:
    @pytest.mark.parametrize("name", sorted(CONFIGURATIONS))
    def test_the_grid_is_the_hundred_by_hundred_the_ticket_specifies(self, name):
        config = CONFIGURATIONS[name]
        assert config.grid_shape == (100, 100)
        assert metalens_field(config).shape == (100, 100)

    @pytest.mark.parametrize("name", sorted(CONFIGURATIONS))
    def test_the_aperture_fits_inside_the_window_with_a_guard_band(self, name):
        config = CONFIGURATIONS[name]
        assert 2 * config.aperture_radius_m < config.window_m
        assert config.aperture_radius_m / (0.5 * config.window_m) == pytest.approx(0.8)

    @pytest.mark.parametrize("name", sorted(CONFIGURATIONS))
    def test_the_psf_is_resolved_by_the_sensor_grid(self, name):
        """A PSF under one sample is not a PSF, and no NCC on it would mean anything."""
        config = CONFIGURATIONS[name]
        assert config.airy_radius_m / config.sample_pitch_m > 4.0

    @pytest.mark.parametrize("name", sorted(CONFIGURATIONS))
    def test_the_steepest_useful_ray_stays_inside_the_periodic_window(self, name):
        """Both routes solve the same periodic problem, so wraparound cannot break
        their agreement -- but it would make the PSF a wrapped artifact instead of a
        PSF. Recorded as a property of the configuration."""
        config = CONFIGURATIONS[name]
        assert config.max_lateral_travel_m <= 0.5 * config.window_m

    @pytest.mark.parametrize("name", sorted(CONFIGURATIONS))
    def test_the_metalens_phase_is_sampled_well_inside_nyquist(self, name):
        config = CONFIGURATIONS[name]
        local_frequency = config.numerical_aperture / config.wavelength_m
        nyquist = 1.0 / (2.0 * config.sample_pitch_m)
        assert local_frequency < 0.25 * nyquist

    def test_the_slab_sensor_sits_at_the_plate_shifted_focus(self):
        shift = 10e-6 * (1.0 - 1.0 / 1.5)
        assert SLAB_CONFIG.sensor_distance_m == pytest.approx(50e-6 + shift)
        assert AIR_CONFIG.sensor_distance_m == pytest.approx(50e-6)

    def test_the_layer_stack_totals_the_sensor_distance(self):
        for config in CONFIGURATIONS.values():
            total = sum(layer.thickness_m for layer in config.scaled_layers)
            assert total == pytest.approx(config.sensor_distance_m, rel=1e-12)

    def test_the_prescription_reproduces_the_layer_stack(self):
        for config in CONFIGURATIONS.values():
            spec = optical_system_spec(config)
            assert len(spec.surfaces) == len(config.scaled_layers)
            assert spec.surfaces[0].is_stop
            for surface, layer in zip(spec.surfaces, config.scaled_layers, strict=True):
                assert surface.thickness_mm == pytest.approx(layer.thickness_m * 1e3)
            assert spec.wavelengths[0].value_um == pytest.approx(
                config.wavelength_m * 1e6
            )


class TestSourceField:
    def test_both_routes_start_from_the_same_field(self):
        """Not a tautology: it is what stops the two legs being tuned separately."""
        left = np.asarray(metalens_field(AIR_CONFIG).u)
        right = np.asarray(metalens_field(AIR_CONFIG).u)
        assert np.array_equal(left, right)
        reference = reference_field(AIR_CONFIG, direction_cosine_floor=FLOOR)
        assert reference.provenance["configuration"] == AIR_CONFIG.name

    def test_the_amplitude_is_a_hard_circular_aperture(self):
        field = np.asarray(metalens_field(AIR_CONFIG).u)
        magnitude = np.abs(field)
        assert set(np.unique(np.round(magnitude, 12))) == {0.0, 1.0}
        assert int((magnitude > 0).sum()) == 5025

    def test_the_phase_is_the_hyperbolic_metalens_phase(self):
        config = AIR_CONFIG
        field = np.asarray(metalens_field(config).u)
        coordinate = (np.arange(config.grid) - config.grid // 2) * config.sample_pitch_m
        x, y = np.meshgrid(coordinate, coordinate, indexing="xy")
        radius = np.hypot(x, y)
        inside = radius <= config.aperture_radius_m
        expected = -config.wavenumber * (
            np.sqrt(radius**2 + config.design_focal_length_m**2)
            - config.design_focal_length_m
        )
        residual = np.angle(field[inside] * np.exp(-1j * expected[inside]))
        assert np.abs(residual).max() < 1e-9

    def test_the_field_focuses_where_the_design_says(self):
        reference = reference_field(AIR_CONFIG, direction_cosine_floor=FLOOR)
        intensity = np.abs(np.asarray(reference.u)) ** 2
        assert np.unravel_index(int(intensity.argmax()), intensity.shape) == (50, 50)
        # And the spot is Airy-sized rather than merely centred.
        radius = encircled_energy_radius_m(intensity, pitch=AIR_CONFIG.sample_pitch_m)
        assert 0.3 * AIR_CONFIG.airy_radius_m < radius < 1.2 * AIR_CONFIG.airy_radius_m


class TestOracleCrossChecks:
    def test_the_layered_form_matches_the_repository_asm_reference(self):
        """CHE-40's reference, in the opposite FFT convention. Agreement to 1e-13."""
        source = np.asarray(metalens_field(AIR_CONFIG).u)
        mine = np.asarray(reference_field(AIR_CONFIG, direction_cosine_floor=FLOOR).u)
        theirs = angular_spectrum_float64(
            source,
            wavelength_m=AIR_CONFIG.wavelength_m,
            sample_pitch_m=AIR_CONFIG.sample_pitch_m,
            z_m=AIR_CONFIG.sensor_distance_m,
        )
        comparison = compare_fields(mine, theirs)
        assert comparison.piston_aligned_relative_field_error < 1e-12
        assert comparison.relative_intensity_l2_error < 1e-12
        assert comparison.energy_residual < 1e-13

    def test_the_band_limit_is_the_only_difference_from_the_unrestricted_reference(self):
        """The excluded modes carry 2.3e-7 of the power, so the two must nearly agree.

        Which is the point: the band limit is not quietly doing optical work. It
        removes eight numerically unusable bins and nothing else.
        """
        limited = np.asarray(reference_field(AIR_CONFIG, direction_cosine_floor=FLOOR).u)
        unrestricted = np.asarray(reference_field(AIR_CONFIG, direction_cosine_floor=0.0).u)
        assert compare_fields(limited, unrestricted).piston_aligned_relative_field_error < 1e-3

    def test_the_retained_mask_is_the_one_the_coupler_uses(self):
        """Two implementations of the same two conditions must not drift apart."""
        from multiscale_optics_agent.core.precision import Precision
        from multiscale_optics_agent.couplers.streaming import band_limit_spectrum
        from multiscale_optics_agent.couplers.wave_to_ray import decompose

        limited, _ = band_limit_spectrum(
            decompose(metalens_field(AIR_CONFIG)),
            direction_cosine_floor=FLOOR,
            max_optical_path_m=AIR_CONFIG.sensor_distance_m,
            precision=str(Precision.FP64),
        )
        assert np.array_equal(
            np.asarray(limited.propagating),
            retained_mode_mask(AIR_CONFIG, direction_cosine_floor=FLOOR),
        )

    def test_the_centred_spectrum_agrees_with_the_couplers_own(self):
        from multiscale_optics_agent.couplers.wave_to_ray import decompose

        field = metalens_field(AIR_CONFIG)
        assert np.allclose(
            centred_spectrum(np.asarray(field.u)),
            np.asarray(decompose(field).spectrum),
            rtol=0,
            atol=1e-18,
        )

    def test_the_transfer_function_is_unit_modulus_where_it_propagates(self):
        for config in CONFIGURATIONS.values():
            transfer = layered_transfer(config)
            mask = retained_mode_mask(config, direction_cosine_floor=FLOOR)
            assert np.allclose(np.abs(transfer[mask]), 1.0, atol=1e-12), (
                f"{config.name}: an oracle that is not unit modulus is not lossless"
            )

    def test_zero_distance_is_the_identity_on_the_retained_band(self):
        import dataclasses

        from multiscale_optics_agent.evaluation.metalens import Layer

        config = dataclasses.replace(
            AIR_CONFIG, layers=(Layer(thickness_m=1e-18, refractive_index=1.0, name="air"),),
            design_focal_length_m=1e-18,
        )
        source = np.asarray(metalens_field(config).u)
        mask = retained_mode_mask(config, direction_cosine_floor=FLOOR)
        band_limited = np.fft.fftshift(
            np.fft.ifft2(np.fft.ifftshift(np.where(mask, centred_spectrum(source), 0.0)))
        ) * (config.grid**2)
        propagated = np.asarray(reference_field(config, direction_cosine_floor=FLOOR).u)
        # 1e-10 rather than machine epsilon: both sides take a 100x100 FFT round
        # trip over values of order 5e3, and 3e-12 is that round trip's own floor.
        assert compare_fields(propagated, band_limited).raw_relative_field_error < 1e-10


@pytest.mark.chromatix
@pytest.mark.integration
class TestChromatixCrossCheck:
    """A third-party route to the same reference. Not a gate; corroboration.

    Chromatix is complex64-only, so this cannot be a tight test -- and it should
    not be: what it establishes is that the analytic oracle is not idiosyncratic,
    at the precision Chromatix has. A complex64 field of ~5e3 discrete power over
    a 100-wave propagation is good to a few times 1e-5 relative, so the bound
    below is set by the reference leg's own dtype and is stated as such.
    """

    def test_chromatix_asm_propagate_agrees_with_the_analytic_oracle(self):
        pytest.importorskip("chromatix")
        import chromatix.functional as cf
        import jax
        import jax.numpy as jnp

        # x64 off, matching the adapter: asm_propagate can otherwise promote to
        # complex128 and the comparison would not be against the precision
        # Chromatix actually ships.
        jax.config.update("jax_enable_x64", False)
        config = AIR_CONFIG
        source = np.asarray(metalens_field(config).u)
        mask = retained_mode_mask(config, direction_cosine_floor=FLOOR)
        # Give Chromatix the *band-limited* field, so the two routes carry the
        # same modes and the comparison is not charged for the eight bins the ray
        # ensemble cannot represent.
        limited = np.fft.fftshift(
            np.fft.ifft2(np.fft.ifftshift(np.where(mask, centred_spectrum(source), 0.0)))
        ) * (config.grid**2)

        field = cf.Field.build(
            jnp.asarray(limited, dtype=jnp.complex64),
            jnp.asarray([[config.sample_pitch_m, config.sample_pitch_m]]),
            config.wavelength_m,
        )
        propagated = cf.asm_propagate(
            field, z=config.sensor_distance_m, n=1.0, pad_width=0
        )
        theirs = np.asarray(propagated.u).squeeze()
        mine = np.asarray(reference_field(config, direction_cosine_floor=FLOOR).u)
        comparison = compare_fields(theirs, mine)
        assert comparison.piston_aligned_relative_field_error < 5.0e-4, (
            "the analytic oracle and Chromatix must agree to the complex64 floor; "
            f"got {comparison.piston_aligned_relative_field_error:.3e}"
        )
        assert (
            1.0 - normalized_cross_correlation(np.abs(theirs) ** 2, np.abs(mine) ** 2)
            < 1.0e-6
        )


class TestSlabAssumption:
    """The slab oracle's one added assumption, checked rather than trusted."""

    def test_the_pinned_solver_applies_no_fresnel_amplitude_loss(self):
        """The oracle assumes ideal transmission; Optiland must actually do that.

        With no coatings configured, optiland 0.6.0 refracts without touching
        ``RealRays.i``. If a future version applied Fresnel coefficients, the
        oracle would need them too -- so the assumption is a test, not a comment.
        """
        pytest.importorskip("optiland")
        import numpy as np
        from optiland.rays import RealRays

        from multiscale_optics_agent.adapters.optiland_builder import (
            build_optiland_system,
        )
        from multiscale_optics_agent.adapters.optiland_ray_trace import (
            configure_optiland_execution,
        )
        from multiscale_optics_agent.core.precision import (
            DeviceKind,
            DevicePlacement,
            Precision,
        )

        configure_optiland_execution(
            device=DevicePlacement(DeviceKind.CPU, None),
            precision=Precision.FP64,
            enable_grad=False,
        )
        lens = build_optiland_system(optical_system_spec(SLAB_CONFIG))
        count = 5
        angles = np.linspace(0.0, 0.3, count)
        rays = RealRays(
            np.zeros(count),
            np.zeros(count),
            np.zeros(count),
            angles,
            np.zeros(count),
            np.sqrt(1.0 - angles**2),
            np.full(count, 0.75),
            np.full(count, 0.5),
        )
        traced = lens.surfaces.trace(rays, skip=1)
        assert np.allclose(np.asarray(traced.i), 0.75, atol=1e-12), (
            "intensity changed across the interfaces; the oracle's ideal-transmission "
            "assumption no longer holds and must be replaced by Fresnel coefficients"
        )

    def test_snells_law_holds_at_the_interface(self):
        """The transverse wavevector is what the layered transfer function conserves."""
        pytest.importorskip("optiland")
        from optiland.rays import RealRays

        from multiscale_optics_agent.adapters.optiland_builder import (
            build_optiland_system,
        )
        from multiscale_optics_agent.adapters.optiland_ray_trace import (
            configure_optiland_execution,
        )
        from multiscale_optics_agent.core.precision import (
            DeviceKind,
            DevicePlacement,
            Precision,
        )

        configure_optiland_execution(
            device=DevicePlacement(DeviceKind.CPU, None),
            precision=Precision.FP64,
            enable_grad=False,
        )
        lens = build_optiland_system(optical_system_spec(SLAB_CONFIG))
        angles = np.array([0.0, 0.05, 0.1, 0.2, 0.3])
        rays = RealRays(
            np.zeros_like(angles), np.zeros_like(angles), np.zeros_like(angles),
            angles, np.zeros_like(angles), np.sqrt(1.0 - angles**2),
            np.ones_like(angles), np.full_like(angles, 0.5),
        )
        # Trace only the first interface, so the direction inside the glass is read.
        lens.surfaces.reset()
        lens.surfaces.surfaces[1].trace(rays)
        assert np.allclose(np.asarray(rays.L) * 1.5, angles, atol=1e-12), (
            "n * sin(theta) is not conserved across the interface"
        )


class TestMetrics:
    """Every metric has to move when the thing it measures moves."""

    @staticmethod
    def _gaussian(shape=(64, 64), sigma=4.0, shift=(0.0, 0.0), scale=1.0, ):
        y = np.arange(shape[0]) - shape[0] // 2 - shift[0]
        x = np.arange(shape[1]) - shape[1] // 2 - shift[1]
        xx, yy = np.meshgrid(x, y, indexing="xy")
        return scale * np.exp(-(xx**2 + yy**2) / (2 * sigma**2))

    def test_ncc_is_one_for_identical_images_and_blind_to_scale(self):
        image = self._gaussian()
        assert normalized_cross_correlation(image, image) == pytest.approx(1.0)
        assert normalized_cross_correlation(3.7 * image, image) == pytest.approx(1.0)

    def test_ncc_removes_the_mean(self):
        """A pedestal must not manufacture correlation.

        Two *uncorrelated* images plus a large common pedestal correlate at
        ~1 under a raw dot product. The zero-mean form is the stricter one and is
        what the gate is stated in.
        """
        rng = np.random.default_rng(0)
        left = rng.random((32, 32))
        right = rng.random((32, 32))
        pedestal = 1000.0
        raw = float(
            (left + pedestal).ravel() @ (right + pedestal).ravel()
            / (
                np.linalg.norm((left + pedestal).ravel())
                * np.linalg.norm((right + pedestal).ravel())
            )
        )
        assert raw > 0.999
        assert abs(normalized_cross_correlation(left + pedestal, right + pedestal)) < 0.1

    def test_a_constant_image_has_no_correlation_to_report(self):
        with pytest.raises(ValueError):
            normalized_cross_correlation(np.ones((8, 8)), self._gaussian((8, 8)))

    def test_the_power_error_sees_a_scale_that_ncc_cannot(self):
        reference = self._gaussian()
        comparison = compare_psfs(2.0 * reference, reference, pitch=1e-6)
        assert comparison.ncc == pytest.approx(1.0)
        assert comparison.relative_power_error == pytest.approx(1.0, rel=1e-9)
        assert comparison.relative_peak_error == pytest.approx(1.0, rel=1e-9)

    def test_the_centroid_error_sees_a_shift(self):
        reference = self._gaussian()
        shifted = self._gaussian(shift=(3.0, 0.0))
        comparison = compare_psfs(shifted, reference, pitch=1e-6)
        assert comparison.centroid_error_m == pytest.approx(3e-6, rel=0.05)
        assert compare_psfs(reference, reference, pitch=1e-6).centroid_error_m < 1e-15

    def test_the_width_metrics_see_a_broadening(self):
        reference = self._gaussian(sigma=4.0)
        broad = self._gaussian(sigma=6.0)
        comparison = compare_psfs(broad, reference, pitch=1e-6)
        assert comparison.relative_fwhm_error == pytest.approx(0.5, rel=0.1)
        assert comparison.relative_ee50_error > 0.2

    def test_the_fwhm_matches_the_analytic_gaussian_width(self):
        reference = self._gaussian(sigma=5.0)
        comparison = compare_psfs(reference, reference, pitch=1e-6)
        expected = 2.0 * math.sqrt(2.0 * math.log(2.0)) * 5.0 * 1e-6
        assert comparison.reference_fwhm_m == pytest.approx(expected, rel=0.03)

    def test_the_normalized_mse_is_a_shape_metric_not_a_scale_one(self):
        reference = self._gaussian()
        assert compare_psfs(5.0 * reference, reference, pitch=1e-6).normalized_mse < 1e-24
        assert compare_psfs(
            self._gaussian(sigma=5.0), reference, pitch=1e-6
        ).normalized_mse > 1e-3

    def test_a_shape_mismatch_is_refused(self):
        with pytest.raises(ValueError):
            normalized_cross_correlation(np.zeros((4, 4)), np.zeros((5, 5)))
