"""The coherent wave -> ray -> Optiland -> wave bridge, against analytic cases (CHE-70).

Phase 4's analytical tests, plus the estimator properties the 100x100 benchmark
would otherwise have to take on trust. Every test here runs on the host in
float64 through the *same* code the GPU sweep runs -- one implementation, two
namespaces -- so a defect that would only show up as a slightly worse NCC on the
GPU shows up here as a hard failure instead.

The oracle throughout is analytic. For a plane-parallel homogeneous stack the
propagation operator is diagonal in the plane-wave basis, so the finite sum over
the grid's own modes *is* the exact answer, not a discretization of it.

Two results in here are the reason the benchmark is built the way it is:

``test_the_exactness_limit_reproduces_the_analytic_oracle``
    Full enumeration under the uniform density collapses the Monte Carlo
    estimator onto the analytic field at 1e-13. Everything about the benchmark
    downstream is then a sampling-error question, which is a much smaller claim.

``test_the_grazing_band_limit_is_what_makes_the_exactness_limit_exact``
    Without the band limit the same enumeration lands at ~3e-9 in float64, and
    the residual is eight bins whose OPL is 4745 m. That is the measurement the
    floor is derived from, kept as a test so a future change that drops the floor
    fails here rather than silently degrading the GPU run.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

pytest.importorskip("optiland")

import optiland.backend as be

from core.boundary import (
    ContractError,
    Frame,
    RayBundle,
    ReferencePlane,
)
from core.capabilities import C_RAY_TO_WAVE_CAPABILITIES
from core.coherent_batch import (
    CoherentRayBatch,
    declared_launch_opl_reference,
    metres_to_micrometres,
    metres_to_millimetres,
    micrometres_to_metres,
    millimetres_to_metres,
)
from core.precision import (
    DeviceKind,
    DevicePlacement,
    Precision,
)
from couplers.ray_to_wave import ray_to_wave
from couplers.streaming import (
    PositionalAngularSampler,
    StreamingReconstruction,
    band_limit_spectrum,
    build_chunk_bundle,
    chunk_plan,
    grazing_floor_for_phase_budget,
    nested_aperture_launch_positions,
)
from couplers.wave_to_ray import (
    SamplingDensity,
    decompose,
    enumerate_indices,
    sampling_density,
    spectrum_to_rays,
)
from solvers.optiland.builder import build_optiland_system
from solvers.optiland.coherent_trace import (
    configure_optiland_execution,
    plan_trace_bridges,
    surface_positions_m,
    trace_ray_batch,
)
from studies.metalens.oracle import (
    AIR_CONFIG,
    SLAB_CONFIG,
    metalens_field,
    normalized_cross_correlation,
    optical_system_spec,
    reference_field,
)

pytestmark = [pytest.mark.coupler, pytest.mark.optiland, pytest.mark.integration]

FLOOR = 1.0e-2
HOST = DevicePlacement(DeviceKind.CPU, None)


@pytest.fixture(scope="module")
def host_optiland() -> None:
    """Optiland on the numpy backend in float64, set explicitly rather than inherited.

    ``set_backend`` / ``set_precision`` are process-global in optiland 0.6.0, so a
    previous test's state is never trusted.
    """
    configure_optiland_execution(device=HOST, precision=Precision.FP64, enable_grad=False)
    yield
    configure_optiland_execution(device=HOST, precision=Precision.FP64, enable_grad=False)


def _traced_field(
    config,
    *,
    launch_positions,
    indices=None,
    density_kind=SamplingDensity.UNIFORM,
    floor=FLOOR,
    normalization=None,
):
    """wave_to_ray -> Optiland -> ray_to_wave for one explicit mode/launch set."""
    field_in = metalens_field(config)
    spectrum, _ = band_limit_spectrum(
        decompose(field_in),
        direction_cosine_floor=floor,
        max_optical_path_m=config.sensor_distance_m,
        precision=str(Precision.FP64),
        phase_budget_rad=1.0e-2,
    )
    density = sampling_density(spectrum, density_kind)
    if indices is None:
        indices = enumerate_indices(density)
    bundle = spectrum_to_rays(
        spectrum, indices, density, launch_positions_xy_m=np.asarray(launch_positions)
    )
    batch = CoherentRayBatch(
        bundle=bundle,
        ray_id=np.arange(bundle.count, dtype=np.int64),
        valid=np.ones(bundle.count, dtype=bool),
    )
    lens = build_optiland_system(optical_system_spec(config))
    sensor = ReferencePlane(name="sensor", z_m=config.sensor_distance_m)
    plans = plan_trace_bridges(batch, home=C_RAY_TO_WAVE_CAPABILITIES, device=HOST)
    traced, diagnostics = trace_ray_batch(batch, lens, image_plane=sensor, plans=plans)
    field_out, reconstruction = ray_to_wave(
        traced.bundle,
        grid_shape=config.grid_shape,
        sample_pitch_m=config.pitch_pair,
        plane=sensor,
        normalization=normalization,
    )
    return field_out, traced, diagnostics, reconstruction


def _relative_field_error(test: np.ndarray, reference: np.ndarray) -> float:
    test = np.asarray(test, dtype=np.complex128)
    reference = np.asarray(reference, dtype=np.complex128)
    return float(np.linalg.norm(test - reference) / np.linalg.norm(reference))


# --- Phase 4 Test A: free space, and the exactness limit ----------------------


class TestExactnessLimit:
    """Enumerate every retained bin and the estimator stops being an estimator."""

    @pytest.mark.parametrize("config", [AIR_CONFIG, SLAB_CONFIG], ids=["air", "slab"])
    def test_the_exactness_limit_reproduces_the_analytic_oracle(self, host_optiland, config):
        field_out, _, _, _ = _traced_field(config, launch_positions=[[0.0, 0.0]])
        reference = reference_field(config, direction_cosine_floor=FLOOR)
        error = _relative_field_error(field_out.u, reference.u)
        assert error < 1.0e-11, (
            f"full enumeration under the uniform density must collapse onto the "
            f"analytic field; got {error:.3e}"
        )

    def test_the_grazing_band_limit_is_what_makes_the_exactness_limit_exact(
        self, host_optiland
    ):
        """CHE-70's measured finding, kept as a falsifiable test.

        Eight bins on this grid satisfy ``d_u^2 + d_v^2 = 1`` exactly (the (30,40)
        Pythagorean triples), survive the strict evanescent cut at
        ``d_n = 1.05e-8``, and accumulate a 4745 m OPL over a 50 um propagation.
        ``C_RAY_TO_WAVE`` forms ``k(OPL - d.x0)``, so their phase is lost to
        cancellation. Removing the floor must make the exactness limit *worse* by
        orders of magnitude -- if it ever stops doing so, the floor's derivation no
        longer describes this grid and the constant needs re-deriving.
        """
        reference = reference_field(AIR_CONFIG, direction_cosine_floor=0.0)
        unlimited, _, _, _ = _traced_field(AIR_CONFIG, launch_positions=[[0.0, 0.0]], floor=0.0)
        without = _relative_field_error(unlimited.u, reference.u)

        limited, _, _, _ = _traced_field(AIR_CONFIG, launch_positions=[[0.0, 0.0]], floor=FLOOR)
        with_floor = _relative_field_error(
            limited.u, reference_field(AIR_CONFIG, direction_cosine_floor=FLOOR).u
        )
        assert without > 1.0e-10, (
            "the grazing bins are supposed to be a measurable problem in float64; "
            f"got {without:.3e}"
        )
        assert with_floor < without / 1.0e3, (
            f"the floor must buy orders of magnitude: {with_floor:.3e} against "
            f"{without:.3e}"
        )

    def test_the_grazing_bins_are_the_ones_the_derivation_names(self):
        field_in = metalens_field(AIR_CONFIG)
        spectrum = decompose(field_in)
        _, band = band_limit_spectrum(
            spectrum,
            direction_cosine_floor=FLOOR,
            max_optical_path_m=AIR_CONFIG.sensor_distance_m,
            precision=str(Precision.FP64),
            phase_budget_rad=1.0e-2,
        )
        assert band.excluded_bin_count == 8
        assert band.excluded_power_fraction < 1.0e-6
        assert band.retained_bin_count == 7825
        # Z / d_n_min, the quantity the floor exists to bound.
        assert band.max_retained_optical_path_m == pytest.approx(5.0e-3)

    def test_the_floor_is_derived_from_the_precision_and_the_distance(self):
        """The constant is a rounded-up derivation, not a taste."""
        derived = grazing_floor_for_phase_budget(
            wavelength_m=AIR_CONFIG.wavelength_m,
            max_optical_path_m=AIR_CONFIG.sensor_distance_m,
            precision=Precision.FP32,
        )
        assert 1.0e-3 < derived < FLOOR, f"float32 derivation moved to {derived:.3e}"
        # float64 needs a floor nine orders of magnitude smaller for the same
        # budget, which is why the *float32* requirement is the one that sets it.
        assert grazing_floor_for_phase_budget(
            wavelength_m=AIR_CONFIG.wavelength_m,
            max_optical_path_m=AIR_CONFIG.sensor_distance_m,
            precision=Precision.FP64,
        ) < derived * 1.0e-6


# --- Phase 4 Test B: two-ray interference ------------------------------------


class TestTwoRayInterference:
    """Two coherent rays, one controlled path difference. Nothing else."""

    @staticmethod
    def _pair(delta_opl_m: float) -> RayBundle:
        wavelength = 500e-9
        return RayBundle(
            positions_m=np.zeros((2, 3)),
            directions=np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]),
            wavelength_m=wavelength,
            reference_plane=ReferencePlane(name="p", z_m=0.0),
            frame=Frame(axis_order="flat per-ray arrays"),
            amplitude=np.array([1.0 + 0.0j, 1.0 + 0.0j]),
            optical_path_length_m=np.array([0.0, delta_opl_m]),
            optical_path_length_reference="zero at the emitting plane 'p'",
            normalization="two unit amplitudes",
            reconstruction_normalization="none",
        )

    def test_zero_path_difference_interferes_constructively(self):
        field, _ = ray_to_wave(
            self._pair(0.0), grid_shape=(4, 4), sample_pitch_m=(1e-6, 1e-6)
        )
        assert np.abs(np.asarray(field.u)) == pytest.approx(2.0, abs=1e-12)

    def test_a_half_wave_path_difference_extinguishes(self):
        field, _ = ray_to_wave(
            self._pair(250e-9), grid_shape=(4, 4), sample_pitch_m=(1e-6, 1e-6)
        )
        assert np.abs(np.asarray(field.u)).max() < 1e-12

    @pytest.mark.parametrize("waves", [0.0, 0.25, 0.5, 0.75, 1.0])
    def test_the_fringe_follows_the_path_difference_exactly(self, waves):
        field, _ = ray_to_wave(
            self._pair(waves * 500e-9), grid_shape=(2, 2), sample_pitch_m=(1e-6, 1e-6)
        )
        expected = abs(1.0 + math.cos(2 * math.pi * waves) + 1j * math.sin(2 * math.pi * waves))
        assert np.abs(np.asarray(field.u)).max() == pytest.approx(expected, abs=1e-12)

    def test_an_optiland_intensity_cannot_stand_in_for_the_amplitude(self):
        """Requirement 4: an intensity-only bundle is refused, not coerced.

        The two rays below have equal ``|a|^2`` and opposite sign. Any path that
        rebuilt the amplitude from an intensity would make them interfere
        constructively; the contract refuses the bundle instead.
        """
        bundle = RayBundle(
            positions_m=np.zeros((2, 3)),
            directions=np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]),
            wavelength_m=500e-9,
            reference_plane=ReferencePlane(name="p", z_m=0.0),
            frame=Frame(axis_order="flat per-ray arrays"),
            weight=np.array([1.0, 1.0]),
            weight_semantics="optiland RealRays.i; explicitly not an amplitude",
            optical_path_length_m=np.zeros(2),
            optical_path_length_reference="zero at 'p'",
        )
        with pytest.raises(ContractError) as raised:
            ray_to_wave(bundle, grid_shape=(2, 2), sample_pitch_m=(1e-6, 1e-6))
        assert raised.value.code == "AMPLITUDE_IS_A_WEIGHT"

        signed = self._pair(0.0)
        opposed = dataclasses.replace(
            signed, amplitude=np.array([1.0 + 0.0j, -1.0 + 0.0j])
        )
        field, _ = ray_to_wave(opposed, grid_shape=(2, 2), sample_pitch_m=(1e-6, 1e-6))
        assert np.abs(np.asarray(field.u)).max() < 1e-12, (
            "two rays of equal |a|^2 and opposite phase must cancel; a surviving "
            "amplitude means the sign was lost"
        )


# --- Phase 4 Test C: launch translation and the Fourier shift phase ----------


class TestLaunchPositions:
    def test_multiple_launch_positions_produce_p_times_s_rays(self):
        """Requirement 1: P is real, and the population is exactly P * S."""
        field_in = metalens_field(AIR_CONFIG)
        spectrum, _ = band_limit_spectrum(
            decompose(field_in),
            direction_cosine_floor=FLOOR,
            max_optical_path_m=AIR_CONFIG.sensor_distance_m,
            precision=str(Precision.FP64),
        )
        density = sampling_density(spectrum, SamplingDensity.MAGNITUDE)
        sampler, _ = PositionalAngularSampler.build(
            spectrum, density_kind=SamplingDensity.MAGNITUDE, seed=3, samples_per_launch=32
        )
        launch = nested_aperture_launch_positions(
            5, aperture_radius_m=AIR_CONFIG.aperture_radius_m
        )
        plan = chunk_plan(launch_count=5, samples_per_launch=32, chunk_size=1024)
        total = 0
        seen_positions = set()
        for items in plan:
            bundle, ids = build_chunk_bundle(spectrum, density, sampler, items, launch)
            total += bundle.count
            for row in np.asarray(bundle.positions_m)[:, :2]:
                seen_positions.add((round(float(row[0]), 15), round(float(row[1]), 15)))
            assert ids.shape == (bundle.count,)
        assert total == 5 * 32
        assert len(seen_positions) == 5, (
            "every launch position must appear in the emitted population; "
            f"found {len(seen_positions)}"
        )

    def test_the_launch_phase_is_the_fourier_shift_phase(self):
        """Requirement 2 / Phase 4 Test C, against the closed form."""
        field_in = metalens_field(AIR_CONFIG)
        spectrum, _ = band_limit_spectrum(
            decompose(field_in),
            direction_cosine_floor=FLOOR,
            max_optical_path_m=AIR_CONFIG.sensor_distance_m,
            precision=str(Precision.FP64),
        )
        density = sampling_density(spectrum, SamplingDensity.UNIFORM)
        indices = np.array([100, 2000, 5000], dtype=np.int64)
        offset = np.array([[3.25e-6, -1.75e-6]])
        centred = spectrum_to_rays(spectrum, indices, density, launch_positions_xy_m=None)
        shifted = spectrum_to_rays(
            spectrum, indices, density, launch_positions_xy_m=offset
        )
        transverse = np.asarray(shifted.directions)[:, :2]
        expected = np.exp(
            1j * spectrum.wavenumber * (transverse @ offset[0])
        )
        ratio = np.asarray(shifted.amplitude) / np.asarray(centred.amplitude)
        assert np.allclose(ratio, expected, rtol=0, atol=1e-12)

    def test_launch_position_cancels_exactly_for_a_shift_invariant_system(
        self, host_optiland
    ):
        """A measured property of this composition, not an assumption.

        For any shift-invariant stack the wavelet sum's launch dependence cancels
        analytically: the ``exp(i k d.r_p)`` applied at emission is undone by the
        ``-d.x0`` term in the reconstruction, because ``x0 = r_p + Z d/d_n``. So
        the same modes launched from anywhere in the aperture must give the *same*
        field to round-off.

        This is why the benchmark draws independent angular samples per launch:
        with modes shared across launches, ``P`` would be an exact no-op here and
        a spatial convergence study would be measuring nothing. Recorded as a test
        so the claim in the report is checked rather than argued.
        """
        one, _, _, _ = _traced_field(AIR_CONFIG, launch_positions=[[0.0, 0.0]])
        many, _, _, _ = _traced_field(
            AIR_CONFIG,
            launch_positions=nested_aperture_launch_positions(
                4, aperture_radius_m=AIR_CONFIG.aperture_radius_m
            ).positions_xy_m,
        )
        error = _relative_field_error(many.u, one.u)
        assert error < 1.0e-11, f"launch position did not cancel: {error:.3e}"

    def test_the_launch_sequence_is_nested_and_inside_the_aperture(self):
        big = nested_aperture_launch_positions(
            128, aperture_radius_m=AIR_CONFIG.aperture_radius_m
        )
        for count in (1, 2, 7, 31, 64):
            assert np.array_equal(
                nested_aperture_launch_positions(
                    count, aperture_radius_m=AIR_CONFIG.aperture_radius_m
                ).positions_xy_m,
                big.positions_xy_m[:count],
            )
        radii = np.hypot(*big.positions_xy_m.T)
        assert radii.max() <= AIR_CONFIG.aperture_radius_m
        # Rejection, not clamping: nothing may pile up on the rim.
        assert (radii > 0.999 * AIR_CONFIG.aperture_radius_m).sum() <= 1


# --- Phase 4 Test D: normalization -------------------------------------------


class TestNormalization:
    """Requirement 8 / Phase 4 Test D: growing P must not multiply the power."""

    def _psf_power(self, launches: int, samples: int, seed: int = 11) -> float:
        config = AIR_CONFIG
        field_in = metalens_field(config)
        spectrum, _ = band_limit_spectrum(
            decompose(field_in),
            direction_cosine_floor=FLOOR,
            max_optical_path_m=config.sensor_distance_m,
            precision=str(Precision.FP64),
        )
        density = sampling_density(spectrum, SamplingDensity.MAGNITUDE)
        sampler, _ = PositionalAngularSampler.build(
            spectrum,
            density_kind=SamplingDensity.MAGNITUDE,
            seed=seed,
            samples_per_launch=samples,
        )
        launch = nested_aperture_launch_positions(
            launches, aperture_radius_m=config.aperture_radius_m
        )
        sensor = ReferencePlane(name="sensor", z_m=config.sensor_distance_m)
        reconstruction = StreamingReconstruction(
            grid_shape=config.grid_shape,
            sample_pitch_m=config.pitch_pair,
            plane=sensor,
            wavelength_m=config.wavelength_m,
            namespace=spectrum.namespace,
            complex_dtype=spectrum.dtype,
            total_rays=launches * samples,
        )
        for items in chunk_plan(
            launch_count=launches, samples_per_launch=samples, chunk_size=4096
        ):
            bundle, ids = build_chunk_bundle(spectrum, density, sampler, items, launch)
            advanced = _advance_analytically(bundle, config.sensor_distance_m, sensor)
            reconstruction.add_chunk(
                CoherentRayBatch(
                    bundle=advanced,
                    ray_id=ids,
                    valid=np.ones(advanced.count, dtype=bool),
                )
            )
        result = reconstruction.finalize()
        return float(np.sum(np.abs(np.asarray(result.field.u)) ** 2))

    def test_growing_the_spatial_count_does_not_multiply_the_power(self):
        oracle = reference_field(AIR_CONFIG, direction_cosine_floor=FLOOR)
        reference = float(np.sum(np.abs(np.asarray(oracle.u)) ** 2))
        powers = {count: self._psf_power(count, 256) for count in (1, 2, 4, 8)}
        ratios = {count: value / reference for count, value in powers.items()}
        # A missing 1/P would make these 1, 2, 4, 8. The Monte Carlo variance bias
        # is positive and *falls* with the ray count, so the ratios must approach 1
        # from above and never scale with P.
        assert all(0.5 < ratio < 6.0 for ratio in ratios.values()), ratios
        assert ratios[8] < ratios[1], f"power error must fall with the ray count: {ratios}"

    def test_the_reconstruction_refuses_a_ray_count_that_does_not_match(self):
        config = AIR_CONFIG
        reconstruction = StreamingReconstruction(
            grid_shape=config.grid_shape,
            sample_pitch_m=config.pitch_pair,
            plane=ReferencePlane(name="s", z_m=config.sensor_distance_m),
            wavelength_m=config.wavelength_m,
            namespace=decompose(metalens_field(config)).namespace,
            complex_dtype=decompose(metalens_field(config)).dtype,
            total_rays=999,
        )
        bundle = RayBundle(
            positions_m=np.zeros((4, 3)),
            directions=np.tile(np.array([0.0, 0.0, 1.0]), (4, 1)),
            wavelength_m=config.wavelength_m,
            reference_plane=ReferencePlane(name="s", z_m=config.sensor_distance_m),
            frame=Frame(axis_order="flat per-ray arrays"),
            amplitude=np.ones(4, dtype=np.complex128),
            optical_path_length_m=np.zeros(4),
            optical_path_length_reference="zero at 's'",
        )
        reconstruction.add_chunk(
            CoherentRayBatch(
                bundle=bundle, ray_id=np.arange(4), valid=np.ones(4, dtype=bool)
            )
        )
        with pytest.raises(ContractError) as raised:
            reconstruction.finalize()
        assert raised.value.code == "SHAPE_MISMATCH"


def _advance_analytically(bundle: RayBundle, distance_m: float, plane: ReferencePlane) -> RayBundle:
    """Straight-line free-space advance, for tests that must exclude the solver."""
    directions = np.asarray(bundle.directions)
    positions = np.asarray(bundle.positions_m)
    axial = directions[:, 2]
    return RayBundle(
        positions_m=np.column_stack(
            [
                positions[:, 0] + distance_m * directions[:, 0] / axial,
                positions[:, 1] + distance_m * directions[:, 1] / axial,
                np.full(bundle.count, plane.z_m),
            ]
        ),
        directions=directions,
        wavelength_m=bundle.wavelength_m,
        reference_plane=plane,
        frame=Frame(axis_order="flat per-ray arrays"),
        amplitude=bundle.amplitude,
        optical_path_length_m=distance_m / axial,
        optical_path_length_reference=bundle.optical_path_length_reference,
        normalization=bundle.normalization,
        reconstruction_normalization=bundle.reconstruction_normalization,
    )


# --- density conventions -----------------------------------------------------


class TestSamplingDensity:
    """Requirements 9 and 10: both densities must converge to the same field."""

    @pytest.mark.parametrize(
        "density", [SamplingDensity.MAGNITUDE, SamplingDensity.UNIFORM]
    )
    def test_both_densities_approach_the_same_analytic_reference(self, density):
        reference = np.abs(
            np.asarray(reference_field(AIR_CONFIG, direction_cosine_floor=FLOOR).u)
        ) ** 2
        errors = []
        for samples in (256, 4096):
            power = _monte_carlo_psf(AIR_CONFIG, 8, samples, density, seed=5)
            errors.append(1.0 - normalized_cross_correlation(power, reference))
        assert errors[-1] < errors[0], (
            f"{density} must improve with the sample count: {errors}"
        )
        assert errors[-1] < 1.0e-2, f"{density} at N=32768 stalled at 1-NCC={errors[-1]:.3e}"

    def test_magnitude_importance_sampling_is_the_more_efficient_one_here(self):
        """Measured, not asserted as folklore.

        The paper reports faster convergence for a concentrated spectrum. This
        one is concentrated (NA 0.196 out of a full hemisphere), so ``p_mag``
        should win -- and if it ever does not, that is a finding about the
        importance weight, not noise to dismiss.
        """
        reference = np.abs(
            np.asarray(reference_field(AIR_CONFIG, direction_cosine_floor=FLOOR).u)
        ) ** 2
        magnitude = 1.0 - normalized_cross_correlation(
            _monte_carlo_psf(AIR_CONFIG, 8, 1024, SamplingDensity.MAGNITUDE, seed=5),
            reference,
        )
        uniform = 1.0 - normalized_cross_correlation(
            _monte_carlo_psf(AIR_CONFIG, 8, 1024, SamplingDensity.UNIFORM, seed=5),
            reference,
        )
        assert magnitude < uniform, (
            f"p_mag {magnitude:.3e} did not beat p_uni {uniform:.3e} on a "
            "concentrated spectrum; investigate the importance weight"
        )

    def test_dropping_the_importance_weight_biases_the_result(self):
        """A negative control that can actually fail (CHE-44).

        Under ``p_mag`` the ``1/p`` factor is not a constant, so omitting it must
        change the answer. Under ``p_uni`` it *is* a constant and omitting it is
        only a global scale -- which is exactly why the control is run on
        ``p_mag``.
        """
        from couplers.wave_to_ray import SamplingPerturbation

        reference = np.abs(
            np.asarray(reference_field(AIR_CONFIG, direction_cosine_floor=FLOOR).u)
        ) ** 2
        good = _monte_carlo_psf(AIR_CONFIG, 4, 1024, SamplingDensity.MAGNITUDE, seed=9)
        bad = _monte_carlo_psf(
            AIR_CONFIG,
            4,
            1024,
            SamplingDensity.MAGNITUDE,
            seed=9,
            perturbation=SamplingPerturbation(apply_importance_weight=False),
        )
        assert normalized_cross_correlation(good, reference) > normalized_cross_correlation(
            bad, reference
        )


def _monte_carlo_psf(config, launches, samples, density_kind, *, seed, perturbation=None):
    """A Monte Carlo PSF through the streaming estimator, advanced analytically."""
    from couplers.wave_to_ray import SamplingPerturbation

    perturbation = perturbation or SamplingPerturbation()
    spectrum, _ = band_limit_spectrum(
        decompose(metalens_field(config)),
        direction_cosine_floor=FLOOR,
        max_optical_path_m=config.sensor_distance_m,
        precision=str(Precision.FP64),
    )
    density = sampling_density(spectrum, density_kind)
    sampler, _ = PositionalAngularSampler.build(
        spectrum, density_kind=density_kind, seed=seed, samples_per_launch=samples
    )
    launch = nested_aperture_launch_positions(
        launches, aperture_radius_m=config.aperture_radius_m
    )
    sensor = ReferencePlane(name="sensor", z_m=config.sensor_distance_m)
    reconstruction = StreamingReconstruction(
        grid_shape=config.grid_shape,
        sample_pitch_m=config.pitch_pair,
        plane=sensor,
        wavelength_m=config.wavelength_m,
        namespace=spectrum.namespace,
        complex_dtype=spectrum.dtype,
        total_rays=launches * samples,
    )
    for items in chunk_plan(
        launch_count=launches, samples_per_launch=samples, chunk_size=8192
    ):
        parts = []
        for item in items:
            indices = sampler.indices(
                launch_index=item.launch_index, start=item.start, stop=item.stop
            )
            parts.append(
                spectrum_to_rays(
                    spectrum,
                    indices,
                    density,
                    launch_positions_xy_m=launch.positions_xy_m[
                        item.launch_index : item.launch_index + 1
                    ],
                    perturbation=perturbation,
                )
            )
        bundle = parts[0] if len(parts) == 1 else _concatenate(parts)
        advanced = _advance_analytically(bundle, config.sensor_distance_m, sensor)
        reconstruction.add_chunk(
            CoherentRayBatch(
                bundle=advanced,
                ray_id=np.arange(advanced.count, dtype=np.int64),
                valid=np.ones(advanced.count, dtype=bool),
            )
        )
    result = reconstruction.finalize()
    return np.abs(np.asarray(result.field.u)) ** 2


def _concatenate(parts: list[RayBundle]) -> RayBundle:
    return RayBundle(
        positions_m=np.concatenate([part.positions_m for part in parts]),
        directions=np.concatenate([part.directions for part in parts]),
        wavelength_m=parts[0].wavelength_m,
        reference_plane=parts[0].reference_plane,
        frame=parts[0].frame,
        amplitude=np.concatenate([part.amplitude for part in parts]),
        optical_path_length_m=np.concatenate([part.optical_path_length_m for part in parts]),
        optical_path_length_reference=parts[0].optical_path_length_reference,
        normalization=parts[0].normalization,
        reconstruction_normalization=parts[0].reconstruction_normalization,
    )


# --- the Optiland handoff ----------------------------------------------------


class TestOptilandHandoff:
    def test_the_complex_amplitude_survives_the_trace_unchanged(self, host_optiland):
        """Requirement 3: bit-identical in, bit-identical out."""
        _, traced, diagnostics, _ = _traced_field(
            AIR_CONFIG,
            launch_positions=[[1e-6, -2e-6]],
            indices=np.arange(0, 7825, 97, dtype=np.int64),
        )
        field_in = metalens_field(AIR_CONFIG)
        spectrum, _ = band_limit_spectrum(
            decompose(field_in),
            direction_cosine_floor=FLOOR,
            max_optical_path_m=AIR_CONFIG.sensor_distance_m,
            precision=str(Precision.FP64),
        )
        density = sampling_density(spectrum, SamplingDensity.UNIFORM)
        original = spectrum_to_rays(
            spectrum,
            np.arange(0, 7825, 97, dtype=np.int64),
            density,
            launch_positions_xy_m=np.array([[1e-6, -2e-6]]),
        )
        assert np.array_equal(
            np.asarray(traced.bundle.amplitude), np.asarray(original.amplitude)
        ), "the amplitude is a sidecar and must come back bit-identical"
        assert "sidecar" in diagnostics["amplitude_handling"]

    def test_ray_identity_and_order_survive_the_pinned_solver(self, host_optiland):
        """Order preservation is *checked*, not assumed (Phase 1)."""
        _, traced, _, _ = _traced_field(
            AIR_CONFIG,
            launch_positions=[[0.0, 0.0], [3e-6, 4e-6]],
            indices=np.arange(0, 400, 7, dtype=np.int64),
        )
        ids = np.asarray(traced.ray_id)
        assert np.array_equal(ids, np.arange(ids.size))
        assert bool(np.asarray(traced.valid).all())

    def test_the_optical_path_matches_the_analytic_geometry(self, host_optiland):
        """Requirement 5: sign, scale and units of the accumulated path.

        The analytic value for a straight ray to a plane at ``Z`` is ``Z / d_n``
        per air layer and ``n t / d_n'`` inside glass, with ``d_n'`` from Snell.
        Both configurations are checked, so the index weighting is exercised and
        not just the geometric distance.
        """
        for config in (AIR_CONFIG, SLAB_CONFIG):
            _, traced, _, _ = _traced_field(
                config,
                launch_positions=[[2e-6, -1e-6]],
                indices=np.arange(0, 7825, 311, dtype=np.int64),
            )
            field_in = metalens_field(config)
            spectrum, _ = band_limit_spectrum(
                decompose(field_in),
                direction_cosine_floor=FLOOR,
                max_optical_path_m=config.sensor_distance_m,
                precision=str(Precision.FP64),
            )
            density = sampling_density(spectrum, SamplingDensity.UNIFORM)
            emitted = spectrum_to_rays(
                spectrum,
                np.arange(0, 7825, 311, dtype=np.int64),
                density,
                launch_positions_xy_m=np.array([[2e-6, -1e-6]]),
            )
            transverse = np.asarray(emitted.directions)[:, :2]
            radial = (transverse**2).sum(axis=1)
            expected = np.zeros(emitted.count)
            for layer in config.scaled_layers:
                index = layer.refractive_index
                axial = np.sqrt(index**2 - radial) / index
                expected += index * layer.thickness_m / axial
            observed = np.asarray(traced.bundle.optical_path_length_m)
            assert np.allclose(observed, expected, rtol=1e-11, atol=1e-15), (
                f"{config.name}: OPL disagrees with the analytic path by "
                f"{np.abs(observed - expected).max():.3e} m"
            )
            assert observed.min() > 0.0, "a forward ray's accumulated path must be positive"

    def test_the_opl_reference_is_declared_and_names_the_launch_plane(self, host_optiland):
        _, traced, _, _ = _traced_field(
            AIR_CONFIG, launch_positions=[[0.0, 0.0]], indices=np.array([0, 1, 2])
        )
        declared = traced.bundle.optical_path_length_reference
        assert "metalens_exit" in declared
        assert "opd = 0" in declared
        assert declared == declared_launch_opl_reference(
            ReferencePlane(name="metalens_exit", z_m=0.0)
        )

    def test_a_launch_plane_that_is_not_the_first_surface_is_refused(self, host_optiland):
        """A silently different optical system is worse than an error."""
        field_in = metalens_field(AIR_CONFIG)
        spectrum, _ = band_limit_spectrum(
            decompose(field_in),
            direction_cosine_floor=FLOOR,
            max_optical_path_m=AIR_CONFIG.sensor_distance_m,
            precision=str(Precision.FP64),
        )
        density = sampling_density(spectrum, SamplingDensity.UNIFORM)
        bundle = spectrum_to_rays(spectrum, np.array([0, 1]), density)
        moved = dataclasses.replace(
            bundle, reference_plane=ReferencePlane(name="wrong", z_m=1e-3)
        )
        batch = CoherentRayBatch(
            bundle=moved, ray_id=np.arange(2), valid=np.ones(2, dtype=bool)
        )
        lens = build_optiland_system(optical_system_spec(AIR_CONFIG))
        plans = plan_trace_bridges(batch, home=C_RAY_TO_WAVE_CAPABILITIES, device=HOST)
        with pytest.raises(ContractError) as raised:
            trace_ray_batch(
                batch,
                lens,
                image_plane=ReferencePlane(name="s", z_m=AIR_CONFIG.sensor_distance_m),
                plans=plans,
            )
        assert raised.value.code == "REFERENCE_PLANE_MISMATCH"

    def test_the_built_system_places_its_surfaces_where_the_configuration_says(self):
        for config in (AIR_CONFIG, SLAB_CONFIG):
            positions = surface_positions_m(build_optiland_system(optical_system_spec(config)))
            assert positions[0] == -math.inf, "surface 0 is the object surface"
            assert positions[1] == pytest.approx(0.0, abs=1e-15)
            assert positions[-1] == pytest.approx(config.sensor_distance_m, rel=1e-12)


class TestUnitBoundary:
    """One conversion boundary, and it round-trips."""

    @pytest.mark.parametrize("value", [0.0, 1e-9, 5e-5, 1.0, 1234.5])
    def test_length_conversions_round_trip(self, value):
        assert millimetres_to_metres(metres_to_millimetres(value)) == pytest.approx(
            value, rel=1e-15, abs=1e-30
        )

    def test_known_values_are_what_the_solver_units_say(self):
        assert metres_to_millimetres(50e-6) == pytest.approx(0.05)
        assert metres_to_micrometres(500e-9) == pytest.approx(0.5)
        assert micrometres_to_metres(0.5) == pytest.approx(500e-9)

    def test_the_wavelength_reaching_the_solver_is_micrometres(self, host_optiland):
        _, traced, _, _ = _traced_field(
            AIR_CONFIG, launch_positions=[[0.0, 0.0]], indices=np.array([0, 1])
        )
        assert traced.bundle.provenance["trace"]["wavelength_um"] == pytest.approx(0.5)
        assert traced.bundle.wavelength_m == pytest.approx(500e-9)

    def test_the_conventions_the_reconstruction_assumes_are_the_ones_declared(self):
        field = reference_field(AIR_CONFIG, direction_cosine_floor=FLOOR)
        assert field.reference_plane.z_m == pytest.approx(AIR_CONFIG.sensor_distance_m)
        assert field.frame.origin_rule
        # +z propagation: every retained ray direction has a positive axial cosine.
        spectrum, _ = band_limit_spectrum(
            decompose(metalens_field(AIR_CONFIG)),
            direction_cosine_floor=FLOOR,
            max_optical_path_m=AIR_CONFIG.sensor_distance_m,
            precision=str(Precision.FP64),
        )
        density = sampling_density(spectrum, SamplingDensity.UNIFORM)
        bundle = spectrum_to_rays(spectrum, enumerate_indices(density), density)
        assert float(np.asarray(bundle.directions)[:, 2].min()) >= FLOOR


class TestBackendState:
    def test_the_backend_state_is_observed_not_echoed(self):
        state = configure_optiland_execution(
            device=HOST, precision=Precision.FP64, enable_grad=False
        )
        assert state.observed_precision == "float64"
        assert state.grad_enabled is False
        assert be.get_precision() in (64, "float64")

    def test_a_float32_request_is_actually_applied(self):
        state = configure_optiland_execution(
            device=HOST, precision=Precision.FP32, enable_grad=False
        )
        assert state.observed_precision == "float32"
        configure_optiland_execution(
            device=HOST, precision=Precision.FP64, enable_grad=False
        )
