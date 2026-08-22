"""Chunking must not change the answer, and must not change the memory (CHE-70).

The two claims are separate and are tested separately, because a chunking bug
that hides behind Monte Carlo noise is exactly the failure this whole module
exists to make impossible:

*Numerical chunk equivalence* -- for a fixed seed, ``chunk_size`` changes neither
the sampled ray population (bit-identical indices) nor the reconstructed field
beyond floating-point summation order.

*Bounded memory* -- drawing ``n`` samples costs ``O(retained_bins + n)`` and never
materializes an ``(n, W)`` conditional table, for any ``n``.

The sampler is also held to the *distribution* it claims, against
``wave_to_ray.sampling_density``, so the memory-safe route is not quietly a
different estimator from the one CHE-25 characterized.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from core.precision import Precision
from couplers.coherent_batch import CoherentRayBatch
from couplers.contracts import ContractError, ReferencePlane
from couplers.streaming import (
    PositionalAngularSampler,
    StreamingReconstruction,
    band_limit_spectrum,
    build_chunk_bundle,
    chunk_plan,
    nested_aperture_launch_positions,
)
from couplers.wave_to_ray import (
    SamplingDensity,
    decompose,
    draw_indices,
    sampling_density,
)
from evaluation.metalens import AIR_CONFIG, metalens_field

pytestmark = [pytest.mark.coupler]

FLOOR = 1.0e-2


@pytest.fixture(scope="module")
def spectrum():
    limited, _ = band_limit_spectrum(
        decompose(metalens_field(AIR_CONFIG)),
        direction_cosine_floor=FLOOR,
        max_optical_path_m=AIR_CONFIG.sensor_distance_m,
        precision=str(Precision.FP64),
        phase_budget_rad=1.0e-2,
    )
    return limited


class TestChunkPlan:
    def test_every_ray_appears_exactly_once(self):
        for launches, samples, chunk in [
            (1, 1, 1), (5, 4, 8), (3, 10, 4), (7, 7, 7), (16, 1024, 4096), (2, 5, 100),
        ]:
            plan = chunk_plan(
                launch_count=launches, samples_per_launch=samples, chunk_size=chunk
            )
            ids = [
                identifier
                for items in plan
                for item in items
                for identifier in range(item.first_ray_id, item.first_ray_id + item.size)
            ]
            assert sorted(ids) == list(range(launches * samples)), (
                f"P={launches} S={samples} chunk={chunk} did not partition the rays"
            )

    def test_the_first_ray_id_matches_the_launch_and_sample_index(self):
        for items in chunk_plan(launch_count=4, samples_per_launch=6, chunk_size=12):
            for item in items:
                assert item.first_ray_id == item.launch_index * 6 + item.start

    def test_chunks_prefer_a_single_shape(self):
        """Constant shapes are what let a JIT compile once for a whole sweep."""
        plan = chunk_plan(launch_count=16, samples_per_launch=256, chunk_size=1024)
        sizes = {sum(item.size for item in items) for items in plan}
        assert sizes == {1024}

    def test_a_launch_larger_than_a_chunk_is_split_into_even_pieces(self):
        plan = chunk_plan(launch_count=2, samples_per_launch=1024, chunk_size=256)
        sizes = sorted({sum(item.size for item in items) for items in plan})
        assert sizes == [256]
        assert len(plan) == 8

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"launch_count": 0, "samples_per_launch": 1, "chunk_size": 1},
            {"launch_count": 1, "samples_per_launch": 0, "chunk_size": 1},
            {"launch_count": 1, "samples_per_launch": 1, "chunk_size": 0},
        ],
    )
    def test_a_nonpositive_count_is_refused(self, kwargs):
        with pytest.raises(ValueError):
            chunk_plan(**kwargs)


class TestPositionalSampler:
    def test_the_drawn_population_does_not_depend_on_the_chunk_boundaries(self, spectrum):
        """Requirement 12/13's first half: bit-identical indices, not merely the same law."""
        sampler, _ = PositionalAngularSampler.build(
            spectrum,
            density_kind=SamplingDensity.MAGNITUDE,
            seed=17,
            samples_per_launch=1000,
        )
        whole = sampler.indices(launch_index=5, start=0, stop=1000)
        for boundaries in ([0, 1000], [0, 1, 999, 1000], [0, 250, 500, 750, 1000],
                           [0, 333, 666, 1000]):
            pieces = [
                sampler.indices(launch_index=5, start=start, stop=stop)
                for start, stop in itertools.pairwise(boundaries)
            ]
            assert np.array_equal(np.concatenate(pieces), whole)

    def test_different_launches_draw_independent_sample_sets(self, spectrum):
        """What makes P a genuine convergence axis rather than an exact no-op."""
        sampler, _ = PositionalAngularSampler.build(
            spectrum,
            density_kind=SamplingDensity.MAGNITUDE,
            seed=17,
            samples_per_launch=512,
        )
        first = sampler.indices(launch_index=0, start=0, stop=512)
        second = sampler.indices(launch_index=1, start=0, stop=512)
        assert not np.array_equal(first, second)

    def test_a_different_seed_is_a_different_realization(self, spectrum):
        left, _ = PositionalAngularSampler.build(
            spectrum, density_kind=SamplingDensity.MAGNITUDE, seed=1, samples_per_launch=256
        )
        right, _ = PositionalAngularSampler.build(
            spectrum, density_kind=SamplingDensity.MAGNITUDE, seed=2, samples_per_launch=256
        )
        assert not np.array_equal(
            left.indices(launch_index=0, start=0, stop=256),
            right.indices(launch_index=0, start=0, stop=256),
        )

    @pytest.mark.parametrize(
        "density_kind", [SamplingDensity.MAGNITUDE, SamplingDensity.UNIFORM]
    )
    def test_the_memory_safe_sampler_reproduces_the_reference_distribution(
        self, spectrum, density_kind
    ):
        """Requirement 14's correctness half, as a chi-square goodness of fit.

        A max-relative-deviation bound would be the wrong test and would be
        flaky for a good reason: over 7825 bins whose expected counts are ~50,
        the *worst* bin is several Poisson sigma out by construction. The
        aggregate statistic is the one with a known null distribution --
        ``chi^2/dof`` is 1 with standard deviation ``sqrt(2/dof)`` = 1.6e-2 here,
        so the bound below is roughly 9 sigma: loose enough never to flake,
        tight enough that any real bias in the inverse-CDF route fails it.

        The same statistic is then computed against ``draw_indices``'s own
        frequencies, so the memory-safe route is held to the implementation
        CHE-25 characterized and not only to the analytic density.
        """
        from scipy import stats

        density = np.asarray(sampling_density(spectrum, density_kind), dtype=np.float64)
        density = density / density.sum()
        count = 400_000
        sampler, _ = PositionalAngularSampler.build(
            spectrum, density_kind=density_kind, seed=99, samples_per_launch=count
        )
        drawn = sampler.indices(launch_index=0, start=0, stop=count)
        assert drawn.min() >= 0 and drawn.max() < density.size

        observed = np.bincount(drawn, minlength=density.size).astype(np.float64)
        expected = density * count
        # The standard chi-square validity condition: bins with an expected count
        # below 5 are pooled rather than dropped, so no probability mass leaves
        # the test.
        keep = expected >= 5.0
        assert keep.sum() > 100
        pooled_observed, pooled_expected = observed[keep], expected[keep]
        if not keep.all():
            # Only pool when there is something to pool: an empty tail would put a
            # zero in the denominator, which is a NaN rather than a failure.
            pooled_observed = np.append(pooled_observed, observed[~keep].sum())
            pooled_expected = np.append(pooled_expected, expected[~keep].sum())
        degrees = pooled_expected.size - 1
        chi_square = float(
            ((pooled_observed - pooled_expected) ** 2 / pooled_expected).sum()
        )
        reduced = chi_square / degrees
        tolerance = 9.0 * np.sqrt(2.0 / degrees)
        assert abs(reduced - 1.0) < tolerance, (
            f"chi^2/dof = {reduced:.4f} against 1 +/- {tolerance:.4f} "
            f"(dof = {degrees}); p = {stats.chi2.sf(chi_square, degrees):.3e}"
        )

        reference = np.bincount(
            draw_indices(density, count, np.random.default_rng(4)), minlength=density.size
        ).astype(np.float64)
        pooled_reference = reference[keep]
        if not keep.all():
            pooled_reference = np.append(pooled_reference, reference[~keep].sum())
        # Two-sample chi-square between two equal-size draws. Both counts are
        # random, so Var(O1 - O2) = 2E while (O1 + O2) estimates 2E: the ratio has
        # expectation 1 per bin, the same null as the one-sample form above.
        two_sample = float(
            (
                (pooled_observed - pooled_reference) ** 2
                / (pooled_observed + pooled_reference).clip(min=1.0)
            ).sum()
        )
        assert abs(two_sample / degrees - 1.0) < 9.0 * np.sqrt(2.0 / degrees), (
            f"the memory-safe sampler and draw_indices differ: "
            f"two-sample chi^2/dof = {two_sample / degrees:.4f}, expected 1"
        )

    def test_drawing_a_huge_sample_allocates_only_the_sample(self, spectrum):
        """Requirement 14's memory half.

        ``O(retained_bins + n)``, checked by measuring the process's own RSS
        growth across a draw far larger than any chunk the sweep uses. An
        ``(n, W)`` conditional table would be ``n * 7825 * 8`` bytes -- 62 GB at
        this n -- so the bound is not a close call.
        """
        from core.resources import process_rss_bytes

        count = 1_000_000
        sampler, _ = PositionalAngularSampler.build(
            spectrum, density_kind=SamplingDensity.MAGNITUDE, seed=5, samples_per_launch=count
        )
        before = process_rss_bytes()
        drawn = sampler.indices(launch_index=0, start=0, stop=count)
        growth = process_rss_bytes() - before
        assert drawn.size == count
        forbidden = count * sampler.bin_count * 8
        assert growth < forbidden / 1000, (
            f"drawing {count} samples grew RSS by {growth} B; an (n, W) table would "
            f"have been {forbidden} B"
        )
        # Positive form of the same bound: a handful of int64/float64 arrays.
        assert growth < 200 * 1024**2

    def test_the_sampler_declares_its_own_properties(self, spectrum):
        sampler, _ = PositionalAngularSampler.build(
            spectrum, density_kind=SamplingDensity.MAGNITUDE, seed=5, samples_per_launch=8
        )
        record = sampler.as_dict()
        assert record["chunk_invariant"] is True
        assert record["bin_count"] == 7825
        assert "O(retained_bins + chunk)" in record["memory_complexity"]

    def test_a_sample_range_outside_the_launch_is_refused(self, spectrum):
        sampler, _ = PositionalAngularSampler.build(
            spectrum, density_kind=SamplingDensity.UNIFORM, seed=1, samples_per_launch=10
        )
        with pytest.raises(ValueError):
            sampler.indices(launch_index=0, start=0, stop=11)
        with pytest.raises(ValueError):
            sampler.indices(launch_index=0, start=5, stop=2)


class TestBandLimit:
    def test_the_limit_only_ever_removes_modes(self, spectrum):
        original = decompose(metalens_field(AIR_CONFIG))
        assert not bool(
            np.any(np.asarray(spectrum.propagating) & ~np.asarray(original.propagating))
        )

    def test_the_input_spectrum_is_not_mutated(self):
        original = decompose(metalens_field(AIR_CONFIG))
        before = int(original.propagating_count)
        band_limit_spectrum(
            original,
            direction_cosine_floor=0.5,
            max_optical_path_m=AIR_CONFIG.sensor_distance_m,
            precision=str(Precision.FP64),
        )
        assert int(original.propagating_count) == before

    def test_the_excluded_power_is_reported_not_hidden(self):
        _, band = band_limit_spectrum(
            decompose(metalens_field(AIR_CONFIG)),
            direction_cosine_floor=0.5,
            max_optical_path_m=AIR_CONFIG.sensor_distance_m,
            precision=str(Precision.FP64),
        )
        assert band.excluded_bin_count > 1000
        assert band.excluded_power_fraction > 0.0

    def test_a_floor_that_excludes_everything_is_refused(self):
        with pytest.raises(ContractError) as raised:
            band_limit_spectrum(
                decompose(metalens_field(AIR_CONFIG)),
                direction_cosine_floor=1.5,
                max_optical_path_m=AIR_CONFIG.sensor_distance_m,
                precision=str(Precision.FP64),
            )
        assert raised.value.code == "EMPTY_ENSEMBLE"


class TestChunkEquivalence:
    """Requirements 12 and 13, on complex fields rather than normalized PSFs."""

    @staticmethod
    def _field(spectrum, launches: int, samples: int, chunk: int, seed: int = 21):
        density = sampling_density(spectrum, SamplingDensity.MAGNITUDE)
        sampler, _ = PositionalAngularSampler.build(
            spectrum,
            density_kind=SamplingDensity.MAGNITUDE,
            seed=seed,
            samples_per_launch=samples,
        )
        launch = nested_aperture_launch_positions(
            launches, aperture_radius_m=AIR_CONFIG.aperture_radius_m
        )
        plane = ReferencePlane(name="p0", z_m=0.0)
        reconstruction = StreamingReconstruction(
            grid_shape=AIR_CONFIG.grid_shape,
            sample_pitch_m=AIR_CONFIG.pitch_pair,
            plane=plane,
            wavelength_m=AIR_CONFIG.wavelength_m,
            namespace=spectrum.namespace,
            complex_dtype=spectrum.dtype,
            total_rays=launches * samples,
        )
        for items in chunk_plan(
            launch_count=launches, samples_per_launch=samples, chunk_size=chunk
        ):
            bundle, ids = build_chunk_bundle(spectrum, density, sampler, items, launch)
            reconstruction.add_chunk(
                CoherentRayBatch(
                    bundle=bundle, ray_id=ids, valid=np.ones(bundle.count, dtype=bool)
                )
            )
        return reconstruction.finalize()

    def test_the_complex_field_is_unchanged_by_the_chunk_size(self, spectrum):
        launches, samples = 8, 512
        total = launches * samples
        whole = self._field(spectrum, launches, samples, total)
        reference = np.asarray(whole.field.u)
        norm = float(np.linalg.norm(reference))
        for divisor in (2, 8, 32):
            other = self._field(spectrum, launches, samples, total // divisor)
            error = float(np.linalg.norm(np.asarray(other.field.u) - reference)) / norm
            assert error < 1.0e-13, (
                f"chunk = N/{divisor} moved the complex field by {error:.3e}; only "
                "summation order may differ"
            )
            assert other.total_rays == whole.total_rays

    def test_the_psf_is_unchanged_by_the_chunk_size(self, spectrum):
        from evaluation.metalens import (
            normalized_cross_correlation,
        )

        launches, samples = 8, 512
        whole = np.abs(np.asarray(self._field(spectrum, launches, samples, 4096).field.u)) ** 2
        eighth = np.abs(np.asarray(self._field(spectrum, launches, samples, 512).field.u)) ** 2
        assert 1.0 - normalized_cross_correlation(eighth, whole) < 1.0e-14

    def test_the_normalization_is_the_estimators_not_the_chunks(self, spectrum):
        """A 1/N_chunk would scale the field by the number of chunks."""
        launches, samples = 4, 256
        whole = self._field(spectrum, launches, samples, launches * samples)
        split = self._field(spectrum, launches, samples, 64)
        assert float(np.abs(np.asarray(split.field.u)).sum()) == pytest.approx(
            float(np.abs(np.asarray(whole.field.u)).sum()), rel=1e-12
        )
        assert split.chunk_count > whole.chunk_count
        record = split.field.provenance["streaming"]
        assert "normalization='none'" in record["normalization_rule"]
        assert "1/N_total" in record["normalization_rule"]

    def test_the_chunk_count_and_sizes_are_recorded(self, spectrum):
        result = self._field(spectrum, 4, 256, 256)
        assert result.chunk_count == 4
        assert result.as_dict()["distinct_chunk_sizes"] == [256]

    def test_finalizing_without_a_chunk_is_refused(self, spectrum):
        reconstruction = StreamingReconstruction(
            grid_shape=AIR_CONFIG.grid_shape,
            sample_pitch_m=AIR_CONFIG.pitch_pair,
            plane=ReferencePlane(name="p", z_m=0.0),
            wavelength_m=AIR_CONFIG.wavelength_m,
            namespace=spectrum.namespace,
            complex_dtype=spectrum.dtype,
            total_rays=4,
        )
        with pytest.raises(ContractError) as raised:
            reconstruction.finalize()
        assert raised.value.code == "EMPTY_ENSEMBLE"


class TestClippedRays:
    """A clipped ray keeps its place in N_total and contributes nothing."""

    def test_zeroed_amplitudes_do_not_change_the_denominator(self, spectrum):
        density = sampling_density(spectrum, SamplingDensity.UNIFORM)
        sampler, _ = PositionalAngularSampler.build(
            spectrum, density_kind=SamplingDensity.UNIFORM, seed=3, samples_per_launch=64
        )
        launch = nested_aperture_launch_positions(
            2, aperture_radius_m=AIR_CONFIG.aperture_radius_m
        )
        items = chunk_plan(launch_count=2, samples_per_launch=64, chunk_size=128)[0]
        bundle, ids = build_chunk_bundle(spectrum, density, sampler, items, launch)

        import dataclasses

        half = np.asarray(bundle.amplitude).copy()
        half[::2] = 0.0
        masked = dataclasses.replace(bundle, amplitude=half)
        valid = np.ones(bundle.count, dtype=bool)
        valid[::2] = False

        plane = ReferencePlane(name="p0", z_m=0.0)
        reconstruction = StreamingReconstruction(
            grid_shape=AIR_CONFIG.grid_shape,
            sample_pitch_m=AIR_CONFIG.pitch_pair,
            plane=plane,
            wavelength_m=AIR_CONFIG.wavelength_m,
            namespace=spectrum.namespace,
            complex_dtype=spectrum.dtype,
            total_rays=128,
        )
        reconstruction.add_chunk(
            CoherentRayBatch(bundle=masked, ray_id=ids, valid=valid)
        )
        result = reconstruction.finalize()
        assert result.total_rays == 128
        assert result.valid_rays == 64
        assert "survival fraction" in result.field.provenance["streaming"]["clipped_ray_policy"]
