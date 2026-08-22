"""Bounded-memory streaming of the coherent ray estimator (CHE-70, Phases 5-13).

The estimator itself is **not** implemented here. ``C_WAVE_TO_RAY``
(:mod:`couplers.wave_to_ray`) and ``C_RAY_TO_WAVE``
(:mod:`couplers.ray_to_wave`) are the verified physics
and are called unchanged. What this module owns is everything needed to run that
estimator over a ray population too large to hold at once:

* a **declared band limit** on the angular spectrum, without which the
  reconstruction's phase is destroyed in float32 (see below -- this is measured,
  not precautionary);
* **nested spatial launch positions** covering the active aperture (Phase 6);
* a **chunk-invariant angular sampler** whose drawn population depends on the
  seed and the ray index, and on nothing else (Phases 8, 12);
* a **streaming reconstruction** that keeps one ``(ny, nx)`` accumulator and
  releases every chunk (Phases 9, 10, 11).

Why a grazing-mode band limit is required, and where the number comes from
-------------------------------------------------------------------------
``C_RAY_TO_WAVE`` forms each ray's constant phase as ``k (OPL_i - d_i . x0_i)``.
For a mode whose axial direction cosine is ``d_n``, propagating a distance ``Z``
makes both terms ``~Z/d_n`` while their difference is ``Z d_n``. Near grazing the
two nearly cancel, so the *relative* precision of the inputs sets the *absolute*
error of the phase: ``delta_phase ~ eps * k * Z / d_n``.

That is not hypothetical. On the CHE-70 100x100 metalens grid, eight bins land
on ``d_u^2 + d_v^2 = 1`` exactly -- the Pythagorean triples ``(30,40)``,
``(40,30)`` and their sign variants -- and survive the ``radial < 1`` evanescent
cut at ``d_n = 1.05e-8``. Their OPL over a 50 um propagation is **4745 m**, and
they carry 2.3e-7 of the field's power. Measured, float64, full enumeration
against the analytic angular-spectrum oracle:

    no band limit          2.8e-09 relative field error
    d_n floor = 1e-2       8.9e-14 relative field error

In float32 the same 4745 m OPL is a phase of 6e10 rad, whose representation error
alone is ~7e3 rad: those eight bins are pure noise injected at full importance
weight. So the floor is a correctness requirement for the GPU path, not a tidying
step, and :func:`grazing_floor_for_phase_budget` derives it from the precision
and the propagation distance rather than picking a round number.

The floor is applied to the ray ensemble **and to the oracle alike**, so the two
routes decompose the same field over the same set of modes. It is a declared
property of the benchmark configuration, not a change to either coupler.

Chunk invariance, and what it is not
------------------------------------
Two different claims are kept apart, because conflating them is how a chunking
bug hides behind Monte Carlo noise:

*Numerical chunk equivalence* -- for a fixed seed, ``chunk_size`` does not change
the sampled ray population at all, and changes the reconstructed field only by
floating-point summation order. This is testable to round-off and
``tests/test_streaming_estimator.py`` tests it.

*Monte Carlo realization variability* -- two different seeds give different
fields. That is the estimator's variance and it is reported as a spread over
seeds, never used to excuse a chunking discrepancy.

Invariance holds because the sampler is *positional*: ray ``j`` of launch ``p``
draws from a PCG64 stream advanced to ``p * S + j``, so its value is a pure
function of ``(seed, p, j)``. Nothing about chunk boundaries enters.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from core.arrays import (
    device_of,
    dtype_of,
    namespace_of,
    numpy_dtype,
    xp_for,
)
from core.boundary import (
    ComplexField,
    ContractCode,
    ContractError,
    Frame,
    RayBundle,
    ReferencePlane,
)
from core.precision import Precision
from couplers.ray_to_wave import (
    DEFAULT_KSPACE_OVERSAMPLE,
    Projection,
    Reconstruction,
    ray_to_wave,
)
from couplers.wave_to_ray import (
    AngularSpectrum,
    SamplingDensity,
    sampling_density,
    spectrum_to_rays,
)

__all__ = [
    "BandLimit",
    "ChunkWorkItem",
    "LaunchGeometry",
    "PositionalAngularSampler",
    "StreamingReconstruction",
    "StreamingResult",
    "band_limit_spectrum",
    "chunk_plan",
    "grazing_floor_for_phase_budget",
    "nested_aperture_launch_positions",
]

#: Phase budget the grazing floor is derived against, in radians. A hundredth of
#: a radian is ~1/600 of a wave: far below anything a PSF comparison at NCC 0.99
#: can resolve, and far above the float32 noise floor of the modes that matter.
DEFAULT_PHASE_BUDGET_RAD = 1.0e-2


def grazing_floor_for_phase_budget(
    *,
    wavelength_m: float,
    max_optical_path_m: float,
    precision: Precision,
    phase_budget_rad: float = DEFAULT_PHASE_BUDGET_RAD,
) -> float:
    """The smallest admissible axial direction cosine, derived not chosen.

    ``delta_phase ~ eps * k * Z / d_n`` (module docstring), so requiring
    ``delta_phase <= phase_budget_rad`` gives
    ``d_n >= eps * k * Z / phase_budget_rad``.

    ``max_optical_path_m`` is the *axial* extent the rays traverse, not the
    grazing OPL -- the grazing OPL is what this bound exists to keep finite.
    """
    if phase_budget_rad <= 0.0:
        raise ValueError("phase_budget_rad must be positive")
    epsilon = float(np.finfo(numpy_dtype(precision.real_dtype)).eps)
    wavenumber = 2.0 * math.pi / wavelength_m
    return epsilon * wavenumber * float(max_optical_path_m) / phase_budget_rad


@dataclass(frozen=True)
class BandLimit:
    """A declared restriction of an angular spectrum, and its measured cost.

    The cost is not optional. Excluding modes discards power, and a benchmark
    that quietly narrowed its own band limit until it agreed with its reference
    would be measuring nothing -- so the excluded fraction and the excluded bin
    count travel with the restricted spectrum and land in the manifest.
    """

    direction_cosine_floor: float
    phase_budget_rad: float
    precision: str
    excluded_bin_count: int
    excluded_power_fraction: float
    retained_bin_count: int
    max_optical_path_m: float
    #: The largest OPL any retained mode can accumulate over the declared axial
    #: extent, ``Z / d_n_min``. The quantity the floor exists to bound.
    max_retained_optical_path_m: float

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def band_limit_spectrum(
    spectrum: AngularSpectrum, *, direction_cosine_floor: float, **provenance: Any
) -> tuple[AngularSpectrum, BandLimit]:
    """Exclude near-grazing modes from ``spectrum``'s propagating set.

    Returns a new :class:`AngularSpectrum` -- the input is frozen and is not
    modified -- whose ``propagating`` mask additionally requires
    ``d_n >= direction_cosine_floor``, with ``evanescent_power_fraction``
    recomputed so it still means "power this decomposition cannot represent as
    rays".

    The same mask must be applied to any reference the result is compared
    against. :func:`retained_mask` returns it for that purpose.
    """
    xp = spectrum.xp
    direction_v, direction_u = xp.meshgrid(
        spectrum.direction_v, spectrum.direction_u, indexing="ij"
    )
    radial = direction_u**2 + direction_v**2
    axial = xp.sqrt(xp.clip(1.0 - radial, 0.0, None))
    retained = spectrum.propagating & (axial >= direction_cosine_floor)
    if not bool(xp.any(retained)):
        raise ContractError(
            ContractCode.EMPTY_ENSEMBLE,
            (
                f"a grazing floor of {direction_cosine_floor!r} excludes every "
                "propagating mode; the field cannot be represented by rays under "
                "this band limit"
            ),
            declaration="direction_cosine_floor",
        )

    mode_power = xp.abs(spectrum.spectrum) ** 2
    total = float(xp.sum(mode_power))
    excluded = spectrum.propagating & ~retained
    limited = dataclasses.replace(
        spectrum,
        propagating=retained,
        evanescent_power_fraction=(
            float(xp.sum(mode_power[~retained]) / total) if total > 0.0 else 0.0
        ),
    )
    band = BandLimit(
        direction_cosine_floor=float(direction_cosine_floor),
        phase_budget_rad=float(provenance.get("phase_budget_rad", float("nan"))),
        precision=str(provenance.get("precision", spectrum.dtype.precision)),
        excluded_bin_count=int(xp.count_nonzero(excluded)),
        excluded_power_fraction=(
            float(xp.sum(mode_power[excluded]) / total) if total > 0.0 else 0.0
        ),
        retained_bin_count=int(xp.count_nonzero(retained)),
        max_optical_path_m=float(provenance.get("max_optical_path_m", float("nan"))),
        max_retained_optical_path_m=(
            float(provenance.get("max_optical_path_m", float("nan")))
            / float(direction_cosine_floor)
            if direction_cosine_floor > 0.0
            else float("inf")
        ),
    )
    return limited, band


@dataclass(frozen=True)
class LaunchGeometry:
    """Spatial launch positions covering an aperture, nested by construction.

    ``positions_xy_m`` is ``(P, 2)`` on the host. Nesting -- ``positions[:P1]``
    is a valid smaller sampling of the same aperture for every ``P1 <= P`` -- is
    what makes a spatial convergence study a refinement rather than a sequence of
    unrelated experiments, and it comes from taking prefixes of one low-discrepancy
    sequence rather than redrawing.

    ``quadrature_weight`` is uniform, and that is a statement about the sequence:
    the R2 sequence is asymptotically uniform over the square, rejection to the
    disc leaves it uniform over the disc, so equal weights are the correct
    quadrature and the ``1/N`` of the estimator carries them. A non-uniform
    spatial density would need its own ``1/q`` factor here.
    """

    positions_xy_m: np.ndarray
    aperture_radius_m: float
    scheme: str
    quadrature_weight: str
    accepted_of_proposed: tuple[int, int]

    @property
    def count(self) -> int:
        return int(self.positions_xy_m.shape[0])

    def prefix(self, count: int) -> LaunchGeometry:
        if count > self.count:
            raise ValueError(f"cannot take {count} of {self.count} launch positions")
        return dataclasses.replace(self, positions_xy_m=self.positions_xy_m[:count])

    def as_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "aperture_radius_m": self.aperture_radius_m,
            "scheme": self.scheme,
            "quadrature_weight": self.quadrature_weight,
            "accepted_of_proposed": list(self.accepted_of_proposed),
            "nested": "positions[:P1] is a valid sampling for every P1 <= P",
        }


#: Roberts' generalized golden ratio in two dimensions: the plastic number, the
#: real root of x^3 = x + 1. The R2 sequence built from it is low-discrepancy,
#: deterministic, needs no state beyond the index, and its prefixes are nested.
_PLASTIC_NUMBER = 1.324717957244746025960908854


def nested_aperture_launch_positions(
    count: int,
    *,
    aperture_radius_m: float,
    offset: float = 0.5,
) -> LaunchGeometry:
    """``count`` nested launch positions inside a circular aperture.

    The R2 sequence is generated over the bounding square and rejected to the
    disc. Rejection preserves nesting because it never reorders: the first
    ``P1`` accepted points are a prefix of the first ``P2`` for ``P1 <= P2``.

    Points outside the aperture are *rejected rather than clamped*. Clamping
    would pile samples onto the rim, and Phase 6's rule is that padded or
    inactive regions must not be sampled as if they were aperture coverage.
    """
    if count <= 0:
        raise ValueError(f"count must be positive, got {count}")
    if aperture_radius_m <= 0.0:
        raise ValueError(f"aperture_radius_m must be positive, got {aperture_radius_m}")

    alpha = np.array([1.0 / _PLASTIC_NUMBER, 1.0 / _PLASTIC_NUMBER**2])
    accepted: list[tuple[float, float]] = []
    index = 1
    # pi/4 acceptance, so 4/pi proposals per point on average; the cap is a
    # generous multiple of that and exists only so a pathological offset cannot
    # spin forever.
    limit = int(8 * count / (math.pi / 4.0)) + 64
    while len(accepted) < count and index <= limit:
        unit = (offset + alpha * index) % 1.0
        point = (2.0 * unit - 1.0) * aperture_radius_m
        if point[0] ** 2 + point[1] ** 2 <= aperture_radius_m**2:
            accepted.append((float(point[0]), float(point[1])))
        index += 1
    if len(accepted) < count:  # pragma: no cover - defensive
        raise RuntimeError(
            f"the R2 sequence yielded only {len(accepted)} of {count} points inside "
            f"the aperture within {limit} proposals"
        )
    return LaunchGeometry(
        positions_xy_m=np.asarray(accepted, dtype=np.float64),
        aperture_radius_m=float(aperture_radius_m),
        scheme=(
            "R2 (Roberts' generalized golden-ratio) low-discrepancy sequence over "
            "the bounding square, rejected to the circular aperture; nested prefixes"
        ),
        quadrature_weight=(
            "uniform: the sequence is uniform over the disc, so equal weights are "
            "the correct quadrature and the estimator's 1/N carries them"
        ),
        accepted_of_proposed=(len(accepted), index - 1),
    )


@dataclass(frozen=True)
class PositionalAngularSampler:
    """Inverse-CDF sampling of spectral bins, addressable by ray index.

    Two properties, both structural rather than engineered:

    **Bounded memory.** Drawing ``n`` samples costs ``O(M + n)`` -- one CDF over
    the ``M`` retained bins plus ``n`` uniforms -- and never forms an
    ``(n, W)`` conditional table. Phase 8's requirement is met by construction
    for any ``n``, so a chunk is the only thing that sizes an allocation.

    **Chunk invariance.** The uniform for global index ``g`` comes from a PCG64
    stream advanced to ``g``. ``numpy``'s PCG64 consumes exactly one 64-bit
    output per ``float64`` draw, so ``advance(g)`` positions the stream exactly;
    the value is therefore a function of ``(seed, g)`` and of nothing else. Two
    runs with different chunk sizes draw the *same* population, not merely
    populations with the same distribution.

    The draw is host work, deliberately. ``numpy.random.Generator`` is what pins
    the seed, and bitwise reproducibility across devices is worth more than
    avoiding one host-to-device copy of a chunk of indices -- the same trade
    ``wave_to_ray.draw_indices`` already documents.
    """

    cumulative: np.ndarray
    seed: int
    samples_per_launch: int
    density_kind: SamplingDensity

    @classmethod
    def build(
        cls,
        spectrum: AngularSpectrum,
        *,
        density_kind: SamplingDensity,
        seed: int,
        samples_per_launch: int,
    ) -> tuple[PositionalAngularSampler, Any]:
        """Build a sampler and return it with the density it samples from.

        The density is the one ``wave_to_ray.sampling_density`` produces --
        including its refusal of a density that is zero where the spectrum is
        not -- so this does not become a second, subtly different definition of
        ``p``.
        """
        density = sampling_density(spectrum, density_kind)
        host = np.asarray(density, dtype=np.float64)
        host = host / host.sum()
        cumulative = np.cumsum(host)
        # Force the last edge to exactly 1 so a uniform of 0.9999... can never
        # fall past the end of the table.
        cumulative[-1] = 1.0
        return (
            cls(
                cumulative=cumulative,
                seed=int(seed),
                samples_per_launch=int(samples_per_launch),
                density_kind=density_kind,
            ),
            density,
        )

    @property
    def bin_count(self) -> int:
        return int(self.cumulative.size)

    def uniforms(self, global_offset: int, count: int) -> np.ndarray:
        bit_generator = np.random.PCG64(self.seed)
        if global_offset:
            bit_generator.advance(int(global_offset))
        return np.random.Generator(bit_generator).random(int(count))

    def indices(self, *, launch_index: int, start: int, stop: int) -> np.ndarray:
        """Bin indices for samples ``[start, stop)`` of launch ``launch_index``."""
        if not 0 <= start <= stop <= self.samples_per_launch:
            raise ValueError(
                f"sample range [{start}, {stop}) is outside "
                f"[0, {self.samples_per_launch}) for launch {launch_index}"
            )
        offset = launch_index * self.samples_per_launch + start
        uniform = self.uniforms(offset, stop - start)
        return np.searchsorted(self.cumulative, uniform, side="left").clip(
            0, self.bin_count - 1
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "samples_per_launch": self.samples_per_launch,
            "density": str(self.density_kind),
            "bin_count": self.bin_count,
            "method": "inverse CDF over the retained bins, positional PCG64 stream",
            "chunk_invariant": True,
            "memory_complexity": "O(retained_bins + chunk)",
        }


@dataclass(frozen=True)
class ChunkWorkItem:
    """One launch position's contiguous slice of samples, and its global ids."""

    launch_index: int
    start: int
    stop: int
    first_ray_id: int

    @property
    def size(self) -> int:
        return self.stop - self.start


def chunk_plan(
    *, launch_count: int, samples_per_launch: int, chunk_size: int
) -> list[list[ChunkWorkItem]]:
    """Split ``launch_count * samples_per_launch`` rays into chunks of work items.

    Chunks are aligned to launch boundaries whenever ``samples_per_launch``
    divides into ``chunk_size``, and a launch larger than a chunk is split into
    equal pieces. Both rules exist for the same reason: every work item then has
    one of very few sizes, so a JIT-compiled kernel is compiled once for the whole
    sweep instead of once per ragged boundary.

    The plan is a pure function of the three counts -- no device state, no RNG --
    so the same ``(P, S, chunk_size)`` always yields the same partition.
    """
    if min(launch_count, samples_per_launch, chunk_size) <= 0:
        raise ValueError(
            f"launch_count={launch_count}, samples_per_launch={samples_per_launch} and "
            f"chunk_size={chunk_size} must all be positive"
        )
    chunks: list[list[ChunkWorkItem]] = []
    if samples_per_launch > chunk_size:
        # One launch spans several chunks. Split into equal pieces so every piece
        # has the same shape.
        pieces = math.ceil(samples_per_launch / chunk_size)
        piece = math.ceil(samples_per_launch / pieces)
        for launch in range(launch_count):
            for start in range(0, samples_per_launch, piece):
                stop = min(start + piece, samples_per_launch)
                chunks.append(
                    [
                        ChunkWorkItem(
                            launch_index=launch,
                            start=start,
                            stop=stop,
                            first_ray_id=launch * samples_per_launch + start,
                        )
                    ]
                )
        return chunks

    launches_per_chunk = max(1, chunk_size // samples_per_launch)
    for first in range(0, launch_count, launches_per_chunk):
        last = min(first + launches_per_chunk, launch_count)
        chunks.append(
            [
                ChunkWorkItem(
                    launch_index=launch,
                    start=0,
                    stop=samples_per_launch,
                    first_ray_id=launch * samples_per_launch,
                )
                for launch in range(first, last)
            ]
        )
    return chunks


@dataclass
class StreamingResult:
    """The reconstructed field plus what the streaming run measured about itself."""

    field: ComplexField
    total_rays: int
    valid_rays: int
    chunk_count: int
    chunk_sizes: tuple[int, ...]
    residency: dict[str, Any]
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "total_rays": self.total_rays,
            "valid_rays": self.valid_rays,
            "chunk_count": self.chunk_count,
            "distinct_chunk_sizes": sorted(set(self.chunk_sizes)),
            "residency": self.residency,
            **self.diagnostics,
        }


class StreamingReconstruction:
    """One ``(ny, nx)`` complex accumulator, fed chunk by chunk (Phases 9, 10).

    The normalization deserves the emphasis Phase 10 gives it. ``C_RAY_TO_WAVE``
    can apply the ``1/N`` of SI eq S5 itself, but ``N`` would then be the *chunk's*
    ray count, which is not the estimator's ``N``. So every chunk is reconstructed
    with ``normalization="none"`` -- an unnormalized wavelet sum -- and the single
    ``1/N_total`` is applied once when the accumulator is finalized. That is the
    mathematically equivalent route Phase 10 permits, and it is the reason chunk
    size cannot change the answer beyond summation order.

    A clipped ray is *not* removed from ``N_total``. It was drawn, and the
    operator being estimated includes whatever vignetting clipped it; dropping it
    from the denominator would rescale the field by the survival fraction.
    """

    def __init__(
        self,
        *,
        grid_shape: tuple[int, int],
        sample_pitch_m: tuple[float, float],
        plane: ReferencePlane,
        wavelength_m: float,
        namespace: Any,
        complex_dtype: Any,
        total_rays: int,
        projection: Projection = Projection.ASM_CONSISTENT,
        reconstruction: Reconstruction = Reconstruction.RAMP_SUM,
        kspace_oversample: float = DEFAULT_KSPACE_OVERSAMPLE,
        kspace_grid_shape: tuple[int, int] | None = None,
    ) -> None:
        self.grid_shape = grid_shape
        self.sample_pitch_m = sample_pitch_m
        self.plane = plane
        self.wavelength_m = wavelength_m
        self.total_rays = int(total_rays)
        self.projection = projection
        # Chunking and the reconstruction algorithm are independent: the 1/N is
        # still applied once at finalize, so a chunk boundary cannot change the
        # answer under either route. See ray_to_wave.Reconstruction for the
        # k-grid's exactness condition, which the *caller* owns because only the
        # caller knows what grid its ray directions were enumerated on.
        self.reconstruction = reconstruction
        self.kspace_oversample = kspace_oversample
        self.kspace_grid_shape = kspace_grid_shape
        self._xp = xp_for(namespace)
        self._complex_dtype = complex_dtype
        self._accumulator = self._xp.zeros(grid_shape, dtype=numpy_dtype(complex_dtype))
        self.valid_rays = 0
        self.chunk_sizes: list[int] = []
        self._first_diagnostics: dict[str, Any] | None = None

    def add_chunk(self, batch: Any) -> dict[str, Any]:
        """Reconstruct one traced chunk and fold it into the accumulator."""
        chunk_field, diagnostics = ray_to_wave(
            batch.bundle,
            grid_shape=self.grid_shape,
            sample_pitch_m=self.sample_pitch_m,
            plane=self.plane,
            # See the class docstring: the 1/N is the estimator's, not the chunk's.
            normalization="none",
            projection=self.projection,
            reconstruction=self.reconstruction,
            kspace_oversample=self.kspace_oversample,
            kspace_grid_shape=self.kspace_grid_shape,
        )
        self._accumulator = self._accumulator + chunk_field.u
        self.valid_rays += batch.valid_count
        self.chunk_sizes.append(batch.count)
        record = diagnostics.as_dict()
        if self._first_diagnostics is None:
            self._first_diagnostics = record
        return record

    def finalize(self, *, provenance: dict[str, Any] | None = None) -> StreamingResult:
        if not self.chunk_sizes:
            raise ContractError(
                ContractCode.EMPTY_ENSEMBLE,
                "no chunks were accumulated; there is no field to finalize",
                declaration="chunks",
            )
        summed = sum(self.chunk_sizes)
        if summed != self.total_rays:
            raise ContractError(
                ContractCode.SHAPE_MISMATCH,
                (
                    f"the chunks carried {summed} rays but the estimator was "
                    f"normalized for {self.total_rays}; a 1/N mismatch rescales "
                    "the whole field"
                ),
                declaration="total_rays",
            )
        u = self._accumulator / self.total_rays
        field_ = ComplexField(
            u=u,
            sample_pitch_m=self.sample_pitch_m,
            wavelength_m=self.wavelength_m,
            reference_plane=self.plane,
            frame=Frame(),
            normalization=(
                "u is complex amplitude; discrete power = sum(|u|^2) * dy * dx; "
                f"ray-sum normalization = one_over_n applied once over "
                f"N_total = {self.total_rays} after chunked accumulation; "
                f"projection = {self.projection}"
            ),
            provenance={
                "coupler": "C_RAY_TO_WAVE",
                "streaming": {
                    "chunk_count": len(self.chunk_sizes),
                    "distinct_chunk_sizes": sorted(set(self.chunk_sizes)),
                    "normalization_rule": (
                        "each chunk reconstructed with normalization='none'; the "
                        "single 1/N_total applied at finalize, so chunk size cannot "
                        "change the estimator"
                    ),
                    "clipped_ray_policy": (
                        "clipped rays keep their place in N_total and contribute "
                        "zero amplitude; removing them would rescale the field by "
                        "the survival fraction"
                    ),
                },
                **(provenance or {}),
            },
        )
        return StreamingResult(
            field=field_,
            total_rays=self.total_rays,
            valid_rays=self.valid_rays,
            chunk_count=len(self.chunk_sizes),
            chunk_sizes=tuple(self.chunk_sizes),
            residency={
                "sensor_accumulator": {
                    "dtype": str(dtype_of(u)),
                    "device": str(device_of(u)),
                    "namespace": str(namespace_of(u)),
                }
            },
            diagnostics={"first_chunk_reconstruction": self._first_diagnostics},
        )


def build_chunk_bundle(
    spectrum: AngularSpectrum,
    density: Any,
    sampler: PositionalAngularSampler,
    items: list[ChunkWorkItem],
    launch: LaunchGeometry,
) -> tuple[RayBundle, np.ndarray]:
    """Emit one chunk's rays through ``C_WAVE_TO_RAY``, unchanged.

    Each work item is one launch position, so ``spectrum_to_rays`` is called with
    a single-row launch array and that item's own drawn bins -- which is what makes
    ``S`` an *independent* angular sample set per launch rather than one set shared
    by every launch. Both variants are unbiased; the independent one is what makes
    ``P`` a genuine convergence axis instead of an exact no-op for a
    shift-invariant system, and it matches the reference implementation's
    ``P * S`` draw count.

    Returns the concatenated bundle and its global ray ids.
    """
    if not items:
        raise ContractError(
            ContractCode.EMPTY_ENSEMBLE, "a chunk needs at least one work item", declaration="items"
        )
    xp = spectrum.xp
    parts: list[RayBundle] = []
    ids: list[np.ndarray] = []
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
            )
        )
        ids.append(np.arange(item.first_ray_id, item.first_ray_id + item.size, dtype=np.int64))

    if len(parts) == 1:
        return parts[0], ids[0]
    merged = RayBundle(
        positions_m=xp.concatenate([part.positions_m for part in parts]),
        directions=xp.concatenate([part.directions for part in parts]),
        wavelength_m=parts[0].wavelength_m,
        reference_plane=parts[0].reference_plane,
        frame=parts[0].frame,
        amplitude=xp.concatenate([part.amplitude for part in parts]),
        optical_path_length_m=xp.concatenate(
            [part.optical_path_length_m for part in parts]
        ),
        optical_path_length_reference=parts[0].optical_path_length_reference,
        normalization=parts[0].normalization,
        reconstruction_normalization=parts[0].reconstruction_normalization,
        provenance={
            **parts[0].provenance,
            "launch_count": len(parts),
            "sampled_indices": "not retained: a chunked run does not keep per-ray state",
            "sampling_probabilities": "not retained: see sampled_indices",
        },
    )
    return merged, np.concatenate(ids)


def iter_chunks(
    plan: list[list[ChunkWorkItem]],
) -> Iterator[tuple[int, list[ChunkWorkItem]]]:
    """Enumerate a chunk plan. Trivial, but it names the loop the phases refer to."""
    yield from enumerate(plan)
