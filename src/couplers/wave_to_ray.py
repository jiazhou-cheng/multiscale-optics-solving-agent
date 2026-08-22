"""C_WAVE_TO_RAY — angular-spectrum decomposition into Monte Carlo rays (CHE-25).

Implements SI S2 (eqs S1–S5) and Algorithm S2 of Cheng et al., ACS Photonics
2026 (DOI 10.1021/acsphotonics.6c00818).

The framing that makes this testable: **this is a quadrature scheme for an
integral whose exact value is known** (SI eq S2), not an approximation of the
physics. Enumerate every propagating bin and the estimator must collapse onto
the deterministic reference at dtype round-off. Only after that check passes is
there any point discussing sampling error -- which is why
``coupler_protocol.yaml`` makes the exactness limit mandatory and first.

Sampling is an **input**, not a side effect
-------------------------------------------
:func:`spectrum_to_rays` takes pre-drawn spectral indices as an argument;
:func:`draw_indices` draws them from an explicit seeded generator. Three
properties follow structurally rather than having to be engineered:

* bitwise determinism, because the core is a pure function of its arguments;
* one implementation serves both the reference and the gradient study, since
  there is no RNG inside to differentiate through;
* SI Algorithm S2's "sampled directions held fixed during backpropagation"
  becomes the shape of the interface, not a ``.detach()`` to remember.

This module imports neither Optiland nor Chromatix.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np

from core.arrays import (
    dtype_of,
    matmul_precision_kwargs,
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
from core.capabilities import C_WAVE_TO_RAY_CAPABILITIES
from core.precision import ArrayNamespace, DType, Precision

__all__ = [
    "AngularSpectrum",
    "SamplingDensity",
    "SamplingPerturbation",
    "compute_precision_for",
    "decompose",
    "draw_indices",
    "sampling_density",
    "spectrum_to_rays",
    "wave_to_ray",
]


class SamplingDensity(StrEnum):
    """Which density ``p(k_u, k_v)`` to draw secondary-ray wavevectors from."""

    #: Uniform over propagating bins. Assumes nothing about the spectrum.
    UNIFORM = "p_uni"
    #: Proportional to ``|U~|``. The paper reports faster convergence for
    #: spectra concentrated in a single lobe (Figure 4a) and comparable rates
    #: for multilobed ones (Figure 4b).
    MAGNITUDE = "p_mag"


@dataclass(frozen=True)
class SamplingPerturbation:
    """Deliberate defects, for negative tests only. Defaults are correct."""

    #: Drop the ``1/p`` importance weight. Biases the estimator under
    #: non-uniform sampling while still producing a plausible-looking field.
    apply_importance_weight: bool = True
    #: Keep evanescent modes. ``k_n`` is then imaginary and the direction is
    #: not a direction.
    discard_evanescent: bool = True
    #: Take the negative root for ``k_n``. Reverses propagation; invisible at z = 0.
    normal_sign: int = 1
    #: Drop the launch-position phase. Invisible for one centred launch point.
    apply_launch_phase: bool = True

    @property
    def is_identity(self) -> bool:
        return (
            self.apply_importance_weight
            and self.discard_evanescent
            and self.normal_sign == 1
            and self.apply_launch_phase
        )

    def describe(self) -> str:
        if self.is_identity:
            return "none"
        parts = []
        if not self.apply_importance_weight:
            parts.append("importance_weight_omitted")
        if not self.discard_evanescent:
            parts.append("evanescent_cut_omitted")
        if self.normal_sign != 1:
            parts.append("kn_sign_flipped")
        if not self.apply_launch_phase:
            parts.append("launch_phase_omitted")
        return "+".join(parts)


@dataclass(frozen=True)
class AngularSpectrum:
    """The propagating plane-wave decomposition of a field on a plane.

    ``spectrum`` uses the **centered** DFT convention: zero frequency sits at
    index ``n // 2`` on each axis, matching the spatial origin rule M1 pinned.
    That pairing is not cosmetic -- with the ordinary un-centered transform the
    reconstruction picks up an ``exp(-i pi m)`` offset per axis and misses the
    field entirely, which is the first thing to check if a round trip fails.
    """

    spectrum: Any
    direction_v: Any
    direction_u: Any
    propagating: Any
    wavelength_m: float
    sample_pitch_m: tuple[float, float]
    grid_shape: tuple[int, int]
    reference_plane: ReferencePlane
    evanescent_power_fraction: float
    total_discrete_power: float

    @property
    def wavenumber(self) -> float:
        return 2.0 * math.pi / self.wavelength_m

    @property
    def xp(self) -> Any:
        """The array module this spectrum belongs to -- NumPy on the host, JAX on a GPU."""
        return xp_for(namespace_of(self.spectrum))

    @property
    def namespace(self) -> ArrayNamespace:
        return namespace_of(self.spectrum)

    @property
    def dtype(self) -> DType:
        return dtype_of(self.spectrum)

    @property
    def real_dtype(self) -> DType:
        """The real dtype matching the spectrum's precision -- float32 for complex64."""
        return self.dtype.precision.real_dtype

    @property
    def propagating_count(self) -> int:
        return int(self.xp.count_nonzero(self.propagating))

    def transverse_directions(self) -> Any:
        """``(M, 2)`` transverse direction cosines of the propagating modes."""
        xp = self.xp
        dv, du = xp.meshgrid(self.direction_v, self.direction_u, indexing="ij")
        return xp.column_stack([du[self.propagating], dv[self.propagating]])

    def propagating_amplitudes(self) -> Any:
        return self.spectrum[self.propagating]

    def as_dict(self) -> dict[str, Any]:
        return {
            "grid_shape": list(self.grid_shape),
            "sample_pitch_m": list(self.sample_pitch_m),
            "wavelength_m": self.wavelength_m,
            "propagating_modes": self.propagating_count,
            "total_modes": int(self.spectrum.size),
            "evanescent_power_fraction": self.evanescent_power_fraction,
            "total_discrete_power": self.total_discrete_power,
        }


def compute_precision_for(spectrum: AngularSpectrum) -> Precision:
    """The precision this coupler decomposes and re-emits in for ``spectrum``.

    Taken from the field's own dtype and floored at the coupler's declared
    minimum, so it is distinct from both the accepted input dtype and the
    emitted output dtype (PB4b section 9).
    """
    floor = C_WAVE_TO_RAY_CAPABILITIES.minimum_compute_precision
    return max([spectrum.dtype.precision, floor], key=lambda p: p.bits)


def decompose(
    field: ComplexField, *, perturbation: SamplingPerturbation = SamplingPerturbation()
) -> AngularSpectrum:
    """Angular spectrum of a field, with the evanescent cut applied and accounted.

    Discarded evanescent power is reported as a named fraction. It is a real
    loss -- an evanescent mode has no propagation direction to give a ray -- and
    a large fraction is the signature of a field that should not be turned into
    rays at all.
    """
    ny, nx = field.shape
    dy, dx = field.sample_pitch_m
    # One implementation, two namespaces: the FFT, the frequency grid and the
    # evanescent cut all execute wherever the field already lives, so a GPU
    # field is decomposed on the GPU with no host round trip and no second
    # copy of this physics.
    xp = field.xp
    real_np = numpy_dtype(field.real_dtype)

    # Centered DFT: ifftshift in, fftshift out. See the class docstring.
    #
    # The 1/(ny*nx) makes the coefficients the *modal amplitudes* themselves,
    # so that a plain sum of modes reproduces the field and no stray inverse-DFT
    # factor has to be remembered downstream. With it, the Monte Carlo estimate
    # (1/N) * sum_i U~[m_i]/p[m_i] is an unbiased estimator of sum_m U~[m],
    # which is the field. Note that Parseval then reads
    # sum|U~|^2 = (1/(ny*nx)) sum|u|^2.
    spectrum = xp.fft.fftshift(xp.fft.fft2(xp.fft.ifftshift(field.u))) / (ny * nx)

    # Transverse direction cosines: d = lambda * f. Built at the field's own
    # real precision so a complex64 field does not acquire float64 axes.
    direction_u = (xp.fft.fftshift(xp.fft.fftfreq(nx, d=dx)) * field.wavelength_m).astype(real_np)
    direction_v = (xp.fft.fftshift(xp.fft.fftfreq(ny, d=dy)) * field.wavelength_m).astype(real_np)
    dv, du = xp.meshgrid(direction_v, direction_u, indexing="ij")
    radial = du**2 + dv**2

    if perturbation.discard_evanescent:
        # Strict inequality excludes the grazing k_n = 0 bin, which is singular
        # for any 1/k_n-type factor and is recorded as an open question in the
        # coupler card rather than silently included.
        propagating = radial < 1.0
    else:
        propagating = xp.ones_like(radial, dtype=bool)

    mode_power = xp.abs(spectrum) ** 2
    total = float(xp.sum(mode_power))
    evanescent_fraction = (
        float(xp.sum(mode_power[~propagating]) / total) if total > 0.0 else 0.0
    )

    return AngularSpectrum(
        spectrum=spectrum,
        direction_v=direction_v,
        direction_u=direction_u,
        propagating=propagating,
        wavelength_m=field.wavelength_m,
        sample_pitch_m=(dy, dx),
        grid_shape=(ny, nx),
        reference_plane=field.reference_plane,
        evanescent_power_fraction=evanescent_fraction,
        total_discrete_power=field.discrete_power(),
    )


def sampling_density(
    spectrum: AngularSpectrum, kind: SamplingDensity = SamplingDensity.UNIFORM
) -> np.ndarray:
    """Normalized probability over propagating bins.

    Returns a vector aligned with :meth:`AngularSpectrum.propagating_amplitudes`.
    A density that is zero where the spectrum is nonzero makes the estimator
    *inconsistent* rather than merely slow -- those modes are never drawn and no
    amount of ``1/p`` reweighting can recover them -- so that case is refused.
    """
    xp = spectrum.xp
    real_np = numpy_dtype(spectrum.real_dtype)
    amplitudes = spectrum.propagating_amplitudes()
    if amplitudes.size == 0:
        raise ContractError(
            ContractCode.EMPTY_ENSEMBLE,
            "no propagating modes survive the evanescent cut",
            declaration="propagating",
            remedy="This field cannot be represented by rays at this wavelength and pitch.",
        )

    if kind is SamplingDensity.UNIFORM:
        density = xp.full(amplitudes.size, 1.0 / amplitudes.size, dtype=real_np)
    else:
        magnitude = xp.abs(amplitudes)
        total = float(xp.sum(magnitude))
        if total <= 0.0:
            raise ContractError(
                ContractCode.EMPTY_ENSEMBLE,
                "spectral magnitude is identically zero; p_mag is undefined",
                declaration="sampling_density",
            )
        density = (magnitude / total).astype(real_np)

    nonzero_spectrum = xp.abs(amplitudes) > 0.0
    if bool(xp.any(nonzero_spectrum & (density <= 0.0))):
        raise ContractError(
            ContractCode.MISSING_DECLARATION,
            (
                "sampling density is zero on a bin where the spectrum is nonzero; "
                "the estimator would be inconsistent, not merely noisy"
            ),
            declaration="sampling_density",
        )
    return density


def draw_indices(density: Any, count: int, rng: np.random.Generator) -> np.ndarray:
    """Draw ``count`` spectral-bin indices from ``density``.

    Deliberately separate from the core so the core stays a pure function. The
    generator is supplied by the caller; the protocol requires an explicit seed.

    The draw itself is host work by construction -- ``numpy.random.Generator``
    is what pins the seed, and bitwise reproducibility across devices is worth
    more here than avoiding one copy of a probability vector. ``np.asarray`` is
    therefore written out rather than left to happen inside ``rng.choice``, so
    the host read is visible in the source and not a surprise in a profile.
    """
    if count <= 0:
        raise ContractError(
            ContractCode.EMPTY_ENSEMBLE,
            f"secondary-ray count must be positive, got {count}",
            declaration="count",
        )
    host_density = np.asarray(density, dtype=np.float64)
    # Renormalize after the float32 -> float64 widening: numpy requires p to sum
    # to 1 within a tight tolerance, and a complex64 spectrum's density does not
    # after the cast.
    host_density = host_density / host_density.sum()
    return rng.choice(host_density.size, size=count, p=host_density)


def enumerate_indices(density: Any) -> np.ndarray:
    """Every propagating bin exactly once — the deterministic exactness limit."""
    return np.arange(density.size, dtype=np.int64)


def spectrum_to_rays(
    spectrum: AngularSpectrum,
    indices: Any,
    density: Any,
    *,
    launch_positions_xy_m: Any = None,
    perturbation: SamplingPerturbation = SamplingPerturbation(),
) -> RayBundle:
    """Turn selected spectral modes into rays (SI eqs S4, Algorithm S1 lines 8-13).

    ``indices`` selects bins; it is an argument rather than something drawn
    here, so this function is pure and bitwise reproducible.

    Each selected mode ``m`` produces, for each launch position ``(x_p, y_p)``:

    * direction ``d = (d_u, d_v, d_n)`` with ``d_n = +sqrt(1 - d_u^2 - d_v^2)``;
    * amplitude ``a = U~[m] / p[m] * exp(i phi)`` with
      ``phi = k (d_u x_p + d_v y_p)`` the launch-position phase;
    * ``OPL = 0``, because the accumulated path restarts at this plane.

    The ``1/p`` factor is what keeps the estimator unbiased under non-uniform
    sampling. It is not an optimization.
    """
    xp = spectrum.xp
    dot = matmul_precision_kwargs(spectrum.namespace)
    real_np = numpy_dtype(spectrum.real_dtype)
    complex_dtype = spectrum.dtype
    complex_np = numpy_dtype(complex_dtype)

    # Bin indices stay host integers whatever the field's namespace: they come
    # from a NumPy Generator (draw_indices) or from arange, they carry no
    # precision, and both NumPy and JAX index correctly with a host integer
    # array. Pushing them to a device would buy nothing and cost a transfer.
    indices = np.asarray(indices, dtype=np.int64)
    if indices.ndim != 1 or indices.size == 0:
        raise ContractError(
            ContractCode.EMPTY_ENSEMBLE,
            f"indices must be a non-empty 1-D array, got shape {indices.shape}",
            declaration="indices",
        )

    density = xp.asarray(density, dtype=real_np)
    all_amplitudes = spectrum.propagating_amplitudes()
    if density.shape != all_amplitudes.shape:
        raise ContractError(
            ContractCode.SHAPE_MISMATCH,
            f"density {density.shape} must align with the propagating modes "
            f"{all_amplitudes.shape}",
            declaration="density",
        )
    # A zero density where the spectrum is nonzero is not slow convergence: those
    # modes are never drawn, and no amount of 1/p reweighting recovers them. The
    # estimator is inconsistent, so it is refused rather than run.
    if bool(xp.any((xp.abs(all_amplitudes) > 0.0) & (density <= 0.0))):
        raise ContractError(
            ContractCode.MISSING_DECLARATION,
            (
                "sampling density is zero on a bin where the spectrum is nonzero; "
                "the estimator would be inconsistent, not merely noisy"
            ),
            declaration="density",
            remedy="Use a density with support everywhere the spectrum is nonzero.",
        )

    transverse = spectrum.transverse_directions()[indices]
    amplitudes = spectrum.propagating_amplitudes()[indices]
    probabilities = density[indices]

    radial = xp.sum(transverse**2, axis=1)
    if perturbation.discard_evanescent and bool(xp.any(radial >= 1.0)):
        raise ContractError(
            ContractCode.NON_UNIT_DIRECTION,
            "selected an evanescent bin; it has no propagation direction",
            declaration="indices",
        )
    normal = perturbation.normal_sign * xp.sqrt(xp.clip(1.0 - radial, 0.0, None))
    directions = xp.column_stack([transverse, normal]).astype(real_np)

    if perturbation.apply_importance_weight:
        weighted = amplitudes / probabilities
    else:
        weighted = amplitudes.astype(complex_np)

    if launch_positions_xy_m is None:
        launch = xp.zeros((1, 2), dtype=real_np)
    else:
        launch = xp.asarray(launch_positions_xy_m, dtype=real_np)
        if launch.ndim != 2 or launch.shape[1] != 2:
            raise ContractError(
                ContractCode.SHAPE_MISMATCH,
                f"launch_positions_xy_m must be (P, 2), got {launch.shape}",
                declaration="launch_positions_xy_m",
            )

    wavenumber = spectrum.wavenumber
    launch_count = launch.shape[0]
    mode_count = indices.size

    # Outer product over (launch position, mode): P * S rays, a budget set by
    # the caller rather than by the incoming ray count. This is what stops the
    # count growing multiplicatively across cascaded planar surfaces
    # (SI Algorithm S1).
    if perturbation.apply_launch_phase:
        # Explicit dtype rather than `1j * ...`: see _cis in ray_to_wave.py for
        # why scalar promotion is not left to decide a contract-visible dtype.
        projected = xp.matmul(launch, transverse.T, **dot)
        launch_phase = xp.exp((wavenumber * projected).astype(complex_np) * 1j)
    else:
        launch_phase = xp.ones((launch_count, mode_count), dtype=complex_np)

    amplitude = (weighted[None, :] * launch_phase).reshape(-1)
    tiled_directions = xp.tile(directions, (launch_count, 1))
    plane_z = spectrum.reference_plane.z_m
    positions = xp.column_stack(
        [
            xp.repeat(launch[:, 0], mode_count),
            xp.repeat(launch[:, 1], mode_count),
            xp.full(launch_count * mode_count, plane_z, dtype=real_np),
        ]
    )

    return RayBundle(
        positions_m=positions,
        directions=tiled_directions,
        wavelength_m=spectrum.wavelength_m,
        reference_plane=spectrum.reference_plane,
        frame=Frame(axis_order="flat per-ray arrays"),
        amplitude=amplitude,
        optical_path_length_m=xp.zeros(launch_count * mode_count, dtype=real_np),
        optical_path_length_reference=(
            f"zero at the emitting plane {spectrum.reference_plane.name!r}"
        ),
        normalization=(
            "amplitudes carry the 1/p importance weight (SI eq S4); the 1/N of "
            "SI eq S5 is applied by the reconstruction, not stored on the rays"
        ),
        # This ensemble is a Monte Carlo estimate of an integral, so a coherent
        # reconstruction from it must divide by the ray count (SI eq S5).
        reconstruction_normalization="one_over_n",
        provenance={
            "coupler": "C_WAVE_TO_RAY",
            "equation": "ACS Photonics 2026 eq 1 / SI eq S4",
            "mode_count": int(mode_count),
            "launch_count": int(launch_count),
            "perturbation": perturbation.describe(),
            "evanescent_power_fraction": spectrum.evanescent_power_fraction,
            "sampled_indices": indices,
            "sampling_probabilities": probabilities,
            # Requested/resolved/actual for this coupler: the field's own
            # precision is what it computed in, and the emitted representation
            # is read off the arrays rather than asserted.
            "execution": {
                "input": {"dtype": str(complex_dtype), "namespace": str(spectrum.namespace)},
                "compute_precision": str(compute_precision_for(spectrum)),
                "output": {"real_dtype": str(real_np), "complex_dtype": str(complex_np)},
            },
        },
    )


def wave_to_ray(
    field: ComplexField,
    *,
    count: int | None = None,
    density_kind: SamplingDensity = SamplingDensity.UNIFORM,
    rng: np.random.Generator | None = None,
    launch_positions_xy_m: Any = None,
    perturbation: SamplingPerturbation = SamplingPerturbation(),
) -> tuple[RayBundle, AngularSpectrum, Any]:
    """Convenience wrapper: decompose, build a density, select, emit rays.

    ``count=None`` enumerates every propagating bin — the deterministic
    exactness limit, with no sampling error at all. Any other value draws
    ``count`` modes from ``rng``, which must then be supplied so the seed is
    explicit rather than implicit.
    """
    spectrum = decompose(field, perturbation=perturbation)
    density = sampling_density(spectrum, density_kind)

    if count is None:
        indices = enumerate_indices(density)
    else:
        if rng is None:
            raise ContractError(
                ContractCode.MISSING_DECLARATION,
                "stochastic sampling requires an explicit seeded generator",
                declaration="rng",
                remedy="Pass numpy.random.default_rng(seed). The protocol forbids an implicit seed.",
            )
        indices = draw_indices(density, count, rng)

    bundle = spectrum_to_rays(
        spectrum,
        indices,
        density,
        launch_positions_xy_m=launch_positions_xy_m,
        perturbation=perturbation,
    )
    return bundle, spectrum, density
