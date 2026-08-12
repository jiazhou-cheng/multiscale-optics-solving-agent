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

from multiscale_optics_agent.couplers.contracts import (
    ComplexField,
    ContractCode,
    ContractError,
    Frame,
    RayBundle,
    ReferencePlane,
)

__all__ = [
    "AngularSpectrum",
    "SamplingDensity",
    "SamplingPerturbation",
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

    spectrum: np.ndarray
    direction_v: np.ndarray
    direction_u: np.ndarray
    propagating: np.ndarray
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
    def propagating_count(self) -> int:
        return int(np.count_nonzero(self.propagating))

    def transverse_directions(self) -> np.ndarray:
        """``(M, 2)`` transverse direction cosines of the propagating modes."""
        dv, du = np.meshgrid(self.direction_v, self.direction_u, indexing="ij")
        return np.column_stack([du[self.propagating], dv[self.propagating]])

    def propagating_amplitudes(self) -> np.ndarray:
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

    # Centered DFT: ifftshift in, fftshift out. See the class docstring.
    #
    # The 1/(ny*nx) makes the coefficients the *modal amplitudes* themselves,
    # so that a plain sum of modes reproduces the field and no stray inverse-DFT
    # factor has to be remembered downstream. With it, the Monte Carlo estimate
    # (1/N) * sum_i U~[m_i]/p[m_i] is an unbiased estimator of sum_m U~[m],
    # which is the field. Note that Parseval then reads
    # sum|U~|^2 = (1/(ny*nx)) sum|u|^2.
    spectrum = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field.u))) / (ny * nx)

    # Transverse direction cosines: d = lambda * f.
    direction_u = np.fft.fftshift(np.fft.fftfreq(nx, d=dx)) * field.wavelength_m
    direction_v = np.fft.fftshift(np.fft.fftfreq(ny, d=dy)) * field.wavelength_m
    dv, du = np.meshgrid(direction_v, direction_u, indexing="ij")
    radial = du**2 + dv**2

    if perturbation.discard_evanescent:
        # Strict inequality excludes the grazing k_n = 0 bin, which is singular
        # for any 1/k_n-type factor and is recorded as an open question in the
        # coupler card rather than silently included.
        propagating = radial < 1.0
    else:
        propagating = np.ones_like(radial, dtype=bool)

    mode_power = np.abs(spectrum) ** 2
    total = float(np.sum(mode_power))
    evanescent_fraction = (
        float(np.sum(mode_power[~propagating]) / total) if total > 0.0 else 0.0
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
    amplitudes = spectrum.propagating_amplitudes()
    if amplitudes.size == 0:
        raise ContractError(
            ContractCode.EMPTY_ENSEMBLE,
            "no propagating modes survive the evanescent cut",
            declaration="propagating",
            remedy="This field cannot be represented by rays at this wavelength and pitch.",
        )

    if kind is SamplingDensity.UNIFORM:
        density = np.full(amplitudes.size, 1.0 / amplitudes.size, dtype=np.float64)
    else:
        magnitude = np.abs(amplitudes)
        total = float(np.sum(magnitude))
        if total <= 0.0:
            raise ContractError(
                ContractCode.EMPTY_ENSEMBLE,
                "spectral magnitude is identically zero; p_mag is undefined",
                declaration="sampling_density",
            )
        density = (magnitude / total).astype(np.float64)

    nonzero_spectrum = np.abs(amplitudes) > 0.0
    if np.any(nonzero_spectrum & (density <= 0.0)):
        raise ContractError(
            ContractCode.MISSING_DECLARATION,
            (
                "sampling density is zero on a bin where the spectrum is nonzero; "
                "the estimator would be inconsistent, not merely noisy"
            ),
            declaration="sampling_density",
        )
    return density


def draw_indices(
    density: np.ndarray, count: int, rng: np.random.Generator
) -> np.ndarray:
    """Draw ``count`` spectral-bin indices from ``density``.

    Deliberately separate from the core so the core stays a pure function. The
    generator is supplied by the caller; the protocol requires an explicit seed.
    """
    if count <= 0:
        raise ContractError(
            ContractCode.EMPTY_ENSEMBLE,
            f"secondary-ray count must be positive, got {count}",
            declaration="count",
        )
    return rng.choice(density.size, size=count, p=density)


def enumerate_indices(density: np.ndarray) -> np.ndarray:
    """Every propagating bin exactly once — the deterministic exactness limit."""
    return np.arange(density.size, dtype=np.int64)


def spectrum_to_rays(
    spectrum: AngularSpectrum,
    indices: np.ndarray,
    density: np.ndarray,
    *,
    launch_positions_xy_m: np.ndarray | None = None,
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
    indices = np.asarray(indices, dtype=np.int64)
    if indices.ndim != 1 or indices.size == 0:
        raise ContractError(
            ContractCode.EMPTY_ENSEMBLE,
            f"indices must be a non-empty 1-D array, got shape {indices.shape}",
            declaration="indices",
        )

    density = np.asarray(density, dtype=np.float64)
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
    if np.any((np.abs(all_amplitudes) > 0.0) & (density <= 0.0)):
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

    radial = np.sum(transverse**2, axis=1)
    if perturbation.discard_evanescent and np.any(radial >= 1.0):
        raise ContractError(
            ContractCode.NON_UNIT_DIRECTION,
            "selected an evanescent bin; it has no propagation direction",
            declaration="indices",
        )
    normal = perturbation.normal_sign * np.sqrt(np.clip(1.0 - radial, 0.0, None))
    directions = np.column_stack([transverse, normal])

    if perturbation.apply_importance_weight:
        weighted = amplitudes / probabilities
    else:
        weighted = amplitudes.astype(np.complex128)

    if launch_positions_xy_m is None:
        launch = np.zeros((1, 2), dtype=np.float64)
    else:
        launch = np.asarray(launch_positions_xy_m, dtype=np.float64)
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
        launch_phase = np.exp(1j * wavenumber * (launch @ transverse.T))
    else:
        launch_phase = np.ones((launch_count, mode_count), dtype=np.complex128)

    amplitude = (weighted[None, :] * launch_phase).reshape(-1)
    tiled_directions = np.tile(directions, (launch_count, 1))
    plane_z = spectrum.reference_plane.z_m
    positions = np.column_stack(
        [
            np.repeat(launch[:, 0], mode_count),
            np.repeat(launch[:, 1], mode_count),
            np.full(launch_count * mode_count, plane_z, dtype=np.float64),
        ]
    )

    return RayBundle(
        positions_m=positions,
        directions=tiled_directions,
        wavelength_m=spectrum.wavelength_m,
        reference_plane=spectrum.reference_plane,
        frame=Frame(axis_order="flat per-ray arrays"),
        amplitude=amplitude,
        optical_path_length_m=np.zeros(launch_count * mode_count, dtype=np.float64),
        optical_path_length_reference=(
            f"zero at the emitting plane {spectrum.reference_plane.name!r}"
        ),
        normalization=(
            "amplitudes carry the 1/p importance weight (SI eq S4); the 1/N of "
            "SI eq S5 is applied by the reconstruction, not stored on the rays"
        ),
        provenance={
            "coupler": "C_WAVE_TO_RAY",
            "equation": "ACS Photonics 2026 eq 1 / SI eq S4",
            "mode_count": int(mode_count),
            "launch_count": int(launch_count),
            "perturbation": perturbation.describe(),
            "evanescent_power_fraction": spectrum.evanescent_power_fraction,
            "sampled_indices": indices,
            "sampling_probabilities": probabilities,
        },
    )


def wave_to_ray(
    field: ComplexField,
    *,
    count: int | None = None,
    density_kind: SamplingDensity = SamplingDensity.UNIFORM,
    rng: np.random.Generator | None = None,
    launch_positions_xy_m: np.ndarray | None = None,
    perturbation: SamplingPerturbation = SamplingPerturbation(),
) -> tuple[RayBundle, AngularSpectrum, np.ndarray]:
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
