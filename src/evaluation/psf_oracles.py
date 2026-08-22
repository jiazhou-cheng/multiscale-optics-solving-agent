"""CHE-37 (M3.8): two independent PSF oracles for the M3 slice.

The shipping path is ``Optiland trace -> C_RAY_TO_WAVE -> Chromatix ASM ->
ComplexField -> PSF measurement``. Nothing in it is checked by anything else in
it, so this module provides two things that are checked by something else.

1. The analytic Airy pattern
----------------------------
Closed form, valid only for an unaberrated circular pupil, so it applies to
``M3SingletRef`` and nothing else in M3. It tests the absolute scale, the
pixel-to-angle mapping and the whole chain's normalization at once, because it
predicts *where* the first null is, not merely that there is one.

**The frozen protocol value is a diameter.**
``systems.M3-SINGLET-REF.airy_radius_um = 12.9746`` is ``1.22 lambda / NA``, the
first-null DIAMETER. The radius is ``0.61 lambda / NA = 6.4873 um``. The same
factor of two is in ``airy_radius_in_pixels = 4.88`` (the radius is ~2.44 px at
the frozen 2x oversampling). This module computes the radius from the first zero
of ``J1`` and never reads the frozen field; see
:data:`AIRY_FIRST_NULL_COEFFICIENT_EXACT`. Recorded as a protocol defect in
``open_structural_items.airy_radius_entry_is_a_diameter``, found by CHE-33.

2. An independent FFT/Fraunhofer PSF
------------------------------------
Same traced OPD and amplitude, completely different implementation. The argument
for independence is structural, not stylistic:

=====================  ================================  ============================
step                   shipping path                     this oracle
=====================  ================================  ============================
pupil field            sum of N plane wavelets, one per   least-squares polynomial fit
                       ray, accumulated onto the grid    of W(x, y), evaluated on the
                       (``couplers/ray_to_wave.py``)     grid; no per-ray accumulation
propagation            Chromatix angular-spectrum         none. A single FFT of the
                       transfer function over z, with     pupil, which is the field at
                       padding and an evanescent policy   the focus of a converging
                                                          wave of radius R
image pixel scale      dx_out = dx_pupil (ASM preserves   lambda * R / (N_fft * dx),
                       the pitch)                         analytic, and *different*
=====================  ================================  ============================

It does not import, wrap or restate ``ray_to_wave``. What the two DO share is the
traced OPD map and the amplitude weights, so a defect in the ray trace or in the
OPL declaration is invisible to this oracle. That shared blind spot is why M3.8
also carries the Maréchal Strehl cross-check.

The reference sphere, and a lesson about removing tilt
------------------------------------------------------
The oracle needs the *aberration* ``W``, not the total pupil OPL: the converging
term that focuses the beam is what the FFT geometry already encodes. ``W`` is
therefore the pupil OPL measured against a reference sphere centred on the
observation point, and for a perfect system ``W`` is constant, because every ray
of a perfect system reaches the focus with equal optical path.

Getting that reference point wrong does not look like an error, it looks like
aberration. While selecting M3.8's aberrated case, subtracting a least-squares
*linear* ramp in ``(x, y)`` from the pupil OPL of an off-axis bundle left a
residual that grew with field angle and reached 1.0 waves RMS at full field --
which read as a strongly aberrated system. It was not. Optiland's own
``Wavefront`` said 0.021 waves, and the traced geometric spot -- which depends on
no reference-sphere convention at all -- was 0.18 Airy radii, so the rays were
landing well inside the diffraction core. The linear ramp is only the leading
term of a lateral shift of the sphere centre; the rest of that shift stays in the
residual and masquerades as aberration. :func:`fit_reference_sphere` therefore
solves for the sphere *centre* instead of subtracting a ramp, and reports how far
it moved.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import Any

import numpy as np

from couplers.contracts import ContractCode, ContractError, RayBundle

__all__ = [
    "AIRY_FIRST_NULL_COEFFICIENT_EXACT",
    "AIRY_FIRST_NULL_COEFFICIENT_ROUNDED",
    "AIRY_J1_FIRST_ZERO",
    "FraunhoferPsf",
    "PupilAberration",
    "ReferenceSphere",
    "airy_first_null_radius_m",
    "airy_intensity_at_radius",
    "airy_psf_on_grid",
    "azimuthal_profile",
    "fit_reference_sphere",
    "fraunhofer_psf",
    "measure_first_null_radius_m",
    "numerical_aperture_from_geometry",
    "pupil_aberration",
    "radial_profile",
    "resample_to_grid",
]

#: First zero of the Bessel function ``J1``. ``scipy.special.jn_zeros(1, 1)[0]``.
AIRY_J1_FIRST_ZERO = 3.8317059702075125

#: ``AIRY_J1_FIRST_ZERO / (2 pi)``. The first-null RADIUS is this times
#: ``lambda / NA``. The familiar 1.22 is the same number doubled -- a diameter.
AIRY_FIRST_NULL_COEFFICIENT_EXACT = AIRY_J1_FIRST_ZERO / (2.0 * math.pi)

#: The rounded coefficient the protocol quotes. Kept so the difference between
#: 0.61 and 0.60983 can be reported rather than absorbed: it is 0.03% of the
#: radius, which is small, but the oracle should not be the place a rounding is
#: introduced silently.
AIRY_FIRST_NULL_COEFFICIENT_ROUNDED = 0.61


def airy_first_null_radius_m(wavelength_m: float, numerical_aperture: float) -> float:
    """The Airy first-null **radius**, ``0.60983 * lambda / NA``.

    Not ``1.22 * lambda / NA``, which is the diameter and is what the frozen
    protocol field ``airy_radius_um`` actually holds.
    """
    if wavelength_m <= 0.0 or numerical_aperture <= 0.0:
        raise ContractError(
            ContractCode.UNIT_NOT_SI,
            f"wavelength and NA must be positive, got {wavelength_m!r} and "
            f"{numerical_aperture!r}",
            declaration="wavelength_m / numerical_aperture",
        )
    return AIRY_FIRST_NULL_COEFFICIENT_EXACT * wavelength_m / numerical_aperture


def numerical_aperture_from_geometry(pupil_radius_m: float, distance_m: float) -> float:
    """``sin`` of the marginal ray angle: ``a / sqrt(a^2 + R^2)``.

    Reported alongside the protocol's frozen NA rather than instead of it. The
    two differ by ~0.5% on M3-SINGLET-REF because a paraxial ratio ``a / R`` is
    not a sine, and 0.5% of the Airy radius is 0.03 um -- far above the profile
    tolerance the residual is compared against, so which definition an oracle
    used is not a detail.
    """
    return pupil_radius_m / math.hypot(pupil_radius_m, distance_m)


def airy_intensity_at_radius(
    radius_m: np.ndarray[Any, Any] | float,
    *,
    wavelength_m: float,
    numerical_aperture: float,
) -> np.ndarray[Any, Any]:
    """``[2 J1(v) / v]^2`` with ``v = 2 pi NA r / lambda``. Peak-normalized."""
    from scipy.special import j1  # type: ignore[import-untyped]

    v = 2.0 * math.pi * numerical_aperture * np.abs(np.asarray(radius_m, dtype=np.float64))
    v = v / wavelength_m
    out = np.ones_like(v)
    nonzero = v > 0.0
    out[nonzero] = (2.0 * j1(v[nonzero]) / v[nonzero]) ** 2
    intensity: np.ndarray[Any, Any] = out
    return intensity


def airy_psf_on_grid(
    *,
    shape: tuple[int, int],
    sample_pitch_m: tuple[float, float],
    wavelength_m: float,
    numerical_aperture: float,
    center_m: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray[Any, Any]:
    """The analytic Airy intensity, peak-normalized, on the pinned origin rule.

    Sampled at pixel centres. At M3 sampling the Airy radius spans ~4.9 px, so
    point sampling a function with a 2.44-px-wide core is itself an
    approximation; M3.8 reports the residual against the sampling term in the
    budget rather than treating this as exact.
    """
    ny, nx = shape
    dy, dx = sample_pitch_m
    y = (np.arange(ny, dtype=np.float64) - ny // 2) * dy - center_m[0]
    x = (np.arange(nx, dtype=np.float64) - nx // 2) * dx - center_m[1]
    r = np.hypot(y[:, None], x[None, :])
    return airy_intensity_at_radius(
        r, wavelength_m=wavelength_m, numerical_aperture=numerical_aperture
    )


@dataclass(frozen=True)
class ReferenceSphere:
    """The sphere the pupil wavefront is measured against."""

    #: Centre, in the ray frame, metres.
    center_m: tuple[float, float, float]
    #: Where the search started -- the declared observation point.
    initial_center_m: tuple[float, float, float]
    #: RMS of the residual at the fitted centre, in waves.
    residual_rms_waves: float
    #: RMS at the initial centre. The ratio is how much the fit mattered.
    initial_residual_rms_waves: float
    iterations: int

    @property
    def shift_m(self) -> tuple[float, float, float]:
        return tuple(
            float(a - b) for a, b in zip(self.center_m, self.initial_center_m, strict=True)
        )  # type: ignore[return-value]

    def as_dict(self) -> dict[str, Any]:
        return {
            "center_m": list(self.center_m),
            "initial_center_m": list(self.initial_center_m),
            "shift_m": list(self.shift_m),
            "residual_rms_waves": self.residual_rms_waves,
            "initial_residual_rms_waves": self.initial_residual_rms_waves,
            "iterations": self.iterations,
        }


def _sphere_residual(
    x: np.ndarray[Any, Any],
    y: np.ndarray[Any, Any],
    z_m: float,
    opl_m: np.ndarray[Any, Any],
    center: tuple[float, float, float],
) -> np.ndarray[Any, Any]:
    """``OPL + |Q - P|``, piston removed. Constant for a perfect system."""
    qx, qy, qz = center
    path = np.sqrt((qx - x) ** 2 + (qy - y) ** 2 + (qz - z_m) ** 2)
    total = opl_m + path
    residual: np.ndarray[Any, Any] = total - total.mean()
    return residual


def fit_reference_sphere(
    *,
    positions_m: np.ndarray[Any, Any],
    plane_z_m: float,
    optical_path_length_m: np.ndarray[Any, Any],
    wavelength_m: float,
    initial_center_m: tuple[float, float, float],
    max_iterations: int = 25,
    tolerance: float = 1e-14,
) -> ReferenceSphere:
    """Solve for the sphere centre that minimizes the wavefront residual.

    Gauss-Newton on the three coordinates of the centre, with the analytic
    Jacobian ``d|Q - P| / dQ = (Q - P) / |Q - P|``.

    This exists instead of "subtract a least-squares tilt", which is the mistake
    documented in this module's docstring: a lateral shift of the sphere centre is
    only *approximately* a linear ramp across the pupil, and the difference grows
    with field angle until it is indistinguishable from real aberration.
    """
    x = np.asarray(positions_m)[:, 0]
    y = np.asarray(positions_m)[:, 1]
    opl = np.asarray(optical_path_length_m, dtype=np.float64)
    center = np.array(initial_center_m, dtype=np.float64)

    initial_rms = float(
        np.std(_sphere_residual(x, y, plane_z_m, opl, tuple(center))) / wavelength_m
    )

    iterations = 0
    for iterations in range(1, max_iterations + 1):  # noqa: B007
        qx, qy, qz = center
        dx_ = qx - x
        dy_ = qy - y
        dz_ = qz - plane_z_m
        path = np.sqrt(dx_**2 + dy_**2 + dz_**2)
        residual = opl + path
        residual = residual - residual.mean()

        # Jacobian of the residual w.r.t. the centre, with the piston projected
        # out (a common shift of every ray is absorbed by the mean removal).
        jacobian = np.stack([dx_ / path, dy_ / path, np.full_like(path, dz_) / path], axis=1)
        jacobian = jacobian - jacobian.mean(axis=0, keepdims=True)

        step, *_ = np.linalg.lstsq(jacobian, -residual, rcond=None)
        center = center + step
        if float(np.max(np.abs(step))) <= tolerance * max(1.0, float(np.max(np.abs(center)))):
            break

    final_rms = float(np.std(_sphere_residual(x, y, plane_z_m, opl, tuple(center))) / wavelength_m)
    return ReferenceSphere(
        center_m=(float(center[0]), float(center[1]), float(center[2])),
        initial_center_m=initial_center_m,
        residual_rms_waves=final_rms,
        initial_residual_rms_waves=initial_rms,
        iterations=iterations,
    )


@dataclass(frozen=True)
class PupilAberration:
    """The wavefront error at the pupil, as scattered per-ray samples."""

    #: (N, 2) ray positions at the pupil plane, metres.
    positions_m: np.ndarray[Any, Any]
    #: (N,) wavefront error in metres, piston removed.
    wavefront_error_m: np.ndarray[Any, Any]
    #: (N,) real amplitude, as declared by the bundle.
    amplitude: np.ndarray[Any, Any]
    wavelength_m: float
    #: The largest traced pupil radius: the aperture the oracle masks with.
    pupil_radius_m: float
    sphere: ReferenceSphere
    plane_z_m: float

    @property
    def rms_waves(self) -> float:
        return float(np.std(self.wavefront_error_m) / self.wavelength_m)

    @property
    def peak_to_valley_waves(self) -> float:
        return float(np.ptp(self.wavefront_error_m) / self.wavelength_m)

    @property
    def marechal_strehl(self) -> float:
        """``exp(-(2 pi sigma)^2)``, sigma in waves.

        Valid only for small aberrations -- conventionally below ~0.1 waves RMS,
        where it agrees with the diffraction integral to a few percent. Above that
        it decays too fast and must be read as an ordering, not a prediction.
        """
        return float(math.exp(-((2.0 * math.pi * self.rms_waves) ** 2)))

    @property
    def marechal_is_in_regime(self) -> bool:
        return self.rms_waves <= 0.1

    def as_dict(self) -> dict[str, Any]:
        return {
            "rms_waves": self.rms_waves,
            "peak_to_valley_waves": self.peak_to_valley_waves,
            "marechal_strehl": self.marechal_strehl,
            "marechal_valid_regime": self.marechal_is_in_regime,
            "marechal_validity_note": (
                "exp(-(2 pi sigma)^2) is a small-aberration expansion. Below ~0.1 "
                "waves RMS it tracks the diffraction integral to a few percent; "
                "above it, it falls too fast and is an ordering only."
            ),
            "rayleigh_quarter_wave_limited": bool(self.peak_to_valley_waves <= 0.25),
            "pupil_radius_m": self.pupil_radius_m,
            "ray_count": int(self.wavefront_error_m.size),
            "reference_sphere": self.sphere.as_dict(),
        }


def pupil_aberration(
    bundle: RayBundle,
    *,
    plane_z_m: float,
    observation_point_m: tuple[float, float, float],
    fit_sphere: bool = True,
) -> PupilAberration:
    """Wavefront error at the pupil, against a sphere centred on the observation.

    ``observation_point_m`` is where the PSF is being formed -- the declared
    observation plane's z, and the lateral point the PSF is centred on. With
    ``fit_sphere`` the lateral/axial centre is refined by
    :func:`fit_reference_sphere`; the shift is reported, because a large shift
    means the declared observation point was not where the light actually goes.

    Nothing here reconstructs a field. It is arithmetic on the bundle the shipping
    coupler consumes, which is the point: the oracle downstream of it must not
    share the reconstruction.
    """
    positions = np.asarray(bundle.positions_m, dtype=np.float64)
    opl = np.asarray(bundle.optical_path_length_m, dtype=np.float64)
    if opl.size == 0:
        raise ContractError(
            ContractCode.EMPTY_ENSEMBLE,
            "an empty bundle has no wavefront",
            declaration="optical_path_length_m",
        )

    sphere = (
        fit_reference_sphere(
            positions_m=positions,
            plane_z_m=plane_z_m,
            optical_path_length_m=opl,
            wavelength_m=bundle.wavelength_m,
            initial_center_m=observation_point_m,
        )
        if fit_sphere
        else ReferenceSphere(
            center_m=observation_point_m,
            initial_center_m=observation_point_m,
            residual_rms_waves=float(
                np.std(
                    _sphere_residual(
                        positions[:, 0], positions[:, 1], plane_z_m, opl, observation_point_m
                    )
                )
                / bundle.wavelength_m
            ),
            initial_residual_rms_waves=float("nan"),
            iterations=0,
        )
    )

    residual = _sphere_residual(
        positions[:, 0], positions[:, 1], plane_z_m, opl, sphere.center_m
    )

    # RayBundle.amplitude is complex, because an amplitude may carry phase. This
    # oracle takes the modulus and refuses a bundle whose amplitude carries phase
    # rather than discarding it: that phase is part of the wavefront, and dropping
    # it would leave the oracle quietly modelling a different pupil than the
    # shipping path does. numpy's ComplexWarning would have been the only notice.
    amplitude = np.asarray(bundle.amplitude)
    if np.iscomplexobj(amplitude):
        phase = np.abs(np.angle(amplitude))
        if float(np.max(phase)) > 1e-12:
            raise ContractError(
                ContractCode.PHASOR_MISMATCH,
                "the bundle's amplitude carries a phase "
                f"(max |arg| = {float(np.max(phase)):.3e} rad), which this oracle "
                "would drop when taking the modulus",
                declaration="amplitude",
                remedy=(
                    "Fold the amplitude phase into the optical path length before "
                    "calling, or extend the oracle to carry a complex pupil "
                    "amplitude. Silently keeping only |A| models a different pupil "
                    "than the shipping reconstruction sums."
                ),
            )
        amplitude = np.abs(amplitude)

    return PupilAberration(
        positions_m=positions[:, :2],
        wavefront_error_m=residual,
        amplitude=np.asarray(amplitude, dtype=np.float64),
        wavelength_m=bundle.wavelength_m,
        pupil_radius_m=float(np.max(np.hypot(positions[:, 0], positions[:, 1]))),
        sphere=sphere,
        plane_z_m=plane_z_m,
    )


def _monomial_design(
    x: np.ndarray[Any, Any], y: np.ndarray[Any, Any], *, order: int
) -> np.ndarray[Any, Any]:
    """Monomials ``x^i y^j`` with ``i + j <= order``, on normalized coordinates."""
    columns = [
        (x**i) * (y ** (total - i))
        for total in range(order + 1)
        for i in range(total + 1)
    ]
    return np.stack(columns, axis=1)


@dataclass(frozen=True)
class FraunhoferPsf:
    """A PSF from a directly constructed pupil and one Fourier transform."""

    intensity: np.ndarray[Any, Any]
    #: ``(dy, dx)`` at the image plane: ``lambda * R / (N_fft * pupil pitch)``.
    sample_pitch_m: tuple[float, float]
    wavelength_m: float
    distance_m: float
    pupil_grid_n: int
    fft_grid_n: int
    pupil_pitch_m: float
    #: Relative residual of the polynomial fit to the sampled wavefront error.
    fit_residual_rms_waves: float
    fit_order: int
    diagnostics: dict[str, Any] = dataclass_field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "oracle": "fraunhofer_fft",
            "independent_of": "C_RAY_TO_WAVE reconstruction and Chromatix ASM propagation",
            "shares_with_shipping_path": "the traced OPD map and amplitude weights",
            "sample_pitch_m": list(self.sample_pitch_m),
            "pixel_scale_rule": "lambda * R / (N_fft * pupil_pitch)",
            "wavelength_m": self.wavelength_m,
            "distance_m": self.distance_m,
            "pupil_grid_n": self.pupil_grid_n,
            "fft_grid_n": self.fft_grid_n,
            "pupil_pitch_m": self.pupil_pitch_m,
            "fit_order": self.fit_order,
            "fit_residual_rms_waves": self.fit_residual_rms_waves,
            **self.diagnostics,
        }


def fraunhofer_psf(
    aberration: PupilAberration,
    *,
    pupil_pitch_m: float,
    pupil_grid_n: int,
    fft_grid_n: int | None = None,
    distance_m: float,
    fit_order: int = 6,
    uniform_amplitude: bool = False,
) -> FraunhoferPsf:
    """The independent oracle: build the pupil, transform it once.

    Steps, none of which the shipping path performs:

    1. Least-squares fit ``W(x, y)`` and the amplitude with monomials up to
       ``fit_order`` on normalized pupil coordinates, from the scattered ray
       samples. The shipping path never fits anything -- it accumulates one plane
       wavelet per ray.
    2. Evaluate the fit on a regular grid and mask to the traced pupil radius.
    3. ``P = A exp(+i 2 pi W / lambda)`` inside the aperture, zero outside. The
       ``+`` is this project's spatial factor ``exp(+i k z)``.
    4. One FFT. The field at the focus of a converging wave of radius ``R`` is the
       Fourier transform of its pupil function; the converging term itself is not
       applied here, it is what the geometry encodes.

    ``uniform_amplitude`` replaces the fitted amplitude with its mean. That is a
    negative control for the blind-spot audit, not a modelling option: under the
    near-uniform hexapolar weights of this slice it should change almost nothing,
    and if it changes nothing at all the amplitude path is untested.
    """
    n_fft = int(fft_grid_n if fft_grid_n is not None else pupil_grid_n)
    if n_fft < pupil_grid_n:
        raise ContractError(
            ContractCode.SHAPE_MISMATCH,
            f"fft_grid_n={n_fft} is smaller than pupil_grid_n={pupil_grid_n}",
            declaration="fft_grid_n",
            remedy="The FFT grid sets the image pixel scale; it may pad, not crop.",
        )

    radius = aberration.pupil_radius_m
    x_s = aberration.positions_m[:, 0] / radius
    y_s = aberration.positions_m[:, 1] / radius
    design = _monomial_design(x_s, y_s, order=fit_order)

    w_coefficients, *_ = np.linalg.lstsq(design, aberration.wavefront_error_m, rcond=None)
    w_fit_at_rays = design @ w_coefficients
    fit_residual = float(
        np.std(aberration.wavefront_error_m - w_fit_at_rays) / aberration.wavelength_m
    )

    amplitude = aberration.amplitude
    if uniform_amplitude:
        a_coefficients = None
    else:
        a_coefficients, *_ = np.linalg.lstsq(design, amplitude, rcond=None)

    # Pupil grid, on the pinned origin rule so it matches every other grid here.
    axis = (np.arange(pupil_grid_n, dtype=np.float64) - pupil_grid_n // 2) * pupil_pitch_m
    gy, gx = np.meshgrid(axis, axis, indexing="ij")
    inside = np.hypot(gy, gx) <= radius

    gxs = (gx / radius)[inside]
    gys = (gy / radius)[inside]
    grid_design = _monomial_design(gxs, gys, order=fit_order)

    w_grid = np.zeros((pupil_grid_n, pupil_grid_n), dtype=np.float64)
    w_grid[inside] = grid_design @ w_coefficients
    a_grid = np.zeros((pupil_grid_n, pupil_grid_n), dtype=np.float64)
    a_grid[inside] = (
        float(np.mean(amplitude)) if a_coefficients is None else grid_design @ a_coefficients
    )
    a_grid = np.clip(a_grid, 0.0, None)  # type: ignore[assignment]

    pupil = np.zeros((n_fft, n_fft), dtype=np.complex128)
    offset = (n_fft - pupil_grid_n) // 2
    pupil[offset : offset + pupil_grid_n, offset : offset + pupil_grid_n] = a_grid * np.exp(
        1j * 2.0 * math.pi * w_grid / aberration.wavelength_m
    )

    spectrum = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(pupil)))
    intensity = np.abs(spectrum) ** 2
    pitch = aberration.wavelength_m * distance_m / (n_fft * pupil_pitch_m)

    return FraunhoferPsf(
        intensity=intensity,
        sample_pitch_m=(pitch, pitch),
        wavelength_m=aberration.wavelength_m,
        distance_m=distance_m,
        pupil_grid_n=int(pupil_grid_n),
        fft_grid_n=n_fft,
        pupil_pitch_m=float(pupil_pitch_m),
        fit_residual_rms_waves=fit_residual,
        fit_order=int(fit_order),
        diagnostics={
            "pupil_samples_inside_aperture": int(inside.sum()),
            "amplitude_source": "mean (uniform control)" if uniform_amplitude else "fitted",
            "fitted_wavefront_rms_waves": float(
                np.std(w_grid[inside]) / aberration.wavelength_m
            ),
            "sampled_wavefront_rms_waves": aberration.rms_waves,
            "spatial_factor": "exp(+i k z), this project's convention",
        },
    )


def radial_profile(
    intensity: np.ndarray[Any, Any],
    *,
    sample_pitch_m: tuple[float, float],
    center_index: tuple[int, int] | None = None,
    bin_width_m: float | None = None,
    max_radius_m: float | None = None,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Azimuthally averaged profile: ``(radii, mean intensity, sample count)``.

    Averaging over azimuth is the right comparison for a rotationally symmetric
    PSF and the wrong one for anything else -- it is exactly the "symmetric radial
    metric" the blind-spot audit names as able to hide an orientation error. Use it
    for the Airy comparison; do not use it alone to decide that two PSFs agree.
    """
    ny, nx = intensity.shape
    dy, dx = sample_pitch_m
    iy, ix = center_index if center_index is not None else (ny // 2, nx // 2)
    y = (np.arange(ny, dtype=np.float64) - iy) * dy
    x = (np.arange(nx, dtype=np.float64) - ix) * dx
    r = np.hypot(y[:, None], x[None, :]).ravel()
    values = np.asarray(intensity, dtype=np.float64).ravel()

    width = bin_width_m if bin_width_m is not None else min(dy, dx)
    limit = max_radius_m if max_radius_m is not None else float(r.max())
    keep = r <= limit
    r_kept, values_kept = r[keep], values[keep]

    index = np.floor(r_kept / width).astype(int)
    counts = np.bincount(index)
    sums = np.bincount(index, weights=values_kept)
    nonempty = counts > 0
    radii = (np.arange(counts.size, dtype=np.float64) + 0.5) * width
    return radii[nonempty], sums[nonempty] / counts[nonempty], counts[nonempty]


def azimuthal_profile(
    intensity: np.ndarray[Any, Any],
    *,
    sample_pitch_m: tuple[float, float],
    center_m: tuple[float, float] = (0.0, 0.0),
    max_radius_m: float,
    radial_samples: int = 512,
    azimuthal_samples: int = 128,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Azimuthally averaged profile on a fine radial grid, by interpolation.

    :func:`radial_profile` bins pixels by radius, so its radial resolution is the
    pixel pitch. That is fatal for locating a first null: at the frozen M3
    sampling the Airy radius spans 2.44 pixels, and binning at that width smears
    the null and biases its position. On the self-check that error was 13%.

    This samples the array on a polar grid with bilinear interpolation instead, so
    the radial resolution is a free parameter. What it cannot do is invent
    information the sampling does not contain: a null in an array sampled at
    2.66 um is only located to a fraction of 2.66 um, and M3.8 reports that
    uncertainty rather than the interpolator's precision.
    """
    from scipy.ndimage import map_coordinates  # type: ignore[import-untyped]

    dy, dx = sample_pitch_m
    ny, nx = intensity.shape
    radii = np.linspace(0.0, max_radius_m, radial_samples)
    angles = np.linspace(0.0, 2.0 * math.pi, azimuthal_samples, endpoint=False)

    y = center_m[0] + radii[:, None] * np.sin(angles)[None, :]
    x = center_m[1] + radii[:, None] * np.cos(angles)[None, :]
    rows = y / dy + ny // 2
    columns = x / dx + nx // 2

    sampled = map_coordinates(
        np.asarray(intensity, dtype=np.float64),
        np.stack([rows.ravel(), columns.ravel()]),
        order=1,
        mode="constant",
        cval=np.nan,
    ).reshape(rows.shape)
    return radii, np.nanmean(sampled, axis=1)


def resample_to_grid(
    intensity: np.ndarray[Any, Any],
    *,
    from_pitch_m: tuple[float, float],
    to_pitch_m: tuple[float, float],
    to_shape: tuple[int, int],
    order: int = 3,
) -> np.ndarray[Any, Any]:
    """Point-sample a finely sampled PSF at another grid's pixel centres.

    Needed to compare the FFT oracle against the shipping PSF. The two are sampled
    on *different* pitches by construction -- ASM preserves the pupil pitch while
    the oracle's is ``lambda R / (N_fft dx)`` -- and comparing their azimuthal
    profiles directly measures the sampling difference, not the physics. On the
    diffraction-limited case that mismatch alone read as a 13.5% disagreement while
    the oracle's own first null was correct to 0.14%.

    Both arrays are point samples of the same continuous intensity, so evaluating
    the fine one at the coarse one's pixel centres is the comparison that means
    something. Interpolating the coarse one up instead would invent the detail the
    coarse grid does not have.
    """
    from scipy.ndimage import map_coordinates

    ny, nx = to_shape
    dy_to, dx_to = to_pitch_m
    dy_from, dx_from = from_pitch_m
    y = (np.arange(ny, dtype=np.float64) - ny // 2) * dy_to
    x = (np.arange(nx, dtype=np.float64) - nx // 2) * dx_to
    rows = y / dy_from + intensity.shape[0] // 2
    columns = x / dx_from + intensity.shape[1] // 2
    grid_rows, grid_columns = np.meshgrid(rows, columns, indexing="ij")
    resampled: np.ndarray[Any, Any] = map_coordinates(
        np.asarray(intensity, dtype=np.float64),
        np.stack([grid_rows.ravel(), grid_columns.ravel()]),
        order=order,
        mode="constant",
        cval=0.0,
    ).reshape(ny, nx)
    return resampled


def first_null_comparison(
    measured: np.ndarray[Any, Any],
    *,
    sample_pitch_m: tuple[float, float],
    wavelength_m: float,
    numerical_aperture: float,
    center_m: tuple[float, float] = (0.0, 0.0),
    center_index: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """Compare a measured first null against the analytic one, bias cancelled.

    A first-null estimator is badly biased at coarse sampling, and the frozen M3
    grid is coarse: 2.45 pixels per Airy radius. Measured on this project's own
    oracle, the same estimator reads the null of an *exactly known* Airy pattern
    **+11.9% high** at that pitch, falling to +1.8% at 9.8 px/radius and +0.02% at
    39 px/radius. So a raw comparison of a measured null against ``0.61 lambda/NA``
    at frozen sampling is dominated by the estimator, not by the slice.

    The fix is to run the same estimator over the analytic pattern sampled on the
    same grid and compare the two estimates. The bias then cancels to first order,
    and ``ratio_measured_over_analytic`` is the number that means something.
    ``analytic_estimator_bias`` is reported alongside so the cancellation is
    visible rather than assumed.
    """
    analytic = airy_psf_on_grid(
        shape=(int(measured.shape[0]), int(measured.shape[1])),
        sample_pitch_m=sample_pitch_m,
        wavelength_m=wavelength_m,
        numerical_aperture=numerical_aperture,
        center_m=center_m,
    )
    predicted = airy_first_null_radius_m(wavelength_m, numerical_aperture)
    limit = 3.0 * predicted

    def _null(array: np.ndarray[Any, Any]) -> float | None:
        radii, profile = azimuthal_profile(
            array / float(np.max(array)),
            sample_pitch_m=sample_pitch_m,
            center_m=center_m,
            max_radius_m=limit,
            radial_samples=1200,
            azimuthal_samples=256,
        )
        return measure_first_null_radius_m(radii, profile)

    measured_null = _null(np.asarray(measured, dtype=np.float64))
    analytic_null = _null(analytic)

    return {
        "predicted_first_null_m": predicted,
        "measured_first_null_m": measured_null,
        "analytic_same_grid_first_null_m": analytic_null,
        "analytic_estimator_bias": (
            None if analytic_null is None else analytic_null / predicted - 1.0
        ),
        "ratio_measured_over_analytic": (
            None
            if (measured_null is None or analytic_null is None or analytic_null == 0.0)
            else measured_null / analytic_null
        ),
        "ratio_measured_over_predicted": (
            None if measured_null is None else measured_null / predicted
        ),
        "pixels_per_airy_radius": predicted / min(sample_pitch_m),
        "center_index_used": list(center_index) if center_index is not None else None,
        "method": (
            "azimuthal average by bilinear interpolation, parabola-refined minimum; "
            "the same estimator is applied to the analytic pattern on the same grid "
            "so its sampling bias cancels in ratio_measured_over_analytic"
        ),
    }


def measure_first_null_radius_m(
    radii: np.ndarray[Any, Any],
    profile: np.ndarray[Any, Any],
) -> float | None:
    """Radius of the first local minimum of a radial profile, parabola-refined.

    Returns ``None`` when the profile has no interior minimum, which is the honest
    answer for a PSF whose first null is unresolved at the sampling in hand -- not
    a number to be compared against an oracle.
    """
    values = np.asarray(profile, dtype=np.float64)
    if values.size < 5:
        return None
    interior = np.arange(1, values.size - 1)
    minima = interior[(values[1:-1] < values[:-2]) & (values[1:-1] <= values[2:])]
    if minima.size == 0:
        return None
    i = int(minima[0])
    y0, y1, y2 = values[i - 1], values[i], values[i + 1]
    denominator = y0 - 2.0 * y1 + y2
    delta = 0.0 if denominator == 0.0 else 0.5 * (y0 - y2) / denominator
    spacing = float(radii[i + 1] - radii[i])
    return float(radii[i] + delta * spacing)
