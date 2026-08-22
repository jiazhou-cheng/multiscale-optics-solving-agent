"""Independent analytic oracles for L1-WAVE-01.

This module imports **no Chromatix and no JAX**. Each oracle is written from
its defining equation so the evaluator can never use the solver under test to
compute its own expectation.

Three oracles, in order of how much they assume:

1. :func:`plane_wave_transfer` -- *exact*. A plane wave whose transverse
   frequency lands exactly on an FFT bin is an eigenmode of the discrete
   angular-spectrum operator, so the propagated field is the input times
   ``exp(i k_z z)`` with no approximation whatsoever: no paraxial expansion,
   no window truncation, no interpolation. Any residual is floating-point
   round-off.
2. :func:`fresnel_focal_field` -- *paraxial*. The Fresnel/Fourier focal field
   of a rectangular pupil behind an ideal thin lens. Exact within the Fresnel
   approximation and within the continuous-aperture idealization.
3. :func:`richards_wolf_focal_field` -- *Debye--Wolf*. Vectorial focal field
   of an aplanatic high-NA objective, by Gauss--Legendre quadrature over the
   convergence angle. Exact within the Debye approximation; its own
   quadrature convergence is measured rather than assumed.

:func:`numpy_angular_spectrum` is a fourth, *non-oracle* reference: a float64
angular-spectrum propagation used only to separate a paraxial oracle's own
model error from the solver's implementation error.

Shared conventions (asserted by the evaluator, recorded in every artifact)
-------------------------------------------------------------------------
- SI units (metres, radians) throughout.
- Array axes ``(y, x)`` row-major; index ``n // 2`` along each spatial axis is
  coordinate zero. This is Chromatix's own grid centering, verified against
  ``Field.grid``; for even ``n`` the sampled window is asymmetric by one
  sample.
- Phasor ``exp(-i omega t)``, so a forward wave carries ``exp(+i k z)``. This
  matches ``compute_asm_propagator``'s ``exp(+i phase)`` for ``z >= 0``, and
  fixes the signs of the converging-lens phase, the wavefront curvature, and
  the Gouy shift.
- Arrays hold complex field *amplitude*. Intensity is ``abs(u) ** 2`` and is
  never substituted for the amplitude.
- Vector fields are returned as ``(E_x, E_y, E_z)``. Chromatix's own
  ``VectorField.u`` uses the opposite order ``(E_z, E_y, E_x)`` on its last
  axis; the conversion happens once, in the benchmark adapter, and never here.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.special import jv


# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------
def grid_coordinates_m(n: int, pitch_m: float) -> np.ndarray:
    """Chromatix grid centering: index ``n // 2`` is coordinate zero."""
    return (np.arange(n, dtype=np.float64) - n // 2) * pitch_m


def grid_xy(n: int, pitch_m: float) -> tuple[np.ndarray, np.ndarray]:
    """Broadcastable ``(x, y)`` for an ``(y, x)`` array: x is a row, y a column."""
    coordinates = grid_coordinates_m(n, pitch_m)
    return coordinates[None, :], coordinates[:, None]


# ===========================================================================
# Case 1 -- exact homogeneous propagation primitive
# ===========================================================================
def fft_bin_frequency(mode: int, n: int, pitch_m: float) -> float:
    """Transverse spatial frequency of FFT bin ``mode`` in cycles per metre."""
    return mode / (n * pitch_m)


def axial_wavenumber(
    frequency_y: float, frequency_x: float, wavelength_m: float, refractive_index: float
) -> float:
    """``k_z = (2 pi n / lambda) sqrt(1 - (lambda/n)^2 (fx^2 + fy^2))``.

    The exact homogeneous-medium dispersion relation -- no paraxial expansion.
    Raises for an evanescent mode, which this benchmark deliberately never
    launches (an evanescent order would test the solver's decay handling, a
    different question from the propagating phase transfer).
    """
    argument = 1.0 - (wavelength_m / refractive_index) ** 2 * (frequency_x**2 + frequency_y**2)
    if argument <= 0.0:
        raise ValueError(
            f"mode (fy={frequency_y}, fx={frequency_x}) is evanescent at "
            f"wavelength {wavelength_m} m, n={refractive_index}; this case only launches "
            "propagating orders."
        )
    return float(2.0 * np.pi * refractive_index / wavelength_m * np.sqrt(argument))


def plane_wave_mode(n: int, mode_y: int, mode_x: int) -> np.ndarray:
    """Unit-amplitude plane wave on exactly FFT bin ``(mode_y, mode_x)``.

    Built from the sample indices, not from physical coordinates, so the field
    is *exactly* periodic on the grid to floating-point precision. That is what
    makes it an eigenmode of the discrete propagator, and it is why this case
    must run unpadded: zero-padding a periodic plane wave manufactures an edge
    the physics does not contain.
    """
    rows = np.arange(n, dtype=np.float64)[:, None]
    columns = np.arange(n, dtype=np.float64)[None, :]
    return np.exp(2j * np.pi * (mode_x * columns + mode_y * rows) / n)


def plane_wave_transfer(field: np.ndarray, axial_wavenumber_rad_per_m: float, z_m: float):
    """Exact propagated field: ``u * exp(i k_z z)``. No approximation."""
    return np.asarray(field, dtype=np.complex128) * np.exp(1j * axial_wavenumber_rad_per_m * z_m)


def float32_phase_round_off(axial_wavenumber_rad_per_m: float, z_m: float) -> float:
    """Predicted phase round-off for a complex64 transfer function, in radians.

    The propagator evaluates ``exp(i k_z z)`` in single precision, so the
    accumulated phase ``k_z z`` -- which reaches thousands of radians here --
    carries a relative error of one float32 epsilon. The resulting absolute
    phase error is therefore ``eps * |k_z z|``, *not* a fixed constant. Case 1
    scales its tolerance with this quantity instead of hard-coding a number,
    so the test stays meaningful at every propagation distance.
    """
    return float(np.finfo(np.float32).eps * abs(axial_wavenumber_rad_per_m * z_m))


# ===========================================================================
# Case 2 -- ideal paraxial focusing of a finite rectangular pupil
# ===========================================================================
def rect_pupil_mask(n: int, pitch_m: float, samples_x: int, samples_y: int) -> np.ndarray:
    """Rectangular aperture of exactly ``samples_x`` by ``samples_y`` samples.

    Odd sample counts are required so the aperture is exactly symmetric about
    the coordinate origin on the ``n // 2``-centred grid. Specifying the
    aperture in *samples* rather than in metres removes a half-pixel ambiguity
    between the sampled pupil and the continuous width the analytic oracle
    assumes.
    """
    if samples_x % 2 == 0 or samples_y % 2 == 0:
        raise ValueError(
            f"aperture sample counts must be odd for exact symmetry about the origin; "
            f"got samples_x={samples_x}, samples_y={samples_y}."
        )
    x, y = grid_xy(n, pitch_m)
    half_x = samples_x * pitch_m / 2.0
    half_y = samples_y * pitch_m / 2.0
    return (np.abs(x) <= half_x) & (np.abs(y) <= half_y)


def thin_lens_pupil_field(
    *,
    n: int,
    pitch_m: float,
    samples_x: int,
    samples_y: int,
    focal_length_m: float,
    tilt_x_rad: float,
    tilt_y_rad: float,
    wavelength_m: float,
    refractive_index: float,
) -> np.ndarray:
    """Field immediately after a rectangular pupil and an ideal thin lens.

    ``rect(x) rect(y) * exp(-i k (x^2+y^2) / 2f) * exp(i k (theta_x x + theta_y y))``

    The converging lens carries a *negative* quadratic phase in the
    ``exp(+i k z)`` convention; flipping that sign turns the lens into a
    diverging one and moves the focus to ``z = -f``, which is one of the
    perturbations the evaluator must catch.

    Chromatix's own ``thin_lens`` is deliberately not used. The lens here is
    part of the *specified input field* that the analytic oracle describes
    exactly, which keeps free-space propagation the only thing under test and
    avoids validating one Chromatix element against another.
    """
    wavenumber = 2.0 * np.pi * refractive_index / wavelength_m
    x, y = grid_xy(n, pitch_m)
    mask = rect_pupil_mask(n, pitch_m, samples_x, samples_y)
    lens = np.exp(-1j * wavenumber * (x**2 + y**2) / (2.0 * focal_length_m))
    tilt = np.exp(1j * wavenumber * (tilt_x_rad * x + tilt_y_rad * y))
    return mask * lens * tilt


def fresnel_focal_field(
    *,
    n: int,
    pitch_m: float,
    aperture_x_m: float,
    aperture_y_m: float,
    focal_length_m: float,
    tilt_x_rad: float,
    tilt_y_rad: float,
    wavelength_m: float,
    refractive_index: float,
) -> np.ndarray:
    """Analytic focal field one focal length behind the pupil of :func:`thin_lens_pupil_field`.

    Fresnel propagation over ``z = f`` cancels the lens's quadratic phase
    exactly, leaving a scaled Fourier transform of the pupil:

    ``U(x') = (e^{ikf} / (i lambda f)) e^{i k r'^2 / 2f}
              L_x sinc(L_x (x' - f theta_x) / (lambda f))
              L_y sinc(L_y (y' - f theta_y) / (lambda f))``

    Two consequences the evaluator tests directly: the focus sits at
    ``+f * theta`` (a *signed* prediction), and the residual quadratic phase
    ``e^{i k r'^2 / 2f}`` survives -- it is not an aberration, and dropping it
    would silently break the complex-field comparison while leaving the
    intensity untouched.
    """
    wavenumber = 2.0 * np.pi * refractive_index / wavelength_m
    x, y = grid_xy(n, pitch_m)
    argument_x = aperture_x_m * (x - focal_length_m * tilt_x_rad) / (wavelength_m * focal_length_m)
    argument_y = aperture_y_m * (y - focal_length_m * tilt_y_rad) / (wavelength_m * focal_length_m)
    prefactor = np.exp(1j * wavenumber * focal_length_m) / (1j * wavelength_m * focal_length_m)
    residual_phase = np.exp(1j * wavenumber * (x**2 + y**2) / (2.0 * focal_length_m))
    # np.sinc is the normalized sinc, sin(pi t) / (pi t) -- the same convention
    # as the analytic rectangular-aperture transform above.
    return (
        prefactor
        * residual_phase
        * aperture_x_m
        * np.sinc(argument_x)
        * aperture_y_m
        * np.sinc(argument_y)
    )


def sinc_squared_reference() -> dict[str, float]:
    """Scale-free shape constants of a ``sinc^2`` line spread function.

    These are pure numbers: they depend on neither wavelength, aperture, nor
    focal length, so a case that reproduces them has the *shape* right
    independently of any scale calibration.
    """
    return {
        # sinc(t) first zero
        "first_null_in_t": 1.0,
        # sinc^2(t) = 1/2 at t = 0.442946..., so FWHM = 2t
        "fwhm_in_t": 0.8858929413,
        # first sidelobe of sinc^2, at t = 1.4303...
        "first_sidelobe_ratio": 0.04718036,
    }


# ===========================================================================
# Case 3 -- vectorial high-NA focusing (Richards-Wolf / Debye-Wolf)
# ===========================================================================
def aplanatic_pupil_amplitude(
    *,
    n: int,
    pitch_m: float,
    focal_length_m: float,
    numerical_aperture: float,
    refractive_index: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Aplanatic (Abbe sine condition) pupil amplitude and its ``cos(theta)`` map.

    A collimated beam entering an aplanatic objective maps onto the reference
    sphere with amplitude scaled by ``sqrt(cos theta)`` (energy projection),
    where ``sin theta = r / f``. Returns ``(amplitude, cos_theta)`` with the
    amplitude already masked to the pupil ``r <= f * NA / n``.
    """
    x, y = grid_xy(n, pitch_m)
    radius_squared = x**2 + y**2
    pupil_radius_m = focal_length_m * numerical_aperture / refractive_index
    mask = radius_squared <= pupil_radius_m**2
    sin_theta_squared = np.clip(radius_squared / focal_length_m**2, 0.0, 1.0)
    cos_theta = np.sqrt(np.clip(1.0 - sin_theta_squared, 0.0, None))
    return mask * np.sqrt(cos_theta), np.where(mask, cos_theta, 1.0)


def richards_wolf_focal_field(
    *,
    radius_m: np.ndarray,
    azimuth_rad: np.ndarray,
    defocus_m: float,
    numerical_aperture: float,
    refractive_index: float,
    wavelength_m: float,
    quadrature_points: int,
    apodization_exponent: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorial focal field of an x-polarized aplanatic objective, in float64.

    Debye--Wolf integral, evaluated by Gauss--Legendre quadrature over the
    convergence angle ``theta`` in ``[0, arcsin(NA/n)]``:

    ``I_0 = int (cos t)^p sin t (1 + cos t) J_0(k rho sin t) e^{i k z cos t} dt``
    ``I_1 = int (cos t)^p sin^2 t     J_1(k rho sin t) e^{i k z cos t} dt``
    ``I_2 = int (cos t)^p sin t (1 - cos t) J_2(k rho sin t) e^{i k z cos t} dt``

    ``E = -i [ I_0 + I_2 cos 2phi,  I_2 sin 2phi,  -2i I_1 cos phi ]``

    ``apodization_exponent`` is ``p``: 0.5 is the aplanatic sine-condition
    objective. It is exposed because it is exactly the factor Chromatix 0.6.0
    omits, so the evaluator can report which apodization the solver actually
    behaves like instead of assuming one.

    Returns ``(E_x, E_y, E_z)``.
    """
    maximum_angle_rad = np.arcsin(numerical_aperture / refractive_index)
    wavenumber = 2.0 * np.pi * refractive_index / wavelength_m

    nodes, weights = np.polynomial.legendre.leggauss(quadrature_points)
    theta = 0.5 * maximum_angle_rad * (nodes + 1.0)
    weights = 0.5 * maximum_angle_rad * weights

    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    common = (
        (cos_theta**apodization_exponent)
        * sin_theta
        * np.exp(1j * wavenumber * defocus_m * cos_theta)
    )

    bessel_argument = wavenumber * np.asarray(radius_m, dtype=np.float64)[..., None] * sin_theta
    integral_0 = np.sum(weights * common * (1.0 + cos_theta) * jv(0, bessel_argument), axis=-1)
    integral_1 = np.sum(weights * common * sin_theta * jv(1, bessel_argument), axis=-1)
    integral_2 = np.sum(weights * common * (1.0 - cos_theta) * jv(2, bessel_argument), axis=-1)

    field_x = -1j * (integral_0 + integral_2 * np.cos(2.0 * azimuth_rad))
    field_y = -1j * (integral_2 * np.sin(2.0 * azimuth_rad))
    field_z = -1j * (-2j * integral_1 * np.cos(azimuth_rad))
    return field_x, field_y, field_z


def richards_wolf_on_grid(
    *,
    n: int,
    pitch_m: float,
    defocus_m: float,
    numerical_aperture: float,
    refractive_index: float,
    wavelength_m: float,
    quadrature_points: int,
    apodization_exponent: float = 0.5,
) -> np.ndarray:
    """:func:`richards_wolf_focal_field` on an ``(y, x)`` grid, stacked as ``(y, x, 3)``."""
    x, y = grid_xy(n, pitch_m)
    radius_m = np.sqrt(x**2 + y**2)
    azimuth_rad = np.arctan2(np.broadcast_to(y, radius_m.shape), np.broadcast_to(x, radius_m.shape))
    components = richards_wolf_focal_field(
        radius_m=radius_m,
        azimuth_rad=azimuth_rad,
        defocus_m=defocus_m,
        numerical_aperture=numerical_aperture,
        refractive_index=refractive_index,
        wavelength_m=wavelength_m,
        quadrature_points=quadrature_points,
        apodization_exponent=apodization_exponent,
    )
    return np.stack(components, axis=-1)


# ===========================================================================
# Independent float64 angular spectrum -- a cross-check, never the oracle
# ===========================================================================
def numpy_angular_spectrum(
    field: np.ndarray,
    *,
    pitch_m: float,
    wavelength_m: float,
    refractive_index: float,
    z_m: float,
    pad_width: int,
) -> np.ndarray:
    """Zero-padded exact (non-paraxial) angular-spectrum propagation in float64.

    Written directly from ``U(f) -> U(f) exp(i k_z z)`` with the same
    ``exp(+i k z)`` convention as the oracles above; evanescent components are
    dropped. Its only job is to split a paraxial oracle's own model error from
    the solver's implementation error -- two exact calculations should differ
    only by the solver's complex64 arithmetic.
    """
    padded = np.pad(np.asarray(field, dtype=np.complex128), pad_width, mode="constant")
    n_padded = padded.shape[0]
    frequency_y = np.fft.fftfreq(n_padded, d=pitch_m)[:, None]
    frequency_x = np.fft.fftfreq(n_padded, d=pitch_m)[None, :]

    argument = 1.0 - (wavelength_m / refractive_index) ** 2 * (frequency_x**2 + frequency_y**2)
    axial = 2.0 * np.pi * refractive_index / wavelength_m * np.sqrt(np.maximum(argument, 0.0))
    transfer = np.where(argument >= 0.0, np.exp(1j * axial * z_m), 0.0)

    spectrum = np.fft.fft2(np.fft.ifftshift(padded))
    return np.fft.fftshift(np.fft.ifft2(spectrum * transfer))


# ===========================================================================
# Sampling validity (measured facts, no verdict)
# ===========================================================================
def sampling_diagnostics(
    *,
    n: int,
    pitch_m: float,
    pad_width: int,
    wavelength_m: float,
    refractive_index: float,
    occupied_frequency_per_m: float,
    z_m: float,
) -> dict[str, Any]:
    """Observable sampling/window facts for one case; the evaluator judges them."""
    nyquist_frequency = 1.0 / (2.0 * pitch_m)
    evanescent_cutoff = refractive_index / wavelength_m
    padded_half_window_m = (n + 2 * pad_width) * pitch_m / 2.0
    geometric_shift_m = abs(z_m) * wavelength_m * occupied_frequency_per_m / refractive_index
    return {
        "input_half_window_m": n * pitch_m / 2.0,
        "padded_half_window_m": padded_half_window_m,
        "nyquist_frequency_per_m": nyquist_frequency,
        "occupied_frequency_per_m": occupied_frequency_per_m,
        "bandwidth_headroom": (
            nyquist_frequency / occupied_frequency_per_m
            if occupied_frequency_per_m > 0
            else float("inf")
        ),
        "evanescent_cutoff_per_m": evanescent_cutoff,
        "grid_reaches_evanescent": bool(nyquist_frequency > evanescent_cutoff),
        "geometric_shift_m": geometric_shift_m,
        "padded_window_over_geometric_shift": (
            padded_half_window_m / geometric_shift_m if geometric_shift_m > 0 else float("inf")
        ),
    }
