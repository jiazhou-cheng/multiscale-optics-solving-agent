"""Curvature-induced error bound for tangent-plane patches — SI S3 (CHE-27).

The ray–wave method extracts a local angular spectrum from a patch treated as
locally planar. On a curved surface that approximation costs accuracy, and SI
S3 bounds the cost. Over a patch of width ``D`` on a surface of local radius
``R``, the sag contributes a quadratic phase (eq S6) whose gradient is a
position-linear spatial frequency (eq S7); the spread at the patch edge (eq S8)
maps through ``sin(theta) = lambda f`` to a bound on the direction error:

    eps_curv <= arcsin( D / (2 R) )                                     (eq S9)

Two properties make it usable as a precondition rather than a footnote: it is
**independent of the DOE phase profile**, and it is monotone in ``D/R``. The
planar limit ``R -> infinity`` gives zero, matching SI S2's statement that
planar patches have no intrinsic upper size bound.

Its own assumptions are validity limits in turn, and are recorded rather than
implied: one principal curvature direction at a time (the 2-D result follows by
applying the argument along both principal axes), quadratic sag, and ``D << R``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from multiscale_optics_agent.couplers.contracts import ContractCode, ContractError

__all__ = [
    "CurvatureBudget",
    "check_patch",
    "curvature_direction_error_bound",
    "curvature_observability_width",
    "max_patch_width_for_error",
    "measured_tangent_plane_direction_error",
]


def curvature_observability_width(wavelength_m: float, radius_m: float) -> float:
    """Patch width below which the curvature error is not spectrally observable.

    Measured while validating eq S9 (CHE-27), and not stated in the paper.

    A patch of width ``D`` resolves directions no finer than ``lambda / D``,
    while the curvature spread it carries is ``D / 2R``. The spread is therefore
    only visible in a local angular spectrum when

        D / (2 R) > lambda / D    <=>    D > sqrt(2 lambda R)

    Below that width the patch's own diffraction limit exceeds the curvature
    effect, so eq S9 remains true but is conservative by construction rather
    than tight: measured/bound ratios of 200:1 are expected there, and are a
    property of the aperture, not a slack bound.

    Practically, this says a curvature budget cannot be validated empirically on
    a patch smaller than ``sqrt(2 lambda R)`` -- the measurement has nothing to
    report. The bound is still the right thing to enforce; it simply cannot be
    confirmed tight in that regime.
    """
    if math.isinf(radius_m):
        return 0.0
    return math.sqrt(2.0 * wavelength_m * radius_m)


def curvature_direction_error_bound(patch_width_m: float, radius_m: float) -> float:
    """``eps_curv <= arcsin(D / 2R)`` in radians (SI eq S9).

    ``radius_m = inf`` is the planar case and returns exactly 0.0.
    """
    if patch_width_m <= 0.0 or not math.isfinite(patch_width_m):
        raise ContractError(
            ContractCode.UNIT_NOT_SI,
            f"patch width must be a positive length in metres, got {patch_width_m!r}",
            declaration="patch_width_m",
        )
    if math.isinf(radius_m):
        return 0.0
    if radius_m <= 0.0 or math.isnan(radius_m):
        raise ContractError(
            ContractCode.UNIT_NOT_SI,
            f"radius of curvature must be positive or infinite, got {radius_m!r}",
            declaration="radius_m",
        )

    ratio = patch_width_m / (2.0 * radius_m)
    if ratio >= 1.0:
        # arcsin is undefined past 1: the patch subtends more than the surface
        # can support, so there is no bound to report rather than a large one.
        raise ContractError(
            ContractCode.SHAPE_MISMATCH,
            (
                f"patch width {patch_width_m:.3e} m exceeds twice the radius "
                f"{radius_m:.3e} m; the tangent-plane picture has no meaning here "
                "and eq S9 gives no bound"
            ),
            declaration="patch_width_m",
            remedy="Reduce the patch width below 2R, and well below it for the bound to be useful.",
        )
    return math.asin(ratio)


def max_patch_width_for_error(error_threshold_rad: float, radius_m: float) -> float:
    """Largest patch width whose bound stays under a caller's threshold.

    The inverse of :func:`curvature_direction_error_bound`, so a caller can size
    a patch from an accuracy requirement instead of guessing and re-checking.
    """
    if error_threshold_rad <= 0.0:
        raise ContractError(
            ContractCode.UNIT_NOT_SI,
            f"error threshold must be positive radians, got {error_threshold_rad!r}",
            declaration="error_threshold_rad",
        )
    if math.isinf(radius_m):
        return math.inf
    if error_threshold_rad >= math.pi / 2:
        return 2.0 * radius_m
    return 2.0 * radius_m * math.sin(error_threshold_rad)


@dataclass(frozen=True)
class CurvatureBudget:
    """The result of checking a patch against a caller's error threshold."""

    patch_width_m: float
    radius_m: float
    error_bound_rad: float
    error_threshold_rad: float
    max_patch_width_m: float
    within_budget: bool
    thin_patch_assumption_holds: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "patch_width_m": self.patch_width_m,
            "radius_m": self.radius_m,
            "error_bound_rad": self.error_bound_rad,
            "error_threshold_rad": self.error_threshold_rad,
            "max_patch_width_m": self.max_patch_width_m,
            "within_budget": self.within_budget,
            "thin_patch_assumption_holds": self.thin_patch_assumption_holds,
            "bound": "eps_curv <= arcsin(D / 2R)  [ACS Photonics 2026 eq 4 / SI eq S9]",
            "independent_of": "the DOE phase profile",
            "assumptions": [
                "one principal curvature direction at a time",
                "quadratic sag",
                "D << R",
            ],
        }


#: ``D << R`` is an assumption of the derivation, not a consequence of it. Past
#: this ratio the quadratic-sag expansion is no longer the leading behaviour, so
#: the bound is reported with that caveat attached rather than silently trusted.
_THIN_PATCH_RATIO = 0.1


def check_patch(
    *,
    patch_width_m: float,
    radius_m: float,
    error_threshold_rad: float,
    enforce: bool = True,
) -> CurvatureBudget:
    """Precondition check for a tangent-plane patch on a curved surface.

    Raises when the bound exceeds the caller's threshold and ``enforce`` is set,
    so a caller cannot silently exceed the tangent-plane approximation. Set
    ``enforce=False`` to measure the regime deliberately -- the budget still
    records that it was exceeded, so a result produced there is never mistaken
    for a valid one.
    """
    bound = curvature_direction_error_bound(patch_width_m, radius_m)
    limit = max_patch_width_for_error(error_threshold_rad, radius_m)
    within = bound <= error_threshold_rad
    thin = math.isinf(radius_m) or (patch_width_m / radius_m) <= _THIN_PATCH_RATIO

    if enforce and not within:
        raise ContractError(
            ContractCode.SHAPE_MISMATCH,
            (
                f"curvature bound {bound:.6e} rad exceeds the requested threshold "
                f"{error_threshold_rad:.6e} rad for a {patch_width_m:.3e} m patch on a "
                f"{radius_m:.3e} m radius"
            ),
            declaration="patch_width_m",
            remedy=(
                f"Use a patch no wider than {limit:.3e} m at this radius, or accept a "
                "larger direction error explicitly."
            ),
        )
    return CurvatureBudget(
        patch_width_m=patch_width_m,
        radius_m=radius_m,
        error_bound_rad=bound,
        error_threshold_rad=error_threshold_rad,
        max_patch_width_m=limit,
        within_budget=within,
        thin_patch_assumption_holds=thin,
    )


def measured_tangent_plane_direction_error(
    *,
    patch_width_m: float,
    radius_m: float,
    wavelength_m: float,
    samples: int = 4096,
    windows: int = 16,
) -> float:
    """Direction error actually incurred by treating a curved patch as planar.

    Builds the sag phase ``phi = (k0 / 2R) x^2`` of SI eq S6, then measures the
    propagation direction the **local** angular spectrum reports at points
    across the patch, and returns the largest deviation from the tangent plane's
    own direction (zero). Eq S9 must bound this; a bound merely plotted
    alongside a measurement is not a bound.

    The measurement is local, not global, and that choice matters. Taking the
    extreme angle present in the *whole* patch's spectrum instead reports the
    aperture's edge diffraction: for a 50-lambda flat patch the first sinc null
    already sits at 0.02 rad, twenty times the curvature spread of a
    100000-lambda radius. Eq S9 bounds the error in the *extracted direction*,
    not the total angular content of a truncated aperture, so the two must not
    be conflated -- and a measurement that conflated them would make the bound
    look violated everywhere.

    Each window's spectral centroid is used, which is unbiased by that window's
    own symmetric diffraction spread.
    """
    if samples < 64:
        raise ContractError(
            ContractCode.SHAPE_MISMATCH,
            f"need at least 64 samples to resolve the sag spectrum, got {samples}",
            declaration="samples",
        )
    if windows < 2 or samples % windows:
        raise ContractError(
            ContractCode.SHAPE_MISMATCH,
            f"samples ({samples}) must divide evenly into at least 2 windows, got {windows}",
            declaration="windows",
        )

    wavenumber = 2.0 * math.pi / wavelength_m
    x = np.linspace(-patch_width_m / 2.0, patch_width_m / 2.0, samples, endpoint=False)
    pitch = float(x[1] - x[0])
    sag_phase = 0.0 if math.isinf(radius_m) else wavenumber * x**2 / (2.0 * radius_m)
    patch = np.exp(1j * sag_phase) * np.ones_like(x)

    window_size = samples // windows
    frequency = np.fft.fftshift(np.fft.fftfreq(window_size, d=pitch))
    direction = frequency * wavelength_m
    propagating = np.abs(direction) < 1.0

    worst = 0.0
    for index in range(windows):
        segment = patch[index * window_size : (index + 1) * window_size]
        spectrum = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(segment)))
        power = np.abs(spectrum[propagating]) ** 2
        total = float(np.sum(power))
        if total <= 0.0:
            continue
        centroid = float(np.sum(direction[propagating] * power) / total)
        worst = max(worst, abs(math.asin(min(max(centroid, -1.0), 1.0))))
    return worst
