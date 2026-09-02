"""The curvature-induced error bound for tangent-plane patches -- SI S3.

CHE-195 (R10.3). Pure arithmetic over a patch width and a radius, with no array
and no representation, which is why it is its own module: the local-patch route
consumes it as a *precondition*, and a precondition that lived inside the thing it
guards would be checkable only by running that thing.

The ray-wave method extracts a local angular spectrum from a patch treated as
locally planar. On a curved surface that approximation costs accuracy, and SI S3
bounds the cost. Over a patch of width `D` on a surface of local radius `R`, the
sag contributes a quadratic phase (eq S6) whose gradient is a position-linear
spatial frequency (eq S7); the spread at the patch edge (eq S8) maps through
`sin(theta) = lambda f` to

    eps_curv <= arcsin( D / (2R) )                                     (eq S9)

Two properties make it usable as a precondition rather than a footnote: it is
**independent of the DOE phase profile**, and it is monotone in `D/R`. The planar
limit `R -> inf` gives exactly zero, matching SI S2's statement that planar
patches have no intrinsic upper size bound.

Its own assumptions are validity limits in turn, and are recorded rather than
implied: one principal curvature direction at a time (the 2-D result follows by
applying the argument along both principal axes), quadratic sag, and `D << R`.

The bound is enforced, not advised
----------------------------------
A patch too wide for the surface's curvature produces a plausible field with a
direction error no intensity metric will show. `require_patch_within_curvature`
therefore **refuses**, and the signed margin travels with the result either way.
That is the same argument R07.3's measure refusal makes, and it fails the same way
under schedule pressure: the honest form of "this might be inaccurate" is a
refusal with the number attached.

`measured_tangent_plane_direction_error` exists because a bound merely plotted
beside a measurement is not a bound. It builds the sag phase and measures what a
*local* angular spectrum actually reports, so eq S9 is checked rather than
trusted.
"""

from __future__ import annotations

import math

import numpy as np

from representations import ContractError

__all__ = [
    "curvature_direction_error_bound",
    "curvature_observability_width",
    "max_patch_width_for_error",
    "measured_tangent_plane_direction_error",
    "require_patch_within_curvature",
]

#: `D << R` is an assumption of the derivation, not a consequence of it. Past this
#: ratio the quadratic-sag expansion is no longer the leading behaviour, so the
#: bound is reported with that caveat attached rather than silently trusted.
THIN_PATCH_RATIO = 0.1


def curvature_observability_width(*, wavelength_m: float, radius_m: float) -> float:
    """Patch width below which the curvature error is not spectrally observable.

    Measured while validating eq S9 (CHE-27), and **not stated in the paper** --
    which is why it is here rather than assumed away.

    A patch of width `D` resolves directions no finer than `lambda / D`, while the
    curvature spread it carries is `D / 2R`. The spread is therefore visible in a
    local angular spectrum only when

        D / (2R) > lambda / D    <=>    D > sqrt(2 lambda R)

    Below that width the patch's own diffraction limit exceeds the curvature
    effect, so eq S9 remains true but is conservative *by construction* rather
    than slack: measured-to-bound ratios of 200:1 are expected there and are a
    property of the aperture.

    Practically: a curvature budget cannot be validated empirically on a patch
    smaller than this. The measurement has nothing to report. The bound is still
    the right thing to enforce; it simply cannot be confirmed tight in that
    regime, and a test that tried would be measuring the aperture.
    """
    if math.isinf(radius_m):
        return 0.0
    return math.sqrt(2.0 * wavelength_m * radius_m)


def curvature_direction_error_bound(*, patch_width_m: float, radius_m: float) -> float:
    """`eps_curv <= arcsin(D / 2R)`, in radians (SI eq S9).

    `radius_m = inf` is the planar case and returns exactly 0.0 -- not a small
    number, zero, because a planar patch has no curvature error to bound.
    """
    if not (patch_width_m > 0.0 and math.isfinite(patch_width_m)):
        raise ContractError(
            "UNIT_NOT_SI",
            f"the patch width must be a positive finite length in metres, got "
            f"{patch_width_m!r}",
            declaration="patch_width_m",
        )
    if math.isinf(radius_m):
        return 0.0
    if not (radius_m > 0.0):
        raise ContractError(
            "UNIT_NOT_SI",
            f"the radius of curvature must be positive or infinite, got {radius_m!r}",
            declaration="radius_m",
        )

    ratio = patch_width_m / (2.0 * radius_m)
    if ratio >= 1.0:
        raise ContractError(
            "SHAPE_MISMATCH",
            f"the patch width {patch_width_m:.3e} m exceeds twice the radius "
            f"{radius_m:.3e} m, so `arcsin` is undefined: the patch subtends more than "
            "the surface can support and the tangent-plane picture has no meaning here. "
            "There is no bound to report rather than a large one.",
            declaration="patch_width_m",
            remedy=(
                "Reduce the patch width below 2R, and well below it for the bound to "
                "be useful at all."
            ),
        )
    return math.asin(ratio)


def max_patch_width_for_error(*, error_threshold_rad: float, radius_m: float) -> float:
    """The largest patch width whose bound stays under a threshold.

    The inverse of `curvature_direction_error_bound`, so a caller sizes a patch
    from an accuracy requirement instead of guessing and re-checking.
    """
    if not (error_threshold_rad > 0.0):
        raise ContractError(
            "UNIT_NOT_SI",
            f"the error threshold must be positive radians, got {error_threshold_rad!r}",
            declaration="error_threshold_rad",
        )
    if math.isinf(radius_m):
        return math.inf
    if error_threshold_rad >= math.pi / 2:
        return 2.0 * radius_m
    return 2.0 * radius_m * math.sin(error_threshold_rad)


def require_patch_within_curvature(
    *, patch_width_m: float, radius_m: float, error_threshold_rad: float
) -> dict[str, object]:
    """Refuse a patch too wide for the surface's curvature; report the signed margin.

    Returns the budget as a mapping -- a `dict` and not a record type, because
    R10.3 budgets **0** classes and this one is consumed by being merged into the
    operator's diagnostics rather than held.

    `margin_rad` is **signed**: `threshold - bound`, so a caller sees how close it
    is to the boundary rather than only whether it crossed one. That is the same
    shape R10.4's three margins take, deliberately.

    There is no `enforce=False`. The reference implementation had one, and a
    validity envelope with an off switch is the advisory bound R10.3's risk
    section is about -- a patch too wide produces a plausible field whose direction
    error no intensity metric will show. A caller who wants to *measure* that
    regime calls `measured_tangent_plane_direction_error`, which is what the
    switch was really for.
    """
    bound = curvature_direction_error_bound(
        patch_width_m=patch_width_m, radius_m=radius_m
    )
    limit = max_patch_width_for_error(
        error_threshold_rad=error_threshold_rad, radius_m=radius_m
    )
    margin = error_threshold_rad - bound
    thin = math.isinf(radius_m) or (patch_width_m / radius_m) <= THIN_PATCH_RATIO

    if margin < 0.0:
        raise ContractError(
            "SHAPE_MISMATCH",
            f"the curvature bound {bound:.6e} rad exceeds the declared threshold "
            f"{error_threshold_rad:.6e} rad for a {patch_width_m:.3e} m patch on a "
            f"{radius_m:.3e} m radius -- a signed margin of {margin:.3e} rad. Refused "
            "rather than warned: a patch too wide for the tangent-plane approximation "
            "produces a plausible field whose direction error no intensity metric shows.",
            declaration="patch_px",
            remedy=(
                f"Use a patch no wider than {limit:.3e} m at this radius, or declare a "
                "larger error threshold and own the accuracy claim."
            ),
        )
    return {
        "patch_width_m": patch_width_m,
        "radius_m": radius_m,
        "error_bound_rad": bound,
        "error_threshold_rad": error_threshold_rad,
        "margin_rad": margin,
        "max_patch_width_m": limit,
        "thin_patch_assumption_holds": thin,
        "bound": "eps_curv <= arcsin(D / 2R)  [ACS Photonics 2026 eq 4 / SI eq S9]",
        "independent_of": "the DOE phase profile",
        "assumptions": [
            "one principal curvature direction at a time",
            "quadratic sag",
            "D << R",
        ],
    }


def measured_tangent_plane_direction_error(
    *,
    patch_width_m: float,
    radius_m: float,
    wavelength_m: float,
    samples: int = 4096,
    windows: int = 16,
) -> float:
    """The direction error a curved patch actually incurs when treated as planar.

    Builds the sag phase `phi = (k0 / 2R) x^2` of SI eq S6, measures the direction
    the **local** angular spectrum reports at points across the patch, and returns
    the largest deviation from the tangent plane's own direction (zero). Eq S9 must
    bound this; a bound merely plotted beside a measurement is not a bound.

    **The measurement is local, not global, and that choice is load-bearing.**
    Taking the extreme angle present in the *whole* patch's spectrum instead
    reports the aperture's edge diffraction: for a 50-lambda flat patch the first
    sinc null already sits at 0.02 rad, twenty times the curvature spread of a
    100000-lambda radius. Eq S9 bounds the error in the *extracted direction*, not
    the total angular content of a truncated aperture, and a measurement that
    conflated the two would make the bound look violated everywhere.

    Each window's spectral **centroid** is used, which is unbiased by that window's
    own symmetric diffraction spread.
    """
    if samples < 64:
        raise ContractError(
            "SHAPE_MISMATCH",
            f"at least 64 samples are needed to resolve the sag spectrum, got {samples}",
            declaration="samples",
        )
    if windows < 2 or samples % windows:
        raise ContractError(
            "SHAPE_MISMATCH",
            f"samples ({samples}) must divide evenly into at least 2 windows, got "
            f"{windows}",
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
