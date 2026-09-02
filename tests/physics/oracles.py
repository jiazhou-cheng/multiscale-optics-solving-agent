"""Analytic PSF oracles, and the metrics that compare a reconstruction to them.

CHE-198 (R11.2). **Evidence, not infrastructure.** These live under `tests/`
because an oracle exists to judge an implementation, and shipping it inside the
implementation is the circularity this repository's working rules name first: a
custom numerical routine must never be the answer key for the same project's
other custom numerical routine. An Airy formula in `src/` would also be a
capability the project claims, maintains and versions; here it is a thing a test
compares against.

The one oracle that decides
---------------------------
**O1 -- the analytic Airy pattern** -- is the only oracle in this file that may
decide a gate. It is paraxial, aberration-free, and shares no code and no traced
data with anything under test: `scipy.special.j1` and a radius, and nothing else.
That independence is exactly what makes it admissible.

An independent numerical propagator of our own -- the reference implementation
had one, "O2", a float64 ASM/Rayleigh-Sommerfeld solver written to check this
same coupler -- is *not* reproduced here and would not be admissible if it were.
Using our own numerics as the answer key for our own numerics is circular
validation, and the reference bundle's disposition records that O2 was
consulted for characterization only and decided nothing.

...and what O1 does not know
----------------------------
O1 is paraxial: it is a function of one number, `NA`, and it knows nothing about
how a real pupil maps radius to angle. So on an *exactly* spherical,
aberration-free converging wavefront it still disagrees, at order `NA^2` -- and
**how much depends on which pupil map that wavefront has.** Measured in
`test_psf_verification.py` at the f/9.7 singlet's NA and 64 rings:

* a **tangent-condition** cone (`rho = f tan(theta)`, the uniform area sampling of
  a flat pupil) -- 8.86e-4, 89 % of the frozen 1.0e-3 gate;
* an **aplanatic / sine-condition** cone (`rho = f sin(theta)`, with its
  `sqrt(cos theta)` apodization) at the same NA -- 1.19e-4.

The floor is therefore not "O1's paraxiality" as one number: it is the
paraxial-versus-real *pupil map*, of which paraxiality is one term, and 87 % of
the tangent figure is the difference between those two maps rather than anything
a coupler did. Nothing here measures which map M3-SINGLET-REF actually has, so
neither figure transfers to the traced system on its own.

What does carry over is the shape of CHE-117's conclusion: this metric is a
statement about the direction-cosine scale, and that scale is what the system
leaves open -- so O1 cannot decide a 1e-3 gate here.

Conventions
-----------
Every function takes SI metres and returns SI metres or a dimensionless ratio.
Grids are `(ny, nx)` in `(y, x)` order on the `n // 2` origin -- the same rule
`representations.Frame` declares and `measurements.psf` reports -- because an
oracle built on a different centring differs from the truth by half a pixel,
which at ordinary PSF sampling is a large fraction of an Airy radius.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.special import j1

__all__ = [
    "AIRY_FIRST_NULL_V",
    "airy_first_null_radius_m",
    "airy_intensity_at_radius",
    "airy_psf_on_grid",
    "disc_mask",
    "measure_first_null_radius_m",
    "numerical_aperture_from_geometry",
    "peak_normalized_disc_relative_l2",
    "pixels_per_airy_radius",
    "radial_profile",
    "relative_l2_intensity",
]

#: The first zero of `J1`, in the reduced radial variable `v = 2 pi NA r / lambda`.
#:
#: Written to sixteen digits rather than computed, and checked against
#: `scipy.optimize.brentq` in the oracle self-tests: a constant a reader can
#: compare with a handbook is worth more here than a root-finder call, and this
#: number sets the length scale every comparison below is measured in.
AIRY_FIRST_NULL_V = 3.8317059702075123


def numerical_aperture_from_geometry(*, semi_aperture_m: float, distance_m: float) -> float:
    """`a / sqrt(a^2 + R^2)` -- the sine of the marginal ray angle.

    The **geometric** declaration, and it is one of at least two defensible ones
    for a real lens. The other is the largest transverse direction cosine the
    trace actually produced. For M3-SINGLET-REF those two differ by 2.9e-3, which
    CHE-117 measured as 4.4x the gate they are used to decide -- so which one is
    passed here is a declaration a caller makes and not a detail.
    """
    return float(semi_aperture_m) / math.hypot(float(semi_aperture_m), float(distance_m))


def airy_first_null_radius_m(*, numerical_aperture: float, wavelength_m: float) -> float:
    """`3.8317 lambda / (2 pi NA)`, the classical `0.61 lambda / NA`."""
    return AIRY_FIRST_NULL_V * float(wavelength_m) / (2.0 * math.pi * float(numerical_aperture))


def airy_intensity_at_radius(
    radius_m: Any, *, numerical_aperture: float, wavelength_m: float
) -> Any:
    """`(2 J1(v) / v)^2`, peak 1 at `r = 0`.

    The limit at `v = 0` is written in rather than reached: `2 J1(v) / v -> 1`,
    and evaluating it numerically is `0 / 0`.
    """
    v = 2.0 * math.pi * float(numerical_aperture) * np.abs(np.asarray(radius_m, dtype=np.float64))
    v = v / float(wavelength_m)
    out = np.ones_like(v)
    lit = v > 0.0
    out[lit] = (2.0 * j1(v[lit]) / v[lit]) ** 2
    return out


def _coordinates(
    shape: tuple[int, int], sample_pitch_m: tuple[float, float]
) -> tuple[Any, Any]:
    """`(y, x)` grids on the `n // 2` origin -- `representations.Frame`'s rule."""
    y = (np.arange(shape[0], dtype=np.float64) - shape[0] // 2) * float(sample_pitch_m[0])
    x = (np.arange(shape[1], dtype=np.float64) - shape[1] // 2) * float(sample_pitch_m[1])
    return np.meshgrid(y, x, indexing="ij")


def airy_psf_on_grid(
    *,
    shape: tuple[int, int],
    sample_pitch_m: tuple[float, float],
    numerical_aperture: float,
    wavelength_m: float,
) -> Any:
    """O1, sampled. Peak-normalized by construction, `1.0` at the origin sample."""
    grid_y, grid_x = _coordinates(shape, sample_pitch_m)
    return airy_intensity_at_radius(
        np.hypot(grid_y, grid_x),
        numerical_aperture=numerical_aperture,
        wavelength_m=wavelength_m,
    )


def disc_mask(
    *, shape: tuple[int, int], sample_pitch_m: tuple[float, float], radius_m: float
) -> Any:
    """The gate region: samples within `radius_m` of the origin.

    A disc rather than the whole window, because the corners of a square window
    are many Airy radii out, where both patterns are at the 1e-5 level and the
    residual there is dominated by whatever the window truncation did.
    """
    grid_y, grid_x = _coordinates(shape, sample_pitch_m)
    return np.hypot(grid_y, grid_x) <= float(radius_m)


def relative_l2_intensity(measured: Any, reference: Any, *, mask: Any = None) -> float:
    """`||m - o|| / ||o||` over `mask`. **Un-normalized**: it sees a global scale.

    The comparison to use when the absolute scale is part of the claim. Nothing
    here divides by a peak, so a reconstruction that is right in shape and wrong
    by a constant factor fails this and passes
    `peak_normalized_disc_relative_l2`.
    """
    m = np.asarray(measured, dtype=np.float64)
    o = np.asarray(reference, dtype=np.float64)
    if mask is not None:
        m, o = m[mask], o[mask]
    return float(np.linalg.norm(m - o) / np.linalg.norm(o))


def peak_normalized_disc_relative_l2(
    measured: Any,
    reference: Any,
    *,
    sample_pitch_m: tuple[float, float],
    radius_m: float,
) -> float:
    """The frozen L2-PSF-01 gate metric: peak-normalize both, then relative L2 on a disc.

    **It cannot see a global scale error**, by construction: both inputs are
    divided by their own maxima first. CHE-117 measured that directly -- rescaling
    the measured intensity by `2^64` moves this number by 1e-14 relative, float64
    round-off -- which is why an absolute check has to be made separately and why
    R11.2's criterion 2 requires one.

    What it is good at is shape, and it is close to linear in a fractional error
    in the Airy scale: slope 1.517-1.532 over four decades, measured by CHE-117.
    So the frozen 1.0e-3 threshold is, read literally, the statement that the
    system's Airy scale is known to 6.5e-4.
    """
    measured_array = np.asarray(measured, dtype=np.float64)
    reference_array = np.asarray(reference, dtype=np.float64)
    mask = disc_mask(
        shape=measured_array.shape, sample_pitch_m=sample_pitch_m, radius_m=radius_m
    )
    return relative_l2_intensity(
        measured_array / measured_array.max(),
        reference_array / reference_array.max(),
        mask=mask,
    )


def radial_profile(
    intensity: Any, *, sample_pitch_m: tuple[float, float], bin_width_m: float
) -> tuple[Any, Any]:
    """`(radius, mean intensity)` in annular bins about the origin.

    Empty bins are dropped rather than returned as NaN: a profile is a sequence of
    measurements and a bin nothing landed in is not one.
    """
    array = np.asarray(intensity, dtype=np.float64)
    grid_y, grid_x = _coordinates(array.shape, sample_pitch_m)
    radius = np.hypot(grid_y, grid_x).ravel()
    index = np.floor(radius / float(bin_width_m)).astype(np.int64)
    counts = np.bincount(index)
    totals = np.bincount(index, weights=array.ravel())
    lit = counts > 0
    centres = (np.nonzero(lit)[0] + 0.5) * float(bin_width_m)
    return centres, totals[lit] / counts[lit]


def measure_first_null_radius_m(intensity: Any, *, sample_pitch_m: tuple[float, float]) -> float:
    """The first local minimum along `+x` from the centre row, sub-pixel.

    Parabolic interpolation through the three samples around the minimum, because
    the whole point of this measurement is to compare against a continuous
    analytic radius and reporting it to the nearest sample would make the
    comparison a statement about the pitch.

    That is *not* enough to make it trustworthy at coarse sampling, and R11.2's
    criterion 4 is about exactly that -- see
    `test_psf_verification.py::test_a_first_null_discrepancy_is_sampling_before_it_is_physics`,
    which measures a **+166 %** error at 2.4 samples per Airy radius on a pattern
    whose shape residual is unaffected. A first-null discrepancy is a statement
    about the grid until the grid is shown not to matter.
    """
    array = np.asarray(intensity, dtype=np.float64)
    row = array[array.shape[0] // 2, array.shape[1] // 2 :]
    for index in range(1, row.size - 1):
        if row[index] < row[index - 1] and row[index] <= row[index + 1]:
            left, centre, right = row[index - 1], row[index], row[index + 1]
            curvature = left - 2.0 * centre + right
            offset = 0.5 * (left - right) / curvature if curvature != 0.0 else 0.0
            return float((index + offset) * sample_pitch_m[1])
    raise ValueError(
        "no local minimum along +x: the window does not reach the first null, so "
        "there is nothing here to measure"
    )


def pixels_per_airy_radius(
    *, sample_pitch_m: tuple[float, float], airy_radius_m: float
) -> float:
    """The sampling figure every grid-convergence claim below is indexed by."""
    return float(airy_radius_m) / math.sqrt(
        float(sample_pitch_m[0]) * float(sample_pitch_m[1])
    )
