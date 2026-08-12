"""Field estimators for L1-WAVE-01. No Chromatix, no JAX, no analytic oracle.

Everything here measures a *sampled field*, and is applied identically to the
solver output and to the analytic field on the same grid. That symmetry is
what lets the evaluator attribute error: whatever an estimator does to the
analytic field is estimator error, not solver error.

Two deliberate omissions:

- There is no ``D4-sigma`` second-moment width. The focal pattern of a hard
  rectangular aperture is a ``sinc^2``, whose second moment *diverges*
  (the tails fall off as ``1/t^2``), so a second-moment width would measure
  the window rather than the beam. Widths here are FWHM by sub-pixel linear
  interpolation of the half-maximum crossings.
- Intensity is never computed from an amplitude implicitly. Callers pass
  ``abs(u) ** 2`` explicitly, so an amplitude/intensity confusion is a visible
  error at the call site rather than a silent factor of two.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Complex-field agreement
# ---------------------------------------------------------------------------
def complex_field_agreement(measured: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    """Compare two complex fields up to one global phase.

    A global piston is not physically meaningful on a single plane, so it is
    removed before the RMS comparison. What survives is the *relative* phase
    structure -- wavefront curvature, Gouy shift, the vectorial phase between
    components -- which is exactly what a convention error corrupts.
    ``overlap`` reaches 1 only if amplitude and relative phase agree
    everywhere.
    """
    a = np.asarray(measured, dtype=np.complex128).ravel()
    b = np.asarray(reference, dtype=np.complex128).ravel()
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        raise ValueError("cannot compare a zero field")
    a = a / norm_a
    b = b / norm_b
    inner = np.vdot(b, a)
    aligned = b * np.exp(1j * np.angle(inner))
    return {
        "overlap": float(np.abs(inner)),
        "normalized_rms_error": float(np.linalg.norm(a - aligned)),
        "amplitude_normalized_rms_error": float(np.linalg.norm(np.abs(a) - np.abs(b))),
    }


def wrapped_phase_difference(measured: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Phase of ``measured`` minus ``reference``, wrapped into ``(-pi, pi]``."""
    difference = np.angle(np.asarray(measured)) - np.angle(np.asarray(reference))
    return (difference + np.pi) % (2.0 * np.pi) - np.pi


def amplitude_ratio_error(measured: np.ndarray, reference: np.ndarray) -> float:
    """``max | |measured| / |reference| - 1 |`` over samples with nonzero reference."""
    a = np.abs(np.asarray(measured))
    b = np.abs(np.asarray(reference))
    nonzero = b > 0.0
    if not np.any(nonzero):
        raise ValueError("reference amplitude is identically zero")
    return float(np.max(np.abs(a[nonzero] / b[nonzero] - 1.0)))


# ---------------------------------------------------------------------------
# Position and width
# ---------------------------------------------------------------------------
def grid_coordinates_m(n: int, pitch_m: float) -> np.ndarray:
    return (np.arange(n, dtype=np.float64) - n // 2) * pitch_m


def intensity_centroid_m(intensity: np.ndarray, pitch_m: float) -> dict[str, float]:
    """Intensity-weighted centroid in metres, on the ``n // 2``-centred grid.

    Unlike the second moment, the *first* moment of a ``sinc^2`` converges by
    symmetry, so this is a valid position estimator for Case 2.
    """
    intensity = np.asarray(intensity, dtype=np.float64)
    n = intensity.shape[0]
    coordinates = grid_coordinates_m(n, pitch_m)
    total = float(intensity.sum())
    if total <= 0.0:
        raise ValueError("intensity map has non-positive total power")
    return {
        "centroid_x_m": float((intensity * coordinates[None, :]).sum() / total),
        "centroid_y_m": float((intensity * coordinates[:, None]).sum() / total),
    }


def fwhm_m(profile: np.ndarray, pitch_m: float) -> float:
    """Full width at half maximum by sub-pixel linear interpolation.

    Walks outward from the peak to the first sample below half maximum on each
    side and interpolates the crossing, so the result is not quantized to the
    sample pitch. Returns ``nan`` if the profile does not fall to half maximum
    inside the window -- a truncated lobe has no measurable FWHM and must not
    be reported as if it did.
    """
    values = np.asarray(profile, dtype=np.float64)
    peak_value = values.max()
    if peak_value <= 0.0:
        return float("nan")
    normalized = values / peak_value
    peak_index = int(np.argmax(normalized))

    crossings = []
    for step in (-1, 1):
        index = peak_index
        while 0 < index < len(normalized) - 1 and normalized[index] > 0.5:
            index += step
        if normalized[index] > 0.5:
            return float("nan")
        previous = index - step
        span = normalized[index] - normalized[previous]
        fraction = 0.0 if span == 0 else (0.5 - normalized[previous]) / span
        crossings.append(abs(previous + fraction * step - peak_index))
    return float(sum(crossings) * pitch_m)


def first_null_radius_m(profile: np.ndarray, pitch_m: float) -> float:
    """Distance from the peak to the first local minimum, to sample resolution.

    Reported as a diagnostic rather than gated: a zero of a ``sinc`` is a
    sharp cusp in intensity, so its position is far less robustly estimated
    from samples than the FWHM.
    """
    values = np.asarray(profile, dtype=np.float64)
    peak_index = int(np.argmax(values))
    index = peak_index
    while index + 1 < len(values) and values[index + 1] < values[index]:
        index += 1
    return float((index - peak_index) * pitch_m)


def first_sidelobe_ratio(profile: np.ndarray) -> float:
    """Peak of the first sidelobe divided by the main peak, sub-pixel refined.

    A scale-free shape number: for a ``sinc^2`` it is 0.047180 regardless of
    wavelength, aperture, or focal length, so it tests the pattern's shape
    independently of any coordinate calibration.
    """
    values = np.asarray(profile, dtype=np.float64)
    peak_value = values.max()
    if peak_value <= 0.0:
        return float("nan")
    normalized = values / peak_value
    peak_index = int(np.argmax(normalized))

    index = peak_index
    while index + 1 < len(normalized) and normalized[index + 1] < normalized[index]:
        index += 1  # descend into the first null
    while index + 1 < len(normalized) and normalized[index + 1] > normalized[index]:
        index += 1  # climb the first sidelobe
    if index <= 0 or index >= len(normalized) - 1:
        return float("nan")

    # Quadratic (three-point) refinement of the sidelobe peak height.
    left, centre, right = normalized[index - 1], normalized[index], normalized[index + 1]
    denominator = left - 2.0 * centre + right
    if denominator == 0.0:
        return float(centre)
    return float(centre - 0.125 * (right - left) ** 2 / denominator)


def radial_intensity_profile(
    intensity: np.ndarray, pitch_m: float, bin_count: int = 128
) -> dict[str, np.ndarray]:
    """Azimuthally averaged intensity vs radius, normalized to its peak."""
    intensity = np.asarray(intensity, dtype=np.float64)
    n = intensity.shape[0]
    coordinates = grid_coordinates_m(n, pitch_m)
    radius = np.sqrt(coordinates[None, :] ** 2 + coordinates[:, None] ** 2).ravel()
    values = intensity.ravel()

    edges = np.linspace(0.0, float(radius.max()), bin_count + 1)
    index = np.clip(np.digitize(radius, edges) - 1, 0, bin_count - 1)
    counts = np.bincount(index, minlength=bin_count)
    sums = np.bincount(index, weights=values, minlength=bin_count)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
    peak = np.nanmax(mean) if np.any(np.isfinite(mean)) else 1.0
    return {
        "radius_m": 0.5 * (edges[:-1] + edges[1:]),
        "normalized_intensity": mean / peak if peak > 0 else mean,
        "sample_count": counts.astype(np.float64),
    }


# ---------------------------------------------------------------------------
# Window and power
# ---------------------------------------------------------------------------
def edge_energy_fraction(field: np.ndarray) -> float:
    """Fraction of ``|u|^2`` on the 1-pixel border of the sampled window."""
    intensity = np.abs(np.asarray(field)) ** 2
    if intensity.ndim == 3:
        intensity = intensity.sum(axis=-1)
    total = float(intensity.sum())
    if total <= 0.0 or min(intensity.shape[:2]) < 3:
        return 0.0
    border = float(
        intensity[0, :].sum()
        + intensity[-1, :].sum()
        + intensity[1:-1, 0].sum()
        + intensity[1:-1, -1].sum()
    )
    return border / total


# ---------------------------------------------------------------------------
# Vector-field structure (Case 3)
# ---------------------------------------------------------------------------
def vector_component_ratios(field_xyz: np.ndarray) -> dict[str, float]:
    """Peak-intensity ratios between the three Cartesian components.

    Dimensionless and independent of any overall amplitude calibration, so
    these survive even when a field's absolute scale is untrustworthy. For an
    x-polarized aplanatic focus they are a sharp NA signature.
    """
    intensity = np.abs(np.asarray(field_xyz)) ** 2
    peak_x = float(intensity[..., 0].max())
    if peak_x <= 0.0:
        raise ValueError("E_x is identically zero; cannot form component ratios")
    return {
        "peak_intensity_x": peak_x,
        "iy_over_ix": float(intensity[..., 1].max()) / peak_x,
        "iz_over_ix": float(intensity[..., 2].max()) / peak_x,
    }


def vector_symmetry_diagnostics(field_xyz: np.ndarray) -> dict[str, float]:
    """Symmetry signatures of an x-polarized aplanatic focus.

    ``E_z`` is proportional to ``cos(phi)``, so it vanishes on axis and is
    antisymmetric under ``x -> -x``; ``E_y`` is proportional to ``sin(2 phi)``
    and so has four lobes and is antisymmetric under either reflection alone.
    These hold for the *shape* of the field and are independent of scale,
    which makes them usable even when the coordinate calibration is in doubt.
    """
    field = np.asarray(field_xyz)
    n = field.shape[0]
    centre = n // 2
    intensity_total = float((np.abs(field) ** 2).sum())
    if intensity_total <= 0.0:
        raise ValueError("vector field is identically zero")

    field_z = field[..., 2]
    field_y = field[..., 1]
    peak_z = float(np.abs(field_z).max())
    # Flip about the origin sample so the comparison is exactly grid-aligned.
    flipped_x = field_z[:, ::-1]
    antisymmetry = float(np.abs(field_z + flipped_x).max()) / (peak_z if peak_z > 0 else 1.0)
    return {
        "ez_on_axis_over_peak": (
            float(np.abs(field_z[centre, centre])) / peak_z if peak_z > 0 else 0.0
        ),
        "ez_x_antisymmetry_residual": antisymmetry,
        "ey_on_axis_over_peak": (
            float(np.abs(field_y[centre, centre])) / float(np.abs(field_y).max())
            if float(np.abs(field_y).max()) > 0
            else 0.0
        ),
    }
