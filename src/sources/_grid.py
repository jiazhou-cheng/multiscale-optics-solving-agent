"""The sampled-grid declaration the wave sources share, and its two refusals.

Private to `sources/`. Nothing here is a new convention: every line is the
arithmetic `plane_wave` landed on CHE-210 (R06.5), lifted out unchanged when
CHE-215 (R06.10) added a second and third source on the same grid.

**Why this module exists rather than three copies.** The two refusals below --
`|k_t| <= n k0` and `|k_t| <= pi/d` per axis -- are the ones that catch a *
plausible* wrong answer: an evanescent field carried as a propagating one, and an
aliased tilt that reads back as a different but entirely believable angle. Two
sources with independently written copies of those bounds diverge the first time
one of them is edited, and the symptom is that one source refuses a geometry the
other accepts -- with no test anywhere comparing the two. CHE-215 names that as
its main risk. So the bound is written once and both wave sources call it.

The same argument applies to the coordinate origin. `grid_coordinates` is the one
place `Frame.origin_index` is consulted, because a half-sample origin shift *is* a
linear phase ramp across the grid -- a tilt -- and a source whose origin drifted
from the rest of the tree by half a sample would emit a field that is wrong in
exactly the quantity these sources exist to state.

Not shared, deliberately: `spherical_wave`'s under-sampling refusal. It is the
same *bound* -- `nyquist_limit_rad_per_m` -- against a local spatial frequency
that varies across the grid, so it calls that function and writes its own
message, because the geometry a caller has to fix (the source position, not a
declared tilt) is different and the message has to name it.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from representations import ContractError, Frame
from representations.contracts import require_positive_si

__all__ = [
    "SOURCE_DTYPE",
    "grid_coordinates",
    "nyquist_limit_rad_per_m",
    "require_grid_shape",
    "require_sample_pitch",
    "require_transverse_wavevector",
]

#: The one storage dtype of this project's wave path. `complex128` is not a
#: choice the wave path has: `numerics.negotiate` refuses it against the measured
#: chromatix row with `LOSSY_DOWNCAST_REQUIRED`, so a float64 field could not be
#: propagated at all. Phase is accumulated in float64 and cast once.
SOURCE_DTYPE = np.complex64


def require_grid_shape(shape: tuple[int, int]) -> tuple[int, int]:
    """`(ny, nx)`, both at least one sample."""
    counts = tuple(int(value) for value in shape)
    if len(counts) != 2 or any(count < 1 for count in counts):
        raise ValueError(f"shape must be (ny, nx) with at least one sample per axis, got {shape!r}")
    return (counts[0], counts[1])


def require_sample_pitch(sample_pitch_m: tuple[float, float]) -> tuple[float, float]:
    """`(dy, dx)` in metres, through the helpers `ScalarField` itself applies.

    Called *before* the refusals rather than left to `ScalarField.__post_init__`,
    because those refusals divide by the pitch: a field built from a bad
    declaration would be refused either way, and doing it here means the message
    names the declaration instead of the NaNs it produced.
    """
    pitch = tuple(
        require_positive_si(value, name=name)
        for value, name in zip(
            sample_pitch_m, ("sample_pitch_m[dy]", "sample_pitch_m[dx]"), strict=True
        )
    )
    return (pitch[0], pitch[1])


def grid_coordinates(
    shape: tuple[int, int], pitch: tuple[float, float], frame: Frame
) -> tuple[np.ndarray[Any, np.dtype[np.float64]], np.ndarray[Any, np.dtype[np.float64]]]:
    """`(y, x)` 1-D float64 coordinate axes in metres, on the frame's own origin.

    `Frame.origin_index` rather than a rewritten `n // 2`: see the module
    docstring on why half a sample matters here specifically.
    """
    ny, nx = shape
    dy, dx = pitch
    y = (np.arange(ny, dtype=np.float64) - frame.origin_index(ny)) * dy
    x = (np.arange(nx, dtype=np.float64) - frame.origin_index(nx)) * dx
    return y, x


def nyquist_limit_rad_per_m(pitch_m: float) -> float:
    """The largest angular spatial frequency a pitch of `pitch_m` represents.

    `pi / d`. Public within the package because `spherical_wave` compares a
    *local* spatial frequency against the same bound, and a second copy of a
    factor of two is how the two sources stop agreeing.
    """
    return math.pi / pitch_m


def require_transverse_wavevector(
    transverse_wavevector_rad_per_m: tuple[float, float],
    *,
    pitch: tuple[float, float],
    wavelength_m: float,
    medium_index: float,
) -> tuple[float, float]:
    """`(k_y, k_x)` in rad/m, refused if it is not a representable illumination.

    Two refusals, in this order, and which one binds is a property of the pitch:
    `pi/d` exceeds `n k0` whenever `d < lambda / (2 n)`, so on a fine grid every
    representable tilt is inside the light cone and only the first can fire.

    Raises:
        ContractError: `REPRESENTATION_INCONSISTENT` for `|k_t| > n k0`
            (evanescent, not an illumination angle) or for `|k_t|` past `pi/d` on
            either axis (an aliased tilt reads back as a different, entirely
            plausible angle).
        ValueError: not a finite `(k_y, k_x)` pair.
    """
    wavevector = tuple(float(value) for value in transverse_wavevector_rad_per_m)
    if len(wavevector) != 2 or not all(math.isfinite(value) for value in wavevector):
        raise ValueError(
            "transverse_wavevector_rad_per_m must be a finite (k_y, k_x) pair in rad/m, got "
            f"{transverse_wavevector_rad_per_m!r}"
        )

    medium_wavenumber = 2.0 * math.pi * medium_index / wavelength_m
    magnitude = math.hypot(*wavevector)
    if magnitude > medium_wavenumber:
        raise ContractError(
            "REPRESENTATION_INCONSISTENT",
            f"|k_t| = {magnitude:.6g} rad/m exceeds n k0 = {medium_wavenumber:.6g} rad/m "
            f"(n = {medium_index}, lambda = {wavelength_m} m). That is an evanescent wave, "
            "not an illumination angle: the field this would build decays along +z and "
            "would be carried as a propagating one.",
            declaration="transverse_wavevector_rad_per_m",
            remedy="Reduce |k_t|, or state the medium index the angle was measured in.",
        )

    for value, step, axis in zip(wavevector, pitch, ("k_y", "k_x"), strict=True):
        nyquist = nyquist_limit_rad_per_m(step)
        if abs(value) > nyquist:
            raise ContractError(
                "REPRESENTATION_INCONSISTENT",
                f"|{axis}| = {abs(value):.6g} rad/m is past this grid's Nyquist limit "
                f"pi/d = {nyquist:.6g} rad/m at a pitch of {step} m. The sampled ramp would "
                "alias, and an aliased tilt reads back as a completely different and "
                "entirely plausible angle -- which is the failure this refusal exists for.",
                declaration="transverse_wavevector_rad_per_m",
                remedy="Refine the pitch, or reduce the tilt.",
            )

    return (wavevector[0], wavevector[1])
