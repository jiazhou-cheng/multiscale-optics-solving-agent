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

**Where the field is accumulated, and where it lands.** CHE-246 (T2) gave the
three sources a `namespace`/`device` target. It did **not** move the arithmetic
into that target, and the reason is the validity line all three records carry:
"the phase ramp is accumulated in float64 before the cast". `jax_enable_x64` is
disabled in every process this project runs -- `backends/chromatix/fields.py`
pins it off on import, and it is process-global and cannot be set after the first
array exists -- so JAX cannot represent float64 at all. A source that accumulated
its ramp in the target namespace would therefore accumulate in float32 whenever
the target was JAX, costing ~6e-5 rad on a 100 um grid where the cast itself
costs ~6e-8, while still returning a field whose descriptor claims float64
accumulation. That is a false declaration, not a performance detail.

So the real quantity -- `phase` for the two ramp sources, `radius` for
`spherical_wave` -- is accumulated in host float64, cast once to `complex64`, and
only then placed. `require_phase_accumulation` makes the claim executable at the
site it is a claim about, and `deliver` performs the single move. The consequence
worth stating is a property rather than a compromise: **every cell's array is
bit-identical to the host cell's**, because they are the same arithmetic and the
same cast with only the buffer moved. A float32 accumulation would break that by
three orders of magnitude, which is what makes the validity line testable from
outside the package.

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

from numerics import ArrayNamespace, DevicePlacement, DType, to_namespace, verify_dtype
from representations import ContractError, Frame
from representations.contracts import require_positive_si

__all__ = [
    "ACCUMULATION_DTYPE",
    "SOURCE_DTYPE",
    "SOURCE_DTYPE_DECLARED",
    "deliver",
    "grid_coordinates",
    "nyquist_limit_rad_per_m",
    "require_grid_shape",
    "require_phase_accumulation",
    "require_sample_pitch",
    "require_transverse_wavevector",
]

#: The one storage dtype of this project's wave path. `complex128` is not a
#: choice the wave path has: `numerics.negotiate` refuses it against the measured
#: chromatix row with `LOSSY_DOWNCAST_REQUIRED`, so a float64 field could not be
#: propagated at all. Phase is accumulated in float64 and cast once.
SOURCE_DTYPE = np.complex64

#: The same dtype in this project's own vocabulary, for the one place a `DType` is
#: what is needed (`deliver`'s explicit request to `to_namespace`). Two spellings
#: of one dtype, so they are pinned to each other rather than left to agree.
SOURCE_DTYPE_DECLARED: DType = DType.COMPLEX64

#: The dtype the real quantity under the exponent is accumulated in, which is the
#: dtype all three records' validity lines name. See the module docstring for why
#: it is not the target namespace's.
ACCUMULATION_DTYPE: DType = DType.FLOAT64

# The two spellings above are pinned to each other, and the storage and
# accumulation dtypes are pinned apart, by `tests/parity/test_sources_parity.py`
# (`test_every_source_declares_the_same_storage_dtype_in_every_cell`) rather than
# by a module-level `assert` here. Nothing else in `src/` asserts at
# import time, and an `assert` is the one refusal in Python that disappears under
# `-O`.


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


def require_phase_accumulation(value: Any, *, source: str) -> Any:
    """The validity line "accumulated in float64", made executable where it is made.

    `numerics.arrays.verify_dtype` and not a new mechanism: that function exists
    for precisely this failure -- "JAX with `jax_enable_x64` disabled accepts
    `astype(float64)` and returns `float32`. No warning, no error" -- and a second
    check with its own message would be a second opinion about the same thing.

    **This is the only executable thing standing behind the float64 declaration,
    and it catches the case the parity gate cannot.** Two regressions would make
    all three records' validity lines false:

    * moving this arithmetic into the target namespace, to avoid the copy
      `deliver` makes. `tests/parity/test_sources_parity.py` sees that one,
      because it makes the cells stop being bit-identical;
    * a *uniform* host float32 accumulation -- a `float32` coordinate axis, a
      `np.float32` wavevector, a `grid_coordinates` that stopped returning
      float64. Every cell degrades identically, so the parity gate stays green
      and the declaration is silently false. Only this check sees it, which is
      why it is at the accumulation site rather than at the boundary.

    The second is the likelier of the two and is the reason this is not a
    formality. `tests/sources/test_plane_wave.py` covers the refusal.
    """
    return verify_dtype(value, ACCUMULATION_DTYPE, context=f"sources.{source}")


def deliver(u: Any, *, namespace: ArrayNamespace, device: DevicePlacement | None) -> Any:
    """The cast field array, moved once into the requested namespace and device.

    The only move any source makes, and it is `numerics.arrays.to_namespace` --
    the production mover -- rather than a local `jnp.asarray`, so a source cannot
    place a buffer by a route nothing else in the tree uses.

    `dtype` is requested explicitly **and the result is verified**, which is two
    steps and not one. Requesting it alone is not enough: `to_namespace`'s NumPy
    branch `astype`s an over-wide input down with no check at all, so a source
    that forgot its own `.astype(SOURCE_DTYPE)` would be silently narrowed here
    rather than refused. `verify_dtype` on the way out is what makes the refusal
    reachable, and it is the same argument as `require_phase_accumulation`'s
    applied to the storage dtype instead of the accumulation dtype.

    A NumPy target with `device=None` is the identity: `_to_numpy` of a NumPy
    array is `np.asarray`, and the dtype already matches, so the default path
    produces the same bytes it did before this argument existed. That is why
    CHE-246's "existing NumPy results are bit-identical" is a property of the code
    rather than a claim about it.

    `device=None` means "wherever this namespace puts a new array", which is the
    host for NumPy and the default backend for JAX. Note the landed limitation
    `tests/conftest.py` records: on the GPU image `to_namespace` cannot reach
    JAX's host device, because `_jax_device` selects from `jax.devices()` -- the
    default backend only -- so `namespace=JAX, device=None` there is a refusal
    rather than a host field. Naming `device` explicitly avoids the question.
    """
    return verify_dtype(
        to_namespace(u, namespace=namespace, device=device, dtype=SOURCE_DTYPE_DECLARED),
        SOURCE_DTYPE_DECLARED,
        context="sources.deliver",
    )


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
