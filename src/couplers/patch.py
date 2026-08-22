"""Patch-based local windowed Fourier transform — the general ray-DOE method.

The point most likely to be missed, so it goes first: **the batched planar step
in `couplers/cascade.py` is a special case of this, not a peer of it.** SI S10
says so about the global aggregation Algorithm S1 performs — *"For conformal
DOEs, this global aggregation before ray-DOE interaction is not applicable
because rays intersect different local tangent planes with position-dependent
coordinate frames and surface normals. We therefore retain the direct
implementation."* The direct implementation is this one. Until now this
repository had the shortcut and not the method.

What it does: each incident ray extracts its own local complex patch of the DOE,
Fourier transforms it (SI eq S1), and emits secondary rays sampled from that
patch's angular spectrum (eqs S3-S5). The field at a downstream plane is the
coherent sum over all secondary rays of all patches.

Why it matters for **planar** DOEs too
--------------------------------------
Not only for curved substrates. The paper's own guidance, main text, Fig 3
discussion: *"For large planar DOEs where the processing of the full-field
planar patch exceeds available GPU memory, we can instead decompose the full
field into multiple smaller overlapping patches, each associated with an
incident ray."* Patch size is a memory-budget dial. SI Table S2 makes that
concrete rather than rhetorical: on the 4032x4032 Grating-Lens DOE the
full-field route is recorded as **OOM on a 48 GB RTX A6000** while the patch
route completes the same system in 4.982 s at 11.492 GB using 40x40 patches.

What is gated, and what is only bounded
---------------------------------------
SI S2 (page S6) makes a claim that is directly testable, and it is the reason
this module can be gated rather than merely characterized:

    "For planar DOEs, there is no intrinsic upper bound on the patch size
    because the tangent-plane approximation is exact everywhere on the surface.
    As long as the ensemble of patches uniformly covers the DOE profile, the
    coherent sum of their responses converges to the full DOE response through
    coherent superposition, consistent with the linearity of the Fourier
    transform."

Two consequences, both asserted in `tests/test_patch_wft.py`:

1. one patch as large as the whole aperture reproduces the full-field route
   exactly, and
2. many smaller patches uniformly covering the aperture converge to the same
   field.

**On a curved substrate neither holds.** Every patch has its own tangent frame
and normal, the exactness relation is gone, and all that remains is the SI S3
bound `eps_curv <= arcsin(D/2R)` in `couplers/curvature.py`. That asymmetry is
the most important scoping fact here: the planar case gets a hard gate, the
conformal case would get a bound and a characterization, and the second must not
inherit the first's confidence. `substrate` is an explicit declaration and
refuses anything but planar.

Three things that must not be re-derived the hard way
-----------------------------------------------------
Each was found by getting it wrong first.

**Under enumeration the density must be uniform (`p = 1/n_modes`), not 1.**
Using `p = 1` divides the exact sum by the mode count and silently breaks the
exactness relation -- it still produces an entirely plausible field.

**`pad_factor` is the wrong parameter, and this is the finding that cost the
most to reach.** A patch's discrete plane-wave sum is *periodic* with period
`pad_px * pitch`. If a periodic replica of any patch lands inside the
reconstruction window, the sum reproduces the replica along with the field, and
the error saturates at O(1) rather than converging -- which is what a
patch-count sweep looks like when it plateaus instead of falling.

Two conditions, both derivable and both measured:

* **Clearance.** A patch centred at `c` has replicas spanning
  `c ± patch/2 + m * pad`. For none to enter a window of half-width `N/2`, the
  requirement is `pad > max|c| + (N + patch)/2`, strictly -- at equality the
  replica's edge sample lands on the window's edge sample, and a patch is rarely
  zero at its own edge. With centres drawn over the aperture dilated by half a
  patch, `max|c| = (N + patch)/2` and this becomes the memorable
  `pad >= N + patch`. Measured on a 3x3 tiling of an 11-px patch over a 33-px
  grid: pad 32 gives 1.44, pad 33 gives **5.6e-15**, pad 23 gives 1.29.

  **The full-aperture single patch is exempt, and it is the only exemption.**
  Clearance is a statement about a *sub*-aperture -- it protects the part of the
  window the patch does not itself cover. When the patch is the window there is
  no such part, and the periodicity a pad would suppress is the same periodicity
  the unpadded reference ASM has. Padding it is not the safe choice: it moves
  the mode grid off the oracle's and the exactness anchor reads 0.57 instead of
  1.4e-12.
* **Centring.** `pad_px - patch_px` must be **even**, so the patch sits on the
  padded array's centre sample. Odd puts it half a sample off, which injects a
  linear phase: pad 34 and pad 44 both satisfy clearance and both give ~1.4.

  Together with the odd-`pad_px` rule below this forces `patch_px` **odd**: an
  odd pad and an even `pad - patch` cannot both hold otherwise. That is not an
  arithmetic accident. A patch with an even side has no centre sample, so
  "centred on a ray" is undefined for it. An even `patch_px` is refused rather
  than rounded, because the paper's sizes (40, 50, 100) are all even and a
  caller transcribing one should be told which value actually ran.

`pad_px` is therefore **derived**, not taken. A caller states `pad_factor` as a
preference and `plan_patches` raises it to whatever satisfies both conditions,
recording the value it used.

The odd-grid rule is a third, independent constraint from the same family: an
even `pad_px` places a mode exactly at `lambda / (2 * pitch)`, which
`ray_to_wave`'s Nyquist guard refuses -- correctly, since a ray at exactly the
Nyquist direction is not representable on the grid it came from. Measured:
pad 66 is refused outright, pad 67 gives 3.0e-15.

**Patch centres are drawn over the aperture dilated by half a patch.** For patch
indicator `w` and centres `c ~ U(D)`, `E[integral_D w(x - c) dc] = A_patch` for
any `x` whose full footprint lies inside `D`. Draw centres only over the
aperture itself and the estimator is biased **at the rim only**, which reads as
a soft edge rather than as a bug.

Apodization
-----------
Windows (hann, hamming, blackman) are available upstream and are **not offered
here**. Any taper below 1 removes field that no other patch replaces, so the
coherent sum stops converging to the full-DOE response: the partition-of-unity
argument that makes relation (2) above true is exactly what a taper breaks. If a
taper is ever wanted it must be declared as trading the exactness guarantee for
a smoother spectrum, not added as an option.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np

from core.boundary import (
    ContractCode,
    ContractError,
    RayBundle,
    ReferencePlane,
)
from couplers.curvature import check_patch

__all__ = [
    "CoverageBasis",
    "PatchDiagnostics",
    "PatchPlan",
    "Substrate",
    "advance_bundle_to_plane",
    "extract_patch",
    "patch_secondary_rays",
    "plan_patches",
    "resolve_pad_px",
]


class Substrate(StrEnum):
    """What the DOE sits on. Only one value executes.

    Declared rather than inferred, because the difference is the difference
    between a hard gate and a bound: on a planar substrate the tangent-plane
    approximation is exact everywhere and the patch sum provably converges to
    the full-field response; on a curved one it does not, and the only statement
    available is `eps_curv <= arcsin(D/2R)`.
    """

    PLANAR = "planar"
    #: Refused. Needs Newton sag intersection, per-hit tangent frames and
    #: position-dependent normals -- a substantial new geometry surface, and the
    #: exactness ladder does not extend to it.
    CONFORMAL = "conformal"


class CoverageBasis(StrEnum):
    """How caller-supplied patch centres were drawn — a required declaration.

    The coverage correction ``A_draw / A_patch`` is only unbiased for centres
    drawn uniformly over the dilated aperture. When :func:`plan_patches` draws
    them itself it knows this; when a caller supplies them -- the paper's actual
    configuration, where each *incident ray* defines a patch -- it does not, and
    the density is not recoverable from the positions alone.

    So it is declared. Guessing would produce a field that is wrong by a
    constant factor and looks entirely plausible, which is the failure mode this
    whole module keeps running into.
    """

    #: Centres are an i.i.d. uniform sample of the dilated aperture's sample
    #: grid. The correction applies as usual.
    UNIFORM_OVER_DILATED_APERTURE = "uniform_over_dilated_aperture"
    #: Centres are the complete set of sample positions of the dilated
    #: aperture, each used once. Deterministic, and the correction is the same
    #: ratio -- it is a count of positions either way.
    ENUMERATED_OVER_DILATED_APERTURE = "enumerated_over_dilated_aperture"
    #: The caller does not know. Refused: an unbiased estimator needs a known
    #: sampling density, and there is no safe default.
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class PatchPlan:
    """Where the patches are and how much of the aperture each one covers."""

    #: (P, 2) patch centres in metres, on the plane's origin rule.
    centers_xy_m: np.ndarray[Any, Any]
    #: Patch side length in samples.
    patch_px: int
    #: Padded transform size in samples. **Derived**, not taken from the
    #: caller: see :func:`resolve_pad_px`. Always odd, always leaves the patch
    #: centred, and always clears the reconstruction window of replicas.
    pad_px: int
    #: ``A_draw / A_patch``: the patch-centre Monte Carlo correction. Each
    #: patch's contribution is multiplied by it so the ensemble is unbiased.
    #:
    #: The direction matters and is easy to invert. For patch indicator ``w``
    #: and centres ``c ~ U(D)``, ``E[sum_i w(x - c_i)] = P * A_patch / A_D``, so
    #: the raw patch sum UNDERSTATES the full response by ``A_patch / A_D``
    #: (the ``1/P`` is already applied by ``reconstruction_normalization``).
    #: Multiplying by ``A_D / A_patch`` cancels it. Inverting this is invisible
    #: on the full-aperture anchor, where the ratio is exactly 1.
    coverage: float
    #: Half-width, in samples, by which the draw region was dilated beyond the
    #: aperture. Equal to ``patch_px // 2``.
    dilation_px: int
    #: ``arcsin(D / 2R)`` for the declared substrate. Exactly 0 for planar,
    #: recorded rather than left implicit.
    curvature_bound_rad: float


@dataclass(frozen=True)
class PatchDiagnostics:
    patch_count: int
    patch_px: int
    pad_px: int
    coverage: float
    secondary_per_patch: int
    outgoing_ray_count: int
    enumerated: bool
    propagating_modes: int
    evanescent_modes: int
    substrate: str
    curvature_bound_rad: float
    apodization: str = "none -- a taper breaks the partition-of-unity exactness argument"
    reconstruction_normalization: str = "one_over_n"

    def as_dict(self) -> dict[str, Any]:
        return {
            "patch_count": self.patch_count,
            "patch_px": self.patch_px,
            "pad_px": self.pad_px,
            "coverage": self.coverage,
            "secondary_per_patch": self.secondary_per_patch,
            "outgoing_ray_count": self.outgoing_ray_count,
            "enumerated": self.enumerated,
            "propagating_modes": self.propagating_modes,
            "evanescent_modes": self.evanescent_modes,
            "substrate": self.substrate,
            "curvature_bound_rad": self.curvature_bound_rad,
            "apodization": self.apodization,
            "reconstruction_normalization": self.reconstruction_normalization,
        }


def resolve_pad_px(
    *, grid_n: int, patch_px: int, pad_factor: int = 2, max_center_px: float = 0.0
) -> int:
    """The smallest padded transform size that reconstructs exactly.

    Three conditions, each measured rather than assumed.

    1. **Clearance**: no periodic replica of a patch may enter the
       reconstruction window. A patch centred at ``c`` has replicas at
       ``c +/- patch/2 + m * pad``, and the window has half-width ``grid_n/2``,
       so the requirement is ``pad > max|c| + patch/2 + grid_n/2``. With a
       single centred patch that is ``(grid_n + patch)/2``; with centres drawn
       over the aperture dilated by half a patch it is ``grid_n + patch``. The
       bound is strict: at equality the replica's edge sample lands on the
       window's edge sample, and a patch is rarely zero at its own edge. The
       full-aperture single patch is exempt -- see the branch below.
       Violating it saturates the error at O(1), which is what a patch-count
       sweep looks like when it plateaus instead of converging: measured on a
       3x3 tiling of an 11-px patch over a 33-px grid (extreme centre ``-11p``,
       so the requirement is ``pad >= 33``), pad 32 gives 1.44, pad 33 gives
       **5.6e-15**, pad 23 gives 1.29.
    2. **Centring**: ``(pad_px - patch_px)`` must be even, so the patch sits on
       the padded array's centre sample. Odd puts it half a sample off and
       injects a linear phase -- pad 34 and pad 44 both clear condition 1 and
       both give ~1.4.
    3. **Oddness**: an even ``pad_px`` places a mode exactly at
       ``lambda / (2 * pitch)``, which ``ray_to_wave``'s Nyquist guard refuses.
       Measured: pad 66 is refused outright, pad 67 gives 3.0e-15.

    ``pad_factor`` is a preference, not an instruction: ``patch_px * pad_factor``
    is a floor and this raises it until all three hold. Returning a value the
    caller did not ask for is right here -- silently using one that violates
    condition 1 produces a plausible field that is wrong by 100%.

    **A larger pad is not automatically safer, and this is the subtlety.**
    Padding beyond the minimum makes the reconstruction *less* periodic and
    therefore more physical -- but the independent oracle
    ``verification/asm_oracle.angular_spectrum_float64`` is an unpadded DFT and
    is itself periodic with the reconstruction grid's period. Comparing a
    weakly-periodic patch route against a strongly-periodic oracle measures the
    difference in wraparound, not an error in either: on the full-aperture
    anchor at z = 1.26 mm, pad 33 gives 1.4e-12 against that oracle and pad 67
    gives 0.57. So an exactness comparison must match the two periodicities, and
    a *physical* comparison must pad both.
    """
    if int(patch_px) % 2 == 0:
        # Conditions 2 and 3 are jointly unsatisfiable for an even patch: an odd
        # pad and an even (pad - patch) cannot both hold. That is not an
        # arithmetic accident -- a patch with an even side has no centre sample,
        # so "centred on a ray" is undefined for it, and the half-sample offset
        # it forces is the same error the other two conditions exist to prevent.
        #
        # Refused rather than silently rounded, because the paper's patch sizes
        # (40, 50, 100) are even and a caller transcribing one should be told
        # that the odd neighbour is what actually ran.
        raise ContractError(
            ContractCode.SHAPE_MISMATCH,
            f"patch_px={patch_px} is even, so the patch has no centre sample and "
            "cannot be centred in an odd padded grid",
            declaration="patch_px",
            remedy=f"Use {int(patch_px) + 1} (or {int(patch_px) - 1}).",
        )
    if int(patch_px) >= int(grid_n) and float(max_center_px) == 0.0:
        # The full-aperture single patch is exempt from clearance, and this is
        # the one exemption. Clearance is a statement about a *sub*-aperture:
        # it keeps a replica of the patch out of the part of the window the
        # patch does not itself cover. Here the patch is the window, nothing
        # lies outside it, and the periodicity the pad would suppress is the
        # same periodicity the unpadded reference ASM has. Padding this case
        # does not make it safer -- it moves the mode grid off the oracle's and
        # the exactness anchor reads 0.57 instead of 1.4e-12.
        floor = int(grid_n)
    else:
        floor = math.floor(max_center_px + (int(patch_px) + int(grid_n)) / 2.0) + 1
    candidate = max(int(patch_px) * int(pad_factor), floor)
    while candidate % 2 == 0 or (candidate - int(patch_px)) % 2 != 0:
        candidate += 1
    return candidate


def plan_patches(
    *,
    grid_shape: tuple[int, int],
    sample_pitch_m: tuple[float, float],
    patch_px: int,
    pad_factor: int = 2,
    patch_count: int | None = None,
    centers_xy_m: np.ndarray[Any, Any] | None = None,
    coverage_basis: CoverageBasis = CoverageBasis.UNKNOWN,
    substrate: Substrate = Substrate.PLANAR,
    radius_m: float = math.inf,
    error_threshold_rad: float = 1e-3,
    rng: np.random.Generator | None = None,
) -> PatchPlan:
    """Choose patch centres and sizes, with the curvature bound enforced first.

    Three placements, and which one is in use is decided by what the caller
    passes rather than by a mode flag:

    * neither ``patch_count`` nor ``centers_xy_m`` -- the single full-aperture
      patch: the exactness anchor, and the configuration relation (1) of SI S2
      is about;
    * ``patch_count`` with an ``rng`` -- centres drawn here, uniformly over the
      dilated aperture's sample grid;
    * ``centers_xy_m`` with a ``coverage_basis`` -- centres supplied by the
      caller, which is the paper's actual configuration: each *incident ray*
      defines a patch. The basis must be declared because the Monte Carlo
      correction is only unbiased for a known density, and the density cannot be
      read back off the positions.

    The curvature check is a **precondition**, not a footnote. On a planar
    substrate it records ``R = inf => bound 0`` explicitly rather than leaving it
    implicit, so a later reader can tell that the zero was established rather
    than assumed.
    """
    if substrate is not Substrate.PLANAR:
        raise ContractError(
            ContractCode.MISSING_DECLARATION,
            f"substrate={substrate.value!r} is not implemented. On a curved "
            "substrate every patch has its own tangent frame and normal, the "
            "exactness relation of SI S2 does not hold, and only the bound "
            "eps_curv <= arcsin(D/2R) remains",
            declaration="substrate",
            remedy=(
                "Use couplers/curvature.py to size a patch against an error "
                "threshold, and treat the result as a characterization rather "
                "than a gate."
            ),
        )
    ny, nx = int(grid_shape[0]), int(grid_shape[1])
    if patch_px <= 0 or patch_px > min(ny, nx):
        raise ContractError(
            ContractCode.SHAPE_MISMATCH,
            f"patch_px={patch_px} must be in 1..{min(ny, nx)}",
            declaration="patch_px",
        )
    if pad_factor < 1:
        raise ContractError(
            ContractCode.SHAPE_MISMATCH,
            f"pad_factor={pad_factor} must be at least 1",
            declaration="pad_factor",
        )

    pitch = float(sample_pitch_m[0])
    # An enforced precondition, not a footnote. `check_patch` raises when the
    # bound exceeds the threshold, and on a planar substrate it records
    # `R = inf => bound 0` explicitly rather than leaving the zero implicit --
    # so a later reader can tell it was established rather than assumed.
    budget = check_patch(
        patch_width_m=patch_px * pitch,
        radius_m=radius_m,
        error_threshold_rad=error_threshold_rad,
    )

    # The clearance floor depends on how far a centre can be from the origin,
    # which is zero for the single full-aperture patch and the dilated
    # half-width for drawn centres. Using the drawn-case floor for the single
    # patch would over-pad it and, against the unpadded oracle, look like a
    # 0.57 error rather than the 1.4e-12 it actually is.
    supplied = centers_xy_m is not None
    if supplied and patch_count is not None:
        raise ContractError(
            ContractCode.MISSING_DECLARATION,
            "centers_xy_m and patch_count both decide where the patches are",
            declaration="centers_xy_m",
            remedy="Supply exactly one.",
        )
    if supplied and coverage_basis is CoverageBasis.UNKNOWN:
        raise ContractError(
            ContractCode.MISSING_DECLARATION,
            "centers_xy_m needs coverage_basis: the Monte Carlo correction "
            "A_draw / A_patch is only unbiased for a known sampling density, and "
            "the density is not recoverable from the positions",
            declaration="coverage_basis",
            remedy=(
                "Declare CoverageBasis.UNIFORM_OVER_DILATED_APERTURE if the "
                "centres are an i.i.d. uniform draw over the dilated aperture's "
                "sample grid, or place them with patch_count instead. A guessed "
                "density gives a field wrong by a constant factor that looks "
                "entirely plausible."
            ),
        )

    max_center_px = (
        0.0
        if patch_count is None and not supplied
        else float(max(ny, nx) // 2 + patch_px // 2)
    )
    pad_px = resolve_pad_px(
        grid_n=max(ny, nx),
        patch_px=patch_px,
        pad_factor=pad_factor,
        max_center_px=max_center_px,
    )

    if patch_count is None and not supplied:
        centers = np.zeros((1, 2), dtype=np.float64)
        coverage = 1.0
        dilation = 0
    elif supplied:
        # The paper's actual configuration: each incident ray defines a patch,
        # so the centres come from the ray bundle rather than from a draw here.
        # They are snapped to the sample grid for the same reason drawn centres
        # are -- `extract_patch` indexes by nearest sample, so an unsnapped
        # centre extracts one patch while the ray launches from another.
        dilation = patch_px // 2
        raw = np.asarray(centers_xy_m, dtype=np.float64)
        if raw.ndim != 2 or raw.shape[1] != 2:
            raise ContractError(
                ContractCode.SHAPE_MISMATCH,
                f"centers_xy_m must be (P, 2), got {raw.shape}",
                declaration="centers_xy_m",
            )
        centers = np.column_stack(
            [
                np.round(raw[:, 0] / float(sample_pitch_m[1])) * float(sample_pitch_m[1]),
                np.round(raw[:, 1] / float(sample_pitch_m[0])) * float(sample_pitch_m[0]),
            ]
        )
        draw_positions = (2 * (ny // 2 + dilation) + 1) * (2 * (nx // 2 + dilation) + 1)
        coverage = float(draw_positions / (patch_px * patch_px))
    else:
        if rng is None:
            raise ContractError(
                ContractCode.MISSING_DECLARATION,
                "drawing patch centres needs an explicit seeded generator",
                declaration="rng",
            )
        # Dilate the draw region by half a patch. Without this the estimator is
        # unbiased in the interior and biased at the rim, which looks like a
        # soft edge rather than like a defect.
        dilation = patch_px // 2
        # Centres are drawn on the SAMPLE GRID, not continuously. `extract_patch`
        # indexes by nearest sample, so a continuous centre would extract a patch
        # at the snapped position while the ray launches at the unsnapped one --
        # a sub-sample offset that injects a linear phase and biases the
        # estimator. Measured: with continuous centres the patch-count sweep
        # plateaus at ~0.28 relative error instead of converging, and 100x more
        # patches buys almost nothing, which is the signature of a bias rather
        # than of variance.
        #
        # This is the third distinct half-sample bug in this module (the other
        # two are the odd-pad rule and the patch-centring rule). They are the
        # same mistake wearing different clothes: a grid has an origin, and every
        # quantity referred to it has to use the same one.
        rows = rng.integers(-(ny // 2 + dilation), ny // 2 + dilation + 1, size=patch_count)
        cols = rng.integers(-(nx // 2 + dilation), nx // 2 + dilation + 1, size=patch_count)
        centers = np.column_stack(
            [cols * float(sample_pitch_m[1]), rows * float(sample_pitch_m[0])]
        ).astype(np.float64)
        # Counted, not integrated: the draw is over a finite set of sample
        # positions, so the coverage correction is a ratio of counts.
        draw_positions = (2 * (ny // 2 + dilation) + 1) * (2 * (nx // 2 + dilation) + 1)
        coverage = float(draw_positions / (patch_px * patch_px))

    return PatchPlan(
        centers_xy_m=centers,
        patch_px=int(patch_px),
        pad_px=pad_px,
        coverage=coverage,
        dilation_px=int(dilation),
        curvature_bound_rad=float(budget.error_bound_rad),
    )


def extract_patch(
    field: np.ndarray[Any, Any],
    *,
    center_xy_m: tuple[float, float],
    patch_px: int,
    sample_pitch_m: tuple[float, float],
) -> np.ndarray[Any, Any]:
    """One ``patch_px x patch_px`` window of the DOE, zero outside it.

    **Zero continuation, not edge-clamp.** A bounded DOE has no field outside
    its aperture; continuing the edge value would invent structure that the
    coherent sum would then faithfully reproduce.

    Nearest-sample indexing on the plane's own origin rule (coordinate zero at
    index ``n // 2``). Bilinear interpolation between samples would smooth the
    DOE, and a DOE is exactly the thing whose sample-level structure matters.
    """
    array = np.asarray(field)
    ny, nx = array.shape
    cy = round(center_xy_m[1] / float(sample_pitch_m[0])) + ny // 2
    cx = round(center_xy_m[0] / float(sample_pitch_m[1])) + nx // 2
    half = patch_px // 2

    out = np.zeros((patch_px, patch_px), dtype=array.dtype)
    y0, y1 = cy - half, cy - half + patch_px
    x0, x1 = cx - half, cx - half + patch_px
    sy0, sx0 = max(y0, 0), max(x0, 0)
    sy1, sx1 = min(y1, ny), min(x1, nx)
    if sy0 < sy1 and sx0 < sx1:
        out[sy0 - y0 : sy1 - y0, sx0 - x0 : sx1 - x0] = array[sy0:sy1, sx0:sx1]
    return out


def patch_secondary_rays(
    doe_field: np.ndarray[Any, Any],
    *,
    plan: PatchPlan,
    sample_pitch_m: tuple[float, float],
    wavelength_m: float,
    plane: ReferencePlane,
    secondary_count: int | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[RayBundle, PatchDiagnostics]:
    """Emit secondary rays from every patch's own angular spectrum.

    ``secondary_count=None`` enumerates every propagating mode of every patch:
    the deterministic limit, and the configuration the exactness relation is
    measured in.

    The emitted bundle declares ``reconstruction_normalization = "one_over_n"``,
    which delegates the ``1 / (N_patches * S)`` of SI eq S5 to `ray_to_wave`
    rather than applying it here. One place owns that factor.
    """
    from core.boundary import Frame

    pitch_y, pitch_x = float(sample_pitch_m[0]), float(sample_pitch_m[1])
    pad = plan.pad_px

    # Spatial frequencies of the PADDED transform, centered to match
    # `ComplexField.coordinates` and `wave_to_ray.decompose`.
    fy = np.fft.fftshift(np.fft.fftfreq(pad, d=pitch_y))
    fx = np.fft.fftshift(np.fft.fftfreq(pad, d=pitch_x))
    grid_fx, grid_fy = np.meshgrid(fx, fy)
    dir_x = grid_fx * wavelength_m
    dir_y = grid_fy * wavelength_m
    radial = dir_x**2 + dir_y**2
    propagating = radial < 1.0
    n_propagating = int(propagating.sum())
    if n_propagating == 0:
        raise ContractError(
            ContractCode.EMPTY_ENSEMBLE,
            "no propagating modes on the padded patch grid",
            declaration="pad_px",
        )

    positions: list[np.ndarray[Any, Any]] = []
    directions: list[np.ndarray[Any, Any]] = []
    amplitudes: list[np.ndarray[Any, Any]] = []

    enumerated = secondary_count is None
    if not enumerated and rng is None:
        raise ContractError(
            ContractCode.MISSING_DECLARATION,
            "stochastic secondary sampling requires an explicit seeded generator",
            declaration="rng",
        )

    for center in np.asarray(plan.centers_xy_m, dtype=np.float64):
        patch = extract_patch(
            doe_field,
            center_xy_m=(float(center[0]), float(center[1])),
            patch_px=plan.patch_px,
            sample_pitch_m=sample_pitch_m,
        )
        padded = np.zeros((pad, pad), dtype=np.complex128)
        off = (pad - plan.patch_px) // 2
        padded[off : off + plan.patch_px, off : off + plan.patch_px] = patch

        # Matches wave_to_ray.decompose exactly, including the 1/n_pad^2, so no
        # stray inverse-DFT factor propagates downstream.
        spectrum = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(padded))) / (pad * pad)

        modal = spectrum[propagating]
        du = dir_x[propagating]
        dv = dir_y[propagating]

        if enumerated:
            # Uniform, NOT 1. Using p = 1 divides the exact sum by the mode
            # count and silently breaks the exactness relation.
            density = np.full(n_propagating, 1.0 / n_propagating)
            picks = np.arange(n_propagating)
        else:
            magnitude = np.abs(modal)
            total = float(magnitude.sum())
            density = (
                magnitude / total
                if total > 0.0
                else np.full(n_propagating, 1.0 / n_propagating)
            )
            picks = rng.choice(n_propagating, size=int(secondary_count), p=density)

        # No launch-position phase here, and the reason is the one thing about
        # this module most likely to be got wrong.
        #
        # `wave_to_ray.spectrum_to_rays` DOES apply `exp(i k (d_u x_p + d_v y_p))`,
        # because there the spectrum belongs to a field whose origin is the
        # plane's origin while the ray launches somewhere else, so the phase
        # between the two has to be carried. Here the padded patch is centred on
        # the patch centre, so the spectrum's own origin IS the launch point,
        # and `ray_to_wave` already references its ramp to each ray's position:
        # `dr_i(x, y) = d_x (x - x0_i) + d_y (y - y0_i)`. Adding the phase again
        # double-counts it.
        #
        # It is invisible on the full-aperture anchor, where the single centre
        # is at the origin and the extra factor is exactly 1 -- which is how it
        # survived until the sub-aperture relation was measured.
        amplitudes.append(plan.coverage * modal[picks] / density[picks])

        normal = np.sqrt(np.clip(1.0 - (du[picks] ** 2 + dv[picks] ** 2), 0.0, None))
        directions.append(np.column_stack([du[picks], dv[picks], normal]))
        positions.append(
            np.column_stack(
                [
                    np.full(picks.size, center[0]),
                    np.full(picks.size, center[1]),
                    np.full(picks.size, plane.z_m),
                ]
            )
        )

    bundle = RayBundle(
        positions_m=np.concatenate(positions),
        directions=np.concatenate(directions),
        wavelength_m=wavelength_m,
        reference_plane=plane,
        frame=Frame(),
        amplitude=np.concatenate(amplitudes),
        optical_path_length_m=np.zeros(sum(p.shape[0] for p in positions)),
        optical_path_length_reference=(
            f"zero at the patch plane {plane.name!r}; each patch's own centre "
            "phase is carried in the amplitude, so the path restarts here"
        ),
        reconstruction_normalization="one_over_n",
    )
    return bundle, PatchDiagnostics(
        patch_count=int(np.asarray(plan.centers_xy_m).shape[0]),
        patch_px=plan.patch_px,
        pad_px=plan.pad_px,
        coverage=plan.coverage,
        secondary_per_patch=(n_propagating if enumerated else int(secondary_count or 0)),
        outgoing_ray_count=int(bundle.count),
        enumerated=enumerated,
        propagating_modes=n_propagating,
        evanescent_modes=int(pad * pad - n_propagating),
        substrate=Substrate.PLANAR.value,
        curvature_bound_rad=plan.curvature_bound_rad,
    )


def advance_bundle_to_plane(bundle: RayBundle, *, target: ReferencePlane) -> RayBundle:
    """Move a bundle to a downstream plane along each ray's own direction.

    Two things happen and both are exact rather than approximate:

    * positions advance to ``z_target`` along ``d``, which for a plane offset
      ``dz`` is an arc length ``s = dz / d_z`` per ray;
    * the optical path advances by ``n * s``, here ``s`` with ``n = 1``.

    The second is what makes it exact: advancing by arc length ``s`` changes the
    per-ray constant phase by ``k s d_z^2``, and ``s d_z^2 = dz d_z``, which is
    precisely the phase an exact plane wave accumulates over ``dz``. It is not a
    paraxial step and no term is dropped.

    Rays travelling away from the target, or exactly parallel to it, are
    refused rather than silently dropped -- a bundle that quietly loses members
    on a transfer produces a plausible field with missing power.
    """
    positions = np.asarray(bundle.positions_m, dtype=np.float64)
    directions = np.asarray(bundle.directions, dtype=np.float64)
    dz = float(target.z_m) - positions[:, 2]
    dn = directions[:, 2]
    if np.any(np.abs(dn) < 1e-12):
        raise ContractError(
            ContractCode.NON_UNIT_DIRECTION,
            "a ray is parallel to the target plane and can never reach it",
            declaration="directions",
        )
    if np.any(dz * dn < 0.0):
        raise ContractError(
            ContractCode.REFERENCE_PLANE_MISMATCH,
            "a ray travels away from the target plane; refusing rather than "
            "dropping it, because a bundle that quietly loses members produces a "
            "plausible field with missing power",
            declaration="directions",
        )
    arc = dz / dn
    advanced = positions + directions * arc[:, None]
    opl = np.asarray(bundle.optical_path_length_m, dtype=np.float64) + arc
    return dataclasses.replace(
        bundle,
        positions_m=advanced,
        optical_path_length_m=opl,
        reference_plane=target,
        optical_path_length_reference=(
            f"{bundle.optical_path_length_reference}, then advanced along each "
            f"ray's own direction to the plane {target.name!r} at z = "
            f"{target.z_m:.6e} m. Exact: advancing by arc length s changes the "
            "per-ray constant phase by k s d_z^2, and s d_z^2 = dz d_z, which is "
            "the phase an exact plane wave accumulates over dz"
        ),
        provenance={**bundle.provenance, "advanced_from_z_m": float(bundle.reference_plane.z_m)},
    )
