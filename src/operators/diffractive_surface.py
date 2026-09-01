"""The diffractive surface as a physical operation: `RayBundle -> RayBundle`.

CHE-194 (R10.2). One public function, one class, and a composition written out
rather than generalized:

```python
operators.diffractive_surface(rays, *, surface, model="full_field", ...)
    -> tuple[RayBundle, dict[str, Any]]
```

```
RayBundle -> [ray_to_scalar] -> ScalarField -> [complex_transmission]
          -> ScalarField -> [scalar_to_ray] -> RayBundle
```

**The whole thing is a physical operator, and the two couplers inside it are
still couplers.** An optical surface changes the physical state; that the
implementation converts representation twice on the way is its *implementation*,
not its identity. A caller asking for this is asking what the surface does to the
light, not for a change of description. The reference implementation reached the
same conclusion and installed it as a rule, because `src/couplers/` had been
using one word for three different kinds of thing.

The shared boundary, which R10.1 settled
----------------------------------------
Everything here happens at the surface's own reference surface, and **this
operation does not propagate the rays to it** -- the bundle must already be
expressed there, which `ray_to_scalar`'s `surface=` expectation check enforces.
R10.1 measured the degenerate case: with `t = 1` the composition is
**bit-identical** to the couplers' own round trip, on one surface. That is the
sharpest available statement that the boundary is shared, and it is why the three
models of R10 can be one operation.

No composite framework, and the condition is stated
---------------------------------------------------
`docs/architecture_principles.md` permits a composite-operator framework only if
at least two *production* compositions immediately need it. **There is one.** So
this composition is hard-coded: three named calls in sequence, with the reasons
for each argument at the call site. If R10.3's patch route turns out to need the
same scaffolding that is the moment to reconsider, and the way to tell will be
that two functions in this module want to share a step -- not that one of them
looks like it could be parameterized.

What the parameter surface deliberately does not have
------------------------------------------------------
The reference implementation's `FullFieldParameters` had eight fields. Three of
them are gone and the reasons are not stylistic:

* `preserve_energy` -- defaulted to `False` with the note that "it should stay
  off: a lossy surface legitimately loses power". A knob that should stay off is
  not a knob. If a caller wants a normalized field they can normalize one, and
  they will then own having done it.
* `pad_width` -- zero padding before the transform. `ScalarField` carries a pad
  state and `scalar_to_ray` transforms on the field's own grid, so padding here
  would be a new numerical capability with no consumer asking for it. The
  surface's grid *is* the transform grid, which is also what removes the
  shape-disagreement the reference's own `DiffractiveSurface` docstring
  complains about.
* `primary_sampling` / `primary_count` -- the primaries are the *incident rays*,
  and they are an argument. There is nothing left to sample.

**And one knob that is deliberately not exposed: `projection`.** The incident
reconstruction is always `ASM_CONSISTENT`. `SENSOR_OBLIQUITY` is a *detector*
model -- R07.1 measured that it does not preserve the field -- so offering it
inside a surface transformation would let a caller select an operator that loses
a few percent off-axis at a place where nothing is being detected. It is fixed
rather than defaulted, and this paragraph is why.

`launch_positions_xy_m` survives, passed through to `scalar_to_ray`, and its
default is one point at the transverse origin. R08.1 measured why that is the
right default: with the mode indices drawn once and reused, `P` launch points
reproduce the identical estimator -- the launch phase cancels `ray_to_scalar`'s
`-d . x0` exactly -- so they cost `P` times as much and buy nothing. It is
exposed because it is the argument a caller would reach for, and documented so
they do not.

Which limitations the emitted bundle inherits, and where it says so
-------------------------------------------------------------------
This is the ticket's named risk: a composition that inherits both couplers'
limitations without declaring either, so a downstream consumer sees a clean ray
bundle with two undeclared approximations baked in. `RayBundle` has no `validity`
field -- only `ScalarField` does -- so the declaration travels two ways:

* in the emitted bundle's `optical_path_reference`, which is free text a
  consumer reads and which already describes where the path is measured from.
  It now also says which model transformed it and what the interior field
  declared. This is the half that survives a caller dropping the record.
* in the returned diagnostics, structured, including the interior field's
  `validity` set verbatim.

The interior field declares `surface_only` and `no_wavefront_curvature_term`
(CHE-50), and may declare a grazing band limit if R07.4's floor excluded modes.
The outgoing rays are a *ray* state again, so `surface_only` no longer binds them
-- `operators.propagate_rays` will happily advance them, and correctly. What does
not come back is the curvature the interior reconstruction never had.

Air only, for now
-----------------
R09 found that neither coupler carries the refractive index in its ramp, and both
now refuse `medium_index != 1`. This composition inherits that refusal from its
parts rather than restating it, which is the right place for it: when the
convention is settled, this module needs no change.

This module imports `couplers` and `operators.transmission`, and no solver and no
backend. The two couplers inside are the same functions R07 and R08 built -- no
private copy and no variant kernel, which
`tests/physics/test_diffractive_surface_full_field.py` asserts by identity.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from couplers import (
    DEFAULT_KSPACE_OVERSAMPLE,
    DrawRule,
    Reconstruction,
    SamplingDensity,
    ray_to_scalar,
    scalar_to_ray,
)
from operators.patch_curvature import (
    curvature_observability_width,
    require_patch_within_curvature,
)
from operators.transmission import complex_transmission
from representations import ContractError, RayBundle, ReferenceSurface, ScalarField

__all__ = [
    "DIFFRACTIVE_MODELS",
    "DiffractiveModel",
    "DiffractiveSurface",
    "diffractive_surface",
]

#: Which granularity computes the interaction. **Never inferred.**
#:
#: A `Literal`, not a `StrEnum`, for the reason every other vocabulary in this
#: tree is one: the class-budget gate counts a `StrEnum` as a class and R10.2
#: budgets +1, which `DiffractiveSurface` spends.
#:
#: **One member, and that is the honest count rather than a placeholder.**
#: `local_patch` arrives with R10.3 and `generalized_snell` with R10.4, each with
#: its own evidence. A vocabulary that named models nothing implements would be a
#: capability claim, which is what `SEMANTIC_TYPES` and `MEASURE_KINDS` are
#: enumerated to prevent.
#:
#: `full_field`
#:     Global angular-spectrum treatment: accumulate every incident ray
#:     coherently onto the one common surface, apply the complex transmission
#:     once, decompose once. SI Algorithm S1. Available when one common plane
#:     exists, which on a planar substrate it does.
#: `local_patch`
#:     Local tangent-plane windowed-Fourier treatment: the illuminated surface is
#:     windowed patch by patch, each patch is decomposed on its own padded grid,
#:     and the patches' rays are concatenated. SI eq S1 and S3-S5. SI S10 calls
#:     this "the direct implementation" and `full_field` the *shortcut* available
#:     when one common plane exists -- so `full_field` is `local_patch` at one
#:     full-aperture patch, which `tests/physics/test_diffractive_surface_patch.py`
#:     measures rather than asserts.
DiffractiveModel = Literal["full_field", "local_patch"]

DIFFRACTIVE_MODELS: tuple[DiffractiveModel, ...] = ("full_field", "local_patch")

#: The patch window, declared because it is a model parameter with one value.
#:
#: Exactly one member, and that is the honest count. Any taper below 1 removes
#: field that no other patch replaces, so the coherent patch sum stops converging
#: to the full-surface response -- the partition-of-unity argument behind the SI S2
#: convergence relation is exactly what a taper breaks. It is offered as a
#: *declaration* so a caller can see that the rectangular choice was made, and a
#: future taper arrives as a new member with its own evidence rather than as a
#: silent default change.
PatchWindow = Literal["rectangular"]


@dataclass(frozen=True)
class DiffractiveSurface:
    """The diffractive surface as one declared argument.

    A class on rule 1: the transmission, the pitch it is sampled at and the
    surface it lives on are one physical object with joint invariants, and the
    failure mode is specific. Before the reference implementation gathered them,
    "the diffractive surface" was four loose arguments repeated at every call
    site -- an array, a grid shape, a pitch and a plane -- so a caller could
    describe one surface to one function and a different one to the next without
    anything noticing.

    Rule 2 as well: it is the public argument schema of the operation, so its
    field names are an interface.

    Conventions, stated because none of them is visible in an intensity:

    * `transmission` is a **complex amplitude transmission** -- not an intensity
      and not a phase. A real array is refused: it is an amplitude mask with an
      undeclared phase, and reading `|t|` as `t` throws away the part that
      diffracts. Use `from_phase` for a phase-only surface.
    * the grid is `(ny, nx)` with coordinate zero at index `n // 2`, matching
      `ScalarField.coordinates`. It is **read off the array** rather than
      declared beside it, which removes the shape disagreement entirely.
    * `reference_surface` is where the transmission lives. This operation does
      not propagate the incident bundle to it; the bundle must already be
      expressed there, and `ray_to_scalar`'s expectation check enforces that.

    """

    #: `(ny, nx)` complex amplitude transmission.
    transmission: Any

    #: `(dy, dx)` sample spacing in metres.
    sample_pitch_m: tuple[float, float]

    #: Where the transmission lives. The incident bundle must already be on it.
    reference_surface: ReferenceSurface

    #: Radius of curvature of the substrate, in metres. `inf` is planar, and is
    #: the default.
    #:
    #: **The radius is the declaration; there is no `Substrate` enum.** The
    #: reference implementation carried both and had to guard against them
    #: disagreeing -- "substrate declares a flat surface but radius_m declares a
    #: curved one". Two fields that must agree are one field, and `inf` says
    #: planar unambiguously.
    #:
    #: Read only by `local_patch`, whose curvature envelope needs it. `full_field`
    #: requires one common plane and refuses a finite radius: on a curved
    #: substrate rays intersect different local tangent planes with
    #: position-dependent frames, so its central accumulation has no meaning there
    #: (SI S10). Refused rather than allowed to fall back, because the
    #: accumulation would still *compute* and would return something that looks
    #: like a diffraction pattern.
    radius_m: float = math.inf

    def __post_init__(self) -> None:
        transmission = np.asarray(self.transmission)
        if transmission.ndim != 2:
            raise ContractError(
                "SHAPE_MISMATCH",
                f"the transmission must be a 2-D (ny, nx) grid, got shape "
                f"{tuple(transmission.shape)}",
                declaration="transmission",
            )
        if not np.iscomplexobj(transmission):
            raise ContractError(
                "DTYPE_KIND_MISMATCH",
                "the transmission must be complex. A real array is an amplitude mask "
                "with an undeclared phase, and the phase is the part that diffracts.",
                declaration="transmission",
                remedy=(
                    "Use DiffractiveSurface.from_phase() for a phase-only surface, or "
                    "supply exp(+i phi) yourself -- note the sign."
                ),
            )
        if not bool(np.all(np.isfinite(transmission))):
            raise ContractError(
                "NON_FINITE",
                "the transmission contains non-finite values, which propagate through "
                "the coherent sum to make the whole field NaN rather than a locally "
                "wrong one",
                declaration="transmission",
            )
        pitch = tuple(float(value) for value in self.sample_pitch_m)
        if len(pitch) != 2 or not all(math.isfinite(p) and p > 0.0 for p in pitch):
            raise ContractError(
                "UNIT_NOT_SI",
                f"sample_pitch_m must be two positive finite lengths in metres, got "
                f"{self.sample_pitch_m!r}",
                declaration="sample_pitch_m",
            )
        radius = float(self.radius_m)
        if not (radius > 0.0) or math.isnan(radius):
            raise ContractError(
                "UNIT_NOT_SI",
                f"radius_m must be a positive length in metres or inf for a planar "
                f"substrate, got {self.radius_m!r}",
                declaration="radius_m",
            )
        object.__setattr__(self, "transmission", transmission)
        object.__setattr__(self, "sample_pitch_m", pitch)
        object.__setattr__(self, "radius_m", radius)

    @property
    def grid_shape(self) -> tuple[int, int]:
        """`(ny, nx)`, read off the transmission rather than declared beside it."""
        shape = np.asarray(self.transmission).shape
        return (int(shape[0]), int(shape[1]))

    @classmethod
    def from_phase(
        cls,
        phase_rad: Any,
        *,
        sample_pitch_m: tuple[float, float],
        reference_surface: ReferenceSurface,
        radius_m: float = math.inf,
    ) -> DiffractiveSurface:
        """A lossless phase-only surface, `t = exp(+i phi)`.

        A classmethod rather than a second class, and it exists for the sign. The
        `+` is `representations.PHASOR`'s, applied in one place instead of at each
        call site: a caller writing `exp(-i phi)` gets a conjugated surface, which
        is a real DOE that focuses on the wrong side of the substrate and looks
        entirely plausible in any intensity.
        """
        phase = np.asarray(phase_rad, dtype=np.float64)
        return cls(
            transmission=np.exp(1j * phase),
            sample_pitch_m=sample_pitch_m,
            reference_surface=reference_surface,
            radius_m=radius_m,
        )


def resolve_pad_px(
    *,
    grid_n: int,
    patch_px: int,
    pad_factor: int = 2,
    max_center_px: int = 0,
    full_aperture: bool = False,
) -> int:
    """The smallest padded transform size that reconstructs exactly.

    Two conditions, both measured by the reference implementation rather than
    assumed, and `pad_factor * patch_px` is a **preference** that is raised until
    they hold. Returning a size the caller did not ask for is right here: silently
    using one that violates condition 1 produces a plausible field that is wrong by
    100 %.

    1. **Clearance.** No periodic replica of a patch may enter the reconstruction
       window. A patch centred at `c` has replicas at `c +- patch/2 + m * pad` and
       the window has half-width `grid_n/2`, so

           pad > max|c| + patch/2 + grid_n/2.

       `max_center_px` is that `max|c|`, in samples, and it is **passed in from the
       centres the tiling actually produced** rather than assumed to be `grid_n/2`.
       A tiling's outermost centres sit outside the grid -- measured, a 25-px patch
       on a 65-px grid puts one at 50 px, needing `pad > 94` where the
       centres-inside-the-grid form gives 91 -- so deriving the bound from
       `grid_n` alone would leave the guarantee to whether those far patches
       happened to be empty.

       The bound is **strict**: at equality the replica's edge sample lands on the
       window's edge sample, and a patch is rarely zero at its own edge. Measured
       on a 3x3 tiling of an 11-px patch over a 33-px grid -- pad 32 gives 1.44,
       pad 33 gives 5.6e-15, pad 23 gives 1.29. That plateau is what a patch-count
       sweep looks like when it stops converging.
    2. **Centring.** `pad_px - patch_px` must be even, so the patch sits on the
       padded array's centre sample. Odd puts it half a sample off and injects a
       linear phase -- pad 34 and pad 44 both clear condition 1 and both give ~1.4.

    The reference implementation had a third condition, oddness, because its
    reconstruction refused a mode landing exactly on `lambda / (2 pitch)`. R07.1's
    Nyquist check admits that mode within a derived edge tolerance, so the
    condition no longer binds -- and with an odd `patch_px` (which the patch route
    requires anyway) condition 2 already forces an odd `pad_px`.

    **A larger pad is not automatically safer**, and this is the subtlety that
    decides how the two routes are compared. Padding beyond the minimum makes the
    reconstruction *less* periodic and therefore more physical -- but the
    full-field route is an unpadded transform on the surface's own grid and is
    periodic with it. Comparing a weakly-periodic patch route against a
    strongly-periodic one measures the difference in wraparound, not an error in
    either: measured here, a 64x64 uniform field padded to 131 reconstructs 13.5 %
    away from its periodic self, with the peak 11 % high from the edge ringing that
    zero padding introduces and periodicity hides.

    `full_aperture=True` is the exemption for exactly that: one patch covering the
    whole grid, padded to itself, so the period *is* the reconstruction window and
    both routes compute the same periodic problem. That is the configuration the
    `full_field == local_patch` identity is measured in, and it is the reference
    implementation's exemption too.
    """
    if full_aperture and patch_px >= grid_n:
        return patch_px
    if patch_px <= 0 or patch_px % 2 == 0:
        raise ContractError(
            "SHAPE_MISMATCH",
            f"patch_px={patch_px} must be positive and odd. An even patch has no centre "
            "sample, so 'centred on a position' is undefined for it.",
            declaration="patch_px",
            remedy=f"Use {max(patch_px + 1, 1)} or {max(patch_px - 1, 1)}.",
        )
    if pad_factor < 1:
        raise ContractError(
            "SHAPE_MISMATCH",
            f"pad_factor={pad_factor} must be at least 1; the padded grid cannot be "
            "smaller than the patch",
            declaration="pad_factor",
        )
    clearance = 2 * int(max_center_px) + patch_px + grid_n + 1
    pad = max(patch_px * int(pad_factor), clearance)
    if (pad - patch_px) % 2:
        pad += 1
    return pad


def _patch_centres(
    *, grid_shape: tuple[int, int], patch_px: int, sample_pitch_m: tuple[float, float]
) -> Any:
    """`(P, 2)` centres of a non-overlapping tiling that covers the whole grid, in metres.

    A **deterministic tiling**, and that is the whole of the coverage story here.
    The reference implementation also drew centres at random over a dilated
    aperture and carried an `A_draw / A_patch` correction with a declared
    `CoverageBasis` to keep the estimator unbiased. None of that is ported: it
    belongs to the patch-emitter estimator R10.3's own avoided list targets
    (`PatchPlan`, `PatchDiagnostics`, `PatchEmitterCostModel`, the thread pool),
    it has no caller in this tree, and a coverage correction with nothing to
    correct is speculative scaffolding.

    A tiling is also the configuration every exactness statement is made in: the
    patches partition the surface, so their windowed fields sum to it exactly and
    the coherent sum of their rays is the whole response with no estimator in the
    way.
    """
    ny, nx = grid_shape
    dy, dx = sample_pitch_m
    half = patch_px // 2

    def multiples(count: int) -> Any:
        """`k` such that the patches at `origin + k * patch_px` contain `[0, count)`.

        Patch `k` covers rows `[o + k p - h, o + k p - h + p)`, so containment needs
        `o + k_min p - h <= 0` and `o + k_max p - h + p - 1 >= count - 1`. Solving
        each and rounding outward is the whole derivation -- and it is written out
        because the obvious integer-division form leaves a band of the surface
        covered by no patch at all, which is silent: the record would report a patch
        count and nothing about the rows nobody looked at.
        """
        origin = count // 2
        lowest = math.floor((half - origin) / patch_px)
        highest = math.ceil((count - 1 + half - origin - patch_px + 1) / patch_px)
        return np.arange(lowest, highest + 1)

    centre_y, centre_x = np.meshgrid(
        multiples(ny) * patch_px * dy, multiples(nx) * patch_px * dx, indexing="ij"
    )
    return np.column_stack([centre_x.ravel(), centre_y.ravel()])


def _windowed_patch(
    field: Any, *, centre_xy_m: Any, patch_px: int, pad_px: int
) -> Any:
    """One `pad_px x pad_px` array holding the patch around `centre_xy_m`, zero elsewhere.

    **Zero continuation, not edge-clamp.** A bounded surface has no field outside
    its aperture, and continuing the edge value would invent structure the coherent
    sum would then faithfully reproduce.

    **Nearest-sample indexing** on the field's own origin rule. Bilinear
    interpolation between samples would smooth the surface, and a diffractive
    surface is exactly the thing whose sample-level structure matters.
    """
    array = np.asarray(field.u)
    ny, nx = array.shape
    dy, dx = field.sample_pitch_m
    centre_row = round(float(centre_xy_m[1]) / dy) + field.frame.origin_index(ny)
    centre_col = round(float(centre_xy_m[0]) / dx) + field.frame.origin_index(nx)
    half = patch_px // 2

    patch = np.zeros((patch_px, patch_px), dtype=array.dtype)
    top, bottom = centre_row - half, centre_row - half + patch_px
    left, right = centre_col - half, centre_col - half + patch_px
    source_top, source_left = max(top, 0), max(left, 0)
    source_bottom, source_right = min(bottom, ny), min(right, nx)
    if source_top < source_bottom and source_left < source_right:
        patch[
            source_top - top : source_bottom - top,
            source_left - left : source_right - left,
        ] = array[source_top:source_bottom, source_left:source_right]

    # Centred zero padding: `(pad - patch)` is even by `resolve_pad_px`, so the
    # patch's own centre sample lands on the padded array's centre sample and no
    # linear phase is injected.
    margin = (pad_px - patch_px) // 2
    return np.pad(patch, margin, mode="constant")


def _decompose_by_patch(
    field: Any,
    *,
    surface: DiffractiveSurface,
    patch_px: int | None,
    pad_factor: int,
    window: PatchWindow,
    error_threshold_rad: float,
    count: int | None,
    density: SamplingDensity,
    draw: DrawRule,
    rng: Any,
    seed: int | None,
) -> tuple[RayBundle, Any, dict[str, Any]]:
    """Window the illuminated surface, decompose each patch, place its rays.

    **The illuminated surface, not the bare transmission**, and that is R10.1's
    correction rather than a detail. The reference implementation's patch branch
    passed `surface.transmission` and never read the incident bundle, so two rays
    with different amplitudes, phases or directions produced the same outgoing
    bundle. `field` here is the *transmitted* field -- the incident reconstruction
    times the transmission -- so the illumination is in it.

    Each patch is decomposed by `scalar_to_ray` on its own padded grid, which puts
    the patch's coefficients in the patch's *own* coordinates. The emitted rays are
    then **translated** to the patch centre rather than launched there: launching
    would apply `exp(+i k d . c)` on top of coefficients that are already centred
    at `c`, and the reconstruction would place the patch back at the global origin.
    Translating is the operation that says "this patch's field lives here".
    """
    if window != "rectangular":
        raise ContractError(
            "MISSING_DECLARATION",
            f"window={window!r} is not implemented. Any taper below 1 removes field that "
            "no other patch replaces, so the coherent patch sum stops converging to the "
            "full-surface response -- the partition-of-unity argument behind the SI S2 "
            "convergence relation is what a taper breaks.",
            declaration="window",
            remedy=(
                "Leave the window rectangular. A taper would have to be declared as "
                "trading the exactness guarantee for a smoother spectrum, with its own "
                "evidence, rather than selected as an option."
            ),
        )
    ny, nx = field.shape
    if patch_px is None:
        # The full-aperture patch: the exactness anchor, and the configuration in
        # which `full_field` is this model's special case (SI S10).
        patch_px = max(ny, nx)
        if patch_px % 2 == 0:
            patch_px += 1
    if patch_px <= 0 or patch_px % 2 == 0:
        raise ContractError(
            "SHAPE_MISMATCH",
            f"patch_px={patch_px} must be positive and odd. An even patch has no centre "
            "sample, so 'centred on a position' is undefined for it.",
            declaration="patch_px",
        )

    if not math.isinf(surface.radius_m):
        raise ContractError(
            "MISSING_DECLARATION",
            f"model='local_patch' cannot yet be applied to a substrate of radius "
            f"{surface.radius_m!r} m. SI S10 identifies this model as *the* applicable "
            "one on a curved substrate, and that is the point: what is missing is the "
            "implementation, not the model. Newton sag intersection, per-hit tangent "
            "frames and position-dependent normals are all absent -- every patch here is "
            "windowed from one planar field and every ray comes back on one planar "
            "surface, so a curved substrate would get a purely planar answer with a "
            "curvature margin attached to it, which is worse than a refusal.",
            declaration="radius_m",
            remedy=(
                "Declare a planar substrate (radius_m = inf). To size a patch against a "
                "curvature you will have later, call "
                "`operators.patch_curvature.require_patch_within_curvature` directly -- "
                "the envelope is implemented and tested; the geometry is not."
            ),
        )

    dy, dx = field.sample_pitch_m
    patch_width_m = float(patch_px) * max(dy, dx)
    # Still evaluated, on a planar substrate, so the record shows the check ran and
    # carries the margin the caller has against the threshold they declared.
    budget = require_patch_within_curvature(
        patch_width_m=patch_width_m,
        radius_m=surface.radius_m,
        error_threshold_rad=error_threshold_rad,
    )

    full_aperture = patch_px >= max(ny, nx)
    centres = (
        np.zeros((1, 2))
        if full_aperture
        else _patch_centres(
            grid_shape=(ny, nx), patch_px=patch_px, sample_pitch_m=(dy, dx)
        )
    )
    max_center_px = int(
        np.max(np.abs(centres / np.asarray([dx, dy]))) if centres.size else 0
    )
    pad_px = resolve_pad_px(
        grid_n=max(ny, nx),
        patch_px=patch_px,
        pad_factor=pad_factor,
        max_center_px=max_center_px,
        full_aperture=full_aperture,
    )

    positions: list[Any] = []
    directions: list[Any] = []
    amplitudes: list[Any] = []
    weights: list[Any] = []
    optical_paths: list[Any] = []
    last_sampling = None
    for centre in centres:
        patch = _windowed_patch(
            field, centre_xy_m=centre, patch_px=patch_px, pad_px=pad_px
        )
        if not bool(np.any(patch)):
            # An empty window's patch is identically zero, so excluding it is exact
            # rather than approximate -- and including it would spend a whole
            # secondary budget on zero-amplitude rays.
            continue
        patch_field = ScalarField(
            u=patch,
            sample_pitch_m=(dy, dx),
            wavelength_m=field.wavelength_m,
            reference_surface=field.reference_surface,
            frame=field.frame,
            validity=field.validity,
        )
        # `seed=None` per patch, deliberately. `scalar_to_ray` refuses a recorded
        # seed whose generator has already been drawn from -- correctly, since it
        # would not regenerate that draw -- so forwarding the caller's seed to every
        # patch made the second one raise and blamed the caller for this function's
        # reuse. The seed belongs to the *operation*, and it is recorded once, at
        # the operator's own level.
        rays, last_sampling = scalar_to_ray(
            patch_field,
            surface=field.reference_surface,
            count=count,
            density=density,
            draw=draw,
            rng=rng,
        )
        offset = np.asarray([float(centre[0]), float(centre[1]), 0.0])
        positions.append(np.asarray(rays.positions_m) + offset)
        directions.append(np.asarray(rays.directions))
        amplitudes.append(np.asarray(rays.amplitude))
        weights.append(np.asarray(rays.measure_weight))
        optical_paths.append(np.asarray(rays.optical_path_m))

    if not positions:
        raise ContractError(
            "EMPTY_ENSEMBLE",
            "every patch of the transmitted field is identically zero, so there is "
            "nothing to emit",
            declaration="transmission",
        )

    # The coverage factor, and it is not optional. `ray_to_scalar` divides an
    # `importance_weight` ensemble by its **total** ray count, but each patch's
    # weights were built for that patch's own mode count -- so `P` patches would
    # come out `N_total / N_p` too small, exactly `1/P` for equal patches. The
    # patches *partition* the surface, so their contributions must sum rather than
    # average, and this is the factor that says so.
    #
    # It is a deterministic tiling's version of the reference implementation's
    # `A_draw / A_patch` correction: with every patch drawn exactly once there is
    # no density to correct for, only the count.
    total_rays = int(sum(int(block.shape[0]) for block in positions))
    weights = [
        block * (total_rays / float(block.shape[0])) for block in weights
    ]

    outgoing = RayBundle(
        positions_m=np.concatenate(positions),
        directions=np.concatenate(directions),
        wavelength_m=field.wavelength_m,
        reference_surface=field.reference_surface,
        frame=field.frame,
        amplitude=np.concatenate(amplitudes),
        optical_path_m=np.concatenate(optical_paths),
        optical_path_reference=(
            f"zero at the emitting surface {field.reference_surface.name!r}; the "
            "accumulated path restarts here"
        ),
        measure_weight=np.concatenate(weights),
        measure_kind="importance_weight",
    )
    return (
        outgoing,
        None,
        {
            "window": window,
            "patch_px": patch_px,
            "pad_px": pad_px,
            "patch_count": len(positions),
            "candidate_patch_count": int(centres.shape[0]),
            "patch_width_m": patch_width_m,
            "max_center_px": max_center_px,
            "seed": seed,
            # Named for what it is. It describes the **last patch**, not the
            # ensemble: its `ray_count` is one patch's and its `grid_shape` is the
            # padded patch grid, neither of which is the emitted bundle's. A record
            # that called itself the ensemble's would be inventing provenance.
            "last_patch_sampling": None if last_sampling is None else last_sampling.as_dict(),
            "curvature": budget,
            "observability_width_m": curvature_observability_width(
                wavelength_m=field.wavelength_m, radius_m=surface.radius_m
            ),
        },
    )


def diffractive_surface(
    rays: RayBundle,
    *,
    surface: DiffractiveSurface,
    model: DiffractiveModel = "full_field",
    reconstruction: Reconstruction = Reconstruction.DIRECT,
    kspace_oversample: float = DEFAULT_KSPACE_OVERSAMPLE,
    kspace_grid_shape: tuple[int, int] | None = None,
    allow_gain: bool = False,
    patch_px: int | None = None,
    pad_factor: int = 2,
    window: PatchWindow = "rectangular",
    error_threshold_rad: float = 1.0e-3,
    count: int | None = None,
    density: SamplingDensity = "uniform",
    draw: DrawRule = "iid",
    rng: np.random.Generator | None = None,
    seed: int | None = None,
    launch_positions_xy_m: Any = None,
) -> tuple[RayBundle, dict[str, Any]]:
    """Transform `rays` through `surface`. Incident coherent rays in, outgoing rays out.

    Parameters
    ----------
    rays
        The incident bundle, already expressed on `surface.reference_surface`.
        Every model reads it -- which is R10.1's correction to the reference
        implementation, whose patch branch read only its wavelength and therefore
        computed the response of the bare surface under implicitly uniform
        illumination.
    surface
        The diffractive surface. Its grid is the reconstruction grid, so the two
        cannot disagree.
    model
        Which granularity computes it. Named, never inferred. One member today.
    reconstruction, kspace_oversample, kspace_grid_shape
        Which realization of the wavelet sum reconstructs the incident field,
        passed to `couplers.ray_to_scalar`. Exposed because it is the **cost**
        knob and this operation is where the cost lands: `DIRECT` is
        `O(N_rays x ny x nx)`, so 512^2 incident rays onto a 512^2 surface is
        about 7e10 term evaluations. `KSPACE` is `O(N_rays + K log K)` and carries
        R07.2's declared interpolation budget -- read it before choosing, because
        at the default oversampling the two routes differ by tens of percent.
    allow_gain
        Passed to the thin element. Exposed only so its refusal's remedy is
        actionable from here; a surface with `|t| > 1` is a modelling error rather
        than a knob, and the default says so.
    count, density, draw, rng, seed
        The outgoing sampling, passed to `couplers.scalar_to_ray`. `count=None`
        enumerates every propagating mode of the transmitted field -- the
        deterministic limit, with no sampling error, and the configuration every
        exactness gate is measured in.
    launch_positions_xy_m
        Where the outgoing rays are launched from, passed through. `None` is one
        point at the transverse origin, and that is the right default rather than
        a lazy one: see the module docstring.

    Returns
    -------
    The outgoing bundle, and a diagnostics mapping.

    The mapping is a `dict` rather than a record type, and that is a deliberate
    non-addition. Its two substantial entries are the `as_dict()` of the typed
    records `ray_to_scalar` and `scalar_to_ray` already produce; a dataclass whose
    only job is to hold two dataclasses is the field creep R10 lists nine class
    names against. What it adds over them is the envelope: which model ran, and
    what the interior field declared about itself.

    Raises
    ------
    ContractError
        Everything the parts refuse, unchanged and not restated: an undeclared
        measure or a non-unit medium index on the way in, a grid that cannot
        represent the steepest ramp, a grazing mode the precision cannot carry, a
        surface the bundle is not on. Plus `MISSING_DECLARATION` for an unknown
        `model` and `SHAPE_MISMATCH` if the transmission does not match the field
        it modulates.
    """
    if model not in DIFFRACTIVE_MODELS:
        raise ContractError(
            "MISSING_DECLARATION",
            f"model must be one of {list(DIFFRACTIVE_MODELS)}, got {model!r}. The model "
            "is named rather than inferred: a caller who names one and is quietly given "
            "another's physics has no way to find out.",
            declaration="model",
        )

    # 1. Ray -> scalar, on the surface's own grid and at the surface the rays are
    # already declared on. Passing `surface=` is the expectation check that makes
    # R10.1's shared boundary executable rather than documented: this operation
    # does not propagate, so a bundle declared elsewhere is refused here.
    incident_field, reconstruction_record = ray_to_scalar(
        rays,
        grid_shape=surface.grid_shape,
        sample_pitch_m=surface.sample_pitch_m,
        surface=surface.reference_surface,
        reconstruction=reconstruction,
        kspace_oversample=kspace_oversample,
        kspace_grid_shape=kspace_grid_shape,
    )

    # 2. The surface, as the thin element it is. `operators.complex_transmission`
    # is R06.6's, unchanged: this composition adds no transmission code of its
    # own, so there is no second place where the phase sign could be written.
    # No shape guard between the two: the reconstruction grid *is*
    # `surface.grid_shape`, read off this same array, so a mismatch is not
    # reachable. A branch that cannot be reached is a claim about a failure path
    # that does not exist, which is what `CONTRACT_CODES` is enumerated against.
    transmission = np.asarray(surface.transmission)
    transmitted_field = complex_transmission(
        incident_field,
        amplitude=np.abs(transmission),
        phase_rad=np.angle(transmission),
        allow_gain=allow_gain,
    )

    # 3. Scalar -> ray. The transmitted field is decomposed into the modes that
    # leave the surface -- globally for `full_field`, patch by patch for
    # `local_patch`. Both call the same `scalar_to_ray`.
    patch_record: dict[str, Any] | None = None
    if model == "local_patch":
        outgoing, sampling, patch_record = _decompose_by_patch(
            transmitted_field,
            surface=surface,
            patch_px=patch_px,
            pad_factor=pad_factor,
            window=window,
            error_threshold_rad=error_threshold_rad,
            count=count,
            density=density,
            draw=draw,
            rng=rng,
            seed=seed,
        )
    else:
        if not math.isinf(surface.radius_m):
            raise ContractError(
                "MISSING_DECLARATION",
                f"model='full_field' cannot be applied to a substrate of radius "
                f"{surface.radius_m!r} m. Its central step is one coherent accumulation "
                "onto the ONE common plane every incident ray crosses; on a curved "
                "substrate rays intersect different local tangent planes with "
                "position-dependent frames and normals, so there is no such plane "
                "(SI S10). Refused rather than allowed to fall back, because the "
                "accumulation would still compute and would return something that looks "
                "like a diffraction pattern.",
                declaration="model",
                remedy="Use model='local_patch', which SI S10 calls the direct implementation.",
            )
        if patch_px is not None:
            raise ContractError(
                "MISSING_DECLARATION",
                f"patch_px={patch_px!r} was supplied with model='full_field', which has "
                "no patches. The model and its parameters are not inferred from each "
                "other in either direction.",
                declaration="patch_px",
            )
        outgoing, sampling = scalar_to_ray(
            transmitted_field,
            surface=surface.reference_surface,
            count=count,
            density=density,
            draw=draw,
            rng=rng,
            seed=seed,
            launch_positions_xy_m=launch_positions_xy_m,
        )

    # The ticket's named risk: the composition must declare what happened inside
    # it. `RayBundle` has no `validity` field, so the statement goes where a
    # consumer already reads provenance -- and it survives a caller dropping the
    # diagnostics.
    interior = sorted(transmitted_field.validity)
    outgoing = dataclasses.replace(
        outgoing,
        optical_path_reference=(
            f"{outgoing.optical_path_reference}, after a {model!r} diffractive-surface "
            f"transformation at {surface.reference_surface.name!r} whose interior field "
            f"declared {interior}"
        ),
    )

    record: dict[str, Any] = {
        "model": model,
        # Verbatim, because the interior field's own declaration is the honest
        # statement of what the reconstruction could and could not carry.
        "interior_field_validity": interior,
        "reconstruction": reconstruction_record.as_dict(),
        # `None` on the patch route: no single typed record describes an ensemble
        # assembled from `P` decompositions, and one that claimed to would be
        # reporting one patch's ray count and the padded patch grid as the
        # ensemble's. `record["patch"]["last_patch_sampling"]` is the honest form.
        "sampling": None if sampling is None else sampling.as_dict(),
    }
    if patch_record is not None:
        record["patch"] = patch_record
    return outgoing, record
