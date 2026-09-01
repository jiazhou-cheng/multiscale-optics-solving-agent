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
expressed there. Two models get that check from `ray_to_scalar`'s `surface=`
expectation; `generalized_snell` forms no field, so it makes the same check
itself rather than inheriting it.
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
#: **Three members, all implemented.** The vocabulary grew one model per ticket --
#: `full_field` with R10.1, `local_patch` with R10.3, `generalized_snell` with
#: R10.4 -- each landing with its own evidence, because a vocabulary that named a
#: model nothing implements would be a capability claim, which is what
#: `SEMANTIC_TYPES` and `MEASURE_KINDS` are enumerated to prevent.
#:
#: **They are not interchangeable.** The first two form a field and emit every
#: order; `generalized_snell` is a *reduction* that returns one ray per ray, and
#: it carries three signed validity margins the other two have no need of. Its
#: declared domain is narrower in four ways they are not bounded by: a **planar**
#: substrate, a **locally constant modulus** (so no Ronchi grating and no hard
#: aperture edge -- both diffract by amplitude, which it cannot see), an order
#: **|m| <= 1**, and rays **on the surface's own sampled extent**. Each is a
#: refusal, not a caveat.
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
#: `generalized_snell`
#:     The **reduced-order** model: no field is formed at all. Each incident ray
#:     is redirected by a local grating equation evaluated at its own transverse
#:     position, and comes out still one ray. Planar substrate only, and the one
#:     model that uses a declared refractive index -- because it never forms a
#:     field, it never reaches the couplers whose ramp is `n = 1`.
#:
#:     One operation with three models does **not** mean three interchangeable
#:     models. This one has a validity domain the other two do not, bounded by
#:     three signed margins, and crossing any of them is a refusal.
DiffractiveModel = Literal["full_field", "local_patch", "generalized_snell"]

DIFFRACTIVE_MODELS: tuple[DiffractiveModel, ...] = (
    "full_field",
    "local_patch",
    "generalized_snell",
)

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

    #: Refractive index on the **transmitted** side. Read only by
    #: `generalized_snell`, which is the one model that uses a declared index --
    #: it forms no field, so it never reaches the couplers whose transverse ramp
    #: is the `n = 1` form (R09). The incident-side index is the one the
    #: reference surface already declares, so there is no second field for it.
    transmitted_index: float = 1.0

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
        transmitted = float(self.transmitted_index)
        if not (transmitted > 0.0) or not math.isfinite(transmitted):
            raise ContractError(
                "UNIT_NOT_SI",
                f"transmitted_index must be a positive finite refractive index, got "
                f"{self.transmitted_index!r}",
                declaration="transmitted_index",
            )
        object.__setattr__(self, "transmitted_index", transmitted)

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
        transmitted_index: float = 1.0,
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
            transmitted_index=transmitted_index,
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


# ---------------------------------------------------------------------------
# The reduced-order model, and the three margins that bound it
# ---------------------------------------------------------------------------


def propagating_order_margin(
    transverse_wavenumber_sq: Any, *, transmitted_index: float, wavenumber: float
) -> Any:
    """Signed margin of `|k_t^out| < n_t k0` -- predicate 1, and a **hard** limit.

    `> 0` the order propagates, `<= 0` refuses: at `0` the order is grazing, so
    `k_n = 0` and it carries no power along the axis, and below it the order does
    not exist as an outgoing ray at all. Fractional, so it is comparable with the
    other two.
    """
    limit_sq = (float(transmitted_index) * float(wavenumber)) ** 2
    return (limit_sq - np.asarray(transverse_wavenumber_sq, dtype=np.float64)) / limit_sq


def local_gradient_smoothness_margin(
    curvature_rad_per_m2: Any, worst_raw_step_rad: Any, *, transverse_scale_m: float
) -> Any:
    """Signed margin of predicate 2: the local plane-wave picture holds at all.

    **The worse of two sub-checks, because either alone misses a real failure.**

    * *Curvature against the declared transverse scale.* The phase the local
      curvature accumulates over `D` must stay well under one fringe -- bounded at
      `pi` for a symmetric two-sided budget -- for "one locally redirected ray" to
      mean anything.
    * *The estimator's own span against the wrap boundary.* The gradient is a
      centred difference over **two** samples, so its argument is the two-sample
      phase difference and it aliases once that exceeds `pi` -- i.e. once the
      *per-sample* step exceeds `pi/2`. A uniformly undersampled grating aliases
      every tap by the same wrong amount, which the curvature check alone reads as
      perfectly smooth: zero curvature, nonsense gradient.

      **`worst_raw_step_rad` must therefore be `2 x` the largest *adjacent-sample*
      step, not the wrapped two-sample step the estimator returns.** The reference
      implementation used the latter, and it has a hole exactly where it matters:
      for a per-sample step `s` the two-sample step is `wrap(2s)`, which tends to
      **zero** as `s -> pi`. Measured on a linear ramp at 250 nm pitch, the old form
      reported margins of +0.54, +0.74, +0.90 and **+0.98** at per-sample steps of
      2.42, 2.73, 2.99 and 3.11 rad -- rising toward maximum confidence precisely as
      the recovered direction cosine collapsed from its true `+0.27` to `-0.011`.
      The predicate was most certain where the answer was most wrong.

      Correcting it moves the failure one octave down rather than removing it: an
      *adjacent* step is wrapped too, so below two samples per period the check is
      fooled again. See `_local_phase_gradient` -- that residue is the array's
      Nyquist limit, not this predicate's.

    A **grazing** margin of exactly `0` refuses, here and in predicate 1. A margin
    is a distance to a boundary, and a caller sitting on one has no distance left.
    """
    curvature = np.abs(np.asarray(curvature_rad_per_m2, dtype=np.float64))
    accumulated = curvature * float(transverse_scale_m) ** 2
    curvature_margin = (math.pi - accumulated) / math.pi
    step_margin = (
        math.pi - np.abs(np.asarray(worst_raw_step_rad, dtype=np.float64))
    ) / math.pi
    return np.minimum(curvature_margin, step_margin)


def single_order_dominance(
    transmission: Any,
    *,
    sample_pitch_m: tuple[float, float],
    centre_xy_m: tuple[float, float],
    patch_px: int,
    wavelength_m: float,
    target_direction_xy: tuple[float, float],
) -> tuple[float, float]:
    """`(dominance, margin)` -- the fraction of local spectral power in one order.

    Predicate 3. A local window of `exp(i phi)` -- the **phase alone** -- is
    transformed, following the same window-then-transform idiom the patch route
    uses. That is legitimate only because `_local_phase_gradient` separately
    refuses a surface whose modulus is not locally constant: where `|t|` is
    constant over the window, `t` and `exp(i phi)` have the same spectrum up to a
    scale, and the ratio this function returns is identical. Without that refusal
    it would be reading the wrong function -- a Ronchi grating has `arg(t) = 0`
    everywhere and would be reported as perfectly single-order while diffracting
    entirely by amplitude.

    `target_direction_xy` is the requested order's **momentum kick**
    `m grad(phi) / k0`, which is where that order sits in this spectrum. It is
    deliberately *not* the outgoing direction cosine: those coincide only at normal
    incidence in vacuum, and centring on `d_out` would slide the disk off the peak
    by the incident tilt and refuse a tilted plane wave on a perfectly single-order
    grating.

    It is evaluated **once**, at the ensemble centroid and the mean kick, where
    predicates 1 and 2 take a per-ray worst case. "How many orders are there" is
    only answerable relative to a window, and one window at the centroid is what
    this reports; a bundle straddling two structurally different regions is
    described by neither. Made worst-of-windows it would cost one FFT per ray.

    Dominance sums power over a **disk** around the requested order's direction,
    not a single bin. `resolve_pad_px` zero-pads well past `patch_px`, which
    *interpolates* the window's own DTFT onto a much finer grid than its native
    resolution, so reading one interpolated bin would report a slice of an order
    rather than the order. The disk radius is the window's native angular
    resolution `lambda / (patch_px * pitch)`, which is the mainlobe width a
    rectangular window of that size actually has.

    **That radius is also the predicate's resolving power, and the caller sets
    it.** Neighbouring orders are `|grad phi| / k0 = lambda / Lambda` apart, so
    unless `patch_px` is at least about `Lambda / pitch` the disk spans several
    orders and a positive margin says only that power lies within one resolution
    element of the requested direction. On a 16-sample ramp at the default
    5-sample window, `m = -1`, `0` and `+1` are then all reported dominant, and
    only `+1` exists. `_generalized_snell` computes `orders_resolved` from exactly
    this comparison and puts it in the record beside the margin; a caller acting on
    the margin has to read it.

    `margin = 2 * dominance - 1`, so a bare majority is the boundary and full
    concentration is `+1`.
    """
    array = np.asarray(transmission)
    phase_only = np.exp(1j * np.angle(array))
    grid_n = int(array.shape[0])
    pad_px = resolve_pad_px(grid_n=grid_n, patch_px=int(patch_px))

    pitch_y, pitch_x = float(sample_pitch_m[0]), float(sample_pitch_m[1])
    half = int(patch_px) // 2
    ny, nx = array.shape
    centre_row = round(float(centre_xy_m[1]) / pitch_y) + ny // 2
    centre_col = round(float(centre_xy_m[0]) / pitch_x) + nx // 2
    window = np.zeros((int(patch_px), int(patch_px)), dtype=np.complex128)
    top, bottom = centre_row - half, centre_row - half + int(patch_px)
    left, right = centre_col - half, centre_col - half + int(patch_px)
    source_top, source_left = max(top, 0), max(left, 0)
    source_bottom, source_right = min(bottom, ny), min(right, nx)
    if source_top < source_bottom and source_left < source_right:
        window[
            source_top - top : source_bottom - top,
            source_left - left : source_right - left,
        ] = phase_only[source_top:source_bottom, source_left:source_right]

    margin_px = (pad_px - int(patch_px)) // 2
    padded = np.pad(window, margin_px, mode="constant")
    spectrum = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(padded))) / (pad_px * pad_px)

    frequency_y = np.fft.fftshift(np.fft.fftfreq(pad_px, d=pitch_y))
    frequency_x = np.fft.fftshift(np.fft.fftfreq(pad_px, d=pitch_x))
    grid_fx, grid_fy = np.meshgrid(frequency_x, frequency_y)
    direction_x = grid_fx * float(wavelength_m)
    direction_y = grid_fy * float(wavelength_m)
    propagating = (direction_x**2 + direction_y**2) < 1.0

    power = np.abs(spectrum) ** 2
    total = float(power[propagating].sum())
    if total <= 0.0:
        return 0.0, -1.0

    target_u, target_v = float(target_direction_xy[0]), float(target_direction_xy[1])
    resolution = float(wavelength_m) / (
        float(patch_px) * math.sqrt(pitch_y * pitch_x)
    )
    inside = propagating & (
        ((direction_x - target_u) ** 2 + (direction_y - target_v) ** 2) < resolution**2
    )
    dominance = float(power[inside].sum()) / total
    return dominance, 2.0 * dominance - 1.0


#: How much `|t|` may change between adjacent samples, anywhere on the surface,
#: before `generalized_snell` refuses it. A fraction of the peak modulus.
#:
#: This model redirects a ray by `m grad(arg t)` and puts `|t|` into the amplitude:
#: **it is blind to diffraction by amplitude.** A Ronchi grating has `arg(t) = 0`
#: everywhere, so every requested order comes out undeflected while the real
#: grating throws its power into `+-1` -- an invalid case admitted with maximum
#: confidence, and the failure mode this ticket exists to make impossible.
#:
#: 5 % is where an amplitude modulation stops mattering rather than where it
#: becomes convenient: a modulation of depth `eps` puts about `(eps / 4)^2` of the
#: power into each amplitude sideband, so a 5 % per-sample step is 1.6e-4 and a
#: Ronchi grating's 100 % is the entire response. A smooth apodization changes by
#: far less than that between neighbours and is admitted, which is the point: this
#: refuses amplitude *structure*, not envelopes.
#:
#: It is measured over the **whole transmission**, not over the samples the rays
#: happen to land on. The first version of this gate read the gradient estimator's
#: own nine taps, and a bundle sampled at the grating's own period -- the normal
#: input to a ray-side operator, whose sampling has no relationship to the DOE's --
#: put every ray at a bar centre, read a variation of exactly 0, and reproduced the
#: whole failure: an undeflected ray from a grating throwing 40 % of its power into
#: `+-1`, with every margin green. A gate whose verdict depends on where the rays
#: sit is not a statement about the surface.
#:
#: Normalizing by the peak rather than by the local value is deliberate: in the
#: tail of a smooth apodization the *relative* step is large while the absolute one
#: is negligible, and this must not refuse an envelope for being dim.
MODULUS_LOCALITY_TOLERANCE = 0.05


def _local_phase_gradient(
    transmission: Any, *, sample_pitch_m: tuple[float, float], rows: Any, cols: Any
) -> tuple[Any, Any, Any, Any, Any]:
    """`(phase, grad_y, grad_x, curvature, worst_raw_step)` at each ray's own sample.

    The gradient is estimated from the **complex** transmission,
    `d phi / du ~= angle(t[+1] conj(t[-1])) / (2 du)`, rather than by unwrapping
    and differencing `angle(t)`. The two agree wherever unwrapping would succeed,
    but the complex form has no unwrap step to fail and is exact to round-off for a
    genuine phase ramp at any pitch, because `angle` of a unit-modulus product is
    exact. It returns the wrong answer -- aliased by a multiple of
    `2 pi / (2 pitch)` -- when the true step between two samples exceeds `pi`, and
    the smoothness predicate exists to catch that.

    **The guarantee is bounded, and the bound is the array's own Nyquist limit.**
    Every quantity here is built from `angle(...)`, which is itself wrapped, so a
    *per-sample* step approaching `2 pi` reads as a small step again: below two
    samples per period the predicate admits the surface and the recovered gradient
    is wrong. That is irreducible rather than a hole to close -- an array sampled
    at 1.2 samples per period does not contain the surface it came from, and no
    estimator reading only that array can say so. It is the caller's sampling that
    has to be right; `tests/physics/test_generalized_snell.py` pins the admitted
    region so it is documented rather than discovered.

    The ray's **own** phase is read at its true nearest sample, clipped only to stay
    in the array -- never at the interior-clamped stencil centre. The stencil needs
    an interior centre to keep every tap in bounds, but using that clamped location
    for the ray's own phase would read a different pixel than its amplitude does,
    a silent mismatch for any ray within two samples of an edge.
    """
    array = np.asarray(transmission)
    ny, nx = array.shape
    pitch_y, pitch_x = float(sample_pitch_m[0]), float(sample_pitch_m[1])
    phase = np.angle(array[np.clip(rows, 0, ny - 1), np.clip(cols, 0, nx - 1)])
    centre_row = np.clip(rows, 2, ny - 3)
    centre_col = np.clip(cols, 2, nx - 3)

    def step(plus: Any, minus: Any) -> Any:
        return np.angle(plus * np.conj(minus))

    step_y = step(array[centre_row + 1, centre_col], array[centre_row - 1, centre_col])
    step_x = step(array[centre_row, centre_col + 1], array[centre_row, centre_col - 1])
    gradient_y = step_y / (2.0 * pitch_y)
    gradient_x = step_x / (2.0 * pitch_x)

    forward_y = step(array[centre_row + 2, centre_col], array[centre_row, centre_col])
    backward_y = step(array[centre_row, centre_col], array[centre_row - 2, centre_col])
    curvature_y = (forward_y - backward_y) / (4.0 * pitch_y**2)
    forward_x = step(array[centre_row, centre_col + 2], array[centre_row, centre_col])
    backward_x = step(array[centre_row, centre_col], array[centre_row, centre_col - 2])
    curvature_x = (forward_x - backward_x) / (4.0 * pitch_x**2)

    curvature = np.sqrt(curvature_y**2 + curvature_x**2)

    # **Adjacent-sample** steps, doubled -- not the two-sample steps above. The
    # gradient is a centred difference over two samples, so what must stay under
    # `pi` is the *unwrapped* two-sample difference, and `2 x` an adjacent step is
    # that quantity whenever the adjacent step is itself unaliased. Reading the
    # two-sample step instead measures `wrap(2s)`, which tends to zero as the
    # per-sample step approaches `pi` -- see `local_gradient_smoothness_margin`.
    adjacent = [
        step(array[centre_row + offset + 1, centre_col], array[centre_row + offset, centre_col])
        for offset in (-2, -1, 0, 1)
    ] + [
        step(array[centre_row, centre_col + offset + 1], array[centre_row, centre_col + offset])
        for offset in (-2, -1, 0, 1)
    ]
    worst_raw_step = 2.0 * np.maximum.reduce([np.abs(value) for value in adjacent])

    return phase, gradient_y, gradient_x, curvature, worst_raw_step


def _generalized_snell(
    rays: RayBundle, *, surface: DiffractiveSurface, order: int, patch_px: int
) -> tuple[RayBundle, dict[str, Any]]:
    """The per-ray local grating equation. No field is formed.

        k_t^out = n_i k0 d_t^in + m grad_t(phi)(x, y)
        k_n^out = sqrt( (n_t k0)^2 - |k_t^out|^2 )
        opl^out = opl^in + m phi(x, y) / k0

    `phi` is read off the same complex `transmission` the other two models use, so
    a caller declares one surface whichever model computes the interaction.

    **`m phi`, not `phi`**, and two code-independent arguments say which is right.
    `exp(i(-1) phi)` and `exp(i(+1)(-phi))` are the same complex factor, so
    `(order=-1, t)` and `(order=+1, conj(t))` must return the same bundle -- with
    `phi` alone they returned the same *direction* and opposite optical paths,
    which is a contradiction rather than a tolerance. And `order=0` is the
    undiffracted transmission, which picks up no ramp at all; with `phi` alone it
    was handed the whole ramp phase on an undeflected ray. That was CHE-148's
    finding, and it is reproduced here rather than re-derived.
    """
    amplitude_in, optical_path_in = rays.require_coherent()
    if rays.reference_surface != surface.reference_surface:
        raise ContractError(
            "FRAME_MISMATCH",
            f"the bundle is declared on {rays.reference_surface!r} but the surface is at "
            f"{surface.reference_surface!r}. This model interacts, it does not propagate, "
            "and it reads the incident medium index off the surface -- so a bundle "
            "declared elsewhere would be given the wrong tangential momentum and then "
            "silently relocated. The other two models get this check from "
            "`couplers.ray_to_scalar`; this one forms no field and so must make it "
            "itself.",
            declaration="reference_surface",
            remedy=(
                "Advance the bundle with `operators.propagate_rays(rays, to=...)` first, "
                "or pass the surface the bundle already declares."
            ),
        )
    if not math.isinf(surface.radius_m):
        raise ContractError(
            "MISSING_DECLARATION",
            f"model='generalized_snell' cannot be applied to a substrate of radius "
            f"{surface.radius_m!r} m: the tangential-momentum equation is evaluated in "
            "the surface's own local tangent frame, and on a curved substrate that "
            "frame is position-dependent. This model has no way to accept one declared "
            "per ray.",
            declaration="radius_m",
            remedy=(
                "Use a planar substrate. A per-ray local frame is a future "
                "declaration, not an inferred one."
            ),
        )
    if abs(order) > 1:
        raise ContractError(
            "MISSING_DECLARATION",
            f"order m={order} is outside this model's domain. It reads one *local* phase "
            "gradient, so the surface it sees is locally linear and has exactly one "
            "fundamental: higher orders come from harmonic content, which is structure "
            "the smoothness predicate refuses before this point. Searching the "
            "fundamental's spectrum at m times its kick finds only whichever real order "
            "happens to fall in the window, which is how m=+2 was admitted and m=-2 "
            "refused on one blazed ramp.",
            declaration="order",
            remedy=(
                "Use model='full_field' or 'local_patch' for a surface with harmonic "
                "content: they form a field and emit every order at once."
            ),
        )
    if patch_px <= 0 or patch_px % 2 == 0:
        raise ContractError(
            "SHAPE_MISMATCH",
            f"patch_px={patch_px} must be positive and odd: it has no centre sample to "
            "declare a transverse scale or a window from.",
            declaration="patch_px",
        )

    transmission = np.asarray(surface.transmission)
    incident_index = float(surface.reference_surface.medium_index)
    transmitted_index = float(surface.transmitted_index)
    wavenumber = rays.wavenumber
    positions_xy = np.asarray(rays.positions_m)[:, :2].astype(np.float64)
    directions_xy = np.asarray(rays.directions)[:, :2].astype(np.float64)

    ny, nx = transmission.shape
    pitch_y, pitch_x = surface.sample_pitch_m
    rows = np.round(positions_xy[:, 1] / pitch_y).astype(np.int64) + ny // 2
    cols = np.round(positions_xy[:, 0] / pitch_x).astype(np.int64) + nx // 2
    outside = (rows < 0) | (rows >= ny) | (cols < 0) | (cols >= nx)
    if bool(np.any(outside)):
        # Clamping them to the edge sample is the tempting answer and it silently
        # extends the DOE to infinity: every off-aperture ray would be handed the
        # edge gradient and the edge modulus and come out deflected by a structure
        # that is not there. Zeroing them is the other answer, and it is equally a
        # guess -- a DOE patch in an open window transmits outside its own extent.
        # The surface does not declare which, so this refuses rather than picks.
        raise ContractError(
            "SHAPE_MISMATCH",
            f"{int(np.count_nonzero(outside))} of {rows.size} rays fall outside the "
            f"transmission grid's {ny} x {nx} sampled extent. The surface declares no "
            "behaviour beyond it: continuing the edge sample would deflect those rays "
            "by structure that is not there, and blocking them would assume an opaque "
            "mount this surface never stated.",
            declaration="positions_m",
            remedy=(
                "Restrict the bundle to the surface's extent, or sample the "
                "transmission over the full extent the bundle covers."
            ),
        )

    # Every estimate below reads `arg(t)`, so a surface that diffracts by `|t|` is
    # invisible to all of them. Checked over the whole surface -- see
    # `MODULUS_LOCALITY_TOLERANCE` for why not over the rays' own samples.
    moduli = np.abs(transmission)
    peak = float(np.max(moduli))
    worst_modulus_variation = (
        max(float(np.max(np.abs(np.diff(moduli, axis=axis)))) for axis in (0, 1)) / peak
        if peak > 0.0
        else 0.0
    )
    if worst_modulus_variation > MODULUS_LOCALITY_TOLERANCE:
        raise ContractError(
            "MISSING_DECLARATION",
            f"the transmission modulus changes by {worst_modulus_variation:.3f} of its "
            f"peak between adjacent samples somewhere on this surface, above the "
            f"{MODULUS_LOCALITY_TOLERANCE} this model admits. It redirects each ray by "
            "m grad(arg t) and puts |t| into the amplitude, so it is blind to "
            "diffraction by amplitude: a surface with amplitude structure -- a Ronchi "
            "grating, or a hard aperture edge -- would come back undeflected, with every "
            "margin reporting full confidence.",
            declaration="transmission",
            remedy=(
                "Use model='full_field' or 'local_patch', which transform the complex "
                "transmission and so see amplitude and phase gratings alike."
            ),
        )

    phase, gradient_y, gradient_x, curvature, worst_raw_step = _local_phase_gradient(
        transmission, sample_pitch_m=(pitch_y, pitch_x), rows=rows, cols=cols
    )

    # Predicate 2 first: if the gradient itself cannot be trusted, the direction it
    # points is not a question worth asking.
    transverse_scale_m = float(patch_px) * math.sqrt(pitch_y * pitch_x)
    smoothness = local_gradient_smoothness_margin(
        curvature, worst_raw_step, transverse_scale_m=transverse_scale_m
    )
    worst_smoothness = float(np.min(smoothness))
    if worst_smoothness <= 0.0:
        raise ContractError(
            "MISSING_DECLARATION",
            "the local phase-gradient estimate is not reliable at the declared "
            f"transverse scale (worst signed margin {worst_smoothness:.3e} against "
            "LOCAL_GRADIENT_SMOOTHNESS): the phase varies too fast, relative to the "
            "sample pitch and the declared patch scale, for a single local plane wave "
            "to describe the response here.",
            declaration="patch_px",
            remedy=(
                "Sample the surface's phase on a finer grid, or declare a smaller "
                "patch_px. This is not the evanescent-order refusal: the gradient "
                "itself cannot be trusted yet, independent of where it points."
            ),
        )

    out_y = incident_index * wavenumber * directions_xy[:, 1] + float(order) * gradient_y
    out_x = incident_index * wavenumber * directions_xy[:, 0] + float(order) * gradient_x
    transverse_sq = out_y**2 + out_x**2

    order_margin = propagating_order_margin(
        transverse_sq, transmitted_index=transmitted_index, wavenumber=wavenumber
    )
    worst_order_margin = float(np.min(order_margin))
    if worst_order_margin <= 0.0:
        raise ContractError(
            "MISSING_DECLARATION",
            "the requested diffraction order is evanescent for at least one ray "
            f"(worst signed margin {worst_order_margin:.3e} against "
            f"PROPAGATING_ORDER_EXISTS; 0 is grazing and refuses too): m={order} has no "
            f"outgoing propagating "
            f"direction at this position for n_incident={incident_index}, "
            f"n_transmitted={transmitted_index}.",
            declaration="order",
            remedy=(
                "Request a different order, or accept that this configuration has no "
                "propagating solution here. Returning a normalized nonsense direction "
                "would be worse than refusing."
            ),
        )

    transmitted_wavenumber = transmitted_index * wavenumber
    axial = np.sqrt(np.clip(transmitted_wavenumber**2 - transverse_sq, 0.0, None))
    directions_out = np.column_stack(
        [out_x, out_y, axial]
    ) / transmitted_wavenumber

    local = transmission[np.clip(rows, 0, ny - 1), np.clip(cols, 0, nx - 1)]
    outgoing = RayBundle(
        positions_m=np.column_stack(
            [
                positions_xy[:, 0],
                positions_xy[:, 1],
                np.full(positions_xy.shape[0], surface.reference_surface.z_m),
            ]
        ),
        directions=directions_out,
        wavelength_m=rays.wavelength_m,
        reference_surface=surface.reference_surface,
        frame=rays.frame,
        amplitude=np.asarray(amplitude_in) * np.abs(local),
        optical_path_m=np.asarray(optical_path_in) + float(order) * phase / wavenumber,
        optical_path_reference=(
            f"{rays.optical_path_reference}; plus this generalized-Snell surface's "
            f"local phase m phi(x, y) / k0 at {surface.reference_surface.name!r}, "
            f"order m={order}"
        ),
        measure_weight=rays.measure_weight,
        measure_kind=rays.measure_kind,
    )

    # Whether predicate 3 can *separate* the requested order from its neighbours at
    # this window. The disk is one window-resolution wide and neighbouring orders are
    # `|grad phi| / k0` apart, so below `patch_px ~ Lambda / pitch` the disk covers
    # several orders at once and a positive margin means only "power lies within one
    # resolution element of the requested direction" -- on a 16-sample ramp at the
    # default window, m = -1, 0 and +1 are then all reported dominant, and only one of
    # them exists. Reported rather than refused, because a caller can fix it by
    # declaring a wider `patch_px`, and because refusing would also refuse the case
    # where the aperture is genuinely shorter than one period and the surface has no
    # order structure to resolve. **A caller acting on the margin must read this flag
    # with it.**
    disk_radius = float(rays.wavelength_m) / (float(patch_px) * math.sqrt(pitch_y * pitch_x))
    order_spacing = (
        float(np.hypot(np.mean(gradient_x), np.mean(gradient_y))) / wavenumber
    )
    dominance, dominance_margin = single_order_dominance(
        transmission,
        sample_pitch_m=(pitch_y, pitch_x),
        centre_xy_m=(float(np.mean(positions_xy[:, 0])), float(np.mean(positions_xy[:, 1]))),
        patch_px=patch_px,
        wavelength_m=rays.wavelength_m,
        # The order's momentum kick, not its outgoing direction cosine: the
        # spectrum this searches is the kick spectrum, and the two differ by the
        # incident tilt and by n_t whenever either is not the trivial one.
        target_direction_xy=(
            float(order) * float(np.mean(gradient_x)) / wavenumber,
            float(order) * float(np.mean(gradient_y)) / wavenumber,
        ),
    )
    if dominance_margin <= 0.0:
        raise ContractError(
            "MISSING_DECLARATION",
            f"no more than half the local spectral power is in the requested order "
            f"(dominance {dominance:.3f}, signed margin {dominance_margin:.3e} against "
            "SINGLE_ORDER_DOMINANCE). This model returns **one** outgoing ray per "
            "incident ray, so applying it where the surface emits several orders "
            "returns one direction for a response that has more than one.",
            declaration="model",
            remedy=(
                "Use model='full_field' or 'local_patch', which form a field and emit "
                "every order. Or declare a smaller patch_px if the extra orders are an "
                "artifact of a window wider than the structure it is measuring -- but "
                "check that the smoothness margin survives it."
            ),
        )

    return outgoing, {
        "order": int(order),
        "incident_index": incident_index,
        "transmitted_index": transmitted_index,
        "patch_px": int(patch_px),
        "transverse_scale_m": transverse_scale_m,
        "worst_modulus_variation": worst_modulus_variation,
        "dominance_disk_radius": disk_radius,
        "order_spacing_direction": order_spacing,
        "orders_resolved": bool(order_spacing > disk_radius),
        # All three signed, so a caller sees how close it is to a boundary rather
        # than only whether it crossed one. The whole value of a reduced-order model
        # is that its domain is bounded and the bounds are visible.
        "propagating_order_margin": worst_order_margin,
        "local_gradient_smoothness_margin": worst_smoothness,
        "single_order_dominance": dominance,
        "single_order_dominance_margin": dominance_margin,
        "opl_convention": (
            "additive: outgoing OPL = incident OPL + m phi(x, y) / k0 at the ray's own "
            "transverse position; the amplitude carries only |t(x, y)|, which is **not** "
            "scaled by the diffraction efficiency -- an admitted surface at dominance "
            "0.51 still emits the full incident amplitude into the requested order, so "
            "read `single_order_dominance` before trusting a radiometric total. The "
            "order "
            "factor m is the same one multiplying grad(phi) in the momentum equation, "
            "because both come from differentiating or evaluating the m-th order's "
            "local plane-wave factor exp(i m phi) -- CHE-148"
        ),
    }


def diffractive_surface(
    rays: RayBundle,
    *,
    surface: DiffractiveSurface,
    model: DiffractiveModel = "full_field",
    reconstruction: Reconstruction = Reconstruction.DIRECT,
    kspace_oversample: float = DEFAULT_KSPACE_OVERSAMPLE,
    kspace_grid_shape: tuple[int, int] | None = None,
    allow_gain: bool = False,
    order: int = 1,
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
        Which granularity computes it. Named, never inferred. The three are not
        interchangeable -- see `DiffractiveModel`.
    reconstruction, kspace_oversample, kspace_grid_shape
        Which realization of the wavelet sum reconstructs the incident field,
        passed to `couplers.ray_to_scalar`. Exposed because it is the **cost**
        knob and this operation is where the cost lands: `DIRECT` is
        `O(N_rays x ny x nx)`, so 512^2 incident rays onto a 512^2 surface is
        about 7e10 term evaluations. `KSPACE` is `O(N_rays + K log K)` and carries
        R07.2's declared interpolation budget -- read it before choosing, because
        at the default oversampling the two routes differ by tens of percent.
    order
        The diffraction order `m`, for `generalized_snell` only. Declared with the
        physically usual default rather than inferred: a caller who wants another
        order says so.
    patch_px
        The local window, in samples, and odd. Under `local_patch` it is the
        decomposition tile; under `generalized_snell` it is the most load-bearing
        knob on the call, because it sets **both** the transverse scale the
        smoothness margin is measured against and the width of the disk dominance
        integrates over -- widen it and the model resolves more orders but trusts
        the local plane-wave picture over a longer distance. It defaults to 5 there
        (a five-sample stencil, the smallest window with a centre and two taps each
        side) and to the tiling `local_patch` declares.

        **The default does not resolve orders on a grating coarser than 5 samples**,
        and predicate 3 is uninformative until `patch_px` reaches roughly the local
        period in samples. The record's `orders_resolved` is that comparison, made
        explicit rather than left to the reader.
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

    if model != "generalized_snell" and order != 1:
        raise ContractError(
            "MISSING_DECLARATION",
            f"order={order!r} was supplied with model={model!r}, which emits every order "
            "the surface has rather than a selected one. Only 'generalized_snell' takes "
            "a diffraction order, and a parameter the named model ignores is a caller "
            "believing something the run did not do.",
            declaration="order",
        )

    if model == "generalized_snell":
        # Straight to the reduced-order model: it forms no field, so none of the
        # steps below apply -- and because it never reaches a coupler, it is the
        # one model that may run in a medium (R09's ramp refusal does not bind it).
        outgoing, snell = _generalized_snell(
            rays,
            surface=surface,
            order=order,
            patch_px=5 if patch_px is None else patch_px,
        )
        return outgoing, {
            "model": model,
            "interior_field_validity": [],
            "reconstruction": None,
            "sampling": None,
            "generalized_snell": snell,
        }

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
