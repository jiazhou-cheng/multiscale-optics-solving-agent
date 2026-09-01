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
from operators.transmission import complex_transmission
from representations import ContractError, RayBundle, ReferenceSurface

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
DiffractiveModel = Literal["full_field"]

DIFFRACTIVE_MODELS: tuple[DiffractiveModel, ...] = ("full_field",)


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

    What is *not* here yet: `substrate` and `radius_m`. They are the curvature
    declaration, and the only model that reads them is the local-patch route
    (R10.3) whose curvature envelope needs them. `full_field` requires one common
    plane, which is what a planar substrate is, so a substrate field on this
    record today would have exactly one legal value -- and a field with one legal
    value is a default pretending to be a declaration.
    """

    #: `(ny, nx)` complex amplitude transmission.
    transmission: Any

    #: `(dy, dx)` sample spacing in metres.
    sample_pitch_m: tuple[float, float]

    #: Where the transmission lives. The incident bundle must already be on it.
    reference_surface: ReferenceSurface

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
        object.__setattr__(self, "transmission", transmission)
        object.__setattr__(self, "sample_pitch_m", pitch)

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
    # leave the surface.
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

    return outgoing, {
        "model": model,
        # Verbatim, because the interior field's own declaration is the honest
        # statement of what the reconstruction could and could not carry.
        "interior_field_validity": interior,
        "reconstruction": reconstruction_record.as_dict(),
        "sampling": sampling.as_dict(),
    }
