"""Where a representation lives, and how its phase is written.

CHE-174 (R02.2). Two data types, added only because both the ray representation
and the scalar-field representation have to *declare* the geometry they are
expressed in before either can be handed to anything else. Nothing else lives
here: this module is the smallest thing R02.3 and R02.4 can both embed.

CHE-175/CHE-176 added `PHASOR` and `SPATIAL_FACTOR` beside the four geometric
constants. They are not geometry, and the module is named for the majority
rather than for them; they are here because both representations validate
against them and neither may import the other. The phasor sign is what decides
whether a converging wavefront reads as converging, so it is a frozen constant
with an equality check, exactly like the axis order.

The conventions are reused from the reference implementation verbatim, not
re-derived. `pre-rewrite-2026-08-30:src/core/boundary.py:83-86` is where they were
pinned and `:456-535` is the pair of types this replaces:

* right-handed Cartesian, propagation along `+z`;
* field arrays are indexed `(y, x)`;
* array index `n // 2` is coordinate zero on each spatial axis;
* SI at the boundary -- metres, radians, seconds.

**The reference surface is planar, in content as well as in name.** No consumer
in the new tree needs a curved reference, and there is no curved-surface subtype,
no abstract surface base and no separate `ReferencePlane`/`ReferenceSurface`
pair. The name is the general one because a curved reference is the plausible
extension and renaming a serialized field later is worse than naming it well now;
the *content* stays planar until something actually needs curvature.
`docs/architecture_principles.md` bans admitting generality nobody uses, and a
`curvature` field that is always zero is exactly that.

Failure vocabulary
------------------
`ContractError` from `representations.contracts`, with a code from
`CONTRACT_CODES`. R02.2 raised plain `ValueError`s here and said they would move
onto the structured diagnostic once one existed and a caller had to branch on it;
R02.3 is that ticket, and leaving half a package raising a code and half raising
prose would mean a coupler's `except ContractError` silently missed a bad frame.
`ContractError` subclasses `ValueError`, so nothing that caught the old shape
stopped working.

The exception is the two lookup helpers below. `field_axis_index("z")` and
`origin_index(0)` are bad *arguments*, not bad *declarations*: no boundary
artifact is being refused, so they stay plain `ValueError`s and cost no contract
code.

This module imports no backend. A frame is not a numeric policy, so it reaches
`numerics` only through `contracts`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from representations.contracts import ContractError

__all__ = [
    "AXIS_ORDER",
    "HANDEDNESS",
    "ORIGIN_RULE",
    "PHASOR",
    "PROPAGATION_AXIS",
    "SPATIAL_FACTOR",
    "Frame",
    "ReferenceSurface",
]

#: Field arrays are `(y, x)`. Row-major with the *slow* axis vertical, so
#: `u[i, j]` is row `i` (y) and column `j` (x).
AXIS_ORDER = "(y, x)"

#: Right-handed Cartesian. With `+z` the propagation axis this fixes the sign of
#: every cross product and every rotation in the tree.
HANDEDNESS = "right-handed"

#: Centring. `n // 2` is the *upper* centre sample for even `n`, which matches
#: `numpy.fft.fftshift` and is why it is stated rather than assumed: `(n - 1) / 2`
#: is the other defensible choice and differs by half a sample, a linear phase
#: ramp across the pupil that no rotationally symmetric test case would show.
ORIGIN_RULE = "array index n//2 is coordinate zero"

#: Light travels toward `+z`, which is the axis the spatial factor below is
#: measured along.
PROPAGATION_AXIS = "+z"

#: The time convention. Every complex amplitude in the tree is the coefficient of
#: `exp(-i omega t)`.
#:
#: Frozen and equality-checked rather than inferred, because the other convention
#: is the complex conjugate of this one and the two are indistinguishable in any
#: intensity: `|U|^2` is identical under conjugation, so a phasor mismatch is
#: invisible right up to the point where a converging wavefront propagates the
#: wrong way. Optiland, Chromatix and this repository's couplers each have to
#: agree, and the agreement has to be checkable.
PHASOR = "exp(-i omega t)"

#: The spatial factor that `PHASOR` and `PROPAGATION_AXIS` together imply: with
#: `exp(-i omega t)` and propagation toward `+z`, forward propagation multiplies
#: by `exp(+i k z)`. Stated separately because it is the form the couplers write,
#: and deriving it at each call site is how a sign gets flipped.
SPATIAL_FACTOR = "exp(+i k z)"

#: Absolute tolerance on `|n| - 1` for a declared unit normal.
#:
#: Reused verbatim from `pre-rewrite-2026-08-30:src/core/boundary.py:99`, where it
#: is the float64 direction-norm floor: about 4e6 times looser than float64
#: round-off, a legacy allowance that catches a normal someone forgot to normalize
#: without failing one that a rotation composed. Normals here are Python floats,
#: so the dtype-dependent widening that file also carries (`64 * eps`, for
#: float32 ray directions) has nothing to apply to and is not reproduced. R02.3
#: owns that one, because it owns the array of directions it was derived for.
_UNIT_NORM_TOLERANCE = 1e-9

#: Each frame field, the one value it may take, and why getting it wrong is
#: silent. Checked as a table because the four are one invariant, not four
#: independent settings -- see the class docstring.
_FRAME_INVARIANT: tuple[tuple[str, str, str], ...] = (
    (
        "axis_order",
        AXIS_ORDER,
        "a transposed field array is invisible in any rotationally symmetric test case",
    ),
    (
        "handedness",
        HANDEDNESS,
        "a left-handed frame mirrors the wavefront, which reads as a sign error in the "
        "phase rather than as a geometry error",
    ),
    (
        "origin_rule",
        ORIGIN_RULE,
        "a half-sample shift in the origin is a linear phase ramp across the pupil, i.e. "
        "a tilt that looks like a decentred system",
    ),
    (
        "propagation_axis",
        PROPAGATION_AXIS,
        "propagation along another axis flips which component of a direction cosine is "
        "the obliquity factor",
    ),
)


@dataclass(frozen=True)
class Frame:
    """The coordinate convention an artifact is expressed in.

    A class rather than four loose strings (rule 1: a shared invariant across
    several fields). Axis order, handedness, origin rule and propagation axis are
    **one** invariant, not four independent settings: each of the four fixes part
    of the same mapping from array indices to physical directions, and getting any
    one of them wrong silently mirrors, transposes or shifts a wavefront rather
    than raising. Carried separately they would be four defaults that four call
    sites could disagree about, and the disagreement would show up as a phase
    error attributed to the physics.

    All four are validated at construction. The reference implementation checked
    only handedness and propagation axis eagerly and left `axis_order` to a
    `require_field_axis_order()` call the consumer had to remember; a declaration
    that is only checked when someone asks is a declaration that can be wrong for
    as long as nobody asks.
    """

    axis_order: str = AXIS_ORDER
    handedness: str = HANDEDNESS
    origin_rule: str = ORIGIN_RULE
    propagation_axis: str = PROPAGATION_AXIS

    def __post_init__(self) -> None:
        for name, expected, why in _FRAME_INVARIANT:
            value = getattr(self, name)
            if value != expected:
                raise ContractError(
                    "FRAME_MISMATCH",
                    f"Frame.{name} must be {expected!r}, got {value!r}. The new tree "
                    f"supports one frame convention; {why}.",
                    declaration=f"Frame.{name}",
                    remedy=(
                        "Convert the data at the boundary that produced it; do not "
                        "re-declare the frame."
                    ),
                )

    @property
    def field_axes(self) -> tuple[str, ...]:
        """The spatial axes of a field array, outermost first.

        Parsed from `axis_order` rather than written out, so the string and the
        behaviour cannot disagree: there is one statement of the order and both
        the declaration and `field_axis_index` read it.
        """
        return tuple(part.strip() for part in self.axis_order.strip("()").split(","))

    def field_axis_index(self, label: str) -> int:
        """The array axis holding `label`, e.g. `"y"` -> 0 under `(y, x)`.

        Use this instead of writing `shape[0]` for the y-extent. The two agree
        today, and the point is that they go on agreeing if the order is ever
        argued into changing.
        """
        axes = self.field_axes
        if label not in axes:
            raise ValueError(f"{label!r} is not a spatial axis of this frame; it declares {axes}.")
        return axes.index(label)

    @property
    def propagation_normal(self) -> tuple[float, float, float]:
        """`propagation_axis` as a unit vector, e.g. `"+z"` -> `(0.0, 0.0, 1.0)`.

        Parsed from the declaration for the same reason `field_axes` is: there is
        one statement of the axis and everything reads it, so the string and the
        vector cannot disagree. Added by CHE-185 (R07.1), which needs to refuse a
        reference surface that is not perpendicular to the propagation axis --
        writing `(0.0, 0.0, 1.0)` at that call site would be a second, silent
        statement of this convention.
        """
        sign, axis = self.propagation_axis[0], self.propagation_axis[1:]
        components = [0.0, 0.0, 0.0]
        components["xyz".index(axis)] = 1.0 if sign == "+" else -1.0
        return (components[0], components[1], components[2])

    def origin_index(self, count: int) -> int:
        """The index of coordinate zero on a spatial axis of `count` samples.

        The one implementation of `ORIGIN_RULE`. A grid builder calls this rather
        than writing `n // 2`, so the rule is a function with a test and not a
        convention repeated in several places that can drift apart.
        """
        if count < 1:
            raise ValueError(f"a spatial axis needs at least one sample, got {count!r}")
        return count // 2


@dataclass(frozen=True)
class ReferenceSurface:
    """The named planar surface a representation is declared on, in SI.

    A class rather than three loose arguments (rule 1: a shared invariant across
    several fields). The axial coordinate, the unit normal and the refractive
    index of the medium are validated *together* because they are only meaningful
    together: an optical path length between two surfaces is `n * s` projected
    onto the normal, so a plausible `z_m` beside a non-unit normal, or beside an
    index nobody set, produces an OPL that is wrong by a factor no downstream
    check can attribute back here. Every artifact crossing a boundary declares
    one.

    Planar. `normal` exists because the `<n, d>` projection factor needs a normal
    to project onto and because a tilted reference plane is expressible, not
    because a curved surface is coming.

    `medium_index` has **no default**. The reference implementation's coupler
    read it from the prescription with the note "never assumed to be 1"
    (`pre-rewrite-2026-08-30:src/couplers/handoff.py:367-392`), and a default of
    1.0 here would put that assumption back one layer down, where it would be
    invisible to the caller that meant to state it.
    """

    #: What this surface *is* -- "exit_pupil", "image_surface", "sensor". Free
    #: text: which names are meaningful is a property of the problem being solved,
    #: and the couplers that match a produced surface against an expected one land
    #: in R07/R08 with the vocabulary they need.
    name: str

    #: Axial coordinate along `+z`, in metres, in the frame the artifact declares.
    z_m: float

    #: Real refractive index of the medium the surface sits in. Dimensionless and
    #: positive; absorption is not modelled here.
    medium_index: float

    #: Unit surface normal `(x, y, z)`. Defaults to the propagation axis.
    normal: tuple[float, float, float] = (0.0, 0.0, 1.0)

    def __post_init__(self) -> None:
        if not self.name:
            raise ContractError(
                "MISSING_DECLARATION",
                "ReferenceSurface.name is empty. A surface has to be named for a "
                "consumer to check that it is the surface it expected, which is the "
                "difference between a defocus and a whole pupil-to-focus distance.",
                declaration="ReferenceSurface.name",
            )

        z_m = float(self.z_m)
        if not math.isfinite(z_m):
            raise ContractError(
                "NON_FINITE",
                f"ReferenceSurface.z_m must be a finite coordinate in metres, got {self.z_m!r}",
                declaration="ReferenceSurface.z_m",
            )
        object.__setattr__(self, "z_m", z_m)

        medium_index = float(self.medium_index)
        if not math.isfinite(medium_index) or medium_index <= 0.0:
            raise ContractError(
                "UNIT_NOT_SI",
                f"ReferenceSurface.medium_index must be positive and finite, got "
                f"{self.medium_index!r}. It is the real index of the medium the surface "
                "sits in, read from the prescription and never assumed to be 1.",
                declaration="ReferenceSurface.medium_index",
            )
        object.__setattr__(self, "medium_index", medium_index)

        components = tuple(float(value) for value in self.normal)
        if len(components) != 3:
            raise ContractError(
                "SHAPE_MISMATCH",
                f"ReferenceSurface.normal must be a 3-vector, got {len(components)} "
                f"component(s): {self.normal!r}",
                declaration="ReferenceSurface.normal",
            )
        if not all(math.isfinite(value) for value in components):
            raise ContractError(
                "NON_FINITE",
                f"ReferenceSurface.normal is not finite: {self.normal!r}",
                declaration="ReferenceSurface.normal",
            )
        norm = math.sqrt(sum(value * value for value in components))
        if abs(norm - 1.0) > _UNIT_NORM_TOLERANCE:
            raise ContractError(
                "NON_UNIT_DIRECTION",
                f"ReferenceSurface.normal must be a unit vector, |n| = {norm!r} "
                f"(tolerance {_UNIT_NORM_TOLERANCE:g}). The normal is a direction, so its "
                "length carries no information; a length that is not 1 means it was never "
                "normalized, and it would scale the <n, d> projection that the optical "
                "path is computed with.",
                declaration="ReferenceSurface.normal",
            )
        object.__setattr__(self, "normal", components)
