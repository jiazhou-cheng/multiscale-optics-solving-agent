"""A spot diagram derived from a `RayBundle`: `RayBundle -> SpotResult`.

CHE-226 (R16). The project-owned half of two deliberately separate paths, and the
separation is the point:

* **a `RayBundle`-derived measurement consumes rays exactly as supplied**, subject
  to explicit provenance gates. That is this module. Its only input is a bundle:
  no `OpticalSetup`, no `SourceSpec`, no field, no conjugate, no backend object,
  and nothing here can generate a ray;
* **a native analysis generates its own rays** from a source description, which is
  `backends/optiland/analysis.py`.

Both produce a spot. Neither is implemented in terms of the other, and supplied
rays are never routed back through a backend's own spot analysis to be measured --
that route would silently re-generate the rays from a field the caller may not
have, which is the conflation this split exists to prevent.

The measurement plane is the bundle's
--------------------------------------
`rays.reference_surface`, and there is no `surface=` argument. A `RayBundle`
declares exactly *one* reference surface -- that is the representation's contract,
not an omission -- so the plane is already chosen by whoever produced the bundle,
and selecting a different one would mean holding rays at several planes at once.
Per-surface ray history is therefore a *sequence* of bundles and remains future
work; when it lands, selecting a surface is choosing which bundle to hand to this
function, not a parameter here.

The gate: an unsplit population, declared and not inferred
----------------------------------------------------------
A geometric spot diagram is a statistic over ray intersections, and it is only a
statistic of *one* light population if each row is one ray of that population.
Ray splitting breaks that: descendants of one incident ray -- across the orders of
a grating, or at a partially reflecting surface -- put several branches in one
array, and an unweighted moment over them measures the branching as much as the
spot. Merging them needs a rule about how the branches combine, and picking one
branch needs a rule about which; neither is this function's to invent.

So `representations.RayBundle.ray_splitting` is required to be `"unsplit"`, and the
other two values are refused with different codes:

* `"split_descendants"` -> `SPLIT_RAYS_NOT_MEASURABLE`. The population is known to
  contain them, and this measurement is the wrong one for it.
* `"undeclared"` -> `RAY_SPLITTING_UNDECLARED`. Nobody said. Refusing is the whole
  reason the field defaults to this value.

**Nothing is inferred numerically.** Not from the amplitude, not from duplicate or
near-duplicate coordinates, not from the ray count against a launch fan's, not
from anything else. A heuristic here would be a guess about provenance dressed as
a measurement, and it would be wrong in both directions: a split population can
have distinct coordinates everywhere and an unsplit one can have coincident rows.

**The gate is orthogonal to survival.** A ray may be clipped by a surface rim
without having split, and a descendant may survive; `RAY_SPLITTINGS` states this
and neither state is derived from the other. What this measurement does with
survival is the separate declaration below.

The three metric definitions, stated once and carried on the result
------------------------------------------------------------------
`SPOT_METRIC_DEFINITIONS` is the text, `SpotResult` carries it, and these are the
choices it records:

*Which rays.* Rays with `|a|^2 > 0`. This project encodes ray survival in the
amplitude on the supplied-ray path -- `backends.optiland.rays.SUPPLIED_RAY_SURVIVAL_RULE`
keeps the row and zeroes the coefficient -- and drops the row entirely on the
generated path, so one rule reads both: a zero-intensity row is a ray the system
did not deliver. It is also, measured, what the pinned Optiland spot analysis does
(`mask = i_g > 0` in `optiland.analysis.spot_diagram`), so the two paths are not
answering different questions about vignetting.

*Which reference.* The intensity-weighted centroid of those rays. Radii are
measured from it, never from the axis and never from a chief ray: this measurement
has no field, no pupil and no system, so it cannot identify a chief ray, and
inventing one from "the row nearest the axis" would be exactly the kind of
heuristic the gate above refuses. A caller that wants a chief-ray reference is
asking for the native analysis, which has the system.

*Which weighting.* `|a_i|^2`, for the centroid and the RMS radius. The geometric
radius is a maximum over the included rays and is unweighted by construction. With
equal amplitudes -- every ray of an unapodized Optiland launch fan, measured
(`backends/optiland/launch.py`) -- the weighted centroid and RMS reduce exactly to
the unweighted ones, so the choice does not silently move the common case.

*What is deliberately NOT applied: `measure_weight`.* A spot diagram is the
distribution of ray *intersections*, and the sampling measure is a declaration
about how the pupil was discretized rather than a per-ray energy. Folding a
`quadrature_area_m2` in would make the RMS radius depend on the pupil sampling
declaration -- the rim ring of a hexapolar fan carries half a cell -- and folding an
`importance_weight` in would make it an estimator owing a `1/N`. The honest
consequence is stated rather than hidden: under a non-uniform sampling density
this is a *sampling-weighted* moment of the irradiance rather than the irradiance
moment, and the density is the caller's declaration to read off `measure_kind`.

What this module is not
-----------------------
Not a renderer. Every number is a field of `SpotResult`; there is no `view()`,
this project has no plotting dependency, and adding one is a separate decision.
And not validated against a backend: the metrics are checked against closed-form
ray geometry in `tests/physics/test_spot_diagram.py` (a ring of known radius, a
known centroid offset), because a backend's spot code and this one over the same
rays is shared-input characterization and not an independent oracle
(`AGENTS.md`, Scientific Non-Negotiables).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import Any

from numerics import array_state, xp_for
from representations import (
    ContractError,
    Frame,
    RayBundle,
    RaySplitting,
    require_finite,
)

__all__ = [
    "SPOT_INCLUSION_RULE",
    "SPOT_METRIC_DEFINITIONS",
    "SPOT_REFERENCE",
    "SPOT_WEIGHTING",
    "SpotResult",
    "spot_diagram",
]

#: Which rays the metrics are taken over. See the module docstring.
SPOT_INCLUSION_RULE = (
    "rays with |a|^2 > 0: a zero-intensity row is a ray the system did not deliver, "
    "which is how survival is encoded on both traced paths"
)

#: What the radii are measured from.
SPOT_REFERENCE = (
    "the intensity-weighted centroid of the included rays -- not the axis and not a "
    "chief ray, which a bundle with no system in scope cannot identify"
)

#: What weights the moments.
SPOT_WEIGHTING = (
    "intensity |a_i|^2 for the centroid and the RMS radius; the geometric radius is "
    "an unweighted maximum. measure_weight is NOT applied -- it is a sampling "
    "declaration, not a per-ray energy"
)

#: The three metrics, verbatim on every result, so a consumer reading only the
#: artifact can never be unsure which definition produced a number. The reference
#: implementation's spot statistics carried none of this, and "RMS spot radius"
#: names at least four different numbers across the literature and the tools:
#: weighted or not, from the centroid or from a chief ray, over surviving rays or
#: over all of them.
SPOT_METRIC_DEFINITIONS: dict[str, str] = {
    "centroid_m": (
        "centroid = (sum_i w_i x_i, sum_i w_i y_i) / sum_i w_i with w_i = |a_i|^2, "
        "in metres, in the bundle's frame at its declared reference surface"
    ),
    "rms_radius_m": (
        "rms_radius = sqrt(sum_i w_i |r_i - centroid|^2 / sum_i w_i), the "
        "intensity-weighted second moment about the centroid, in metres"
    ),
    "geometric_radius_m": (
        "geometric_radius = max_i |r_i - centroid| over the INCLUDED rays only, in "
        "metres: the smallest circle about the centroid containing every delivered "
        "ray. Unweighted -- a maximum has no weighting -- and an outlier moves it"
    ),
}

_GATE_REMEDY = (
    "This measurement takes rays exactly as supplied and will not decide how "
    "branches combine. Measure each branch as its own bundle, or use an analysis "
    "that models the splitting; do not re-declare the bundle to get past this."
)


@dataclass(frozen=True)
class SpotResult:
    """Ray intersections at one plane, plus the three metrics and their definitions.

    A class on rule 2, the same rule as `PsfResult`: it is the public record a
    consumer reads back, and half of R16's acceptance criteria are statements about
    what it carries -- which rays were included, from which reference, under which
    weighting. The alternative is a free-form mapping, which is the provenance dict
    R02.4 removed from `ScalarField` for exactly this reason.

    **The arrays are row-aligned with the bundle**, including the excluded rays,
    which are present with `intensity == 0.0`. That is deliberate and it is the
    same choice `SUPPLIED_RAY_SURVIVAL_RULE` makes one layer down: a caller can see
    what was dropped and can index back into the bundle it came from. Filtering for
    a scatter plot is `intensity > 0`, one line, at the point where the caller knows
    whether it wants to see the clipped rays.
    """

    #: `(N,)` transverse coordinates at the measurement plane, in metres, in the
    #: declared frame. Column 0 and column 1 of `rays.positions_m`, untouched: this
    #: measurement resamples nothing and transforms no coordinates.
    x_m: Any
    y_m: Any

    #: `(N,)` `|a_i|^2`. Non-negative, and zero exactly where a ray was excluded.
    intensity: Any

    wavelength_m: float

    #: `ReferenceSurface.name` of the plane this was measured at.
    surface: str

    centroid_m: tuple[float, float]
    rms_radius_m: float
    geometric_radius_m: float

    #: Rows in the bundle, and rows the metrics were taken over. Equal when nothing
    #: was clipped; recorded separately because "the spot of the 3 rays that got
    #: through" and "the spot of 1027 rays" are different measurements and a
    #: normalized plot looks the same either way.
    ray_count: int
    included_count: int

    #: The declarations above, verbatim.
    inclusion_rule: str = SPOT_INCLUSION_RULE
    reference: str = SPOT_REFERENCE
    weighting: str = SPOT_WEIGHTING

    #: The provenance the gate accepted. Always `"unsplit"` today -- it is the only
    #: admissible value -- and carried anyway so the artifact states the invariant it
    #: was measured under rather than leaving it to this module's docstring.
    ray_splitting: RaySplitting = "unsplit"

    frame: Frame = dataclass_field(default_factory=Frame)

    def __post_init__(self) -> None:
        """The invariants, executed rather than asserted by an edge.

        Not vacuous even though `spot_diagram` only ever constructs this from
        `|a|^2` and a finite bundle: this is a public frozen dataclass, and an
        amplitude stored where an intensity was expected -- or a NaN metric from a
        caller's own arithmetic -- is exactly the substitution that yields a
        plausible-looking spot with impossible numbers in it.
        """
        require_finite(self.x_m, name="x_m")
        require_finite(self.y_m, name="y_m")
        require_finite(self.intensity, name="intensity")

        shapes = {
            name: tuple(getattr(self, name).shape) for name in ("x_m", "y_m", "intensity")
        }
        if len(set(shapes.values())) != 1 or len(shapes["x_m"]) != 1:
            raise ContractError(
                "SHAPE_MISMATCH",
                f"x_m, y_m and intensity must be one agreeing (N,) each, got {shapes!r}",
                declaration="intensity",
            )

        xp = xp_for(array_state(self.intensity).namespace)
        if bool(xp.any(self.intensity < 0.0)):
            raise ContractError(
                "NEGATIVE_INTENSITY",
                "intensity has negative entries, so it is not |a|^2. A complex "
                "amplitude stored here would also be refused by the finiteness check "
                "only by accident",
                declaration="intensity",
            )

        for name in ("rms_radius_m", "geometric_radius_m"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ContractError(
                    "UNIT_NOT_SI",
                    f"{name}={value!r} is not a finite non-negative radius in metres",
                    declaration=name,
                )

        if not 0 < self.included_count <= self.ray_count:
            raise ContractError(
                "EMPTY_ENSEMBLE",
                f"included_count={self.included_count} of ray_count={self.ray_count}: a "
                "spot with no included ray has no centroid, and more included than "
                "supplied is not a subset",
                declaration="included_count",
            )


def spot_diagram(rays: RayBundle) -> SpotResult:
    """Measure the spot `rays` make at the surface they are declared on.

    No system, no field, no source and no backend: the rays are the input in full,
    and the only thing asked of them beyond the geometry is the amplitude, because
    that is where per-ray intensity and ray survival both live.

    The provenance gate comes first, before any arithmetic, so a refusal is about
    the population rather than about a number derived from it. See the module
    docstring for why it is a declaration and not a heuristic.

    Raises:
        ContractError: `SPLIT_RAYS_NOT_MEASURABLE` or `RAY_SPLITTING_UNDECLARED` if
            the bundle is not declared unsplit; `MISSING_DECLARATION` if it carries
            no amplitude, so neither the intensity nor the survival of a ray is
            stated; `EMPTY_ENSEMBLE` if every ray was clipped.
    """
    if rays.ray_splitting == "split_descendants":
        raise ContractError(
            "SPLIT_RAYS_NOT_MEASURABLE",
            "this bundle is declared to contain ray-splitting descendants, so its rows "
            "are branches of one incident population rather than rays of one. An "
            "unweighted moment over them is a statistic of the branching",
            declaration="ray_splitting",
            remedy=_GATE_REMEDY,
        )
    if rays.ray_splitting != "unsplit":
        raise ContractError(
            "RAY_SPLITTING_UNDECLARED",
            f"this bundle declares ray_splitting={rays.ray_splitting!r}, so whether it "
            "contains ray-splitting descendants is not established. An ordinary "
            "geometric spot diagram requires an unsplit population, and that is not "
            "something this measurement can read off the numbers -- coincident rows are "
            "neither necessary nor sufficient for splitting",
            declaration="ray_splitting",
            remedy=(
                "Declare it at the producer that knows: a sequential geometric trace "
                "divides no ray, which is what backends/optiland/ states as 'unsplit'."
            ),
        )

    if rays.amplitude is None:
        raise ContractError(
            "MISSING_DECLARATION",
            "this bundle carries no amplitude, so it states neither the intensity of a "
            "ray nor whether the ray survived -- and survival is encoded in the "
            "amplitude on the supplied-ray trace path. Weighting every row equally "
            "instead would measure clipped rays as delivered ones",
            declaration="amplitude",
            remedy=(
                "Trace or construct the bundle with the amplitude its producer decided "
                "on. A measure_weight is a sampling weight and is not a substitute."
            ),
        )

    xp = rays.xp
    x = rays.positions_m[:, 0]
    y = rays.positions_m[:, 1]

    # `|a|^2` in the bundle's own namespace, device and precision, taken once. Every
    # number below is a reduction of it, which is the discipline `measurements.psf`
    # states at length: two reductions of two separately-squared arrays is how the
    # reference tree ended up with two answers for one PSF.
    intensity = xp.abs(rays.amplitude) ** 2
    total = float(xp.sum(intensity))
    if not total > 0.0:
        raise ContractError(
            "EMPTY_ENSEMBLE",
            f"all {rays.count} rays have zero intensity, so there is no spot: either "
            "every ray was clipped by a surface rim, or the bundle was constructed with "
            "a zero amplitude",
            declaration="amplitude",
            remedy=(
                "A trace in which nothing survives is a result about the system, not a "
                "spot to measure. Check the aperture, the field and the launch."
            ),
        )

    centroid_x = float(xp.sum(intensity * x)) / total
    centroid_y = float(xp.sum(intensity * y)) / total

    squared_radius = (x - centroid_x) ** 2 + (y - centroid_y) ** 2
    rms_radius = math.sqrt(float(xp.sum(intensity * squared_radius)) / total)

    # The one place the inclusion rule has to be applied explicitly. The weighted
    # moments above exclude a zero-intensity ray by arithmetic -- its weight is zero
    # -- but a maximum does not: a clipped ray at the edge of the field would set the
    # geometric radius while contributing nothing to the RMS, which is the one
    # combination that would look self-consistent and be wrong.
    included = intensity > 0.0
    zero = xp.zeros_like(squared_radius)
    geometric_radius = math.sqrt(
        float(xp.max(xp.where(included, squared_radius, zero)))
    )
    included_count = int(float(xp.sum(xp.where(included, xp.ones_like(zero), zero))))

    return SpotResult(
        x_m=x,
        y_m=y,
        intensity=intensity,
        wavelength_m=rays.wavelength_m,
        surface=rays.reference_surface.name,
        centroid_m=(centroid_x, centroid_y),
        rms_radius_m=rms_radius,
        geometric_radius_m=geometric_radius,
        ray_count=rays.count,
        included_count=included_count,
        ray_splitting=rays.ray_splitting,
        frame=rays.frame,
    )
