"""`RayBundle -> ScalarField`: the wavelet sum, and the conventions it stands on.

CHE-185 (R07.1) and CHE-186 (R07.2). One public function, two `StrEnum`s and one
diagnostics record:

```python
couplers.ray_to_scalar(rays, *, grid_shape, sample_pitch_m, surface=None,
                       projection=Projection.ASM_CONSISTENT,
                       reconstruction=Reconstruction.DIRECT,
                       kspace_oversample=1.5, kspace_grid_shape=None)
    -> tuple[ScalarField, ReconstructionDiagnostics]
```

The route is an **argument**, not a second operation. Both realizations compute the
same sum with the same semantic port pair, so a second public name, a second
descriptor or a module-level route registry would put two answers behind one
question. See `Reconstruction` for what the two actually differ by, which is a
discretization one of them has and the other does not.

The operator is

```
U(r) = sum_i  a_i * w_i * exp[ i k ( OPL_i + dr_i(r) ) ]
```

with `dr_i(r) = d_u_i (x - x0_i) + d_v_i (y - y0_i)` the extra path from the
ray's own intersection point on the surface to the field point, along the
wavelet direction.

**A ray is a plane wavelet, so each ray contributes a linear phase ramp across
the whole surface, not a point.** That is the one structural fact an
implementation can get wrong while still producing plausible output: depositing
energy at `(x0_i, y0_i)` is a different operator, and no intensity plot says so.

The sum is a discretization of a surface integral over the aperture, which is
why `w_i` -- the declared sampling measure -- is in it. See §"The measure is in
the kernel now" below; this is the one place the new tree departs from the
reference implementation's arithmetic, and it departs from it by moving a factor
rather than by changing a number.

Kind: a coupler, and why that is not obvious
--------------------------------------------
Heavy numerics do not make an operator (`docs/architecture_principles.md` §2).
This is `O(N_rays x N_pixels)` of complex exponentials and it is still a
*representation change*: the physical state before and after is the same field on
the same surface, described first as an ensemble of wavelets and then as a
sampled complex amplitude. Nothing propagates, no surface interaction happens,
and `z_m` does not move. The evidence that it really is a representation change
rather than an approximation of one is the 7.1e-15 round trip cited below --
which is also why the projection convention is not a free choice.

Two projection conventions, and only one of them is a coupler
-------------------------------------------------------------
Main-text eq 2 of Cheng et al., ACS Photonics 2026 (DOI
10.1021/acsphotonics.6c00818) carries a factor `<n_hat, d_hat>`; SI eq S5, which
derives the same sum as an estimator of the angular-spectrum integral, does not.
They are **different operators** and the paper does not flag the difference.

CHE-25 measured which one preserves a field: summing every propagating mode of a
random field on a 16x16 grid reproduces that field to `7.1e-15` without the
factor and misses it by `2.2 %` of peak amplitude with it, tracking the smallest
`cos(theta)` on the grid. A representation change must preserve the field, so
`Projection.ASM_CONSISTENT` is the default and the only one a coupler may use.
`Projection.SENSOR_OBLIQUITY` is retained as an explicitly named **detector**
model -- it is main-text eq 2, and it models an angle-dependent detector
response, not a field.

Both are implemented and the choice is reported, because `2.2 %` off-axis is
exactly the size of error that reads as a tolerance problem. Picking one
silently produces a coupler that quietly loses a few percent, round-trips
inexactly, and gives no test a name to fail under.

The measure is in the kernel now
--------------------------------
`RayBundle` separates physical amplitude from sampling measure and defaults
`measure_kind` to `"undeclared"` so that a consumer can **refuse** rather than
invent a quadrature (`representations/rays.py`, R02.3). The reference
implementation instead folded the area element into the amplitude at the
producer (CHE-47: `amplitude = sqrt(w) * dA`) and its kernel summed whatever
amplitude arrived. R05.2 moved the declaration; this kernel is where the factor
is applied again, so:

* `a_i` is `sqrt(intensity)` -- phase-free, no area factor;
* `w_i` is the physical quadrature area in m^2, or a dimensionless importance
  weight, and which one it is decides whether the sum owes a `1/N`;
* `a_i * w_i` is the launch amplitude the frozen convention calls
  `a = sqrt(I) * dA`.

Same physics as the reference, same number out. A kernel ported from the old tree
without this multiply looks correct and is short by `dA` per ray -- convergent in
shape, wrong in scale, which is the failure CHE-33 measured as
`(ray count)^2.0024` raw-power scaling.

Normalization is derived, not an argument. A declared quadrature is a *given*
ensemble and takes no `1/N` (main-text eq 2); a declared importance weight makes
the sum a Monte-Carlo estimator of a spectrum and owes the `1/N` of SI eqs
S3/S5. Both are stated by `measure_kind`, so a second argument saying the same
thing could only ever disagree with it.

**An undeclared measure is refused, and that refusal is not fussiness.**
`sum_i dA_i -> pi a^2` as the ring count grows, so the reconstructed discrete
power *converges* under ray refinement. Without the area element the sum is pinned
to the ray count instead of to the aperture: measured on a hexapolar fan from 217
to 49 537 rays, `d log P / d log N` is **2.0038** with an equal weight per ray and
**-0.0002** with the area element. Peak-normalized metrics cannot see the
difference, which is exactly why the wrong convention survived three milestones.

**What the weight is not.** Not an apodization, and not a substitute for one: it
is fixed by how the pupil was *sampled*, not by the physics of the aperture. So
the non-uniform launch amplitude it produces at the rim does **not** exercise this
kernel's response to a physically apodized, vignetted or Fresnel-weighted pupil.
Those remain untested and must not be claimed. It is also not the cause of
sampling artifacts: on the reference implementation's frozen configuration the
weight appears to shift the measured first-null radius by 3.6 %, and grid
refinement showed that to be 2.44 pixels per Airy radius rather than the weight.
That, like the paragraph below, is a **transcribed** finding -- nothing in the new
tree has re-measured it.

**Open item, carried forward rather than closed (CHE-187 criterion 6).** On an
off-axis field of the reference implementation's four-element benchmark lens, the
residual against the analytic Airy profile went from `1.48e-3` to `1.11e-2` when
the weight was introduced, and removing the weight improves it 7.5x. On axis the
same metric barely moved (`5.87e-3 -> 5.51e-3`). The rim taper is a *plausible*
cause -- a uniform-pupil Airy oracle sees a tapered pupil as a mismatch, and an
off-axis pupil is sampled asymmetrically -- but that is a **hypothesis, not a
measurement**, and it is not resolved by widening a tolerance. Neither that
system nor an off-axis Airy oracle exists in the new tree, so nothing here
re-measures it. The numbers are recorded so the ticket that lands the comparison
inherits them; the system is named in
`tests/physics/test_ray_to_scalar_refusals.py`, because a benchmark lens is
measured evidence and `tests/problems/test_fixtures.py` keeps its name out of
production for exactly that reason.

The absolute scale, and what it is not
--------------------------------------
Ray-density-independent but **not** SI-absolute. The kernel omits the
`1/(i lambda z)` Kirchhoff prefactor, so `U` is `i lambda z` times the SI field
and no `A_0` is ever declared. Reporting propagated power in watts would be wrong
by about eighteen orders of magnitude. Every power in
`ReconstructionDiagnostics` is therefore **relative** -- comparable between two
runs of the same configuration, and not a physical power.

The oracle for that scale is stationary phase, which is why the scale is
checkable at all: `int dA exp(i k rho^2 / 2R) = i lambda R (1 - exp(i pi a^2 / (lambda R)))`
over a disc of radius `a`, so a unit-amplitude-density pupil converging at `R`
reconstructs a focal peak of `lambda R * |1 - exp(i pi a^2 / (lambda R))|`, and
exactly `lambda R` when `a^2 = lambda R / 3`. If that stops holding, the launch
amplitude scale has moved. `tests/physics/test_ray_to_scalar.py` is the gate.

Declared limitation: no wavefront-curvature term (CHE-50)
---------------------------------------------------------
The sum is linear in the transverse coordinate, so the reconstructed field
carries **no** `exp(i k r^2 / 2R)` term. It is valid *at* the declared reference
surface with zero further propagation, and every emitted field says so in
`validity` -- `surface_only` and `no_wavefront_curvature_term`, the typed flags
R02.4 introduced precisely so this limitation is branchable instead of a
provenance string. Measured cost: about 1.2 rad of phase against an exact
spherical-wave reference at the 5-Airy-radius gate edge while the intensity
residual sits at 1e-3, so `|U|^2` will not warn a consumer who propagates it.

The remedy is not a correction term. Advance the **ray** state to the new surface
and reconstruct there -- exact, not an approximation.

The declared error budget between the two routes
-----------------------------------------------
`Reconstruction.DIRECT` evaluates each ray's ramp at its exact direction, so it
is exact per ray and is the oracle. `Reconstruction.KSPACE` bins each ray's
transverse wavevector bilinearly onto a k-grid and inverse-transforms once,
which removes the `O(N_rays x N_pixels)` cost and adds a discretization the
direct route does not have.

Measured speedup, with the configuration, because a performance claim without one
is not a performance claim: **61x** for 60 000 rays onto a 256x256 grid, 1.275 s
to 0.021 s, NumPy on the host in the pinned CPU container. The ratio is the point
rather than the seconds -- direct cost is `O(N_rays x ny x nx)` and k-space cost
is `O(N_rays + K log K)`, so it grows with the pixel count.

The budget:

* **On-node they agree to dtype round-off.** When every ray's `(k_x, k_y)` lands
  on a k-grid node the bilinear weights collapse to `(1, 0)` and the two routes
  are the same arithmetic. Measured 1.6e-15 of peak on an enumerated 16x16
  spectrum with `kspace_grid_shape` equal to the grid the modes were enumerated
  on -- and, for the same reason, at every integer oversampling of it.
* **Off-node the disagreement is the interpolation's, and it is not small.**
  Measured on 2 000 random directions in a 0.15 rad cone onto a 64x64 grid, as a
  fraction of peak amplitude against the direct route:

  | oversample | 1.0 | 1.5 | 2 | 4 | 8 | 16 | 32 |
  | -- | -- | -- | -- | -- | -- | -- | -- |
  | vs direct | 7.5e-1 | 4.0e-1 | 2.8e-1 | 7.3e-2 | 1.8e-2 | 4.9e-3 | 1.2e-3 |

  It falls as `oversample^-2`, which is bilinear interpolation's own rate and is
  what identifies the disagreement as the interpolation rather than as a defect.
  **At the default 1.5x the routes are not interchangeable**: 40 % of peak is not
  a tolerance, it is a different answer, and the caller either owns an
  enumeration and names `kspace_grid_shape`, or oversamples until the budget
  above is small enough for what it is doing.

The two routes also differ at the **top** band edge. A ray's fractional k-index is
`d_u K dx / lambda + K // 2`, so the representable band is asymmetric for even
`K` -- `[-lambda / (2 dx), lambda / (2 dx) (1 - 2 / K)]`, one bin short of the
output grid's Nyquist limit on the positive side, at any oversampling -- and
symmetric at `+- lambda / (2 dx) (1 - 1 / K)` for odd `K`, which the `n // 2`
origin rule makes the natural consequence rather than a special case. Either way a
mode sitting exactly on `+lambda / (2 dx)` is evaluated by the direct route and
**dropped, and counted in both rays and launch power, and reported** by the
k-space one. That is the declared rule for a mode the k-grid cannot hold: never
folded into the wrong bin, and never silently discarded either.

The trap in that first bullet is that exactness is a statement about *two* grids.
The enumerated modes of a padded patch land on nodes only if the k-grid period is
also the pad period; reconstructing them on a differently sized output grid at
some oversampling factor puts every mode off-node and converts an exactness
measurement into an interpolation error, with neither route at fault. **A caller
that owns an enumeration passes `kspace_grid_shape` outright** rather than relying
on `kspace_oversample`.

Grid Nyquist is a refusal, not a warning
-----------------------------------------
A wavelet with transverse direction cosine `d_t` writes a ramp of spatial
frequency `d_t / lambda`, which the output grid resolves only below
`1 / (2 * pitch)`. Beyond that the ramp folds into the wrong bin, which is
indistinguishable from a real feature. So it is refused. The condition is
**per axis**: a diagonal FFT bin has `|d_t| = sqrt(2) * lambda / (2 * pitch)`
and is exactly representable because each component sits at its own axis limit,
and testing the norm instead rejects the corner modes of any square spectrum.

This module imports no solver and no backend
--------------------------------------------
The coupler core is the physics under test; if it could import an engine, a
coupler defect could be misattributed to engine behaviour. One implementation
serves every device: the array module is taken from the bundle, so a NumPy
bundle executes on the host and a JAX bundle executes wherever its arrays live.
`tests/physics/test_ray_to_scalar.py` asserts both halves -- `sys.modules` in a
fresh interpreter, and an AST walk of this package's own imports, on top of the
walks in `tests/solvers/test_optiland_boundary.py` that already read every module
here for native names.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any, Literal

import numpy as np

from numerics import (
    PHASE_ACCUMULATION_FLOOR,
    ArrayNamespace,
    ArrayState,
    DType,
    Precision,
    device_of,
    dtype_of,
    matmul_precision_kwargs,
    namespace_of,
    numpy_dtype,
    verify_dtype,
)
from representations import (
    ContractError,
    RayBundle,
    ReferenceSurface,
    ScalarField,
    direction_norm_tolerance,
)

__all__ = [
    "DEFAULT_KSPACE_OVERSAMPLE",
    "REFUSALS",
    "SCALE_NOTE",
    "Normalization",
    "Projection",
    "Reconstruction",
    "ReconstructionDiagnostics",
    "grid_nyquist_direction_limit",
    "ray_to_scalar",
]


class Projection(StrEnum):
    """Which of the paper's two wavelet-sum conventions to apply.

    A `StrEnum` and not a boolean, because the two names are the argument: a
    `bool` called `apply_obliquity` says which factor is multiplied and not
    which physical object is being modelled, and the whole finding here is that
    those are two operators rather than one operator with a switch.
    """

    #: SI eq S5. No obliquity factor. Reproduces the angular-spectrum field on
    #: the surface to round-off, so it round-trips. The only correct choice for a
    #: representation change, and the default.
    ASM_CONSISTENT = "asm_consistent"

    #: Main-text eq 2. Applies `<n_hat, d_hat>`. Models an angle-dependent
    #: detector response -- a sensor, not a field.
    SENSOR_OBLIQUITY = "sensor_obliquity"


class Reconstruction(StrEnum):
    """How the ramp sum is evaluated. The same operator, two numerical realizations.

    A plane wavelet *is* a delta in k-space, so the sum can be evaluated either
    directly in real space or as a scatter onto a k-grid followed by one inverse
    FFT. Which one a caller wants depends on what it is for, and the two are not
    interchangeable -- see the module docstring's error budget.
    """

    #: Direct real-space evaluation at each ray's exact direction. Cost
    #: `O(N_rays x ny x nx)`, contracted from two `O(N n)` factors because the
    #: ramp is separable. **Exact per ray**, so this is the oracle every
    #: analytic gate in `tests/physics/test_ray_to_scalar.py` is measured
    #: through, and it stays the default.
    DIRECT = "direct"

    #: Bilinear scatter onto a k-grid plus one inverse FFT. Cost
    #: `O(N_rays + K log K)` -- per-ray cost stops scaling with pixel count,
    #: which is the whole point. The price is that each ray's direction is
    #: quantized onto the k-grid.
    KSPACE = "kspace"


#: Oversampling of the k-grid relative to the output grid when
#: `kspace_grid_shape` is not given. The reference implementation's value, kept
#: rather than raised: it is *characterized* against measured off-node error
#: (`tests/physics/test_ray_to_scalar_kspace.py`) rather than tuned to hit a
#: target, and raising it would hide the interpolation instead of reporting it.
DEFAULT_KSPACE_OVERSAMPLE = 1.5

#: Floor on how close to a k-grid node a ray must fall to count as on it, in bins.
#:
#: Measured as the distance to the *nearest* node, not the floor's fractional
#: part. A mode whose fractional index rounds to `4.999999999999999` sits on a
#: node and is reconstructed exactly, but its floor fraction is `1 - 1e-15`;
#: measuring the floor would report it as off-node and make the diagnostic depend
#: on whether the k-grid period happens to be a power of two.
#:
#: A floor rather than the value, because `_bin_tolerance` widens it for the dtype:
#: a fractional index has magnitude up to `K`, so its representation error is
#: `K * eps` bins, and at the FP32 floor a genuinely on-node bundle would
#: otherwise report `on_node_fraction = 0`.
_ON_NODE_BINS = 1.0e-9

#: How far outside the k-grid band a ray may fall and still be kept, in bins.
#:
#: The edge bins land *on* the boundary and arrive there through floating point:
#: a mode at index `-K//2` computes its fractional index as `0 - 1e-14`, and a
#: bare `>= 0` test discards it. The reference implementation measured that on an
#: enumerated patch it silently dropped 397 of 39 601 rays -- exactly the
#: outermost row and column -- and turned a 7.1e-13 exactness anchor into 8.5e-2.
#: A ray outside the band by a millionth of a bin is not distinguishable from one
#: on the edge, so it is kept and clipped; a ray genuinely outside is still
#: dropped, and still counted.
#:
#: Also a floor, widened for the dtype by `_bin_tolerance`. The frozen float64
#: value is unchanged by that widening (`64 * eps * K` is 3e-11 bins at `K = 2048`),
#: which is the point of keeping it as the floor rather than replacing it.
_KSPACE_EDGE_BINS = 1.0e-6


def _bin_tolerance(floor: float, *, bins: int, real_dtype: DType) -> float:
    """`floor`, widened to the representation error of a fractional index this large.

    A fractional k-index runs to `K`, so it carries about `K * eps` bins of
    round-off before anything is compared. `64 * eps` is the same allowance
    `representations.direction_norm_tolerance` derives for a unit vector, scaled
    by the magnitude actually being represented -- the module's one rule for this,
    applied to a bin index instead of a length.
    """
    return max(floor, 64.0 * float(np.finfo(numpy_dtype(real_dtype)).eps) * bins)


#: Whether the sum carries the `1/N` of a Monte-Carlo estimator.
#:
#: A `Literal`, not an argument and not a class: it is *derived* from the
#: bundle's `measure_kind` by `_resolve_measure` and reported, so it exists as a
#: vocabulary for the diagnostics rather than as a knob.
Normalization = Literal["none", "one_over_n"]

#: What each `measure_kind` implies for the sum. The table is the whole of the
#: normalization decision, and it is a table so that adding a fourth measure kind
#: to `representations.rays` fails here loudly instead of defaulting.
_NORMALIZATION_FOR_MEASURE: dict[str, Normalization] = {
    "quadrature_area_m2": "none",
    "importance_weight": "one_over_n",
}

#: Above this ray count the pairwise nearest-neighbour scan behind the
#: ray-density diagnostic is skipped rather than run at `O(N^2)`. The diagnostic
#: then reports `not_computed_above_scan_limit` instead of a number, because a
#: fabricated estimate of a sampling condition is worse than an absent one.
_NEAREST_NEIGHBOUR_SCAN_LIMIT = 4096

#: Every contract code raised *in this module*, plus the two `RayBundle` raises on
#: the way in -- with what each means at this boundary.
#:
#: Deliberately not "everything a caller can catch". A `ContractError` can also
#: arrive from `ScalarField.__post_init__` on the way out -- `NON_FINITE` if a
#: coherent sum overflows the compute dtype, for instance -- and enumerating another
#: type's intake here would be a copy that drifts. What this table is complete over
#: is the refusals this function *decides*, which is the set a caller branches on to
#: fix its own declaration.
#:
#: A dict and not a class: it is a table, and R07.3 budgets no class. Enumerated
#: for the reason `representations.CONTRACT_CODES` and `numerics.REFUSAL_CODES` are
#: -- a declared failure a caller could branch on, with nothing able to raise it,
#: is a claim about a path that does not exist. The reference implementation caught
#: a real one that way (`test_contract_code_reachability.py`), and
#: `tests/physics/test_ray_to_scalar_refusals.py` ports the principle in both
#: directions: every code here is reachable *through this function*, and every
#: `ContractError` code raised anywhere in this module is listed here.
#:
#: The last two are raised by `RayBundle.require_coherent()` rather than by this
#: module, and they are in the table because they are part of *this* boundary's
#: contract: a caller of `ray_to_scalar` branches on them here, and where the
#: `raise` statement physically lives is not something it can see.
REFUSALS: dict[str, str] = {
    "MEASURE_UNDECLARED": (
        "the bundle states no integration measure, so the quadrature the sum is taken "
        "with is unknown. Refused, never defaulted to uniform."
    ),
    "UNKNOWN_MEASURE_KIND": (
        "the bundle declares a measure kind this coupler has no normalization rule for. "
        "Whether the sum owes a 1/N is a property of the measure."
    ),
    "FRAME_MISMATCH": (
        "the caller expected a surface the bundle is not on, the surface is not "
        "perpendicular to the propagation axis, or the rays are not on it. This coupler "
        "does not propagate."
    ),
    "SHAPE_MISMATCH": (
        "the output grid is non-positive, cannot represent the steepest wavelet ramp per "
        "axis, or is larger than the k-grid it would have to be cropped out of."
    ),
    "UNIT_NOT_SI": (
        "a sample pitch is not a positive length in metres. Checked here rather than left "
        "to the emitted field, because the pitch is divided by to get the Nyquist limit "
        "before the field exists."
    ),
    "COHERENT_STATE_INCOMPLETE": (
        "the bundle carries no complex amplitude or no optical path. A real intensity "
        "weight is not an amplitude and this boundary will not choose the mapping."
    ),
    "OPL_REFERENCE_UNVERIFIED": (
        "the optical path is carried with its reference declared 'unverified', so its "
        "sign is not established. A wrong sign conjugates the wavefront."
    ),
}


#: The one sentence about what the reported powers are. Stated once, carried on
#: every diagnostics record, and asserted by `tests/physics/test_ray_to_scalar.py`
#: never to name a watt.
SCALE_NOTE = (
    "relative: no 1/(i lambda z) Kirchhoff prefactor and no declared A_0, so U is "
    "i lambda z times the SI field. Comparable between two runs of the same "
    "configuration; not a physical power."
)


@dataclass(frozen=True)
class ReconstructionDiagnostics:
    """Everything measured during a reconstruction, reported rather than judged.

    A class on rule 2: it is the public record a caller reads back, and the
    ticket's own acceptance criteria are statements about its fields -- the
    route that produced the field, the measure that was applied, the excluded
    power. Made of plain scalars so `as_dict()` is JSON-shaped without a
    serializer.

    It judges nothing. `grid_nyquist_satisfied` is `False` only on a path that
    already refused, and `ray_density_status` names a condition rather than
    passing or failing it: the wavelet picture holds locally while adjacent rays
    differ by less than half a cycle, and whether a caller cares is the caller's.
    """

    ray_count: int
    wavelength_m: float
    grid_shape: tuple[int, int]
    sample_pitch_m: tuple[float, float]

    #: The convention applied, and the equation it is.
    projection: str
    equation: str

    #: The measure that was declared, the `1/N` it implied, and its total. The
    #: sum is `sum_i w_i` -- in m^2 for a quadrature, dimensionless for an
    #: importance weight -- and it is the number that converges to the aperture
    #: area under ray refinement.
    measure_kind: str
    measure_sum: float
    normalization: str

    #: Amplitude and measure reported **separately**, because the contract is
    #: that nothing folds one into the other before the kernel. The launch power
    #: sum is what the kernel actually summed, i.e. after the multiply.
    incident_amplitude_power_sum: float
    launch_amplitude_power_sum: float

    #: The two sampling conditions, which fail independently: the grid's ability
    #: to represent the steepest ramp, and the ensemble's density.
    #:
    #: The first is reported as the `(v, u)` pair it is *enforced* as, not as a
    #: worst-of. Collapsing it to `max(|d_u|, |d_v|)` against `min(limit_x,
    #: limit_y)` reads as a violation on any anisotropic pitch -- `d_u = 1.0` on
    #: the `(0.30, 0.25) um` fixture is inside its own axis limit of 1.100 and
    #: outside the other axis's 0.917 -- so the summary number would contradict
    #: `grid_nyquist_satisfied` on a bundle that is perfectly well sampled.
    max_transverse_direction: tuple[float, float]
    grid_nyquist_direction_limit: tuple[float, float]
    grid_nyquist_satisfied: bool
    ray_spacing_estimate_m: float | None
    max_adjacent_ray_phase_rad: float | None
    ray_density_status: str

    #: `<n_hat, d_hat>` over the ensemble, always measured and only sometimes
    #: applied. Reported either way so a caller can see how much the two
    #: projections would differ on this bundle.
    min_projection_factor: float
    max_projection_factor: float

    #: Requested / resolved / actual, all three distinguishable. Under JAX
    #: without x64 a complex128 request comes back complex64 in silence, so the
    #: output dtype is read off the array that was produced.
    compute_precision: str
    input_state: dict[str, str]
    output_state: dict[str, str]

    #: Which of the two numerical realizations produced the field, so a
    #: downstream consumer can tell without being told.
    reconstruction: str

    reconstructed_discrete_power: float
    scale: str = SCALE_NOTE

    #: The k-grid, the splatted and dropped ray counts and the on-node fraction --
    #: or `None` on the direct route, which has no k-grid and drops nothing.
    #: Rays that cannot be represented on the k-grid are counted and reported,
    #: never silently dropped: a reconstruction missing 30 % of its rays and one
    #: missing none must not produce the same record.
    kspace: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        """A JSON-shaped mapping. `asdict` is enough because every field is plain."""
        record = asdict(self)
        for name in (
            "grid_shape",
            "sample_pitch_m",
            "max_transverse_direction",
            "grid_nyquist_direction_limit",
        ):
            record[name] = list(getattr(self, name))
        return record


def grid_nyquist_direction_limit(wavelength_m: float, pitch_m: float) -> float:
    """Largest transverse direction cosine a grid of this pitch can represent.

    A wavelet with transverse direction cosine `d_t` writes a phase ramp of
    spatial frequency `d_t / lambda` onto the surface, which the grid resolves
    only below the Nyquist frequency `1 / (2 * pitch)`:

        |d_t| <= lambda / (2 * pitch)

    A condition on the **output grid**, distinct from whether the ray ensemble
    samples the wavefront densely enough. Both can fail independently and
    refining one does not fix the other, which is the usual wasted debugging
    step -- so both are reported, separately, on the diagnostics record.

    Public because a caller choosing an output pitch needs it *before* calling
    the coupler, and because deriving it again at a call site is how the factor
    of two goes missing.
    """
    return wavelength_m / (2.0 * pitch_m)


def _compute_precision(rays: RayBundle) -> Precision:
    """The precision this coupler will accumulate phase in for `rays`.

    Taken from the data the bundle actually carries -- geometry, amplitude and
    optical path -- and floored at `numerics.PHASE_ACCUMULATION_FLOOR`. Distinct
    from the input dtype on purpose: a bundle handed in as float16 is computed in
    float32, and calling that "float16 support" would be advertising a cast.
    """
    precisions = [dtype_of(rays.positions_m).precision]
    for optional in (rays.amplitude, rays.optical_path_m):
        if optional is not None:
            precisions.append(dtype_of(optional).precision)
    return max([*precisions, PHASE_ACCUMULATION_FLOOR], key=lambda p: p.bits)


def _cis(xp: Any, phase: Any, complex_dtype: DType) -> Any:
    """`exp(i * phase)` in an explicitly chosen complex dtype.

    Written out rather than left to `xp.exp(1j * phase)`, which relies on scalar
    promotion rules that differ between NumPy versions and between NumPy and
    JAX-without-x64. The dtype of a reconstructed field is part of this coupler's
    contract, so it is stated rather than inherited.
    """
    return xp.exp(phase.astype(numpy_dtype(complex_dtype)) * 1j)


def _resolve_measure(rays: RayBundle) -> tuple[Any, Normalization]:
    """The declared sampling measure, or a refusal. Never a default.

    `representations.rays` makes `"undeclared"` the default value of
    `measure_kind` so that this refusal is what happens when nobody thought about
    it. The pressure the ticket names is real -- a fixture forgets to declare its
    measure and the cheapest repair is to treat the weight as uniform -- and
    giving in turns the contract into a default. Uniform is not a synonym for
    unknown: a coupler that assumes it has invented a quadrature, and the
    invented one differs from the true one by the aperture area.
    """
    if rays.measure_kind == "undeclared":
        raise ContractError(
            "MEASURE_UNDECLARED",
            "this bundle declares no integration measure, so the quadrature the wavelet "
            "sum is taken with is unknown. U(r) = sum_i a_i w_i exp[i k (...)] is a "
            "surface integral over the aperture; without w_i the sum is pinned to the "
            "ray count instead of to the aperture, and the reconstructed power scales as "
            "(ray count)^2 rather than converging (CHE-33 measured 2.0024).",
            declaration="measure_kind",
            remedy=(
                "Declare the measure at the producer -- 'quadrature_area_m2' for a pupil "
                "area element in m^2, 'importance_weight' for a Monte-Carlo 1/p. A "
                "trusted ray-to-wave conversion refuses rather than inventing one; do "
                "not treat an unknown measure as uniform."
            ),
        )
    normalization = _NORMALIZATION_FOR_MEASURE.get(rays.measure_kind)
    if normalization is None:
        # Unreachable through `RayBundle`, which validates `measure_kind` against
        # `MEASURE_KINDS` -- and all three of those have a row. This is the window
        # between a fourth kind landing in `representations/` and its row landing
        # here, which R08 may open. `tests/physics/test_ray_to_scalar_refusals.py`
        # reaches it through this helper, so it is covered rather than claimed.
        raise ContractError(
            "UNKNOWN_MEASURE_KIND",
            f"measure_kind {rays.measure_kind!r} has no declared normalization in this "
            "coupler. Whether the sum owes a 1/N is a property of the measure, and a "
            "measure kind that lands in representations/ lands with its row here.",
            declaration="measure_kind",
        )
    return rays.measure_weight, normalization


def _require_declared_surface(
    rays: RayBundle, surface: ReferenceSurface | None, xp: Any
) -> ReferenceSurface:
    """The surface the field is emitted on: the bundle's, checked three ways.

    Not an override. The kernel does not propagate, so the reconstructed field
    lives exactly where the rays are declared; emitting it on a *different*
    surface would relabel the geometry the optical path was measured in and turn
    a whole pupil-to-focus distance into a plausible-looking defocus. `surface`
    is therefore an expectation a caller may state and have checked -- the role
    the reference implementation's `_check_plane` played -- and a mismatch is
    refused. To reconstruct somewhere else, advance the ray state there and call
    again: exact, not an approximation.

    The other two checks are about geometry rather than labels, and they exist
    because the kernel's ramp is purely transverse. `dr_i(r)` carries no `d_z`
    term, so:

    * a **tilted** reference is refused. The sample grid this function's caller
      builds is the `z = z_m` plane, and on a tilted surface those points are not
      on the surface the emitted field declares. The missing `d_z` term is a pure
      phase error with no symptom in `|U|^2`. `ReferenceSurface.normal` stays
      general because a tilted reference is expressible; this coupler is the
      consumer that says it cannot use one.
    * a bundle whose rays are **not on the plane** is refused. Ray `i`'s optical
      path is measured to *its own* intersection point, so a ray sitting at `z_i`
      contributes as though it were at `z_m` and loses `k d_z (z_m - z_i)`. The
      allowance is float round-off on the plane's own coordinate and nothing
      more -- being on the plane is a yes/no fact, not a budget.
    """
    if surface is not None and surface != rays.reference_surface:
        raise ContractError(
            "FRAME_MISMATCH",
            f"the caller expects the surface {surface!r} but the bundle is declared on "
            f"{rays.reference_surface!r}. This coupler changes representation and does not "
            "propagate, so the field can only be emitted where the rays already are.",
            declaration="surface",
            remedy=(
                "Advance the ray state to the expected surface and reconstruct there -- "
                "exact, not an approximation -- or pass the surface the bundle declares."
            ),
        )
    declared = rays.reference_surface
    if declared.normal != rays.frame.propagation_normal:
        raise ContractError(
            "FRAME_MISMATCH",
            f"the bundle is declared on a surface whose normal is {declared.normal}, not "
            f"{rays.frame.propagation_normal}. The wavelet ramp this coupler evaluates is "
            "purely transverse, so it can only reconstruct on a plane perpendicular to the "
            "propagation axis; on a tilted one the sample points are not on the surface the "
            "emitted field would declare, and the missing d_z term is a phase error that "
            "|U|^2 cannot show.",
            declaration="reference_surface.normal",
        )

    axial = rays.positions_m[:, 2]
    # Round-off on the plane's own coordinate, at the dtype the positions are
    # stored in. `64 * eps` is the same allowance `direction_norm_tolerance`
    # derives for a unit vector, applied to a length by scaling it with the
    # magnitude being represented.
    scale = max(abs(declared.z_m), float(xp.max(xp.abs(axial))), 1.0)
    tolerance = 64.0 * float(np.finfo(numpy_dtype(dtype_of(axial))).eps) * scale
    deviation = float(xp.max(xp.abs(axial - declared.z_m)))
    if deviation > tolerance:
        raise ContractError(
            "FRAME_MISMATCH",
            f"the rays are not on the surface they are declared on: the worst axial "
            f"deviation is {deviation:.3e} m from z_m = {declared.z_m!r}, against a "
            f"round-off allowance of {tolerance:.3e} m. Each ray's optical path is "
            "measured to its own intersection point and the ramp is purely transverse, so "
            "a ray off the plane contributes as though it were on it and loses "
            "k d_z (z_m - z_i) of phase.",
            declaration="positions_m",
            remedy=(
                "Advance the ray state to the declared plane, or declare the plane the "
                "rays are actually on."
            ),
        )
    return declared


def _ray_density(positions_xy: Any, directions_xy: Any, wavenumber: float, xp: Any) -> tuple[
    float | None, float | None, str
]:
    """Estimate the worst phase step between neighbouring rays on the surface.

    The wavelet picture holds locally only while adjacent rays differ by less
    than half a cycle. For each ray, find its nearest neighbour and evaluate the
    phase disagreement between the two ramps over their separation,
    `k * |(d_i - d_j) . (r_i - r_j)|`.
    """
    count = int(positions_xy.shape[0])
    if count < 2:
        return None, None, "not_applicable_single_ray"
    if count > _NEAREST_NEIGHBOUR_SCAN_LIMIT:
        return None, None, "not_computed_above_scan_limit"

    delta = positions_xy[:, None, :] - positions_xy[None, :, :]
    distances = xp.linalg.norm(delta, axis=2)
    # `fill_diagonal` mutates in place, which a JAX array does not support. The
    # functional form masks the self-distance and is identical in both.
    distances = xp.where(xp.eye(count, dtype=bool), xp.inf, distances)
    neighbour = xp.argmin(distances, axis=1)
    separation = distances[xp.arange(count), neighbour]

    direction_difference = directions_xy - directions_xy[neighbour]
    offsets = positions_xy - positions_xy[neighbour]
    phase_step = wavenumber * xp.abs(xp.sum(direction_difference * offsets, axis=1))

    worst = float(xp.max(phase_step))
    return (
        float(xp.mean(separation)),
        worst,
        "wavelet_approximation_holds" if worst < math.pi else "adjacent_ray_phase_step_exceeds_pi",
    )


def _scatter_add(
    xp: Any, namespace: ArrayNamespace, size: int, indices: Any, values: Any, dtype: Any
) -> Any:
    """`out[indices] += values` on a flat array of length `size`.

    Scatter-add is outside the array-API surface `xp` otherwise provides, so this
    is the one operation in the module that has to name its namespace -- the same
    concession `numerics.matmul_precision_kwargs` makes, for the same reason.

    Both branches accumulate **in `dtype`**, not in a wider one. Worth stating
    because `np.bincount` would be faster on the host and would accumulate in
    float64 regardless of what it was handed, which would make the NumPy path
    quietly more accurate than the JAX path and turn a namespace disagreement into
    a mystery.
    """
    if namespace is ArrayNamespace.JAX:
        return xp.zeros(size, dtype=dtype).at[indices].add(values)
    out = np.zeros(size, dtype=dtype)
    np.add.at(out, indices, values)
    return out


def _reconstruct_kspace(
    xp: Any,
    namespace: ArrayNamespace,
    *,
    coefficient: Any,
    directions_xy: Any,
    wavenumber: float,
    grid_shape: tuple[int, int],
    sample_pitch_m: tuple[float, float],
    oversample: float,
    explicit_shape: tuple[int, int] | None,
    origin_index: Callable[[int], int],
    real_dtype: DType,
    complex_dtype: DType,
) -> tuple[Any, dict[str, Any]]:
    """Evaluate the wavelet sum as one k-space scatter plus one inverse FFT.

    Each ray is a plane wavelet, so in k-space it is a delta at
    `(k_x, k_y) = k (d_u, d_v)` weighted by the same `coefficient` the direct
    route uses. Depositing those deltas bilinearly on the grid whose period is
    `(K_y dy, K_x dx)` and inverse-transforming gives

        u[y, x] = sum_p Chat[p] exp(i (k_x_p x + k_y_p y))

    which is the ramp sum with each ray's direction snapped, by interpolation,
    onto the k-grid.

    Two departures from the reference implementation's upstream source, both
    deliberate and both kept:

    1. **The crop offset is `origin(K) - origin(n)`, not `(K - n) // 2`.** They
       agree whenever `K` and `n` share parity and differ by one sample otherwise,
       which would put the reconstructed origin one pixel off the coordinate origin
       this repository pins -- a half-pixel tilt, not a visible failure.
       `origin_index` is `Frame.origin_index`, passed in rather than written out, so
       this route and the field it produces cannot adopt different centrings.
    2. **A ray at the top k-bin is kept, not dropped.** Requiring a full
       four-neighbour footprint discards it; here the upper neighbour index is
       clamped instead, and at `i == K - 1` the interpolation weight on that
       neighbour is identically zero, so clamping adds nothing and keeps a ray the
       direct route would have used.

    The upstream aperture crop -- dropping rays that land outside the grid extent
    before accumulation -- is **not** ported. A wavelet contributes a ramp across
    the whole surface, so where it happens to cross does not bound where it
    contributes; that crop is a sensor model, the direct route has none, and
    porting it would make the two routes disagree for a reason unrelated to the
    algorithm.
    """
    ny, nx = grid_shape
    dy, dx = sample_pitch_m
    if explicit_shape is not None:
        ky_n, kx_n = int(explicit_shape[0]), int(explicit_shape[1])
    else:
        ky_n, kx_n = math.ceil(ny * oversample), math.ceil(nx * oversample)
    if ky_n < ny or kx_n < nx:
        raise ContractError(
            "SHAPE_MISMATCH",
            f"the k-grid ({ky_n}, {kx_n}) is smaller than the output grid ({ny}, {nx}), so "
            "the field cannot be cropped out of it",
            declaration="kspace_grid_shape",
            remedy=(
                "Pass kspace_oversample >= 1.0, or a kspace_grid_shape at least as large "
                "as the output shape on each axis."
            ),
        )

    real_np, complex_np = numpy_dtype(real_dtype), numpy_dtype(complex_dtype)
    count = int(directions_xy.shape[0])

    # Fractional, fftshifted grid index of each ray's transverse wavevector. `dk`
    # is set by the *k-grid* period, which is why oversampling changes which rays
    # land on a node and which do not.
    delta_ky = 2.0 * math.pi / (ky_n * dy)
    delta_kx = 2.0 * math.pi / (kx_n * dx)
    fractional_y = (wavenumber * directions_xy[:, 1]) / delta_ky + origin_index(ky_n)
    fractional_x = (wavenumber * directions_xy[:, 0]) / delta_kx + origin_index(kx_n)

    # Unrepresentable rays are **zero-weighted, not removed**. Compacting with a
    # boolean index produces a data-dependent shape, which under JAX means a host
    # synchronization plus a gather in the middle of the pipeline; the reference
    # implementation measured that serialization making the fast path slower than
    # the one it replaces. Keeping every shape static is what puts the asymptotics
    # in the wall clock.
    edge = _bin_tolerance(_KSPACE_EDGE_BINS, bins=max(ky_n, kx_n), real_dtype=real_dtype)
    representable = (
        (fractional_y >= -edge)
        & (fractional_y <= ky_n - 1 + edge)
        & (fractional_x >= -edge)
        & (fractional_x <= kx_n - 1 + edge)
    )
    fractional_y = xp.clip(fractional_y, 0.0, float(ky_n - 1))
    fractional_x = xp.clip(fractional_x, 0.0, float(kx_n - 1))
    weights = coefficient * representable.astype(complex_np)

    # int32, deliberately. These index a flat array of ky_n * kx_n elements, so
    # int32 covers any k-grid that can be allocated at all, and asking for int64
    # under JAX without x64 is silently truncated to int32 anyway.
    lower_y = xp.floor(fractional_y).astype(np.int32)
    lower_x = xp.floor(fractional_x).astype(np.int32)
    frac_y = (fractional_y - lower_y.astype(real_np)).astype(real_np)
    frac_x = (fractional_x - lower_x.astype(real_np)).astype(real_np)
    upper_y = xp.minimum(lower_y + 1, ky_n - 1)
    upper_x = xp.minimum(lower_x + 1, kx_n - 1)

    # One allocation and one scatter for all four corners rather than four of
    # each: the fused form is the difference between one transient corner array
    # per chunk and four.
    indices = xp.concatenate(
        [
            lower_y * kx_n + lower_x,
            lower_y * kx_n + upper_x,
            upper_y * kx_n + lower_x,
            upper_y * kx_n + upper_x,
        ]
    )
    corner_weights = xp.concatenate(
        [
            (1.0 - frac_y) * (1.0 - frac_x),
            (1.0 - frac_y) * frac_x,
            frac_y * (1.0 - frac_x),
            frac_y * frac_x,
        ]
    ).astype(complex_np)
    flat = _scatter_add(
        xp,
        namespace,
        ky_n * kx_n,
        indices,
        xp.concatenate([weights, weights, weights, weights]) * corner_weights,
        complex_np,
    )

    # `ifft2` carries `1 / (K_y K_x)`; the sum being evaluated does not. That
    # factor belongs to the transform and is unrelated to the estimator's
    # `1 / N_rays`, which the caller applies once afterwards.
    padded = xp.fft.fftshift(xp.fft.ifft2(xp.fft.ifftshift(flat.reshape(ky_n, kx_n)))) * (
        ky_n * kx_n
    )
    start_y = origin_index(ky_n) - origin_index(ny)
    start_x = origin_index(kx_n) - origin_index(nx)
    u = padded[start_y : start_y + ny, start_x : start_x + nx].astype(complex_np)

    node_distance = xp.maximum(
        xp.minimum(frac_y, 1.0 - frac_y).astype(real_np),
        xp.minimum(frac_x, 1.0 - frac_x).astype(real_np),
    )
    # Both reductions are read after the field, so the only device-to-host
    # synchronization here happens once and after all the work is enqueued.
    on_node_bins = _bin_tolerance(_ON_NODE_BINS, bins=max(ky_n, kx_n), real_dtype=real_dtype)
    kept = int(xp.sum(representable))
    on_node = int(xp.sum(representable & (node_distance < on_node_bins)))
    # Power as well as rays. A ray count is the number a consumer reaches for, and
    # on an ensemble with non-uniform amplitude the two are not interchangeable:
    # the rays nearest the band edge carry the extreme wavevectors, so a 0.1 % ray
    # drop on an apodized or vignetted pupil can be a large amplitude drop.
    launch_power = float(xp.sum(xp.abs(coefficient) ** 2))
    dropped_power = float(xp.sum(xp.abs(coefficient * (~representable).astype(complex_np)) ** 2))
    record: dict[str, Any] = {
        "kspace_grid_shape": [ky_n, kx_n],
        "rays_splatted": kept,
        "rays_dropped_out_of_band": count - kept,
        "dropped_fraction": (count - kept) / count if count else 0.0,
        "dropped_launch_power_fraction": (
            dropped_power / launch_power if launch_power > 0.0 else 0.0
        ),
        # Fraction of *splatted* rays on a node. 1.0 certifies exactness only when
        # read together with `dropped_fraction == 0.0`; dropped rays are excluded
        # from this ratio rather than counted against it.
        "on_node_fraction": (on_node / kept) if kept else 0.0,
    }
    if kept == 0:
        record["note"] = "no ray was representable on the k-grid; the field is identically zero"
    return u, record


def ray_to_scalar(
    rays: RayBundle,
    *,
    grid_shape: tuple[int, int],
    sample_pitch_m: tuple[float, float],
    surface: ReferenceSurface | None = None,
    projection: Projection = Projection.ASM_CONSISTENT,
    reconstruction: Reconstruction = Reconstruction.DIRECT,
    kspace_oversample: float = DEFAULT_KSPACE_OVERSAMPLE,
    kspace_grid_shape: tuple[int, int] | None = None,
) -> tuple[ScalarField, ReconstructionDiagnostics]:
    """Reconstruct the scalar field a ray bundle describes on its own surface.

    Parameters
    ----------
    rays
        Rays carrying a complex amplitude, an optical path with a declared
        reference, and a declared sampling measure. A bundle carrying only a
        real intensity weight, or an optical path whose reference is
        `"unverified"`, is refused by `RayBundle.require_coherent()`; a bundle
        whose `measure_kind` is `"undeclared"` is refused here.
    grid_shape, sample_pitch_m
        Output grid as `(ny, nx)` and `(dy, dx)` in metres. Coordinate zero is at
        index `n // 2` on each axis, read from `Frame.origin_index` rather than
        written out, so the coupler cannot adopt a different centring than the
        field it emits.
    surface
        The surface the caller expects the rays to be declared on, checked and
        not applied. See `_require_declared_surface`.
    projection
        `ASM_CONSISTENT` (the default) reconstructs the field;
        `SENSOR_OBLIQUITY` applies `<n_hat, d_hat>` and models a detector. They
        are different operators -- see the module docstring.
    reconstruction
        Which realization evaluates the sum. `DIRECT` (the default) is exact per
        ray and costs `O(N_rays x ny x nx)`; `KSPACE` costs
        `O(N_rays + K log K)` and quantizes each ray's direction onto the k-grid.
        The default is the default because every analytic gate is measured
        through it.
    kspace_oversample, kspace_grid_shape
        The k-grid, for `KSPACE` only. `kspace_grid_shape` names it outright and
        wins; otherwise it is `ceil(oversample * grid_shape)` per axis. A caller
        reconstructing an *enumeration* must name it: exactness holds only when
        the k-grid period equals the grid the modes were enumerated on, and an
        oversampling factor that happens to miss that period converts an
        exactness measurement into an interpolation error.

    Returns
    -------
    The reconstructed field, and the diagnostics measured while producing it.
    The diagnostics are returned rather than attached because `ScalarField`
    carries typed `validity` and no provenance dict, by R02.4's decision; a
    caller that wants the record keeps it, and one that does not is not handed a
    free-form mapping that quietly becomes load-bearing.

    Raises
    ------
    ContractError
        `COHERENT_STATE_INCOMPLETE` / `OPL_REFERENCE_UNVERIFIED` from the bundle,
        `MEASURE_UNDECLARED` for an undeclared measure, `FRAME_MISMATCH` for a
        surface the bundle is not on, `SHAPE_MISMATCH` for a non-positive grid or
        one that cannot represent the steepest wavelet ramp.
    """
    amplitude, optical_path_m = rays.require_coherent()
    measure_weight, normalization = _resolve_measure(rays)
    emitted_surface = _require_declared_surface(rays, surface, rays.xp)

    ny, nx = int(grid_shape[0]), int(grid_shape[1])
    dy, dx = float(sample_pitch_m[0]), float(sample_pitch_m[1])
    if ny <= 0 or nx <= 0:
        raise ContractError(
            "SHAPE_MISMATCH",
            f"grid_shape must be positive, got {grid_shape!r}",
            declaration="grid_shape",
        )
    if not (dy > 0.0 and dx > 0.0) or not (math.isfinite(dy) and math.isfinite(dx)):
        # Checked here and not left to `ScalarField`, which would also refuse it:
        # the pitch is divided by to get the Nyquist limit several steps earlier, so
        # a zero would surface as a bare `ZeroDivisionError` with no code on it.
        raise ContractError(
            "UNIT_NOT_SI",
            f"sample_pitch_m must be two positive finite lengths in metres, got "
            f"{sample_pitch_m!r}",
            declaration="sample_pitch_m",
        )

    # The one place the execution representation is chosen. Everything below is
    # written against `xp` and the two dtypes, so the same source runs on the
    # host and on a device.
    xp = rays.xp
    namespace = namespace_of(rays.positions_m)
    # A dot product must compute at the dtype it claims; on a device that takes an
    # explicit request. See numerics.arrays.matmul_precision_kwargs.
    dot = matmul_precision_kwargs(namespace)
    precision = _compute_precision(rays)
    real_dtype = precision.real_dtype
    complex_dtype = precision.complex_dtype
    if complex_dtype is None:  # pragma: no cover - unreachable behind the FP32 floor
        # Not a `ContractError`. FP16 is the only family with no complex dtype and
        # `_compute_precision` floors at `PHASE_ACCUMULATION_FLOOR` (FP32), so
        # reaching here means that floor changed, which is a programming error in
        # this module rather than a bad declaration by a caller. Declaring it as a
        # refusal code would put a branch in `REFUSALS` that no test can reach.
        raise RuntimeError(
            f"{precision} has no complex dtype to reconstruct a field in; the phase "
            "accumulation floor must stay at FP32 or above"
        )
    real_np, complex_np = numpy_dtype(real_dtype), numpy_dtype(complex_dtype)

    wavenumber = rays.wavenumber
    positions_xy = rays.positions_m[:, :2].astype(real_np)
    directions_xy = rays.directions[:, :2].astype(real_np)

    # The grid must be able to represent the steepest ramp before anything is
    # summed onto it, per axis rather than on the direction norm.
    limit_y = grid_nyquist_direction_limit(rays.wavelength_m, dy)
    limit_x = grid_nyquist_direction_limit(rays.wavelength_m, dx)
    max_du = float(xp.max(xp.abs(directions_xy[:, 0])))
    max_dv = float(xp.max(xp.abs(directions_xy[:, 1])))
    # The corner bins of an enumerated spectrum land *on* the limit and arrive
    # there through floating point: `d_u = lambda / (2 * dx)` computed from a
    # renormalized direction is a ulp either side of it, and a bare `<=` then
    # refuses a mode that is exactly representable. The allowance is
    # `representations.direction_norm_tolerance` rather than a constant of this
    # module's own, because it is the same question -- how far a direction cosine
    # may drift at this dtype before the drift is real -- and it is already
    # derived and tested there.
    edge = direction_norm_tolerance(dtype_of(directions_xy))
    nyquist_satisfied = bool(max_du <= limit_x + edge and max_dv <= limit_y + edge)
    if not nyquist_satisfied:
        raise ContractError(
            "SHAPE_MISMATCH",
            f"the output grid cannot represent the steepest wavelet ramp: |d_u|max = "
            f"{max_du:.6f} against limit {limit_x:.6f}, |d_v|max = {max_dv:.6f} against "
            f"limit {limit_y:.6f} (lambda / (2 * pitch), per axis). Beyond the limit a "
            "ramp folds into the wrong bin, which is indistinguishable from a real "
            "feature, so it is refused rather than aliased.",
            declaration="sample_pitch_m",
            remedy=(
                "Refine the output pitch, or restrict the ray directions. Adding rays "
                "will not help: this is a grid condition, not a ray-density one."
            ),
        )

    # `<n_hat, d_hat>`. Always computed so it can be reported, applied only under
    # the sensor convention -- see the module docstring for why the coupler
    # default omits it.
    normal = xp.asarray(emitted_surface.normal, dtype=real_np)
    projection_factor = xp.matmul(rays.directions.astype(real_np), normal, **dot)
    weight = (
        projection_factor.astype(complex_np)
        if projection is Projection.SENSOR_OBLIQUITY
        else xp.ones(rays.count, dtype=complex_np)
    )

    # Amplitude times measure: the launch amplitude `a = sqrt(I) * dA` of the
    # frozen convention, formed here and nowhere earlier. The bundle carries the
    # two apart so that an undeclared measure is refusable at all.
    launch = amplitude.astype(complex_np) * measure_weight.astype(real_np).astype(complex_np)

    # Constant per-ray phase: the optical path, minus the ramp evaluated back at
    # the ray's own intersection point, so `dr_i` is measured from there.
    constant_phase = optical_path_m.astype(real_np) - xp.sum(
        directions_xy * positions_xy, axis=1
    )
    coefficient = launch * weight * _cis(xp, wavenumber * constant_phase, complex_dtype)

    # Grid coordinates on the `n // 2` origin, built here because the field they
    # belong to does not exist yet. `Frame.origin_index` is the one
    # implementation of the rule and is what `ScalarField.coordinates()` reads,
    # so the ramps and the emitted field's own axes cannot disagree.
    origin_y = rays.frame.origin_index(ny)
    origin_x = rays.frame.origin_index(nx)
    y = (xp.arange(ny, dtype=real_np) - origin_y) * dy
    x = (xp.arange(nx, dtype=real_np) - origin_x) * dx

    kspace_record: dict[str, Any] | None = None
    if reconstruction is Reconstruction.KSPACE:
        # One scatter and one FFT. Nothing of size (N, ny) or (N, nx) is formed,
        # which is the entire point: per-ray cost stops scaling with pixels.
        u, kspace_record = _reconstruct_kspace(
            xp,
            namespace,
            coefficient=coefficient,
            directions_xy=directions_xy,
            wavenumber=wavenumber,
            grid_shape=(ny, nx),
            sample_pitch_m=(dy, dx),
            oversample=kspace_oversample,
            explicit_shape=kspace_grid_shape,
            origin_index=rays.frame.origin_index,
            real_dtype=real_dtype,
            complex_dtype=complex_dtype,
        )
    else:
        # `exp(i k (d_u x + d_v y))` is separable, so the O(N ny nx) sum contracts
        # from two O(N n) factors instead of materializing an (N, ny, nx) tensor.
        ramp_y = _cis(xp, wavenumber * xp.outer(directions_xy[:, 1], y), complex_dtype)
        ramp_x = _cis(xp, wavenumber * xp.outer(directions_xy[:, 0], x), complex_dtype)
        u = xp.einsum("n,ny,nx->yx", coefficient, ramp_y, ramp_x, optimize=True, **dot)

    if normalization == "one_over_n":
        u = u / rays.count

    # Resolved versus actual, checked rather than assumed: under JAX without x64 a
    # complex128 request comes back complex64 in silence, and a coupler reporting
    # its *resolved* dtype would be reporting a precision it did not compute in.
    u = verify_dtype(u, complex_dtype, context="couplers.ray_to_scalar")

    field = ScalarField(
        u=u,
        sample_pitch_m=(dy, dx),
        wavelength_m=rays.wavelength_m,
        reference_surface=emitted_surface,
        frame=rays.frame,
        # CHE-50, declared rather than silently carried. Typed, so a consumer
        # branches on it instead of parsing a provenance string.
        validity=frozenset({"surface_only", "no_wavefront_curvature_term"}),
    )

    spacing, max_phase_step, density_status = _ray_density(
        positions_xy, directions_xy, wavenumber, xp
    )
    diagnostics = ReconstructionDiagnostics(
        ray_count=rays.count,
        wavelength_m=rays.wavelength_m,
        grid_shape=(ny, nx),
        sample_pitch_m=(dy, dx),
        projection=str(projection),
        equation=(
            "ACS Photonics 2026 SI eq S5 (no obliquity factor)"
            if projection is Projection.ASM_CONSISTENT
            else "ACS Photonics 2026 main text eq 2 (with <n_hat, d_hat> obliquity)"
        ),
        measure_kind=rays.measure_kind,
        measure_sum=float(xp.sum(measure_weight)),
        normalization=normalization,
        incident_amplitude_power_sum=float(xp.sum(xp.abs(amplitude) ** 2)),
        launch_amplitude_power_sum=float(xp.sum(xp.abs(launch) ** 2)),
        max_transverse_direction=(max_dv, max_du),
        grid_nyquist_direction_limit=(limit_y, limit_x),
        grid_nyquist_satisfied=nyquist_satisfied,
        ray_spacing_estimate_m=spacing,
        max_adjacent_ray_phase_rad=max_phase_step,
        ray_density_status=density_status,
        min_projection_factor=float(xp.min(projection_factor)),
        max_projection_factor=float(xp.max(projection_factor)),
        compute_precision=str(precision),
        input_state=rays.state.as_dict(),
        output_state=ArrayState(dtype_of(u), device_of(u), namespace_of(u)).as_dict(),
        reconstruction=str(reconstruction),
        reconstructed_discrete_power=field.discrete_power(),
        kspace=kspace_record,
    )
    return field, diagnostics
