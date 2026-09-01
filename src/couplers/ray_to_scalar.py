"""`RayBundle -> ScalarField`: the wavelet sum, and the conventions it stands on.

CHE-185 (R07.1). One public function, one `StrEnum` and one diagnostics record:

```python
couplers.ray_to_scalar(rays, *, grid_shape, sample_pitch_m, surface=None,
                       projection=Projection.ASM_CONSISTENT)
    -> tuple[ScalarField, ReconstructionDiagnostics]
```

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
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any, Literal

import numpy as np

from numerics import (
    PHASE_ACCUMULATION_FLOOR,
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
    "SCALE_NOTE",
    "Normalization",
    "Projection",
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

    reconstructed_discrete_power: float
    scale: str = SCALE_NOTE

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
    if normalization is None:  # pragma: no cover - a new measure kind lands with its row
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


def ray_to_scalar(
    rays: RayBundle,
    *,
    grid_shape: tuple[int, int],
    sample_pitch_m: tuple[float, float],
    surface: ReferenceSurface | None = None,
    projection: Projection = Projection.ASM_CONSISTENT,
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
    if complex_dtype is None:  # pragma: no cover - guarded by the FP32 floor
        raise ContractError(
            "MISSING_DECLARATION",
            f"{precision} has no complex dtype to reconstruct a field in",
            declaration="compute_precision",
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
        reconstructed_discrete_power=field.discrete_power(),
    )
    return field, diagnostics
