"""`RayBundle(S1) -> RayBundle(S2)`: physical evolution through one declared medium.

CHE-192 (R09.2). One public function and no classes:

```python
operators.propagate_rays(rays, *, to, phase_budget_rad=1e-2) -> RayBundle
```

Propagation is *physical evolution through a declared medium from one declared
reference surface to another, without changing representation and without
separately applying a discrete optical-surface interaction.* Three consequences,
and each of them is a decision this module makes rather than inherits:

* It is **not** a coupler. The representation on both sides is a `RayBundle`;
  what changes is the state. The reference implementation kept ray propagation in
  `couplers/`, and under this definition that was wrong -- the module even said
  so, listing propagation as the third of three things `couplers/` held.
* It lives in `operators/`, which is where a project-owned *independently
  selectable* operation belongs, and it is selectable: it is CHE-50's declared
  remedy (below), and it is the step a caller takes between two reconstructions.
* Wave propagation is **not** here. Chromatix owns those numerics and
  `backends.chromatix.propagate` exposes them; a wrapper in this package added
  purely to relocate semantic ownership would do no numerical work, and R09.1
  bans exactly that.

Why this operation carries more weight than its size
----------------------------------------------------
CHE-50's declared remedy for the wavelet sum's missing wavefront-curvature term
is *this* operation: "to obtain a field on a different plane, advance the ray
state to that plane and reconstruct there. That is exact, not an approximation."
So `propagate_rays` is not a convenience. It is the sanctioned alternative to
something `couplers.ray_to_scalar` refuses to do, and that refusal is only
defensible if this is correct.

The exactness claim, worked
---------------------------
Each ray advances along its own direction by the arc length `s = dz / d_z` for a
plane offset `dz`, and its optical path grows by `n s`. That is exact rather than
paraxial, and here is why. `couplers.ray_to_scalar` forms each wavelet's constant
phase as `k (OPL - n d_t . x0_t)`; after the advance,

    C2 = OPL + n s - n d_t . (x0_t + d_t s)  =  C1 + n s (1 - |d_t|^2)  =  C1 + n s d_z^2,

and `s d_z^2 = dz d_z`, so the phase changed by `n k d_z dz` -- precisely what a
plane wave of wavevector `n k d_hat` accumulates over an axial offset `dz`. No
term is dropped. The reconstruction at the new surface is not an approximation of
the field there; it is the same superposition of plane wavelets evaluated there.

**That composed claim is executable at any `n`.** The derivation above writes the
coupler's constant phase as `k (OPL - n d_t . x0_t)`, which is what
`couplers.ray_to_scalar` now forms -- the `n` in its transverse ramp and in its
launch-ramp subtraction is the CHE-192 follow-up, made after R09 found it missing.
So a bundle advanced through glass has an optical path the reconstruction reads on
the same convention, and the pair composes in a medium as well as in air.

**The refractive index is the whole risk of this ticket**, and the arithmetic
above is why: `n` appears twice and has to appear in both places. A version that
adds the geometric distance `s` to the optical path is right in air and silently
wrong everywhere else -- including inside a lens, which is where a sequential
trace actually uses it. The reference implementation hard-coded `n = 1` in both of
its copies and had no test that would have caught it;
`tests/physics/test_ray_propagation.py` is that test, with the negative control.

The medium index in the reconstruction kernel, found here
---------------------------------------------------------
Deriving the above turned up that neither coupler carried the index:
`couplers.ray_to_scalar` wrote `k0 (OPL + d_hat . dr)` where the medium form is
`k0 OPL + n k0 d_hat . dr`, and `couplers.scalar_to_ray`'s direction cosines were
`lambda_vacuum f`, the `n = 1` form. Both were undeclared assumptions rather than
wrong numbers -- every case the tree had run was in air.

R09 refused rather than fixed, because the fix alters a landed physical convention
in two already-committed tickets. The owner then took it: `n` is in both couplers,
a no-op at `n = 1`, and the refusals are gone. This operation's `n s` and their
`n k0` ramps are one convention now, which is what the exactness claim above
depends on.

One medium, asserted rather than assumed
----------------------------------------
`to.medium_index` must equal the index the rays are declared in. Two surfaces
that declare different media do not bound one medium, so a single propagation
step between them is not defined -- reaching the second would require a discrete
surface interaction, which this operation excludes by definition. Refused rather
than resolved by picking one of the two indices, which is the move that would put
a silent `n = 1` back.

The measure does not change, and that is not an omission
--------------------------------------------------------
`measure_weight` and `measure_kind` pass through untouched. The quadrature that
established each wavelet's coefficient was taken at the surface the rays were
*originally* declared on, and a plane wavelet is an infinite plane wave whose
coefficient is fixed once. Advancing where the wavelet is *stated* to cross a
plane does not change what it is. A version that rescaled the cell areas by a
geometric Jacobian would be double-counting the convergence that the phases
already carry.

What is refused, and why each refusal rather than a drop
--------------------------------------------------------
A bundle that quietly loses members on a transfer produces a plausible field with
missing power, so nothing is dropped:

* a ray travelling **away** from the target, or exactly parallel to it, never
  reaches it. The reference implementation's two copies disagreed here -- one
  refused, the other returned a *negative* arc and silently propagated the ray
  backwards, subtracting from its optical path. That divergence is why the
  refusing one is authoritative (R09.1);
* a ray whose `d_z` is small enough that its arc length carries an optical path
  the compute precision cannot represent. The floor is **derived, not chosen**:
  it is `couplers.grazing_floor_for_phase_budget` with the axial offset as the
  extent, which is the same `eps k Z / d_n` bound R07.4 ported -- the same physics,
  because a 5 mm offset at `d_z = 1e-12` is a 5e9 m optical path. Both reference
  copies admitted that ray, one behind a fixed `1e-12` absolute cut and the other
  behind exact equality with zero;
* a bundle with no optical path. This operation exists to evolve one, so there is
  nothing for it to do; advancing the positions and leaving the path absent would
  produce a bundle no reconstruction can read.

This module imports `couplers` for that floor, and it is the first use of the
`operators -> couplers` edge the allowlist has always permitted. It imports no
solver and no backend: advancing a ray along its own direction happens in whatever
array namespace the bundle already carries.
"""

from __future__ import annotations

import dataclasses

from couplers import DEFAULT_PHASE_BUDGET_RAD, grazing_floor_for_phase_budget
from numerics import dtype_of, numpy_dtype
from representations import ContractError, RayBundle, ReferenceSurface

__all__ = ["propagate_rays"]


def _require_one_medium(rays: RayBundle, to: ReferenceSurface) -> float:
    """The index of the single medium between the two surfaces, or a refusal."""
    source = rays.reference_surface
    if to.normal != rays.frame.propagation_normal:
        raise ContractError(
            "FRAME_MISMATCH",
            f"the target surface declares the normal {to.normal}, not "
            f"{rays.frame.propagation_normal}. The advance is `s = dz / d_z` along the "
            "propagation axis, which is defined only for a plane perpendicular to it; a "
            "tilted target needs a plane-intersection solve this operation does not do.",
            declaration="to.normal",
        )
    if to.medium_index != source.medium_index:
        raise ContractError(
            "MISSING_DECLARATION",
            f"the rays are declared in a medium of index {source.medium_index!r} and the "
            f"target surface in one of {to.medium_index!r}, so the two do not bound a "
            "single medium and there is no one index for the optical path to grow by. "
            "Reaching the target would require a discrete surface interaction, which this "
            "operation excludes by definition.",
            declaration="to.medium_index",
            remedy=(
                "Propagate to a surface in the same medium and apply the interaction "
                "there, or declare the medium the rays actually travel through."
            ),
        )
    return float(source.medium_index)


def propagate_rays(
    rays: RayBundle,
    *,
    to: ReferenceSurface,
    phase_budget_rad: float = DEFAULT_PHASE_BUDGET_RAD,
) -> RayBundle:
    """Advance `rays` to the surface `to`, along each ray's own direction.

    Parameters
    ----------
    rays
        A bundle carrying an optical path with a declared reference. The amplitude
        is not required and is passed through unchanged -- propagation through a
        transparent medium does not change a wavelet's coefficient.
    to
        The target surface. Must be perpendicular to the propagation axis and must
        declare the same medium the rays are in.
    phase_budget_rad
        The phase budget the `d_z` floor is derived against, in radians. The same
        constant and the same derivation `couplers.ray_to_scalar` uses, because it
        is the same hazard: a near-grazing ray's arc length is an optical path
        whose phase the compute precision cannot carry.

    Returns
    -------
    A bundle on `to`, with positions advanced, directions unchanged, the optical
    path grown by `n * s` per ray, and the sampling measure passed through. The
    optical-path reference records the advance, so a consumer can see that the path
    is no longer measured to where the rays were first declared.

    Raises
    ------
    ContractError
        `MISSING_DECLARATION` for a target in a different medium or a bundle with
        no optical path; `FRAME_MISMATCH` for a tilted target or a ray that never
        reaches it; `GRAZING_PHASE_UNREPRESENTABLE` for a ray whose arc length
        carries an unrepresentable phase.
    """
    medium_index = _require_one_medium(rays, to)
    if rays.optical_path_m is None:
        raise ContractError(
            "MISSING_DECLARATION",
            "this bundle carries no optical path, and evolving one is what this operation "
            "does. Advancing the positions while leaving the path absent would produce a "
            "bundle no reconstruction can read as coherent.",
            declaration="optical_path_m",
            remedy="Declare the optical path and its reference at the producer.",
        )

    xp = rays.xp
    real_np = numpy_dtype(rays.state.dtype.precision.real_dtype)
    positions = rays.positions_m
    directions = rays.directions
    axial_offset = float(to.z_m) - positions[:, 2]
    axial_direction = directions[:, 2]

    # Never reaches the target: parallel to it, or pointing away from it. Refused
    # rather than dropped, because a bundle that quietly loses members on a
    # transfer produces a plausible field with missing power.
    unreachable = (axial_direction == 0.0) | (axial_offset * axial_direction < 0.0)
    if bool(xp.any(unreachable)):
        raise ContractError(
            "FRAME_MISMATCH",
            f"{int(xp.sum(unreachable))} of {rays.count} rays never reach the surface "
            f"{to.name!r} at z = {to.z_m:.6e} m: they are parallel to it or travel away "
            "from it. Refused rather than dropped -- a bundle that quietly loses members "
            "produces a plausible field with missing power.",
            declaration="directions",
            remedy="Propagate to a surface the whole bundle reaches, or split the bundle.",
        )

    # The floor on |d_z|, derived from the phase the arc length would carry rather
    # than picked. `grazing_floor_for_phase_budget` solves `eps k Z / d_n <= budget`
    # for `d_n` with `Z` the axial extent; here that extent is the offset each ray
    # traverses, so the worst offset in the bundle is what the floor is taken at.
    worst_offset_m = float(xp.max(xp.abs(axial_offset)))
    floor = (
        grazing_floor_for_phase_budget(
            wavelength_m=rays.wavelength_m,
            max_optical_path_m=medium_index * worst_offset_m,
            precision=dtype_of(rays.optical_path_m).precision,
            phase_budget_rad=phase_budget_rad,
        )
        if worst_offset_m > 0.0
        else 0.0
    )
    below = xp.abs(axial_direction) < floor
    if bool(xp.any(below)):
        raise ContractError(
            "GRAZING_PHASE_UNREPRESENTABLE",
            f"{int(xp.sum(below))} of {rays.count} rays have |d_z| below {floor:.3e}, so "
            f"their arc length over the {worst_offset_m:.3e} m offset carries an optical "
            "path whose phase this precision cannot represent to within "
            f"{phase_budget_rad} rad. The floor is derived from that bound, not chosen; "
            "the reference implementation's fixed 1e-12 cut admitted a ray whose arc was "
            "5e9 m.",
            declaration="directions",
            remedy=(
                "Restrict the bundle's directions, propagate a shorter distance, or "
                "compute the optical path in a higher precision."
            ),
        )

    arc = (axial_offset / axial_direction).astype(real_np)
    return dataclasses.replace(
        rays,
        positions_m=positions + directions * arc[:, None],
        # Directions unchanged: this is propagation of the ray *state*, not a change
        # of the ray model, and nothing here refracts.
        reference_surface=to,
        optical_path_m=rays.optical_path_m + (medium_index * arc).astype(real_np),
        optical_path_reference=(
            f"{rays.optical_path_reference}, then advanced along each ray's own direction "
            f"to {to.name!r} at z = {to.z_m:.6e} m through a medium of index "
            f"{medium_index!r} (optical path grew by n * s, s the arc length)"
        ),
    )
