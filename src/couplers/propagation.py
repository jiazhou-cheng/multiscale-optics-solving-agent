"""Propagation: moving a bundle between planes, and nothing else (CHE-142).

This module exists to make one distinction structural rather than merely
documented. ``src/couplers/`` holds three kinds of operation and they are not
interchangeable:

* a **representation transition** changes what the light *is described by* --
  ``couplers/ray_to_wave.py``, ``couplers/wave_to_ray.py``;
* a **diffractive interaction** is physics at a surface: incident coherent rays
  meet a diffractive surface and coherent rays come out --
  ``couplers/interaction.py`` and the two models under it;
* **propagation** moves an existing representation from one plane to another and
  changes neither the representation nor the physical content.

:func:`advance_bundle_to_plane` is the third, and it lived in
``couplers/patch.py`` -- inside a diffractive-interaction model -- purely
because the patch route was its first caller. Nothing about it is patch-specific
and nothing about it diffracts. ``couplers.patch`` re-exports it so every
existing caller keeps working; see the note there for how long.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from core.boundary import ContractCode, ContractError, RayBundle, ReferencePlane

__all__ = ["advance_bundle_to_plane"]


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
