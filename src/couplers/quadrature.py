"""Per-ray pupil quadrature weight for a hexapolar ray fan (CHE-38 S14/S15, CHE-47).

CHE-38 (M3.9R) found that the dominant sensor-plane residual of ``C_RAY_TO_WAVE``
is not a kernel defect: the coherent wavelet sum

    U(r) = sum_i a_i * exp[ i k ( OPL_i + dr_i(r) ) ]

is a quadrature over the ray ensemble, and treating every ray as an equal-weight
sample is the wrong quadrature at the aperture boundary. A hexapolar ring set is
very nearly equal-area in the interior -- ring ``j`` carries ``6j`` rays over an
annulus of area proportional to ``j`` -- but wrong at the two edges: the central
ray represents a smaller cell than a naive uniform assignment implies, and the
outermost ring sits exactly on the aperture rim and represents only the inner
half of its cell (there is no ray beyond it to average with). CHE-38 measured
that a diagnostic radial-trapezoid correction (center 3/4, rim 1/2, interior 1)
collapses the residual from ``3.84e-3`` to a converged ``4.07e-4`` at 787 969
rays, on a synthetic aberration-free bundle. Nothing in that ticket changed
production code (CHE-38 section 14); this module is the ticket that does.

This module is pure math over ring indices and carries no unit assumptions.
The absolute area scale (:func:`hexapolar_area_weight_m2`) is what resolves
CHE-33's ``N^2.0024`` raw-power scaling: assigning ray ``i`` the *physical*
area element ``dA_i`` its quadrature cell represents, rather than a bare
relative correction factor, makes ``sum_i dA_i -> aperture area`` as the ring
count grows, instead of staying pinned at a constant regardless of ray density.
That is what makes the reconstructed discrete power converge under ray
refinement rather than grow as (ray count)^2.

Where the weight is applied
----------------------------
Nowhere in ``C_RAY_TO_WAVE`` (``ray_to_wave.py``). The kernel sums whatever
amplitude a bundle declares; this module only helps a *producer* -- the
Optiland adapter, via ``optiland_handoff.py`` -- declare a better amplitude.
That keeps the validated sensor-side kernel semantics (CHE-24, CHE-38)
untouched, per this ticket's requirement.
"""

from __future__ import annotations

import math

import numpy as np

from core.boundary import ContractCode, ContractError

__all__ = [
    "hexapolar_area_weight_m2",
    "hexapolar_ring_index",
]

#: Absolute tolerance, in units of ring spacing (``1 / num_rings``), for a ray's
#: normalized pupil radius to be accepted as landing exactly on a ring. Optiland's
#: hexapolar distribution places ring ``j`` at exactly ``rho = j / num_rings`` in
#: float64 (``optiland.distribution.HexagonalDistribution``: ``r = linspace(0, 1,
#: num_rings + 1)``), so a real (non-vignetted) hexapolar ray fan agrees with this
#: to float64 round-off. A hand-built or vignetted bundle that does not land
#: within tolerance is refused rather than silently mis-binned.
_RING_TOLERANCE = 1.0e-6


def hexapolar_ring_index(pupil_x: np.ndarray, pupil_y: np.ndarray, num_rings: int) -> np.ndarray:
    """Recover each ray's hexapolar ring index from its normalized pupil coordinate.

    ``pupil_x``, ``pupil_y`` are the normalized entrance-pupil coordinates
    Optiland calls ``Px``, ``Py`` -- the unit disk a hexapolar distribution with
    ``num_rings`` rings was sampled on, ring ``j`` at radius ``j / num_rings``
    for ``j = 0 .. num_rings`` (``j = 0`` is the single central ray).

    Raises:
        ContractError: with :attr:`ContractCode.NON_HEXAPOLAR_SAMPLING` if any
            ray's normalized radius does not land within tolerance of
            ``j / num_rings`` for an integer ``j`` -- e.g. the bundle was
            vignetted (a ray dropped, shifting which rows correspond to which
            ring) or was never a hexapolar fan at all.
    """
    if num_rings < 1:
        raise ContractError(
            ContractCode.NON_HEXAPOLAR_SAMPLING,
            f"num_rings must be >= 1, got {num_rings!r}",
            declaration="num_rings",
        )
    pupil_x = np.asarray(pupil_x, dtype=np.float64)
    pupil_y = np.asarray(pupil_y, dtype=np.float64)
    if pupil_x.shape != pupil_y.shape:
        raise ContractError(
            ContractCode.SHAPE_MISMATCH,
            f"pupil_x {pupil_x.shape} must match pupil_y {pupil_y.shape}",
            declaration="pupil_y",
        )

    rho = np.hypot(pupil_x, pupil_y)
    ring_float = rho * num_rings
    ring_index = np.round(ring_float).astype(np.int64)
    residual = np.abs(ring_float - ring_index)
    bad = residual > _RING_TOLERANCE
    if np.any(bad):
        worst = int(np.argmax(residual))
        raise ContractError(
            ContractCode.NON_HEXAPOLAR_SAMPLING,
            (
                f"{int(np.sum(bad))} of {pupil_x.size} rays do not land on a "
                f"hexapolar ring of a {num_rings}-ring fan; worst offender has "
                f"normalized radius {float(rho[worst]):.9f} against the nearest "
                f"ring at {float(ring_index[worst]) / num_rings:.9f} "
                f"(residual {float(residual[worst]):.3e} rings). This usually means "
                "the bundle was vignetted (a ray dropped, so row order no longer "
                "matches ring order) or is not a hexapolar fan at all."
            ),
            declaration="pupil_x, pupil_y",
            remedy=(
                "Only assign a quadrature weight to an un-vignetted hexapolar "
                "fan, where the traced ray count equals 1 + 3*num_rings*"
                "(num_rings + 1). Otherwise leave the weight unavailable."
            ),
        )
    out_of_range = (ring_index < 0) | (ring_index > num_rings)
    if np.any(out_of_range):
        raise ContractError(
            ContractCode.NON_HEXAPOLAR_SAMPLING,
            "a ray's recovered ring index falls outside [0, num_rings]; the "
            "pupil coordinates exceed the unit disk",
            declaration="pupil_x, pupil_y",
        )
    return ring_index


def hexapolar_area_weight_m2(
    ring_index: np.ndarray, num_rings: int, aperture_radius_m: float
) -> np.ndarray:
    """Absolute per-ray quadrature (area) weight, in square metres.

    Ring ``j``'s nominal cell area is ``pi * a^2 / (3 * num_rings^2)`` -- the
    aperture area divided by the number of rays a hexapolar fan asymptotically
    approaches (``3 * num_rings^2`` for large ``num_rings``, exactly
    ``3*n^2 + 3*n + 1`` including the center). Every interior ring's ``6j``
    points share that nominal cell exactly, which is the right quadrature
    there (CHE-38: the residual against this assumption alone falls as
    ``rings^-0.87``, a boundary rate, not an interior one). Two boundary
    corrections, CHE-38's measured trapezoid weights:

    * the central ray (``ring_index == 0``) gets ``3/4`` of the nominal cell:
      it represents a disk of radius ``a / (2 * num_rings)``, i.e. area
      ``pi a^2 / (4 num_rings^2)``, three quarters of the nominal cell.
    * the outermost ring (``ring_index == num_rings``) gets ``1/2`` the nominal
      cell: it sits exactly on ``rho = a`` and represents only the inner half
      of its annular cell, since there is no ray beyond the rim to average
      with.

    With these weights ``sum_i weight_i = pi * a^2 * (1 + 1/(4 * num_rings^2))``
    exactly (worked from the ring counts ``6j``), converging to the true
    aperture area as ``num_rings`` grows -- which is what makes a reconstructed
    field's discrete power converge under ray refinement instead of growing
    as ``(ray count)^2`` (CHE-33's finding, closed by this function).
    """
    if num_rings < 1:
        raise ContractError(
            ContractCode.NON_HEXAPOLAR_SAMPLING,
            f"num_rings must be >= 1, got {num_rings!r}",
            declaration="num_rings",
        )
    if not math.isfinite(aperture_radius_m) or aperture_radius_m <= 0.0:
        raise ContractError(
            ContractCode.UNIT_NOT_SI,
            f"aperture_radius_m must be a positive finite value, got {aperture_radius_m!r}",
            declaration="aperture_radius_m",
        )
    ring_index = np.asarray(ring_index)

    nominal_area_m2 = math.pi * aperture_radius_m**2 / (3.0 * num_rings**2)
    weight = np.full(ring_index.shape, nominal_area_m2, dtype=np.float64)
    weight[ring_index == 0] = 0.75 * nominal_area_m2
    weight[ring_index == num_rings] = 0.5 * nominal_area_m2
    return weight
