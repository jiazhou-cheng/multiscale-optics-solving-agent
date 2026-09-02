"""The collimated ensemble against the analytic wavelet sum, and against `plane_wave`.

CHE-215 (R06.10), the physics half of item 1, re-pointed by CHE-219 (R05.8) at
`fixtures.ray_bundles` -- the same arithmetic, moved out of `src/sources/` because
a launch `RayBundle` built from caller-supplied points has no optical system in
scope and therefore cannot be a source. `test_collimated_ensemble.py` holds the
declaration contract; this file is the one that measures whether the bundle the
builder produces is *physically* the plane-wave mode it claims to be.

Oracle discipline, stated because the two checks here are not the same kind
of evidence (`AGENTS.md`, "Scientific Non-Negotiables")
-------------------------------------------------------------------------------
**The gate** is the closed form. A collimated ensemble whose every ray carries
`OPL_j = n (d_hat . r_j)` *is* one plane-wave mode, so the coherent wavelet sum
`couplers.ray_to_scalar` performs has an analytic value:

    U(r) = N dA exp(+i n k0 d_hat . r)

That number came off paper, not out of this repository, so the tolerance is dtype
round-off rather than a choice. It is the same oracle `tests/physics/
ray_support.collimated_bundle` documents -- which is the point: that helper
delegates to the same builder, so the reconstruction gates R07 landed and the
checks here run against one ensemble rather than two hand-built twins.

**Not a gate**, and labelled so: the agreement between `collimated_bundle ->
ray_to_scalar` and `sources.plane_wave` at the matching `k_t`. Both sides are
repository numerical code, so this is *shared-code characterization* -- useful
evidence that the ray path and the wave path state the same illumination in the
same units, and specifically that the `(x, y, z)` direction cosine and the
`(y, x)` wavevector describe one mode and not two. It cannot establish that either
side is right. It is asserted at a loose tolerance for that reason and the
closed-form gate above is what would fail if the physics were wrong.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from fixtures.ray_bundles import collimated_bundle, direction_from_angle

from couplers.ray_to_scalar import ray_to_scalar
from representations import Frame, ReferenceSurface
from sources import plane_wave, transverse_wavevector_from_angle

WAVELENGTH_M = 0.532e-6

#: Non-square in both count and pitch: an axis-symmetric fixture cannot fail on a
#: transposed `(x, y)` column stack, which is this source's declared trap.
SHAPE = (32, 40)
PITCH_M = (0.30e-6, 0.25e-6)


def a_surface(*, medium_index: float = 1.0) -> ReferenceSurface:
    return ReferenceSurface(name="handoff", z_m=0.0, medium_index=medium_index)


def launch_points(shape: tuple[int, int], pitch: tuple[float, float]) -> np.ndarray:
    """`(N, 3)` points as `(x, y, z)` columns from a `(ny, nx)` grid of pitch `(dy, dx)`."""
    frame = Frame()
    y = (np.arange(shape[0], dtype=np.float64) - frame.origin_index(shape[0])) * pitch[0]
    x = (np.arange(shape[1], dtype=np.float64) - frame.origin_index(shape[1])) * pitch[1]
    grid_y, grid_x = np.meshgrid(y, x, indexing="ij")
    return np.column_stack([grid_x.ravel(), grid_y.ravel(), np.zeros(grid_x.size)])


def a_collimated_field(
    direction: tuple[float, float, float],
    *,
    medium_index: float = 1.0,
    shape: tuple[int, int] = SHAPE,
    pitch: tuple[float, float] = PITCH_M,
):
    """Reconstruct the source's bundle onto the same grid its points came from."""
    positions = launch_points(shape, pitch)
    area = pitch[0] * pitch[1]
    rays = collimated_bundle(
        positions,
        direction=direction,
        wavelength_m=WAVELENGTH_M,
        reference_surface=a_surface(medium_index=medium_index),
        measure_weight=np.full(positions.shape[0], area),
        measure_kind="quadrature_area_m2",
    )
    field, diagnostics = ray_to_scalar(rays, grid_shape=shape, sample_pitch_m=pitch)
    return rays, field, diagnostics, area


# ---------------------------------------------------------------------------
# The gate: the closed-form wavelet sum
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("theta", [0.0, 0.20, 0.35])
@pytest.mark.parametrize("phi", [0.0, 0.5 * math.pi, 0.7])
def test_the_source_reconstructs_to_the_analytic_plane_wave_mode(
    theta: float, phi: float
) -> None:
    """`U(r) = N dA exp(+i k0 d_hat . r)`, to dtype round-off.

    The direction comes from `direction_from_angle`, so the azimuth convention is
    on the gate rather than only in a unit test: `phi = 0` tilts in `+x` and
    `phi = pi/2` in `+y`, and swapping them would move the reconstructed ramp onto
    the other axis, which on this non-square grid the oracle catches.

    Normal incidence (`theta = 0`) is in the sweep because it is the default, and
    it is the one case where a wrong optical path is *invisible* -- every `OPL_j`
    is zero, so the mode is flat either way.
    """
    direction = direction_from_angle(theta, phi)
    rays, field, diagnostics, area = a_collimated_field(direction)

    y, x = field.coordinates()
    grid_y, grid_x = np.meshgrid(y, x, indexing="ij")
    oracle = (
        rays.count
        * area
        * np.exp(1j * rays.wavenumber * (direction[0] * grid_x + direction[1] * grid_y))
    )

    residual = np.max(np.abs(np.asarray(field.u) - oracle)) / np.max(np.abs(oracle))
    assert residual < 1e-13
    assert diagnostics.measure_kind == "quadrature_area_m2"
    assert diagnostics.normalization == "none"
    assert field.validity == frozenset({"surface_only", "no_wavefront_curvature_term"})


@pytest.mark.parametrize("medium_index", [1.336, 1.5168])
def test_the_optical_path_the_source_writes_carries_the_medium(medium_index: float) -> None:
    """In a medium the oracle is `N dA exp(+i n k0 d_hat . r)`, and both halves move.

    The source multiplies the geometric projection by `n` and the coupler's
    transverse ramp carries `n` (CHE-192). Those are two independent places, and if
    only one of them had the index the reconstruction would be wrong by
    `(n - 1) k0 d_t . dr` -- unbounded in waves. Air is covered by the sweep above,
    so this is the case that could not pass by accident.
    """
    direction = direction_from_angle(0.25, 0.4)
    rays, field, _, area = a_collimated_field(direction, medium_index=medium_index)

    y, x = field.coordinates()
    grid_y, grid_x = np.meshgrid(y, x, indexing="ij")
    oracle = (
        rays.count
        * area
        * np.exp(
            1j
            * medium_index
            * rays.wavenumber
            * (direction[0] * grid_x + direction[1] * grid_y)
        )
    )

    assert np.max(np.abs(np.asarray(field.u) - oracle)) / np.max(np.abs(oracle)) < 1e-13


def test_a_conjugated_optical_path_fails_the_same_gate() -> None:
    """The negative control: the gate has to be able to fail.

    A tolerance of 1e-13 against an analytic oracle means nothing unless a wrong
    ensemble misses it, and the wrong ensemble worth testing is the *conjugate* --
    `OPL -> -OPL`, which reverses every wavelet's phase and turns a converging
    wavefront into a diverging one with no signature in `|U|`. Built here by
    negating the source's own output, so what is measured is that the sign the
    source chose is load-bearing.
    """
    import dataclasses

    direction = direction_from_angle(0.35, 0.0)
    rays, field, _, area = a_collimated_field(direction)

    conjugated = dataclasses.replace(rays, optical_path_m=-np.asarray(rays.optical_path_m))
    wrong, _ = ray_to_scalar(conjugated, grid_shape=SHAPE, sample_pitch_m=PITCH_M)

    y, x = field.coordinates()
    grid_y, grid_x = np.meshgrid(y, x, indexing="ij")
    oracle = (
        rays.count
        * area
        * np.exp(1j * rays.wavenumber * (direction[0] * grid_x + direction[1] * grid_y))
    )

    residual = np.max(np.abs(np.asarray(wrong.u) - oracle)) / np.max(np.abs(oracle))
    assert residual > 0.5, "the conjugate ensemble must not pass the gate it is a control for"
    # ...and it is invisible in intensity, which is why the control is needed.
    assert np.allclose(np.abs(np.asarray(wrong.u)), np.abs(oracle), rtol=1e-6)


# ---------------------------------------------------------------------------
# Not a gate: consistency between two repository paths
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("phi", [0.0, 0.5 * math.pi])
def test_the_ray_path_and_the_wave_path_state_the_same_illumination(phi: float) -> None:
    """Shared-code characterization, **not** an independent oracle. See the module docstring.

    Both sides are this repository's numerical code, so agreement cannot establish
    that either is right -- `AGENTS.md` forbids reading it that way. What it does
    establish is that the package's two angle converters describe **one** mode:
    `direction_from_angle` returns `(d_x, d_y, d_z)` and
    `transverse_wavevector_from_angle` returns `(k_y, k_x)`, opposite axis orders,
    and a caller who transposed one would get a plausible tilt on the wrong axis.
    `phi = pi/2` is in the sweep for exactly that reason.
    """
    theta = 0.30
    direction = direction_from_angle(theta, phi)
    rays, field, _, area = a_collimated_field(direction)

    illumination = plane_wave(
        SHAPE,
        sample_pitch_m=PITCH_M,
        wavelength_m=WAVELENGTH_M,
        reference_surface=a_surface(),
        transverse_wavevector_rad_per_m=transverse_wavevector_from_angle(
            theta, phi, wavelength_m=WAVELENGTH_M, medium_index=1.0
        ),
    )

    scale = rays.count * area
    residual = np.max(
        np.abs(np.asarray(field.u) / scale - np.asarray(illumination.u))
    )
    # Loose on purpose: `plane_wave` stores complex64 while the reconstruction is
    # float64, so ~1e-7 is the cast and not a physical disagreement.
    assert residual < 1e-5
