"""R09.2: ray propagation, the refractive index, and the test the old tree lacked.

CHE-192. This is one of the few places the rewrite *adds* coverage rather than
porting it. The reference implementation had two `advance_bundle_to_plane`
implementations and no dedicated test file for either; the behaviour was covered
incidentally inside the coupler and adapter tests, and neither of those exercised
a medium of index other than 1.

**The refractive index is the whole risk.** Advancing a ray a geometric distance
`s` through index `n` grows its optical path by `n s`, and a version that adds `s`
is right in air and silently wrong everywhere else -- including inside a lens,
which is where a sequential trace actually uses it. Both reference copies
hard-coded `n = 1`. Every acceptance number below is therefore taken at `n = 1.5`
as well as in air, and the negative control is the geometric-distance version.

Why the operation matters more than its size
--------------------------------------------
CHE-50's declared remedy for the wavelet sum's missing wavefront-curvature term is
this operation: advance the ray state and reconstruct there, which is exact rather
than an approximation. `test_advancing_then_reconstructing_is_exact` is the gate
that makes that remedy trustworthy, and
`test_the_two_routes_diverge_where_the_curvature_term_would` measures where the
alternative it replaces fails.
"""

from __future__ import annotations

import ast
import dataclasses
import math
from pathlib import Path

import numpy as np
import pytest
from ray_support import WAVELENGTH_M, a_surface, collimated_bundle

from couplers import DEFAULT_PHASE_BUDGET_RAD, ray_to_scalar
from operations import CATALOG, OperationKind, resolve
from operators import propagate_rays
from representations import ContractError, RayBundle, ReferenceSurface

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"

SHAPE = (24, 32)
PITCH_M = (0.30e-6, 0.25e-6)


def a_bundle(
    *,
    thetas=(0.0, 0.08, 0.15),
    z_m: float = 0.0,
    medium_index: float = 1.0,
    optical_path_m=None,
    dtype=np.float64,
) -> RayBundle:
    """A few rays at declared angles, on a surface in a declared medium."""
    directions = np.array(
        [[math.sin(t), 0.0, math.cos(t)] for t in thetas], dtype=np.float64
    )
    count = directions.shape[0]
    positions = np.column_stack(
        [np.linspace(-1e-4, 1e-4, count), np.zeros(count), np.full(count, z_m)]
    )
    return RayBundle(
        positions_m=positions.astype(dtype),
        directions=directions.astype(dtype),
        wavelength_m=WAVELENGTH_M,
        reference_surface=a_surface("start", z_m=z_m, medium_index=medium_index),
        amplitude=np.ones(count, dtype=np.complex128),
        optical_path_m=(
            np.zeros(count) if optical_path_m is None else np.asarray(optical_path_m)
        ).astype(dtype),
        optical_path_reference="the starting surface",
        measure_weight=np.full(count, 1.0e-12, dtype=dtype),
        measure_kind="quadrature_area_m2",
    )


def arc_lengths(rays: RayBundle, z_m: float) -> np.ndarray:
    """`s = dz / d_z` per ray, written out here rather than read from the operator."""
    positions = np.asarray(rays.positions_m, dtype=np.float64)
    directions = np.asarray(rays.directions, dtype=np.float64)
    return (z_m - positions[:, 2]) / directions[:, 2]


# ---------------------------------------------------------------------------
# 1. The optical path, against a closed form, in a medium
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("medium_index", [1.0, 1.5, 1.336])
def test_the_optical_path_grows_by_n_times_the_arc_length(medium_index: float) -> None:
    """Criterion 1. `OPL' = OPL + n s`, with `s = dz / d_z` -- closed form, any n.

    The geometry is closed-form per ray and needs no oracle library: a ray at polar
    angle `theta` crossing an axial offset `dz` travels `dz / cos(theta)`, and its
    optical path grows by `n` times that. Checked at three indices including air, so
    the `n = 1` case cannot be the only one that passes.
    """
    offset_m = 4.0e-3
    rays = a_bundle(medium_index=medium_index)
    target = a_surface("end", z_m=offset_m, medium_index=medium_index)

    advanced = propagate_rays(rays, to=target)

    expected_arc = offset_m / np.cos([0.0, 0.08, 0.15])
    assert np.allclose(arc_lengths(rays, offset_m), expected_arc, rtol=1e-12)
    assert np.allclose(
        np.asarray(advanced.optical_path_m),
        np.asarray(rays.optical_path_m) + medium_index * expected_arc,
        rtol=1e-12,
    )
    # Geometry: each ray lands on the target plane, displaced by its own direction.
    positions = np.asarray(advanced.positions_m)
    assert np.allclose(positions[:, 2], offset_m, rtol=0, atol=1e-18)
    assert np.allclose(
        positions[:, 0],
        np.asarray(rays.positions_m)[:, 0] + np.sin([0.0, 0.08, 0.15]) * expected_arc,
        rtol=1e-12,
    )
    # Directions and the sampling measure pass through untouched.
    assert np.array_equal(np.asarray(advanced.directions), np.asarray(rays.directions))
    assert np.array_equal(
        np.asarray(advanced.measure_weight), np.asarray(rays.measure_weight)
    )
    assert advanced.measure_kind == rays.measure_kind
    assert advanced.reference_surface == target


def test_the_negative_control_adding_geometric_distance_fails_in_a_medium() -> None:
    """Criterion 2. The version both reference copies shipped, and where it is wrong.

    Adding the geometric arc length instead of `n s` is **exactly right in air** and
    wrong by `(n - 1) s` everywhere else. At `n = 1.5` over a 4 mm offset that is
    2 mm of optical path -- about 3600 waves at 550 nm, so it is not a tolerance
    question. The point of the control is that no test in air could ever see it,
    which is why the reference implementation's incidental coverage did not.
    """
    offset_m = 4.0e-3
    medium_index = 1.5
    rays = a_bundle(medium_index=medium_index)
    target = a_surface("end", z_m=offset_m, medium_index=medium_index)

    advanced = propagate_rays(rays, to=target)
    arc = arc_lengths(rays, offset_m)
    geometric_only = np.asarray(rays.optical_path_m) + arc

    assert not np.allclose(np.asarray(advanced.optical_path_m), geometric_only)
    shortfall = np.asarray(advanced.optical_path_m) - geometric_only
    assert np.allclose(shortfall, (medium_index - 1.0) * arc, rtol=1e-12)
    assert shortfall.min() > 1.9e-3
    waves = float(shortfall.min()) / WAVELENGTH_M
    assert waves > 3.4e3, waves

    # ...and in air the two are identical, which is the half that let it survive.
    in_air = a_bundle(medium_index=1.0)
    air_target = a_surface("end", z_m=offset_m, medium_index=1.0)
    air = propagate_rays(in_air, to=air_target)
    assert np.allclose(
        np.asarray(air.optical_path_m),
        np.asarray(in_air.optical_path_m) + arc_lengths(in_air, offset_m),
        rtol=1e-12,
    )


def test_two_advances_compose_into_one() -> None:
    """A propagation is a propagation: S1 -> S2 -> S3 equals S1 -> S3.

    Not a tautology of the arithmetic -- it holds only because the directions are
    unchanged, so it is the property that says nothing refracts here. A version
    that perturbed the directions would break it while still looking like a
    propagation.
    """
    medium_index = 1.5
    rays = a_bundle(medium_index=medium_index)
    middle = a_surface("middle", z_m=1.5e-3, medium_index=medium_index)
    end = a_surface("end", z_m=4.0e-3, medium_index=medium_index)

    stepwise = propagate_rays(propagate_rays(rays, to=middle), to=end)
    direct = propagate_rays(rays, to=end)

    assert np.allclose(
        np.asarray(stepwise.positions_m), np.asarray(direct.positions_m), rtol=1e-12
    )
    assert np.allclose(
        np.asarray(stepwise.optical_path_m), np.asarray(direct.optical_path_m), rtol=1e-12
    )
    assert stepwise.reference_surface == direct.reference_surface == end


def test_a_zero_offset_advance_is_the_identity() -> None:
    """Propagating to the surface the rays are already on changes nothing."""
    rays = a_bundle(medium_index=1.5)
    same = propagate_rays(rays, to=rays.reference_surface)
    assert np.array_equal(np.asarray(same.positions_m), np.asarray(rays.positions_m))
    assert np.array_equal(np.asarray(same.optical_path_m), np.asarray(rays.optical_path_m))


def test_a_bundle_travelling_toward_minus_z_still_gains_optical_path() -> None:
    """A negative axial offset is legal; a negative optical path is not.

    The refusal is on rays travelling *away from the target*, not on a negative
    offset. A bundle pointing at `-z` legitimately reaches a surface behind it: the
    offset `dz` and the cosine `d_z` are both negative, so the arc `s = dz / d_z` is
    **positive** and the optical path **grows** by `n s`. Light does not un-travel.

    That is the sign the reference implementation's two copies disagreed about, and
    it is why this has its own test: the copy that accepted a mixed bundle returned a
    *negative* arc for the rays travelling away from its target and subtracted from
    their optical path. Here `s < 0` is unreachable, because `dz * d_z < 0` is
    refused before the division -- which is the whole content of the disagreement.
    """
    medium_index = 1.5
    rays = a_bundle(medium_index=medium_index, z_m=4.0e-3, optical_path_m=[1e-2] * 3)
    rays = dataclasses.replace(
        rays, directions=-np.asarray(rays.directions)
    )
    target = a_surface("back", z_m=0.0, medium_index=medium_index)

    advanced = propagate_rays(rays, to=target)
    arc = arc_lengths(rays, 0.0)
    # Both dz and d_z are negative, so the arc is positive and the path grows.
    assert np.all(np.asarray(rays.directions)[:, 2] < 0.0)
    assert arc_lengths(rays, 0.0).min() > 0.0
    assert np.all(arc > 0.0)
    assert np.allclose(
        np.asarray(advanced.optical_path_m),
        np.asarray(rays.optical_path_m) + medium_index * arc,
        rtol=1e-12,
    )


# ---------------------------------------------------------------------------
# 2. CHE-50's remedy: advance then reconstruct
# ---------------------------------------------------------------------------


def test_advancing_then_reconstructing_is_exact() -> None:
    """Criterion 3, and the gate CHE-50's remedy rests on.

    A collimated bundle carries one angular mode, so the reconstruction at any
    surface has the analytic value `N dA exp(+i k d_hat . r)`. Advancing the rays to
    a surface `dz` downstream and reconstructing there must therefore give that same
    plane wave with the extra `exp(+i k d_z dz)` and nothing else -- which is what
    "exact, not an approximation" means. Measured 3.5e-15 of peak.

    The arithmetic behind it: the advance changes each wavelet's constant phase by
    `k s d_z^2` and `s d_z^2 = dz d_z`, so the phase moves by exactly what a plane
    wave accumulates over `dz`. No term is dropped and none is added.
    """
    step_m = 6.0e-6
    theta = 0.2
    rays, direction, area = collimated_bundle(
        shape=SHAPE,
        sample_pitch_m=PITCH_M,
        direction=(math.sin(theta), 0.0, math.cos(theta)),
    )
    target = a_surface("downstream", z_m=step_m)

    advanced = propagate_rays(rays, to=target)
    field, _ = ray_to_scalar(advanced, grid_shape=SHAPE, sample_pitch_m=PITCH_M)

    y, x = field.coordinates()
    grid_y, grid_x = np.meshgrid(np.asarray(y), np.asarray(x), indexing="ij")
    oracle = (
        rays.count
        * area
        * np.exp(
            1j
            * rays.wavenumber
            * (direction[0] * grid_x + direction[1] * grid_y + direction[2] * step_m)
        )
    )
    residual = float(
        np.max(np.abs(np.asarray(field.u) - oracle)) / np.max(np.abs(oracle))
    )
    assert residual < 1e-13, residual
    assert field.reference_surface == target


def test_advancing_a_pupil_bundle_reproduces_the_focal_bundle_exactly() -> None:
    """Criterion 3, the positive half: the advance *is* the focal-plane construction.

    A converging bundle can be written two ways -- rays leaving the pupil toward a
    focus, or rays already at the focus carrying the pupil-to-focus optical path --
    and `propagate_rays` must turn the first into the second. It does, to
    round-off: every ray lands on the axis (`px + d_x s = px - px = 0`, since
    `s = sqrt(rho^2 + R^2)` and `d_x = -px / s`) and the optical path becomes
    `sqrt(rho^2 + R^2)` exactly.

    So the `lambda R` focal-peak oracle that R07.1 measures on the directly-built
    focal bundle also holds on the *advanced* one -- which is CHE-50's remedy
    checked end to end: advance the ray state, reconstruct there, and the analytic
    focal amplitude comes out.
    """
    from ray_support import (
        FOCAL_M,
        converging_bundle,
        hexapolar_disc,
        plateau_radius_m,
    )
    from ray_support import (
        WAVELENGTH_M as LAMBDA,
    )

    radius = plateau_radius_m()
    rings = 32
    rho, phi, area = hexapolar_disc(rings, radius)
    count = rho.size
    pupil_x, pupil_y = rho * np.cos(phi), rho * np.sin(phi)
    directions = np.column_stack([-pupil_x, -pupil_y, np.full(count, FOCAL_M)])
    directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)

    at_pupil = RayBundle(
        positions_m=np.column_stack([pupil_x, pupil_y, np.zeros(count)]),
        directions=directions,
        wavelength_m=LAMBDA,
        reference_surface=a_surface("pupil", z_m=0.0),
        amplitude=np.ones(count),
        optical_path_m=np.zeros(count),
        optical_path_reference="the pupil plane",
        measure_weight=area,
        measure_kind="quadrature_area_m2",
    )

    advanced = propagate_rays(at_pupil, to=a_surface("focal plane", z_m=FOCAL_M))
    direct, _ = converging_bundle(rings=rings, radius_m=radius)

    assert np.allclose(np.asarray(advanced.positions_m), np.asarray(direct.positions_m), atol=1e-18)
    assert np.allclose(
        np.asarray(advanced.optical_path_m), np.asarray(direct.optical_path_m), rtol=1e-12
    )

    from ray_support import focal_peak_oracle

    field, _ = ray_to_scalar(advanced, grid_shape=(9, 9), sample_pitch_m=(0.2e-6, 0.2e-6))
    assert float(abs(np.asarray(field.u)[4, 4])) == pytest.approx(
        focal_peak_oracle(radius_m=radius), rel=1e-3
    )


def test_propagating_the_reconstructed_field_instead_is_refused_by_the_tree() -> None:
    """Criterion 3, and the reason this operation is *the* route rather than one of two.

    The alternative -- reconstruct at the original surface and propagate the field
    -- is not merely less accurate here, it is **structurally refused**:
    `couplers.ray_to_scalar` emits every field with `surface_only` in its typed
    validity (CHE-50), and `solvers.chromatix.propagate` refuses a `surface_only`
    field. Two packages, one declaration, and a caller cannot compose the wrong
    route by accident.

    That is CHE-50's remedy made executable rather than documented, and it is why
    `propagate_rays` had to be correct: it is the only route left.
    """
    from solvers.chromatix import propagate

    rays, _, _ = collimated_bundle(
        shape=SHAPE, sample_pitch_m=PITCH_M, direction=(0.0, 0.0, 1.0)
    )
    field, _ = ray_to_scalar(rays, grid_shape=SHAPE, sample_pitch_m=PITCH_M)
    assert "surface_only" in field.validity
    assert "no_wavefront_curvature_term" in field.validity

    with pytest.raises(ContractError) as raised:
        propagate(
            field,
            distance_m=6e-6,
            model={"method": "asm", "pad_width": 0, "target_surface": "downstream"},
        )
    assert raised.value.declaration == "validity"
    assert "surface_only" in str(raised.value)


def test_where_the_two_routes_would_agree_they_do() -> None:
    """Criterion 3's regime statement, measured on the case where both are valid.

    The blanket refusal above cannot tell which regime a caller is in, so it refuses
    all of them. For a **single angular mode** both routes really are valid: a plane
    wave carries no wavefront curvature, so there is nothing for the reconstruction
    to have lost. Demonstrated by stripping `surface_only` deliberately -- a
    test-side act, stated as one -- and comparing:

    * advance the rays by `dz`, then reconstruct;
    * reconstruct at `z = 0`, then propagate the field by `dz` with the ASM.

    The tilt is placed on an exact spectral bin so the ASM is a single-coefficient
    multiply and contributes no interpolation of its own. The bundle is float32
    because the wave path is complex64-only -- `numerics` refuses the complex128
    field a float64 bundle would produce, with the measured evidence that
    `chromatix.ScalarField.__init__` casts unconditionally -- so the agreement is
    quoted at that precision rather than at float64's.

    Where they diverge is the next test, and it is measured rather than transcribed.
    """
    from solvers.chromatix import propagate

    step_m = 6.0e-6
    # An exact spectral bin of the output grid: d_u = lambda * m / (nx dx).
    modes = 3
    d_u = WAVELENGTH_M * modes / (SHAPE[1] * PITCH_M[1])
    rays, direction, _ = collimated_bundle(
        shape=SHAPE,
        sample_pitch_m=PITCH_M,
        direction=(d_u, 0.0, math.sqrt(1.0 - d_u**2)),
        dtype=np.float32,
    )
    assert direction[0] == pytest.approx(d_u, rel=1e-12)

    advanced, _ = ray_to_scalar(
        propagate_rays(rays, to=a_surface("downstream", z_m=step_m)),
        grid_shape=SHAPE,
        sample_pitch_m=PITCH_M,
    )
    at_source, _ = ray_to_scalar(rays, grid_shape=SHAPE, sample_pitch_m=PITCH_M)
    propagatable = dataclasses.replace(
        at_source, validity=at_source.validity - {"surface_only"}
    )
    propagated = propagate(
        propagatable,
        distance_m=step_m,
        # `pad_width=0` on purpose: the reconstruction of a single on-bin mode is
        # exactly periodic on this grid, so a plain transfer-function multiply is the
        # exact propagation. Zero-padding would truncate the plane wave and add edge
        # diffraction -- measured at 1.05 of peak, i.e. it dominates everything -- so
        # padding here would be comparing against a different physical problem.
        model={"method": "asm", "pad_width": 0, "target_surface": "downstream"},
    )

    reference = np.asarray(advanced.u)
    residual = float(
        np.max(np.abs(np.asarray(propagated.u) - reference)) / np.max(np.abs(reference))
    )
    assert residual < 1e-5, residual


@pytest.mark.parametrize(
    ("focal_m", "radius_m", "expected_divergence"),
    [(100e-6, 4.0e-6, 0.371), (60e-6, 3.0e-6, 0.214)],
)
def test_where_the_two_routes_diverge_the_advance_is_the_one_that_is_right(
    focal_m: float, radius_m: float, expected_divergence: float
) -> None:
    """Criterion 3's divergence half, measured on a geometry the window supports.

    A **converging** bundle is where the two routes part, and the reason is
    CHE-50's: the field reconstructed at the pupil carries no `exp(i k r^2 / 2R)`
    term, so propagating *it* propagates something that has already lost the
    curvature, while advancing the rays and reconstructing at the focus keeps it.

    Which route is right is settled by an oracle outside both of them --
    stationary phase over the pupil, `lambda R |1 - exp(i pi a^2 / (lambda R))|`:

    | R, a | advance then reconstruct | reconstruct then propagate | analytic |
    | -- | -- | -- | -- |
    | 100 um, 4 um | 4.8554e-11 (1.0004x) | 6.0137e-11 (1.24x) | 4.8534e-11 |
    | 60 um, 3 um | 2.7431e-11 (1.0005x) | 3.2951e-11 (1.20x) | 2.7417e-11 |

    The advance is exact to 5e-4; the field route is **20-24 % high in peak
    amplitude** and differs from the advance by 21-37 % of peak. So this is not a
    tolerance to be tightened: one route answers the question and the other does
    not, which is why `solvers.chromatix.propagate` refuses a `surface_only` field
    outright rather than doing it less accurately.

    The geometry is chosen so the measurement means something. R07's own focal
    fixture is a 4.8 mm propagation on a 1.8 um window, where an ASM comparison
    would be dominated by wraparound rather than by the curvature term; at
    `R = 100 um` with a 12.8 um window the converging cone stays inside the grid
    and the residual is the physics. R07.1 records the reference implementation's
    figure for the same limitation -- about 1.2 rad of phase against an exact
    spherical-wave reference at the 5-Airy-radius gate edge, with the intensity
    residual at 1e-3 so `|U|^2` does not warn -- and that number is transcribed
    while these are measured here.
    """
    from ray_support import hexapolar_disc

    from solvers.chromatix import propagate

    shape, pitch = (64, 64), (0.2e-6, 0.2e-6)
    rho, phi, area = hexapolar_disc(16, radius_m)
    count = rho.size
    pupil_x, pupil_y = rho * np.cos(phi), rho * np.sin(phi)
    directions = np.column_stack([-pupil_x, -pupil_y, np.full(count, focal_m)])
    directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)
    at_pupil = RayBundle(
        positions_m=np.column_stack([pupil_x, pupil_y, np.zeros(count)]).astype(np.float32),
        directions=directions.astype(np.float32),
        wavelength_m=WAVELENGTH_M,
        reference_surface=a_surface("pupil", z_m=0.0),
        amplitude=np.ones(count, dtype=np.complex64),
        optical_path_m=np.zeros(count, dtype=np.float32),
        optical_path_reference="the pupil plane",
        measure_weight=area.astype(np.float32),
        measure_kind="quadrature_area_m2",
    )

    # Route A: advance the ray state, reconstruct at the focus. CHE-50's remedy.
    advanced, _ = ray_to_scalar(
        propagate_rays(at_pupil, to=a_surface("focus", z_m=focal_m)),
        grid_shape=shape,
        sample_pitch_m=pitch,
    )
    # Route B: reconstruct at the pupil and propagate the field. `surface_only` is
    # stripped deliberately -- the tree refuses this route, and stripping it is how
    # the cost of taking it anyway gets measured.
    at_pupil_field, _ = ray_to_scalar(at_pupil, grid_shape=shape, sample_pitch_m=pitch)
    propagated = propagate(
        dataclasses.replace(
            at_pupil_field, validity=at_pupil_field.validity - {"surface_only"}
        ),
        distance_m=focal_m,
        model={"method": "asm", "pad_width": 64, "target_surface": "focus"},
    )

    phase_max = math.pi * radius_m**2 / (WAVELENGTH_M * focal_m)
    oracle = WAVELENGTH_M * focal_m * abs(1.0 - complex(math.cos(phase_max), math.sin(phase_max)))

    advance_peak = float(np.max(np.abs(np.asarray(advanced.u))))
    field_peak = float(np.max(np.abs(np.asarray(propagated.u))))

    # The advance is right, against an oracle neither route produced.
    assert advance_peak == pytest.approx(oracle, rel=2e-3)
    # The field route is not, and by an amount no tolerance covers.
    assert field_peak / oracle > 1.15
    divergence = float(
        np.max(np.abs(np.asarray(propagated.u) - np.asarray(advanced.u))) / advance_peak
    )
    assert divergence == pytest.approx(expected_divergence, rel=0.15), divergence


# ---------------------------------------------------------------------------
# 3. What it refuses
# ---------------------------------------------------------------------------


def test_a_target_in_a_different_medium_is_refused() -> None:
    """Two surfaces in different media do not bound one medium.

    There is no single `n` for the optical path to grow by, and reaching the target
    would need a discrete surface interaction -- which this operation excludes by
    definition. Refused rather than resolved by picking one of the two indices,
    which is the move that puts a silent `n = 1` back.
    """
    rays = a_bundle(medium_index=1.5)
    with pytest.raises(ContractError) as raised:
        propagate_rays(rays, to=a_surface("end", z_m=4e-3, medium_index=1.0))
    assert raised.value.code == "MISSING_DECLARATION"
    assert raised.value.declaration == "to.medium_index"
    # ...and the same target in the same medium is accepted, so this is a pairing
    # rule and not a ban on a medium.
    propagate_rays(rays, to=a_surface("end", z_m=4e-3, medium_index=1.5))


def test_a_tilted_target_is_refused() -> None:
    """`s = dz / d_z` is defined only for a plane perpendicular to the propagation axis."""
    rays = a_bundle()
    tilted = ReferenceSurface(
        name="tilted",
        z_m=4e-3,
        medium_index=1.0,
        normal=(0.0, math.sin(0.1), math.cos(0.1)),
    )
    with pytest.raises(ContractError) as raised:
        propagate_rays(rays, to=tilted)
    assert raised.value.code == "FRAME_MISMATCH"


def test_a_ray_that_never_reaches_the_target_is_refused_not_dropped() -> None:
    """A bundle that quietly loses members produces a plausible field with missing power.

    This is where the reference implementation's two copies disagreed. One refused;
    the other returned a **negative** arc for the offending ray and silently
    propagated it backwards, subtracting from its optical path. The refusing one is
    authoritative and this is why.
    """
    rays = a_bundle()
    # One ray turned around: it now travels away from a target downstream.
    mixed_directions = np.asarray(rays.directions).copy()
    mixed_directions[1] = -mixed_directions[1]
    mixed = dataclasses.replace(rays, directions=mixed_directions)

    with pytest.raises(ContractError) as raised:
        propagate_rays(mixed, to=a_surface("end", z_m=4e-3))
    assert raised.value.code == "FRAME_MISMATCH"
    assert "never reach" in str(raised.value)
    assert "1 of 3" in str(raised.value)

    # A ray exactly parallel to the target never arrives either.
    parallel = np.asarray(rays.directions).copy()
    parallel[2] = [1.0, 0.0, 0.0]
    with pytest.raises(ContractError) as raised:
        propagate_rays(
            dataclasses.replace(rays, directions=parallel), to=a_surface("end", z_m=4e-3)
        )
    assert raised.value.code == "FRAME_MISMATCH"


def test_a_ray_whose_arc_length_is_unrepresentable_is_refused() -> None:
    """The floor on `|d_z|` is derived, not chosen, and it is R07.4's own bound.

    A 4 mm offset at `d_z = 1e-12` is a 4e9 m arc length, whose phase at 550 nm is
    4.6e16 rad -- pure noise at any precision. Both reference copies admitted it:
    one behind a fixed `1e-12` absolute cut, the other behind exact equality with
    zero. Here the floor comes from `couplers.grazing_floor_for_phase_budget`, the
    same `eps k Z / d_n <= budget` derivation the reconstruction kernel uses, so the
    two cannot drift.
    """
    offset_m = 4.0e-3
    grazing = np.array([[1.0 - 5e-25, 0.0, 1e-12], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    rays = dataclasses.replace(a_bundle(), directions=grazing)

    with pytest.raises(ContractError) as raised:
        propagate_rays(rays, to=a_surface("end", z_m=offset_m))
    assert raised.value.code == "GRAZING_PHASE_UNREPRESENTABLE"

    # The floor is the operation's own derivation, so a caller can compute it: at
    # float64 over 4 mm with the default budget it is ~1.1e-9, and a ray just above
    # it is accepted.
    from couplers import grazing_floor_for_phase_budget
    from numerics import Precision

    floor = grazing_floor_for_phase_budget(
        wavelength_m=WAVELENGTH_M,
        max_optical_path_m=offset_m,
        precision=Precision.FP64,
        phase_budget_rad=DEFAULT_PHASE_BUDGET_RAD,
    )
    assert 1e-11 < floor < 1e-7, floor
    admitted = np.array(
        [
            [math.sqrt(1.0 - (2.0 * floor) ** 2), 0.0, 2.0 * floor],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ]
    )
    propagate_rays(
        dataclasses.replace(a_bundle(), directions=admitted),
        to=a_surface("end", z_m=offset_m),
    )


def test_a_bundle_with_no_optical_path_is_refused() -> None:
    """Evolving one is what this operation does, so there is nothing for it to do."""
    rays = a_bundle()
    with pytest.raises(ContractError) as raised:
        propagate_rays(
            dataclasses.replace(rays, optical_path_m=None, optical_path_reference=None),
            to=a_surface("end", z_m=4e-3),
        )
    assert raised.value.code == "MISSING_DECLARATION"
    assert raised.value.declaration == "optical_path_m"


def test_the_optical_path_reference_records_the_advance() -> None:
    """A consumer can see the path is no longer measured to where the rays started."""
    rays = a_bundle(medium_index=1.5)
    advanced = propagate_rays(rays, to=a_surface("end", z_m=4e-3, medium_index=1.5))
    reference = advanced.optical_path_reference or ""
    assert reference.startswith(rays.optical_path_reference or "")
    assert "'end'" in reference
    assert "1.5" in reference
    assert "n * s" in reference


def test_both_couplers_carry_the_medium_index_this_operation_uses() -> None:
    """R09's central finding, fixed rather than refused (CHE-192).

    `couplers.ray_to_scalar`'s transverse ramp was `k0 d_hat . dr`, the `n = 1` form
    of `n k0 d_hat . dr`, and `couplers.scalar_to_ray`'s direction cosines were
    `lambda_vacuum f`, likewise. R09 found both while deriving this operation --
    whose optical path grows by `n s`, so composing them through glass would have
    put a medium-aware path against a vacuum ramp -- and refused rather than alter a
    landed convention overnight. The owner took the fix: `n` is in both, a no-op at
    `n = 1`.

    This test is what the refusal became. Both calls it used to reject now return,
    which is the half a reader of this file needs; the ramp and the cosines are
    graded against their own analytic oracles in the two couplers' test files.
    """
    from ray_support import a_random_field

    from couplers import scalar_to_ray

    in_water = 1.336

    rays = a_bundle(medium_index=in_water)
    field, reconstruction = ray_to_scalar(rays, grid_shape=(8, 8), sample_pitch_m=PITCH_M)
    assert field.reference_surface.medium_index == in_water
    # The Nyquist limit is on the medium wavelength, tightened by exactly `n`.
    assert reconstruction.grid_nyquist_direction_limit[1] == pytest.approx(
        WAVELENGTH_M / (in_water * 2.0 * PITCH_M[1])
    )

    submerged = dataclasses.replace(
        a_random_field(shape=(8, 8), sample_pitch_m=PITCH_M),
        reference_surface=ReferenceSurface(name="in water", z_m=0.0, medium_index=in_water),
    )
    emitted, _ = scalar_to_ray(submerged)
    assert emitted.reference_surface.medium_index == in_water
    assert np.allclose(np.linalg.norm(np.asarray(emitted.directions), axis=1), 1.0)


def test_the_advance_and_the_reconstruction_compose_in_a_medium() -> None:
    """The exactness claim of this module, executed at `n != 1` rather than argued.

    The derivation in `operators/ray_propagation.py` writes the coupler's constant
    phase as `k0 (OPL - n d_t . x0_t)` and gets `C2 = C1 + n s d_z^2` after the
    advance, so reconstructing at the new plane must equal reconstructing at the old
    one times `exp(i n k0 d_z dz)` -- exactly what a plane wave of wavevector
    `n k0 d_hat` accumulates over an axial offset `dz`. One ray, so the factor is
    uniform across the grid and can be read off rather than fitted.

    This is the composition R09 could not run. Both `n`s are load-bearing: the one
    on the optical path here and the one on the ramp in the coupler, and the vacuum
    phase is asserted *not* to be the answer.
    """
    in_water = 1.336
    offset_m = 4.0e-3
    rays = a_bundle(thetas=(0.15,), medium_index=in_water)
    axial = float(np.asarray(rays.directions)[0, 2])

    before, _ = ray_to_scalar(rays, grid_shape=SHAPE, sample_pitch_m=PITCH_M)
    advanced = propagate_rays(
        rays, to=a_surface("end", z_m=offset_m, medium_index=in_water)
    )
    after, _ = ray_to_scalar(advanced, grid_shape=SHAPE, sample_pitch_m=PITCH_M)

    ratio = np.asarray(after.u) / np.asarray(before.u)
    expected = np.exp(1j * in_water * rays.wavenumber * axial * offset_m)
    assert np.allclose(ratio, expected, rtol=0.0, atol=1e-8)
    # The vacuum phase is 1.4e4 rad away from it, so this is not a factor the test
    # would have accepted either way.
    vacuum = np.exp(1j * rays.wavenumber * axial * offset_m)
    assert not np.allclose(ratio, vacuum, rtol=0.0, atol=1e-3)


# ---------------------------------------------------------------------------
# 4. Exactly one implementation, and the record
# ---------------------------------------------------------------------------


def test_exactly_one_advance_to_a_plane_exists_in_the_tree() -> None:
    """Criterion 5. The reference implementation had two; this tree has one.

    A name walk over every production module, because the failure this guards is a
    second copy appearing under a slightly different name in whichever package
    needs it next -- which is exactly how the reference implementation's pair came
    about, one of them created inside a patch model because that was its first
    caller.
    """
    advancing = [
        f"{module.relative_to(SRC)}::{node.name}"
        for module in sorted(SRC.rglob("*.py"))
        for node in ast.walk(ast.parse(module.read_text(encoding="utf-8")))
        if isinstance(node, ast.FunctionDef)
        and any(
            token in node.name
            for token in ("advance", "propagate_rays", "to_plane", "to_surface")
        )
    ]
    assert advancing == ["operators/ray_propagation.py::propagate_rays"], advancing


def test_ray_propagation_registers_as_a_physical_operator() -> None:
    """Criterion 4. `ray_bundle -> ray_bundle`: the state changes, the representation
    does not.

    Not a coupler, and the `kind` is where that is said.

    The descriptor used to be constructed here, inside a fixture that emptied the
    registry, because `operators/` may not import `operations/` and there was no
    production registration site anywhere. CHE-221 (R03.4) put one *inside*
    `operations/`: the catalog names the implementation as a
    `"module.path:attribute"` string, so it needs no dependency edge in either
    direction, and the allowlist is unchanged. What is read below is the shipped
    record rather than a copy this file kept in step by hand.

    One line of the migrated `validity` was **corrected rather than copied**, and
    the correction is flagged on CHE-221: the fixture said the reconstruction
    kernel "implements the n = 1 ramp and refuses n != 1, so a bundle advanced
    through a medium cannot yet be reconstructed (recorded on CHE-192)". CHE-192's
    follow-up put the `n` in and lifted the refusal, so migrating that line
    verbatim would have put a false validity claim into the canonical catalog.
    """
    descriptor = next(d for d in CATALOG if d.operation_id == "O_PROPAGATE_RAYS")
    assert descriptor.kind is OperationKind.PHYSICAL_OPERATOR
    assert descriptor.kind is not OperationKind.COUPLER
    assert descriptor.input == descriptor.output == "ray_bundle"
    assert descriptor.capabilities is None
    assert not any("cannot yet be reconstructed" in c for c in descriptor.validity), (
        "the n != 1 reconstruction refusal was lifted by CHE-192"
    )
    assert resolve("O_PROPAGATE_RAYS") is propagate_rays


def test_the_module_defines_no_class() -> None:
    """Criterion 6. Class delta 0: an advance is a function of a bundle and a surface."""
    source = (SRC / "operators" / "ray_propagation.py").read_text(encoding="utf-8")
    assert [
        node.name for node in ast.walk(ast.parse(source)) if isinstance(node, ast.ClassDef)
    ] == []


def test_no_wave_propagation_forwarding_wrapper_landed() -> None:
    """R09.1 criterion 4. Chromatix owns wave propagation and exposes it itself.

    A function in `operators/` that only called `solvers.chromatix.propagate` would
    do no numerical work; relocating semantic ownership is not a reason for a
    function to exist. Checked as an import rule, which is also what the allowlist
    says: `operators -> solvers` is forbidden.
    """
    package = SRC / "operators"
    for module in sorted(package.rglob("*.py")):
        tree = ast.parse(module.read_text(encoding="utf-8"), filename=str(module))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = {alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = {node.module.split(".")[0]}
            else:
                continue
            assert "solvers" not in names, module.relative_to(SRC)
