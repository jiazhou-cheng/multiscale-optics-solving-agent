"""The collimated ensemble as a contract: every convention, and four refusals.

CHE-215 (R06.10), item 1. `tests/physics/test_collimated_source.py` holds the
physics -- the analytic wavelet-sum oracle and the labelled cross-path consistency
check -- because that needs a coupler and these need nothing.

**CHE-219 (R05.8) moved what this file tests.** It was `sources.collimated_bundle`
and it is now `fixtures.ray_bundles.collimated_bundle`: a launch `RayBundle` built
from caller-supplied points and a shared direction has no optical system in scope,
so it cannot say whether those points are the entrance pupil, the stop, the first
traced surface, a valid finite-conjugate aim, or anything in the constructed
system -- and a system launch is `backends.optiland.launch`'s. Nothing about the
arithmetic or these assertions changed with the move; two tests that asserted
properties of a *production* module went with it, and are noted below.

What this file guards is that **nothing is defaulted and nothing is guessed**.
This builder creates a representation out of nothing, so every convention it does
not take as an argument is one it invented, and the two that would do real damage
silently are the optical path (which is what makes the ensemble one mode rather
than N unrelated wavelets) and the measure (which R05/R07 established scales every
downstream reconstruction).
"""

from __future__ import annotations

import ast
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from fixtures.ray_bundles import collimated_bundle, direction_from_angle

from representations import PHASOR, ContractError, Frame, RayBundle, ReferenceSurface

WAVELENGTH_M = 0.532e-6

#: Deliberately non-square in both count and pitch. Positions are `(x, y, z)`
#: columns while grids are `(y, x)`, and a square fixture cannot fail on a swap.
SHAPE = (5, 8)
PITCH_M = (0.30e-6, 0.25e-6)

MODULE = Path(__file__).resolve().parents[1] / "fixtures" / "ray_bundles.py"


def a_surface(*, z_m: float = 0.0, medium_index: float = 1.0) -> ReferenceSurface:
    return ReferenceSurface(name="entrance_pupil", z_m=z_m, medium_index=medium_index)


def grid_positions(
    shape: tuple[int, int] = SHAPE, pitch: tuple[float, float] = PITCH_M, *, z_m: float = 0.0
) -> np.ndarray:
    """`(N, 3)` launch points from a `(ny, nx)` grid, x column-stacked first.

    Test-side on purpose: the source takes explicit points precisely so that the
    rectangular aperture model lives in the caller. Written here the way the
    package docstring's example writes it, so the example is exercised.
    """
    ny, nx = shape
    dy, dx = pitch
    y = (np.arange(ny, dtype=np.float64) - Frame().origin_index(ny)) * dy
    x = (np.arange(nx, dtype=np.float64) - Frame().origin_index(nx)) * dx
    grid_y, grid_x = np.meshgrid(y, x, indexing="ij")
    return np.column_stack([grid_x.ravel(), grid_y.ravel(), np.full(grid_x.size, z_m)])


def a_bundle(**overrides: object) -> RayBundle:
    arguments: dict[str, object] = {
        "wavelength_m": WAVELENGTH_M,
        "reference_surface": a_surface(),
    }
    arguments.update(overrides)
    positions = arguments.pop("positions_m", None)
    if positions is None:
        positions = grid_positions()
    return collimated_bundle(positions, **arguments)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 1. Every convention is declared
# ---------------------------------------------------------------------------


def test_the_bundle_is_the_launch_points_the_caller_gave_it() -> None:
    """Positions are preserved *exactly*, and the count is one ray per point.

    Exactly rather than to a tolerance: a source that resampled, reordered or
    projected the caller's points onto the surface's `z_m` would be propagating,
    and the residual would be attributed to the coupler downstream.
    """
    positions = grid_positions(z_m=1e-3)
    rays = a_bundle(positions_m=positions, reference_surface=a_surface(z_m=0.0))

    assert rays.count == SHAPE[0] * SHAPE[1]
    assert np.array_equal(np.asarray(rays.positions_m), positions)
    # The surface says z = 0 and the rays were launched from z = 1 mm. The source
    # does not reconcile them: moving a ray onto a plane is a propagation.
    assert rays.reference_surface.z_m == 0.0
    assert np.all(np.asarray(rays.positions_m)[:, 2] == 1e-3)


def test_a_nonsquare_grid_catches_an_axis_swap() -> None:
    """`(x, y, z)` columns against a `(y, x)` grid -- the trap, asserted.

    The fixture is 5 x 8 with unequal pitches, so `x` spans 8 samples of 0.25 um
    and `y` spans 5 of 0.30 um. A transposed column-stack would put the wider
    span on the wrong axis, which is invisible on any square grid and invisible on
    a square pitch.
    """
    positions = np.asarray(a_bundle().positions_m)
    x_column, y_column = positions[:, 0], positions[:, 1]

    assert len(np.unique(x_column)) == SHAPE[1]
    assert len(np.unique(y_column)) == SHAPE[0]
    assert np.ptp(x_column) == pytest.approx((SHAPE[1] - 1) * PITCH_M[1])
    assert np.ptp(y_column) == pytest.approx((SHAPE[0] - 1) * PITCH_M[0])


def test_every_ray_shares_one_normalized_direction() -> None:
    """One mode, not N rays that happen to point the same way.

    `direction` is normalized inside the source, so a caller stating `(0, 0, 2)`
    or `(1, 1, 1)` gets a legal bundle rather than a `NON_UNIT_DIRECTION` refusal
    they have to fix by dividing by a norm the representation never showed them.
    """
    rays = a_bundle(direction=(0.0, 0.0, 2.0))
    directions = np.asarray(rays.directions)

    assert directions.shape == (rays.count, 3)
    assert np.array_equal(directions, np.tile(directions[0], (rays.count, 1)))
    assert np.allclose(np.linalg.norm(directions, axis=1), 1.0, atol=0.0, rtol=1e-15)

    unnormalized = a_bundle(direction=(1.0, 1.0, 1.0))
    assert np.allclose(np.asarray(unnormalized.directions)[0], 1.0 / math.sqrt(3.0))


def test_normal_incidence_is_the_same_primitive() -> None:
    """`(0, 0, 1)` is the default, and there is no second function for it."""
    from fixtures import ray_bundles

    for name in ("normal_bundle", "collimated_bundle_at_angle", "tilted_bundle"):
        assert not hasattr(ray_bundles, name)

    default = a_bundle()
    explicit = a_bundle(direction=(0.0, 0.0, 1.0))
    assert np.array_equal(np.asarray(default.directions), np.asarray(explicit.directions))
    # At normal incidence every launch point is on the same wavefront, so the
    # optical path is identically zero -- not merely constant.
    assert np.all(np.asarray(default.optical_path_m) == 0.0)


def test_a_tilt_is_built_from_the_angle_converter() -> None:
    """Tilted illumination, with the direction coming from `direction_from_angle`.

    And the two converters are asserted to describe the *same* mode:
    `k_t = n k0 (d_y, d_x)`. They are separate functions because one returns a
    direction cosine in `(x, y, z)` and the other a wavevector in `(y, x)`, and a
    caller who mixed the orders would get a plausible tilt on the wrong axis. They
    no longer sit in the same package -- CHE-219 kept
    `sources.transverse_wavevector_from_angle`, because `k_t` on a `ScalarField`
    grid is not a ray aim, and moved `direction_from_angle` out with the launch --
    so this cross-check matters more rather than less.
    """
    from sources import transverse_wavevector_from_angle

    theta, phi = 0.35, 0.7
    direction = direction_from_angle(theta, phi)
    rays = a_bundle(direction=direction)
    d_x, d_y, d_z = np.asarray(rays.directions)[0]

    assert (d_x, d_y, d_z) == pytest.approx(direction)
    assert d_z == pytest.approx(math.cos(theta))
    assert math.hypot(d_x, d_y) == pytest.approx(math.sin(theta))
    # phi from +x toward +y, the same convention the wavevector converter uses.
    assert math.atan2(d_y, d_x) == pytest.approx(phi)

    medium_wavenumber = 2.0 * math.pi / WAVELENGTH_M
    k_y, k_x = transverse_wavevector_from_angle(
        theta, phi, wavelength_m=WAVELENGTH_M, medium_index=1.0
    )
    assert (k_y, k_x) == pytest.approx((medium_wavenumber * d_y, medium_wavenumber * d_x))


def test_the_wavelength_and_the_medium_index_are_carried_through() -> None:
    """Neither is defaulted, and `medium_index` is not defaulted one layer down.

    `ReferenceSurface.medium_index` has no default anywhere in the tree by an
    explicit R02 decision, and a source is exactly where such a default would go
    unnoticed -- so the surface is a required keyword and nothing here supplies one.
    """
    rays = a_bundle(reference_surface=a_surface(medium_index=1.515))

    assert rays.wavelength_m == WAVELENGTH_M
    assert rays.wavenumber == pytest.approx(2.0 * math.pi / WAVELENGTH_M)
    assert rays.reference_surface.medium_index == 1.515
    assert rays.frame == Frame()
    assert rays.phasor == PHASOR

    with pytest.raises(TypeError):
        collimated_bundle(grid_positions(), wavelength_m=WAVELENGTH_M)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        collimated_bundle(grid_positions(), reference_surface=a_surface())  # type: ignore[call-arg]


@pytest.mark.parametrize("medium_index", [1.0, 1.336, 1.515])
def test_the_optical_path_is_n_times_the_geometric_projection(medium_index: float) -> None:
    """`optical_path_m = n (d_hat . r)`, with its reference stated.

    This is the load-bearing assertion of the file. Those phases are what make the
    ensemble one plane-wave mode: with them `couplers.ray_to_scalar` reconstructs
    `N dA exp(+i n k0 d_hat . r)` in closed form, and without them there is no
    analytic form to compare against at all.

    `n` multiplies the projection because an optical path *is* `n` times a
    geometric one -- the same convention `operators.propagate_rays` advances by and
    `couplers.ray_to_scalar` reads back. Three indices including air, so the
    `n = 1` case cannot be the only one that passes.
    """
    positions = grid_positions()
    direction = direction_from_angle(0.35, 0.7)
    rays = a_bundle(
        positions_m=positions,
        direction=direction,
        reference_surface=a_surface(medium_index=medium_index),
    )

    expected = medium_index * (positions @ np.asarray(direction, dtype=np.float64))
    assert np.allclose(np.asarray(rays.optical_path_m), expected, rtol=1e-15, atol=0.0)
    # `RayBundle` refuses a path with no reference, so this string is not optional.
    assert rays.optical_path_reference == "the global origin, along d_hat"


def test_the_measure_is_left_undeclared_rather_than_guessed() -> None:
    """The default is `None` / `"undeclared"`, and the coupler's refusal follows.

    Not a gap. From explicit positions there is no `dA` to derive -- the same
    `(N, 3)` array is a uniform grid, a hexapolar pupil with unequal cells, and an
    importance-weighted draw, and those differ by the aperture area and by whether
    the reconstruction owes a `1/N`. R05 moved the quadrature weight off the
    amplitude and R07's kernel applies `measure_weight` itself, so a defaulted `dA`
    would scale every reconstruction by a factor no intensity check can see.
    """
    default = a_bundle()
    assert default.measure_weight is None
    assert default.measure_kind == "undeclared"

    # A caller who knows the sampling states it, and both kinds are accepted.
    area = PITCH_M[0] * PITCH_M[1]
    declared = a_bundle(
        measure_weight=np.full(default.count, area), measure_kind="quadrature_area_m2"
    )
    assert np.allclose(np.asarray(declared.measure_weight), area)
    assert declared.measure_kind == "quadrature_area_m2"

    # And the representation still refuses a half-declared measure, through this
    # source rather than around it.
    with pytest.raises(ContractError) as excinfo:
        a_bundle(measure_weight=np.full(default.count, area))
    assert excinfo.value.code == "MEASURE_UNDECLARED"
    with pytest.raises(ContractError) as excinfo:
        a_bundle(measure_kind="quadrature_area_m2")
    assert excinfo.value.code == "MISSING_DECLARATION"


def test_the_amplitude_is_a_uniform_peak_and_nothing_renormalizes_it() -> None:
    """One real number per ray, unnormalized, widened to a phase-free complex."""
    unit = a_bundle()
    assert np.allclose(np.asarray(unit.amplitude), 1.0)

    doubled = a_bundle(amplitude=2.0)
    assert np.allclose(np.asarray(doubled.amplitude), 2.0)
    # No 1/N, no 1/sqrt(N), no power normalization: 2 A is exactly 2 A.
    assert np.allclose(np.asarray(doubled.amplitude), 2.0 * np.asarray(unit.amplitude))
    # A real amplitude is a phase-free launch, widened rather than reinterpreted.
    assert np.iscomplexobj(np.asarray(unit.amplitude))
    assert np.all(np.asarray(unit.amplitude).imag == 0.0)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_the_dtype_is_read_off_the_positions(dtype: type) -> None:
    """A float32 request really is a float32 bundle, and it passes on the first try.

    `RayBundle` checks direction norms against `direction_norm_tolerance(dtype)`,
    which is a bound on round-off. Normalizing in float64 before the cast is what
    makes a float32 bundle land inside it without the caller knowing the tolerance
    exists -- and the projection is accumulated in float64 for the same reason
    `plane_wave` accumulates its ramp there: `k` is ~1.2e7 rad/m.
    """
    positions = grid_positions().astype(dtype)
    rays = a_bundle(positions_m=positions, direction=(1.0, 1.0, 1.0))

    assert np.asarray(rays.positions_m).dtype == dtype
    assert np.asarray(rays.directions).dtype == dtype
    assert np.asarray(rays.optical_path_m).dtype == dtype
    # Complex of the *same* precision, never widened to complex128.
    assert np.asarray(rays.amplitude).dtype == (
        np.complex64 if dtype is np.float32 else np.complex128
    )


# ---------------------------------------------------------------------------
# 2. The refusals
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "direction",
    [(0.0, 0.0, 0.0), (math.nan, 0.0, 1.0), (0.0, math.inf, 1.0), (0.0, 1.0)],
)
def test_a_direction_that_states_nothing_is_refused(direction: tuple[float, ...]) -> None:
    """A zero, non-finite or wrong-length direction, before it becomes NaN geometry.

    The zero vector is the one worth naming: it would normalize to NaN, and a
    bundle of NaN directions is refused by `RayBundle` several frames later with a
    message about a norm rather than about the argument that was wrong.
    """
    with pytest.raises(ValueError) as excinfo:
        a_bundle(direction=direction)  # type: ignore[arg-type]
    assert "direction" in str(excinfo.value)


@pytest.mark.parametrize(
    "positions",
    [
        np.zeros((4, 2)),
        np.zeros((4, 4)),
        np.zeros(4),
        np.zeros((2, 4, 3)),
    ],
)
def test_positions_that_are_not_n_by_3_are_refused(positions: np.ndarray) -> None:
    """`(N, 3)`, and the message names the `(x, y, z)` column order.

    Refused here rather than left to `RayBundle`, because the projection
    `positions @ d_hat` runs first and a `(4, 2)` array would fail with a matmul
    shape error that names neither the declaration nor the convention.
    """
    with pytest.raises(ContractError) as excinfo:
        a_bundle(positions_m=positions)
    assert excinfo.value.code == "SHAPE_MISMATCH"
    assert "(x, y, z)" in str(excinfo.value)


def test_the_ray_bundle_contract_refuses_the_rest() -> None:
    """The wavelength, the emptiness and the finiteness go through `RayBundle`.

    Not through a second copy of its validation: a source that pre-checked what
    the representation already checks is where the two definitions drift.
    """
    for bad_wavelength in (0.0, -WAVELENGTH_M, math.nan):
        with pytest.raises(ContractError):
            a_bundle(wavelength_m=bad_wavelength)

    with pytest.raises(ContractError) as excinfo:
        a_bundle(positions_m=np.zeros((0, 3)))
    assert excinfo.value.code == "EMPTY_ENSEMBLE"

    with pytest.raises(ContractError) as excinfo:
        a_bundle(positions_m=np.full((4, 3), math.nan))
    assert excinfo.value.code == "NON_FINITE"

    for bad_amplitude in (0.0, -1.0, math.nan):
        with pytest.raises(ContractError):
            a_bundle(amplitude=bad_amplitude)


def test_the_angle_converter_refuses_an_ambiguous_argument() -> None:
    """Past `pi/2` a backward ray would come back as a forward tilt.

    `cos(theta)` goes negative while `sin(theta)` starts decreasing again, so the
    returned triple is a perfectly well-formed direction of the wrong magnitude
    pointing the wrong way -- which is why this is a refusal and not a wrap.
    """
    for theta in (0.51 * math.pi, -0.51 * math.pi, math.pi):
        with pytest.raises(ValueError) as excinfo:
            direction_from_angle(theta, 0.0)
        assert "pi/2" in str(excinfo.value)

    for bad in ((math.nan, 0.0), (0.0, math.inf)):
        with pytest.raises(ValueError):
            direction_from_angle(*bad)

    # Grazing itself, |theta| = pi/2, is the boundary and is permitted.
    grazing = direction_from_angle(0.5 * math.pi, 0.0)
    assert grazing[2] == pytest.approx(0.0, abs=1e-16)
    assert grazing[0] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 3. The boundaries this source may not cross
# ---------------------------------------------------------------------------


def test_no_backend_is_on_the_path() -> None:
    """No Optiland and no Chromatix, asserted two ways.

    Statically, because an import inside a branch nobody exercised would not show
    up dynamically; and dynamically, because reaching for a solver at call time is
    what would make this ensemble a launch. It is arithmetic on the caller's own
    points and nothing else -- aiming one at an entrance pupil needs the stop, the
    pupil, the system NA and the aimer, which is `backends.optiland.launch`.

    This survived the CHE-219 move because it is the assertion that keeps the
    builder honest about what it is *not*. What did not survive was a class-budget
    check: `BUDGETS["sources"]` no longer covers this file, and
    `tests/unit/test_class_budget.py` scans `src/` rather than `tests/`.
    """
    tree = ast.parse(MODULE.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])

    assert imported <= {"__future__", "math", "typing", "numpy", "numerics", "representations"}
    for forbidden in ("optiland", "chromatix", "backends", "couplers", "operators", "problems"):
        assert forbidden not in imported

    for forbidden in ("optiland", "chromatix"):
        before = forbidden in sys.modules
        a_bundle()
        assert (forbidden in sys.modules) == before


def test_the_module_defines_no_class() -> None:
    """Two functions and no object, which is what kept the move a file move.

    A `CollimatedLaunch` frozen dataclass was the candidate with a real argument --
    the direction and the medium index are coupled, and the optical path depends on
    both. It did not land because the coupling is not an *invariant* the object
    could check: `n (d_hat . r)` is a computation over the caller's points, so the
    object would carry three fields and validate strictly less than this function.
    Kept after CHE-219 as a property of the helper rather than as a budget check,
    because a test fixture that grew a class hierarchy would be the same mistake
    one directory over.
    """
    source = MODULE.read_text(encoding="utf-8")
    assert [n.name for n in ast.walk(ast.parse(source)) if isinstance(n, ast.ClassDef)] == []
