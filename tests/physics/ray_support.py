"""The three ray ensembles R07's tests reconstruct from, and their analytic oracles.

A module rather than fixtures in a `conftest.py`, following
`tests/solvers/chromatix_support.py`: four test files build the same bundles and a
conftest would load them for the rest of the suite as well.

Every builder here is **analytic**. None of them calls a backend solver, and --
the part that carries the independence claim -- **none of the oracles came out of
this repository's numerical code**: every one below is a closed form written from
the physics, which is what makes these gates independent evidence rather than
characterization (`AGENTS.md`, "Scientific Non-Negotiables").

One caveat on where the *ensemble* arithmetic lives, stated rather than left
implicit. `collimated_bundle` here delegates to `fixtures.ray_bundles`, which is
the same code CHE-215 (R06.10) landed as `sources.collimated_bundle` and CHE-219
(R05.8) moved out of `src/` -- a launch `RayBundle` built from caller-supplied
points with no optical system in scope is not a source, because nothing in it can
say whether those points are the entrance pupil, the stop, a valid aim, or the
system at all. Nothing about these gates changed with the move: the arithmetic is
byte-for-byte the same and the oracle side was always independent, which is where
the gate's force lives. The `n (d_hat . r)` optical path is asserted directly
against the closed form in `tests/physics/test_collimated_ensemble.py`, so a
defect in the ensemble builder shows up there as well and not only as a residual
here.

The three ensembles and the oracle each one exists for:

`collimated_bundle`
    SI Figure S1c: one angular mode, many launch points, each given the optical
    path its lateral position implies, `OPL_j = d_hat . r_j`. With those phases
    the ensemble *is* a single plane-wave mode and the oracle
    `exp(+i k d_hat . r)` is exact, so the tolerance comes from dtype round-off
    rather than from a choice. Without them it is not a mode at all, which is
    what makes this the sharpest available check on the ray-to-scalar direction.

`mode_bundle`
    Every propagating plane-wave mode of a given field, as one ray each. The
    oracle is the truncated inverse DFT of the field's own spectrum, so a
    reconstruction that preserves the field must return it to round-off. This is
    the ensemble the projection-convention finding was measured on, and the one
    that reaches grazing incidence when the grid pitch is fine enough.

`a_random_field`
    A seeded random complex field on a small non-square grid. The source of the
    round-trip gates: a random field has content in every propagating mode, so a
    decomposition that drops or mis-weights any of them shows up.

`converging_bundle`
    A hexapolar-sampled uniform pupil converging at `focal_m`, declared **at the
    focal plane**. The oracle is stationary phase:
    `int dA exp(i k rho^2 / 2R) = i lambda R (1 - exp(i pi a^2 / (lambda R)))`,
    so the focal peak is `lambda R * |1 - exp(i phi_max)|` and exactly
    `lambda R` when `a^2 = lambda R / 3`. It fixes the *absolute scale* of the
    launch amplitude, which no peak-normalized metric can see.

The measure each one declares, and why
--------------------------------------
`RayBundle.measure_kind` has to be declared or the coupler refuses, so each
builder states the measure its own construction implies -- and they are not the
same measure, which is the point:

* the two aperture-sampled ensembles carry `quadrature_area_m2`, a physical cell
  area in m^2, and take no `1/N`;
* the mode enumeration carries `importance_weight`, dimensionless, because a
  retained *subset* of a grid's modes is an estimator of the whole spectrum and
  owes the `1/N` an estimator owes.

  Its weight is derived, not chosen. The truncated inverse DFT the oracle uses
  is `(1 / (ny nx)) sum_kept Chat exp(...)`; the coupler computes
  `(1 / N_kept) sum_kept Chat w exp(...)`. Equating them gives
  `w = N_kept / (ny nx)` for every mode. Note that this is **not** `1 / p` for a
  uniform draw over the retained set, which would be `N_kept`: the estimator
  being formed is of the *whole grid's* inverse transform, not of the retained
  subset's mean, so the reciprocal density that appears is the retained
  fraction. The `1/N` is still real and still load-bearing -- dropping it, or
  the weight, changes the answer by `ny nx / N_kept`.

The `n // 2` origin, stated once
--------------------------------
The coupler puts coordinate zero at array index `n // 2`, so a field built with
index 0 at the array origin has to be `fftshift`ed before it can be compared
against a reconstruction. `shifted_inverse_dft` does that in one place; doing it
at each call site is how a half-grid roll becomes "a coupler bug".
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any

import numpy as np
from fixtures.ray_bundles import collimated_bundle as collimated_source

from representations import RayBundle, ReferenceSurface

#: The M3 reference wavelength and the singlet focal length the frozen `lambda R`
#: record was taken at, reused so the reproduced number is comparable to it.
WAVELENGTH_M = 0.55e-6
FOCAL_M = 4.8375e-3


def a_surface(name: str = "handoff", *, z_m: float = 0.0, medium_index: float = 1.0):
    return ReferenceSurface(name=name, z_m=z_m, medium_index=medium_index)


def shifted_inverse_dft(spectrum: Any) -> Any:
    """The inverse DFT of `spectrum`, on the `n // 2` coordinate origin.

    `exp(i k d_u x)` with `x = (n - nx // 2) dx` equals the DFT kernel times
    `exp(-2 pi i q (nx // 2) / nx)`, i.e. a roll by half the grid -- which is
    `fftshift` for the even grids used here. Written once because a missing roll
    reads as a completely wrong field rather than as an indexing convention.
    """
    return np.fft.fftshift(np.fft.ifft2(spectrum))


def collimated_bundle(
    *,
    shape: tuple[int, int],
    sample_pitch_m: tuple[float, float],
    direction: tuple[float, float, float],
    wavelength_m: float = WAVELENGTH_M,
    z_m: float = 0.0,
    length_scale: float = 1.0,
    optical_path_sign: float = 1.0,
    medium_index: float = 1.0,
    dtype: Any = np.float64,
) -> tuple[RayBundle, Any, float]:
    """One angular mode launched from every point of a grid. Returns `(rays, d_hat, dA)`.

    `length_scale` multiplies every length -- pitch, position and optical path --
    and is the metre-for-millimetre twin of checklist item 4: at `1000.0` the
    same geometry is read in the wrong unit and `k * OPL` scales by a thousand.

    `optical_path_sign` negates the optical path, and is the negative twin of
    checklist item 1: it conjugates every wavelet, so a converging wavefront
    reconstructs as a diverging one and no intensity check can tell.

    `medium_index` declares the surface's medium *and* scales the optical path,
    which is `n` times the geometric one -- the two have to move together or the
    ensemble is not a plane-wave mode in that medium. Its oracle is
    `N dA exp(i n k0 d_hat . r)`.
    """
    ny, nx = shape
    dy, dx = sample_pitch_m[0] * length_scale, sample_pitch_m[1] * length_scale
    y = (np.arange(ny) - ny // 2) * dy
    x = (np.arange(nx) - nx // 2) * dx
    grid_y, grid_x = np.meshgrid(y, x, indexing="ij")
    positions = np.column_stack(
        [grid_x.ravel(), grid_y.ravel(), np.full(grid_x.size, z_m * length_scale)]
    )
    d_hat = np.asarray(direction, dtype=np.float64)
    d_hat = d_hat / np.linalg.norm(d_hat)
    count = positions.shape[0]

    # One ensemble builder for the whole suite (`fixtures.ray_bundles`), so these
    # gates and `test_collimated_ensemble.py` reconstruct from the same code. The
    # rectangular grid stays here on purpose: the builder takes explicit (N, 3)
    # points because binding it to a rectangular aperture model is what it exists
    # not to do.
    rays = collimated_source(
        positions.astype(dtype),
        direction=direction,
        wavelength_m=wavelength_m,
        reference_surface=a_surface(z_m=z_m * length_scale, medium_index=medium_index),
        measure_weight=np.full(count, dy * dx, dtype=dtype),
        measure_kind="quadrature_area_m2",
    )
    if optical_path_sign != 1.0:
        # The conjugate twin, applied *on top* of the honest bundle rather than
        # inside the builder. A negative control has to stay at the call site: a
        # builder that could emit a conjugated wavefront on request is the failure
        # this control exists to detect.
        rays = dataclasses.replace(
            rays, optical_path_m=(optical_path_sign * rays.optical_path_m).astype(dtype)
        )
    return (rays, d_hat, dy * dx)


def mode_bundle(
    field: Any,
    *,
    sample_pitch_m: tuple[float, float],
    wavelength_m: float = WAVELENGTH_M,
    z_m: float = 0.0,
    propagate_m: float = 0.0,
    direction_cosine_floor: float | None = None,
    dtype: Any = np.float64,
) -> tuple[RayBundle, Any, Any]:
    """Every propagating mode of `field` as one ray. Returns `(rays, retained_mask, spectrum)`.

    `propagate_m` advances each mode by that axial distance before the handoff,
    which is what makes the ensemble a *test of the constant phase*: the ray's
    intersection point moves to `(d_u, d_v) Z / d_n` and its optical path to
    `Z / d_n`, so the kernel forms `Z / d_n - (d_u^2 + d_v^2) Z / d_n = Z d_n` as
    a difference of two large numbers. For a near-grazing mode both are enormous
    and the difference is tiny, which is the H4 cancellation.

    `direction_cosine_floor` drops modes with `d_n` below it. The identical mask
    is returned so a comparison oracle can apply it too -- an oracle carrying
    modes the reconstruction excluded measures the exclusion, not the kernel.
    """
    ny, nx = field.shape
    dy, dx = sample_pitch_m
    spectrum = np.fft.fft2(field)
    direction_v, direction_u = np.meshgrid(
        wavelength_m * np.fft.fftfreq(ny, dy),
        wavelength_m * np.fft.fftfreq(nx, dx),
        indexing="ij",
    )
    radial = direction_u**2 + direction_v**2
    axial = np.sqrt(np.clip(1.0 - radial, 0.0, None))
    retained = radial < 1.0
    if direction_cosine_floor is not None:
        retained = retained & (axial >= direction_cosine_floor)

    d_u, d_v, d_n = direction_u[retained], direction_v[retained], axial[retained]
    count = int(d_u.size)
    # Each mode's ray leaves the plane z_m and is caught `propagate_m` later, so
    # its intersection point and its optical path both carry the 1 / d_n that
    # cancels in the constant phase.
    lateral = propagate_m / d_n
    positions = np.column_stack(
        [d_u * lateral, d_v * lateral, np.full(count, z_m + propagate_m)]
    )
    return (
        RayBundle(
            positions_m=positions.astype(dtype),
            directions=np.column_stack([d_u, d_v, d_n]).astype(dtype),
            wavelength_m=wavelength_m,
            reference_surface=a_surface("plane", z_m=z_m + propagate_m),
            # The complex counterpart of `dtype`, so a float32 request really is a
            # float32 bundle: `_compute_precision` takes the max over geometry,
            # amplitude and optical path, and a complex128 amplitude beside float32
            # geometry would silently pull the whole reconstruction to FP64.
            amplitude=spectrum[retained].astype(
                np.complex64 if np.dtype(dtype) == np.float32 else np.complex128
            ),
            optical_path_m=(propagate_m / d_n).astype(dtype),
            optical_path_reference="the plane z_m, along each mode's own direction",
            measure_weight=np.full(count, count / (ny * nx), dtype=dtype),
            measure_kind="importance_weight",
        ),
        retained,
        spectrum,
    )


def hexapolar_disc(rings: int, radius_m: float) -> tuple[Any, Any, Any]:
    """`(rho, phi, area)` for a hexapolar fan on a disc. Ring `j` carries `6j` rays.

    The area rule is the frozen one: interior cells are `pi a^2 / (3 J^2)`, the
    central ray gets `3/4` of a nominal cell because it represents a disc of
    radius `a / 2J`, and the rim ring gets `1/2` because it sits exactly on
    `rho = a` with no ray beyond it to average with. Those weights make
    `sum_i dA_i = pi a^2 (1 + 1 / (4 J^2))` exactly, which is what converges to
    the aperture area under ring refinement instead of staying pinned to the ray
    count.

    Written out here from the closed form rather than imported from
    `solvers.optiland.rays`, which produces the same numbers: importing it would
    pull torch into a coupler test, and would make the solver's implementation the
    oracle for the coupler that consumes it.
    """
    nominal_m2 = math.pi * radius_m**2 / (3.0 * rings**2)
    rho = [0.0]
    phi = [0.0]
    area = [0.75 * nominal_m2]
    for ring in range(1, rings + 1):
        for spoke in range(6 * ring):
            rho.append(radius_m * ring / rings)
            phi.append(2.0 * math.pi * spoke / (6 * ring))
            area.append(nominal_m2 * (0.5 if ring == rings else 1.0))
    return np.asarray(rho), np.asarray(phi), np.asarray(area)


def converging_bundle(
    *,
    rings: int,
    radius_m: float,
    focal_m: float = FOCAL_M,
    wavelength_m: float = WAVELENGTH_M,
    length_scale: float = 1.0,
    measure_kind: str = "quadrature_area_m2",
    dtype: Any = np.float64,
) -> tuple[RayBundle, Any]:
    """A uniform pupil converging at `focal_m`, declared at the focal plane.

    Every ray passes through the focus, so the intersection points coincide and
    the whole reconstruction at `r = 0` is the constant phase: `sum_i dA_i
    exp(i k sqrt(R^2 + rho_i^2))`, the discrete form of the Fresnel integral whose
    closed form fixes the launch amplitude scale.
    """
    rho, phi, area = hexapolar_disc(rings, radius_m * length_scale)
    focal = focal_m * length_scale
    pupil_x, pupil_y = rho * np.cos(phi), rho * np.sin(phi)
    count = int(rho.size)
    directions = np.column_stack([-pupil_x, -pupil_y, np.full(count, focal)])
    directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)
    weight = area if measure_kind == "quadrature_area_m2" else np.ones(count)
    return (
        RayBundle(
            positions_m=np.column_stack(
                [np.zeros(count), np.zeros(count), np.full(count, focal)]
            ).astype(dtype),
            directions=directions.astype(dtype),
            wavelength_m=wavelength_m,
            reference_surface=a_surface("focal plane", z_m=focal),
            amplitude=np.ones(count, dtype=dtype),
            optical_path_m=np.hypot(rho, focal).astype(dtype),
            optical_path_reference="the pupil plane at z = 0, along each ray",
            measure_weight=weight.astype(dtype),
            measure_kind=measure_kind,  # type: ignore[arg-type]
        ),
        area,
    )


def focal_peak_oracle(
    *, radius_m: float, focal_m: float = FOCAL_M, wavelength_m: float = WAVELENGTH_M
) -> float:
    """`lambda R * |1 - exp(i pi a^2 / (lambda R))|` -- stationary phase, closed form.

    Exactly `lambda R` when `a^2 = lambda R / 3`, which is the aperture
    `test_ray_to_scalar.py` uses so the gate is a bare number rather than a
    truncation factor someone has to trust.
    """
    phase = math.pi * radius_m**2 / (wavelength_m * focal_m)
    return wavelength_m * focal_m * abs(1.0 - complex(math.cos(phase), math.sin(phase)))


def plateau_radius_m(*, focal_m: float = FOCAL_M, wavelength_m: float = WAVELENGTH_M) -> float:
    """The pupil radius at which the truncation factor is exactly 1: `a^2 = lambda R / 3`."""
    return math.sqrt(wavelength_m * focal_m / 3.0)


def single_mode_bundle(
    *,
    axial_cosine: float,
    propagate_m: float,
    wavelength_m: float,
    dtype: Any = np.float64,
) -> RayBundle:
    """One plane-wave mode with axial cosine `d_n`, caught `propagate_m` downstream.

    The sharpest possible form of the grazing-mode defect, because there is nothing
    else in the ensemble to average it away. The ray leaves the origin, so its
    intersection point is `(d_u Z / d_n, 0, Z)` and its optical path is `Z / d_n`:
    both scale as `1 / d_n` while the constant phase they differ by is `k Z d_n`,
    which *shrinks* as `d_n` does. The analytic value at the coordinate origin is
    therefore `exp(+i k Z d_n)` exactly, with no truncation and no quadrature -- so
    the realized phase error can be read off directly rather than inferred from a
    field residual.
    """
    transverse = math.sqrt(1.0 - axial_cosine * axial_cosine)
    lateral = transverse * propagate_m / axial_cosine
    complex_dtype = np.complex64 if np.dtype(dtype) == np.float32 else np.complex128
    return RayBundle(
        positions_m=np.array([[lateral, 0.0, propagate_m]]).astype(dtype),
        directions=np.array([[transverse, 0.0, axial_cosine]]).astype(dtype),
        wavelength_m=wavelength_m,
        reference_surface=a_surface("plane", z_m=propagate_m),
        amplitude=np.ones(1, dtype=complex_dtype),
        optical_path_m=np.array([propagate_m / axial_cosine]).astype(dtype),
        optical_path_reference="the plane z = 0, along the mode's own direction",
        measure_weight=np.ones(1, dtype=dtype),
        measure_kind="quadrature_area_m2",
    )


def a_random_field(
    *,
    shape: tuple[int, int] = (24, 32),
    sample_pitch_m: tuple[float, float] = (0.40e-6, 0.35e-6),
    wavelength_m: float = WAVELENGTH_M,
    dtype: Any = np.complex128,
    seed: int = 8,
):
    """A seeded random complex field. Non-square in both count and pitch, on purpose.

    Imported here rather than in the test files so the round-trip gates on both
    sides of the coupler pair compare against the same object. `ScalarField` is
    imported lazily inside the function for no reason other than keeping this
    module's import list to the two names the ray builders need -- see below.
    """
    from representations import ScalarField

    rng = np.random.default_rng(seed)
    u = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(dtype)
    return ScalarField(
        u=u,
        sample_pitch_m=sample_pitch_m,
        wavelength_m=wavelength_m,
        reference_surface=a_surface("plane"),
    )


def propagating_only(field: Any) -> Any:
    """`field` with its evanescent modes removed -- the oracle a round trip must hit.

    An evanescent mode has no propagation direction to give a ray, so a
    `ScalarField -> RayBundle -> ScalarField` round trip cannot return it and must
    not be graded as though it could. Written here from the closed-form transform
    rather than by calling `scalar_to_ray`, so the oracle is not the code under
    test: centred DFT, strict `radial < 1` cut, centred inverse.

    The cut is taken on the **medium** direction cosines `lambda_0 f / n`, read
    from the field's own reference surface: `|k_t| < n k0` is a wider circle in a
    medium, so an oracle written at `n = 1` would grade a submerged round trip
    against the wrong mode set.
    """
    u = np.asarray(field.u)
    ny, nx = u.shape
    dy, dx = field.sample_pitch_m
    wavelength_m = field.wavelength_m / field.reference_surface.medium_index
    spectrum = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(u)))
    direction_v, direction_u = np.meshgrid(
        np.fft.fftshift(np.fft.fftfreq(ny, dy)) * wavelength_m,
        np.fft.fftshift(np.fft.fftfreq(nx, dx)) * wavelength_m,
        indexing="ij",
    )
    keep = direction_u**2 + direction_v**2 < 1.0
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(np.where(keep, spectrum, 0.0))))
