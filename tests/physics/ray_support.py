"""The three ray ensembles R07's tests reconstruct from, and their analytic oracles.

A module rather than fixtures in a `conftest.py`, following
`tests/solvers/chromatix_support.py`: four test files build the same bundles and a
conftest would load them for the rest of the suite as well.

Every builder here is **analytic**. None of them calls a solver, so none of the
numbers a test compares against came out of this repository's numerical code --
which is the rule that makes these gates independent evidence rather than
characterization (`AGENTS.md`, "Scientific Non-Negotiables"). The three ensembles
and the oracle each one exists for:

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

import math
from typing import Any

import numpy as np

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
    dtype: Any = np.float64,
) -> tuple[RayBundle, Any, float]:
    """One angular mode launched from every point of a grid. Returns `(rays, d_hat, dA)`.

    `length_scale` multiplies every length -- pitch, position and optical path --
    and is the metre-for-millimetre twin of checklist item 4: at `1000.0` the
    same geometry is read in the wrong unit and `k * OPL` scales by a thousand.

    `optical_path_sign` negates the optical path, and is the negative twin of
    checklist item 1: it conjugates every wavelet, so a converging wavefront
    reconstructs as a diverging one and no intensity check can tell.
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
    return (
        RayBundle(
            positions_m=positions.astype(dtype),
            directions=np.tile(d_hat, (count, 1)).astype(dtype),
            wavelength_m=wavelength_m,
            reference_surface=a_surface(z_m=z_m * length_scale),
            amplitude=np.ones(count, dtype=dtype),
            optical_path_m=(optical_path_sign * (positions @ d_hat)).astype(dtype),
            optical_path_reference="the global origin, along d_hat",
            measure_weight=np.full(count, dy * dx, dtype=dtype),
            measure_kind="quadrature_area_m2",
        ),
        d_hat,
        dy * dx,
    )


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
