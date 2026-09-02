"""A collimated ensemble as a `RayBundle`, for the tests that need one. Not a source.

CHE-219 (R05.8) moved this out of `src/sources/`, where CHE-215 (R06.10) had
landed it as `sources.collimated_bundle`. Two functions, unchanged arithmetic:

```python
collimated_bundle(positions_m, *, direction=(0, 0, 1), wavelength_m,
                  reference_surface, amplitude=1.0,
                  measure_weight=None, measure_kind="undeclared") -> RayBundle
direction_from_angle(theta_rad, phi_rad) -> tuple[float, float, float]
```

Why it is a test helper and not production architecture
--------------------------------------------------------
Because it builds a launch `RayBundle` from caller-supplied points and a
direction, **with no optical system in scope**. That operation cannot say whether
those points are the entrance pupil, the stop, the first traced surface, a valid
finite-conjugate aim, or anything in the constructed system at all -- and R05.8's
rule is that a source may be described without a system while a ray *launch* may
not. The actual launch positions and directions depend on the stop, the entrance
pupil's location and diameter, every surface preceding the stop, the object
distance, the field, the backend's pupil map and the ray aimer, so a system-launch
`RayBundle` is produced by `solvers.optiland.launch` and nowhere else.

Keeping this here rather than deleting it costs nothing and loses no coverage: it
is what `tests/physics/ray_support.py` builds its plane-wave-mode ensembles from
and what `tests/physics/test_collimated_ensemble.py` holds to the closed form
`OPL_j = n (d_hat . r_j)`. What it must not be is the *architecture* -- a second,
system-independent way to initialize rays living beside the system-aware one is
exactly the ambiguity R05.8 removed.

Explicit positions, not a grid
-------------------------------
The primitive takes `(N, 3)` launch points and nothing else. A grid-only signature
would be smaller to call and would be the wrong primitive twice over: it cannot
express the hexapolar pupil sampling `tests/physics/ray_support.converging_bundle`
already uses, and it would tie the ensemble to a rectangular aperture model.
Callers that want a rectangular grid build the positions themselves.

**Axis order is a real trap and is stated once here.** Positions are `(x, y, z)`
*columns*, matching `RayBundle.positions_m` and the `(x, y, z)` order of
`direction`. Grids in this project are `(y, x)`. So a caller building positions
from a `(ny, nx)` meshgrid has to column-stack `x` before `y`, and a swap is
invisible on any square grid -- which is why the tests here use `ny != nx`.

The optical path is the physics, not decoration
------------------------------------------------
`optical_path_m = n (d_hat . r_j)` with `n = reference_surface.medium_index`, and
`optical_path_reference` is stated rather than left off. Those phases are what
make the ensemble *one plane-wave mode* rather than N independent wavelets that
happen to point the same way: with them, `couplers.ray_to_scalar` reconstructs
`N dA exp(+i n k0 d_hat . r)` exactly, and without them it reconstructs something
with no analytic form at all. `n` multiplies the geometric projection because an
optical path is `n` times a geometric one -- the same convention
`operators.propagate_rays` advances by (`n s`) and the same one
`couplers.ray_to_scalar` reads back.

`RayBundle` refuses an optical path with no declared reference, so the string is
not optional. It is fixed here rather than taken as an argument: this is what
chose the origin, so a caller-supplied reference could only disagree with the
arithmetic above.

The measure is left undeclared, deliberately
---------------------------------------------
`measure_weight=None`, `measure_kind="undeclared"` by default -- `RayBundle`'s own
defaults, which make `couplers.ray_to_scalar` refuse the bundle until someone
states what the samples are.

From explicit positions there is no `dA` to derive: the same `(N, 3)` array is a
uniform grid of cell area `dy dx`, a hexapolar pupil whose cells are not all
equal, and an importance-weighted draw, and those three differ by the aperture
area and by whether the reconstruction owes a `1/N`. R05 moved the quadrature
weight off the amplitude and R07's kernel applies `measure_weight` itself, so a
silently defaulted `dA` would scale *every* downstream reconstruction by a factor
no intensity check can see. A caller who knows the sampling passes the weight and
its kind; this function will not guess.

Every other convention, also declared
--------------------------------------
* **Wavelength** -- vacuum `lambda` in metres, one value: a bundle is
  monochromatic per evaluation and a spectrum is several bundles.
* **Direction** -- one shared unit vector, normalized here in float64 so the
  bundle passes `direction_norm_tolerance(dtype)` on the first try rather than
  making the caller guess the tolerance.
* **Reference surface** -- caller-declared, carrying `z_m` and `medium_index`.
  `medium_index` has no default anywhere in the tree and does not get one here.
  Note that this function does **not** move the positions onto `z_m`: the launch
  points are the caller's declaration, and silently projecting them would be a
  propagation.
* **Amplitude** -- a uniform real peak amplitude, one number, unnormalized. An
  amplitude and never an intensity. `RayBundle` widens a real amplitude to the
  complex dtype of the same precision, which is what a phase-free launch is.
* **Frame and phasor** -- the project's single `Frame` and `PHASOR`, both from
  `RayBundle`'s own defaults.
* **Dtype** -- read off the positions the caller passed, so a float32 request
  really is a float32 bundle. The projection `d_hat . r` is accumulated in float64
  and cast once, the same discipline `sources.plane_wave` applies to its phase
  ramp: `k` is ~1.2e7 rad/m, so a millimetre of path is ~1e4 rad and float32
  arithmetic in the accumulation would cost far more than the cast does.

**No backend.** No Optiland, no system geometry, no pupil, no aperture, no stop --
which is precisely why this is not a launch.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from numerics import array_state, dtype_of, numpy_dtype, xp_for
from representations import ContractError, RayBundle, ReferenceSurface
from representations.contracts import adopt_array, require_positive_si
from representations.rays import MeasureKind

__all__ = ["collimated_bundle", "direction_from_angle"]


def direction_from_angle(theta_rad: float, phi_rad: float) -> tuple[float, float, float]:
    """The unit direction cosine `(d_x, d_y, d_z)` at polar `theta`, azimuth `phi`.

    The twin of `sources.transverse_wavevector_from_angle`, and deliberately the
    same convention rather than a second one:

        d_x = sin(theta) cos(phi)   d_y = sin(theta) sin(phi)   d_z = cos(theta)

    `phi` is measured from `+x` toward `+y`, so `phi = 0` tilts in `+x` and
    `phi = pi/2` tilts in `+y`. Because the two functions agree, a collimated
    bundle at `direction_from_angle(theta, phi)` and a plane wave at
    `transverse_wavevector_from_angle(theta, phi, ...)` describe the same mode:
    `k_t = n k0 (d_y, d_x)`, in that axis order.

    Pure floats, no array and no representation, so it can be called before a
    bundle exists. The return order is `(x, y, z)`, matching
    `RayBundle.positions_m` columns -- *not* the `(y, x)` order the grid-shaped
    arguments elsewhere in this package use.

    Raises:
        ValueError: a non-finite argument, or `|theta| > pi/2`. Past `pi/2`,
            `cos(theta)` goes negative while `sin(theta)` starts decreasing again,
            so a backward-going ray would be returned as a plausible forward tilt
            of the wrong magnitude. Rotate `phi` by `pi` instead.
    """
    theta = float(theta_rad)
    phi = float(phi_rad)
    if not all(math.isfinite(value) for value in (theta, phi)):
        raise ValueError(
            f"direction_from_angle needs finite arguments, got theta_rad={theta_rad!r}, "
            f"phi_rad={phi_rad!r}"
        )
    if abs(theta) > 0.5 * math.pi:
        raise ValueError(
            f"theta_rad={theta_rad!r} is beyond pi/2, which would return a backward-going "
            "ray as a forward tilt of the wrong magnitude. Rotate phi by pi instead of "
            "taking theta past grazing."
        )

    radial = math.sin(theta)
    return (radial * math.cos(phi), radial * math.sin(phi), math.cos(theta))


def _require_unit_direction(
    direction: tuple[float, float, float],
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """`direction` as an exactly normalized float64 `(3,)`.

    Normalizing here rather than requiring a unit vector from the caller is what
    makes `direction=(0, 0, 1)` and `direction=(1, 1, 1)` both legal statements of
    intent. `RayBundle` checks the norm against `direction_norm_tolerance(dtype)`,
    which is a bound on *round-off* and not a licence to hand it something that
    misses by a percent.

    Raises:
        ValueError: not a finite three-vector, or a zero vector -- which states no
            direction at all and would normalize to NaN.
    """
    values = tuple(float(value) for value in direction)
    if len(values) != 3 or not all(math.isfinite(value) for value in values):
        raise ValueError(
            f"direction must be a finite (d_x, d_y, d_z) triple, got {direction!r}"
        )
    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0.0:
        raise ValueError(
            "direction is the zero vector, which states no propagation direction; "
            "normal incidence is (0, 0, 1)"
        )
    return np.asarray(values, dtype=np.float64) / norm


def collimated_bundle(
    positions_m: Any,
    *,
    direction: tuple[float, float, float] = (0.0, 0.0, 1.0),
    wavelength_m: float,
    reference_surface: ReferenceSurface,
    amplitude: float = 1.0,
    measure_weight: Any | None = None,
    measure_kind: MeasureKind = "undeclared",
) -> RayBundle:
    """One angular mode launched from every point of `positions_m`.

    Args:
        positions_m: `(N, 3)` launch points in metres, as `(x, y, z)` **columns**.
            See the module docstring: grids in this project are `(y, x)`, so this
            is the one place the two orders meet. The array's dtype, device and
            namespace are the bundle's.
        direction: one `(d_x, d_y, d_z)` shared by every ray, normalized here.
            `(0, 0, 1)` is normal incidence. Build it from an angle with
            `direction_from_angle`.
        wavelength_m: vacuum wavelength in metres.
        reference_surface: the surface these rays are declared on. Its
            `medium_index` is the `n` in the optical path and is not defaulted
            here or anywhere else. The positions are *not* projected onto its
            `z_m`; that would be a propagation.
        amplitude: uniform real peak amplitude per ray, unnormalized. An
            amplitude, never an intensity.
        measure_weight: `(N,)` sampling measure, or `None`. Not derived from the
            positions -- see the module docstring on why guessing a `dA` would
            scale every downstream reconstruction.
        measure_kind: what `measure_weight` is. `"undeclared"` with no weight is
            the default, and `couplers.ray_to_scalar` refuses such a bundle.

    Returns:
        A `RayBundle` of `N` rays sharing one direction, carrying
        `optical_path_m = n (d_hat . r)` with `optical_path_reference` stated.

    Raises:
        ContractError: `positions_m` is not `(N, 3)`, or whatever the `RayBundle`
            contract refuses about the wavelength, the measure pair, the
            finiteness of the arrays or their agreement in dtype/device/namespace.
        ValueError: a non-finite or zero `direction`, or a non-finite /
            non-positive `amplitude`.
    """
    positions = adopt_array(positions_m, name="positions_m", complex_=False)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ContractError(
            "SHAPE_MISMATCH",
            f"positions_m must be (N, 3) launch points as (x, y, z) columns, got "
            f"{tuple(positions.shape)}. Note the column order: grids in this project are "
            "(y, x), so a meshgrid has to be column-stacked x before y.",
            declaration="positions_m",
        )

    unit = _require_unit_direction(direction)
    peak = require_positive_si(amplitude, name="amplitude")

    xp = xp_for(array_state(positions).namespace)
    real_np = numpy_dtype(dtype_of(positions))
    count = int(positions.shape[0])

    # float64 for the projection, cast once -- see the module docstring's dtype
    # note. `d_hat . r` is a sum of three products, so this costs nothing and it
    # is the number `k` multiplies.
    high_precision = positions.astype(np.float64)
    projection = high_precision @ xp.asarray(unit, dtype=high_precision.dtype)
    optical_path = (reference_surface.medium_index * projection).astype(real_np)

    return RayBundle(
        positions_m=positions,
        directions=xp.tile(xp.asarray(unit, dtype=real_np), (count, 1)),
        wavelength_m=require_positive_si(wavelength_m, name="wavelength_m"),
        reference_surface=reference_surface,
        amplitude=xp.full((count,), peak, dtype=real_np),
        optical_path_m=optical_path,
        # The plane through the global origin normal to `d_hat`, which is what
        # `n (d_hat . r)` measures from. The same string the test helper this
        # source replaced declared, so a reconstruction's oracle is unchanged.
        optical_path_reference="the global origin, along d_hat",
        measure_weight=measure_weight,
        measure_kind=measure_kind,
    )
