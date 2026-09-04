"""An analytic spherical wave, as a fully declared `ScalarField`.

CHE-215 (R06.10), item 3. One public function:

```python
sources.spherical_wave(shape, *, sample_pitch_m, wavelength_m, reference_surface,
                       source_position_m, amplitude=1.0, converging=False,
                       namespace=ArrayNamespace.NUMPY, device=None) -> ScalarField
```

    E(r) = A (R_ref / R) exp(+/- i n k0 R),    R = |r - r_s|,    R_ref = 1 m

sampled on the plane `reference_surface.z_m`. This is the project's point emitter:
a **wave-optics** point source is its analytic spherical field with explicit
geometry and checked sampling, not a single nonzero pixel on a grid -- which is a
delta whose spectrum is flat to the grid's Nyquist limit and therefore aliased by
construction.

**This module reverses part of a landed declaration.** `sources/__init__.py` used
to exclude point sources. The exclusion was lifted on the owner's decision for
this ticket; the reasoning and what is still excluded are recorded in that package
docstring, which is this package's canonical prose.

The sign, pinned to the phasor convention
------------------------------------------
The project's conventions are `PHASOR = exp(-i omega t)` and `SPATIAL_FACTOR =
exp(+i k z)`, so a wave travelling in `+z` accumulates `+i k` per metre. Read
against that:

* **diverging** (`converging=False`, the default) is `exp(+i n k0 R)`. The phase
  grows with distance *from* the source, so the wave travels away from it. For the
  field on this plane to be travelling forward, the source belongs **upstream**:
  `source_position_m[2] < reference_surface.z_m`.
* **converging** (`converging=True`) is `exp(-i n k0 R)`. The phase grows toward
  the source, so the wave travels *into* it, and the focus belongs **downstream**:
  `source_position_m[2] > reference_surface.z_m`.

The two are exact complex conjugates of each other, which is precisely why the sign
is dangerous: conjugating the field turns a converging wavefront into a diverging
one with **no signature in any intensity** anywhere. `tests/physics/ray_support`
carries the same trap as a deliberate negative control (`optical_path_sign`), and
`tests/sources/test_spherical_wave.py` asserts the conjugation relation rather than
trusting it.

A boolean rather than a `direction="diverging"` string: one bit, and no string
vocabulary for two call sites to spell differently. Note that the pairing above is
**documented and not refused** -- `converging=False` with a downstream source is
the conjugate field, i.e. one travelling in `-z`, which this project's forward
`SPATIAL_FACTOR` cannot carry. It is left expressible because refusing it is a
convention decision this ticket was not scoped to make.

`amplitude` is dimensional, and that is declared
-------------------------------------------------
The `1/R` makes `A` an amplitude **at a reference distance**, not a peak. The
declared convention, decided by the owner on this ticket because the tree had no
precedent:

    A is the field amplitude at R = R_ref = 1 metre from the source.

So the value on the grid is `A / R` with `R` in metres -- a source 10 mm away
gives 100 A, and a source 1 m away gives A. The alternative, peak-normalizing to
`A` at the grid's closest sample, was rejected: it hides the absolute `1/R` scale,
and the absolute scale is the one thing an intensity-only or peak-normalized check
cannot recover. `plane_wave` and `gaussian_beam` do state a *peak* amplitude; the
difference is real and is why it is spelled out here rather than assumed to carry
over.

Under-sampling is refused, not warned
--------------------------------------
The local transverse spatial frequency of `exp(i n k0 R)` is its own phase
gradient, per axis:

    k_y(r) = n k0 (y - y_s) / R        k_x(r) = n k0 (x - x_s) / R

which grows with lateral offset from the source and is largest at whichever grid
corner is furthest from it. Compared against `pi / d` per axis -- the *same* bound
`sources._grid.nyquist_limit_rad_per_m` gives `plane_wave` -- and refused with
`REPRESENTATION_INCONSISTENT` naming the geometry that failed.

The same failure, and the same refusal, as an aliased tilt: an under-sampled
spherical wave reads back as a *different and entirely plausible* geometry, since
the aliased local frequency corresponds to a real ray at a smaller angle. One
source refusing this while the other warned would be the convention drift this
ticket set out to avoid.

The geometric limit that follows, worth stating in a form a caller can size a grid
with:

    |rho - rho_s| / R  <  lambda_0 / (2 n d)        i.e.  sin(theta_local) < lambda_0 / (2 n d)

and since `NA = n sin(theta)`, the numerical-aperture form is `NA < lambda_0 /
(2 d)` -- independent of `n`, because a medium shortens the wavelength and widens
the acceptance angle by the same factor. At `lambda_0 = 532 nm` a pitch of 200 nm
therefore carries up to `NA = 1.33`, and a pitch of 1 um carries `NA = 0.27`.

Two more refusals, both about the `1/R` singularity
----------------------------------------------------
A source **on** the sampled plane is refused outright: `R` reaches zero if the
source lands on a sample, and where it does not, `|rho - rho_s| / R` approaches 1
and the sampling refusal above fires anyway with a less useful message. And any
geometry whose smallest `R` on the grid falls below one sample pitch is refused,
because a `1/R` evaluated inside a sample is a number the grid does not represent
-- the field varies by orders of magnitude between adjacent samples.

No aperture argument, ever
---------------------------
Truncation **composes**: `spherical_wave(...)` followed by the thin-element
operator R06.6 landed is how a truncated spherical wave is expressed, and it is
strictly more expressive than an `aperture=` argument here would be (any complex
mask, not just a hard disc). No source in this package inspects downstream
elements for an NA, a stop or a launch cone, and none infers one. That is the
solver/problem layer's job (CHE-207, R05.5).

`complex64` with a float64 radius and phase, `Frame.origin_index` for the grid
origin, `validity=frozenset()` because an analytic spherical wave is exact at its
declared surface -- all as `plane_wave`.
"""

from __future__ import annotations

import math

import numpy as np

from numerics import ArrayNamespace, DevicePlacement
from representations import ContractError, Frame, ReferenceSurface, ScalarField
from representations.contracts import require_positive_si
from sources._grid import (
    SOURCE_DTYPE,
    deliver,
    grid_coordinates,
    nyquist_limit_rad_per_m,
    require_grid_shape,
    require_phase_accumulation,
    require_sample_pitch,
)

__all__ = ["spherical_wave"]

#: The declared distance `amplitude` is the field amplitude at. One metre, in SI,
#: stated as a constant so the docstring and the arithmetic cannot drift.
REFERENCE_DISTANCE_M = 1.0


def spherical_wave(
    shape: tuple[int, int],
    *,
    sample_pitch_m: tuple[float, float],
    wavelength_m: float,
    reference_surface: ReferenceSurface,
    source_position_m: tuple[float, float, float],
    amplitude: float = 1.0,
    converging: bool = False,
    namespace: ArrayNamespace = ArrayNamespace.NUMPY,
    device: DevicePlacement | None = None,
) -> ScalarField:
    """A diverging or converging spherical wave sampled on `reference_surface`.

    Args:
        shape: `(ny, nx)`.
        sample_pitch_m: `(dy, dx)` in metres. No default and no inference.
        wavelength_m: vacuum wavelength in metres.
        reference_surface: the plane the field is sampled on. Its `z_m` is the
            plane's axial coordinate and its `medium_index` is the `n` in
            `n k0 R`; neither is defaulted.
        source_position_m: `(x_s, y_s, z_s)` in metres -- **`(x, y, z)` order**,
            matching `RayBundle.positions_m` columns, *not* the `(y, x)` order of
            this module's grid-shaped arguments. Must be off the sampled plane;
            see the docstring on which side each sign belongs on.
        amplitude: the field amplitude at `R = 1 m` from the source, **not** a
            peak. Dimensional; see the module docstring.
        converging: `False` (default) gives `exp(+i n k0 R)`, diverging;
            `True` gives `exp(-i n k0 R)`, converging. The two are complex
            conjugates and no intensity can tell them apart.
        namespace: which array namespace the field is returned in. `numpy` (the
            default) reproduces this function's behaviour before CHE-246 (T2)
            exactly. `jax` is the wave path's GPU entry point; `torch` is refused,
            because a representation holds data in a compute namespace.
        device: where the returned buffer lives. `None` means wherever the
            namespace puts a new array. **The arithmetic does not move**: `R` is
            accumulated in host float64 and cast once, because JAX cannot
            represent float64 in this process and a source that accumulated in
            the target would silently break the float64 validity line this record
            carries. See `sources/_grid.py`.

    Returns:
        A `ScalarField` of `complex64`, `validity=frozenset()`.

    Raises:
        ContractError: `REPRESENTATION_INCONSISTENT` for a source on the sampled
            plane, a minimum `R` below one sample pitch, or a local transverse
            spatial frequency past `pi/d` on either axis -- each naming the
            geometry that failed. Also whatever the `ScalarField` contract refuses
            about the pitch, the wavelength and the array.
        ValueError: a non-positive axis length, a non-finite `source_position_m`,
            or a non-finite / non-positive `amplitude`.
    """
    counts = require_grid_shape(shape)
    pitch = require_sample_pitch(sample_pitch_m)
    wavelength = require_positive_si(wavelength_m, name="wavelength_m")
    reference_amplitude = require_positive_si(amplitude, name="amplitude")

    position = tuple(float(value) for value in source_position_m)
    if len(position) != 3 or not all(math.isfinite(value) for value in position):
        raise ValueError(
            f"source_position_m must be a finite (x_s, y_s, z_s) triple in metres, got "
            f"{source_position_m!r}"
        )

    index = reference_surface.medium_index
    x_s, y_s, z_s = position
    axial_offset = float(reference_surface.z_m) - z_s
    if axial_offset == 0.0:
        raise ContractError(
            "REPRESENTATION_INCONSISTENT",
            f"the source sits on the sampled plane (z_s = {z_s} m = reference_surface.z_m). "
            "R reaches zero if it lands on a sample, and the local transverse frequency "
            "approaches n k0 everywhere else, so no grid represents this field. A "
            "diverging source belongs upstream of the plane and a converging focus "
            "downstream of it.",
            declaration="source_position_m",
            remedy="Move the source off the plane, or declare the field on another surface.",
        )

    frame = Frame()
    dy, dx = pitch
    y, x = grid_coordinates(counts, pitch, frame)
    # float64 throughout, cast once. `n k0 R` is ~1e7 rad per metre of radius, so
    # the radius is the quantity that has to be accumulated in double.
    lateral_y = y[:, None] - y_s
    lateral_x = x[None, :] - x_s
    radius = np.sqrt(lateral_y**2 + lateral_x**2 + axial_offset**2)

    closest = float(np.min(radius))
    finest_pitch = min(dy, dx)
    if closest < finest_pitch:
        raise ContractError(
            "REPRESENTATION_INCONSISTENT",
            f"the closest grid sample is {closest:.6g} m from the source, inside one sample "
            f"pitch ({finest_pitch} m). The 1/R amplitude changes by orders of magnitude "
            "between adjacent samples there, so the sampled field is a discretization of a "
            "singularity rather than a spherical wave.",
            declaration="source_position_m",
            remedy="Move the source further from the plane, or refine the pitch.",
        )

    # The local transverse spatial frequency is the phase gradient itself, per
    # axis, and it is largest at whichever corner is furthest from the source.
    medium_wavenumber = 2.0 * math.pi * index / wavelength
    for lateral, step, axis, label in (
        (lateral_y, dy, "k_y", "dy"),
        (lateral_x, dx, "k_x", "dx"),
    ):
        local = medium_wavenumber * float(np.max(np.abs(lateral) / radius))
        nyquist = nyquist_limit_rad_per_m(step)
        if local > nyquist:
            sine = local / medium_wavenumber
            raise ContractError(
                "REPRESENTATION_INCONSISTENT",
                f"the largest local |{axis}| = {local:.6g} rad/m is past this grid's Nyquist "
                f"limit pi/d = {nyquist:.6g} rad/m at a {label} of {step} m. The geometry "
                f"that does it: a source at (x, y, z) = {position} m against a plane at "
                f"z = {reference_surface.z_m} m, giving sin(theta_local) = {sine:.4g} at the "
                f"worst sample where this grid carries at most "
                f"{wavelength / (2.0 * index * step):.4g} (NA "
                f"{wavelength / (2.0 * step):.4g}). The sampled phase would alias, and an "
                "aliased spherical wave reads back as a different and entirely plausible "
                "geometry.",
                declaration="source_position_m",
                remedy=(
                    "Refine the pitch, shrink the grid, or move the source further from "
                    "the plane."
                ),
            )

    sign = -1.0 if converging else 1.0
    # `radius` and not a phase: for this source the accumulated real quantity IS
    # the radius, and `n k0 R` is ~1e7 rad per metre of it at visible wavelengths.
    # Assigned rather than called for its check, as in the other two sources, so
    # that a `verify_dtype` which ever returned a converted array is not ignored.
    radius = require_phase_accumulation(radius, source="spherical_wave")
    u = (
        reference_amplitude
        * (REFERENCE_DISTANCE_M / radius)
        * np.exp(1j * sign * medium_wavenumber * radius)
    ).astype(SOURCE_DTYPE)

    return ScalarField(
        u=deliver(u, namespace=namespace, device=device),
        sample_pitch_m=(dy, dx),
        wavelength_m=wavelength,
        reference_surface=reference_surface,
        frame=frame,
        validity=frozenset(),
    )
