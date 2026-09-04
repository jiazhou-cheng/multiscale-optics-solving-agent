"""A Gaussian beam at its waist plane, as a fully declared `ScalarField`.

CHE-215 (R06.10), item 2. One public function:

```python
sources.gaussian_beam(shape, *, sample_pitch_m, wavelength_m, reference_surface,
                      waist_radius_m, center_m=(0.0, 0.0),
                      transverse_wavevector_rad_per_m=(0.0, 0.0), amplitude=1.0,
                      namespace=ArrayNamespace.NUMPY, device=None) -> ScalarField
```

    E(y, x) = A exp(-((y - y0)^2 + (x - x0)^2) / w0^2) exp(i(k_y y + k_x x))

A real Gaussian envelope times exactly the carrier ramp `plane_wave` writes, so a
`waist_radius_m` much larger than the grid reproduces `plane_wave` to round-off.

**This module reverses part of a landed declaration.** `sources/__init__.py` used
to exclude "Gaussian beams as a source primitive". The exclusion was lifted on the
owner's decision for this ticket; the reasoning and what is still excluded are
recorded in that package docstring, which is this package's canonical prose.

`w0` is the `1/e` **amplitude** radius
---------------------------------------
Stated first because it is the single most misread parameter in Gaussian optics.
The envelope is `exp(-rho^2 / w0^2)`, so at `rho = w0` the **amplitude** is `A/e`
and the **intensity** is `A^2/e^2` -- `w0` is the `1/e` amplitude radius and
therefore the `1/e^2` intensity radius, which is the usual laser-catalogue
convention. The competing reading, `exp(-rho^2 / (2 w0^2))`, differs by a factor of
`sqrt(2)` in the waist and produces an entirely plausible-looking beam of the
wrong size, with no signature anywhere downstream. `tests/sources/
test_gaussian_beam.py` asserts the amplitude at `rho = w0` is exactly `A/e` so the
reading is measured rather than documented.

Waist only, and why arbitrary `z` is a separate ticket
-------------------------------------------------------
At the waist the field is a real envelope times the carrier and nothing else: no
wavefront curvature, no Gouy phase, no `w(z)`. That makes it **exact at its
declared surface**, so `validity=frozenset()` is an honest claim -- the same claim
`plane_wave` makes.

An off-waist Gaussian is not exact: it is a *paraxial* solution, and
`ValidityFlag` in `src/representations/scalar.py` is exactly three tokens --
`surface_only`, `no_wavefront_curvature_term`, `carrier_removed_phase` -- none of
which says "paraxial". Landing `w(z)`, `R(z)` and the Gouy phase therefore means
extending a shared representation contract vocabulary, which is a wider-review
change with its own gate work. So this function takes no `z` argument at all
rather than taking one and mis-declaring its validity; the axial coordinate is the
surface's `z_m`, and the beam is at its waist there by construction. Arbitrary `z`
is a follow-up that owns the new flag, its `VALIDITY_NOTES` entry and the
measurement behind it.

Truncation is documented, not refused
--------------------------------------
A grid of half-extent `L` truncates the envelope at `exp(-L^2 / w0^2)`. The
fraction of the beam's power outside a square of half-width `L` is roughly
`1 - erf(sqrt(2) L / w0)^2` for the intensity `exp(-2 rho^2 / w0^2)`, so a grid
half-extent of `1.5 w0` leaves ~1e-4 of the power outside, `2 w0` leaves ~1e-7,
and `1 w0` leaves ~2e-2. That is a *modelling* choice with a visible consequence
-- a truncated Gaussian rings when it is propagated -- and callers who mean to
clip should be able to. It is therefore stated here and not refused.

What **is** refused is an *unresolved* waist: fewer than two samples across `w0`
on either axis is not a Gaussian beam, it is a discretization artifact that will
read back as a beam of whatever size the grid can represent. That is the same
class of failure as an aliased tilt, so it is refused the same way.

The two carrier refusals are shared, not copied
------------------------------------------------
`|k_t| <= n k0` and `|k_t| <= pi/d` per axis come from `sources._grid`, the same
function `plane_wave` calls. CHE-215 names divergence between two sources'
independently written refusals as its main risk: the symptom would be one source
accepting a geometry the other refuses, with nothing in the suite comparing them.

`complex64` with a float64 envelope and ramp, `Frame.origin_index` for the origin,
peak amplitude with no renormalization (chromatix's `functional.gaussian_beam`
carries a `power=` that this tree does not inherit) -- all as `plane_wave`.
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
    require_grid_shape,
    require_phase_accumulation,
    require_sample_pitch,
    require_transverse_wavevector,
)

__all__ = ["gaussian_beam"]

#: Samples across `w0` below which the waist is not resolved. Two is the Nyquist
#: floor and nothing more: it is the point below which the envelope's own spatial
#: content aliases, not a recommendation for how to sample a beam.
_MINIMUM_SAMPLES_ACROSS_WAIST = 2.0


def gaussian_beam(
    shape: tuple[int, int],
    *,
    sample_pitch_m: tuple[float, float],
    wavelength_m: float,
    reference_surface: ReferenceSurface,
    waist_radius_m: float,
    center_m: tuple[float, float] = (0.0, 0.0),
    transverse_wavevector_rad_per_m: tuple[float, float] = (0.0, 0.0),
    amplitude: float = 1.0,
    namespace: ArrayNamespace = ArrayNamespace.NUMPY,
    device: DevicePlacement | None = None,
) -> ScalarField:
    """A Gaussian beam at its waist, on `reference_surface`.

    Args:
        shape: `(ny, nx)`.
        sample_pitch_m: `(dy, dx)` in metres. No default and no inference.
        wavelength_m: vacuum wavelength in metres.
        reference_surface: the plane the waist is on. Its `medium_index` is the
            `n` in `|k_t| <= n k0` and is not defaulted here or anywhere else.
        waist_radius_m: `w0`, the `1/e` **amplitude** radius -- equivalently the
            `1/e^2` intensity radius. See the module docstring; this is the
            parameter a factor of `sqrt(2)` hides in.
        center_m: `(y0, x0)` envelope centre in metres, in the frame's own
            coordinates. `(y, x)` order, matching the array axes. Shifting the
            centre moves the envelope only -- the carrier ramp is unchanged, so
            this is a translation and not a tilt.
        transverse_wavevector_rad_per_m: `(k_y, k_x)` in **rad/m**, exactly as
            `plane_wave`. `(0, 0)` is normal incidence. Build it from an angle with
            `transverse_wavevector_from_angle`.
        amplitude: peak amplitude at the envelope centre. Unnormalized.
        namespace: which array namespace the field is returned in. `numpy` (the
            default) reproduces this function's behaviour before CHE-246 (T2)
            exactly. `jax` is the wave path's GPU entry point; `torch` is refused,
            because a representation holds data in a compute namespace.
        device: where the returned buffer lives. `None` means wherever the
            namespace puts a new array. **The arithmetic does not move**: the
            real quantity under the exponent is accumulated in host float64 and
            cast once, because JAX cannot represent float64 in this process and a
            source that accumulated in the target would silently break the
            float64 validity line this record carries. See `sources/_grid.py`.

    Returns:
        A `ScalarField` of `complex64`, `validity=frozenset()`.

    Raises:
        ContractError: `|k_t| > n k0`, `|k_t|` past `pi/d` on either axis, or a
            waist spanning fewer than two samples on either axis. Also whatever
            the `ScalarField` contract refuses about the pitch, the wavelength and
            the array.
        ValueError: a non-positive axis length, a non-finite or non-positive
            `waist_radius_m` or `amplitude`, or a non-finite `center_m`.
    """
    counts = require_grid_shape(shape)
    pitch = require_sample_pitch(sample_pitch_m)
    wavelength = require_positive_si(wavelength_m, name="wavelength_m")
    peak = require_positive_si(amplitude, name="amplitude")
    waist = require_positive_si(waist_radius_m, name="waist_radius_m")
    ky, kx = require_transverse_wavevector(
        transverse_wavevector_rad_per_m,
        pitch=pitch,
        wavelength_m=wavelength,
        medium_index=reference_surface.medium_index,
    )

    center = tuple(float(value) for value in center_m)
    if len(center) != 2 or not all(math.isfinite(value) for value in center):
        raise ValueError(f"center_m must be a finite (y0, x0) pair in metres, got {center_m!r}")

    # An unresolved waist is refused, per axis, for the same reason an aliased
    # tilt is: the field that comes back is a plausible beam of the wrong size.
    for step, axis in zip(pitch, ("dy", "dx"), strict=True):
        samples = waist / step
        if samples < _MINIMUM_SAMPLES_ACROSS_WAIST:
            raise ContractError(
                "REPRESENTATION_INCONSISTENT",
                f"the waist w0 = {waist:.6g} m spans {samples:.3g} samples along {axis} at a "
                f"pitch of {step} m, below the {_MINIMUM_SAMPLES_ACROSS_WAIST:g} this grid "
                "needs to represent it. An unresolved waist is not a Gaussian beam, it is a "
                "discretization artifact that reads back as a beam of whatever size the grid "
                "can carry.",
                declaration="waist_radius_m",
                remedy="Refine the pitch, or widen the waist.",
            )

    frame = Frame()
    dy, dx = pitch
    y0, x0 = center
    # float64 throughout, cast once. The envelope and the ramp are accumulated
    # separately and multiplied, so a wide beam reduces to `plane_wave` exactly.
    y, x = grid_coordinates(counts, pitch, frame)
    # Both quantities, because this record's validity line names both: "the
    # envelope and the phase ramp are accumulated in float64 before the cast".
    # Guarding only the ramp would leave half of a declaration unchecked.
    radial_squared = require_phase_accumulation(
        (y[:, None] - y0) ** 2 + (x[None, :] - x0) ** 2, source="gaussian_beam"
    )
    envelope = require_phase_accumulation(
        np.exp(-radial_squared / (waist * waist)), source="gaussian_beam"
    )
    phase = require_phase_accumulation(
        ky * y[:, None] + kx * x[None, :], source="gaussian_beam"
    )
    u = (peak * envelope * np.exp(1j * phase)).astype(SOURCE_DTYPE)

    return ScalarField(
        u=deliver(u, namespace=namespace, device=device),
        sample_pitch_m=(dy, dx),
        wavelength_m=wavelength,
        reference_surface=reference_surface,
        frame=frame,
        validity=frozenset(),
    )
