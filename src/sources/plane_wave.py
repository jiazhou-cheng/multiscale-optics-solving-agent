"""A declared illumination, as a fully declared `ScalarField`.

CHE-210 (R06.5). Two public functions:

```python
sources.plane_wave(shape, *, sample_pitch_m, wavelength_m, reference_surface,
                   transverse_wavevector_rad_per_m=(0.0, 0.0), amplitude=1.0) -> ScalarField
sources.transverse_wavevector_from_angle(theta_rad, phi_rad, *,
                                         wavelength_m, medium_index) -> tuple[float, float]
```

Normal incidence is `k_t = (0, 0)` -- the same primitive, not a second function.

Why this is a capability and not a test helper
-----------------------------------------------
Every field in the tree before this ticket was built by hand inside a test
module, which is correct for a test and is not a capability: nothing outside a
test could *state* an illumination, and Fourier ptychography (R06.8) is precisely
a sweep over stated illuminations. A source is also the only operation that
creates a representation out of nothing, so its conventions have to be pinned
before anything consumes them.

**No backend.** A plane wave is `A exp(i(k_y y + k_x x))` on the project's own
grid, in float64, cast once. `chromatix.functional.plane_wave` is therefore a
cross-check and not the implementation -- see the unit note below, and note that
it also carries a `power=1.0` default that renormalizes the amplitude, which a
project with no radiometric normalization anywhere
(`ScalarField.discrete_power`) must not inherit by accident.

The angle unit, which is the load-bearing declaration
------------------------------------------------------
The project states the illumination as the **transverse wavevector `k_t = (k_y,
k_x)` in rad/m** -- angular wavenumber -- so the field is literally
`exp(i(k_y y + k_x x))` and nothing is converted at the point of use.

That is not a free choice. `pre-rewrite-2026-08-30:knowledge/solvers/chromatix/
conventions.md` records (CHE-57, measured by sweeping three values against
unclipped lateral displacement) that the *same argument name* `kykx` means
**angular wavenumber** on `plane_wave` and **spatial frequency, cycles per
length** on `asm_propagate` -- one package, one name, a factor of `2 pi` apart.
Re-read from the pinned build: `plane_wave` computes `exp(1j * sum(kykx *
field.grid))`, which is rad/m. A wrong-by-`2 pi` illumination angle produces a
perfectly well-formed image at the wrong place in the pupil, and in Fourier
ptychography that is the one parameter the whole method sweeps over. So the
project declares one unit, converts at the boundary, and
`tests/physics/test_coherent_sources.py` asserts the factor of `2 pi` is
measurable rather than trusting that the two readings happen to agree.

`transverse_wavevector_from_angle` is the converter: `|k_t| = (2 pi n / lambda)
sin(theta)`, with `phi` the azimuth from `+x` toward `+y`, so `k_x = |k_t| cos
phi` and `k_y = |k_t| sin phi`.

Every other convention, also declared
--------------------------------------
* **Wavelength** -- vacuum `lambda` in metres, one value per field.
* **Sampling** -- `(dy, dx)` and `(ny, nx)` are caller-supplied, with no default
  and no inference, and coordinate zero comes from `Frame.origin_index` rather
  than from a rewritten `n // 2`.
* **Frame and phasor** -- the project's single `Frame`, `PHASOR =
  exp(-i omega t)`, `SPATIAL_FACTOR = exp(+i k z)`.
* **Reference surface** -- caller-declared, carrying `z_m` and `medium_index`.
  `medium_index` has no default anywhere in the tree, and the wavevector
  magnitude here is `n k0`, not `k0`.
* **Propagation direction** -- `+z`. The source is the field *at* its declared
  surface, already travelling forward.
* **Amplitude** -- `amplitude` is the **peak amplitude**, uniform across the
  grid. `discrete_power` is left as whatever it comes out as, because it is a
  relative quantity by its own docstring and there is no radiometric
  normalization in this tree to make it anything else. Chromatix's `power=`
  convention is not imported.
* **Coherence** -- fully coherent, monochromatic, scalar. Partial coherence is
  not modelled and is not implied by the word "source".
* **Validity** -- `frozenset()`. An analytic plane wave has no declared
  limitation, and the empty set is that claim rather than the absence of one.

complex64, deliberately
------------------------
The returned array is `complex64`, and the phase ramp is accumulated in
**float64** before the single cast. Both halves matter. `complex128` is not a
storage choice the wave path has -- `numerics.negotiate` refuses it against the
measured chromatix row with `LOSSY_DOWNCAST_REQUIRED` -- so a float64 source
could not be propagated at all, which would make criterion 1 unmeasurable.
Accumulating in float64 first is what keeps the cast honest: `k` is ~1.2e7 rad/m
and a 100 um grid puts ~1e3 rad through the exponent, where float32 arithmetic
would cost ~6e-5 rad instead of the ~6e-8 the cast itself costs.

Placement
---------
`src/sources/` is a **new package and a deliberate architecture change**, made
with the owner's decision on CHE-210: a source maps a problem statement into a
representation, which is the definition of a `solver`, but it has no external
backend and `solvers/<backend>/` is organized per backend. The alternatives were
worse in ways the ticket names: `representations/` would own physics it exists
only to declare, `operators/` is wrong by definition because an operator consumes
a representation, and quietly widening an existing package's remit is the move
the architecture rules exist to prevent. The same change adds the row `sources/
-> problems, representations, numerics` to `docs/architecture_principles.md`
section 3 and to `ALLOWED`/`LANDED` in `scripts/check_dependencies.py`. Only two
of the three declared edges are exercised today: nothing here imports
`problems/`, because the illumination *declaration* that would live there has no
consumer yet.
"""

from __future__ import annotations

import math

import numpy as np

from representations import ContractError, Frame, ReferenceSurface, ScalarField
from representations.contracts import require_positive_si

__all__ = ["plane_wave", "transverse_wavevector_from_angle"]

#: The one storage dtype of the project's wave path. See the module docstring.
_SOURCE_DTYPE = np.complex64


def transverse_wavevector_from_angle(
    theta_rad: float, phi_rad: float, *, wavelength_m: float, medium_index: float
) -> tuple[float, float]:
    """`(k_y, k_x)` in rad/m for a plane wave at polar `theta`, azimuth `phi`.

    Pure float64 arithmetic, no array, no representation:

        |k_t| = (2 pi n / lambda) sin(theta)
        k_x   = |k_t| cos(phi)      k_y = |k_t| sin(phi)

    `phi` is measured from `+x` toward `+y`, so `phi = 0` tilts the beam in `+x`
    and `phi = pi/2` tilts it in `+y`. The return order is `(k_y, k_x)`, matching
    the project's `(y, x)` array axis order, so a caller never has to transpose a
    pair between this function and `plane_wave`.

    Raises:
        ValueError: a non-finite argument, a non-positive wavelength or index, or
            `|theta| > pi/2`. Beyond `pi/2`, `sin(theta)` starts *decreasing*
            again, so a backward-going direction would be returned as a small
            forward tilt -- a plausible answer for a wave travelling the other way.
    """
    theta = float(theta_rad)
    phi = float(phi_rad)
    wavelength = float(wavelength_m)
    index = float(medium_index)
    if not all(math.isfinite(value) for value in (theta, phi, wavelength, index)):
        raise ValueError(
            f"transverse_wavevector_from_angle needs finite arguments, got "
            f"theta_rad={theta_rad!r}, phi_rad={phi_rad!r}, "
            f"wavelength_m={wavelength_m!r}, medium_index={medium_index!r}"
        )
    if wavelength <= 0.0:
        raise ValueError(f"wavelength_m={wavelength_m!r} must be positive")
    if index <= 0.0:
        raise ValueError(f"medium_index={medium_index!r} must be positive")
    if abs(theta) > 0.5 * math.pi:
        raise ValueError(
            f"theta_rad={theta_rad!r} is beyond pi/2. sin(theta) decreases again past "
            "pi/2, so a backward-going direction would come back as a small forward "
            "tilt. Rotate phi by pi instead of taking theta past grazing."
        )

    magnitude = (2.0 * math.pi * index / wavelength) * math.sin(theta)
    return (magnitude * math.sin(phi), magnitude * math.cos(phi))


def plane_wave(
    shape: tuple[int, int],
    *,
    sample_pitch_m: tuple[float, float],
    wavelength_m: float,
    reference_surface: ReferenceSurface,
    transverse_wavevector_rad_per_m: tuple[float, float] = (0.0, 0.0),
    amplitude: float = 1.0,
) -> ScalarField:
    """A fully declared coherent plane wave at `reference_surface`.

    Args:
        shape: `(ny, nx)`.
        sample_pitch_m: `(dy, dx)` in metres. No default and no inference: a
            shape is not an extent.
        wavelength_m: vacuum wavelength in metres.
        reference_surface: the plane the field is declared on. Its
            `medium_index` is the `n` in `|k_t| <= n k0` and is not defaulted
            here or anywhere else.
        transverse_wavevector_rad_per_m: `(k_y, k_x)` in **rad/m** -- angular
            wavenumber, not cycles/m. `(0, 0)` is normal incidence. Build it from
            an angle with `transverse_wavevector_from_angle`.
        amplitude: uniform peak amplitude. Dimensionless and relative; see the
            module docstring on normalization.

    Returns:
        A `ScalarField` of `complex64` on the `n // 2` origin, declaring
        `validity=frozenset()`.

    Raises:
        ContractError: `|k_t| > n k0` (evanescent, not an illumination angle), or
            `|k_t|` past the grid's Nyquist `pi/d` on either axis (an aliased
            tilt reads back as a completely different, entirely plausible angle).
            Also whatever the `ScalarField` contract refuses about the pitch, the
            wavelength and the array.
        ValueError: a non-positive axis length, or a non-finite / non-positive
            amplitude.
    """
    counts = tuple(int(value) for value in shape)
    if len(counts) != 2 or any(count < 1 for count in counts):
        raise ValueError(f"shape must be (ny, nx) with at least one sample per axis, got {shape!r}")

    # The same helpers `ScalarField.__post_init__` applies, called early because
    # the refusals below divide by the pitch and the wavelength. A field built
    # from a bad declaration would be refused either way; doing it here means the
    # message names the declaration rather than the NaNs it produced.
    pitch = tuple(
        require_positive_si(value, name=name)
        for value, name in zip(
            sample_pitch_m, ("sample_pitch_m[dy]", "sample_pitch_m[dx]"), strict=True
        )
    )
    wavelength = require_positive_si(wavelength_m, name="wavelength_m")
    peak = require_positive_si(amplitude, name="amplitude")

    wavevector = tuple(float(value) for value in transverse_wavevector_rad_per_m)
    if len(wavevector) != 2 or not all(math.isfinite(value) for value in wavevector):
        raise ValueError(
            "transverse_wavevector_rad_per_m must be a finite (k_y, k_x) pair in rad/m, got "
            f"{transverse_wavevector_rad_per_m!r}"
        )

    index = reference_surface.medium_index
    medium_wavenumber = 2.0 * math.pi * index / wavelength
    magnitude = math.hypot(*wavevector)
    if magnitude > medium_wavenumber:
        raise ContractError(
            "REPRESENTATION_INCONSISTENT",
            f"|k_t| = {magnitude:.6g} rad/m exceeds n k0 = {medium_wavenumber:.6g} rad/m "
            f"(n = {index}, lambda = {wavelength} m). That is an evanescent wave, not an "
            "illumination angle: the field this would build decays along +z and would be "
            "carried as a propagating one.",
            declaration="transverse_wavevector_rad_per_m",
            remedy="Reduce |k_t|, or state the medium index the angle was measured in.",
        )

    for value, step, axis in zip(wavevector, pitch, ("k_y", "k_x"), strict=True):
        nyquist = math.pi / step
        if abs(value) > nyquist:
            raise ContractError(
                "REPRESENTATION_INCONSISTENT",
                f"|{axis}| = {abs(value):.6g} rad/m is past this grid's Nyquist limit "
                f"pi/d = {nyquist:.6g} rad/m at a pitch of {step} m. The sampled ramp would "
                "alias, and an aliased tilt reads back as a completely different and "
                "entirely plausible angle -- which is the failure this refusal exists for.",
                declaration="transverse_wavevector_rad_per_m",
                remedy="Refine the pitch, or reduce the tilt.",
            )

    frame = Frame()
    ky, kx = wavevector
    dy, dx = pitch
    ny, nx = counts
    # float64 throughout, cast once. `Frame.origin_index` rather than `n // 2`:
    # a half-sample origin shift is a linear phase ramp across the grid, i.e. a
    # tilt, which is exactly the quantity this function exists to state.
    y = (np.arange(ny, dtype=np.float64) - frame.origin_index(ny)) * dy
    x = (np.arange(nx, dtype=np.float64) - frame.origin_index(nx)) * dx
    phase = ky * y[:, None] + kx * x[None, :]
    u = (peak * np.exp(1j * phase)).astype(_SOURCE_DTYPE)

    return ScalarField(
        u=u,
        sample_pitch_m=(dy, dx),
        wavelength_m=wavelength,
        reference_surface=reference_surface,
        frame=frame,
        validity=frozenset(),
    )
