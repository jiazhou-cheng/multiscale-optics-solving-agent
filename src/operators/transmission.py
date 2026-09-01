"""One thin element: `U_out = U_in * A * exp(i phi)`, and the masks that feed it.

CHE-211 (R06.6). Three public functions and one operator:

```python
operators.complex_transmission(field, *, amplitude=1.0, phase_rad=0.0,
                               target_surface=None, allow_gain=False) -> ScalarField
operators.circular_aperture_amplitude(shape, *, sample_pitch_m, radius_m, edge) -> array
operators.numerical_aperture_radius_m(numerical_aperture, *, focal_length_m,
                                      medium_index) -> float
```

Amplitude-only objects (`phase_rad=0`), phase-only objects (`amplitude=1`) and
finite-NA pupils (`amplitude` = an aperture mask) are **arguments**, not three
APIs. `amplitude` and `phase_rad` each accept a scalar or a full array, and that
is the whole of what makes the three cases one implementation.

The reason it is one primitive is architectural. The pressure to add
`phase_mask`, `amplitude_mask`, `pupil` and `grating` as separate public
operations is constant and each one looks cheap; the reference implementation's
recorded failure mode was exactly that family, each member growing its own
parameters, diagnostics and result type. `A exp(i phi)` is the general thin
element and the special cases have nothing of their own to hold.

Kind, and where it lives
------------------------
A `physical_operator`, not a coupler. The representation on both sides is a
`ScalarField` at the same reference surface and nothing is re-described: the
*state* changes, which is `docs/architecture_principles.md` section 2's
distinction and the one the retired `C_FIELD_TO_PSF` got wrong in the other
direction.

It needs **no backend**. An elementwise multiply happens in whatever array
namespace the field already carries, so this module imports NumPy for a mask grid
and nothing else -- a NumPy field and a JAX field both work and both come back in
the namespace they arrived in. `tests/solvers/test_chromatix_boundary.py`'s AST
and `sys.modules` walks already cover `operators/` for that, because they walk
every module under `src/`.

The approximation, stated where a reviewer checks it
-----------------------------------------------------
The element is infinitely thin and acts **at the field's own reference surface**.
`z_m` does not advance, no propagation happens inside it, and there is no
thickness, no multiple scattering, no polarization and no angular dependence of
the transmission. That sentence is in the descriptor's `approximation` field
(`tests/operators/test_transmission.py`), not only here.

That approximation is also what makes R06.8's whole method exact rather than
approximate: `O(x) exp(i k_t x)` has spectrum `O~(k - k_t)` **exactly** for a
thin, angle-independent transmission, so "tilt the illumination" and "shift the
object spectrum" are the same statement. They would not be for a thick or
angle-dependent sample.

The pupil is not a lens argument
---------------------------------
`chromatix.functional.ff_lens(NA=...)` applies `circular_pupil(field, D=2fNA/n)`
to the **incoming** field -- the front focal plane, not the Fourier plane where a
system stop belongs (read from the pinned build, 2026-08-31). Adopting that
shortcut would build a physically different system that still produces a
plausible image, and it would hard-wire "the pupil is a property of the lens" into
the vocabulary at the point where the project needs the opposite. So the operator
takes a complex transmission and a pupil is a **mask builder** that produces one.
Arbitrary pupils, aberrated pupils, Zernike phase, SLM patterns and coded
apertures are then new mask builders against an unchanged operator: no new
operator, no new registration, no new boundary.

`circular_pupil` is worth reading and not calling for a second reason: it is
`l2_sq_norm(field.grid) <= ((central_lambda/lambda) w/2)**2`, a hard-edged mask
with a chromatic radius scaling a monochromatic path does not want.

The mask edge is a declared parameter, never a hidden default
--------------------------------------------------------------
`edge=` on `circular_aperture_amplitude` has **no default**, because both edges
are legitimate physics and which one a result used has to be visible in the call.
The measurement behind that: `tests/physics/test_scalar_wave_propagation.py`'s
ASM round trip lands at 2.75e-7 with the soft `exp(-(r/R)^8)` edge and at 2.2e-2
-- four orders of magnitude worse, against a 1e-5 gate -- with a hard disc, because
a step edge puts real power past the light cone and the evanescent orders decay
on the way out and cannot come back.

`validity` is inherited unchanged, and `surface_only` is not refused
--------------------------------------------------------------------
Multiplying by a mask neither adds nor removes a declared limitation: a
`carrier_removed_phase` field stays carrier-removed, and a
`no_wavefront_curvature_term` field stays that too. `surface_only` is the subtle
one and it is **permitted here**: it means the field is valid at its declared
reference surface and nowhere else, and acting exactly at that surface is the one
operation it allows. `propagate` and `focal_plane_transform` refuse it because
they move the field off that surface; this operator does not move it at all.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from numerics import numpy_dtype
from representations import ContractError, Frame, ReferenceSurface, ScalarField

__all__ = [
    "EDGES",
    "circular_aperture_amplitude",
    "complex_transmission",
    "numerical_aperture_radius_m",
]

#: The edge profiles `circular_aperture_amplitude` builds. Two, because two is
#: what the recorded measurement covers; a Gaussian or a raised-cosine edge is a
#: third mask builder in the ticket that needs one, against this same operator.
#:
#: * `'hard'` -- `A = 1` strictly inside `radius_m`, `0` outside. The analytic
#:   aperture, and the one whose spectrum does not fit on any finite grid.
#: * `'soft_r8'` -- `A = exp(-(r/R)^8)`, which is 0.37 at `r = R` and falls to
#:   1e-3 by `r = 1.3 R`. Bandlimited enough for a propagation round trip; not the
#:   aperture an analytic `sinc` or Airy oracle is written for.
EDGES: tuple[str, ...] = ("hard", "soft_r8")


def _factor(
    value: Any,
    *,
    name: str,
    field: ScalarField,
    complex_ok: bool,
) -> Any:
    """Adopt a scalar-or-full-array factor into the field's namespace, or refuse.

    The shape rule is the point: an array factor must have **exactly** the
    field's shape. NumPy would happily broadcast a `(1, nx)` row or an `(nx,)`
    vector across it, and a mask that broadcasts is the silent failure this
    refusal exists for -- it produces a well-formed field with the aperture
    applied along one axis only.
    """
    xp = field.xp
    array = xp.asarray(value)
    if array.ndim not in (0, 2):
        raise ContractError(
            "SHAPE_MISMATCH",
            f"{name} must be a scalar or a full 2-D (y, x) array, got shape "
            f"{tuple(array.shape)}",
            declaration=name,
        )
    if array.ndim == 2 and tuple(int(n) for n in array.shape) != field.shape:
        raise ContractError(
            "SHAPE_MISMATCH",
            f"{name} has shape {tuple(int(n) for n in array.shape)} against a field of "
            f"{field.shape}. A broadcastable-but-wrong mask is not refused by the "
            "arithmetic: it would apply along one axis and leave a plausible field.",
            declaration=name,
        )
    if not complex_ok and xp.iscomplexobj(array):
        raise ContractError(
            "DTYPE_KIND_MISMATCH",
            f"{name} is complex. `amplitude` is a real, non-negative modulus and the "
            "phase is `phase_rad`; accepting a complex amplitude would let one physical "
            "quantity be specified two ways and disagree.",
            declaration=name,
            remedy="Pass abs() as `amplitude` and angle() as `phase_rad`.",
        )
    if not bool(xp.all(xp.isfinite(array))):
        raise ContractError(
            "NON_FINITE",
            f"{name} is not finite everywhere. A NaN or an inf in a thin element "
            "propagates into every sample of every later plane.",
            declaration=name,
        )
    return array


def complex_transmission(
    field: ScalarField,
    *,
    amplitude: Any = 1.0,
    phase_rad: Any = 0.0,
    target_surface: str | None = None,
    allow_gain: bool = False,
) -> ScalarField:
    """Multiply `field` by `amplitude * exp(1j * phase_rad)` at its own surface.

    The one thin-element operator. `amplitude=1.0` gives a pure phase element,
    `phase_rad=0.0` a pure amplitude element, and an aperture mask in `amplitude`
    a finite-NA stop -- one implementation behind all three.

    Args:
        field: the field at the plane the element occupies.
        amplitude: real, non-negative modulus `A`. A scalar or a full `(ny, nx)`
            array; nothing in between, because a broadcastable shape is a silent
            partial application.
        phase_rad: real phase `phi`, in radians, added under the project's
            `exp(-i omega t)` phasor. A scalar or a full `(ny, nx)` array.
        target_surface: an optional new **name** for the reference surface, for
            when the same plane acquires a role ("pupil", "object"). `z_m`, the
            medium index and the normal are unchanged: an infinitely thin element
            does not move the field.
        allow_gain: permit `A > 1`. Off by default, so gain is a stated claim
            rather than an arithmetic accident.

    Returns:
        A `ScalarField` on the same surface, at the same pitch and pad state, in
        the namespace and dtype the input arrived in, with `validity` inherited
        unchanged.

    Raises:
        ContractError: a factor is the wrong shape, non-finite, or complex
            (`amplitude`); `A` is negative; or `A > 1` without `allow_gain`.
    """
    a = _factor(amplitude, name="amplitude", field=field, complex_ok=False)
    phi = _factor(phase_rad, name="phase_rad", field=field, complex_ok=False)
    xp = field.xp

    if bool(xp.any(a < 0.0)):
        raise ContractError(
            "REPRESENTATION_INCONSISTENT",
            "amplitude has a negative value. `A` is a modulus; a negative entry is a "
            "pi phase shift written in the wrong field, and it would pass every "
            "intensity check downstream.",
            declaration="amplitude",
            remedy="Move the sign into `phase_rad` as a pi offset.",
        )
    if not allow_gain and bool(xp.any(a > 1.0)):
        raise ContractError(
            "REPRESENTATION_INCONSISTENT",
            f"amplitude reaches {float(xp.max(a))!r} > 1. A passive thin element cannot "
            "amplify. Pass allow_gain=True to state the claim deliberately.",
            declaration="amplitude",
            remedy="allow_gain=True, or normalize the mask.",
        )

    # The phasor is built in the field's own dtype, and the product is cast back
    # to it: a float64 `phase_rad` handed to a complex64 field would otherwise
    # promote the whole array to complex128 and silently leave the one dtype the
    # wave backend has.
    target_dtype = numpy_dtype(field.state.dtype)
    u = (field.u * (a * xp.exp(1j * phi))).astype(target_dtype)

    surface = field.reference_surface
    if target_surface is not None:
        if not str(target_surface).strip():
            raise ContractError(
                "MISSING_DECLARATION",
                "target_surface is empty. Pass None to keep the field's own surface name; "
                "an unnamed plane cannot be checked against the one a consumer expected.",
                declaration="target_surface",
            )
        surface = ReferenceSurface(
            name=str(target_surface),
            z_m=surface.z_m,
            medium_index=surface.medium_index,
            normal=surface.normal,
        )

    return ScalarField(
        u=u,
        sample_pitch_m=field.sample_pitch_m,
        wavelength_m=field.wavelength_m,
        reference_surface=surface,
        frame=field.frame,
        # Inherited unchanged, including `surface_only`: see the module docstring.
        validity=field.validity,
        pad_width=field.pad_width,
        padded=field.padded,
    )


def circular_aperture_amplitude(
    shape: tuple[int, int],
    *,
    sample_pitch_m: tuple[float, float],
    radius_m: float,
    edge: str,
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """A circular amplitude mask of radius `radius_m` on the `n // 2` origin.

    A mask is an array plus the grid it was built on, and the grid is the caller's
    -- so this takes a shape and a pitch rather than a field, which is what lets
    the same builder produce a stop in a *Fourier* plane whose pitch is not the
    input plane's.

    Returned in float64 NumPy. `complex_transmission` converts it into the
    field's namespace and dtype, so a JAX field gets a JAX mask without this
    function knowing a backend exists.

    Args:
        shape: `(ny, nx)`.
        sample_pitch_m: `(dy, dx)` in metres.
        radius_m: `R`, in metres.
        edge: `'hard'` or `'soft_r8'`. **Required**: both are legitimate physics
            and which one a recorded result used must be visible in the call. See
            the module docstring for the 2.75e-7 vs 2.2e-2 measurement.

    Raises:
        ValueError: an unknown `edge`, a non-positive radius or pitch, or a shape
            with a non-positive axis.
    """
    if edge not in EDGES:
        raise ValueError(
            f"edge={edge!r} is not one of {list(EDGES)}. There is no default: a hard edge "
            "and a soft edge are different physics and the choice belongs in the call."
        )
    radius = float(radius_m)
    if not (math.isfinite(radius) and radius > 0.0):
        raise ValueError(f"radius_m={radius_m!r} must be a positive, finite length in metres")

    frame = Frame()
    axes = []
    for count, pitch in zip(shape, sample_pitch_m, strict=True):
        count = int(count)
        pitch = float(pitch)
        if count < 1 or not (math.isfinite(pitch) and pitch > 0.0):
            raise ValueError(
                f"an axis needs at least one sample and a positive pitch, got {count!r} "
                f"samples at {pitch!r} m"
            )
        axes.append((np.arange(count, dtype=np.float64) - frame.origin_index(count)) * pitch)

    y, x = axes
    radius_grid = np.hypot(y[:, None], x[None, :])
    if edge == "hard":
        return (radius_grid <= radius).astype(np.float64)
    # exp(-(r/R)^8): 0.37 at r = R, 1e-3 by 1.3 R. Written with the 8th power of
    # the *ratio* rather than of the radius so a large grid does not overflow.
    return np.exp(-((radius_grid / radius) ** 8))


def numerical_aperture_radius_m(
    numerical_aperture: float, *, focal_length_m: float, medium_index: float
) -> float:
    """The stop radius that realizes `NA`, in the **Fourier plane's** coordinates.

    Pure float64 arithmetic, and the derivation is two steps:

        f_c  = NA / lambda            the cutoff spatial frequency of the stop
        x    = lambda f f_x / n       an optical Fourier transform's frequency-to-
                                      position map (see `fourier_plane_pitch_m`)
        =>  R = lambda f (NA/lambda) / n = f NA / n

    The wavelength cancels, which is worth stating: the stop radius that
    implements a given NA is achromatic even though the cutoff frequency it
    imposes is not.

    This exists so that a caller places the stop where it belongs rather than
    trusting a lens argument -- `ff_lens(NA=)` in the pinned build stops the
    *incoming* field, at the front focal plane.

    Raises:
        ValueError: a non-finite or non-positive argument, or `NA > medium_index`.
            `NA = n sin(theta)` is bounded by `n`; a larger value is not a
            demanding aperture, it is an impossible one.
    """
    na = float(numerical_aperture)
    focal_length = float(focal_length_m)
    index = float(medium_index)
    if not (math.isfinite(na) and na > 0.0):
        raise ValueError(f"numerical_aperture={numerical_aperture!r} must be positive and finite")
    if not (math.isfinite(focal_length) and focal_length > 0.0):
        raise ValueError(
            f"focal_length_m={focal_length_m!r} must be a positive, finite length in metres"
        )
    if not (math.isfinite(index) and index > 0.0):
        raise ValueError(f"medium_index={medium_index!r} must be positive and finite")
    if na > index:
        raise ValueError(
            f"numerical_aperture={na!r} exceeds medium_index={index!r}. NA = n sin(theta) "
            "is bounded by the index of the medium; a larger value describes no aperture."
        )
    return focal_length * na / index
