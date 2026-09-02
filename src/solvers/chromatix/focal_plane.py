"""The ideal lens, as the transformation between its two focal planes.

CHE-209 (R06.4). One public function:

```python
solvers.chromatix.focal_plane_transform(field, *, focal_length_m, model) -> ScalarField
```

A `representations.ScalarField` at the front focal plane goes in and a
`ScalarField` at the back focal plane comes out, on a **different, declared**
sample pitch. It lives in the solver package for the reason `solver.py` already
records: backend ownership beats taxonomy, and a forwarding wrapper under
`operators/` would put a second name on one implementation without adding a
boundary. It registers as a `physical_operator` -- the state at the back focal
plane is a different physical state, not a re-description of the front one.

Named for the physics, not for the backend
-------------------------------------------
This is deliberately not "an ideal lens". A lens is an element with a thickness,
a pupil, aberrations and a position; the part with a closed form is the
transformation *between two conjugate focal planes* of an ideal one. Naming the
transformation is what leaves the pupil somewhere else -- `operators/` (R06.6) --
so an aberrated or coded pupil later is a different mask against an unchanged
operator rather than a new argument here.

**`NA=` is not exposed, and that is a physics decision.** The backend's `ff_lens`
takes one, and (read from the pinned build) it applies `circular_pupil` to the
*incoming* field -- the front focal plane -- not to the Fourier plane where a
system stop belongs. Accepting that argument would silently build a different
optical system that still produces a plausible image. So this function calls the
transform alone and the stop is placed, separately and visibly, at the plane it
actually occupies.

Padding is refused rather than chosen
---------------------------------------
The output pitch is `lambda f / (n N dx)`, so `N` -- the number of samples
actually transformed -- is in the denominator. Padding an optical Fourier
transform therefore changes its output sampling, unlike padding an ASM
propagation, which changes only how much wraparound it suffers. A `pad_width`
would be a quiet regrid, so `model=` refuses the key instead of picking a
meaning for it.

The phase this carries, said in the type
------------------------------------------
The textbook front-to-back focal-plane relation is

    U_b(x) = exp(i k n 2f) / (i lambda f / n)
             * integral U_f(x') exp(-i 2 pi x x' n / (lambda f)) dx'

and the backend computes it **without** the leading constant: its normalization
is `-i * dy dx / (lambda f / n)` and there is no `exp` factor anywhere in
`optical_fft`. The returned field's phase is therefore relative to a removed
piston of `carrier_phase_rad(wavelength_m=lambda, distance_m=2f,
refractive_index=n)`, which is exactly what `'carrier_removed_phase'` declares,
so the returned field declares it. `|U|^2` cannot see the difference, which is
why it is in the type rather than in this docstring alone.

Which oracle decides
--------------------
The closed forms in `tests/physics/test_focal_plane_transform.py`: the discrete
sampling relation, a delta transforming to a linear phase ramp of analytic slope,
a tilted plane wave focusing to `f sin(theta)`, a rectangular aperture's `sinc`
nulls and sidelobe ratio, unitarity of the forward/inverse pair, and discrete
Parseval for the power. No repository FFT gates anything.

**The focus lands at `f sin(theta)`, not `f tan(theta)`.** That is a property of
the model, not a defect in it: a single optical Fourier transform maps spatial
frequency to position linearly, and a plane wave at `theta` carries `f_x = n
sin(theta) / lambda`. The two differ at order `theta^3` -- 5.3 output samples at
20 degrees on the test grid -- and the difference is the sine-condition/paraxial
content of the ideal-lens model. It is in the descriptor's `approximation` and in
the test, which asserts the measurement rejects `f tan(theta)`.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from representations import ReferenceSurface, ScalarField, ValidityFlag
from representations.contracts import ContractError
from solvers.chromatix.fields import (
    fourier_plane_pitch_m,
    from_native,
    import_backend,
    to_native,
)

__all__ = ["DIRECTIONS", "focal_plane_transform"]

#: Which way through the lens. Two forward transforms are what make a 4f relay's
#: image inverted; a second leg run as `'inverse'` gives an upright image and a
#: system that is not a 4f relay, so the direction is named rather than inferred.
DIRECTIONS: tuple[str, ...] = ("forward", "inverse")

#: `model=` keys, checked rather than trusted, for the reason `solver.py` gives:
#: a misspelled key that is silently discarded is a different physical run
#: reported as the requested one.
_MODEL_REQUIRED = frozenset({"target_surface"})
_MODEL_OPTIONAL = frozenset({"direction"})


def _require_model(model: Mapping[str, Any]) -> tuple[str, str]:
    """Refuse a `model=` that is misspelled, incomplete or over-specified."""
    if not isinstance(model, Mapping):
        raise TypeError(f"model= must be a mapping, got {type(model).__name__}")
    keys = set(model)
    missing = _MODEL_REQUIRED - keys
    unknown = keys - _MODEL_REQUIRED - _MODEL_OPTIONAL
    if missing or unknown:
        detail = []
        if "pad_width" in unknown:
            detail.append(
                "does not take 'pad_width': the output pitch is lambda f / (n N dx), so "
                "padding changes N and therefore silently regrids the Fourier plane. Pad "
                "the free-space legs, not the transform"
            )
            unknown = unknown - {"pad_width"}
        if unknown:
            detail.append(
                f"does not take {sorted(unknown)} -- an unrecognized key would be "
                "silently discarded, which is a different transform reported as this one"
            )
        if missing:
            detail.append(f"needs {sorted(missing)}")
        raise ValueError("model= " + "; and ".join(detail))

    direction = str(model.get("direction", "forward"))
    if direction not in DIRECTIONS:
        raise ValueError(f"model['direction']={direction!r} is not one of {list(DIRECTIONS)}")

    target_surface = str(model["target_surface"])
    if not target_surface.strip():
        raise ValueError(
            "model['target_surface'] is empty. The plane a field lands on has to be named "
            "for a consumer to check it is the plane it expected -- here, the difference "
            "between a Fourier plane and an image plane."
        )
    return direction, target_surface


def focal_plane_transform(
    field: ScalarField, *, focal_length_m: float, model: Mapping[str, Any]
) -> ScalarField:
    """Transform `field` from one focal plane of an ideal lens to the other.

    The medium is read from `field.reference_surface.medium_index` rather than
    taken as an argument, exactly as `propagate` does: the index of the medium a
    field sits in is already a declaration the representation carries, and a
    second one here could disagree with it.

    Args:
        field: the field at the starting focal plane. Its declared pitch, shape,
            wavelength and surface are the transform's boundary conditions, and
            the output pitch is computed from them before the backend is called.
        focal_length_m: focal length of the ideal lens, in metres, positive. The
            direction through it is `model['direction']`, not the sign of `f`.
        model: `target_surface`, the name of the plane the result is declared on,
            and optional `direction` (`'forward'`, the default, or `'inverse'`).

    Returns:
        A `ScalarField` on the target surface at `z +/- 2f`, sampled at
        `lambda f / (n N dx)` per axis, in the array namespace the input arrived
        in, declaring `'carrier_removed_phase'` -- the `exp(i k n 2f)` piston of
        the textbook relation is not carried.

    Raises:
        ValueError: a `model=` key is missing or unrecognized (`pad_width` names
            its own reason), the direction is unknown, or the focal length is not
            positive and finite. A dtype or device outside the measured capability
            row also raises here, carrying a `code`.
        ContractError: the field is declared `surface_only` or is still padded, or
            the backend returned a pitch that is not the declared one.
        ImportError: chromatix or jax is not installed.
    """
    direction, target_surface = _require_model(model)

    focal_length = float(focal_length_m)
    if not math.isfinite(focal_length) or focal_length <= 0.0:
        raise ValueError(
            f"focal_length_m={focal_length_m!r} must be a positive, finite length in metres. "
            "Which way the field goes through the lens is model['direction'], not the sign "
            "of f -- a negative f here would make the two statements able to disagree."
        )

    if "surface_only" in field.validity:
        raise ContractError(
            "REPRESENTATION_INCONSISTENT",
            "the field declares `surface_only`: it is valid at its own reference surface "
            "and nowhere else, so transforming it to the conjugate focal plane is not a "
            "loss of accuracy but a different physical claim.",
            declaration="validity",
            remedy="Re-derive the field at the plane you want, or drop the flag deliberately.",
        )
    if field.padded:
        raise ContractError(
            "PAD_STATE_UNKNOWN",
            "the field is still padded, and a Fourier transform mixes every sample into "
            "every other one, so the window its producer modelled is not recoverable "
            "afterwards -- and the pad samples are in N, which sets the output pitch.",
            declaration="padded",
            remedy="Crop the field back to its modelled window before transforming it.",
        )

    medium_index = field.reference_surface.medium_index
    expected_pitch_m = fourier_plane_pitch_m(
        field.sample_pitch_m,
        field.shape,
        wavelength_m=field.wavelength_m,
        focal_length_m=focal_length,
        medium_index=medium_index,
    )

    native, requested = to_native(field)
    _, _, cf = import_backend()
    # The backend selects fft vs ifft from the sign of the distance; `ff_lens`
    # with no NA is this call and nothing else, so this is the same physics
    # without the pupil-in-the-wrong-plane argument.
    signed = focal_length if direction == "forward" else -focal_length
    out = cf.optical_fft(native, signed, medium_index)

    validity: frozenset[ValidityFlag] = field.validity | {"carrier_removed_phase"}
    surface = field.reference_surface
    return from_native(
        out,
        source=field,
        requested=requested,
        expected_pitch_m=expected_pitch_m,
        reference_surface=ReferenceSurface(
            name=target_surface,
            # Front focal plane -> lens -> back focal plane is 2f of path. An
            # inverse transform walks the same geometry backwards.
            z_m=surface.z_m
            + (2.0 * focal_length if direction == "forward" else -2.0 * focal_length),
            medium_index=medium_index,
            normal=surface.normal,
        ),
        validity=validity,
        # A transform mixes every sample into every other one: no part of the
        # output is "the padding", so a pad width carried through would describe
        # nothing. Padded inputs are refused above rather than silently flattened.
        pad_width=0,
        padded=False,
    )
