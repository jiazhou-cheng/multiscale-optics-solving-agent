"""Scalar-wave propagation, and an unambiguous answer to which phase it carries.

CHE-184 (R06.2). One public function:

```python
solvers.chromatix.propagate(field, *, distance_m, model) -> ScalarField
```

A `representations.ScalarField` goes in and a `ScalarField` comes out. This is
semantically physical propagation and it lives in the solver package anyway:
backend ownership beats taxonomy, and a forwarding wrapper under `operators/`
would put a second name on one implementation without adding a boundary. The
operation *registers* as a `physical_operator` -- never as a coupler, which
changes representation while preserving physical state, which this does not.

What is not here
----------------
No `CarrierRemovedPropagation` result object, no standalone baseline route, no
`WaveHandoffError`. The request is the arguments, the result is the return value,
and a boundary that cannot hold is a `ContractError` or a `numerics` refusal --
both of which already exist and already carry a machine-readable code.

The carrier-phase question, answered in the type
------------------------------------------------
Two propagations of the same field over the same distance differ by the constant
`k n z`, and `|U|^2` cannot see it. That is the whole risk of this boundary: a
composition that adds a carrier-removed field to an absolute one, or reports its
phase as optical phase, is wrong by a piston that nothing downstream measures.

So `model['method']` names which one is computed and the returned field **says
which one it carries** in its typed `validity`, not in prose:

* `'asm'` -- the backend's own `asm_propagate`. Absolute phase. `validity` gains
  nothing.
* `'asm_carrier_removed'` -- the same padding, the same frequency grid, the same
  FFT pair and the same evanescent policy, with the constant `exp(i k n z)`
  factored out of the transfer function. `validity` gains
  `'carrier_removed_phase'`.

Why the second method exists at all is a precision fact, not a preference. The
transfer function is `exp(i z k_z)`, whose phase magnitude is `k n z` -- ~5.4e5
rad at 47 mm -- while the part that carries the diffraction, `z (k_z - k)`, is
~2.5e3 rad on the same grid. complex64 rounds whichever number it is handed, so
representing the constant costs about one float32 epsilon per radian of a
quantity that cannot change the physics. Removing it is an **exact algebraic
rewrite**, through the identity `k_z - k = -(k_x^2 + k_y^2) / (k_z + k)` so that
no cancellation of two nearly equal numbers occurs; no term is dropped and it is
not a paraxial approximation.

`carrier_phase_rad` returns the removed constant in float64, and it is
deliberately never folded back into the field: doing that in complex64 would
reintroduce exactly the rounding the method removes. A consumer that needs
absolute optical phase adds it to `angle(u)` at float64 or better.

Which oracle decides
--------------------
The analytic closed forms in `tests/physics/test_scalar_wave_propagation.py` --
a plane wave's `k_z z`, a Gaussian's `w(z)`, a tilted beam's `z tan(theta)`, and
the unitarity of free-space propagation. This repository's own float64 ASM
implementation is **not** ported and gates nothing: letting custom numerical code
certify numerical code is circular validation.

Gradients
---------
None are claimed. `DERIVATIVE` is `forward_only` and there is no argument that
changes it. `jax_enable_x64` is pinned off on every backend import, which is a
determinism decision rather than a differentiability one.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from representations import ReferenceSurface, ScalarField, ValidityFlag
from representations.contracts import ContractError
from solvers.chromatix.fields import (
    CAPABILITIES,
    EDGE_ENERGY_REPORTING_THRESHOLD,
    edge_energy_fraction,
    from_native,
    import_backend,
    padded_field_bytes,
    padded_shape,
    to_native,
)

__all__ = [
    "CAPABILITIES",
    "DERIVATIVE",
    "EDGE_ENERGY_REPORTING_THRESHOLD",
    "MODELS",
    "carrier_phase_rad",
    "edge_energy_fraction",
    "padded_field_bytes",
    "padded_shape",
    "propagate",
]

#: What may be claimed about differentiating through a propagation.
DERIVATIVE = "forward_only"

#: The propagation methods this package exposes. Both are the exact
#: (non-paraxial) angular spectrum; they differ in which phase the returned field
#: carries, and the returned field says which.
MODELS: tuple[str, ...] = ("asm", "asm_carrier_removed")

#: `model=` keys: the required set and the optional set.
#:
#: Checked rather than trusted, and a mapping rather than a `TypedDict`, because
#: a `TypedDict` is a static annotation that disappears at run time and this
#: package adds no class. An unrecognized key is refused for the reason the ray
#: solver refuses one: a misspelling that is silently discarded is a different
#: physical run reported as the requested one.
_MODEL_REQUIRED = frozenset({"method", "pad_width", "target_surface"})
_MODEL_OPTIONAL = frozenset({"crop"})


def carrier_phase_rad(
    *, wavelength_m: float, distance_m: float, refractive_index: float
) -> float:
    """`k n z`, in float64, **unwrapped** -- the constant `'asm_carrier_removed'` drops.

    Unwrapped on purpose. The wrapped value is what a complex64 field could hold;
    the unwrapped one is the physical accumulated phase, and a consumer
    reconstructing absolute phase needs it before any modular reduction.

    Pure arithmetic: it imports no backend and takes no array, so it is also the
    prediction a test can hold the two methods' difference to.
    """
    return 2.0 * math.pi * float(refractive_index) * float(distance_m) / float(wavelength_m)


def _require_model(model: Mapping[str, Any]) -> tuple[str, int, str, bool]:
    """Refuse a `model=` that is misspelled, incomplete or over-specified."""
    if not isinstance(model, Mapping):
        raise TypeError(f"model= must be a mapping, got {type(model).__name__}")
    keys = set(model)
    missing = _MODEL_REQUIRED - keys
    unknown = keys - _MODEL_REQUIRED - _MODEL_OPTIONAL
    if missing or unknown:
        detail = []
        if unknown:
            detail.append(
                f"does not take {sorted(unknown)} -- an unrecognized key would be "
                "silently discarded, which is a different propagation reported as this one"
            )
        if missing:
            detail.append(f"needs {sorted(missing)}")
        raise ValueError("model= " + "; and ".join(detail))

    method = str(model["method"])
    if method not in MODELS:
        raise ValueError(f"model['method']={method!r} is not one of {list(MODELS)}")

    pad_width = int(model["pad_width"])
    if pad_width < 0:
        raise ValueError(
            f"model['pad_width']={pad_width!r} must be a non-negative sample count. It is "
            "required and has no default: padding is what decides both the wraparound "
            "error and the memory cost, and neither is recoverable from the input shape."
        )

    target_surface = str(model["target_surface"])
    if not target_surface.strip():
        raise ValueError(
            "model['target_surface'] is empty. The plane a field lands on has to be named "
            "for a consumer to check it is the plane it expected -- the difference "
            "between a defocus and a whole pupil-to-focus distance."
        )

    crop = bool(model.get("crop", True))
    return method, pad_width, target_surface, crop


def _carrier_removed_propagator(native: Any, *, distance_m: float, refractive_index: float) -> Any:
    """The exact ASM transfer function with the constant `exp(i k n z)` factored out.

    Mirrors the backend's own `compute_asm_propagator` term for term -- the same
    `f_grid`, the same complex `sqrt` so evanescent orders decay rather than being
    zeroed, the same negative-`z` conjugation and the same trailing `ifftshift`
    into FFT-natural order -- and differs only in the phase it exponentiates:

        z (k_z - k) = -2 pi |z| (lambda / n) f^2 / (delay + 1),   delay = k_z / k

    which is the exact identity `k_z - k = -(k_x^2 + k_y^2) / (k_z + k)`, written
    so no cancellation of two nearly equal numbers occurs.
    """
    _, jnp, _ = import_backend()
    from chromatix.utils import l2_sq_norm

    wavelength = native.broadcasted_wavelength
    frequency_squared = l2_sq_norm(native.f_grid)
    axial_ratio = 1.0 - (wavelength / refractive_index) ** 2 * frequency_squared
    delay = jnp.sqrt(jnp.complex64(axial_ratio))
    relative_phase = (
        -2.0
        * jnp.pi
        * jnp.abs(distance_m)
        * (wavelength / refractive_index)
        * frequency_squared
        / (delay + 1.0)
    )
    kernel = jnp.exp(1j * relative_phase)
    kernel = jnp.where(distance_m >= 0, kernel, jnp.conj(kernel))
    return jnp.fft.ifftshift(kernel, axes=native.spatial_dims)


def propagate(
    field: ScalarField, *, distance_m: float, model: Mapping[str, Any]
) -> ScalarField:
    """Propagate `field` by `distance_m` along `+z` and return a neutral `ScalarField`.

    The medium is read from `field.reference_surface.medium_index` rather than
    taken as an argument: the index of the medium a field sits in is already a
    declaration the representation carries, and accepting a second one here would
    let the two disagree.

    Args:
        field: the incoming field. Its declared pitch, wavelength, frame and
            surface are the propagation's boundary conditions, and none is
            inferred from the array.
        distance_m: signed distance along `+z`, in metres. Negative propagates
            backward, which conjugates the transfer function.
        model: `method` (`'asm'` or `'asm_carrier_removed'`), `pad_width` in
            samples per side, and `target_surface`, the name of the plane the
            result is declared on. Optional `crop` (default `True`) returns the
            input window; `crop=False` returns the padded array with `padded=True`
            and the pad width recorded, so the modelled window is recoverable.

    Returns:
        A `ScalarField` on the target surface, in the array namespace the input
        arrived in, whose `validity` gains `'carrier_removed_phase'` exactly when
        the carrier-removed method was used.

    Raises:
        ValueError: a `model=` key is missing or unrecognized, the distance is not
            finite, or the field's dtype or device is outside the measured
            capability table (those carry a `code`; `complex128` is
            `LOSSY_DOWNCAST_REQUIRED` and is refused before jax is imported).
        ContractError: the field is declared `surface_only`, or the backend
            returned a pitch that is not the declared one.
        ImportError: chromatix or jax is not installed.
    """
    method, pad_width, target_surface, crop = _require_model(model)

    distance = float(distance_m)
    if not math.isfinite(distance):
        raise ValueError(f"distance_m={distance_m!r} is not a finite distance in metres")

    if "surface_only" in field.validity:
        raise ContractError(
            "REPRESENTATION_INCONSISTENT",
            "the field declares `surface_only`: it is valid at its own reference surface "
            "and nowhere else, so propagating it is not a loss of accuracy but a different "
            "physical claim. Nothing in the result would record that.",
            declaration="validity",
            remedy="Re-derive the field at the surface you want, or drop the flag deliberately.",
        )

    refractive_index = field.reference_surface.medium_index

    native, requested = to_native(field)
    _, _, cf = import_backend()

    if method == "asm":
        # The backend's own entry point. With `output_dx` and `output_shape`
        # unset it is exactly pad -> compute_asm_propagator -> kernel_propagate
        # -> crop, which is the same machinery the branch below drives, so any
        # difference between the two methods is attributable to the kernel.
        out = cf.asm_propagate(
            native,
            z=distance,
            n=refractive_index,
            pad_width=pad_width,
            mode="same" if crop else "full",
        )
    else:
        from chromatix.functional.propagation import crop as native_crop
        from chromatix.functional.propagation import kernel_propagate
        from chromatix.functional.propagation import pad as native_pad

        padded_in = native_pad(native, pad_width)
        propagator = _carrier_removed_propagator(
            padded_in, distance_m=distance, refractive_index=refractive_index
        )
        out = kernel_propagate(padded_in, propagator)
        if crop:
            out = native_crop(out, pad_width)

    validity: frozenset[ValidityFlag] = field.validity
    if method == "asm_carrier_removed":
        validity = validity | {"carrier_removed_phase"}

    surface = field.reference_surface
    return from_native(
        out,
        source=field,
        requested=requested,
        reference_surface=ReferenceSurface(
            name=target_surface,
            z_m=surface.z_m + distance,
            medium_index=refractive_index,
            normal=surface.normal,
        ),
        validity=validity,
        pad_width=pad_width,
        padded=bool(not crop and pad_width > 0),
    )
