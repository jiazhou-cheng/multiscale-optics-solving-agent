"""Carrier-removed exact ASM over Chromatix's own propagation machinery (CHE-40).

M3.2 measured Chromatix's ``complex64`` angular spectrum against a float64
reference and found a relative field error growing as ``eps32 * 2*pi*z/lambda``:
2.5e-5 at 40 um, 6.3e-2 at 47 mm. That rejected a 48 mm-focal-length reference
singlet and made the optical system's absolute *size* a protocol decision.

M3.2A's question is whether that error is a property of the wave engine or a
property of the *number being represented*. Chromatix evaluates

    H(f) = exp(i z k_z),    k_z = sqrt(k^2 - k_x^2 - k_y^2)

whose phase magnitude is ``k z``, ~5.4e5 rad at 47 mm. But ``H`` factors exactly:

    H = exp(i k z) * exp(i z (k_z - k))

and the first factor is constant over the whole spectrum. It is a global piston:
it cannot change intensity, and along a single propagation path it cannot change
relative phase either. Only the second factor carries the diffraction. Its phase
magnitude is ``max |z (k_z - k)|``, which on the M3 grids is ~2.5e3 rad at the
same 47 mm -- 200x smaller, and it is the number float32 then has to round.

This module evaluates the second factor through the exact identity

    k_z - k = -(k_x^2 + k_y^2) / (k_z + k)

rather than by subtracting two nearly equal numbers. **This is an algebraic
rewrite of exact ASM, not a paraxial approximation** -- no term is dropped, and
:func:`tests.test_carrier_removed_asm` pins the equality in float64.

Everything else is Chromatix's. The kernel is handed to
``chromatix.functional.propagation.kernel_propagate`` and the padding to
``pad``/``crop``, so the FFT convention, the ``ifftshift`` of the kernel, the
spatial-frequency grid, the normalization, and the propagating/evanescent policy
(``sqrt`` of a complex argument, so evanescent orders decay rather than being
zeroed) are the same objects the baseline path uses. Only the transfer function
differs, which is the point: any measured difference is attributable.

## Global-phase policy (M3.2A AC6)

The removed ``exp(i k z)`` is **retained as metadata** on
:class:`CarrierRemovedPropagation.removed_carrier_phase_rad`, computed in
float64, and is **deliberately not reapplied to the field**. Reapplying it in
``complex64`` would reintroduce precisely the rounding this module exists to
avoid, so a "convenience" reconstruction would silently undo the fix.

Consequences a consumer must respect:

- The returned field's **relative** phase is physical. Its **absolute** phase is
  offset by ``-k z`` and is not usable as an optical phase.
- Intensity, PSF, MTF, and any single-path interference within one field are
  unaffected: a global piston cancels.
- Anything that interferes this field against a *separately* propagated field,
  or that reports absolute optical phase, must add
  ``removed_carrier_phase_rad`` back in float64 first.
- :func:`reconstruct_absolute_phase` does that, in float64, and is the only
  supported route. It is not called automatically.
- One trap the implementation surfaces rather than hides: ``Field.spectrum`` is
  itself ``float32``, so a carrier phase derived from the field's own wavelength
  is only good to ~3e-8 relative -- ~0.02 rad at 47 mm. Pass ``wavelength_m``
  explicitly when the reconstructed absolute phase has to mean anything, and read
  ``wavelength_source`` to see which value was used.

No gradient is claimed through this path. It is forward-only, like every other
JAX path in this repository (AGENTS.md).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "CARRIER_REMOVED_ASM_ID",
    "GLOBAL_PHASE_POLICY",
    "CarrierRemovedPropagation",
    "carrier_removed_asm_propagate",
    "carrier_removed_asm_propagator",
    "pin_wave_engine_precision",
    "reconstruct_absolute_phase",
]

CARRIER_REMOVED_ASM_ID = "ASM-CARRIER-REMOVED-V1"

GLOBAL_PHASE_POLICY = "retained_as_metadata_not_reapplied"
"""The removed carrier is recorded, never silently restored. See the module docstring."""


def pin_wave_engine_precision() -> None:
    """Force ``jax_enable_x64`` off, the way ``chromatix_adapter`` already does.

    ``jax_enable_x64`` is process-global mutable state, and ``sax`` turns it *on*
    as an import side effect that Python will not re-trigger. Under x64 the FFTs
    behind ``kernel_propagate`` promote to ``complex128`` and this module's error
    figures silently improve by orders of magnitude -- which would make every
    number in ``outputs/M3/carrier-phase/`` depend on whether some unrelated
    adapter had been imported first.

    Exported rather than private because a caller that builds the ``Field``
    itself must pin the flag *before* doing so: this function cannot retroactively
    downcast a field that was already constructed under x64.
    """
    import jax

    jax.config.update("jax_enable_x64", False)


def _chromatix() -> tuple[Any, Any, Any, Any]:
    """Import Chromatix's own propagation pieces, so only the kernel differs."""
    import jax.numpy as jnp
    from chromatix.functional.propagation import crop, kernel_propagate, pad
    from chromatix.utils import l2_sq_norm

    pin_wave_engine_precision()
    return jnp, kernel_propagate, (pad, crop), l2_sq_norm


@dataclass(frozen=True, slots=True)
class CarrierRemovedPropagation:
    """A propagated field plus the global phase that was taken out of it.

    ``field`` is a ``chromatix.Field``; it is typed loosely here so that
    importing this module does not require Chromatix at module scope.
    """

    field: Any
    removed_carrier_phase_rad: float
    z_m: float
    refractive_index: float
    wavelength_m: float
    wavelength_source: str
    global_phase_policy: str = GLOBAL_PHASE_POLICY
    implementation_id: str = CARRIER_REMOVED_ASM_ID

    @property
    def absolute_phase_is_physical(self) -> bool:
        """False by construction -- the field carries relative phase only."""
        return False


def carrier_removed_asm_propagator(field: Any, z_m: float, refractive_index: float = 1.0) -> Any:
    """The carrier-removed exact ASM kernel for ``field``, ready for ``kernel_propagate``.

    Mirrors ``chromatix.functional.propagation.compute_asm_propagator`` term for
    term -- same ``f_grid``, same ``sqrt`` of a complex argument so evanescent
    orders decay identically, same negative-``z`` conjugation, same trailing
    ``ifftshift`` into FFT-natural order -- and differs only in the phase it
    exponentiates.
    """
    jnp, _, _, l2_sq_norm = _chromatix()

    wavelength = field.broadcasted_wavelength
    frequency_squared = l2_sq_norm(field.f_grid)
    axial_ratio = 1.0 - (wavelength / refractive_index) ** 2 * frequency_squared
    # Complex sqrt, not a clipped real one: this is Chromatix's default
    # `remove_evanescent=False` policy, under which evanescent orders decay
    # instead of being discarded. Preserved rather than improved.
    delay = jnp.sqrt(jnp.complex64(axial_ratio))

    # z*(k_z - k) = -2*pi*z*lambda*f^2 / (n*(delay+1)), the exact identity.
    # Nothing here is ever as large as k*z, which is the whole point.
    relative_phase = (
        -2.0
        * jnp.pi
        * jnp.abs(z_m)
        * (wavelength / refractive_index)
        * frequency_squared
        / (delay + 1.0)
    )
    kernel = jnp.exp(1j * relative_phase)
    kernel = jnp.where(z_m >= 0, kernel, jnp.conj(kernel))
    return jnp.fft.ifftshift(kernel, axes=field.spatial_dims)


def carrier_removed_asm_propagate(
    field: Any,
    *,
    z_m: float,
    refractive_index: float = 1.0,
    pad_width: int = 0,
    cval: float = 0.0,
    mode: str = "full",
    wavelength_m: float | None = None,
) -> CarrierRemovedPropagation:
    """Propagate ``field`` by ``z_m`` with the common carrier phase removed.

    Signature and defaults follow ``chromatix.functional.asm_propagate`` for the
    arguments M3 uses, so a caller can swap one for the other without changing
    padding or output shape. Options the M3 slice does not use (``kykx``,
    ``bandlimit``, ``shift_yx``, ``output_dx``, absorbing boundaries) are
    deliberately absent rather than silently ignored: none of them is exercised
    or tested here, and accepting an argument that is not honoured is how a
    convention drifts.

    Args:
        wavelength_m: the wavelength the *recorded carrier phase* is computed
            from. It does not enter the propagation, which uses the field's own
            spectrum exactly as Chromatix does. It exists because
            ``Field.spectrum`` is stored in **float32**: reading the wavelength
            back off the field caps ``removed_carrier_phase_rad`` at ~3e-8
            relative, which is ~0.02 rad of absolute phase at 47 mm -- larger
            than everything else this module removes. A caller that will
            reconstruct absolute phase should pass its float64 wavelength here.
            ``wavelength_source`` on the result records which route was taken, so
            the limitation is visible rather than inferred.

    Returns:
        A :class:`CarrierRemovedPropagation`. Read
        ``removed_carrier_phase_rad`` before treating the field's phase as
        absolute; see the module docstring.
    """
    _, kernel_propagate, (pad, crop), _ = _chromatix()

    if mode not in ("full", "same"):
        raise ValueError(f"mode must be 'full' or 'same'; got {mode!r}")

    padded = pad(field, pad_width, cval=cval)
    propagator = carrier_removed_asm_propagator(padded, z_m, refractive_index)
    propagated = kernel_propagate(padded, propagator)
    if mode == "same":
        propagated = crop(propagated, pad_width)

    if wavelength_m is None:
        carrier_wavelength_m = float(np.asarray(field.broadcasted_wavelength).ravel()[0])
        wavelength_source = "chromatix Field.spectrum (float32, ~3e-8 relative)"
    else:
        carrier_wavelength_m = float(wavelength_m)
        wavelength_source = "caller (float64)"

    return CarrierRemovedPropagation(
        field=propagated,
        removed_carrier_phase_rad=_carrier_phase_rad(
            wavelength_m=carrier_wavelength_m, z_m=z_m, refractive_index=refractive_index
        ),
        z_m=float(z_m),
        refractive_index=float(refractive_index),
        wavelength_m=carrier_wavelength_m,
        wavelength_source=wavelength_source,
    )


def _carrier_phase_rad(*, wavelength_m: float, z_m: float, refractive_index: float) -> float:
    """``k z``, in float64, unwrapped.

    Unwrapped on purpose. The wrapped value is what a complex64 field could
    represent; the unwrapped value is the physical accumulated phase, and a
    consumer reconstructing absolute phase needs it before any modular
    reduction, in float64.
    """
    phase = 2.0 * np.pi * refractive_index * abs(z_m) / wavelength_m
    return float(phase if z_m >= 0.0 else -phase)


def reconstruct_absolute_phase(result: CarrierRemovedPropagation) -> np.ndarray:
    """The absolute optical phase of a carrier-removed field, in float64.

    Returns the *unwrapped* absolute phase ``arg(E) + k z``, as a NumPy float64
    array. It is not returned as a complex field, and the carrier is not folded
    back into ``result.field``, because doing that in ``complex64`` would
    reintroduce the rounding this module removes. A consumer that genuinely
    needs an absolute-phase complex field must build it at float64 or better.
    """
    relative_phase = np.angle(np.asarray(result.field.u, dtype=np.complex128))
    return relative_phase + result.removed_carrier_phase_rad
