"""Narrow Chromatix operations used only by the L1-WAVE-01 accuracy suite.

Mirrors ``optiland_benchmark_adapter.py``: this module holds the external
solver imports and exposes solver-native arrays, so the benchmark evaluator
cannot accidentally use Chromatix to compute its own expectations. The
analytic oracles live in ``benchmarks/level1/L1-WAVE-01/oracles.py``, which
imports neither Chromatix nor JAX.

Why this exists separately from ``chromatix_adapter.ChromatixAdapter``
----------------------------------------------------------------------
The CHE-14 standalone baseline is scalar-only by design and rejects
``field_kind="vector"`` outright (``CHROMATIX_UNSUPPORTED_FIELD_KIND``),
because no vector path in Chromatix has been validated in this repository.
L1-WAVE-01 Case 3 needs exactly that unvalidated path in order to *test* it.
Putting it here keeps the frozen CHE-14 contract untouched and keeps the
vector surface clearly marked as benchmark-only. Cases 1 and 2 do not use this
module at all -- they run through ``ChromatixAdapter.run_standalone``.

Vector component order (verified against the installed package)
---------------------------------------------------------------
``chromatix.core.field.VectorField.u`` has shape ``(y, x, 3)`` with the last
axis ordered ``(E_z, E_y, E_x)`` -- confirmed by reading
``chromatix.functional.lenses.cartesian_to_spherical``, which writes
``field.u[..., 2]`` as the x component and returns
``concatenate([e_inf_z, e_inf_y, e_inf_x])``. This module converts to and from
the project's ``(E_x, E_y, E_z)`` order exactly once, at this boundary, so no
downstream code has to remember Chromatix's ordering.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from multiscale_optics_agent.core.errors import AdapterDependencyError

CHROMATIX_COMPONENT_ORDER = ("E_z", "E_y", "E_x")
PROJECT_COMPONENT_ORDER = ("E_x", "E_y", "E_z")


@dataclass(frozen=True)
class VectorFocusResult:
    """Solver-native result of one high-NA vectorial focusing call."""

    field_xyz: np.ndarray  # (y, x, 3), ordered (E_x, E_y, E_z)
    reported_pitch_m: float
    output_shape: tuple[int, int]
    pupil_samples: int
    pupil_pitch_m: float
    metadata: dict[str, Any]


def _imports() -> tuple[Any, Any, Any, Any]:
    try:
        import jax

        # Match the CHE-14 baseline's precision contract: complex64, x64 off.
        jax.config.update("jax_enable_x64", False)  # type: ignore[no-untyped-call]
        import chromatix.functional as cf
        import jax.numpy as jnp
        from chromatix.core.field import VectorField
        from chromatix.core.spectrum import MonoSpectrum
    except Exception as exc:
        raise AdapterDependencyError(
            "L1-WAVE-01 Case 3 requires the pinned chromatix 0.6.0 "
            "(git commit d24bdf0022835bb8ce1cdcc6aeafbc7fcb39daee) and jax: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    return jax, jnp, cf, (VectorField, MonoSpectrum)


def high_na_vector_focus(
    *,
    pupil_amplitude_x: np.ndarray,
    pupil_pitch_m: float,
    wavelength_m: float,
    refractive_index: float,
    numerical_aperture: float,
    focal_length_m: float,
    output_shape: tuple[int, int],
    output_pitch_m: float,
) -> VectorFocusResult:
    """Focus an x-polarized pupil field through ``chromatix.functional.high_na_ff_lens``.

    ``pupil_amplitude_x`` is the scalar x-polarized amplitude on the pupil
    plane, ``(y, x)``; the y and z components entering the objective are zero.
    Chromatix applies the aplanatic *polarization rotation* itself (in
    ``cartesian_to_spherical``) but applies no ``sqrt(cos theta)`` apodization,
    so the caller supplies whatever pupil apodization the physics requires.

    Returns the focal field in project ``(E_x, E_y, E_z)`` order together with
    the sampling Chromatix reports. The evaluator treats that reported pitch as
    a *claim to be tested*, not as ground truth.
    """
    jax, jnp, cf, (VectorField, MonoSpectrum) = _imports()

    amplitude = np.asarray(pupil_amplitude_x)
    if amplitude.ndim != 2 or amplitude.shape[0] != amplitude.shape[1]:
        raise ValueError(
            "pupil_amplitude_x must be a square 2-D (y, x) array; got shape "
            f"{amplitude.shape}. chromatix.functional.high_na_ff_lens carries a "
            "'What about non-square cases?' TODO in the pinned source and is not "
            "exercised off-square here."
        )

    pupil = np.zeros((*amplitude.shape, 3), dtype=np.complex64)
    pupil[..., 2] = amplitude.astype(np.complex64)  # index 2 is E_x in chromatix order

    field = VectorField(jnp.asarray(pupil), pupil_pitch_m, 0.0, MonoSpectrum(wavelength_m))
    focused = cf.high_na_ff_lens(
        field,
        f=focal_length_m,
        n=refractive_index,
        NA=numerical_aperture,
        output_shape=tuple(int(v) for v in output_shape),
        output_dx=output_pitch_m,
    )

    raw = np.asarray(jax.device_get(focused.u))
    # (E_z, E_y, E_x) -> (E_x, E_y, E_z)
    field_xyz = np.ascontiguousarray(raw[..., ::-1])
    reported_pitch = float(np.asarray(jax.device_get(focused.dx)).reshape(-1)[0])

    return VectorFocusResult(
        field_xyz=field_xyz,
        reported_pitch_m=reported_pitch,
        output_shape=(int(raw.shape[0]), int(raw.shape[1])),
        pupil_samples=int(amplitude.shape[0]),
        pupil_pitch_m=float(pupil_pitch_m),
        metadata={
            "function": "chromatix.functional.high_na_ff_lens",
            "chromatix_component_order": list(CHROMATIX_COMPONENT_ORDER),
            "returned_component_order": list(PROJECT_COMPONENT_ORDER),
            "wavelength_m": float(wavelength_m),
            "refractive_index": float(refractive_index),
            "numerical_aperture": float(numerical_aperture),
            "focal_length_m": float(focal_length_m),
            "requested_output_pitch_m": float(output_pitch_m),
            "pupil_radius_m": float(focal_length_m * numerical_aperture / refractive_index),
            "dtype": "complex64",
            "device": "cpu",
            "apodization_applied_by_chromatix": (
                "none. cartesian_to_spherical applies the aplanatic polarization "
                "rotation but no sqrt(cos theta) energy-projection factor; the "
                "caller must supply it in pupil_amplitude_x."
            ),
        },
    )
