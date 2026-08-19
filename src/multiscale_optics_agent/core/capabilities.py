"""The one authoritative device/dtype capability declaration per component (CHE-61).

Every claim below was executed against the pinned installs in the
``agent_solver`` / ``agent_solver_gpu`` images, not read off a framework's
documentation. Where a package's API *exposes* a dtype it cannot actually
compute in, the table records what it computes in.

This module is the source of truth. ``registry/models.yaml`` and
``registry/couplers.yaml`` are downstream reflections of it, updated only after
the executable tests pass -- never the other way round -- and
``tests/test_precision_contract.py`` fails if the two disagree.

Measured facts behind the entries
---------------------------------
Optiland 0.6.0 (``tmp_probes/pb4b_probe.py``, GPU image):
    ``set_precision`` is literally ``Literal['float32','float64']`` and raises
    ``ValueError("Precision must be 'float32' or 'float64'.")`` for anything
    else -- so there is **no float16 path** and this project will not invent
    one. ``set_device`` raises ``BackendCapabilityError`` on the numpy backend,
    so CUDA is reachable only through the torch backend. With
    ``set_backend('torch'); set_device('cuda')``, ``be.array([...])`` returns a
    ``Tensor`` on ``cuda:0`` in the selected precision (both float32 and
    float64 confirmed).

Chromatix 0.6.0 @ d24bdf0 (``tmp_probes/pb4b_probe2.py``, GPU image):
    ``ScalarField.__init__`` does ``jnp.asarray(u, dtype=jnp.complex64)``
    unconditionally. Handing ``Field.build`` a ``complex128`` array *with*
    ``jax_enable_x64=True`` still yields ``field.u.dtype == complex64``. There
    is therefore **no complex128 path at any device**, and the project does not
    claim one. ``asm_propagate`` on the GPU image runs on ``cuda:0`` and
    returns ``complex64`` on ``cuda:0``.

Couplers:
    Pure array math with no package dependency, so their capability is set by
    what the shared implementation is written against: the NumPy/JAX compute
    namespaces of ``core.arrays``. Both real and complex data cross them, and
    the phase accumulation ``k * OPL`` sets a float32 floor on the compute
    dtype regardless of what arrives (``minimum_compute_precision``).
"""

from __future__ import annotations

from typing import Any

from multiscale_optics_agent.core.precision import (
    ArrayNamespace,
    CapabilityError,
    ComponentCapabilities,
    DeviceKind,
    DType,
    Precision,
)

__all__ = [
    "CHROMATIX_CAPABILITIES",
    "COMPONENT_CAPABILITIES",
    "C_RAY_TO_WAVE_CAPABILITIES",
    "C_WAVE_TO_RAY_CAPABILITIES",
    "OPTILAND_CAPABILITIES",
    "capabilities_for",
    "capability_matrix",
]


_REAL_DTYPES = frozenset({DType.FLOAT32, DType.FLOAT64})
_FIELD_DTYPES = frozenset({DType.FLOAT32, DType.FLOAT64, DType.COMPLEX64, DType.COMPLEX128})


OPTILAND_CAPABILITIES = ComponentCapabilities(
    component="M_RAY_OPTILAND",
    devices=frozenset({DeviceKind.CPU, DeviceKind.CUDA}),
    precisions=frozenset({Precision.FP32, Precision.FP64}),
    accepted_input_dtypes=_REAL_DTYPES,
    native_compute_dtypes=_REAL_DTYPES,
    output_dtypes=_REAL_DTYPES,
    namespaces=frozenset({ArrayNamespace.NUMPY, ArrayNamespace.TORCH}),
    # set_device raises BackendCapabilityError on the numpy backend, so CUDA is
    # a torch-backend-only capability. Declaring it here rather than as an
    # `if backend == ...` branch is what keeps the two from drifting.
    device_namespaces={
        DeviceKind.CPU: frozenset({ArrayNamespace.NUMPY, ArrayNamespace.TORCH}),
        DeviceKind.CUDA: frozenset({ArrayNamespace.TORCH}),
    },
    minimum_compute_precision=Precision.FP32,
    evidence=(
        "optiland 0.6.0 set_precision is Literal['float32','float64'] and "
        "set_device raises BackendCapabilityError on the numpy backend "
        "(tmp_probes/pb4b_probe.py, agent_solver_gpu, RTX A6000, "
        "torch 2.13.0+cu126)"
    ),
    notes=(
        "float16 is refused, not promoted: geometry, OPL and direction cosines "
        "all accumulate, and Optiland has no float16 mode to execute even if "
        "the project wanted one."
    ),
)


CHROMATIX_CAPABILITIES = ComponentCapabilities(
    component="M_WAVE_CHROMATIX",
    devices=frozenset({DeviceKind.CPU, DeviceKind.CUDA}),
    # FP32 only. Not a policy choice -- there is no complex128 storage in the
    # package, so an FP64 request has nothing to execute.
    precisions=frozenset({Precision.FP32}),
    accepted_input_dtypes=frozenset({DType.COMPLEX64}),
    native_compute_dtypes=frozenset({DType.COMPLEX64}),
    output_dtypes=frozenset({DType.COMPLEX64}),
    namespaces=frozenset({ArrayNamespace.JAX}),
    device_namespaces={
        DeviceKind.CPU: frozenset({ArrayNamespace.JAX}),
        DeviceKind.CUDA: frozenset({ArrayNamespace.JAX}),
    },
    # complex128 is physically ingestible and silently truncated by Chromatix
    # itself. Keeping it out of accepted_input_dtypes is what makes the bridge
    # refuse it under SAFE and record it as lossy under ALLOW_DOWNCAST, instead
    # of letting the loss happen inside ScalarField where nothing measures it.
    lossy_input_dtypes=frozenset({DType.COMPLEX128}),
    minimum_compute_precision=Precision.FP32,
    evidence=(
        "chromatix 0.6.0 @ d24bdf0 ScalarField.__init__ is "
        "`jnp.asarray(u, dtype=jnp.complex64)`; Field.build(complex128 array) "
        "returns complex64 even under jax_enable_x64=True, and asm_propagate "
        "returns complex64 on cuda:0 (tmp_probes/pb4b_probe2.py, "
        "agent_solver_gpu, jax 0.6.2 backend gpu)"
    ),
    notes=(
        "Requested device must never be reported as actual: PB4a measured "
        "klujax (a SAX dependency) pinning jax_platform_name='cpu' at import "
        "time, which produces a successful complex64 run on the host while the "
        "caller asked for CUDA."
    ),
)


C_RAY_TO_WAVE_CAPABILITIES = ComponentCapabilities(
    component="C_RAY_TO_WAVE",
    devices=frozenset({DeviceKind.CPU, DeviceKind.CUDA}),
    precisions=frozenset({Precision.FP32, Precision.FP64}),
    accepted_input_dtypes=_FIELD_DTYPES,
    native_compute_dtypes=_FIELD_DTYPES,
    output_dtypes=frozenset({DType.COMPLEX64, DType.COMPLEX128}),
    namespaces=frozenset({ArrayNamespace.NUMPY, ArrayNamespace.JAX}),
    device_namespaces={
        DeviceKind.CPU: frozenset({ArrayNamespace.NUMPY, ArrayNamespace.JAX}),
        # NumPy cannot hold device memory, so the CUDA path is JAX-only. Torch
        # ray output is bridged in (DLPack, staying on the device).
        DeviceKind.CUDA: frozenset({ArrayNamespace.JAX}),
    },
    minimum_compute_precision=Precision.FP32,
    evidence=(
        "one shared xp-parameterized implementation in couplers/ray_to_wave.py; "
        "dtype/device preservation and GPU residency asserted in "
        "tests/test_precision_execution_matrix.py"
    ),
    notes=(
        "float16 is not accepted: it is promoted to float32 by the bridge under "
        "SAFE, which is a promotion and is reported as one. The coupler never "
        "computes phase in float16."
    ),
)


C_WAVE_TO_RAY_CAPABILITIES = ComponentCapabilities(
    component="C_WAVE_TO_RAY",
    devices=frozenset({DeviceKind.CPU, DeviceKind.CUDA}),
    precisions=frozenset({Precision.FP32, Precision.FP64}),
    accepted_input_dtypes=frozenset({DType.COMPLEX64, DType.COMPLEX128}),
    native_compute_dtypes=_FIELD_DTYPES,
    output_dtypes=_FIELD_DTYPES,
    namespaces=frozenset({ArrayNamespace.NUMPY, ArrayNamespace.JAX}),
    device_namespaces={
        DeviceKind.CPU: frozenset({ArrayNamespace.NUMPY, ArrayNamespace.JAX}),
        DeviceKind.CUDA: frozenset({ArrayNamespace.JAX}),
    },
    minimum_compute_precision=Precision.FP32,
    evidence=(
        "one shared xp-parameterized implementation in couplers/wave_to_ray.py; "
        "the FFT, the evanescent cut and the 1/p weighting all execute in the "
        "input namespace (tests/test_precision_execution_matrix.py)"
    ),
    notes=(
        "Input is a complex field by definition; the emitted ray bundle carries "
        "real geometry at the matching real precision and a complex amplitude."
    ),
)


COMPONENT_CAPABILITIES: dict[str, ComponentCapabilities] = {
    capability.component: capability
    for capability in (
        OPTILAND_CAPABILITIES,
        CHROMATIX_CAPABILITIES,
        C_RAY_TO_WAVE_CAPABILITIES,
        C_WAVE_TO_RAY_CAPABILITIES,
    )
}


def capabilities_for(component: str) -> ComponentCapabilities:
    """Look up a component's declaration, or fail naming what exists."""
    try:
        return COMPONENT_CAPABILITIES[component]
    except KeyError as exc:
        raise CapabilityError(
            code="UNKNOWN_COMPONENT",
            component=component,
            message="no executable capability declaration exists for this component.",
            requested=component,
            supported=COMPONENT_CAPABILITIES,
            remedy=(
                "Add one to core/capabilities.py with the probe evidence behind "
                "it. A component with no declaration has no validated device or "
                "dtype support, and this project will not guess one."
            ),
        ) from exc


def capability_matrix() -> list[dict[str, Any]]:
    """The executable form of the PB4b section-15 truth table.

    Generated from the declarations rather than written alongside them, so a
    documented claim cannot outlive the capability it describes.
    """
    return [
        COMPONENT_CAPABILITIES[name].capability_row()
        for name in (
            "M_RAY_OPTILAND",
            "C_RAY_TO_WAVE",
            "M_WAVE_CHROMATIX",
            "C_WAVE_TO_RAY",
        )
    ]
