"""The one authoritative device/dtype capability declaration per component (CHE-61).

Every claim below was executed against the pinned installs in the
``agent_solver`` / ``agent_solver_gpu`` images, not read off a framework's
documentation. Where a package's API *exposes* a dtype it cannot actually
compute in, the table records what it computes in.

This module is the source of truth. ``registry/models.yaml`` and
``registry/couplers.yaml`` are downstream reflections of it, updated only after
the executable tests pass -- never the other way round -- and
``tests/test_registry_matches_capabilities.py`` fails if the two disagree.

Measured facts behind the entries
---------------------------------
Optiland 0.6.0 (``benchmarks/probes/precision/optiland_capability.py``, GPU):
    ``set_precision`` is literally ``Literal['float32','float64']`` and raises
    ``ValueError("Precision must be 'float32' or 'float64'.")`` for anything
    else -- so there is **no float16 path** and this project will not invent
    one. ``set_device`` raises ``BackendCapabilityError`` on the numpy backend,
    so CUDA is reachable only through the torch backend. With
    ``set_backend('torch'); set_device('cuda')``, ``be.array([...])`` returns a
    ``Tensor`` on ``cuda:0`` in the selected precision (both float32 and
    float64 confirmed).

Chromatix 0.6.0 @ d24bdf0
(``benchmarks/probes/precision/chromatix_capability.py``, GPU):
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

from core.precision import (
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
    "C_PATCH_WFT_CAPABILITIES",
    "C_PLANAR_DOE_STEP_CAPABILITIES",
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
        "(benchmarks/probes/precision/optiland_capability.py, agent_solver_gpu, "
        "RTX A6000, torch 2.13.0+cu126)"
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
        "returns complex64 on cuda:0 "
        "(benchmarks/probes/precision/chromatix_capability.py, agent_solver_gpu, "
        "jax 0.6.2 backend gpu)"
    ),
    notes=(
        "Requested device must never be reported as actual: a process-global JAX "
        "platform pin produces a successful complex64 run on the host while the "
        "caller asked for CUDA, with no error raised. PB4a measured this via a "
        "third-party import-time pin (removed in CHE-72); the rule is not "
        "specific to that cause."
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


C_PLANAR_DOE_STEP_CAPABILITIES = ComponentCapabilities(
    component="C_PLANAR_DOE_STEP",
    devices=frozenset({DeviceKind.CPU, DeviceKind.CUDA}),
    precisions=frozenset({Precision.FP32, Precision.FP64}),
    # A ray bundle in, a ray bundle out -- but the step passes through a complex
    # field in the middle, so it accepts what either side of that can carry.
    accepted_input_dtypes=_FIELD_DTYPES,
    native_compute_dtypes=_FIELD_DTYPES,
    output_dtypes=_FIELD_DTYPES,
    namespaces=frozenset({ArrayNamespace.NUMPY, ArrayNamespace.JAX}),
    device_namespaces={
        DeviceKind.CPU: frozenset({ArrayNamespace.NUMPY, ArrayNamespace.JAX}),
        DeviceKind.CUDA: frozenset({ArrayNamespace.JAX}),
    },
    minimum_compute_precision=Precision.FP32,
    evidence=(
        "composed from C_RAY_TO_WAVE and C_WAVE_TO_RAY, which is why its "
        "capability is theirs intersected rather than an independent claim: the "
        "step accumulates through couplers/ray_to_wave.py and resamples through "
        "couplers/wave_to_ray.py, so any device or dtype either of them refuses "
        "is refused here too. The exactness gate -- full enumeration reproducing "
        "the transmitted field to dtype round-off -- is asserted in "
        "tests/test_coupler_round_trip.py"
    ),
    notes=(
        "The intermediate DOE multiply happens in the accumulated field's own "
        "dtype: this step introduces no cast of its own, which is what makes "
        "the composition's precision the min of its two halves rather than "
        "something new."
    ),
)


C_PATCH_WFT_CAPABILITIES = ComponentCapabilities(
    component="C_PATCH_WFT",
    devices=frozenset({DeviceKind.CPU}),
    precisions=frozenset({Precision.FP64}),
    # Accepts all four: the paper's own phase masks are float32, and refusing
    # them would refuse the data this operator exists to reproduce. The upcast
    # to complex128 is lossless, and declared rather than silent.
    accepted_input_dtypes=_FIELD_DTYPES,
    # The patch transform is complex128 throughout and says so. Not caution: the
    # exactness relation it exists to satisfy is measured at 1.4e-12 relative
    # field error, which is below float32 epsilon, so a float32 path could not
    # be gated by the one test that makes this operator trustworthy. Emitting a
    # float32 spectrum would be a different operator with a different claim.
    native_compute_dtypes=frozenset({DType.COMPLEX128}),
    output_dtypes=frozenset({DType.FLOAT64, DType.COMPLEX128}),
    namespaces=frozenset({ArrayNamespace.NUMPY}),
    device_namespaces={DeviceKind.CPU: frozenset({ArrayNamespace.NUMPY})},
    minimum_compute_precision=Precision.FP64,
    evidence=(
        "couplers/patch.py builds every patch spectrum with numpy's float64 "
        "FFT; the full-aperture enumeration reproduces the independent float64 "
        "discrete ASM at 1.4e-12 relative field error and complete enumeration "
        "is unbiased at 5.9e-15 (tests/test_patch_wft.py). No CUDA or JAX path "
        "has been executed, so none is declared -- see the note."
    ),
    notes=(
        "CPU-only is a statement about THIS step, not about the route. The "
        "declaration is unchanged and still correct -- no CUDA or JAX path has "
        "executed, so none is declared -- but the COST ARGUMENT that used to "
        "sit here has been overtaken twice and is corrected rather than left. "
        "It read: the expensive half of a patch run is the O(rays x pixels) "
        "reconstruction, and the patch transform 'is not where the time goes'. "
        "CHE-101 made the reconstruction 9.6x faster (7% of demo3), and CHE-119 "
        "measured the transform and its draw at 42% -- so for a while this note "
        "said the opposite of the truth. "
        "CHE-119's response was NOT a device port: the per-patch transform and "
        "draw are independent per patch and numpy releases the GIL in both, so "
        "eight host threads took the stage 2.5x faster with every emitted ray "
        "bitwise unchanged. That is why this declaration still does not widen. "
        "A CUDA or float32 emitter would be a different operator with a "
        "different claim -- the exactness relation is measured at 1.4e-12, below "
        "float32 epsilon -- and would need its own entry, its own evidence and "
        "its own gate."
    ),
)


COMPONENT_CAPABILITIES: dict[str, ComponentCapabilities] = {
    capability.component: capability
    for capability in (
        OPTILAND_CAPABILITIES,
        CHROMATIX_CAPABILITIES,
        C_RAY_TO_WAVE_CAPABILITIES,
        C_WAVE_TO_RAY_CAPABILITIES,
        C_PLANAR_DOE_STEP_CAPABILITIES,
        C_PATCH_WFT_CAPABILITIES,
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
            "C_PLANAR_DOE_STEP",
            "C_PATCH_WFT",
        )
    ]
