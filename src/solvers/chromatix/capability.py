"""Why a request is refused, decided before any solver runs.

`check_capability` raises and computes nothing. AGENTS.md requires an
unsupported request to be refused eagerly, and Chromatix has more to refuse than
most: it is FP32-only with no complex128 storage at any device, so a precision
request that another solver would merely round is one this one cannot execute.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from core.errors import (
    UnsupportedCapabilityError,
)
from core.precision import (
    CapabilityError,
    DeviceKind,
    DType,
)
from core.specs import ArtifactKind, ModelSpec
from solvers.base import (
    ModelRunRequest,
)
from solvers.chromatix.constants import (
    _SUPPORTED_PROPAGATION,
    MODEL_ID,
)
from solvers.chromatix.execution import (
    _jax_gpu_unavailable_reason,
    _plan_input_bridge,
    _resolve_chromatix_execution,
)

if TYPE_CHECKING:
    pass



def check_capability(spec: ModelSpec, request: ModelRunRequest) -> None:
    config = request.config

    propagation = config.get("propagation")
    if propagation != _SUPPORTED_PROPAGATION:
        raise UnsupportedCapabilityError(
            "M_WAVE_CHROMATIX adapter only implements "
            f"config['propagation'] == {_SUPPORTED_PROPAGATION!r} "
            "(chromatix.functional.asm_propagate, angular spectrum). Got "
            f"{propagation!r}. transform_propagate/Fresnel and any other "
            "kernel are not implemented (see conventions.md: "
            "transform_propagate changes sample pitch and has only a "
            "narrow, unrelated gradient probe, not a forward-propagation "
            "regression oracle)."
        )

    field_kind = config.get("field_kind", "scalar")
    if field_kind != "scalar":
        raise UnsupportedCapabilityError(
            "M_WAVE_CHROMATIX adapter only implements scalar fields "
            f"(chromatix ScalarField); got field_kind={field_kind!r}. "
            "Vector/polarized propagation has not been probed (see "
            "knowledge/solvers/chromatix/capability_notes.md)."
        )

    if request.require_gradients:
        raise UnsupportedCapabilityError(
            "M_WAVE_CHROMATIX adapter does not claim a verified derivative "
            "for asm_propagate. Only one narrow probe "
            "(thin_lens focal length -> transform_propagate -> intensity) "
            "passed a directional-derivative check "
            "(knowledge/solvers/chromatix/expected/gradient_probe.json); "
            "that path is not implemented by this adapter and asm_propagate "
            "has no such evidence. require_gradients=True is rejected."
        )

    # CHE-61 (PB4b) replaces CHE-55's blanket `device != 'cpu'` rejection.
    # The guard is not removed, it is narrowed to what is actually true:
    # Chromatix executes on CPU and on CUDA (measured: asm_propagate returns
    # complex64 on cuda:0), and it has NO complex128 path at any device, so an
    # FP64 request is refused on precision grounds rather than device ones.
    # CHE-55's stated worry -- that an installed GPU-enabled jaxlib could
    # silently change what this adapter executes -- is now addressed by
    # placing the field explicitly and reporting the OBSERVED output device,
    # instead of by refusing the GPU outright.
    resolved = _resolve_chromatix_execution(config)
    if resolved.device.kind is DeviceKind.CUDA:
        reason = _jax_gpu_unavailable_reason()
        if reason is not None:
            raise CapabilityError(
                code="CHROMATIX_CUDA_UNAVAILABLE",
                component=MODEL_ID,
                message=(
                    f"config['device']={str(resolved.device)!r} is executable for "
                    f"this adapter, but not in this process: {reason}."
                ),
                requested=resolved.device,
                supported=["cpu"],
                remedy=(
                    "Run in the CUDA container (`./run.sh --gpu ...`, see "
                    "docs/testing/gpu_environment.md). There is deliberately "
                    "no silent fallback to the CPU."
                ),
            )

    if "optical_surface" in request.inputs:
        raise UnsupportedCapabilityError(
            "M_WAVE_CHROMATIX adapter does not implement fusion of an "
            "'optical_surface' input with the propagated field in this "
            "scope; only free-space scalar angular-spectrum propagation "
            "of 'input_field' is supported."
        )

    # The input dtype gate, planned eagerly from the record's own declaration
    # so a refused conversion is refused before chromatix is imported rather
    # than mid-propagation. The same planner runs again on the loaded array in
    # _run_asm_propagate, where the observed dtype -- not the declared one --
    # is what gets recorded as provenance.
    declared = request.inputs.get("input_field")
    if declared is not None and declared.dtype:
        _plan_input_bridge(DType.parse(declared.dtype), config)

    input_record = request.inputs.get("input_field")
    if input_record is not None and input_record.kind != ArtifactKind.COMPLEX_FIELD:
        raise UnsupportedCapabilityError(
            "M_WAVE_CHROMATIX adapter expects input_field artifact kind "
            f"{ArtifactKind.COMPLEX_FIELD.value!r}; got "
            f"{input_record.kind.value!r} (e.g. a vector_field is out of "
            "scope for this adapter)."
        )

# ------------------------------------------------------------------
# ModelAdapter protocol
# ------------------------------------------------------------------
