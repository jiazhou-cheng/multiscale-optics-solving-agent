"""The M1 standalone ray baseline, kept out of the graph-facing adapter.

``run_standalone`` implements the frozen ``M1-BASELINE-CPU-V2`` contract: a
fresh process, no coupler, a fixed artifact set, and a structured blocker rather
than a fabricated value when the solver refuses. It is not the ``ModelAdapter``
protocol and nothing in a graph reaches it.

CHE-91 considered archiving it with the gen1 suites and did not: it has a live
consumer outside them in
``benchmarks/probes/optiland/standalone_baseline.py``, which is the
executable evidence behind a card claim. Moving it out of the adapter is the same
separation without losing the evidence.

It takes the adapter as an argument rather than being a method on it. The only
thing it needs is ``run``, and saying so makes the direction of the dependency
visible: the baseline uses the adapter, not the other way round.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from typing import Any

from pydantic import ValidationError

from core.errors import (
    AdapterDependencyError,
    UnsupportedCapabilityError,
)
from core.optical_system import (
    OpticalSystemSpec,
)
from solvers.base import (
    ModelRunRequest,
    RunStatus,
)
from solvers.optiland.builder import build_optiland_system

# Re-exported for callers that reach for them on this module. CHE-91 moved the
# definitions into cohesive siblings, but `solvers.optiland.adapter` stays the
# addressable surface: several tests patch `_import_optiland` and `_resolve_lens`
# *here*, and a patch target is part of a module's contract even when the name is
# private. Keeping the binding means the split needed no test edits, which is the
# whole standard a characterization refactor is held to.
from solvers.optiland.constants import (  # noqa: F401
    _BASELINE_SEED,
    _DEFAULT_HANDOFF_PLANE,
    _DEFAULT_HX,
    _DEFAULT_HY,
    _DEFAULT_NUM_RAYS,
    _DEFAULT_WAVELENGTH,
    _DIRECTION_NORM_TOLERANCE,
    _GEOMETRY_M_PER_MM,
    _MISSING_WAVEFRONT_METADATA,
    _OPD_WARNING,
    _SUPPORTED_BACKENDS,
    _SUPPORTED_HANDOFF_PLANES,
    _SUPPORTED_SAMPLES,
    _VALIDATED_DESIGN_PARAMETER_PATTERN,
    _WAVELENGTH_M_PER_UM,
    MODEL_ID,
)
from solvers.optiland.provenance import (
    _cpu_device_name,
)
from solvers.optiland.requests import (
    OptilandRayFailure,
    OptilandRayRequest,
    OptilandRayResult,
)


def _resolve_lens(spec: OpticalSystemSpec) -> Any:
    """Build the system through the one generic construction path."""
    return build_optiland_system(spec)





def run_standalone(
    adapter: Any, request: OptilandRayRequest | Mapping[str, Any]
) -> OptilandRayResult:
    """Run the one deterministic CHE-13 CPU baseline and persist its summary.

    A mapping is accepted at the process/CLI boundary so malformed input can
    be returned as a structured diagnostic. Valid input is immediately
    converted to the typed request above.
    """
    started = time.perf_counter()
    try:
        typed = (
            request
            if isinstance(request, OptilandRayRequest)
            else OptilandRayRequest.model_validate(request)
        )
    except ValidationError as exc:
        return OptilandRayResult(
            status=RunStatus.FAILED,
            runtime_seconds=time.perf_counter() - started,
            failure=OptilandRayFailure(
                code="OPTILAND_INVALID_BASELINE_REQUEST",
                message=str(exc),
                stage="request_validation",
                exception_type=type(exc).__name__,
            ),
        )

    model_request = ModelRunRequest(
        run_id="che13-standalone",
        node_id="optiland-ray-baseline",
        inputs={},
        config={
            "sample": typed.prescription,
            "backend": typed.backend,
            "device": typed.device,
            "dtype": typed.dtype,
            "wavelength": typed.wavelength_um,
            "Hx": typed.field_hx,
            "Hy": typed.field_hy,
            "num_rays": typed.pupil_sampling,
            "output_directory": str(typed.output_directory),
            "seed": typed.seed,
        },
        design_parameters={},
        require_gradients=typed.require_gradients,
    )
    try:
        result = adapter.run(model_request)
    except (AdapterDependencyError, UnsupportedCapabilityError) as exc:
        code = (
            "OPTILAND_DEPENDENCY_UNAVAILABLE"
            if isinstance(exc, AdapterDependencyError)
            else "OPTILAND_UNSUPPORTED_BASELINE_REQUEST"
        )
        return OptilandRayResult(
            status=RunStatus.FAILED,
            backend=typed.backend,
            device=typed.device,
            cpu_device=_cpu_device_name(),
            dtype=typed.dtype,
            requested_sampling=typed.pupil_sampling,
            runtime_seconds=time.perf_counter() - started,
            output_directory=str(typed.output_directory),
            failure=OptilandRayFailure(
                code=code,
                message=str(exc),
                stage="dependency_or_capability_gate",
                exception_type=type(exc).__name__,
            ),
        )

    runtime_seconds = time.perf_counter() - started
    if result.status is not RunStatus.SUCCEEDED:
        diagnostic_code = str(result.diagnostics.get("code", "OPTILAND_BASELINE_FAILED"))
        return OptilandRayResult(
            status=RunStatus.FAILED,
            package_version=result.diagnostics.get("package_version"),
            backend=typed.backend,
            device=typed.device,
            cpu_device=_cpu_device_name(),
            dtype=typed.dtype,
            requested_sampling=typed.pupil_sampling,
            runtime_seconds=runtime_seconds,
            output_directory=str(typed.output_directory),
            warnings=result.warnings,
            failure=OptilandRayFailure(
                code=diagnostic_code,
                message=result.error_message or "Optiland baseline failed without a message.",
                stage=str(result.diagnostics.get("stage", "adapter_run")),
                exception_type=result.error_type,
            ),
        )

    rays_artifact = result.outputs["rays"]
    summary_metrics = dict(result.diagnostics["summary_metrics"])
    summary = {
        "schema_version": 1,
        "prescription": typed.prescription,
        "backend": typed.backend,
        "device": typed.device,
        "dtype": typed.dtype,
        "wavelength_um": typed.wavelength_um,
        "field_hx": typed.field_hx,
        "field_hy": typed.field_hy,
        "requested_sampling": typed.pupil_sampling,
        "seed": typed.seed,
        "seed_semantics": (
            "recorded; Optiland hexapolar sampler is deterministic and uses no RNG"
        ),
        "surviving_ray_count": int(rays_artifact.shape[0]),
        "scientific_array_sha256": result.diagnostics["scientific_array_sha256"],
        "summary_metrics": summary_metrics,
        "conventions": rays_artifact.metadata["conventions"],
    }
    summary_path = typed.output_directory / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    return OptilandRayResult(
        status=RunStatus.SUCCEEDED,
        package_version=result.diagnostics["package_version"],
        backend=typed.backend,
        device=typed.device,
        cpu_device=result.diagnostics["cpu_device"],
        dtype=typed.dtype,
        requested_sampling=typed.pupil_sampling,
        surviving_ray_count=int(rays_artifact.shape[0]),
        runtime_seconds=runtime_seconds,
        output_directory=str(typed.output_directory),
        arrays_path=rays_artifact.uri,
        summary_path=str(summary_path),
        scientific_array_sha256=result.diagnostics["scientific_array_sha256"],
        summary_metrics=summary_metrics,
        warnings=result.warnings,
    )
