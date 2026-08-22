"""Why a request is refused, decided before any solver runs.

``capability_problems`` returns ``(code, message)`` pairs and executes nothing.
That property is the reason it has its own module: AGENTS.md requires an
unsupported request to be refused *eagerly*, and a refusal sharing a file with
the trace is one careless edit away from being computed after it.

Every code it can report is in the inventory pinned by
``tests/test_solver_adapter_characterization.py``.
"""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from core.optical_system import (
    OPTICAL_SYSTEM_SPEC_VERSION,
    OpticalSystemSpec,
    PrescriptionError,
)
from core.precision import (
    CapabilityError,
    DeviceKind,
)
from solvers.base import (
    ModelRunRequest,
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
from solvers.optiland.execution import (
    _cuda_unavailable_reason,
    _resolve_optiland_execution,
)
from solvers.optiland.requests import (
    _prescription_from_config,
)


def _resolve_lens(spec: OpticalSystemSpec) -> Any:
    """Build the system through the one generic construction path."""
    return build_optiland_system(spec)





def capability_problems(request: ModelRunRequest) -> list[tuple[str, str]]:
    """Return (code, message) pairs for every deliberately-unimplemented request feature.

    Non-empty output means run()/estimate() must raise
    UnsupportedCapabilityError before touching optiland.
    """
    problems: list[tuple[str, str]] = []
    backend_name = request.config.get("backend", "numpy")

    if backend_name not in _SUPPORTED_BACKENDS:
        problems.append(
            (
                "OPTILAND_UNSUPPORTED_BACKEND",
                f"config['backend']={backend_name!r} is not supported; use 'numpy' or 'torch'.",
            )
        )

    if request.require_gradients and backend_name != "torch":
        problems.append(
            (
                "OPTILAND_GRADIENTS_REQUIRE_TORCH_BACKEND",
                "require_gradients=True but config['backend'] is not "
                "explicitly 'torch'. The default numpy backend has "
                "optiland.backend.supports_gradients=False (see "
                "knowledge/solvers/optiland/conventions.md); set "
                "config['backend']='torch' explicitly to opt into the "
                "differentiable path. This adapter never silently returns "
                "a non-differentiable numpy result for a requested "
                "gradient.",
            )
        )
    if request.require_gradients and backend_name == "torch" and not request.design_parameters:
        problems.append(
            (
                "OPTILAND_GRADIENTS_REQUIRE_DESIGN_PARAMETERS",
                "require_gradients=True needs at least one entry in "
                "design_parameters to attach an autograd leaf to (this "
                "adapter validates only "
                "'surfaces.surfaces[<index>].geometry.radius'); with no "
                "design parameters there is nothing for a caller to call "
                ".backward() against.",
            )
        )

    # CHE-61: device and precision are negotiated against Optiland's real
    # capability declaration instead of compared with two string constants.
    # What changes for the caller: 'cuda' and 'float32' are now accepted
    # where Optiland can execute them, and 'float16' is refused with the
    # reason -- set_precision is literally Literal['float32','float64'], so
    # there is no float16 path to promote into. What does NOT change: the
    # defaults, so an existing request means exactly what it always meant.
    if backend_name in _SUPPORTED_BACKENDS:
        try:
            resolved = _resolve_optiland_execution(request.config)
        except CapabilityError as exc:
            problems.append((f"OPTILAND_{exc.code}", str(exc)))
        else:
            if resolved.device.kind is DeviceKind.CUDA:
                reason = _cuda_unavailable_reason()
                if reason is not None:
                    problems.append(
                        (
                            "OPTILAND_CUDA_UNAVAILABLE",
                            f"config['device']={str(resolved.device)!r} is executable "
                            "for this adapter, but not in this container: "
                            f"{reason}. Run `./run.sh --gpu ...` (see "
                            "docs/testing/gpu_environment.md). There is "
                            "deliberately no silent fallback to the CPU.",
                        )
                    )

    # System construction: either a registered canonical prescription named
    # by config['sample'], or one supplied inline through
    # config['prescription']. Both are validated here, before any solver
    # import, so a malformed prescription can never produce a partially
    # constructed lens (CHE-56).
    try:
        _prescription_from_config(request.config)
    except PrescriptionError as exc:
        code = (
            "OPTILAND_UNSUPPORTED_SAMPLE"
            if exc.code == "PRESCRIPTION_NAME_UNKNOWN"
            else "OPTILAND_INVALID_PRESCRIPTION"
        )
        problems.append((code, str(exc)))
    except ValidationError as exc:
        problems.append(
            (
                "OPTILAND_INVALID_PRESCRIPTION",
                "config['prescription'] is not a valid canonical optical-system "
                f"specification ({OPTICAL_SYSTEM_SPEC_VERSION}): {exc}",
            )
        )

    handoff_plane = request.config.get("handoff_plane", _DEFAULT_HANDOFF_PLANE)
    if handoff_plane not in _SUPPORTED_HANDOFF_PLANES:
        problems.append(
            (
                "OPTILAND_UNSUPPORTED_HANDOFF_PLANE",
                f"config['handoff_plane']={handoff_plane!r} is not one of "
                f"{_SUPPORTED_HANDOFF_PLANES!r}. A reference sphere in "
                "particular is not implemented: the ray-to-wave coupler "
                "accumulates onto a plane (M3.2). (The coupler is named here "
                "only in prose -- benchmarks/verify_m1_independence.py fails "
                "the ray branch if its identifier appears in this source.)",
            )
        )

    if "system" in request.inputs:
        problems.append(
            (
                "OPTILAND_CUSTOM_SYSTEM_NOT_IMPLEMENTED",
                "The optional 'system' input port is not implemented: it "
                "would carry an arbitrary solver object, which has no typed "
                "contract and no validation. Since CHE-56 a custom lens IS "
                "supported -- as a canonical prescription "
                f"({OPTICAL_SYSTEM_SPEC_VERSION}) passed through "
                "config['prescription'], or by name through "
                "config['sample']. See "
                "docs/prescriptions/canonical_optical_systems.md.",
            )
        )

    for name in request.design_parameters:
        if not _VALIDATED_DESIGN_PARAMETER_PATTERN.match(name):
            problems.append(
                (
                    "OPTILAND_UNSUPPORTED_DESIGN_PARAMETER",
                    f"design_parameters key {name!r} is not one of the "
                    "parameter paths this adapter has validated (only "
                    "'surfaces.surfaces[<index>].geometry.radius' on the "
                    "selected sample lens, matching "
                    "knowledge/solvers/optiland/probes/gradient_probe.py).",
                )
            )

    return problems
