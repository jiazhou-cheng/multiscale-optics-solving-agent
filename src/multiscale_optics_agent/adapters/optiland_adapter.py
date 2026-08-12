"""Adapter for ``M_RAY_OPTILAND`` (Optiland sequential ray tracer, pinned 0.6.0).

Grounded in ``knowledge/solvers/optiland/`` (``solver_card.yaml``,
``conventions.md``, ``capability_notes.md``, ``api_minimal_examples.md``,
``failure_guide.md``) and the probes/expected fixtures under that directory,
all captured against the ``agent_solver`` container on 2026-07-30. Nothing
below relies on training-data memory of an older/different Optiland API.

Scope (deliberately narrow -- see current Optiland scope "do not broaden scope
while a P0 model/coupler lacks tests")
------------------------------------------------------------------------
- Only the bundled sample lens systems that have actually been probed are
  supported (currently just ``optiland.samples.objectives.ReverseTelephoto``,
  selected via ``config["sample"]``, default ``"ReverseTelephoto"``). A
  custom, hand-built lens prescription supplied through the optional
  ``system`` input port is NOT implemented: building one from scratch has not
  been exercised against this pinned install (see
  ``knowledge/solvers/optiland/capability_notes.md``, "Not yet exercised"),
  so this adapter raises ``UnsupportedCapabilityError`` rather than guessing
  at an untested construction API.
- Only ``config["device"] == "cpu"`` and ``config["dtype"] == "float64"`` are
  supported (matching what was actually probed); GPU execution was never
  tested (no CUDA device in the probing container) and is rejected eagerly
  rather than silently attempted.
- Only one design-parameter path is supported for the differentiable
  (torch-backend) case: ``surfaces.surfaces[<index>].geometry.radius`` on the
  selected sample lens, matching
  ``knowledge/solvers/optiland/probes/gradient_probe.py`` exactly. This is
  narrower than the registry's ``derivative.parameters: ["*"]`` claim in
  ``registry/models.yaml`` -- that field describes an aspiration for the
  model class in general, not a guarantee that this adapter implementation
  has validated every parameter path. Any other design-parameter key is
  rejected eagerly.

Backend policy (no silent convention changes or unverified gradient claims)
------------------------------------------------------------------------
Optiland has exactly two numerical backends, selected process-globally at
runtime via ``optiland.backend.set_backend(...)`` -- NOT per-object and NOT
at install time (verbatim from ``optiland.backend.__doc__``, see
``conventions.md``). The default (and the only backend present after a bare
``pip install optiland``) is NumPy, which has **no autodiff**
(``optiland.backend.supports_gradients`` is ``False``). ``torch`` is not a
declared optiland dependency at all (confirmed via
``importlib.metadata.distribution('optiland').requires`` --
``knowledge/solvers/optiland/failure_guide.md``).

Consequently:
- This adapter defaults to ``config.get("backend", "numpy")``.
- If ``request.require_gradients`` is ``True`` but ``config["backend"]`` is
  not explicitly ``"torch"``, ``run()``/``estimate()`` raise
  ``UnsupportedCapabilityError`` *before* importing optiland or torch. This
  adapter never silently substitutes a non-differentiable NumPy result for a
  requested gradient (repository scientific-contract requirements).
- Every call to ``run()`` explicitly calls ``be.set_backend(backend_name)``,
  even when ``backend_name == "numpy"``. ``set_backend`` mutates global,
  process-wide module state and is documented as **not thread-safe**
  (``conventions.md``); relying on "whatever the default already is" would
  silently pick up a backend left behind by a previous run in the same
  process. This adapter must never be called concurrently from multiple
  threads with different backends -- that restriction is inherited directly
  from optiland, not invented here.
- Converting a torch tensor to NumPy for on-disk ``ArtifactRecord``
  persistence goes through ``optiland.backend.utils.to_numpy``, which
  internally calls ``tensor.detach().cpu().numpy()`` (verified by reading its
  source in the pinned install). That is a real derivative-boundary detach
  (repository scientific-contract requirements) and is recorded explicitly in
  ``ModelRunResult.warnings``/``diagnostics`` on every torch-backend run; the
  live, still-differentiable tensor is additionally kept (undetached) in
  ``diagnostics["objective_tensor"]``/``diagnostics["design_parameter_tensors"]``
  when ``require_gradients=True``, specifically so a caller can call
  ``.backward()`` on it -- ``ArtifactRecord`` itself has no field that could
  hold a live tensor with an attached autograd graph.
- The one gradient path this adapter exposes has a directional-derivative
  relative error of 1.11e-03 against centered finite difference (see
  ``knowledge/solvers/optiland/expected/gradient_probe.json``), looser than
  every JAX-based solver in this repository and not yet root-caused.
  ``registry/models.yaml`` already declares ``derivative.verified: false``
  for ``M_RAY_OPTILAND``; nothing in this adapter changes that, and this
  module never sets ``verified: true`` anywhere.

Units (repository scientific-contract requirements)
------------------------------------------------------------------------
CHE-12 verified that Optiland geometry is expressed in millimetres and trace
wavelength in micrometres. ``config["wavelength"]`` remains a native-
micrometre solver input for compatibility, while persisted position and
wavelength arrays cross the adapter boundary in SI using exactly ``1e-3
m/mm`` and ``1e-6 m/um``. ``RealRays.opd`` is deliberately preserved as
``opd_native``: its reference and sign semantics remain unverified and it is
not presented as an absolute OPL/OPD oracle.

Exception-handling convention (per ``core/errors.py`` docstring, mirroring
``fdtdx_adapter.py``)
------------------------------------------------------------------------
- ``AdapterDependencyError`` is raised (propagates) when ``optiland`` cannot
  be imported at all, or when ``torch`` cannot be imported for a request that
  needs the torch backend.
- ``UnsupportedCapabilityError`` is raised (propagates) *eagerly*, before any
  solver call, for: ``require_gradients=True`` without ``backend="torch"``;
  ``require_gradients=True`` with no ``design_parameters`` to differentiate
  against; an unsupported ``backend``/``device``/``dtype``; an unsupported
  ``sample``; a custom ``system`` input; or an unrecognized
  ``design_parameters`` key.
- Failures once the solver has actually been invoked (an exception from
  ``Optic.trace`` or from setting an attribute on the lens) are caught at the
  ``run()`` boundary and reported as
  ``ModelRunResult(status=RunStatus.FAILED, error_type=..., error_message=...)``
  rather than raised, per ``SolverExecutionError``'s docstring.

Output ports and known metadata gaps (repository scientific-contract requirements: no
fabricated solver output)
------------------------------------------------------------------------
``registry/models.yaml`` declares that the ``rays`` output port provides
``amplitude``/``polarization`` metadata and that ``wavefront`` additionally
provides ``optical_path_length``/``pupil_mask``. The traced object returned
by ``Optic.trace`` is ``optiland.rays.real_rays.RealRays``, which was
inspected directly against the pinned install and exposes exactly
``x, y, z, L, M, N, i, w, opd`` -- a real-valued per-ray *intensity* (``i``),
not a complex amplitude or Jones vector, and no pupil-mask array. This
adapter never fabricates the missing fields: it populates only
``i``/``opd``/coordinates/direction/wavelength, and records the gap
explicitly (``metadata["missing_declared_metadata"]`` plus a matching
``ModelRunResult`` warning) rather than inventing placeholder values.
"""

from __future__ import annotations

import hashlib
import json
import platform
import re
import tempfile
import time
import uuid
from collections.abc import Mapping
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from multiscale_optics_agent.adapters.base import (
    CostEstimate,
    ModelRunRequest,
    ModelRunResult,
    RunStatus,
)
from multiscale_optics_agent.core.artifacts import ArtifactRecord
from multiscale_optics_agent.core.errors import (
    AdapterDependencyError,
    UnsupportedCapabilityError,
)
from multiscale_optics_agent.core.graph import Severity, ValidationIssue, ValidationReport
from multiscale_optics_agent.core.specs import ArtifactKind, Device, Framework, ModelSpec
from multiscale_optics_agent.registry.loader import Registry

MODEL_ID = "M_RAY_OPTILAND"

_SUPPORTED_BACKENDS = ("numpy", "torch")
_SUPPORTED_SAMPLES = ("ReverseTelephoto",)
_SUPPORTED_DEVICE = "cpu"
_SUPPORTED_DTYPE = "float64"
_DIRECTION_NORM_TOLERANCE = 1e-12
_GEOMETRY_M_PER_MM = 1e-3
_WAVELENGTH_M_PER_UM = 1e-6
_BASELINE_SEED = 20260811

# The only design-parameter path characterized by
# knowledge/solvers/optiland/probes/gradient_probe.py.
_VALIDATED_DESIGN_PARAMETER_PATTERN = re.compile(r"^surfaces\.surfaces\[(\d+)\]\.geometry\.radius$")

_DEFAULT_WAVELENGTH = 0.55  # micrometres; verified by CHE-12
_DEFAULT_NUM_RAYS = 16
_DEFAULT_HX = 0.0
_DEFAULT_HY = 0.0

_MISSING_WAVEFRONT_METADATA = ["amplitude", "polarization", "pupil_mask"]

_OPD_WARNING = (
    "RealRays.opd is preserved in Optiland-native values because its reference "
    "and sign convention remain unverified; it is regression evidence, not an "
    "absolute OPL/OPD oracle."
)


class OptilandRayRequest(BaseModel):
    """Typed contract for the single CHE-13 standalone ray baseline."""

    model_config = ConfigDict(extra="forbid")

    prescription: Literal["ReverseTelephoto"] = "ReverseTelephoto"
    backend: Literal["numpy"] = "numpy"
    device: Literal["cpu"] = "cpu"
    dtype: Literal["float64"] = "float64"
    wavelength_um: float = Field(default=_DEFAULT_WAVELENGTH, gt=0)
    field_hx: float = Field(default=_DEFAULT_HX, ge=-1.0, le=1.0)
    field_hy: float = Field(default=_DEFAULT_HY, ge=-1.0, le=1.0)
    pupil_sampling: int = Field(default=_DEFAULT_NUM_RAYS, gt=0)
    output_directory: Path
    seed: int = _BASELINE_SEED
    require_gradients: Literal[False] = False


class OptilandRayFailure(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str
    message: str
    stage: str
    exception_type: str | None = None


class OptilandRayResult(BaseModel):
    """Structured success/failure result for :class:`OptilandRayRequest`."""

    model_config = ConfigDict(extra="forbid")

    status: RunStatus
    package_version: str | None = None
    backend: str | None = None
    device: str | None = None
    cpu_device: str | None = None
    dtype: str | None = None
    requested_sampling: int | None = None
    surviving_ray_count: int | None = None
    runtime_seconds: float | None = None
    output_directory: str | None = None
    arrays_path: str | None = None
    summary_path: str | None = None
    scientific_array_sha256: str | None = None
    summary_metrics: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    failure: OptilandRayFailure | None = None


def _import_optiland(*, need_torch: bool) -> tuple[Any, Any, Any, Any]:
    """Lazily import optiland (and torch, only if requested).

    Never called at module import time -- only from run()/estimate(). Returns
    (optiland.backend, optiland.samples.objectives, optiland.backend.utils,
    torch-module-or-None).
    """
    try:
        import optiland.backend as be  # type: ignore[import-untyped]
        import optiland.backend.utils as be_utils  # type: ignore[import-untyped]
        from optiland.samples import objectives  # type: ignore[import-untyped]
    except Exception as exc:
        raise AdapterDependencyError(
            f"optiland could not be imported: {type(exc).__name__}: {exc}. "
            "Install it via `pip install optiland==0.6.0` or this project's "
            "'torch' extra (`pip install .[torch]`, which also pins torch)."
        ) from exc

    torch_module: Any = None
    if need_torch:
        try:
            import torch

            torch_module = torch
        except Exception as exc:
            raise AdapterDependencyError(
                "config['backend']='torch' (or require_gradients=True) needs "
                "the optional torch package. torch is NOT a declared optiland "
                "dependency (knowledge/solvers/optiland/failure_guide.md) and "
                f"must be installed separately: {type(exc).__name__}: {exc}"
            ) from exc

    return be, objectives, be_utils, torch_module


@lru_cache(maxsize=1)
def _load_spec() -> ModelSpec:
    return Registry.from_package().models[MODEL_ID]


def _cpu_device_name() -> str:
    """Return an observable CPU description without claiming core isolation."""
    model = platform.processor().strip()
    if not model:
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.lower().startswith("model name"):
                    model = line.split(":", 1)[1].strip()
                    break
        except OSError:
            pass
    return model or platform.machine() or "cpu"


def _scientific_array_hash(arrays: Mapping[str, Any]) -> str:
    """Hash names, dtype, shape, and contiguous bytes independent of NPZ metadata."""
    import numpy as np

    digest = hashlib.sha256()
    for name in sorted(arrays):
        array = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(json.dumps(list(array.shape)).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


class OptilandAdapter:
    """``ModelAdapter`` for ``M_RAY_OPTILAND`` (Optiland 0.6.0). See module docstring."""

    @property
    def spec(self) -> ModelSpec:
        return _load_spec()

    def run_standalone(self, request: OptilandRayRequest | Mapping[str, Any]) -> OptilandRayResult:
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
            result = self.run(model_request)
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

    def estimate(self, request: ModelRunRequest) -> CostEstimate:
        problems = self._capability_problems(request)
        if problems:
            raise UnsupportedCapabilityError("; ".join(message for _, message in problems))

        num_rays = int(request.config.get("num_rays", _DEFAULT_NUM_RAYS))
        return CostEstimate(
            wall_time_s=None,
            peak_memory_bytes=None,
            solver_calls=1,
            confidence="low",
            notes=[
                "Heuristic only -- optiland is not imported for this estimate.",
                "Registry cost_model: O(number_of_surfaces * number_of_rays); "
                "this estimator does not read the actual surface count.",
                f"requested num_rays={num_rays}; the traced ray count returned "
                "by Optic.trace() is typically much larger due to pupil "
                "sampling (see knowledge/solvers/optiland/failure_guide.md) "
                "and is not known until after the solver call.",
            ],
        )

    def validate_request(self, request: ModelRunRequest) -> ValidationReport:
        issues = [
            ValidationIssue(severity=Severity.ERROR, code=code, message=message, location="request")
            for code, message in self._capability_problems(request)
        ]

        num_rays = request.config.get("num_rays", _DEFAULT_NUM_RAYS)
        if not isinstance(num_rays, int | float) or num_rays <= 0:
            issues.append(
                ValidationIssue(
                    severity=Severity.ERROR,
                    code="OPTILAND_INVALID_NUM_RAYS",
                    message=f"config['num_rays'] must be a positive number, got {num_rays!r}.",
                    location="config.num_rays",
                )
            )
        wavelength = request.config.get("wavelength", _DEFAULT_WAVELENGTH)
        if not isinstance(wavelength, int | float) or wavelength <= 0:
            issues.append(
                ValidationIssue(
                    severity=Severity.ERROR,
                    code="OPTILAND_INVALID_WAVELENGTH",
                    message=f"config['wavelength'] must be a positive number, got {wavelength!r}.",
                    location="config.wavelength",
                )
            )

        if not issues:
            issues.append(
                ValidationIssue(
                    severity=Severity.INFO,
                    code="OPTILAND_REQUEST_OK",
                    message="Request is within the scope this adapter implements.",
                )
            )
        return ValidationReport(issues=issues)

    def run(self, request: ModelRunRequest) -> ModelRunResult:
        # Eager, pre-solver capability gate: this must happen before any
        # optiland/torch import so a
        # caller never silently receives a non-differentiable numpy result
        # when it asked for gradients, and never triggers an untested code
        # path (custom system, unsupported sample/device/dtype/parameter).
        problems = self._capability_problems(request)
        if problems:
            raise UnsupportedCapabilityError("; ".join(message for _, message in problems))

        request_report = self.validate_request(request)
        if not request_report.valid:
            return ModelRunResult(
                status=RunStatus.FAILED,
                error_type="ValueError",
                error_message="; ".join(issue.message for issue in request_report.errors),
                diagnostics={
                    "code": "OPTILAND_INVALID_REQUEST",
                    "stage": "request_validation",
                    "validation_codes": [issue.code for issue in request_report.errors],
                },
            )

        started = time.perf_counter()
        backend_name = str(request.config.get("backend", "numpy"))
        sample_name = str(request.config.get("sample", "ReverseTelephoto"))
        wavelength = float(request.config.get("wavelength", _DEFAULT_WAVELENGTH))
        hx = float(request.config.get("Hx", _DEFAULT_HX))
        hy = float(request.config.get("Hy", _DEFAULT_HY))
        num_rays = int(request.config.get("num_rays", _DEFAULT_NUM_RAYS))

        be, objectives, be_utils, torch_module = _import_optiland(
            need_torch=backend_name == "torch"
        )

        try:
            package_version = version("optiland")
        except PackageNotFoundError:
            package_version = "unknown"

        # optiland.backend.set_backend mutates global, process-wide module
        # state and is documented as not thread-safe (conventions.md). Set it
        # explicitly on every run, even for the numpy default, so a previous
        # run's backend choice in this process can never silently leak in.
        be.set_backend(backend_name)

        warnings: list[str] = [_OPD_WARNING]
        if backend_name == "numpy":
            warnings.append(
                "backend='numpy' (default): optiland.backend.supports_gradients "
                "is False for this run -- no gradients are available, "
                "regardless of the registry's derivative.mode=native_autodiff "
                "(that mode only applies once backend='torch' is selected)."
            )

        try:
            sample_cls = getattr(objectives, sample_name)
            lens = sample_cls()

            design_parameter_tensors: dict[str, Any] = {}
            for name, value in request.design_parameters.items():
                match = _VALIDATED_DESIGN_PARAMETER_PATTERN.match(name)
                assert match is not None  # already enforced by _capability_problems
                surface_index = int(match.group(1))
                surface = lens.surfaces.surfaces[surface_index]
                if backend_name == "torch":
                    tensor = torch_module.tensor(
                        float(value), dtype=torch_module.float64, requires_grad=True
                    )
                    surface.geometry.radius = tensor
                    design_parameter_tensors[name] = tensor
                else:
                    surface.geometry.radius = float(value)

            rays = lens.trace(Hx=hx, Hy=hy, wavelength=wavelength, num_rays=num_rays)
        except (AdapterDependencyError, UnsupportedCapabilityError):
            raise
        except Exception as exc:
            return ModelRunResult(
                status=RunStatus.FAILED,
                error_type=type(exc).__name__,
                error_message=str(exc),
                warnings=warnings,
                diagnostics={
                    "code": "OPTILAND_TRACE_FAILED",
                    "stage": "optiland_build_or_trace",
                    "sample": sample_name,
                    "backend_used": backend_name,
                    "package_version": package_version,
                },
            )

        diagnostics: dict[str, Any] = {
            "backend_used": backend_name,
            "sample": sample_name,
            "requested_num_rays": num_rays,
            "Hx": hx,
            "Hy": hy,
            "wavelength_native_units": wavelength,
            "package_version": package_version,
            "device": _SUPPORTED_DEVICE,
            "cpu_device": _cpu_device_name(),
            "dtype": _SUPPORTED_DTYPE,
            "seed": int(request.config.get("seed", _BASELINE_SEED)),
            "seed_semantics": (
                "recorded; Optiland hexapolar sampler is deterministic and uses no RNG"
            ),
        }

        if backend_name == "torch" and request.require_gradients:
            objective_tensor = (rays.x**2 + rays.y**2).mean()
            diagnostics["objective_description"] = (
                "mean(x^2 + y^2) over traced rays at the requested field point "
                "-- an RMS-spot-size proxy, the only design_parameter -> "
                "objective path characterized for this adapter "
                "(knowledge/solvers/optiland/probes/gradient_probe.py)."
            )
            # Kept undetached on purpose (repository scientific-contract requirements): a
            # caller needing the gradient must call .backward() on this exact
            # tensor object. ArtifactRecord has no field for a live autograd
            # graph, so it cannot be exposed through `outputs` below.
            diagnostics["objective_tensor"] = objective_tensor
            diagnostics["design_parameter_tensors"] = design_parameter_tensors
            warnings.append(
                "diagnostics['objective_tensor'] is a live torch.Tensor with "
                "an attached autograd graph (undetached). Every "
                "ArtifactRecord in `outputs` is instead built from a detached "
                "NumPy copy (optiland.backend.utils.to_numpy, which calls "
                "tensor.detach().cpu().numpy() internally) for on-disk "
                "persistence -- that conversion is the recorded derivative "
                "boundary (repository scientific-contract requirements), and it does not carry "
                "gradient information."
            )

        try:
            configured_output = request.config.get("output_directory")
            if configured_output is None:
                run_dir = Path(
                    tempfile.mkdtemp(prefix=f"optiland_{request.run_id}_{request.node_id}_")
                )
            else:
                run_dir = Path(str(configured_output))
                run_dir.mkdir(parents=True, exist_ok=True)
            final_surface = lens.surfaces.surfaces[-1]
            reference_plane_z_mm = float(be_utils.to_numpy(final_surface.geometry.cs.z))
            rays_artifact = self._build_ray_bundle_artifact(
                request,
                rays,
                be_utils,
                backend_name,
                sample_name,
                wavelength,
                hx,
                hy,
                num_rays,
                run_dir,
                reference_plane_z_mm,
            )
            wavefront_artifact, wavefront_warnings = self._build_wavefront_artifact(
                request, rays, be_utils, backend_name, sample_name, wavelength, run_dir
            )
        except Exception as exc:
            return ModelRunResult(
                status=RunStatus.FAILED,
                error_type=type(exc).__name__,
                error_message=str(exc),
                warnings=warnings,
                diagnostics={
                    **diagnostics,
                    "code": "OPTILAND_INVALID_OR_EMPTY_OUTPUT",
                    "stage": "output_validation_or_persistence",
                },
            )

        warnings.extend(wavefront_warnings)
        diagnostics.update(
            {
                "actual_surviving_ray_count": int(rays_artifact.shape[0]),
                "runtime_seconds": time.perf_counter() - started,
                "scientific_array_sha256": rays_artifact.metadata["scientific_array_sha256"],
                "summary_metrics": rays_artifact.metadata["summary_metrics"],
            }
        )
        return ModelRunResult(
            status=RunStatus.SUCCEEDED,
            outputs={"rays": rays_artifact, "wavefront": wavefront_artifact},
            diagnostics=diagnostics,
            warnings=warnings,
        )

    def _capability_problems(self, request: ModelRunRequest) -> list[tuple[str, str]]:
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

        device = request.config.get("device", _SUPPORTED_DEVICE)
        if device != _SUPPORTED_DEVICE:
            problems.append(
                (
                    "OPTILAND_UNSUPPORTED_DEVICE",
                    f"config['device']={device!r} is not implemented; only "
                    "'cpu' has been exercised for this adapter (no CUDA "
                    "device was available when probing this pinned install -- "
                    "see knowledge/solvers/optiland/capability_notes.md).",
                )
            )

        dtype = request.config.get("dtype", _SUPPORTED_DTYPE)
        if dtype != _SUPPORTED_DTYPE:
            problems.append(
                (
                    "OPTILAND_UNSUPPORTED_DTYPE",
                    f"config['dtype']={dtype!r} is not implemented; only "
                    f"{_SUPPORTED_DTYPE!r} has been exercised for this adapter.",
                )
            )

        sample_name = request.config.get("sample", "ReverseTelephoto")
        if sample_name not in _SUPPORTED_SAMPLES:
            problems.append(
                (
                    "OPTILAND_UNSUPPORTED_SAMPLE",
                    f"config['sample']={sample_name!r} is not in the validated "
                    f"set {_SUPPORTED_SAMPLES!r}; only bundled "
                    "optiland.samples.objectives systems that have actually "
                    "been probed are supported.",
                )
            )

        if "system" in request.inputs:
            problems.append(
                (
                    "OPTILAND_CUSTOM_SYSTEM_NOT_IMPLEMENTED",
                    "A custom lens prescription via the optional 'system' "
                    "input port is not implemented/validated by this adapter "
                    "(knowledge/solvers/optiland/capability_notes.md lists a "
                    "hand-built, non-sample lens prescription as 'not yet "
                    "exercised'); only the bundled sample selected via "
                    "config['sample'] is supported.",
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

    @staticmethod
    def _build_ray_bundle_artifact(
        request: ModelRunRequest,
        rays: Any,
        be_utils: Any,
        backend_name: str,
        sample_name: str,
        wavelength: float,
        hx: float,
        hy: float,
        num_rays: int,
        run_dir: Path,
        reference_plane_z_mm: float,
    ) -> ArtifactRecord:
        import numpy as np

        native = {
            "x": np.asarray(be_utils.to_numpy(rays.x), dtype=np.float64),
            "y": np.asarray(be_utils.to_numpy(rays.y), dtype=np.float64),
            "z": np.asarray(be_utils.to_numpy(rays.z), dtype=np.float64),
            "L": np.asarray(be_utils.to_numpy(rays.L), dtype=np.float64),
            "M": np.asarray(be_utils.to_numpy(rays.M), dtype=np.float64),
            "N": np.asarray(be_utils.to_numpy(rays.N), dtype=np.float64),
            "intensity": np.asarray(be_utils.to_numpy(rays.i), dtype=np.float64),
            "wavelength_um": np.asarray(be_utils.to_numpy(rays.w), dtype=np.float64),
            "opd_native": np.asarray(be_utils.to_numpy(rays.opd), dtype=np.float64),
        }
        shapes = {name: array.shape for name, array in native.items()}
        if not native["x"].size:
            raise ValueError("Optiland returned an empty surviving-ray set.")
        if len(set(shapes.values())) != 1 or native["x"].ndim != 1:
            raise ValueError(f"RealRays arrays must be equal-length 1-D arrays; got {shapes!r}.")
        nonfinite = {
            name: int(np.count_nonzero(~np.isfinite(array))) for name, array in native.items()
        }
        if any(nonfinite.values()):
            raise ValueError(f"Optiland returned non-finite scientific output: {nonfinite!r}.")

        direction_norm = np.sqrt(native["L"] ** 2 + native["M"] ** 2 + native["N"] ** 2)
        max_direction_norm_error = float(np.max(np.abs(direction_norm - 1.0)))
        if backend_name == "numpy" and max_direction_norm_error > _DIRECTION_NORM_TOLERANCE:
            raise ValueError(
                "Optiland direction vectors are not unit norm: "
                f"max error {max_direction_norm_error:.17g} exceeds "
                f"{_DIRECTION_NORM_TOLERANCE:.1e}."
            )

        arrays = {
            "x_m": native["x"] * _GEOMETRY_M_PER_MM,
            "y_m": native["y"] * _GEOMETRY_M_PER_MM,
            "z_m": native["z"] * _GEOMETRY_M_PER_MM,
            "L": native["L"],
            "M": native["M"],
            "N": native["N"],
            "intensity": native["intensity"],
            "wavelength_m": native["wavelength_um"] * _WAVELENGTH_M_PER_UM,
            "opd_native": native["opd_native"],
            # The trace API exposes survivors only. This derived boolean states
            # the membership of each exported row; it is not an Optiland pupil mask.
            "survived": np.ones(native["x"].shape, dtype=np.bool_),
        }
        scientific_hash = _scientific_array_hash(arrays)
        summary_metrics = {
            "max_direction_norm_error": max_direction_norm_error,
            "direction_norm_tolerance": _DIRECTION_NORM_TOLERANCE,
            "all_finite": True,
            "intensity_min": float(np.min(native["intensity"])),
            "intensity_max": float(np.max(native["intensity"])),
            "intensity_sum": float(np.sum(native["intensity"])),
            "opd_native_min": float(np.min(native["opd_native"])),
            "opd_native_max": float(np.max(native["opd_native"])),
            "x_m_min": float(np.min(arrays["x_m"])),
            "x_m_max": float(np.max(arrays["x_m"])),
            "y_m_min": float(np.min(arrays["y_m"])),
            "y_m_max": float(np.max(arrays["y_m"])),
            "z_m_min": float(np.min(arrays["z_m"])),
            "z_m_max": float(np.max(arrays["z_m"])),
        }

        path = run_dir / "rays.npz"
        np.savez(path, **arrays)

        digest = hashlib.sha256(path.read_bytes()).hexdigest()

        return ArtifactRecord(
            id=f"{request.node_id}-rays-{uuid.uuid4().hex[:8]}",
            kind=ArtifactKind.RAY_BUNDLE,
            uri=str(path),
            sha256=digest,
            shape=tuple(native["x"].shape),
            dtype=str(native["x"].dtype),
            framework=Framework.PYTORCH if backend_name == "torch" else Framework.NUMPY,
            device=Device.CPU,
            units="SI for position and wavelength; dimensionless direction/intensity; native OPD",
            metadata={
                "length_unit": "m",
                "native_length_unit": "mm",
                "native_to_si_scale": _GEOMETRY_M_PER_MM,
                "wavelength_unit": "m",
                "native_wavelength_unit": "um",
                "native_wavelength_to_si_scale": _WAVELENGTH_M_PER_UM,
                "wavelength_m": float(arrays["wavelength_m"][0]),
                "coordinate_fields": ["x_m", "y_m", "z_m"],
                "direction_fields": ["L", "M", "N"],
                "intensity_field": "intensity",
                "intensity_is_not_amplitude": (
                    "RealRays.i is a real-valued per-ray intensity, not a "
                    "complex amplitude or Jones vector; no 'amplitude' or "
                    "'polarization' array is produced by this adapter (see "
                    "module docstring)."
                ),
                "requested_Hx": hx,
                "requested_Hy": hy,
                "requested_num_rays": num_rays,
                "traced_num_rays": int(native["x"].shape[0]),
                "survival_field": "survived",
                "survival_semantics": (
                    "Every exported row survived the sequential trace. Optic.trace "
                    "does not expose rejected input candidates, so invalid and "
                    "vignetted counts before survivor filtering are unavailable."
                ),
                "sample": sample_name,
                "backend": backend_name,
                "scientific_array_sha256": scientific_hash,
                "summary_metrics": summary_metrics,
                "conventions": {
                    "axes": "x,y,z right-handed Cartesian; propagation is +z",
                    "handedness": "right-handed",
                    "direction": "(L,M,N) direction cosines in the same frame",
                    "reference_plane": "final traced image surface, surface index 14",
                    "reference_plane_z_m": reference_plane_z_mm * _GEOMETRY_M_PER_MM,
                    "opd_field": "opd_native",
                    "opd_unit": "Optiland native value (geometry-scale mm expected)",
                    "opd_reference": "unverified",
                    "opd_sign": "unverified",
                    "polarization": "missing; RealRays provides no polarization state",
                    "coherence": "missing; sequential rays are not a coherent complex field",
                    "normalization": "raw Optiland ray intensity/weight; not normalized",
                    "sampling": (
                        "hexapolar pupil distribution; requested value is density, not output count"
                    ),
                },
            },
        )

    @staticmethod
    def _build_wavefront_artifact(
        request: ModelRunRequest,
        rays: Any,
        be_utils: Any,
        backend_name: str,
        sample_name: str,
        wavelength: float,
        run_dir: Path,
    ) -> tuple[ArtifactRecord, list[str]]:
        import numpy as np

        x = be_utils.to_numpy(rays.x) * _GEOMETRY_M_PER_MM
        y = be_utils.to_numpy(rays.y) * _GEOMETRY_M_PER_MM
        opd = be_utils.to_numpy(rays.opd)
        traced_wavelength = be_utils.to_numpy(rays.w) * _WAVELENGTH_M_PER_UM

        path = run_dir / "wavefront.npz"
        np.savez(path, x_m=x, y_m=y, opd_native=opd, wavelength_m=traced_wavelength)
        import hashlib

        digest = hashlib.sha256(path.read_bytes()).hexdigest()

        warnings = [
            "Output port 'wavefront' registry metadata declares "
            f"{_MISSING_WAVEFRONT_METADATA!r} but "
            "optiland.rays.real_rays.RealRays exposes neither a polarization "
            "state nor a pupil mask (only x, y, z, L, M, N, i, opd, w). These "
            "metadata keys are intentionally left unpopulated rather than "
            "fabricated (repository scientific-contract requirements)."
        ]

        artifact = ArtifactRecord(
            id=f"{request.node_id}-wavefront-{uuid.uuid4().hex[:8]}",
            kind=ArtifactKind.WAVEFRONT_SAMPLES,
            uri=str(path),
            sha256=digest,
            shape=tuple(x.shape),
            dtype=str(x.dtype),
            framework=Framework.PYTORCH if backend_name == "torch" else Framework.NUMPY,
            device=Device.CPU,
            units=None,
            metadata={
                "length_unit": "m",
                "wavelength_unit": "m",
                "wavelength": float(traced_wavelength[0]) if traced_wavelength.size else None,
                "coordinate_fields": ["x_m", "y_m"],
                "optical_path_length_source": (
                    "RealRays.opd -- convention not independently verified "
                    "(absolute optical path length vs. OPD relative to a "
                    "chief/reference ray); see conventions.md."
                ),
                "missing_declared_metadata": _MISSING_WAVEFRONT_METADATA,
                "sample": sample_name,
                "backend": backend_name,
            },
        )
        return artifact, warnings


def get_adapter() -> OptilandAdapter:
    return OptilandAdapter()
