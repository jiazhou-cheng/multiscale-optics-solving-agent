"""Adapter for ``M_RAY_OPTILAND`` (Optiland sequential ray tracer, pinned 0.6.0).

**Responsibility: the graph-facing ``ModelAdapter`` protocol, and nothing else.**
``spec``, ``estimate``, ``validate_request`` and ``run``. Everything ``run``
needs on the way through lives in a sibling with its own stated responsibility:

===================  =========================================================
``constants.py``     pinned values, supported sets, unit conversions
``requests.py``      the standalone request/result/failure contract, and
                     turning a config into an ``OpticalSystemSpec``
``execution.py``     backend/device/precision resolution, and the lazy import
``capability.py``    why a request is refused -- decided before anything runs
``pupil.py``         exit pupil, image space, reference-plane geometry
``artifacts.py``     building and persisting the two exported records
``provenance.py``    what ran, on what, and the array hash
``baseline.py``      the frozen M1 standalone contract
``builder.py``       canonical prescription -> Optiland system
``coherent_trace.py``  tracing a caller-supplied coherent ray population
===================  =========================================================

This file remains the address callers use: several names above are re-bound
here because tests patch them at this path, and a patch target is part of a
module's contract even when the name is private. The conventions below apply to
the integration as a whole, not only to this file.

Grounded in ``knowledge/solvers/optiland/`` (``card.yaml``,
``conventions.md``, ``usage_notes.md``, ``api_minimal_examples.md``,
``failure_guide.md``) and the probes/expected fixtures under that directory,
all captured against the ``agent_solver`` container on 2026-07-30. Nothing
below relies on training-data memory of an older/different Optiland API.

Scope (deliberately narrow -- see current Optiland scope "do not broaden scope
while a P0 model/coupler lacks tests")
------------------------------------------------------------------------
- Optical systems are built from canonical prescriptions
  (``core/optical_system.py``, schema ``optical-system-spec/1``) through the one
  generic builder in ``solvers/optiland/builder.py`` -- CHE-56 (PB5). A request
  either names a registered prescription via ``config["sample"]`` (default
  ``"ReverseTelephoto"``; the registry is ``registry/prescriptions.py``) or
  supplies one inline via ``config["prescription"]``, as an
  ``OpticalSystemSpec`` or its serialized mapping. There is no longer a
  bundled-sample construction path and a separate hand-written one.
  What remains refused is the optional ``system`` **input port**: it would
  carry an arbitrary Optiland object with no typed contract, and the canonical
  schema exists precisely so that is unnecessary. Prescription features outside
  the Phase 3 supported set (plane/spherical/even-asphere geometry, refractive
  and grating interactions, air/ideal/catalog materials, one stop, EPD
  aperture, angular fields, wavelengths with one primary) are rejected eagerly
  with a structured ``PrescriptionError`` rather than approximated.
- Device and precision are negotiated against Optiland's real capability
  declaration (``core/capabilities.py::OPTILAND_CAPABILITIES``) rather than
  compared with two string constants -- CHE-61 (PB4b), replacing CHE-55's
  blanket gate. ``float32``/``float64`` on ``cpu``, and either precision on
  ``cuda`` through the torch backend, are executable; the defaults are still
  ``cpu``/``float64``, so an existing request means exactly what it always
  meant. ``float16`` is refused because Optiland has no float16 path to promote
  into (``set_precision`` is typed ``Literal['float32','float64']``), and
  ``cuda`` on the numpy backend is refused because ``set_device`` raises
  ``BackendCapabilityError`` there. A ``cuda`` request in a container without a
  device fails with a distinct code from a ``cuda`` request that is malformed.
- Only one design-parameter path is supported for the differentiable
  (torch-backend) case: ``surfaces.surfaces[<index>].geometry.radius`` on the
  selected sample lens, matching
  ``benchmarks/probes/optiland/gradient_probe.py`` exactly. This is
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
- **The torch backend defaults to float32; the numpy backend defaults to
  float64.** Measured on the pinned install (``be.get_precision()`` returns 32
  after ``set_backend('torch')`` and 64 after ``set_backend('numpy')`` --
  ``benchmarks/probes/precision/default_precision.py``). Before CHE-61 this adapter never
  called ``set_precision`` at all while reporting ``dtype: 'float64'`` in its
  diagnostics, so **every torch-backend run traced in float32 under a float64
  label**. The recorded gradient-probe evidence in
  ``benchmarks/probes/records/optiland/gradient_probe.json`` is therefore the
  float32 path, and ``config['dtype']='float32'`` reproduces it bit-identically;
  the float64 default now genuinely runs float64 and differs from that record by
  1.3e-05 relative on the objective and 2.3e-06 on the gradient
  (``benchmarks/probes/precision/grad_precision.py``). ``set_precision`` is now called
  explicitly on every run, like ``set_backend``, so the label and the arithmetic
  agree.
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
  ``benchmarks/probes/records/optiland/gradient_probe.json``), looser than
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

Exception-handling convention (per the ``core/errors.py`` docstring)
--------------------------------------------------------------------
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

import tempfile
import time
from collections.abc import Mapping
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from core.arrays import array_state
from core.errors import (
    AdapterDependencyError,
    UnsupportedCapabilityError,
)
from core.graph import Severity, ValidationIssue, ValidationReport
from core.optical_system import (
    OPTICAL_SYSTEM_SPEC_VERSION,
    OpticalSystemSpec,
)
from core.precision import (
    ExecutionRequest,
)
from core.specs import ModelSpec
from solvers.base import (
    CostEstimate,
    ModelRunRequest,
    ModelRunResult,
    RunStatus,
)
from solvers.optiland.artifacts import build_ray_bundle_artifact, build_wavefront_artifact
from solvers.optiland.baseline import run_standalone as _run_standalone
from solvers.optiland.builder import build_optiland_system
from solvers.optiland.capability import capability_problems

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
from solvers.optiland.execution import (  # noqa: F401
    _apply_optiland_execution,
    _cuda_unavailable_reason,
    _import_optiland,
    _resolve_optiland_execution,
)
from solvers.optiland.provenance import (  # noqa: F401
    _cpu_device_name,
    _load_spec,
    _scientific_array_hash,
)
from solvers.optiland.pupil import (
    HandoffPlaneError,
    _resolve_exit_pupil,
    _resolve_image_space,
    _resolve_object_space_reference,
    _resolve_ray_pupil_sampling,
)
from solvers.optiland.requests import (
    OptilandRayRequest,
    OptilandRayResult,
    _prescription_from_config,
)


def _resolve_lens(spec: OpticalSystemSpec) -> Any:
    """Build the system through the one generic construction path."""
    return build_optiland_system(spec)




# --- The module's public surface -------------------------------------------
#
# CHE-91 moved these definitions into cohesive siblings, but
# `solvers.optiland.adapter` stays the address callers use. Two kinds of caller
# make that a contract rather than a convenience:
#
#   * imports -- `_resolve_lens` and `_scientific_array_hash` are imported from
#     here by benchmark probes and tests;
#   * `unittest.mock.patch` targets -- several tests patch
#     `solvers.optiland.adapter._import_optiland` and `._resolve_lens`, and a
#     patch target is part of a module's contract even when the name is private.
#
# Re-binding them here is what let the split land with no test edits, which is
# the standard CHE-91 holds a characterization refactor to. Anything listed
# below is load-bearing; deleting it because this file no longer *uses* it is
# the mistake this comment exists to prevent.

class OptilandAdapter:
    """``ModelAdapter`` for ``M_RAY_OPTILAND`` (Optiland 0.6.0). See module docstring."""

    @property
    def spec(self) -> ModelSpec:
        return _load_spec()


    def run_standalone(self, request: OptilandRayRequest | Mapping[str, Any]) -> OptilandRayResult:
        """The frozen M1 standalone baseline. Implementation in `baseline.py`.

        Kept as a method because that is how every caller reaches it.
        """
        return _run_standalone(self, request)

    def estimate(self, request: ModelRunRequest) -> CostEstimate:
        problems = capability_problems(request)
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
            for code, message in capability_problems(request)
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
        problems = capability_problems(request)
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
        # Already validated by _capability_problems, so this cannot raise here.
        prescription = _prescription_from_config(request.config)
        sample_name = prescription.name
        wavelength = float(request.config.get("wavelength", _DEFAULT_WAVELENGTH))
        hx = float(request.config.get("Hx", _DEFAULT_HX))
        hy = float(request.config.get("Hy", _DEFAULT_HY))
        num_rays = int(request.config.get("num_rays", _DEFAULT_NUM_RAYS))
        handoff_plane = str(request.config.get("handoff_plane", _DEFAULT_HANDOFF_PLANE))

        # Negotiated before any solver call. Cannot raise here: the same call
        # already ran inside _capability_problems above.
        resolved = _resolve_optiland_execution(request.config)
        requested_execution = ExecutionRequest.from_config(MODEL_ID, request.config)

        be, be_utils, torch_module = _import_optiland(need_torch=backend_name == "torch")

        try:
            package_version = version("optiland")
        except PackageNotFoundError:
            package_version = "unknown"

        # set_backend / set_precision / set_device all mutate global, process-wide
        # module state and none is thread-safe (conventions.md). All three are set
        # explicitly on every run, even at the defaults, so a previous run's
        # choices in this process can never silently leak in. This is also the
        # first time this adapter has driven the latter two at all: before CHE-61
        # it reported "cpu"/"float64" while leaving Optiland on whatever it
        # happened to be set to.
        applied_execution = _apply_optiland_execution(be, torch_module, resolved, backend_name)

        warnings: list[str] = [_OPD_WARNING]
        if backend_name == "numpy":
            warnings.append(
                "backend='numpy' (default): optiland.backend.supports_gradients "
                "is False for this run -- no gradients are available, "
                "regardless of the registry's derivative.mode=native_autodiff "
                "(that mode only applies once backend='torch' is selected)."
            )

        try:
            lens = _resolve_lens(prescription)

            design_parameter_tensors: dict[str, Any] = {}
            for name, value in request.design_parameters.items():
                match = _VALIDATED_DESIGN_PARAMETER_PATTERN.match(name)
                assert match is not None  # already enforced by _capability_problems
                surface_index = int(match.group(1))
                surface = lens.surfaces.surfaces[surface_index]
                if backend_name == "torch":
                    # The leaf must match the precision the trace runs in, or
                    # torch promotes mid-graph and the gradient is computed
                    # against a different number than the one that was set.
                    tensor = torch_module.tensor(
                        float(value),
                        dtype=getattr(torch_module, str(resolved.precision.real_dtype)),
                        requires_grad=True,
                        device=str(resolved.device),
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
            # CHE-56: which prescription actually built the system, and a digest
            # of its canonical normalization. Two runs that report the same
            # fingerprint built the same optical system.
            "prescription_spec_version": OPTICAL_SYSTEM_SPEC_VERSION,
            "prescription_fingerprint": prescription.fingerprint(),
            "prescription_source": (
                "config['prescription']" if request.config.get("prescription") is not None
                else "config['sample']"
            ),
            "requested_num_rays": num_rays,
            "Hx": hx,
            "Hy": hy,
            "wavelength_native_units": wavelength,
            "package_version": package_version,
            # Requested / resolved / actual, kept apart (PB4b section 12). The
            # flat "device"/"dtype" keys are retained for compatibility and now
            # carry the RESOLVED values rather than two hard-coded constants.
            "device": str(resolved.device),
            "cpu_device": _cpu_device_name(),
            "dtype": str(resolved.precision.real_dtype),
            "execution": {
                "requested": requested_execution.as_dict(),
                "resolved": resolved.as_dict(),
                "applied_to_optiland": applied_execution,
                # "actual" is filled in below, from the traced arrays themselves.
            },
            "seed": int(request.config.get("seed", _BASELINE_SEED)),
            "seed_semantics": (
                "recorded; Optiland hexapolar sampler is deterministic and uses no RNG"
            ),
        }

        # PB4b section 13: the observed placement, read off the traced arrays, and
        # a mismatch against the request made visible rather than reported as
        # success. Optiland has no equivalent of the JAX platform-pin hazard that
        # motivated the rule, but the rule is the same one and it is cheap to hold
        # here too.
        actual_state = array_state(rays.x)
        diagnostics["execution"]["actual"] = actual_state.as_dict()
        mismatches = []
        if actual_state.device.kind is not resolved.device.kind:
            mismatches.append(f"device resolved {resolved.device} but traced {actual_state.device}")
        if actual_state.dtype is not resolved.precision.real_dtype:
            mismatches.append(
                f"precision resolved {resolved.precision.real_dtype} "
                f"but traced {actual_state.dtype}"
            )
        diagnostics["execution"]["mismatches"] = mismatches
        if mismatches:
            warnings.append(
                "requested/resolved execution does not match what Optiland actually "
                f"produced: {'; '.join(mismatches)}. The artifact records the ACTUAL "
                "values; treat any downstream precision or device claim as the actual one."
            )

        if backend_name == "torch" and request.require_gradients:
            objective_tensor = (rays.x**2 + rays.y**2).mean()
            diagnostics["objective_description"] = (
                "mean(x^2 + y^2) over traced rays at the requested field point "
                "-- an RMS-spot-size proxy, the only design_parameter -> "
                "objective path characterized for this adapter "
                "(benchmarks/probes/optiland/gradient_probe.py)."
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
            image_plane_z_mm = float(be_utils.to_numpy(final_surface.geometry.cs.z))

            if handoff_plane == "exit_pupil":
                exit_pupil = _resolve_exit_pupil(lens, be_utils, image_plane_z_mm)
                reference_plane_z_mm = exit_pupil["z_mm"]
            else:
                exit_pupil = None
                reference_plane_z_mm = image_plane_z_mm

            import numpy as _np

            object_space_reference = _resolve_object_space_reference(
                lens,
                be,
                be_utils,
                hx=hx,
                hy=hy,
                wavelength_um=wavelength,
                num_rays=num_rays,
                traced_count=int(_np.asarray(be_utils.to_numpy(rays.x)).size),
            )
            ray_pupil_sampling = _resolve_ray_pupil_sampling(
                lens,
                be_utils,
                num_rays=num_rays,
                traced_count=int(_np.asarray(be_utils.to_numpy(rays.x)).size),
            )
            rays_artifact = build_ray_bundle_artifact(
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
                handoff_plane,
                exit_pupil,
                len(lens.surfaces.surfaces) - 1,
                _resolve_image_space(lens, be_utils, wavelength),
                object_space_reference,
                ray_pupil_sampling,
            )
            wavefront_artifact, wavefront_warnings = build_wavefront_artifact(
                request, rays, be_utils, backend_name, sample_name, wavelength, run_dir
            )
        except HandoffPlaneError as exc:
            # A plane that cannot be resolved is a structured failure, not a crash
            # and not a silent fallback to the image surface: a caller that asked
            # for the exit pupil and got the image plane back would be off by the
            # whole pupil-to-focus distance with nothing to notice it by.
            return ModelRunResult(
                status=RunStatus.FAILED,
                error_type=type(exc).__name__,
                error_message=str(exc),
                warnings=warnings,
                diagnostics={
                    **diagnostics,
                    "code": exc.code,
                    "stage": "handoff_plane_resolution",
                    "requested_handoff_plane": handoff_plane,
                },
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


def get_adapter() -> OptilandAdapter:
    return OptilandAdapter()
