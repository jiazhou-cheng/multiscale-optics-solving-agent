"""Adapter for ``M_RAY_OPTILAND`` (Optiland sequential ray tracer, pinned 0.6.0).

Grounded in ``knowledge/solvers/optiland/`` (``solver_card.yaml``,
``conventions.md``, ``capability_notes.md``, ``api_minimal_examples.md``,
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
- **The torch backend defaults to float32; the numpy backend defaults to
  float64.** Measured on the pinned install (``be.get_precision()`` returns 32
  after ``set_backend('torch')`` and 64 after ``set_backend('numpy')`` --
  ``benchmarks/probes/precision/default_precision.py``). Before CHE-61 this adapter never
  called ``set_precision`` at all while reporting ``dtype: 'float64'`` in its
  diagnostics, so **every torch-backend run traced in float32 under a float64
  label**. The recorded gradient-probe evidence in
  ``knowledge/solvers/optiland/expected/gradient_probe.json`` is therefore the
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

from core.arrays import array_state, dtype_of, numpy_dtype
from core.artifacts import ArtifactRecord
from core.capabilities import OPTILAND_CAPABILITIES
from core.errors import (
    AdapterDependencyError,
    UnsupportedCapabilityError,
)
from core.graph import Severity, ValidationIssue, ValidationReport
from core.optical_system import (
    OPTICAL_SYSTEM_SPEC_VERSION,
    OpticalSystemSpec,
    PrescriptionError,
)
from core.precision import (
    ArrayNamespace,
    CapabilityError,
    DeviceKind,
    DType,
    ExecutionRequest,
    ResolvedExecution,
)
from core.specs import ArtifactKind, Device, Framework, ModelSpec
from registry.loader import Registry
from registry.prescriptions import (
    prescription_names,
    resolve_prescription,
)
from solvers.base import (
    CostEstimate,
    ModelRunRequest,
    ModelRunResult,
    RunStatus,
)
from solvers.optiland.builder import build_optiland_system

MODEL_ID = "M_RAY_OPTILAND"

_SUPPORTED_BACKENDS = ("numpy", "torch")

# Since CHE-56 (PB5) every supported system -- bundled or adapter-owned -- is a
# canonical prescription in registry/prescriptions.py, built by the single
# generic builder in optiland_builder.py. There is no longer one construction
# path for bundled samples and another for adapter-owned ones, and the name
# `sample` survives only as the lookup key for a canonical prescription.
#
# CHE-32 (M3.3)'s reason for owning a prescription at all still holds: no
# bundled system qualifies as M3.2's diffraction-limited reference.
# tmp_probes/optiland_exit_pupil_probe.py measured every system in
# optiland.samples.objectives on axis at 550 nm and the best is WideAngle100FOV
# at 0.36 waves peak-to-valley, against Rayleigh's 0.25.
#
# What CHE-56 changes is that an unnamed prescription is no longer refused: a
# caller may pass a canonical `OpticalSystemSpec` (or its serialized mapping)
# through `config['prescription']`. What is still refused is an arbitrary
# Python Optiland object through the `system` input port -- that is not a
# validated contract, and the canonical schema exists precisely so it does not
# need to be.
_SUPPORTED_SAMPLES = prescription_names()

# M3-SINGLET-REF's numbers (frozen by M3.2 in benchmarks/slice_protocol.yaml)
# moved to the prescription itself, registry/prescriptions.py, so there is one
# definition rather than a copy here and a construction site there.

#: The DEFAULT device and precision, no longer the only supported ones (CHE-61).
#: Optiland's own API is `set_precision(Literal['float32','float64'])` and
#: `set_device(str)` (torch backend only, `BackendCapabilityError` otherwise), so
#: what this adapter may execute is declared once in
#: `core/capabilities.py::OPTILAND_CAPABILITIES` and validated from there. These
#: two constants remain the defaults, which is what keeps every existing request
#: -- and L1-RAY-01's recorded fingerprint -- byte-identical.
_DEFAULT_DEVICE = "cpu"
_DEFAULT_DTYPE = "float64"
_SUPPORTED_DEVICE = _DEFAULT_DEVICE
_SUPPORTED_DTYPE = _DEFAULT_DTYPE

#: float64 direction-norm bound, unchanged. A float32 trace cannot meet it and
#: must not be asked to: see `_direction_norm_tolerance`.
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

# CHE-32: which plane the exported rays are referenced to. The default stays
# "image_surface" so that L1-RAY-01's recorded scientific fingerprint
# (43dab1ee...) reproduces bit-identically -- M3.3 adds a plane, it does not move
# the existing one.
_DEFAULT_SAMPLE = "ReverseTelephoto"
_SUPPORTED_HANDOFF_PLANES = ("image_surface", "exit_pupil")
_DEFAULT_HANDOFF_PLANE = "image_surface"

_OPD_WARNING = (
    "RealRays.opd is preserved in Optiland-native values. CHE-30 established the "
    "convention -- absolute accumulated optical path in the geometry unit (mm), "
    "index-weighted, referenced to the ray launch state -- but for an infinite "
    "object that launch plane is aperture-dependent, so the exported value is a "
    "piston of order 1e4 waves plus the wavefront. It must not be read as a phase "
    "without subtracting a declared reference; see conventions.opd_reference."
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


_BACKEND_NAMESPACE = {
    "numpy": ArrayNamespace.NUMPY,
    "torch": ArrayNamespace.TORCH,
}


def _direction_norm_tolerance(dtype: DType) -> float:
    """Direction unit-norm bound appropriate to the precision actually traced in.

    The float64 constant (1e-12) stays exactly as it was. A float32 trace cannot
    satisfy it and it would be wrong to want it to: Optiland normalizes in
    float32, so ``|d| - 1`` sits at a few float32 epsilons before this adapter
    ever sees the rays. ``64 * eps`` is that round-off with headroom, derived
    rather than chosen, and it reduces to the historical value for float64.
    """
    import numpy as np

    eps = float(np.finfo(numpy_dtype(dtype)).eps)
    return max(_DIRECTION_NORM_TOLERANCE, 64.0 * eps)


def _resolve_optiland_execution(config: Mapping[str, Any]) -> ResolvedExecution:
    """Negotiate ``config['device'] / config['dtype'] / config['backend']``.

    One place, one capability table. Before CHE-61 this was two ``!=``
    comparisons against string constants, which is why the adapter reported
    "cpu"/"float64" whatever Optiland was actually told to do -- and it was never
    told anything, because ``set_device``/``set_precision`` were never called.

    ``config['backend']`` selects the array namespace, and that pairing is
    enforced rather than assumed: ``set_device`` raises
    ``BackendCapabilityError`` on the numpy backend, so ``device='cuda'`` with
    ``backend='numpy'`` is refused here instead of failing inside Optiland.
    """
    backend_name = str(config.get("backend", "numpy"))
    namespace = _BACKEND_NAMESPACE.get(backend_name)
    execution = ExecutionRequest.from_config(MODEL_ID, config)
    return OPTILAND_CAPABILITIES.resolve(
        ExecutionRequest(
            component=MODEL_ID,
            precision=execution.precision,
            device=execution.device,
            namespace=namespace,
            bridge_policy=execution.bridge_policy,
        )
    )


def _apply_optiland_execution(
    be: Any, torch_module: Any, resolved: ResolvedExecution, backend_name: str
) -> dict[str, Any]:
    """Drive Optiland's real execution controls and report what it actually did.

    ``set_backend``, ``set_precision`` and ``set_device`` are all process-global
    and none is thread-safe (Optiland's own docs). They are therefore set
    explicitly on every run -- never inherited from whatever a previous run in
    this process left behind -- exactly as ``set_backend`` already was.

    The return value is *observed*: ``get_device()``/``get_precision()`` read
    back from Optiland rather than echoing the request, because a request and a
    result are different facts and PB4a showed what happens when a project
    conflates them.
    """
    be.set_backend(backend_name)

    applied: dict[str, Any] = {
        "set_backend": backend_name,
        "set_precision": str(resolved.precision.real_dtype),
        "set_device": None,
    }
    # Optiland spells precision as a dtype name; the project spells it as a
    # policy. This is the one place the two vocabularies meet.
    be.set_precision(str(resolved.precision.real_dtype))

    if resolved.namespace is ArrayNamespace.TORCH:
        device_string = str(resolved.device)
        if resolved.device.kind is DeviceKind.CUDA:
            _require_cuda(torch_module, resolved)
        be.set_device(device_string)
        applied["set_device"] = device_string
        applied["get_device"] = str(be.get_device())
    else:
        # set_device raises BackendCapabilityError on the numpy backend, so it is
        # not called at all rather than called-and-caught.
        applied["get_device"] = "cpu (numpy backend has no device concept)"
    # get_precision returns an int width (32 / 64) on the pinned install, NOT the
    # dtype name set_precision takes -- an asymmetry worth normalizing here so the
    # diagnostics read in one vocabulary. Both spellings are accepted because the
    # setter's and getter's disagreement is exactly the kind of thing a minor
    # release changes, and a diagnostics field is not worth crashing a trace over.
    observed = be.get_precision()
    applied["get_precision_raw"] = observed
    applied["get_precision"] = (
        f"float{int(observed)}" if isinstance(observed, int) else str(observed)
    )
    return applied


def _cuda_unavailable_reason() -> str | None:
    """Why torch cannot reach a CUDA device here, or ``None`` if it can.

    Called only when a request actually asks for CUDA, so the default CPU path
    never pays for a torch import. Importing torch *is* how this question gets
    answered, which is why the eager gate can be import-free for every other
    request but not for this one -- and it is still eager, since it happens
    before any Optiland call.
    """
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - torch is pinned in both images
        return f"torch is not importable ({exc})"
    if torch.version.cuda is None or "+cpu" in torch.__version__:
        return (
            f"torch is a CPU-only build ({torch.__version__}); the default "
            "agent_solver image installs it from the CPU wheel index"
        )
    if not torch.cuda.is_available():
        return "torch.cuda.is_available() is False (no CUDA device attached to this container)"
    return None


def _require_cuda(torch_module: Any, resolved: ResolvedExecution) -> None:
    """Refuse a CUDA request the installed torch cannot actually serve.

    The default ``agent_solver`` image installs torch from the CPU-only wheel
    index, where ``torch.cuda.is_available()`` is ``False`` and
    ``be.set_device('cuda')`` would either raise from deep inside torch or -- on
    a half-provisioned image -- succeed and then fail at the first kernel. Both
    are worse than a named capability error here.
    """
    if torch_module is None:  # pragma: no cover - guarded by _import_optiland
        raise CapabilityError(
            code="OPTILAND_CUDA_REQUIRES_TORCH",
            component=MODEL_ID,
            message="a CUDA request needs the torch backend, which was not imported.",
            requested=resolved.device,
        )
    if not torch_module.cuda.is_available():
        raise CapabilityError(
            code="OPTILAND_CUDA_UNAVAILABLE",
            component=MODEL_ID,
            message=(
                f"config['device']={str(resolved.device)!r} was requested but this "
                f"torch install cannot reach a CUDA device (torch "
                f"{torch_module.__version__}, torch.version.cuda="
                f"{torch_module.version.cuda!r}, torch.cuda.is_available()=False)."
            ),
            requested=resolved.device,
            supported=["cpu"],
            evidence="docker/Dockerfile installs torch from the CPU-only wheel index",
            remedy=(
                "Run in the CUDA image: `./run.sh --gpu ...` (see "
                "docs/testing/gpu_environment.md). There is deliberately no "
                "silent fallback to the CPU."
            ),
        )


def _host_array(be_utils: Any, value: Any, *, dtype: Any = None) -> Any:
    """Copy solver data to the host for persistence, preserving its precision.

    This replaces ``np.asarray(be_utils.to_numpy(x), dtype=np.float64)``, which
    did three separable things in one expression: a device-to-host transfer, a
    precision force, and (for a torch tensor) an autodiff graph break. Only the
    first is actually required to write a ``.npz``.

    Dropping the ``dtype=np.float64`` is a no-op for the default numpy/float64
    path -- the arrays already are float64, so L1-RAY-01's recorded fingerprint
    is unchanged -- and is what lets a float32 trace be persisted as float32
    rather than silently gaining ten digits it never computed.

    ``dtype`` is still available for the few quantities that are deliberately
    computed at reference precision regardless of the trace; each such call site
    says why.
    """
    import numpy as np

    host = np.asarray(be_utils.to_numpy(value))
    return host if dtype is None else host.astype(dtype)


def _import_optiland(*, need_torch: bool) -> tuple[Any, Any, Any]:
    """Lazily import optiland (and torch, only if requested).

    Never called at module import time -- only from run()/estimate(). Returns
    (optiland.backend, optiland.backend.utils, torch-module-or-None).

    ``optiland.samples`` is deliberately absent since CHE-56: system
    construction goes through the canonical prescription registry and the
    generic builder, which imports the construction API it needs itself. This
    function covers the backend state and array-conversion surface that the
    trace/export path uses.
    """
    try:
        import optiland.backend as be  # type: ignore[import-untyped]
        import optiland.backend.utils as be_utils  # type: ignore[import-untyped]
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

    return be, be_utils, torch_module


class HandoffPlaneError(RuntimeError):
    """The requested handoff plane could not be resolved from the system.

    Carried as an exception rather than a sentinel so the caller cannot mistake
    an unresolved plane for one at z = 0. `run()` converts it to a structured
    failure; it is never allowed to reach the export.
    """

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _prescription_from_config(config: Mapping[str, Any]) -> OpticalSystemSpec:
    """The canonical prescription this request names, or supplies inline.

    ``config['prescription']`` accepts either an :class:`OpticalSystemSpec` or a
    serialized mapping, which is parsed through
    :meth:`OpticalSystemSpec.from_dict` so its schema version is checked rather
    than assumed. ``config['sample']`` names one of the registered canonical
    prescriptions. Supplying both is a conflict, not a precedence question, and
    is rejected.
    """
    inline = config.get("prescription")
    sample_name = config.get("sample")
    if inline is not None and sample_name is not None:
        raise PrescriptionError(
            "PRESCRIPTION_CONFLICTING_SOURCES",
            "config['prescription'] and config['sample'] both name a system",
            path="config",
            expected=(
                "exactly one of config['prescription'] (an inline canonical "
                "prescription) or config['sample'] (a registered prescription name)"
            ),
        )
    if inline is not None:
        if isinstance(inline, OpticalSystemSpec):
            return inline
        return OpticalSystemSpec.from_dict(inline)
    return resolve_prescription(str(sample_name or _DEFAULT_SAMPLE))


def _resolve_lens(spec: OpticalSystemSpec) -> Any:
    """Build the system through the one generic construction path."""
    return build_optiland_system(spec)


def _resolve_exit_pupil(lens: Any, be_utils: Any, image_plane_z_mm: float) -> dict[str, Any]:
    """Read the exit pupil from the system, and say what the reading means.

    `Paraxial.XPL()` is signed and measured **from the image surface**, not from
    the global origin, so the plane is `image_z + XPL`
    (tmp_probes/optiland_exit_pupil_probe.py).

    The pupil is frequently *virtual* -- on `ReverseTelephoto` it lands at
    z = 2.15 mm with five refracting surfaces beyond it. That does not make the
    plane wrong, but it does change what a position at that plane is, and the
    returned metadata says so rather than leaving a reader to assume.
    """
    import numpy as np

    try:
        location_from_image_mm = float(
            np.asarray(be_utils.to_numpy(lens.paraxial.XPL())).ravel()[0]
        )
        diameter_mm = float(np.asarray(be_utils.to_numpy(lens.paraxial.XPD())).ravel()[0])
    except Exception as exc:
        raise HandoffPlaneError(
            "OPTILAND_EXIT_PUPIL_UNRESOLVED",
            "config['handoff_plane']='exit_pupil' was requested but Optiland's "
            f"paraxial solver could not supply XPL()/XPD(): {type(exc).__name__}: {exc}. "
            "The plane is read from the system, never guessed, so the run fails here "
            "rather than exporting rays against an invented reference.",
        ) from exc

    if not (np.isfinite(location_from_image_mm) and np.isfinite(diameter_mm)):
        raise HandoffPlaneError(
            "OPTILAND_EXIT_PUPIL_UNRESOLVED",
            "Optiland returned a non-finite exit pupil for this system "
            f"(XPL={location_from_image_mm!r}, XPD={diameter_mm!r}); a telecentric or "
            "degenerate configuration has no finite exit pupil plane, and this "
            "adapter will not substitute one.",
        )

    pupil_z_mm = image_plane_z_mm + location_from_image_mm
    surface_z = [
        float(np.asarray(be_utils.to_numpy(surface.geometry.cs.z)).ravel()[0])
        for surface in lens.surfaces.surfaces[:-1]
    ]
    beyond = [z for z in surface_z if np.isfinite(z) and z > pupil_z_mm]

    return {
        "z_mm": pupil_z_mm,
        "location_from_image_mm": location_from_image_mm,
        "diameter_mm": diameter_mm,
        "is_virtual": bool(beyond),
        "refracting_surfaces_beyond_pupil_z_mm": beyond,
    }


def _resolve_image_space(lens: Any, be_utils: Any, wavelength_um: float) -> dict[str, Any]:
    """Read the three facts a downstream OPL declaration must not assume.

    CHE-33 needs the image-space index to move an optical path between the traced
    image surface and any other plane in image space, and needs the entrance pupil
    diameter because CHE-30 showed the OPL zero of an infinite-object system moves
    with the aperture.

    Every value is read from the prescription and is ``None`` when the installed
    package does not expose it. It is not defaulted: ``n = 1`` happens to hold for
    both M3 systems, and a silent 1.0 for a system with a cover glass or an
    immersion medium would be a wrong optical path with nothing to notice it by.
    """
    import numpy as np

    def _scalar(thunk: Any) -> float | None:
        # Everything, including the attribute lookup, happens inside the guard:
        # this runs against fake lenses in the adapter's own failure tests, and a
        # missing attribute must degrade to "not available" rather than turn an
        # unrelated structured failure into a crash.
        try:
            value = float(np.asarray(be_utils.to_numpy(thunk())).ravel()[0])
        except Exception:
            return None
        return value if np.isfinite(value) else None

    index = _scalar(lambda: lens.surfaces.surfaces[-1].material_pre.n(wavelength_um))
    entrance_pupil_diameter_mm = _scalar(lambda: lens.paraxial.EPD())

    try:
        object_at_infinity: bool | None = bool(lens.object_surface.is_infinite)
    except Exception:
        object_at_infinity = None

    return {
        "image_space_refractive_index": index,
        "entrance_pupil_diameter_m": (
            entrance_pupil_diameter_mm * _GEOMETRY_M_PER_MM
            if entrance_pupil_diameter_mm is not None
            else None
        ),
        "object_at_infinity": object_at_infinity,
    }


def _project_rays_to_plane(rays: Any, be_utils: Any, target_z_mm: float) -> dict[str, Any]:
    """Advance each ray along its own image-space direction to `target_z_mm`.

    Returns the ray's image-space **asymptote** at that plane, which is what the
    exit pupil is defined by, not a physical intersection: for a virtual pupil the
    line being extended passes back through glass the ray never travelled in that
    state. See the probe for why that is nonetheless the right construction.

    Directions are unchanged -- this is a reparameterization along each ray, not a
    propagation, so no OPL is added or removed here. OPL is M3.4's.
    """
    import numpy as np

    # Precision preserved: this projection feeds the EXPORTED exit-pupil
    # positions, so widening here would make the exit_pupil handoff plane report
    # float64 for a float32 trace while image_surface reported float32.
    x = _host_array(be_utils, rays.x)
    y = _host_array(be_utils, rays.y)
    z = _host_array(be_utils, rays.z)
    direction_z = _host_array(be_utils, rays.N)

    if np.any(direction_z == 0.0):
        raise HandoffPlaneError(
            "OPTILAND_HANDOFF_PLANE_UNREACHABLE",
            "at least one traced ray has N = 0 and therefore never reaches the "
            "requested handoff plane; the projection is undefined for it.",
        )

    step_mm = (target_z_mm - z) / direction_z
    return {
        "x_mm": x + _host_array(be_utils, rays.L) * step_mm,
        "y_mm": y + _host_array(be_utils, rays.M) * step_mm,
        "z_mm": np.full_like(x, target_z_mm),
        "max_abs_step_mm": float(np.max(np.abs(step_mm))),
    }


def _resolve_object_space_reference(
    lens: Any,
    be: Any,
    be_utils: Any,
    *,
    hx: float,
    hy: float,
    wavelength_um: float,
    num_rays: int,
    traced_count: int,
) -> dict[str, Any]:
    """The optical path from an incoming *wavefront* to each ray's launch point.

    CHE-41. ``RealRays.opd`` is seeded to zero at the launch state, and for an
    object at infinity ``angle.py`` launches every ray on one plane
    **perpendicular to z** at ``positions[1] - (EPD - min(positions[1:-1]))``.
    A plane perpendicular to z is a wavefront only for a bundle travelling along
    z. For a bundle tilted by ``theta`` the two surfaces differ by
    ``n_object * (d0 . r_launch)``, which is *linear in the launch coordinate* --
    a tilt, not a piston. CHE-30 characterized the same launch plane and recorded
    only the piston consequence, because on axis that is all there is.

    Nothing is corrected here. The term is measured from the launch state and
    exported so that a *consumer* can declare its reference; the accumulated
    ``opd_native`` is left exactly as Optiland produced it.

    The launch state is regenerated through the public entry point
    ``ray_tracer.ray_generator.generate_rays`` over the same hexapolar
    distribution ``Optic.trace`` builds, which is the only reason this is
    possible at all: ``Optic.trace`` returns the traced rays and keeps no record
    of where they started.

    Every precondition the term depends on is *checked rather than assumed*, and
    a failed check returns ``available=False`` with the reason. It never returns
    a term it could not verify: an unavailable term is a structured refusal
    downstream, and a wrong one is a wavefront aimed at the wrong image point.
    """
    import numpy as np

    def unavailable(reason: str) -> dict[str, Any]:
        return {
            "available": False,
            "unavailable_reason": reason,
            "offset_native": None,
        }

    try:
        object_at_infinity = bool(lens.object_surface.is_infinite)
    except Exception as exc:  # pragma: no cover - defensive
        return unavailable(
            f"the object surface could not be read ({type(exc).__name__}), so the "
            "launch geometry is unknown"
        )
    if not object_at_infinity:
        return unavailable(
            "the object is at a finite distance, so the launch state is a POINT "
            "rather than a plane. A point source is already a common wavefront and "
            "the term would be zero -- but no system in this repository exercises "
            "that path, and an untested zero is still an untested claim."
        )

    try:
        from optiland.distribution import create_distribution

        distribution = create_distribution("hexapolar")
        distribution.generate_points(num_rays)
        pupil_x = distribution.x
        pupil_y = distribution.y
        pupil_points = int(np.asarray(be_utils.to_numpy(pupil_x)).size)
        field_x = be.atleast_1d(be.array(float(hx)))
        field_y = be.atleast_1d(be.array(float(hy)))
        launch = lens.ray_tracer.ray_generator.generate_rays(
            be.repeat(field_x, pupil_points),
            be.repeat(field_y, pupil_points),
            pupil_x,
            pupil_y,
            wavelength_um,
        )
    except Exception as exc:
        return unavailable(
            "the launch state could not be regenerated from "
            f"ray_tracer.ray_generator.generate_rays ({type(exc).__name__}: {exc}); "
            "Optic.trace does not retain it, so there is nothing to measure the "
            "object-space reference from"
        )

    def to_array(value: Any) -> Any:
        # Deliberately float64, and NOT the trace's precision: this is a
        # regenerated launch state used to compute the object-space OPL
        # reference, which is a piston-and-tilt correction of order 1e4 waves.
        # Computing that reference in float32 would inject an error larger than
        # the wavefront it corrects. Declared here rather than inherited.
        return np.asarray(be_utils.to_numpy(value), dtype=np.float64)

    x0, y0, z0 = to_array(launch.x), to_array(launch.y), to_array(launch.z)
    l0, m0, n0 = to_array(launch.L), to_array(launch.M), to_array(launch.N)

    if x0.size != traced_count:
        return unavailable(
            f"the regenerated launch state has {x0.size} rays but the trace exported "
            f"{traced_count}; the two cannot be matched row for row"
        )
    if not (
        np.all(np.isfinite(x0))
        and np.all(np.isfinite(y0))
        and np.all(np.isfinite(z0))
        and np.all(np.isfinite(l0))
        and np.all(np.isfinite(m0))
        and np.all(np.isfinite(n0))
    ):
        return unavailable("the regenerated launch state is not finite")

    direction_spread = max(float(np.ptp(l0)), float(np.ptp(m0)), float(np.ptp(n0)))
    plane_spread = float(np.ptp(z0))
    if direction_spread > _DIRECTION_NORM_TOLERANCE:
        return unavailable(
            "the launch directions are not common to every ray (spread "
            f"{direction_spread:.3e}), so the incoming bundle is not collimated and "
            "a single plane wavefront does not describe it"
        )
    if plane_spread > 0.0:
        return unavailable(
            f"the launch points do not lie on one plane (z spread {plane_spread:.3e} "
            "in native units), so the seeded reference surface is not the plane this "
            "term assumes"
        )

    index = None
    try:
        index = float(
            np.asarray(
                be_utils.to_numpy(lens.surfaces.surfaces[0].material_post.n(wavelength_um))
            ).ravel()[0]
        )
    except Exception:
        index = None
    if index is None or not np.isfinite(index) or index <= 0.0:
        return unavailable(
            "the object-space refractive index could not be read from the "
            "prescription, and the optical path from a wavefront to the launch "
            "plane is index-weighted"
        )

    # d0 . r_launch, index-weighted. The N0 * z0 part is common to every ray
    # because the launch plane is flat; it is retained rather than dropped so the
    # exported quantity is the optical path from ONE stated wavefront (the one
    # through the global origin, perpendicular to d0) rather than from an
    # unstated one.
    offset_native = index * (l0 * x0 + m0 * y0 + n0 * z0)

    return {
        "available": True,
        "unavailable_reason": None,
        "offset_native": offset_native,
        "launch_x_native": x0,
        "launch_y_native": y0,
        "launch_z_native": z0,
        "launch_direction": [float(l0[0]), float(m0[0]), float(n0[0])],
        "launch_plane_z_native": float(z0[0]),
        "object_space_refractive_index": index,
        "span_native": float(np.ptp(offset_native)),
    }


def _resolve_ray_pupil_sampling(
    lens: Any,
    be_utils: Any,
    *,
    num_rays: int,
    traced_count: int,
) -> dict[str, Any]:
    """The raw hexapolar pupil coordinates CHE-47's quadrature weight needs.

    CHE-38 found that the wavelet sum's dominant sensor-plane residual is a
    per-ray quadrature-weight error, not a kernel defect (section 14/15), and
    CHE-47 is the ticket that supplies the weight. Computing it needs to know
    which pupil ring each traced ray came from, and ``Optic.trace`` keeps no
    record of that -- so the same hexapolar distribution is regenerated from
    ``optiland.distribution.create_distribution`` and matched row for row
    against the trace, exactly as :func:`_resolve_object_space_reference`
    regenerates the launch state above.

    This function returns only the RAW normalized pupil coordinates, the ring
    count, and the aperture radius -- not a ring index or a weight. The actual
    quadrature math (`couplers.quadrature`) is coupler
    physics, not adapter physics, and this module must import no coupler: the
    M1 independence check (`benchmarks/level1/L1-RAY-01`) asserts that tracing
    a ray bundle loads no `couplers.*` module, and an
    import here would violate that for every caller of this adapter, not only
    CHE-47's. `optiland_handoff.py` (already coupler-side) computes the ring
    index and area weight from what this function returns.

    Every precondition is checked rather than assumed, exactly as CHE-41's
    object-space term is: a row-count mismatch (a vignetted ray, so the
    regenerated pupil no longer lines up one-to-one with the traced set) or an
    unreadable aperture diameter returns ``available=False`` with the reason,
    never a fabricated value.
    """
    import numpy as np

    def unavailable(reason: str) -> dict[str, Any]:
        return {
            "available": False,
            "unavailable_reason": reason,
            "pupil_x": None,
            "pupil_y": None,
        }

    try:
        from optiland.distribution import create_distribution

        distribution = create_distribution("hexapolar")
        distribution.generate_points(num_rays)
        # Also a float64 reference by declaration: these are regenerated
        # normalized pupil coordinates used to assign hexapolar RING INDICES by
        # comparing r against j / num_rings (CHE-47). That comparison is a
        # tolerance test on a ratio, so it is computed at reference precision
        # independently of what the trace ran in.
        pupil_x = np.asarray(be_utils.to_numpy(distribution.x), dtype=np.float64)
        pupil_y = np.asarray(be_utils.to_numpy(distribution.y), dtype=np.float64)
    except Exception as exc:
        return unavailable(
            "the hexapolar pupil sampling could not be regenerated from "
            f"optiland.distribution.create_distribution ({type(exc).__name__}: {exc})"
        )

    if pupil_x.size != traced_count:
        return unavailable(
            f"the regenerated pupil sampling has {pupil_x.size} points but the trace "
            f"exported {traced_count}; at least one ray was vignetted (or num_rays did "
            "not request a hexapolar fan), so a ring index cannot be assigned row for row"
        )

    try:
        epd_mm = float(np.asarray(be_utils.to_numpy(lens.paraxial.EPD())).ravel()[0])
    except Exception as exc:
        return unavailable(
            f"the entrance pupil diameter could not be read ({type(exc).__name__}: {exc}), "
            "so the physical aperture area a quadrature weight scales to is unknown"
        )
    if not np.isfinite(epd_mm) or epd_mm <= 0.0:
        return unavailable(f"entrance pupil diameter is not a positive finite value ({epd_mm!r})")
    aperture_radius_m = (epd_mm / 2.0) * _GEOMETRY_M_PER_MM

    return {
        "available": True,
        "unavailable_reason": None,
        "pupil_x": pupil_x,
        "pupil_y": pupil_y,
        "num_rings": num_rays,
        "aperture_radius_m": aperture_radius_m,
    }


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
                handoff_plane,
                exit_pupil,
                len(lens.surfaces.surfaces) - 1,
                _resolve_image_space(lens, be_utils, wavelength),
                object_space_reference,
                ray_pupil_sampling,
            )
            wavefront_artifact, wavefront_warnings = self._build_wavefront_artifact(
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
        handoff_plane: str = _DEFAULT_HANDOFF_PLANE,
        exit_pupil: dict[str, Any] | None = None,
        image_surface_index: int = 14,
        image_space: dict[str, Any] | None = None,
        object_space_reference: dict[str, Any] | None = None,
        ray_pupil_sampling: dict[str, Any] | None = None,
    ) -> ArtifactRecord:
        import numpy as np

        image_space = image_space or {}
        image_space_refractive_index = image_space.get("image_space_refractive_index")
        entrance_pupil_diameter_m = image_space.get("entrance_pupil_diameter_m")
        object_at_infinity = image_space.get("object_at_infinity")

        # The explicit execution -> serialization boundary for the trace. The
        # dtype the solver produced is preserved: forcing float64 here was the
        # single line that made a float32 or GPU trace indistinguishable from a
        # float64 host one downstream. For the default numpy/float64 path this is
        # a no-op, so L1-RAY-01's recorded fingerprint is unchanged.
        traced_state = array_state(rays.x)
        native = {
            name: _host_array(be_utils, value)
            for name, value in (
                ("x", rays.x),
                ("y", rays.y),
                ("z", rays.z),
                ("L", rays.L),
                ("M", rays.M),
                ("N", rays.N),
                ("intensity", rays.i),
                ("wavelength_um", rays.w),
                ("opd_native", rays.opd),
            )
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
        # Bound scaled to the precision the trace ACTUALLY ran in. The float64
        # value is untouched; a float32 trace is held to float32 round-off
        # instead of being failed for arithmetic it never claimed to do.
        norm_tolerance = _direction_norm_tolerance(dtype_of(native["L"]))
        if backend_name == "numpy" and max_direction_norm_error > norm_tolerance:
            raise ValueError(
                "Optiland direction vectors are not unit norm: "
                f"max error {max_direction_norm_error:.17g} exceeds "
                f"{norm_tolerance:.1e} for {dtype_of(native['L'])}."
            )

        # At the image surface the exported coordinates are the traced ones,
        # untouched -- which is what keeps L1-RAY-01's fingerprint bit-identical.
        # At the exit pupil they are each ray's image-space asymptote evaluated at
        # the pupil plane; directions are unchanged either way.
        if handoff_plane == "exit_pupil":
            projected = _project_rays_to_plane(rays, be_utils, reference_plane_z_mm)
            position = {
                "x": projected["x_mm"],
                "y": projected["y_mm"],
                "z": projected["z_mm"],
            }
            max_projection_step_mm = projected["max_abs_step_mm"]
        else:
            position = {"x": native["x"], "y": native["y"], "z": native["z"]}
            max_projection_step_mm = 0.0

        arrays = {
            "x_m": position["x"] * _GEOMETRY_M_PER_MM,
            "y_m": position["y"] * _GEOMETRY_M_PER_MM,
            "z_m": position["z"] * _GEOMETRY_M_PER_MM,
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
        # CHE-41. The object-space reference travels in the SAME file but under a
        # separate hash, and `arrays` is left untouched: `scientific_array_sha256`
        # is the frozen identity of the traced ray set (CHE-32 pinned
        # e494af41... on M3-SINGLET-REF, L1-RAY-01's 43dab1ee... downstream of the
        # same function), and adding a column to it would move a fingerprint that
        # nothing about this change has any business moving. The new arrays are
        # additional *object-space* data, not a revision of the traced output.
        object_space = object_space_reference or {"available": False}
        object_space_arrays: dict[str, Any] = {}
        if object_space.get("available"):
            object_space_arrays = {
                "object_space_reference_offset_m": (
                    np.asarray(object_space["offset_native"], dtype=np.float64) * _GEOMETRY_M_PER_MM
                ),
                "launch_x_m": (
                    np.asarray(object_space["launch_x_native"], dtype=np.float64)
                    * _GEOMETRY_M_PER_MM
                ),
                "launch_y_m": (
                    np.asarray(object_space["launch_y_native"], dtype=np.float64)
                    * _GEOMETRY_M_PER_MM
                ),
                "launch_z_m": (
                    np.asarray(object_space["launch_z_native"], dtype=np.float64)
                    * _GEOMETRY_M_PER_MM
                ),
            }
            if object_space_arrays["object_space_reference_offset_m"].shape != arrays["x_m"].shape:
                raise ValueError(
                    "the object-space reference term does not match the exported ray "
                    f"count: {object_space_arrays['object_space_reference_offset_m'].shape} "
                    f"vs {arrays['x_m'].shape}."
                )

        # CHE-47 (M3.9R extension). Same reasoning as the object-space block above:
        # additional per-ray columns, not a revision of the traced set, so they do
        # not enter `scientific_hash` below. Raw pupil coordinates only -- the ring
        # index and area weight are coupler physics, computed downstream in
        # optiland_handoff.py, never here (see _resolve_ray_pupil_sampling).
        quadrature = ray_pupil_sampling or {"available": False}
        quadrature_arrays: dict[str, Any] = {}
        if quadrature.get("available"):
            quadrature_arrays = {
                "pupil_normalized_x": np.asarray(quadrature["pupil_x"], dtype=np.float64),
                "pupil_normalized_y": np.asarray(quadrature["pupil_y"], dtype=np.float64),
            }
            if quadrature_arrays["pupil_normalized_x"].shape != arrays["x_m"].shape:
                raise ValueError(
                    "the regenerated pupil sampling does not match the exported ray "
                    f"count: {quadrature_arrays['pupil_normalized_x'].shape} vs "
                    f"{arrays['x_m'].shape}."
                )

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
        np.savez(path, **arrays, **object_space_arrays, **quadrature_arrays)

        digest = hashlib.sha256(path.read_bytes()).hexdigest()

        return ArtifactRecord(
            id=f"{request.node_id}-rays-{uuid.uuid4().hex[:8]}",
            kind=ArtifactKind.RAY_BUNDLE,
            uri=str(path),
            sha256=digest,
            shape=tuple(native["x"].shape),
            dtype=str(native["x"].dtype),
            framework=Framework.PYTORCH if backend_name == "torch" else Framework.NUMPY,
            # OBSERVED, from the traced tensors, not from the request. A torch
            # trace pinned to cuda:0 says so here; before CHE-61 this was
            # hard-coded to CPU and a GPU trace was indistinguishable from a host
            # one in the record it produced.
            device=traced_state.device.to_spec_device(),
            units="SI for position and wavelength; dimensionless direction/intensity; native OPD",
            metadata={
                # Where the trace actually executed, and the fact that the .npz
                # alongside it is an explicit persistence copy rather than
                # evidence that the computation happened on the host.
                "execution": traced_state.as_dict(),
                "serialization": {
                    "boundary": "explicit_persistence",
                    "host_copy": traced_state.namespace is not ArrayNamespace.NUMPY,
                    "kind": (
                        "serialization"
                        if traced_state.namespace is not ArrayNamespace.NUMPY
                        else "already_on_host"
                    ),
                    "reason": "npz persistence requires host bytes",
                    "mechanism": (
                        "optiland.backend.utils.to_numpy, which calls "
                        "tensor.detach().cpu().numpy() -- a host transfer AND an "
                        "autodiff graph break, both confined to this boundary"
                    ),
                },
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
                # CHE-32: M3.3 must state how the pupil boundary is represented,
                # and the answer is that Optiland does not represent it. RealRays
                # carries no mask (probed: no `mask`/`pupil_mask`/`vignetted`
                # attribute) and Optic.trace returns survivors only, so the
                # boundary is implicit in WHICH rows exist. The measured extent
                # below is derived from the survivors; it is emphatically not an
                # Optiland pupil mask, and `survived` -- an all-true derived
                # boolean -- must not be promoted into one.
                "pupil_boundary": {
                    "representation": "implicit_in_surviving_rays",
                    "mask_available_from_optiland": False,
                    "derived_from": "measured extent of the traced survivors",
                    "measured_semi_extent_x_m": float(np.max(np.abs(arrays["x_m"]))),
                    "measured_semi_extent_y_m": float(np.max(np.abs(arrays["y_m"]))),
                    "paraxial_semi_diameter_m": (
                        exit_pupil["diameter_mm"] / 2.0 * _GEOMETRY_M_PER_MM
                        if exit_pupil is not None
                        else None
                    ),
                    "warning": (
                        "the measured extent is a property of the traced set, not of "
                        "the aperture, and it does not bracket the paraxial diameter "
                        "in a predictable direction: sampling density pulls it inward "
                        "while pupil aberration pushes real marginal rays outward, and "
                        "on both M3 systems the measured value lands slightly ABOVE "
                        "the paraxial semi-diameter. A consumer needing the aperture "
                        "must use the paraxial diameter, not this."
                    ),
                },
                "sample": sample_name,
                "backend": backend_name,
                "scientific_array_sha256": scientific_hash,
                "summary_metrics": summary_metrics,
                "conventions": {
                    "axes": "x,y,z right-handed Cartesian; propagation is +z",
                    "handedness": "right-handed",
                    "direction": "(L,M,N) direction cosines in the same frame",
                    "reference_plane": (
                        f"exit pupil, read from Paraxial.XPL()/XPD() "
                        f"({'virtual' if exit_pupil and exit_pupil['is_virtual'] else 'real'})"
                        if handoff_plane == "exit_pupil"
                        else (f"final traced image surface, surface index {image_surface_index}")
                    ),
                    "reference_plane_z_m": reference_plane_z_mm * _GEOMETRY_M_PER_MM,
                    "handoff_plane": handoff_plane,
                    "exit_pupil": (
                        {
                            "source": "optic.paraxial.XPL() and XPD(), read not constructed",
                            "location_from_image_m": (
                                exit_pupil["location_from_image_mm"] * _GEOMETRY_M_PER_MM
                            ),
                            "z_m": exit_pupil["z_mm"] * _GEOMETRY_M_PER_MM,
                            "diameter_m": exit_pupil["diameter_mm"] * _GEOMETRY_M_PER_MM,
                            "is_virtual": exit_pupil["is_virtual"],
                            "refracting_surfaces_beyond_pupil_z_m": [
                                z * _GEOMETRY_M_PER_MM
                                for z in exit_pupil["refracting_surfaces_beyond_pupil_z_mm"]
                            ],
                            "position_semantics": (
                                "x_m/y_m are each ray's IMAGE-SPACE ASYMPTOTE evaluated "
                                "at the pupil plane, not a physical intersection. The "
                                "exit pupil is the image of the stop in image space and "
                                "is frequently virtual -- when is_virtual is true the "
                                "extended line passes back through glass the ray never "
                                "travelled in that state. This is the construction the "
                                "exit pupil is defined by, and it is what a "
                                "wavefront-over-the-pupil calculation wants, but it is "
                                "not 'where the ray is'."
                            ),
                            "max_projection_step_m": (max_projection_step_mm * _GEOMETRY_M_PER_MM),
                            "directions_unchanged": (
                                "the projection is a reparameterization along each ray, "
                                "so (L,M,N) are the traced values and no optical path "
                                "was added or removed; OPL handling is M3.4's (CHE-33)"
                            ),
                        }
                        if exit_pupil is not None
                        else None
                    ),
                    "opd_field": "opd_native",
                    "opd_unit": "Optiland native value (geometry-scale mm expected)",
                    # CHE-30 established all four parts of this convention against
                    # manufactured geometries with closed-form answers, each with the
                    # competing hypothesis it rules out; M1's "unverified" is
                    # superseded. Declaring it here does NOT admit opd_native as an
                    # optical path length: the contract layer still refuses a bundle
                    # whose OPL was never declared, because the value carried below is
                    # absolute and its zero moves with the aperture.
                    "opd_reference": (
                        "ray launch state, where RealRays seeds the accumulator to "
                        "zero. For this infinite-object system Optic.trace aims that "
                        "plane at positions[1] - (EPD - min(positions[1:-1])), so the "
                        "zero MOVES when the aperture changes -- the value is only "
                        "meaningful alongside entrance_pupil_diameter_m below."
                    ),
                    "opd_sign": (
                        "non-negative accumulation; larger means longer optical path. "
                        "standard_surface.py adds be.abs(t * n_pre) per surface, so it "
                        "never decreases along a purely refractive path."
                    ),
                    "opd_quantity": "absolute_accumulated_optical_path_length",
                    "opd_is_relative_to_chief_ray": False,
                    # CHE-41 tested this off axis, where it is testable: the chief
                    # ray's own opd_native is 1.0e4 waves rather than zero, and the
                    # accumulator's zero sits on a plane, so the field name's
                    # promise of a chief-ray-relative OPD remains false. What CHE-30
                    # could not see on axis is the SHAPE of that plane's failure,
                    # stated in the next two entries.
                    "opd_is_relative_to_chief_ray_verified_off_axis": "CHE-41, Hy = 0.2",
                    "opd_reference_surface": (
                        "a plane PERPENDICULAR TO Z at the launch, which is a "
                        "wavefront only for a bundle travelling along z"
                    ),
                    "opd_omits_incoming_wavefront_tilt": True,
                    "opd_omitted_term": (
                        "n_object * (d0 . r_launch), exported below as "
                        "object_space_reference_offset_m. Linear in the launch "
                        "coordinate, so on axis it is a piston that cancels in any "
                        "chief-ray subtraction and off axis it is the whole "
                        "convergence tilt: on M3-REVERSE-TELEPHOTO at Hy = 0.2 the "
                        "pupil OPL carries 0.13% of the tilt the geometry requires "
                        "without it (CHE-41, CHE-37)."
                    ),
                    "opd_convention_verified_by": "CHE-30, extended off axis by CHE-41",
                    # CHE-41: the object-space information Optic.trace throws away.
                    # Present as a declaration even when unavailable, because an
                    # absent term is what a consumer must refuse an off-axis field
                    # on, and it cannot refuse on a key that is not there.
                    "object_space_reference": {
                        "available": bool(object_space.get("available")),
                        "unavailable_reason": object_space.get("unavailable_reason"),
                        "array": (
                            "object_space_reference_offset_m"
                            if object_space.get("available")
                            else None
                        ),
                        "unit": "m",
                        "quantity": (
                            "n_object * (d0 . r_launch): the optical path from the "
                            "plane wavefront of the incoming collimated bundle that "
                            "passes through the global origin, to each ray's launch "
                            "point. ADD it to the native accumulated path to obtain a "
                            "path measured from that wavefront. It is not applied "
                            "here; opd_native is exported exactly as Optiland "
                            "produced it."
                        ),
                        "launch_geometry": (
                            "collimated_bundle_launched_on_a_plane_perpendicular_to_z"
                            if object_space.get("available")
                            else None
                        ),
                        "launch_direction": object_space.get("launch_direction"),
                        "launch_plane_z_m": (
                            object_space["launch_plane_z_native"] * _GEOMETRY_M_PER_MM
                            if object_space.get("available")
                            else None
                        ),
                        "object_space_refractive_index": object_space.get(
                            "object_space_refractive_index"
                        ),
                        "span_m": (
                            object_space["span_native"] * _GEOMETRY_M_PER_MM
                            if object_space.get("available")
                            else None
                        ),
                        "span_is_zero_on_axis": (
                            "a zero span means the term is a pure piston, which the "
                            "consumer's chief-ray subtraction removes exactly. That is "
                            "the measured statement of 'this field is on axis', and it "
                            "is what keeps the on-axis declaration bit-identical to "
                            "CHE-33's."
                        ),
                        "source": (
                            "ray_tracer.ray_generator.generate_rays over the same "
                            "hexapolar distribution Optic.trace builds. Optic.trace "
                            "returns only the traced rays and retains no launch state, "
                            "so it is regenerated; the regeneration is checked to be "
                            "collimated, planar, finite and row-matched before the "
                            "term is offered, and CHE-41's probe re-traces it and "
                            "requires x, y and opd to match the shipping trace exactly."
                        ),
                        "verified_by": "CHE-41",
                        "sha256": (
                            _scientific_array_hash(object_space_arrays)
                            if object_space_arrays
                            else None
                        ),
                    },
                    # CHE-47 (M3.9R extension): the RAW hexapolar pupil coordinates a
                    # per-ray quadrature weight is computed from downstream (CHE-38
                    # identified the missing weight as the dominant sensor-plane
                    # residual). Present as a declaration even when unavailable, for the
                    # same reason as object_space_reference above. This adapter exports
                    # coordinates only, never a ring index or a weight: that computation
                    # is coupler physics (couplers.quadrature),
                    # and this module must import no coupler -- see
                    # _resolve_ray_pupil_sampling.
                    "quadrature_weight": {
                        "available": bool(quadrature.get("available")),
                        "unavailable_reason": quadrature.get("unavailable_reason"),
                        "pupil_x_array": (
                            "pupil_normalized_x" if quadrature.get("available") else None
                        ),
                        "pupil_y_array": (
                            "pupil_normalized_y" if quadrature.get("available") else None
                        ),
                        "pupil_coordinate_unit": "dimensionless, normalized to the unit disk",
                        "quantity": (
                            "the normalized entrance-pupil coordinates (Px, Py) Optic.trace "
                            "sampled the hexapolar fan on, regenerated because Optic.trace "
                            "keeps no record of them. A CONSUMER computes each ray's ring "
                            "index and the absolute pupil/phase-space area element that ring "
                            "represents (couplers.quadrature."
                            "hexapolar_ring_index / hexapolar_area_weight_m2), radial-"
                            "trapezoid corrected at the two boundaries (center 3/4, outer "
                            "ring 1/2, interior 1x the nominal cell pi*a^2/(3*num_rings^2)), "
                            "then multiplies it onto sqrt(intensity) to get a per-ray "
                            "amplitude whose coherent sum is a converged quadrature of the "
                            "aperture. Not applied here; intensity is exported exactly as "
                            "Optiland produced it."
                        ),
                        "num_rings": quadrature.get("num_rings"),
                        "aperture_radius_m": quadrature.get("aperture_radius_m"),
                        "source": (
                            "optiland.distribution.create_distribution('hexapolar') "
                            "regenerated over the same num_rings Optic.trace used, matched "
                            "row for row against the traced set (CHE-38 sections 14-15, "
                            "CHE-47)."
                        ),
                        "verified_by": "CHE-47",
                        "sha256": (
                            _scientific_array_hash(quadrature_arrays) if quadrature_arrays else None
                        ),
                    },
                    "entrance_pupil_diameter_m": entrance_pupil_diameter_m,
                    "object_at_infinity": object_at_infinity,
                    # Read from the prescription, not assumed. A consumer moving an
                    # optical path between the image surface and any other plane in
                    # image space needs this index, and "it is air" is a property of
                    # these two systems rather than of lenses.
                    "image_space_refractive_index": image_space_refractive_index,
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
