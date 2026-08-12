"""Adapter for ``M_WAVE_CHROMATIX``: Chromatix scalar angular-spectrum propagation.

Scope
-----
This adapter implements exactly one physical path, chosen because it is the
only one with any evidence behind it in
``knowledge/solvers/chromatix/expected/propagation_probe.json``:

    ScalarField (2D, monochromatic)
        -> ``chromatix.functional.asm_propagate`` (angular-spectrum method)
        -> ScalarField (2D, monochromatic, padded per Chromatix's own
           ``compute_padding_transfer`` estimate unless ``pad_width`` is
           given explicitly)

Two entry points expose that one path:

- :meth:`ChromatixAdapter.run` -- the graph-facing ``ModelAdapter`` protocol
  method, driven by ``ModelRunRequest``/``ArtifactRecord``. Capability and
  dependency failures *raise*; solver failures come back as
  ``ModelRunResult(status=FAILED, ...)``.
- :meth:`ChromatixAdapter.run_standalone` -- the CHE-14 standalone wave
  baseline, driven by the typed :class:`ChromatixWaveRequest` and returning
  :class:`ChromatixWaveResult`. It never raises: every rejected capability,
  invalid convention/sampling value, unreadable input, resource-estimate
  overrun, and solver failure is returned as a structured
  :class:`ChromatixWaveFailure`, and no field, power, or convergence value is
  fabricated on a failure path. It writes a self-contained, hashable bundle
  (``input_field.npy``, ``output_field.npy``, ``summary.json``) so two runs
  can be compared for bit-identical scientific output.

Any other propagation kernel (``transform_propagate``/Fresnel, or anything
else in ``chromatix.functional``), any vector/polarized field, and any
gradient request are deliberately unimplemented and raise
``UnsupportedCapabilityError`` *before* Chromatix is imported or called. This
is a narrower surface than the ``M_WAVE_CHROMATIX`` registry entry describes
(``approximation: vector_wave``, ``dtypes: [complex64, complex128]``); see
the module-level ``KNOWN_REGISTRY_DISCREPANCIES`` note below for specifics
this adapter discovered that the registry entry does not yet reflect.

Conventions declared by this adapter (repository scientific-contract requirements)
--------------------------------------------------------------------
- Units: the project's canonical SI convention (meters) is used for
  ``wavelength`` and ``sample_pitch`` at the adapter boundary. Chromatix
  itself is unit-scale-agnostic (see
  ``knowledge/solvers/chromatix/conventions.md``); this adapter does not
  rescale, it simply forwards the meter-valued numbers into Chromatix calls
  that only require internal consistency, not a particular magnitude.
- Axis order: arrays are ``(y, x)`` (height, width), matching
  ``chromatix.core.field.Field.spatial_dims`` (``dims.y = -2``,
  ``dims.x = -1``). A ``sample_pitch`` given as a 2-tuple is interpreted as
  ``(pitch_y, pitch_x)`` in that same order; a bare scalar means an isotropic
  (square) pixel.
- Phasor convention: the project canonical convention is
  ``exp(-i omega t)`` (repository scientific conventions). Chromatix's ``Field`` has no
  explicit time dependence and its spatial kernels use ``exp(+i k.r)``,
  which ``knowledge/solvers/chromatix/conventions.md`` notes is *consistent
  with, but not cross-checked against*, this project's convention. This
  adapter does not attempt any sign correction; it forwards the input
  ``phasor`` metadata unchanged and emits a run-time warning if it is not
  exactly ``"exp(-i omega t)"``.
- Complex fields store amplitude, not intensity (repository scientific conventions). This
  adapter does not accept a real-valued input array and silently promote it
  to complex; a non-complex ``.npy`` payload is treated as a solver-execution
  failure (`SolverExecutionError`), not silently corrected.
- Padding: ``asm_propagate`` returns a padded array (see
  ``knowledge/solvers/chromatix/conventions.md``); this adapter does not crop
  the result back to the input shape. The returned ``ArtifactRecord.shape``
  and ``metadata["padded"]`` reflect the true (possibly larger) output shape.

Derivative policy (repository gradient policy)
----------------------------------------
``M_WAVE_CHROMATIX`` is registered with ``derivative.mode: native_autodiff``
and ``derivative.verified: false``. The only directional-derivative evidence
in this repository
(``knowledge/solvers/chromatix/expected/gradient_probe.json``) exercises
``thin_lens`` -> ``transform_propagate``, a path this adapter does not
implement. No evidence exists for a gradient through ``asm_propagate``.
Accordingly this adapter raises ``UnsupportedCapabilityError`` for any
request with ``require_gradients=True``, and never sets
``derivative.verified: true`` anywhere.

Artifact storage boundary (repository scientific-contract requirements)
---------------------------------------------------------
There is no shared JAX-native artifact store in this repository yet
(content-addressed run storage is not implemented). Pending that, this adapter
reads the input field as a plain NumPy ``.npy`` file at ``ArtifactRecord.uri``
and writes the output field the same way under
``<config['output_dir'] or 'runs'>/<run_id>/<node_id>/output_field.npy``.
Converting the JAX output array to NumPy for this write is a host-copy
derivative boundary; it is recorded in ``ModelRunResult.diagnostics`` and is
consistent with this adapter never claiming a differentiable output for this
path (see "Derivative policy" above).

Exception policy (per ``multiscale_optics_agent.core.errors``)
------------------------------------------------------------------
- ``AdapterDependencyError``: chromatix/jax cannot be imported. Raised
  eagerly and left to propagate.
- ``UnsupportedCapabilityError``: the request asks for something outside the
  scope above (wrong propagation kernel, vector field, gradients, an
  ``optical_surface`` input this adapter does not fuse). Raised eagerly,
  before any Chromatix call, and left to propagate.
- Anything else that goes wrong while actually running Chromatix (missing
  input file, non-complex array, Chromatix raising internally, NaN/Inf
  output) is caught at the ``run()`` boundary and returned as
  ``ModelRunResult(status=RunStatus.FAILED, error_type="SolverExecutionError", ...)``
  rather than raised, per the task instructions for this adapter.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import time
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

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

if TYPE_CHECKING:
    import numpy as np

MODEL_ID = "M_WAVE_CHROMATIX"

_SUPPORTED_PROPAGATION = "angular_spectrum"
_EXPECTED_PHASOR = "exp(-i omega t)"
_SUPPORTED_DTYPES = {"complex64", "complex128"}

# ---------------------------------------------------------------------------
# CHE-14 standalone wave baseline constants
# ---------------------------------------------------------------------------
_BASELINE_SEED = 20260811
_BASELINE_DEVICE = "cpu"
_BASELINE_DTYPE = "complex64"
_BASELINE_FIELD_KIND = "scalar"
_PINNED_COMMIT = "d24bdf0022835bb8ce1cdcc6aeafbc7fcb39daee"
_PINNED_VERSION = "0.6.0"
_PADDING_POLICIES = ("explicit", "auto_transfer", "none")
_OUTPUT_MODES = ("full", "same")
# 4096**2 complex64 = 128 MiB for the output array alone, and asm_propagate
# holds several arrays of that size simultaneously during the FFT pair. Above
# this the baseline returns a structured resource diagnostic instead of
# attempting the run: compute_padding_transfer is a worst-case (full-bandwidth)
# estimator and routinely proposes grids two orders of magnitude larger than a
# band-limited input actually needs (see conventions.md).
_DEFAULT_MAX_OUTPUT_PIXELS = 16_777_216

# Discovered while implementing this adapter, not yet reflected in
# src/multiscale_optics_agent/registry/models.yaml (which this task does not
# own). See the final implementation report for the literal proposed diff.
KNOWN_REGISTRY_DISCREPANCIES = (
    "registry declares dtypes: [complex64, complex128], but "
    "chromatix.core.field.ScalarField.__init__ unconditionally does "
    "`self.u = jnp.asarray(u, dtype=jnp.complex64)`; a complex128 input is "
    "silently downcast to complex64 by Chromatix itself. This adapter "
    "cannot prevent that (it is inside Chromatix's own Field constructor); "
    "it only surfaces a warning when this will happen."
)


def _do_import_chromatix() -> tuple[Any, Any, Any, Any, Any]:
    """Unprotected import step; isolated so tests can force an ``ImportError``.

    Do not call this directly outside :func:`_import_chromatix` -- it does
    not translate the failure into ``AdapterDependencyError``.
    """
    import chromatix  # type: ignore[import-untyped]
    import chromatix.functional as cf  # type: ignore[import-untyped]
    import jax
    import jax.numpy as jnp
    from chromatix.functional.propagation import (  # type: ignore[import-untyped]
        compute_padding_transfer,
    )

    # jax_enable_x64 is process-global mutable state (like optiland's
    # set_backend -- never call this concurrently across threads). Another
    # adapter in the same process (e.g. sax, which requires x64 enabled) may
    # have already flipped it on as an import side effect that -- because
    # Python only executes module bodies once -- will NOT be re-triggered by
    # importing sax again later. Asserting our own requirement explicitly,
    # every call, makes this adapter correct regardless of import/call order
    # instead of depending on ambient state: Chromatix's own
    # ScalarField.__init__ force-casts input to complex64 either way, but
    # downstream FFT-based propagation (asm_propagate) can still promote to
    # complex128 under x64, which would not match probe evidence captured
    # with x64 disabled.
    jax.config.update("jax_enable_x64", False)  # type: ignore[no-untyped-call]

    return jax, jnp, chromatix, cf, compute_padding_transfer


def _import_chromatix() -> tuple[Any, Any, Any, Any, Any]:
    """Lazily import jax/chromatix, converting failure to ``AdapterDependencyError``.

    Never called at module import time (see ``adapters/__init__.py``
    convention); only called from inside ``run()``/``estimate()``.
    """
    try:
        return _do_import_chromatix()
    except ImportError as exc:
        raise AdapterDependencyError(
            "chromatix and/or jax could not be imported. This adapter requires "
            "the pinned commit in "
            "knowledge/solvers/chromatix/solver_card.yaml "
            "(git+https://github.com/chromatix-team/chromatix.git@"
            "d24bdf0022835bb8ce1cdcc6aeafbc7fcb39daee, tag 0.6.0); the PyPI "
            "package literally named 'chromatix' is an unrelated namesquat. "
            f"Underlying error: {exc!r}"
        ) from exc


def _map_device(backend: str) -> Device:
    if backend == "gpu":
        return Device.GPU
    if backend == "tpu":
        return Device.TPU
    return Device.CPU


def _pitch_to_pair(sample_pitch: Any) -> tuple[float, float]:
    """Normalize ``sample_pitch`` metadata to ``(pitch_y, pitch_x)`` in meters."""
    if isinstance(sample_pitch, int | float):
        return (float(sample_pitch), float(sample_pitch))
    pitch_y, pitch_x = sample_pitch
    return (float(pitch_y), float(pitch_x))


def _uri_to_path(uri: str) -> Path:
    if uri.startswith("file://"):
        uri = uri[len("file://") :]
    return Path(uri)


# ---------------------------------------------------------------------------
# CHE-14: typed standalone wave-baseline contract
# ---------------------------------------------------------------------------


class ChromatixWaveRequest(BaseModel):
    """Typed contract for the CHE-14 standalone scalar wave baseline.

    Exactly one scalar, monochromatic complex field enters, either in memory
    (``input_field_array``, used by the L1-WAVE-01 benchmark) or from a
    ``.npy`` file (``input_field_path``, used by the reproducible probe
    command). Everything else on this model is either an SI physical
    parameter or an explicitly declared convention that the result records
    verbatim -- nothing about the field is inferred.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    # --- field source: exactly one of these two ---------------------------
    input_field_path: Path | None = None
    input_field_array: Any = None

    # --- SI physical parameters ------------------------------------------
    wavelength_m: float
    # (dy, dx) in that order, matching the (y, x) array axis order below.
    sample_pitch_m: tuple[float, float]
    z_m: float
    refractive_index: float = 1.0

    # --- declared conventions (recorded verbatim, validated non-empty) ----
    phasor: str = _EXPECTED_PHASOR
    coordinate_frame: str = (
        "right-handed Cartesian; array axes (y, x) row-major; +z is the propagation direction"
    )
    origin: str = "coordinate origin at array index n//2 along each spatial axis"
    reference_plane: str = "input plane at z=0; output plane at z=z_m"
    normalization: str = (
        "u stores complex field amplitude, not intensity; "
        "power = sum(|u|^2) * dy * dx in the supplied length units"
    )

    # --- solver-path selection (only one path is implemented) -------------
    propagation: str = _SUPPORTED_PROPAGATION
    field_kind: str = _BASELINE_FIELD_KIND
    device: str = _BASELINE_DEVICE
    dtype: str = _BASELINE_DTYPE
    require_gradients: bool = False

    # --- grid / padding policy -------------------------------------------
    padding_policy: str = "explicit"
    pad_width: int | None = None
    output_mode: str = "full"
    max_output_pixels: int = _DEFAULT_MAX_OUTPUT_PIXELS

    # --- run bookkeeping --------------------------------------------------
    output_directory: Path
    seed: int = _BASELINE_SEED


class ChromatixWaveFailure(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str
    message: str
    stage: str
    exception_type: str | None = None


class ChromatixWaveResult(BaseModel):
    """Structured success/failure result for :class:`ChromatixWaveRequest`."""

    model_config = ConfigDict(extra="forbid")

    status: RunStatus
    package_version: str | None = None
    package_commit: str | None = None
    propagation: str | None = None
    device: str | None = None
    cpu_device: str | None = None
    jax_backend: str | None = None
    dtype: str | None = None
    input_shape: tuple[int, int] | None = None
    output_shape: tuple[int, int] | None = None
    input_sample_pitch_m: tuple[float, float] | None = None
    output_sample_pitch_m: tuple[float, float] | None = None
    pad_width: int | None = None
    padded: bool | None = None
    cropped: bool | None = None
    runtime_seconds: float | None = None
    output_directory: str | None = None
    input_field_path: str | None = None
    output_field_path: str | None = None
    summary_path: str | None = None
    input_field_sha256: str | None = None
    output_field_sha256: str | None = None
    scientific_array_sha256: str | None = None
    summary_metrics: dict[str, Any] = Field(default_factory=dict)
    field_metadata: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    failure: ChromatixWaveFailure | None = None


class _BaselineError(Exception):
    """Internal control-flow signal carrying a structured baseline diagnostic."""

    def __init__(self, code: str, message: str, stage: str, exception_type: str | None = None):
        super().__init__(message)
        self.failure = ChromatixWaveFailure(
            code=code, message=message, stage=stage, exception_type=exception_type
        )


def _installed_chromatix_provenance() -> tuple[str | None, str | None, list[str]]:
    """Return the *actually installed* chromatix (version, commit) plus warnings.

    The commit is read from the installed distribution's ``direct_url.json``
    (PEP 610), which pip writes for a VCS install. This is the real installed
    revision, not the value copied into ``solver_card.yaml``; a mismatch
    between the two is surfaced as a warning rather than silently ignored.
    """
    warnings: list[str] = []
    try:
        distribution = importlib.metadata.distribution("chromatix")
    except importlib.metadata.PackageNotFoundError:
        return None, None, ["chromatix distribution metadata is not installed."]

    version = distribution.version
    commit: str | None = None
    for entry in distribution.files or []:
        if entry.name == "direct_url.json":
            try:
                commit = json.loads(entry.read_text())["vcs_info"]["commit_id"]
            except (OSError, ValueError, KeyError, TypeError):
                commit = None
            break

    if version != _PINNED_VERSION:
        warnings.append(
            f"installed chromatix version {version!r} differs from the pinned "
            f"{_PINNED_VERSION!r} in knowledge/solvers/chromatix/solver_card.yaml."
        )
    if commit is None:
        warnings.append(
            "installed chromatix commit could not be read from direct_url.json; "
            "the pinned-commit claim cannot be verified from this environment."
        )
    elif commit != _PINNED_COMMIT:
        warnings.append(
            f"installed chromatix commit {commit!r} differs from the pinned "
            f"{_PINNED_COMMIT!r} in knowledge/solvers/chromatix/solver_card.yaml."
        )
    return version, commit, warnings


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
    """Hash names, dtype, shape, and contiguous bytes independent of file metadata."""
    import numpy as np

    digest = hashlib.sha256()
    for name in sorted(arrays):
        array = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(json.dumps(list(array.shape)).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


class ChromatixAdapter:
    """``ModelAdapter`` implementation for ``M_WAVE_CHROMATIX`` (scalar ASM only)."""

    def __init__(self) -> None:
        self._spec: ModelSpec | None = None

    @property
    def spec(self) -> ModelSpec:
        if self._spec is None:
            self._spec = Registry.from_package().models[MODEL_ID]
        return self._spec

    # ------------------------------------------------------------------
    # CHE-14 standalone wave baseline
    # ------------------------------------------------------------------
    def run_standalone(
        self, request: ChromatixWaveRequest | Mapping[str, Any]
    ) -> ChromatixWaveResult:
        """Run the one deterministic CHE-14 CPU wave baseline and persist its bundle.

        Unlike :meth:`run`, this entry point never raises for a rejected or
        failed request: every deliberately-unimplemented capability, invalid
        metadata value, unreadable input, resource-estimate overrun, and
        solver failure comes back as ``status=FAILED`` with a structured
        :class:`ChromatixWaveFailure`. No field, power, or convergence value
        is ever fabricated on a failure path.
        """
        started = time.perf_counter()
        try:
            typed = (
                request
                if isinstance(request, ChromatixWaveRequest)
                else ChromatixWaveRequest.model_validate(request)
            )
        except ValidationError as exc:
            return ChromatixWaveResult(
                status=RunStatus.FAILED,
                runtime_seconds=time.perf_counter() - started,
                failure=ChromatixWaveFailure(
                    code="CHROMATIX_INVALID_BASELINE_REQUEST",
                    message=str(exc),
                    stage="request_validation",
                    exception_type=type(exc).__name__,
                ),
            )

        try:
            return self._execute_baseline(typed, started)
        except _BaselineError as exc:
            return ChromatixWaveResult(
                status=RunStatus.FAILED,
                device=typed.device,
                cpu_device=_cpu_device_name(),
                dtype=typed.dtype,
                propagation=typed.propagation,
                runtime_seconds=time.perf_counter() - started,
                output_directory=str(typed.output_directory),
                failure=exc.failure,
            )
        except Exception as exc:  # unexpected: still structured, never invented output
            return ChromatixWaveResult(
                status=RunStatus.FAILED,
                device=typed.device,
                cpu_device=_cpu_device_name(),
                dtype=typed.dtype,
                propagation=typed.propagation,
                runtime_seconds=time.perf_counter() - started,
                output_directory=str(typed.output_directory),
                failure=ChromatixWaveFailure(
                    code="CHROMATIX_SOLVER_EXECUTION_FAILED",
                    message=str(exc),
                    stage="baseline_execution",
                    exception_type=type(exc).__name__,
                ),
            )

    @staticmethod
    def _baseline_problems(typed: ChromatixWaveRequest) -> list[tuple[str, str, str]]:
        """Return (code, message, stage) for every gate this baseline rejects.

        Pure Python: runs before chromatix/jax is imported so an unsupported
        request never reaches, or pays for, the solver.
        """
        import math

        problems: list[tuple[str, str, str]] = []

        if typed.propagation != _SUPPORTED_PROPAGATION:
            problems.append(
                (
                    "CHROMATIX_UNSUPPORTED_PROPAGATION",
                    f"propagation={typed.propagation!r} is not implemented; this baseline "
                    f"implements only {_SUPPORTED_PROPAGATION!r} "
                    "(chromatix.functional.asm_propagate). transform_propagate changes the "
                    "sample pitch and has no forward-propagation oracle in this repository.",
                    "capability_gate",
                )
            )
        if typed.field_kind != _BASELINE_FIELD_KIND:
            problems.append(
                (
                    "CHROMATIX_UNSUPPORTED_FIELD_KIND",
                    f"field_kind={typed.field_kind!r} is not implemented; only "
                    f"{_BASELINE_FIELD_KIND!r} (chromatix ScalarField) has been probed. "
                    "No Jones-basis ordering or vector propagation frame has been verified "
                    "(knowledge/solvers/chromatix/capability_notes.md).",
                    "capability_gate",
                )
            )
        if typed.require_gradients:
            problems.append(
                (
                    "CHROMATIX_GRADIENTS_NOT_SUPPORTED",
                    "require_gradients=True is rejected: no directional-derivative evidence "
                    "exists for asm_propagate. The only gradient probe in this repository "
                    "covers thin_lens -> transform_propagate, which this baseline does not "
                    "implement. This baseline never claims a derivative it has not tested.",
                    "capability_gate",
                )
            )
        if typed.device != _BASELINE_DEVICE:
            problems.append(
                (
                    "CHROMATIX_UNSUPPORTED_DEVICE",
                    f"device={typed.device!r} is not implemented; only {_BASELINE_DEVICE!r} "
                    "has been exercised (no GPU was available in the probing container).",
                    "capability_gate",
                )
            )
        if typed.dtype != _BASELINE_DTYPE:
            problems.append(
                (
                    "CHROMATIX_UNSUPPORTED_DTYPE",
                    f"dtype={typed.dtype!r} is not implemented; chromatix "
                    "ScalarField.__init__ unconditionally casts to complex64, so "
                    f"{_BASELINE_DTYPE!r} is the only honest declaration for this path.",
                    "capability_gate",
                )
            )

        sources = [typed.input_field_path is not None, typed.input_field_array is not None]
        if sum(sources) != 1:
            problems.append(
                (
                    "CHROMATIX_INVALID_BASELINE_REQUEST",
                    "exactly one of input_field_path / input_field_array must be supplied; "
                    f"got input_field_path={typed.input_field_path!r} and "
                    f"input_field_array set={typed.input_field_array is not None}.",
                    "request_validation",
                )
            )

        if not math.isfinite(typed.wavelength_m) or typed.wavelength_m <= 0.0:
            problems.append(
                (
                    "CHROMATIX_INVALID_SAMPLING",
                    f"wavelength_m must be a finite positive value in meters; "
                    f"got {typed.wavelength_m!r}.",
                    "sampling_validation",
                )
            )
        for axis, pitch in zip(("dy", "dx"), typed.sample_pitch_m, strict=True):
            if not math.isfinite(pitch) or pitch <= 0.0:
                problems.append(
                    (
                        "CHROMATIX_INVALID_SAMPLING",
                        f"sample_pitch_m[{axis}] must be a finite positive value in meters; "
                        f"got {pitch!r}.",
                        "sampling_validation",
                    )
                )
        if not math.isfinite(typed.z_m):
            problems.append(
                (
                    "CHROMATIX_INVALID_SAMPLING",
                    f"z_m must be a finite propagation distance in meters; got {typed.z_m!r}.",
                    "sampling_validation",
                )
            )
        if not math.isfinite(typed.refractive_index) or typed.refractive_index <= 0.0:
            problems.append(
                (
                    "CHROMATIX_INVALID_SAMPLING",
                    "refractive_index must be a finite positive value; "
                    f"got {typed.refractive_index!r}.",
                    "sampling_validation",
                )
            )

        for name in ("phasor", "coordinate_frame", "origin", "reference_plane", "normalization"):
            if not str(getattr(typed, name)).strip():
                problems.append(
                    (
                        "CHROMATIX_INVALID_METADATA",
                        f"declared convention {name!r} must be a non-empty string. Every model "
                        "boundary in this repository must declare its conventions explicitly.",
                        "metadata_validation",
                    )
                )

        if typed.padding_policy not in _PADDING_POLICIES:
            problems.append(
                (
                    "CHROMATIX_INVALID_PADDING",
                    f"padding_policy={typed.padding_policy!r} must be one of "
                    f"{list(_PADDING_POLICIES)!r}.",
                    "padding_validation",
                )
            )
        if typed.padding_policy == "explicit" and typed.pad_width is None:
            problems.append(
                (
                    "CHROMATIX_INVALID_PADDING",
                    "padding_policy='explicit' requires pad_width. Use 'auto_transfer' to "
                    "delegate to chromatix.compute_padding_transfer, or 'none' for pad_width=0.",
                    "padding_validation",
                )
            )
        if typed.padding_policy != "explicit" and typed.pad_width is not None:
            problems.append(
                (
                    "CHROMATIX_INVALID_PADDING",
                    f"pad_width={typed.pad_width!r} was supplied but padding_policy is "
                    f"{typed.padding_policy!r}; that would silently ignore the requested policy.",
                    "padding_validation",
                )
            )
        if typed.pad_width is not None and typed.pad_width < 0:
            problems.append(
                (
                    "CHROMATIX_INVALID_PADDING",
                    f"pad_width must be non-negative; got {typed.pad_width!r}.",
                    "padding_validation",
                )
            )
        if typed.output_mode not in _OUTPUT_MODES:
            problems.append(
                (
                    "CHROMATIX_INVALID_PADDING",
                    f"output_mode={typed.output_mode!r} must be one of {list(_OUTPUT_MODES)!r}.",
                    "padding_validation",
                )
            )
        if typed.max_output_pixels <= 0:
            problems.append(
                (
                    "CHROMATIX_INVALID_BASELINE_REQUEST",
                    f"max_output_pixels must be positive; got {typed.max_output_pixels!r}.",
                    "request_validation",
                )
            )

        return problems

    @staticmethod
    def _load_baseline_field(typed: ChromatixWaveRequest) -> Any:
        """Load and structurally validate the input complex field. Never coerces."""
        import numpy as np

        if typed.input_field_path is not None:
            path = _uri_to_path(str(typed.input_field_path))
            if not path.exists():
                raise _BaselineError(
                    "CHROMATIX_INPUT_FIELD_UNREADABLE",
                    f"input_field_path does not exist: {path}",
                    "input_field_load",
                    "FileNotFoundError",
                )
            try:
                array = np.load(path)
            except Exception as exc:
                raise _BaselineError(
                    "CHROMATIX_INPUT_FIELD_UNREADABLE",
                    f"input_field_path could not be read as a .npy array ({path}): {exc}",
                    "input_field_load",
                    type(exc).__name__,
                ) from exc
        else:
            array = np.asarray(typed.input_field_array)

        if array.ndim != 2:
            raise _BaselineError(
                "CHROMATIX_INPUT_FIELD_NOT_2D",
                f"expected a 2D scalar field with axes (y, x); got ndim={array.ndim} "
                f"shape={tuple(array.shape)}.",
                "input_field_validation",
            )
        if not np.iscomplexobj(array):
            raise _BaselineError(
                "CHROMATIX_INPUT_FIELD_NOT_COMPLEX",
                f"input field dtype {array.dtype} is not complex. Complex fields in this "
                "repository store amplitude, not intensity; this baseline does not promote a "
                "real array to a complex field, because doing so would silently reinterpret "
                "an intensity map as an amplitude.",
                "input_field_validation",
            )
        if not np.all(np.isfinite(array)):
            nonfinite = int(np.count_nonzero(~np.isfinite(array)))
            raise _BaselineError(
                "CHROMATIX_INPUT_FIELD_NOT_FINITE",
                f"input field contains {nonfinite} non-finite sample(s) (NaN/Inf).",
                "input_field_validation",
            )
        if array.size == 0:
            raise _BaselineError(
                "CHROMATIX_INPUT_FIELD_NOT_2D",
                "input field is empty.",
                "input_field_validation",
            )
        return array

    def _execute_baseline(self, typed: ChromatixWaveRequest, started: float) -> ChromatixWaveResult:
        problems = self._baseline_problems(typed)
        if problems:
            code, message, stage = problems[0]
            joined = "; ".join(item[1] for item in problems)
            raise _BaselineError(code, joined if len(problems) > 1 else message, stage)

        try:
            jax, jnp, _chromatix, cf, compute_padding_transfer = _import_chromatix()
        except AdapterDependencyError as exc:
            raise _BaselineError(
                "CHROMATIX_DEPENDENCY_UNAVAILABLE",
                str(exc),
                "dependency_gate",
                type(exc).__name__,
            ) from exc

        import numpy as np

        u_raw = self._load_baseline_field(typed)
        warnings: list[str] = []

        version, commit, provenance_warnings = _installed_chromatix_provenance()
        warnings.extend(provenance_warnings)

        if str(u_raw.dtype) != _BASELINE_DTYPE:
            warnings.append(
                f"input field dtype {u_raw.dtype} was cast to {_BASELINE_DTYPE} at the baseline "
                "boundary. chromatix.core.field.ScalarField.__init__ performs this cast "
                "unconditionally anyway; doing it here makes the saved input artifact identical "
                "to the array Chromatix actually propagated."
            )
        u_in = np.ascontiguousarray(u_raw.astype(np.complex64))

        if typed.phasor != _EXPECTED_PHASOR:
            warnings.append(
                f"declared phasor {typed.phasor!r} is not the project canonical "
                f"{_EXPECTED_PHASOR!r}. Chromatix declares no time convention and its spatial "
                "kernel is exp(+i k.r); this baseline applies no sign correction and forwards "
                "the field unchanged."
            )

        pitch_y_m, pitch_x_m = typed.sample_pitch_m
        height, width = int(u_in.shape[0]), int(u_in.shape[1])

        # --- padding policy ------------------------------------------------
        if typed.padding_policy == "none":
            pad_width = 0
        elif typed.padding_policy == "explicit":
            pad_width = int(typed.pad_width or 0)
        else:
            if pitch_y_m != pitch_x_m:
                raise _BaselineError(
                    "CHROMATIX_INVALID_PADDING",
                    "padding_policy='auto_transfer' needs square pixels: "
                    "chromatix.compute_padding_transfer takes a single scalar dx, but "
                    f"sample_pitch_m is ({pitch_y_m!r}, {pitch_x_m!r}). Pass an explicit "
                    "pad_width instead.",
                    "padding_validation",
                )
            pad_width = int(
                compute_padding_transfer(height, typed.wavelength_m, pitch_y_m, typed.z_m)
            )
            warnings.append(
                f"pad_width={pad_width} came from chromatix.compute_padding_transfer, a "
                "worst-case full-bandwidth estimator. For a band-limited input it is often "
                "far larger than necessary; the padded extent is recorded so a reviewer can "
                "check it against the physically occupied bandwidth."
            )

        padded_height = height + 2 * pad_width
        padded_width = width + 2 * pad_width
        padded_pixels = padded_height * padded_width
        if padded_pixels > typed.max_output_pixels:
            raise _BaselineError(
                "CHROMATIX_RESOURCE_ESTIMATE_EXCEEDED",
                f"padding policy {typed.padding_policy!r} implies a "
                f"{padded_height}x{padded_width} = {padded_pixels} pixel propagation grid, "
                f"above max_output_pixels={typed.max_output_pixels}. At complex64 the output "
                f"array alone would be {padded_pixels * 8 / 2**20:.1f} MiB and asm_propagate "
                "holds several such arrays during the FFT pair. Nothing was executed and no "
                "field was produced; raise max_output_pixels deliberately or reduce pad_width.",
                "resource_estimate",
            )

        # --- solver call -----------------------------------------------------
        try:
            field_in = cf.Field.build(
                jnp.asarray(u_in, dtype=jnp.complex64),
                jnp.asarray([[pitch_y_m, pitch_x_m]]),
                typed.wavelength_m,
            )
            field_out = cf.asm_propagate(
                field_in,
                z=typed.z_m,
                n=typed.refractive_index,
                pad_width=pad_width,
                mode=typed.output_mode,
            )
            u_out = np.ascontiguousarray(np.asarray(jax.device_get(field_out.u)).squeeze())
            dx_out = tuple(
                float(v) for v in np.asarray(jax.device_get(field_out.dx)).reshape(-1).tolist()
            )
            power_in = float(np.asarray(jax.device_get(field_in.power)).reshape(-1)[0])
            power_out = float(np.asarray(jax.device_get(field_out.power)).reshape(-1)[0])
        except Exception as exc:
            raise _BaselineError(
                "CHROMATIX_SOLVER_EXECUTION_FAILED",
                f"chromatix.functional.asm_propagate failed: {exc}",
                "solver_call",
                type(exc).__name__,
            ) from exc

        if u_out.ndim != 2:
            raise _BaselineError(
                "CHROMATIX_SOLVER_EXECUTION_FAILED",
                f"expected a 2D output field; got shape {tuple(u_out.shape)}.",
                "output_validation",
            )
        if not np.all(np.isfinite(u_out)):
            nonfinite = int(np.count_nonzero(~np.isfinite(u_out)))
            raise _BaselineError(
                "CHROMATIX_NONFINITE_OUTPUT",
                f"propagated field contains {nonfinite} non-finite sample(s) (NaN/Inf); "
                "no field or power metric is reported for this run.",
                "output_validation",
            )

        return self._persist_baseline(
            typed=typed,
            started=started,
            u_in=u_in,
            u_out=u_out,
            dx_out=dx_out,
            pad_width=pad_width,
            power_in=power_in,
            power_out=power_out,
            version=version,
            commit=commit,
            jax_backend=jax.default_backend(),
            warnings=warnings,
        )

    @staticmethod
    def _edge_energy_fraction(u: Any) -> float:
        """Fraction of |u|^2 sitting on the 1-pixel border of the sampled window.

        This is the observable finite-window indicator: a value far above the
        floor means the window truncates the field and any power or
        second-moment metric taken on this grid is window-limited.
        """
        import numpy as np

        intensity = np.abs(u) ** 2
        total = float(intensity.sum())
        if total <= 0.0 or min(intensity.shape) < 3:
            return 0.0
        border = float(
            intensity[0, :].sum()
            + intensity[-1, :].sum()
            + intensity[1:-1, 0].sum()
            + intensity[1:-1, -1].sum()
        )
        return border / total

    def _persist_baseline(
        self,
        *,
        typed: ChromatixWaveRequest,
        started: float,
        u_in: Any,
        u_out: Any,
        dx_out: tuple[float, ...],
        pad_width: int,
        power_in: float,
        power_out: float,
        version: str | None,
        commit: str | None,
        jax_backend: str,
        warnings: list[str],
    ) -> ChromatixWaveResult:
        import numpy as np

        output_directory = Path(typed.output_directory)
        output_directory.mkdir(parents=True, exist_ok=True)
        input_path = output_directory / "input_field.npy"
        output_path = output_directory / "output_field.npy"
        np.save(input_path, u_in)
        np.save(output_path, u_out)
        input_sha = hashlib.sha256(input_path.read_bytes()).hexdigest()
        output_sha = hashlib.sha256(output_path.read_bytes()).hexdigest()
        scientific_sha = _scientific_array_hash({"input_field": u_in, "output_field": u_out})

        input_shape = (int(u_in.shape[0]), int(u_in.shape[1]))
        output_shape = (int(u_out.shape[0]), int(u_out.shape[1]))
        expected_full = (input_shape[0] + 2 * pad_width, input_shape[1] + 2 * pad_width)
        cropped = typed.output_mode == "same" and pad_width > 0
        if not cropped and output_shape != expected_full:
            warnings.append(
                f"output shape {output_shape} does not match the padded shape {expected_full} "
                f"implied by pad_width={pad_width}; chromatix resized the grid in a way this "
                "baseline did not request."
            )

        pitch_y_m, pitch_x_m = typed.sample_pitch_m
        summary_metrics: dict[str, Any] = {
            "input_shape": list(input_shape),
            "output_shape": list(output_shape),
            "input_sample_pitch_m": [pitch_y_m, pitch_x_m],
            "output_sample_pitch_m": list(dx_out),
            "sample_pitch_unchanged": bool(
                len(dx_out) == 2
                and np.isclose(dx_out[0], pitch_y_m, rtol=1e-6)
                and np.isclose(dx_out[1], pitch_x_m, rtol=1e-6)
            ),
            "pad_width": pad_width,
            "padding_policy": typed.padding_policy,
            "padded": bool(pad_width > 0),
            "output_mode": typed.output_mode,
            "cropped": bool(cropped),
            "resampled": False,
            "input_extent_m": [input_shape[0] * pitch_y_m, input_shape[1] * pitch_x_m],
            "output_extent_m": [
                output_shape[0] * (dx_out[0] if len(dx_out) == 2 else pitch_y_m),
                output_shape[1] * (dx_out[1] if len(dx_out) == 2 else pitch_x_m),
            ],
            "power_in": power_in,
            "power_out": power_out,
            "power_conservation_ratio": (power_out / power_in) if power_in else None,
            "input_edge_energy_fraction": self._edge_energy_fraction(u_in),
            "output_edge_energy_fraction": self._edge_energy_fraction(u_out),
            "finite_window_interpretation": (
                "power_in and power_out are discrete sums over the sampled window only "
                "(sum(|u|^2) * dy * dx), not radiometric watts. Energy that leaves the window "
                "is lost from power_out, so power_conservation_ratio is a window-truncation "
                "diagnostic, not a physical conservation law. The edge-energy fractions bound "
                "how much field is sitting at the window boundary on each plane."
            ),
            "input_amplitude_max": float(np.abs(u_in).max()),
            "output_amplitude_max": float(np.abs(u_out).max()),
            "all_finite": True,
        }

        field_metadata: dict[str, Any] = {
            "axis_order": "(y, x)",
            "array_layout": "row-major; array[i, j] is y=coords_y[i], x=coords_x[j]",
            "origin": typed.origin,
            "origin_index_rule": "index n//2 along each spatial axis is coordinate 0",
            "handedness": "right-handed",
            "plus_z": "propagation direction (+z), verified against asm_propagate for z >= 0",
            "coordinate_frame": typed.coordinate_frame,
            "reference_plane": typed.reference_plane,
            "wavelength_m": typed.wavelength_m,
            "sample_pitch_m": [pitch_y_m, pitch_x_m],
            "z_m": typed.z_m,
            "refractive_index": typed.refractive_index,
            "phasor": typed.phasor,
            "spatial_kernel_sign": "exp(+i k.r) (chromatix compute_asm_propagator, z >= 0)",
            "normalization": typed.normalization,
            "values_are": "complex field amplitude (never intensity); intensity is abs(u)**2",
            "polarization": "scalar (chromatix ScalarField; no polarization state tracked)",
            "coherence": "monochromatic, fully coherent single-wavelength field",
            "dtype": _BASELINE_DTYPE,
            "device": _BASELINE_DEVICE,
            "cpu_device": _cpu_device_name(),
            "jax_backend": jax_backend,
            "jax_enable_x64": False,
            "propagation_method": "chromatix.functional.asm_propagate",
            "package": "chromatix",
            "package_version": version,
            "package_commit": commit,
        }

        summary = {
            "schema_version": 1,
            "baseline_id": "CHE-14-CHROMATIX-WAVE",
            "propagation": typed.propagation,
            "field_kind": typed.field_kind,
            "device": typed.device,
            "dtype": typed.dtype,
            "seed": typed.seed,
            "seed_semantics": (
                "recorded; this baseline builds its field analytically and uses no RNG"
            ),
            "package_version": version,
            "package_commit": commit,
            "summary_metrics": summary_metrics,
            "field_metadata": {
                key: value
                for key, value in field_metadata.items()
                # cpu_device/jax_backend are environment facts, not part of the
                # deterministic scientific summary compared across two runs.
                if key not in {"cpu_device", "jax_backend"}
            },
            "input_field_sha256": input_sha,
            "output_field_sha256": output_sha,
            "scientific_array_sha256": scientific_sha,
        }
        summary_path = output_directory / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

        return ChromatixWaveResult(
            status=RunStatus.SUCCEEDED,
            package_version=version,
            package_commit=commit,
            propagation=typed.propagation,
            device=typed.device,
            cpu_device=field_metadata["cpu_device"],
            jax_backend=jax_backend,
            dtype=typed.dtype,
            input_shape=input_shape,
            output_shape=output_shape,
            input_sample_pitch_m=(pitch_y_m, pitch_x_m),
            output_sample_pitch_m=(dx_out[0], dx_out[1]) if len(dx_out) == 2 else None,
            pad_width=pad_width,
            padded=bool(pad_width > 0),
            cropped=bool(cropped),
            runtime_seconds=time.perf_counter() - started,
            output_directory=str(output_directory),
            input_field_path=str(input_path),
            output_field_path=str(output_path),
            summary_path=str(summary_path),
            input_field_sha256=input_sha,
            output_field_sha256=output_sha,
            scientific_array_sha256=scientific_sha,
            summary_metrics=summary_metrics,
            field_metadata=field_metadata,
            warnings=warnings,
        )

    # ------------------------------------------------------------------
    # Capability gate -- pure Python, no chromatix/jax import required.
    # ------------------------------------------------------------------
    def _check_capability(self, request: ModelRunRequest) -> None:
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

        if "optical_surface" in request.inputs:
            raise UnsupportedCapabilityError(
                "M_WAVE_CHROMATIX adapter does not implement fusion of an "
                "'optical_surface' input with the propagated field in this "
                "scope; only free-space scalar angular-spectrum propagation "
                "of 'input_field' is supported."
            )

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
    def validate_request(self, request: ModelRunRequest) -> ValidationReport:
        issues: list[ValidationIssue] = []

        try:
            self._check_capability(request)
        except UnsupportedCapabilityError as exc:
            issues.append(
                ValidationIssue(
                    severity=Severity.ERROR,
                    code="CHROMATIX_UNSUPPORTED_CAPABILITY",
                    message=str(exc),
                )
            )

        input_record = request.inputs.get("input_field")
        if input_record is None:
            issues.append(
                ValidationIssue(
                    severity=Severity.ERROR,
                    code="CHROMATIX_MISSING_INPUT",
                    message="Request is missing the required 'input_field' artifact.",
                    location="inputs.input_field",
                )
            )
        else:
            missing_metadata = sorted(
                {"wavelength", "sample_pitch", "coordinate_frame", "phasor"}
                - set(input_record.metadata)
            )
            if missing_metadata:
                issues.append(
                    ValidationIssue(
                        severity=Severity.ERROR,
                        code="CHROMATIX_MISSING_METADATA",
                        message=f"input_field is missing required metadata: {missing_metadata!r}.",
                        location="inputs.input_field.metadata",
                    )
                )
            if input_record.dtype is not None and input_record.dtype not in _SUPPORTED_DTYPES:
                issues.append(
                    ValidationIssue(
                        severity=Severity.ERROR,
                        code="CHROMATIX_UNSUPPORTED_DTYPE",
                        message=(
                            f"input_field dtype {input_record.dtype!r} is not one of "
                            f"{sorted(_SUPPORTED_DTYPES)!r}."
                        ),
                        location="inputs.input_field.dtype",
                    )
                )

        if "z_m" not in request.config:
            issues.append(
                ValidationIssue(
                    severity=Severity.ERROR,
                    code="CHROMATIX_MISSING_CONFIG",
                    message="config['z_m'] (propagation distance in meters) is required.",
                    location="config.z_m",
                )
            )

        if not issues:
            issues.append(
                ValidationIssue(
                    severity=Severity.INFO,
                    code="CHROMATIX_REQUEST_VALID",
                    message="Request satisfies the scalar angular-spectrum propagation contract.",
                )
            )
        return ValidationReport(issues=issues)

    def estimate(self, request: ModelRunRequest) -> CostEstimate:
        self._check_capability(request)

        notes: list[str] = []
        input_record = request.inputs.get("input_field")
        pad_width = request.config.get("pad_width")

        if pad_width is not None:
            pad_width = int(pad_width)
        elif input_record is not None and input_record.shape and "z_m" in request.config:
            try:
                _, _, _, _, compute_padding_transfer = _import_chromatix()
                pitch_y, pitch_x = _pitch_to_pair(input_record.metadata["sample_pitch"])
                if pitch_y != pitch_x:
                    notes.append(
                        "non-square sample_pitch cannot be estimated by "
                        "chromatix.compute_padding_transfer (it takes one scalar dx); "
                        "cost estimate falls back to the unpadded shape."
                    )
                else:
                    wavelength_m = float(input_record.metadata["wavelength"])
                    z_m = float(request.config["z_m"])
                    height = input_record.shape[-2]
                    pad_width = int(compute_padding_transfer(height, wavelength_m, pitch_y, z_m))
                    notes.append(
                        f"pad_width estimated via chromatix.compute_padding_transfer: {pad_width}"
                    )
            except AdapterDependencyError as exc:
                notes.append(f"chromatix unavailable for a precise estimate: {exc}")
            except (KeyError, TypeError, ValueError) as exc:
                notes.append(f"could not estimate pad_width from inputs/config: {exc}")

        peak_memory_bytes: int | None = None
        confidence = "low"
        if input_record is not None and input_record.shape and len(input_record.shape) >= 2:
            height, width = input_record.shape[-2], input_record.shape[-1]
            if pad_width is not None:
                out_h, out_w = height + 2 * pad_width, width + 2 * pad_width
                confidence = "medium"
            else:
                out_h, out_w = height, width
                notes.append("pad_width unknown; memory estimate ignores asm_propagate padding.")
            # complex64 input + padded complex64 output, rough order of magnitude only.
            peak_memory_bytes = 8 * (height * width + out_h * out_w)

        return CostEstimate(
            wall_time_s=None,
            peak_memory_bytes=peak_memory_bytes,
            solver_calls=1,
            confidence=confidence,
            notes=notes,
        )

    def run(self, request: ModelRunRequest) -> ModelRunResult:
        # Capability gate: must raise (not be swallowed) before any solver call.
        self._check_capability(request)
        # Dependency gate: must raise (not be swallowed) if chromatix/jax is unusable.
        jax, jnp, _chromatix, cf, compute_padding_transfer = _import_chromatix()

        try:
            return self._run_asm_propagate(request, jax, jnp, cf, compute_padding_transfer)
        except (AdapterDependencyError, UnsupportedCapabilityError):
            raise
        except Exception as exc:
            return ModelRunResult(
                status=RunStatus.FAILED,
                error_type="SolverExecutionError",
                error_message=str(exc),
                diagnostics={"exception_class": type(exc).__name__},
            )

    # ------------------------------------------------------------------
    # Internal execution
    # ------------------------------------------------------------------
    def _run_asm_propagate(
        self,
        request: ModelRunRequest,
        jax: Any,
        jnp: Any,
        cf: Any,
        compute_padding_transfer: Any,
    ) -> ModelRunResult:
        # Only reached after _import_chromatix() succeeded, so numpy (a jax
        # dependency) is guaranteed importable here even though it is not a
        # base dependency of this project.
        import numpy as np

        config = request.config
        if "input_field" not in request.inputs:
            raise KeyError("request.inputs is missing required 'input_field'")
        input_record = request.inputs["input_field"]

        u_in = self._load_complex_array(input_record)
        if u_in.ndim != 2:
            raise ValueError(
                f"expected a 2D scalar field array (y, x), got ndim={u_in.ndim} shape={u_in.shape}"
            )
        if not np.iscomplexobj(u_in):
            raise ValueError(
                f"input_field array dtype {u_in.dtype} is not complex; this adapter "
                "does not silently promote a real array to a complex field "
                "(repository scientific conventions: complex fields store amplitude, "
                "not intensity)."
            )

        warnings: list[str] = []
        if str(u_in.dtype) == "complex128":
            warnings.append(
                "input array dtype is complex128, but "
                "chromatix.core.field.ScalarField.__init__ unconditionally casts to "
                "complex64 (`jnp.asarray(u, dtype=jnp.complex64)`); precision beyond "
                "complex64 will be silently lost inside Chromatix itself. "
                "See KNOWN_REGISTRY_DISCREPANCIES in this module."
            )

        wavelength_m = float(input_record.metadata["wavelength"])
        pitch_y_m, pitch_x_m = _pitch_to_pair(input_record.metadata["sample_pitch"])
        phasor = input_record.metadata.get("phasor")
        if phasor != _EXPECTED_PHASOR:
            warnings.append(
                f"input phasor metadata {phasor!r} is not the project canonical "
                f"{_EXPECTED_PHASOR!r} (repository scientific conventions). Chromatix declares no "
                "explicit time convention (see conventions.md); this adapter does "
                "not attempt a sign correction and forwards the field unchanged."
            )

        refractive_index = float(config.get("refractive_index", 1.0))
        z_m = float(config["z_m"])

        pad_width = config.get("pad_width")
        if pad_width is None:
            if pitch_y_m != pitch_x_m:
                raise ValueError(
                    "pad_width was not given and sample_pitch is non-square "
                    f"({pitch_y_m!r}, {pitch_x_m!r}); "
                    "chromatix.compute_padding_transfer only accepts a single scalar "
                    "dx, so automatic padding is only supported for square pixels in "
                    "this adapter. Pass config['pad_width'] explicitly."
                )
            pad_width = int(compute_padding_transfer(u_in.shape[0], wavelength_m, pitch_y_m, z_m))
        else:
            pad_width = int(pad_width)

        # chromatix.utils.shapes._broadcast_dx_to_grid requires a non-square dx
        # for a monochromatic field to be shaped (1, 2) -- i.e. (wavelengths,
        # 2) -- not a bare (2,) array (verified against the installed
        # package; a bare (2,) array raises "Number of wavelengths does not
        # match" because it is interpreted as one dx value per wavelength).
        field_in = cf.Field.build(
            jnp.asarray(u_in, dtype=jnp.complex64),
            jnp.asarray([[pitch_y_m, pitch_x_m]]),
            wavelength_m,
        )
        field_out = cf.asm_propagate(field_in, z=z_m, n=refractive_index, pad_width=pad_width)

        u_out = np.asarray(jax.device_get(field_out.u))
        dx_out = tuple(
            float(v) for v in np.asarray(jax.device_get(field_out.dx)).reshape(-1).tolist()
        )
        power_in = float(np.asarray(jax.device_get(field_in.power)).reshape(-1)[0])
        power_out = float(np.asarray(jax.device_get(field_out.power)).reshape(-1)[0])

        output_root = Path(config.get("output_dir", "runs")) / request.run_id / request.node_id
        output_root.mkdir(parents=True, exist_ok=True)
        output_path = output_root / "output_field.npy"
        sha256 = self._write_array(u_out, output_path)

        output_metadata = {
            "wavelength": wavelength_m,
            "sample_pitch": dx_out,
            "coordinate_frame": (
                "axes=(y, x) row-major (chromatix Field.spatial_dims convention); "
                "right-handed Cartesian; +z is the propagation direction"
            ),
            "phasor": input_record.metadata.get("phasor", _EXPECTED_PHASOR),
            "polarization": "scalar (chromatix ScalarField; no polarization state tracked)",
            "normalization": (
                "u stores complex field amplitude, not intensity. field.power = "
                "sum(|u|^2) * prod(dx) is Chromatix's own bookkeeping in the length "
                "units supplied for dx/wavelength (meters, per project convention); "
                "it is not a calibrated SI radiometric power (W) unless the input "
                "amplitude already carried that convention."
            ),
            "propagation_method": "asm_propagate",
            "z_m": z_m,
            "refractive_index": refractive_index,
            "pad_width": pad_width,
            "padded": tuple(u_out.shape) != tuple(u_in.shape),
            "input_shape": tuple(int(s) for s in u_in.shape),
        }

        output_record = ArtifactRecord(
            id=f"{request.node_id}:output_field",
            kind=ArtifactKind.COMPLEX_FIELD,
            uri=str(output_path),
            sha256=sha256,
            shape=tuple(int(s) for s in u_out.shape),
            dtype=str(u_out.dtype),
            framework=Framework.JAX,
            device=_map_device(jax.default_backend()),
            units=None,
            metadata=output_metadata,
        )

        diagnostics = {
            "power_in": power_in,
            "power_out": power_out,
            "power_conservation_ratio": (power_out / power_in) if power_in else None,
            "chromatix_pinned_version": self.spec.source.pinned_version
            if self.spec.source
            else None,
            "jax_default_backend": jax.default_backend(),
            "derivative_note": (
                "field_out.u was converted from a JAX array to NumPy and written to "
                "disk (no shared JAX-native artifact store exists yet in this "
                "repository); this is a derivative boundary recorded under the repository "
                "scientific contract. require_gradients is rejected for this node before any "
                "solver call (see UnsupportedCapabilityError), so no differentiable "
                "path claim is made or broken by this conversion."
            ),
        }

        return ModelRunResult(
            status=RunStatus.SUCCEEDED,
            outputs={"output_field": output_record},
            diagnostics=diagnostics,
            warnings=warnings,
        )

    @staticmethod
    def _load_complex_array(record: ArtifactRecord) -> np.ndarray[Any, Any]:
        import numpy as np

        path = _uri_to_path(record.uri)
        if not path.exists():
            raise FileNotFoundError(
                f"input_field artifact file not found for {record.id!r}: {path}"
            )
        array: np.ndarray[Any, Any] = np.load(path)
        return array

    @staticmethod
    def _write_array(array: np.ndarray[Any, Any], path: Path) -> str:
        import numpy as np

        np.save(path, array)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        return digest


def get_adapter() -> ChromatixAdapter:
    return ChromatixAdapter()
