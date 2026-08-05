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

Any other propagation kernel (``transform_propagate``/Fresnel, or anything
else in ``chromatix.functional``), any vector/polarized field, and any
gradient request are deliberately unimplemented and raise
``UnsupportedCapabilityError`` *before* Chromatix is imported or called. This
is a narrower surface than the ``M_WAVE_CHROMATIX`` registry entry describes
(``approximation: vector_wave``, ``dtypes: [complex64, complex128]``); see
the module-level ``KNOWN_REGISTRY_DISCREPANCIES`` note below for specifics
this adapter discovered that the registry entry does not yet reflect.

Conventions declared by this adapter (CLAUDE.md section 3, rule 1)
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
  ``exp(-i omega t)`` (CLAUDE.md section 7). Chromatix's ``Field`` has no
  explicit time dependence and its spatial kernels use ``exp(+i k.r)``,
  which ``knowledge/solvers/chromatix/conventions.md`` notes is *consistent
  with, but not cross-checked against*, this project's convention. This
  adapter does not attempt any sign correction; it forwards the input
  ``phasor`` metadata unchanged and emits a run-time warning if it is not
  exactly ``"exp(-i omega t)"``.
- Complex fields store amplitude, not intensity (CLAUDE.md section 7). This
  adapter does not accept a real-valued input array and silently promote it
  to complex; a non-complex ``.npy`` payload is treated as a solver-execution
  failure (`SolverExecutionError`), not silently corrected.
- Padding: ``asm_propagate`` returns a padded array (see
  ``knowledge/solvers/chromatix/conventions.md``); this adapter does not crop
  the result back to the input shape. The returned ``ArtifactRecord.shape``
  and ``metadata["padded"]`` reflect the true (possibly larger) output shape.

Derivative policy (CLAUDE.md section 6)
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

Artifact storage boundary (CLAUDE.md section 3, rule 3)
---------------------------------------------------------
There is no shared JAX-native artifact store in this repository yet
(``docs/ARCHITECTURE.md`` section 5 describes a future "content-addressed run
directory" that is not implemented). Pending that, this adapter reads the
input field as a plain NumPy ``.npy`` file at ``ArtifactRecord.uri`` and
writes the output field the same way under
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
from pathlib import Path
from typing import TYPE_CHECKING, Any

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
                "(CLAUDE.md section 7: complex fields store amplitude, not intensity)."
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
                f"{_EXPECTED_PHASOR!r} (CLAUDE.md section 7). Chromatix declares no "
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
                "repository); this is a derivative boundary (CLAUDE.md section 3, "
                "rule 3). require_gradients is rejected for this node before any "
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
