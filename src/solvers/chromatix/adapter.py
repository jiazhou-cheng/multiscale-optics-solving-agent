"""Adapter for ``M_WAVE_CHROMATIX``: Chromatix scalar angular-spectrum propagation.

**Responsibility: the graph-facing ``ModelAdapter`` protocol, and nothing else.**
``spec``, ``estimate``, ``validate_request`` and ``run``. Everything else lives
in a sibling with its own stated responsibility:

=========================  ===================================================
``constants.py``           pinned values, supported sets, declared conventions
``requests.py``            the standalone request/result/failure contract
``execution.py``           the lazy import, device selection, precision bridge
``capability.py``          why a request is refused, before anything runs
``propagation.py``         the ASM propagation and the plane arithmetic
``provenance.py``          what was installed, what ran, the array hash
``baseline.py``            the frozen M1 standalone contract
``carrier_removed_asm.py`` the carrier-removed propagation variant
=========================  ===================================================

Two names are re-bound here because callers read them at this path, and two are
deliberately *not* -- see the note above the class. The conventions below apply
to the integration as a whole, not only to this file.

Scope
-----
This adapter implements exactly one physical path, chosen because it is the
only one with any evidence behind it in
``benchmarks/probes/records/chromatix/propagation_probe.json``:

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
else in ``chromatix.functional``), any vector/polarized field, any
gradient request, and any ``config['device']`` other than ``'cpu'`` are
deliberately unimplemented and raise ``UnsupportedCapabilityError`` *before*
Chromatix is imported or called -- this matches the registered
``M_WAVE_CHROMATIX`` entry (``approximation: scalar_wave``,
``devices: [cpu]``, ``dtypes: [complex64]``). A ``complex128`` input array
is accepted rather than rejected: Chromatix's own ``ScalarField.__init__``
unconditionally downcasts it to ``complex64`` before propagation, and this
adapter reports the resulting numeric truncation in
``ModelRunResult.diagnostics["complex64_input_truncation"]`` (CHE-35) rather
than silently absorbing or refusing it.

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
(``benchmarks/probes/records/chromatix/gradient_probe.json``) exercises
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

Exception policy (per ``core.errors``)
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

from collections.abc import Mapping
from typing import Any

from core.errors import (
    AdapterDependencyError,
    UnsupportedCapabilityError,
)
from core.graph import Severity, ValidationIssue, ValidationReport
from core.precision import CapabilityError  # noqa: F401  -- re-exported; see the note below
from core.specs import ModelSpec
from registry.loader import Registry
from solvers.base import (
    CostEstimate,
    ModelRunRequest,
    ModelRunResult,
    RunStatus,
)
from solvers.chromatix.baseline import run_standalone as _run_standalone
from solvers.chromatix.capability import check_capability

# `_PINNED_COMMIT` is re-exported rather than used here. It is read as
# `adapter._PINNED_COMMIT` by the standalone-baseline test, which checks that a
# run records the *commit* it ran against -- the thing that makes this a pinned
# integration rather than a package name that happened to import. `ruff --fix`
# will prune a re-export this file does not use, so the noqa is load-bearing.
from solvers.chromatix.constants import (  # noqa: F401
    _CHROMATIX_SPATIAL_FACTOR,
    _EXPECTED_PHASOR,
    _PINNED_COMMIT,
    _SUPPORTED_DTYPES,
    MODEL_ID,
)
from solvers.chromatix.execution import (
    _import_chromatix,
    _pitch_to_pair,
)
from solvers.chromatix.propagation import WaveHandoffError, run_asm_propagate
from solvers.chromatix.requests import (
    ChromatixWaveRequest,
    ChromatixWaveResult,
)

# Deliberately NOT re-exported here: `_do_import_chromatix` and
# `_jax_gpu_unavailable_reason`. They live in `solvers.chromatix.execution`, and
# every call site reads them there. Binding them on this module would create a
# name that looks patchable and is not -- `monkeypatch.setattr(adapter, ...)`
# would rebind a copy the running code never consults, and two of this
# integration's tests assert that the solver is *never imported*, so a patch that
# quietly did nothing would pass. An AttributeError is the better failure.

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
        """The frozen M1 standalone baseline. Implementation in `baseline.py`."""
        return _run_standalone(self, request)

    def _check_capability(self, request: ModelRunRequest) -> None:
        check_capability(self.spec, request)

    def _run_asm_propagate(self, *args: Any, **kwargs: Any) -> ModelRunResult:
        """Kept as a method because a test reaches for it at this path.

        The implementation is `propagation.run_asm_propagate`, which takes the
        spec explicitly rather than reading it off an adapter -- propagation
        needs to know the model it is running, not the object that owns it.
        """
        result: ModelRunResult = run_asm_propagate(self.spec, *args, **kwargs)
        return result

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

        if "z_m" not in request.config and "target_plane_z_m" not in request.config:
            issues.append(
                ValidationIssue(
                    severity=Severity.ERROR,
                    code="CHROMATIX_MISSING_CONFIG",
                    message=(
                        "config['z_m'] (propagation distance in metres) or "
                        "config['target_plane_z_m'] (absolute target plane, from which the "
                        "distance is derived against the input field's own plane) is required."
                    ),
                    location="config.z_m",
                )
            )
        if input_record is not None and input_record.metadata.get("phasor") not in (
            None,
            _EXPECTED_PHASOR,
        ):
            issues.append(
                ValidationIssue(
                    severity=Severity.ERROR,
                    code="CHROMATIX_PHASOR_MISMATCH",
                    message=(
                        f"input phasor {input_record.metadata.get('phasor')!r} is not the "
                        f"project canonical {_EXPECTED_PHASOR!r}. CHE-35 established that "
                        f"Chromatix's ASM implements {_CHROMATIX_SPATIAL_FACTOR}, so a "
                        "mismatched field focuses in the wrong direction rather than being "
                        "merely mislabelled."
                    ),
                    location="inputs.input_field.metadata.phasor",
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
        except WaveHandoffError as exc:
            # A declared boundary condition failed. Structured refusal, never a
            # field: a wrong plane or a wrong phasor produces output that looks
            # entirely ordinary, so silence here is the dangerous option.
            return ModelRunResult(
                status=RunStatus.FAILED,
                error_type=type(exc).__name__,
                error_message=str(exc),
                diagnostics={"code": exc.code, "stage": "wave_handoff_validation"},
            )
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
def get_adapter() -> ChromatixAdapter:
    return ChromatixAdapter()
