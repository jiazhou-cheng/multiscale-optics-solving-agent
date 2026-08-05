"""Adapter for ``M_CIRCUIT_SAX`` -- differentiable photonic-circuit S-parameter
composition through the `sax <https://gdsfactory.github.io/sax/>`_ package
(pinned version ``0.18.2``, PyPI-authoritative; see
``knowledge/solvers/sax/solver_card.yaml``).

Scope of this adapter (deliberately narrow, per CLAUDE.md section 15 --
"do not attempt to integrate every tool before the first end-to-end
benchmark works"):

* ``config["mode"] == "component"`` evaluates a single ``sax.models.coupler_ideal``
  scattering model directly (no netlist assembly).
* ``config["mode"] == "mzi_circuit"`` assembles exactly the two-coupler,
  two-waveguide Mach-Zehnder interferometer netlist from
  ``knowledge/solvers/sax/probes/circuit_probe.py`` via ``sax.circuit`` --
  the only circuit topology independently verified in this repository
  against an analytic oracle (``sin^2(dphi/2)``, 1.36e-15 relative error;
  energy conservation and reciprocity exact). Numeric parameters
  (couplings, ``neff``/``ng``, arm lengths, loss) are configurable; the
  topology itself is not, to avoid making an unverified claim about an
  arbitrary user-supplied netlist.

Any other ``mode``, or an unsupported ``component.model`` name in
``"component"`` mode, is a deliberately unimplemented capability and raises
``UnsupportedCapabilityError`` *before* any call into ``sax``/``jax``
(CLAUDE.md section 3 rule 5 / adapters/__init__.py convention).

Conventions declared explicitly (CLAUDE.md section 3 rule 1):

* Units: this project is SI-meters internally (CLAUDE.md section 7).
  SAX's built-in models (``coupler_ideal``, ``straight``) take ``wl``,
  ``wl0``, and ``length`` in **microns** by community convention (see
  ``knowledge/solvers/sax/conventions.md``, "Units" section) -- nothing in
  SAX itself enforces this. This adapter converts every length-like
  quantity from the request's SI-meter config value by multiplying by
  ``1e6`` before calling ``sax``, and records both the SI and micron values
  in output metadata/diagnostics so the conversion is never silent.
* Phase/propagation sign: SAX's ``straight`` model was independently
  verified (see ``knowledge/solvers/sax/conventions.md``) to use
  ``exp(+i * 2*pi*n*L/wavelength)``, consistent with this project's
  canonical ``exp(-i*omega*t)`` time convention (CLAUDE.md section 7). This
  has only been checked for ``straight``; it is declared here, not
  re-derived, for every run.
* Port naming: ``sax.set_port_naming_strategy`` is global, process-wide,
  mutable state (see ``knowledge/solvers/sax/conventions.md``). This
  adapter never assumes a default. It reads
  ``config.get("port_naming_strategy", "optical")``, calls
  ``sax.set_port_naming_strategy`` with that value, and always records the
  value returned by ``sax.get_port_naming_strategy()`` afterwards in the
  output ``ArtifactRecord.metadata["port_naming_strategy"]`` and in
  ``ModelRunResult.diagnostics``. Discovered while building this adapter
  (not previously recorded in the knowledge pack): the installed 0.18.2
  package's actual *default* strategy (before any call to
  ``set_port_naming_strategy``) is ``"inout"``, not ``"optical"`` -- see
  the note appended to ``knowledge/solvers/sax/conventions.md``.
* Host-copy / derivative boundary (CLAUDE.md section 3 rule 3): SAX's
  ``SDict`` S-matrix values are JAX array scalars. To store them in a
  JSON-serializable ``ArtifactRecord``, this adapter converts every entry
  with ``complex(value)``, which is a host copy and a hard derivative
  boundary -- gradients cannot flow through the returned
  ``ArtifactRecord``. This is recorded explicitly in
  ``ModelRunResult.warnings`` and in
  ``ArtifactRecord.metadata["derivative_boundary"]`` on every run, not
  just when gradients are requested.
* Discovered dtype heterogeneity (not previously recorded in the knowledge
  pack): a bare ``sax.models.coupler_ideal`` call returns an ``SDict``
  whose entries have *mixed* per-key dtypes (some ``float64`` "thru" terms,
  some ``complex128`` "cross" terms) for a real-valued coupling ratio,
  whereas the same model assembled inside ``sax.circuit`` returns uniformly
  ``complex128`` entries. This adapter records the raw observed dtype set
  in ``ArtifactRecord.metadata["raw_dtypes_observed"]`` and always declares
  the serialized ``ArtifactRecord.dtype`` as ``"complex128"`` (the values
  are stored as ``[real, imag]`` float pairs regardless of the source
  dtype), so no information is silently dropped.

Failure-handling convention used in this module (CLAUDE.md section 3 rule 5
/ ``core/errors.py``):

* ``AdapterDependencyError`` -- raised (propagates as a real exception) if
  ``sax``/``jax`` cannot be imported.
* ``UnsupportedCapabilityError`` -- raised (propagates as a real exception),
  *before* calling into ``sax``, for a deliberately unimplemented request:
  unknown ``mode``, unsupported ``component.model``, missing required
  numeric config keys, or ``request.require_gradients=True`` (this
  adapter's ``run()`` only computes a forward S-parameter response; it does
  not implement a gradient path -- see the module docstring's gradient
  notes and the registry entry's ``derivative.verified: false``).
* Everything else that fails only once ``sax`` is actually called (for
  example an invalid ``port_naming_strategy`` string, which ``sax`` itself
  rejects with ``ValueError``, or a malformed netlist port/connection
  reference, which ``sax.circuit`` rejects with ``KeyError``) is treated as
  a solver-execution-time failure: this adapter returns
  ``ModelRunResult(status=RunStatus.FAILED, error_type=..., ...)`` rather
  than raising ``SolverExecutionError``, so that graph execution can
  inspect and report the failure structurally instead of unwinding the
  call stack.

Gradient status: ``M_CIRCUIT_SAX.derivative.verified`` is (and must remain,
per this task's scope) ``false``. The only *repository-recorded* gradient
evidence prior to this adapter was one closed-form path (``coupler_ideal``'s
``coupling`` parameter, evaluated directly, not through ``sax.circuit``),
with a passing AD-vs-finite-difference check (relative error 4.4e-13, see
``knowledge/solvers/sax/expected/gradient_probe.json``). While building
this adapter, a single additional point-check (see
``tests/test_sax_adapter.py::test_gradient_through_assembled_circuit_not_yet_verified``,
marked ``xfail(strict=False)``) found that ``jax.grad`` *does* appear to
propagate correctly through an assembled ``sax.circuit`` away from a
degenerate critical point -- see the addendum in
``knowledge/solvers/sax/conventions.md``. That is one point, not the
CLAUDE.md section 6.2 bundle (multiple step sizes, a convergence table, an
ill-conditioned case), so it does not change ``derivative.verified``.
Regardless, ``run()`` never computes or exposes a gradient itself, for
either mode, and rejects ``require_gradients=True`` eagerly.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from multiscale_optics_agent.adapters.base import (
    CostEstimate,
    ModelRunRequest,
    ModelRunResult,
    RunStatus,
)
from multiscale_optics_agent.core.artifacts import ArtifactRecord
from multiscale_optics_agent.core.errors import AdapterDependencyError, UnsupportedCapabilityError
from multiscale_optics_agent.core.graph import Severity, ValidationIssue, ValidationReport
from multiscale_optics_agent.core.specs import ArtifactKind, Device, Framework, ModelSpec
from multiscale_optics_agent.registry.loader import Registry

MODEL_ID = "M_CIRCUIT_SAX"

_SUPPORTED_MODES = frozenset({"component", "mzi_circuit"})
_VALID_PORT_NAMING_STRATEGIES = frozenset({"optical", "inout"})
_MZI_REQUIRED_KEYS = ("coupling1", "coupling2", "neff", "ng", "length_short_m", "length_long_m")


def _is_positive_number(value: Any) -> bool:
    """True if ``value`` is a non-bool int/float strictly greater than zero."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0


# Netlist topology from knowledge/solvers/sax/probes/circuit_probe.py -- the
# only circuit assembly independently verified in this repository. Kept
# fixed (not user-configurable) so this adapter never assembles an
# unverified topology; see module docstring.
_MZI_NETLIST: dict[str, Any] = {
    "instances": {
        "c1": "coupler_ideal",
        "wg_short": "straight",
        "wg_long": "straight",
        "c2": "coupler_ideal",
    },
    "connections": {
        "c1,o3": "wg_short,o1",
        "c1,o4": "wg_long,o1",
        "wg_short,o2": "c2,o1",
        "wg_long,o2": "c2,o2",
    },
    "ports": {
        "in0": "c1,o1",
        "in1": "c1,o2",
        "out0": "c2,o3",
        "out1": "c2,o4",
    },
}


def _import_sax() -> tuple[Any, Any, Any, Any]:
    """Lazily import jax/sax. Never called at module import time.

    Raises ``AdapterDependencyError`` if the ``circuit`` extra
    (``pip install -e '.[circuit]'``) is not installed.
    """

    try:
        import jax
        import jax.numpy as jnp
        import sax  # type: ignore[import-untyped]
        import sax.models as sax_models  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover - exercised via monkeypatch in tests
        raise AdapterDependencyError(
            "sax (and/or its jax dependency) could not be imported. Install the "
            "'circuit' extra: `pip install -e '.[circuit]'`."
        ) from exc

    # sax.saxtypes.core sets jax_enable_x64=True as an import side effect,
    # but Python only executes a module body once per process: if something
    # else already imported sax earlier (e.g. pytest collection importing
    # this test module), that side effect already fired and this `import
    # sax` above is a sys.modules cache hit that runs no code. Meanwhile
    # jax_enable_x64 is process-global mutable state another adapter (e.g.
    # chromatix, which requires it disabled) may have since turned back off.
    # Assert our own requirement explicitly, every call, instead of relying
    # on an unrepeatable import side effect (like optiland's set_backend,
    # never call this concurrently across threads).
    jax.config.update("jax_enable_x64", True)  # type: ignore[no-untyped-call]

    return jax, jnp, sax, sax_models


_SDictSerialization = tuple[dict[str, list[float]], list[str], list[str]]


def _serialize_sdict(sdict: dict[tuple[str, str], Any]) -> _SDictSerialization:
    """Convert a SAX ``SDict`` into a JSON-serializable payload.

    Returns ``(serialized, port_names, raw_dtypes_observed)``. Converting
    each JAX array scalar with ``complex(...)`` is a host copy and a
    derivative boundary -- see module docstring.
    """

    serialized: dict[str, list[float]] = {}
    ports: set[str] = set()
    raw_dtypes: set[str] = set()
    for (port_a, port_b), value in sdict.items():
        ports.add(port_a)
        ports.add(port_b)
        c = complex(value)
        serialized[f"{port_a}->{port_b}"] = [c.real, c.imag]
        raw_dtypes.add(str(getattr(value, "dtype", "unknown")))
    return serialized, sorted(ports), sorted(raw_dtypes)


def _device_from_jax(jax_mod: Any) -> Device:
    try:
        platform = str(jax_mod.devices()[0].platform).lower()
    except Exception:  # pragma: no cover - defensive; jax.devices() should not fail here
        return Device.CPU
    return {"cpu": Device.CPU, "gpu": Device.GPU, "tpu": Device.TPU}.get(platform, Device.CPU)


class SaxAdapter:
    """``ModelAdapter`` implementation for ``M_CIRCUIT_SAX``.

    Structurally implements ``multiscale_optics_agent.adapters.base.ModelAdapter``.
    """

    def __init__(self) -> None:
        self._spec: ModelSpec | None = None

    @property
    def spec(self) -> ModelSpec:
        if self._spec is None:
            self._spec = Registry.from_package().models[MODEL_ID]
        return self._spec

    # ------------------------------------------------------------------
    # Cost estimation
    # ------------------------------------------------------------------
    def estimate(self, request: ModelRunRequest) -> CostEstimate:
        mode = request.config.get("mode", "component")
        if mode == "mzi_circuit":
            return CostEstimate(
                wall_time_s=0.2,
                peak_memory_bytes=10_000_000,
                solver_calls=1,
                confidence="low",
                notes=[
                    "Heuristic only, not benchmarked: registry cost_model declares "
                    "O(number_of_ports^3) for the dense circuit solve; a 4-port MZI "
                    "is small enough that JAX/XLA dispatch overhead likely dominates.",
                ],
            )
        if mode == "component":
            return CostEstimate(
                wall_time_s=0.02,
                peak_memory_bytes=1_000_000,
                solver_calls=1,
                confidence="low",
                notes=[
                    "Heuristic only, not benchmarked: a single closed-form "
                    "sax.models.coupler_ideal evaluation, no matrix solve.",
                ],
            )
        return CostEstimate(
            solver_calls=1,
            confidence="unknown",
            notes=[f"Unrecognized mode {mode!r}; cost cannot be estimated."],
        )

    # ------------------------------------------------------------------
    # Request validation (does not itself call sax/jax)
    # ------------------------------------------------------------------
    def validate_request(self, request: ModelRunRequest) -> ValidationReport:
        issues: list[ValidationIssue] = []
        config = request.config
        mode = config.get("mode", "component")

        if mode not in _SUPPORTED_MODES:
            issues.append(
                ValidationIssue(
                    severity=Severity.ERROR,
                    code="SAX_UNKNOWN_MODE",
                    message=(
                        f"Unsupported mode {mode!r}; expected one of {sorted(_SUPPORTED_MODES)}."
                    ),
                    location="config.mode",
                )
            )

        if request.require_gradients:
            issues.append(
                ValidationIssue(
                    severity=Severity.ERROR,
                    code="SAX_GRADIENT_NOT_IMPLEMENTED",
                    message=(
                        "This adapter's run() does not compute gradients. Only one "
                        "closed-form component path (coupler_ideal.coupling) has an "
                        "AD-vs-finite-difference check; the assembled sax.circuit "
                        "matrix-solve path has never been gradient-probed."
                    ),
                    location="require_gradients",
                )
            )

        wavelength_m = config.get("wavelength_m")
        if not _is_positive_number(wavelength_m):
            issues.append(
                ValidationIssue(
                    severity=Severity.ERROR,
                    code="SAX_INVALID_WAVELENGTH",
                    message="config.wavelength_m must be a positive number in SI meters.",
                    location="config.wavelength_m",
                )
            )

        strategy = config.get("port_naming_strategy", "optical")
        if strategy not in _VALID_PORT_NAMING_STRATEGIES:
            issues.append(
                ValidationIssue(
                    severity=Severity.ERROR,
                    code="SAX_INVALID_PORT_NAMING_STRATEGY",
                    message=(
                        f"port_naming_strategy {strategy!r} is not one of "
                        f"{sorted(_VALID_PORT_NAMING_STRATEGIES)} (sax.set_port_naming_strategy "
                        "would raise ValueError for this value)."
                    ),
                    location="config.port_naming_strategy",
                )
            )

        if mode == "component":
            component_cfg = config.get("component")
            if not isinstance(component_cfg, dict) or component_cfg.get("model") != "coupler_ideal":
                issues.append(
                    ValidationIssue(
                        severity=Severity.ERROR,
                        code="SAX_UNSUPPORTED_COMPONENT",
                        message=(
                            "mode='component' only implements 'coupler_ideal' in this "
                            f"adapter version; got component={component_cfg!r}."
                        ),
                        location="config.component",
                    )
                )
            elif "coupling" not in component_cfg.get("params", {}):
                issues.append(
                    ValidationIssue(
                        severity=Severity.ERROR,
                        code="SAX_MISSING_COUPLING",
                        message="config.component.params.coupling is required for coupler_ideal.",
                        location="config.component.params.coupling",
                    )
                )
        elif mode == "mzi_circuit":
            missing = [key for key in _MZI_REQUIRED_KEYS if key not in config]
            if missing:
                issues.append(
                    ValidationIssue(
                        severity=Severity.ERROR,
                        code="SAX_MISSING_MZI_PARAMS",
                        message=f"mode='mzi_circuit' requires config keys {missing}.",
                        location="config",
                    )
                )

        if not issues:
            issues.append(
                ValidationIssue(
                    severity=Severity.INFO,
                    code="SAX_REQUEST_VALID",
                    message="Request passed adapter-level validation.",
                )
            )
        return ValidationReport(issues=issues)

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------
    def run(self, request: ModelRunRequest) -> ModelRunResult:
        config = request.config
        mode = config.get("mode", "component")

        # Eager, pre-solver checks for deliberately unimplemented capabilities.
        if request.require_gradients:
            raise UnsupportedCapabilityError(
                "M_CIRCUIT_SAX.run() does not implement gradient computation; the "
                "assembled sax.circuit matrix-solve path has never been "
                "gradient-probed (derivative.verified=false in the registry)."
            )
        if mode not in _SUPPORTED_MODES:
            raise UnsupportedCapabilityError(
                f"Unsupported mode {mode!r}; this adapter implements only "
                f"{sorted(_SUPPORTED_MODES)}."
            )

        wavelength_m = config.get("wavelength_m")
        if not _is_positive_number(wavelength_m):
            raise UnsupportedCapabilityError(
                "config.wavelength_m (a positive number in SI meters) is required."
            )

        if mode == "component":
            component_cfg = config.get("component")
            if not isinstance(component_cfg, dict) or component_cfg.get("model") != "coupler_ideal":
                raise UnsupportedCapabilityError(
                    "mode='component' only implements 'coupler_ideal' in this "
                    f"adapter version; got component={component_cfg!r}."
                )
            params = component_cfg.get("params", {})
            if "coupling" not in params:
                raise UnsupportedCapabilityError(
                    "config.component.params.coupling is required for coupler_ideal."
                )
        else:  # mode == "mzi_circuit"
            missing = [key for key in _MZI_REQUIRED_KEYS if key not in config]
            if missing:
                raise UnsupportedCapabilityError(
                    f"mode='mzi_circuit' requires config keys {missing}."
                )

        jax_mod, _jnp, sax, sax_models = _import_sax()
        requested_strategy = config.get("port_naming_strategy", "optical")

        try:
            if mode == "component":
                outputs, diagnostics, warnings = self._run_component(
                    request, jax_mod, sax, sax_models, config, requested_strategy
                )
            else:
                outputs, diagnostics, warnings = self._run_mzi_circuit(
                    request, jax_mod, sax, sax_models, config, requested_strategy
                )
        except Exception as exc:
            return ModelRunResult(
                status=RunStatus.FAILED,
                error_type=type(exc).__name__,
                error_message=str(exc),
                diagnostics={
                    "mode": mode,
                    "requested_port_naming_strategy": requested_strategy,
                    "stage": "sax_execution",
                },
            )

        return ModelRunResult(
            status=RunStatus.SUCCEEDED,
            outputs=outputs,
            diagnostics=diagnostics,
            warnings=warnings,
        )

    def _run_component(
        self,
        request: ModelRunRequest,
        jax_mod: Any,
        sax: Any,
        sax_models: Any,
        config: dict[str, Any],
        requested_strategy: str,
    ) -> tuple[dict[str, ArtifactRecord], dict[str, Any], list[str]]:
        sax.set_port_naming_strategy(requested_strategy)
        active_strategy = sax.get_port_naming_strategy()

        wavelength_m = float(config["wavelength_m"])
        wl_um = wavelength_m * 1e6
        coupling = config["component"]["params"]["coupling"]

        sdict = sax_models.coupler_ideal(wl=wl_um, coupling=coupling)
        serialized, port_names, raw_dtypes = _serialize_sdict(sdict)

        artifact = self._build_artifact(
            request=request,
            output_name="response",
            serialized=serialized,
            port_names=port_names,
            raw_dtypes=raw_dtypes,
            wavelength_m=wavelength_m,
            port_naming_strategy=active_strategy,
            device=_device_from_jax(jax_mod),
            extra_metadata={
                "mode": "component",
                "component_model": "coupler_ideal",
                "coupling": coupling,
            },
        )
        diagnostics = {
            "mode": "component",
            "sax_version": sax.__version__,
            "jax_version": jax_mod.__version__,
            "requested_port_naming_strategy": requested_strategy,
            "active_port_naming_strategy": active_strategy,
            "wavelength_m": wavelength_m,
            "wavelength_um": wl_um,
        }
        warnings = [
            "S-matrix values were converted from JAX array scalars to Python "
            "complex for serialization: this is a host-copy derivative boundary "
            "(CLAUDE.md section 3 rule 3); gradients cannot flow through this "
            "ArtifactRecord.",
        ]
        return {"response": artifact}, diagnostics, warnings

    def _run_mzi_circuit(
        self,
        request: ModelRunRequest,
        jax_mod: Any,
        sax: Any,
        sax_models: Any,
        config: dict[str, Any],
        requested_strategy: str,
    ) -> tuple[dict[str, ArtifactRecord], dict[str, Any], list[str]]:
        sax.set_port_naming_strategy(requested_strategy)
        active_strategy = sax.get_port_naming_strategy()

        wavelength_m = float(config["wavelength_m"])
        wl_um = wavelength_m * 1e6
        loss_dB_cm = config.get("loss_dB_cm", 0.0)
        length_short_um = float(config["length_short_m"]) * 1e6
        length_long_um = float(config["length_long_m"]) * 1e6
        neff = config["neff"]
        ng = config["ng"]

        models = {"coupler_ideal": sax_models.coupler_ideal, "straight": sax_models.straight}
        mzi, info = sax.circuit(_MZI_NETLIST, models)
        result = mzi(
            wl=wl_um,
            c1={"coupling": config["coupling1"]},
            c2={"coupling": config["coupling2"]},
            wg_short={
                "length": length_short_um,
                "neff": neff,
                "wl0": wl_um,
                "ng": ng,
                "loss_dB_cm": loss_dB_cm,
            },
            wg_long={
                "length": length_long_um,
                "neff": neff,
                "wl0": wl_um,
                "ng": ng,
                "loss_dB_cm": loss_dB_cm,
            },
        )
        serialized, port_names, raw_dtypes = _serialize_sdict(result)

        artifact = self._build_artifact(
            request=request,
            output_name="response",
            serialized=serialized,
            port_names=port_names,
            raw_dtypes=raw_dtypes,
            wavelength_m=wavelength_m,
            port_naming_strategy=active_strategy,
            device=_device_from_jax(jax_mod),
            extra_metadata={
                "mode": "mzi_circuit",
                "topology": "two coupler_ideal + two straight (see circuit_probe.py)",
                "coupling1": config["coupling1"],
                "coupling2": config["coupling2"],
                "neff": neff,
                "ng": ng,
                "length_short_m": config["length_short_m"],
                "length_long_m": config["length_long_m"],
                "loss_dB_cm": loss_dB_cm,
            },
        )
        diagnostics = {
            "mode": "mzi_circuit",
            "sax_version": sax.__version__,
            "jax_version": jax_mod.__version__,
            "requested_port_naming_strategy": requested_strategy,
            "active_port_naming_strategy": active_strategy,
            "wavelength_m": wavelength_m,
            "wavelength_um": wl_um,
            "circuit_info": str(info),
        }
        warnings = [
            "S-matrix values were converted from JAX array scalars to Python "
            "complex for serialization: this is a host-copy derivative boundary "
            "(CLAUDE.md section 3 rule 3); gradients cannot flow through this "
            "ArtifactRecord.",
            "Gradient flow through this assembled sax.circuit matrix-solve path "
            "has never been probed (registry derivative.verified=false); do not "
            "treat this circuit as differentiable end-to-end.",
        ]
        return {"response": artifact}, diagnostics, warnings

    @staticmethod
    def _build_artifact(
        *,
        request: ModelRunRequest,
        output_name: str,
        serialized: dict[str, list[float]],
        port_names: list[str],
        raw_dtypes: list[str],
        wavelength_m: float,
        port_naming_strategy: str,
        device: Device,
        extra_metadata: dict[str, Any],
    ) -> ArtifactRecord:
        n_ports = len(port_names)
        payload = json.dumps(serialized, sort_keys=True).encode("utf-8")
        sha256 = hashlib.sha256(payload).hexdigest()
        metadata: dict[str, Any] = {
            "s_parameters": serialized,
            "port_names": port_names,
            "wavelength": wavelength_m,
            "wave_definition": {
                "phasor_convention": "exp(-i*omega*t)",
                "propagation_phase_sign": (
                    "exp(+i*2*pi*n*L/wavelength) (verified for sax.models.straight only)"
                ),
                "wavelength_units": (
                    "meters (SI); sax internally uses microns, converted by this adapter"
                ),
            },
            "port_naming_strategy": port_naming_strategy,
            "raw_dtypes_observed": raw_dtypes,
            "derivative_boundary": "host_copy_via_complex_conversion_for_serialization",
            "n_recorded_s_parameter_entries": len(serialized),
            **extra_metadata,
        }
        return ArtifactRecord(
            id=f"{request.node_id}:{output_name}",
            kind=ArtifactKind.CIRCUIT_RESPONSE,
            uri=f"memory://sax/{request.run_id}/{request.node_id}/{output_name}",
            sha256=sha256,
            shape=(n_ports, n_ports),
            dtype="complex128",
            framework=Framework.INTERNAL,
            device=device,
            units=None,
            metadata=metadata,
        )


def get_adapter() -> SaxAdapter:
    return SaxAdapter()
