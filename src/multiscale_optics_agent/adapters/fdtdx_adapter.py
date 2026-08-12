"""Forward-only adapter for ``M_EM_FDTDX`` (FDTDX, pinned version 0.6.2).

Scope
-----
This adapter is deliberately **forward-only**. Two independent gradient
probes against this exact pinned install both failed outright and are
documented in ``knowledge/solvers/fdtdx/`` (``conventions.md``,
``capability_notes.md``, ``failure_guide.md``,
``probes/gradient_probe.py`` + ``expected/gradient_probe.json``):

1. ``jax.grad`` with respect to the source wavelength (closed over as a
   traced scalar passed into ``WaveCharacter(wavelength=...)``) returns
   exactly ``0.0``, while a centered finite-difference estimate at the same
   point gives a large nonzero value (~``-1.02e6``). This is a phantom zero
   gradient, plausibly from an internal ``round()``-based conversion from
   wavelength to an integer time-step count.
2. ``jax.grad`` with respect to background permittivity (passed directly
   into ``Material(permittivity=<traced>)`` before ``place_objects``)
   raises ``jax.errors.ConcretizationTypeError``, because
   ``fdtdx.place_objects`` performs concrete Python-level introspection of
   material properties (``math.isclose`` in
   ``fdtdx/materials.py::_is_property_isotropic``) that cannot be traced.

Neither bug is fixed or worked around here -- that is explicitly out of
scope for this adapter (see the calling task's scope boundary). The
apparently-correct pattern -- build the object graph once with concrete
values via ``place_objects``, then differentiate only through
``apply_params(arrays, objects, params, key)`` with respect to the
``params``/``ParameterContainer`` pytree, or via ``fdtdx.full_backward`` --
has not been worked out or tested against this pinned version. Because no
differentiable path has passed the required directional-derivative test, this
adapter raises ``UnsupportedCapabilityError`` eagerly
for any request with ``require_gradients=True``, before importing fdtdx or
touching the solver, rather than letting a caller discover either failure
mode itself deep inside its own ``jax.grad`` call. This does **not**
contradict the registry's ``derivative.mode: native_autodiff`` claim (which
already declares ``verified: false`` and documents both failures in
``derivative.notes``) -- it only means this particular adapter
implementation does not expose any code path that would let a caller
attempt to differentiate through it.

Exception-handling convention (per ``core/errors.py`` docstring)
------------------------------------------------------------------
- ``AdapterDependencyError`` is raised (propagates to the caller) when
  ``fdtdx``/``jax`` cannot be imported.
- ``UnsupportedCapabilityError`` is raised (propagates to the caller)
  *eagerly*, before any solver call, for ``require_gradients=True`` and for
  spatially-varying material inputs (see below).
- Failures that occur once the solver has actually been invoked
  (``place_objects`` / ``apply_params`` / ``run_fdtd`` raising, NaN/Inf
  output) are caught at the ``run()`` boundary and reported as
  ``ModelRunResult(status=RunStatus.FAILED, error_type=..., error_message=...)``
  rather than raised, per ``SolverExecutionError``'s docstring.

Conventions recorded on every output artifact (see
``knowledge/solvers/fdtdx/conventions.md``)
------------------------------------------------------------------
- Field array axis order is ``(component, x, y, z)`` -- the leading axis of
  size 3 is (Ex, Ey, Ez), **not** a trailing component axis. This adapter
  never transposes the array; the axis order is recorded verbatim in
  ``ArtifactRecord.metadata["axis_order"]``.
- The returned field is the raw, real-valued, time-domain field snapshot at
  the final simulation time step (``arrays_out.fields.E``/``H``), not a
  monochromatic phasor. This adapter does not attempt to extract a
  phasor/near-field surface (that would require ``PhasorDetector``, which
  has not been probed against this pinned version -- see
  ``capability_notes.md`` "Not yet exercised") and does not attempt to
  compute absorbed power density (would require a Poynting-flux/absorption
  calculation, also not probed). Only the ``vector_field`` output port is
  populated; ``near_field`` and ``absorbed_power`` are deliberately left
  absent from ``ModelRunResult.outputs`` rather than fabricated.
- The monitor/field output is **not power-normalized**. This adapter
  appends an explicit warning to ``ModelRunResult.warnings`` rather than
  silently normalizing or silently leaving it ambiguous.
- Units are SI throughout (meters, seconds), matching
  ``knowledge/solvers/fdtdx/conventions.md`` "Units".

Material input handling
------------------------
The registry declares a ``material`` input port (``material_field``
artifact). Building a spatially-varying material field would require the
untested ``Device``/``ParameterContainer`` differentiable-construction
pattern (see module docstring above) even for a forward-only run, since
that is FDTDX's real mechanism for attaching a material array to the
object graph. That pattern has not been worked out in this repository, so
this adapter only supports a spatially **uniform** background permittivity,
supplied either via ``request.config["background_permittivity"]`` /
``request.design_parameters["background_permittivity"]`` (concrete Python
float, default ``1.0``) or via
``request.inputs["material"].metadata["uniform_permittivity"]``. Any other
``material`` input (missing that metadata key, implying a genuinely
spatially-varying field) raises ``UnsupportedCapabilityError`` eagerly.
"""

from __future__ import annotations

import hashlib
import tempfile
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any

from multiscale_optics_agent.adapters.base import (
    CostEstimate,
    ModelRunRequest,
    ModelRunResult,
    RunStatus,
)
from multiscale_optics_agent.core.artifacts import ArtifactRecord
from multiscale_optics_agent.core.errors import (
    AdapterDependencyError,
    SolverExecutionError,
    UnsupportedCapabilityError,
)
from multiscale_optics_agent.core.graph import Severity, ValidationIssue, ValidationReport
from multiscale_optics_agent.core.specs import ArtifactKind, Device, Framework, ModelSpec
from multiscale_optics_agent.registry.loader import Registry

MODEL_ID = "M_EM_FDTDX"

_GRADIENT_CAPABILITY_MESSAGE = (
    "M_EM_FDTDX (fdtdx 0.6.2) has no verified differentiable path. Two "
    "documented gradient probes both failed: jax.grad w.r.t. source "
    "wavelength returns a phantom exact 0.0 (vs. a large nonzero finite- "
    "difference estimate), and jax.grad w.r.t. background permittivity "
    "raises jax.errors.ConcretizationTypeError inside fdtdx.place_objects. "
    "See knowledge/solvers/fdtdx/capability_notes.md and conventions.md "
    "for the full account. This adapter is forward-only; request "
    "require_gradients=False."
)

_SPATIAL_MATERIAL_MESSAGE = (
    "This forward-only M_EM_FDTDX adapter only supports a spatially "
    "uniform, concrete background permittivity (via "
    "config/design_parameters['background_permittivity'] or "
    "inputs['material'].metadata['uniform_permittivity']). Attaching a "
    "genuinely spatially-varying material_field would require fdtdx's "
    "untested Device/ParameterContainer construction pattern -- see "
    "knowledge/solvers/fdtdx/capability_notes.md."
)

# Defaults mirror knowledge/solvers/fdtdx/probes/propagation_probe.py exactly,
# the only configuration validated against the pinned 0.6.2 install (see
# knowledge/solvers/fdtdx/expected/propagation_probe.json).
_DEFAULT_TIME_S = 5e-15
_DEFAULT_RESOLUTION_M = 100e-9
_DEFAULT_BACKEND = "cpu"
_DEFAULT_COURANT_FACTOR = 0.99
_DEFAULT_VOLUME_SHAPE_M = (3.0e-6, 3.0e-6, 3.0e-6)
_DEFAULT_BACKGROUND_PERMITTIVITY = 1.0
_DEFAULT_WAVELENGTH_M = 1.0e-6
_DEFAULT_SOURCE_POLARIZATION = (1, 0, 0)
_DEFAULT_SOURCE_PARTIAL_REAL_SHAPE = (2e-6, 2e-6, None)
_DEFAULT_SOURCE_RADIUS_M = 1e-6
_DEFAULT_SOURCE_STD = 1.0 / 3.0
_DEFAULT_SOURCE_DIRECTION = "+"
_DEFAULT_SEED = 0


def _import_fdtdx() -> tuple[Any, Any, Any]:
    """Lazily import fdtdx/jax/numpy. Never call at module import time."""
    try:
        import fdtdx  # type: ignore[import-untyped]
        import jax
        import jax.numpy as jnp
    except Exception as exc:  # pragma: no cover - exercised only when missing
        raise AdapterDependencyError(
            f"fdtdx/jax could not be imported: {type(exc).__name__}: {exc}"
        ) from exc
    return fdtdx, jax, jnp


@lru_cache(maxsize=1)
def _load_spec() -> ModelSpec:
    return Registry.from_package().models[MODEL_ID]


def _resolve_config(request: ModelRunRequest) -> dict[str, Any]:
    """Merge declared defaults, config, and design_parameters (concrete only).

    design_parameters override config, which overrides the defaults above.
    Every value is coerced through float()/tuple(float(...)) to fail loudly
    (rather than silently accepting) if a caller passes a traced JAX value
    -- this adapter's whole premise is that construction is concrete.
    """
    merged: dict[str, Any] = {
        "time_s": _DEFAULT_TIME_S,
        "resolution_m": _DEFAULT_RESOLUTION_M,
        "backend": _DEFAULT_BACKEND,
        "courant_factor": _DEFAULT_COURANT_FACTOR,
        "volume_shape_m": _DEFAULT_VOLUME_SHAPE_M,
        "background_permittivity": _DEFAULT_BACKGROUND_PERMITTIVITY,
        "wavelength_m": _DEFAULT_WAVELENGTH_M,
        "seed": _DEFAULT_SEED,
    }
    merged.update(request.config)
    merged.update(request.design_parameters)

    try:
        time_s = float(merged["time_s"])
        resolution_m = float(merged["resolution_m"])
        courant_factor = float(merged["courant_factor"])
        volume_shape_m = tuple(float(v) for v in merged["volume_shape_m"])
        background_permittivity = float(merged["background_permittivity"])
        wavelength_m = float(merged["wavelength_m"])
        seed = int(merged["seed"])
        backend = str(merged["backend"])
    except (TypeError, ValueError) as exc:
        raise SolverExecutionError(
            f"M_EM_FDTDX config/design_parameters could not be resolved to "
            f"concrete scalars (this adapter does not support traced "
            f"values): {type(exc).__name__}: {exc}"
        ) from exc

    if len(volume_shape_m) != 3:
        raise SolverExecutionError(
            f"volume_shape_m must have exactly 3 entries, got {volume_shape_m!r}"
        )

    return {
        "time_s": time_s,
        "resolution_m": resolution_m,
        "backend": backend,
        "courant_factor": courant_factor,
        "volume_shape_m": volume_shape_m,
        "background_permittivity": background_permittivity,
        "wavelength_m": wavelength_m,
        "seed": seed,
    }


def _resolve_permittivity(request: ModelRunRequest, resolved_config: dict[str, Any]) -> float:
    material = request.inputs.get("material")
    if material is None:
        return float(resolved_config["background_permittivity"])
    if "uniform_permittivity" not in material.metadata:
        raise UnsupportedCapabilityError(_SPATIAL_MATERIAL_MESSAGE)
    try:
        return float(material.metadata["uniform_permittivity"])
    except (TypeError, ValueError) as exc:
        raise UnsupportedCapabilityError(
            f"{_SPATIAL_MATERIAL_MESSAGE} (metadata['uniform_permittivity'] "
            f"was not a concrete scalar: {exc})"
        ) from exc


class FdtdxAdapter:
    """Forward-only ``ModelAdapter`` for ``M_EM_FDTDX`` (fdtdx 0.6.2)."""

    @property
    def spec(self) -> ModelSpec:
        return _load_spec()

    def estimate(self, request: ModelRunRequest) -> CostEstimate:
        if request.require_gradients:
            raise UnsupportedCapabilityError(_GRADIENT_CAPABILITY_MESSAGE)

        fdtdx, _jax, _jnp = _import_fdtdx()
        resolved = _resolve_config(request)

        # Cheap: constructing SimulationConfig does not run the solver, only
        # computes derived quantities like time_steps_total.
        try:
            config = fdtdx.SimulationConfig(
                time=resolved["time_s"],
                resolution=resolved["resolution_m"],
                backend=resolved["backend"],
                courant_factor=resolved["courant_factor"],
            )
            time_steps_total = int(config.time_steps_total)
        except Exception as exc:
            return CostEstimate(
                confidence="unknown",
                notes=[
                    "Could not construct fdtdx.SimulationConfig to estimate "
                    f"time_steps_total: {type(exc).__name__}: {exc}"
                ],
            )

        grid_dims = tuple(
            max(1, round(extent / resolved["resolution_m"]))
            for extent in resolved["volume_shape_m"]
        )
        grid_cells = 1
        for dim in grid_dims:
            grid_cells *= dim

        # float32, 3 vector components, E and H fields held simultaneously.
        bytes_per_field_snapshot = grid_cells * 3 * 4
        peak_memory_bytes = int(2 * bytes_per_field_snapshot)

        return CostEstimate(
            wall_time_s=None,
            peak_memory_bytes=peak_memory_bytes,
            solver_calls=1,
            confidence="low",
            notes=[
                "Uncalibrated O(time_steps * grid_cells) estimate from "
                "config metadata only; no measured wall-clock timing was "
                "collected for this adapter.",
                f"time_steps_total={time_steps_total}, grid_dims={grid_dims}, "
                f"grid_cells={grid_cells}.",
                "peak_memory_bytes only accounts for E and H field arrays "
                "at one time step, not solver working memory or "
                "checkpointing overhead.",
            ],
        )

    def validate_request(self, request: ModelRunRequest) -> ValidationReport:
        issues: list[ValidationIssue] = []

        if request.require_gradients:
            issues.append(
                ValidationIssue(
                    severity=Severity.ERROR,
                    code="UNSUPPORTED_GRADIENTS",
                    message=_GRADIENT_CAPABILITY_MESSAGE,
                    location="request.require_gradients",
                )
            )

        material = request.inputs.get("material")
        if material is not None:
            required_metadata = ["spatial_grid", "permittivity_convention", "length_unit"]
            missing = [key for key in required_metadata if key not in material.metadata]
            if missing:
                issues.append(
                    ValidationIssue(
                        severity=Severity.WARNING,
                        code="MATERIAL_METADATA_INCOMPLETE",
                        message=(
                            f"inputs['material'] is missing registry-required "
                            f"metadata keys {missing}; the M_EM_FDTDX registry "
                            f"port declares requires_metadata={required_metadata}."
                        ),
                        location="inputs.material.metadata",
                    )
                )
            if "uniform_permittivity" not in material.metadata:
                issues.append(
                    ValidationIssue(
                        severity=Severity.ERROR,
                        code="UNSUPPORTED_SPATIAL_MATERIAL",
                        message=_SPATIAL_MATERIAL_MESSAGE,
                        location="inputs.material.metadata",
                    )
                )

        try:
            resolved = _resolve_config(request)
        except SolverExecutionError as exc:
            issues.append(
                ValidationIssue(
                    severity=Severity.ERROR,
                    code="INVALID_CONFIG",
                    message=str(exc),
                    location="config",
                )
            )
            return ValidationReport(issues=issues)

        if resolved["time_s"] <= 0:
            issues.append(
                ValidationIssue(
                    severity=Severity.ERROR,
                    code="INVALID_TIME",
                    message=f"time_s must be positive, got {resolved['time_s']}.",
                    location="config.time_s",
                )
            )
        if resolved["resolution_m"] <= 0:
            issues.append(
                ValidationIssue(
                    severity=Severity.ERROR,
                    code="INVALID_RESOLUTION",
                    message=f"resolution_m must be positive, got {resolved['resolution_m']}.",
                    location="config.resolution_m",
                )
            )
        if not (0 < resolved["courant_factor"] <= 1.0):
            issues.append(
                ValidationIssue(
                    severity=Severity.ERROR,
                    code="INVALID_COURANT_FACTOR",
                    message=(
                        f"courant_factor must be in (0, 1], got {resolved['courant_factor']}."
                    ),
                    location="config.courant_factor",
                )
            )
        if resolved["resolution_m"] > 0:
            for axis, extent in zip("xyz", resolved["volume_shape_m"], strict=True):
                if extent < resolved["resolution_m"]:
                    issues.append(
                        ValidationIssue(
                            severity=Severity.ERROR,
                            code="VOLUME_SMALLER_THAN_RESOLUTION",
                            message=(
                                f"volume_shape_m[{axis}]={extent} is smaller than "
                                f"resolution_m={resolved['resolution_m']}; grid "
                                f"would have zero cells along this axis."
                            ),
                            location="config.volume_shape_m",
                        )
                    )

        return ValidationReport(issues=issues)

    def run(self, request: ModelRunRequest) -> ModelRunResult:
        # Eager, pre-solver capability gate -- must happen before any import
        # of fdtdx/jax so a caller never reaches either documented gradient
        # failure mode through this adapter.
        if request.require_gradients:
            raise UnsupportedCapabilityError(_GRADIENT_CAPABILITY_MESSAGE)

        resolved = _resolve_config(request)
        permittivity = _resolve_permittivity(request, resolved)

        fdtdx, jax, jnp = _import_fdtdx()

        warnings: list[str] = [
            "Output field is the raw solver monitor snapshot and is NOT "
            "power-normalized (no impedance/intensity normalization has "
            "been applied). Treat values as relative field amplitude only.",
            "Only the 'vector_field' output port is populated. 'near_field' "
            "(phasor, reference-plane) and 'absorbed_power' are not "
            "produced by this adapter -- PhasorDetector and "
            "Poynting-flux/absorption extraction have not been probed "
            "against fdtdx 0.6.2 in this repository.",
        ]

        try:
            key = jax.random.PRNGKey(resolved["seed"])

            config = fdtdx.SimulationConfig(
                time=resolved["time_s"],
                resolution=resolved["resolution_m"],
                backend=resolved["backend"],
                dtype=jnp.float32,
                courant_factor=resolved["courant_factor"],
            )

            constraints: list[Any] = []
            object_list: list[Any] = []

            volume = fdtdx.SimulationVolume(
                partial_real_shape=resolved["volume_shape_m"],
                material=fdtdx.Material(permittivity=permittivity),
            )
            object_list.append(volume)

            bound_cfg = fdtdx.BoundaryConfig.from_uniform_bound(boundary_type="periodic")
            bound_dict, boundary_constraints = fdtdx.boundary_objects_from_config(bound_cfg, volume)
            constraints.extend(boundary_constraints)
            object_list.extend(list(bound_dict.values()))

            source = fdtdx.GaussianPlaneSource(
                partial_grid_shape=(None, None, 1),
                partial_real_shape=_DEFAULT_SOURCE_PARTIAL_REAL_SHAPE,
                fixed_E_polarization_vector=_DEFAULT_SOURCE_POLARIZATION,
                wave_character=fdtdx.WaveCharacter(wavelength=resolved["wavelength_m"]),
                radius=_DEFAULT_SOURCE_RADIUS_M,
                std=_DEFAULT_SOURCE_STD,
                direction=_DEFAULT_SOURCE_DIRECTION,
            )
            constraints.append(
                source.place_relative_to(
                    volume,
                    axes=(0, 1, 2),
                    own_positions=(0, 0, 0),
                    other_positions=(0, 0, 0),
                )
            )
            object_list.append(source)

            energy_detector = fdtdx.EnergyDetector(name="energy")
            constraints.extend(energy_detector.same_position_and_size(volume))
            object_list.append(energy_detector)

            key, subkey = jax.random.split(key)
            objects, arrays, params, config, _place_info = fdtdx.place_objects(
                object_list=object_list, config=config, constraints=constraints, key=subkey
            )

            arrays2, new_objects, _apply_info = fdtdx.apply_params(arrays, objects, params, subkey)
            final_state = fdtdx.run_fdtd(
                arrays=arrays2, objects=new_objects, config=config, key=subkey
            )
            _, arrays_out = final_state

            e_field = arrays_out.fields.E
            h_field = arrays_out.fields.H
            time_steps_total = int(config.time_steps_total)
        except (AdapterDependencyError, UnsupportedCapabilityError):
            raise
        except Exception as exc:
            return ModelRunResult(
                status=RunStatus.FAILED,
                error_type=type(exc).__name__,
                error_message=str(exc),
                warnings=warnings,
                diagnostics={"resolved_config": resolved, "permittivity": permittivity},
            )

        any_nan = bool(jnp.any(jnp.isnan(e_field)) or jnp.any(jnp.isnan(h_field)))
        max_abs_e = float(jnp.max(jnp.abs(e_field)))
        if any_nan:
            return ModelRunResult(
                status=RunStatus.FAILED,
                error_type="SolverExecutionError",
                error_message="fdtdx produced NaN field values.",
                warnings=warnings,
                diagnostics={
                    "resolved_config": resolved,
                    "permittivity": permittivity,
                    "max_abs_E": max_abs_e,
                },
            )

        # Host copy / derivative-boundary note: converting the JAX
        # array to NumPy here is a real host copy, recorded explicitly. This
        # is a forward-only adapter (require_gradients is rejected above),
        # so no differentiable path is broken by this conversion.
        import numpy as np

        e_field_np = np.asarray(jax.device_get(e_field))
        h_field_np = np.asarray(jax.device_get(h_field))

        run_dir = Path(tempfile.mkdtemp(prefix=f"fdtdx_{request.run_id}_{request.node_id}_"))
        e_path = run_dir / "E_field.npy"
        h_path = run_dir / "H_field.npy"
        np.save(e_path, e_field_np)
        np.save(h_path, h_field_np)
        artifact_sha256 = hashlib.sha256(e_path.read_bytes()).hexdigest()

        vector_field_artifact = ArtifactRecord(
            id=f"{request.node_id}-vector_field-{uuid.uuid4().hex[:8]}",
            kind=ArtifactKind.VECTOR_FIELD,
            uri=str(e_path),
            sha256=artifact_sha256,
            shape=tuple(e_field_np.shape),
            dtype=str(e_field_np.dtype),
            framework=Framework.NUMPY,
            device=Device.CPU,
            units="V/m (uncalibrated; not power-normalized, see warnings)",
            metadata={
                "axis_order": "(component, x, y, z)",
                "field_components": ["Ex", "Ey", "Ez"],
                "h_field_uri": str(h_path),
                "h_field_axis_order": "(component, x, y, z)",
                "wavelength": resolved["wavelength_m"],
                "sample_pitch": resolved["resolution_m"],
                "length_unit": "meter",
                "phasor": (
                    "not_applicable: real-valued time-domain field snapshot "
                    "at the final simulation time step, not a monochromatic "
                    "phasor"
                ),
                "time_domain": True,
                "time_steps_total": time_steps_total,
                "final_time_s": resolved["time_s"],
                "background_permittivity": permittivity,
                "permittivity_convention": "relative permittivity, isotropic scalar",
                "coordinate_frame": (
                    "matches fdtdx's internal Yee-grid (x, y, z) axis labeling; "
                    "handedness/orientation relative to a lab frame has not "
                    "been independently confirmed in this repository"
                ),
                "power_normalized": False,
                "boundary_type": "periodic",
                "backend": resolved["backend"],
            },
        )

        return ModelRunResult(
            status=RunStatus.SUCCEEDED,
            outputs={"vector_field": vector_field_artifact},
            diagnostics={
                "resolved_config": resolved,
                "permittivity": permittivity,
                "time_steps_total": time_steps_total,
                "max_abs_E": max_abs_e,
                "any_nan": any_nan,
                "jax_devices": [str(d) for d in jax.devices()],
                "jax_default_backend": jax.default_backend(),
                "host_copy": (
                    "E/H fields converted from JAX device arrays to NumPy "
                    "via jax.device_get for on-disk artifact persistence "
                    "(forward-only path; no gradient claim through this "
                    "boundary)."
                ),
            },
            warnings=warnings,
        )


def get_adapter() -> FdtdxAdapter:
    return FdtdxAdapter()
