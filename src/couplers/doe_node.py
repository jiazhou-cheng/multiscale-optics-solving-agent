"""``C_PLANAR_DOE_STEP`` as an executable graph edge (CHE-95).

``couplers/cascade.py::planar_doe_step`` has implemented SI Algorithm S1 since
CHE-26 and was **library-only**: no registry entry, no capability declaration,
no graph node, not exported from ``couplers/__init__.py``. Its only callers were
one test file and a benchmark runner. So the one operator that bounds ray count
across stacked DOEs could not appear in a graph.

This module makes it reachable. Like ``couplers/node.py`` it is deliberately a
**wrapper**: it reads and writes ``ArtifactRecord`` and adds no physics, which
is what keeps the core's verification evidence -- the full-enumeration exactness
limit above all -- binding on what a graph executes.

Why a ray -> ray edge is still a coupler
----------------------------------------
The source and target artifacts are both ``ray_bundle``, which at first reading
makes this look like a model rather than a coupler. It is a coupler because it
*changes representation and back*, and carries physical assumptions belonging to
neither side: the accumulation onto a common plane is only valid because the
surface is planar, and the interference that survives it is the entire reason
the step exists. A model would leave the representation alone.

What the edge declares, and why the declarations are the interesting part
-------------------------------------------------------------------------
Three things change across the step and none is visible in an intensity:

* **OPL resets to zero.** The incident path is already in the accumulated
  field's phase; carrying it forward would double-count it. The consequence is
  that the phase reference is rebased to this plane.
* **Amplitude becomes a spectral amplitude.** ``U~[m]/p[m]``, with no per-ray
  correspondence to the incident weights.
* **Power is not conserved unless asked.** ``preserve_energy`` is off by
  default because a lossy DOE legitimately loses power, and renormalizing hides
  exactly the case a conservation check exists to catch.

All three are emitted in the result's diagnostics on every run, not only in this
docstring, because a consumer reads the record.

Refusal, not repair
-------------------
Every precondition returns a ``CouplerRunResult`` with ``status=FAILED``, a
``ContractCode`` and a remedy. ``validate_request`` and ``transform`` call the
same :meth:`PlanarDoeStepCoupler.diagnose`, so they cannot disagree about which
requests are acceptable -- the alternative, two parallel checklists, is how a
validator comes to bless a request that then fails.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from core.artifacts import ArtifactRecord
from core.boundary import ContractCode, ContractError, RayBundle, ReferencePlane
from core.execution import CostEstimate, RunStatus
from core.graph import Severity, ValidationIssue, ValidationReport
from core.specs import ArtifactKind, CouplerSpec
from couplers.base import DEFAULT_SOURCE_PORT, CouplerRunRequest, CouplerRunResult
from couplers.cascade import PrimarySampling, planar_doe_step
from couplers.node import _grid_shape, _sample_pitch
from couplers.wave_to_ray import SamplingDensity

__all__ = ["COUPLER_ID", "DOE_PORT", "PlanarDoeStepCoupler", "get_coupler"]

COUPLER_ID = "C_PLANAR_DOE_STEP"

#: The second source port. The step consumes an incident bundle *and* a DOE
#: transmission on the same plane, which is why `CouplerRunRequest` grew a
#: `sources` mapping: a single unnamed source cannot express it.
DOE_PORT = "doe_transmission"

_ISSUE_CODE_PREFIX = "COUPLER_"


class PlanarDoeStepCoupler:
    """Runnable ``C_PLANAR_DOE_STEP``: rays plus a DOE in, rays out."""

    def __init__(self, spec: CouplerSpec | None = None) -> None:
        self._spec = spec

    @property
    def spec(self) -> CouplerSpec:
        if self._spec is None:
            from registry.loader import Registry

            self._spec = Registry.from_package().couplers[COUPLER_ID]
        return self._spec

    # ------------------------------------------------------------------
    # Preconditions
    # ------------------------------------------------------------------
    def diagnose(self, request: CouplerRunRequest) -> list[ContractError]:
        """Every refusal reachable without running the accumulation."""
        errors: list[ContractError] = []
        config = dict(request.config)

        record = request.sources.get(DEFAULT_SOURCE_PORT)
        if record is None:
            errors.append(
                ContractError(
                    ContractCode.MISSING_DECLARATION,
                    f"no {DEFAULT_SOURCE_PORT!r} port: the incident ray bundle",
                    declaration="sources",
                )
            )
        elif record.kind is not ArtifactKind.RAY_BUNDLE:
            errors.append(
                ContractError(
                    ContractCode.ARTIFACT_KIND_MISMATCH,
                    f"{COUPLER_ID} consumes {ArtifactKind.RAY_BUNDLE.value!r} on "
                    f"{DEFAULT_SOURCE_PORT!r}, got {record.kind.value!r}",
                    declaration=f"sources.{DEFAULT_SOURCE_PORT}.kind",
                )
            )

        if DOE_PORT not in request.sources and "doe_transmission_uri" not in config:
            errors.append(
                ContractError(
                    ContractCode.MISSING_DECLARATION,
                    "no DOE transmission: supply it on the "
                    f"{DOE_PORT!r} port or as config['doe_transmission_uri']",
                    declaration=f"sources.{DOE_PORT}",
                    remedy=(
                        "The transmission must be complex. A real array is an "
                        "amplitude mask with an undeclared phase, not a transmission."
                    ),
                )
            )

        for check in (_grid_shape, _sample_pitch):
            try:
                check(config)
            except Exception as exc:  # _Refusal from couplers.node
                errors.append(getattr(exc, "error", None) or ContractError(
                    ContractCode.MISSING_DECLARATION, str(exc), declaration="config"
                ))

        if config.get("plane_z_m") is None:
            errors.append(
                ContractError(
                    ContractCode.MISSING_DECLARATION,
                    "config['plane_z_m'] is required: the step accumulates onto a "
                    "declared plane and does not choose one",
                    declaration="config.plane_z_m",
                    remedy=(
                        "State where the DOE is. The step does NOT propagate the "
                        "incident bundle to the plane -- the bundle must already be "
                        "expressed there."
                    ),
                )
            )

        has_positions = config.get("launch_positions_xy_m") is not None
        has_sampler = config.get("primary_sampling") is not None
        if has_positions and has_sampler:
            errors.append(
                ContractError(
                    ContractCode.MISSING_DECLARATION,
                    "config['launch_positions_xy_m'] and config['primary_sampling'] "
                    "both specify where the outgoing rays launch from",
                    declaration="config.launch_positions_xy_m",
                    remedy="Supply exactly one.",
                )
            )
        elif not has_positions and not has_sampler:
            errors.append(
                ContractError(
                    ContractCode.MISSING_DECLARATION,
                    "the edge declares no primary launch positions",
                    declaration="config.launch_positions_xy_m",
                    remedy=(
                        "Set config['launch_positions_xy_m'] as a (P, 2) list, or "
                        "config['primary_sampling'] ('uniform_on_grid' or "
                        "'incident_positions') with config['primary_count']. There is "
                        "no default: where the outgoing rays launch from is a modelling "
                        "choice, and the uniform option is not established as the right "
                        "one -- see the coupler card."
                    ),
                )
            )
        elif has_sampler and config.get("primary_count") is None:
            errors.append(
                ContractError(
                    ContractCode.MISSING_DECLARATION,
                    "config['primary_sampling'] needs config['primary_count']",
                    declaration="config.primary_count",
                )
            )
        if has_sampler and str(config["primary_sampling"]) == "uniform_on_grid" \
                and config.get("seed") is None:
            errors.append(
                ContractError(
                    ContractCode.MISSING_DECLARATION,
                    "uniform_on_grid draws positions and so needs config['seed']",
                    declaration="config.seed",
                )
            )

        if record is not None and record.kind is ArtifactKind.RAY_BUNDLE:
            metadata = record.metadata
            if not (
                metadata.get("optical_path_length_field")
                and metadata.get("amplitude_field")
                and metadata.get("optical_path_length_reference")
            ):
                errors.append(
                    ContractError(
                        ContractCode.OPL_REFERENCE_UNVERIFIED,
                        "the incident record does not declare both an amplitude and an "
                        "optical path length with its reference",
                        declaration="metadata.optical_path_length_reference",
                        remedy=(
                            "Promote the bundle explicitly first. This step resets OPL to "
                            "zero, so a guessed reference would be absorbed into the "
                            "accumulated phase and could not be audited afterwards -- which "
                            "is why this edge is stricter than C_RAY_TO_WAVE, where the "
                            "promotion survives into the field's phase with the record "
                            "still naming the plane."
                        ),
                    )
                )

        secondary = config.get("secondary_count")
        if secondary is not None and int(secondary) >= 2 and config.get("seed") is None:
            errors.append(
                ContractError(
                    ContractCode.MISSING_DECLARATION,
                    "stochastic resampling needs config['seed']; the protocol requires "
                    "an explicit seed rather than an implicit one",
                    declaration="config.seed",
                    remedy=(
                        "Set a seed, or omit secondary_count to take the deterministic "
                        "full-enumeration limit."
                    ),
                )
            )
        return errors

    def validate_request(self, request: CouplerRunRequest) -> ValidationReport:
        issues = [
            ValidationIssue(
                severity=Severity.ERROR,
                code=f"{_ISSUE_CODE_PREFIX}{error.code}",
                message=str(error),
                location=f"edges.{request.edge_id}",
            )
            for error in self.diagnose(request)
        ]
        if request.require_gradients:
            issues.append(
                ValidationIssue(
                    severity=Severity.ERROR,
                    code=f"{_ISSUE_CODE_PREFIX}GRADIENT_NOT_VERIFIED",
                    message=(
                        "require_gradients=True is refused. This step inherits "
                        "C_WAVE_TO_RAY's fixed-direction estimator, which is a "
                        "deliberately biased surrogate; the registry records "
                        "derivative.mode=surrogate with verified=false, and "
                        "registering the step as a graph node did not change that."
                    ),
                    location=f"edges.{request.edge_id}",
                )
            )
        if not issues:
            issues.append(
                ValidationIssue(
                    severity=Severity.INFO,
                    code=f"{_ISSUE_CODE_PREFIX}REQUEST_VALID",
                    message=f"Request satisfies the {COUPLER_ID} contract.",
                    location=f"edges.{request.edge_id}",
                )
            )
        return ValidationReport(issues=issues)

    # ------------------------------------------------------------------
    # Cost
    # ------------------------------------------------------------------
    def estimate(self, request: CouplerRunRequest) -> CostEstimate:
        """Three terms, and the first one usually dominates.

        Accumulation is ``rays x pixels``; the transform is
        ``pixels log pixels``; resampling is ``P x S``. Reporting only the
        budget would understate a step whose incident count is large, which is
        precisely the configuration this operator exists to make survivable.
        """
        config = request.config
        notes: list[str] = []
        record = request.sources.get(DEFAULT_SOURCE_PORT)
        rays = int(record.shape[0]) if record is not None and record.shape else None
        grid: tuple[int, int] | None
        try:
            grid = _grid_shape(dict(config))
        except Exception:
            grid = None

        if rays is None or grid is None:
            notes.append("ray count or grid shape unavailable; no cost estimate is invented.")
            return CostEstimate(solver_calls=1, confidence="low", notes=notes)

        ny, nx = grid
        pixels = ny * nx
        budget = int(config.get("primary_count", 1)) * int(config.get("secondary_count") or pixels)
        notes.append(
            f"accumulation {rays} x {pixels} ray-pixel products dominates; transform "
            f"{pixels} log {pixels}; resampling {budget} rays. The outgoing count is the "
            "budget and does not depend on the incident count -- that is the property "
            "this step exists for."
        )
        return CostEstimate(
            peak_memory_bytes=16 * (rays * ny + rays * nx + pixels + budget),
            solver_calls=1,
            confidence="low",
            notes=notes,
        )

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------
    def transform(self, request: CouplerRunRequest) -> CouplerRunResult:
        errors = self.diagnose(request)
        if errors:
            first = errors[0]
            return CouplerRunResult(
                status=RunStatus.FAILED,
                error_type=str(first.code),
                error_message=str(first),
                diagnostics={
                    "coupler": COUPLER_ID,
                    "edge_id": request.edge_id,
                    "refusals": [error.as_diagnostic() for error in errors],
                },
            )

        config = dict(request.config)
        record = request.sources[DEFAULT_SOURCE_PORT]
        grid_shape = _grid_shape(config)
        pitch = _sample_pitch(config)
        plane = ReferencePlane(
            name=str(config.get("plane_name", "doe")), z_m=float(config["plane_z_m"])
        )
        seed = config.get("seed")
        rng = np.random.default_rng(int(seed)) if seed is not None else None

        primary_sampling = config.get("primary_sampling")
        launch = config.get("launch_positions_xy_m")

        try:
            bundle = _load_bundle(record)
            transmission = _load_transmission(request, config)
            outgoing, transmitted, diagnostics = planar_doe_step(
                bundle,
                transmission,
                grid_shape=grid_shape,
                sample_pitch_m=pitch,
                plane=plane,
                launch_positions_xy_m=(
                    np.asarray(launch, dtype=np.float64) if launch is not None else None
                ),
                primary_sampling=(
                    PrimarySampling(str(primary_sampling)) if primary_sampling else None
                ),
                primary_count=(
                    int(config["primary_count"]) if config.get("primary_count") else None
                ),
                secondary_count=(
                    int(config["secondary_count"])
                    if config.get("secondary_count") is not None
                    else None
                ),
                density_kind=SamplingDensity(
                    str(config.get("density_kind", SamplingDensity.UNIFORM.value))
                ),
                preserve_energy=bool(config.get("preserve_energy", False)),
                pad_width=int(config.get("pad_width", 0)),
                rng=rng,
            )
        except ContractError as error:
            return CouplerRunResult(
                status=RunStatus.FAILED,
                error_type=str(error.code),
                error_message=str(error),
                diagnostics={
                    "coupler": COUPLER_ID,
                    "edge_id": request.edge_id,
                    "undiagnosed_refusal": True,
                    "note": (
                        "validate_request did not predict this refusal; the two are "
                        "supposed to agree, so this is a defect in "
                        "PlanarDoeStepCoupler."
                    ),
                    "refusals": [error.as_diagnostic()],
                },
            )

        output_root = Path(str(config.get("output_dir", "runs"))) / request.run_id / request.edge_id
        target = outgoing.to_artifact_record(
            artifact_id=f"{request.edge_id}:ray_bundle",
            uri=output_root / "outgoing_rays.npz",
        )

        warnings: list[str] = []
        if diagnostics.collapsed_to_mean_wavevector:
            warnings.append(
                "secondary_count <= 1 selected the collapsed preview: one ray along "
                "the power-weighted mean wavevector. This is a preview, not an "
                "approximation with a stated error; do not measure a PSF from it."
            )
        if diagnostics.energy_preservation_factor is not None:
            warnings.append(
                "preserve_energy renormalized the transmitted field by "
                f"{diagnostics.energy_preservation_factor:.6g}. The recorded power is "
                "therefore a policy, not a measurement, and cannot be used to check "
                "whether the DOE conserves power."
            )
        if diagnostics.evanescent_power_fraction > 0.0:
            warnings.append(
                f"{diagnostics.evanescent_power_fraction:.3e} of the transmitted power "
                "is evanescent and carries no propagation direction, so it is not "
                "represented in the outgoing rays. This is a real loss, reported."
            )

        return CouplerRunResult(
            status=RunStatus.SUCCEEDED,
            target=target,
            warnings=warnings,
            diagnostics={
                "coupler": COUPLER_ID,
                "edge_id": request.edge_id,
                "cascade": diagnostics.as_dict(),
                "transmitted_discrete_power": transmitted.discrete_power(),
                "source_artifact_id": record.id,
                "gradient_claim": (
                    "none. derivative.mode=surrogate, verified=false: the secondary "
                    "directions are held fixed, so the estimator is knowingly biased."
                ),
            },
        )


def _load_bundle(record: ArtifactRecord) -> RayBundle:
    """The incident bundle, which must already declare its OPL and amplitude.

    Unlike `C_RAY_TO_WAVE`, this edge will **not** promote an undeclared record
    against a declared plane. The reason is specific rather than conservative:
    this step resets OPL to zero on the way out, so an incident OPL whose
    reference was guessed would be absorbed into the accumulated phase and then
    become unrecoverable. C_RAY_TO_WAVE's promotion is auditable afterwards
    because the OPL survives into the field's phase with the record still
    naming the plane; here it does not survive at all.
    """
    metadata = record.metadata
    opl_field = metadata.get("optical_path_length_field")
    amplitude_field = metadata.get("amplitude_field")
    reference = metadata.get("optical_path_length_reference")
    if not (opl_field and amplitude_field and reference):
        raise ContractError(
            ContractCode.OPL_REFERENCE_UNVERIFIED,
            "the incident record does not declare both an amplitude and an optical "
            "path length with its reference",
            declaration="metadata.optical_path_length_reference",
            remedy=(
                "Promote the bundle explicitly first -- through C_RAY_TO_WAVE's "
                "handoff declaration, or by writing a record that states them. This "
                "step resets OPL to zero, so a guessed reference would be absorbed "
                "into the accumulated phase and could not be audited afterwards."
            ),
        )
    data = dict(np.load(record.uri))
    return (
        RayBundle.from_artifact_record(record, arrays=data)
        .with_amplitude_from_weight(
            mapping=str(metadata.get("amplitude_mapping", "amplitude supplied by the producer")),
            amplitude=data[str(amplitude_field)],
        )
        .with_declared_optical_path_length(data[str(opl_field)], reference=str(reference))
    )


def _load_transmission(
    request: CouplerRunRequest, config: dict[str, Any]
) -> np.ndarray[Any, Any]:
    record = request.sources.get(DOE_PORT)
    if record is not None:
        return np.asarray(np.load(record.uri))
    return np.asarray(np.load(str(config["doe_transmission_uri"])))


def get_coupler() -> PlanarDoeStepCoupler:
    return PlanarDoeStepCoupler()
