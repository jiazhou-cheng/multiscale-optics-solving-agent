"""``C_PATCH_WFT`` as an executable graph edge (CHE-96).

``couplers/patch.py`` implements the general ray-DOE method: each incident ray
extracts its own local complex patch of the DOE, transforms it, and emits
secondary rays sampled from that patch's angular spectrum. This module makes it
reachable from a graph, and like ``couplers/node.py`` and ``couplers/doe_node.py``
it is deliberately a **wrapper** -- it reads and writes ``ArtifactRecord`` and
adds no physics, which is what keeps the core's evidence binding on what a graph
executes.

Why this edge is stricter than ``C_PLANAR_DOE_STEP``
-----------------------------------------------------
It refuses three things that edge accepts, and each refusal is the graph-level
form of a mistake that produced a plausible wrong field during development:

* **An even ``patch_px``.** The paper's own sizes are 40, 50 and 100, so this is
  the refusal a caller is most likely to hit, and it is the one most worth
  hitting: an even patch has no centre sample, so "centred on a ray" is
  undefined for it.
* **Caller-supplied patch centres with no declared coverage basis.** The
  ``A_draw / A_patch`` correction is unbiased only for a known sampling density.
  Guessing one scales the whole field by a constant.
* **A non-planar substrate.** Refused, not approximated. On a curved substrate
  the SI S2 exactness relation does not hold at all, and the planar case's
  confidence must not be inherited by the curved one.

``pad_factor`` is a preference, not an instruction
---------------------------------------------------
The edge derives ``pad_px`` and reports the value it used in the diagnostics. A
caller asking for a pad that violates clearance gets a larger one rather than a
field that is wrong by 100%, and can see that it happened.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from core.boundary import ContractCode, ContractError, ReferencePlane
from core.execution import CostEstimate, RunStatus
from core.graph import Severity, ValidationIssue, ValidationReport
from core.specs import ArtifactKind, CouplerSpec
from couplers.base import DEFAULT_SOURCE_PORT, CouplerRunRequest, CouplerRunResult
from couplers.doe_node import DOE_PORT, _load_bundle, _load_transmission
from couplers.node import _grid_shape, _sample_pitch
from couplers.patch import (
    CoverageBasis,
    Substrate,
    patch_secondary_rays,
    plan_patches,
)

__all__ = ["COUPLER_ID", "DOE_PORT", "PatchWftCoupler", "get_coupler"]

COUPLER_ID = "C_PATCH_WFT"

_ISSUE_CODE_PREFIX = "COUPLER_"


class PatchWftCoupler:
    """Runnable ``C_PATCH_WFT``: rays plus a DOE in, secondary rays out."""

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
        """Every refusal reachable without transforming a single patch."""
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
                errors.append(
                    getattr(exc, "error", None)
                    or ContractError(
                        ContractCode.MISSING_DECLARATION, str(exc), declaration="config"
                    )
                )

        if config.get("plane_z_m") is None:
            errors.append(
                ContractError(
                    ContractCode.MISSING_DECLARATION,
                    "config['plane_z_m'] is required: the patches live on a declared "
                    "plane and the edge does not choose one",
                    declaration="config.plane_z_m",
                    remedy=(
                        "State where the DOE is. The edge does NOT propagate the "
                        "incident bundle to the plane."
                    ),
                )
            )

        patch_px = config.get("patch_px")
        if patch_px is None:
            errors.append(
                ContractError(
                    ContractCode.MISSING_DECLARATION,
                    "config['patch_px'] is required: patch size is the memory/accuracy "
                    "dial this operator exists to expose, and there is no default",
                    declaration="config.patch_px",
                    remedy=(
                        "Use the largest patch that fits the available memory (the "
                        "paper's guidance), or the full grid width for the "
                        "exactness anchor."
                    ),
                )
            )
        elif int(patch_px) % 2 == 0:
            errors.append(
                ContractError(
                    ContractCode.SHAPE_MISMATCH,
                    f"config['patch_px']={patch_px} is even, so the patch has no "
                    "centre sample and cannot be centred in an odd padded grid",
                    declaration="config.patch_px",
                    remedy=(
                        f"Use {int(patch_px) + 1}. The paper's sizes (40, 50, 100) are "
                        "all even, so this is refused rather than rounded -- a caller "
                        "transcribing one should be told which value actually ran."
                    ),
                )
            )

        substrate = str(config.get("substrate", Substrate.PLANAR.value))
        if substrate != Substrate.PLANAR.value:
            errors.append(
                ContractError(
                    ContractCode.MISSING_DECLARATION,
                    f"config['substrate']={substrate!r} is not implemented",
                    declaration="config.substrate",
                    remedy=(
                        "On a curved substrate every patch has its own tangent frame "
                        "and normal, the SI S2 exactness relation does not hold, and "
                        "only the bound eps_curv <= arcsin(D/2R) remains. That is a "
                        "characterization, not this edge's gate."
                    ),
                )
            )

        placement = str(config.get("patch_placement", "incident_positions"))
        if placement not in {"incident_positions", "drawn", "full_aperture"}:
            errors.append(
                ContractError(
                    ContractCode.MISSING_DECLARATION,
                    f"config['patch_placement']={placement!r} is not one of "
                    "'incident_positions', 'drawn', 'full_aperture'",
                    declaration="config.patch_placement",
                )
            )
        if placement == "incident_positions" and not config.get("coverage_basis"):
            errors.append(
                ContractError(
                    ContractCode.MISSING_DECLARATION,
                    "placing patches on the incident rays needs "
                    "config['coverage_basis']: the A_draw / A_patch correction is "
                    "unbiased only for a known sampling density, and the density "
                    "cannot be read back off the positions",
                    declaration="config.coverage_basis",
                    remedy=(
                        "Declare 'uniform_over_dilated_aperture' if the incident rays "
                        "were drawn that way. A guessed density scales the whole field "
                        "by a constant and looks entirely plausible."
                    ),
                )
            )
        if placement == "drawn":
            if config.get("patch_count") is None:
                errors.append(
                    ContractError(
                        ContractCode.MISSING_DECLARATION,
                        "patch_placement='drawn' needs config['patch_count']",
                        declaration="config.patch_count",
                    )
                )
            if config.get("seed") is None:
                errors.append(
                    ContractError(
                        ContractCode.MISSING_DECLARATION,
                        "drawing patch centres needs config['seed']; the protocol "
                        "requires an explicit seed rather than an implicit one",
                        declaration="config.seed",
                    )
                )

        secondary = config.get("secondary_count")
        if secondary is not None and config.get("seed") is None:
            errors.append(
                ContractError(
                    ContractCode.MISSING_DECLARATION,
                    "stochastic secondary sampling needs config['seed']",
                    declaration="config.seed",
                    remedy=(
                        "Set a seed, or omit secondary_count to take the deterministic "
                        "full-enumeration limit."
                    ),
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
                            "Promote the bundle explicitly first. This edge resets OPL "
                            "to zero, so a guessed reference would be absorbed into the "
                            "patch phases and could not be audited afterwards."
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
                        "require_gradients=True is refused. This edge inherits "
                        "C_WAVE_TO_RAY's fixed-direction estimator, a deliberately "
                        "biased surrogate; the registry records "
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
        """Two terms, and the *downstream* one is the one that hurts.

        The transform is ``patches x pad^2 log pad``. Emission is
        ``patches x S``. Neither is the reason a patch run is expensive: the
        reconstruction that consumes this bundle is ``rays x pixels``, and at
        the paper's Table S2 parameters that is 1.6e8 against 1e4. Reporting
        only this edge's own cost would understate the run by orders of
        magnitude, so the emitted ray count is reported as the quantity the
        caller has to budget against.
        """
        config = dict(request.config)
        notes: list[str] = []
        patch_px = config.get("patch_px")
        try:
            grid = _grid_shape(config)
        except Exception:
            grid = None
        if patch_px is None or grid is None:
            notes.append(
                "patch size or grid shape unavailable; no cost estimate is invented."
            )
            return CostEstimate(solver_calls=1, confidence="low", notes=notes)

        record = request.sources.get(DEFAULT_SOURCE_PORT)
        placement = str(config.get("patch_placement", "incident_positions"))
        if placement == "full_aperture":
            patches = 1
        elif placement == "drawn":
            patches = int(config.get("patch_count") or 0)
        else:
            patches = int(record.shape[0]) if record is not None and record.shape else 0

        pad = int(patch_px) * int(config.get("pad_factor", 2))
        secondary = config.get("secondary_count")
        per_patch = int(secondary) if secondary is not None else pad * pad
        emitted = patches * per_patch
        notes.append(
            f"{patches} patch transforms of {pad}^2; {emitted} secondary rays emitted. "
            "The dominant cost is DOWNSTREAM: reconstruction is O(rays x pixels), so "
            f"{emitted} rays against a 1e4-pixel sensor is {emitted * 10000:.2e} "
            "ray-pixel products. Batch accordingly."
        )
        return CostEstimate(
            peak_memory_bytes=16 * (pad * pad + 4 * emitted),
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
        placement = str(config.get("patch_placement", "incident_positions"))

        try:
            bundle = _load_bundle(record)
            transmission = np.asarray(_load_transmission(request, config))
            centers: np.ndarray[Any, Any] | None = None
            if placement == "incident_positions":
                centers = np.asarray(bundle.positions_m, dtype=np.float64)[:, :2]

            plan = plan_patches(
                grid_shape=grid_shape,
                sample_pitch_m=pitch,
                patch_px=int(config["patch_px"]),
                pad_factor=int(config.get("pad_factor", 2)),
                patch_count=(
                    int(config["patch_count"]) if placement == "drawn" else None
                ),
                centers_xy_m=centers,
                coverage_basis=CoverageBasis(
                    str(config.get("coverage_basis", CoverageBasis.UNKNOWN.value))
                ),
                substrate=Substrate(str(config.get("substrate", Substrate.PLANAR.value))),
                radius_m=float(config.get("radius_m", math.inf)),
                error_threshold_rad=float(config.get("error_threshold_rad", 1e-3)),
                rng=rng,
            )
            outgoing, diagnostics = patch_secondary_rays(
                transmission,
                plan=plan,
                sample_pitch_m=pitch,
                wavelength_m=float(bundle.wavelength_m),
                plane=plane,
                secondary_count=(
                    int(config["secondary_count"])
                    if config.get("secondary_count") is not None
                    else None
                ),
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
                        "supposed to agree, so this is a defect in PatchWftCoupler."
                    ),
                    "refusals": [error.as_diagnostic()],
                },
            )

        output_root = (
            Path(str(config.get("output_dir", "runs"))) / request.run_id / request.edge_id
        )
        target = outgoing.to_artifact_record(
            artifact_id=f"{request.edge_id}:ray_bundle",
            uri=output_root / "secondary_rays.npz",
        )

        warnings: list[str] = []
        requested_pad = int(config["patch_px"]) * int(config.get("pad_factor", 2))
        if plan.pad_px != requested_pad:
            warnings.append(
                f"pad_factor asked for pad {requested_pad}; the edge used "
                f"{plan.pad_px}. pad_factor is a preference and the pad is derived "
                "from clearance, centring and oddness -- a pad that violates "
                "clearance produces a plausible field that is wrong by 100%."
            )
        if diagnostics.evanescent_modes > 0:
            warnings.append(
                f"{diagnostics.evanescent_modes} of {plan.pad_px ** 2} modes are "
                "evanescent and carry no propagation direction, so they are not "
                "represented in the outgoing rays. This is a real loss, reported."
            )
        if not diagnostics.enumerated:
            warnings.append(
                "secondary directions were sampled, so this bundle is one Monte "
                "Carlo realization. A single realization is not a result: report an "
                "ensemble, per SI S6."
            )

        return CouplerRunResult(
            status=RunStatus.SUCCEEDED,
            target=target,
            warnings=warnings,
            diagnostics={
                "coupler": COUPLER_ID,
                "edge_id": request.edge_id,
                "patch": diagnostics.as_dict(),
                "patch_placement": placement,
                "coverage_basis": str(
                    config.get("coverage_basis", CoverageBasis.UNKNOWN.value)
                ),
                "pad_requested": requested_pad,
                "pad_used": plan.pad_px,
                "opl_convention": (
                    "reset to zero at the patch plane; each patch's phase is already "
                    "in its spectral amplitude"
                ),
                "source_artifact_id": record.id,
                "gradient_claim": (
                    "none. derivative.mode=surrogate, verified=false: the secondary "
                    "directions are held fixed, so the estimator is knowingly biased."
                ),
            },
        )


def get_coupler() -> PatchWftCoupler:
    return PatchWftCoupler()
