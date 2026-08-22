"""CHE-34 (M3.5): ``C_RAY_TO_WAVE`` as an executable graph edge.

``couplers/base.py`` has declared ``Coupler.transform`` since M2 and nothing has
ever defined it, so ``GraphValidator`` has been validating edges that cannot run.
This module defines exactly one, for the M3 slice.

It is deliberately a **wrapper**, and the test suite proves it added nothing: the
field it writes is bit-identical to calling
:func:`couplers.ray_to_wave.ray_to_wave` directly on the
same input. All the physics stayed in the core, which is what keeps M2's
verification evidence -- the 7.82e-14 plane-wave oracle, the enumeration limits,
the independent Chromatix cross-check -- still binding on what a graph executes.

Where the node lives, and why not in the core
---------------------------------------------
``ray_to_wave.py`` may import no solver engine, and ``benchmarks/
coupler_protocol.yaml`` freezes that rule because M1's independence evidence
stops bounding the search if a coupler defect can be misattributed to an engine.
This module imports no engine either -- it reads and writes ``ArtifactRecord`` --
but it does depend on the record layer and on CHE-33's Optiland declarations, so
it is a separate module rather than an addition to the core. The AST and
``sys.modules`` checks cover both.

Refusal, not repair
-------------------
Every precondition returns a :class:`CouplerRunResult` with
``status=FAILED``, a ``ContractCode``, and a remedy. Nothing here raises past the
caller and nothing here fabricates a field. The set of refusals is:

============================  =====================================
Nyquist violation (per axis)  ``SHAPE_MISMATCH``
Undeclared OPL                ``OPL_REFERENCE_UNVERIFIED``
Plane mismatch                ``REFERENCE_PLANE_MISMATCH``
Empty bundle                  ``EMPTY_ENSEMBLE``
Non-finite input              ``NON_FINITE``
Wrong artifact kind           ``ARTIFACT_KIND_MISMATCH``
Missing / unusable config     ``MISSING_DECLARATION``
============================  =====================================

:meth:`RayToWaveCoupler.validate_request` and
:meth:`RayToWaveCoupler.transform` cannot disagree about that set, because both
call the same :meth:`RayToWaveCoupler.diagnose`. A test asserts it over every
refusal above; the alternative -- two parallel checklists -- is how a validator
comes to bless a request that then fails.

The declaration is edge configuration
-------------------------------------
A ray record straight out of the ray model is not coherent, by design (CHE-33).
This node will promote one, but only when the edge explicitly declares the
handoff plane; with no declaration it refuses. That keeps the M3.4 gate intact
while letting the graph express the whole step as one edge, which is what it
physically is: the change of representation *is* where those assumptions enter.

What a downstream edge inherits (CHE-50)
----------------------------------------
The field this edge writes is valid **at** the declared handoff plane and carries
no ``exp(i k r^2 / 2R)`` wavefront-curvature term, because the core's sum is
linear in the transverse coordinate. A graph that reads this record and only
measures an intensity or a PSF is unaffected. A graph that propagates it a
further distance is outside what CHE-24/CHE-38 verified in phase, and the
discrepancy will not show up in ``|U|^2``.

CHE-50 dispositioned this as a tracked known limitation with no kernel change,
to be re-examined when a propagation-sensitive hybrid composition requires it.
It is not silent: the emitted ``ComplexField`` states it in
``provenance["validity"]``, and the reasoning and measured numbers live in
``couplers/ray_to_wave.py``'s module docstring and on the coupler card. To hand
off on a different plane, declare that plane and let the core reconstruct there
from advanced ray state -- do not propagate this record instead.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from core.artifacts import ArtifactRecord
from core.boundary import (
    ContractCode,
    ContractError,
    RayBundle,
)
from core.execution import CostEstimate, RunStatus
from core.graph import Severity, ValidationIssue, ValidationReport
from core.specs import ArtifactKind, CouplerSpec
from couplers.base import (
    DEFAULT_SOURCE_PORT,
    CouplerRunRequest,
    CouplerRunResult,
)
from couplers.handoff import (
    DeclaredHandoffPlane,
    declare_coherent_bundle,
)
from couplers.ray_to_wave import (
    Projection,
    grid_nyquist_direction_limit,
    ray_to_wave,
)

__all__ = ["COUPLER_ID", "RayToWaveCoupler", "get_coupler"]

COUPLER_ID = "C_RAY_TO_WAVE"

#: Measured on this project's CPU by M3.2's feasibility probe and frozen in
#: benchmarks/protocols/slice_protocol.yaml. Used only to turn the cost driver into
#: seconds; the driver itself -- rays x grid points -- is the honest part.
_RAY_PIXEL_PRODUCTS_PER_SECOND = 5.5e8

_ISSUE_CODE_PREFIX = "COUPLER_"


class _Refusal(Exception):
    """Internal carrier so every precondition takes the same exit."""

    def __init__(self, error: ContractError) -> None:
        super().__init__(str(error))
        self.error = error


def _refuse(
    code: ContractCode, message: str, *, declaration: str | None = None, remedy: str | None = None
) -> _Refusal:
    return _Refusal(ContractError(code, message, declaration=declaration, remedy=remedy))


class RayToWaveCoupler:
    """Runnable ``C_RAY_TO_WAVE``: one ``ArtifactRecord`` in, one out."""

    def __init__(self, spec: CouplerSpec | None = None) -> None:
        self._spec = spec

    @property
    def spec(self) -> CouplerSpec:
        if self._spec is None:
            from registry.loader import Registry

            self._spec = Registry.from_package().couplers[COUPLER_ID]
        return self._spec

    # ------------------------------------------------------------------
    # Preconditions -- the single source both validate_request and transform use
    # ------------------------------------------------------------------
    def diagnose(self, request: CouplerRunRequest) -> list[ContractError]:
        """Every refusal this coupler can make without running the sum.

        Returns them all rather than the first, so a caller fixing a request is
        not led through them one at a time. Order is stable: structural problems
        (kind, config) precede physical ones (declaration, sampling), because a
        Nyquist verdict computed from a record of the wrong kind would be noise.
        """
        errors: list[ContractError] = []
        try:
            record = self._require_record(request)
            config = self._require_config(request)
        except _Refusal as refusal:
            return [refusal.error]

        try:
            bundle = self._resolve_bundle(record, config)
        except _Refusal as refusal:
            return [refusal.error]
        except ContractError as error:
            # Raised by the contract layer itself -- an empty bundle, a
            # non-finite array, a unit that is not SI. Reported, not re-wrapped:
            # the code and remedy it already carries are the right ones.
            return [error]

        errors.extend(self._sampling_errors(bundle, config))
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
            # The registry records derivative.mode=finite_difference,
            # verified=false. A forward result would still be produced, so this
            # is a warning on the request rather than a refusal of it -- but it
            # must be said out loud, because AGENTS.md forbids claiming a
            # gradient across an untested boundary.
            issues.append(
                ValidationIssue(
                    severity=Severity.ERROR,
                    code=f"{_ISSUE_CODE_PREFIX}GRADIENT_NOT_VERIFIED",
                    message=(
                        "require_gradients=True is refused: C_RAY_TO_WAVE declares "
                        "derivative.mode=finite_difference with verified=false, and M3 is "
                        "forward-only by protocol. No gradient is claimed across this edge."
                    ),
                    location=f"edges.{request.edge_id}",
                )
            )
        if not issues:
            issues.append(
                ValidationIssue(
                    severity=Severity.INFO,
                    code=f"{_ISSUE_CODE_PREFIX}REQUEST_VALID",
                    message="Request satisfies the C_RAY_TO_WAVE contract.",
                    location=f"edges.{request.edge_id}",
                )
            )
        return ValidationReport(issues=issues)

    # ------------------------------------------------------------------
    # Cost
    # ------------------------------------------------------------------
    def estimate(self, request: CouplerRunRequest) -> CostEstimate:
        """Report the real driver: rays x grid points.

        Not ``O(rays) + O(pixels)``, which is what the registry's cost model said
        and which is wrong by a factor of the grid size. The separable
        contraction in the core turns the naive ``(N, ny, nx)`` tensor into two
        ``(N, n)`` factors, so *memory* is additive -- but the work is not, and a
        graph rejecting an infeasible configuration needs the product.
        """
        notes: list[str] = []
        record = request.sources.get(DEFAULT_SOURCE_PORT)
        config = request.config

        ray_count = None
        if record is not None and record.shape:
            ray_count = int(record.shape[0])
        try:
            ny, nx = _grid_shape(config)
        except _Refusal as refusal:
            notes.append(f"grid unknown: {refusal.error}")
            ny = nx = None

        wall_time_s = None
        peak_memory_bytes = None
        confidence = "low"
        if ray_count is not None and ny is not None and nx is not None:
            products = ray_count * ny * nx
            wall_time_s = products / _RAY_PIXEL_PRODUCTS_PER_SECOND
            # The two separable ramp factors plus the output grid, complex128.
            peak_memory_bytes = 16 * (ray_count * ny + ray_count * nx + ny * nx)
            confidence = "medium"
            notes.append(
                f"cost driver = rays x grid points = {ray_count} x {ny * nx} = {products:.3g} "
                f"ray-pixel products at a measured ~{_RAY_PIXEL_PRODUCTS_PER_SECOND:.1e}/s "
                "(shared unpinned machine, M3.2; an ordering figure, not a regression envelope)"
            )
        else:
            notes.append("ray count or grid shape unavailable; no cost estimate is invented.")

        return CostEstimate(
            wall_time_s=wall_time_s,
            peak_memory_bytes=peak_memory_bytes,
            solver_calls=1,
            confidence=confidence,
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

        record = request.sources[DEFAULT_SOURCE_PORT]
        config = request.config
        bundle = self._resolve_bundle(record, config)
        grid_shape = _grid_shape(config)
        pitch = _sample_pitch(config)
        projection = Projection(str(config.get("projection", Projection.ASM_CONSISTENT.value)))

        try:
            field, diagnostics = ray_to_wave(
                bundle,
                grid_shape=grid_shape,
                sample_pitch_m=pitch,
                projection=projection,
                normalization=config.get("normalization"),
            )
        except ContractError as error:
            # diagnose() is meant to have caught everything reachable here. If it
            # did not, say so plainly instead of presenting it as an ordinary
            # refusal -- an undiagnosed refusal is a defect in this class.
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
                        "supposed to agree, so this is a defect in RayToWaveCoupler."
                    ),
                    "refusals": [error.as_diagnostic()],
                },
            )

        output_root = Path(str(config.get("output_dir", "runs"))) / request.run_id / request.edge_id
        target = field.to_artifact_record(
            artifact_id=f"{request.edge_id}:complex_field",
            uri=output_root / "pupil_field.npy",
        )

        warnings: list[str] = []
        if diagnostics.ray_density_status == "adjacent_ray_phase_step_exceeds_pi":
            warnings.append(
                "adjacent rays disagree by more than pi at the plane, so the wavelet "
                "picture is not locally valid; refine the ray count (this is a ray-density "
                "condition, and refining the grid will not fix it)."
            )

        return CouplerRunResult(
            status=RunStatus.SUCCEEDED,
            target=target,
            warnings=warnings,
            diagnostics={
                "coupler": COUPLER_ID,
                "edge_id": request.edge_id,
                "reconstruction": diagnostics.as_dict(),
                "handoff": bundle.provenance.get("handoff_diagnostics"),
                "declarations": bundle.provenance.get("handoff"),
                "source_artifact_id": record.id,
                "gradient_claim": (
                    "none. Forward-only, per benchmarks/protocols/slice_protocol.yaml; "
                    "derivative.verified is false for this coupler."
                ),
            },
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    @staticmethod
    def _require_record(request: CouplerRunRequest) -> ArtifactRecord:
        try:
            record = request.require_source()
        except KeyError as exc:
            raise _refuse(
                ContractCode.MISSING_DECLARATION,
                str(exc),
                declaration="sources",
                remedy=f"Supply the {DEFAULT_SOURCE_PORT!r} port.",
            ) from None
        if record.kind is not ArtifactKind.RAY_BUNDLE:
            raise _refuse(
                ContractCode.ARTIFACT_KIND_MISMATCH,
                f"C_RAY_TO_WAVE consumes {ArtifactKind.RAY_BUNDLE.value!r}, got "
                f"{record.kind.value!r}",
                declaration="sources.source.kind",
                remedy=(
                    "The wavelet sum needs a direction per sample. wavefront_samples "
                    "carries pupil coordinates only and can never supply one."
                ),
            )
        return record

    @staticmethod
    def _require_config(request: CouplerRunRequest) -> dict[str, Any]:
        config = dict(request.config)
        _grid_shape(config)
        _sample_pitch(config)
        projection = str(config.get("projection", Projection.ASM_CONSISTENT.value))
        if projection not in {member.value for member in Projection}:
            raise _refuse(
                ContractCode.MISSING_DECLARATION,
                f"config['projection']={projection!r} is not a known convention",
                declaration="config.projection",
                remedy=(
                    "Use 'asm_consistent' (field-preserving, correct for a coupler) or "
                    "'sensor_obliquity' (main-text eq 2's detector model)."
                ),
            )
        normalization = config.get("normalization")
        if normalization is not None and normalization not in {"none", "one_over_n"}:
            raise _refuse(
                ContractCode.MISSING_DECLARATION,
                f"config['normalization']={normalization!r} must be 'none' or 'one_over_n'",
                declaration="config.normalization",
                remedy="Omit it to take the bundle's own reconstruction_normalization.",
            )
        return config

    @staticmethod
    def _resolve_bundle(record: ArtifactRecord, config: dict[str, Any]) -> RayBundle:
        """Take a coherent record as-is; promote an undeclared one only on request."""
        metadata = record.metadata
        already_coherent = bool(metadata.get("optical_path_length_field")) and bool(
            metadata.get("amplitude_field")
        )
        if already_coherent:
            reference = metadata.get("optical_path_length_reference")
            if not reference:
                raise _refuse(
                    ContractCode.OPL_REFERENCE_UNVERIFIED,
                    "the record supplies an optical path length array but names no reference "
                    "for it",
                    declaration="metadata.optical_path_length_reference",
                    remedy=(
                        "A producer that writes an OPL must write the plane or ray it is "
                        "measured from. An unlabelled OPL is not usable as a phase."
                    ),
                )
            data = dict(np.load(record.uri))
            return (
                RayBundle.from_artifact_record(record, arrays=data)
                .with_amplitude_from_weight(
                    mapping=str(
                        metadata.get("amplitude_mapping", "amplitude supplied by the producer")
                    ),
                    amplitude=data[str(metadata["amplitude_field"])],
                )
                .with_declared_optical_path_length(
                    data[str(metadata["optical_path_length_field"])], reference=str(reference)
                )
            )

        plane_kind = config.get("handoff_plane")
        plane_z_m = config.get("handoff_plane_z_m")
        if plane_kind is None or plane_z_m is None:
            raise _refuse(
                ContractCode.OPL_REFERENCE_UNVERIFIED,
                (
                    "the source record carries no declared optical path length or amplitude, "
                    "and the edge declares no handoff plane to promote it against"
                ),
                declaration="config.handoff_plane / config.handoff_plane_z_m",
                remedy=(
                    "Either supply a ray bundle that already declares its OPL reference and "
                    "amplitude, or declare config['handoff_plane'] and "
                    "config['handoff_plane_z_m'] on the edge so the OPL convention is stated "
                    "rather than assumed. A coupler will not choose an OPL reference: a wrong "
                    "sign conjugates the wavefront and is indistinguishable downstream."
                ),
            )
        if not isinstance(plane_z_m, (int, float)) or not math.isfinite(float(plane_z_m)):
            raise _refuse(
                ContractCode.MISSING_DECLARATION,
                f"config['handoff_plane_z_m']={plane_z_m!r} is not a finite number of metres",
                declaration="config.handoff_plane_z_m",
            )
        declared = DeclaredHandoffPlane(handoff_plane=str(plane_kind), z_m=float(plane_z_m))
        return declare_coherent_bundle(record, declared_plane=declared).bundle

    @staticmethod
    def _sampling_errors(bundle: RayBundle, config: dict[str, Any]) -> list[ContractError]:
        """The per-axis Nyquist condition, evaluated before the sum is paid for.

        Per axis, not on the direction norm: a diagonal FFT bin has
        ``|d| = sqrt(2) lambda / (2 pitch)`` and is exactly representable. M2 fixed
        that after a round trip could not enumerate its own bins.
        """
        pitch_y, pitch_x = _sample_pitch(config)
        limit_y = grid_nyquist_direction_limit(bundle.wavelength_m, pitch_y)
        limit_x = grid_nyquist_direction_limit(bundle.wavelength_m, pitch_x)
        max_du = float(np.max(np.abs(bundle.directions[:, 0])))
        max_dv = float(np.max(np.abs(bundle.directions[:, 1])))
        if max_du <= limit_x and max_dv <= limit_y:
            return []
        return [
            ContractError(
                ContractCode.SHAPE_MISMATCH,
                (
                    "output grid cannot represent the steepest wavelet ramp: "
                    f"|d_u|max = {max_du:.6f} against limit {limit_x:.6f}, "
                    f"|d_v|max = {max_dv:.6f} against limit {limit_y:.6f} "
                    "(lambda / (2 * pitch), per axis)"
                ),
                declaration="config.sample_pitch_m",
                remedy=(
                    "Refine the output pitch, or restrict the ray directions. Adding more "
                    "rays will not help: this is a grid condition, not a ray-density one."
                ),
            )
        ]


def _grid_shape(config: dict[str, Any]) -> tuple[int, int]:
    shape = config.get("grid_shape")
    if shape is None:
        grid_n = config.get("grid_n")
        if grid_n is None:
            raise _refuse(
                ContractCode.MISSING_DECLARATION,
                "the edge declares no output grid",
                declaration="config.grid_shape / config.grid_n",
                remedy="Set config['grid_n'] (square) or config['grid_shape'] as (ny, nx).",
            )
        shape = (grid_n, grid_n)
    try:
        ny, nx = int(shape[0]), int(shape[1])
    except (TypeError, ValueError, IndexError):
        raise _refuse(
            ContractCode.SHAPE_MISMATCH,
            f"config['grid_shape']={shape!r} is not an (ny, nx) pair",
            declaration="config.grid_shape",
        ) from None
    if ny <= 0 or nx <= 0:
        raise _refuse(
            ContractCode.SHAPE_MISMATCH,
            f"grid shape must be positive, got {(ny, nx)!r}",
            declaration="config.grid_shape",
        )
    return ny, nx


def _sample_pitch(config: dict[str, Any]) -> tuple[float, float]:
    pitch = config.get("sample_pitch_m", config.get("target_sample_pitch_m"))
    if pitch is None:
        raise _refuse(
            ContractCode.MISSING_DECLARATION,
            "the edge declares no output sample pitch",
            declaration="config.sample_pitch_m",
            remedy="Set config['sample_pitch_m'] as a scalar or an (dy, dx) pair, in metres.",
        )
    values = (
        (float(pitch[0]), float(pitch[1]))
        if isinstance(pitch, (list, tuple))
        else (float(pitch), float(pitch))
    )
    if not all(math.isfinite(value) and value > 0.0 for value in values):
        raise _refuse(
            ContractCode.UNIT_NOT_SI,
            f"sample pitch must be positive metres, got {values!r}",
            declaration="config.sample_pitch_m",
        )
    return values


def get_coupler() -> RayToWaveCoupler:
    return RayToWaveCoupler()
