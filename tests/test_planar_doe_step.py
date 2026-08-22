"""The batched planar DOE step as a graph node, and the options CHE-95 added.

`couplers/cascade.py::planar_doe_step` has implemented SI Algorithm S1 since M2
and was **library-only** for the whole of M3: no registry entry, no capability
declaration, no graph node, not exported. Its only callers were one test file
and a benchmark runner, so the one operator that bounds ray count across stacked
DOEs could not appear in a graph.

Promotion is the risky half of that. The step has 14 round-trip tests but no
analytic oracle of its own beyond the full-enumeration limit, and making it
graph-reachable makes it usable *unattended*. So the enumeration gate is treated
as a hard precondition of promotion rather than as one test among several, and
it is the first thing below.

Everything CHE-95 added defaults to the behaviour the function already had.
`tests/test_coupler_round_trip.py` is the evidence: its 14 tests were written
against the old signature and pass unchanged.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.artifacts import ArtifactRecord
from core.boundary import ComplexField, ContractError, RayBundle, ReferencePlane
from core.execution import RunStatus
from core.specs import ArtifactKind
from couplers.base import CouplerRunRequest
from couplers.cascade import PrimarySampling, planar_doe_step, sample_primary_positions
from couplers.doe_node import COUPLER_ID, DOE_PORT, PlanarDoeStepCoupler
from couplers.wave_to_ray import (
    decompose,
    enumerate_indices,
    sampling_density,
    spectrum_to_rays,
)

pytestmark = pytest.mark.coupler

WAVELENGTH_M = 500e-9
PITCH_M = 1e-6
N = 16
GRID = (N, N)
PITCH = (PITCH_M, PITCH_M)
PLANE = ReferencePlane(name="doe", z_m=0.0)


def _field(seed: int = 20260822) -> ComplexField:
    rng = np.random.default_rng(seed)
    return ComplexField(
        u=(rng.normal(size=GRID) + 1j * rng.normal(size=GRID)).astype(np.complex128),
        sample_pitch_m=PITCH,
        wavelength_m=WAVELENGTH_M,
        reference_plane=PLANE,
    )


def _incident(seed: int = 20260822) -> RayBundle:
    spectrum = decompose(_field(seed))
    density = sampling_density(spectrum)
    return spectrum_to_rays(spectrum, enumerate_indices(density), density)


def _phase_doe(seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.exp(1j * rng.uniform(-np.pi, np.pi, size=GRID))


# ---------------------------------------------------------------------------
# The precondition for promotion
# ---------------------------------------------------------------------------

def test_full_enumeration_still_reproduces_the_transmitted_field() -> None:
    """The exactness limit, re-measured after promotion.

    With every propagating bin enumerated there is no sampling error, so
    re-accumulating the outgoing rays must reproduce the transmitted field to
    dtype round-off. This is the gate `benchmarks/protocols/coupler_protocol.yaml`
    makes mandatory and first, and it is the only oracle this composed step has:
    promoting an operator whose exactness limit had drifted would put an
    unvalidated transformation into a graph.
    """
    from couplers.ray_to_wave import Projection, ray_to_wave

    incident = _incident()
    outgoing, transmitted, diagnostics = planar_doe_step(
        incident,
        _phase_doe(),
        grid_shape=GRID,
        sample_pitch_m=PITCH,
        plane=PLANE,
        launch_positions_xy_m=np.zeros((1, 2)),
        secondary_count=None,
    )
    assert diagnostics.enumerated

    rebuilt, _ = ray_to_wave(
        outgoing,
        grid_shape=GRID,
        sample_pitch_m=PITCH,
        plane=PLANE,
        projection=Projection.ASM_CONSISTENT,
    )
    error = float(
        np.linalg.norm(rebuilt.u - transmitted.u) / np.linalg.norm(transmitted.u)
    )
    assert error < 1e-12, (
        f"the full-enumeration limit no longer reproduces the transmitted field "
        f"(relative L2 {error:.3e}). This is the only oracle the composed step has; "
        "promoting it to a graph node is conditional on this holding."
    )


# ---------------------------------------------------------------------------
# The property the step exists for
# ---------------------------------------------------------------------------

def test_two_stacked_does_keep_the_outgoing_count_at_the_budget() -> None:
    """The whole motivation, asserted directly on a stack rather than inferred.

    Per-ray branching would give `P*S` after one surface and `P*S*S'` after two.
    The batched step accumulates first, so the count after the second surface is
    the second surface's budget -- and is *independent of the incident count*,
    which is the property that makes a stack survivable.
    """
    incident = _incident()
    launches = np.zeros((4, 2))
    rng = np.random.default_rng(7)

    first, _, _ = planar_doe_step(
        incident, _phase_doe(1), grid_shape=GRID, sample_pitch_m=PITCH, plane=PLANE,
        launch_positions_xy_m=launches, secondary_count=32, rng=rng,
    )
    second, _, diagnostics = planar_doe_step(
        first, _phase_doe(2), grid_shape=GRID, sample_pitch_m=PITCH, plane=PLANE,
        launch_positions_xy_m=np.zeros((3, 2)), secondary_count=16, rng=rng,
    )
    third, _, _ = planar_doe_step(
        second, _phase_doe(3), grid_shape=GRID, sample_pitch_m=PITCH, plane=PLANE,
        launch_positions_xy_m=np.zeros((3, 2)), secondary_count=16, rng=rng,
    )

    assert first.count == 4 * 32
    assert second.count == 3 * 16
    assert third.count == 3 * 16, (
        "the third surface's count differs from the second's at the same budget, "
        "so the count is tracking the input after all"
    )
    assert diagnostics.incident_ray_count == first.count


@pytest.mark.parametrize("incident_seed", [1, 2, 3])
def test_the_outgoing_count_does_not_depend_on_the_incident_count(incident_seed: int) -> None:
    """Same budget, different incident populations, same outgoing count."""
    incident = _incident(incident_seed)
    outgoing, _, _ = planar_doe_step(
        incident, _phase_doe(), grid_shape=GRID, sample_pitch_m=PITCH, plane=PLANE,
        launch_positions_xy_m=np.zeros((5, 2)), secondary_count=11,
        rng=np.random.default_rng(3),
    )
    assert outgoing.count == 5 * 11


# ---------------------------------------------------------------------------
# The options, each defaulting to the prior behaviour
# ---------------------------------------------------------------------------

def test_primary_positions_can_be_sampled_instead_of_supplied() -> None:
    positions = sample_primary_positions(
        PrimarySampling.UNIFORM_ON_GRID,
        count=32,
        grid_shape=GRID,
        sample_pitch_m=PITCH,
        rng=np.random.default_rng(11),
    )
    assert positions.shape == (32, 2)
    # On the grid's own origin rule: coordinate zero at index n // 2, matching
    # ComplexField.coordinates. Off-by-half-a-pitch here is invisible in an
    # intensity and is a real phase error.
    half_extent = (N // 2) * PITCH_M
    assert np.all(np.abs(positions) <= half_extent + 1e-18)
    residual = np.abs(positions / PITCH_M - np.round(positions / PITCH_M))
    assert float(residual.max()) < 1e-9, "positions are not on the sample grid"


def test_incident_position_sampling_reuses_the_incoming_transverse_positions() -> None:
    incident = _incident()
    positions = sample_primary_positions(
        PrimarySampling.INCIDENT_POSITIONS,
        count=6,
        grid_shape=GRID,
        sample_pitch_m=PITCH,
        bundle=incident,
    )
    assert np.allclose(positions, np.asarray(incident.positions_m)[:6, :2])


def test_asking_for_more_incident_positions_than_exist_is_refused() -> None:
    """Refusal rather than replacement: an invented position is a fabricated ray."""
    incident = _incident()
    with pytest.raises(ContractError, match="cannot invent one"):
        sample_primary_positions(
            PrimarySampling.INCIDENT_POSITIONS,
            count=incident.count + 1,
            grid_shape=GRID,
            sample_pitch_m=PITCH,
            bundle=incident,
        )


def test_supplying_both_position_sources_is_a_conflict_not_a_precedence() -> None:
    with pytest.raises(ContractError, match="both specify where"):
        planar_doe_step(
            _incident(), _phase_doe(), grid_shape=GRID, sample_pitch_m=PITCH, plane=PLANE,
            launch_positions_xy_m=np.zeros((2, 2)),
            primary_sampling=PrimarySampling.UNIFORM_ON_GRID, primary_count=2,
            secondary_count=None,
        )


def test_preserve_energy_is_off_by_default_and_reported_when_on() -> None:
    """A lossy DOE must be allowed to lose power, and a fix must be visible.

    The mask below transmits half the aperture and blocks the rest, so the
    transmitted power *should* fall. With `preserve_energy` on it does not --
    which is exactly why the applied factor is reported: without it, a
    renormalized run and a genuinely lossless DOE are indistinguishable in the
    record.
    """
    incident = _incident()
    lossy = np.zeros(GRID, dtype=np.complex128)
    lossy[: N // 2, :] = 1.0

    _, plain, plain_diag = planar_doe_step(
        incident, lossy, grid_shape=GRID, sample_pitch_m=PITCH, plane=PLANE,
        launch_positions_xy_m=np.zeros((1, 2)), secondary_count=None,
    )
    assert plain_diag.energy_preservation_factor is None
    assert plain.discrete_power() < plain_diag.incident_discrete_power

    _, kept, kept_diag = planar_doe_step(
        incident, lossy, grid_shape=GRID, sample_pitch_m=PITCH, plane=PLANE,
        launch_positions_xy_m=np.zeros((1, 2)), secondary_count=None,
        preserve_energy=True,
    )
    assert kept_diag.energy_preservation_factor is not None
    assert kept_diag.energy_preservation_factor > 1.0
    assert kept.discrete_power() == pytest.approx(kept_diag.incident_discrete_power, rel=1e-12)


def test_padding_widens_the_grid_without_moving_the_power() -> None:
    """Zero-padding interpolates the spectrum; it does not add or remove field.

    Zero, not edge-clamp: a bounded DOE has no field outside it, and continuing
    the edge value would invent one.
    """
    incident = _incident()
    _, unpadded, _ = planar_doe_step(
        incident, _phase_doe(), grid_shape=GRID, sample_pitch_m=PITCH, plane=PLANE,
        launch_positions_xy_m=np.zeros((1, 2)), secondary_count=None,
    )
    _, padded, diagnostics = planar_doe_step(
        incident, _phase_doe(), grid_shape=GRID, sample_pitch_m=PITCH, plane=PLANE,
        launch_positions_xy_m=np.zeros((1, 2)), secondary_count=None, pad_width=N // 2,
    )
    assert diagnostics.pad_width == N // 2
    assert padded.u.shape == (2 * N, 2 * N)
    assert padded.discrete_power() == pytest.approx(unpadded.discrete_power(), rel=1e-12)
    assert diagnostics.propagating_modes > 0


def test_the_collapsed_mode_emits_one_ray_per_launch_position() -> None:
    """A cheap preview, and labelled as one.

    The single direction is the bin nearest the power-weighted mean wavevector
    rather than a synthesised off-grid direction, so the ray is still an actual
    mode of the transmitted field and every downstream invariant holds.
    """
    incident = _incident()
    outgoing, _, diagnostics = planar_doe_step(
        incident, _phase_doe(), grid_shape=GRID, sample_pitch_m=PITCH, plane=PLANE,
        launch_positions_xy_m=np.zeros((3, 2)), secondary_count=1,
    )
    assert diagnostics.collapsed_to_mean_wavevector
    assert diagnostics.secondary_count == 1
    assert outgoing.count == 3
    norms = np.linalg.norm(np.asarray(outgoing.directions), axis=1)
    assert np.allclose(norms, 1.0, atol=1e-12)


def test_the_conventions_are_reported_on_every_result() -> None:
    """OPL reset and the amplitude meaning travel with the record, not just the docs."""
    _, _, diagnostics = planar_doe_step(
        _incident(), _phase_doe(), grid_shape=GRID, sample_pitch_m=PITCH, plane=PLANE,
        launch_positions_xy_m=np.zeros((1, 2)), secondary_count=None,
    )
    payload = diagnostics.as_dict()
    assert "reset to 0 at this plane" in payload["opl_convention"]
    assert "double-count" in payload["opl_convention"]
    assert "U~[m]/p[m]" in payload["amplitude_convention"]
    assert payload["launch_source"] == "caller"


def test_the_outgoing_optical_path_length_is_actually_zero() -> None:
    """The convention is stated; this checks the arrays agree with it."""
    outgoing, _, _ = planar_doe_step(
        _incident(), _phase_doe(), grid_shape=GRID, sample_pitch_m=PITCH, plane=PLANE,
        launch_positions_xy_m=np.zeros((2, 2)), secondary_count=None,
    )
    assert np.allclose(np.asarray(outgoing.optical_path_length_m), 0.0)


# ---------------------------------------------------------------------------
# The graph node
# ---------------------------------------------------------------------------

def _write_records(tmp_path):
    incident = _incident()
    rays_uri = tmp_path / "rays.npz"
    record = incident.to_artifact_record(artifact_id="incident", uri=rays_uri)
    doe_uri = tmp_path / "doe.npy"
    np.save(doe_uri, _phase_doe())
    doe_record = ArtifactRecord(
        id="doe",
        kind=ArtifactKind.COMPLEX_FIELD,
        uri=str(doe_uri),
        shape=list(GRID),
        dtype="complex128",
    )
    return record, doe_record


def test_the_node_runs_and_reports_the_conventions(tmp_path) -> None:
    source, doe = _write_records(tmp_path)
    result = PlanarDoeStepCoupler().transform(
        CouplerRunRequest(
            run_id="che95",
            edge_id="doe_step",
            sources={"source": source, DOE_PORT: doe},
            config={
                "grid_shape": list(GRID),
                "sample_pitch_m": list(PITCH),
                "plane_z_m": 0.0,
                "launch_positions_xy_m": [[0.0, 0.0]],
                "output_dir": str(tmp_path / "out"),
            },
        )
    )
    assert result.status is RunStatus.SUCCEEDED, result.error_message
    assert result.target is not None
    assert result.target.kind is ArtifactKind.RAY_BUNDLE
    cascade = result.diagnostics["cascade"]
    assert cascade["enumerated"] is True
    assert "reset to 0" in cascade["opl_convention"]
    assert "surrogate" in result.diagnostics["gradient_claim"]


def test_the_node_refuses_a_stochastic_request_with_no_seed(tmp_path) -> None:
    source, doe = _write_records(tmp_path)
    request = CouplerRunRequest(
        run_id="che95",
        edge_id="doe_step",
        sources={"source": source, DOE_PORT: doe},
        config={
            "grid_shape": list(GRID),
            "sample_pitch_m": list(PITCH),
            "plane_z_m": 0.0,
            "launch_positions_xy_m": [[0.0, 0.0]],
            "secondary_count": 8,
        },
    )
    coupler = PlanarDoeStepCoupler()
    result = coupler.transform(request)
    assert result.status is RunStatus.FAILED
    assert "seed" in (result.error_message or "")
    # validate_request must agree with transform, or a validator blesses a
    # request that then fails.
    codes = {issue.code for issue in coupler.validate_request(request).issues}
    assert any("MISSING_DECLARATION" in code for code in codes)


def test_the_node_refuses_an_incident_record_with_no_declared_opl(tmp_path) -> None:
    """Stricter than C_RAY_TO_WAVE, for a stated reason.

    This step resets OPL to zero, so an incident OPL whose reference was guessed
    is absorbed into the accumulated phase and cannot be audited afterwards.
    C_RAY_TO_WAVE's promotion survives into the field's phase with the record
    still naming the plane; this one does not survive at all.
    """
    incident = _incident()
    uri = tmp_path / "bare.npz"
    record = incident.to_artifact_record(artifact_id="bare", uri=uri)
    stripped = record.model_copy(
        update={"metadata": {k: v for k, v in record.metadata.items()
                             if k != "optical_path_length_reference"}}
    )
    np.save(tmp_path / "doe.npy", _phase_doe())
    doe = ArtifactRecord(
        id="doe", kind=ArtifactKind.COMPLEX_FIELD, uri=str(tmp_path / "doe.npy"),
        shape=list(GRID), dtype="complex128",
    )
    result = PlanarDoeStepCoupler().transform(
        CouplerRunRequest(
            run_id="che95", edge_id="e", sources={"source": stripped, DOE_PORT: doe},
            config={
                "grid_shape": list(GRID), "sample_pitch_m": list(PITCH), "plane_z_m": 0.0,
                "launch_positions_xy_m": [[0.0, 0.0]],
            },
        )
    )
    assert result.status is RunStatus.FAILED
    assert "OPL" in str(result.error_type) or "opl" in (result.error_message or "").lower()


def test_the_node_refuses_a_gradient_request(tmp_path) -> None:
    source, doe = _write_records(tmp_path)
    report = PlanarDoeStepCoupler().validate_request(
        CouplerRunRequest(
            run_id="che95", edge_id="e", sources={"source": source, DOE_PORT: doe},
            require_gradients=True,
            config={
                "grid_shape": list(GRID), "sample_pitch_m": list(PITCH), "plane_z_m": 0.0,
                "launch_positions_xy_m": [[0.0, 0.0]],
            },
        )
    )
    codes = {issue.code for issue in report.issues}
    assert "COUPLER_GRADIENT_NOT_VERIFIED" in codes


def test_the_registry_entry_and_the_node_agree_on_the_id() -> None:
    from registry.loader import Registry

    spec = Registry.from_package().couplers[COUPLER_ID]
    assert spec.id == COUPLER_ID
    assert spec.source.artifact is ArtifactKind.RAY_BUNDLE
    assert spec.target.artifact is ArtifactKind.RAY_BUNDLE
    assert spec.lossy is True
    assert spec.derivative.verified is False
