"""CHE-34 (M3.5): C_RAY_TO_WAVE as an executable graph edge.

The ticket is narrow on purpose -- an ``ArtifactRecord -> ArtifactRecord`` wrapper
over physics M2 already verified -- so the tests are about the three things a
wrapper can get wrong:

1. **It changed the numbers.** Guarded by a bit-identity check against the direct
   function call. Not `allclose`: a wrapper that perturbs the last bit has still
   inserted itself into the physics.
2. **It blesses a request it then fails.** ``validate_request`` and ``transform``
   are driven off one ``diagnose``, and every refusal below is asserted through
   both. Two parallel checklists is how a validator comes to approve something
   that cannot run.
3. **It pulled an engine into the coupler.** The static and dynamic
   engine-agnosticism checks are extended to cover this module, because M1's
   independence evidence stops bounding the search the moment it does not.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from multiscale_optics_agent.core.artifacts import ArtifactRecord
from multiscale_optics_agent.core.graph import GraphValidator
from multiscale_optics_agent.core.specs import ArtifactKind
from multiscale_optics_agent.couplers.base import CouplerRunRequest
from multiscale_optics_agent.couplers.contracts import ContractCode
from multiscale_optics_agent.couplers.optiland_handoff import (
    DeclaredHandoffPlane,
    declare_coherent_bundle,
)
from multiscale_optics_agent.couplers.ray_to_wave import Projection, ray_to_wave
from multiscale_optics_agent.couplers.ray_to_wave_node import COUPLER_ID, RayToWaveCoupler
from multiscale_optics_agent.registry.loader import Registry

pytest.importorskip("optiland")

from multiscale_optics_agent.adapters.base import ModelRunRequest, RunStatus
from multiscale_optics_agent.adapters.optiland_adapter import get_adapter

ROOT = Path(__file__).resolve().parents[1]

PUPIL_Z_M = 6.814345991561233e-05
PITCH_M = 2.6587352810843895e-06
GRID_N = 64


def _config(**overrides):
    config = {
        "handoff_plane": "exit_pupil",
        "handoff_plane_z_m": PUPIL_Z_M,
        "grid_n": GRID_N,
        "target_sample_pitch_m": PITCH_M,
        "projection": "asm_consistent",
    }
    config.update(overrides)
    return config


@pytest.fixture(scope="module")
def ray_record(tmp_path_factory):
    out = tmp_path_factory.mktemp("m35-rays")
    result = get_adapter().run(
        ModelRunRequest(
            run_id="che34",
            node_id="lens",
            config={
                "sample": "M3SingletRef",
                "num_rays": 8,
                "wavelength": 0.55,
                "handoff_plane": "exit_pupil",
                "output_directory": str(out),
            },
        )
    )
    assert result.status is RunStatus.SUCCEEDED
    return result.outputs["rays"]


@pytest.fixture
def coupler() -> RayToWaveCoupler:
    return RayToWaveCoupler()


def _request(record, tmp_path, **overrides) -> CouplerRunRequest:
    return CouplerRunRequest(
        run_id="che34",
        edge_id="pupil_reconstruction",
        source=record,
        config=_config(output_dir=str(tmp_path), **overrides),
    )


# --- The wrapper added nothing ------------------------------------------------


def test_node_output_is_bit_identical_to_the_direct_call(coupler, ray_record, tmp_path):
    """This is a wrapper; prove it.

    Bit-identity, not a tolerance. A tolerance would let the node quietly acquire
    its own arithmetic -- a cast, a reordering, an extra normalization -- and M2's
    verification evidence would stop applying to what a graph actually runs.
    """
    result = coupler.transform(_request(ray_record, tmp_path))
    assert result.status is RunStatus.SUCCEEDED, result.error_message

    bundle = declare_coherent_bundle(
        ray_record,
        declared_plane=DeclaredHandoffPlane(handoff_plane="exit_pupil", z_m=PUPIL_Z_M),
    ).bundle
    direct, _ = ray_to_wave(
        bundle,
        grid_shape=(GRID_N, GRID_N),
        sample_pitch_m=(PITCH_M, PITCH_M),
        projection=Projection.ASM_CONSISTENT,
    )

    written = np.load(result.target.uri)
    assert written.dtype == direct.u.dtype
    assert np.array_equal(written, direct.u)


def test_the_emitted_record_declares_everything_the_wave_node_reads(coupler, ray_record, tmp_path):
    """The target record must be self-describing; M3.6 consumes it unchanged."""
    result = coupler.transform(_request(ray_record, tmp_path))
    metadata = result.target.metadata

    assert result.target.kind is ArtifactKind.COMPLEX_FIELD
    for key in ("wavelength", "sample_pitch", "coordinate_frame", "phasor", "normalization"):
        assert metadata.get(key) is not None, key
    # Pad state is a required declaration: shape alone is not physical extent.
    assert metadata["pad_width"] == 0
    assert metadata["padded"] is False
    assert "projection = asm_consistent" in metadata["normalization"]
    # The declarations made upstream travel with the result, including the versioned
    # OPL reference CHE-41 added: a consumer must be able to tell v1's declared path
    # from v2's, because off axis they differ by the whole convergence tilt.
    declarations = result.diagnostics["declarations"]
    assert declarations["issue"].startswith("CHE-33 (M3.4)")
    assert declarations["opl_reference_version"].endswith("v2 (CHE-41)")
    assert declarations["superseded_opl_reference"]["version"].endswith("v1 (CHE-33)")
    assert "none" in result.diagnostics["gradient_claim"]


def test_an_already_coherent_record_needs_no_edge_declaration(coupler, ray_record, tmp_path):
    """A producer that declares its own OPL and amplitude is taken as-is.

    The edge declaration exists to promote an *undeclared* bundle. A bundle that
    already states its reference must not be re-declared -- re-deriving a chief
    ray from a record that names one would let the node quietly override an
    upstream physical decision.
    """
    bundle = declare_coherent_bundle(
        ray_record,
        declared_plane=DeclaredHandoffPlane(handoff_plane="exit_pupil", z_m=PUPIL_Z_M),
    ).bundle
    coherent = bundle.to_artifact_record(artifact_id="coherent-rays", uri=tmp_path / "coherent.npz")

    request = CouplerRunRequest(
        run_id="che34",
        edge_id="e",
        source=coherent,
        config={
            "grid_n": GRID_N,
            "target_sample_pitch_m": PITCH_M,
            "output_dir": str(tmp_path),
        },
    )
    assert coupler.validate_request(request).valid
    result = coupler.transform(request)
    assert result.status is RunStatus.SUCCEEDED, result.error_message

    direct, _ = ray_to_wave(bundle, grid_shape=(GRID_N, GRID_N), sample_pitch_m=(PITCH_M, PITCH_M))
    assert np.array_equal(np.load(result.target.uri), direct.u)


# --- Refusals, and validate/transform agreement -------------------------------


def _empty_ray_record(tmp_path) -> ArtifactRecord:
    path = tmp_path / "empty.npz"
    empty = np.zeros(0, dtype=np.float64)
    np.savez(
        path,
        x_m=empty,
        y_m=empty,
        z_m=empty,
        L=empty,
        M=empty,
        N=empty,
        intensity=empty,
        opd_native=empty,
    )
    return ArtifactRecord(
        id="empty",
        kind=ArtifactKind.RAY_BUNDLE,
        uri=str(path),
        shape=(0,),
        metadata={
            "length_unit": "m",
            "wavelength_m": 5.5e-7,
            "conventions": {
                "handoff_plane": "exit_pupil",
                "reference_plane": "exit pupil",
                "reference_plane_z_m": PUPIL_Z_M,
                "image_space_refractive_index": 1.0,
                "exit_pupil": {"location_from_image_m": -4.837461300309598e-3},
            },
        },
    )


def _non_finite_ray_record(tmp_path) -> ArtifactRecord:
    record = _empty_ray_record(tmp_path)
    path = tmp_path / "nonfinite.npz"
    ones = np.ones(3, dtype=np.float64)
    np.savez(
        path,
        x_m=np.array([0.0, np.nan, 0.0]),
        y_m=np.zeros(3),
        z_m=np.full(3, PUPIL_Z_M),
        L=np.zeros(3),
        M=np.zeros(3),
        N=ones,
        intensity=ones,
        opd_native=ones,
    )
    return record.model_copy(update={"id": "nonfinite", "uri": str(path), "shape": (3,)})


def _wrong_kind_record(tmp_path) -> ArtifactRecord:
    path = tmp_path / "field.npy"
    np.save(path, np.ones((4, 4), dtype=np.complex128))
    return ArtifactRecord(id="field", kind=ArtifactKind.COMPLEX_FIELD, uri=str(path), shape=(4, 4))


REFUSALS = {
    "undeclared_opl": (
        {"handoff_plane": None, "handoff_plane_z_m": None},
        ContractCode.OPL_REFERENCE_UNVERIFIED,
    ),
    "plane_kind_mismatch": (
        {"handoff_plane": "image_surface"},
        ContractCode.REFERENCE_PLANE_MISMATCH,
    ),
    "plane_position_mismatch": (
        {"handoff_plane_z_m": PUPIL_Z_M + 1e-6},
        ContractCode.REFERENCE_PLANE_MISMATCH,
    ),
    # lambda / (2 * 1e-4) = 2.75e-3, well under this system's 0.0517 marginal
    # direction cosine, so the grid provably cannot write the steepest ramp.
    "nyquist_violation": ({"target_sample_pitch_m": 1e-4}, ContractCode.SHAPE_MISMATCH),
    "missing_grid": ({"grid_n": None}, ContractCode.MISSING_DECLARATION),
    "missing_pitch": ({"target_sample_pitch_m": None}, ContractCode.MISSING_DECLARATION),
    "unknown_projection": ({"projection": "eq_2_ish"}, ContractCode.MISSING_DECLARATION),
    "bad_normalization": (
        {"normalization": "one_over_n_squared"},
        ContractCode.MISSING_DECLARATION,
    ),
}


@pytest.mark.parametrize("name", sorted(REFUSALS))
def test_a_bad_request_is_refused_structurally_and_identically_by_both_entry_points(
    coupler, ray_record, tmp_path, name
):
    """`transform` refuses, and `validate_request` said so first, with the same code.

    Both are driven from one `diagnose`, so this is a structural guarantee rather
    than two lists kept in step by hand -- but a guarantee nobody exercises is
    just a comment, so every refusal the class can make is run through both here.
    """
    overrides, expected = REFUSALS[name]
    config_overrides = {key: value for key, value in overrides.items() if value is not None}
    request = _request(ray_record, tmp_path, **config_overrides)
    for key, value in overrides.items():
        if value is None:
            request.config.pop(key, None)

    report = coupler.validate_request(request)
    assert not report.valid, name
    assert [issue.code for issue in report.errors] == [f"COUPLER_{expected}"]

    result = coupler.transform(request)
    assert result.status is RunStatus.FAILED
    assert result.error_type == str(expected)
    assert result.target is None, "a refused transform must not emit an artifact"
    assert result.diagnostics["refusals"][0]["remedy"], "a refusal must say what to do"
    assert not result.diagnostics.get("undiagnosed_refusal")


@pytest.mark.parametrize(
    ("builder", "expected"),
    [
        (_empty_ray_record, ContractCode.EMPTY_ENSEMBLE),
        (_non_finite_ray_record, ContractCode.NON_FINITE),
        (_wrong_kind_record, ContractCode.ARTIFACT_KIND_MISMATCH),
    ],
    ids=["empty_bundle", "non_finite_input", "wrong_artifact_kind"],
)
def test_a_bad_source_record_is_refused_structurally(coupler, tmp_path, builder, expected):
    request = _request(builder(tmp_path), tmp_path)
    report = coupler.validate_request(request)
    assert not report.valid
    assert [issue.code for issue in report.errors] == [f"COUPLER_{expected}"]

    result = coupler.transform(request)
    assert result.status is RunStatus.FAILED
    assert result.error_type == str(expected)
    assert result.target is None


def test_a_missing_source_port_is_refused_rather_than_raising(coupler, tmp_path):
    request = CouplerRunRequest(
        run_id="che34",
        edge_id="e",
        sources={"not_the_port": _wrong_kind_record(tmp_path)},
        config=_config(),
    )
    result = coupler.transform(request)
    assert result.status is RunStatus.FAILED
    assert result.error_type == str(ContractCode.MISSING_DECLARATION)


def test_a_gradient_request_is_refused(coupler, ray_record, tmp_path):
    """M3 is forward-only and this coupler's derivative is unverified."""
    request = _request(ray_record, tmp_path)
    request.require_gradients = True
    report = coupler.validate_request(request)
    assert not report.valid
    assert any(issue.code.endswith("GRADIENT_NOT_VERIFIED") for issue in report.errors)


# --- Cost -----------------------------------------------------------------


def test_estimate_reports_the_product_not_the_sum(coupler, ray_record, tmp_path):
    """A graph rejecting an infeasible configuration needs rays x pixels.

    The registry's cost model said `O(rays + pixels)`, which understates the work
    by a factor of the grid size. Memory *is* additive, because the core contracts
    two separable ramps instead of building an (N, ny, nx) tensor -- so the two
    scale differently and the estimate must reflect both.
    """
    small = coupler.estimate(_request(ray_record, tmp_path, grid_n=64))
    large = coupler.estimate(_request(ray_record, tmp_path, grid_n=256))

    assert small.wall_time_s is not None and large.wall_time_s is not None
    assert large.wall_time_s == pytest.approx(16.0 * small.wall_time_s, rel=1e-9)
    # Memory grows with the separable factors and the output grid, not with the
    # product of rays and pixels.
    assert large.peak_memory_bytes < 16 * small.peak_memory_bytes
    assert "ray-pixel products" in " ".join(large.notes)


def test_estimate_invents_nothing_when_the_grid_is_unknown(coupler, ray_record, tmp_path):
    request = _request(ray_record, tmp_path)
    request.config.pop("grid_n")
    estimate = coupler.estimate(request)
    assert estimate.wall_time_s is None
    assert estimate.confidence == "low"


# --- The graph edge -----------------------------------------------------------


def test_graph_validator_accepts_the_slice_edge_chain():
    """M_RAY_OPTILAND -> C_RAY_TO_WAVE -> M_WAVE_CHROMATIX, as shipped."""
    registry = Registry.from_package()
    graph = Registry.load_graph(ROOT / "examples" / "graphs" / "ray_to_wave.yaml")
    report = GraphValidator(registry).validate(graph)
    assert report.valid, [issue.model_dump() for issue in report.errors]


def test_the_example_graph_edge_config_actually_runs(coupler, ray_record, tmp_path):
    """The validator's approval has to mean something, so run the shipped config.

    Only the grid is shrunk, and only for runtime. Everything the declaration
    turns on -- the plane, its axial position, the pitch, the projection -- is
    taken from the file rather than restated here, so a divergence between what
    the example claims and what executes fails this test.
    """
    graph = Registry.load_graph(ROOT / "examples" / "graphs" / "ray_to_wave.yaml")
    edge = next(edge for edge in graph.edges if edge.coupler == COUPLER_ID)
    config = dict(edge.config)
    config["grid_n"] = GRID_N
    config["output_dir"] = str(tmp_path)

    result = coupler.transform(
        CouplerRunRequest(run_id="che34", edge_id=edge.id, source=ray_record, config=config)
    )
    assert result.status is RunStatus.SUCCEEDED, result.error_message
    assert result.diagnostics["reconstruction"]["grid_nyquist_satisfied"] is True


def test_the_registry_port_matches_what_the_implementation_consumes(coupler):
    """The port type and the code must not drift apart again.

    Before CHE-34 the registry declared `wavefront_samples`, which the implemented
    physics could never accept: a plane-wavelet sum needs a direction per sample
    and `WavefrontSamples` carries none.
    """
    spec = coupler.spec
    assert spec.source.artifact is ArtifactKind.RAY_BUNDLE
    assert spec.target.artifact is ArtifactKind.COMPLEX_FIELD
    assert spec.derivative.verified is False
    # The declaration requirement must be stated where a graph author reads it.
    assert any("declarations must be supplied" in item for item in spec.validity.assumptions)


# --- Engine agnosticism survives ---------------------------------------------


def test_the_runnable_node_imports_no_solver_engine():
    tree = ast.parse(
        (ROOT / "src/multiscale_optics_agent/couplers/ray_to_wave_node.py").read_text()
    )
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert not imported & {"optiland", "chromatix"}, sorted(imported)


def test_the_runnable_node_loads_no_engine_at_runtime():
    """Dynamic half: constructing and validating through the node pulls in neither.

    The static scan alone is not enough -- an engine can arrive through a
    transitive import, which is exactly what `adapters.base` would have done had
    it not been protocol-only.
    """
    script = (
        "import sys\n"
        "from multiscale_optics_agent.couplers.ray_to_wave_node import RayToWaveCoupler\n"
        "from multiscale_optics_agent.couplers.base import CouplerRunRequest\n"
        "from multiscale_optics_agent.core.artifacts import ArtifactRecord\n"
        "from multiscale_optics_agent.core.specs import ArtifactKind\n"
        "c = RayToWaveCoupler()\n"
        "c.spec\n"
        "r = CouplerRunRequest(run_id='r', edge_id='e', source=ArtifactRecord("
        "id='x', kind=ArtifactKind.RAY_BUNDLE, uri='missing.npz'), config={})\n"
        "c.validate_request(r)\n"
        "c.transform(r)\n"
        "print([m for m in sys.modules if m.split('.')[0] in {'optiland', 'chromatix'}])\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script], cwd=ROOT, capture_output=True, text=True, check=True
    )
    assert completed.stdout.strip() == "[]", completed.stdout
