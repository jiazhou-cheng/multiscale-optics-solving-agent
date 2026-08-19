"""Tests for the M_RAY_OPTILAND adapter (Optiland 0.6.0).

Scope reminder (see module docstring of
``multiscale_optics_agent.adapters.optiland_adapter``): this adapter only
supports the bundled ``ReverseTelephoto`` sample lens, the ``numpy``
(default) and ``torch`` (opt-in) backends, and exactly one design-parameter
path for gradients. Every numeric expectation below is read from
``knowledge/solvers/optiland/expected/*.json``, captured by actually running
``knowledge/solvers/optiland/probes/*.py`` against the pinned install --
nothing here is a re-derived or assumed oracle.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]

optiland = pytest.importorskip("optiland")

from conftest import load_probe_expected  # noqa: E402

from multiscale_optics_agent.adapters import optiland_adapter  # noqa: E402
from multiscale_optics_agent.adapters.base import ModelRunRequest, RunStatus  # noqa: E402
from multiscale_optics_agent.adapters.optiland_adapter import (  # noqa: E402
    OptilandAdapter,
    OptilandRayRequest,
)
from multiscale_optics_agent.core.errors import (  # noqa: E402
    AdapterDependencyError,
    UnsupportedCapabilityError,
)
from multiscale_optics_agent.core.precision import (  # noqa: E402
    ArrayNamespace,
    DeviceKind,
    Precision,
)
from multiscale_optics_agent.core.specs import ArtifactKind  # noqa: E402

pytestmark = [pytest.mark.integration, pytest.mark.optiland]

#: The traced ray set: the arrays `scientific_array_sha256` covers, unchanged since
#: M1. CHE-41 adds object-space arrays to the same file under a separate hash, so
#: the two groups are named separately here rather than merged into one set.
TRACED_ARRAY_NAMES = frozenset(
    {
        "x_m",
        "y_m",
        "z_m",
        "L",
        "M",
        "N",
        "intensity",
        "wavelength_m",
        "opd_native",
        "survived",
    }
)

#: CHE-41: the launch state Optic.trace discards, and the wavefront-reference term
#: computed from it. Present whenever the launch geometry could be characterized.
OBJECT_SPACE_ARRAY_NAMES = frozenset(
    {
        "object_space_reference_offset_m",
        "launch_x_m",
        "launch_y_m",
        "launch_z_m",
    }
)

#: CHE-47: the RAW hexapolar pupil coordinates a quadrature weight is computed
#: from downstream, regenerated the same way as the object-space term above.
#: Present whenever the trace is an un-vignetted hexapolar fan (both M3 samples,
#: at every ray count M1/M2/M3 exercise). The adapter exports coordinates only,
#: never a ring index or a weight -- see optiland_adapter._resolve_ray_pupil_sampling.
QUADRATURE_ARRAY_NAMES = frozenset(
    {
        "pupil_normalized_x",
        "pupil_normalized_y",
    }
)


def _smoke_request(**config_overrides) -> ModelRunRequest:
    """Mirror knowledge/solvers/optiland/probes/raytrace_probe.py exactly.

    Passing an empty config relies on the adapter's declared defaults
    (wavelength=0.55, num_rays=16, Hx=Hy=0, sample=ReverseTelephoto, backend=
    numpy), which match the probe's parameters, so this reproduces the
    validated probe run.
    """
    return ModelRunRequest(
        run_id="test-run",
        node_id="lens",
        inputs={},
        config=dict(config_overrides),
        design_parameters={},
        require_gradients=False,
    )


# ---------------------------------------------------------------------------
# 1. Smoke test
# ---------------------------------------------------------------------------


def test_smoke_run_succeeds_with_ray_and_wavefront_outputs() -> None:
    adapter = OptilandAdapter()
    result = adapter.run(_smoke_request())

    assert result.status is RunStatus.SUCCEEDED, result.error_message
    assert set(result.outputs) == {"rays", "wavefront"}

    rays_artifact = result.outputs["rays"]
    wavefront_artifact = result.outputs["wavefront"]
    assert rays_artifact.kind is ArtifactKind.RAY_BUNDLE
    assert wavefront_artifact.kind is ArtifactKind.WAVEFRONT_SAMPLES

    # knowledge/solvers/optiland/failure_guide.md: trace(num_rays=N) does NOT
    # return exactly N rays (pupil/aperture sampling changes the survivor
    # count) -- verify that directly rather than assuming num_rays==16.
    requested_num_rays = 16
    assert rays_artifact.shape[0] > 0
    assert rays_artifact.shape[0] != requested_num_rays
    assert result.diagnostics["requested_num_rays"] == requested_num_rays

    # Declared-but-unavailable metadata (amplitude/polarization/pupil_mask)
    # must be recorded as missing, never fabricated.
    assert "missing_declared_metadata" in wavefront_artifact.metadata
    assert any("not power-normalized" not in w for w in result.warnings)  # sanity: no fdtdx leakage
    assert any("intensity" in w or "length_unit" in w or "backend" in w for w in result.warnings)

    # The persisted artifact on disk must be loadable and non-empty,
    # independent of the in-process diagnostics dict.
    saved_rays = np.load(rays_artifact.uri)
    assert saved_rays["x_m"].shape == rays_artifact.shape


# ---------------------------------------------------------------------------
# 2. Independent-comparison test against the recorded probe evidence
# ---------------------------------------------------------------------------


def test_matches_recorded_raytrace_probe_evidence() -> None:
    expected = load_probe_expected("optiland", "raytrace_probe")

    adapter = OptilandAdapter()
    result = adapter.run(_smoke_request())

    assert result.status is RunStatus.SUCCEEDED, result.error_message
    rays_artifact = result.outputs["rays"]

    assert list(rays_artifact.shape) == expected["rays_x_shape"]
    assert rays_artifact.dtype == expected["rays_x_dtype"]
    assert result.diagnostics["sample"] == expected["lens_class"]

    saved = np.load(rays_artifact.uri)
    assert saved["x_m"].shape == tuple(expected["rays_x_shape"])
    assert str(saved["x_m"].dtype) == expected["rays_x_dtype"]


# ---------------------------------------------------------------------------
# 3. Gradient regression test (opt-in torch backend)
# ---------------------------------------------------------------------------


@pytest.mark.torch
@pytest.mark.parametrize(
    ("dtype", "objective_rel", "grad_rel"),
    [
        # The recorded probe never called set_precision, so it ran at Optiland's
        # torch DEFAULT, which is float32 (measured: be.get_precision() -> 32
        # after set_backend('torch'); benchmarks/probes/precision/default_precision.py).
        # Asking for float32 therefore reproduces the record bit-identically, and
        # this row is the unchanged regression lock.
        ("float32", 1e-6, 1e-6),
        # The adapter's default is float64 and, since CHE-61, it now genuinely
        # traces in float64 instead of reporting float64 over a float32 trace.
        # These two bounds are the MEASURED difference between the two paths
        # (1.3e-05 on the objective, 2.3e-06 on the gradient --
        # benchmarks/probes/precision/grad_precision.py), not a relaxation: the float64
        # result is the more accurate of the two, and the record is what moved
        # relative to it.
        ("float64", 1e-4, 1e-5),
    ],
)
def test_gradient_matches_recorded_probe_within_known_tolerance(
    dtype, objective_rel, grad_rel
) -> None:
    """Reproduce knowledge/solvers/optiland/probes/gradient_probe.py through the adapter.

    This is a regression lock on a *recorded* directional-derivative check,
    not the full repository gradient-verification bundle (that
    would need multiple finite-difference step sizes, a convergence table,
    and a deliberately ill-conditioned case -- out of scope here). The
    relative-error tolerance below (2e-3) is intentionally looser than every
    JAX-based solver in this repository (typically 1e-4 to 1e-13) because the
    recorded probe evidence itself shows 1.11e-03 relative error between
    torch autodiff and centered finite difference for this exact path, and
    that gap has not been root-caused (see
    knowledge/solvers/optiland/conventions.md, "derivative" section, and
    registry/models.yaml's derivative.notes for M_RAY_OPTILAND). Do not
    tighten this tolerance without re-investigating the discrepancy.
    """
    torch = pytest.importorskip("torch")
    expected = load_probe_expected("optiland", "gradient_probe")

    parameter_name = "surfaces.surfaces[1].geometry.radius"
    adapter = OptilandAdapter()
    request = ModelRunRequest(
        run_id="grad-test",
        node_id="lens",
        inputs={},
        config={
            "backend": "torch",
            "dtype": dtype,
            "wavelength": 0.55,
            "num_rays": 64,
            "Hx": 0.0,
            "Hy": 0.0,
        },
        design_parameters={parameter_name: expected["r0"]},
        require_gradients=True,
    )

    result = adapter.run(request)
    assert result.status is RunStatus.SUCCEEDED, result.error_message

    objective_tensor = result.diagnostics["objective_tensor"]
    design_tensor = result.diagnostics["design_parameter_tensors"][parameter_name]
    assert isinstance(objective_tensor, torch.Tensor)
    assert design_tensor.requires_grad

    assert objective_tensor.item() == pytest.approx(expected["objective_value"], rel=objective_rel)

    objective_tensor.backward()
    grad_ad = design_tensor.grad.item()

    assert grad_ad == pytest.approx(expected["grad_native_autodiff"], rel=grad_rel)

    relative_error = abs(grad_ad - expected["grad_finite_difference"]) / abs(
        expected["grad_finite_difference"]
    )
    assert relative_error < 2e-3, (
        f"AD-vs-finite-difference relative error {relative_error:.3e} exceeds the "
        "known, not-yet-root-caused 1.11e-03 tolerance recorded in "
        "knowledge/solvers/optiland/expected/gradient_probe.json."
    )

    # The new guarantee: the precision the run REPORTS is the precision it ran
    # in, observed from the traced tensors rather than echoed from the request.
    execution = result.diagnostics["execution"]
    assert execution["requested"]["precision"] == Precision.parse(dtype)
    assert execution["resolved"]["precision"] == Precision.parse(dtype)
    assert execution["actual"]["dtype"] == dtype
    assert execution["applied_to_optiland"]["get_precision"] == dtype
    assert execution["mismatches"] == []
    # And the autograd leaf matches it, so torch cannot promote mid-graph and
    # differentiate a different number than the one that was set.
    assert str(design_tensor.dtype) == f"torch.{dtype}"


# ---------------------------------------------------------------------------
# 4. Failure/resource test: gradients requested without opting into torch
#    must be rejected before any solver call.
# ---------------------------------------------------------------------------


def test_require_gradients_without_explicit_torch_backend_raises() -> None:
    adapter = OptilandAdapter()
    request = ModelRunRequest(
        run_id="fail-test",
        node_id="lens",
        inputs={},
        config={},  # backend left at the numpy default
        design_parameters={},
        require_gradients=True,
    )

    with patch("multiscale_optics_agent.adapters.optiland_adapter._import_optiland") as mock_import:
        mock_import.side_effect = AssertionError(
            "solver import must not be attempted when gradients are requested "
            "without an explicit torch backend"
        )
        with pytest.raises(UnsupportedCapabilityError, match="backend"):
            adapter.run(request)

    mock_import.assert_not_called()

    # Same eager rejection must be visible to compile-time graph validation.
    report = adapter.validate_request(request)
    assert not report.valid
    assert any(issue.code == "OPTILAND_GRADIENTS_REQUIRE_TORCH_BACKEND" for issue in report.errors)


# ---------------------------------------------------------------------------
# Additional capability-scope tests (not in the 5 required categories, but
# cheap and directly exercise the eager UnsupportedCapabilityError gate).
# ---------------------------------------------------------------------------


def test_custom_system_input_is_rejected_eagerly() -> None:
    from multiscale_optics_agent.core.artifacts import ArtifactRecord

    adapter = OptilandAdapter()
    request = ModelRunRequest(
        run_id="test-run",
        node_id="lens",
        inputs={
            "system": ArtifactRecord(
                id="custom-system",
                kind=ArtifactKind.OPTICAL_SYSTEM,
                uri="memory://custom-lens",
            )
        },
        config={},
        design_parameters={},
        require_gradients=False,
    )
    with pytest.raises(UnsupportedCapabilityError, match="system"):
        adapter.run(request)


def test_unvalidated_design_parameter_path_is_rejected_eagerly() -> None:
    adapter = OptilandAdapter()
    request = ModelRunRequest(
        run_id="test-run",
        node_id="lens",
        inputs={},
        config={},
        design_parameters={"surfaces.surfaces[0].geometry.conic": 0.0},
        require_gradients=False,
    )
    with pytest.raises(UnsupportedCapabilityError, match="design_parameters"):
        adapter.run(request)


def test_cuda_rejected_eagerly_on_the_numpy_backend() -> None:
    """CHE-61 (PB4b) replaces CHE-55's blanket device gate with the real reason.

    A CUDA request is no longer refused because "no GPU was available when
    probing"; it is refused on the numpy backend because Optiland's own
    `set_device` raises `BackendCapabilityError` there (measured,
    benchmarks/probes/precision/optiland_capability.py). The request is wrong in
    a way no container can fix, and the error now says which knob is missing
    rather than which machine was unavailable.
    """
    adapter = OptilandAdapter()
    request = ModelRunRequest(
        run_id="test-run",
        node_id="lens",
        inputs={},
        config={"device": "gpu"},
        design_parameters={},
        require_gradients=False,
    )
    with patch("multiscale_optics_agent.adapters.optiland_adapter._import_optiland") as mock_import:
        mock_import.side_effect = AssertionError(
            "solver import must not be attempted for an unsupported device"
        )
        with pytest.raises(UnsupportedCapabilityError, match="cuda"):
            adapter.run(request)
    mock_import.assert_not_called()

    report = adapter.validate_request(request)
    assert not report.valid
    codes = {issue.code for issue in report.errors}
    assert "OPTILAND_UNSUPPORTED_NAMESPACE_FOR_DEVICE" in codes
    message = next(
        issue.message
        for issue in report.errors
        if issue.code == "OPTILAND_UNSUPPORTED_NAMESPACE_FOR_DEVICE"
    )
    assert "torch" in message


def test_float32_is_now_a_supported_precision_not_a_rejected_one() -> None:
    """The inverse of CHE-55's dtype gate, and the point of PB4b.

    `set_precision` accepts float32 and float64. Refusing float32 was a project
    limitation, not a package one, and PB4b removes it -- so this asserts
    acceptance where the old test asserted rejection.
    """
    adapter = OptilandAdapter()
    for dtype in ("float32", "float64"):
        request = ModelRunRequest(
            run_id="test-run",
            node_id="lens",
            inputs={},
            config={"dtype": dtype},
            design_parameters={},
            require_gradients=False,
        )
        report = adapter.validate_request(request)
        assert report.valid, f"{dtype} should be executable: {report.errors}"


def test_float16_is_rejected_because_optiland_has_no_float16_path() -> None:
    """Not a policy decision this project is free to revisit by adding a cast.

    `optiland.backend.set_precision` is typed `Literal['float32','float64']` and
    raises `ValueError("Precision must be 'float32' or 'float64'.")` for anything
    else. Accepting float16 at the project boundary and casting it up would be
    advertising support for a computation that never happens.
    """
    adapter = OptilandAdapter()
    request = ModelRunRequest(
        run_id="test-run",
        node_id="lens",
        inputs={},
        config={"dtype": "float16"},
        design_parameters={},
        require_gradients=False,
    )
    with patch("multiscale_optics_agent.adapters.optiland_adapter._import_optiland") as mock_import:
        mock_import.side_effect = AssertionError(
            "solver import must not be attempted for an unsupported precision"
        )
        with pytest.raises(UnsupportedCapabilityError, match="fp16"):
            adapter.run(request)
    mock_import.assert_not_called()

    report = adapter.validate_request(request)
    assert not report.valid
    codes = {issue.code for issue in report.errors}
    assert "OPTILAND_UNSUPPORTED_PRECISION" in codes
    message = next(
        issue.message for issue in report.errors if issue.code == "OPTILAND_UNSUPPORTED_PRECISION"
    )
    # The remedy names what Optiland can do, so the caller can act on it.
    assert "float32" in message and "float64" in message


def test_cuda_on_the_torch_backend_is_gated_on_the_container_not_the_request() -> None:
    """The two reasons a CUDA request can fail are now distinguishable.

    Optiland *can* execute on CUDA through its torch backend, so this request is
    well-formed. Whether it runs depends on whether the container has a device --
    an environment fact, reported as one. Either outcome is acceptable here and
    the test asserts the right one for the image it is running in; what it will
    not tolerate is a silent CPU fallback, which is why the success branch
    asserts the resolved device really is cuda.
    """
    adapter = OptilandAdapter()
    request = ModelRunRequest(
        run_id="test-run",
        node_id="lens",
        inputs={},
        config={"backend": "torch", "device": "cuda"},
        design_parameters={"surfaces.surfaces[1].geometry.radius": 2.5},
        require_gradients=True,
    )
    report = adapter.validate_request(request)
    reason = optiland_adapter._cuda_unavailable_reason()

    if reason is None:
        assert report.valid, report.errors
        resolved = optiland_adapter._resolve_optiland_execution(request.config)
        assert resolved.device.kind is DeviceKind.CUDA
        assert resolved.namespace is ArrayNamespace.TORCH
        return

    assert not report.valid
    codes = {issue.code for issue in report.errors}
    assert "OPTILAND_CUDA_UNAVAILABLE" in codes
    message = next(
        issue.message for issue in report.errors if issue.code == "OPTILAND_CUDA_UNAVAILABLE"
    )
    # Names the container, not the capability, and points at the way to get one.
    assert "--gpu" in message
    assert "no silent fallback" in message
    with pytest.raises(UnsupportedCapabilityError, match=r"CUDA|cuda"):
        adapter.run(request)


def test_spec_matches_registry_entry(registry) -> None:
    adapter = OptilandAdapter()
    assert adapter.spec.id == "M_RAY_OPTILAND"
    assert (
        adapter.spec is registry.models["M_RAY_OPTILAND"]
        or adapter.spec == registry.models["M_RAY_OPTILAND"]
    )


def test_standalone_contract_is_deterministic_and_complete(tmp_path) -> None:
    adapter = OptilandAdapter()
    first = adapter.run_standalone(OptilandRayRequest(output_directory=tmp_path / "first"))
    second = adapter.run_standalone(OptilandRayRequest(output_directory=tmp_path / "second"))

    assert first.status is RunStatus.SUCCEEDED, first.failure
    assert second.status is RunStatus.SUCCEEDED, second.failure
    assert first.package_version == "0.6.0"
    assert first.backend == "numpy"
    assert first.device == "cpu"
    assert first.dtype == "float64"
    assert first.requested_sampling == 16
    assert first.surviving_ray_count == 817
    assert first.runtime_seconds is not None and first.runtime_seconds < 10.0
    assert first.scientific_array_sha256 == second.scientific_array_sha256
    assert first.summary_metrics == second.summary_metrics
    assert json.loads((tmp_path / "first" / "summary.json").read_text()) == json.loads(
        (tmp_path / "second" / "summary.json").read_text()
    )

    saved = np.load(first.arrays_path)
    # The TRACED ray set, which is what scientific_array_sha256 covers and what M1
    # pinned. CHE-41 adds an object-space group to the same file under its own hash
    # (see the second assertion); it does not add a column to this set, because that
    # would move a fingerprint no traced ray moved.
    assert set(saved.files) >= TRACED_ARRAY_NAMES
    assert (
        set(saved.files) - TRACED_ARRAY_NAMES == OBJECT_SPACE_ARRAY_NAMES | QUADRATURE_ARRAY_NAMES
    )
    assert np.all(saved["survived"])
    assert np.max(np.abs(np.sqrt(saved["L"] ** 2 + saved["M"] ** 2 + saved["N"] ** 2) - 1)) <= 1e-12
    assert np.all(saved["wavelength_m"] == pytest.approx(0.55e-6))

    expected = load_probe_expected("optiland", "standalone_baseline")
    assert first.scientific_array_sha256 == expected["stable_result"]["scientific_array_sha256"]
    assert first.summary_metrics == expected["stable_result"]["summary_metrics"]


@pytest.mark.parametrize(
    ("overrides", "expected_code"),
    [
        ({"prescription": "not-a-lens"}, "OPTILAND_INVALID_BASELINE_REQUEST"),
        ({"backend": "torch"}, "OPTILAND_INVALID_BASELINE_REQUEST"),
        ({"require_gradients": True}, "OPTILAND_INVALID_BASELINE_REQUEST"),
    ],
)
def test_standalone_invalid_or_unsupported_request_is_structured(
    tmp_path, overrides, expected_code
) -> None:
    payload = {"output_directory": tmp_path / "failed", **overrides}
    result = OptilandAdapter().run_standalone(payload)

    assert result.status is RunStatus.FAILED
    assert result.failure is not None
    assert result.failure.code == expected_code
    assert result.failure.stage == "request_validation"
    assert result.arrays_path is None


def test_standalone_missing_dependency_is_structured(tmp_path) -> None:
    adapter = OptilandAdapter()
    with patch(
        "multiscale_optics_agent.adapters.optiland_adapter._import_optiland",
        side_effect=AdapterDependencyError("missing pinned package"),
    ):
        result = adapter.run_standalone({"output_directory": tmp_path / "failed"})

    assert result.status is RunStatus.FAILED
    assert result.failure is not None
    assert result.failure.code == "OPTILAND_DEPENDENCY_UNAVAILABLE"
    assert result.arrays_path is None


def _fake_optiland_import(values, *, paraxial=None):
    """A stand-in optiland whose trace returns exactly `values`.

    `paraxial` is left absent by default so that a test which does not opt into
    the exit-pupil path cannot accidentally depend on a fabricated pupil.

    Returns the `_import_optiland` tuple and the fake lens separately: since
    CHE-56 the adapter builds its system through the canonical prescription
    builder rather than by looking a class up on `optiland.samples`, so
    substituting a lens means patching `_resolve_lens`, not the sample module.
    """

    class Backend:
        """Models the backend surface the adapter actually drives.

        CHE-61 added `set_precision`/`get_precision` to that surface: the adapter
        now tells Optiland which precision to execute in instead of reporting
        "float64" while leaving the real setting untouched. `set_device` is
        deliberately absent, mirroring the real numpy backend, where it raises
        `BackendCapabilityError` -- so a test would fail loudly if the adapter
        ever called it off the torch path.
        """

        precision = "float64"

        @staticmethod
        def set_backend(name):
            assert name == "numpy"

        @classmethod
        def set_precision(cls, precision):
            assert precision in ("float32", "float64")
            cls.precision = precision

        @classmethod
        def get_precision(cls):
            # An int width, matching the real package: set_precision takes
            # 'float64' but get_precision answers 64. The double reproduces the
            # asymmetry rather than smoothing it over, so the adapter is exercised
            # against the API it actually calls.
            return 64 if cls.precision == "float64" else 32

    class BackendUtils:
        @staticmethod
        def to_numpy(value):
            return np.asarray(value)

    class Lens:
        surfaces = SimpleNamespace(
            surfaces=[SimpleNamespace(geometry=SimpleNamespace(cs=SimpleNamespace(z=0.0)))]
        )

        def trace(self, **kwargs):
            return values

    if paraxial is not None:
        Lens.paraxial = paraxial

    return (Backend, BackendUtils, None), Lens()


@contextmanager
def _patched_optiland(values, *, paraxial=None):
    """Patch both extracted collaborators: the import probe and the builder."""
    modules, lens = _fake_optiland_import(values, paraxial=paraxial)
    with (
        patch(
            "multiscale_optics_agent.adapters.optiland_adapter._import_optiland",
            return_value=modules,
        ),
        patch(
            "multiscale_optics_agent.adapters.optiland_adapter._resolve_lens",
            return_value=lens,
        ),
    ):
        yield


@pytest.mark.parametrize("bad_value", [np.array([], dtype=float), np.array([np.nan])])
def test_standalone_empty_or_nonfinite_output_is_structured(tmp_path, bad_value) -> None:
    rays = SimpleNamespace(
        x=bad_value,
        y=bad_value,
        z=bad_value,
        L=bad_value,
        M=bad_value,
        N=bad_value,
        i=bad_value,
        w=bad_value,
        opd=bad_value,
    )
    with _patched_optiland(rays):
        result = OptilandAdapter().run_standalone({"output_directory": tmp_path / "failed"})

    assert result.status is RunStatus.FAILED
    assert result.failure is not None
    assert result.failure.code == "OPTILAND_INVALID_OR_EMPTY_OUTPUT"
    assert result.arrays_path is None


# ---------------------------------------------------------------------------
# CHE-32 (M3.3): the exit-pupil handoff plane, and the adapter-owned
# M3SingletRef system.
#
# Two oracles are used deliberately, and they are independent of each other:
#
#   * benchmarks/slice_protocol.yaml -- M3.2's frozen exit-pupil geometry.
#     Read here from the protocol file, not from the probe fixture, so the
#     export is checked against the milestone's own declared numbers.
#   * knowledge/solvers/optiland/expected/exit_pupil_handoff.json -- recorded by
#     knowledge/solvers/optiland/probes/exit_pupil_handoff.py against the pinned
#     install; this is the M1-standard determinism/hash evidence.
#
# Everything below drives the ordinary export path (OptilandAdapter.run ->
# _build_ray_bundle_artifact) and asserts on the exported artifact and its
# metadata. Calling Paraxial.XPL()/XPD() directly in a test would prove that
# Optiland has an exit pupil, not that this adapter hands rays off at it.
# ---------------------------------------------------------------------------

_M3_SYSTEMS = [
    ("ReverseTelephoto", "M3-REVERSE-TELEPHOTO"),
    ("M3SingletRef", "M3-SINGLET-REF"),
]


def _frozen_protocol_system(protocol_system_id: str) -> dict:
    """The `derived` block M3.2 froze for one system in benchmarks/slice_protocol.yaml."""
    import yaml

    protocol = yaml.safe_load((ROOT / "benchmarks" / "slice_protocol.yaml").read_text())
    for entry in protocol["systems"]:
        if entry["id"] == protocol_system_id:
            return entry["derived"]
    raise AssertionError(f"{protocol_system_id!r} is not in benchmarks/slice_protocol.yaml")


def _exported(sample: str, handoff_plane: str | None, output_directory) -> tuple:
    """Run the adapter and return (ray artifact, loaded arrays).

    `handoff_plane=None` omits the key entirely rather than passing the default
    explicitly -- the backward-compatibility test below depends on that
    difference being real.
    """
    config = {"sample": sample, "output_directory": str(output_directory)}
    if handoff_plane is not None:
        config["handoff_plane"] = handoff_plane

    result = OptilandAdapter().run(_smoke_request(**config))
    assert result.status is RunStatus.SUCCEEDED, result.error_message
    artifact = result.outputs["rays"]
    return artifact, np.load(artifact.uri)


@pytest.mark.parametrize(("sample", "protocol_system_id"), _M3_SYSTEMS)
def test_exit_pupil_handoff_reproduces_the_frozen_protocol_plane(
    tmp_path, sample, protocol_system_id
) -> None:
    """The exported reference plane is M3.2's frozen exit pupil, in SI."""
    frozen = _frozen_protocol_system(protocol_system_id)
    artifact, arrays = _exported(sample, "exit_pupil", tmp_path / "pupil")

    conventions = artifact.metadata["conventions"]
    exit_pupil = conventions["exit_pupil"]
    assert conventions["handoff_plane"] == "exit_pupil"
    assert "exit pupil" in conventions["reference_plane"]

    frozen_pupil_z_m = frozen["exit_pupil_z_mm"] * 1e-3
    frozen_diameter_m = frozen["exit_pupil_diameter_mm"] * 1e-3
    frozen_image_z_m = frozen["image_plane_z_mm"] * 1e-3

    assert conventions["reference_plane_z_m"] == pytest.approx(frozen_pupil_z_m, rel=1e-12)
    assert exit_pupil["z_m"] == pytest.approx(frozen_pupil_z_m, rel=1e-12)
    assert exit_pupil["diameter_m"] == pytest.approx(frozen_diameter_m, rel=1e-12)

    # XPL is signed and measured FROM THE IMAGE SURFACE, not from the origin --
    # the single fact the plane depends on. Recovering the frozen image plane by
    # subtracting it back out is what distinguishes a correct reading from one
    # that treated XPL as an absolute coordinate and happened to look plausible.
    assert exit_pupil["z_m"] - exit_pupil["location_from_image_m"] == pytest.approx(
        frozen_image_z_m, rel=1e-12
    )
    assert exit_pupil["z_m"] != pytest.approx(frozen_image_z_m, rel=1e-6)

    # The pupil is virtual on both M3 systems; the metadata must say so rather
    # than let a consumer assume the rays physically pass through this plane.
    assert exit_pupil["is_virtual"] is True
    assert exit_pupil["refracting_surfaces_beyond_pupil_z_m"]
    assert "ASYMPTOTE" in exit_pupil["position_semantics"]

    # The exported rays are actually on the declared plane, exactly.
    assert np.all(arrays["z_m"] == conventions["reference_plane_z_m"])

    # Paraxial aperture is reported from XPD; the measured survivor extent is
    # kept separate and labelled derived (it must not be read as an aperture).
    boundary = artifact.metadata["pupil_boundary"]
    assert boundary["paraxial_semi_diameter_m"] == pytest.approx(frozen_diameter_m / 2.0, rel=1e-12)
    assert boundary["mask_available_from_optiland"] is False
    assert boundary["representation"] == "implicit_in_surviving_rays"


@pytest.mark.parametrize(("sample", "protocol_system_id"), _M3_SYSTEMS)
def test_exit_pupil_projection_moves_each_ray_along_its_own_line(
    tmp_path, sample, protocol_system_id
) -> None:
    """The projection is a reparameterization along each ray, not a propagation."""
    del protocol_system_id
    image_artifact, at_image = _exported(sample, "image_surface", tmp_path / "image")
    pupil_artifact, at_pupil = _exported(sample, "exit_pupil", tmp_path / "pupil")

    direction = np.stack([at_image["L"], at_image["M"], at_image["N"]], axis=1)
    start = np.stack([at_image["x_m"], at_image["y_m"], at_image["z_m"]], axis=1)
    end = np.stack([at_pupil["x_m"], at_pupil["y_m"], at_pupil["z_m"]], axis=1)
    displacement = end - start
    step = np.linalg.norm(displacement, axis=1)

    # Collinearity, not a re-run of x + L*(z_target - z)/N: recomputing the
    # implementation's own formula would restate it rather than test it.
    residual = np.linalg.norm(np.cross(direction, displacement), axis=1) / step
    assert float(np.max(residual)) < 1e-12

    # The projection has to actually do something, or collinearity is vacuous.
    assert float(np.min(step)) > 1e-4
    assert not np.allclose(at_pupil["x_m"], at_image["x_m"])
    assert (
        pupil_artifact.metadata["scientific_array_sha256"]
        != image_artifact.metadata["scientific_array_sha256"]
    )

    # Directions, intensity and OPD are carried across untouched: no optical
    # path is added or removed here (OPL handling is M3.4/CHE-33).
    for field in ("L", "M", "N", "intensity", "opd_native", "wavelength_m", "survived"):
        assert np.array_equal(at_pupil[field], at_image[field]), field

    exit_pupil = pupil_artifact.metadata["conventions"]["exit_pupil"]
    assert exit_pupil["max_projection_step_m"] == pytest.approx(float(np.max(step)), rel=1e-9)
    assert at_pupil["z_m"].shape == at_image["z_m"].shape


def test_m3_singlet_ref_matches_recorded_m1_standard_evidence(tmp_path) -> None:
    """M3SingletRef is pinned to the same standard as every bundled system.

    Deterministic trace, finite output, unit-norm direction cosines, a stable
    scientific-array hash, and a survivor count -- at both handoff planes,
    compared against knowledge/solvers/optiland/expected/exit_pupil_handoff.json,
    which was recorded by running the probe against the pinned install.
    """
    expected = load_probe_expected("optiland", "exit_pupil_handoff")
    assert expected["status"] == "passed"
    assert expected["package_version"] == "0.6.0"
    frozen_planes = expected["systems"]["M3SingletRef"]["planes"]
    assert set(frozen_planes) == {"image_surface", "exit_pupil"}

    for plane, frozen in frozen_planes.items():
        assert frozen["deterministic"] is True

        first, first_arrays = _exported("M3SingletRef", plane, tmp_path / plane / "a")
        second, _ = _exported("M3SingletRef", plane, tmp_path / plane / "b")

        # Deterministic in this process, and identical to the recorded evidence.
        first_hash = first.metadata["scientific_array_sha256"]
        assert first_hash == second.metadata["scientific_array_sha256"], plane
        assert first_hash == frozen["scientific_array_sha256"], plane
        assert first.metadata["summary_metrics"] == frozen["summary_metrics"], plane
        assert int(first.shape[0]) == frozen["surviving_ray_count"]
        assert first.dtype == frozen["dtype"] == "float64"
        assert first.metadata["conventions"]["handoff_plane"] == plane
        assert first.metadata["conventions"]["reference_plane_z_m"] == pytest.approx(
            frozen["reference_plane_z_m"], rel=1e-12
        )

        # The M1 invariants, re-derived from the persisted arrays rather than
        # taken from the adapter's own summary of them.
        scientific = {name: first_arrays[name] for name in TRACED_ARRAY_NAMES}
        assert (
            set(first_arrays.files) - TRACED_ARRAY_NAMES
            == OBJECT_SPACE_ARRAY_NAMES | QUADRATURE_ARRAY_NAMES
        )
        for name, array in scientific.items():
            if name == "survived":
                assert np.all(array)
                continue
            assert np.all(np.isfinite(array)), (plane, name)
        norm = np.sqrt(scientific["L"] ** 2 + scientific["M"] ** 2 + scientific["N"] ** 2)
        assert float(np.max(np.abs(norm - 1.0))) <= 1e-12
        assert np.all(scientific["wavelength_m"] == pytest.approx(0.55e-6))


def test_m3_singlet_ref_is_a_supported_sample_not_a_custom_prescription(tmp_path) -> None:
    """The `system` input port stays refused, and an unknown name is rejected.

    CHE-56 opened a *typed* custom-prescription path
    (``config['prescription']``), which is exercised in
    ``test_optiland_canonical_prescriptions.py``. It did not open the untyped
    one: an arbitrary solver object on the ``system`` input port has no contract
    to validate, so it is still refused, and a name that is not in the
    prescription registry is still rejected by the same eager gate.
    """
    adapter = OptilandAdapter()

    # A prescription arriving through the input port stays refused even when the
    # named sample is the adapter-owned one.
    from multiscale_optics_agent.core.artifacts import ArtifactRecord

    request = ModelRunRequest(
        run_id="test-run",
        node_id="lens",
        inputs={
            "system": ArtifactRecord(
                id="custom-system", kind=ArtifactKind.OPTICAL_SYSTEM, uri="memory://custom-lens"
            )
        },
        config={"sample": "M3SingletRef"},
        design_parameters={},
        require_gradients=False,
    )
    with pytest.raises(UnsupportedCapabilityError, match="system"):
        adapter.run(request)

    # And an unregistered name is still rejected by the same gate, now naming
    # the supported alternatives rather than only refusing.
    with pytest.raises(UnsupportedCapabilityError, match="not a canonical prescription"):
        adapter.run(_smoke_request(sample="M3SingletRefV2"))
    with pytest.raises(UnsupportedCapabilityError, match="ReverseTelephoto, M3SingletRef"):
        adapter.run(_smoke_request(sample="M3SingletRefV2"))

    report = adapter.validate_request(_smoke_request(sample="M3SingletRefV2"))
    assert not report.valid
    assert any(issue.code == "OPTILAND_UNSUPPORTED_SAMPLE" for issue in report.errors)

    # The supported one runs.
    artifact, _ = _exported("M3SingletRef", "image_surface", tmp_path / "ok")
    assert artifact.metadata["sample"] == "M3SingletRef"


# ---------------------------------------------------------------------------
# CHE-32: structured failures. An unresolvable or unreachable plane must be a
# reported code, never a crash and never a silent fallback to the image
# surface -- that fallback would be wrong by the whole pupil-to-focus distance
# with nothing to notice it by.
# ---------------------------------------------------------------------------


def _unit_norm_fake_rays(*, n_values) -> SimpleNamespace:
    """Finite, unit-norm rays so a failure cannot be blamed on the earlier gates."""
    n = np.asarray(n_values, dtype=float)
    count = n.size
    return SimpleNamespace(
        x=np.zeros(count),
        y=np.zeros(count),
        z=np.zeros(count),
        L=np.sqrt(1.0 - n**2),
        M=np.zeros(count),
        N=n,
        i=np.ones(count),
        w=np.full(count, 0.55),
        opd=np.zeros(count),
    )


def _run_with_fake_optiland(tmp_path, rays, *, paraxial, handoff_plane="exit_pupil"):
    with _patched_optiland(rays, paraxial=paraxial):
        return OptilandAdapter().run(
            _smoke_request(handoff_plane=handoff_plane, output_directory=str(tmp_path / "out"))
        )


def _raises():
    """A paraxial accessor that fails the way an unsolvable system would."""

    def _call():
        raise RuntimeError("paraxial solve did not converge")

    return _call


@pytest.mark.parametrize(
    ("paraxial", "why"),
    [
        (SimpleNamespace(XPL=_raises(), XPD=lambda: 2.0), "paraxial solver raised"),
        (SimpleNamespace(XPL=lambda: np.inf, XPD=lambda: 2.0), "XPL is non-finite"),
        (SimpleNamespace(XPL=lambda: -1.0, XPD=lambda: np.nan), "XPD is non-finite"),
    ],
)
def test_unresolvable_exit_pupil_is_a_structured_failure(tmp_path, paraxial, why) -> None:
    result = _run_with_fake_optiland(
        tmp_path, _unit_norm_fake_rays(n_values=[1.0, 0.8]), paraxial=paraxial
    )

    assert result.status is RunStatus.FAILED, why
    assert result.diagnostics["code"] == "OPTILAND_EXIT_PUPIL_UNRESOLVED"
    assert result.diagnostics["stage"] == "handoff_plane_resolution"
    assert result.diagnostics["requested_handoff_plane"] == "exit_pupil"
    assert not result.outputs
    # Never a silent fallback to the image surface.
    assert "reference_plane_z_m" not in result.diagnostics


def test_ray_that_never_reaches_the_handoff_plane_is_a_structured_failure(tmp_path) -> None:
    """N = 0 means the ray is parallel to the plane; the projection is undefined."""
    result = _run_with_fake_optiland(
        tmp_path,
        _unit_norm_fake_rays(n_values=[1.0, 0.0]),
        paraxial=SimpleNamespace(XPL=lambda: -1.0, XPD=lambda: 2.0),
    )

    assert result.status is RunStatus.FAILED
    assert result.diagnostics["code"] == "OPTILAND_HANDOFF_PLANE_UNREACHABLE"
    assert result.diagnostics["stage"] == "handoff_plane_resolution"
    assert not result.outputs

    # The same rays at the default plane are exported normally, which is what
    # makes this a property of the requested plane rather than of the rays.
    ok = _run_with_fake_optiland(
        tmp_path,
        _unit_norm_fake_rays(n_values=[1.0, 0.0]),
        paraxial=SimpleNamespace(XPL=lambda: -1.0, XPD=lambda: 2.0),
        handoff_plane="image_surface",
    )
    assert ok.status is RunStatus.SUCCEEDED, ok.error_message


@pytest.mark.parametrize("plane", ["reference_sphere", "entrance_pupil", "", None])
def test_unsupported_handoff_plane_is_rejected_eagerly(plane) -> None:
    adapter = OptilandAdapter()
    request = _smoke_request(handoff_plane=plane)

    with patch("multiscale_optics_agent.adapters.optiland_adapter._import_optiland") as mock_import:
        mock_import.side_effect = AssertionError("optiland must not be imported for a bad plane")
        with pytest.raises(UnsupportedCapabilityError, match="handoff_plane"):
            adapter.run(request)
    mock_import.assert_not_called()

    report = adapter.validate_request(request)
    assert not report.valid
    assert any(issue.code == "OPTILAND_UNSUPPORTED_HANDOFF_PLANE" for issue in report.errors)


# ---------------------------------------------------------------------------
# CHE-32: backward compatibility. M3.3 adds a plane; it does not move the
# existing one.
# ---------------------------------------------------------------------------


def test_omitting_handoff_plane_preserves_the_default_image_surface_fingerprint(tmp_path) -> None:
    """No `handoff_plane` key at all must still be the L1 default-path export.

    The expected hash is the one already frozen by CHE-13 in
    knowledge/solvers/optiland/expected/standalone_baseline.json -- the same
    value L1-RAY-01 is built on -- not a value re-recorded for this ticket.
    """
    frozen_hash = load_probe_expected("optiland", "standalone_baseline")["stable_result"][
        "scientific_array_sha256"
    ]

    omitted_request = _smoke_request(output_directory=str(tmp_path / "omitted"))
    assert "handoff_plane" not in omitted_request.config  # the point of the test

    result = OptilandAdapter().run(omitted_request)
    assert result.status is RunStatus.SUCCEEDED, result.error_message
    artifact = result.outputs["rays"]
    conventions = artifact.metadata["conventions"]

    assert conventions["handoff_plane"] == "image_surface"
    assert conventions["reference_plane"] == "final traced image surface, surface index 14"
    assert conventions["exit_pupil"] is None
    assert artifact.metadata["pupil_boundary"]["paraxial_semi_diameter_m"] is None
    assert artifact.metadata["scientific_array_sha256"] == frozen_hash

    # Naming the default explicitly is the same run, byte for byte...
    explicit, _ = _exported("ReverseTelephoto", "image_surface", tmp_path / "explicit")
    assert explicit.metadata["scientific_array_sha256"] == frozen_hash

    # ...and the new plane really is a different export, so the two assertions
    # above are not both passing for the trivial reason.
    pupil, _ = _exported("ReverseTelephoto", "exit_pupil", tmp_path / "pupil")
    assert pupil.metadata["scientific_array_sha256"] != frozen_hash
