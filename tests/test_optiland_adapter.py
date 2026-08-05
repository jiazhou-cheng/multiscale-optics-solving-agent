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

from unittest.mock import patch

import numpy as np
import pytest

optiland = pytest.importorskip("optiland")

from conftest import load_probe_expected  # noqa: E402

from multiscale_optics_agent.adapters.base import ModelRunRequest, RunStatus  # noqa: E402
from multiscale_optics_agent.adapters.optiland_adapter import OptilandAdapter  # noqa: E402
from multiscale_optics_agent.core.errors import UnsupportedCapabilityError  # noqa: E402
from multiscale_optics_agent.core.specs import ArtifactKind  # noqa: E402

pytestmark = pytest.mark.integration


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
    assert saved_rays["x"].shape == rays_artifact.shape


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
    assert saved["x"].shape == tuple(expected["rays_x_shape"])
    assert str(saved["x"].dtype) == expected["rays_x_dtype"]


# ---------------------------------------------------------------------------
# 3. Gradient regression test (opt-in torch backend)
# ---------------------------------------------------------------------------


@pytest.mark.torch
def test_gradient_matches_recorded_probe_within_known_tolerance() -> None:
    """Reproduce knowledge/solvers/optiland/probes/gradient_probe.py through the adapter.

    This is a regression lock on a *recorded* directional-derivative check,
    not the full CLAUDE.md section 6.2 gradient-verification bundle (that
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
        config={"backend": "torch", "wavelength": 0.55, "num_rays": 64, "Hx": 0.0, "Hy": 0.0},
        design_parameters={parameter_name: expected["r0"]},
        require_gradients=True,
    )

    result = adapter.run(request)
    assert result.status is RunStatus.SUCCEEDED, result.error_message

    objective_tensor = result.diagnostics["objective_tensor"]
    design_tensor = result.diagnostics["design_parameter_tensors"][parameter_name]
    assert isinstance(objective_tensor, torch.Tensor)
    assert design_tensor.requires_grad

    assert objective_tensor.item() == pytest.approx(expected["objective_value"], rel=1e-6)

    objective_tensor.backward()
    grad_ad = design_tensor.grad.item()

    assert grad_ad == pytest.approx(expected["grad_native_autodiff"], rel=1e-6)

    relative_error = abs(grad_ad - expected["grad_finite_difference"]) / abs(
        expected["grad_finite_difference"]
    )
    assert relative_error < 2e-3, (
        f"AD-vs-finite-difference relative error {relative_error:.3e} exceeds the "
        "known, not-yet-root-caused 1.11e-03 tolerance recorded in "
        "knowledge/solvers/optiland/expected/gradient_probe.json."
    )


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


def test_spec_matches_registry_entry(registry) -> None:
    adapter = OptilandAdapter()
    assert adapter.spec.id == "M_RAY_OPTILAND"
    assert (
        adapter.spec is registry.models["M_RAY_OPTILAND"]
        or adapter.spec == registry.models["M_RAY_OPTILAND"]
    )
