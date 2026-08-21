"""Tests for the forward-only M_EM_FDTDX adapter.

Scope reminder (see module docstring of
``multiscale_optics_agent.adapters.fdtdx_adapter``): this adapter is
forward-only. Two independent gradient probes against fdtdx 0.6.2 both
failed and are documented in ``knowledge/solvers/fdtdx/``. This test module
does not attempt to fix or work around either failure; the two
``xfail(strict=True)`` tests below exist to lock in the known-broken
behavior so the suite fails loudly (demanding human review) if upstream
fdtdx/jax ever changes it.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

fdtdx = pytest.importorskip("fdtdx")
jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from conftest import load_probe_expected  # noqa: E402

from multiscale_optics_agent.adapters.base import ModelRunRequest, RunStatus  # noqa: E402
from multiscale_optics_agent.adapters.fdtdx_adapter import FdtdxAdapter  # noqa: E402
from multiscale_optics_agent.core.errors import UnsupportedCapabilityError  # noqa: E402

pytestmark = [pytest.mark.integration, pytest.mark.jax, pytest.mark.fdtdx]


def _smoke_request(*, require_gradients: bool = False) -> ModelRunRequest:
    """Mirror knowledge/solvers/fdtdx/probes/propagation_probe.py exactly.

    Passing an empty config relies on the adapter's declared defaults, which
    are set to the same values as the probe (see fdtdx_adapter.py's
    ``_DEFAULT_*`` constants), so this reproduces the validated probe run.
    """
    return ModelRunRequest(
        run_id="test-run",
        node_id="fdtd",
        inputs={},
        config={},
        design_parameters={},
        require_gradients=require_gradients,
    )


# ---------------------------------------------------------------------------
# 1. Smoke test
# ---------------------------------------------------------------------------


def test_smoke_run_succeeds_with_documented_axis_order() -> None:
    adapter = FdtdxAdapter()
    result = adapter.run(_smoke_request())

    assert result.status is RunStatus.SUCCEEDED, result.error_message
    assert "vector_field" in result.outputs

    artifact = result.outputs["vector_field"]
    assert artifact.metadata["axis_order"] == "(component, x, y, z)"
    assert artifact.shape == (3, 30, 30, 30)
    assert artifact.metadata["field_components"] == ["Ex", "Ey", "Ez"]

    # Un-power-normalized warning must be present (adapter deliberately does
    # not normalize the monitor output).
    assert any("power-normalized" in warning for warning in result.warnings)


# ---------------------------------------------------------------------------
# 2. Independent-comparison test against the recorded probe evidence
# ---------------------------------------------------------------------------


def test_matches_recorded_propagation_probe_evidence() -> None:
    expected = load_probe_expected("fdtdx", "propagation_probe")

    adapter = FdtdxAdapter()
    result = adapter.run(_smoke_request())

    assert result.status is RunStatus.SUCCEEDED, result.error_message
    artifact = result.outputs["vector_field"]

    assert list(artifact.shape) == expected["E_shape_after_run"]
    assert artifact.dtype == expected["E_dtype"]
    assert result.diagnostics["time_steps_total"] == expected["time_steps_total"]
    assert result.diagnostics["any_nan"] == expected["any_nan"]
    assert result.diagnostics["max_abs_E"] == pytest.approx(
        expected["max_abs_E"], rel=1e-5, abs=1e-8
    )

    # The saved artifact on disk must reproduce the same value independently
    # of the in-process diagnostics dict.
    import numpy as np

    saved = np.load(artifact.uri)
    assert saved.shape == tuple(expected["E_shape_after_run"])
    assert float(np.max(np.abs(saved))) == pytest.approx(expected["max_abs_E"], rel=1e-5, abs=1e-8)


# ---------------------------------------------------------------------------
# 3 & 4. Regression locks for the two documented gradient failures.
#
# These call fdtdx directly (the same way
# knowledge/solvers/fdtdx/probes/gradient_probe.py did), NOT through
# FdtdxAdapter, because the adapter's run()/validate_request() deliberately
# reject require_gradients=True before ever reaching fdtdx -- there is no
# code path through the adapter that could reproduce either failure. These
# tests exist purely to detect, loudly, if fdtdx/jax upstream ever changes
# this behavior (at which point the adapter's forward-only restriction
# should be revisited by a human, not silently lifted).
# ---------------------------------------------------------------------------


def _base_objects(volume, wavelength):
    constraints, object_list = [], [volume]
    bound_cfg = fdtdx.BoundaryConfig.from_uniform_bound(boundary_type="periodic")
    bound_dict, c_list = fdtdx.boundary_objects_from_config(bound_cfg, volume)
    constraints.extend(c_list)
    object_list.extend(list(bound_dict.values()))
    source = fdtdx.GaussianPlaneSource(
        partial_grid_shape=(None, None, 1),
        partial_real_shape=(2e-6, 2e-6, None),
        fixed_E_polarization_vector=(1, 0, 0),
        wave_character=fdtdx.WaveCharacter(wavelength=wavelength),
        radius=1e-6,
        std=1 / 3,
        direction="+",
    )
    constraints.append(
        source.place_relative_to(
            volume, axes=(0, 1, 2), own_positions=(0, 0, 0), other_positions=(0, 0, 0)
        )
    )
    object_list.append(source)
    return constraints, object_list


def _run_with_wavelength(wavelength):
    key = jax.random.PRNGKey(0)
    config = fdtdx.SimulationConfig(
        time=5e-15, resolution=100e-9, backend="cpu", dtype=jnp.float32, courant_factor=0.99
    )
    volume = fdtdx.SimulationVolume(
        partial_real_shape=(3.0e-6, 3.0e-6, 3.0e-6), material=fdtdx.Material(permittivity=1.0)
    )
    constraints, object_list = _base_objects(volume, wavelength)
    key, subkey = jax.random.split(key)
    objects, arrays, params, config, _ = fdtdx.place_objects(
        object_list=object_list, config=config, constraints=constraints, key=subkey
    )
    arrays2, new_objects, _info = fdtdx.apply_params(arrays, objects, params, subkey)
    final_state = fdtdx.run_fdtd(arrays=arrays2, objects=new_objects, config=config, key=subkey)
    _, arrays_out = final_state
    return jnp.sum(arrays_out.fields.E**2)


def _run_with_permittivity(permittivity):
    key = jax.random.PRNGKey(0)
    config = fdtdx.SimulationConfig(
        time=5e-15, resolution=100e-9, backend="cpu", dtype=jnp.float32, courant_factor=0.99
    )
    volume = fdtdx.SimulationVolume(
        partial_real_shape=(3.0e-6, 3.0e-6, 3.0e-6),
        material=fdtdx.Material(permittivity=permittivity),
    )
    constraints, object_list = _base_objects(volume, 1.0e-6)
    key, subkey = jax.random.split(key)
    objects, arrays, params, config, _ = fdtdx.place_objects(
        object_list=object_list, config=config, constraints=constraints, key=subkey
    )
    arrays2, new_objects, _info = fdtdx.apply_params(arrays, objects, params, subkey)
    final_state = fdtdx.run_fdtd(arrays=arrays2, objects=new_objects, config=config, key=subkey)
    _, arrays_out = final_state
    return jnp.sum(arrays_out.fields.E**2)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Known fdtdx 0.6.2 bug: jax.grad w.r.t. source wavelength returns a "
        "phantom exact 0.0 against a large nonzero finite-difference "
        "estimate (suspected internal round()-based step-count "
        "discretization). See knowledge/solvers/fdtdx/conventions.md "
        "'A real, reproducible zero-gradient case'. If this starts passing, "
        "fdtdx/jax has changed upstream and the adapter's eager "
        "require_gradients rejection must be re-evaluated by a human."
    ),
)
def test_wavelength_gradient_matches_finite_difference_lock() -> None:
    w0 = 1.0e-6
    grad_ad = float(jax.grad(_run_with_wavelength)(w0))
    eps = 1e-9
    grad_fd = float((_run_with_wavelength(w0 + eps) - _run_with_wavelength(w0 - eps)) / (2 * eps))
    # Currently fails: grad_ad is exactly 0.0 while grad_fd is ~-1.02e6.
    assert grad_ad == pytest.approx(grad_fd, rel=0.05)


@pytest.mark.xfail(
    strict=True,
    raises=jax.errors.ConcretizationTypeError,
    reason=(
        "Known fdtdx 0.6.2 bug: fdtdx.place_objects() performs concrete "
        "Python-level introspection of material properties (math.isclose "
        "on permittivity components) and cannot be traced, so jax.grad "
        "w.r.t. a Material(permittivity=<traced>) raises "
        "ConcretizationTypeError. See "
        "knowledge/solvers/fdtdx/conventions.md 'Object graph construction "
        "is NOT traceable'. If this starts passing (or raises a different "
        "exception), fdtdx/jax has changed upstream and the adapter's "
        "eager require_gradients rejection must be re-evaluated by a human."
    ),
)
def test_permittivity_gradient_raises_concretization_error_lock() -> None:
    jax.grad(_run_with_permittivity)(1.0)


# ---------------------------------------------------------------------------
# 5. Failure/resource test: require_gradients=True must be rejected before
#    any solver call.
# ---------------------------------------------------------------------------


def test_require_gradients_rejected_before_any_solver_call() -> None:
    adapter = FdtdxAdapter()
    request = _smoke_request(require_gradients=True)

    with patch("multiscale_optics_agent.adapters.fdtdx_adapter._import_fdtdx") as mock_import:
        mock_import.side_effect = AssertionError(
            "solver import must not be attempted when require_gradients=True"
        )
        with pytest.raises(UnsupportedCapabilityError):
            adapter.run(request)

    mock_import.assert_not_called()

    # Same eager rejection must be visible to compile-time graph validation,
    # not just at run() time.
    report = adapter.validate_request(request)
    assert not report.valid
    assert any(issue.code == "UNSUPPORTED_GRADIENTS" for issue in report.errors)
