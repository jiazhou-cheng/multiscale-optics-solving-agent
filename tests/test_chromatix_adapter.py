"""Tests for the M_WAVE_CHROMATIX adapter (chromatix_adapter.py).

Scope reminder (see the adapter module docstring for the full contract):
this adapter implements exactly one path -- scalar, monochromatic
angular-spectrum propagation via ``chromatix.functional.asm_propagate`` --
and deliberately rejects everything else (other propagation kernels, vector
fields, gradients) with ``UnsupportedCapabilityError`` before any solver
call. Every test below is marked ``@pytest.mark.jax`` and
``@pytest.mark.integration`` because even the "failure" tests import the
adapter module, which is only meaningfully exercised together with the real
pinned chromatix/jax install described in
``knowledge/solvers/chromatix/solver_card.yaml``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from conftest import load_probe_expected

from multiscale_optics_agent.adapters import chromatix_adapter as mod
from multiscale_optics_agent.adapters.base import ModelRunRequest, RunStatus
from multiscale_optics_agent.core.artifacts import ArtifactRecord
from multiscale_optics_agent.core.errors import AdapterDependencyError, UnsupportedCapabilityError
from multiscale_optics_agent.core.specs import ArtifactKind

pytestmark = [pytest.mark.jax, pytest.mark.integration]

_PHASOR = "exp(-i omega t)"
_COORD_FRAME = "axes=(y, x) row-major; right-handed; +z propagation"


# NOTE: jax_enable_x64 isolation is handled inside
# chromatix_adapter._do_import_chromatix(), which explicitly forces it off
# on every call rather than relying on ambient process state -- see that
# function's docstring. A test-only fixture here or in conftest.py cannot
# fix this correctly: sax.saxtypes.core sets jax_enable_x64=True as an
# import side effect that (because Python only runs a module body once) does
# not re-fire on a cache-hit `import sax`, so a "reset before every test"
# fixture would end up breaking sax's own precision requirements instead.


def _make_input_record(
    path: Path,
    array: np.ndarray,
    *,
    wavelength: float,
    sample_pitch: float,
) -> ArtifactRecord:
    np.save(path, array)
    return ArtifactRecord(
        id="in-field",
        kind=ArtifactKind.COMPLEX_FIELD,
        uri=str(path),
        shape=tuple(array.shape),
        dtype=str(array.dtype),
        metadata={
            "wavelength": wavelength,
            "sample_pitch": sample_pitch,
            "coordinate_frame": _COORD_FRAME,
            "phasor": _PHASOR,
        },
    )


# ---------------------------------------------------------------------------
# 1. Smoke test
# ---------------------------------------------------------------------------


def test_smoke_scalar_angular_spectrum_propagation(tmp_path: Path) -> None:
    shape = (16, 16)
    dx = 1.0e-6
    wavelength = 5.32e-7
    rng = np.random.default_rng(0)
    u_in = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(np.complex64)

    input_record = _make_input_record(
        tmp_path / "input_field.npy", u_in, wavelength=wavelength, sample_pitch=dx
    )
    adapter = mod.get_adapter()
    request = ModelRunRequest(
        run_id="smoke-run",
        node_id="wave",
        inputs={"input_field": input_record},
        config={
            "propagation": "angular_spectrum",
            "z_m": 5.0e-5,
            "pad_width": 8,
            "output_dir": str(tmp_path / "runs"),
        },
    )

    result = adapter.run(request)

    assert result.status == RunStatus.SUCCEEDED, result.error_message
    assert "output_field" in result.outputs
    out = result.outputs["output_field"]
    assert out.kind == ArtifactKind.COMPLEX_FIELD

    output_port = adapter.spec.output_port("output_field")
    assert output_port is not None
    for key in output_port.provides_metadata:
        assert key in out.metadata, f"missing required output metadata key {key!r}"

    # The field data itself round-trips through the adapter's declared storage.
    u_out = np.load(out.uri)
    assert u_out.shape == out.shape
    assert np.iscomplexobj(u_out)


# ---------------------------------------------------------------------------
# 2. Independent comparison against the recorded propagation probe
# ---------------------------------------------------------------------------


def test_matches_propagation_probe_asm_propagate(tmp_path: Path) -> None:
    expected = load_probe_expected("chromatix", "propagation_probe")
    shape = tuple(expected["shape"])
    dx = expected["dx"]
    wavelength = expected["wavelength"]
    n = expected["n"]
    asm_expected = expected["asm_propagate"]
    z = asm_expected["z"]

    jax, _jnp, _chromatix, cf, _compute_padding_transfer = mod._do_import_chromatix()

    # Build exactly the same plane-wave input the probe used, so the adapter
    # is exercised on the identical physical configuration recorded in
    # knowledge/solvers/chromatix/expected/propagation_probe.json.
    plane = cf.plane_wave(shape=shape, dx=dx, spectrum=wavelength, power=1.0)
    u_in = np.asarray(jax.device_get(plane.u))

    input_record = _make_input_record(
        tmp_path / "input_field.npy", u_in, wavelength=wavelength, sample_pitch=dx
    )
    adapter = mod.get_adapter()
    request = ModelRunRequest(
        run_id="probe-cmp",
        node_id="wave",
        inputs={"input_field": input_record},
        config={
            "propagation": "angular_spectrum",
            "z_m": z,
            "refractive_index": n,
            "output_dir": str(tmp_path / "runs"),
            # pad_width intentionally omitted: exercise the adapter's own
            # compute_padding_transfer call and check it reproduces the
            # probe's recorded pad width.
        },
    )

    result = adapter.run(request)

    assert result.status == RunStatus.SUCCEEDED, result.error_message
    out = result.outputs["output_field"]

    assert out.metadata["pad_width"] == asm_expected["computed_pad_width"]
    assert tuple(out.shape) == tuple(asm_expected["u_shape"])
    assert out.dtype == asm_expected["u_dtype"]
    assert out.metadata["sample_pitch"] == pytest.approx(tuple(asm_expected["dx"]), rel=1e-5)
    assert result.diagnostics["power_out"] == pytest.approx(asm_expected["power"], rel=1e-4)


# ---------------------------------------------------------------------------
# 3. Gradient regression check (narrow scalar path only)
# ---------------------------------------------------------------------------


def test_gradient_probe_regression_thin_lens_transform_propagate() -> None:
    """Regression-check the one narrow gradient path the knowledge pack probed.

    This does NOT exercise the adapter's ``run()`` -- the adapter only
    implements ``asm_propagate`` (forward, no gradient claim; see
    ``ChromatixAdapter._check_capability``, which rejects
    ``require_gradients=True`` outright). ``gradient_probe.json`` instead
    covers ``thin_lens`` -> ``transform_propagate``, a path this adapter does
    not implement at all. This test reproduces that probe's exact
    computation directly against chromatix (reusing the adapter's lazy
    import helper) so a future chromatix upgrade that silently changes this
    narrow, previously-verified path is caught. Per CLAUDE.md section 6.2,
    comparing native autodiff to a *finite-difference* directional
    derivative (not another reverse-mode call) is the independent check;
    this is exactly what is reproduced here.
    """
    expected = load_probe_expected("chromatix", "gradient_probe")

    jax, jnp, _chromatix, cf, _compute_padding_transfer = mod._do_import_chromatix()

    shape = (64, 64)
    dx = 1.0
    wavelength = 0.532
    n = 1.0

    def objective(f: float):
        field = cf.plane_wave(shape=shape, dx=dx, spectrum=wavelength, power=1.0)
        field = cf.thin_lens(field, f=f, n=n)
        field = cf.transform_propagate(field, z=f, n=n, pad_width=32)
        return jnp.sum(field.intensity)

    f0 = expected["f0"]
    eps = expected["fd_step"]

    value = float(objective(f0))
    grad_ad = float(jax.grad(objective)(f0))
    grad_fd = float((objective(f0 + eps) - objective(f0 - eps)) / (2 * eps))
    relative_error = abs(grad_ad - grad_fd) / abs(grad_fd)

    assert value == pytest.approx(expected["objective_value"], rel=1e-4)
    assert grad_ad == pytest.approx(expected["grad_native_autodiff"], rel=1e-4)
    assert grad_fd == pytest.approx(expected["grad_finite_difference"], rel=1e-4)
    # Regression bound on the AD-vs-finite-difference agreement itself,
    # matching the tolerance the probe established (do not tighten this
    # blindly -- it is a directional-derivative check on a single point,
    # not a converged Richardson extrapolation).
    assert relative_error < 10 * expected["relative_error"]


# ---------------------------------------------------------------------------
# 4a. Dependency failure surfaces as AdapterDependencyError
# ---------------------------------------------------------------------------


def test_missing_chromatix_dependency_surfaces_as_adapter_dependency_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _boom() -> None:
        raise ImportError("simulated: chromatix is not installed in this environment")

    monkeypatch.setattr(mod, "_do_import_chromatix", _boom)

    adapter = mod.get_adapter()
    request = ModelRunRequest(
        run_id="dep-fail",
        node_id="wave",
        inputs={},
        config={"propagation": "angular_spectrum", "z_m": 1.0e-4},
    )

    with pytest.raises(AdapterDependencyError):
        adapter.run(request)


# ---------------------------------------------------------------------------
# 4b. Unsupported-capability requests are rejected before any solver call
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "config",
    [
        pytest.param({"propagation": "fresnel", "z_m": 1.0e-4}, id="wrong-propagation-kernel"),
        pytest.param(
            {"propagation": "angular_spectrum", "field_kind": "vector", "z_m": 1.0e-4},
            id="vector-field-requested",
        ),
        pytest.param({}, id="propagation-key-missing"),
    ],
)
def test_unsupported_capability_rejected_before_any_chromatix_call(
    config: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _must_not_be_called() -> None:
        raise AssertionError(
            "chromatix must not be imported for a request _check_capability should reject"
        )

    monkeypatch.setattr(mod, "_do_import_chromatix", _must_not_be_called)

    adapter = mod.get_adapter()
    request = ModelRunRequest(run_id="cap-fail", node_id="wave", inputs={}, config=config)

    with pytest.raises(UnsupportedCapabilityError):
        adapter.run(request)


def test_require_gradients_rejected_before_any_chromatix_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _must_not_be_called() -> None:
        raise AssertionError("chromatix must not be imported when require_gradients=True")

    monkeypatch.setattr(mod, "_do_import_chromatix", _must_not_be_called)

    adapter = mod.get_adapter()
    request = ModelRunRequest(
        run_id="grad-fail",
        node_id="wave",
        inputs={},
        config={"propagation": "angular_spectrum", "z_m": 1.0e-4},
        require_gradients=True,
    )

    with pytest.raises(UnsupportedCapabilityError):
        adapter.run(request)


# ---------------------------------------------------------------------------
# Bonus: validate_request / estimate sanity (not strictly required, cheap)
# ---------------------------------------------------------------------------


def test_validate_request_reports_missing_z_m() -> None:
    adapter = mod.get_adapter()
    request = ModelRunRequest(
        run_id="r",
        node_id="wave",
        inputs={},
        config={"propagation": "angular_spectrum"},
    )
    report = adapter.validate_request(request)
    assert not report.valid
    codes = {issue.code for issue in report.errors}
    assert "CHROMATIX_MISSING_CONFIG" in codes
    assert "CHROMATIX_MISSING_INPUT" in codes


def test_spec_loaded_from_registry(registry) -> None:
    adapter = mod.get_adapter()
    assert adapter.spec.id == mod.MODEL_ID
    assert adapter.spec.model_dump() == registry.models[mod.MODEL_ID].model_dump()
