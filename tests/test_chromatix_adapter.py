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

import json
from pathlib import Path

import numpy as np
import pytest
from conftest import load_probe_expected

from multiscale_optics_agent.adapters import chromatix_adapter as mod
from multiscale_optics_agent.adapters.base import ModelRunRequest, RunStatus
from multiscale_optics_agent.core.artifacts import ArtifactRecord
from multiscale_optics_agent.core.errors import AdapterDependencyError, UnsupportedCapabilityError
from multiscale_optics_agent.core.specs import ArtifactKind

pytestmark = [pytest.mark.jax, pytest.mark.integration, pytest.mark.chromatix]

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


@pytest.mark.slow
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
    narrow, previously-verified path is caught. Under the repository gradient-verification policy,
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
        # CHE-55 (M3.5): the graph-facing path used to have no device gate at
        # all, unlike run_standalone's CHROMATIX_UNSUPPORTED_DEVICE check --
        # it would silently report whatever jax.default_backend() happened to
        # be. Proves that gap is closed.
        pytest.param(
            {"propagation": "angular_spectrum", "z_m": 1.0e-4, "device": "gpu"},
            id="gpu-device-requested",
        ),
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


# ---------------------------------------------------------------------------
# 5. CHE-14 standalone wave baseline (ChromatixWaveRequest / run_standalone)
# ---------------------------------------------------------------------------

# The canonical CHE-14 case. `knowledge/` is not an importable package, so
# these mirror the constants in
# knowledge/solvers/chromatix/probes/standalone_baseline.py;
# test_standalone_baseline_case_matches_the_probe_that_generated_it below
# fails if the two ever drift apart.
WAVELENGTH_M = 532e-9
REFRACTIVE_INDEX = 1.0
WAIST_M = 10e-6
GRID = 128
PITCH_M = 0.5e-6
PAD_WIDTH = 256


def rayleigh_range_m(waist_m: float, wavelength_m: float, refractive_index: float) -> float:
    return float(np.pi * waist_m**2 * refractive_index / wavelength_m)


def gaussian_waist_field(grid: int, pitch_m: float, waist_m: float) -> np.ndarray:
    coordinates = (np.arange(grid) - grid // 2) * pitch_m
    x = coordinates[None, :]
    y = coordinates[:, None]
    return np.exp(-(x**2 + y**2) / waist_m**2).astype(np.complex64)


def test_standalone_baseline_case_matches_the_probe_that_generated_it() -> None:
    case = load_probe_expected("chromatix", "standalone_baseline")["case"]
    assert case["wavelength_m"] == WAVELENGTH_M
    assert case["refractive_index"] == REFRACTIVE_INDEX
    assert case["waist_m"] == WAIST_M
    assert case["grid"] == GRID
    assert case["sample_pitch_m"] == PITCH_M
    assert case["pad_width"] == PAD_WIDTH
    assert case["z_m"] == pytest.approx(
        rayleigh_range_m(WAIST_M, WAVELENGTH_M, REFRACTIVE_INDEX), rel=1e-12
    )


def _baseline_payload(output_directory: Path, **overrides) -> dict:
    payload = {
        "input_field_array": gaussian_waist_field(GRID, PITCH_M, WAIST_M),
        "wavelength_m": WAVELENGTH_M,
        "sample_pitch_m": (PITCH_M, PITCH_M),
        "z_m": rayleigh_range_m(WAIST_M, WAVELENGTH_M, REFRACTIVE_INDEX),
        "refractive_index": REFRACTIVE_INDEX,
        "padding_policy": "explicit",
        "pad_width": PAD_WIDTH,
        "output_directory": output_directory,
    }
    payload.update(overrides)
    return payload


def test_standalone_baseline_contract_and_determinism(tmp_path: Path) -> None:
    adapter = mod.ChromatixAdapter()
    first = adapter.run_standalone(_baseline_payload(tmp_path / "first"))
    second = adapter.run_standalone(_baseline_payload(tmp_path / "second"))

    assert first.status is RunStatus.SUCCEEDED, first.failure
    assert second.status is RunStatus.SUCCEEDED, second.failure

    # Pinned package identity is read from the *installed* distribution.
    assert first.package_version == "0.6.0"
    assert first.package_commit == mod._PINNED_COMMIT
    assert first.propagation == "angular_spectrum"
    assert first.device == "cpu"
    assert first.dtype == "complex64"

    # Shapes/spacing are reported; padding is recorded and nothing is cropped.
    assert first.input_shape == (GRID, GRID)
    assert first.output_shape == (GRID + 2 * PAD_WIDTH, GRID + 2 * PAD_WIDTH)
    assert first.pad_width == PAD_WIDTH
    assert first.padded is True
    assert first.cropped is False
    assert first.summary_metrics["resampled"] is False
    assert first.summary_metrics["sample_pitch_unchanged"] is True

    # Power bookkeeping and the finite-window interpretation are emitted.
    assert first.summary_metrics["power_conservation_ratio"] == pytest.approx(1.0, abs=1e-4)
    assert first.summary_metrics["input_edge_energy_fraction"] < 1e-6
    assert "not radiometric watts" in first.summary_metrics["finite_window_interpretation"]

    # Runtime budget for the canonical smoke case.
    assert first.runtime_seconds is not None and first.runtime_seconds < 10.0

    # Determinism: identical summary metrics, metadata, and field hashes.
    assert first.summary_metrics == second.summary_metrics
    assert first.scientific_array_sha256 == second.scientific_array_sha256
    assert first.output_field_sha256 == second.output_field_sha256
    assert json.loads(Path(str(first.summary_path)).read_text()) == json.loads(
        Path(str(second.summary_path)).read_text()
    )

    # The saved artifact holds inspectable complex amplitudes, not intensities.
    u_out = np.load(str(first.output_field_path))
    assert np.iscomplexobj(u_out)
    assert u_out.dtype == np.complex64
    assert u_out.shape == first.output_shape
    assert np.any(u_out.imag != 0.0)
    u_in = np.load(str(first.input_field_path))
    assert np.iscomplexobj(u_in) and u_in.shape == (GRID, GRID)


def test_standalone_baseline_declares_every_required_convention(tmp_path: Path) -> None:
    result = mod.ChromatixAdapter().run_standalone(_baseline_payload(tmp_path / "run"))
    assert result.status is RunStatus.SUCCEEDED, result.failure

    metadata = result.field_metadata
    required = {
        "axis_order",
        "origin",
        "handedness",
        "plus_z",
        "coordinate_frame",
        "reference_plane",
        "wavelength_m",
        "sample_pitch_m",
        "phasor",
        "normalization",
        "dtype",
        "device",
        "package_version",
        "package_commit",
    }
    assert required <= set(metadata)
    assert metadata["axis_order"] == "(y, x)"
    assert metadata["handedness"] == "right-handed"
    assert metadata["phasor"] == _PHASOR
    assert metadata["dtype"] == "complex64"
    assert metadata["device"] == "cpu"
    assert "amplitude" in metadata["values_are"]

    # summary.json carries the same contract and excludes environment facts,
    # so two runs on different hosts still compare equal.
    summary = json.loads(Path(str(result.summary_path)).read_text())
    assert "cpu_device" not in summary["field_metadata"]
    assert "jax_backend" not in summary["field_metadata"]
    assert summary["field_metadata"]["axis_order"] == "(y, x)"


def test_standalone_baseline_matches_recorded_expected_fixture(tmp_path: Path) -> None:
    expected = load_probe_expected("chromatix", "standalone_baseline")
    assert expected["status"] == "passed"
    assert expected["deterministic"] is True
    assert expected["identical_field_arrays"] is True

    result = mod.ChromatixAdapter().run_standalone(_baseline_payload(tmp_path / "run"))
    assert result.status is RunStatus.SUCCEEDED, result.failure
    assert result.summary_metrics == expected["stable_result"]["summary_metrics"]
    assert result.scientific_array_sha256 == expected["stable_result"]["scientific_array_sha256"]
    assert result.output_field_sha256 == expected["stable_result"]["output_field_sha256"]


@pytest.mark.parametrize(
    ("overrides", "expected_code", "expected_stage"),
    [
        pytest.param(
            {"propagation": "fresnel"},
            "CHROMATIX_UNSUPPORTED_PROPAGATION",
            "capability_gate",
            id="wrong-propagation-kernel",
        ),
        pytest.param(
            {"field_kind": "vector"},
            "CHROMATIX_UNSUPPORTED_FIELD_KIND",
            "capability_gate",
            id="vector-field",
        ),
        pytest.param(
            {"require_gradients": True},
            "CHROMATIX_GRADIENTS_NOT_SUPPORTED",
            "capability_gate",
            id="gradient-request",
        ),
        pytest.param(
            {"device": "gpu"}, "CHROMATIX_UNSUPPORTED_DEVICE", "capability_gate", id="gpu-device"
        ),
        pytest.param(
            {"phasor": "  "},
            "CHROMATIX_INVALID_METADATA",
            "metadata_validation",
            id="blank-phasor-metadata",
        ),
        pytest.param(
            {"coordinate_frame": ""},
            "CHROMATIX_INVALID_METADATA",
            "metadata_validation",
            id="blank-coordinate-frame",
        ),
        pytest.param(
            {"wavelength_m": 0.0},
            "CHROMATIX_INVALID_SAMPLING",
            "sampling_validation",
            id="zero-wavelength",
        ),
        pytest.param(
            {"sample_pitch_m": (PITCH_M, -1.0)},
            "CHROMATIX_INVALID_SAMPLING",
            "sampling_validation",
            id="negative-pitch",
        ),
        pytest.param(
            {"padding_policy": "explicit", "pad_width": None},
            "CHROMATIX_INVALID_PADDING",
            "padding_validation",
            id="explicit-padding-without-width",
        ),
        pytest.param(
            {"padding_policy": "auto_transfer", "pad_width": None},
            "CHROMATIX_RESOURCE_ESTIMATE_EXCEEDED",
            "resource_estimate",
            id="excessive-resource-estimate",
        ),
    ],
)
def test_standalone_baseline_failures_are_structured_and_produce_no_field(
    tmp_path: Path, overrides: dict, expected_code: str, expected_stage: str
) -> None:
    result = mod.ChromatixAdapter().run_standalone(
        _baseline_payload(tmp_path / "failed", **overrides)
    )

    assert result.status is RunStatus.FAILED
    assert result.failure is not None
    assert result.failure.code == expected_code
    assert result.failure.stage == expected_stage
    # No fabricated science on a failure path.
    assert result.output_field_path is None
    assert result.summary_metrics == {}
    assert result.output_shape is None
    assert result.summary_metrics.get("power_out") is None


def test_standalone_baseline_rejects_real_and_non_finite_input_fields(tmp_path: Path) -> None:
    adapter = mod.ChromatixAdapter()
    field = gaussian_waist_field(GRID, PITCH_M, WAIST_M)

    real_valued = adapter.run_standalone(
        _baseline_payload(tmp_path / "real", input_field_array=np.abs(field).astype(np.float32))
    )
    assert real_valued.status is RunStatus.FAILED
    assert real_valued.failure is not None
    assert real_valued.failure.code == "CHROMATIX_INPUT_FIELD_NOT_COMPLEX"

    corrupted = field.copy()
    corrupted[3, 7] = np.inf
    non_finite = adapter.run_standalone(
        _baseline_payload(tmp_path / "nonfinite", input_field_array=corrupted)
    )
    assert non_finite.status is RunStatus.FAILED
    assert non_finite.failure is not None
    assert non_finite.failure.code == "CHROMATIX_INPUT_FIELD_NOT_FINITE"

    missing = adapter.run_standalone(
        _baseline_payload(
            tmp_path / "missing",
            input_field_array=None,
            input_field_path=tmp_path / "does-not-exist.npy",
        )
    )
    assert missing.status is RunStatus.FAILED
    assert missing.failure is not None
    assert missing.failure.code == "CHROMATIX_INPUT_FIELD_UNREADABLE"


def test_standalone_baseline_missing_dependency_is_structured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _boom() -> None:
        raise ImportError("simulated: chromatix is not installed in this environment")

    monkeypatch.setattr(mod, "_do_import_chromatix", _boom)

    result = mod.ChromatixAdapter().run_standalone(_baseline_payload(tmp_path / "failed"))

    assert result.status is RunStatus.FAILED
    assert result.failure is not None
    assert result.failure.code == "CHROMATIX_DEPENDENCY_UNAVAILABLE"
    assert result.failure.stage == "dependency_gate"
    assert result.output_field_path is None


def test_standalone_baseline_reads_field_from_npy_path(tmp_path: Path) -> None:
    """The file-backed source must reproduce the in-memory source exactly."""
    field = gaussian_waist_field(GRID, PITCH_M, WAIST_M)
    field_path = tmp_path / "input.npy"
    np.save(field_path, field)

    adapter = mod.ChromatixAdapter()
    from_memory = adapter.run_standalone(_baseline_payload(tmp_path / "memory"))
    from_disk = adapter.run_standalone(
        _baseline_payload(tmp_path / "disk", input_field_array=None, input_field_path=field_path)
    )

    assert from_disk.status is RunStatus.SUCCEEDED, from_disk.failure
    assert from_disk.scientific_array_sha256 == from_memory.scientific_array_sha256


def test_standalone_baseline_output_mode_same_records_the_crop(tmp_path: Path) -> None:
    """A crop only happens when asked for, and is never silent."""
    result = mod.ChromatixAdapter().run_standalone(
        _baseline_payload(tmp_path / "same", output_mode="same")
    )

    assert result.status is RunStatus.SUCCEEDED, result.failure
    assert result.output_shape == (GRID, GRID)
    assert result.cropped is True
    assert result.summary_metrics["cropped"] is True
    assert result.summary_metrics["pad_width"] == PAD_WIDTH
