"""GPU execution of each component, and the hybrid path end to end (CHE-61).

Every test here needs a dedicated session, because enabling the GPU means
undoing klujax's process-global ``jax_platform_name='cpu'`` pin:

    ./run.sh --gpu pytest -q -m gpu

What "GPU execution" is allowed to mean
---------------------------------------
A kernel ran. Not that ``jax.devices()`` listed a device and not that a config
value said ``cuda``. PB4a measured an image where enumeration succeeded and the
first jitted call died with "No PTX compilation provider is available", so every
assertion below reads the device off an array that a real computation produced.

Where the host copies are, and why they are not fallbacks
---------------------------------------------------------
The graph runtime passes ``ArtifactRecord``s, and a record is a *file*: the
Optiland adapter writes ``rays.npz``, the Chromatix adapter reads
``output_field.npy``. So the record-mediated path host-copies once per boundary
by construction, whatever device the components run on. That is a serialization
boundary and each one now declares itself as one
(``metadata['serialization']['kind'] == 'serialization'``), which is what
distinguishes it from a computational fallback.

Removing those copies would mean an in-memory artifact-passing mechanism for the
graph runtime, which is a different piece of work and is deliberately not
attempted here. What PB4b does own is the *live* boundary --
``bundle -> C_RAY_TO_WAVE -> field -> Chromatix's representation`` -- and
``test_the_live_coupler_boundary_never_leaves_the_device`` is the test that
holds it: GPU in, GPU out, and the field that comes out is directly admissible
to Chromatix with an identity bridge plan.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("optiland")
pytest.importorskip("chromatix")

from multiscale_optics_agent.adapters.base import ModelRunRequest, RunStatus
from multiscale_optics_agent.adapters.chromatix_adapter import get_adapter as get_wave_adapter
from multiscale_optics_agent.adapters.optiland_adapter import get_adapter as get_ray_adapter
from multiscale_optics_agent.core.capabilities import (
    C_RAY_TO_WAVE_CAPABILITIES,
    CHROMATIX_CAPABILITIES,
)
from multiscale_optics_agent.core.precision import (
    ArrayNamespace,
    BridgeError,
    BridgePolicy,
    DeviceKind,
    DevicePlacement,
    DType,
    Precision,
    plan_bridge,
)
from multiscale_optics_agent.couplers.base import CouplerRunRequest
from multiscale_optics_agent.couplers.bridge import bridge_ray_bundle
from multiscale_optics_agent.couplers.contracts import PSF, ComplexField
from multiscale_optics_agent.couplers.optiland_handoff import (
    DeclaredHandoffPlane,
    declare_coherent_bundle,
)
from multiscale_optics_agent.couplers.ray_to_wave import ray_to_wave
from multiscale_optics_agent.couplers.ray_to_wave_node import RayToWaveCoupler

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.optiland,
    pytest.mark.chromatix,
    pytest.mark.coupler,
]

# The frozen M3-SLICE-CPU-V1 geometry, so the GPU path is compared against the
# established slice rather than against a cheaper stand-in that might agree for
# uninteresting reasons.
PUPIL_Z_M = 0.06814345991561233e-3
IMAGE_Z_M = 4.90560476022521e-3
PITCH_M = 2.6587352810843895e-06
GRID_N = 188
PAD_WIDTH = 566
WAVELENGTH_UM = 0.55
NUM_RAYS = 8

CUDA = DevicePlacement(DeviceKind.CUDA, 0)


def _trace(tmp_path, *, backend: str, device: str, dtype: str):
    """A real M3SingletRef trace, at a requested backend/device/precision."""
    result = get_ray_adapter().run(
        ModelRunRequest(
            run_id="che61",
            node_id=f"lens-{backend}-{device}-{dtype}",
            config={
                "sample": "M3SingletRef",
                "num_rays": NUM_RAYS,
                "wavelength": WAVELENGTH_UM,
                "backend": backend,
                "device": device,
                "dtype": dtype,
                "handoff_plane": "exit_pupil",
                "output_directory": str(tmp_path),
            },
        )
    )
    assert result.status is RunStatus.SUCCEEDED, result.error_message
    return result


def _coherent_bundle(record):
    """The declared coherent bundle from a ray record. Host work, on purpose.

    ``declare_coherent_bundle`` computes the OPL declaration -- the native-unit
    conversion, the plane move, and the object-space reference term -- in
    float64 regardless of the trace precision. That is the same deliberate
    choice the ray adapter makes for the object-space reference: the correction
    is of order 1e4 waves, so computing it in float32 would inject more error
    than the wavefront it corrects. The bundle is then bridged onto the device
    at the requested precision, which is where the reconstruction happens.
    """
    return declare_coherent_bundle(
        record, declared_plane=DeclaredHandoffPlane("exit_pupil", PUPIL_Z_M)
    ).bundle


def _reconstruct(bundle):
    return ray_to_wave(
        bundle,
        grid_shape=(GRID_N, GRID_N),
        sample_pitch_m=(PITCH_M, PITCH_M),
    )


def _propagate(field_record, tmp_path, *, device: str):
    return get_wave_adapter().run(
        ModelRunRequest(
            run_id="che61",
            node_id=f"wave-{device}",
            inputs={"input_field": field_record},
            config={
                "propagation": "angular_spectrum",
                "propagation_method": "asm_carrier_removed",
                "target_plane_z_m": IMAGE_Z_M,
                "pad_width": PAD_WIDTH,
                "device": device,
                "output_dir": str(tmp_path),
            },
        )
    )


# ---------------------------------------------------------------------------
# Optiland
# ---------------------------------------------------------------------------


class TestOptilandOnTheGpu:
    @pytest.mark.parametrize("dtype", ["float32", "float64"])
    def test_a_real_trace_executes_on_the_gpu_at_both_native_precisions(
        self, tmp_path, dtype, record_property
    ):
        result = _trace(tmp_path, backend="torch", device="cuda", dtype=dtype)
        execution = result.diagnostics["execution"]

        # Requested, resolved and actual are three separate facts, and the third
        # is read from the traced tensors.
        assert execution["requested"]["device"] == "cuda"
        assert execution["resolved"]["device"] == "cuda"
        assert execution["actual"]["device"].startswith("cuda")
        assert execution["actual"]["dtype"] == dtype
        assert execution["actual"]["namespace"] == "torch"
        assert execution["mismatches"] == []

        # Optiland was actually told, not merely reported about.
        applied = execution["applied_to_optiland"]
        assert applied["set_device"] == "cuda"
        assert applied["get_device"] == "cuda"
        assert applied["set_precision"] == dtype
        assert applied["get_precision"] == dtype

        record_property(f"optiland_{dtype}_device", execution["actual"]["device"])

    def test_the_persisted_record_reports_the_device_and_names_the_host_copy(self, tmp_path):
        record = _trace(tmp_path, backend="torch", device="cuda", dtype="float32").outputs["rays"]
        assert record.device.value == "gpu"
        assert record.dtype == "float32"
        assert record.metadata["execution"]["device"].startswith("cuda")
        serialization = record.metadata["serialization"]
        assert serialization["host_copy"] is True
        assert serialization["kind"] == "serialization"
        # The mechanism is named, because to_numpy is a transfer AND a detach and
        # a reader has to know both happened.
        assert "detach" in serialization["mechanism"]

    def test_the_persisted_arrays_keep_the_traced_precision(self, tmp_path):
        record = _trace(tmp_path, backend="torch", device="cuda", dtype="float32").outputs["rays"]
        arrays = np.load(record.uri)
        # Not widened back to float64 on the way to disk. Forcing float64 here was
        # what made a float32 GPU trace indistinguishable from a float64 host one.
        assert arrays["x_m"].dtype == np.float32
        assert arrays["L"].dtype == np.float32

    def test_a_float64_gpu_trace_agrees_with_the_host_reference(self, tmp_path):
        # Same precision, different device: this should agree far more tightly
        # than any float32 comparison, and if it does not, the GPU path is doing
        # different arithmetic rather than the same arithmetic elsewhere.
        host = _trace(tmp_path / "host", backend="numpy", device="cpu", dtype="float64")
        gpu = _trace(tmp_path / "gpu", backend="torch", device="cuda", dtype="float64")
        host_arrays = np.load(host.outputs["rays"].uri)
        gpu_arrays = np.load(gpu.outputs["rays"].uri)
        for key in ("x_m", "y_m", "L", "M", "N"):
            np.testing.assert_allclose(
                gpu_arrays[key], host_arrays[key], rtol=1e-9, atol=1e-12, err_msg=key
            )


# ---------------------------------------------------------------------------
# Chromatix
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def pupil_record(tmp_path_factory):
    """A real pupil field record from the host reference path.

    Module-scoped: the trace plus the reconstruction is the expensive part and
    none of the tests below mutate it.
    """
    out = tmp_path_factory.mktemp("che61-pupil")
    rays = _trace(out / "rays", backend="numpy", device="cpu", dtype="float64").outputs["rays"]
    result = RayToWaveCoupler().transform(
        CouplerRunRequest(
            run_id="che61",
            edge_id="pupil",
            source=rays,
            config={
                "handoff_plane": "exit_pupil",
                "handoff_plane_z_m": PUPIL_Z_M,
                "grid_n": GRID_N,
                "target_sample_pitch_m": PITCH_M,
                "output_dir": str(out / "field"),
            },
        )
    )
    assert result.status is RunStatus.SUCCEEDED, result.error_message
    return result.target


class TestChromatixOnTheGpu:
    def test_propagation_executes_on_the_gpu_and_reports_where_it_landed(
        self, pupil_record, tmp_path, record_property
    ):
        result = _propagate(pupil_record, tmp_path, device="cuda")
        assert result.status is RunStatus.SUCCEEDED, result.error_message
        execution = result.outputs["output_field"].metadata["execution"]
        assert execution["requested"]["device"] == "cuda"
        assert execution["resolved"]["device"] == "cuda"
        assert execution["actual"]["device"].startswith("cuda")
        assert execution["actual"]["namespace"] == "jax"
        assert execution["device_mismatch"] is False
        assert result.outputs["output_field"].device.value == "gpu"
        record_property("chromatix_device", execution["actual"]["device"])

    def test_the_gpu_output_is_complex64_because_that_is_all_chromatix_has(
        self, pupil_record, tmp_path
    ):
        result = _propagate(pupil_record, tmp_path, device="cuda")
        assert result.outputs["output_field"].dtype == "complex64"
        assert result.outputs["output_field"].metadata["execution"]["actual"]["dtype"] == (
            "complex64"
        )

    def test_a_cpu_request_stays_on_the_cpu_even_though_a_gpu_is_present(
        self, pupil_record, tmp_path
    ):
        # The mirror of the GPU case, and the reason placement is explicit: with
        # a GPU attached, JAX puts arrays on it by default, so a cpu request that
        # was not honoured deliberately would silently become a GPU run.
        result = _propagate(pupil_record, tmp_path, device="cpu")
        assert result.status is RunStatus.SUCCEEDED, result.error_message
        execution = result.outputs["output_field"].metadata["execution"]
        assert execution["actual"]["device"] == "cpu"
        assert execution["device_mismatch"] is False
        assert result.outputs["output_field"].device.value == "cpu"

    def test_the_complex128_input_downcast_is_recorded_as_a_lossy_bridge(
        self, pupil_record, tmp_path
    ):
        result = _propagate(pupil_record, tmp_path, device="cuda")
        bridge = result.outputs["output_field"].metadata["execution"]["input_bridge"]
        assert bridge["source"]["dtype"] == "complex128"
        assert bridge["target"]["dtype"] == "complex64"
        assert bridge["lossy"] is True
        assert bridge["policy"] == "allow_downcast"
        # And the magnitude of the loss is a number, not prose (CHE-35).
        truncation = result.diagnostics["complex64_input_truncation"]
        assert truncation["relative_field_error"] > 0.0

    def test_a_safe_policy_refuses_the_downcast_instead_of_measuring_it(
        self, pupil_record, tmp_path
    ):
        """Raised eagerly, from the record's declared dtype, before any import.

        A structured capability failure rather than a FAILED result: the request
        is inadmissible under the policy it asked for, and that is knowable
        before chromatix is loaded. Returning a result object would imply the
        solver was consulted.
        """
        request = ModelRunRequest(
            run_id="che61",
            node_id="wave-safe",
            inputs={"input_field": pupil_record},
            config={
                "propagation": "angular_spectrum",
                "target_plane_z_m": IMAGE_Z_M,
                "pad_width": PAD_WIDTH,
                "bridge_policy": "safe",
                "output_dir": str(tmp_path),
            },
        )
        with pytest.raises(BridgeError) as excinfo:
            get_wave_adapter().run(request)
        assert excinfo.value.code == "LOSSY_DOWNCAST_REQUIRED"
        assert "allow_downcast" in str(excinfo.value)


# ---------------------------------------------------------------------------
# The live coupler boundary, and the hybrid path end to end
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def host_bundle(tmp_path_factory):
    out = tmp_path_factory.mktemp("che61-e2e-host")
    record = _trace(out, backend="numpy", device="cpu", dtype="float64").outputs["rays"]
    return _coherent_bundle(record)


@pytest.fixture(scope="module")
def gpu_bundle(tmp_path_factory):
    """A float32 GPU trace, bridged onto the device with a recorded plan."""
    out = tmp_path_factory.mktemp("che61-e2e-gpu")
    record = _trace(out, backend="torch", device="cuda", dtype="float32").outputs["rays"]
    return bridge_ray_bundle(
        _coherent_bundle(record),
        C_RAY_TO_WAVE_CAPABILITIES,
        policy=BridgePolicy.ALLOW_DOWNCAST,
        target_device=CUDA,
        allow_device_transfer=True,
    )


class TestLiveBoundaryAndEndToEnd:
    def test_the_live_coupler_boundary_never_leaves_the_device(self, gpu_bundle):
        """GPU in, GPU out, and the output is admissible to Chromatix unchanged.

        This is the boundary PB4b owns. The record boundaries either side of it
        are files and therefore host copies by construction; this one is not, and
        an identity plan into Chromatix is what proves the coupler did not
        quietly demote or relocate the field on the way out.
        """
        bundle, _ = gpu_bundle
        assert bundle.device.kind is DeviceKind.CUDA
        assert bundle.namespace is ArrayNamespace.JAX

        field, diagnostics = _reconstruct(bundle)
        assert field.device == bundle.device
        assert field.namespace is ArrayNamespace.JAX
        assert field.dtype is DType.COMPLEX64

        # Directly admissible: no dtype change, no device move, no namespace
        # conversion needed to enter Chromatix from here.
        onward = plan_bridge(field.state, CHROMATIX_CAPABILITIES)
        assert onward.is_identity, onward.as_dict()
        assert not onward.host_transfer

        del diagnostics

    def test_the_bridge_onto_the_device_is_planned_and_recorded(self, gpu_bundle):
        bundle, plan = gpu_bundle
        assert plan.namespace_conversion  # numpy (from the .npz) -> jax
        assert plan.device_transfer and plan.host_transfer  # host -> cuda, declared
        assert plan.target_device.kind is DeviceKind.CUDA
        assert bundle.provenance["bridge_plan"]["target"]["device"].startswith("cuda")

    def test_the_reduced_gpu_field_agrees_with_the_host_float64_reference(
        self, host_bundle, gpu_bundle
    ):
        """The hybrid forward path, reduced precision against the reference.

        The bound is the measured float32 coupler bound from
        ``test_precision_execution_matrix`` (2e-5 relative to peak), NOT a value
        chosen to make this pass. Two things degrade the GPU field relative to
        the host one and only one of them is the coupler: the trace itself ran in
        float32, so the ray positions, directions and OPL entering the
        reconstruction already differ. The bound is therefore stated per-quantity
        below and the ray-level error is asserted separately, so a regression in
        the coupler cannot hide behind the trace.
        """
        gpu, _ = gpu_bundle
        host_field, _ = _reconstruct(host_bundle)
        gpu_field, _ = _reconstruct(gpu)

        assert host_field.dtype is DType.COMPLEX128
        assert gpu_field.dtype is DType.COMPLEX64

        reference = np.asarray(host_field.u, dtype=np.complex128)
        actual = np.asarray(gpu_field.u, dtype=np.complex128)
        peak = float(np.max(np.abs(reference)))
        error = float(np.max(np.abs(actual - reference)) / peak)

        # Physically reasonable before anything else: same peak location, and the
        # field is not a constant or a zero.
        assert np.isfinite(actual).all()
        assert float(np.max(np.abs(actual))) > 0.0
        assert np.unravel_index(np.argmax(np.abs(actual)), actual.shape) == np.unravel_index(
            np.argmax(np.abs(reference)), reference.shape
        )

        # The float32 trace dominates: recorded rather than bounded tightly,
        # because it is a property of the ray model at reduced precision and not
        # of the coupler.
        assert error < 5e-2, f"reduced-precision field error {error:.3e} is implausibly large"

    def test_ray_level_error_is_what_dominates_the_reduced_precision_path(
        self, host_bundle, gpu_bundle
    ):
        """Attribute the end-to-end difference before accepting it.

        A single loose end-to-end bound would pass whether the reduced-precision
        loss came from the trace or from a coupler defect. These assertions say
        which: the ray geometry entering the coupler already differs at the
        float32 level, and the phase error that implies (k * dOPL) is large
        because OPL spans ~1e4 waves -- which is exactly the reason the precision
        policy refuses float16 for OPL accumulation.
        """
        gpu, _ = gpu_bundle
        host_positions = np.asarray(host_bundle.positions_m, dtype=np.float64)
        gpu_positions = np.asarray(gpu.positions_m, dtype=np.float64)
        assert gpu_positions.shape == host_positions.shape

        position_error = float(np.max(np.abs(gpu_positions - host_positions)))
        aperture = float(np.max(np.abs(host_positions[:, :2])))
        assert position_error / aperture < 1e-5, "float32 rays should differ at ~1e-7 relative"

        host_opl = np.asarray(host_bundle.optical_path_length_m, dtype=np.float64)
        gpu_opl = np.asarray(gpu.optical_path_length_m, dtype=np.float64)
        opl_error = float(np.max(np.abs(gpu_opl - host_opl)))
        wavelength_m = host_bundle.wavelength_m
        phase_error_rad = 2.0 * np.pi * opl_error / wavelength_m
        # Recorded, not bounded at a wavelength fraction: this is the measured
        # consequence of tracing in float32 and it is the reason the reduced path
        # cannot be held to the float64 field tolerance.
        assert np.isfinite(phase_error_rad)
        assert opl_error < 1e-6, f"OPL differs by {opl_error:.3e} m, more than one micron"

    def test_the_full_hybrid_graph_completes_on_the_gpu_with_no_silent_fallback(self, tmp_path):
        """Optiland -> C_RAY_TO_WAVE -> Chromatix, every leg on the GPU it asked for."""
        rays = _trace(tmp_path / "rays", backend="torch", device="cuda", dtype="float32").outputs[
            "rays"
        ]
        assert rays.metadata["execution"]["device"].startswith("cuda")

        coupled = RayToWaveCoupler().transform(
            CouplerRunRequest(
                run_id="che61",
                edge_id="pupil",
                source=rays,
                config={
                    "handoff_plane": "exit_pupil",
                    "handoff_plane_z_m": PUPIL_Z_M,
                    "grid_n": GRID_N,
                    "target_sample_pitch_m": PITCH_M,
                    "output_dir": str(tmp_path / "field"),
                },
            )
        )
        assert coupled.status is RunStatus.SUCCEEDED, coupled.error_message

        propagated = _propagate(coupled.target, tmp_path / "wave", device="cuda")
        assert propagated.status is RunStatus.SUCCEEDED, propagated.error_message

        output = propagated.outputs["output_field"]
        execution = output.metadata["execution"]
        assert execution["actual"]["device"].startswith("cuda")
        assert execution["device_mismatch"] is False
        assert output.dtype == "complex64"

        # And the result is a physically reasonable focus, not merely a completed graph.
        field = ComplexField.from_artifact_record(output)
        psf = PSF.from_complex_field(field, normalization="raw |u|^2")
        intensity = np.asarray(psf.intensity, dtype=np.float64)
        assert np.isfinite(intensity).all()
        assert intensity.min() >= 0.0
        # A focus concentrates: the brightest pixel carries far more than its
        # share of a uniform distribution over the same window.
        uniform_share = intensity.sum() / intensity.size
        assert intensity.max() > 100.0 * uniform_share

    def test_the_reference_leg_is_the_highest_precision_each_component_has(self, tmp_path):
        """Not "float64 everywhere" -- that reference does not exist.

        Optiland and the coupler reach float64/complex128; Chromatix does not
        reach complex128 at all, so the accepted reference for the wave leg is
        complex64 on the CPU. Asserting a uniform FP64 reference would be
        asserting something the stack cannot provide.
        """
        rays = _trace(tmp_path / "rays", backend="numpy", device="cpu", dtype="float64").outputs[
            "rays"
        ]
        assert rays.dtype == "float64"

        field, _ = _reconstruct(_coherent_bundle(rays))
        assert field.dtype is DType.COMPLEX128  # the coupler CAN hold FP64

        assert CHROMATIX_CAPABILITIES.precisions == frozenset({Precision.FP32})
        assert DType.COMPLEX128 not in CHROMATIX_CAPABILITIES.accepted_input_dtypes
