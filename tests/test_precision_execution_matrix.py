"""Executed dtype/device/namespace matrix for the couplers (CHE-61).

Two things are checked here and they are kept apart on purpose, in the order
PB4b section 18 requires:

1. **Semantics.** Does the output carry the dtype, device and namespace the
   inputs and the resolved execution imply? A number that is numerically fine
   while claiming the wrong precision is the failure this whole ticket exists to
   prevent, so it is checked first and independently.

2. **Numerics.** Only then, does reduced precision agree with the high-precision
   reference to within a tolerance *derived from measurement*?

Every tolerance below is annotated with the number it came from
(``tmp_probes/pb4b_tolerance.py`` on the host and
``tmp_probes/pb4b_tolerance_gpu.py`` on an RTX A6000; 16x16 grid, 500 nm,
1 um pitch, all errors relative to peak against a float64 reference):

=============================  ==========  ====================
path                           max error   reference
=============================  ==========  ====================
NumPy   float64 -> complex128   3.5e-15    analytic exp(i k d.r)
NumPy   float32 -> complex64    2.1e-06    analytic
JAX cpu float32 -> complex64    4.0e-06    analytic
JAX gpu float32 -> complex64    1.9e-06    analytic
round trip complex128 (host)    1.6e-15    the input field (L2)
round trip complex64  (host)    8.3e-07    the input field (L2)
round trip complex64  (gpu)     8.3e-07    the input field (L2)
=============================  ==========  ====================

The FP32 bound is set at 2e-5, five times the worst measured value, which leaves
room for a different BLAS or FFT ordering without leaving room for a precision
bug. It is emphatically *not* the float64 bound relaxed until things passed.

Note what the GPU column does *not* show: a penalty. The same bound holds on the
host and on the device, and it took a fix to get there. XLA:GPU's default
precision for an ``f32``/``c64`` dot on Ampere is TF32, which made the first GPU
measurement of the round trip 2.2e-4 -- 170x the host value -- while the array
still reported ``complex64``. Loosening this bound would have "passed" that and
buried a 1500x accuracy loss in a tolerance constant. The cause and the fix are
in ``core.arrays.matmul_precision_kwargs``.

The GPU half of this matrix lives in ``TestGpuResidency`` and needs a dedicated
session: ``./run.sh --gpu pytest -q -m gpu``.
"""

from __future__ import annotations

import numpy as np
import pytest

from multiscale_optics_agent.core.arrays import array_state, xp_for
from multiscale_optics_agent.core.capabilities import (
    CHROMATIX_CAPABILITIES,
    C_RAY_TO_WAVE_CAPABILITIES,
)
from multiscale_optics_agent.core.precision import (
    ArrayNamespace,
    BridgeError,
    BridgePolicy,
    CapabilityError,
    ComponentCapabilities,
    DeviceKind,
    DevicePlacement,
    DType,
    Precision,
    plan_bridge,
)
from multiscale_optics_agent.couplers.bridge import bridge_complex_field, bridge_ray_bundle
from multiscale_optics_agent.couplers.contracts import (
    ComplexField,
    ContractCode,
    ContractError,
    Frame,
    PSF,
    RayBundle,
    ReferencePlane,
)
from multiscale_optics_agent.couplers.ray_to_wave import (
    collimated_bundle,
    compute_precision_for,
    ray_to_wave,
)
from multiscale_optics_agent.couplers.wave_to_ray import decompose, wave_to_ray

pytestmark = pytest.mark.coupler

WAVELENGTH_M = 500e-9
PITCH_M = 1e-6
N_GRID = 16
GRID = (N_GRID, N_GRID)
PITCH = (PITCH_M, PITCH_M)
PLANE = ReferencePlane(name="matrix plane", z_m=0.0)

#: Derived from the measured 4.0e-6 worst case with 5x headroom. See the module
#: docstring for the full measurement table.
FP32_MAX_REL_ERROR = 2e-5
#: The established float64 path, unchanged by CHE-61. Measured 3.5e-15.
FP64_MAX_REL_ERROR = 1e-13

_DIRECTION = (0.10, -0.05, float(np.sqrt(1.0 - 0.01 - 0.0025)))

#: A destination that executes only in FP32, real and complex. Stands in for the
#: general "reduced-precision backend" case without borrowing a real component's
#: dtype set, so a policy failure here cannot be confused with a Chromatix or
#: Optiland capability change.
_FP32_ONLY = ComponentCapabilities(
    component="T_FP32_ONLY",
    devices=frozenset({DeviceKind.CPU, DeviceKind.CUDA}),
    precisions=frozenset({Precision.FP32}),
    accepted_input_dtypes=frozenset({DType.FLOAT32, DType.COMPLEX64}),
    native_compute_dtypes=frozenset({DType.FLOAT32, DType.COMPLEX64}),
    output_dtypes=frozenset({DType.COMPLEX64}),
    namespaces=frozenset({ArrayNamespace.NUMPY, ArrayNamespace.JAX}),
    device_namespaces={
        DeviceKind.CPU: frozenset({ArrayNamespace.NUMPY, ArrayNamespace.JAX}),
        DeviceKind.CUDA: frozenset({ArrayNamespace.JAX}),
    },
    evidence="test fixture for reduced-precision destinations",
)


def _launch_positions() -> np.ndarray:
    axis = (np.arange(N_GRID) - N_GRID // 2) * PITCH_M
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    return np.column_stack([xx.ravel(), yy.ravel()])


def _bundle(precision: Precision, namespace: ArrayNamespace | None = None) -> RayBundle:
    return collimated_bundle(
        positions_xy_m=_launch_positions(),
        direction=_DIRECTION,
        wavelength_m=WAVELENGTH_M,
        precision=precision,
        namespace=namespace,
    )


def _analytic_field() -> np.ndarray:
    """``exp(+i k d_hat . r) * N`` -- exact, evaluated in float64 on the host.

    The oracle is always float64 whatever the path under test: comparing a
    float32 result against a float32 oracle would hide exactly the error being
    measured.
    """
    k = 2.0 * np.pi / WAVELENGTH_M
    axis = (np.arange(N_GRID, dtype=np.float64) - N_GRID // 2) * PITCH_M
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    count = N_GRID * N_GRID
    return np.exp(1j * k * (_DIRECTION[0] * xx + _DIRECTION[1] * yy)) * count


def _relative_peak_error(actual, expected: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=np.complex128)
    return float(np.max(np.abs(actual - expected)) / np.max(np.abs(expected)))


def _band_limited_field(dtype: DType, namespace: ArrayNamespace | None = None) -> ComplexField:
    """A random field with every evanescent bin removed, at a chosen dtype."""
    rng = np.random.default_rng(20260819)
    base = rng.standard_normal((N_GRID, N_GRID)) + 1j * rng.standard_normal((N_GRID, N_GRID))
    spectrum = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(base)))
    frequency = np.fft.fftshift(np.fft.fftfreq(N_GRID, d=PITCH_M)) * WAVELENGTH_M
    dv, du = np.meshgrid(frequency, frequency, indexing="ij")
    spectrum = np.where(du**2 + dv**2 < 1.0, spectrum, 0.0)
    u = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(spectrum))).astype(str(dtype))
    xp = np if namespace is None else xp_for(namespace)
    return ComplexField(
        u=xp.asarray(u),
        sample_pitch_m=PITCH,
        wavelength_m=WAVELENGTH_M,
        reference_plane=PLANE,
        frame=Frame(),
    )


# ---------------------------------------------------------------------------
# Artifact representation is preserved, not normalized
# ---------------------------------------------------------------------------


class TestArtifactPreservesRepresentation:
    def test_a_python_list_still_becomes_host_float64(self):
        # The historical default. Nothing that used to work changes meaning.
        bundle = RayBundle(
            positions_m=[[0.0, 0.0, 0.0]],
            directions=[[0.0, 0.0, 1.0]],
            wavelength_m=WAVELENGTH_M,
            reference_plane=PLANE,
        )
        assert bundle.state == array_state(np.zeros((1, 3)))
        assert bundle.dtype is DType.FLOAT64

    def test_float32_geometry_is_kept_as_float32(self):
        bundle = _bundle(Precision.FP32)
        assert bundle.dtype is DType.FLOAT32
        assert bundle.amplitude.dtype == np.complex64
        assert bundle.optical_path_length_m.dtype == np.float32

    def test_complex64_field_is_kept_as_complex64(self):
        field = _band_limited_field(DType.COMPLEX64)
        assert field.dtype is DType.COMPLEX64
        assert field.real_dtype is DType.FLOAT32

    def test_jax_namespace_survives_construction(self):
        bundle = _bundle(Precision.FP32, ArrayNamespace.JAX)
        assert bundle.namespace is ArrayNamespace.JAX
        assert bundle.device.kind is DeviceKind.CPU

    def test_a_real_amplitude_widens_to_the_matching_complex_precision(self):
        # float32 weight -> complex64 amplitude, NOT complex128. Preserving
        # precision through a real->complex widening is the whole point.
        bundle = RayBundle(
            positions_m=np.zeros((2, 3), dtype=np.float32),
            directions=np.tile(np.asarray([0.0, 0.0, 1.0], dtype=np.float32), (2, 1)),
            wavelength_m=WAVELENGTH_M,
            reference_plane=PLANE,
            weight=np.asarray([1.0, 4.0], dtype=np.float32),
            weight_semantics="power",
        )
        promoted = bundle.with_amplitude_from_weight(
            mapping="amplitude = sqrt(weight); weight is a power"
        )
        assert promoted.amplitude.dtype == np.complex64

    def test_a_mixed_namespace_artifact_is_refused(self):
        jnp = xp_for(ArrayNamespace.JAX)
        with pytest.raises(ContractError) as excinfo:
            RayBundle(
                positions_m=jnp.zeros((1, 3), dtype=np.float32),
                directions=jnp.asarray([[0.0, 0.0, 1.0]], dtype=np.float32),
                wavelength_m=WAVELENGTH_M,
                reference_plane=PLANE,
                amplitude=np.asarray([1.0 + 0j], dtype=np.complex64),
            )
        assert excinfo.value.code is ContractCode.REPRESENTATION_INCONSISTENT

    def test_a_torch_array_must_cross_the_bridge_rather_than_be_adopted(self):
        torch = pytest.importorskip("torch")
        with pytest.raises(ContractError) as excinfo:
            RayBundle(
                positions_m=torch.zeros((1, 3), dtype=torch.float64),
                directions=torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64),
                wavelength_m=WAVELENGTH_M,
                reference_plane=PLANE,
            )
        assert excinfo.value.code is ContractCode.REPRESENTATION_INCONSISTENT
        assert "explicit bridge" in str(excinfo.value)

    def test_direction_tolerance_follows_the_dtype(self):
        # A float32 bundle whose directions are only unit-norm to float32
        # accuracy is valid. Under the old fixed 1e-9 bound it was not, which
        # made float32 unusable for a reason unrelated to the physics.
        bundle = _bundle(Precision.FP32)
        norms = np.linalg.norm(np.asarray(bundle.directions, dtype=np.float64), axis=1)
        worst = float(np.max(np.abs(norms - 1.0)))
        assert worst > 0.0, "float32 cast should perturb the norm at all"
        assert worst > 1e-9, "otherwise this test proves nothing about the tolerance"


class TestSerializationBoundaryIsExplicit:
    def test_record_reports_the_execution_representation_and_the_host_copy(self, tmp_path):
        field = _band_limited_field(DType.COMPLEX64)
        record = field.to_artifact_record(artifact_id="f", uri=tmp_path / "f.npy")
        assert record.dtype == "complex64"
        assert record.metadata["execution"]["dtype"] == "complex64"
        serialization = record.metadata["serialization"]
        assert serialization["boundary"] == "explicit_persistence"
        # Already on the host, so no copy was needed and none is claimed.
        assert serialization["host_copy"] is False
        assert serialization["kind"] == "already_on_host"

    def test_ray_bundle_record_reports_its_amplitude_dtype_separately(self, tmp_path):
        bundle = _bundle(Precision.FP32)
        record = bundle.to_artifact_record(artifact_id="r", uri=tmp_path / "r.npz")
        assert record.dtype == "float32"
        assert record.metadata["amplitude_dtype"] == "complex64"

    def test_persisted_dtype_is_not_widened_on_reload(self, tmp_path):
        bundle = _bundle(Precision.FP32)
        record = bundle.to_artifact_record(artifact_id="r", uri=tmp_path / "r.npz")
        reloaded = RayBundle.from_artifact_record(record)
        assert reloaded.dtype is DType.FLOAT32


# ---------------------------------------------------------------------------
# C_RAY_TO_WAVE across namespaces and precisions
# ---------------------------------------------------------------------------


class TestRayToWaveExecutionMatrix:
    @pytest.mark.parametrize(
        ("precision", "namespace", "expected_dtype"),
        [
            (Precision.FP64, None, DType.COMPLEX128),
            (Precision.FP32, None, DType.COMPLEX64),
            (Precision.FP32, ArrayNamespace.JAX, DType.COMPLEX64),
        ],
    )
    def test_output_dtype_and_namespace_follow_the_input(
        self, precision, namespace, expected_dtype
    ):
        bundle = _bundle(precision, namespace)
        field, _ = ray_to_wave(bundle, grid_shape=GRID, sample_pitch_m=PITCH)
        assert field.dtype is expected_dtype
        assert field.namespace is bundle.namespace
        assert field.device == bundle.device

    def test_compute_precision_is_derived_from_the_data(self):
        assert compute_precision_for(_bundle(Precision.FP64)) is Precision.FP64
        assert compute_precision_for(_bundle(Precision.FP32)) is Precision.FP32

    def test_diagnostics_distinguish_input_compute_and_actual_output(self):
        bundle = _bundle(Precision.FP32)
        field, _ = ray_to_wave(bundle, grid_shape=GRID, sample_pitch_m=PITCH)
        execution = field.provenance["execution"]
        assert execution["input"]["dtype"] == "float32"
        assert execution["compute_precision"] == "fp32"
        assert execution["compute_dtype"] == "complex64"
        # Observed, from the array that was produced -- not copied from above.
        assert execution["output"]["dtype"] == "complex64"
        assert execution["output"]["namespace"] == "numpy"

    def test_float64_host_path_still_matches_the_analytic_oracle(self):
        field, _ = ray_to_wave(_bundle(Precision.FP64), grid_shape=GRID, sample_pitch_m=PITCH)
        assert _relative_peak_error(field.u, _analytic_field()) < FP64_MAX_REL_ERROR

    @pytest.mark.parametrize("namespace", [None, ArrayNamespace.JAX])
    def test_float32_agrees_with_the_analytic_oracle_at_a_measured_tolerance(self, namespace):
        field, _ = ray_to_wave(
            _bundle(Precision.FP32, namespace), grid_shape=GRID, sample_pitch_m=PITCH
        )
        error = _relative_peak_error(field.u, _analytic_field())
        assert error < FP32_MAX_REL_ERROR
        # And it is genuinely worse than float64 -- otherwise the reduced path is
        # not actually running in reduced precision.
        assert error > FP64_MAX_REL_ERROR

    def test_numpy_and_jax_agree_at_the_same_precision(self):
        # The "one physics implementation" claim, made falsifiable: two
        # namespaces, same source, same answer to float32 round-off.
        host, _ = ray_to_wave(_bundle(Precision.FP32), grid_shape=GRID, sample_pitch_m=PITCH)
        jax_field, _ = ray_to_wave(
            _bundle(Precision.FP32, ArrayNamespace.JAX), grid_shape=GRID, sample_pitch_m=PITCH
        )
        assert _relative_peak_error(jax_field.u, np.asarray(host.u, np.complex128)) < (
            FP32_MAX_REL_ERROR
        )


class TestWaveToRayExecutionMatrix:
    @pytest.mark.parametrize(
        ("dtype", "namespace", "real_dtype"),
        [
            (DType.COMPLEX128, None, "float64"),
            (DType.COMPLEX64, None, "float32"),
            (DType.COMPLEX64, ArrayNamespace.JAX, "float32"),
        ],
    )
    def test_emitted_bundle_matches_the_field_precision(self, dtype, namespace, real_dtype):
        field = _band_limited_field(dtype, namespace)
        bundle, spectrum, _ = wave_to_ray(field)
        assert str(bundle.dtype) == real_dtype
        assert bundle.namespace is field.namespace
        assert spectrum.dtype is dtype

    def test_decomposition_stays_in_the_field_namespace(self):
        field = _band_limited_field(DType.COMPLEX64, ArrayNamespace.JAX)
        spectrum = decompose(field)
        assert spectrum.namespace is ArrayNamespace.JAX
        assert array_state(spectrum.direction_u).dtype is DType.FLOAT32

    @pytest.mark.parametrize(
        ("dtype", "bound"),
        [(DType.COMPLEX128, FP64_MAX_REL_ERROR), (DType.COMPLEX64, FP32_MAX_REL_ERROR)],
    )
    def test_round_trip_reproduces_the_field_at_its_own_precision(self, dtype, bound):
        # Enumerating every propagating bin is the deterministic exactness
        # limit: no sampling error, so what is left is dtype round-off alone.
        field = _band_limited_field(dtype)
        bundle, _, _ = wave_to_ray(field)
        restored, _ = ray_to_wave(bundle, grid_shape=GRID, sample_pitch_m=PITCH)
        assert restored.dtype is dtype
        assert _relative_peak_error(restored.u, np.asarray(field.u, np.complex128)) < bound

    def test_stochastic_draw_is_reproducible_from_a_complex64_spectrum(self):
        # draw_indices renormalizes after widening float32 -> float64; without
        # that numpy rejects p as not summing to 1.
        field = _band_limited_field(DType.COMPLEX64)
        first, _, _ = wave_to_ray(field, count=8, rng=np.random.default_rng(3))
        second, _, _ = wave_to_ray(field, count=8, rng=np.random.default_rng(3))
        np.testing.assert_array_equal(
            first.provenance["sampled_indices"], second.provenance["sampled_indices"]
        )


class TestPsfFollowsTheField:
    def test_a_complex64_field_yields_a_float32_psf(self):
        field = _band_limited_field(DType.COMPLEX64)
        psf = PSF.from_complex_field(field, normalization="raw |u|^2")
        assert psf.dtype is DType.FLOAT32
        assert psf.namespace is ArrayNamespace.NUMPY

    def test_a_complex128_field_still_yields_a_float64_psf(self):
        field = _band_limited_field(DType.COMPLEX128)
        assert PSF.from_complex_field(field, normalization="raw |u|^2").dtype is DType.FLOAT64


# ---------------------------------------------------------------------------
# Bridging into a component with different capabilities
# ---------------------------------------------------------------------------


class TestBridgeExecution:
    def test_an_admissible_field_is_returned_untouched(self):
        field = _band_limited_field(DType.COMPLEX64)
        bridged, plan = bridge_complex_field(field, C_RAY_TO_WAVE_CAPABILITIES)
        assert bridged is field  # identity, not a rebuild
        assert plan.is_identity

    def test_complex128_into_chromatix_is_refused_by_default(self):
        field = _band_limited_field(DType.COMPLEX128)
        with pytest.raises(BridgeError) as excinfo:
            bridge_complex_field(field, CHROMATIX_CAPABILITIES)
        assert excinfo.value.code == "LOSSY_DOWNCAST_REQUIRED"

    def test_explicit_downcast_into_chromatix_converts_and_records_the_loss(self):
        field = _band_limited_field(DType.COMPLEX128)
        bridged, plan = bridge_complex_field(
            field, CHROMATIX_CAPABILITIES, policy=BridgePolicy.ALLOW_DOWNCAST
        )
        assert bridged.dtype is DType.COMPLEX64
        assert bridged.namespace is ArrayNamespace.JAX
        assert plan.lossy
        assert bridged.provenance["bridge_plan"]["lossy"] is True
        assert bridged.provenance["bridge_plan"]["source"]["dtype"] == "complex128"

    def test_a_ray_bundle_has_no_home_in_chromatix_at_all(self):
        # Not a precision question: Chromatix accepts no real dtype, so real ray
        # geometry cannot enter it under any policy. The failure names the kind
        # mismatch rather than proposing a downcast.
        with pytest.raises(BridgeError) as excinfo:
            bridge_ray_bundle(
                _bundle(Precision.FP64),
                CHROMATIX_CAPABILITIES,
                policy=BridgePolicy.ALLOW_DOWNCAST,
            )
        assert excinfo.value.code == "NO_COMPATIBLE_DTYPE_KIND"

    def test_bridging_a_bundle_keeps_geometry_and_amplitude_at_one_precision(self):
        # An FP32-only destination: the geometry goes float64 -> float32 and the
        # amplitude must follow into complex64, not stay at complex128. An
        # artifact at two precisions at once is what this prevents.
        bridged, plan = bridge_ray_bundle(
            _bundle(Precision.FP64), _FP32_ONLY, policy=BridgePolicy.ALLOW_DOWNCAST
        )
        assert plan.lossy and plan.downcast
        assert bridged.dtype is DType.FLOAT32
        assert array_state(bridged.amplitude).dtype is DType.COMPLEX64
        assert bridged.provenance["bridge_plan"]["lossy"] is True

    def test_a_downcast_bundle_still_reconstructs_within_the_fp32_bound(self):
        bridged, _ = bridge_ray_bundle(
            _bundle(Precision.FP64), _FP32_ONLY, policy=BridgePolicy.ALLOW_DOWNCAST
        )
        field, _ = ray_to_wave(bridged, grid_shape=GRID, sample_pitch_m=PITCH)
        assert field.dtype is DType.COMPLEX64
        assert _relative_peak_error(field.u, _analytic_field()) < FP32_MAX_REL_ERROR


class TestJaxSixtyFourBitIsNotSilentlyDropped:
    def test_requesting_float64_from_jax_without_x64_is_a_named_failure(self):
        # The hazard: jnp.asarray(x, dtype=float64) returns float32 with no
        # warning when jax_enable_x64 is off, which is the state the Chromatix
        # adapter enforces on every call. Silently accepting that would report
        # FP64 for an FP32 computation.
        import jax

        if bool(jax.config.read("jax_enable_x64")):
            pytest.skip("this session has x64 enabled, so the downcast cannot occur")
        with pytest.raises(CapabilityError) as excinfo:
            collimated_bundle(
                positions_xy_m=_launch_positions(),
                direction=_DIRECTION,
                wavelength_m=WAVELENGTH_M,
                precision=Precision.FP64,
                namespace=ArrayNamespace.JAX,
            )
        assert excinfo.value.code == "SILENT_DTYPE_DOWNCAST"
        assert "jax_enable_x64" in str(excinfo.value)


# ---------------------------------------------------------------------------
# GPU residency -- dedicated session only
# ---------------------------------------------------------------------------


@pytest.mark.gpu
class TestGpuResidency:
    """GPU execution must be proved by a kernel, not by device enumeration.

    PB4a measured an image where ``jax.devices()`` reported ``CudaDevice(id=0)``
    while the first jitted call died with "No PTX compilation provider is
    available". Every assertion below therefore reads the device off an array
    that a real computation produced.
    """

    def _cuda_bundle(self, precision: Precision) -> RayBundle:
        import jax

        bundle = _bundle(precision, ArrayNamespace.JAX)
        gpu = next(device for device in jax.devices() if device.platform == "gpu")
        return RayBundle(
            positions_m=jax.device_put(bundle.positions_m, gpu),
            directions=jax.device_put(bundle.directions, gpu),
            wavelength_m=bundle.wavelength_m,
            reference_plane=bundle.reference_plane,
            frame=bundle.frame,
            amplitude=jax.device_put(bundle.amplitude, gpu),
            optical_path_length_m=jax.device_put(bundle.optical_path_length_m, gpu),
            optical_path_length_reference=bundle.optical_path_length_reference,
        )

    def test_ray_to_wave_output_lands_on_the_gpu(self, record_property):
        bundle = self._cuda_bundle(Precision.FP32)
        assert bundle.device.kind is DeviceKind.CUDA
        field, _ = ray_to_wave(bundle, grid_shape=GRID, sample_pitch_m=PITCH)
        assert field.device.kind is DeviceKind.CUDA
        assert field.dtype is DType.COMPLEX64
        record_property("ray_to_wave_output_device", str(field.device))

    def test_no_host_round_trip_across_the_coupler(self):
        bundle = self._cuda_bundle(Precision.FP32)
        field, _ = ray_to_wave(bundle, grid_shape=GRID, sample_pitch_m=PITCH)
        # Same device ordinal in and out: the reconstruction never touched the
        # host, which is what "preserve GPU residency" has to mean.
        assert field.device == bundle.device
        assert field.provenance["execution"]["output"]["device"] == str(bundle.device)

    def test_wave_to_ray_round_trip_stays_on_the_gpu(self):
        import jax

        gpu = next(device for device in jax.devices() if device.platform == "gpu")
        host = _band_limited_field(DType.COMPLEX64, ArrayNamespace.JAX)
        field = ComplexField(
            u=jax.device_put(host.u, gpu),
            sample_pitch_m=PITCH,
            wavelength_m=WAVELENGTH_M,
            reference_plane=PLANE,
            frame=Frame(),
        )
        assert field.device.kind is DeviceKind.CUDA
        bundle, spectrum, _ = wave_to_ray(field)
        assert bundle.device.kind is DeviceKind.CUDA
        assert spectrum.namespace is ArrayNamespace.JAX
        restored, _ = ray_to_wave(bundle, grid_shape=GRID, sample_pitch_m=PITCH)
        assert restored.device.kind is DeviceKind.CUDA
        assert _relative_peak_error(restored.u, np.asarray(host.u, np.complex128)) < (
            FP32_MAX_REL_ERROR
        )

    def test_torch_cuda_rays_bridge_into_jax_without_leaving_the_device(self):
        torch = pytest.importorskip("torch")
        if not torch.cuda.is_available():  # pragma: no cover - guarded by the gpu marker
            pytest.skip("no CUDA torch build")
        from multiscale_optics_agent.core.arrays import to_namespace

        tensor = torch.arange(6, dtype=torch.float32, device="cuda").reshape(2, 3)
        converted = to_namespace(tensor, namespace=ArrayNamespace.JAX)
        state = array_state(converted)
        assert state.namespace is ArrayNamespace.JAX
        assert state.device.kind is DeviceKind.CUDA
        assert state.dtype is DType.FLOAT32
        np.testing.assert_allclose(np.asarray(converted), tensor.cpu().numpy())

    def test_gpu_serialization_records_a_host_copy(self, tmp_path):
        bundle = self._cuda_bundle(Precision.FP32)
        field, _ = ray_to_wave(bundle, grid_shape=GRID, sample_pitch_m=PITCH)
        record = field.to_artifact_record(artifact_id="gpu", uri=tmp_path / "gpu.npy")
        serialization = record.metadata["serialization"]
        assert serialization["host_copy"] is True
        assert serialization["kind"] == "serialization"
        # The record still tells the truth about where the computation happened.
        assert record.metadata["execution"]["device"].startswith("cuda")
        assert record.device.value == "gpu"
        # And the live artifact was not dragged to the host by persisting it.
        assert field.device.kind is DeviceKind.CUDA

    def test_a_gpu_source_cannot_silently_enter_a_cpu_only_target(self):
        host_only = ComponentCapabilities(
            component="T_HOST_ONLY",
            devices=frozenset({DeviceKind.CPU}),
            precisions=frozenset({Precision.FP32}),
            accepted_input_dtypes=frozenset({DType.COMPLEX64}),
            native_compute_dtypes=frozenset({DType.COMPLEX64}),
            output_dtypes=frozenset({DType.COMPLEX64}),
            namespaces=frozenset({ArrayNamespace.NUMPY}),
            evidence="fixture",
        )
        bundle = self._cuda_bundle(Precision.FP32)
        field, _ = ray_to_wave(bundle, grid_shape=GRID, sample_pitch_m=PITCH)
        with pytest.raises(BridgeError) as excinfo:
            plan_bridge(field.state, host_only)
        assert excinfo.value.code == "DEVICE_INCOMPATIBLE"

        plan = plan_bridge(field.state, host_only, allow_device_transfer=True)
        assert plan.device_transfer and plan.host_transfer
        assert plan.target_device == DevicePlacement(DeviceKind.CPU)
