"""Precision/dtype/device vocabulary, capability negotiation and bridge planning (CHE-61).

Deliberately free of optical physics and of every external solver: bridge
resolution is a decision procedure over a capability table, and it is testable
-- and must be tested -- without running a propagation. If these tests needed
Chromatix to pass, a policy bug and a physics bug would be indistinguishable.

The seven cases PB4b section 16 requires are grouped under
``TestRequiredBridgeCases`` and named after the section, so a reviewer can map
them one for one.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.arrays import (
    array_state,
    device_of,
    dtype_of,
    namespace_of,
    to_host_numpy,
    xp_for,
)
from core.capabilities import (
    C_RAY_TO_WAVE_CAPABILITIES,
    CHROMATIX_CAPABILITIES,
    COMPONENT_CAPABILITIES,
    OPTILAND_CAPABILITIES,
    capabilities_for,
    capability_matrix,
)
from core.errors import UnsupportedCapabilityError
from core.precision import (
    ArrayNamespace,
    ArrayState,
    BridgeError,
    BridgePolicy,
    CapabilityError,
    ComponentCapabilities,
    DeviceKind,
    DevicePlacement,
    DType,
    ExecutionRequest,
    Precision,
    plan_bridge,
)
from core.specs import Device

CPU = DevicePlacement(DeviceKind.CPU)
CUDA0 = DevicePlacement(DeviceKind.CUDA, 0)


def _state(dtype: DType, device: DevicePlacement = CPU, namespace=ArrayNamespace.NUMPY):
    return ArrayState(dtype, device, namespace)


#: A synthetic target used by the section-16 cases. Real components are tested
#: separately; a policy test that depends on a real component's dtype set fails
#: for two different reasons at once.
def _target(
    accepted: set[DType],
    *,
    devices: set[DeviceKind] | None = None,
    namespaces: set[ArrayNamespace] | None = None,
) -> ComponentCapabilities:
    accepted_frozen = frozenset(accepted)
    return ComponentCapabilities(
        component="T_SYNTHETIC",
        devices=frozenset(devices or {DeviceKind.CPU}),
        precisions=frozenset({d.precision for d in accepted_frozen}),
        accepted_input_dtypes=accepted_frozen,
        native_compute_dtypes=accepted_frozen,
        output_dtypes=accepted_frozen,
        namespaces=frozenset(namespaces or {ArrayNamespace.NUMPY, ArrayNamespace.JAX}),
        evidence="synthetic fixture for policy tests",
    )


# ---------------------------------------------------------------------------
# Step 1 -- vocabulary
# ---------------------------------------------------------------------------


class TestPrecisionIsNotDtype:
    def test_one_precision_family_covers_a_real_and_a_complex_dtype(self):
        assert Precision.FP32.real_dtype is DType.FLOAT32
        assert Precision.FP32.complex_dtype is DType.COMPLEX64
        assert Precision.FP64.real_dtype is DType.FLOAT64
        assert Precision.FP64.complex_dtype is DType.COMPLEX128

    def test_fp16_has_no_complex_dtype_rather_than_an_invented_one(self):
        # Symmetry is not a reason to declare a representation no framework in
        # this stack provides.
        assert Precision.FP16.real_dtype is DType.FLOAT16
        assert Precision.FP16.complex_dtype is None

    def test_complex64_is_fp32_because_accuracy_follows_the_component_not_the_word(self):
        assert DType.COMPLEX64.precision is Precision.FP32
        assert DType.COMPLEX64.component_bits == 32
        assert DType.COMPLEX128.precision is Precision.FP64

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("fp32", Precision.FP32),
            ("FP64", Precision.FP64),
            ("float64", Precision.FP64),  # the legacy config spelling
            ("complex64", Precision.FP32),
            ("double", Precision.FP64),
        ],
    )
    def test_parse_accepts_precision_and_dtype_spellings(self, text, expected):
        assert Precision.parse(text) is expected

    def test_parse_rejects_an_unknown_precision_structurally(self):
        with pytest.raises(CapabilityError) as excinfo:
            Precision.parse("bfloat16")
        assert excinfo.value.code == "UNKNOWN_PRECISION"

    def test_unsupported_dtype_is_refused_rather_than_coerced(self):
        with pytest.raises(CapabilityError) as excinfo:
            DType.parse("int32")
        assert excinfo.value.code == "UNSUPPORTED_DTYPE"


class TestDeviceIsSeparate:
    def test_ordinal_survives_parsing_and_printing(self):
        device = DevicePlacement.parse("cuda:1")
        assert (device.kind, device.index) == (DeviceKind.CUDA, 1)
        assert str(device) == "cuda:1"

    def test_gpu_and_cuda_are_the_same_device_kind(self):
        assert DevicePlacement.parse("gpu").kind is DevicePlacement.parse("cuda").kind

    def test_projects_onto_the_coarse_registry_enum(self):
        assert CUDA0.to_spec_device() is Device.GPU
        assert CPU.to_spec_device() is Device.CPU

    def test_unknown_device_is_refused(self):
        with pytest.raises(CapabilityError) as excinfo:
            DevicePlacement.parse("tpu")
        assert excinfo.value.code == "UNSUPPORTED_DEVICE"


class TestNamespaceIsSeparateFromDevice:
    def test_numpy_cannot_leave_the_host_but_jax_and_torch_can(self):
        assert not ArrayNamespace.NUMPY.can_leave_host
        assert ArrayNamespace.JAX.can_leave_host
        assert ArrayNamespace.TORCH.can_leave_host

    def test_numpy_array_state_is_observed_not_declared(self):
        state = array_state(np.zeros((2, 2), dtype=np.complex64))
        assert state == ArrayState(DType.COMPLEX64, CPU, ArrayNamespace.NUMPY)

    def test_dtype_device_namespace_are_read_off_the_buffer(self):
        array = np.ones(3, dtype=np.float32)
        assert dtype_of(array) is DType.FLOAT32
        assert device_of(array) == CPU
        assert namespace_of(array) is ArrayNamespace.NUMPY

    def test_python_lists_default_to_the_historical_host_representation(self):
        assert namespace_of([1.0, 2.0]) is ArrayNamespace.NUMPY

    def test_torch_is_not_a_compute_namespace(self):
        with pytest.raises(CapabilityError) as excinfo:
            xp_for(ArrayNamespace.TORCH)
        assert excinfo.value.code == "NAMESPACE_NOT_A_COMPUTE_NAMESPACE"

    def test_numpy_compute_namespace_is_numpy_itself(self):
        assert xp_for(ArrayNamespace.NUMPY) is np


# ---------------------------------------------------------------------------
# Step 2 -- capability declarations reflect measured package reality
# ---------------------------------------------------------------------------


class TestDeclaredCapabilities:
    def test_optiland_supports_exactly_its_two_native_precisions(self):
        assert OPTILAND_CAPABILITIES.precisions == frozenset({Precision.FP32, Precision.FP64})
        assert Precision.FP16 not in OPTILAND_CAPABILITIES.precisions

    def test_optiland_reaches_cuda_only_through_torch(self):
        assert OPTILAND_CAPABILITIES.namespaces_for(DeviceKind.CUDA) == frozenset(
            {ArrayNamespace.TORCH}
        )
        assert ArrayNamespace.NUMPY in OPTILAND_CAPABILITIES.namespaces_for(DeviceKind.CPU)

    def test_chromatix_declares_complex64_only(self):
        assert CHROMATIX_CAPABILITIES.native_compute_dtypes == frozenset({DType.COMPLEX64})
        assert DType.COMPLEX128 not in CHROMATIX_CAPABILITIES.accepted_input_dtypes
        assert CHROMATIX_CAPABILITIES.precisions == frozenset({Precision.FP32})

    def test_chromatix_records_complex128_as_lossy_rather_than_supported(self):
        # The package swallows it and returns complex64. Naming that as
        # "accepted" would make the loss invisible to the planner.
        assert CHROMATIX_CAPABILITIES.lossy_input_dtypes == frozenset({DType.COMPLEX128})

    def test_every_declaration_carries_its_evidence(self):
        for capability in COMPONENT_CAPABILITIES.values():
            assert capability.evidence, f"{capability.component} declares no evidence"

    def test_the_capability_matrix_covers_every_declared_component(self):
        """Order is the pipeline's, and completeness is derived rather than typed.

        The rows read in execution order -- ray model, ray-to-wave, wave model,
        wave-to-ray -- because that is how a reader traces a field through the
        stack, and the composed batched DOE step comes last because its
        capability is the intersection of two rows above it.

        The completeness half is asserted against `COMPONENT_CAPABILITIES`
        rather than against a second hand-written list. A declared component
        missing from the documented matrix is a component whose support nobody
        can look up, and a hard-coded list would have to be edited to notice.
        """
        rows = capability_matrix()
        assert [row["component"] for row in rows] == [
            "M_RAY_OPTILAND",
            "C_RAY_TO_WAVE",
            "M_WAVE_CHROMATIX",
            "C_WAVE_TO_RAY",
            "C_PLANAR_DOE_STEP",
            "C_PATCH_WFT",
        ]
        assert {row["component"] for row in rows} == set(COMPONENT_CAPABILITIES), (
            "the documented capability matrix and the declarations have diverged; "
            "a declared component absent from the matrix is one whose support "
            "nobody can look up."
        )

    def test_unknown_component_fails_structurally(self):
        with pytest.raises(CapabilityError) as excinfo:
            capabilities_for("M_NOT_A_MODEL")
        assert excinfo.value.code == "UNKNOWN_COMPONENT"


class TestCapabilityNegotiation:
    def test_optiland_accepts_fp32_and_fp64_on_both_devices(self):
        for precision in (Precision.FP32, Precision.FP64):
            for device in ("cpu", "cuda"):
                resolved = OPTILAND_CAPABILITIES.resolve(
                    ExecutionRequest(
                        "M_RAY_OPTILAND",
                        precision=precision,
                        device=DevicePlacement.parse(device),
                    )
                )
                assert resolved.precision is precision
                assert resolved.device.kind is DevicePlacement.parse(device).kind

    def test_optiland_cuda_resolves_to_the_torch_namespace(self):
        resolved = OPTILAND_CAPABILITIES.resolve(
            ExecutionRequest("M_RAY_OPTILAND", precision=Precision.FP32, device=CUDA0)
        )
        assert resolved.namespace is ArrayNamespace.TORCH

    def test_optiland_fp16_is_a_structured_capability_failure(self):
        with pytest.raises(CapabilityError) as excinfo:
            OPTILAND_CAPABILITIES.resolve(
                ExecutionRequest("M_RAY_OPTILAND", precision=Precision.FP16)
            )
        assert excinfo.value.code == "UNSUPPORTED_PRECISION"
        assert "float32" in str(excinfo.value)
        # Adapters must be able to catch this as the project's existing
        # capability error without knowing about the new hierarchy.
        assert isinstance(excinfo.value, UnsupportedCapabilityError)

    def test_chromatix_fp64_is_a_structured_capability_failure(self):
        with pytest.raises(CapabilityError) as excinfo:
            CHROMATIX_CAPABILITIES.resolve(
                ExecutionRequest("M_WAVE_CHROMATIX", precision=Precision.FP64)
            )
        assert excinfo.value.code == "UNSUPPORTED_PRECISION"

    def test_a_precision_one_backend_lacks_is_not_globally_rejected(self):
        # The whole point of per-backend capability: FP64 is refused by
        # Chromatix and accepted by Optiland in the same process.
        with pytest.raises(CapabilityError):
            CHROMATIX_CAPABILITIES.resolve(
                ExecutionRequest("M_WAVE_CHROMATIX", precision=Precision.FP64)
            )
        assert (
            OPTILAND_CAPABILITIES.resolve(
                ExecutionRequest("M_RAY_OPTILAND", precision=Precision.FP64)
            ).precision
            is Precision.FP64
        )

    def test_request_from_legacy_config_keeps_meaning_what_it_meant(self):
        request = ExecutionRequest.from_config(
            "M_RAY_OPTILAND", {"device": "cpu", "dtype": "float64"}
        )
        assert request.precision is Precision.FP64
        assert request.device == CPU
        assert request.bridge_policy is BridgePolicy.SAFE

    def test_default_request_is_the_historical_host_float64_behaviour(self):
        request = ExecutionRequest.from_config("M_RAY_OPTILAND", {})
        assert (request.precision, request.device) == (Precision.FP64, CPU)


class TestComputePrecisionIsDistinctFromIO:
    def test_float16_input_is_computed_in_float32_not_float16(self):
        assert C_RAY_TO_WAVE_CAPABILITIES.compute_dtype_for(DType.FLOAT16) is DType.FLOAT32

    def test_a_dtype_above_the_floor_is_not_dragged_up(self):
        assert C_RAY_TO_WAVE_CAPABILITIES.compute_dtype_for(DType.FLOAT64) is DType.FLOAT64
        assert C_RAY_TO_WAVE_CAPABILITIES.compute_dtype_for(DType.COMPLEX64) is DType.COMPLEX64

    def test_float16_is_not_in_any_component_native_compute_set(self):
        for capability in COMPONENT_CAPABILITIES.values():
            assert DType.FLOAT16 not in capability.native_compute_dtypes


# ---------------------------------------------------------------------------
# Step 4 -- the seven bridge cases PB4b section 16 requires
# ---------------------------------------------------------------------------


class TestRequiredBridgeCases:
    def test_exact_preservation(self):
        plan = plan_bridge(_state(DType.FLOAT32), _target({DType.FLOAT32, DType.FLOAT64}))
        assert plan.target_dtype is DType.FLOAT32
        assert plan.is_identity
        assert not plan.dtype_conversion
        assert plan.effects == ()

    def test_rule_a_does_not_round_trip_a_compatible_dtype(self):
        # float32 -> float64 -> float32 "for convenience" is the failure this
        # asserts against: the widest accepted dtype is not the chosen one.
        plan = plan_bridge(_state(DType.FLOAT32), _target({DType.FLOAT32, DType.FLOAT64}))
        assert plan.target_dtype is not DType.FLOAT64

    def test_safe_widening_picks_the_minimum_necessary_promotion(self):
        plan = plan_bridge(
            _state(DType.FLOAT16),
            _target({DType.FLOAT32, DType.FLOAT64}),
            policy=BridgePolicy.SAFE,
        )
        assert plan.target_dtype is DType.FLOAT32  # rule B: not float64
        assert plan.promotion and not plan.downcast and not plan.lossy
        assert "float16 -> float32" in " ".join(plan.effects)

    def test_unsafe_downcast_is_refused_under_safe(self):
        with pytest.raises(BridgeError) as excinfo:
            plan_bridge(
                _state(DType.FLOAT64), _target({DType.FLOAT32}), policy=BridgePolicy.SAFE
            )
        assert excinfo.value.code == "LOSSY_DOWNCAST_REQUIRED"
        assert "allow_downcast" in str(excinfo.value)

    def test_explicit_lossy_downcast_is_permitted_and_recorded(self):
        plan = plan_bridge(
            _state(DType.FLOAT64),
            _target({DType.FLOAT32}),
            policy=BridgePolicy.ALLOW_DOWNCAST,
        )
        assert plan.target_dtype is DType.FLOAT32
        assert plan.lossy and plan.downcast and not plan.promotion
        assert "LOSSY" in plan.reason

    def test_strict_refuses_any_representation_mismatch(self):
        with pytest.raises(BridgeError) as excinfo:
            plan_bridge(
                _state(DType.FLOAT16),
                _target({DType.FLOAT32, DType.FLOAT64}),
                policy=BridgePolicy.STRICT,
            )
        assert excinfo.value.code == "STRICT_REPRESENTATION_MISMATCH"

    def test_strict_allows_an_already_admissible_representation(self):
        plan = plan_bridge(
            _state(DType.FLOAT32), _target({DType.FLOAT32}), policy=BridgePolicy.STRICT
        )
        assert plan.is_identity

    def test_device_preservation_keeps_a_gpu_source_on_the_gpu(self):
        plan = plan_bridge(
            _state(DType.COMPLEX64, CUDA0, ArrayNamespace.JAX),
            _target(
                {DType.COMPLEX64},
                devices={DeviceKind.CPU, DeviceKind.CUDA},
                namespaces={ArrayNamespace.JAX},
            ),
        )
        assert plan.target_device == CUDA0
        assert not plan.device_transfer and not plan.host_transfer
        assert plan.is_identity

    def test_device_incompatibility_fails_rather_than_falling_back_to_cpu(self):
        with pytest.raises(BridgeError) as excinfo:
            plan_bridge(
                _state(DType.COMPLEX64, CUDA0, ArrayNamespace.JAX),
                _target({DType.COMPLEX64}, devices={DeviceKind.CPU}),
            )
        assert excinfo.value.code == "DEVICE_INCOMPATIBLE"
        assert "no implicit CPU fallback" in str(excinfo.value)

    def test_device_transfer_happens_only_when_explicitly_permitted(self):
        plan = plan_bridge(
            _state(DType.COMPLEX64, CUDA0, ArrayNamespace.JAX),
            _target({DType.COMPLEX64}, devices={DeviceKind.CPU}),
            allow_device_transfer=True,
        )
        assert plan.device_transfer and plan.host_transfer
        assert plan.target_device.kind is DeviceKind.CPU
        assert any("host transfer" in effect for effect in plan.effects)


class TestBridgeNamespaceRules:
    def test_torch_gpu_into_a_jax_gpu_target_stays_on_the_device(self):
        # The Optiland -> C_RAY_TO_WAVE hop. A namespace change, but no host
        # round trip: DLPack keeps the buffer on cuda:0.
        plan = plan_bridge(
            _state(DType.FLOAT32, CUDA0, ArrayNamespace.TORCH),
            C_RAY_TO_WAVE_CAPABILITIES,
        )
        assert plan.namespace_conversion
        assert plan.target_namespace is ArrayNamespace.JAX
        assert plan.target_device == CUDA0
        assert not plan.host_transfer and not plan.device_transfer

    def test_torch_to_numpy_is_reported_as_a_graph_break(self):
        plan = plan_bridge(
            _state(DType.FLOAT64, CPU, ArrayNamespace.TORCH),
            _target({DType.FLOAT64}, namespaces={ArrayNamespace.NUMPY}),
        )
        assert plan.namespace_conversion and plan.graph_break
        assert any("graph break" in effect for effect in plan.effects)

    def test_strict_refuses_a_namespace_conversion(self):
        with pytest.raises(BridgeError) as excinfo:
            plan_bridge(
                _state(DType.FLOAT64, CPU, ArrayNamespace.TORCH),
                _target({DType.FLOAT64}, namespaces={ArrayNamespace.NUMPY}),
                policy=BridgePolicy.STRICT,
            )
        assert excinfo.value.code == "STRICT_NAMESPACE_MISMATCH"

    def test_numpy_source_into_a_cuda_target_is_a_host_transfer(self):
        plan = plan_bridge(
            _state(DType.COMPLEX64, CPU, ArrayNamespace.NUMPY),
            CHROMATIX_CAPABILITIES,
            target_device=CUDA0,
            allow_device_transfer=True,
        )
        assert plan.target_namespace is ArrayNamespace.JAX
        assert plan.host_transfer and plan.device_transfer


class TestBridgeAgainstRealComponents:
    def test_complex128_into_chromatix_is_refused_by_default(self):
        with pytest.raises(BridgeError) as excinfo:
            plan_bridge(
                _state(DType.COMPLEX128, CPU, ArrayNamespace.NUMPY), CHROMATIX_CAPABILITIES
            )
        assert excinfo.value.code == "LOSSY_DOWNCAST_REQUIRED"

    def test_complex128_into_chromatix_under_explicit_downcast_is_recorded_lossy(self):
        plan = plan_bridge(
            _state(DType.COMPLEX128, CPU, ArrayNamespace.NUMPY),
            CHROMATIX_CAPABILITIES,
            policy=BridgePolicy.ALLOW_DOWNCAST,
        )
        assert plan.target_dtype is DType.COMPLEX64
        assert plan.lossy and plan.downcast
        assert "LOSSY" in plan.reason

    def test_complex64_into_chromatix_is_preserved_exactly(self):
        plan = plan_bridge(
            _state(DType.COMPLEX64, CUDA0, ArrayNamespace.JAX), CHROMATIX_CAPABILITIES
        )
        assert plan.is_identity and plan.target_device == CUDA0

    def test_a_real_dtype_has_no_home_in_chromatix(self):
        with pytest.raises(BridgeError) as excinfo:
            plan_bridge(_state(DType.FLOAT64), CHROMATIX_CAPABILITIES)
        assert excinfo.value.code == "NO_COMPATIBLE_DTYPE_KIND"

    def test_plan_serializes_every_question_the_contract_must_answer(self):
        plan = plan_bridge(
            _state(DType.COMPLEX128, CPU, ArrayNamespace.NUMPY),
            CHROMATIX_CAPABILITIES,
            policy=BridgePolicy.ALLOW_DOWNCAST,
            compute_dtype=DType.COMPLEX64,
        )
        payload = plan.as_dict()
        for key in (
            "source",
            "target",
            "policy",
            "compute_dtype",
            "dtype_conversion",
            "promotion",
            "downcast",
            "lossy",
            "device_transfer",
            "host_transfer",
            "namespace_conversion",
            "reason",
        ):
            assert key in payload, key
        assert payload["source"]["dtype"] == "complex128"
        assert payload["target"]["dtype"] == "complex64"


class TestSerializationBoundary:
    def test_host_copy_is_a_named_operation_not_an_incidental_astype(self):
        array = np.ones((2, 2), dtype=np.complex64)
        host = to_host_numpy(array, reason="ArtifactRecord persistence")
        assert isinstance(host, np.ndarray)
        assert host.dtype == np.complex64  # serialization does not change precision
