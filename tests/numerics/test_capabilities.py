"""The capability table is measured, cited, and cannot be widened past what it states.

CHE-173 (R02.1), acceptance criteria 2 and 3. The risk this ticket names is
re-deriving the table from the packages' documentation instead of from the
probes: a row copied from an API signature is worse than no row, because it will
be trusted. Nothing can check that a probe was *run*; what can be checked is that
every row names one, that the row is internally consistent with what it claims,
and that the specific measured facts the ticket lists are what the table says.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

from numerics.precision import (
    CHROMATIX_CAPABILITIES,
    COMPONENT_CAPABILITIES,
    OPTILAND_CAPABILITIES,
    PHASE_ACCUMULATION_FLOOR,
    ArrayNamespace,
    ArrayState,
    ComponentCapabilities,
    DeviceKind,
    DevicePlacement,
    DType,
    Precision,
    capabilities_for,
    capability_rows,
    compute_dtype,
)


def _row_of(component: str) -> ComponentCapabilities:
    return COMPONENT_CAPABILITIES[component]


# --- criterion 2: every entry cites the probe that measured it -------------


@pytest.mark.parametrize("component", sorted(COMPONENT_CAPABILITIES))
def test_every_entry_cites_a_precision_probe(component: str) -> None:
    capability = _row_of(component)
    assert capability.probe.startswith("benchmarks/probes/precision/"), (
        f"{component} cites {capability.probe!r}, which is not a precision probe"
    )
    assert capability.probe.endswith(".py")


@pytest.mark.parametrize("component", sorted(COMPONENT_CAPABILITIES))
def test_every_entry_states_a_measurement_and_where_it_ran(component: str) -> None:
    """Evidence must say what was observed, not that support exists.

    The weakest useful check that distinguishes a measurement from a claim: the
    sentence has to name the pinned version it was measured against and the
    frozen tag the probe is reproducible from.
    """
    evidence = _row_of(component).evidence
    assert len(evidence) > 80, f"{component} evidence is too short to be a measurement"
    assert "pre-rewrite-2026-08-30" in evidence, (
        f"{component} evidence does not say which frozen revision the probe is at"
    )
    assert re.search(r"\d+\.\d+\.\d+", evidence), (
        f"{component} evidence names no pinned version"
    )


#: The tag the probe paths resolve against. `benchmarks/` was deleted from the
#: working tree by the greenfield rewrite, so the citation is only as good as the
#: revision it names -- which is why the test below resolves it rather than
#: trusting the string to look like a path.
PROBE_TAG = "pre-rewrite-2026-08-30"
ROOT = Path(__file__).resolve().parents[2]


def _tag_exists() -> bool:
    return (
        subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "--verify", f"{PROBE_TAG}^{{commit}}"],
            capture_output=True,
        ).returncode
        == 0
    )


@pytest.mark.parametrize("component", sorted(COMPONENT_CAPABILITIES))
def test_every_cited_probe_resolves_at_the_frozen_tag(component: str) -> None:
    """The citation is a reference, not a sentence that looks like one.

    A probe path that no longer resolves is the same defect as a capability with
    no probe: nothing can be re-run to confirm or falsify the row. Checked with
    `git cat-file` because the file is deliberately not in the working tree.
    """
    if not _tag_exists():
        pytest.skip(f"{PROBE_TAG} is not present in this checkout")
    probe = _row_of(component).probe
    resolved = subprocess.run(
        ["git", "-C", str(ROOT), "cat-file", "-e", f"{PROBE_TAG}:{probe}"],
        capture_output=True,
    )
    assert resolved.returncode == 0, (
        f"{component} cites {probe}, which does not exist at {PROBE_TAG}. Re-run the probe "
        "and cite where it lives now, or drop the row."
    )


def test_a_capability_without_a_probe_is_refused() -> None:
    """The detection half of criterion 2."""
    with pytest.raises(ValueError) as caught:
        ComponentCapabilities(
            component="M_MADE_UP",
            devices=frozenset({DeviceKind.CPU}),
            precisions=frozenset({Precision.FP32}),
            accepted_input_dtypes=frozenset({DType.FLOAT32}),
            native_compute_dtypes=frozenset({DType.FLOAT32}),
            output_dtypes=frozenset({DType.FLOAT32}),
            device_namespaces={DeviceKind.CPU: frozenset({ArrayNamespace.NUMPY})},
            probe="",
            evidence="the docs say it supports float32",
        )
    assert caught.value.code == "INVALID_CAPABILITY_DECLARATION"
    assert "benchmarks/probes/" in str(caught.value)


def test_a_capability_with_no_evidence_is_refused() -> None:
    with pytest.raises(ValueError, match="evidence"):
        ComponentCapabilities(
            component="M_MADE_UP",
            devices=frozenset({DeviceKind.CPU}),
            precisions=frozenset({Precision.FP32}),
            accepted_input_dtypes=frozenset({DType.FLOAT32}),
            native_compute_dtypes=frozenset({DType.FLOAT32}),
            output_dtypes=frozenset({DType.FLOAT32}),
            device_namespaces={DeviceKind.CPU: frozenset({ArrayNamespace.NUMPY})},
            probe="benchmarks/probes/precision/made_up.py",
            evidence="   ",
        )


def test_a_declared_device_with_no_way_to_reach_it_is_refused() -> None:
    """Widening by device: claiming CUDA without saying which namespace drives it.

    This is exactly how the Optiland row could go wrong -- `set_device` raises on
    the numpy backend, so CUDA is a torch-only capability, and a row that
    declares CUDA without pinning it to a namespace has widened past the probe.
    """
    with pytest.raises(ValueError, match="device_namespaces"):
        ComponentCapabilities(
            component="M_MADE_UP",
            devices=frozenset({DeviceKind.CPU, DeviceKind.CUDA}),
            precisions=frozenset({Precision.FP32}),
            accepted_input_dtypes=frozenset({DType.FLOAT32}),
            native_compute_dtypes=frozenset({DType.FLOAT32}),
            output_dtypes=frozenset({DType.FLOAT32}),
            device_namespaces={DeviceKind.CPU: frozenset({ArrayNamespace.NUMPY})},
            probe="benchmarks/probes/precision/made_up.py",
            evidence="measured on the host only, which is why CUDA is a widening",
        )


def test_computing_in_a_dtype_the_component_refuses_is_refused() -> None:
    """Widening by dtype: a native compute dtype outside the accepted set."""
    with pytest.raises(ValueError, match="native_compute_dtypes"):
        ComponentCapabilities(
            component="M_MADE_UP",
            devices=frozenset({DeviceKind.CPU}),
            precisions=frozenset({Precision.FP32, Precision.FP64}),
            accepted_input_dtypes=frozenset({DType.FLOAT32}),
            native_compute_dtypes=frozenset({DType.FLOAT32, DType.FLOAT64}),
            output_dtypes=frozenset({DType.FLOAT32}),
            device_namespaces={DeviceKind.CPU: frozenset({ArrayNamespace.NUMPY})},
            probe="benchmarks/probes/precision/made_up.py",
            evidence="measured at float32 only, so float64 compute is a widening",
        )


def test_a_precision_with_nothing_to_execute_it_in_is_refused() -> None:
    """Widening by precision: the Chromatix FP64 mistake, made in advance."""
    with pytest.raises(ValueError, match="no native compute dtype"):
        ComponentCapabilities(
            component="M_MADE_UP",
            devices=frozenset({DeviceKind.CPU}),
            precisions=frozenset({Precision.FP32, Precision.FP64}),
            accepted_input_dtypes=frozenset({DType.COMPLEX64}),
            native_compute_dtypes=frozenset({DType.COMPLEX64}),
            output_dtypes=frozenset({DType.COMPLEX64}),
            device_namespaces={DeviceKind.CPU: frozenset({ArrayNamespace.JAX})},
            probe="benchmarks/probes/precision/made_up.py",
            evidence="complex64 only, so an FP64 row has nothing to run",
        )


def test_a_lossy_dtype_cannot_also_be_accepted() -> None:
    with pytest.raises(ValueError, match="lossy_input_dtypes"):
        ComponentCapabilities(
            component="M_MADE_UP",
            devices=frozenset({DeviceKind.CPU}),
            precisions=frozenset({Precision.FP32}),
            accepted_input_dtypes=frozenset({DType.COMPLEX64, DType.COMPLEX128}),
            native_compute_dtypes=frozenset({DType.COMPLEX64}),
            output_dtypes=frozenset({DType.COMPLEX64}),
            device_namespaces={DeviceKind.CPU: frozenset({ArrayNamespace.JAX})},
            lossy_input_dtypes=frozenset({DType.COMPLEX128}),
            probe="benchmarks/probes/precision/made_up.py",
            evidence="complex128 is truncated on intake, so it is not admissible",
        )


def test_the_namespace_set_is_derived_and_cannot_disagree_with_the_devices() -> None:
    """One place for the namespace set, so there is one place to widen it."""
    assert OPTILAND_CAPABILITIES.namespaces == frozenset(
        {ArrayNamespace.NUMPY, ArrayNamespace.TORCH}
    )
    assert CHROMATIX_CAPABILITIES.namespaces == frozenset({ArrayNamespace.JAX})


# --- the measured facts the ticket lists as reuse-without-re-derivation ----


def test_optiland_has_no_float16_path() -> None:
    """`set_precision` is `Literal['float32','float64']` and raises otherwise."""
    assert OPTILAND_CAPABILITIES.precisions == frozenset({Precision.FP32, Precision.FP64})
    assert DType.FLOAT16 not in OPTILAND_CAPABILITIES.accepted_input_dtypes
    assert DType.FLOAT16 not in OPTILAND_CAPABILITIES.native_compute_dtypes


def test_optiland_reaches_cuda_only_through_torch() -> None:
    """`set_device` raises `BackendCapabilityError` on the numpy backend."""
    assert OPTILAND_CAPABILITIES.namespaces_for(DeviceKind.CUDA) == frozenset(
        {ArrayNamespace.TORCH}
    )
    assert ArrayNamespace.NUMPY in OPTILAND_CAPABILITIES.namespaces_for(DeviceKind.CPU)


def test_chromatix_has_no_complex128_path_at_any_device() -> None:
    """`ScalarField.__init__` is `jnp.asarray(u, dtype=jnp.complex64)`, unconditionally."""
    assert CHROMATIX_CAPABILITIES.precisions == frozenset({Precision.FP32})
    for device in CHROMATIX_CAPABILITIES.devices:
        assert CHROMATIX_CAPABILITIES.namespaces_for(device) == frozenset({ArrayNamespace.JAX})
    assert CHROMATIX_CAPABILITIES.output_dtypes == frozenset({DType.COMPLEX64})
    assert DType.COMPLEX128 not in CHROMATIX_CAPABILITIES.accepted_input_dtypes


def test_chromatix_complex128_is_declared_lossy_rather_than_accepted() -> None:
    """The distinction that makes the truncation happen where something records it."""
    assert CHROMATIX_CAPABILITIES.lossy_input_dtypes == frozenset({DType.COMPLEX128})
    state = ArrayState(DType.COMPLEX128, DevicePlacement(DeviceKind.CPU), ArrayNamespace.JAX)
    assert not CHROMATIX_CAPABILITIES.accepts(state)


def test_the_table_declares_no_row_for_code_that_does_not_exist() -> None:
    """R02.1 has no coupler; a coupler row here would be a claim about nothing.

    The reference implementation's table had seven rows. Five described coupler
    and operator implementations, whose capability is set by what their shared
    implementation is written against -- so they belong to the tickets that write
    them, with their own evidence.
    """
    assert set(COMPONENT_CAPABILITIES) == {"M_RAY_OPTILAND", "M_WAVE_CHROMATIX"}


def test_an_undeclared_component_is_refused_by_name() -> None:
    with pytest.raises(ValueError) as caught:
        capabilities_for("C_RAY_TO_WAVE")
    assert caught.value.code == "UNKNOWN_COMPONENT"
    assert "M_RAY_OPTILAND" in str(caught.value)


def test_the_matrix_is_generated_from_the_declarations() -> None:
    rows = capability_rows()
    assert [row["component"] for row in rows] == sorted(COMPONENT_CAPABILITIES)
    for row in rows:
        assert row["probe"].startswith("benchmarks/probes/precision/")
        assert row["evidence"]


# --- criterion 3: float32 is the floor for phase accumulation --------------


def test_float32_is_the_minimum_compute_precision_for_phase_accumulation() -> None:
    """`k * OPL` gets float32 headroom whatever arrives."""
    assert PHASE_ACCUMULATION_FLOOR is Precision.FP32
    assert compute_dtype(DType.FLOAT16) is DType.FLOAT32


def test_the_floor_promotes_without_changing_kind_and_never_narrows() -> None:
    assert compute_dtype(DType.FLOAT32) is DType.FLOAT32
    assert compute_dtype(DType.FLOAT64) is DType.FLOAT64
    assert compute_dtype(DType.COMPLEX64) is DType.COMPLEX64
    assert compute_dtype(DType.COMPLEX128) is DType.COMPLEX128


def test_no_declared_component_computes_below_the_phase_floor() -> None:
    for component, capability in COMPONENT_CAPABILITIES.items():
        assert capability.minimum_compute_precision.bits >= PHASE_ACCUMULATION_FLOOR.bits, (
            f"{component} declares a compute floor below float32"
        )
        for dtype in capability.accepted_input_dtypes:
            assert capability.compute_dtype_for(dtype).component_bits >= 32


def test_a_promotion_is_not_native_support() -> None:
    """float16 in -> float32 compute is a promotion, and must read as one.

    The whole point of separating `accepted_input_dtypes` from
    `native_compute_dtypes`: `compute_dtype_for` answering float32 for a float16
    input is not a statement that float16 is supported.
    """
    assert compute_dtype(DType.FLOAT16) not in (DType.FLOAT16,)
    assert DType.FLOAT16 not in OPTILAND_CAPABILITIES.native_compute_dtypes
