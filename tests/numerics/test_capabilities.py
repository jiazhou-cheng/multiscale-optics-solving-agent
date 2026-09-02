"""The capability *contract*: what a declaration may claim, on synthetic rows.

CHE-173 (R02.1), acceptance criteria 2 and 3, narrowed by CHE-223 (R03.6). The
risk R02.1 named is re-deriving a capability from the packages' documentation
instead of from the probes: a row copied from an API signature is worse than no
row, because it will be trusted. Nothing can check that a probe was *run*; what
can be checked is that every declaration names one and is internally consistent
with what it claims.

What CHE-223 took out of this file
----------------------------------
Every assertion **about the two measured records** -- the probe citations, the
`git cat-file` resolution, the four pinned measured facts, two-rows-not-seven --
moved to `tests/knowledge/test_capability_pack.py`, because the rows are now data
under `knowledge/capabilities/` and `numerics/` knows no backend's name. That file
accounts for the move item by item.

What is here is the contract, and every row below is **synthetic**: `M_MADE_UP`,
built to be refused. That is the right shape for these tests and always was -- a
refusal test that widened a *real* row would be asserting something about
Optiland rather than about the rule. `tests/knowledge/test_capability_loader.py`
proves the same refusals fire on a record read from disk, which is what makes the
loader a construction path and not a bypass.

**Ten rule groups, each with a named test below**, and the count is stated because
CHE-206 carried a wrong one (five) for a while:

 1. `probe` cites a path under `benchmarks/probes/`
 2. `evidence` is non-empty
 3. `probe_tag` is non-empty (added by CHE-223)
 4. five required sets are non-empty -- one loop there, one parametrized test here
 5. `device_namespaces` keys are exactly `devices`, and no namespace set is empty
 6. CUDA needs a namespace that can leave the host
 7. `native_compute_dtypes` is within `accepted_input_dtypes`
 8. `lossy_input_dtypes` is disjoint from `accepted_input_dtypes`
 9. every declared precision has a native compute dtype
10. no declared precision is below the declared minimum

They are collected and raised together as `INVALID_CAPABILITY_DECLARATION`, so a
row with three problems takes one edit to find out about rather than three.
"""

from __future__ import annotations

import pytest

from numerics.precision import (
    PHASE_ACCUMULATION_FLOOR,
    ArrayNamespace,
    ComponentCapabilities,
    DeviceKind,
    DType,
    Precision,
    compute_dtype,
)


#: A minimal declaration that is *accepted*, so every refusal below is known to be
#: caused by the one field it changes rather than by the row being malformed in
#: some other way. This is the positive control the file previously lacked: six of
#: the seven refusal tests wrote out a full row by hand, and a typo in one would
#: have passed for the refusal it was testing.
def a_capability(**overrides: object) -> ComponentCapabilities:
    fields: dict[str, object] = {
        "component": "M_MADE_UP",
        "devices": frozenset({DeviceKind.CPU}),
        "precisions": frozenset({Precision.FP32}),
        "accepted_input_dtypes": frozenset({DType.FLOAT32}),
        "native_compute_dtypes": frozenset({DType.FLOAT32}),
        "output_dtypes": frozenset({DType.FLOAT32}),
        "device_namespaces": {DeviceKind.CPU: frozenset({ArrayNamespace.NUMPY})},
        "probe": "benchmarks/probes/precision/made_up.py",
        "probe_tag": "a-frozen-tag",
        "evidence": "measured on the host at float32; a synthetic row for the rule tests",
    }
    fields.update(overrides)
    return ComponentCapabilities(**fields)  # type: ignore[arg-type]


def test_the_synthetic_row_is_accepted() -> None:
    """The positive control. Without it every refusal below could be a typo."""
    capability = a_capability()
    assert capability.component == "M_MADE_UP"
    assert capability.namespaces == frozenset({ArrayNamespace.NUMPY})


# --- criterion 2: a declaration must cite what measured it -----------------


def test_a_capability_without_a_probe_is_refused() -> None:
    """The detection half of criterion 2."""
    with pytest.raises(ValueError) as caught:
        a_capability(probe="", evidence="the docs say it supports float32")
    assert caught.value.code == "INVALID_CAPABILITY_DECLARATION"
    assert "benchmarks/probes/" in str(caught.value)


def test_a_capability_with_no_evidence_is_refused() -> None:
    with pytest.raises(ValueError, match="evidence"):
        a_capability(evidence="   ")


def test_a_capability_with_no_probe_tag_is_refused() -> None:
    """CHE-223's tenth rule, and the reason the field exists.

    A probe path with no revision to resolve it against is not a citation: the
    file was deleted from the working tree by the greenfield rewrite, so the path
    alone names nothing. The tag used to be a module constant interpolated into
    the evidence prose; it is now a field of the measurement, and this is what
    keeps it from being blank.
    """
    with pytest.raises(ValueError, match="probe_tag"):
        a_capability(probe_tag="  ")


# --- the widening refusals -------------------------------------------------


@pytest.mark.parametrize(
    "field",
    [
        "devices",
        "precisions",
        "accepted_input_dtypes",
        "native_compute_dtypes",
        "output_dtypes",
    ],
)
def test_an_empty_required_set_is_refused(field: str) -> None:
    """Five sets, one loop in `__post_init__`, five cases here.

    An empty set is not a narrower capability, it is an unusable one: nothing can
    enter, nothing can execute, or nothing can come out. Previously untested as a
    group -- the file asserted the other rules and left this loop to inspection.
    """
    with pytest.raises(ValueError, match=field):
        a_capability(**{field: frozenset()})


def test_a_declared_device_with_no_way_to_reach_it_is_refused() -> None:
    """Widening by device: claiming CUDA without saying which namespace drives it.

    This is exactly how the Optiland record could go wrong -- `set_device` raises
    on the numpy backend, so CUDA is a torch-only capability, and a declaration
    that names CUDA without pinning it to a namespace has widened past the probe.
    """
    with pytest.raises(ValueError, match="device_namespaces"):
        a_capability(
            devices=frozenset({DeviceKind.CPU, DeviceKind.CUDA}),
            evidence="measured on the host only, which is why CUDA is a widening",
        )


def test_a_device_with_an_empty_namespace_set_is_refused() -> None:
    """The other half of the same rule: the key is present and says nothing."""
    with pytest.raises(ValueError, match=r"device_namespaces\[cpu\]"):
        a_capability(device_namespaces={DeviceKind.CPU: frozenset()})


def test_cuda_driven_only_by_a_host_only_namespace_is_refused() -> None:
    """NumPy cannot hold device memory, so declaring it as CUDA's driver is a fiction.

    Refused here rather than handled downstream, which is what lets
    `_negotiate_namespace` have no unreachable branch.
    """
    with pytest.raises(ValueError, match="cannot hold device memory"):
        a_capability(
            devices=frozenset({DeviceKind.CPU, DeviceKind.CUDA}),
            device_namespaces={
                DeviceKind.CPU: frozenset({ArrayNamespace.NUMPY}),
                DeviceKind.CUDA: frozenset({ArrayNamespace.NUMPY}),
            },
            evidence="a numpy-only row cannot reach a device; declaring CUDA is a fiction",
        )


def test_computing_in_a_dtype_the_component_refuses_is_refused() -> None:
    """Widening by dtype: a native compute dtype outside the accepted set."""
    with pytest.raises(ValueError, match="native_compute_dtypes"):
        a_capability(
            precisions=frozenset({Precision.FP32, Precision.FP64}),
            native_compute_dtypes=frozenset({DType.FLOAT32, DType.FLOAT64}),
            evidence="measured at float32 only, so float64 compute is a widening",
        )


def test_a_precision_with_nothing_to_execute_it_in_is_refused() -> None:
    """Widening by precision: the Chromatix FP64 mistake, made in advance."""
    with pytest.raises(ValueError, match="no native compute dtype"):
        a_capability(
            precisions=frozenset({Precision.FP32, Precision.FP64}),
            accepted_input_dtypes=frozenset({DType.COMPLEX64}),
            native_compute_dtypes=frozenset({DType.COMPLEX64}),
            output_dtypes=frozenset({DType.COMPLEX64}),
            device_namespaces={DeviceKind.CPU: frozenset({ArrayNamespace.JAX})},
            evidence="complex64 only, so an FP64 row has nothing to run",
        )


def test_a_lossy_dtype_cannot_also_be_accepted() -> None:
    with pytest.raises(ValueError, match="lossy_input_dtypes"):
        a_capability(
            accepted_input_dtypes=frozenset({DType.COMPLEX64, DType.COMPLEX128}),
            native_compute_dtypes=frozenset({DType.COMPLEX64}),
            output_dtypes=frozenset({DType.COMPLEX64}),
            device_namespaces={DeviceKind.CPU: frozenset({ArrayNamespace.JAX})},
            lossy_input_dtypes=frozenset({DType.COMPLEX128}),
            evidence="complex128 is truncated on intake, so it is not admissible",
        )


def test_a_precision_below_the_declared_floor_is_refused() -> None:
    """The floor is a claim about accumulation, so a row may not undercut its own."""
    with pytest.raises(ValueError, match="below the declared minimum"):
        a_capability(
            precisions=frozenset({Precision.FP16, Precision.FP32}),
            accepted_input_dtypes=frozenset({DType.FLOAT16, DType.FLOAT32}),
            native_compute_dtypes=frozenset({DType.FLOAT16, DType.FLOAT32}),
            output_dtypes=frozenset({DType.FLOAT32}),
            minimum_compute_precision=Precision.FP32,
            evidence="an fp16 row under an fp32 floor claims what the floor forbids",
        )


def test_every_problem_is_reported_at_once() -> None:
    """Three faults, one refusal -- not three edit-and-rerun cycles."""
    with pytest.raises(ValueError) as caught:
        a_capability(probe="", evidence="  ", probe_tag="")
    message = str(caught.value)
    assert "benchmarks/probes/" in message
    assert "evidence" in message
    assert "probe_tag" in message


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


def test_a_promotion_is_not_native_support() -> None:
    """float16 in -> float32 compute is a promotion, and must read as one.

    The whole point of separating `accepted_input_dtypes` from
    `native_compute_dtypes`: `compute_dtype_for` answering float32 for a float16
    input is not a statement that float16 is supported. Asserted on a synthetic row
    -- the same claim about the *measured* Optiland record is
    `tests/knowledge/test_capability_pack.py::test_optiland_has_no_float16_path`.
    """
    capability = a_capability()
    assert compute_dtype(DType.FLOAT16) is DType.FLOAT32
    assert DType.FLOAT16 not in capability.native_compute_dtypes
    assert capability.compute_dtype_for(DType.FLOAT16) is DType.FLOAT32
