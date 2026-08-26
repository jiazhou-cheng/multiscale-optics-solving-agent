"""B1-RAY on real CUDA hardware: was the device used, and do the numbers agree.

CHE-106 (M1.1). The CPU-only half of this criterion is asserted in
``tests/test_b1_ray_instances.py``: a CUDA request is refused rather than
downgraded, and no comparison involving CUDA is reported as agreeing. That is a
statement about the refusal path and it cannot close the criterion, which asks
for a measured CPU-vs-CUDA agreement.

This file is the other half and it only runs where a device exists::

    MOA_GPUS=device=6 ./run.sh --gpu pytest -q -m gpu

What it establishes that the CPU session cannot
-----------------------------------------------
The CUDA rows *executed*, and they executed on the device -- read off the live
traced tensor via ``core.arrays.array_state``, never off the request. That
distinction is not academic here: the previous implementation read placement out
of the ``.npz`` beside the record, and because ``np.savez`` requires host bytes it
reported a genuine ``cuda:0`` float64 trace as ``{'namespace': 'numpy',
'device': 'cpu'}`` and therefore as ``honoured_device: false``. A real CUDA
execution was recorded as a downgrade, on the very host that had the device. The
observation was the defect, not the execution, and it was invisible without a GPU.

The four comparison classes the criterion names are measured here rather than
asserted from a committed file: CPU/FP64 vs CUDA/FP64, CPU/FP32 vs CUDA/FP32,
CPU/FP32 vs CPU/FP64, and CUDA/FP32 vs CUDA/FP64 -- plus numpy-vs-torch on the
host, which is what makes it a statement about both supported backends.

Measured on an RTX A6000, torch 2.13.0+cu126, Optiland 0.6.0: the two float64
arms agree to 1.4e-17 of the axial lever arm, i.e. exactly, and the cross-dtype
arms to ~5.5e-7 against a 7.6e-6 float32 floor.
"""

from __future__ import annotations

import importlib.util
import json
import sys

import numpy as np
import pytest

from core.paths import repository_root

pytestmark = [pytest.mark.gpu, pytest.mark.integration, pytest.mark.optiland]

RECORD = (
    repository_root()
    / "benchmarks"
    / "probes"
    / "records"
    / "optiland"
    / "b1_ray_device_precision.json"
)


def _driver():
    name = "b1_ray_driver"
    if name in sys.modules:
        return sys.modules[name]
    path = repository_root() / "benchmarks" / "instances" / "b1_ray.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def cuda_available() -> None:
    import torch

    if not torch.cuda.is_available():  # pragma: no cover - guarded by the gpu marker
        pytest.skip("no CUDA device")


@pytest.fixture(scope="module")
def matrix(cuda_available):
    return _driver().device_precision_matrix()


@pytest.fixture(scope="module")
def agreement(matrix):
    return _driver().device_precision_agreement(matrix)


def test_the_cuda_rows_actually_executed_on_the_device(matrix) -> None:
    """The assertion a CPU session structurally cannot make.

    Both declared CUDA rows must have run, and every one of the three independent
    readings of where they ran must say CUDA: the live array's own
    ``array_state``, Optiland's ``get_device()`` read back after ``set_device``,
    and the ``ArtifactRecord.device`` the adapter stamped from the same
    observation. Three readings because the failure mode is a request echoed as a
    result, and one reading cannot tell the difference.
    """
    cuda_rows = [
        r
        for r in matrix["rows"]
        if r["requested"]["device"] == "cuda" and r["requested"]["backend"] == "torch"
    ]
    assert len(cuda_rows) == 2, cuda_rows
    for row in cuda_rows:
        assert row["outcome"] == "executed", row
        assert row["observed"]["device"].startswith("cuda"), row
        assert row["observed"]["namespace"] == "torch", row
        assert row["requested"]["dtype"] in row["observed"]["dtype"], row
        assert row["applied_to_optiland"]["get_device"].startswith("cuda"), row
        assert row["applied_to_optiland"]["get_precision"] == row["requested"]["dtype"], row
        assert row["record_device"] == "gpu", row
        assert row["record_framework"] == "pytorch", row
        assert row["adapter_reported_mismatches"] == [], row
        assert row["honoured_device"] and row["honoured_dtype"], row
    assert matrix["cuda_executed"] is True


def test_a_cuda_request_cannot_pass_by_executing_on_the_cpu(matrix) -> None:
    """The failure the criterion forbids, checked on the host that could hide it.

    A CUDA row that executed with host arrays is a silent downgrade. On a CPU-only
    session the adapter refuses instead, so this can only be tested here, and it
    is the reason this file exists rather than a parametrization of the CPU one.

    The check is one-sided on purpose: it does not merely require the observation
    to match the request -- which would pass if both were wrong together -- but
    that on a host with a working device, a row asking for CUDA is never observed
    on the CPU.
    """
    import torch

    assert torch.cuda.is_available()
    downgraded = [
        row
        for row in matrix["executed"]
        if row["requested"]["device"] == "cuda"
        and not row["observed"]["device"].startswith("cuda")
    ]
    assert not downgraded, (
        "a CUDA request executed and came back with host arrays. This is the silent "
        f"downgrade the criterion forbids: {downgraded}"
    )
    # And the converse bookkeeping error: nothing may claim CUDA it did not use.
    overclaimed = [
        row
        for row in matrix["executed"]
        if row["observed"]["device"].startswith("cuda")
        and row["requested"]["device"] != "cuda"
    ]
    assert not overclaimed, overclaimed


def test_the_observation_source_is_the_live_array_not_the_npz(matrix) -> None:
    """The specific bug, pinned where only a GPU can pin it.

    On the CPU the two sources are indistinguishable, which is why this went
    unnoticed. Here the ``.npz`` copy reads ``numpy``/``cpu`` for a trace whose
    live tensor was ``torch``/``cuda:0``, and the row records both -- so a future
    reader can see that the persisted copy is a serialization artifact rather than
    the execution device, and a future edit that reverts the source fails here.
    """
    cuda_rows = [
        r for r in matrix["executed"] if r["observed"]["device"].startswith("cuda")
    ]
    assert cuda_rows
    for row in cuda_rows:
        assert row["observed"]["namespace"] == "torch"
        assert row["observed"]["device"].startswith("cuda")
        # The persisted copy is host bytes, and says so.
        assert row["persisted"]["namespace"] == "numpy"
        assert row["persisted"]["device"] == "cpu"
        # The dtype is the one thing that must survive persistence.
        assert row["persisted"]["dtype"] == row["observed"]["dtype"]


def test_every_comparison_class_is_measured_on_this_host(agreement) -> None:
    """All four the criterion names, plus both backends, with nothing unavailable."""
    measured = {c["comparison_id"] for c in agreement["measured"]}
    assert measured == {
        "cpu_fp64_vs_cuda_fp64",
        "cpu_fp32_vs_cuda_fp32",
        "cpu_fp32_vs_cpu_fp64",
        "cuda_fp32_vs_cuda_fp64",
        "numpy_cpu_fp64_vs_torch_cpu_fp64",
    }, sorted(measured)
    assert agreement["unavailable"] == (), agreement["unavailable"]
    assert set(agreement["cuda_comparisons_measured"]) == {
        "cpu_fp64_vs_cuda_fp64",
        "cpu_fp32_vs_cuda_fp32",
        "cuda_fp32_vs_cuda_fp64",
    }


def test_the_agreement_thresholds_hold_on_measured_outputs(agreement) -> None:
    """Recomputed from the arrays, not compared against a recorded constant.

    Each comparison carries both arms' actual placement, the reference and
    compared value of every quantity at the element where they differ most, the
    absolute error, the normalized error and a tolerance derived from the coarser
    of the two precisions. The verdict is recomputed here from those values, so a
    record that agreed with itself but not with the arrays would fail.
    """
    eps32 = float(np.finfo(np.float32).eps)
    eps64 = float(np.finfo(np.float64).eps)
    for comparison in agreement["measured"]:
        dtypes = {
            comparison["actual_reference"]["dtype"],
            comparison["actual_compared"]["dtype"],
        }
        expected = 64.0 * (eps32 if any("float32" in d for d in dtypes) else eps64)
        assert comparison["tolerance"]["threshold"] == pytest.approx(expected), comparison
        for name, quantity in comparison["quantities"].items():
            label = f"{comparison['comparison_id']}/{name}"
            assert abs(
                quantity["reference_value"] - quantity["compared_value"]
            ) == pytest.approx(quantity["absolute_error"], rel=1e-12, abs=1e-300), label
            assert quantity["normalized_error"] == pytest.approx(
                quantity["absolute_error"] / quantity["normalization_scale"], rel=1e-12
            ), label
            assert quantity["normalized_error"] <= expected, (
                label,
                quantity["normalized_error"],
                expected,
            )
        assert comparison["met"], comparison["comparison_id"]
    assert agreement["all_measured_met"]


def test_float64_across_the_two_devices_is_essentially_exact(agreement) -> None:
    """The physically interesting result, separated from the bookkeeping.

    The same float64 refraction arithmetic on a CPU and on an A6000 is not merely
    inside a round-off bound -- it agrees to a small multiple of eps, because
    nothing in a sequential ray trace reassociates a reduction the way a matmul
    does. Stating that as its own assertion is what would catch a GPU kernel that
    silently used a lower internal precision: a TF32-style downgrade would show up
    here at ~1e-3 relative while still passing a 64*eps(float32) bound.
    """
    comparison = next(
        c for c in agreement["measured"] if c["comparison_id"] == "cpu_fp64_vs_cuda_fp64"
    )
    assert comparison["actual_reference"]["device"] == "cpu"
    assert comparison["actual_compared"]["device"].startswith("cuda")
    assert comparison["worst_normalized_error"] < 100 * float(np.finfo(np.float64).eps), (
        comparison["worst_normalized_error"],
        comparison["quantities"],
    )


def test_the_cross_dtype_cost_is_real_and_bounded(agreement) -> None:
    """A float32 arm must differ from a float64 one, and by about the right amount.

    An agreement that came out exactly zero across precisions would mean the dtype
    request was not honoured -- which is the same defect as the device one and
    deserves the same treatment. So the cross-dtype comparisons are asserted to be
    nonzero AND inside the float32 floor, and the CUDA and CPU arms are asserted to
    pay a similar cost, because a device-specific precision surprise would show as
    an asymmetry between them.
    """
    floor = 64.0 * float(np.finfo(np.float32).eps)
    costs = {}
    for comparison_id in ("cpu_fp32_vs_cpu_fp64", "cuda_fp32_vs_cuda_fp64"):
        comparison = next(
            c for c in agreement["measured"] if c["comparison_id"] == comparison_id
        )
        worst = comparison["worst_normalized_error"]
        assert 0.0 < worst <= floor, (comparison_id, worst, floor)
        costs[comparison_id] = worst
    ratio = costs["cuda_fp32_vs_cuda_fp64"] / costs["cpu_fp32_vs_cpu_fp64"]
    assert 0.1 < ratio < 10.0, costs


def test_the_committed_record_reproduces_on_this_host(agreement) -> None:
    """The persisted evidence, re-measured.

    The record under ``benchmarks/probes/records/optiland/`` is what makes this
    criterion checkable without a GPU, and a committed measurement nobody re-runs
    is exactly the artifact that goes stale silently. So the GPU session compares
    the live agreement against the committed one: same comparison set, same
    verdicts, and the same errors to within a factor of a few -- not bit-exact,
    because the record may have been produced on a different device of the same
    class, and pretending otherwise would make this fail for the wrong reason.
    """
    assert RECORD.is_file()
    recorded = json.loads(RECORD.read_text())
    assert recorded["environment"]["cuda_executed"] is True
    by_id = {
        c["comparison_id"]: c
        for c in recorded["agreement"]["comparisons"]
        if c["status"] == "measured"
    }
    live = {c["comparison_id"]: c for c in agreement["measured"]}
    assert set(by_id) == set(live), (sorted(by_id), sorted(live))
    for comparison_id, comparison in live.items():
        stored = by_id[comparison_id]
        assert stored["met"] is comparison["met"], comparison_id
        assert stored["tolerance"]["threshold"] == pytest.approx(
            comparison["tolerance"]["threshold"]
        ), comparison_id
        stored_worst = float(stored["worst_normalized_error"])
        live_worst = comparison["worst_normalized_error"]
        if max(stored_worst, live_worst) > 1e-15:
            assert stored_worst == pytest.approx(live_worst, rel=10.0), (
                comparison_id,
                stored_worst,
                live_worst,
            )


def test_the_gated_families_still_close_on_the_device(cuda_available) -> None:
    """The ray gates are declared float64-only, and this says what that costs.

    ``RAY_EXECUTION`` declares ``dtypes={FLOAT64}`` and ``devices={CPU, CUDA}``,
    so the gating instances are entitled to run on a device and must still meet
    their tolerances there. Two are checked rather than all eight, because the
    point is that the device does not change the physics and the cheapest
    demonstration of that is the machine-precision one plus the corrected
    conservation law.

    Note what this does NOT do: it does not re-derive the gates or re-record them.
    The canonical records are the CPU ones, and a GPU run that agreed with them is
    evidence about the device, not a second source of truth about the optics.
    """
    driver = _driver()
    for instance_id in ("B1-RAY-SNELL-01", "B1-RAY-LAGRANGE-01"):
        run = driver.run_instance(instance_id)
        gating = [m for m in run.result.physics_accuracy if m.tolerance_may_gate]
        assert gating, instance_id
        for metric in gating:
            assert metric.met, (instance_id, metric.metric, metric.measured.value)
