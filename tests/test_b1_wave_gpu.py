"""Chromatix on real CUDA: did the propagation run there, and does the label come
from the array.

CHE-107 (M1.2). The host-only half of this criterion is asserted in
``tests/test_b1_wave_instances.py``: a CUDA request is REFUSED rather than
quietly served from the CPU, with "no silent fallback" in the detail. That is a
statement about the refusal path, and it cannot close a criterion that asks for
a GPU run whose device is read off the output array -- because on a CPU-only
image there is no output array that could have been on a device.

This file is the other half and it only runs where a device exists::

    MOA_GPUS=device=6 ./run.sh --gpu pytest -q -m gpu

What this session can establish that a CPU session structurally cannot
----------------------------------------------------------------------
The hazard the registry note names is a *successful* run: a process-global JAX
platform pin serves a caller who asked for CUDA from the host, raises nothing,
and returns an entirely ordinary field. Every symptom of that failure is absent
by construction, so the only thing that distinguishes it from a real GPU run is
where the observation came from. On the host both candidate sources agree, and
the wrong one is therefore invisible; here they disagree, and the disagreement
is the test:

* ``metadata['execution']['actual']`` is ``core.arrays.array_state(field_out.u)``
  taken before ``jax.device_get`` -- ``jax``/``cuda:0``/``complex64``.
* the persisted ``.npy`` beside it is host bytes because ``np.save`` requires
  them -- ``numpy``/``cpu``, for the identical run.

The driver used to read the second one. On the ray side the same mistake
reported a genuine ``cuda:0`` float64 trace as a host downgrade: the observation
was the defect, not the execution, and no CPU session could have seen it.

A third asymmetry only this session has: ``jax.default_backend()`` is ``gpu``
here, and the CPU-requested row is still observed on the CPU. That is the
converse claim -- the placement is not being read off a process-wide default
either.

Measured on an RTX A6000, jax 0.6.2 + jax-cuda12-plugin, Chromatix 0.6.0: both
devices propagate B1-WAVE-GAUSS-01's field 100 um and agree to 3.30e-5 against a
1.41e-4 complex64 floor.
"""

from __future__ import annotations

import importlib.util
import json
import sys

import pytest

from core.paths import repository_root

pytestmark = [pytest.mark.gpu, pytest.mark.integration, pytest.mark.chromatix]

RECORD = (
    repository_root()
    / "benchmarks"
    / "probes"
    / "records"
    / "chromatix"
    / "b1_wave_device_observation.json"
)


def _driver():
    name = "b1_wave_driver"
    if name in sys.modules:
        return sys.modules[name]
    path = repository_root() / "benchmarks" / "instances" / "b1_wave.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def cuda_available() -> None:
    from solvers.chromatix.execution import _jax_gpu_unavailable_reason

    reason = _jax_gpu_unavailable_reason()
    if reason is not None:  # pragma: no cover - guarded by the gpu marker
        pytest.skip(f"no usable CUDA device for jax: {reason}")


@pytest.fixture(scope="module")
def observation(cuda_available):
    return _driver().device_observation()


@pytest.fixture(scope="module")
def rows(observation):
    return {row["requested_device"]: row for row in observation["rows"]}


def test_the_cuda_request_actually_executed_on_the_device(rows) -> None:
    """The assertion this criterion has been open on, taken on a device.

    Both readings of the one observation must say CUDA -- the live array's
    ``array_state`` and the ``ArtifactRecord`` the adapter stamped from it -- and
    the adapter must report no device mismatch. Two readings rather than one
    because the failure mode is a request echoed back as a result, and a single
    reading cannot tell an echo from an observation.
    """
    cuda = rows["cuda"]
    assert cuda["outcome"] == "executed", cuda
    assert cuda["observed"]["device"] == "cuda:0", cuda["observed"]
    assert cuda["observed"]["namespace"] == "jax", cuda["observed"]
    # complex64 is not a choice: ScalarField.__init__ casts unconditionally, and
    # this is that fact read off the output rather than off the capability table.
    assert cuda["observed"]["dtype"] == "complex64", cuda["observed"]
    assert cuda["artifact"]["device"] == "gpu", cuda["artifact"]
    assert cuda["artifact"]["framework"] == "jax", cuda["artifact"]
    assert cuda["device_mismatch"] is False, cuda
    assert cuda["honoured"] is True, cuda


def test_a_cuda_request_cannot_pass_by_executing_on_the_host(observation) -> None:
    """The silent downgrade, checked on the host that could hide it.

    One-sided on purpose. Asserting that observation matches request would pass
    if both were wrong together; what is asserted instead is that on a host with
    a working device, no row asking for CUDA is ever observed on the CPU, and
    nothing claims a CUDA it did not use.
    """
    assert observation["cuda_executed"] is True
    downgraded = [
        row
        for row in observation["executed"]
        if row["requested_device"] == "cuda"
        and not row["observed"]["device"].startswith("cuda")
    ]
    assert not downgraded, (
        "a CUDA request executed and came back on the host. This is the silent "
        f"fallback the registry note forbids: {downgraded}"
    )
    overclaimed = [
        row
        for row in observation["executed"]
        if row["observed"]["device"].startswith("cuda") and row["requested_device"] != "cuda"
    ]
    assert not overclaimed, overclaimed


def test_the_observation_source_is_the_live_array_not_the_persisted_npy(rows) -> None:
    """The specific defect, pinned where only a device can pin it.

    On the CPU these two sources are indistinguishable, which is exactly why this
    class of bug survives review. Here the persisted copy reads ``numpy``/``cpu``
    for a run whose live array was ``jax``/``cuda:0``, and the row carries both --
    so a reader can see that the ``.npy`` placement is a serialization artifact,
    and an edit that reverts the source to ``np.load(...)`` fails here.
    """
    cuda = rows["cuda"]
    assert cuda["observed"]["device"] == "cuda:0"
    assert cuda["persisted"]["namespace"] == "numpy"
    assert cuda["persisted"]["device"] == "cpu"
    # If these two ever agreed on this host, the test below would be vacuous.
    assert cuda["persisted"]["device"] != cuda["observed"]["device"]
    # The dtype is the one property that must survive persistence, and does.
    assert cuda["persisted"]["dtype"] == cuda["observed"]["dtype"] == "complex64"
    assert "array_state" in cuda["observation_source"]
    assert "not the request" in cuda["observation_source"]


def test_a_host_run_is_not_mislabelled_cuda_on_a_gpu_host(rows) -> None:
    """The converse, and it needs a GPU host to mean anything.

    ``jax.default_backend()`` is ``gpu`` in this session, so a placement read off
    the process default rather than off the array would report the CPU row as
    CUDA. It does not. That is the half of "read it off the array" that a
    CPU-only session cannot check at all, because there the default backend and
    the correct answer are the same string.
    """
    cpu = rows["cpu"]
    assert cpu["outcome"] == "executed"
    assert cpu["jax_default_backend"] == "gpu", (
        "this assertion is only meaningful when the process default disagrees "
        "with the row's actual placement"
    )
    assert cpu["observed"]["device"] == "cpu", cpu["observed"]
    assert cpu["artifact"]["device"] == "cpu", cpu["artifact"]
    assert cpu["device_mismatch"] is False
    assert cpu["honoured"] is True


def test_the_two_devices_compute_the_same_field(observation) -> None:
    """Placement is not execution, and this is the difference.

    Every assertion above would be satisfied by a stub that returned zeros on the
    device: they establish where an array was, not that it holds a propagation.
    So the two arms are differenced. The bound is the complex64 floor -- one
    float32 epsilon per radian of accumulated phase, the basis B1-WAVE-FWDBWD
    uses -- because both devices run complex64 and the only admissible difference
    is a reassociated FFT reduction.

    It is asserted to be nonzero as well. An exactly identical result across two
    FFT implementations would mean one arm did not run where it said it did.
    """
    agreement = observation["agreement"]
    assert agreement["status"] == "measured", agreement
    assert agreement["met"], agreement
    assert 0.0 < agreement["measured"] <= agreement["threshold"], agreement
    assert agreement["identical"] is False


def test_the_gated_gaussian_instance_still_meets_its_tolerance(cuda_available) -> None:
    """The device does not change the physics, said as a gate rather than a diff.

    ``WAVE_EXECUTION`` declares CUDA as a supported device, so an instance that
    gates on the host is entitled to gate here. One instance rather than nine:
    the point is that placement is orthogonal to correctness, and the cheapest
    honest demonstration is the family whose oracle is a closed form.

    Note what this does NOT do -- it does not re-record the gate. The canonical
    records stay CPU-produced, because a record whose numbers depend on which
    image happened to run it is not a canonical record.
    """
    run = _driver().run_instance("B1-WAVE-GAUSS-01")
    metric = next(
        m for m in run.result.physics_accuracy if m.metric == "gaussian_radius_relative_error"
    )
    assert metric.met is True, f"{metric.measured.value:.3e}"


def test_the_committed_record_reproduces_on_this_host(observation) -> None:
    """The persisted evidence, re-measured rather than trusted.

    The record under ``benchmarks/probes/records/chromatix/`` is what makes this
    criterion inspectable without a GPU, and a committed measurement nobody
    re-runs is precisely the artifact that goes stale in silence. Compared loosely
    on the number -- within a factor of a few -- because it may have been produced
    on a different device of the same class, and strictly on everything
    categorical, which is where a regression would actually show.
    """
    assert RECORD.is_file(), f"{RECORD} is missing; run `--gpu python ... --device`"
    recorded = json.loads(RECORD.read_text())
    assert recorded["environment"]["cuda_executed"] is True
    assert recorded["environment"]["jax"]["gpu_unavailable_reason"] is None

    by_device = {row["requested_device"]: row for row in recorded["rows"]}
    for device, row in {r["requested_device"]: r for r in observation["rows"]}.items():
        stored = by_device[device]
        assert stored["outcome"] == row["outcome"], device
        if row["outcome"] != "executed":
            continue
        assert stored["observed"] == row["observed"], device
        assert stored["artifact"] == row["artifact"], device
        assert stored["persisted"] == row["persisted"], device
        assert stored["honoured"] == row["honoured"], device

    stored_agreement = recorded["agreement"]
    live_agreement = observation["agreement"]
    assert stored_agreement["status"] == live_agreement["status"] == "measured"
    assert stored_agreement["met"] is live_agreement["met"] is True
    assert stored_agreement["threshold"] == pytest.approx(live_agreement["threshold"])
    assert float(stored_agreement["measured"]) == pytest.approx(
        live_agreement["measured"], rel=10.0
    )


def test_the_record_cannot_be_downgraded_by_a_host_only_session(tmp_path) -> None:
    """Why the committed CUDA-positive record survives the next CPU run.

    A host-only session produces a structurally complete record whose CUDA row
    says "refused". Writing it over this one would delete the evidence for an
    acceptance criterion and leave a file that still looks finished, which is the
    worst of the available failures. The writer refuses instead.

    Checked here rather than on the CPU because it needs a genuine CUDA-positive
    record to try to overwrite; the tmp_path copy stands in for the committed one
    so the committed one is never at risk from the test itself.
    """
    target = tmp_path / "b1_wave_device_observation.json"
    target.write_text(json.dumps({"environment": {"cuda_executed": True}}))
    module = _driver()
    # Force the host-only shape of the observation without needing a CPU image.
    original = module.device_observation
    module.device_observation = lambda: {
        "rows": ({"requested_device": "cuda", "outcome": "refused", "detail": "x"},),
        "executed": (),
        "refused": ({"requested_device": "cuda", "outcome": "refused", "detail": "x"},),
        "cuda_executed": False,
        "agreement": {"status": "unavailable", "reason": "executed arms: []"},
        "outputs": {},
    }
    try:
        with pytest.raises(RuntimeError, match="Refusing to overwrite"):
            module.write_device_observation_record(tmp_path)
        assert json.loads(target.read_text())["environment"]["cuda_executed"] is True
        # And the deliberate escape hatch does work, so this is a guard and not a lock.
        written = module.write_device_observation_record(tmp_path, allow_downgrade=True)
        assert json.loads(written.read_text())["environment"]["cuda_executed"] is False
    finally:
        module.device_observation = original


def test_the_canonical_records_stay_host_produced() -> None:
    """The GPU session answers a placement question; it does not own the physics.

    Stated as a property of the committed evidence rather than as a note. Every
    canonical B1-WAVE record must report ``cpu``, because a record whose numbers
    depend on which image happened to run it is not canonical -- and because a
    missing device should leave this criterion OPEN rather than make the family's
    evidence disappear. If a GPU session ever writes one of these, this fails.
    """
    records = sorted(
        (repository_root() / "benchmarks" / "instances" / "records").glob("B1-WAVE-*.json")
    )
    assert records
    for path in records:
        cost = json.loads(path.read_text())["verification"]["resource_cost"]
        assert cost["device"] == "cpu", f"{path.name} was produced on {cost['device']}"
