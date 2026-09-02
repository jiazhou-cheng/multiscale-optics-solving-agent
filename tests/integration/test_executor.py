"""A plan runs, and the record says what happened. Never whether it was right.

CHE-200 (R13.2). Acceptance criteria:

1. the same plan twice gives bit-identical physical results and records differing
   only in declared-volatile fields;
2. a refused or failed node produces structured diagnostics -- never a fabricated
   result, never invented convergence or provenance;
3. caching only against a measured cost. The measurement is in the commit and
   `test_the_environment_fingerprint_is_read_once_per_run` pins the one thing that
   *is* remembered; `tests/unit/test_provenance_separation.py` asserts no result
   cache exists;
4. the memory guard is present and the swap-growth condition terminates the run
   and reports a resource failure;
5. resource state is released deterministically, and two `Executor` instances in
   one process do not interfere;
6. class delta +1, rule 3 -- `scripts/class_budget.py` owns the count.

The risk this file is written against is `Executor` becoming the new
`GraphExecutor`: every one of that class's 800 lines was added for a reason that
seemed local. So the tests below are about the contract -- what a record contains,
what a failure returns, when the resource is released -- and not about internal
structure that would make a refactor fail.
"""

from __future__ import annotations

import functools
import json
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fixtures.systems import singlet_ref, singlet_source

import runtime
from measurements import psf
from representations import ReferenceSurface
from runtime import ExecutionRecord, Executor, execute, memory_snapshot, strip_volatile
from runtime.executor import RUNTIME_CODES
from sources import gaussian_beam

ROOT = Path(__file__).resolve().parents[2]
SURFACE = ReferenceSurface(name="emitting surface", z_m=0.0, medium_index=1.0)

#: A two-node wave plan: an analytic source into a measurement. Backend-free,
#: exactly reproducible, and cheap enough to run many times in this file.
WAVE_PLAN = ("S_SOURCE_GAUSSIAN_BEAM", "M_PSF")
WAVE_REQUEST: dict[str, Any] = {
    "shape": (32, 32),
    "sample_pitch_m": (1.0e-6, 1.0e-6),
    "wavelength_m": 5.5e-7,
    "reference_surface": SURFACE,
    "waist_radius_m": 6.0e-6,
    "normalization": "peak",
}


def _physical(record: ExecutionRecord) -> Any:
    """Re-derive the plan's physical result, so a record can be compared against it."""
    field = gaussian_beam(
        WAVE_REQUEST["shape"],
        sample_pitch_m=WAVE_REQUEST["sample_pitch_m"],
        wavelength_m=WAVE_REQUEST["wavelength_m"],
        reference_surface=SURFACE,
        waist_radius_m=WAVE_REQUEST["waist_radius_m"],
    )
    assert record.status == "completed"
    return np.asarray(psf(field, normalization="peak").intensity)


# ---------------------------------------------------------------------------
# 1. Criterion 1 -- twice, identically
# ---------------------------------------------------------------------------


def test_a_plan_runs_and_the_record_describes_it() -> None:
    record = execute(WAVE_PLAN, request=WAVE_REQUEST)
    assert record.status == "completed"
    assert record.route == WAVE_PLAN
    assert tuple(node.operation_id for node in record.nodes) == WAVE_PLAN
    assert all(node.status == "completed" for node in record.nodes)
    assert all(node.diagnostics == "" for node in record.nodes)
    assert set(record.provenance) == {
        "schema_version",
        "code_fingerprint",
        "environment_fingerprint",
        "runtime_seconds",
        "resources",
    }


def test_the_same_plan_twice_differs_only_in_declared_volatile_fields() -> None:
    """Criterion 1, and the comparison is the whole of it.

    `strip_volatile` projects out `runtime_seconds` and `resources` -- a run's
    duration and its peak memory are measurements of the run, not of the
    computation -- so what is left has to be identical. If it were not, a
    fingerprint over it would report a change nobody made.
    """
    first = execute(WAVE_PLAN, request=WAVE_REQUEST)
    second = execute(WAVE_PLAN, request=WAVE_REQUEST)

    assert first.fingerprinted == second.fingerprinted
    assert strip_volatile(dict(first.provenance)) == strip_volatile(dict(second.provenance))
    # And the volatile fields are really there, so the projection is doing work
    # rather than hiding two empty dicts.
    assert "runtime_seconds" in first.provenance
    assert first.provenance["resources"]["samples"] >= 1
    assert first.provenance != second.provenance or (
        first.provenance["runtime_seconds"] == second.provenance["runtime_seconds"]
    )


def test_the_physical_result_is_bit_identical_across_runs() -> None:
    """The other half of criterion 1: the numbers, not just the record.

    Compared against the physics run outside the executor entirely, which is the
    stronger statement -- the executor does not perturb what it runs.
    """
    reference = _physical(execute(WAVE_PLAN, request=WAVE_REQUEST))
    again = _physical(execute(WAVE_PLAN, request=WAVE_REQUEST))
    assert np.array_equal(reference, again)

    outside = gaussian_beam(
        WAVE_REQUEST["shape"],
        sample_pitch_m=WAVE_REQUEST["sample_pitch_m"],
        wavelength_m=WAVE_REQUEST["wavelength_m"],
        reference_surface=SURFACE,
        waist_radius_m=WAVE_REQUEST["waist_radius_m"],
    )
    assert np.array_equal(np.asarray(psf(outside, normalization="peak").intensity), reference)


def test_the_record_serializes_and_reads_back() -> None:
    record = execute(WAVE_PLAN, request=WAVE_REQUEST)
    # `request` holds a `ReferenceSurface`, which is not JSON. That is honest
    # rather than a defect: the record's `request` is the caller's typed inputs,
    # and serializing one is the caller's problem. What must round-trip is a
    # record whose request is serializable.
    serializable = ExecutionRecord(
        route=record.route,
        request={"normalization": "peak"},
        nodes=record.nodes,
        provenance=record.provenance,
    )
    assert runtime.from_json(runtime.to_json(serializable)) == serializable
    assert json.loads(runtime.to_json(serializable))["provenance"]["resources"]["samples"] >= 1


# ---------------------------------------------------------------------------
# 2. Criterion 2 -- a failure returns diagnostics, never a result
# ---------------------------------------------------------------------------


def test_a_request_missing_a_required_argument_is_refused_by_name() -> None:
    """Criterion 2. A runtime does not choose a physical parameter for a caller.

    `psf`'s `normalization` is the case that makes that a rule: which
    normalization was used is the subject of three of R11's acceptance criteria,
    and a runtime that picked one would be inventing the answer to them.
    """
    incomplete = {key: value for key, value in WAVE_REQUEST.items() if key != "normalization"}
    record = execute(WAVE_PLAN, request=incomplete)

    assert record.status == "refused"
    assert len(record.nodes) == 2, "the source ran; the measurement is what refused"
    assert record.nodes[0].status == "completed"
    refused = record.nodes[1]
    assert refused.operation_id == "M_PSF"
    assert refused.status == "refused"
    assert "normalization" in refused.diagnostics
    assert "INCOMPLETE_REQUEST" in refused.diagnostics


@pytest.mark.filterwarnings("ignore:overflow encountered:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:invalid value encountered:RuntimeWarning")
def test_a_boundary_refusal_is_recorded_as_a_refusal_and_not_a_failure() -> None:
    """The distinction the record keeps: "the boundary said no" is not "it broke".

    `psf` refuses a non-finite intensity with a `ContractError` carrying a `code`,
    and the executor reads that code -- so a refusal is a fact about the problem
    and a raised exception without one is a fact about the run.
    """
    record = execute(("M_PSF",), request={"field": _overflowing_field(), "normalization": "peak"})
    assert record.status == "refused"
    node = record.nodes[0]
    assert node.status == "refused"
    assert node.diagnostics.startswith("NON_FINITE"), node.diagnostics
    assert "intensity" in node.diagnostics


def _overflowing_field() -> Any:
    """A valid complex64 field whose |u|^2 overflows to inf. R11.1's own case."""
    from representations import ScalarField

    u = np.zeros((8, 8), dtype=np.complex64)
    u[1, 1] = np.complex64(2.0e19)
    u[2, 2] = np.complex64(1.0)
    return ScalarField(
        u=u,
        sample_pitch_m=(1.0e-6, 1.0e-6),
        wavelength_m=5.5e-7,
        reference_surface=SURFACE,
    )


def test_a_raised_exception_without_a_code_is_recorded_as_a_failure() -> None:
    """The other arm, so the two statuses are known to be distinguishable."""
    record = execute(
        ("M_PSF",), request={"field": "not a field at all", "normalization": "peak"}
    )
    assert record.status == "failed"
    node = record.nodes[0]
    assert node.status == "failed"
    assert node.diagnostics, "a failed node returns diagnostics, never silence"
    assert ":" in node.diagnostics, "the diagnostic names the exception type"


def test_nothing_downstream_of_a_refusal_runs_and_nothing_is_invented() -> None:
    """The record is shorter than the route, which is how `status` knows.

    And there is no fabricated node for what did not run: a record that padded the
    route with plausible-looking successes is the exact failure AGENTS.md's "never
    invent fields, metrics, convergence or provenance" names.
    """
    record = execute(
        ("M_PSF", "S_SOURCE_GAUSSIAN_BEAM"),
        request={"field": "not a field at all", "normalization": "peak"},
    )
    assert len(record.route) == 2
    assert len(record.nodes) == 1
    assert record.status == "failed"


def test_a_plan_naming_an_operation_the_catalog_does_not_have_is_refused() -> None:
    with pytest.raises(ValueError, match="the catalog does not have"):
        execute(("X_INVENTED",), request={})
    with pytest.raises(ValueError, match="at least one operation"):
        execute((), request={})


def test_a_route_starting_mid_graph_needs_its_input_named() -> None:
    """`M_PSF` consumes a field, so a plan starting there must be handed one."""
    record = execute(("M_PSF",), request={"normalization": "peak"})
    assert record.status == "refused"
    assert "field" in record.nodes[0].diagnostics
    assert "consumes a representation" in record.nodes[0].diagnostics


# ---------------------------------------------------------------------------
# 3. Criterion 3 -- what is remembered, and what is not
# ---------------------------------------------------------------------------


def test_the_environment_fingerprint_is_read_once_per_run() -> None:
    """The one memoization, and the measurement that justified it.

    `environment_fingerprint()` is 7.5 ms -- `importlib.metadata.version` scans the
    filesystem once per package and there are seven -- and computing it per record
    made a two-node run 391% slower than the physics. Installed versions cannot
    change inside a process, so it is read when the executor opens.

    Counted rather than timed, because a timing assertion in the default gate is a
    flake. Two runs through one executor read it once; two executors read it twice,
    which is the property that keeps it per-lifecycle rather than global.
    """
    calls = 0
    real = runtime.records.environment_fingerprint

    def counted() -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return real()

    import runtime.executor as executor_module

    original = executor_module.environment_fingerprint
    executor_module.environment_fingerprint = counted  # type: ignore[assignment]
    try:
        with Executor() as first:
            first.execute(WAVE_PLAN, request=WAVE_REQUEST)
            first.execute(WAVE_PLAN, request=WAVE_REQUEST)
        assert calls == 1, calls
        with Executor() as second:
            second.execute(WAVE_PLAN, request=WAVE_REQUEST)
        assert calls == 2, calls
    finally:
        executor_module.environment_fingerprint = original  # type: ignore[assignment]


def test_an_implementation_the_executor_cannot_bind_fails_loudly() -> None:
    """The variadic refusal, and the mistake that made it necessary.

    A `*args, **kwargs` implementation has no parameter a request can fill by name,
    and classifying the variadics as "required" made the executor demand arguments
    called `args` and `kwargs` -- a diagnostic naming a stub's parameters that told
    nobody anything. Nothing in the catalog is variadic, so this is latent.
    """
    psf_module = sys.modules["measurements.psf"]
    original = psf_module.psf

    def unbindable(*args: Any, **kwargs: Any) -> Any:  # no functools.wraps, on purpose
        raise AssertionError("must not be called")

    psf_module.psf = unbindable  # type: ignore[assignment]
    try:
        record = execute(("M_PSF",), request={"normalization": "peak"})
    finally:
        psf_module.psf = original  # type: ignore[assignment]

    assert record.status == "failed"
    assert record.nodes[0].diagnostics.startswith("UNBINDABLE_SIGNATURE")
    assert "variadic" in record.nodes[0].diagnostics
    # And every real implementation binds, which is what makes the above latent.
    from operations import CATALOG
    from operations import resolve as resolve_operation
    from runtime.executor import _port_and_parameters

    for descriptor in CATALOG:
        _port_and_parameters(resolve_operation(descriptor.operation_id))


def test_two_runs_of_one_plan_both_execute_the_physics() -> None:
    """No result cache: the second run computes, it does not replay.

    Asserted by counting calls into the implementation, because the observable
    result would be identical either way -- which is exactly why a silent cache
    here would be undetectable and is refused.
    """
    calls = 0
    real = psf

    # `functools.wraps` is load-bearing, not decoration: `inspect.signature`
    # follows `__wrapped__`, and without it the stub's own `*args, **kwargs`
    # signature is what the executor binds against -- which it now refuses
    # outright (`UNBINDABLE_SIGNATURE`). The first version of this test hit exactly
    # that and reported "missing ['a', 'k']".
    @functools.wraps(real)
    def counted(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        return real(*args, **kwargs)

    # `sys.modules[...]` and not `import measurements.psf as m`: the package
    # re-exports the function under the module's own name, so the attribute is the
    # callable and the module is only reachable through `sys.modules`. Patching the
    # module is what matters -- `operations.resolve` imports it and reads the
    # attribute off it, so a patch on the package would not be seen.
    psf_module = sys.modules["measurements.psf"]
    original = psf_module.psf
    psf_module.psf = counted  # type: ignore[assignment]
    try:
        with Executor() as executor:
            executor.execute(WAVE_PLAN, request=WAVE_REQUEST)
            executor.execute(WAVE_PLAN, request=WAVE_REQUEST)
    finally:
        psf_module.psf = original  # type: ignore[assignment]
    assert calls == 2, f"the second run did not execute the physics ({calls} call(s))"


# ---------------------------------------------------------------------------
# 4. Criterion 4 -- the memory guard and the swap-growth stop condition
# ---------------------------------------------------------------------------


def test_the_guard_is_present_and_reports_what_it_observed() -> None:
    """A resource record on every run, whether or not anything tripped."""
    record = execute(WAVE_PLAN, request=WAVE_REQUEST)
    resources = record.provenance["resources"]
    assert resources["samples"] >= 1
    assert resources["peak_rss_bytes"] > 0
    assert set(resources) == {
        "samples",
        "peak_rss_bytes",
        "peak_cgroup_swap_bytes",
        "swap_growth_bytes",
        "resource_failure",
        "baseline",
    }
    assert resources["resource_failure"] is None, "nothing tripped on a clean run"
    # The host's swap is recorded beside the container's, so a report can show that
    # the host number moves for other tenants rather than leaving it unexplained.
    assert "host_swap_used_bytes" in resources["baseline"]


def test_swap_growth_terminates_the_run_and_reports_a_resource_failure() -> None:
    """Criterion 4. Growth at all, not a threshold.

    Simulated by moving the *baseline* down after it was taken, which is the same
    arithmetic as swap growing above it and is the only way to test this without
    actually swapping on a shared server. The tolerance is zero, so one byte is
    enough -- and using one byte is the point: this is a stop condition, not a
    slowdown to tolerate.
    """
    with Executor() as executor:
        assert executor.resource_failure() is None, "nothing tripped before the simulation"
        baseline = dict(executor._baseline or {})
        observed = executor._peak_swap_bytes
        if observed is None:
            # No cgroup v2 swap file: the guard reads `None` and cannot trip, which
            # is the honest behaviour off a container. Assert that rather than skip,
            # so the test says what it measured.
            assert executor.swap_growth_bytes is None
            pytest.skip("this platform exposes no cgroup swap counter")
        executor._baseline = {**baseline, "cgroup_swap_bytes": int(observed) - 1}

        assert executor.swap_growth_bytes == 1
        failure = executor.resource_failure()
        assert failure is not None
        assert "RESOURCE_LIMIT" in failure
        assert "swap grew by 1 bytes" in failure

        record = executor.execute(WAVE_PLAN, request=WAVE_REQUEST)

    assert record.status == "failed"
    assert len(record.nodes) == 1, "the run stopped before the first operation"
    node = record.nodes[0]
    assert node.operation_id == WAVE_PLAN[0], "the node names what was NOT run"
    assert node.status == "failed"
    assert "RESOURCE_LIMIT" in node.diagnostics
    assert record.provenance["resources"]["swap_growth_bytes"] == 1


def test_growth_during_the_only_node_is_reported_rather_than_read_as_success() -> None:
    """The gap the review found: a one-node plan could never trip the guard.

    The boundary check runs *before* each node, so for a single-operation plan --
    the shape of every heavy solver run -- growth during the only node was observed
    by the sampler and acted on by nobody, and the record read `completed`. That is
    a run that tripped a stop condition reading as a clean success, which
    `AGENTS.md` does not allow.

    It is reported as an observation on the `resources` block rather than as an
    extra failed node, because the operation *did* run: fabricating a failed record
    for a completed operation would be inventing an outcome, which is the other
    half of the same rule.
    """
    field = gaussian_beam(
        WAVE_REQUEST["shape"],
        sample_pitch_m=WAVE_REQUEST["sample_pitch_m"],
        wavelength_m=WAVE_REQUEST["wavelength_m"],
        reference_surface=SURFACE,
        waist_radius_m=WAVE_REQUEST["waist_radius_m"],
    )
    with Executor() as executor:
        if (executor._baseline or {}).get("cgroup_swap_bytes") is None:
            pytest.skip("this platform exposes no cgroup swap counter")

        # A one-node plan, with the growth appearing *during* it: the sampler's
        # peak is moved as the operation runs, which is what a real swap event
        # looks like from here.
        real = sys.modules["measurements.psf"].psf

        @functools.wraps(real)
        def swapping(*args: Any, **kwargs: Any) -> Any:
            executor._peak_swap_bytes = int(
                (executor._baseline or {})["cgroup_swap_bytes"] or 0
            ) + 4096
            return real(*args, **kwargs)

        sys.modules["measurements.psf"].psf = swapping  # type: ignore[assignment]
        try:
            record = executor.execute(
                ("M_PSF",), request={"field": field, "normalization": "peak"}
            )
        finally:
            sys.modules["measurements.psf"].psf = real  # type: ignore[assignment]

    # The operation ran and completed, and the record says so -- and it also says
    # the stop condition tripped, which is the part that was missing.
    assert [node.status for node in record.nodes] == ["completed"]
    failure = record.provenance["resources"]["resource_failure"]
    assert failure is not None, "growth during the only node was not reported at all"
    assert "RESOURCE_LIMIT" in failure
    assert record.provenance["resources"]["swap_growth_bytes"] == 4096


def test_the_guard_watches_the_container_and_not_the_host() -> None:
    """On a shared server the host's swap moves for other tenants' reasons.

    Guarding on it would terminate runs that used no swap at all, which is the
    reason `AGENTS.md`'s policy is about the workload's swap.
    """
    with Executor() as executor:
        baseline = dict(executor._baseline or {})
        if baseline.get("cgroup_swap_bytes") is None:
            pytest.skip("this platform exposes no cgroup swap counter")
        # The baseline is moved *below* the current host reading, so a guard that
        # watched the host would see growth. Setting it to 0 -- the first version --
        # could not discriminate on a host whose swap is currently unused, which is
        # this one.
        current_host = memory_snapshot()["host_swap_used_bytes"]
        executor._baseline = {
            **baseline,
            "host_swap_used_bytes": (0 if current_host is None else int(current_host)) - 4096,
        }
        assert executor.resource_failure() is None, (
            "a change in the host's swap tripped the guard"
        )


def test_execute_outside_a_context_manager_is_refused() -> None:
    """Without the sampler running, the stop condition would not be evaluated."""
    executor = Executor()
    with pytest.raises(RuntimeError, match="inside `with`"):
        executor.execute(WAVE_PLAN, request=WAVE_REQUEST)


def test_reentering_one_executor_is_refused() -> None:
    with Executor() as executor, pytest.raises(RuntimeError, match="already open"):
        executor.__enter__()


# ---------------------------------------------------------------------------
# 5. Criterion 5 -- deterministic release, and two executors
# ---------------------------------------------------------------------------


def test_reopening_an_executor_resets_its_observations() -> None:
    """A second `with` must not carry the first run's peaks against a new baseline.

    The review's finding: `__exit__` cleared the thread and not the peaks, so if
    the container reclaimed swap between runs, `swap_growth_bytes` came out
    positive and the second run was refused for swap it never used -- a fabricated
    resource failure, which is worse than no guard.
    """
    executor = Executor()
    with executor:
        executor.execute(WAVE_PLAN, request=WAVE_REQUEST)
        if executor._peak_swap_bytes is None:
            pytest.skip("this platform exposes no cgroup swap counter")
        # Leave a high peak behind, as a run that really did swap would.
        executor._peak_swap_bytes = int(executor._peak_swap_bytes) + 1_000_000
        assert executor.resource_failure() is not None

    with executor:
        assert executor._peak_swap_bytes is not None
        assert executor.resource_failure() is None, (
            "the reopened executor inherited the previous run's peak"
        )
        assert executor._samples >= 1
        record = executor.execute(WAVE_PLAN, request=WAVE_REQUEST)
    assert record.status == "completed"


def test_the_sampling_thread_is_joined_before_the_context_exits() -> None:
    """Criterion 5. Deterministic, which is why this is a context manager.

    A run that left a thread behind would make the *next* run's baseline and peaks
    unattributable, and a leaked thread per run is how a long session runs out of
    them.
    """
    before = {thread.name for thread in threading.enumerate()}
    with Executor() as executor:
        executor.execute(WAVE_PLAN, request=WAVE_REQUEST)
        during = {thread.name for thread in threading.enumerate()}
        assert "runtime-memory" in during
    after = {thread.name for thread in threading.enumerate()}
    assert "runtime-memory" not in after, "the sampler outlived the context"
    assert after <= before | {"runtime-memory"} or after == before


def test_the_sampler_really_polls() -> None:
    """The thread does work, not just start and stop.

    `__enter__` and `__exit__` each take a reading, so a sampler that never ran
    would still report two samples. A short interval and a brief wait is what
    distinguishes a running thread from a started one.
    """
    with Executor(sample_interval_s=0.01) as executor:
        time.sleep(0.15)
        samples = executor._samples
    assert samples > 3, f"the sampler took {samples} reading(s); the thread is not polling"


def test_two_executors_in_one_process_do_not_interfere() -> None:
    """Criterion 5's second half, with their nodes interleaved.

    Each owns its own baseline, peaks and sampler; neither holds module state, and
    tripping one's guard does not trip the other's.

    **The honest limit**, stated because the first version of this docstring
    overclaimed: this runs the backend-free wave plan, so it establishes
    independence of *this class's* state and nothing about the backend. The
    executor does not configure a backend -- it cannot, `runtime` may not import
    `backends` -- and the reason two runs asking for different precisions are safe
    is that `configure_execution` sets all three on every call rather than
    inheriting. That guarantee belongs to the solver operation and is tested there.
    """
    with Executor() as first, Executor() as second:
        first_a = first.execute(WAVE_PLAN, request=WAVE_REQUEST)
        second_a = second.execute(WAVE_PLAN, request=WAVE_REQUEST)
        first_b = first.execute(WAVE_PLAN, request=WAVE_REQUEST)

        assert first_a.status == second_a.status == first_b.status == "completed"
        assert first_a.fingerprinted == second_a.fingerprinted == first_b.fingerprinted
        # Separate lifecycles: tripping one's guard does not trip the other's.
        baseline = dict(first._baseline or {})
        observed = first._peak_swap_bytes
        if observed is not None:
            first._baseline = {**baseline, "cgroup_swap_bytes": int(observed) - 1}
            assert first.resource_failure() is not None
            assert second.resource_failure() is None

    assert "runtime-memory" not in {thread.name for thread in threading.enumerate()}


def test_a_failed_run_still_releases_the_resource() -> None:
    """The `with` block's whole job: release on the way out, however it goes."""
    with pytest.raises(ValueError), Executor() as executor:
        executor.execute(("X_INVENTED",), request={})
    assert "runtime-memory" not in {thread.name for thread in threading.enumerate()}


# ---------------------------------------------------------------------------
# 6. A real multi-scale route, through the real solvers
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_a_ray_trace_crossing_to_the_wave_model_runs_end_to_end() -> None:
    """The route R12's graph discovers, executed: rays, coupled, measured.

    `S_RAY_OPTILAND -> C_RAY_TO_SCALAR -> M_PSF`. Every step is a landed
    implementation with its own physics tests, and the point here is only that the
    executor binds and carries them: it resolves three operations across two
    packages, hands each the arguments the request names, passes the coupler's
    *primary* return onward while dropping its diagnostics tuple element, and
    records what happened.

    Marked `slow` because it imports optiland and traces a real system. The
    executor's other criteria are all established above on the wave plan, so the
    default gate does not depend on this one.
    """
    record = execute(
        ("S_RAY_OPTILAND", "C_RAY_TO_SCALAR", "M_PSF"),
        request={
            "setup": singlet_ref(),
            "source": singlet_source(),
            "sampling": {"num_rings": 6, "reference_surface": "exit_pupil"},
            "execution": {"device": "cpu", "precision": "fp64"},
            "grid_shape": (32, 32),
            "sample_pitch_m": (2.0e-7, 2.0e-7),
            "normalization": "peak",
        },
        sources=[ROOT / "src" / "couplers" / "ray_to_scalar.py"],
    )
    assert record.status == "completed", [
        (node.operation_id, node.status, node.diagnostics) for node in record.nodes
    ]
    assert len(record.nodes) == 3
    # The coupler returns `(field, diagnostics)`; the executor passed element 0 on,
    # which is the only reason the measurement could run at all.
    assert record.nodes[2].operation_id == "M_PSF"
    # And the code fingerprint covers the source that was declared.
    assert "src/couplers/ray_to_scalar.py" in record.provenance["code_fingerprint"]["files"]


def test_every_declared_runtime_code_is_reachable() -> None:
    """The closed vocabulary, and a code nobody can trigger is a code nobody needs.

    `representations.contracts` states the rule this follows: "a code invented
    locally by each consumer is how the free-form provenance string this module
    replaced came back". These three are the runtime's own -- not contract codes,
    not `numerics` refusal codes -- so they are enumerated and every one has a path
    here.
    """
    reached: set[str] = set()

    # INCOMPLETE_REQUEST
    partial = {key: value for key, value in WAVE_REQUEST.items() if key != "normalization"}
    incomplete = execute(WAVE_PLAN, request=partial)
    reached.add(incomplete.nodes[-1].diagnostics.split(":")[0])

    # UNBINDABLE_SIGNATURE
    psf_module = sys.modules["measurements.psf"]
    original = psf_module.psf

    def unbindable(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("must not be called")

    psf_module.psf = unbindable  # type: ignore[assignment]
    try:
        reached.add(execute(("M_PSF",), request={}).nodes[0].diagnostics.split(":")[0])
    finally:
        psf_module.psf = original  # type: ignore[assignment]

    # RESOURCE_LIMIT
    with Executor() as executor:
        if executor._peak_swap_bytes is None:
            reached.add("RESOURCE_LIMIT")  # unreachable off a cgroup v2 container
        else:
            executor._baseline = {
                **(executor._baseline or {}),
                "cgroup_swap_bytes": int(executor._peak_swap_bytes) - 1,
            }
            failure = executor.resource_failure() or ""
            reached.add(failure.split(":")[0])

    assert reached == set(RUNTIME_CODES), sorted(reached)


def test_a_diagnostic_carrying_a_uuid_does_not_move_the_records_fingerprint() -> None:
    """B0-META-01's defect, closed where it can now recur.

    `fingerprinted` includes diagnostics, and the executor writes arbitrary backend
    exception text into them -- so a refusal message carrying a `uuid4` would
    rehash the record on every run and nothing could tell that from a real change.
    That is exactly what happened to B0-META-01. `stabilize_diagnostics` replaces
    the identifier with a placeholder, keeping the message's meaning and losing only
    the part that identified the run.
    """
    import uuid

    psf_module = sys.modules["measurements.psf"]
    original = psf_module.psf

    def raising_with_a_uuid(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError(f"backend gave up, correlation id {uuid.uuid4()}")

    stubbed = functools.wraps(original)(raising_with_a_uuid)
    psf_module.psf = stubbed  # type: ignore[assignment]
    try:
        first = execute(WAVE_PLAN, request=WAVE_REQUEST)
        second = execute(WAVE_PLAN, request=WAVE_REQUEST)
    finally:
        psf_module.psf = original  # type: ignore[assignment]

    assert first.status == "failed"
    assert "<uuid>" in first.nodes[-1].diagnostics
    assert "correlation id" in first.nodes[-1].diagnostics, "the message kept its meaning"
    assert first.fingerprinted == second.fingerprinted, (
        "two runs of the same failure fingerprinted differently, which is the "
        "B0-META-01 defect"
    )
    runtime.require_stable_payload(first.fingerprinted, where="a failed run's record")


def test_the_placement_pair_is_read_from_the_execution_request() -> None:
    """Where device and precision actually travel, which is not the top level.

    Every landed operation that takes a placement takes it as an `execution`
    mapping. The first version read top-level `request["device"]`, which nothing
    populates and which the executor also drops from the call -- so a record could
    claim a request that reached nothing.
    """
    from runtime.executor import _requested_placement

    assert _requested_placement({"execution": {"device": "cpu", "precision": "fp64"}}) == {
        "device": "cpu",
        "precision": "fp64",
    }
    assert _requested_placement({"device": "cuda"}) == {}, (
        "a top-level device is not what any operation reads, so it is not recorded "
        "as requested either"
    )
    assert _requested_placement({}) == {}
    # And on a real run the pair is empty, because the wave plan takes no execution.
    record = execute(WAVE_PLAN, request=WAVE_REQUEST)
    assert all(node.requested == {} for node in record.nodes)
    assert all(node.placement_disagreement == () for node in record.nodes)


def test_the_executor_is_the_only_class_in_the_module() -> None:
    """Criterion 6. `GraphExecutor` had nine companions; this has none.

    `scripts/class_budget.py` owns the count -- `runtime` is 3, and the third is
    this one on rule 3. What this adds is the *names*: a budget records what
    exists and only a test can record what was avoided.
    """
    import ast

    avoided = (
        "GraphExecutor",
        "ExecutorError",
        "ProcessModel",
        "SolverStateProtocol",
        "ExecutionCache",
        "InMemoryCache",
        "AdapterResolver",
        "CouplerResolver",
        "_LazyNames",
        "MemoryWatchdog",
        "MemoryWatchdogVerdict",
        "HostMemorySnapshot",
    )
    source = (ROOT / "src" / "runtime" / "executor.py").read_text(encoding="utf-8")
    defined = {
        node.name for node in ast.walk(ast.parse(source)) if isinstance(node, ast.ClassDef)
    }
    assert defined == {"Executor"}, sorted(defined)
    assert not defined & set(avoided)
    # The two resolver protocols existed because the old executor could not import
    # what it ran. `operations.resolve` replaces both, and this is where that is
    # recorded rather than inferred.
    assert "resolve" in source and "Resolver" not in source


def test_the_record_carries_no_verdict_about_the_physics() -> None:
    """The runtime records what happened. Whether it was right is not its question.

    A record with a `passed` field is a record someone will read as a scientific
    conclusion, and nothing in this package is entitled to one.
    """
    record = execute(WAVE_PLAN, request=WAVE_REQUEST)
    text = runtime.to_json(
        ExecutionRecord(
            route=record.route,
            request={"normalization": "peak"},
            nodes=record.nodes,
            provenance=record.provenance,
        )
    )
    for verdict in ("verdict", "reproduces", "passed", "correct", "converged", "verified"):
        assert verdict not in text, f"the serialized record contains {verdict!r}"


# ---------------------------------------------------------------------------
# 7. A plan of node instances: a repeated operation, and what came out
# ---------------------------------------------------------------------------
#
# The capability the flat request could not express. A plan is an ordered sequence
# of node instances, so one operation may appear twice with different arguments --
# and the measured failure before it could is the one these tests are written
# against: the six-node 4f plan bound f1 to *both* focal-plane transforms and an
# open pupil to *both* masks, and reported every node `completed`.

#: A four-node plan that repeats `O_COMPLEX_TRANSMISSION` at two amplitudes.
#:
#: The repeated operation is a **backend-free** one on purpose. The plan that
#: motivated per-node requests repeats `O_FOCAL_PLANE_TRANSFORM` at f1 and f2, and
#: writing that here imports Chromatix into this worker --
#: `tests/operations/test_descriptors.py` asserts `"chromatix" not in sys.modules`
#: and fails when pytest puts the two files on one worker. This module's docstring
#: already declares the backend-free plan as its limit, so the repetition is
#: expressed with an operation that keeps it. The f1/f2 case is covered where it
#: belongs, in `benchmarks/systems/b4f_ideal.py`, whose record carries both focal
#: lengths read off `node_requests` and whose magnification gate is what would fail
#: if they were shared.
#:
#: Two attenuations compose exactly: `A = 0.5 * second`, and the raw intensity is
#: `A^2` times the source's. A shared flat request would apply *one* of them twice,
#: giving `0.25` or `second^2` -- both far from the product, so this is arithmetic
#: rather than a tolerance.
def _attenuator_plan(second: float) -> tuple[Any, ...]:
    return (
        (
            "S_SOURCE_GAUSSIAN_BEAM",
            {
                "shape": (32, 32),
                "sample_pitch_m": (1.0e-6, 1.0e-6),
                "wavelength_m": 5.5e-7,
                "reference_surface": SURFACE,
                "waist_radius_m": 6.0e-6,
            },
        ),
        ("O_COMPLEX_TRANSMISSION", {"amplitude": 0.5}),
        ("O_COMPLEX_TRANSMISSION", {"amplitude": second}),
        ("M_PSF", {"normalization": "raw"}),
    )


def test_a_repeated_operation_binds_each_occurrence_independently() -> None:
    """The failure the per-node request form exists to end, as a physical check.

    Two elements at 0.5 and 0.25. The peak intensity is `(0.5 * 0.25)^2` of the
    source's, and an executor that bound one `amplitude` to both nodes would return
    `0.25^2` or `0.0625^2` instead -- which is what happened, silently, with every
    node reading `completed`.
    """
    with Executor() as executor:
        record = executor.execute(_attenuator_plan(0.25))
        measured = executor.result
    assert record.status == "completed", [
        (node.operation_id, node.status, node.diagnostics) for node in record.nodes
    ]
    assert [entry["amplitude"] for entry in record.node_requests[1:3]] == [0.5, 0.25]
    assert float(np.max(np.asarray(measured.intensity))) == pytest.approx(
        (0.5 * 0.25) ** 2, rel=1e-5
    )


def test_the_two_occurrences_are_not_the_same_system_and_the_record_says_so() -> None:
    """A fingerprint that ignored `node_requests` would call these one run.

    Both plans are the same route, the same shared request (empty) and the same
    node statuses. The *only* thing that distinguishes them is the per-node
    arguments, so leaving those out of `fingerprinted` would answer "nothing
    changed" about a physically different system.
    """
    with Executor() as executor:
        weak = executor.execute(_attenuator_plan(0.25))
        weak_peak = float(np.max(np.asarray(executor.result.intensity)))
        strong = executor.execute(_attenuator_plan(0.5))
        strong_peak = float(np.max(np.asarray(executor.result.intensity)))
    assert weak.route == strong.route
    assert weak_peak != strong_peak
    assert weak.fingerprinted != strong.fingerprinted


def test_a_nodes_own_request_does_not_fall_back_to_the_shared_one() -> None:
    """The rule that makes a forgotten argument loud rather than silently shared.

    `M_PSF` requires `normalization` and this node's request does not name it, while
    the shared request does. A merge would run it with the shared value; the refusal
    is the point -- an argument meant for one node reaching another is the failure
    mode, not a convenience.
    """
    record = execute((("M_PSF", {"field": _overflowing_field()}),), request=WAVE_REQUEST)
    assert record.status == "refused"
    assert record.nodes[0].diagnostics.startswith("INCOMPLETE_REQUEST")
    assert "normalization" in record.nodes[0].diagnostics


def test_a_bare_id_still_reads_the_shared_request() -> None:
    """The original form, unchanged, and mixed with the new one in one plan.

    A bare id is bound from `request` exactly as before, which is what keeps every
    caller that predates per-node requests running. Mixed with a pair, each node
    reads its own source -- and `node_requests` then records the effective request
    of every step, so nothing is left to be inferred about the bare one.
    """
    plan = ("S_SOURCE_GAUSSIAN_BEAM", ("M_PSF", {"normalization": "peak"}))
    record = execute(plan, request=WAVE_REQUEST)
    assert record.status == "completed"
    assert len(record.node_requests) == 2
    assert record.node_requests[0]["waist_radius_m"] == WAVE_REQUEST["waist_radius_m"]
    assert record.node_requests[1] == {"normalization": "peak"}


def test_the_result_is_what_the_last_node_produced_and_is_not_in_the_record() -> None:
    """Criterion: the physical result comes back on the public API, beside the record.

    `ExecutionRecord` is the serialized provenance model and a `PsfResult` cannot
    go in it, so the result is read off the executor. What this pins is that the
    two are consistent: a completed run has a result of the last node's type, and
    the record still carries no field holding it.
    """
    from measurements import PsfResult

    with Executor() as executor:
        record = executor.execute(WAVE_PLAN, request=WAVE_REQUEST)
        assert isinstance(executor.result, PsfResult)
        assert np.array_equal(
            np.asarray(executor.result.intensity), _physical(record)
        )
    assert not any("result" in name for name in record.__dataclass_fields__)


def test_a_run_that_did_not_complete_has_no_result() -> None:
    """A refusal must not leave a previous node's state readable as the answer.

    The plan's first node completes and the second refuses. `result` is `None`, not
    the field the source produced -- which would be a plausible-looking wrong
    answer for the plan that was asked for.
    """
    with Executor() as executor:
        record = executor.execute(("S_SOURCE_GAUSSIAN_BEAM", "M_PSF"), request={
            key: value for key, value in WAVE_REQUEST.items() if key != "normalization"
        })
        assert record.status == "refused"
        assert executor.result is None


def test_the_result_is_cleared_by_the_next_run_and_by_reopening() -> None:
    """It is a run's output, not something the executor keeps.

    A second `execute` that refuses must not hand back the first run's result, and
    neither may a reopened executor or a closed one. Every one of those is the
    "fabricated result" failure with an extra step in front of it.
    """
    executor = Executor()
    with executor:
        executor.execute(WAVE_PLAN, request=WAVE_REQUEST)
        assert executor.result is not None
        executor.execute(("M_PSF",), request={})
        assert executor.result is None
    with executor:
        assert executor.result is None
        executor.execute(WAVE_PLAN, request=WAVE_REQUEST)
        assert executor.result is not None
    assert executor.result is None, "a closed executor still held the run's output"


@pytest.mark.parametrize(
    "rejected",
    [(), (("M_PSF",),), ("X_INVENTED",)],
    ids=["empty", "malformed", "uncatalogued"],
)
def test_a_plan_that_execute_rejects_clears_the_previous_result(rejected: Any) -> None:
    """The raise paths clear it too, and that is not incidental.

    `execute` refuses an empty plan, a malformed step and an uncatalogued id by
    raising. A caller that catches one and then reads `result` would otherwise be
    handed the *previous* run's object as the answer for a plan that never ran --
    the same hazard as a pre-refusal node's state, one level up, and the reason the
    clear happens before validation rather than after it.
    """
    with Executor() as executor:
        executor.execute(WAVE_PLAN, request=WAVE_REQUEST)
        assert executor.result is not None
        with pytest.raises(ValueError):
            executor.execute(rejected, request=WAVE_REQUEST)
        assert executor.result is None


@pytest.mark.parametrize(
    ("plan", "expected"),
    [
        ((), "at least one operation"),
        ((("M_PSF",),), "plan step 0"),
        ((("M_PSF", 5),), "not a mapping"),
        (((1, {}),), "not an operation id"),
    ],
)
def test_a_malformed_plan_is_refused_by_step(plan: Any, expected: str) -> None:
    """`normalize_plan` names the step, because a plan is long enough that it must.

    Public for this reason: a caller -- or, later, whatever generates a plan -- can
    be told which step is wrong without running anything.
    """
    with pytest.raises(ValueError, match=expected):
        runtime.normalize_plan(plan)


def test_per_node_requests_survive_the_serialization_round_trip() -> None:
    """They are inputs, so a record that lost them would not describe its own run."""
    record = execute(
        ("S_SOURCE_GAUSSIAN_BEAM", ("M_PSF", {"normalization": "peak"})),
        request=WAVE_REQUEST,
    )
    serializable = ExecutionRecord(
        route=record.route,
        request={"normalization": "peak"},
        node_requests=({"waist_radius_m": 6.0e-6}, {"normalization": "peak"}),
        nodes=record.nodes,
        provenance={"code_fingerprint": {"files": {}}},
    )
    assert runtime.from_json(runtime.to_json(serializable)) == serializable
    assert json.loads(runtime.to_json(serializable))["node_requests"][1] == {
        "normalization": "peak"
    }


def test_a_per_node_request_that_does_not_align_with_the_route_is_refused() -> None:
    """Aligned by index, so a short list would attribute a request to the wrong node."""
    with pytest.raises(ValueError, match="aligned with `route` by index"):
        ExecutionRecord(
            route=("S_SOURCE_GAUSSIAN_BEAM", "M_PSF"),
            node_requests=({"normalization": "peak"},),
        )
