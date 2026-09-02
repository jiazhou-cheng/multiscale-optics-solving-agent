"""Execute a route and record what happened. One class, and the resource it owns.

CHE-200 (R13.2). `runtime.execute(plan, *, request)` runs a plan and returns an
`ExecutionRecord`. It never records whether the result was *right*: that is the
verification layer's question and this module has no field for an answer to it.

What a plan is, and what this module is not
-------------------------------------------
A plan is an **ordered sequence of node instances**: a step is an operation id, or
an `(operation_id, request)` pair giving that occurrence its own arguments. So a
plan may repeat an operation -- two focal-plane transforms at f1 and then f2 --
and the two occurrences are independent nodes rather than one entry in a shared
mapping. `Executor.execute` records what that fixed, measured on the plan that
made it necessary.

The executor is the generic mechanism; **choosing** the plan is not its job. A
plan may come from `planning.routes`, from a caller writing it down, or later from
an agent generating one, and this module cannot tell the difference and does not
try: it checks that every step is catalogued and runs them in order. That
separation is the point -- `planning.routes` still cannot enumerate a plan with a
repeated operation, and a plan that names one runs here anyway.

The one class in the tree justified by rule 3, and what it owns
--------------------------------------------------------------
`Executor` owns a **mutable resource lifecycle**, which is `AGENTS.md`'s rule 3 and
the only place in the new architecture it applies. The resource, named concretely
and singular:

**a memory sampling thread.** A daemon thread started on `__enter__` and joined on
`__exit__`, polling host and container memory every `sample_interval_s`. The
shared-server policy makes container swap growth a stop condition, and sampling
only at node boundaries would see nothing during one long operation.

**Released deterministically** on `__exit__`, which is the whole reason this is a
context manager rather than a function with module state: the thread is joined
before `__exit__` returns, so a second run cannot see it, and the observation
state is reset on the next `__enter__` rather than carried across. `runtime.execute`
opens and closes one for a single run, so a caller who wants no lifecycle does not
get one.

**What this class does *not* own, corrected.** An earlier version of this docstring
claimed it owned process-global backend state -- optiland's backend/device/precision
and chromatix's platform pin -- and that it "applies its own before every node".
That was false in a way worth recording rather than quietly deleting, because it
was also the justification checked into `scripts/class_budget.py`: this package
**cannot** touch backend configuration, since `check_dependencies` gives `runtime`
only `{planning, operations, representations}` and there is no `backends` import
here. The ordering safety is real and is owned where it belongs:
`backends.optiland.configure_execution` sets all three on **every call** and never
inherits what a previous call left behind, which its own docstring states. So two
executors are safe with respect to the backend because the solver operation makes
them safe, not because this class sequences anything.

Rule 3 still holds, on the sampler alone: a thread is a resource, its lifetime is
this object's, and releasing it deterministically is what a context manager is for.

Two executors in one process do not interfere: each owns its own sampler, baseline
and peaks, and neither holds module-level state.
`tests/integration/test_executor.py` runs their nodes interleaved -- on the
backend-free wave plan, which is the honest limit of that test: it establishes
independence of *this* class's state, and says so rather than implying it covers
the backend channel it cannot reach.

No result cache, and the measurement that decided it
----------------------------------------------------
Criterion 3 admits a cache only against a measured cost, so the cost was measured
before the question was answered. On a 256x256 two-node plan
(`gaussian_beam -> psf`): the physics is **2.2 ms**, and running it through the
first version of this executor took **11.0 ms** -- 391% overhead, which is not the
number the docstring originally guessed.

All of it was one thing: `environment_fingerprint()` at **7.5 ms**, because
`importlib.metadata.version` scans the filesystem once per package and there are
seven, called once per record. That is not a cost a *result* cache would have
fixed; it is a fact being recomputed that cannot change. So the executor reads the
environment fingerprint once when it opens, and the overhead becomes
`source_fingerprint` (0.02 ms with no sources declared) plus resolve, bind, sample
and record. Re-measured after the change: 3.1 ms.

What that leaves for a result cache to save is *re-running the physics*, and
skipping that is a correctness decision rather than a performance one: two calls
with equal arguments are equal only if nothing in the process-global backend state
moved between them, which is exactly what this class exists because it can. So
there is **no result cache, no cache key and no graph fingerprint**. The reference
executor's four fingerprints account for most of its 1145 lines and nine classes,
and `tests/unit/test_provenance_separation.py` asserts none of them is here.

Binding a call, and why the signature is read rather than the descriptor
-----------------------------------------------------------------------
A descriptor carries `requires` and `optional` as parameter *names* and `inputs` as
port *types*, but not which parameter the port is -- CHE-222 recorded that gap and
left it to this ticket. Rather than widen the schema, `_bind` reads the resolved
callable's own signature: the parameter whose annotation names a representation is
the port, and everything else is filled from that node's request by name. That is one
source of truth (the code) rather than two, and a request missing a required
argument is refused by name before anything runs.

Everything is passed by keyword. The two Optiland entry points take `setup` first
positionally, so a port-first positional call would bind the wrong argument; every
parameter in the catalog is `POSITIONAL_OR_KEYWORD` or `KEYWORD_ONLY`, so keyword
binding is universal and order-independent.
"""

from __future__ import annotations

import inspect
import os
import threading
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

from operations import CATALOG, OperationDescriptor, resolve
from runtime.records import (
    ExecutionRecord,
    NodeRecord,
    environment_fingerprint,
    record_provenance,
    stabilize_diagnostics,
)

__all__ = [
    "RUNTIME_CODES",
    "SAMPLE_INTERVAL_S",
    "Executor",
    "PlanNode",
    "execute",
    "memory_snapshot",
    "normalize_plan",
]

#: One step of a plan: an operation id, or an id paired with that step's own request.
#:
#: A type alias and not a class. An agent generating a plan writes
#: `["O_FOCAL_PLANE_TRANSFORM", {"focal_length_m": 0.02, "model": {...}}]`, which is
#: already JSON -- a `PlanNode` dataclass would add a construction step between the
#: generator and the executor and would buy no invariant that `normalize_plan` does
#: not check on the way in. `AGENTS.md`'s rule is that a class earns itself by a
#: shared invariant, a versioned public data model, a resource lifecycle, runtime
#: polymorphism or a plugin boundary, and a two-element pair is none of them.
PlanNode = str | tuple[str, Mapping[str, Any]]

#: The diagnostic codes this module can put on a node record, as a closed set.
#:
#: A code invented locally by each consumer is how the free-form provenance string
#: `representations.contracts` replaced came back, and its docstring says so. These
#: three are not `CONTRACT_CODES` -- a contract code is a representation refusing to
#: describe state, and none of these is that -- and not `REFUSAL_CODES`, which is
#: `numerics`' device/dtype vocabulary. They are the runtime's own, so they are
#: enumerated here and `tests/integration/test_executor.py` reaches every one.
#:
#: * `INCOMPLETE_REQUEST` -- the request does not name an argument the operation
#:   requires. A refusal: nothing broke, the caller has not said enough.
#: * `UNBINDABLE_SIGNATURE` -- the implementation cannot be called by keyword from
#:   a request at all. A failure, and latent: nothing in the catalog is like this.
#: * `RESOURCE_LIMIT` -- the shared-server swap-growth stop condition tripped.
#:
#: A code a *node's own* boundary raised passes through unchanged and is not in
#: this set -- `NON_FINITE`, `MEASURE_UNDECLARED` and the rest belong to the layer
#: that refused, and relabelling one here would hide which layer said no.
RUNTIME_CODES: tuple[str, ...] = (
    "INCOMPLETE_REQUEST",
    "RESOURCE_LIMIT",
    "UNBINDABLE_SIGNATURE",
)

#: How often the sampler reads memory while a node runs.
#:
#: A quarter second, the reference implementation's own interval. Fast enough that
#: a node long enough to swap is sampled many times, slow enough that the thread
#: costs nothing measurable against an operation.
SAMPLE_INTERVAL_S = 0.25

#: Container swap growth tolerated before the run stops. **Zero.**
#:
#: `AGENTS.md`'s shared-GPU policy: "Never use swap as working memory; workload
#: swap growth is a stop condition." Not a slowdown to tolerate and not a
#: threshold to tune -- growth at all, above the baseline taken when the executor
#: opened, terminates the run and is reported as a resource failure.
SWAP_GROWTH_TOLERANCE_BYTES = 0

_PROC_MEMINFO = Path("/proc/meminfo")
_PROC_SELF_STATM = Path("/proc/self/statm")
_CGROUP_SWAP_CURRENT = Path("/sys/fs/cgroup/memory.swap.current")
_CGROUP_MEMORY_CURRENT = Path("/sys/fs/cgroup/memory.current")


def _read_int(path: Path) -> int | None:
    try:
        return int(path.read_text().split()[0])
    except (OSError, ValueError, IndexError):
        return None


def _meminfo_bytes(key: str) -> int | None:
    try:
        for line in _PROC_MEMINFO.read_text().splitlines():
            if line.startswith(f"{key}:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def _rss_bytes() -> int:
    """This process's resident set size, in bytes.

    From `/proc/self/statm` rather than `VmRSS` in `/proc/self/status`, because
    statm is a single short line and this is sampled on a polling thread. The
    reference implementation's reason, unchanged.
    """
    try:
        fields = _PROC_SELF_STATM.read_text().split()
        return int(fields[1]) * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError):  # pragma: no cover - /proc absent
        return 0


def memory_snapshot() -> dict[str, int | None]:
    """One instant of host and container memory state, as a plain mapping.

    A dict and not a `HostMemorySnapshot` class: there is no invariant across the
    fields, nothing subclasses it, and CHE-200 budgets exactly one class. The
    reference implementation spent one here and eleven more on a performance
    harness that belongs in `scripts/`.

    `cgroup_swap_bytes` is the **guarded** quantity. `host_swap_used_bytes` is
    recorded beside it precisely so a report can show that the host number is
    non-zero and moving for unrelated reasons -- other tenants on a shared server
    -- rather than leaving a reader to wonder why it was ignored.

    Every field is `None` when the platform does not expose it, so this works
    outside a cgroup v2 container instead of raising there.

    `mem_total_bytes`, `mem_available_bytes` and `cgroup_memory_bytes` are recorded
    and **not guarded on**, which is a deliberate asymmetry rather than an
    oversight: they appear in the record's `baseline` so a resource report has the
    context to be read, and adding a host-memory-reserve trip condition on them
    would be a second guard with no measured need. The reference implementation had
    one; CHE-200 asks for the swap condition, and a guard nobody has measured a need
    for is the speculative field the ticket's risk section is about.
    """
    swap_total = _meminfo_bytes("SwapTotal")
    swap_free = _meminfo_bytes("SwapFree")
    return {
        "rss_bytes": _rss_bytes(),
        "mem_total_bytes": _meminfo_bytes("MemTotal"),
        "mem_available_bytes": _meminfo_bytes("MemAvailable"),
        "cgroup_memory_bytes": _read_int(_CGROUP_MEMORY_CURRENT),
        "cgroup_swap_bytes": _read_int(_CGROUP_SWAP_CURRENT),
        "host_swap_total_bytes": swap_total,
        "host_swap_used_bytes": (
            None if swap_total is None or swap_free is None else swap_total - swap_free
        ),
    }


def normalize_plan(
    plan: Iterable[PlanNode], *, request: Mapping[str, Any] = MappingProxyType({})
) -> tuple[tuple[str, Mapping[str, Any]], ...]:
    """A plan as `((operation_id, node_request), ...)`, refusing a malformed step.

    The one place the two step forms become one, so nothing downstream has to know
    which was written. A bare id takes the shared `request`; a `(id, mapping)` pair
    takes its own mapping and **not** a merge with the shared one -- see
    `Executor.execute` for why a fallback would reintroduce exactly the silent
    parameter sharing the pair form exists to end.

    Public because a caller that wants to check or print a plan before running it
    should not have to reproduce this, and because a plan is the representation an
    agent generates: a generator can hand its output through here and be told which
    step is malformed, by index, without a run.

    An id is **validated** as a string rather than coerced with `str()`, which the
    pre-plan version did: coercion turned a typo of the wrong type into an
    operation id nobody wrote, and the catalog check then reported the coerced
    spelling. A pair's request is copied, so a caller's dict cannot change under a
    run already started; a bare id's request is the shared mapping **by reference**,
    which is the behaviour this form has always had and is safe because `execute`
    is synchronous.

    Raises:
        ValueError: the plan is empty, or a step is neither an id nor an
            `(id, mapping)` pair. Named by index, because a plan is long enough
            that "a step is malformed" is not a diagnostic.
    """
    steps: list[tuple[str, Mapping[str, Any]]] = []
    for index, item in enumerate(plan):
        if isinstance(item, str):
            steps.append((item, request))
            continue
        try:
            operation_id, node_request = item
        except (TypeError, ValueError):
            raise ValueError(
                f"plan step {index} is {item!r}. A step is either an operation id or a "
                "two-element (operation_id, request) pair; the pair form is what gives one "
                "occurrence of a repeated operation its own arguments."
            ) from None
        if not isinstance(operation_id, str):
            raise ValueError(
                f"plan step {index} names {operation_id!r}, which is not an operation id"
            )
        try:
            steps.append((operation_id, dict(node_request)))
        except (TypeError, ValueError):
            raise ValueError(
                f"plan step {index} ({operation_id}) carries {node_request!r} as its "
                "request, which is not a mapping of argument names to values"
            ) from None
    if not steps:
        raise ValueError("a plan names at least one operation")
    return tuple(steps)


#: The annotations that name a **representation port**.
#:
#: The same two `tests/operations/test_catalog_signatures.py` derives against, and
#: the duplication is deliberate rather than shared: that module is a *gate* over
#: the catalog and this one is a *binder* at runtime, and importing one into the
#: other would make the gate unable to fail independently. What keeps them in step
#: is that the gate runs over every catalogued operation on every suite run --
#: `test_an_implementation_the_executor_cannot_bind_fails_loudly` additionally
#: drives *this* derivation over all fourteen.
_PORT_ANNOTATIONS: tuple[str, ...] = ("RayBundle", "ScalarField")


def _port_and_parameters(
    implementation: Any,
) -> tuple[str | None, tuple[str, ...], tuple[str, ...]]:
    """The port's parameter name, the required names, and the optional names.

    Read off the signature rather than the descriptor -- see the module docstring.
    A parameter whose annotation names a representation is the port; anything else
    is required or optional depending on whether it has a default.

    **A variadic parameter is refused rather than classified.** `*args` has no name
    a request could fill and `**kwargs` would swallow anything, so treating either
    as "required" would make the executor demand an argument called `args`. That
    is not hypothetical: it is what happened the first time a test wrapped an
    implementation in a `*a, **k` stub, and the run refused with
    "`request` is missing ['a', 'k']" -- a diagnostic that named the stub's
    parameters and told nobody anything. Nothing in the catalog is variadic, so
    this is a loud failure on the day one is rather than a wrong demand.

    Raises:
        TypeError: the implementation takes `*args` or `**kwargs`.
    """
    port: str | None = None
    required: list[str] = []
    optional: list[str] = []
    for name, parameter in inspect.signature(implementation).parameters.items():
        if parameter.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            raise TypeError(
                f"{getattr(implementation, '__name__', implementation)!r} takes a variadic "
                f"parameter ({parameter}), which a request cannot fill by name. An operation "
                "the executor can bind names every argument it needs."
            )
        if parameter.annotation is inspect.Parameter.empty:
            raise TypeError(
                f"{getattr(implementation, '__name__', implementation)!r} has an unannotated "
                f"parameter ({name!r}), so it cannot be told from a representation port. An "
                "unannotated port would be filed as a request parameter, the upstream "
                "node's output would be dropped, and the node would still record "
                "`completed` -- silently wrong rather than loudly."
            )
        annotation = str(parameter.annotation)
        if annotation in _PORT_ANNOTATIONS:
            port = name
        elif any(known in annotation for known in _PORT_ANNOTATIONS):
            raise TypeError(
                f"{getattr(implementation, '__name__', implementation)!r} annotates {name!r} as "
                f"{annotation!r}, which mentions a representation but is not exactly one. "
                "Filing it as a request parameter would hide a port; widen "
                "`_PORT_ANNOTATIONS` deliberately, or give the operation an exact port."
            )
        elif parameter.default is not inspect.Parameter.empty:
            optional.append(name)
        else:
            required.append(name)
    return port, tuple(required), tuple(optional)


@dataclass
class Executor:
    """One run's resource lifecycle, and the loop that fills a record.

    Minimality rule 3, and the only class in the new architecture on it. See the
    module docstring for the resource -- process-global backend state and a memory
    sampling thread -- and for when it is released.

    Not frozen and not a value: it is mutable by definition, which is the point of
    the rule. Use it as a context manager; `execute` outside one raises rather than
    silently running without a guard.
    """

    #: The repository root, for the source fingerprint. An argument so a test can
    #: point it somewhere else, and so nothing here reads a global.
    root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2])
    #: Which sources the run's code fingerprint covers. Passed rather than read off
    #: `sys.modules`: see `records.record_provenance` on which trade is whose.
    sources: tuple[Path, ...] = ()
    #: How often the sampler polls. An argument because
    #: `tests/integration/test_executor.py::test_the_sampler_really_polls` needs a
    #: short one to observe more than the two readings `__enter__` and `__exit__`
    #: take -- without it, "the thread samples" would be untested and the thread
    #: would be doing nothing observable.
    sample_interval_s: float = SAMPLE_INTERVAL_S

    #: The environment fingerprint, read once when this executor opens. Measured:
    #: computing it per record cost 7.5 ms against 2.2 ms of physics for a two-node
    #: plan -- `importlib.metadata.version` scans the filesystem once per package.
    #: Installed versions cannot change inside a process, so reading it once is
    #: correct as well as cheaper. This is the only thing here that is remembered
    #: rather than recomputed, and it is an immutable process fact rather than a
    #: result. See the module docstring on why there is no result cache.
    _environment: dict[str, Any] | None = field(default=None, init=False, repr=False)
    _baseline: dict[str, int | None] | None = field(default=None, init=False, repr=False)
    _peak_rss_bytes: int = field(default=0, init=False, repr=False)
    _peak_swap_bytes: int | None = field(default=None, init=False, repr=False)
    _samples: int = field(default=0, init=False, repr=False)
    #: What the last **completed** run produced, held so `result` can hand it back.
    #: Cleared on `__enter__` and at the start of every `execute`, and set only once
    #: the whole route has completed -- see `result` for why that is not "the last
    #: node that ran".
    _produced: Any = field(default=None, init=False, repr=False)
    #: Guards the peak/sample updates, which run on both the sampler thread and the
    #: caller's. `+=` and `max(...)` are read-modify-write, so two threads can lose
    #: a sample; the window is tiny and the consequence is a slightly low peak, but
    #: this is the safety mechanism and a lost *swap* peak is a guard that did not
    #: trip.
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _thread: threading.Thread | None = field(default=None, init=False, repr=False)
    _stop: threading.Event = field(default_factory=threading.Event, init=False, repr=False)

    # -- lifecycle ----------------------------------------------------------

    def __enter__(self) -> Executor:
        """Take the baseline and start the sampler. The resource is acquired here.

        **The observation state is reset**, not carried over. A second `with` on the
        same instance would otherwise keep the first run's peak swap against a fresh
        baseline, and if the container had reclaimed swap in between,
        `swap_growth_bytes` would come out positive and refuse a run for swap it
        never used. That is a fabricated resource failure, which is worse than no
        guard.
        """
        if self._thread is not None:
            raise RuntimeError(
                "this Executor is already open. One executor owns one run's resources; "
                "open a second Executor rather than re-entering this one."
            )
        self._peak_rss_bytes = 0
        self._peak_swap_bytes = None
        self._samples = 0
        self._produced = None
        self._environment = environment_fingerprint()
        self._baseline = memory_snapshot()
        self._absorb(self._baseline)
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._sample_until_stopped, daemon=True, name="runtime-memory"
        )
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        """Stop and join the sampler, take a final reading, release the resource.

        Deterministic: the thread is joined before this returns, so a second run in
        the same process cannot see it. The join has a timeout because a hung
        sampler must not hang the caller -- and a daemon thread would not keep the
        interpreter alive in any case.

        The final reading updates the peaks, which a record taken inside the `with`
        has already read. That is deliberate rather than an oversight: a record
        describes the run it belongs to, and a reading taken after the last
        operation finished belongs to no run. It is here so `swap_growth_bytes` is
        answerable *after* the block, for a caller inspecting the executor rather
        than the record.
        """
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=5.0)
        self._absorb(memory_snapshot())
        # The run's output is released here too. It is not a resource in the sense
        # rule 3 means -- nothing has to be closed -- but a grid-sized array kept
        # alive by a closed executor is the same shape of leak, and `result` is
        # scoped to a run rather than to the object. Read it inside the block.
        self._produced = None

    def _sample_until_stopped(self) -> None:  # pragma: no cover - thread body
        while not self._stop.wait(self.sample_interval_s):
            self._absorb(memory_snapshot())

    def _absorb(self, snapshot: Mapping[str, int | None]) -> None:
        with self._lock:
            self._samples += 1
            rss = snapshot.get("rss_bytes") or 0
            self._peak_rss_bytes = max(self._peak_rss_bytes, int(rss))
            swap = snapshot.get("cgroup_swap_bytes")
            if swap is not None:
                current = int(swap)
                self._peak_swap_bytes = (
                    current
                    if self._peak_swap_bytes is None
                    else max(self._peak_swap_bytes, current)
                )

    # -- the guard ----------------------------------------------------------

    @property
    def swap_growth_bytes(self) -> int | None:
        """Container swap growth since the baseline, or `None` if unreadable.

        The **container's** swap, not the host's. On a shared server the host
        number moves for other tenants' reasons, and guarding on it would stop
        runs that used no swap at all.
        """
        if self._baseline is None:
            return None
        base = self._baseline.get("cgroup_swap_bytes")
        if base is None or self._peak_swap_bytes is None:
            return None
        return max(0, self._peak_swap_bytes - int(base))

    def resource_failure(self) -> str | None:
        """The diagnostic for a tripped guard, or `None` while it holds.

        One condition today: container swap growth above the baseline. Not a
        threshold to tune -- `AGENTS.md` makes any growth a stop condition -- and
        not a slowdown to tolerate.
        """
        growth = self.swap_growth_bytes
        if growth is not None and growth > SWAP_GROWTH_TOLERANCE_BYTES:
            return (
                f"RESOURCE_LIMIT: container swap grew by {growth} bytes above the baseline "
                f"taken when this run opened (tolerance {SWAP_GROWTH_TOLERANCE_BYTES}). The "
                "shared-server policy makes swap growth a stop condition, not a slowdown to "
                "tolerate: the run was terminated here rather than allowed to continue "
                "swapping. Host swap is recorded beside the container's and is deliberately "
                "not guarded, because on a shared server it moves for other tenants."
            )
        return None

    def resource_record(self) -> dict[str, Any]:
        """What the guard observed. An observation, never an input."""
        return {
            "samples": self._samples,
            "peak_rss_bytes": self._peak_rss_bytes,
            "peak_cgroup_swap_bytes": self._peak_swap_bytes,
            "swap_growth_bytes": self.swap_growth_bytes,
            "baseline": dict(self._baseline or {}),
        }

    # -- what the run produced ----------------------------------------------

    @property
    def result(self) -> Any:
        """The physical object the last `execute` finished with, or `None`.

        The **output** of a run, and deliberately not part of its record.
        `ExecutionRecord` is the serialized provenance model: a `ScalarField` or a
        `PsfResult` cannot go in it, and a record that carried a summary of one
        instead would be a second, lossy copy of the physics with nothing to keep
        it honest. So the record says what ran and this says what came out, and the
        two are read together.

        This is not a stored result to be handed back instead of running: it is
        cleared at the start of every `execute` and on `__enter__`, nothing
        consults it to decide whether to run, and two `execute` calls with equal
        arguments both execute their physics -- which is the property
        `tests/integration/test_executor.py::test_two_runs_of_one_plan_both_execute_the_physics`
        already pins, and the reason the module docstring's argument against
        keeping results still holds.

        `None` after a run that refused or failed, and that is a decision rather
        than a consequence: the state of the node *before* a refusal exists and is
        deliberately not handed back, because a source's field returned for a plan
        that was meant to measure a PSF is a plausible-looking wrong answer. Also
        `None` after `execute` *raised* -- on a malformed plan or an uncatalogued
        id -- and after `__exit__`. So this is the plan's answer or nothing, and
        every other outcome reads the same way.
        """
        return self._produced

    # -- the run ------------------------------------------------------------

    def execute(
        self, plan: Iterable[PlanNode], *, request: Mapping[str, Any] = MappingProxyType({})
    ) -> ExecutionRecord:
        """Run a plan and return the record. Never a fabricated result.

        Each operation is resolved, bound **from its own node's request** by
        parameter name, called, and recorded. The physical state produced by one
        node is the port argument of the next; the first node's port -- if it has
        one -- comes from that node's request under the port's own parameter name,
        which is how a plan starting mid-graph is given its input. What the last
        node produced is `self.result`, read beside this record rather than in it.

        A plan is an ordered sequence of *node instances*, and a step is either an
        operation id or `(operation_id, node_request)`. A bare id is bound from the
        shared `request`, which is the original behaviour and is why every caller
        that predates per-node requests still runs unchanged. A pair is bound from
        its own mapping **alone**, with no fallback to the shared one:

            plan = (
                ("S_SOURCE_PLANE_WAVE", {"shape": ..., "wavelength_m": ...}),
                ("O_FOCAL_PLANE_TRANSFORM", {"focal_length_m": f1, "model": {...}}),
                ("O_COMPLEX_TRANSMISSION", {"amplitude": stop}),
                ("O_FOCAL_PLANE_TRANSFORM", {"focal_length_m": f2, "model": {...}}),
            )

        **A repeated operation is a first-class case, and this is what it needed.**
        One flat mapping keyed by parameter name cannot say which occurrence of
        `O_FOCAL_PLANE_TRANSFORM` an entry called `focal_length_m` belongs to.
        Measured before the pair form existed: that exact six-node 4f plan bound f1
        to *both* legs and `amplitude=1.0` to *both* masks, and reported every node
        `completed` -- a unit relay with an open pupil, recorded as a magnifying
        relay with a stop, with nothing anywhere saying f2 had never been read. A
        silently different optical system is the failure mode `AGENTS.md` puts
        first, so the pair form does not fall back: a node with its own request that
        omits a required argument is refused by name, which is loud, rather than
        inheriting a value meant for another node, which is not.

        `planning.routes` cannot enumerate a plan like that one -- no operation may
        repeat in a route -- and this method deliberately does not require that it
        could. A plan is *given* to the executor; where it came from, an
        enumeration or a caller writing it down, is the planner's question and not
        this one's. What is checked here is only that every step is in the catalog.

        A node that refuses or raises stops the run and is recorded with its own
        diagnostics. Nothing downstream of it runs and nothing is invented for it:
        the record is shorter than the route, `ExecutionRecord.status` reads
        `failed` for that reason alone, and `result` is `None`.

        The resource guard is evaluated at every boundary, including before the
        first node. When it trips, the operation that was about to run is recorded
        as `failed` with the resource diagnostic -- "we did not run this" -- rather
        than the one that just finished, which did run.

        Raises:
            RuntimeError: called outside `with`. The guard would not be running.
            ValueError: the plan is empty, malformed, or names an operation the
                catalog does not have.
        """
        if self._thread is None:
            raise RuntimeError(
                "Executor.execute must be called inside `with`: outside it the memory "
                "sampler is not running, so the swap-growth stop condition the "
                "shared-server policy requires would not be evaluated."
            )
        # Cleared **before** the plan is validated, not after. Every refusal below
        # raises, and a caller that catches one and reads `result` would otherwise
        # get the previous run's object as the answer for a plan that never ran --
        # the same hazard as handing back a pre-refusal node's state, one level up.
        self._produced = None
        # Materialized once: `plan` may be a generator, and it is read twice below
        # -- for the steps and for whether any step carried its own request.
        given = tuple(plan)
        steps = normalize_plan(given, request=request)
        route = tuple(operation_id for operation_id, _ in steps)
        catalogued = {descriptor.operation_id: descriptor for descriptor in CATALOG}
        unknown = [step for step in route if step not in catalogued]
        if unknown:
            raise ValueError(
                f"the plan names {unknown}, which the catalog does not have. Every step of "
                "a plan is a catalogued operation id."
            )
        # `()` when nothing gave a node its own request, so a record of a
        # single-request run carries no per-node copy of it to disagree with.
        per_node = (
            tuple(node_request for _, node_request in steps)
            if any(not isinstance(item, str) for item in given)
            else ()
        )

        started = time.monotonic()
        nodes: list[NodeRecord] = []
        state: Any = None
        for index, (operation_id, node_request) in enumerate(steps):
            failure = self.resource_failure()
            if failure is not None:
                nodes.append(
                    NodeRecord(operation_id=operation_id, status="failed", diagnostics=failure)
                )
                break
            node, state, stop = self._run_one(
                catalogued[operation_id],
                request=node_request,
                incoming=state,
                first=index == 0,
            )
            nodes.append(node)
            if stop:
                break

        # The guard once more, **after** the last node. Without this a
        # single-operation plan -- the shape of every heavy solver run -- could
        # never trip it: the boundary check runs before each node, so growth during
        # the only node was observed by the sampler and acted on by nobody, and the
        # record read `completed`. AGENTS.md makes swap growth a stop condition, and
        # a run that tripped it must not read as a clean success.
        #
        # Reported as an observation on the `resources` block rather than as an
        # extra failed node: every operation in the route *did* run, so fabricating
        # a failed one for a completed operation would be inventing an outcome. The
        # caller sees `resource_failure` beside the growth that caused it.
        trailing_failure = self.resource_failure()

        # `result` is the *plan's* answer, so it is set only when the plan ran to
        # its end. Assigning it per node -- which the first version of this did --
        # leaves the state of the node before a refusal readable as though it were
        # what was asked for: a source's field handed back for a plan that was
        # meant to measure a PSF, with `record.status` reading `refused` beside it.
        # A caller who reads the result first and the status second would get a
        # plausible-looking wrong answer, which is the one outcome this module is
        # written to make impossible.
        # `tests/integration/test_executor.py::test_a_run_that_did_not_complete_has_no_result`
        # is the test that caught it.
        self._produced = (
            state
            if len(nodes) == len(route) and all(node.status == "completed" for node in nodes)
            else None
        )

        return ExecutionRecord(
            route=route,
            request=dict(request),
            node_requests=per_node,
            nodes=tuple(nodes),
            provenance={
                **record_provenance(
                    sources=self.sources, root=self.root, environment=self._environment
                ),
                # Both under keys `records.VOLATILE_KEYS` projects out: they
                # identify a run, not a computation, and a fingerprint that moved
                # because a run was slower would claim a change nobody made.
                "runtime_seconds": time.monotonic() - started,
                "resources": {**self.resource_record(), "resource_failure": trailing_failure},
            },
        )

    def _run_one(
        self,
        descriptor: OperationDescriptor,
        *,
        request: Mapping[str, Any],
        incoming: Any,
        first: bool,
    ) -> tuple[NodeRecord, Any, bool]:
        """One operation: resolve, bind, call, record. Returns `(node, state, stop)`."""
        requested = _requested_placement(request)
        try:
            implementation = resolve(descriptor.operation_id)
        except Exception as exc:
            # `Exception` and not a named tuple of types, symmetrically with the
            # call path below. Importing a backend can raise `OSError` for a
            # missing `libcudart` or `RuntimeError` for an absent driver, and both
            # are realistic here; a narrower clause let those escape `execute()`
            # with no record at all, which is the one outcome this method exists to
            # prevent.
            return (
                NodeRecord(
                    operation_id=descriptor.operation_id,
                    status="failed",
                    diagnostics=f"{type(exc).__name__}: {exc}",
                    requested=requested,
                ),
                None,
                True,
            )

        try:
            port, required, optional = _port_and_parameters(implementation)
        except (TypeError, ValueError) as exc:
            return (
                NodeRecord(
                    operation_id=descriptor.operation_id,
                    status="failed",
                    diagnostics=f"UNBINDABLE_SIGNATURE: {exc}",
                    requested=requested,
                ),
                None,
                True,
            )
        arguments: dict[str, Any] = {}
        if port is not None:
            if first:
                # `incoming` is always None on the first node, so the condition is
                # on `first` alone: a route's opening port comes from the request.
                if port not in request:
                    return (
                        self._refused(
                            descriptor,
                            requested,
                            f"the route starts at {descriptor.operation_id}, which consumes a "
                            f"representation, and `request` has no {port!r} to start it from",
                        ),
                        None,
                        True,
                    )
                arguments[port] = request[port]
            else:
                arguments[port] = incoming
        missing = [name for name in required if name not in request]
        if missing:
            return (
                self._refused(
                    descriptor,
                    requested,
                    f"`request` is missing {missing} for {descriptor.operation_id}, whose "
                    f"required arguments are {list(required)}. A runtime does not choose a "
                    "physical parameter for a caller -- and `psf`'s normalization is the "
                    "case that makes that a rule rather than a preference.",
                ),
                None,
                True,
            )
        arguments.update({name: request[name] for name in required})
        arguments.update({name: request[name] for name in optional if name in request})

        try:
            returned = implementation(**arguments)
        except Exception as exc:
            # Everything a backend or a boundary can raise, captured as
            # diagnostics. A `refused` status means the exception **carried a
            # project code** -- `ContractError`, or the ValueError-with-`code` that
            # `numerics.refusal` builds -- because this project declining to compute
            # something it cannot describe is a fact about the problem rather than
            # about the run.
            #
            # The honest limit, which is narrower than "every refusal is recorded as
            # one": a good deal of the project's own validation raises a **plain**
            # `ValueError` -- `sources/_grid.py`, `sources/plane_wave.py`,
            # `problems/ray_trace.py` -- and those arrive here as `failed`. That
            # under-reports refusals rather than over-reporting them, which is the
            # safe direction, and the fix is for those layers to carry a code rather
            # than for this one to guess from a message.
            code = getattr(exc, "code", None)
            status = "refused" if code is not None else "failed"
            diagnostics = f"{code}: {exc}" if code is not None else f"{type(exc).__name__}: {exc}"
            return (
                NodeRecord(
                    operation_id=descriptor.operation_id,
                    status=status,  # type: ignore[arg-type]
                    # Sanitized, because `fingerprinted` includes diagnostics and a
                    # backend message carrying a uuid would rehash the record every
                    # run. B0-META-01 is that defect, and this is the point where it
                    # can recur now that arbitrary exception text reaches a record.
                    diagnostics=stabilize_diagnostics(diagnostics),
                    requested=requested,
                ),
                None,
                True,
            )

        # An operation with an auxiliary return hands back a tuple; the primary
        # semantic value is element 0. Read off the descriptor rather than from the
        # shape of what came back, so a single value that happens to be a tuple is
        # not silently unpacked.
        state = returned[0] if descriptor.returns_auxiliary else returned
        return (
            NodeRecord(
                operation_id=descriptor.operation_id,
                status="completed",
                requested=requested,
                observed=_observed_placement(state, requested),
            ),
            state,
            False,
        )

    @staticmethod
    def _refused(
        descriptor: OperationDescriptor, requested: Mapping[str, str], why: str
    ) -> NodeRecord:
        return NodeRecord(
            operation_id=descriptor.operation_id,
            status="refused",
            diagnostics=f"INCOMPLETE_REQUEST: {why}",
            requested=dict(requested),
        )


def _requested_placement(request: Mapping[str, Any]) -> dict[str, str]:
    """The device and precision the request asked for, from where they travel.

    Read out of `request["execution"]` and not from top-level `request["device"]`,
    which is where the first version looked and where nothing puts them: every
    landed operation that takes a placement takes it as an `execution` mapping
    (`S_RAY_OPTILAND`, `O_RAY_TRACE`), so top-level keys were both never
    populated *and* silently dropped from the call -- neither port, required nor
    optional -- while still being recorded as "requested". A record claiming a
    request that reached nothing is worse than one claiming nothing.

    Empty when the request names no placement, which is most of the catalog: a
    coupler or an analytic source runs in whatever namespace it was handed, and
    `NodeRecord.placement_disagreement` is empty for an unobserved pair rather than
    reporting agreement.
    """
    execution = request.get("execution")
    if not isinstance(execution, Mapping):
        return {}
    return {
        key: str(execution[key]) for key in ("device", "precision") if key in execution
    }


def _observed_placement(state: Any, requested: Mapping[str, str]) -> dict[str, str]:
    """Where the result actually is, when the result can say.

    A requested device is not evidence of an actual one -- a process-global JAX
    platform pin produces a successful host run while the caller asked for CUDA,
    with no error raised -- so this reads the returned representation's own array
    state rather than echoing the request. Empty when the result carries no state
    to read, which is honest: an unobserved placement must not look like an
    agreeing one.
    """
    if not requested:
        return {}
    array_state = getattr(state, "state", None)
    if array_state is None:
        return {}
    observed: dict[str, str] = {}
    device = getattr(array_state, "device", None)
    if device is not None and "device" in requested:
        observed["device"] = str(device)
    dtype = getattr(array_state, "dtype", None)
    precision = getattr(dtype, "precision", None)
    if precision is not None and "precision" in requested:
        observed["precision"] = str(precision)
    return observed


def execute(
    plan: Iterable[PlanNode],
    *,
    request: Mapping[str, Any] = MappingProxyType({}),
    root: Path | None = None,
    sources: Iterable[Path] = (),
) -> ExecutionRecord:
    """Run one plan under a fresh `Executor`, and close it. The public entry point.

    The signature CHE-165 names. A function rather than a method a caller has to
    reach for, so the common case -- one run, one record -- does not require
    knowing that a resource lifecycle exists; and a context manager underneath, so
    that when it matters the release is deterministic.

    **Returns the record and not the physical result**, because the executor it
    opened is closed by the time this returns and the result belongs to that run.
    A caller who needs what the plan produced opens the context itself, which is
    also the cheaper shape for many runs -- the environment fingerprint is read
    once per `__enter__`, at 7.5 ms:

        with Executor() as executor:
            record = executor.execute(plan)
            field = executor.result

    Two entry points would be the alternative and it is worse: `execute` is called
    in 37 places that want a record, and a second function differing only in its
    return type is the kind of near-duplicate this package has none of.
    """
    # `root=None` means "the Executor's own default", which is the same repository
    # root -- restating the expression here would be a second place to change it.
    executor = (
        Executor(sources=tuple(sources))
        if root is None
        else Executor(root=root, sources=tuple(sources))
    )
    with executor:
        return executor.execute(plan, request=request)
