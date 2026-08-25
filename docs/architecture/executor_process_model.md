# The executor's process model, and what it does with global solver state

CHE-113 (M3.1). A decision record, because the choice is not local: it
constrains caching, artifact serialization and what a reproducible fingerprint
means, and whichever answer is chosen has to appear in every run record.

## The problem

A single process running a ray node and then a wave node inherits every piece of
process-global state either solver owns.

* **Optiland.** `set_backend`, `set_precision` and `set_device` mutate
  process-wide module state and are documented as not thread-safe. The ordering
  hazard is real: `configure_optiland_execution` must precede
  `build_optiland_system` or the trace dies inside an Optiland geometry class.
* **JAX.** The platform pin and `jax_enable_x64` are process-global. `x64` is
  pinned `False` everywhere in this repository, and a process that flipped it
  would change every recorded number in that process.
* **The observable symptom.** M0.1's flaky exactness gate was this, and the fix
  was to make the state deterministic rather than to hope.

## The decision

**Single process, with a strict state-transition protocol, recorded per node.**

The protocol is not new. CHE-61 already made the Optiland adapter set all three
globals explicitly on *every* run, at the defaults included, precisely so a
previous run's choices cannot leak into the next. That is the transition
protocol, and it lives where it belongs — in the adapter that owns the state.

What the executor adds is **observation**. `SolverStateProtocol` records, per
node, what was requested and what the adapter reported applying, and both go
into `ExecutionRecord.provenance["solver_state"]`. An adapter that reports
nothing leaves the applied state `None`, and that is information rather than an
assumption that it matched: a node whose applied state is unknown cannot be
cache-matched, and the cache key reflects that.

`jax_enable_x64` is **refused**, not honoured. A node asking for it gets
`RefusalKind.INVALID_CONFIGURATION` naming the declaration, before any solver is
imported.

## What was rejected, and why it is not settled

**Process-per-node.** `src/studies/metalens/controller.py` does this and for
good reasons, all of which still apply: CUDA memory is returned at process exit,
there is no allocator fragmentation across differently-shaped runs, and a dying
candidate cannot poison the next.

It is rejected *for now* on one concrete blocker: it needs artifact
serialization across the process boundary. `ArtifactRecord` deliberately keeps
arrays in solver-owned storage — the record is a reference, and the live object
travels in `ExecutionRecord.artifacts`, which does not cross a fork. Making
process-per-node work means deciding how a `ComplexField` or a `RayBundle`
crosses that boundary, which is a real design question about the artifact layer
and not a flag on the executor.

So `ProcessModel.PROCESS_PER_NODE` is declared and **raises**. Running
in-process while reporting process-per-node would make every record's
`process_model` field a lie, and that field is part of what makes a fingerprint
comparable.

## What this costs, stated

* **CUDA memory is not reclaimed between nodes.** A GPU graph holds every
  node's allocations for the life of the run. The memory watchdog will trip on
  a graph that would have fitted under process-per-node, and that is a real
  limitation rather than a safety margin.
* **A node cannot use a different array backend from its neighbour without the
  adapter re-establishing it.** Both current adapters do; a future one that does
  not would be a silent correctness hazard, and the recorded `solver_state` is
  what would make it visible.
* **The 10-run reproducibility evidence is CPU-only.** The GPU path is not
  covered by it.

## When to revisit

When any of these becomes true:

1. a graph needs two nodes on CUDA whose combined allocation does not fit;
2. an adapter arrives that does not re-establish its global state per run;
3. artifact serialization lands for another reason — a distributed runner, a
   cache that survives the process, a resumable run — at which point
   process-per-node is nearly free.

## Related

* `src/core/executor.py` — the implementation and the same decision in short.
* `tests/test_executor.py::test_asking_for_an_unimplemented_process_model_raises`
* `tests/test_executor_integration.py::test_the_chain_is_reproducible_across_consecutive_runs`
* `src/studies/metalens/controller.py` — the working process-per-candidate
  orchestration this decision declines to copy yet.
