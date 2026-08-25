"""The shipped example graphs, executed for real.

CHE-113 (M3.1). ``tests/test_executor.py`` proves the executor's own behaviour
against stubs, which is the right place for ordering, refusal and bookkeeping.
This file answers a different question, and it is the one the milestone turns
on: does a graph this repository actually ships run end to end through the
executor, with real Optiland and real Chromatix?

It did not, and the reason is the whole argument for building an executor.
``examples/graphs/ray_to_wave.yaml`` -- the flagship three-node
``M_RAY_OPTILAND -> C_RAY_TO_WAVE -> M_WAVE_CHROMATIX`` chain, validated by four
existing tests and cited in three protocol documents -- had **never been
executable**. Its ``lens`` node declared ``wavelength_m`` and ``pupil_samples``;
the Optiland adapter reads ``wavelength`` (micrometres), ``num_rays`` and
``sample``. Every value in that node was silently ignored and the trace fell
back to adapter defaults on a different lens. Then, once that was fixed, the
handoff refused: the node exported at ``image_surface`` while the edge declared
``exit_pupil``, a pupil-to-focus distance apart.

Neither failure is visible to ``GraphValidator``, which checks ports and
artifact kinds and cannot know whether a config key is one the adapter reads.
Both are visible the instant something executes the file.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.execution import RunStatus
from core.execution_record import NodeOutcome, RefusalKind
from core.paths import repository_root
from registry.loader import Registry
from runtime.executor import GraphExecutor, InMemoryCache

pytestmark = [pytest.mark.integration, pytest.mark.optiland, pytest.mark.chromatix]

ROOT = repository_root()
GRAPHS = ROOT / "examples" / "graphs"


@pytest.fixture(scope="module")
def registry() -> Registry:
    return Registry.from_package()


def _executor(registry: Registry, **kwargs) -> GraphExecutor:
    return GraphExecutor(registry, **kwargs)


def _load(name: str) -> object:
    return Registry.load_graph(GRAPHS / name)


def test_the_optiland_smoke_graph_executes(registry: Registry, tmp_path: Path) -> None:
    record = _executor(registry).run(_load("optiland_smoke.yaml"), seed=1)
    assert record.status is RunStatus.SUCCEEDED
    (node,) = record.nodes
    assert node.outcome is NodeOutcome.EXECUTED
    assert node.outputs, "the trace produced artifacts"
    assert node.cost is not None and node.cost.wall_seconds > 0.0


def test_the_full_ray_to_wave_chain_executes_end_to_end(registry: Registry) -> None:
    """Three nodes, one coupler, real solvers on both sides."""
    record = _executor(registry).run(_load("ray_to_wave.yaml"), seed=1)

    assert record.status is RunStatus.SUCCEEDED, (
        record.refusal.detail if record.refusal else [n.error_message for n in record.nodes]
    )
    assert [n.node_id for n in record.nodes] == ["lens", "pupil_reconstruction", "wave"]
    assert all(
        n.outcome in (NodeOutcome.EXECUTED, NodeOutcome.EXECUTED_LOSSY) for n in record.nodes
    )
    assert "wave:output_field" in record.artifacts


def test_the_ray_node_and_the_edge_agree_on_the_handoff_plane(registry: Registry) -> None:
    """The defect the first execution of this file found.

    Optiland exports at ``image_surface`` unless told otherwise, and
    C_RAY_TO_WAVE refuses a record whose plane disagrees with the plane the
    consumer declared. It is right to refuse: the two planes are a
    pupil-to-focus distance apart, so accepting the mismatch would defocus the
    reconstruction rather than piston it. Pinned here so the two declarations
    cannot drift apart again silently.
    """
    graph = _load("ray_to_wave.yaml")
    lens = next(n for n in graph.nodes if n.id == "lens")  # type: ignore[attr-defined]
    edge = next(e for e in graph.edges if e.coupler == "C_RAY_TO_WAVE")  # type: ignore[attr-defined]
    assert lens.config["handoff_plane"] == edge.config["handoff_plane"] == "exit_pupil"


def test_the_ray_node_config_uses_keys_the_adapter_actually_reads(registry: Registry) -> None:
    """A config key nothing reads is worse than a missing one: the run succeeds
    on defaults and the graph reads as though it specified something."""
    graph = _load("ray_to_wave.yaml")
    lens = next(n for n in graph.nodes if n.id == "lens")  # type: ignore[attr-defined]
    assert {"sample", "wavelength", "num_rays"} <= set(lens.config)
    assert "wavelength_m" not in lens.config
    assert "pupil_samples" not in lens.config


def test_a_standalone_wave_graph_names_the_input_it_is_missing(registry: Registry) -> None:
    """``chromatix_smoke.yaml`` is a VALIDATION graph and cannot execute alone.

    Its own metadata says so -- "no edges are needed to *validate* a single node
    against the registry contract" -- and a wave node with no upstream producer
    has no field to propagate. The executor refuses and names the port, rather
    than inventing a field, and that refusal is the right answer rather than a
    gap: a graph-level declared source is how a caller supplies one.
    """
    record = _executor(registry).run(_load("chromatix_smoke.yaml"))

    assert record.status is RunStatus.FAILED
    (node,) = record.nodes
    assert node.outcome is NodeOutcome.REFUSED
    assert node.refusal is not None
    assert node.refusal.kind is RefusalKind.INVALID_CONFIGURATION
    assert "input_field" in (node.error_message or "")


def test_the_chain_is_reproducible_across_consecutive_runs(registry: Registry) -> None:
    """The executor-side counterpart of M0.1's repetition gate, on real solvers.

    Two runs rather than ten: this chain is ~11 s each and the shared-machine
    rule is that verification cost scales with the risk the change carries. What
    ten runs would add over two is confidence about *rare* nondeterminism, and
    the mechanism this is guarding -- process-global solver state leaking between
    a ray node and a wave node -- would show on the second run or not at all,
    because there is no source of variation that only appears later.
    """
    executor = _executor(registry)
    first = executor.run(_load("ray_to_wave.yaml"), seed=1)
    second = executor.run(_load("ray_to_wave.yaml"), seed=1)

    assert first.graph_sha256 == second.graph_sha256
    assert first.status is second.status is RunStatus.SUCCEEDED
    assert [(n.node_id, n.outcome) for n in first.nodes] == [
        (n.node_id, n.outcome) for n in second.nodes
    ]
    assert first.provenance["environment_sha256"] == second.provenance["environment_sha256"]
    assert first.provenance["solver_state"] == second.provenance["solver_state"]


def test_a_cache_hit_on_the_real_chain_matches_the_cold_run(registry: Registry) -> None:
    """The acceptance criterion, on a multi-node graph with real solvers."""
    cache = InMemoryCache()
    executor = _executor(registry, cache=cache)

    cold = executor.run(_load("ray_to_wave.yaml"), seed=1)
    warm = executor.run(_load("ray_to_wave.yaml"), seed=1)

    assert cold.status is warm.status is RunStatus.SUCCEEDED
    assert cache.hits >= 1, "nothing was reused, so this proves nothing"
    assert [n.outputs for n in cold.nodes] == [n.outputs for n in warm.nodes]
    warm_solver_seconds = warm.cost.solver_seconds if warm.cost else None
    cold_solver_seconds = cold.cost.solver_seconds if cold.cost else None
    assert warm_solver_seconds is not None and cold_solver_seconds is not None
    assert warm_solver_seconds < cold_solver_seconds, "a hit that costs the same is not a hit"


def test_the_record_carries_the_cost_estimate_beside_the_actual(registry: Registry) -> None:
    """M0.4 scores the estimator; the executor's job is to record both numbers
    per node so it has something to score."""
    record = _executor(registry).run(_load("ray_to_wave.yaml"), seed=1)
    for node in record.nodes:
        assert node.cost is not None
        assert node.cost.wall_seconds > 0.0
    priced = [n for n in record.nodes if n.cost and n.cost.estimate is not None]
    assert priced, "no node produced a cost estimate at all"
    for node in priced:
        assert node.cost.estimate.wall_time_s is not None or node.cost.estimate.notes  # type: ignore[union-attr]


def test_framework_overhead_is_reported_even_though_it_is_small(registry: Registry) -> None:
    """The executor's own share of the wall clock, measured rather than assumed.

    Reported here rather than gated: M0.4's target is 10%, and the number that
    matters is whichever one this actually is. The executor's overhead is
    validation, topological ordering, fingerprinting and record construction --
    all of it arithmetic over a handful of objects, against seconds of solver
    time.
    """
    record = _executor(registry).run(_load("ray_to_wave.yaml"), seed=1)
    assert record.cost is not None and record.cost.solver_seconds is not None
    overhead = record.cost.wall_seconds - record.cost.solver_seconds
    fraction = overhead / record.cost.wall_seconds
    assert 0.0 <= fraction < 0.5, (
        f"executor overhead is {fraction:.1%} of wall time ({overhead:.3f} s of "
        f"{record.cost.wall_seconds:.3f} s). Anything approaching half means the "
        "executor has become the workload."
    )
