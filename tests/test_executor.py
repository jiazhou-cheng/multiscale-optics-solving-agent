"""The executor runs a graph and records what happened. Nothing more.

CHE-113 (M3.1). Every test here uses stub adapters and couplers, and that is
deliberate rather than cheap: what the executor owns is ordering, plumbing,
refusal, bookkeeping and resource policy, and a real solver in the loop would
make a failure here ambiguous between the executor and the physics. The
executor's behaviour against real adapters is CHE-115's substrate proof, which
is a different question and needs different evidence.

The line these tests defend hardest is the one the architecture turns on: the
executor records facts, and the verifier says what they mean. A ``passed`` field
here would move the scientific decision into the thing that ran the code.
"""

from __future__ import annotations

import ast
import re
from typing import ClassVar

import pytest

from core.artifacts import ArtifactRecord
from core.execution import CostEstimate, RunStatus
from core.execution_record import ExecutionRecord, NodeOutcome, RefusalKind
from runtime.executor import (
    EXECUTOR_VERSION,
    ExecutorError,
    GraphExecutor,
    InMemoryCache,
    ProcessModel,
    graph_fingerprint,
    topological_order,
)
from core.graph import Severity, ValidationIssue, ValidationReport
from core.paths import repository_root
from core.specs import (
    ApproximationClass,
    ArtifactKind,
    CouplerSpec,
    DerivativeMode,
    DerivativeSpec,
    Device,
    Framework,
    GraphSpec,
    ModelSpec,
    PortSpec,
)
from solvers.base import ModelRunRequest, ModelRunResult

ROOT = repository_root()


# --------------------------------------------------------------------------- #
# A two-node graph over stub components
# --------------------------------------------------------------------------- #


def _artifact(artifact_id: str, kind: ArtifactKind, **overrides) -> ArtifactRecord:
    payload = {
        "id": artifact_id,
        "kind": kind,
        "uri": f"memory://{artifact_id}",
        "dtype": "float64",
        "device": Device.CPU,
        "framework": Framework.NUMPY,
    }
    payload.update(overrides)
    return ArtifactRecord(**payload)


SOURCE_MODEL = ModelSpec(
    id="M_RAY_OPTILAND",
    version="1.0.0",
    description="stub ray model",
    framework=Framework.NUMPY,
    approximation=ApproximationClass.GEOMETRIC_OPTICS,
    outputs=[PortSpec(name="rays", artifact=ArtifactKind.RAY_BUNDLE)],
    derivative=DerivativeSpec(mode=DerivativeMode.NONE, verified=False),
)

SINK_MODEL = ModelSpec(
    id="M_WAVE_CHROMATIX",
    version="1.0.0",
    description="stub wave model",
    framework=Framework.JAX,
    approximation=ApproximationClass.SCALAR_WAVE,
    inputs=[PortSpec(name="input_field", artifact=ArtifactKind.COMPLEX_FIELD)],
    outputs=[PortSpec(name="output_field", artifact=ArtifactKind.COMPLEX_FIELD)],
    derivative=DerivativeSpec(mode=DerivativeMode.NATIVE_AUTODIFF, verified=False),
)

BRIDGE = CouplerSpec(
    id="C_RAY_TO_WAVE",
    version="1.0.0",
    description="stub coupler",
    framework=Framework.NUMPY,
    source=PortSpec(name="source", artifact=ArtifactKind.RAY_BUNDLE),
    target=PortSpec(name="target", artifact=ArtifactKind.COMPLEX_FIELD),
    derivative=DerivativeSpec(mode=DerivativeMode.NONE, verified=False),
)


class _Registry:
    models: ClassVar[dict] = {SOURCE_MODEL.id: SOURCE_MODEL, SINK_MODEL.id: SINK_MODEL}
    couplers: ClassVar[dict] = {BRIDGE.id: BRIDGE}


class _Adapter:
    """A stub model adapter. Records every request it was handed."""

    def __init__(self, model: ModelSpec, port: str, kind: ArtifactKind, **behaviour):
        self._spec = model
        self.port = port
        self.kind = kind
        self.requests: list[ModelRunRequest] = []
        self.raises: BaseException | None = behaviour.get("raises")
        self.fails: bool = behaviour.get("fails", False)
        self.estimate_seconds: float | None = behaviour.get("estimate_seconds", 0.01)
        self.actual_dtype: str = behaviour.get("actual_dtype", "float64")
        self.actual_device: Device = behaviour.get("actual_device", Device.CPU)
        self.diagnostics: dict = behaviour.get("diagnostics", {})
        self.calls = 0

    @property
    def spec(self) -> ModelSpec:
        return self._spec

    def estimate(self, request: ModelRunRequest) -> CostEstimate:
        return CostEstimate(wall_time_s=self.estimate_seconds, confidence="stub")

    def validate_request(self, request: ModelRunRequest) -> ValidationReport:
        return ValidationReport(
            issues=[ValidationIssue(severity=Severity.INFO, code="OK", message="ok")]
        )

    def run(self, request: ModelRunRequest) -> ModelRunResult:
        self.calls += 1
        self.requests.append(request)
        if self.raises is not None:
            raise self.raises
        if self.fails:
            return ModelRunResult(
                status=RunStatus.FAILED,
                error_type="UnsupportedCapabilityError",
                error_message="the stub declined",
                diagnostics={"code": "UNSUPPORTED_CAPABILITY", "stage": "capability"},
            )
        return ModelRunResult(
            status=RunStatus.SUCCEEDED,
            outputs={
                self.port: _artifact(
                    f"{request.node_id}-{self.port}",
                    self.kind,
                    dtype=self.actual_dtype,
                    device=self.actual_device,
                )
            },
            diagnostics=self.diagnostics,
        )


class _Coupler:
    def __init__(self, **behaviour):
        self._spec = BRIDGE
        self.refuses: bool = behaviour.get("refuses", False)
        self.raises: BaseException | None = behaviour.get("raises")
        self.calls = 0
        self.configs: list[dict] = []

    @property
    def spec(self) -> CouplerSpec:
        return self._spec

    def estimate(self, request) -> CostEstimate:  # type: ignore[no-untyped-def]
        return CostEstimate(wall_time_s=0.001, confidence="stub")

    def validate_request(self, request) -> ValidationReport:  # type: ignore[no-untyped-def]
        return ValidationReport(issues=[])

    def transform(self, request):  # type: ignore[no-untyped-def]
        from couplers.base import CouplerRunResult

        self.calls += 1
        self.configs.append(dict(request.config))
        if self.raises is not None:
            raise self.raises
        if self.refuses or "handoff_plane" not in request.config:
            return CouplerRunResult(
                status=RunStatus.FAILED,
                error_type="ContractError",
                error_message=(
                    "the handoff plane is not declared; the coupler will not default it"
                ),
                diagnostics={
                    "code": "OPL_REFERENCE_UNVERIFIED",
                    "declaration": "config.handoff_plane",
                    "remedy": "declare handoff_plane and handoff_plane_z_m on the edge",
                },
            )
        return CouplerRunResult(
            status=RunStatus.SUCCEEDED,
            target=_artifact("bridged-field", ArtifactKind.COMPLEX_FIELD, dtype="complex64"),
            diagnostics={"code": "HANDOFF_DECLARED"},
        )


def _graph(**overrides) -> GraphSpec:
    payload = {
        "task_id": "stub-chain",
        "nodes": [
            {"id": "lens", "model": "M_RAY_OPTILAND", "config": {"num_rays": 8}},
            {"id": "wave", "model": "M_WAVE_CHROMATIX", "config": {"z_m": 1e-3}},
        ],
        "edges": [
            {
                "id": "bridge",
                "coupler": "C_RAY_TO_WAVE",
                "source": {"node": "lens", "port": "rays"},
                "target": {"node": "wave", "port": "input_field"},
                "config": {"handoff_plane": "exit_pupil", "handoff_plane_z_m": 1e-4},
            }
        ],
    }
    payload.update(overrides)
    return GraphSpec(**payload)


@pytest.fixture
def parts():
    source = _Adapter(SOURCE_MODEL, "rays", ArtifactKind.RAY_BUNDLE)
    sink = _Adapter(SINK_MODEL, "output_field", ArtifactKind.COMPLEX_FIELD)
    coupler = _Coupler()
    return source, sink, coupler


def _executor(parts, **overrides) -> GraphExecutor:
    source, sink, coupler = parts
    adapters = {"M_RAY_OPTILAND": source, "M_WAVE_CHROMATIX": sink}
    kwargs = {
        "registry": _Registry(),
        "watchdog_interval_s": 0.0,
        "adapter_resolver": lambda model_id: adapters[model_id],
        "coupler_resolver": lambda coupler_id: coupler,
    }
    kwargs.update(overrides)
    return GraphExecutor(**kwargs)


# --------------------------------------------------------------------------- #
# The boundary
# --------------------------------------------------------------------------- #


def test_the_execution_record_carries_no_verdict() -> None:
    """The executor records facts; the verifier says what they mean.

    A ``passed`` field here would move the scientific decision into the thing
    that ran the code, which is the arrangement where a run grades itself.
    """
    banned = {"passed", "pass", "success", "score", "met", "tolerance", "metric"}
    assert not (set(ExecutionRecord.model_fields) & banned)
    from core.execution_record import NodeRecord

    assert not (set(NodeRecord.model_fields) & banned)


def test_the_executor_module_knows_nothing_about_tolerances() -> None:
    """No tolerance, oracle or verdict *identifier* anywhere in the code.

    Checked over identifiers and string literals rather than over the raw text,
    so that the module can explain in prose why it has none of them -- which is
    the more useful thing for a reader than silence.
    """
    tree = ast.parse((ROOT / "src/runtime/executor.py").read_text(encoding="utf-8"))
    banned = {"tolerance", "tolerances", "oracle", "verify", "met", "passed"}
    offenders: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id.lower() in banned:
            offenders.append(f"name {node.id} at line {node.lineno}")
        elif isinstance(node, ast.Attribute) and node.attr.lower() in banned:
            offenders.append(f"attribute .{node.attr} at line {node.lineno}")
        elif isinstance(node, ast.arg) and node.arg.lower() in banned:
            offenders.append(f"argument {node.arg} at line {node.lineno}")
    assert not offenders, (
        "runtime/executor.py touches " + ", ".join(offenders) + ". Deciding what a number "
        "means is verification/verifier.py's job, and the split is the point."
    )


def test_the_executor_does_not_import_the_verifier() -> None:
    """One direction only. The verifier reads a record; the executor must not
    know what will be done with one."""
    tree = ast.parse((ROOT / "src/runtime/executor.py").read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
        elif isinstance(node, ast.Import):
            imported.extend(a.name for a in node.names)
    assert not [m for m in imported if m.startswith("verification")]


# --------------------------------------------------------------------------- #
# Ordering
# --------------------------------------------------------------------------- #


def test_nodes_run_in_topological_order() -> None:
    assert topological_order(_graph()) == ("lens", "wave")


def test_a_cycle_is_refused_rather_than_ordered_arbitrarily() -> None:
    spec = _graph(
        edges=[
            {
                "id": "forward",
                "coupler": "C_RAY_TO_WAVE",
                "source": {"node": "lens", "port": "rays"},
                "target": {"node": "wave", "port": "input_field"},
            },
            {
                "id": "back",
                "coupler": "C_RAY_TO_WAVE",
                "source": {"node": "wave", "port": "output_field"},
                "target": {"node": "lens", "port": "rays"},
            },
        ]
    )
    with pytest.raises(ExecutorError, match="cycle"):
        topological_order(spec)


# --------------------------------------------------------------------------- #
# Validation before any solver
# --------------------------------------------------------------------------- #


def test_an_invalid_graph_is_refused_without_invoking_a_solver(parts) -> None:
    source, sink, coupler = parts
    spec = _graph(
        nodes=[{"id": "lens", "model": "M_RAY_OPTILAND"}, {"id": "wave", "model": "M_UNKNOWN"}]
    )
    record = _executor(parts).run(spec)

    assert record.status is RunStatus.FAILED
    assert record.refusal is not None
    assert record.refusal.kind is RefusalKind.INVALID_CONFIGURATION
    assert record.nodes == []
    assert source.calls == sink.calls == coupler.calls == 0, (
        "an invalid graph must be refused before anything is executed"
    )


def test_a_streaming_node_is_refused_with_the_consequence_stated(parts) -> None:
    """demo3 cannot run through the executor, and that is a decision rather than
    an omission: running a 60M-ray workload un-chunked is worse than refusing."""
    spec = _graph(
        nodes=[
            {"id": "lens", "model": "M_RAY_OPTILAND", "config": {"streaming": True}},
            {"id": "wave", "model": "M_WAVE_CHROMATIX"},
        ]
    )
    record = _executor(parts).run(spec)
    assert record.refusal is not None
    assert record.refusal.kind is RefusalKind.UNSUPPORTED_CAPABILITY
    assert "streaming" in record.refusal.declaration
    assert record.refusal.remedy and "probe" in record.refusal.remedy
    assert parts[0].calls == 0


def test_a_request_to_flip_jax_x64_is_refused_rather_than_honoured(parts) -> None:
    """It is process-global and pinned False everywhere; flipping it would change
    every recorded number in the process."""
    spec = _graph(
        nodes=[
            {"id": "lens", "model": "M_RAY_OPTILAND"},
            {"id": "wave", "model": "M_WAVE_CHROMATIX", "config": {"jax_enable_x64": True}},
        ]
    )
    record = _executor(parts).run(spec)
    assert record.refusal is not None
    assert record.refusal.kind is RefusalKind.INVALID_CONFIGURATION
    assert "jax_enable_x64" in record.refusal.declaration


def test_requiring_verified_gradients_that_do_not_exist_is_refused(parts) -> None:
    record = _executor(parts).run(_graph(require_verified_gradients=True))
    assert record.refusal is not None
    assert record.refusal.kind is RefusalKind.UNVERIFIED_DERIVATIVE
    assert parts[0].calls == 0


# --------------------------------------------------------------------------- #
# The happy path
# --------------------------------------------------------------------------- #


def test_a_valid_graph_executes_end_to_end_and_records_every_node(parts) -> None:
    record = _executor(parts).run(_graph(), seed=7)

    assert record.status is RunStatus.SUCCEEDED
    assert [n.node_id for n in record.nodes] == ["lens", "bridge", "wave"]
    assert all(n.outcome is NodeOutcome.EXECUTED for n in record.nodes)
    assert record.seeds == [7]
    assert record.cost is not None and record.cost.wall_seconds > 0.0
    assert all(n.cost is not None and n.cost.estimate is not None for n in record.nodes)
    assert record.provenance["process_model"] == ProcessModel.IN_PROCESS.value
    assert record.provenance["executor_version"] == EXECUTOR_VERSION


def test_the_coupler_output_becomes_the_downstream_node_input(parts) -> None:
    _source, sink, _coupler = parts
    _executor(parts).run(_graph())
    (sink_request,) = sink.requests
    assert set(sink_request.inputs) == {"input_field"}
    assert sink_request.inputs["input_field"].id == "bridged-field"


def test_the_seed_is_threaded_into_every_node(parts) -> None:
    source, sink, _ = parts
    _executor(parts).run(_graph(), seed=1234)
    assert source.requests[0].config["seed"] == 1234
    assert sink.requests[0].config["seed"] == 1234


def test_a_node_that_already_declares_a_seed_keeps_it(parts) -> None:
    """The graph is more specific than the run. Overwriting a declared seed
    would make a node's own reproducibility depend on how it was invoked."""
    source, _, _ = parts
    spec = _graph(
        nodes=[
            {"id": "lens", "model": "M_RAY_OPTILAND", "config": {"seed": 99}},
            {"id": "wave", "model": "M_WAVE_CHROMATIX"},
        ]
    )
    _executor(parts).run(spec, seed=1234)
    assert source.requests[0].config["seed"] == 99


def test_two_runs_at_one_seed_produce_the_same_graph_fingerprint(parts) -> None:
    spec = _graph()
    first = _executor(parts).run(spec, seed=5)
    second = _executor(parts).run(spec, seed=5)
    assert first.graph_sha256 == second.graph_sha256 == graph_fingerprint(spec)
    assert first.run_id != second.run_id, "the run id is volatile and must not be the fingerprint"


def test_the_record_carries_what_the_verifier_needs(parts) -> None:
    """The M0.5.4 amendment's requirement, as a checklist."""
    record = _executor(parts).run(_graph(), seed=3, instance_id="inst-1")
    assert record.instance_id == "inst-1"
    assert record.observed_parameters["lens.num_rays"] == 8
    assert record.observed_parameters["bridge.handoff_plane"] == "exit_pupil"
    wave = record.node("wave")
    assert wave is not None and wave.device_precision is not None
    assert wave.device_precision.actual_device == "cpu"
    assert record.provenance["environment_sha256"]
    assert record.provenance["solver_state"]


# --------------------------------------------------------------------------- #
# Refusal, not inference
# --------------------------------------------------------------------------- #


def test_a_missing_handoff_declaration_is_refused_with_the_coupler_code(parts) -> None:
    """The couplers already refuse an undeclared handoff plane with
    OPL_REFERENCE_UNVERIFIED. The executor must not paper over that with a
    default: a graph missing a declaration is an invalid graph."""
    _source, sink, _coupler = parts
    spec = _graph(
        edges=[
            {
                "id": "bridge",
                "coupler": "C_RAY_TO_WAVE",
                "source": {"node": "lens", "port": "rays"},
                "target": {"node": "wave", "port": "input_field"},
                "config": {},
            }
        ]
    )
    record = _executor(parts).run(spec)

    bridge = record.node("bridge")
    assert bridge is not None
    assert bridge.outcome is NodeOutcome.REFUSED
    assert bridge.refusal is not None
    assert bridge.refusal.kind is RefusalKind.MISSING_EDGE_DECLARATION
    assert "OPL_REFERENCE_UNVERIFIED" in bridge.contract_codes
    assert bridge.refusal.remedy and "handoff_plane" in bridge.refusal.remedy
    assert sink.calls == 0, "the downstream node must not run on a refused handoff"


def test_an_upstream_failure_preserves_the_results_already_computed(parts) -> None:
    """M6's repair loop reads the partial record; a run that discarded what it
    had would make repair impossible."""
    _source, _sink, coupler = parts
    coupler.refuses = True
    record = _executor(parts).run(_graph())

    assert record.status is RunStatus.PARTIAL
    lens = record.node("lens")
    assert lens is not None and lens.outcome is NodeOutcome.EXECUTED
    assert lens.outputs == ["lens:rays"]
    assert "lens:rays" in record.artifacts, "the upstream artifact survives the failure"
    wave = record.node("wave")
    assert wave is not None and wave.outcome is NodeOutcome.NOT_REACHED


def test_a_node_that_raises_produces_no_partially_populated_artifact(parts) -> None:
    source, sink, coupler = parts
    source.raises = RuntimeError("the trace died inside a geometry class")
    record = _executor(parts).run(_graph())

    lens = record.node("lens")
    assert lens is not None
    assert lens.outcome is NodeOutcome.RAISED
    assert lens.error_type == "RuntimeError"
    assert lens.outputs == []
    assert record.artifacts == {}
    assert coupler.calls == 0 and sink.calls == 0


def test_a_node_that_declines_is_a_refusal_not_a_crash(parts) -> None:
    source, _sink, _coupler = parts
    source.fails = True
    record = _executor(parts).run(_graph())

    lens = record.node("lens")
    assert lens is not None
    assert lens.outcome is NodeOutcome.REFUSED
    assert lens.refusal is not None
    assert lens.refusal.kind is RefusalKind.UNSUPPORTED_CAPABILITY
    assert "UNSUPPORTED_CAPABILITY" in lens.contract_codes


# --------------------------------------------------------------------------- #
# Device and precision, read off the arrays
# --------------------------------------------------------------------------- #


def test_a_downcast_is_recorded_as_lossy_rather_than_as_the_request(parts) -> None:
    """Requested is not evidence of actual. Chromatix casts to complex64
    unconditionally, so a node that asked for FP64 and reported FP64 because
    that is what it asked for would have recorded a fiction."""
    _source, sink, _coupler = parts
    sink.actual_dtype = "complex64"
    spec = _graph(
        nodes=[
            {"id": "lens", "model": "M_RAY_OPTILAND"},
            {"id": "wave", "model": "M_WAVE_CHROMATIX", "config": {"dtype": "complex128"}},
        ]
    )
    record = _executor(parts).run(spec)

    wave = record.node("wave")
    assert wave is not None
    assert wave.outcome is NodeOutcome.EXECUTED_LOSSY
    assert wave.device_precision is not None
    assert wave.device_precision.requested_dtype == "complex128"
    assert wave.device_precision.actual_dtype == "complex64"
    assert not wave.device_precision.honoured
    assert wave.device_precision.measured_loss_relative is None
    assert "not measured by the executor" in (
        wave.device_precision.measured_loss_basis or ""
    ), "an unmeasured loss must say so rather than reading as zero"


def test_the_applied_solver_state_is_recorded_when_the_adapter_reports_it(parts) -> None:
    source, _sink, _coupler = parts
    source.diagnostics = {"execution": {"backend": "numpy", "precision": "float64"}}
    record = _executor(parts).run(_graph())
    states = {s["node"]: s for s in record.provenance["solver_state"]}
    assert states["lens"]["applied"] == {"backend": "numpy", "precision": "float64"}
    assert states["wave"]["applied"] == {}, (
        "an adapter that reports nothing leaves the applied state unknown, which is "
        "information rather than an assumption that it matched the request"
    )


# --------------------------------------------------------------------------- #
# Cache
# --------------------------------------------------------------------------- #


def test_a_cache_hit_produces_the_same_outputs_as_a_cold_run(parts) -> None:
    source, sink, _coupler = parts
    cache = InMemoryCache()
    executor = _executor(parts, cache=cache)

    cold = executor.run(_graph())
    warm = executor.run(_graph())

    assert cold.status is warm.status is RunStatus.SUCCEEDED
    assert [n.outputs for n in cold.nodes] == [n.outputs for n in warm.nodes]
    assert cache.hits == 2 and source.calls == 1 and sink.calls == 1


def test_the_cache_refuses_to_serve_across_a_different_solver_configuration(parts) -> None:
    """A cache that returned a result computed under different precision is
    worse than no cache: it turns a precision question into an invisible one."""
    source, _sink, _coupler = parts
    cache = InMemoryCache()
    executor = _executor(parts, cache=cache)

    executor.run(_graph())
    calls_after_cold = source.calls

    fp32 = _graph(
        nodes=[
            {"id": "lens", "model": "M_RAY_OPTILAND", "config": {"precision": "float32"}},
            {"id": "wave", "model": "M_WAVE_CHROMATIX"},
        ]
    )
    executor.run(fp32)
    assert source.calls == calls_after_cold + 1, "a different precision must miss"


def test_the_cache_refuses_to_serve_across_a_different_seed(parts) -> None:
    source, _sink, _coupler = parts
    executor = _executor(parts, cache=InMemoryCache())
    executor.run(_graph(), seed=1)
    executor.run(_graph(), seed=2)
    assert source.calls == 2


# --------------------------------------------------------------------------- #
# Resource guard
# --------------------------------------------------------------------------- #


def test_swap_growth_aborts_the_run_with_a_resource_failure(parts, monkeypatch) -> None:
    """Growth in the container's swap is a stop condition, not a slowdown.

    Driven through the watchdog's own verdict rather than by allocating memory:
    the policy under test is the executor's response to a breach, and actually
    exhausting a shared machine to prove it would be the wrong experiment.
    """
    from runtime import executor as executor_module
    from core.resources import MemoryWatchdogVerdict

    class _TrippingWatchdog:
        def __init__(self, *args, **kwargs) -> None:
            self.verdict = MemoryWatchdogVerdict()
            self.peak_rss_bytes = 123
            self._samples = 0

        def start(self):
            return self

        def stop(self):
            return self

        def sample(self):
            self._samples += 1
            self.verdict = MemoryWatchdogVerdict(
                breached=True,
                reason="swap_growth",
                detail="container swap grew by 4 MiB above baseline",
                observed_bytes=4 << 20,
                limit_bytes=0,
            )
            return None

    monkeypatch.setattr(executor_module, "MemoryWatchdog", _TrippingWatchdog)
    record = _executor(parts, watchdog_interval_s=0.25).run(_graph())

    assert record.status is RunStatus.FAILED
    assert record.refusal is not None
    assert record.refusal.kind is RefusalKind.RESOURCE_GUARD
    assert "swap" in record.refusal.detail
    assert record.refusal.remedy and "smaller chunk" in record.refusal.remedy
    assert any(d["code"] == "RESOURCE_GUARD_TRIPPED" for d in record.diagnostics)
    lens = record.node("lens")
    assert lens is not None and lens.outcome is NodeOutcome.EXECUTED, (
        "the work already done is preserved"
    )


# --------------------------------------------------------------------------- #
# Process model
# --------------------------------------------------------------------------- #


def test_asking_for_an_unimplemented_process_model_raises(parts) -> None:
    """Running in-process while reporting process-per-node would make the
    record's process_model field a lie."""
    with pytest.raises(ExecutorError, match="declared and not implemented"):
        _executor(parts, process_model=ProcessModel.PROCESS_PER_NODE)


def test_the_process_model_is_part_of_provenance(parts) -> None:
    """Whichever model is used, the record states it, because it is part of what
    makes a fingerprint reproducible."""
    record = _executor(parts).run(_graph())
    assert record.provenance["process_model"] == "in_process"


def test_ten_consecutive_runs_agree_on_everything_that_is_not_volatile(parts) -> None:
    """The executor-side counterpart of M0.1's repetition gate.

    Run ids, timings and memory snapshots are volatile by construction; the
    graph fingerprint, the node outcomes, the artifact ids and the solver state
    are not, and drift in any of them would mean the executor itself is a source
    of irreproducibility.
    """
    executor = _executor(parts)

    def scientific(record) -> tuple:  # type: ignore[no-untyped-def]
        return (
            record.graph_sha256,
            record.status,
            tuple((n.node_id, n.outcome, tuple(n.outputs)) for n in record.nodes),
            tuple(sorted(record.artifacts)),
            record.provenance["process_model"],
            record.provenance["environment_sha256"],
        )

    signatures = {scientific(executor.run(_graph(), seed=11)) for _ in range(10)}
    assert len(signatures) == 1


# --------------------------------------------------------------------------- #
# Cost
# --------------------------------------------------------------------------- #


def test_estimate_versus_actual_is_recorded_per_node(parts) -> None:
    record = _executor(parts).run(_graph())
    for node in record.nodes:
        assert node.cost is not None
        assert node.cost.estimate is not None
        assert node.cost.estimate_ratio is not None


def test_an_estimator_that_raises_does_not_fail_the_run(parts) -> None:
    """Failing a run over its cost model would make the model harder to improve
    than to remove."""
    source, _sink, _coupler = parts

    def _explode(request):  # type: ignore[no-untyped-def]
        raise RuntimeError("uncalibrated environment")

    source.estimate = _explode  # type: ignore[method-assign]
    record = _executor(parts).run(_graph())
    assert record.status is RunStatus.SUCCEEDED
    lens = record.node("lens")
    assert lens is not None and lens.cost is not None and lens.cost.estimate is None


# --------------------------------------------------------------------------- #
# Graph-level declared sources
# --------------------------------------------------------------------------- #


def test_a_declared_source_feeds_a_node_with_no_upstream_producer(parts) -> None:
    """A single-node wave graph consumes a field it did not compute.

    Resolved exactly like an upstream output, so the node cannot tell the
    difference and the executor does not need two plumbing paths.
    """
    _source, sink, _coupler = parts
    spec = GraphSpec(nodes=[{"id": "wave", "model": "M_WAVE_CHROMATIX"}])
    field = _artifact("supplied-field", ArtifactKind.COMPLEX_FIELD, dtype="complex64")

    record = _executor(parts).run(spec, inputs={"wave.input_field": field})

    assert record.status is RunStatus.SUCCEEDED
    (request,) = sink.requests
    assert request.inputs["input_field"].id == "supplied-field"


def test_a_declared_source_without_a_port_is_refused(parts) -> None:
    """Without the port the executor would have to guess which input it feeds,
    and guessing which port an artifact belongs to is exactly the kind of
    inference this executor refuses."""
    spec = GraphSpec(nodes=[{"id": "wave", "model": "M_WAVE_CHROMATIX"}])
    field = _artifact("supplied-field", ArtifactKind.COMPLEX_FIELD)
    with pytest.raises(ExecutorError, match=re.escape("<node>.<port>")):
        _executor(parts).run(spec, inputs={"wave": field})


def test_an_upstream_output_wins_over_a_declared_source(parts) -> None:
    """A declared source is a *fallback* for a port nothing produces. If an edge
    produced one, using the declaration instead would silently discard the
    computation the graph asked for."""
    _source, sink, _coupler = parts
    field = _artifact("supplied-field", ArtifactKind.COMPLEX_FIELD, dtype="complex64")
    _executor(parts).run(_graph(), inputs={"wave.input_field": field})
    (request,) = sink.requests
    assert request.inputs["input_field"].id == "bridged-field"


def test_a_coupler_with_no_graph_node_is_refused_before_anything_runs(parts) -> None:
    """C_WAVE_TO_RAY is the live case, and the answer to M2.2's question.

    It is declared in ``registry/couplers.yaml`` -- which per AGENTS.md is a
    statement that a graph may address it -- and it has no ``get_coupler()``
    module: it is a library component that the patch and DOE couplers wrap
    internally, not something composable as an edge. Before this, a graph
    declaring it validated and then died with a resolver traceback partway
    through. Now it is a structured refusal naming what will work instead.
    """
    _source, _sink, _coupler = parts
    registry = _Registry()
    registry.couplers = {**registry.couplers, "C_WAVE_TO_RAY": BRIDGE.model_copy(
        update={"id": "C_WAVE_TO_RAY"}
    )}
    spec = _graph(
        edges=[
            {
                "id": "bridge",
                "coupler": "C_WAVE_TO_RAY",
                "source": {"node": "lens", "port": "rays"},
                "target": {"node": "wave", "port": "input_field"},
            }
        ]
    )
    record = _executor(parts, registry=registry).run(spec)

    assert record.refusal is not None
    assert record.refusal.kind is RefusalKind.UNSUPPORTED_CAPABILITY
    assert "C_WAVE_TO_RAY" in record.refusal.declaration
    assert record.refusal.remedy and "C_RAY_TO_WAVE" in record.refusal.remedy
    assert parts[0].calls == 0
