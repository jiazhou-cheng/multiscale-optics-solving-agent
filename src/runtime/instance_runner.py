"""Execute a canonical benchmark instance. Records what happened; judges nothing.

CHE-106/107/108/109/110/111/112 (M1, M2). Every family driver needs the same
four pieces of plumbing, and none of them should be reimplemented per family:
hand a ``GraphSpec`` to ``GraphExecutor`` keyed to the instance so the record and
the instance cannot describe different computations; build a declared graph
source for a node with no upstream producer; read a device and a dtype off an
array rather than off a request; and turn a refusal that a shipping component
actually raised into an ``ExecutionRecord``.

**Nothing here decides whether a number is right**, and that is enforced rather
than intended: ``tests/test_package_dependencies.py`` fails if ``runtime/``
imports ``verification/``, on the argument that an executor which could grade its
own run makes the ``ExecutionRecord`` boundary meaningless. The evidence side --
fitting a convergence exponent, judging a broken twin, hashing a scientific
fingerprint, writing a record -- lives in ``verification/evidence.py``, which
imports this module and not the other way round.

The device and dtype rule, implemented rather than restated
-----------------------------------------------------------
``observed_placement`` reads the namespace, device and dtype **off an array**.
Nothing here copies a requested device into a reported one, and
``placement_disagreement`` is what a caller uses to turn "asked for cuda, ran on
cpu" into a refusal instead of a successful CUDA run. A process-global JAX
platform pin makes that the default failure mode rather than an exotic one.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.boundary import ContractError
from core.errors import UnsupportedCapabilityError
from core.execution import RunStatus
from core.execution_record import (
    DevicePrecisionObservation,
    ExecutionRecord,
    NodeOutcome,
    NodeRecord,
    Refusal,
    RefusalKind,
    ResourceCost,
)
from core.precision import CapabilityError
from core.specs import GraphSpec
from registry.loader import Registry
from runtime.executor import GraphExecutor, _refusal_kind_for

__all__ = [
    "RUNNER_VERSION",
    "ProbedRefusal",
    "execute",
    "field_source",
    "observed_placement",
    "placement_disagreement",
    "probe_refusal",
    "record_from_probe",
]

#: Bumped when this module changes what a driver's record means. Carried in
#: every written record beside the executor and verifier versions.
RUNNER_VERSION = "1.0.0"


def _repository_root() -> Path:
    from core.paths import repository_root

    return repository_root()


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


def execute(
    graph: GraphSpec,
    instance: Any,
    *,
    seed: int | None = None,
    inputs: Mapping[str, Any] | None = None,
    registry: Registry | None = None,
) -> ExecutionRecord:
    """Run the graph, keyed to the instance.

    The instance id and fingerprint travel into the record so a later
    verification cannot silently compare a record against an instance it was not
    produced for -- the verifier reports that as a diagnostic rather than
    trusting the pairing.
    """
    executor = GraphExecutor(registry or Registry.from_package())
    return executor.run(
        graph,
        seed=seed,
        instance_id=instance.instance_id,
        instance_fingerprint=instance.fingerprint,
        inputs=inputs,
    )



def field_source(
    array: Any,
    *,
    wavelength_m: float,
    sample_pitch_m: float | tuple[float, float],
    directory: Path,
    artifact_id: str,
    plane_name: str = "input plane",
    plane_z_m: float = 0.0,
) -> Any:
    """A declared graph source for a wave node, from an array.

    Graph plumbing rather than physics: a single-node wave graph has no upstream
    producer, and the executor's ``inputs`` want an ``ArtifactRecord``. Building
    it goes through ``ComplexField`` rather than writing a record by hand, so the
    source a family drives is subject to the same boundary contract as anything
    a solver produced -- pitch in metres, a declared phasor, a declared pad
    state, a declared normalization.
    """
    from core.boundary import ComplexField, ReferencePlane

    pitch = (
        (float(sample_pitch_m), float(sample_pitch_m))
        if isinstance(sample_pitch_m, int | float)
        else (float(sample_pitch_m[0]), float(sample_pitch_m[1]))
    )
    field = ComplexField(
        u=array,
        sample_pitch_m=pitch,
        wavelength_m=float(wavelength_m),
        reference_plane=ReferencePlane(name=plane_name, z_m=float(plane_z_m)),
    )
    directory.mkdir(parents=True, exist_ok=True)
    return field.to_artifact_record(
        artifact_id=artifact_id, uri=directory / f"{artifact_id}.npy"
    )



@dataclass(frozen=True)
class ProbedRefusal:
    """What shipping code raised, in the shape a record needs."""

    code: str
    kind: RefusalKind
    detail: str
    declaration: str | None
    remedy: str | None
    exception_type: str
    supported: tuple[str, ...] = ()

    def as_refusal(self) -> Refusal:
        return Refusal(
            kind=self.kind,
            detail=self.detail,
            declaration=self.declaration,
            remedy=self.remedy,
        )


def probe_refusal(thunk: Callable[[], Any]) -> tuple[ProbedRefusal | None, Any]:
    """Call ``thunk``; report the structured refusal it raised, if it raised one.

    Returns ``(None, value)`` when the call succeeded -- which is itself an
    answer, and for the two silent-hazard instances it is the *expected* one:
    their declared contract status is ``ok`` and the physics is wrong.

    An exception that is not one of the repository's structured errors is
    re-raised. An unstructured traceback is one of the two things M1.3 says a
    caller must never receive, so swallowing it into a record would hide exactly
    the defect this is looking for.
    """
    try:
        return None, thunk()
    except ContractError as exc:
        code = str(exc.code)
        return (
            ProbedRefusal(
                code=code,
                # The executor's own map, so a probed refusal and a graph refusal
                # for the same code carry the same kind. What that kind MEANS --
                # which of the five negative outcomes it is -- is the verifier's
                # question and is answered from the code, not from here.
                kind=_refusal_kind_for([code]),
                detail=str(exc),
                declaration=exc.declaration,
                remedy=exc.remedy,
                exception_type=type(exc).__name__,
            ),
            None,
        )
    except CapabilityError as exc:
        # ``BridgeError`` is a ``CapabilityError``, and both mean the same thing
        # for status purposes: the component cannot execute this, so the caller
        # must change the request rather than fix it.
        return (
            ProbedRefusal(
                code=str(exc.code),
                kind=RefusalKind.UNSUPPORTED_CAPABILITY,
                detail=str(exc),
                declaration=exc.component,
                remedy=exc.remedy
                or (
                    f"choose from the supported set: {exc.supported}"
                    if exc.supported
                    else None
                ),
                exception_type=type(exc).__name__,
                supported=tuple(exc.supported or ()),
            ),
            None,
        )
    except UnsupportedCapabilityError as exc:
        # The base class, raised by the adapters' eager capability gates. It
        # carries no structured code -- the message is the whole diagnostic -- so
        # the code is the exception's own name rather than an invented one.
        # Caught AFTER CapabilityError, which subclasses it and does carry one.
        return (
            ProbedRefusal(
                code="UNSUPPORTED_CAPABILITY",
                kind=RefusalKind.UNSUPPORTED_CAPABILITY,
                detail=str(exc),
                declaration=None,
                remedy=None,
                exception_type=type(exc).__name__,
            ),
            None,
        )


def record_from_probe(
    instance: Any,
    *,
    component: str,
    node_id: str,
    refusal: ProbedRefusal | None,
    observed_parameters: Mapping[str, Any] | None = None,
    device_precision: DevicePrecisionObservation | None = None,
    lossy: bool = False,
    wall_seconds: float = 0.0,
    diagnostics: Sequence[Mapping[str, Any]] = (),
) -> ExecutionRecord:
    """An ``ExecutionRecord`` for a contract case run outside the graph model.

    Carries the instance fingerprint like any other record, so the verifier can
    still tell whether the record and the instance describe the same
    computation. ``artifacts`` is deliberately empty on a refusal: there is no
    partial result, and a downstream node must not be able to find one.
    """
    if refusal is None:
        outcome = NodeOutcome.EXECUTED_LOSSY if lossy else NodeOutcome.EXECUTED
        status = RunStatus.SUCCEEDED
    else:
        outcome = NodeOutcome.REFUSED
        # No REFUSED run status exists: RunStatus is succeeded/failed/partial,
        # and a refusal produced nothing, so FAILED is the honest one. Which
        # KIND of failure it was lives on the refusal, which is where a caller
        # reads it.
        status = RunStatus.FAILED

    node = NodeRecord(
        node_id=node_id,
        component=component,
        outcome=outcome,
        refusal=None if refusal is None else refusal.as_refusal(),
        error_type=None if refusal is None else refusal.exception_type,
        error_message=None if refusal is None else refusal.detail,
        contract_codes=[] if refusal is None else [refusal.code],
        device_precision=device_precision,
        cost=ResourceCost(wall_seconds=wall_seconds),
    )
    return ExecutionRecord(
        run_id=f"probe-{uuid.uuid4().hex[:12]}",
        status=status,
        instance_id=instance.instance_id,
        instance_fingerprint=instance.fingerprint,
        nodes=[node],
        observed_parameters=dict(observed_parameters or {}),
        device_precision=device_precision,
        refusal=None if refusal is None else refusal.as_refusal(),
        diagnostics=[dict(d) for d in diagnostics],
        provenance={
            "producer": "runtime.instance_runner.record_from_probe",
            "runner_version": RUNNER_VERSION,
            "why_not_a_graph": (
                "this instance's question is answered before a graph could run -- a "
                "capability intersection, a bridge policy, or an artifact-level "
                "declaration -- so the shipping refusal path is the execution"
            ),
        },
    )




# ---------------------------------------------------------------------------
# Device and dtype, read off the data
# ---------------------------------------------------------------------------


def observed_placement(array: Any) -> dict[str, str]:
    """What an array actually is: namespace, device, dtype.

    Read off the object. Never off a request -- that is the whole point, and the
    live hazard is a process-global JAX platform pin producing a successful host
    run for a caller who asked for CUDA, with nothing raised.
    """
    dtype = str(getattr(array, "dtype", "unknown"))
    namespace = type(array).__module__.split(".")[0]

    device = "cpu"
    raw = getattr(array, "device", None)
    if raw is not None:
        # torch: ``device`` is an object with a ``type``. jax: ``device`` may be
        # a method on older versions and a property on newer ones; either way
        # the string form names the platform.
        if callable(raw):
            try:
                raw = raw()
            except Exception:  # pragma: no cover - defensive on unknown backends
                raw = None
        device = str(getattr(raw, "type", None) or raw or "cpu")
    else:
        devices = getattr(array, "devices", None)
        if callable(devices):
            try:
                device = str(next(iter(devices())))
            except Exception:  # pragma: no cover - defensive
                device = "cpu"

    return {"namespace": namespace, "device": device.lower(), "dtype": dtype}


def placement_disagreement(requested: Mapping[str, Any], observed: Mapping[str, str]) -> str | None:
    """``None`` if the run honoured the request; otherwise what it did instead.

    Compared on the platform prefix rather than the full string, so ``cuda`` and
    ``cuda:0`` agree and ``cuda`` and ``cpu`` do not. A caller turns a non-None
    answer into ``REPRESENTATION_INCONSISTENT``; reporting the request as the
    actual placement is the failure this exists to make impossible.
    """
    problems: list[str] = []
    want_device = str(requested.get("device", "")).lower().split(":")[0]
    got_device = str(observed.get("device", "")).lower().split(":")[0]
    if want_device and want_device != got_device:
        problems.append(f"device: requested {want_device!r}, array is on {got_device!r}")

    want_dtype = str(requested.get("dtype", "")).lower()
    got_dtype = str(observed.get("dtype", "")).lower()
    if want_dtype and want_dtype not in got_dtype:
        problems.append(f"dtype: requested {want_dtype!r}, array is {got_dtype!r}")

    return "; ".join(problems) or None


