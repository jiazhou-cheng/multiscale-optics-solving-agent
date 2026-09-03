"""What happened, written down. Never whether it was right.

CHE-199 (R13.1), with `stabilize_diagnostics` added by CHE-200 (R13.2). Two records
and eight functions, and one rule that the whole module exists to make executable:

> **Deleting any provenance field changes no physical result.**

If deleting a field changed physics, that information was never provenance -- it
was an input, and it belongs in the typed representation, the problem or the
request. `tests/unit/test_provenance_separation.py` executes that on every field
rather than asserting it in prose, and the reference tree had a field sitting on
the line: the CHE-50 wavefront-curvature limitation travelled in
`provenance["validity"]` as a *warning* until R02.4 moved it to a typed field on
`ScalarField`.

The split the records are built around
--------------------------------------
An `ExecutionRecord` has two halves and they answer different questions:

* `request`, `node_requests` and `route` -- **what to run.** Typed inputs and
  operation ids. A re-run reads these and nothing else, which is what makes the
  rule above checkable rather than merely stated: the deletion test re-derives the
  physics *from the record* after stripping each provenance field, so a field that
  physics read would show up as a changed number. `node_requests` is in this half
  and not the other because a repeated operation's arguments are an input: a route
  that transforms at f1 and then at f2 is a different system from one that
  transforms twice at f1, and nothing else in the record can tell them apart.
* `nodes` and `provenance` -- **what happened.** Statuses, structured
  diagnostics, the device and precision actually observed, and the fingerprints.
  Nothing in this half is an input to anything.

No verdict. The runtime records what happened; whether it was *right* is the
verification layer's question, and this module has no field for an answer to it --
no `reproduces`, no `verdict`, no `passed`. The reference tree's `RecordVerdict`
lived in `core/provenance.py` and is deliberately not here.

Reused without re-derivation, from `pre-rewrite-2026-08-30:src/core/provenance.py`
---------------------------------------------------------------------------------
Four results, each measured there and carried across unchanged in substance:

* **the normalized source digest.** Hashing raw bytes is simpler and wrong in a
  way that matters: a record would be invalidated by a typo fix in a docstring,
  the regeneration would be ceremony, and the mechanism would be routed around
  within a month. Parse to an AST, drop docstring expressions, dump. What survives
  is the code. It can still over-trigger -- renaming a local changes the dump --
  and that is the correct direction to fail in;
* **`strip_volatile`**, and the key names it projects out. A fingerprint must not
  move because a record was regenerated on a different day;
* **the environment fingerprint fields**: the interpreter version and the pinned
  array/solver package versions only. Not the whole environment, which would make
  every image rebuild invalidate every record regardless of relevance;
* **the `uuid4`-in-a-refusal-message finding.** A record whose payload contained a
  fresh uuid rehashed on every run, and B0-META-01 read as a changed measurement
  every time. That is a defect in the record, not a measurement change, so
  `require_stable_payload` refuses a uuid or a timestamp reaching a fingerprinted
  payload rather than leaving it to be discovered by a fingerprint that never
  settles.

What is deliberately not here
-----------------------------
* **No caching, no cache key, no graph fingerprint.** The reference executor's
  four fingerprints -- per-node cache key, graph, environment, source -- account
  for most of its 1145 lines and nine classes; each was individually reasonable.
  CHE-199 admits a cache only if CHE-200 measures a cost that justifies one.
* **No `src/io/` package.** The target shape named `src/io/artifacts.py`, and under
  a flat `src/` namespace root a top-level `io` package shadows the standard
  library's. Serialization is `to_json`/`from_json` here.
* No `NodeOutcome`, `RefusalKind`, `Refusal`, `DevicePrecisionObservation` or
  `ResourceCost` classes -- the reference tree's seven in `core/execution_record.py`
  are a `Literal`, two strings and two mappings here. No `RunProvenance`, no
  `RecordVerdict`, no `ArtifactRecord` module, no `ProbedRefusal`, and none of the
  eleven of `core/performance.py`: a performance harness is a script, not
  production.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.metadata
import json
import platform
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

__all__ = [
    "FINGERPRINTED_PACKAGES",
    "NODE_STATUSES",
    "PROVENANCE_SCHEMA_VERSION",
    "VOLATILE_KEYS",
    "ExecutionRecord",
    "NodeRecord",
    "environment_fingerprint",
    "from_json",
    "record_provenance",
    "require_stable_payload",
    "source_fingerprint",
    "stabilize_diagnostics",
    "strip_volatile",
    "to_json",
]

#: What a node's execution came to. Three outcomes, and the distinction between
#: the last two is worth keeping: a **refusal** is this project declining to
#: compute something it cannot describe, and a **failure** is something breaking.
#: A caller reading a record needs to tell "the boundary said no" from "the
#: backend raised", because the first is a fact about the problem and the second is
#: a fact about the run.
#:
#: A `Literal` rather than an enum, for the reason `measurements.psf` gives for
#: `PsfNormalization`: `scripts/class_budget.py` counts a `StrEnum` as a class, and
#: three strings need no type.
NODE_STATUSES: tuple[str, ...] = ("completed", "refused", "failed")

#: Keys that identify a *run* rather than a computation, dropped before hashing.
#:
#: Carried over verbatim from the reference implementation's `VOLATILE_KEYS`,
#: including the spellings: a probe record already uses `timestamp_utc`, and
#: renaming it here would leave the old records unprojectable. Device and dtype are
#: **not** in this list on purpose -- they change what was computed, so a
#: fingerprint that ignored them would claim reproducibility across a real change.
VOLATILE_KEYS: tuple[str, ...] = (
    # CHE-200 (R13.2) added this one; the rest are the reference implementation's,
    # spellings included. `resources` holds the memory guard's observations -- peak
    # RSS, peak container swap, the baseline -- and they are a measurement of *this
    # run* rather than of the computation, which is the same category every other
    # key here is in. Two runs of the same plan legitimately differ in peak RSS, so
    # a fingerprint that included it would report a change nobody made; and the
    # whole subtree goes, because a nested baseline reading is no less volatile than
    # the peak it is compared against.
    #
    # One latent edge, since this filters by key name at **every** depth including
    # inside `request`: a future request key spelled `resources`, `run_id` or
    # `output_directory` would vanish from `fingerprinted`, so an *input* would
    # become invisible to the fingerprint. None of these is a plausible physical
    # parameter name, which is why the filter is by name at all -- but it is the
    # thing to check before adding a key here.
    "resources",
    "runtime_seconds",
    "process_wall_seconds",
    "worker_process_seconds",
    "import_seconds",
    "setup_seconds",
    "timestamp_utc",
    "run_id",
    "output_directory",
)

#: Packages whose version changes the numbers an operation produces.
#:
#: Pinned solvers and array libraries only. Not the whole environment: that would
#: make every image rebuild invalidate every record regardless of relevance, which
#: is the failure that gets a staleness mechanism switched off.
FINGERPRINTED_PACKAGES: tuple[str, ...] = (
    "numpy",
    "scipy",
    "jax",
    "jaxlib",
    "torch",
    "optiland",
    "chromatix",
)

#: The provenance block's schema version. Read by `from_json`, which refuses a
#: version it does not know -- a version nothing validates reads as a
#: compatibility guarantee and is not one.
#:
#: Bumped to 2 when `node_requests` joined the "what to run" half, so a
#: version-1 payload is missing a key this reader requires. Nothing serialized
#: under version 1 is committed anywhere in the repository, so the bump strands no
#: artifact -- and a reader that silently defaulted the new key would claim a run
#: had no per-node arguments when the record simply predates them.
PROVENANCE_SCHEMA_VERSION = 2

#: What a fingerprinted payload may not contain, and the two shapes that put it
#: there. A uuid or an ISO timestamp anywhere in a hashed payload makes the hash
#: change on every run, which is how B0-META-01 came to read as a changed
#: measurement every time it was regenerated.
_UUID = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")
_TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}")


@dataclass(frozen=True, slots=True)
class NodeRecord:
    """What happened at one operation. Minimality rule 2: a serialized model.

    Five fields, and each has a reader. `operation_id` and `status` are what a
    caller scans; `diagnostics` is the structured refusal or failure text, which is
    the thing a failed path must return *instead of* a plausible-looking result;
    `requested` and `observed` are the device/precision pair, kept separate because
    a requested device is not evidence of an actual one -- a process-global JAX
    platform pin produces a successful host run while the caller asked for CUDA,
    with no error raised, and only a comparison of the two can see it.

    `requested` and `observed` are plain string mappings rather than a
    `DevicePrecisionObservation` class. There is nothing to enforce that
    `numerics.ArrayState` does not already enforce where the values come from, and
    the reference tree spent a class on each of them.
    """

    operation_id: str
    status: Literal["completed", "refused", "failed"]
    diagnostics: str = ""
    requested: Mapping[str, str] = field(default_factory=dict)
    observed: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        problems: list[str] = []
        if not self.operation_id.strip():
            problems.append("`operation_id` is empty")
        if self.status not in NODE_STATUSES:
            problems.append(f"`status` is {self.status!r}; the outcomes are {list(NODE_STATUSES)}")
        elif self.status == "completed" and self.diagnostics:
            problems.append(
                "a completed node carries diagnostics. A diagnostic is what a refusal or a "
                "failure returns instead of a result; on a completed node it reads as a "
                "warning the record has no vocabulary for."
            )
        elif self.status != "completed" and not self.diagnostics.strip():
            problems.append(
                f"a {self.status} node carries no diagnostics. A failed path returns "
                "diagnostics, never a plausible-looking result and never silence."
            )
        # Normalized to plain dicts of str->str, so a record cannot carry a mutable
        # mapping on a frozen dataclass and cannot carry an unserializable value.
        for name in ("requested", "observed"):
            raw = getattr(self, name)
            try:
                normalized = {str(key): str(value) for key, value in dict(raw).items()}
            except (TypeError, ValueError):
                problems.append(f"`{name}` is not a mapping of strings")
                continue
            object.__setattr__(self, name, normalized)
        if problems:
            raise ValueError(
                f"node record {self.operation_id!r} is not usable:\n  " + "\n  ".join(problems)
            )

    @property
    def placement_disagreement(self) -> tuple[str, ...]:
        """Keys where the observed device or precision differs from the requested.

        The check the reference implementation called `placement_disagreement`, kept
        because the failure it catches is silent: a run that reports success while
        having computed somewhere the caller did not ask for. Empty when they agree
        or when nothing was observed.
        """
        return tuple(
            sorted(
                key
                for key, value in self.requested.items()
                if key in self.observed and self.observed[key] != value
            )
        )


@dataclass(frozen=True, slots=True)
class ExecutionRecord:
    """One execution, as what to re-run plus what happened.

    Minimality rule 2: this is the public serialized model, and the two halves are
    the whole design -- see the module docstring. `route`, `request` and
    `node_requests` are inputs a re-run reads; `nodes` and `provenance` are
    observations nothing reads back.

    `status` is derived rather than stored: a record whose `status` could disagree
    with its nodes has two answers to "did this run", and the nodes are the ones
    with evidence.
    """

    route: tuple[str, ...]
    request: Mapping[str, Any] = field(default_factory=dict)
    #: One request per step of `route`, when the plan gave each occurrence its own,
    #: and `()` when every node was bound from the shared `request`.
    #:
    #: **An input, not an observation**, which is why it sits in this half beside
    #: `route` and not on `NodeRecord`. It exists because a route may repeat an
    #: operation -- two focal-plane transforms at f1 and f2 -- and one flat mapping
    #: keyed by parameter name cannot say which occurrence `focal_length_m` belongs
    #: to. Measured before it existed: the six-node 4f plan bound f1 to *both* legs
    #: and reported `completed`, with nothing in the record saying f2 had never been
    #: read. So the per-node requests are recorded for the same reason `request` is
    #: -- they are what a re-run reads -- and a record that dropped them would
    #: describe a system it did not run.
    #:
    #: Aligned with `route` by index and validated to its length, so an entry cannot
    #: belong to an operation other than the one it is written beside.
    node_requests: tuple[Mapping[str, Any], ...] = ()
    nodes: tuple[NodeRecord, ...] = ()
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "route", tuple(str(item) for item in self.route))
        object.__setattr__(self, "nodes", tuple(self.nodes))
        problems: list[str] = []
        if not self.route:
            problems.append("`route` is empty; a record describes operations that ran")
        for index, node in enumerate(self.nodes):
            if not isinstance(node, NodeRecord):
                problems.append(f"`nodes[{index}]` is {type(node).__name__}, not a NodeRecord")
        if not problems and len(self.nodes) > len(self.route):
            problems.append(
                f"{len(self.nodes)} node record(s) for a {len(self.route)}-operation route; "
                "a node is one operation of the route and a run stops at the first refusal"
            )
        if not problems:
            for index, node in enumerate(self.nodes):
                if node.operation_id != self.route[index]:
                    problems.append(
                        f"`nodes[{index}]` records {node.operation_id!r} but the route's "
                        f"step {index} is {self.route[index]!r}; the nodes are the route's "
                        "operations in order"
                    )
        for name in ("request", "provenance"):
            try:
                object.__setattr__(self, name, dict(getattr(self, name)))
            except (TypeError, ValueError):
                problems.append(f"`{name}` is not a mapping")
        try:
            per_node = tuple(dict(entry) for entry in self.node_requests)
        except (TypeError, ValueError):
            problems.append("`node_requests` is not a sequence of mappings")
        else:
            object.__setattr__(self, "node_requests", per_node)
            if per_node and len(per_node) != len(self.route):
                problems.append(
                    f"{len(per_node)} per-node request(s) for a {len(self.route)}-operation "
                    "route. `node_requests` is aligned with `route` by index, so a short "
                    "one would silently attribute a request to the wrong occurrence of a "
                    "repeated operation -- which is the failure it was added to end. It is "
                    "either empty, meaning every node was bound from the shared `request`, "
                    "or one entry per step."
                )
        if problems:
            raise ValueError(
                "execution record is not usable:\n  " + "\n  ".join(problems)
            )

    @property
    def status(self) -> str:
        """`completed` only when every operation of the route completed.

        Derived, so it cannot disagree with the nodes. A run that stopped early is
        not `completed` even if every node it *did* record succeeded, which is why
        the route length is part of the answer.
        """
        if len(self.nodes) != len(self.route):
            return "failed"
        for node in self.nodes:
            if node.status != "completed":
                return node.status
        return "completed"

    @property
    def fingerprinted(self) -> dict[str, Any]:
        """The part of this record a fingerprint is taken over.

        `route`, `request`, `node_requests` and each node's outcome -- with
        `strip_volatile` applied and `provenance` excluded entirely, because
        provenance *contains* the fingerprints and hashing it would be circular.
        This is what `require_stable_payload` checks and what a caller compares
        between two runs to ask "did the physics change".

        `node_requests` is in here for exactly that reason: it is what makes two
        runs of one route with different per-node arguments -- an f1/f2 relay
        against an f1/f1 one -- fingerprint differently. Left out, the two would
        hash identically and the fingerprint would answer "nothing changed" about a
        different optical system.
        """
        payload: dict[str, Any] = strip_volatile(
            {
                "route": list(self.route),
                "request": dict(self.request),
                "node_requests": [dict(entry) for entry in self.node_requests],
                "nodes": [
                    {
                        "operation_id": node.operation_id,
                        "status": node.status,
                        "diagnostics": node.diagnostics,
                        "requested": dict(node.requested),
                        "observed": dict(node.observed),
                    }
                    for node in self.nodes
                ],
            }
        )
        return payload

    def without_provenance_field(self, name: str) -> ExecutionRecord:
        """This record with one provenance key removed.

        Exists for `tests/unit/test_provenance_separation.py`, and it is a
        production method rather than a test helper on purpose: the rule it serves
        -- that no provenance field can reach physics -- is a claim about this
        record's contract, so the operation that tests it belongs on the record.
        """
        if name not in self.provenance:
            raise KeyError(
                f"{name!r} is not a provenance key of this record; it carries "
                f"{sorted(self.provenance)}"
            )
        return replace(
            self, provenance={key: value for key, value in self.provenance.items() if key != name}
        )


def strip_volatile(value: Any) -> Any:
    """Recursively drop run-identity keys from a nested structure.

    Applied before hashing. Descends dicts and lists and leaves scalars alone; the
    filter is by key name at *every* depth, because a nested per-node record
    carries its own timings.
    """
    if isinstance(value, dict):
        return {
            key: strip_volatile(item) for key, item in value.items() if key not in VOLATILE_KEYS
        }
    if isinstance(value, list):
        return [strip_volatile(item) for item in value]
    return value


def _normalized_source_digest(path: Path) -> str:
    """A digest of what a source file *does*, with comments and docstrings out.

    See the module docstring: hashing raw bytes invalidates a record on a typo fix
    in a docstring, which is how a staleness mechanism gets routed around.
    """
    source = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source)
    except SyntaxError:
        # Not importable Python; the bytes are the best available statement.
        return hashlib.sha256(source.encode("utf-8")).hexdigest()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        body = node.body
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            node.body = body[1:] or [ast.Pass()]
    return hashlib.sha256(ast.dump(tree).encode("utf-8")).hexdigest()


def source_fingerprint(paths: Iterable[Path], *, root: Path) -> dict[str, Any]:
    """Fingerprint a set of source files, keeping the per-file digests.

    The per-file digests are kept and not just the combined hash, so a stale record
    can say *which* module moved rather than only that something did. That is the
    difference between a failure someone can act on and one they re-derive by
    bisection.
    """
    root = root.resolve()
    digests: dict[str, str] = {}
    for path in sorted({Path(item).resolve() for item in paths}):
        if not path.is_file():
            continue
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError:
            continue
        digests[relative] = _normalized_source_digest(path)
    combined = hashlib.sha256(
        "\n".join(f"{name}:{digest}" for name, digest in sorted(digests.items())).encode("utf-8")
    ).hexdigest()
    return {"combined_sha256": combined, "files": digests}


def environment_fingerprint() -> dict[str, Any]:
    """Interpreter and pinned solver versions -- what changes speed and numbers.

    A package that is not installed is *absent* rather than recorded as `None`, so
    the fingerprint of an image without torch is stable rather than carrying a
    placeholder that a later install would change into a version.
    """
    versions: dict[str, str] = {}
    for name in FINGERPRINTED_PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    payload: dict[str, Any] = {
        "python_version": platform.python_version(),
        "packages": versions,
    }
    payload["combined_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return payload


def record_provenance(
    *,
    sources: Iterable[Path],
    root: Path,
    timestamp_utc: str | None = None,
    environment: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The provenance block a record carries. Observations only.

    `sources` is passed in rather than read off `sys.modules` here. The reference
    implementation read it, deliberately, so that a hand-maintained dependency list
    could not go stale -- and paid for it with over-capture, since anything else in
    the process was swept in. Which trade is right depends on the caller, so the
    choice stays with the caller: `runtime.executor` (CHE-200) decides what its
    run's sources are, and this function hashes what it is given.

    `timestamp_utc` is an argument and has no default clock reading. A function
    that stamped `datetime.now()` would make this module's own output unhashable
    by `require_stable_payload`, and the name is the one `VOLATILE_KEYS` already
    projects out.

    `environment` may be supplied by a caller that already has one. Measured for
    CHE-200: `environment_fingerprint()` is **7.5 ms**, because
    `importlib.metadata.version` scans the filesystem once per package and there
    are seven, and computing it per record made a two-node run 391% slower than
    the physics it ran. Installed versions cannot change inside a process, so
    `runtime.Executor` reads it once when it opens and passes it here. That is the
    one memoization in this package and it is of an immutable process fact, not of
    a result -- see `executor.py` on why there is no result cache.
    """
    payload: dict[str, Any] = {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "code_fingerprint": source_fingerprint(sources, root=root),
        "environment_fingerprint": (
            environment_fingerprint() if environment is None else dict(environment)
        ),
    }
    if timestamp_utc is not None:
        payload["timestamp_utc"] = timestamp_utc
    return payload


def require_stable_payload(payload: Any, *, where: str) -> None:
    """Refuse a fingerprinted payload that cannot hash to the same value twice.

    A `uuid4` or an ISO timestamp inside a hashed payload makes the hash change on
    every run. Measured, not hypothetical: a uuid inside a *refusal message* made
    B0-META-01 read as a changed measurement on every regeneration, and nothing
    could tell that from a real change. So the record refuses it rather than
    leaving it to be discovered by a fingerprint that never settles.

    Searched over the payload's JSON text, so a uuid nested in a diagnostic string
    is caught -- which is where the real one was.

    Raises:
        ValueError: naming which pattern was found and where.
    """
    text = json.dumps(payload, sort_keys=True, default=str)
    for name, pattern in (("a uuid", _UUID), ("a timestamp", _TIMESTAMP)):
        found = pattern.search(text)
        if found is not None:
            raise ValueError(
                f"{where} contains {name} ({found.group(0)!r}), so its fingerprint would "
                "change on every run and nothing could tell that from a real change. A run "
                f"identifier belongs in a key {list(VOLATILE_KEYS)} projects out, not in a "
                "hashed payload -- and a uuid inside a diagnostic message is the case that "
                "actually happened."
            )


def stabilize_diagnostics(text: str) -> str:
    """One diagnostic message with any uuid or timestamp replaced by a placeholder.

    The B0-META-01 defect, closed at the point where it can now recur. `runtime`'s
    executor writes arbitrary backend and boundary exception text into
    `NodeRecord.diagnostics`, and `ExecutionRecord.fingerprinted` **includes**
    diagnostics -- so a refusal message carrying a `uuid4` would make the record's
    fingerprint change on every run, and nothing would be able to tell that from a
    real change. That is exactly what happened to B0-META-01, and
    `require_stable_payload` refuses it; the executor calls this first so a
    legitimate run is not refused for a message it did not write.

    The message keeps its meaning: what is lost is the specific identifier, which
    identified the run rather than the problem. Replaced rather than stripped, so a
    reader can see that something was there.
    """
    stabilized = _UUID.sub("<uuid>", text)
    return _TIMESTAMP.sub("<timestamp>", stabilized)


def to_json(record: ExecutionRecord) -> str:
    """One record as JSON: the artifact serialization, and the whole of it.

    Sorted keys and a trailing newline, so two runs that produced the same record
    produce the same bytes and a diff is readable. `json` and not a serialization
    framework, and not its own module: `src/io/artifacts.py` from the target shape
    would be a top-level `io` package shadowing the standard library's.
    """
    return json.dumps(
        {
            "schema_version": PROVENANCE_SCHEMA_VERSION,
            "route": list(record.route),
            "request": dict(record.request),
            "node_requests": [dict(entry) for entry in record.node_requests],
            "nodes": [
                {
                    "operation_id": node.operation_id,
                    "status": node.status,
                    "diagnostics": node.diagnostics,
                    "requested": dict(node.requested),
                    "observed": dict(node.observed),
                }
                for node in record.nodes
            ],
            "provenance": dict(record.provenance),
        },
        indent=2,
        sort_keys=True,
    ) + "\n"


def from_json(text: str) -> ExecutionRecord:
    """The round trip's other half, refusing a payload it does not understand.

    Strict for the reason `numerics.knowledge` is: a serialized record is not
    type-checked, so the loader is the only thing between a hand-edited file and a
    record that claims something. Unknown keys are refused and missing keys are
    refused, so a typo cannot become a silently defaulted field.

    Raises:
        ValueError: the payload is not this schema, or a record built from it would
            not be usable -- in which case `ExecutionRecord`'s own refusal is what
            reaches the caller.
    """
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("an execution record is a JSON object")
    expected = {"schema_version", "route", "request", "node_requests", "nodes", "provenance"}
    missing, unknown = expected - set(payload), set(payload) - expected
    if missing or unknown:
        raise ValueError(
            f"the record's keys are wrong: missing {sorted(missing)}, "
            f"unrecognized {sorted(unknown)}"
        )
    if payload["schema_version"] != PROVENANCE_SCHEMA_VERSION:
        raise ValueError(
            f"schema_version is {payload['schema_version']!r} and this reader understands "
            f"{PROVENANCE_SCHEMA_VERSION}"
        )
    node_keys = {"operation_id", "status", "diagnostics", "requested", "observed"}
    nodes = []
    for raw in payload["nodes"]:
        if not isinstance(raw, dict) or set(raw) != node_keys:
            raise ValueError(f"a node record's keys must be exactly {sorted(node_keys)}")
        nodes.append(NodeRecord(**raw))
    return ExecutionRecord(
        route=tuple(payload["route"]),
        request=payload["request"],
        node_requests=tuple(payload["node_requests"]),
        nodes=tuple(nodes),
        provenance=payload["provenance"],
    )
