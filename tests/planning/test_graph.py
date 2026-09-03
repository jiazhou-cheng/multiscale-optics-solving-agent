"""What composes, and the four things the graph must not become.

CHE-164 (R12). Acceptance criteria:

1. constructible and enumerable with **no backend in `sys.modules`** -- asserted
   in a fresh interpreter, not stated;
2. every field the graph reasons over comes from an `OperationDescriptor`; the
   graph owns no facts of its own;
3. no workflow engine, no agent, no LLM call;
4. at least one real multi-step route -- rays to scalar to propagated scalar to
   PSF -- discoverable end to end.

The ticket's main risk is the specification for what is *absent*: reasoning over
validity and cost before a planner consumes it produces descriptor fields nobody
reads, which is how the reference implementation's `discovery/api.py` reached 944
lines and ten pydantic view models answering three questions for one caller. So
several tests below assert absence, and each says what it is refusing rather than
naming a file that no longer exists.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest

from operations import CATALOG, SEMANTIC_TYPES, OperationDescriptor, OperationKind
from planning import ENTRY, capability_graph, routes

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "src" / "planning"

#: The route acceptance criterion 4 names, as operation ids.
#:
#: `ray_bundle` -> `scalar_field` (the wavelet-sum coupler) -> `scalar_field` (the
#: angular spectrum) -> `psf` (the measurement). Three operations, three edges, and
#: every one of them a landed implementation rather than a dummy -- which is the
#: ticket's "do not build this against dummy operations".
CRITERION_4_ROUTE = ("C_RAY_TO_SCALAR", "O_ASM_PROPAGATE", "M_PSF")


# ---------------------------------------------------------------------------
# 1. Criterion 1 -- no backend, in a fresh interpreter
# ---------------------------------------------------------------------------

BACKENDS = ("jax", "jaxlib", "torch", "optiland", "chromatix")


def test_building_and_enumerating_the_graph_pulls_no_backend() -> None:
    """Criterion 1. The whole point of routing over metadata.

    Asking what this project can compose must not load what it would compose it
    with. Checked against `sys.modules` in a subprocess for the same reason
    `operations` is: the failure is transitive, and reading the imports of
    `graph.py` will not show a backend pulled three levels down.

    The enumeration is deliberately the widest one available -- every ordered pair
    of semantic types, plus every entry route -- so this is a statement about a
    real call rather than about an import.
    """
    source = """
import planning
from operations import SEMANTIC_TYPES

graph = planning.capability_graph()
assert graph, graph
total = 0
for target in SEMANTIC_TYPES:
    for start in (None, *SEMANTIC_TYPES):
        for route in planning.routes(frm=start, to=target):
            total += len(route)
assert total > 100, total
import sys, json
print(json.dumps(sorted({m.split('.')[0] for m in sys.modules})))
"""
    completed = subprocess.run(
        [sys.executable, "-c", source], capture_output=True, text=True, check=True
    )
    loaded = set(json.loads(completed.stdout))
    assert not loaded & set(BACKENDS), (
        f"enumerating the capability graph loaded {sorted(loaded & set(BACKENDS))}"
    )
    # And it did run, so the assertion above is not about an empty enumeration.
    assert {"planning", "operations"} <= loaded


# ---------------------------------------------------------------------------
# 2. Criterion 2 -- the graph owns no facts
# ---------------------------------------------------------------------------


def test_every_edge_is_read_off_a_descriptor() -> None:
    """Criterion 2, as an identity between the graph and the catalog.

    Rebuilt here from `CATALOG` by a different route -- iterating descriptors and
    grouping -- and compared. A graph that had acquired an edge of its own, or
    dropped one, fails; and because the expected value is computed rather than
    written down, there is no third place for the adjacency to live.
    """
    expected: dict[str | None, set[str]] = {}
    for descriptor in CATALOG:
        keys = [ENTRY] if descriptor.is_graph_entry else list(descriptor.inputs)
        for key in keys:
            expected.setdefault(key, set()).add(descriptor.operation_id)

    graph = capability_graph()
    assert {key: set(value) for key, value in graph.items()} == expected
    # Sorted, so the structure is deterministic and readable in a diff.
    for value in graph.values():
        assert list(value) == sorted(value)


def _string_constants(path: Path) -> set[str]:
    """Every string constant in a module's code, docstrings excluded.

    Read as AST constants and **not** by searching the unparsed text for
    `"value"`: `ast.unparse` renders string literals with single quotes, so a
    substring check written with double quotes cannot fail. That was this test's
    first version, and `test_the_literal_check_can_fail` below is the meta-check
    that keeps it from being that again.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    docstrings: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef):
            continue
        body = node.body
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            docstrings.add(id(body[0].value))
    return {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in docstrings
    }


def test_the_graph_module_states_no_port_no_validity_and_no_cost() -> None:
    """Criterion 2's other direction: the module cannot contain a fact to own.

    Ports are read as AST string constants, so the quoting style cannot hide one.
    Operation ids and the descriptor-owned field names are substring checks over
    the code with docstrings stripped, because the prose has to be able to name
    what it is refusing and does at length.

    What a violation looks like: a semantic type written as a literal, which would
    be a port the graph decided rather than one it read; or a cost or validity
    table keyed by operation id.
    """
    literals = _string_constants(PACKAGE / "graph.py")
    named_ports = literals & set(SEMANTIC_TYPES)
    assert named_ports == set(), (
        f"graph.py names the semantic type(s) {sorted(named_ports)} as literals. Ports "
        "come from descriptors; a literal here is a port the graph decided."
    )

    code = _code_of(PACKAGE / "graph.py")
    for descriptor in CATALOG:
        assert descriptor.operation_id not in code, (
            f"graph.py names the operation {descriptor.operation_id!r}. The graph "
            "enumerates the catalog; naming a member of it is a fact of its own."
        )
    for owned in ("cost", "validity", "approximation", "evidence", "capabilities"):
        assert owned not in code, f"graph.py reasons over {owned!r}, which is a descriptor's"


def test_the_literal_check_can_fail(tmp_path: Path) -> None:
    """The meta-check, because the first version of the check above could not fail.

    It searched `ast.unparse` output for `"ray_bundle"` with double quotes;
    `ast.unparse` emits single quotes, so `PORT = "ray_bundle"` -- exactly the
    violation the assertion names -- passed. Both quotings are exercised here.
    """
    for source in ('PORT = "ray_bundle"\n', "PORT = 'ray_bundle'\n", 'T = {"psf": 1.0}\n'):
        module = tmp_path / "probe.py"
        module.write_text(source)
        assert _string_constants(module) & set(SEMANTIC_TYPES), source
    # And a docstring mention is deliberately not a violation.
    module = tmp_path / "probe.py"
    module.write_text('"""Routes a ray_bundle to a psf."""\n')
    assert _string_constants(module) & set(SEMANTIC_TYPES) == set()


def test_a_route_names_operations_and_carries_nothing_about_them() -> None:
    """A route is ids, so everything else stays one attribute access away.

    The alternative -- returning a view model per operation -- is exactly the
    `discovery/api.py` shape the ticket names: ten pydantic models restating
    descriptor fields for one caller. A caller that wants the approximation reads
    it off the descriptor, and that is a strictly wider answer than any view model
    would have offered.
    """
    route = routes(frm="ray_bundle", to="psf")[0]
    assert all(isinstance(operation_id, str) for operation_id in route)
    catalogued = {descriptor.operation_id: descriptor for descriptor in CATALOG}
    for operation_id in route:
        descriptor = catalogued[operation_id]
        assert isinstance(descriptor, OperationDescriptor)
        assert descriptor.approximation and descriptor.implementation


# ---------------------------------------------------------------------------
# 3. Criterion 4 -- a real multi-step route
# ---------------------------------------------------------------------------


def test_the_rays_to_propagated_scalar_to_psf_route_is_discoverable() -> None:
    """Criterion 4, on the exact route the ticket names.

    Not "some route of length 3": the specific composition
    `ray_bundle -> scalar_field -> scalar_field -> psf`, through the wavelet-sum
    coupler, the angular spectrum and the PSF measurement. Every one is a landed
    implementation with its own physics tests.
    """
    found = routes(frm="ray_bundle", to="psf")
    assert CRITERION_4_ROUTE in found, found

    # And it composes: each step's output is the next step's input, read off the
    # descriptors rather than assumed from the route's existence.
    catalogued = {descriptor.operation_id: descriptor for descriptor in CATALOG}
    state: str | None = "ray_bundle"
    for operation_id in CRITERION_4_ROUTE:
        descriptor = catalogued[operation_id]
        assert state in descriptor.inputs, (operation_id, state, descriptor.inputs)
        state = descriptor.primary_output
    assert state == "psf"


def test_the_shortest_route_is_first_and_the_ordering_is_stable() -> None:
    """Length then lexicographic, which is an enumeration order and not a cost claim.

    `graph.py` says so explicitly: there is no cost model here, so "shortest first"
    is a reading convenience. Stability matters because a caller comparing two runs
    would otherwise see set iteration order.
    """
    found = routes(frm="ray_bundle", to="psf")
    assert found[0] == ("C_RAY_TO_SCALAR", "M_PSF")
    assert list(found) == sorted(found, key=lambda route: (len(route), route))
    assert routes(frm="ray_bundle", to="psf") == found


def test_a_source_reaches_a_psf_from_no_upstream_state() -> None:
    """`frm=ENTRY`, which is what `inputs=()` made expressible (CHE-222 / R03.5).

    "How do I get a PSF at all" is the question a planner asks first, and before
    R03.5 the schema could not answer it: a source declared the representation it
    *produces* as its input, so it looked like a consumer of what it emits.
    """
    found = routes(frm=ENTRY, to="psf")
    assert ("S_SOURCE_PLANE_WAVE", "M_PSF") in found
    # Every route from ENTRY starts at an operation that consumes no upstream
    # representation. Checked against the catalog rather than against the id
    # spelling: `route[0].startswith("S_")` was the old form and it broke on
    # CHE-225's `SO_RAY_LAUNCH_TRACE`, which is a graph entry whose prefix is a
    # composition rather than a primitive kind. The id was never the property under
    # test.
    entries = {r.operation_id for r in CATALOG if r.is_graph_entry}
    for route in found:
        assert route[0] in entries, route
    # The fused launch-and-trace is the other entry, and it enters at `ray_bundle`.
    assert routes(frm=ENTRY, to="ray_bundle", max_steps=1) == (("SO_RAY_LAUNCH_TRACE",),)


def test_a_state_with_nothing_composing_to_it_returns_no_route() -> None:
    """An empty result is an answer here, unlike in `operations.find`.

    Nothing consumes a `psf` -- an observable is terminal, which
    `OperationDescriptor.__post_init__` enforces -- so no route leaves one. That is
    a real fact about the tree and it comes back as `()`. The two type arguments
    are validated precisely so a typo cannot be mistaken for this.
    """
    assert routes(frm="psf", to="scalar_field") == ()
    assert routes(frm="psf", to="psf") == ()


# ---------------------------------------------------------------------------
# 4. Termination, bounds and refusals
# ---------------------------------------------------------------------------


def test_the_graph_has_cycles_and_the_no_repeat_rule_is_what_terminates() -> None:
    """Both cycles, and the fact that makes an unbounded default safe.

    `scalar_field -> scalar_field` through four operations and
    `ray_bundle -> ray_bundle` through three. Termination therefore does **not**
    come from `max_steps` -- it comes from the no-repeat rule, which bounds a route
    by the catalog size. Asserted because `routes` now defaults to no bound on the
    strength of exactly that argument.

    **The scalar count was four until CHE-224 (R15.1), and losing one is the point
    rather than a regression.** The fourth was `S_WAVE_CHROMATIX`, which resolved to
    the same callable as `O_ASM_PROPAGATE` and existed only because `kind` was being
    asked both "which library runs this" and "what happens to the state". A planner
    enumerating this graph therefore saw two distinguishable-looking candidates for
    one function, differing in no field it could route on. `backend` answers the
    first question as a field, the pair is merged, and the graph now has one edge
    per thing the tree can actually do.

    **And it is four again since CHE-228 (R06.11), for the opposite reason.** The
    new fourth is `O_FRESNEL_PROPAGATE`, which is a distinct callable running a
    distinct kernel, so it is exactly the edge the merged pair was not: two things
    the tree can do, not one thing named twice. The count returning to its old value
    is a coincidence worth stating, because the two fours mean opposite things.
    """
    graph = capability_graph()
    catalogued = {descriptor.operation_id: descriptor for descriptor in CATALOG}
    cycles = {
        state: sorted(
            operation_id
            for operation_id in ids
            if catalogued[operation_id].primary_output == state
        )
        for state, ids in graph.items()
        if state is not None
    }
    assert len(cycles["scalar_field"]) == 4, cycles["scalar_field"]
    assert len(cycles["ray_bundle"]) == 3, cycles["ray_bundle"]

    # And the unbounded search returns: finite, and bounded by the catalog size.
    unbounded = routes(frm=ENTRY, to="psf")
    assert unbounded
    assert max(len(route) for route in unbounded) <= len(CATALOG)


def test_the_unbounded_default_keeps_the_canonical_multi_scale_route() -> None:
    """Why there is no default bound, stated as the route a bound would have lost.

    `max_steps=4` was the first default here, and it silently omitted
    `SO_RAY_LAUNCH_TRACE -> O_PROPAGATE_RAYS -> C_RAY_TO_SCALAR -> O_ASM_PROPAGATE ->
    M_PSF`: trace the system, advance the rays, cross to the wave model, propagate
    the field, measure. That is the project's whole reason for existing, and a
    default that dropped it while its own comment claimed headroom was worse than
    verbose output.
    """
    canonical = (
        "SO_RAY_LAUNCH_TRACE",
        "O_PROPAGATE_RAYS",
        "C_RAY_TO_SCALAR",
        "O_ASM_PROPAGATE",
        "M_PSF",
    )
    assert canonical in routes(frm=ENTRY, to="psf")
    assert canonical not in routes(frm=ENTRY, to="psf", max_steps=4)


def test_no_operation_appears_twice_in_one_route() -> None:
    """Two propagations in a row is one route with a distance argument.

    Enumerating the permutations would be noise rather than a different
    composition, and the argument that distinguishes them is not this module's to
    model. Checked over every reachable route rather than one.
    """
    for target in SEMANTIC_TYPES:
        for start in (ENTRY, *SEMANTIC_TYPES):
            for route in routes(frm=start, to=target):
                assert len(set(route)) == len(route), route


def test_max_steps_bounds_the_route_length() -> None:
    for bound in (1, 2, 3):
        for route in routes(frm="ray_bundle", to="psf", max_steps=bound):
            assert len(route) <= bound
    assert routes(frm="ray_bundle", to="psf", max_steps=1) == ()
    assert routes(frm="ray_bundle", to="psf", max_steps=2) == (("C_RAY_TO_SCALAR", "M_PSF"),)


def test_an_unknown_semantic_type_is_refused_rather_than_returning_nothing() -> None:
    """The same rule `operations.find` follows, and for the same reason.

    `routes(frm="rays", to="psf")` returning `()` would be indistinguishable from
    a correct "nothing composes", which is a real answer here -- so a typo has to
    be an error or the empty result means two things.
    """
    with pytest.raises(ValueError, match="semantic type"):
        routes(frm="rays", to="psf")
    with pytest.raises(ValueError, match="semantic type"):
        routes(frm="ray_bundle", to="point_spread_function")
    # `frm=None` is ENTRY and must not be caught by that check; `to=None` is not a
    # state and must be.
    assert routes(frm=None, to="psf")
    with pytest.raises(ValueError, match="semantic type"):
        routes(frm="ray_bundle", to=None)  # type: ignore[arg-type]


def test_a_max_steps_below_one_is_refused() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        routes(frm="ray_bundle", to="psf", max_steps=0)


def test_routing_over_a_synthetic_catalog_needs_no_monkeypatching() -> None:
    """The `catalog=` argument, and why it is not a second source.

    A three-operation chain nothing in the tree ships, routed end to end. This is
    what makes the algorithm testable independently of what happens to be
    catalogued -- and it is an argument rather than a module-level indirection, so
    production code has exactly one catalog to pass.
    """
    def descriptor(
        operation_id: str, inputs: tuple[str, ...], returns: tuple[str, ...], kind: OperationKind
    ) -> OperationDescriptor:
        return OperationDescriptor(
            operation_id=operation_id,
            kind=kind,
            inputs=inputs,
            returns=returns,
            implementation="tests.planning.nothing:run",
            approximation="none; a synthetic operation for the routing tests",
            evidence=("tests/planning/test_graph.py",),
        )

    synthetic = (
        descriptor("X_ENTRY", (), ("ray_bundle",), OperationKind.SOURCE),
        descriptor("X_CROSS", ("ray_bundle",), ("scalar_field",), OperationKind.COUPLER),
        descriptor("X_OBSERVE", ("scalar_field",), ("psf",), OperationKind.MEASUREMENT),
    )
    assert routes(frm=ENTRY, to="psf", catalog=synthetic) == (
        ("X_ENTRY", "X_CROSS", "X_OBSERVE"),
    )
    assert routes(frm="ray_bundle", to="psf", catalog=synthetic) == (
        ("X_CROSS", "X_OBSERVE"),
    )
    # A synthetic catalog does not leak into the production one.
    assert "X_CROSS" not in capability_graph()["ray_bundle"]


# ---------------------------------------------------------------------------
# 5. Criterion 3, and the classes that did not land
# ---------------------------------------------------------------------------


def test_a_multi_port_operation_is_refused_rather_than_filed_under_each_port() -> None:
    """The day a two-input operation lands, this fails loudly instead of lying.

    Filing it under both ports is the naive reading, and it would make `routes`
    report a composition that cannot execute: arriving at one of two required
    states is not enough to call it. Nothing in the catalog has two ports, and
    `operations.descriptors` says the tuple exists so one can land without a schema
    change -- so the refusal is what turns that into a failure here rather than a
    wrong answer downstream.
    """
    two_ports = OperationDescriptor(
        operation_id="X_INTERFERE",
        kind=OperationKind.PHYSICAL_OPERATOR,
        inputs=("scalar_field", "scalar_field"),
        returns=("scalar_field",),
        implementation="tests.planning.nothing:run",
        approximation="none; a synthetic two-port operation for this refusal",
        evidence=("tests/planning/test_graph.py",),
    )
    with pytest.raises(ValueError, match="representation ports"):
        capability_graph((two_ports,))
    with pytest.raises(ValueError, match="representation ports"):
        routes(frm="scalar_field", to="scalar_field", catalog=(two_ports,))
    # The production catalog has none, so the refusal is latent rather than active.
    assert all(len(descriptor.inputs) <= 1 for descriptor in CATALOG)


def _identifiers(path: Path) -> set[str]:
    """Every name a module defines, imports, calls or reads as an attribute.

    Identifiers rather than a substring search over the source, which is what this
    check first was and what a *refusal message* defeats: `capability_graph`'s
    message legitimately contains the word "execute" while explaining that a
    multi-port route could not be executed. What the criterion is about is whether
    an executor, a step loop or a model client has *arrived* in the package, and
    that is a statement about names.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            names.add(node.name)
        elif isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.add((node.module or "").split(".")[0])
            names.update(alias.name for alias in node.names)
    return names


def test_there_is_no_workflow_engine_no_agent_and_no_llm_call() -> None:
    """Criterion 3, checked on the package's identifiers rather than asserted.

    Matched on names rather than on behaviour, which is the honest limit: what this
    catches is the shape -- an executor, a step loop, a model client -- arriving in
    a package whose job is to answer a metadata question. A name is the right unit
    for that, and it is what the package's own prose can discuss freely without
    tripping the check.
    """
    names = _identifiers(PACKAGE / "graph.py") | _identifiers(PACKAGE / "__init__.py")
    forbidden = {
        "execute", "run", "run_plan", "invoke", "step", "schedule",
        "Workflow", "workflow", "Agent", "agent", "Executor", "executor",
        "llm", "openai", "anthropic", "prompt", "completion", "chat",
    }
    present = names & forbidden
    assert present == set(), f"planning/ defines, imports or calls {sorted(present)}"


def test_the_identifier_check_can_fail(tmp_path: Path) -> None:
    """The meta-check, since the assertion above is a negative.

    Also the record of what the substring version got wrong: it flagged the word
    "execute" inside a refusal message, so the fix had to be a real structural
    check rather than a longer exemption list.
    """
    module = tmp_path / "probe.py"
    for source, expected in (
        ("def execute(plan): ...\n", "execute"),
        ("import openai\n", "openai"),
        ("class Executor: ...\n", "Executor"),
        ("x = client.chat()\n", "chat"),
    ):
        module.write_text(source)
        assert expected in _identifiers(module), source
    # A message mentioning the word is not a violation, which is the whole point.
    module.write_text('raise ValueError("could not execute that route")\n')
    assert "execute" not in _identifiers(module)


def test_the_avoided_classes_did_not_land() -> None:
    """A budget records what exists; only a test can record what was avoided.

    The twenty-three names the reference implementation used for this job, from
    `core/graph.py` (458 lines) and `discovery/api.py` (944). `CapabilityGraph` is
    in the list on purpose: the ticket budgeted it and `graph.py` records why it
    fails all five of `AGENTS.md`'s class rules, so its absence is the decision and
    not an oversight.

    `AVOIDED` below is a **record**, not something the assertion consults: the
    package defines no class at all, which subsumes every name in it. Kept because
    a budget can only count what exists, and this is the list a future ticket would
    otherwise re-derive.
    """
    avoided = (
        "CapabilityGraph",
        "GraphValidator",
        "ValidationIssue",
        "ValidationReport",
        "Severity",
        "GraphSpec",
        "NodeSpec",
        "EdgeSpec",
        "PortRef",
        "DesignVariableSpec",
        "ObjectiveSpec",
        "VerificationSpec",
        "ComponentIndex",
        "PortView",
        "SuitabilityRecord",
        "RefusalView",
        "FamilyCoverage",
        "KnowledgeView",
        "ComponentDescription",
        "ConnectionReport",
        "Handover",
        "RouteCapability",
        "ValidityAnswer",
    )
    assert len(avoided) == 23
    defined = {
        node.name
        for path in sorted(PACKAGE.rglob("*.py"))
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, ast.ClassDef)
    }
    assert defined == set(), f"planning/ defines a class: {sorted(defined)}"


def test_the_package_reasons_over_metadata_and_imports_no_representation() -> None:
    """A route is a statement about descriptors, not about physical state.

    `scripts/check_dependencies.py` permits `planning -> operations` only, so this
    is the structural rule restated as an executed one -- and it is what keeps a
    route from quietly becoming something that holds a `RayBundle`.
    """
    code = _code_of(PACKAGE / "graph.py") + _code_of(PACKAGE / "__init__.py")
    for forbidden in ("representations", "RayBundle", "ScalarField", "numerics", "backends"):
        assert forbidden not in code, f"planning/ imports or names {forbidden!r}"


def _code_of(path: Path) -> str:
    """A module's source with every docstring removed.

    The prose has to be able to name what the package is not, and does at length.
    Checking the code is the only way to hold it to that.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef):
            continue
        body = node.body
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            node.body = body[1:] or [ast.Pass()]
    return ast.unparse(tree)
