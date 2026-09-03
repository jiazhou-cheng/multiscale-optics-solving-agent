"""The production catalog, and the gate that a landed operation cannot escape it.

CHE-221 (R03.4). Two properties, and they are different properties:

* **completeness** -- bidirectional, mechanical, and checked here. Every name in a
  package's `OPERATIONS` has a catalog record, and every catalog record's
  implementation belongs to a package whose `OPERATIONS` lists its attribute.
  Neither direction is checked against a hand-written expected-name list in this
  file, because a list here would be a third source of truth to keep in step with
  the other two.
* **resolution** -- that each `implementation` string names the callable it claims.
  That needs the backends, so it is `test_catalog_resolution.py` and it is
  deliberately in a separate module: this one must stay importable and runnable
  without loading torch or JAX, which is the property `operations/` exists for.

What no gate here can catch
---------------------------
Two things, stated rather than papered over.

**Semantic drift.** A record whose `approximation` prose no longer matches the
code passes every check below. A catalog is structurally a second source of truth
beside the implementation, and the mechanical checks bound the *structural* half
of that risk only.

**An operation that is in neither place.** `OPERATIONS` is hand-maintained, so
someone who lands a public callable and adds it to neither `OPERATIONS` nor the
catalog is invisible to a check that compares the two against each other. That is
the honest limit, and the alternative -- deriving coverage from `__all__` -- was
measured and rejected: `couplers.__all__` has 20 names of which 2 are operations,
so it would demand a descriptor for `DrawRule` and `SamplingDiagnostics`.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

import operations
from operations import CATALOG, OperationDescriptor, OperationKind
from operations.registry import _build_index

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"

#: Every package surface that declares `OPERATIONS`, as an import path.
#:
#: `backends/` has no package-level `__init__` operation surface -- it is a
#: namespace over backend subpackages -- so the two backends declare their own and
#: the gate walks both. Discovered rather than listed: any package under `src/`
#: whose `__init__.py` defines `OPERATIONS` is in scope, so a seventh
#: operation-bearing package joins this gate by existing.
def _operation_surfaces() -> tuple[str, ...]:
    found: list[str] = []
    for path in sorted(SRC.rglob("__init__.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        declares = any(
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "OPERATIONS"
            for node in tree.body
        )
        if declares:
            found.append(".".join(path.parent.relative_to(SRC).parts))
    return tuple(found)


SURFACES = _operation_surfaces()


def test_the_gate_found_the_surfaces_it_is_supposed_to_walk() -> None:
    """The meta-check: a discovery that found nothing cannot fail anything."""
    assert set(SURFACES) == {
        "couplers",
        "measurements",
        "operators",
        "backends.chromatix",
        "backends.optiland",
        "sources",
    }, SURFACES


# ---------------------------------------------------------------------------
# 1. The catalog itself
# ---------------------------------------------------------------------------


def test_the_catalog_is_the_canonical_declaration_and_holds_only_descriptors() -> None:
    assert isinstance(CATALOG, tuple)
    assert CATALOG, "an empty catalog is what CHE-221 exists to end"
    assert all(isinstance(record, OperationDescriptor) for record in CATALOG)
    assert operations.find() == tuple(sorted(CATALOG, key=lambda d: d.operation_id))
    assert operations.registered_ids() == tuple(sorted(d.operation_id for d in CATALOG))


def test_every_implementation_is_a_string_and_nothing_is_imported_to_hold_it() -> None:
    """`implementation` is the whole reason the catalog can live in `operations/`."""
    for record in CATALOG:
        assert isinstance(record.implementation, str)
        module_path, separator, attribute = record.implementation.partition(":")
        assert separator == ":" and module_path and attribute, record.implementation


def test_a_duplicate_operation_id_raises_rather_than_overwriting() -> None:
    """Acceptance criterion 7, on the function the import path actually runs.

    `CATALOG` is a tuple rather than a dict literal keyed by `operation_id`
    precisely so this check can exist: a dict literal would keep the last of two
    entries silently, and which one you got would depend on declaration order.
    """
    duplicated = (*CATALOG, CATALOG[0])
    with pytest.raises(ValueError, match="appears twice in the catalog"):
        _build_index(duplicated)
    # And the real catalog does not trip it, so the check above is about a
    # duplicate rather than about the function always raising.
    assert len(_build_index(CATALOG)) == len(CATALOG)


def test_the_index_refuses_something_that_is_not_a_descriptor() -> None:
    with pytest.raises(TypeError, match="not an"):
        _build_index(("S_NOT_A_RECORD",))  # type: ignore[arg-type]


def test_there_is_no_public_register_any_more() -> None:
    """The registration site moved into the package, so its entry point went away.

    Kept as an assertion rather than left to be noticed: a `register()` that still
    existed would be a second way to put an operation in the index, and the
    catalog's claim to be canonical would be a convention rather than a fact.
    """
    assert not hasattr(operations, "register")
    assert "register" not in operations.__all__
    import operations.registry as registry_module

    assert not hasattr(registry_module, "register")


# ---------------------------------------------------------------------------
# 2. Bidirectional completeness
# ---------------------------------------------------------------------------


def _declared_operations() -> dict[str, tuple[str, ...]]:
    """`{package: OPERATIONS}` for every surface, read by import.

    Importing these six packages does load their own dependencies, which is fine:
    the property under test in `test_registry_imports_no_backend.py` is about
    importing `operations`, not about this test module.
    """
    return {name: tuple(importlib.import_module(name).OPERATIONS) for name in SURFACES}


def _catalog_by_package() -> dict[str, set[tuple[str, OperationKind]]]:
    """`{package: {(attribute, kind)}}` from the catalog's implementation strings.

    The package is the implementation module's own package, which for
    `backends.optiland.solver:trace` is `backends.optiland` and not `backends`: the
    surface that declares `OPERATIONS` is the backend subpackage.
    """
    grouped: dict[str, set[tuple[str, OperationKind]]] = {}
    for record in CATALOG:
        module_path, _, attribute = record.implementation.partition(":")
        package = module_path.rsplit(".", 1)[0]
        grouped.setdefault(package, set()).add((attribute, record.kind))
    return grouped


def test_every_declared_operation_has_a_catalog_record() -> None:
    """Direction 1. A landed operation cannot exist without a descriptor."""
    catalogued = _catalog_by_package()
    missing = [
        f"{package}.{name}"
        for package, names in _declared_operations().items()
        for name in names
        if not any(attribute == name for attribute, _kind in catalogued.get(package, set()))
    ]
    assert missing == [], (
        "these operations are declared in a package's OPERATIONS and have no record in "
        "operations.catalog.CATALOG, so operations.find() cannot see them:\n  "
        + "\n  ".join(missing)
    )


def test_every_catalog_record_names_a_declared_operation() -> None:
    """Direction 2. A record cannot name something no package admits to shipping."""
    declared = _declared_operations()
    orphans = [
        f"{record.operation_id} -> {record.implementation}"
        for record in CATALOG
        if record.implementation.partition(":")[2]
        not in declared.get(record.implementation.partition(":")[0].rsplit(".", 1)[0], ())
    ]
    assert orphans == [], (
        "these catalog records name an attribute that is not in its package's OPERATIONS, "
        "so either the operation was renamed or the tuple was not updated:\n  "
        + "\n  ".join(orphans)
    )


def test_one_record_per_implementation() -> None:
    """Acceptance criterion 4, no longer needing a reconciliation with criterion 9.

    Criterion 4 asks for "exactly one record" per declared operation; criterion 9
    required `S_WAVE_CHROMATIX` and `O_ASM_PROPAGATE` to be two records over one
    callable, so uniqueness was keyed on `(implementation, kind)` to let both hold.
    CHE-224 (R15.1) removed the conflict at its source rather than keeping the
    compound key: the two records existed because `kind` was answering both "which
    library runs" and "what happens to the state", and `backend` answers the first
    now. The pair is merged, so the key is `implementation` alone.

    The compound key was a real loophole while it lasted -- any callable could
    acquire a second record under a different kind silently, which is exactly what
    the "exactly one record" wording was guarding against.
    """
    seen: dict[str, str] = {}
    for record in CATALOG:
        previous = seen.get(record.implementation)
        assert previous is None, (
            f"{record.operation_id} and {previous} are both records over "
            f"{record.implementation}. One callable, one record: the pair that needed "
            "two was two answers to two questions, and the second question is a field."
        )
        seen[record.implementation] = record.operation_id


def test_the_counts_this_justification_rests_on() -> None:
    """The `__all__`-is-mostly-not-operations measurement, asserted not narrated.

    "`couplers.__all__` has 20 names of which 2 are operations" is the whole reason
    `OPERATIONS` exists rather than coverage being derived from `__all__`, and it is
    repeated in three docstrings. Nothing checked it, and CHE-221 itself made it
    drift by 1 -- appending `OPERATIONS` to each `__all__`. So the ratio is measured
    here, and a change to any of these surfaces has to come past this test and
    update the prose it is quoted in.
    """
    import couplers
    import operators
    import sources

    for package, total, operational in (
        (couplers, 20, 2),
        (operators, 10, 3),
        (sources, 5, 3),
    ):
        assert len(package.__all__) == total, (package.__name__, package.__all__)
        assert len(package.OPERATIONS) == operational, package.OPERATIONS
        assert set(package.OPERATIONS) < set(package.__all__), (
            "OPERATIONS must be a strict subset of __all__: an operation nobody "
            "exports is not a public operation"
        )


def test_a_new_operation_without_a_descriptor_fails_the_gate() -> None:
    """Acceptance criterion 5, demonstrated rather than asserted by inspection.

    The gate above reads `OPERATIONS` by import, so the falsifier is a package
    whose tuple names something the catalog does not. Built as data rather than by
    monkeypatching a real package, so the check under test is the comparison and
    not an import side effect.
    """
    catalogued = _catalog_by_package()
    declared = {**_declared_operations(), "couplers": ("ray_to_scalar", "scalar_to_ray", "warp")}
    missing = [
        f"{package}.{name}"
        for package, names in declared.items()
        for name in names
        if not any(attribute == name for attribute, _kind in catalogued.get(package, set()))
    ]
    assert missing == ["couplers.warp"], missing


# ---------------------------------------------------------------------------
# 3. The two deliberate exclusions
# ---------------------------------------------------------------------------


def test_the_optiland_launch_is_excluded_on_purpose() -> None:
    """Acceptance criterion 10, with the package's own reason cited.

    `launch` takes native solver state -- a constructed `Optic` -- and is
    package-facing by construction, which `src/backends/optiland/__init__.py`
    records. It is neither in `__all__` nor in `OPERATIONS`, and no catalog record
    names it. A public launch operation needs a neutral signature first.
    """
    import backends.optiland as optiland_package

    assert "launch" not in optiland_package.OPERATIONS
    assert "launch" not in optiland_package.__all__
    assert not any("launch" in record.implementation for record in CATALOG)

    reason = (SRC / "backends" / "optiland" / "__init__.py").read_text(encoding="utf-8")
    assert "launch" in reason, "the exclusion has to be written down where the package is"


def test_configure_execution_is_not_an_operation() -> None:
    """The other exclusion, and the reason completeness is not read off `__all__`.

    It sets process-global backend state and returns no representation. A gate
    derived from `__all__` would have demanded a descriptor with an `input` and an
    `output` for it, and there is no honest pair of semantic types to give.
    """
    import backends.optiland as optiland_package

    assert "configure_execution" in optiland_package.__all__
    assert "configure_execution" not in optiland_package.OPERATIONS
    assert not any("configure_execution" in r.implementation for r in CATALOG)


# ---------------------------------------------------------------------------
# 4. The records the migration owed
# ---------------------------------------------------------------------------


def test_the_pair_of_records_over_one_callable_is_gone() -> None:
    """CHE-224 (R15.1) merged them, and this is the assertion that they stay merged.

    `S_WAVE_CHROMATIX` (`solver`) and `O_ASM_PROPAGATE` (`physical_operator`) both
    resolved to `backends.chromatix.solver:propagate`. The old version of this test
    pinned that as intended -- "one answers what backend does this project drive,
    the other what happens to the physical state" -- and naming two questions was
    the diagnosis rather than the justification. `backend` answers the first one now.

    So what is pinned is the merge: the surviving record is the physical operator,
    it declares the backend as a field, and it carries the scalar-model sentence the
    deleted record contributed. `S_WAVE_CHROMATIX` must not come back, and the
    uniqueness gate above -- one record per `implementation` -- is what stops any
    callable acquiring a second record silently.
    """
    index = {record.operation_id: record for record in CATALOG}
    assert "S_WAVE_CHROMATIX" not in index

    operator = index["O_ASM_PROPAGATE"]
    assert operator.implementation == "backends.chromatix.solver:propagate"
    assert operator.kind is OperationKind.PHYSICAL_OPERATOR
    assert operator.backend == "chromatix"
    assert operator.capabilities == "M_WAVE_CHROMATIX"
    # The one claim the deleted record made that this one did not: what the SCALAR
    # model omits, as distinct from what the angular-spectrum kernel approximates.
    # Migrated rather than dropped, which is the half of a merge that gets lost.
    for carried in ("no polarization", "no vectorial coupling", "complex64"):
        assert carried in operator.approximation, carried

    # And no callable has two records any more.
    counts: dict[str, int] = {}
    for record in CATALOG:
        counts[record.implementation] = counts.get(record.implementation, 0) + 1
    assert {name for name, n in counts.items() if n > 1} == set()


def test_the_three_operations_that_had_no_descriptor_now_have_one() -> None:
    """The gap CHE-221 measured: `trace_rays`, `gaussian_beam`, `spherical_wave`.

    Eleven descriptors existed in test fixtures and three landed operations had
    none at all. Named individually here because "13 records" would pass with the
    wrong thirteen.
    """
    by_implementation = {record.implementation: record for record in CATALOG}
    for implementation in (
        "backends.optiland.solver:trace_rays",
        "sources.gaussian_beam:gaussian_beam",
        "sources.spherical_wave:spherical_wave",
    ):
        record = by_implementation[implementation]
        assert record.approximation.strip()
        assert record.validity, "each of the three states its own applicability"
        assert record.evidence, "each cites the tests that already cover it"


def test_no_record_claims_a_gradient() -> None:
    """`forward_only` across the board, which is the project's rule and not a habit.

    A `differentiable` record with no `derivative_evidence` is refused at
    construction (R03.1), so what this adds is that no record has *acquired*
    evidence quietly: every cross-framework handoff in this tree is forward-only
    until an independent finite-difference validation supports the claim.
    """
    claiming = [r.operation_id for r in CATALOG if r.derivative != "forward_only"]
    assert claiming == [], claiming
    assert all(r.derivative_evidence is None for r in CATALOG)


def test_the_capability_citations_are_the_two_measured_rows_or_none() -> None:
    """A citation is validated at construction; this pins *which* rows are cited.

    Only the operations that drive an external backend cite a row. Everything else
    is `None`, which is the honest citation rather than a missing one: a coupler
    runs in whatever namespace the field it was handed carries, so citing the
    chromatix row would claim a measurement taken about something else.

    Note that this is **not** the same question as `backend`, which CHE-224 (R15.1)
    added: `capabilities` cites a *measured* device/dtype row, and a
    backend-driving operation with no measured row of its own would carry a
    `backend` and `capabilities=None`. The two happen to coincide across all
    thirteen records today, and
    `test_the_backend_field_and_the_capability_citation_are_different_questions` is
    where that coincidence is stated as a coincidence.
    """
    cited = {r.implementation.rsplit(".", 1)[0]: r.capabilities for r in CATALOG}
    assert cited["backends.optiland"] == "M_RAY_OPTILAND"
    assert cited["backends.chromatix"] == "M_WAVE_CHROMATIX"
    for record in CATALOG:
        if not record.implementation.startswith("backends."):
            assert record.capabilities is None, record.operation_id


# ---------------------------------------------------------------------------
# 5. The fixture-owned copies are gone, and cannot come back quietly
# ---------------------------------------------------------------------------

#: Where a descriptor may legitimately be constructed under `tests/`, with the
#: reason for each.
#:
#: The rule this enforces is "no **production** record living in a test", which is
#: the state CHE-221 ended: eleven real descriptors defined in implementation test
#: modules, each of which had to be kept in step with the code by hand. A record
#: built to be *refused*, or to stand in for the catalog while an algorithm over it
#: is tested, is the opposite -- it is the subject rather than a copy.
#:
#: Every entry has to be argued in. Two so far:
DESCRIPTOR_HOMES: dict[Path, str] = {
    # The schema and the index themselves: `test_descriptors.py` builds records to
    # prove which ones are refused, and `test_registry.py` builds a dummy to test
    # `find`/`resolve` behaviour over a synthetic index.
    Path("tests") / "operations": "the schema and the registry are the subjects",
    # CHE-164 (R12). `tests/planning/` routes over a synthetic three-operation
    # catalog, which is what makes the routing algorithm testable independently of
    # what the tree happens to ship -- the ids are `X_*` and no production record
    # is restated. Without it the only way to test the algorithm would be against
    # the live catalog, so a change to what is catalogued would change what the
    # graph tests mean.
    Path("tests") / "planning": "the routing algorithm over a catalog is the subject",
}

#: Where the **private registry index** may be touched: only this directory.
#:
#: A narrower set than `DESCRIPTOR_HOMES`, and the narrowing is the point. The
#: `tests/planning` entry above is argued entirely in terms of *constructing* a
#: synthetic catalog, and `planning.routes` takes a `catalog=` argument precisely so
#: that no test ever needs to reach into `registry._BY_ID`. Letting one exemption
#: grant the other would hand out a permission its own reason disclaims.
INDEX_HOMES: dict[Path, str] = {
    Path("tests") / "operations": "the registry itself is the subject",
}


def _names_the_descriptor(func: ast.expr) -> bool:
    """Whether a call target is `OperationDescriptor`, bare or qualified."""
    if isinstance(func, ast.Name):
        return func.id == "OperationDescriptor"
    return isinstance(func, ast.Attribute) and func.attr == "OperationDescriptor"


def _test_modules_outside(homes: dict[Path, str]) -> list[Path]:
    """Every test module not inside one of `homes`."""
    resolved = {ROOT / home for home in homes}
    found = sorted(
        path
        for path in (ROOT / "tests").rglob("*.py")
        if "__pycache__" not in str(path) and not resolved & set(path.parents)
    )
    assert len(found) > 40, "the walk found almost nothing, so it cannot fail"
    return found


def test_no_test_outside_a_declared_home_constructs_a_descriptor() -> None:
    """Acceptance criterion 8. Eleven of these existed at `a7db487`.

    Checked on the AST rather than on the text, so a mention in a docstring or a
    comment -- of which the migrated tests have several -- is not a violation and
    only a real call is. Both spellings count: a bare `OperationDescriptor(...)`
    and a qualified `operations.OperationDescriptor(...)`, which is an
    `ast.Attribute` rather than an `ast.Name` and would otherwise walk straight
    past a check written for the bare form.
    """
    offenders = []
    for path in _test_modules_outside(DESCRIPTOR_HOMES):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and _names_the_descriptor(node.func):
                offenders.append(f"{path.relative_to(ROOT)}:{node.lineno}")
    assert offenders == [], (
        "a production descriptor is being constructed in a test again. The catalog is "
        "the canonical declaration; read the record from `operations.CATALOG`:\n  "
        + "\n  ".join(offenders)
    )


def test_no_test_outside_a_declared_home_touches_the_private_index() -> None:
    """Acceptance criterion 8's other half: the ten `isolated_registry` copies.

    `isolated_registry` -- save `dict(registry._REGISTERED)`, clear, yield, restore
    -- was copied verbatim into ten implementation test modules. That was not
    incidental duplication: it is the shape a test takes when a production home
    does not exist. The home exists, so the copies are gone, and reaching into the
    index from outside this directory is what would bring them back.

    `INDEX_HOMES` and not `DESCRIPTOR_HOMES`: `tests/planning/` may construct a
    synthetic descriptor -- that is the routing algorithm's subject -- and may not
    touch the index, because `planning.routes` takes a `catalog=` argument so that
    it never has to. One exemption must not grant the other.
    """
    offenders = [
        f"{path.relative_to(ROOT)}: {name}"
        for path in _test_modules_outside(INDEX_HOMES)
        for name in ("_REGISTERED", "_BY_ID", "isolated_registry")
        if name in path.read_text(encoding="utf-8")
    ]
    assert offenders == [], (
        "private registry state is reachable from a test outside tests/operations/:\n  "
        + "\n  ".join(offenders)
    )


def test_the_walk_would_catch_a_violation() -> None:
    """The meta-check for both walks above, since both assert an empty list."""
    for source in (
        "d = OperationDescriptor(operation_id='X')\n",
        "d = operations.OperationDescriptor(operation_id='X')\n",
    ):
        calls = [
            node
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Call) and _names_the_descriptor(node.func)
        ]
        assert len(calls) == 1, source
    # And a docstring mention is deliberately not one.
    documented = ast.parse('"""Mentions OperationDescriptor( in prose."""\n')
    assert not [n for n in ast.walk(documented) if isinstance(n, ast.Call)]


# ---------------------------------------------------------------------------
# 6. The nine planner questions, answered from metadata alone. CHE-222 (R03.5).
# ---------------------------------------------------------------------------
#
# Each of these is one of the nine questions `operations/descriptors.py` names as
# the specification for the schema's field list. Question 9 -- which backend
# executes this -- is CHE-224 (R15.1)'s, and it is section 7 below, because its
# gate is an agreement between two fields rather than a property of one.
#
# They are asserted against the **shipped catalog**, from metadata only -- no
# signature inspection, no import of an implementation -- because that is the
# situation a planner is in.
#
# `tests/operations/test_catalog_signatures.py` is the other half: it derives the
# same four tuples from `inspect.signature` and compares, so what is asserted here
# as metadata is separately known to match the code.

#: The graph entries: the three analytic sources, the fused launch-and-trace, and
#: the two native analyses, which begin with a source stage inside the solver.
#: Written out because a count would pass with the wrong members -- and the count
#: has moved twice since it was four (CHE-226, CHE-236), which is the argument for
#: the set.
GRAPH_ENTRIES = {
    "SOM_PSF",
    "SOM_SPOT_DIAGRAM",
    "SO_RAY_LAUNCH_TRACE",
    "S_SOURCE_GAUSSIAN_BEAM",
    "S_SOURCE_PLANE_WAVE",
    "S_SOURCE_SPHERICAL_WAVE",
}

#: The three operations that return a 2-tuple.
AUXILIARY_RETURNERS = {"C_RAY_TO_SCALAR", "C_SCALAR_TO_RAY", "O_DIFFRACTIVE_SURFACE"}


def test_question_1_is_an_upstream_representation_edge_required() -> None:
    """Empty for the three sources and for `trace`; non-empty for the other ten.

    This is the question the old schema answered *wrongly*:
    `S_SOURCE_PLANE_WAVE` declared `input="scalar_field"` for a function that
    consumes no field, and `S_RAY_OPTILAND` declared `input="ray_bundle"` for one
    that consumes no bundle -- contradicting the code, `sources/__init__.py` and
    `docs/architecture_principles.md` §2, all three of which say a source is the
    one operation with no input.
    """
    entries = {record.operation_id for record in CATALOG if record.is_graph_entry}
    assert entries == GRAPH_ENTRIES
    assert len(CATALOG) - len(entries) == 11
    # Every entry BEGINS with a source, which is what `ENTRY_KINDS` enforces. Read
    # off `entry_stage` and not `kind` since CHE-225 (R15.2): `SO_RAY_LAUNCH_TRACE`
    # is `physical_operator`-kind because that is where it leaves the state, and it
    # is still an entry because its first stage is a source.
    for record in CATALOG:
        if record.is_graph_entry:
            assert record.entry_stage is OperationKind.SOURCE, record.operation_id
    assert operations.find(entry=True) == tuple(
        sorted((r for r in CATALOG if r.is_graph_entry), key=lambda r: r.operation_id)
    )


def test_question_2_the_two_one_port_operations_are_distinguishable() -> None:
    """`trace_rays` and `propagate_rays` both take one bundle. They differ in what else.

    This is the case CHE-216 could only hypothesize and CHE-217 (R05.6) made real:
    one representation port plus one required non-representation input. A schema
    with a single `input` string made these two records identical on every field a
    planner would use to tell them apart.
    """
    index = {record.operation_id: record for record in CATALOG}
    supplied = index["O_RAY_TRACE"]
    advanced = index["O_PROPAGATE_RAYS"]

    assert supplied.inputs == advanced.inputs == ("ray_bundle",)
    assert supplied.primary_output == advanced.primary_output == "ray_bundle"
    # Identical on both port fields, and distinguished by `requires` -- which is
    # the whole point of the field. Asserted as the two exact tuples; an inequality
    # between two literals a line above each other cannot fail.
    assert supplied.requires == ("setup", "execution")
    assert advanced.requires == ("to",)
    # The premise, so the comparison above is known to be the discriminating one.
    assert (supplied.inputs, supplied.returns) == (advanced.inputs, advanced.returns)


def test_question_3_every_required_value_is_named() -> None:
    """Eleven of the thirteen need a value the old schema never mentioned.

    The ticket says nine, from a table written before three of these records
    existed; the measured figure was twelve of fourteen and is eleven of thirteen
    since CHE-224 (R15.1) merged `S_WAVE_CHROMATIX` away. The two exceptions are
    named at the bottom of this test. Asserted as the exact set rather than a count, because "n
    records have a requirement" would pass with the wrong n. `psf` is the sharpest small case:
    `normalization` is keyword-only with no default and which one was used is the
    subject of three of R11's acceptance criteria, so a runtime must not pick.
    """
    requiring = {r.operation_id for r in CATALOG if r.requires}
    assert requiring == {
        "M_PSF",
        "O_ASM_PROPAGATE",
        "O_DIFFRACTIVE_SURFACE",
        "O_FOCAL_PLANE_TRANSFORM",
        # CHE-228 (R06.11). `distance_m` and `model`, the same two O_ASM_PROPAGATE
        # needs -- and its `model` is the shorter one, because there is no `method`
        # to choose between.
        "O_FRESNEL_PROPAGATE",
        "O_PROPAGATE_RAYS",
        "O_RAY_TRACE",
        "SO_RAY_LAUNCH_TRACE",
        "S_SOURCE_GAUSSIAN_BEAM",
        "S_SOURCE_PLANE_WAVE",
        "S_SOURCE_SPHERICAL_WAVE",
        "C_RAY_TO_SCALAR",
        # CHE-226 (R16). The native analysis needs four; `M_SPOT_DIAGRAM` needs none
        # and is one of the exceptions named below, because a bundle is a complete
        # call for it -- the rays carry their own plane, wavelength and intensity.
        "SOM_SPOT_DIAGRAM",
        # CHE-236 (R16.1). Five: the same four plus `method`, which selects which
        # of three propagations runs. It is required rather than defaulted because
        # the three are not interchangeable at coarse sampling -- measured, and on
        # the record's own `method_definitions` -- so a runtime must not pick one.
        "SOM_PSF",
    }
    index = {record.operation_id: record for record in CATALOG}
    assert index["M_PSF"].requires == ("normalization",)
    assert index["O_ASM_PROPAGATE"].requires == ("distance_m", "model")
    # The units in the names are part of the contract, not decoration.
    assert index["O_FOCAL_PLANE_TRANSFORM"].requires == ("focal_length_m", "model")
    # And the two records with no requirement at all say so: a field plus nothing
    # else is a complete call for both.
    assert index["O_COMPLEX_TRANSMISSION"].requires == ()
    assert index["C_SCALAR_TO_RAY"].requires == ()
    assert index["M_SPOT_DIAGRAM"].requires == ()


def test_question_4_the_optional_set_is_names_and_not_values() -> None:
    """`diffractive_surface` reports 16; `trace_rays` (`O_RAY_TRACE`) reports none.

    Names only, checked by absence: no `optional` member carries an `=`, a repr or
    anything else that would be a mirrored default. Seventeen mirrored defaults
    would drift against the signature, which is the failure this restraint avoids.
    """
    index = {record.operation_id: record for record in CATALOG}
    assert len(index["O_DIFFRACTIVE_SURFACE"].optional) == 16
    assert index["O_RAY_TRACE"].optional == ()
    assert index["SO_RAY_LAUNCH_TRACE"].optional == ("aiming",)
    for record in CATALOG:
        for name in record.optional:
            assert name.isidentifier(), (record.operation_id, name)


def test_question_5_the_primary_result_is_one_field_access() -> None:
    """No `operation_id` switch anywhere, for any of the thirteen."""
    for record in CATALOG:
        assert record.primary_output == record.returns[0]
        assert record.primary_output in ("ray_bundle", "scalar_field", "psf", "spot")


def test_question_6_auxiliary_returns_are_exactly_the_three_that_have_them() -> None:
    """True for the two couplers and the diffractive surface; false for the other 10.

    `output="ray_bundle"` used to read identically for `propagate_rays`, which
    returns a bundle, and `diffractive_surface`, which returns a 2-tuple. A runtime
    reading only the descriptor would have unpacked the wrong one.
    """
    auxiliary = {record.operation_id for record in CATALOG if record.returns_auxiliary}
    assert auxiliary == AUXILIARY_RETURNERS
    index = {record.operation_id: record for record in CATALOG}
    assert index["C_RAY_TO_SCALAR"].returns == ("scalar_field", "reconstruction_diagnostics")
    assert index["C_SCALAR_TO_RAY"].returns == ("ray_bundle", "sampling_diagnostics")
    assert index["O_DIFFRACTIVE_SURFACE"].returns == ("ray_bundle", "diagnostics")
    assert index["O_PROPAGATE_RAYS"].returns == ("ray_bundle",)
    # No auxiliary name is a semantic type, so none can be mistaken for an edge.
    for record in CATALOG:
        for name in record.returns[1:]:
            assert name not in ("ray_bundle", "scalar_field", "psf"), record.operation_id


def test_question_8_no_record_declares_a_port_by_a_name_alone() -> None:
    """The metadata half of criterion 8; the code half is test_catalog_signatures.py.

    What is checkable here without importing anything: every declared port is a
    representation and never the observable, and a graph entry declares no port at
    all rather than a port it does not have. The claim that these agree with the
    real signatures is the other module's, and it derives rather than restates.
    """
    for record in CATALOG:
        for port in record.inputs:
            assert port in ("ray_bundle", "scalar_field"), record.operation_id
        if record.is_graph_entry:
            assert record.inputs == (), record.operation_id


# ---------------------------------------------------------------------------
# 7. The backend axis. CHE-224 (R15.1).
# ---------------------------------------------------------------------------
#
# Question 9 of the schema's specification, and the three gates that keep it from
# collapsing back into `kind`. Before this ticket a backend was an operation kind,
# which put "who executes" and "what happens to physical state" in one field; the
# three checks below are the ones that make the separation checkable rather than
# a convention.


def test_g1_the_backend_field_agrees_with_the_module_path() -> None:
    """A record's declared backend matches where its implementation lives.

    Declared rather than derived -- `operations/descriptors.py` gives the reason,
    which is `check_dependencies.LANDED`'s -- so this is the agreement check that
    makes the declaration safe to trust.

    **The string is parsed, never imported.** `operations/` has no edge to
    `backends/`, and resolving these paths would load torch and JAX, which is the
    one property the package exists to provide. That is also why this test lives
    here rather than in `test_catalog_resolution.py`.
    """
    for record in CATALOG:
        module = record.implementation.split(":", 1)[0]
        if module.startswith("backends."):
            expected = module.split(".")[1]
            assert record.backend == expected, (
                f"{record.operation_id} implements {module} but declares "
                f"backend={record.backend!r}; the package it lives in says {expected!r}"
            )
        else:
            assert record.backend is None, (
                f"{record.operation_id} declares backend={record.backend!r} but "
                f"implements {module}, which is project-owned code driving no external "
                "library. `None` is what a record with no backend says."
            )

    # Not vacuous in either direction: some records declare a backend and some
    # declare none, so neither branch above is the only one that ever runs.
    declared = {r.backend for r in CATALOG}
    assert declared == {None, "optiland", "chromatix"}


def test_g2_the_id_prefix_agrees_with_the_kind() -> None:
    """`S_` source, `O_` physical operator, `C_` coupler, `M_` measurement.

    The defect this closes is that `S_` used to mean two things: `S_RAY_OPTILAND`
    was `S_` for solver and `S_SOURCE_PLANE_WAVE` was `S_` for source, and `kind`
    read `solver` for both, so nothing could tell them apart. Once `SOURCE` exists
    the prefix can carry one meaning, and `S_RAY_OPTILAND_BUNDLE` -- a `ray_bundle`
    port, so not a source under any reading -- was renamed to `O_RAY_TRACE` in the
    same change rather than left as the defect from the other side.
    """
    primitive = {
        "S_": OperationKind.SOURCE,
        "O_": OperationKind.PHYSICAL_OPERATOR,
        "C_": OperationKind.COUPLER,
        "M_": OperationKind.MEASUREMENT,
    }
    #: Composite prefixes, spelled as the stages they fuse. CHE-225 (R15.2) added
    #: `SO_`, source-then-operator; CHE-226 (R16) adds `SOM_`, the native spot
    #: analysis, which generates rays, traces them and reduces them in one call. A
    #: composite prefix is NOT a fifth kind -- every stage is a primitive from the
    #: enum, and the prefix is the stages spelled in order.
    composite = {
        "SO_": (OperationKind.SOURCE, OperationKind.PHYSICAL_OPERATOR),
        "SOM_": (
            OperationKind.SOURCE,
            OperationKind.PHYSICAL_OPERATOR,
            OperationKind.MEASUREMENT,
        ),
    }

    for record in CATALOG:
        # The stage prefix is everything up to the first underscore, which is what
        # makes a two-letter and a three-letter composite prefix the same parse.
        # Slicing `[:3]` worked only while every composite prefix was two letters
        # long, and `SOM_RAY...` would have read as the primitive prefix `SO`.
        stage_prefix = record.operation_id.split("_", 1)[0] + "_"
        if stage_prefix in composite:
            assert record.composes == composite[stage_prefix], (
                f"{record.operation_id} has the composite prefix {stage_prefix!r}, which "
                f"means {[k.value for k in composite[stage_prefix]]}, but declares "
                f"{[k.value for k in record.composes]}"
            )
            continue
        prefix = stage_prefix
        assert prefix in primitive, (
            f"{record.operation_id} starts with {prefix!r}, which is not one of "
            f"{sorted(primitive)} and not a declared composite prefix "
            f"{sorted(composite)}"
        )
        assert record.kind is primitive[prefix], (
            f"{record.operation_id} is {record.kind.value} but its {prefix!r} prefix "
            f"says {primitive[prefix].value}"
        )
        assert record.composes == (), (
            f"{record.operation_id} declares a composition "
            f"{[k.value for k in record.composes]} under the single-primitive prefix "
            f"{prefix!r}. A record that fuses stages says so in its prefix."
        )
    # Every kind is actually exercised, so the loop is not passing on three of four.
    assert {r.kind for r in CATALOG} == set(OperationKind)


def test_g3_every_entry_begins_with_a_source_and_every_source_start_is_an_entry() -> None:
    """The `entry_stage` version, CHE-225 (R15.2). Both directions, over the catalog.

    `ENTRY_KINDS` refuses `inputs=()` on a record whose entry stage is not a source
    at construction, so that direction cannot fail. The other direction is what this
    adds: a record that *begins* with a source and also declares a port would
    satisfy every construction check, and it is incoherent -- a source stage
    consumes nothing.

    The `kind`-keyed version of this test could not survive CHE-225, and that is the
    point rather than a weakening: `SO_RAY_LAUNCH_TRACE` is an entry whose `kind` is
    `physical_operator`, so "every entry is a `SOURCE`" is now false while "every
    entry BEGINS with a source" is exactly true. The old wording is what let
    CHE-224 put a false `kind` on a record and still pass.
    """
    source_started = {r.operation_id for r in CATALOG if r.entry_stage is OperationKind.SOURCE}
    entries = {r.operation_id for r in CATALOG if r.is_graph_entry}
    assert source_started == entries == GRAPH_ENTRIES
    for record in CATALOG:
        assert (record.entry_stage is OperationKind.SOURCE) == (record.inputs == ()), (
            record.operation_id,
            record.entry_stage.value,
            record.inputs,
        )
    # And the composite is genuinely one of them, so this is not four plain sources.
    index = {r.operation_id: r for r in CATALOG}
    fused = index["SO_RAY_LAUNCH_TRACE"]
    assert fused.is_graph_entry and fused.kind is OperationKind.PHYSICAL_OPERATOR


def test_the_backend_field_and_the_capability_citation_are_different_questions() -> None:
    """They coincide across all thirteen records, and that is a fact not a rule.

    `backend` names the library that runs; `capabilities` cites a *measured*
    device/dtype row. A backend-driving operation whose device behaviour nobody has
    probed would carry a `backend` and `capabilities=None`, and nothing should
    prevent it -- so the coincidence is asserted as the current state rather than
    enforced as an invariant, and the schema is checked for not conflating them.
    """
    import dataclasses

    fields = {f.name for f in dataclasses.fields(OperationDescriptor)}
    assert {"backend", "capabilities"} <= fields, "two fields, two questions"

    for record in CATALOG:
        if record.capabilities is not None:
            assert record.backend is not None, (
                f"{record.operation_id} cites a measured row but drives no backend, "
                "which is possible in principle and is not the current state"
            )
    assert {r.operation_id for r in CATALOG if r.backend and not r.capabilities} == set()


def test_no_record_and_no_id_says_solver_any_more() -> None:
    """The acceptance criterion, stated over the shipped catalog.

    `solver` is not a kind, not an id prefix and not a package name. The word may
    still appear in prose -- an external ray-tracing library is a solver, and
    `backends.optiland.solver` is a module -- and what must not appear is a *kind*
    or an *identity* claiming it.
    """
    assert "solver" not in {kind.value for kind in OperationKind}
    for record in CATALOG:
        assert "SOLVER" not in record.operation_id, record.operation_id
        assert record.kind.value != "solver"


# ---------------------------------------------------------------------------
# 8. Composition. CHE-225 (R15.2).
# ---------------------------------------------------------------------------
#
# Question 10, and the two gates that keep the composite honest. `composes` exists
# because one landed record's `kind` was otherwise a false claim -- `trace`
# initializes its rays and then refracts them through every surface -- and not
# because composition is an interesting shape to model.


def test_g5_a_composition_is_well_formed_and_kind_is_its_terminal_stage() -> None:
    """The invariant that makes the scalar `kind` mean something on a composite.

    Without it, `kind` on a fused record would be a free choice between its stages,
    and CHE-224's mistake -- picking the first stage and calling the record a source
    -- would be expressible again. `kind` is the TERMINAL stage: where the operation
    leaves the state, and therefore which boundary the output sits at.

    Enforced at construction as well, so this is the catalog-wide half.
    """
    for record in CATALOG:
        if not record.composes:
            continue
        assert len(record.composes) >= 2, (
            f"{record.operation_id} declares a one-stage composition; `()` already "
            "means 'exactly its kind'"
        )
        assert record.composes[-1] is record.kind, (
            f"{record.operation_id} fuses {[k.value for k in record.composes]} but its "
            f"kind is {record.kind.value}; kind is the terminal stage"
        )
        assert all(stage in set(OperationKind) for stage in record.composes), (
            f"{record.operation_id} fuses something that is not a primitive kind"
        )


def test_g5_the_construction_checks_actually_refuse_a_malformed_composition() -> None:
    """The detection half, since every assertion above is over a catalog that passes.

    Three ways a composition can be a false claim, each refused at construction.
    Built as data rather than by editing a real record.
    """
    from tests.operations.test_descriptors import a_descriptor

    # `kind` disagreeing with the last stage -- CHE-224's mistake, made expressible.
    with pytest.raises(ValueError, match="TERMINAL stage"):
        a_descriptor(
            kind=OperationKind.SOURCE,
            inputs=(),
            composes=(OperationKind.SOURCE, OperationKind.PHYSICAL_OPERATOR),
        )
    # A one-stage "composition", which states one fact twice.
    with pytest.raises(ValueError, match="single stage"):
        a_descriptor(kind=OperationKind.COUPLER, composes=(OperationKind.COUPLER,))
    # A stage that is not a primitive kind.
    with pytest.raises(ValueError, match="primitive kinds"):
        a_descriptor(kind=OperationKind.COUPLER, composes=("source", "solver"))


def test_g6_only_the_pinned_records_declare_a_composition() -> None:
    """A composite is a modelling claim, not something to acquire quietly.

    The same discipline CHE-221 applied to "this is the only callable with two
    records". `composes` is cheap to add to a record and expensive to be wrong
    about, so the set is pinned and a new member has to come past this test.

    **CHE-226 (R16) is the second member, and it came past this test rather than
    around it.** `SOM_SPOT_DIAGRAM` is `(source, physical_operator, measurement)`:
    the pinned solver's own spot analysis generates its rays from the declared
    field, refracts them through every surface and reduces the intersections, all
    inside one call. A bare `measurement` kind with `inputs=()` is refused at
    construction -- only a source may begin a graph entry -- so the schema itself
    rejected the simpler claim. Note what is NOT a composite: `M_SPOT_DIAGRAM`,
    the other spot path, consumes a `RayBundle` and observes it, which is one
    primitive stage and the whole truth about it.

    `O_DIFFRACTIVE_SURFACE` is the interesting exclusion and it is deliberate: it
    is internally coupler -> operator -> coupler (it imports `couplers` and
    `operators.transmission`, and its `approximation` describes accumulate ->
    transmit -> decompose), but its input and output representation types do not
    change and it presents a single operator-like transformation at its boundary, so
    its net primitive kind is the whole truth about it at the ports. Whether it
    should nonetheless expose that structure is a recorded follow-up design
    question -- see the `composes` field docstring -- and not a defect this gate is
    tolerating.
    """
    composites = {r.operation_id for r in CATALOG if r.composes}
    assert composites == {"SO_RAY_LAUNCH_TRACE", "SOM_SPOT_DIAGRAM", "SOM_PSF"}, composites
    index = {r.operation_id: r for r in CATALOG}
    assert index["O_DIFFRACTIVE_SURFACE"].composes == ()
    assert index["M_SPOT_DIAGRAM"].composes == ()
    # And the other thirteen fuse nothing.
    assert len([r for r in CATALOG if not r.composes]) == 14


def test_the_fused_record_says_what_it_fuses_and_why_that_is_not_a_source() -> None:
    """The record CHE-225 exists for, pinned against being quietly re-collapsed.

    `S_RAY_OPTILAND` claimed `kind=SOURCE` for a callable that refracts through
    every surface. What makes the claim checkably false is the record's own
    `approximation`, which describes a state change -- so this asserts the two
    agree now rather than contradict.
    """
    index = {r.operation_id: r for r in CATALOG}
    assert "S_RAY_OPTILAND" not in index, (
        "the collapsed record is back. `trace` initializes rays AND evolves them; "
        "declaring either half alone is a false claim."
    )
    fused = index["SO_RAY_LAUNCH_TRACE"]
    assert fused.composes == (OperationKind.SOURCE, OperationKind.PHYSICAL_OPERATOR)
    assert fused.kind is OperationKind.PHYSICAL_OPERATOR
    assert fused.entry_stage is OperationKind.SOURCE
    assert fused.implementation == "backends.optiland.solver:trace"
    assert fused.backend == "optiland"
    assert fused.capabilities == "M_RAY_OPTILAND"
    # The prose that made the old `kind` falsifiable, still present and still
    # describing a state change.
    assert "refraction at a real interface" in fused.approximation
