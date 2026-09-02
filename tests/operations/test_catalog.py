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


def test_one_record_per_implementation_and_kind_not_per_implementation() -> None:
    """Acceptance criteria 4 and 9, which conflict literally and are reconciled here.

    Criterion 4 asks for "exactly one record" per declared operation; criterion 9
    requires `S_WAVE_CHROMATIX` and `O_ASM_PROPAGATE` to be two records over one
    callable. Both hold once uniqueness is keyed on `(implementation, kind)`
    instead of on `implementation`: a callable may carry one record per kind -- the
    backend it is, and the physical operation it performs -- and a second record of
    the *same* kind over the same callable is the accidental duplicate the
    criterion is protecting against.
    """
    seen: dict[tuple[str, OperationKind], str] = {}
    for record in CATALOG:
        key = (record.implementation, record.kind)
        previous = seen.get(key)
        assert previous is None, (
            f"{record.operation_id} and {previous} are both {record.kind.value} records "
            f"over {record.implementation}. One callable may carry one record per KIND; "
            "two of one kind is a duplicate."
        )
        seen[key] = record.operation_id


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


def test_two_records_may_share_one_callable_and_stay_distinct() -> None:
    """Acceptance criterion 9, pinned as intended rather than tolerated.

    One answers "what backend does this project drive, and in which measured
    capability row"; the other answers "what happens to the physical state".
    Neither is a coupler, and that is the substantive half: a coupler changes
    representation while preserving physical state, and this changes physical state
    while preserving the representation.
    """
    index = {record.operation_id: record for record in CATALOG}
    solver = index["S_WAVE_CHROMATIX"]
    operator = index["O_ASM_PROPAGATE"]

    assert solver.implementation == operator.implementation
    assert solver.kind is OperationKind.SOLVER
    assert operator.kind is OperationKind.PHYSICAL_OPERATOR
    assert OperationKind.COUPLER not in {solver.kind, operator.kind}
    assert solver.approximation != operator.approximation
    assert solver.validity != operator.validity

    # And this is the ONLY callable with two records. Without this, the
    # `(implementation, kind)` key above would let any callable acquire a second
    # record under a different kind silently, which is the loophole the
    # "exactly one record" wording was guarding. Sharing a callable is a real
    # modelling claim -- that one function is both a backend and a physical
    # operation -- and it should have to be argued into this set.
    counts: dict[str, int] = {}
    for record in CATALOG:
        counts[record.implementation] = counts.get(record.implementation, 0) + 1
    assert {name for name, n in counts.items() if n > 1} == {
        "backends.chromatix.solver:propagate"
    }


def test_the_three_operations_that_had_no_descriptor_now_have_one() -> None:
    """The gap CHE-221 measured: `trace_rays`, `gaussian_beam`, `spherical_wave`.

    Eleven descriptors existed in test fixtures and three landed operations had
    none at all. Named individually here because "14 records" would pass with the
    wrong fourteen.
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

    Only the two operations that drive an external backend cite a row. Everything
    else is `None`, which is the honest citation rather than a missing one: a
    coupler runs in whatever namespace the field it was handed carries, so citing
    the chromatix row would claim a measurement taken about something else.
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
# 6. The eight planner questions, answered from metadata alone. CHE-222 (R03.5).
# ---------------------------------------------------------------------------
#
# Each of these is one of the eight questions `operations/descriptors.py` names as
# the specification for the schema's field list. They are asserted against the
# **shipped catalog**, from metadata only -- no signature inspection, no import of
# an implementation -- because that is the situation a planner is in.
#
# `tests/operations/test_catalog_signatures.py` is the other half: it derives the
# same four tuples from `inspect.signature` and compares, so what is asserted here
# as metadata is separately known to match the code.

#: The four graph entries: the three analytic sources, and the problem-driven ray
#: solve. Written out because "four" would pass with the wrong four.
GRAPH_ENTRIES = {
    "S_RAY_OPTILAND",
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
    assert len(CATALOG) - len(entries) == 10
    # Every entry is `solver`-kind, which is what `ENTRY_KINDS` enforces.
    for record in CATALOG:
        if record.is_graph_entry:
            assert record.kind is OperationKind.SOLVER, record.operation_id
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
    supplied = index["S_RAY_OPTILAND_BUNDLE"]
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
    """Twelve of the fourteen need a value the old schema never mentioned.

    The ticket says nine, from a table written before three of these records
    existed; the measured figure is twelve, and the two exceptions are named at the
    bottom of this test. Asserted as the exact set rather than a count, because "n
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
        "O_PROPAGATE_RAYS",
        "S_RAY_OPTILAND",
        "S_RAY_OPTILAND_BUNDLE",
        "S_SOURCE_GAUSSIAN_BEAM",
        "S_SOURCE_PLANE_WAVE",
        "S_SOURCE_SPHERICAL_WAVE",
        "S_WAVE_CHROMATIX",
        "C_RAY_TO_SCALAR",
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


def test_question_4_the_optional_set_is_names_and_not_values() -> None:
    """`diffractive_surface` reports 16; `trace_rays` reports none.

    Names only, checked by absence: no `optional` member carries an `=`, a repr or
    anything else that would be a mirrored default. Seventeen mirrored defaults
    would drift against the signature, which is the failure this restraint avoids.
    """
    index = {record.operation_id: record for record in CATALOG}
    assert len(index["O_DIFFRACTIVE_SURFACE"].optional) == 16
    assert index["S_RAY_OPTILAND_BUNDLE"].optional == ()
    assert index["S_RAY_OPTILAND"].optional == ("aiming",)
    for record in CATALOG:
        for name in record.optional:
            assert name.isidentifier(), (record.operation_id, name)


def test_question_5_the_primary_result_is_one_field_access() -> None:
    """No `operation_id` switch anywhere, for any of the fourteen."""
    for record in CATALOG:
        assert record.primary_output == record.returns[0]
        assert record.primary_output in ("ray_bundle", "scalar_field", "psf")


def test_question_6_auxiliary_returns_are_exactly_the_three_that_have_them() -> None:
    """True for the two couplers and the diffractive surface; false for the other 11.

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
