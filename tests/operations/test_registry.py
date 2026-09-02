"""Query and resolution behaviour, over a synthetic index rather than the catalog.

CHE-178 (R03.2), rewired by CHE-221 (R03.4). The criteria covered here:

1. discovery is declared -- no filename discovery, no import-time scan;
3. `resolve` is the only function that imports an implementation;
5. a dummy operation is found by input, by output and by kind, and resolves lazily.

Criterion 2 (`sys.modules` free of every backend after enumerating everything) is
in `test_registry_imports_no_backend.py`, which needs a fresh interpreter to mean
anything. Criterion 4 (the dependency direction) is the R01.1 gate.

What CHE-221 changed in this file
---------------------------------
`register()` is gone: the registration site moved into `operations.catalog`, so
the index is *derived* at import from `CATALOG` and there is no call that adds to
it. Three tests went with it -- the ones that asserted `register` returns its
argument, refuses a duplicate id and refuses a non-descriptor. The behaviours did
not go: `registry._build_index` kept the duplicate refusal and the type refusal,
and `tests/operations/test_catalog.py` asserts both against the real catalog,
which is a stronger place for them than a synthetic dummy was.

`test_the_shipped_registry_is_empty` also went, because the shipped index is no
longer empty. Its replacement is
`test_importing_operations_populates_the_catalog` below: the property actually
worth pinning was never emptiness, it was that whatever is there arrives at import
without a registration call and without a backend.

**Why the index is swapped for a synthetic dict** rather than tested against the
production catalog: the subject here is `find`/`resolve` *behaviour* -- id
ordering, port filtering, laziness, four distinct error paths -- and a dummy
descriptor pointing at a generated module is what makes the laziness assertion
mean something. Reaching into `registry._BY_ID` is private-state access and it is
confined to this directory on purpose; `tests/operations/test_catalog.py` asserts
that nothing outside `tests/operations/` does it.

**Why the dummy implementation is written to a temp directory** rather than
committed as a fixture module: the assertion is that a module is *absent* from
`sys.modules` until `resolve` is called, and any committed module is one stray
import away from being loaded by something else in the session. Generating it per
test gives a name nothing else in the process could have touched.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from operations import descriptors as descriptors_module
from operations import registry
from operations.descriptors import OperationDescriptor, OperationKind

MODULE_SOURCE = '''
"""A stand-in for a backend-heavy implementation module."""

IMPORTED = True


def run(value):
    return f"ran:{value}"


NOT_CALLABLE = 3
'''


@pytest.fixture(autouse=True)
def synthetic_index(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Swap the catalog-derived index for an empty one, for the duration of a test.

    The index is module-level state on purpose (there is one process and one set of
    operations), so the isolation belongs here rather than in a `Registry` object
    nothing else needs. `monkeypatch.setattr` rather than clear-and-restore,
    because the real index is no longer mutable by anything and should not start
    being mutated by a fixture.
    """
    monkeypatch.setattr(registry, "_BY_ID", {})
    yield


def put(descriptor: OperationDescriptor) -> OperationDescriptor:
    """Place one descriptor in the synthetic index, and return it.

    What `register()` used to be, for this file's purposes only. Deliberately not
    a public function on `registry`: the production index is derived from the
    catalog, and a second way to add to it is exactly what CHE-221 removed.
    """
    registry._BY_ID[descriptor.operation_id] = descriptor
    return descriptor


@pytest.fixture
def lazy_module(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> str:
    """An importable module name that nothing has imported yet."""
    name = f"dummy_impl_{abs(hash(request.node.nodeid)) % 10**8}"
    (tmp_path / f"{name}.py").write_text(MODULE_SOURCE, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, name, raising=False)
    assert name not in sys.modules
    return name


def a_descriptor(**overrides: object) -> OperationDescriptor:
    fields: dict[str, object] = {
        "operation_id": "X_DUMMY",
        "kind": OperationKind.COUPLER,
        "input": "ray_bundle",
        "output": "scalar_field",
        "implementation": "somewhere:run",
        "approximation": "none; a dummy operation for the registry tests",
        "evidence": ("tests/operations/test_registry.py",),
    }
    fields.update(overrides)
    return OperationDescriptor(**fields)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Criterion 1 -- explicit registration
# ---------------------------------------------------------------------------


def test_importing_operations_populates_the_catalog() -> None:
    """What replaced `test_the_shipped_registry_is_empty`, and why.

    Emptiness was never the property worth pinning -- it was an accident of no
    operation having landed, and `operations.find()` returning `()` for a project
    that can trace rays and propagate fields was the debt CHE-221 repaid. What
    matters is that the index arrives *at import*, from a declaration inside the
    package, with no registration call and no backend loaded.

    Run in a fresh interpreter because the autouse fixture above has already
    replaced the index by the time this test body runs, and because the claim is
    about what importing does.
    """
    import json
    import subprocess

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import operations, json, sys; print(json.dumps({"
            "'ids': list(operations.registered_ids()), "
            "'catalog': len(operations.CATALOG), "
            "'backends': sorted({m.split('.')[0] for m in sys.modules} & "
            "{'jax', 'jaxlib', 'torch', 'optiland', 'chromatix'})}))",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    report = json.loads(completed.stdout)
    assert report["ids"], "importing operations left the index empty"
    assert len(report["ids"]) == report["catalog"] == 14
    assert report["ids"] == sorted(report["ids"]), "find() and registered_ids() sort by id"
    assert report["backends"] == [], report["backends"]


def test_there_is_no_registry_class_and_no_directory_scan() -> None:
    assert not hasattr(registry, "Registry")
    assert not hasattr(registry, "ComponentIndex")
    source = Path(registry.__file__).read_text(encoding="utf-8")
    for scan in ("rglob", "glob(", "iterdir", "pkgutil", "walk_packages", "entry_points"):
        assert scan not in source, (
            f"{scan!r} in registry.py is discovery. A module is registered because a "
            "registration site names it."
        )


# ---------------------------------------------------------------------------
# Criterion 5 -- found by port and by kind
# ---------------------------------------------------------------------------


def test_find_with_no_filter_enumerates_everything_in_id_order() -> None:
    put(a_descriptor(operation_id="X_SECOND"))
    put(a_descriptor(operation_id="X_FIRST"))
    assert [d.operation_id for d in registry.find()] == ["X_FIRST", "X_SECOND"]


def test_find_by_input_output_and_kind() -> None:
    forward = put(a_descriptor(operation_id="X_FORWARD"))
    backward = put(
        a_descriptor(
            operation_id="X_BACKWARD",
            input="scalar_field",
            output="ray_bundle",
            kind=OperationKind.PHYSICAL_OPERATOR,
        )
    )
    assert registry.find(input="ray_bundle") == (forward,)
    assert registry.find(output="ray_bundle") == (backward,)
    assert registry.find(kind="coupler") == (forward,)
    assert registry.find(kind=OperationKind.PHYSICAL_OPERATOR) == (backward,)
    assert registry.find(input="scalar_field", output="ray_bundle") == (backward,)
    assert registry.find(input="ray_bundle", kind="physical_operator") == ()


def test_a_typo_in_a_query_is_an_error_not_an_empty_result() -> None:
    """The failure this guards: `find(input="rays")` reading as "no such capability"."""
    put(a_descriptor())
    with pytest.raises(ValueError, match="semantic type"):
        registry.find(input="rays")
    with pytest.raises(ValueError, match="semantic type"):
        registry.find(output="mueller_matrix")
    with pytest.raises(ValueError, match="physical_operator"):
        registry.find(kind="propagator")


def test_the_query_vocabulary_is_the_descriptor_vocabulary() -> None:
    """One source: `find` filters on the same tuple a descriptor validates against."""
    assert registry.SEMANTIC_TYPES is descriptors_module.SEMANTIC_TYPES


# ---------------------------------------------------------------------------
# Criterion 3 -- resolve is the only importing call
# ---------------------------------------------------------------------------


def test_registering_and_finding_do_not_import_the_implementation(lazy_module: str) -> None:
    put(a_descriptor(implementation=f"{lazy_module}:run"))
    assert lazy_module not in sys.modules

    found = registry.find(input="ray_bundle", output="scalar_field", kind="coupler")
    assert [d.operation_id for d in found] == ["X_DUMMY"]
    assert found[0].implementation == f"{lazy_module}:run"
    assert lazy_module not in sys.modules, (
        "listing the registry imported the implementation. This is the property the "
        "whole layer exists for."
    )


def test_resolve_imports_and_returns_the_callable(lazy_module: str) -> None:
    put(a_descriptor(implementation=f"{lazy_module}:run"))
    implementation = registry.resolve("X_DUMMY")
    assert lazy_module in sys.modules
    assert callable(implementation)
    assert implementation("x") == "ran:x"


def test_resolving_an_unregistered_id_names_what_is_registered() -> None:
    put(a_descriptor())
    with pytest.raises(KeyError) as caught:
        registry.resolve("X_MISSING")
    assert "X_DUMMY" in str(caught.value)


def test_resolving_a_module_that_does_not_exist_says_which() -> None:
    put(a_descriptor(implementation="no_such_module_at_all:run"))
    with pytest.raises(ImportError, match="no_such_module_at_all"):
        registry.resolve("X_DUMMY")


def test_resolving_a_missing_attribute_says_which(lazy_module: str) -> None:
    put(a_descriptor(implementation=f"{lazy_module}:absent"))
    with pytest.raises(AttributeError, match="absent"):
        registry.resolve("X_DUMMY")


def test_resolving_something_that_is_not_callable_is_refused(lazy_module: str) -> None:
    put(a_descriptor(implementation=f"{lazy_module}:NOT_CALLABLE"))
    with pytest.raises(TypeError, match="not callable"):
        registry.resolve("X_DUMMY")
