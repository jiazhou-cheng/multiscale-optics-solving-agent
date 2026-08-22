"""The invariants CHE-84 established, as checks rather than as a memory.

Most of the epic's invariants already have a home -- import direction in
`test_package_dependencies.py`, flat-layout integrity in `test_flat_layout.py`,
registry honesty in `test_registry_matches_capabilities.py`, knowledge integrity
in `test_solver_knowledge_pack.py`, generated artifacts in
`test_generated_artifacts.py`. This file is the sweep: the checks that only
become possible once the whole structure exists, plus the ones no earlier phase
had a natural place for.

Design follows `tests/test_suite_layout.py`, whose docstring makes the argument:
pin *boundaries*, not inventories. "A guard that pins the file list would fail
on every legitimate change until someone edited it -- which teaches people to
edit it, defeating the guard." So nothing here asserts that a particular file,
test or module exists; deleting or renaming one is normal work.

Static by construction -- AST and the file tree, no solver import. The fast
subset is ~44 s and this file must not move it: a guard that spawns a pytest
importing jax, chromatix and optiland to re-derive facts the file tree already
determines is the wrong trade.
"""

from __future__ import annotations

import ast
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


# ---------------------------------------------------------------------------
# Registry honesty: registration is a declaration, not a filename
# ---------------------------------------------------------------------------

def _module_level_assignment(path: Path, name: str) -> str | None:
    """The value a module assigns to `name` at module scope, if it is a literal."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if (
                    isinstance(target, ast.Name)
                    and target.id == name
                    and isinstance(node.value, ast.Constant)
                    and isinstance(node.value.value, str)
                ):
                    return node.value.value
    return None


def test_the_registration_map_equals_the_set_of_modules_declaring_a_model_id() -> None:
    """Two ways of saying "this is an adapter" must not disagree.

    CHE-87 replaced filename discovery with an explicit map. That fixed
    over-inclusion -- three benchmark harnesses ending in `_adapter.py` were
    imported on every lookup while registering nothing -- but introduced the
    opposite risk: a module that declares a `MODEL_ID` and is simply forgotten
    in the map is now invisible, with no error anywhere.
    """
    from solvers.registry import _REGISTRATIONS

    declared = {
        model_id: path
        for path in sorted(SRC.rglob("*.py"))
        if (model_id := _module_level_assignment(path, "MODEL_ID")) is not None
    }
    registered = {model_id for model_id, _ in _REGISTRATIONS}

    unregistered = {
        mid: str(p.relative_to(ROOT)) for mid, p in declared.items() if mid not in registered
    }
    assert not unregistered, (
        f"these modules declare a MODEL_ID and are not in solvers/registry.py's "
        f"_REGISTRATIONS: {unregistered}. An adapter nothing registers is "
        "unreachable, and nothing else will say so."
    )
    phantom = registered - set(declared)
    assert not phantom, (
        f"_REGISTRATIONS names {sorted(phantom)}, which no module declares. "
        "get_adapter_for_model already raises on this, but at call time -- which "
        "is later than it needs to be."
    )


def test_a_duplicate_registration_raises_rather_than_resolving_silently() -> None:
    """The failure mode filename scanning could not detect at all.

    Under the old `pkgutil` scan, two modules claiming one id were resolved by
    whichever came later in directory order.
    """
    import solvers.registry as registry_module

    original = registry_module._REGISTRATIONS
    registry_module._registry.cache_clear()
    try:
        registry_module._REGISTRATIONS = (
            ("M_DUPLICATE", "solvers.optiland.adapter"),
            ("M_DUPLICATE", "solvers.chromatix.adapter"),
        )
        with pytest.raises(RuntimeError, match="Duplicate adapter registration"):
            registry_module.available_model_ids()
    finally:
        registry_module._REGISTRATIONS = original
        registry_module._registry.cache_clear()


def test_no_module_outside_solvers_is_named_like_an_adapter() -> None:
    """The naming trap, closed at the source rather than at the discovery site.

    Three gen1 benchmark harnesses were named `*_adapter.py`, which under
    filename discovery made them importable entry points that registered
    nothing. CHE-87 stopped discovery reading filenames and CHE-88 archived the
    files, but the *name* is still misleading to a human, and a future one could
    reintroduce the confusion in a directory the map does not cover.
    """
    offenders = [
        str(path.relative_to(ROOT))
        for path in sorted(SRC.rglob("*_adapter.py"))
        if not path.is_relative_to(SRC / "solvers")
    ]
    assert not offenders, (
        f"{offenders} are named like adapters and are not under solvers/. "
        "Name a module for what it is: a benchmark harness that ends in "
        "`_adapter.py` reads as an entry point to every human who sees it, and "
        "that is how three of them ended up being imported on every lookup."
    )


# ---------------------------------------------------------------------------
# Generated and declared artifacts
# ---------------------------------------------------------------------------

def test_every_example_graph_validates_against_the_packaged_registry() -> None:
    """`examples/graphs/` is documentation that executes.

    `scripts/validate_package.py` checks this, but a script nobody runs in the
    suite is a check nobody runs. CHE-87 had to delete two example graphs in the
    same commit as the registry entries they named, precisely because this
    coupling is real.
    """
    from core.graph import GraphValidator
    from registry.loader import Registry

    registry = Registry.from_package()
    validator = GraphValidator(registry)
    graphs = sorted((ROOT / "examples" / "graphs").glob("*.yaml"))
    assert graphs, "examples/graphs/ is empty"

    for path in graphs:
        report = validator.validate(Registry.load_graph(path))
        assert report.valid, (
            f"{path.relative_to(ROOT)} does not validate:\n"
            + "\n".join(f"  {issue.code}: {issue.message}" for issue in report.errors)
        )


# ---------------------------------------------------------------------------
# Scope of the checkers themselves
# ---------------------------------------------------------------------------

def test_the_repository_checkers_do_not_scan_the_frozen_archives() -> None:
    """A checker that walks `archive/` reports failures nobody may fix.

    The frozen trees are preserved with their stale paths on purpose. Before
    CHE-86, `validate_package.py` walked `ROOT.rglob` with no exclusions and
    validated `archive/`, `docs/archive/` and gitignored scratch -- so its
    "all YAML files" claim covered material whose contents must not change.
    """
    source = (ROOT / "scripts" / "validate_package.py").read_text(encoding="utf-8")
    assert "EXCLUDED_DIRS" in source and '"archive"' in source, (
        "scripts/validate_package.py no longer excludes the frozen trees."
    )
    assert "--ignored" in source, (
        "the ignore list is no longer derived from git. Reimplementing "
        ".gitignore semantics drifts from the file it mirrors; asking git cannot."
    )


def test_no_active_file_needs_a_ticket_id_to_be_understood() -> None:
    """CHE-92's rule, checked in the form that is actually checkable.

    Ticket IDs are legitimate in three places: a structured `verified_by:` or
    `issue:` field, a historical report, and an attribution standing beside a
    stated finding. What is forbidden is a reference that *is* the explanation --
    "see CHE-N", with what CHE-N found left in Linear.

    This does not count references. There are over a thousand and counting them
    would pin an inventory, which `test_suite_layout.py` argues against at
    length. It matches the shape of a load-bearing citation, which is the thing
    that makes a document unreadable without an account on another service.
    """
    load_bearing = re.compile(
        r"(?:^|[\s(])(?:see|per|cf\.?|from|following)\s+CHE-\d+\s*(?:[.;,)\]]|$)",
        re.IGNORECASE,
    )
    # Historical records legitimately cite the ticket that produced them.
    exempt = (
        "archive/",
        "docs/archive/",
        "benchmarks/reports/",
        "benchmarks/protocols/",
        "benchmarks/roadmap.md",
        "docs/architecture/cleanup_baseline.md",
    )
    offenders: list[str] = []
    for base in ("src", "tests", "knowledge", "docs", "benchmarks", "examples", "scripts"):
        for path in sorted((ROOT / base).rglob("*")):
            if not path.is_file() or path.suffix not in {".py", ".md", ".yaml", ".yml"}:
                continue
            rel = path.relative_to(ROOT).as_posix()
            if rel.startswith(exempt) or rel == "tests/test_architecture_invariants.py":
                continue
            for lineno, line in enumerate(
                path.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1
            ):
                if load_bearing.search(line):
                    offenders.append(f"{rel}:{lineno}: {line.strip()[:90]}")
    assert not offenders, (
        "these cite a ticket instead of stating what it found:\n  "
        + "\n  ".join(offenders)
        + "\n\nReplace the reference with the measurement: what was measured, on "
        "which version and device, the observed value, the tolerance, the record "
        "path, and the test that pins it. A reader without Linear access should "
        "not be blocked."
    )
