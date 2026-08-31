"""The dependency direction of the new production tree, as an executable gate.

CHE-171 (R01.1). The reference implementation had this same rule as a test
(`tests/test_package_dependencies.py`) and the AST-walk mechanism there is
reused; none of its asserted package names are. Two things are deliberately
different:

**This is an allowlist, not a denylist.** The old check enumerated, per package,
what it must *not* import. That fails open: a package added later is unconstrained
until someone remembers to add it to every other package's forbidden set. Here
each package declares what it *may* import and everything else is a violation, so
a new package is forbidden by default and has to be argued into the graph.

**It refuses to pass vacuously.** The failure mode this script is most likely to
have is walking zero files and reporting success, because at R01 the new tree is
nearly empty. `verify()` therefore fails when a declared package is missing from
disk or when no module was inspected at all. A gate that cannot fail is not a
gate, and this one is landed before the code it guards precisely so it is in place
when the code arrives.

Run directly (`python scripts/check_dependencies.py`) for a report, or through
`tests/unit/test_dependency_direction.py`, which is what puts it in the default
suite.
"""

from __future__ import annotations

import ast
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

#: What each package of the new tree may import from this project.
#:
#: The direction is the project's, not this script's. Two entries carry the
#: reasoning that is easiest to get wrong:
#:
#: * `numerics` is empty. It is the bottom of the graph, and anything it imported
#:   would become part of the bottom of the graph.
#: * `operations` may import `numerics` and nothing else -- in particular not
#:   `solvers` and not `couplers`. It holds import paths as *strings* and resolves
#:   them lazily, which is what makes "reading the registry imports no backend" a
#:   structural fact rather than a discipline someone maintains.
ALLOWED: dict[str, frozenset[str]] = {
    "numerics": frozenset(),
    "representations": frozenset({"numerics"}),
    "problems": frozenset({"representations", "numerics"}),
    "operations": frozenset({"numerics"}),
    "solvers": frozenset({"problems", "representations", "numerics"}),
    "couplers": frozenset({"representations", "numerics"}),
    "operators": frozenset({"representations", "couplers", "numerics"}),
    "measurements": frozenset({"representations", "numerics"}),
    "planning": frozenset({"operations"}),
    "runtime": frozenset({"planning", "operations", "representations"}),
}

#: Which packages of the new tree have actually been authored.
#:
#: This cannot be derived from the filesystem, and that is the whole subtlety of
#: the migration: `src/couplers/`, `src/solvers/` and `src/runtime/` exist right
#: now and belong to the **old** tree, whose names the new architecture reuses.
#: Detecting "present" by directory existence made this gate walk 52 old modules
#: and report 50 violations that were really just old code importing old code.
#:
#: So the migration state is declared. R01 lands `numerics` and `representations`
#: only -- the minimum for the first real implementation (R02), and the two target
#: names that do not collide with the old tree. Every later ticket adds its package
#: here as it authors it; R14 deletes the old tree, after which this equals
#: `ALLOWED.keys()`.
LANDED: frozenset[str] = frozenset({"numerics", "representations"})

#: Names declared by the new architecture that the old tree currently occupies.
#: An import of one of these from a landed package would resolve to old code, so
#: `_classify` refuses it by name rather than letting it silently work.
#:
#: **A known blind spot, stated rather than left to be discovered.** These three
#: names are exempt from the "on disk but not in LANDED" structural check below,
#: because they are on disk as *old* code and always will be until R14. So when
#: R05/R09/R12 author new-tree code under `src/couplers/`, `src/solvers/` or
#: `src/runtime/`, forgetting to add the name to LANDED raises no error and both
#: gates then walk and count *zero* of that new code. The ticket that lands one of
#: these must add it to LANDED in the same change. R14 empties this set, after
#: which the exemption -- and this hazard -- disappear.
SHARED_NAMES: frozenset[str] = frozenset({"couplers", "solvers", "runtime"})

#: Top-level names in `src/` that belong to the old tree and to nothing else.
#: An import of any of these from the new tree is the violation R01 exists to
#: prevent: the new implementation must not be built on the tree it replaces.
OLD_TREE_ONLY: frozenset[str] = frozenset(
    {"agent", "cli", "core", "discovery", "registry", "studies", "verification"}
)

#: `src` itself, which is the shortest way to defeat everything above.
#:
#: `src/` has no `__init__.py`, so PEP 420 makes `src.core`, `src.couplers` and
#: `src.solvers` importable namespace paths whenever the repository root is on
#: `sys.path` -- which it is for a bare `python`, `python -m` or pytest run from
#: the root. `from src.core.boundary import RayBundle` therefore *resolves*, and an
#: earlier revision of this script classified its first segment (`"src"`) as
#: third-party and passed it. Review caught it; this closes it. Nothing in either
#: tree should ever import through `src.`: the namespace root is on the path, so
#: the package name alone is the import.
NAMESPACE_ROOT = "src"

#: Solver backends. Only `solvers/<backend>/` may import one. A representation or
#: a coupler that imports a backend has stopped being neutral ground -- this is
#: the "representations/ -> any backend" forbidden edge, generalized to every
#: package that is not a solver adapter.
BACKENDS: frozenset[str] = frozenset({"optiland", "chromatix"})


@dataclass(frozen=True)
class Violation:
    module: str
    imported: str
    rule: str

    def __str__(self) -> str:
        return f"  {self.module}\n      imports {self.imported!r} -- {self.rule}"


def _imports(path: Path) -> set[str]:
    """Top-level names imported by one module, from its AST.

    A relative import (`level > 0`) cannot cross above its own package root, so
    it is not this check's concern.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            found.add(node.module.split(".")[0])
    return found


def _modules_of(package: str) -> list[Path]:
    base = SRC / package
    return sorted(p for p in base.rglob("*.py") if "__pycache__" not in str(p))


def _classify(package: str, imported: str) -> str | None:
    """Return the rule broken by `package` importing `imported`, or None."""
    if imported == NAMESPACE_ROOT:
        return (
            "'src' is the namespace root, not a package. `src.core...` resolves through "
            "PEP 420 whenever the repository root is on sys.path, so importing through it "
            "reaches the OLD tree while naming none of it -- which is how this check was "
            "bypassed before. Import the package directly; the root is already on the path."
        )
    if imported in OLD_TREE_ONLY:
        return (
            f"{imported!r} is the old production tree. The new implementation must not "
            "be built on the tree it replaces (R01 acceptance criterion 3); reuse the "
            "physics by re-deriving it from the pre-rewrite tag, not by importing it."
        )
    if imported in BACKENDS and package != "solvers":
        return (
            f"{imported!r} is a solver backend. Only solvers/<backend>/ may import one; "
            "everywhere else it makes the package depend on an external solver's "
            "conventions."
        )
    if imported not in ALLOWED:
        return None  # third-party or stdlib; not this check's business.
    allowed = ALLOWED[package]
    if imported == package:
        return None
    if imported not in allowed:
        return (
            f"{package}/ may import {sorted(allowed) or 'nothing in this project'}. "
            f"{imported!r} is not in that set, and this is an allowlist: a new edge has "
            "to be argued into ALLOWED, not assumed."
        )
    if imported not in LANDED:
        collides = " The old tree still owns that name." if imported in SHARED_NAMES else ""
        return (
            f"{package}/ -> {imported}/ is an allowed direction, but {imported!r} does not "
            "exist in the new tree yet, so the import would resolve to the OLD "
            f"src/{imported}/.{collides} Land the new package first."
        )
    return None


def verify() -> tuple[list[Violation], list[str], int]:
    """Check every module of the new tree.

    Returns the violations, the structural problems that would make a pass
    meaningless, and the number of modules actually inspected.
    """
    violations: list[Violation] = []
    structural: list[str] = []
    inspected = 0

    if not LANDED:
        structural.append(
            "no package of the new tree has been landed, so this check would pass "
            "without inspecting anything. Declared packages: " + ", ".join(sorted(ALLOWED))
        )

    # LANDED is hand-maintained, so it is checked rather than trusted. Three ways
    # it can go wrong, all of which make the gate quietly weaker:
    undeclared = ALLOWED.keys() - LANDED - SHARED_NAMES
    for name in sorted(undeclared):
        if (SRC / name / "__init__.py").is_file():
            structural.append(
                f"src/{name}/ exists and is a new-architecture package name, but is not in "
                "LANDED, so nothing checks its imports. Add it to LANDED."
            )
    # Every top-level name under src/ must be accounted for by one of the three
    # sets, derived from the tree rather than trusted. An unaccounted name is
    # classified as third-party and silently importable -- the same failure shape
    # as the `src.` bypass.
    on_disk = {p.name for p in SRC.iterdir() if p.is_dir() and (p / "__init__.py").is_file()}
    on_disk |= {p.stem for p in SRC.glob("*.py")}
    unaccounted = on_disk - OLD_TREE_ONLY - SHARED_NAMES - LANDED
    for name in sorted(unaccounted):
        structural.append(
            f"src/{name}/ is a top-level name this check does not classify, so an import of "
            f"it from the new tree would be treated as third-party and allowed. Add it to "
            "OLD_TREE_ONLY, SHARED_NAMES or LANDED."
        )

    for name in sorted(LANDED - ALLOWED.keys()):
        structural.append(f"LANDED names {name!r}, which has no entry in ALLOWED")
    for name in sorted(LANDED):
        if not (SRC / name / "__init__.py").is_file():
            structural.append(f"LANDED names {name!r}, but src/{name}/__init__.py does not exist")

    for package in sorted(LANDED):
        modules = _modules_of(package)
        if not modules:
            structural.append(f"src/{package}/ has no Python module, not even __init__.py")
        for module in modules:
            inspected += 1
            name = str(module.relative_to(ROOT))
            for imported in sorted(_imports(module)):
                rule = _classify(package, imported)
                if rule is not None:
                    violations.append(Violation(name, imported, rule))

    if inspected == 0 and not structural:
        structural.append("zero modules inspected; a check that walks nothing cannot fail")

    return violations, structural, inspected


def _report() -> int:
    violations, structural, inspected = verify()
    print(f"new-tree packages present: {', '.join(sorted(LANDED)) or '(none)'}")
    print(f"modules inspected: {inspected}")
    if structural:
        print("\nSTRUCTURAL PROBLEM -- this gate cannot be trusted as it stands:")
        for problem in structural:
            print(f"  {problem}")
    if violations:
        print(f"\n{len(violations)} forbidden import(s):")
        for violation in violations:
            print(violation)
    if not structural and not violations:
        print("\nOK: dependency direction holds.")
    return 1 if (structural or violations) else 0


if __name__ == "__main__":
    sys.exit(_report())
