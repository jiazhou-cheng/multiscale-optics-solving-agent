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

**Why this is a script *and* a test, since the pair looks redundant.** They are
two layers with different jobs and neither subsumes the other:

* **This file is the CLI and report layer.** A standalone AST pass with no pytest
  dependency, for `make check-arch`, a pre-commit hook, or a developer who wants
  to see *which* import in *which* module broke the rule without reading an
  assertion diff. `_report()` prints the whole graph state, including the
  packages that are fine.
* **`tests/unit/test_dependency_direction.py` is the gate and the meta-test.** It
  runs `verify()` in the default suite so CI cannot skip it, and -- the part this
  file cannot do for itself -- it drives `_classify` against synthetic inputs to
  prove each violation class is actually *detected*. During bootstrap the real
  tree is small enough that "the tree passes" is nearly free information; the
  synthetic cases are what make the gate mean something.

So: this file decides the rule, that file proves the rule is enforced. Change a
rule here and the detection test there is what fails if you got it wrong.

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
#: **Declared, not derived from the filesystem**, and that distinction still earns
#: its keep now that the reference tree is gone. A directory under `src/` is not
#: the same fact as a package being part of the new architecture: a stray
#: checkout, a scratch directory, or a package created ahead of the code that
#: justifies it would all read as "landed" to a filesystem probe. Declaring it
#: means joining the graph is an edit someone reviews.
#:
#: Current state:
#:
#: * `numerics` -- landed with real code by CHE-173 (R02.1): `precision.py` and
#:   `arrays.py`. This is the first package the gate walks with anything in it, so
#:   it is also the first time the allowlist has done non-vacuous work.
#: * `representations` -- landed as an empty package by CHE-153 (R01); first real
#:   module by CHE-174 (R02.2): `geometry.py`, holding `Frame` and
#:   `ReferenceSurface`. CHE-175/CHE-176 (R02.3/R02.4) added `contracts.py`,
#:   `rays.py` and `scalar.py`, and those do import `numerics`, so the
#:   `representations -> numerics` edge is exercised.
#: * `operations` -- landed by CHE-177/CHE-178 (R03.1/R03.2): `descriptors.py` and
#:   `registry.py`. It imports `numerics` (one name, `COMPONENT_CAPABILITIES`, which
#:   a descriptor cites rather than copies) and nothing else in the project. The
#:   forbidden edges matter more here than anywhere else so far: `operations ->
#:   solvers` or `-> couplers` would be the end of "listing the registry imports no
#:   backend", which is the single property the package exists to provide. This
#:   allowlist entry and `tests/operations/test_registry_imports_no_backend.py` are the two
#:   halves of that -- the structural rule and the executed check.
#:
#: * `solvers` -- landed by CHE-179/CHE-180/CHE-181 (R05.1/R05.2/R05.3) as
#:   `solvers/optiland/`: `system.py`, `rays.py`, `solver.py`. The first package
#:   with a *permitted* backend import, and the only one there will ever be a
#:   permitted backend import from: `_classify` exempts `solvers` from the
#:   `BACKENDS` rule, which is the same fact stated as a rule about every other
#:   package rather than as a privilege granted to this one. It imports `problems`
#:   (the neutral problem it consumes), `representations` (the neutral bundle it
#:   emits) and `numerics` (the capability row it executes within), so all three
#:   of its allowed edges are now exercised. The forbidden edge that matters is
#:   `solvers -> operations`: a solver holding its own descriptor would make
#:   listing the registry import a backend, which is the one property
#:   `operations/` exists to provide. It is also why the trace's descriptor is not
#:   in this package -- see the R05 report.
#:
#: * `problems` -- landed by CHE-156 (R04): `ray_trace.py`, the neutral sequential
#:   ray-tracing problem. Its allowlist permits `representations` and `numerics`
#:   and it imports neither, which is correct rather than incomplete: a
#:   prescription is physical *intent*, not physical state at a boundary. What the
#:   gate is really guarding here is the other direction -- a problem that imported
#:   a solver would be a problem statable only by someone who has that solver
#:   installed, which is the entanglement R04 exists to undo.
#:
#: Every later ticket adds its package here as it authors it, in the same change.
#: When the graph is complete this equals `ALLOWED.keys()`.
LANDED: frozenset[str] = frozenset(
    {"numerics", "operations", "problems", "representations", "solvers"}
)

#: Target-architecture names that the deleted reference tree also used.
#:
#: `solvers` is now on disk again -- landed fresh by R05, not revived -- and the
#: other two are not. Membership here has no effect on whether a name is checked;
#: it only sharpens the message a violation carries. They are kept as a record
#: of *which* target names were previously occupied, because that is the set most
#: likely to be resurrected by a partial revert, a `git checkout` of an old path,
#: or a copy from the `pre-rewrite-2026-08-30` tag. A package appearing under one
#: of these names is a package that has to be argued in like any other.
#:
#: **The exemption these names used to carry has been removed.** While the old
#: tree was on disk they were skipped by the "on disk but not in LANDED"
#: structural check below, because they were permanently present as old code.
#: That skip was a documented hazard: authoring new code under `src/couplers/`
#: and forgetting to add `couplers` to LANDED raised no error, and both gates then
#: walked and counted *zero* of it. With the old tree gone the skip has no
#: remaining benefit, so it is gone too -- these names are now checked exactly
#: like every other, and landing one still requires the same LANDED edit.
SHARED_NAMES: frozenset[str] = frozenset({"couplers", "solvers", "runtime"})

#: Top-level names that belonged to the reference implementation and to nothing
#: else. An import of one from the new tree is the violation R01 exists to
#: prevent: the new implementation must not be built on the tree it replaces.
#:
#: **Kept deliberately now that none of them exist.** The obvious reading is that
#: this set is dead weight -- `import core` raises `ModuleNotFoundError` on its
#: own, so what is there to guard? Two things. A reintroduced `src/core/` is
#: caught here as an architecture violation with the reason attached, instead of
#: passing this gate and failing later as a confusing runtime error; and these
#: names are the ones a physics-recovery task is most likely to reach for, since
#: `docs/rewrite/reference_inventory.md` cites them by path. Re-deriving from the
#: tag is the supported route, importing is not, and this is where that is said.
#: The set is an anti-pollution guard, not a description of the current disk.
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
        reserved = (
            " That name was also used by the reference implementation, so a partial revert "
            "could make it resolve to code the new tree must not be built on."
            if imported in SHARED_NAMES
            else ""
        )
        return (
            f"{package}/ -> {imported}/ is an allowed direction, but {imported!r} has not "
            "been landed, so there is no new-tree package for the import to reach."
            f"{reserved} Land the package -- and add it to LANDED in the same change -- first."
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
    undeclared = ALLOWED.keys() - LANDED
    for name in sorted(undeclared):
        if (SRC / name / "__init__.py").is_file():
            structural.append(
                f"src/{name}/ exists and is a new-architecture package name, but is not in "
                "LANDED, so nothing checks its imports. Add it to LANDED."
            )
    # Every top-level name under src/ must be accounted for, derived from the tree
    # rather than trusted. An unaccounted name is classified as third-party and
    # silently importable -- the same failure shape as the `src.` bypass. Note that
    # SHARED_NAMES is *not* subtracted here: a directory appearing under one of
    # those names is a new package that has to be landed, not a pre-approved one.
    on_disk = {p.name for p in SRC.iterdir() if p.is_dir() and (p / "__init__.py").is_file()}
    on_disk |= {p.stem for p in SRC.glob("*.py")}
    unaccounted = on_disk - OLD_TREE_ONLY - LANDED
    for name in sorted(unaccounted):
        structural.append(
            f"src/{name}/ is a top-level name this check does not classify, so an import of "
            f"it from the new tree would be treated as third-party and allowed. If it is a "
            "new-architecture package, add it to LANDED (and to ALLOWED); if it is a revived "
            "reference-implementation package, it does not belong under src/ at all."
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
