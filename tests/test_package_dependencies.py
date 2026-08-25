"""Which package may import which, enforced rather than reviewed.

CHE-90 broke two real cycles between `solvers/` and `couplers/`, and the cause
was historical rather than sloppy: the boundary artifacts landed in `couplers/`
because a coupler was their first consumer, and `RunStatus`/`CostEstimate`
landed in the solver base because a solver adapter was the first thing to need a
run status. Both then had to be imported *backwards* by the other side.

A convention that has to be remembered gets forgotten the next time something is
"the first consumer" of something else. So the direction is a test.

These are AST checks over the source tree. Nothing here imports a physics
solver, so the whole file is milliseconds and belongs in the fast subset --
`tests/test_suite_layout.py` argues that trade at length and it applies equally
here.
"""

from __future__ import annotations

import ast
from collections import defaultdict
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

#: Top-level names this distribution owns, derived from the tree.
OWNED = frozenset(
    [p.name for p in SRC.iterdir() if p.is_dir() and (p / "__init__.py").is_file()]
    + [p.stem for p in SRC.glob("*.py")]
)

#: What each package may import from, and *why* that direction and not the other.
#:
#: The rule for `core` is the strongest and the one worth stating plainly: it is
#: the shared vocabulary -- boundary artifacts, precision policy, execution
#: status, graph specs -- so anything it imported would become vocabulary too.
FORBIDDEN: dict[str, tuple[frozenset[str], str]] = {
    "core": (
        frozenset({"solvers", "couplers", "verification", "studies", "agent", "cli"}),
        "core/ is the shared vocabulary every other package speaks. A dependency "
        "from here on any of them makes that package part of the vocabulary, and "
        "the four boundary artifacts stop being neutral ground.",
    ),
    "solvers": (
        frozenset({"couplers", "verification", "studies", "agent", "cli"}),
        "a solver adapter wraps one external package; it must not know what a "
        "coupler is. The artifacts both sides exchange live in core/boundary.py "
        "precisely so neither has to import the other.",
    ),
    "couplers": (
        frozenset({"solvers", "verification", "studies", "agent", "cli"}),
        "couplers/ holds physics -- the ray-wave mathematics -- and nothing else. "
        "A coupler that imports a solver has stopped being a representation "
        "change and become an integration.",
    ),
    "verification": (
        frozenset({"studies", "agent", "cli"}),
        "an oracle that depends on a study is not independent of it. Independence "
        "is the whole reason verification/ exists as its own package.",
    ),
    "discovery": (
        frozenset({"studies", "agent", "cli"}),
        "discovery/ joins the registry, the capability table, the validator, the "
        "ledger and the families into one queryable surface -- so it imports all of "
        "them, and that is why it is not in registry/ (which must not import "
        "verification) or in runtime/ (same rule). What it must not know is anything "
        "about an agent: it answers questions, and who is asking is not one of them.",
    ),
    "runtime": (
        frozenset({"verification", "studies", "agent", "cli"}),
        "runtime/ orchestrates solvers and couplers -- that is what it is for, and "
        "why the executor is not in core/. What it must NOT know is what the "
        "numbers mean: an executor that imported verification/ could grade its own "
        "run, and the whole point of the ExecutionRecord is that something else does.",
    ),
    "registry": (
        frozenset({"solvers", "couplers", "verification", "studies", "agent", "cli"}),
        "the registry declares what exists; it does not execute any of it. "
        "solvers/registry.py is the runtime map and imports adapters lazily.",
    ),
}


def _imports(path: Path) -> set[str]:
    """Top-level package names imported by one module, from its AST."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            # `level > 0` is a relative import: it cannot cross a package
            # boundary upward past its own root, so it is not this file's
            # concern.
            if node.level == 0 and node.module:
                found.add(node.module.split(".")[0])
    return found


def _modules_of(package: str) -> list[Path]:
    base = SRC / package
    return sorted(base.rglob("*.py")) if base.is_dir() else [SRC / f"{package}.py"]


@pytest.mark.parametrize("package", sorted(FORBIDDEN))
def test_package_does_not_import_what_it_must_not(package: str) -> None:
    forbidden, reason = FORBIDDEN[package]
    violations: dict[str, set[str]] = defaultdict(set)
    for module in _modules_of(package):
        offending = _imports(module) & forbidden
        if offending:
            violations[str(module.relative_to(ROOT))] |= offending
    assert not violations, (
        f"{package}/ imports what it must not:\n"
        + "\n".join(f"  {mod}: {sorted(names)}" for mod, names in sorted(violations.items()))
        + f"\n\nWhy: {reason}\n"
        "If the shared thing genuinely belongs to neither side, move it into "
        "core/ -- that is what CHE-90 did with the boundary artifacts and with "
        "RunStatus/CostEstimate. Do not add an exception here."
    )


def test_the_rule_table_covers_every_package_that_has_one() -> None:
    """Guards the guard: a new package with no rule is silently unconstrained."""
    unruled = OWNED - set(FORBIDDEN) - {"cli", "studies", "agent"}
    assert not unruled, (
        f"these packages have no dependency rule: {sorted(unruled)}. Add one to "
        "FORBIDDEN, or add it to the deliberate-exemption set below with a "
        "reason. `cli`, `studies` and `agent` are exempt because they are the "
        "top of the stack -- they compose everything and are imported by nothing."
    )


def test_no_cycle_between_top_level_packages() -> None:
    """The property the two directional rules exist to produce, checked directly.

    A rule table can be individually satisfied and still admit a cycle through a
    package pair nobody thought to constrain, so this derives the graph and looks
    for one rather than trusting the table to be complete.
    """
    edges: dict[str, set[str]] = {}
    for package in sorted(OWNED):
        deps: set[str] = set()
        for module in _modules_of(package):
            deps |= _imports(module) & OWNED
        edges[package] = deps - {package}

    colour: dict[str, int] = dict.fromkeys(edges, 0)  # 0 unvisited, 1 on stack, 2 done
    cycle: list[str] = []

    def visit(node: str, stack: list[str]) -> bool:
        colour[node] = 1
        for nxt in sorted(edges.get(node, ())):
            if colour.get(nxt, 2) == 1:
                cycle.extend([*stack[stack.index(nxt) :], nxt])
                return True
            if colour.get(nxt, 2) == 0 and visit(nxt, [*stack, nxt]):
                return True
        colour[node] = 2
        return False

    for package in sorted(edges):
        if colour[package] == 0 and visit(package, [package]):
            break

    assert not cycle, (
        f"import cycle between top-level packages: {' -> '.join(cycle)}. "
        "Whatever the two sides share belongs in core/, not in a mutual import."
    )


def test_core_boundary_is_where_the_four_artifacts_live() -> None:
    """The move that broke the cycle, pinned as a location rather than a habit.

    AGENTS.md's "Initial Artifact Boundary" makes RayBundle, WavefrontSamples,
    ComplexField and PSF vocabulary of the whole system. They spent a while in
    `couplers/contracts.py` because a coupler consumed them first, which is why
    `solvers/optiland/coherent_trace.py` had to import from `couplers/`.
    """
    from core.boundary import PSF, ComplexField, RayBundle, WavefrontSamples

    for artifact in (RayBundle, WavefrontSamples, ComplexField, PSF):
        assert artifact.__module__ == "core.boundary", (
            f"{artifact.__name__} is defined in {artifact.__module__}, not core.boundary. "
            "The four boundary artifacts are core vocabulary; a package that owns "
            "them becomes a dependency of everything that speaks it."
        )


def test_no_module_locates_the_repository_root_by_counting_parents() -> None:
    """`parents[N]` for the repository root is a silent dependency on file depth.

    It went wrong twice in one night. CHE-89 flattened `src/` and two `parents[3]`
    walks had to become `parents[2]`; CHE-90 then moved `metalens_controller.py`
    one level deeper, and the same expression started pointing at `src/` instead
    of the root. Nothing failed at import, and no test caught the second one --
    the call site was a subprocess `cwd` on a GPU path that skips by default.

    `core.paths.repository_root()` finds the root by marker and raises when there
    is none, so the failure mode is an exception rather than a plausible wrong
    directory. This test is what keeps the next `parents[N]` from being written.

    Only `src/` is checked, and only for a depth of 2 or more: `parents[0]` and
    `parents[1]` reach a sibling file or the package directory, which cannot
    escape the package however the tree is rearranged. The check is on the AST,
    not the text, so prose in a docstring explaining this rule does not trip it.
    """
    offenders: list[str] = []
    for module in sorted(SRC.rglob("*.py")):
        tree = ast.parse(module.read_text(encoding="utf-8"), filename=str(module))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Subscript):
                continue
            value = node.value
            if not (isinstance(value, ast.Attribute) and value.attr == "parents"):
                continue
            index = node.slice
            if not (isinstance(index, ast.Constant) and isinstance(index.value, int)):
                continue
            if index.value >= 2:
                offenders.append(
                    f"{module.relative_to(ROOT)}:{node.lineno}: .parents[{index.value}]"
                )
    assert not offenders, (
        "these locate a directory by counting parents:\n  "
        + "\n  ".join(offenders)
        + "\n\nUse core.paths.repository_root(), which finds the root by marker and "
        "raises when there is none. A parent count survives a file move by "
        "pointing somewhere plausible, which is the worst available outcome."
    )
