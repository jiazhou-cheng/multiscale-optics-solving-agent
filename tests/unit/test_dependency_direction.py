"""The new tree's dependency direction holds, and the check that says so works.

CHE-171 (R01.1). Two halves, and the second is the one that matters at R01:

* the real tree passes;
* the checker **detects** each violation class, proved against synthetic inputs
  rather than by trusting it.

At R01 the new tree is two empty packages, so "the real tree passes" is nearly
free information — a checker that returned "OK" unconditionally would also pass
it. The detection tests are what make the gate meaningful before there is any
code to guard, and they are why this file exists now rather than in R02.

Nothing here imports the old production tree; that is the rule being enforced, so
the enforcement cannot depend on it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.check_dependencies import (  # noqa: E402
    ALLOWED,
    BACKENDS,
    LANDED,
    NAMESPACE_ROOT,
    OLD_TREE_ONLY,
    SHARED_NAMES,
    _classify,
    verify,
)


def test_the_new_tree_has_no_forbidden_import() -> None:
    violations, _, _ = verify()
    assert not violations, "forbidden imports in the new tree:\n" + "\n".join(
        str(v) for v in violations
    )


def test_the_check_is_not_structurally_vacuous() -> None:
    """The failure mode this gate is most likely to have.

    A dependency checker that walks zero files reports success, and at R01 the new
    tree is nearly empty, so this is not a hypothetical. `verify()` reports such a
    state as a structural problem instead of a pass.
    """
    _, structural, inspected = verify()
    assert not structural, "structural problems:\n" + "\n".join(f"  {s}" for s in structural)
    assert inspected > 0, "the check inspected no modules, so it cannot have failed"


def test_every_landed_package_is_actually_walked() -> None:
    """`LANDED` is hand-maintained, so what it claims is checked against the disk."""
    for package in sorted(LANDED):
        assert (ROOT / "src" / package / "__init__.py").is_file(), (
            f"LANDED names {package!r} but src/{package}/__init__.py is missing"
        )
        assert package in ALLOWED, f"LANDED names {package!r} with no direction declared"


# --- the detection half: each violation class, against synthetic inputs ---


def test_importing_the_old_tree_is_a_violation() -> None:
    """R01 acceptance criterion 3, and the reason this gate exists at all."""
    for old in sorted(OLD_TREE_ONLY):
        rule = _classify("representations", old)
        assert rule is not None, f"importing old-tree {old!r} was not flagged"
        assert "old production tree" in rule


def test_importing_through_the_src_namespace_root_is_a_violation() -> None:
    """The shortest way to defeat every other rule in this file.

    `src/` has no `__init__.py`, so PEP 420 makes `src.core.boundary` an importable
    namespace path whenever the repository root is on `sys.path` -- which it is for
    a bare `python`, `python -m` or pytest run from the root. The import *resolves*
    to the old tree while naming none of it, and an earlier revision of the checker
    classified the first segment (`"src"`) as third-party and let it through. Review
    found it; this pins it closed.
    """
    rule = _classify("representations", NAMESPACE_ROOT)
    assert rule is not None, (
        "importing through 'src.' was not flagged, so `from src.core.boundary import "
        "RayBundle` in the new tree would pass while resolving to the old tree"
    )
    assert "namespace root" in rule


def test_every_top_level_name_in_src_is_classified() -> None:
    """An unclassified package is silently importable, which is the same hole.

    `_classify` returns None for anything it does not recognise, on the assumption
    that it is third-party. That assumption is only safe while every top-level name
    under `src/` is in one of the three sets, so the sets are checked against the
    tree instead of being trusted.
    """
    src = ROOT / "src"
    on_disk = {p.name for p in src.iterdir() if p.is_dir() and (p / "__init__.py").is_file()}
    on_disk |= {p.stem for p in src.glob("*.py")}
    unaccounted = on_disk - OLD_TREE_ONLY - SHARED_NAMES - LANDED
    assert not unaccounted, (
        f"top-level names under src/ that the dependency check does not classify: "
        f"{sorted(unaccounted)}. An import of one from the new tree would be treated "
        "as third-party and allowed."
    )


def test_importing_a_solver_backend_outside_solvers_is_a_violation() -> None:
    for backend in sorted(BACKENDS):
        assert _classify("representations", backend) is not None
        assert _classify("couplers", backend) is not None
        # A solver adapter is the one place a backend belongs.
        assert _classify("solvers", backend) is None


def test_numerics_may_import_nothing_in_the_project() -> None:
    """`numerics/` is the bottom of the graph; anything it imported would join it."""
    assert ALLOWED["numerics"] == frozenset()
    for other in sorted(ALLOWED.keys() - {"numerics"}):
        assert _classify("numerics", other) is not None, (
            f"numerics/ importing {other!r} was not flagged"
        )


def test_operations_may_not_import_a_solver_or_a_coupler() -> None:
    """The forbidden edge that makes lazy resolution structural.

    `operations/` holds import paths as strings. If it could import `solvers/`,
    "reading the registry pulls no backend" would be a discipline someone
    maintains rather than a fact about the import graph.
    """
    for target in ("solvers", "couplers"):
        rule = _classify("operations", target)
        assert rule is not None, f"operations/ -> {target}/ was not flagged"


def test_an_allowed_direction_to_an_unlanded_package_is_still_refused() -> None:
    """Because it would resolve to the old tree instead.

    `couplers/` is an allowed target for `operators/`, and `src/couplers/` exists —
    as *old* code. Allowing the import because the direction is legal would wire
    the new tree to the old one through a name collision.
    """
    assert "couplers" in ALLOWED["operators"]
    assert "couplers" in SHARED_NAMES
    assert "couplers" not in LANDED
    rule = _classify("operators", "couplers")
    assert rule is not None
    assert "does not exist in the new tree yet" in rule
    assert "old" in rule.lower()


def test_a_third_party_import_is_not_this_checks_business() -> None:
    for name in ("numpy", "pytest", "pathlib", "dataclasses"):
        assert _classify("representations", name) is None


@pytest.mark.parametrize("package", sorted(ALLOWED))
def test_every_declared_package_has_a_direction(package: str) -> None:
    """An allowlist with a missing entry fails open, which defeats the point."""
    assert isinstance(ALLOWED[package], frozenset)
    assert ALLOWED[package] <= ALLOWED.keys(), (
        f"{package}/ is allowed to import something that is not a declared package"
    )
    assert package not in ALLOWED[package], f"{package}/ lists itself as a dependency"
