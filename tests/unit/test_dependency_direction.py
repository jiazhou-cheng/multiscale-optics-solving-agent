"""The new tree's dependency direction holds, and the check that says so works.

CHE-171 (R01.1). This is the **gate and meta-test layer** over
`scripts/check_dependencies.py`, which is the CLI and report layer. The two are
not redundant and neither replaces the other:

* the script decides the rule and prints a readable report for `make check-arch`,
  a pre-commit hook, or a developer who wants to see which import in which module
  broke it;
* this file puts `verify()` in the default suite so CI cannot skip it, and — the
  part the script cannot do for itself — drives `_classify` against **synthetic**
  inputs to prove each violation class is genuinely detected.

Two halves, and the second is still the one carrying the weight:

* the real tree passes;
* the checker detects each violation class, proved rather than trusted.

The new tree is small — `numerics/` (two modules, CHE-173 / R02.1) and an empty
`representations/` — so "the real tree passes" remains nearly free information. A
checker that returned "OK" unconditionally would also pass it. The detection
tests are what make the gate meaningful while there is little code to guard.

Nothing here imports the reference implementation; that is the rule being
enforced, so the enforcement cannot depend on it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts import check_dependencies  # noqa: E402
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

    A dependency checker that walks zero files reports success, and the new tree is
    four modules, so this is not a hypothetical. `verify()` reports such a state as
    a structural problem instead of a pass -- as it does a `LANDED` entry with no
    package on disk, or a top-level name under `src/` that nothing classifies.
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
    """R01 acceptance criterion 3, and the reason this gate exists at all.

    These packages no longer exist, so such an import would also fail at runtime.
    The rule is kept -- and tested -- so that reviving one is caught here, with the
    architectural reason attached, rather than surfacing later as a bare
    `ModuleNotFoundError` that reads like a missing dependency.
    """
    for old in sorted(OLD_TREE_ONLY):
        rule = _classify("representations", old)
        assert rule is not None, f"importing old-tree {old!r} was not flagged"
        assert "old production tree" in rule


def test_importing_through_the_src_namespace_root_is_a_violation() -> None:
    """The shortest way to defeat every other rule in this file.

    `src/` has no `__init__.py`, so PEP 420 makes `src.anything` an importable
    namespace path whenever the repository root is on `sys.path` -- which it is for
    a bare `python`, `python -m` or pytest run from the root. An import written that
    way reaches its target while naming none of the packages this check classifies,
    and an earlier revision classified the first segment (`"src"`) as third-party and
    let it through. Review found it; this pins it closed.

    The concrete case that motivated it was `from src.core.boundary import
    RayBundle` reaching the reference tree. That tree is gone, but the bypass is
    structural rather than tied to any package: `src.couplers`, `src.backends` or
    anything a later ticket adds would resolve the same way.

    (`src.solvers` was the second example until CHE-224 (R15.1) renamed the
    package; the point is that the spelling of the target is irrelevant.)
    """
    rule = _classify("representations", NAMESPACE_ROOT)
    assert rule is not None, (
        "importing through 'src.' was not flagged, so any `from src.<pkg> import ...` "
        "in the new tree would bypass every other rule in this file"
    )
    assert "namespace root" in rule


def test_every_top_level_name_in_src_is_classified() -> None:
    """An unclassified package is silently importable, which is the same hole.

    `_classify` returns None for anything it does not recognise, on the assumption
    that it is third-party. That assumption is only safe while every top-level name
    under `src/` is accounted for, so the sets are checked against the tree instead
    of being trusted.

    `SHARED_NAMES` is deliberately **not** subtracted here, mirroring `verify()`.
    Those names carried an exemption while the reference tree occupied them; with
    that tree deleted, a directory appearing under one of them is a new package
    that must be landed like any other, not a pre-approved one.
    """
    src = ROOT / "src"
    on_disk = {p.name for p in src.iterdir() if p.is_dir() and (p / "__init__.py").is_file()}
    on_disk |= {p.stem for p in src.glob("*.py")}
    unaccounted = on_disk - OLD_TREE_ONLY - LANDED
    assert not unaccounted, (
        f"top-level names under src/ that the dependency check does not classify: "
        f"{sorted(unaccounted)}. An import of one from the new tree would be treated "
        "as third-party and allowed."
    )


def test_importing_a_solver_backend_outside_backends_is_a_violation() -> None:
    for backend in sorted(BACKENDS):
        assert _classify("representations", backend) is not None
        assert _classify("couplers", backend) is not None
        # A solver adapter is the one place a backend belongs.
        assert _classify("backends", backend) is None


def test_numerics_may_import_nothing_in_the_project() -> None:
    """`numerics/` is the bottom of the graph; anything it imported would join it."""
    assert ALLOWED["numerics"] == frozenset()
    for other in sorted(ALLOWED.keys() - {"numerics"}):
        assert _classify("numerics", other) is not None, (
            f"numerics/ importing {other!r} was not flagged"
        )


def test_operations_may_not_import_a_solver_or_a_coupler() -> None:
    """The forbidden edge that makes lazy resolution structural.

    `operations/` holds import paths as strings. If it could import `backends/`,
    "reading the registry pulls no backend" would be a discipline someone
    maintains rather than a fact about the import graph.
    """
    for target in ("backends", "couplers"):
        rule = _classify("operations", target)
        assert rule is not None, f"operations/ -> {target}/ was not flagged"


def test_an_allowed_direction_to_an_unlanded_package_is_still_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A legal direction is not the same fact as a package to import.

    Passing an import because the *direction* is legal would let a ticket depend on
    a package it had not written. `LANDED` is the second condition, and this pins
    that it is applied.

    **The exemplar has run out of real pairs, so the rule is exercised rather than
    observed.** It was `operators/ -> couplers/` until CHE-185 (R07.1) landed
    `couplers`, then `runtime/ -> planning/` until CHE-164 (R12) landed `planning`.
    `runtime` is now the only unlanded package and nothing may import it, so no
    pair of real packages can demonstrate the rule any more. `planning` is
    un-landed for the length of this test -- the same technique the test below
    already uses -- because what is pinned is the rule, not which packages happen
    to exist today. The next package to land makes this test's *pair* stale again
    and its rule no less true.
    """
    assert "planning" in ALLOWED["runtime"]
    monkeypatch.setattr(
        check_dependencies, "LANDED", LANDED - {"planning"}, raising=True
    )
    rule = _classify("runtime", "planning")
    assert rule is not None
    assert "has not been landed" in rule


def test_a_reference_tree_name_carries_the_extra_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A name the reference implementation also used says so in the refusal.

    The reviver of one gets the warning that a partial revert could have put old
    code there, rather than only the generic "not landed". `couplers` is landed
    now, so the message is exercised by un-landing it for the length of this test:
    what is pinned is the message, not which packages happen to exist today.
    """
    assert "couplers" in SHARED_NAMES
    monkeypatch.setattr(check_dependencies, "LANDED", LANDED - {"couplers"})
    rule = _classify("operators", "couplers")
    assert rule is not None
    assert "has not been landed" in rule
    assert "reference implementation" in rule


def test_a_package_under_a_reference_tree_name_is_not_pre_approved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The exemption `SHARED_NAMES` used to carry, proved gone.

    While the reference tree occupied `couplers/`, `solvers/` and `runtime/`, those
    names were skipped by the "on disk but not in LANDED" structural check, because
    they were permanently present as old code. The documented cost was that
    authoring new code under one of them and forgetting the LANDED edit raised no
    error, and both gates then walked and counted *zero* of it -- a package that
    looked guarded and was not.

    With that tree deleted the skip has no benefit, so it was removed. This drives
    `verify()` against a synthetic tree to show the hole is actually closed, rather
    than inferring it from the set arithmetic.

    **`SHARED_NAMES` is down to two, and both are landed** -- `couplers` at CHE-185
    (R07.1) and `runtime` at CHE-199 (R13.1) -- so neither can play the unlanded
    package any more. The third was `solvers`, landed at R05; CHE-224 (R15.1)
    renamed that package to `backends/` and moved the name to `OLD_TREE_ONLY`,
    because a name the reference tree used and the target architecture does not is
    that set's definition. `backends` is not in `SHARED_NAMES`: the reference tree
    never used it, so there is nothing shared to record.

    `runtime` is un-landed for the length of this test, which is the same technique
    `test_an_allowed_direction_to_an_unlanded_package_is_still_refused` uses and for
    the same reason: what is pinned is the rule, not which packages happen to exist.
    """
    landed = LANDED - {"runtime"}
    src = tmp_path / "src"
    for package in [*sorted(landed), "runtime"]:
        (src / package).mkdir(parents=True)
        (src / package / "__init__.py").write_text("")
    monkeypatch.setattr(check_dependencies, "ROOT", tmp_path)
    monkeypatch.setattr(check_dependencies, "SRC", src)
    monkeypatch.setattr(check_dependencies, "LANDED", landed)

    _, structural, _ = check_dependencies.verify()
    assert any("src/runtime/" in problem and "LANDED" in problem for problem in structural), (
        "a new package under a reference-tree name was not reported as unguarded:\n"
        + "\n".join(f"  {problem}" for problem in structural)
    )


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
