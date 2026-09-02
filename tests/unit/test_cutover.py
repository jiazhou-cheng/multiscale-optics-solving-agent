"""`src/` is the new tree and only the new tree, checked rather than asserted.

CHE-201 (R14.1). The cut-over ticket's premise is that the previous production
source is removed and the tag is the archive, and **that had already happened**
before this ticket ran: the greenfield rewrite deleted `src/verification/`,
`src/couplers/` (the old one), `src/core/`, `src/studies/`, `src/agent/`,
`src/discovery/`, `src/registry/`, `src/cli.py`, `archive/` and the old `docs/`
tree, and `docs/rewrite/reference_inventory.md` is R00's record of what was
extracted from each first.

So what this module is for is the part a deletion cannot do: making the *state*
enforceable, so that it cannot drift back. Four of R14.1's acceptance criteria are
statements about the tree rather than about a change, and none of them had an
executable form:

1. `src/` contains only the new tree;
2. none of the junk-drawer or legacy names appears under `src/`, **at any depth**;
4. `pyproject.toml`'s package list and the coverage source list name only the new
   packages;
5. nothing imports a deleted module -- "checked by import, not by grep", which is
   what the subprocess below does.

Criterion 3 -- every deletion of a physics-bearing module cites the R00 inventory
line recording what was extracted -- is a claim about the deleting commits and is
checked here only in the weak form its artifact allows: that the inventory exists
and carries the extraction the ticket's named risk turns on. See
`test_the_claim_ledger_extraction_the_risk_names_is_in_the_inventory`.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"

sys.path.insert(0, str(ROOT))

from scripts.check_dependencies import LANDED, OLD_TREE_ONLY  # noqa: E402

#: Names that may not appear anywhere under `src/`, with the rule each violates.
#:
#: Two groups. The first is the reference implementation's own top-level packages,
#: read from `check_dependencies.OLD_TREE_ONLY` rather than restated -- that set is
#: already the one place they are named, and a second copy here is the arrangement
#: this whole rewrite is about.
#:
#: The second is the **junk-drawer** names, and they are the interesting half.
#: `AGENTS.md`: "`numerics/` is intended to be the bottom, not a generic `core/`",
#: and `src/numerics/__init__.py` records why -- "'core' names no domain, and a
#: package that names no domain accumulates whatever has no other home. That is how
#: the reference implementation reached 110 classes in `core/`." So the names are
#: banned by what they would *become*, not by what they were.
#:
#: Checked at **any** depth, not only at the top level: `src/couplers/utils.py` is
#: the same failure one directory in, and it is the more likely one now that the
#: top-level names are gated by `check_dependencies`.
JUNK_DRAWER_NAMES: tuple[str, ...] = (
    "archive",
    "baseline",
    "common",
    "compatibility",
    "deprecated",
    "helpers",
    "legacy",
    "misc",
    "old",
    "shared",
    "standalone",
    "util",
    "utils",
)

#: The old-tree names are forbidden **at the top level only**, and the difference
#: is not a loophole -- it is what the two rules mean.
#:
#: A revived `src/registry/` would be the manually duplicated registry §14 bans; a
#: module named `registry.py` *inside* a package is an ordinary module name, and
#: `src/operations/registry.py` is one: the by-id index over the catalog, landed by
#: CHE-178. Banning the word at any depth would have forced that module to be
#: renamed to make a rule pass, which is the tail wagging the dog.
#:
#: The junk-drawer names are the opposite: they are banned by what a package
#: called `utils` *becomes*, and that happens just as readily one directory in.
FORBIDDEN_NAMES: frozenset[str] = frozenset(JUNK_DRAWER_NAMES)


# ---------------------------------------------------------------------------
# 1. Criterion 1 -- `src/` is exactly the landed tree
# ---------------------------------------------------------------------------


def test_src_contains_exactly_the_landed_packages() -> None:
    """Nothing else, and nothing missing.

    `LANDED` is the declared membership and this is the check that the disk agrees
    -- in both directions, because `check_dependencies` reports an *undeclared*
    directory as a structural problem but a declared package that vanished would
    only surface as an import error somewhere later.
    """
    on_disk = {
        path.name for path in SRC.iterdir() if path.is_dir() and (path / "__init__.py").is_file()
    }
    assert on_disk == set(LANDED), {
        "on disk only": sorted(on_disk - set(LANDED)),
        "declared only": sorted(set(LANDED) - on_disk),
    }
    # No loose top-level modules either: every production module belongs to a
    # package, which is what makes the dependency allowlist total.
    assert [path.name for path in SRC.glob("*.py")] == []
    assert not (SRC / "__init__.py").exists(), (
        "src/ is a namespace root, not a package; an __init__.py here would make "
        "`import src.couplers` work and defeat the allowlist"
    )


@pytest.mark.parametrize("name", sorted(FORBIDDEN_NAMES))
def test_no_junk_drawer_name_appears_anywhere_under_src(name: str) -> None:
    """Criterion 2's "any depth" half, as a directory or as a module.

    A `src/utils.py` and a `src/couplers/helpers.py` are the same defect one
    directory apart. The reference implementation's `core/` reached 110 classes, and
    it did so one plausible addition at a time.
    """
    offenders = [
        str(path.relative_to(ROOT))
        for path in SRC.rglob("*")
        if "__pycache__" not in str(path)
        and (path.name == name or path.stem == name)
    ]
    assert offenders == [], (
        f"{name!r} appears under src/, which is a name banned by what it would become "
        "rather than by what it was:\n  " + "\n  ".join(offenders)
    )


@pytest.mark.parametrize("name", sorted(OLD_TREE_ONLY))
def test_no_reference_tree_package_exists_at_the_top_level(name: str) -> None:
    """Criterion 2's top-level half, for the names the reference tree occupied.

    Top-level only, deliberately: see `FORBIDDEN_NAMES` on why
    `src/operations/registry.py` is an ordinary module and a revived `src/registry/`
    would not be. `check_dependencies` already refuses an *import* of one of these
    and reports an undeclared directory; this is the same rule stated as a property
    of the disk, so it holds even if that script's structural check changes shape.
    """
    assert not (SRC / name).exists(), f"src/{name}/ is a reference-implementation package"
    assert not (SRC / f"{name}.py").exists(), f"src/{name}.py is one too"


def test_the_forbidden_name_check_would_catch_a_violation(tmp_path: Path) -> None:
    """The meta-check, since every assertion above is an empty list.

    Both spellings the walk has to see: a directory and a module.
    """
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "utils").mkdir()
    (tmp_path / "helpers.py").write_text("")
    found = {
        path.stem
        for path in tmp_path.rglob("*")
        if path.name in FORBIDDEN_NAMES or path.stem in FORBIDDEN_NAMES
    }
    assert found == {"utils", "helpers"}


# ---------------------------------------------------------------------------
# 2. Criterion 5 -- checked by import, not by grep
# ---------------------------------------------------------------------------


def test_importing_the_whole_tree_pulls_in_no_deleted_module() -> None:
    """Criterion 5, in a fresh interpreter, on `sys.modules`.

    A grep for `from core import` would miss a deleted module reached three levels
    down, or one imported inside a function -- which is where this tree puts every
    backend import. So every landed package is imported and `sys.modules` is read.

    The check is not vacuous in the other direction either: it asserts the packages
    *did* import, so a tree that failed to load would fail here rather than
    reporting a clean absence.
    """
    packages = sorted(LANDED)
    source = (
        "import importlib, json, sys\n"
        f"for name in {packages!r}:\n"
        "    importlib.import_module(name)\n"
        "print(json.dumps(sorted({m.split('.')[0] for m in sys.modules})))\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", source], capture_output=True, text=True, check=True
    )
    loaded = set(json.loads(completed.stdout))
    assert set(packages) <= loaded, "the tree did not import, so the absence below is empty"
    revived = loaded & OLD_TREE_ONLY
    assert revived == set(), (
        f"importing the new tree loaded {sorted(revived)}, which is a reference-"
        "implementation package. The tag is the archive; nothing here may be built on it."
    )


@pytest.mark.slow
def test_every_module_of_the_tree_imports_on_its_own() -> None:
    """Each module in a fresh interpreter, so an import cycle cannot hide.

    Importing the packages together lets a module that only works because another
    one loaded first pass. This is the same check per module, and it is what would
    have caught the shape of failure the old tree's `_LazyNames` existed to work
    around.

    Marked `slow` on a measurement: forty-six subprocesses cost **13 s**, against
    90 s for the whole default gate before this file existed. That is a seventh of
    the fast gate for a check whose failure mode -- an import cycle -- is not
    something an ordinary edit introduces, and the aggregate check above (one
    subprocess, importing every package) runs by default and would catch a package
    that stopped importing at all. `make test-slow` runs this one.
    """
    modules = sorted(
        ".".join(path.relative_to(SRC).with_suffix("").parts).removesuffix(".__init__")
        for path in SRC.rglob("*.py")
        if "__pycache__" not in str(path)
    )
    assert len(modules) > 30, modules
    failures: list[str] = []
    for name in modules:
        completed = subprocess.run(
            [sys.executable, "-c", f"import {name}"], capture_output=True, text=True
        )
        if completed.returncode != 0:
            failures.append(f"{name}: {completed.stderr.strip().splitlines()[-1]}")
    assert failures == [], "\n  ".join(failures)


# ---------------------------------------------------------------------------
# 3. Criterion 4 -- the packaging metadata names only the new tree
# ---------------------------------------------------------------------------


def _pyproject() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def test_the_package_list_names_exactly_the_landed_packages() -> None:
    include = _pyproject()["tool"]["setuptools"]["packages"]["find"]["include"]
    assert {entry.removesuffix("*") for entry in include} == set(LANDED), include


def test_the_coverage_source_list_names_exactly_the_landed_packages() -> None:
    """The one criterion-4 item that was actually stale.

    It named `numerics` and `representations` -- the two packages that existed when
    it was written -- so nine landed packages' coverage was silently not measured. A
    coverage source list that lags the tree reports a number about a fraction of it,
    which is worse than reporting none.
    """
    source = _pyproject()["tool"]["coverage"]["run"]["source"]
    assert set(source) == set(LANDED), {
        "listed only": sorted(set(source) - set(LANDED)),
        "landed only": sorted(set(LANDED) - set(source)),
    }


def test_the_makefile_and_run_sh_name_no_deleted_target() -> None:
    """Criterion 4's other half: no command that cannot run.

    A target that invokes a deleted tree reads as a supported command right up until
    someone tries it, which the Makefile's own comment records as measured
    (`make test-agent-benchmark` aborted in 0.11 s on the host).
    """
    for path in (ROOT / "Makefile", ROOT / "run.sh"):
        # Comment lines are dropped, because a *prose* mention of what was deleted
        # is exactly what these files should carry and the Makefile has several --
        # including the measurement that `make test-agent-benchmark` aborted in
        # 0.11 s on the host, which is why the rule exists. What must not appear is
        # an invocation.
        commands = [
            line
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        assert commands, path.name
        text = "\n".join(commands)
        for name in sorted(OLD_TREE_ONLY):
            for invocation in (f"-m {name}", f"src/{name}", f"{name}/"):
                assert invocation not in text, f"{path.name} invokes {invocation!r}"


# ---------------------------------------------------------------------------
# 4. Criterion 3, in the form its artifact allows
# ---------------------------------------------------------------------------


def test_the_claim_ledger_extraction_the_risk_names_is_in_the_inventory() -> None:
    """R14.1's named risk, and the evidence that it was discharged before the delete.

    The risk: deleting `verification/claim_ledger.py` (1,729 LOC) removes "the only
    machine-readable claim -> oracle -> tolerance -> gate map", which `AGENTS.md`
    ranks as a source of truth -- and "a tolerance with no recorded justification is
    exactly what a future ticket widens to make a benchmark pass". The ticket says
    to verify the extraction *before* deleting.

    The deletion happened in the greenfield rewrite, so this asserts the extraction
    exists rather than gating it. R00.3's §6 is that extraction: 90 claims, 45 with a
    numeric tolerance, six derivations transcribed verbatim and the remaining bases
    enumerated. What this test protects is the *file*: the inventory is now the only
    record of those derivations, so a change that removed §6 would lose them
    silently.
    """
    inventory = ROOT / "docs" / "rewrite" / "reference_inventory.md"
    assert inventory.is_file(), "the R00 inventory is the archive's index; it may not go"
    text = inventory.read_text(encoding="utf-8")
    assert "claim_ledger" in text
    assert "6. Tolerance and oracle extraction" in text
    for evidence in ("tolerance_basis", "gate-deciding", "analytic_closed_form"):
        assert evidence in text, f"the extraction no longer records {evidence!r}"
    assert "45 carry a numeric tolerance" in text, (
        "the extraction's own count is gone, so nothing says how much was extracted"
    )


def test_the_tag_is_the_archive_and_there_is_no_archive_directory() -> None:
    """No `archive/`, no `legacy/`, no compatibility shim. The tag holds it.

    Checked at the repository root as well as under `src/`, because an archive
    directory beside the source is the same thing one level out -- and it is where
    the reference tree's 54 tracked files were.
    """
    for name in ("archive", "legacy", "old", "deprecated", "compatibility"):
        assert not (ROOT / name).exists(), f"{name}/ exists at the repository root"
    tags = subprocess.run(
        ["git", "-C", str(ROOT), "tag", "-l", "pre-rewrite-2026-08-30"],
        capture_output=True,
        text=True,
    )
    if tags.returncode == 0 and tags.stdout.strip():
        # The archive is reachable, which is the condition that makes deleting the
        # tree recoverable rather than final.
        resolved = subprocess.run(
            ["git", "-C", str(ROOT), "cat-file", "-e", "pre-rewrite-2026-08-30:src/core/graph.py"],
            capture_output=True,
        )
        assert resolved.returncode == 0, (
            "the tag exists but the old tree is not in it, so nothing is archived"
        )
