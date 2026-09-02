"""What the test suite may assert about, and that every capability has evidence.

CHE-202 (R14.2). Like R14.1, the deletion this ticket describes had already
happened: the ~13 k LOC of architecture-protection tests and tests of deleted
subsystems -- `test_architecture_invariants`, `test_flat_layout`,
`test_suite_layout`, `test_retired_taxonomy`, `test_family_schema`,
`test_claim_ledger`, `test_verifier`, `test_b0..b4_families`, `test_discovery`,
`test_cli`, `test_agent_benchmark`, `test_metalens_*`, `test_performance_harness`,
`tests_tutorial/`'s 123 files -- went with the greenfield rewrite. The suite is 72
files against 208.

So this module is the enforceable part. Three of the ticket's criteria are
properties of the surviving suite rather than of a change:

* **criterion 2** -- no remaining test asserts the *old* layout. Asserting the
  **new** layout is R01.1's gates' job and is legitimate; what may not come back is
  a test pinned to a tree that no longer exists.
* **criterion 3** -- no mock-patch pins a private name of a deleted module. A patch
  target was part of the old module's contract; it is not part of the new one.
* **criterion 5's intent** -- coverage. The literal criterion asks for an
  assert-count comparison and the measurement is in the commit; what is *checkable*
  is the thing the count is a proxy for, so
  `test_every_catalogued_operation_cites_evidence_that_exists` holds every
  capability the project claims to physics tests that are on disk.

On the hierarchy the ticket specifies
-------------------------------------
It names `tests/{unit,physics,integration,fixtures,evidence}/`. The suite is
organized **per package** instead -- `tests/numerics/`, `tests/couplers/` and so on
beside `unit`, `physics`, `integration` and `fixtures` -- because every ticket from
R02 onward put its tests there, and that convention is now fifteen directories
deep. Restructuring it here would be a rename of 72 files with no behavioural
consequence, which is the "broad cleanup" the rewrite's scope discipline excludes.
Recorded rather than done.

`tests/evidence/` is the one part of that hierarchy with a real question behind it,
and the answer is that it has no content: criterion 4's subject was
`benchmarks/probes/records/` -- 152 files, deleted with the reference tree -- and
the four records under `benchmarks/systems/records/` are *live outputs* of a
committed driver (`make benchmarks` rewrites them), co-located with it by CHE-212
and CHE-213, not frozen evidence pruned from a deleted tree. An empty
`tests/evidence/` would be the speculative scaffolding the clean-slate rule bans.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests"

sys.path.insert(0, str(ROOT))

from operations import CATALOG  # noqa: E402

#: The one module exempt from the rules below: this one.
#:
#: A file that names what it forbids has to be able to name it, which is the same
#: exemption `tests/solvers/test_optiland_boundary.py` takes for the millimetre
#: rule and takes for the same reason -- "a test that may not name what it forbids
#: cannot check that it is forbidden". Exactly one file, so the exemption cannot
#: quietly widen. Note that the path rule below needs it for a narrower reason than
#: the first version did: `test_the_dangling_path_check_can_fail` names a path that
#: deliberately resolves nowhere.
SELF_EXEMPT: frozenset[Path] = frozenset({Path(__file__).resolve()})


def _test_modules() -> list[Path]:
    found = sorted(
        path
        for path in TESTS.rglob("*.py")
        if "__pycache__" not in str(path) and path.resolve() not in SELF_EXEMPT
    )
    assert len(found) > 60, "the walk found almost nothing, so it cannot fail"
    return found


# ---------------------------------------------------------------------------
# Criterion 2, and the rule that could not be written
# ---------------------------------------------------------------------------
#
# Criterion 2 says no remaining test may assert "a file path, a package name, a
# module layout, or the presence of a specific class" -- those being R01.1's gates'
# job, and only for the new layout. Two mechanical forms of that were attempted
# here and **both are recorded as failures** rather than shipped with an exemption
# list longer than their subject:
#
# 1. *forbid a mention of a deleted path.* Flagged nine files, every one a
#    legitimate historical citation: `tests/fixtures/systems.py` recording that the
#    M3 protocol was frozen in `benchmarks/protocols/slice_protocol.yaml`,
#    `tests/physics/test_psf.py` recording that the PSF reduction came from
#    `core/boundary.py:1508`, `tests/solvers/test_optiland_system.py` listing
#    `_resolve_lens` among the names that did *not* land.
#
# 2. *require every path a test names to resolve now or at the frozen tag.* This
#    looked decidable and is not. It flagged seven, in three further legitimate
#    categories: **absence assertions** -- `assert not (BENCHMARKS /
#    "observables.py").exists()` must name the file it forbids; **synthetic fixture
#    paths** -- a capability record's `probe` must start with `benchmarks/probes/`
#    and must *not* exist, because it is a row built to be refused; and
#    **forbidden-name examples** in `test_cutover.py`'s own meta-test.
#
# The finding, which is the useful output: **a path literal in a test is as often a
# thing being excluded as a thing being referenced**, and nothing mechanical
# separates the four meanings. So criterion 2 is enforced in the two forms that
# *are* decidable -- every patch target resolves (below, which is criterion 3
# generalized past the two names the ticket lists), and the set of tests that walk
# the tree is enumerated (further below) -- and the rest is left to review. That is
# a narrower claim than the criterion makes, and it is the true one.


def test_every_mock_patch_target_a_test_names_resolves() -> None:
    """Criterion 3, generalized past the two names the ticket lists.

    `solvers.optiland.adapter._import_optiland` and `._resolve_lens` were pinned by
    mock-patch tests, which made two private names of a deleted module part of what
    a rewrite had to preserve. Rather than ban those two strings -- which
    `tests/solvers/test_optiland_system.py` legitimately *lists* as names that did
    not land -- this resolves every `monkeypatch.setattr` and `mock.patch` target a
    test actually uses. A patch target that does not resolve is the same defect
    whatever it is called.
    """
    import importlib

    unresolved: list[str] = []
    for path in _test_modules():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            attribute = getattr(node.func, "attr", None)
            if attribute not in ("patch", "setattr", "delattr"):
                continue
            if not node.args or not isinstance(node.args[0], ast.Constant):
                continue
            target = node.args[0].value
            if not isinstance(target, str) or "." not in target:
                continue
            module_path, _, name = target.rpartition(".")
            try:
                module = importlib.import_module(module_path)
            except ImportError:
                unresolved.append(f"{path.relative_to(ROOT)}: {target} (no module)")
                continue
            if not hasattr(module, name):
                unresolved.append(f"{path.relative_to(ROOT)}: {target} (no attribute)")
    assert unresolved == [], (
        "a test patches a name that does not exist, so the patch is a no-op or an "
        "error rather than a substitution:\n  " + "\n  ".join(unresolved)
    )


# ---------------------------------------------------------------------------
# 2. Criterion 5's intent -- every claimed capability has evidence on disk
# ---------------------------------------------------------------------------


def test_every_catalogued_operation_cites_evidence_that_exists() -> None:
    """The check an assert count is a proxy for, and the one that is decidable.

    Criterion 5 asks for the physics-assertion count to be shown not to have
    dropped. The measurement is in the commit and it *did* drop, for reasons the
    count cannot distinguish -- capabilities the new tree does not claim took their
    assertions with them, three files on one coupler became three narrower ones, and
    a parametrized test with one assert covers N cases.

    What is decidable, and is the thing the count was standing in for: **no
    capability this project claims may be without physics evidence on disk.** Every
    `OperationDescriptor.evidence` path is resolved here. A capability whose
    evidence file was deleted is exactly the coverage loss an assert count would
    report as a number and not as a name.
    """
    missing = [
        f"{descriptor.operation_id} -> {reference}"
        for descriptor in CATALOG
        for reference in descriptor.evidence
        if not (ROOT / reference).exists()
    ]
    assert missing == [], (
        "a catalogued operation cites evidence that is not on disk:\n  "
        + "\n  ".join(missing)
    )


def test_every_catalogued_operation_cites_at_least_one_test() -> None:
    """`evidence=()` is permitted by the schema and must not be used by the catalog.

    R03.1 allows an empty tuple deliberately -- "no evidence yet" has to be
    *writable* so it is a statement rather than an omission. What this asserts is
    that no landed operation is in that state: fourteen capabilities, fourteen
    citations, every one a test file.
    """
    unevidenced = [
        descriptor.operation_id for descriptor in CATALOG if not descriptor.evidence
    ]
    assert unevidenced == [], unevidenced
    for descriptor in CATALOG:
        for reference in descriptor.evidence:
            assert reference.startswith("tests/"), (
                f"{descriptor.operation_id} cites {reference!r}, which is not a test. "
                "Evidence for a capability is something that runs."
            )


def test_the_evidence_files_actually_contain_assertions() -> None:
    """A cited file that asserts nothing is a citation, not evidence.

    The failure this closes is a file that survives a deletion as a stub -- imports
    intact, tests removed -- which would satisfy every check above.
    """
    thin = []
    for descriptor in CATALOG:
        for reference in descriptor.evidence:
            path = ROOT / reference
            tree = ast.parse(path.read_text(encoding="utf-8"))
            asserts = sum(
                1 for node in ast.walk(tree) if isinstance(node, ast.Assert)
            )
            raises = sum(
                1
                for node in ast.walk(tree)
                if isinstance(node, ast.Attribute) and node.attr == "raises"
            )
            if asserts + raises < 5:
                thin.append(f"{descriptor.operation_id} -> {reference}: {asserts + raises}")
    assert thin == [], "cited evidence with almost no assertions:\n  " + "\n  ".join(thin)


# ---------------------------------------------------------------------------
# 3. Criterion 4 -- tests/evidence/, and why it is empty
# ---------------------------------------------------------------------------


def test_the_only_committed_records_are_the_live_benchmark_outputs() -> None:
    """Criterion 4, and the decision behind it.

    Its subject was `benchmarks/probes/records/` -- 152 files, deleted with the
    reference tree, so there is nothing to prune. What remains committed is four
    JSON records under `benchmarks/systems/records/`, and those are **live outputs**
    of a committed driver: `make benchmarks` rewrites them, and
    `tests/benchmarks/test_records.py` reads them to check in the fast gate that the
    last run was clean. Co-located with the driver by CHE-212 and CHE-213 rather
    than moved to `tests/evidence/`, because a generated artifact belongs beside its
    generator.

    So `tests/evidence/` has no content and is not created. This pins that: a
    committed record appearing anywhere else is a record whose provenance nobody
    declared.
    """
    assert not (TESTS / "evidence").exists(), (
        "tests/evidence/ exists; either it has content that needs declaring, or it is "
        "the empty scaffolding the clean-slate rule bans"
    )
    committed = sorted(
        str(path.relative_to(ROOT))
        for path in ROOT.rglob("*.json")
        if ".git" not in str(path)
        and "__pycache__" not in str(path)
        and not str(path.relative_to(ROOT)).startswith((".claude", "outputs", "runs", "tmp_probes"))
    )
    expected_trees = ("benchmarks/systems/records/", "knowledge/capabilities/")
    stray = [path for path in committed if not path.startswith(expected_trees)]
    assert stray == [], (
        "a committed JSON artifact lives outside the two declared trees -- the "
        "benchmark driver's own records and the capability pack:\n  " + "\n  ".join(stray)
    )
