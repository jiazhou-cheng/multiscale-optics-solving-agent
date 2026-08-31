"""CHE-67: the default suite cannot silently re-absorb an archived or opt-in test.

CHE-67 split the tests into four groups (default / on-demand tutorial / archived /
GPU-own-session). Three of those boundaries are *configuration* -- `testpaths`,
`norecursedirs`, and a directory layout -- and configuration drifts back. The
failure mode is quiet in both directions: a tutorial file moved into `tests/`
adds 33 minutes to every PR run, and an archived milestone file moved back starts
gating current work on a frozen 2026-08 requirement. Neither announces itself.

These are static checks by construction. Asserting on a real collection would
mean spawning a pytest that imports jax, chromatix and optiland -- tens of
seconds, in the tier that is supposed to stay at ~30 s -- to re-derive facts that
are fully determined by the config and the file tree. What a subprocess *would*
add over these checks is confirmation that pytest honors its own `testpaths`, and
that is not this repository's invariant to defend. CHE-67 verified the collection
behavior empirically once (see docs/testing/test_archive.md); this file keeps the
inputs to that behavior from moving afterwards.

Deliberately not asserted here: that any *particular* test still exists. Deleting
or renaming an active test is normal work, and a guard that pins the file list
would fail on every legitimate change until someone edited it -- which teaches
people to edit it, defeating the guard. These tests pin only the *boundaries*.
"""

from __future__ import annotations

import ast
import tomllib
from pathlib import Path
from typing import Any, cast

import pytest

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"
TUTORIAL_SUITE = ROOT / "tests_tutorial"
ARCHIVE_ROOT = ROOT / "archive"
ARCHIVE_GENERATIONS = ARCHIVE_ROOT / "tests"

#: What CHE-67 archived, as (generation-relative path) -> why. The paths are the
#: original ones, so this doubles as the restore map: unarchiving is
#: `git mv archive/tests/gen1/<path> <original dir>`.
ARCHIVED_GEN1 = {
    "tests/benchmarks/test_l1_ray_scaling.py": "superseded benchmark suite",
    "tests/benchmarks/test_l1_wave_accuracy.py": "superseded benchmark suite",
    "tests/benchmarks/test_l1_wave_scaling.py": "superseded benchmark suite",
    "tests/benchmarks/test_l2_coupler_bundle.py": "superseded benchmark suite",
    "tests/benchmarks/test_l2_psf_bundle.py": "superseded benchmark suite",
    "tests/benchmarks/test_m1_bundle_projection.py": "superseded benchmark suite",
    "tests/benchmarks/test_m1_report.py": "superseded benchmark suite",
    "tests/benchmarks/test_m1_reproducibility.py": "superseded benchmark suite",
    "tests/test_optiland_ray_benchmark.py": "superseded benchmark suite",
    "tests/test_fdtdx_adapter.py": "out-of-scope solver",
    "tests/test_fmmax_adapter.py": "out-of-scope solver",
    # `tests/test_sax_adapter.py` was archived here by CHE-67 and then deleted
    # outright by CHE-72, which removed the SAX integration and its klujax
    # dependency. It is not listed: there is no longer a package for it to test,
    # so "restorable" would be false, and this dict is the restore map.
    "tests/test_m1_protocol.py": "outdated milestone suite",
    "tests/test_m2_coupler_protocol.py": "outdated milestone suite",
    "tests/test_m3_slice_protocol.py": "outdated milestone suite",
    "tests/test_m3_pupil_to_focus.py": "outdated milestone suite",
    # `test_m3_psf_measurement.py` and `test_m3_psf_verification.py` were
    # unarchived by CHE-88 and are now `tests/test_psf_measurement.py` and
    # `tests/test_psf_verification.py`. They are not listed because this dict is
    # the restore map and there is nothing left to restore.
    #
    # Why they came back: CHE-67 archived them as milestone evidence, but they
    # were the *only* guard on `verification/psf_oracles.py` (807 lines, the
    # independent Airy and Fraunhofer oracles) and `verification/psf_measurement.py`
    # (448 lines, named as the terminal measurement by the registry and the
    # shipped example graph). An unguarded verification oracle is worse than
    # either keeping or deleting it: it silently becomes the thing it was meant
    # to check. Both files run unmodified against the committed probe record --
    # 46 tests, 6.6 s -- so restoring beat rewriting them.
    "tests/test_m3_convergence.py": "outdated milestone suite",
    "tests/test_m3_off_axis_handoff.py": "outdated milestone suite",
    "tests/test_m3_quadrature_weight.py": "outdated milestone suite",
    "tests/test_m3r_sensor_handoff.py": "outdated milestone suite",
}

TUTORIAL_FILES = ("test_chromatix_tutorials.py", "test_optiland_tutorials.py")


@pytest.fixture(scope="module")
def pytest_config() -> dict[str, Any]:
    """The `[tool.pytest.ini_options]` table, read from the file rather than from
    `pytest.Config`.

    `pytest.Config` would report the *effective* settings, including anything the
    invoking command overrode -- so a run that passed `-p no:cacheprovider` or an
    explicit path would still see whatever pyproject declared. What these tests
    defend is the declaration itself, since that is what a bare `pytest` uses.
    """
    parsed = tomllib.loads((ROOT / "pyproject.toml").read_text())
    return cast(dict[str, Any], parsed["tool"]["pytest"]["ini_options"])


def test_testpaths_is_only_the_default_suite(pytest_config: dict[str, Any]) -> None:
    """`testpaths` is what makes the other trees opt-in; widening it defeats both."""
    assert pytest_config["testpaths"] == ["tests"], (
        "testpaths must stay ['tests']. Adding 'tests_tutorial' puts 33 min of "
        "pinned-dependency reproduction into every default run; adding 'archive' "
        "un-archives every historical test at once. Name the directory on the "
        "command line instead: `pytest tests_tutorial`."
    )


def test_the_opt_in_trees_are_not_swept_up_by_a_root_directory_run(
    pytest_config: dict[str, Any],
) -> None:
    """`pytest .` must find neither tree, since an explicit path beats `testpaths`.

    Measured in CHE-67: before this entry, `pytest .` collected 625 tests instead
    of 565 -- the whole 33-minute tutorial suite. `norecursedirs` is not applied
    to a path named on the command line, so `pytest tests_tutorial` is unaffected
    and stays the way in.

    Also pins the pytest defaults back in: `norecursedirs` *replaces* them, so
    dropping them here would silently start collecting `build/`, `venv/` and
    friends.
    """
    norecurse = pytest_config["norecursedirs"]
    assert "archive" in norecurse
    assert "tests_tutorial" in norecurse
    # CHE-71 added a third opt-in tree on the same principle: it runs an agent, so
    # it is nondeterministic and consumes model tokens. Its *harness* is in the
    # default suite (tests/test_agent_benchmark.py) deliberately, because the
    # grader decides whether an agent passed.
    assert "benchmarks/agents" in norecurse
    for default in ("*.egg", ".*", "_darcs", "build", "CVS", "dist", "node_modules", "venv"):
        assert default in norecurse, f"norecursedirs dropped the pytest default {default!r}"


def test_no_archived_test_reappeared_in_the_active_suite() -> None:
    """An archived file back under `tests/` gates current work on a frozen requirement."""
    active = {path.name for path in TESTS.rglob("test_*.py")}
    returned = sorted(
        f"{name} ({reason})"
        for path, reason in ARCHIVED_GEN1.items()
        if (name := Path(path).name) in active
    )
    assert not returned, (
        "archived test file(s) are active again: "
        + ", ".join(returned)
        + ". If that is intended, it is an unarchive: remove the entry from "
        "ARCHIVED_GEN1 here and say in the issue why the behavior is required "
        "again (archive/tests/gen1/README.md)."
    )


def test_archived_files_are_preserved_and_restorable() -> None:
    """Archiving preserves; it does not delete. Every file must still be there, intact."""
    generation = ARCHIVE_GENERATIONS / "gen1"
    missing = [path for path in ARCHIVED_GEN1 if not (generation / path).is_file()]
    assert not missing, (
        f"archived test file(s) missing from {generation.relative_to(ROOT)}: {missing}. "
        "CHE-67 archived these for history -- they are preserved, not deleted."
    )
    empty = [path for path in ARCHIVED_GEN1 if (generation / path).stat().st_size == 0]
    assert not empty, f"archived test file(s) emptied rather than preserved: {empty}"


def test_every_archive_generation_carries_the_collection_abort() -> None:
    """The guard that survives a mistyped path must exist in every generation.

    `testpaths` and `norecursedirs` both stop at *directory* level; naming an
    archived file directly walks past both (measured in CHE-67). The
    per-generation conftest is what turns that into a usage error, so it is the
    one part of the archive that a future generation must not forget to copy.

    Scoped to every generation under `archive/`, not just `archive/tests/`.
    CHE-88 added `archive/benchmarks/gen1/`, and an archive tree that happens to
    contain no test file today is exactly the one that will acquire one later
    without anyone remembering the guard.
    """
    generations = sorted(
        path
        for family in sorted(p for p in ARCHIVE_ROOT.glob("*") if p.is_dir())
        for path in sorted(p for p in family.glob("*") if p.is_dir())
    )
    assert generations, "archive/ has no generation directories"
    for generation in generations:
        conftest = generation / "conftest.py"
        assert conftest.is_file(), f"{generation.relative_to(ROOT)} has no conftest.py guard"
        source = conftest.read_text()
        assert "pytest_collection_modifyitems" in source and "UsageError" in source, (
            f"{conftest.relative_to(ROOT)} no longer aborts collection. Removing this "
            "guard is not unarchiving -- it just makes a mistyped path run frozen tests."
        )


def test_tutorial_suite_lives_outside_the_default_tree() -> None:
    """The tutorial files are active code, in their own directory, not under `tests/`."""
    assert TUTORIAL_SUITE.is_dir(), "the on-demand tutorial suite directory is gone"
    for name in TUTORIAL_FILES:
        assert (TUTORIAL_SUITE / name).is_file(), f"tutorial suite lost {name}"
        assert not (TESTS / name).exists(), (
            f"{name} is back in tests/ -- that puts the tutorial reproductions into "
            "every default run. Keep it in tests_tutorial/ and invoke it explicitly."
        )


def _applied_markers(path: Path) -> set[str]:
    """Marker names this module really applies, read off the AST.

    An AST pass, not a substring search: this very file names `pytest.mark.tutorial`
    in its own assertion messages, and a guard that its own error text trips is a
    guard people delete.
    """
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(inner := node.value, ast.Attribute)
            and inner.attr == "mark"
            and isinstance(inner.value, ast.Name)
            and inner.value.id == "pytest"
        ):
            found.add(node.attr)
    return found


def _module_level_markers(path: Path) -> set[str]:
    """The marker names in this module's top-level `pytestmark` assignment."""
    module = ast.parse(path.read_text())
    for node in module.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "pytestmark" for t in node.targets
        ):
            value = node.value
            elements = value.elts if isinstance(value, (ast.List, ast.Tuple)) else [value]
            return {e.attr for e in elements if isinstance(e, ast.Attribute)}
    return set()


def test_no_test_under_tests_carries_the_tutorial_marker() -> None:
    """`tutorial` means "member of the on-demand suite", so it must not appear here.

    A tutorial-marked test under `tests/` would run by default, since no documented
    command filters `tutorial` out -- the directory split is the whole mechanism.
    """
    offenders = [
        str(path.relative_to(ROOT))
        for path in sorted(TESTS.rglob("*.py"))
        if "tutorial" in _applied_markers(path)
    ]
    assert not offenders, (
        f"tutorial-marked test(s) inside the default suite: {offenders}. "
        "Tutorial reproductions belong in tests_tutorial/."
    )


def test_the_tutorial_suite_is_selectable_as_a_whole() -> None:
    """`-m tutorial` must name the entire suite, including its own consistency tests.

    Both files carry `tutorial` in their module-level `pytestmark` rather than only
    on the parametrized reproductions, so `pytest tests_tutorial -m tutorial` and
    `pytest tests_tutorial` select the same 60 tests. Without the module-level mark
    the three non-parametrized guard tests in those files would fall outside the
    marker, and "run the tutorial suite" would quietly mean "most of it".
    """
    for name in TUTORIAL_FILES:
        markers = _module_level_markers(TUTORIAL_SUITE / name)
        assert "tutorial" in markers, (
            f"{name} no longer carries `tutorial` in its module-level pytestmark "
            f"(found {sorted(markers)}), so `-m tutorial` would miss its "
            "non-parametrized tests"
        )


# --------------------------------------------------------------------------- #
# CHE-140: the fourth boundary -- what the default *selection* excludes
# --------------------------------------------------------------------------- #
#
# The three boundaries above are about directories. This one is about `addopts`,
# and it drifts the same way for a worse reason: the failure is invisible in both
# directions. Drop `-m "not slow"` and the PR gate silently goes back to six
# minutes; drop the `slow` marker off a test and 60 s of convergence sweep
# silently rejoins every run. Neither announces itself, and both were the actual
# state of this repository before CHE-140 -- `slow` was declared, documented and
# applied to 19 tests, and then selected anyway.


def test_the_default_selection_excludes_the_slow_marker(pytest_config: dict[str, Any]) -> None:
    """The deselection is the single largest thing keeping the gate under a minute.

    Measured (CHE-140): 151 s of the 375 s baseline was tests already carrying
    `slow`, plus 61 s more once the B2 stochastic transitions were marked. What
    `slow` means here is "expensive numerical characterization or convergence
    measurement", not "less important" -- see `test_the_slow_suite_has_a_command`.
    """
    addopts = pytest_config["addopts"]
    assert "not slow" in addopts, (
        "addopts no longer deselects `slow`. That marker is declared in this same "
        "file and applied to ~39 tests; without the deselection it is decoration "
        "and the default suite is back to ~6 minutes. If slow tests must run by "
        "default, delete the marker rather than leaving it selected -- a marker "
        "that filters nothing is worse than none."
    )


def test_the_default_selection_shards_by_file_and_not_finer(
    pytest_config: dict[str, Any],
) -> None:
    """`--dist loadfile` is a correctness requirement, not a tuning choice.

    About forty files in this suite use a module-scoped fixture that executes a
    benchmark driver or builds a solver system once for the whole file. pytest-xdist
    scopes fixtures per worker, so under a finer distribution two tests from one
    file can land on two workers and each re-runs that fixture. For
    `tests/test_b2_transition_instances.py` that fixture is a 61 s stochastic
    sweep, so the finer mode is both slower and a silent multiplication of real
    solver compute.
    """
    addopts = pytest_config["addopts"]
    assert "-n" in addopts.split(), (
        "addopts no longer shards the suite. Serial, this suite is ~165 s; the "
        "sharding is what puts it under a minute."
    )
    assert "--dist loadfile" in addopts, (
        "addopts must pin `--dist loadfile`. xdist's default distribution splits "
        "per test, which re-runs every module-scoped fixture on every worker that "
        "receives one of that file's tests -- including the 61 s B2 stochastic "
        "sweep. Use `-n 0` to run serially instead."
    )


def test_the_slow_suite_has_a_command(pytest_config: dict[str, Any]) -> None:
    """Nothing is allowed to leave the default suite without a way back in.

    A deselected test with no documented command is a deleted test that still
    shows up in the file count, which is the worst of both. `slow` covers the
    coupler gradient-bias characterization, the Monte-Carlo convergence fits and
    the B2 stochastic-transition benchmark -- all of them exit gates for the
    couplers, none of them per-PR.
    """
    makefile = (ROOT / "Makefile").read_text()
    assert "test-slow:" in makefile, (
        "`make test-slow` is gone. It is the only way the ~39 slow tests get run "
        "at all now that addopts deselects them."
    )
    assert '-m "slow"' in makefile or "-m slow" in makefile, (
        "`make test-slow` no longer selects by the marker, so it and addopts can "
        "disagree about what the slow suite is"
    )


def _requested_workers(recipe: str) -> list[int]:
    """Worker counts requested by a pytest command line.

    Handles the three spellings xdist accepts -- `-n 8`, `-n8` and `-n=8` -- because
    a parser that only understood the attached forms would read `-n 8` as "no count
    given" and pass the thing it exists to catch. `auto`/`logical` are reported as a
    large number: they are emphatically not one worker.
    """
    parts = recipe.split()
    counts: list[int] = []
    for index, part in enumerate(parts):
        if not part.startswith("-n"):
            continue
        value = part[2:].lstrip("=").strip()
        if not value:
            value = parts[index + 1] if index + 1 < len(parts) else ""
        counts.append(9999 if value in ("auto", "logical") else int(value or 0))
    return counts


def test_the_gpu_suite_has_a_command_and_is_not_sharded(pytest_config: dict[str, Any]) -> None:
    """`addopts` reaches invocations the CPU gate never sees, and the GPU is one.

    CHE-107 found this: the default `-n N --dist loadfile` is right for the CPU
    gate and wrong on a host with one device, where N workers each open their own
    CUDA context and JAX preallocates a large fraction of device memory per
    process.
    AGENTS.md's shared-server policy settles it -- one workload per GPU -- so this
    is a correctness and resource-safety constraint, not tuning. It is the same
    shape as the swap-guard finding: a resource mechanism degrading silently
    because a default reached it.

    So `make test-gpu` must exist, must select the marker, and must NOT inherit
    the sharding. Asserted on the Makefile rather than on a run, because the run
    needs hardware and this boundary drifts on a laptop.
    """
    makefile = (ROOT / "Makefile").read_text()
    assert "test-gpu:" in makefile, (
        "`make test-gpu` is gone. A bare `./run.sh --gpu pytest -q -m gpu` "
        "inherits addopts and is not a correct GPU invocation."
    )
    recipe = next(
        line for line in makefile.splitlines() if "pytest" in line and "--gpu" in line
    )
    assert "-m gpu" in recipe or '-m "gpu"' in recipe, recipe
    assert "-o addopts=" in recipe, (
        f"the GPU recipe does not override addopts, so it inherits the sharding "
        f"the default gate needs and one device cannot survive: {recipe}"
    )
    # Checked as "any worker count above zero" rather than as the literal `-n 12`:
    # CHE-171 re-measured the constant from 12 to 8, and a guard pinned to the old
    # number would have silently stopped catching an inherited `-n`. `-n 0` is
    # explicitly allowed -- it is serial, which is what one device needs, and the
    # Makefile comment records it as the alternative once the GPU image is rebuilt
    # with the xdist pin.
    workers = _requested_workers(recipe)
    assert all(w == 0 for w in workers), (
        f"the GPU recipe requests {workers} worker(s) on a host with one device: {recipe}"
    )
    assert "--dist" not in recipe, recipe
    assert "gpu" in {entry.split(":", 1)[0] for entry in pytest_config["markers"]}, (
        "the `gpu` marker lost its declaration, so `-m gpu` would match nothing "
        "and the GPU suite would report a clean pass having run none of it"
    )


def test_the_slow_marker_is_still_declared(pytest_config: dict[str, Any]) -> None:
    """`--strict-markers` is not set, so an unregistered marker silently matches
    nothing -- and `-m "not slow"` would then select everything while looking
    correct."""
    declared = {entry.split(":", 1)[0] for entry in pytest_config["markers"]}
    assert "slow" in declared, (
        "the `slow` marker lost its declaration. addopts filters on it, and an "
        "undeclared marker name is not an error here -- it just quietly stops "
        "matching, which would put the whole slow suite back into every run."
    )
