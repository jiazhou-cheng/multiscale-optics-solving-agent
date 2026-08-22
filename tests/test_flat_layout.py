"""The flat layout's one real cost, made loud.

CHE-89 removed the `multiscale_optics_agent/` package level, so this
distribution owns the top-level names `adapters`, `benchmarks`, `core`,
`couplers`, `evaluation`, `registry` and the module `cli`. Several of those are
common enough to exist on PyPI under other ownership, and one -- `benchmarks` --
also names a directory at this repository's root.

That was accepted deliberately in exchange for shorter imports, and the trade is
only safe while it is *checked*. A shadowing install does not raise: `import
core` succeeds, returns somebody else's package, and the failure surfaces later
as a missing attribute or, worse, as a plausible wrong number.

These are static-ish checks -- they import the packages but no physics solver --
and they belong to the fast subset.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

#: Every top-level name this distribution ships. Derived from the tree rather
#: than typed out, so adding a package cannot quietly escape the check.
TOP_LEVEL = sorted(
    [p.name for p in SRC.iterdir() if p.is_dir() and (p / "__init__.py").is_file()]
    + [p.stem for p in SRC.glob("*.py")]
)


def test_the_tree_actually_has_top_level_packages() -> None:
    """Guards the guard: an empty `TOP_LEVEL` would make every check below vacuous."""
    assert len(TOP_LEVEL) >= 5, TOP_LEVEL
    assert "core" in TOP_LEVEL and "couplers" in TOP_LEVEL and "cli" in TOP_LEVEL


@pytest.mark.parametrize("name", TOP_LEVEL)
def test_every_top_level_name_resolves_inside_this_repository(name: str) -> None:
    """A same-named install in site-packages must fail here, not silently win."""
    module = importlib.import_module(name)
    origin = getattr(module, "__file__", None)
    assert origin is not None, f"{name} resolved to a namespace package, not this repository's"
    resolved = Path(origin).resolve()
    assert SRC in resolved.parents, (
        f"`import {name}` resolved to {resolved}, which is outside {SRC}. Another "
        f"distribution owns the top-level name `{name}` in this environment. The "
        "flat layout accepts that risk deliberately; it does not accept it "
        "silently. Uninstall the shadowing package, or rename ours."
    )


def test_benchmarks_resolves_to_the_package_not_the_repository_directory() -> None:
    """The one collision that exists inside this repository, not outside it.

    `src/benchmarks/` and `<root>/benchmarks/` share a name, and the repository
    root is on `sys.path` for a bare `pytest` or `python -m`. PEP 420 resolves
    this correctly -- a directory without `__init__.py` is recorded as a
    namespace *portion* and the scan continues, so the regular package at
    `src/benchmarks/` wins -- but that outcome depends on the root directory
    never acquiring an `__init__.py`.

    If it ever does, `import benchmarks` starts returning a package containing
    `probes/` and `roadmap.md` instead of the agent suite, and the error is a
    confusing `AttributeError` a long way from the cause. Phase 5 removes the
    collision by renaming these to `studies/metalens/` and `agent/`; until then
    this is the guard.
    """
    import benchmarks

    assert Path(benchmarks.__file__).resolve().parent == SRC / "benchmarks"
    assert not (ROOT / "benchmarks" / "__init__.py").exists(), (
        "the repository-root benchmarks/ directory has acquired an __init__.py, "
        "which turns it into a regular package that shadows src/benchmarks/ for "
        "anything run from the repository root. Delete it."
    )


def test_the_packaged_registry_resolves_through_importlib_resources() -> None:
    """`Registry.from_package()` is a wheel path, not a source-tree path.

    It goes through `importlib.resources.files("registry")`, which is why the
    package-data declaration in `pyproject.toml` had to move with the flatten.
    Reading the YAML off disk would have kept working regardless, so a test that
    did that would prove nothing.
    """
    from registry.loader import Registry

    registry = Registry.from_package()
    assert "M_RAY_OPTILAND" in registry.models
    assert "C_RAY_TO_WAVE" in registry.couplers


def test_no_module_still_imports_the_old_package_name() -> None:
    """The re-rooting is complete, and `git grep` is the check.

    Scoped to Python under the active tree: `archive/` is frozen with its old
    import paths on purpose, and prose references are Phase 7 and 8's to update.

    The match is the bare name, not an import-shaped regex. That is deliberate:
    a regex for `from X import` misses `importlib.import_module("X.y")`,
    `sys.modules["X"]`, and a path string like `src/X/core/specs.py`, all of
    which appeared in this tree before the flatten. "The name does not occur"
    needs no cases. The cost is that a comment mentioning the old level fails
    this test -- which is the right trade, because a comment can be reworded and
    a missed call site cannot be found.
    """
    result = subprocess.run(
        [
            "git", "grep", "-l", "multiscale_optics_agent", "--",
            "src/*.py", "src/**/*.py", "tests/*.py", "scripts/*.py",
            "benchmarks/**/*.py", "benchmarks_agent/*.py", "tests_tutorial/*.py",
            # This file names the old level in its own docstring and in the
            # pattern above, so it would match itself. `git grep` only searches
            # tracked files, which meant the self-match appeared the moment this
            # test was committed rather than when it was written -- the worst
            # time to discover it.
            ":!tests/test_flat_layout.py",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    offenders = [line for line in result.stdout.splitlines() if line.strip()]
    assert not offenders, (
        f"these files still name the removed package level: {offenders}. "
        "CHE-89 flattened src/, so the import root is `core`, not "
        "`multiscale_optics_agent.core`."
    )


def test_the_console_script_entry_point_is_importable() -> None:
    """`[project.scripts] multiscale-optics = "cli:app"` must actually resolve.

    `cli` is a loose module rather than a package member, and `packages.find`
    does not collect loose modules -- it needs `py-modules`. Getting that wrong
    installs a console script whose entry point raises `ModuleNotFoundError` at
    first invocation, which no import-based test would catch.
    """
    from importlib.metadata import entry_points

    scripts = {ep.name: ep for ep in entry_points(group="console_scripts")}
    assert "multiscale-optics" in scripts, sorted(scripts)
    app = scripts["multiscale-optics"].load()
    assert callable(app)


def test_python_dash_m_reaches_the_cli() -> None:
    """`python -m cli` is the form the Makefile uses, and it is not the same path."""
    result = subprocess.run(
        [sys.executable, "-m", "cli", "list-models"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin", "HOME": "/tmp", "COLUMNS": "200"},
    )
    assert result.returncode == 0, result.stderr
    assert "M_RAY_OPTILAND" in result.stdout


@pytest.mark.slow
def test_a_non_editable_install_ships_the_registry_yaml(tmp_path: Path) -> None:
    """Build a wheel, install it away from the source tree, and read the registry.

    The editable install in the container is a `.pth` file pointing at
    `/workspace/src`, so *every* in-process check above passes whether or not the
    package data is declared correctly -- the YAML is simply on disk next to the
    module. Only a real wheel distinguishes them, which is why this is worth its
    ~10 s and its `slow` marker.

    Two things it pins that nothing else can:

    * `[tool.setuptools.package-data] "registry" = ["*.yaml"]` moved with the
      flatten. Get it wrong and `Registry.from_package()` raises in a wheel and
      nowhere else.
    * `cli` is a loose module, and `packages.find` does not collect loose
      modules. Without `py-modules = ["cli"]` the console script installs and
      then fails to import.
    """
    build = tmp_path / "build"
    build.mkdir()
    wheel = subprocess.run(
        [sys.executable, "-m", "pip", "wheel", "--no-deps", "-q", str(ROOT), "-w", str(build)],
        capture_output=True,
        text=True,
    )
    assert wheel.returncode == 0, wheel.stderr
    wheels = list(build.glob("*.whl"))
    assert len(wheels) == 1, wheels

    target = tmp_path / "site"
    install = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "--no-deps",
         "--target", str(target), str(wheels[0])],
        capture_output=True,
        text=True,
    )
    assert install.returncode == 0, install.stderr

    # cwd is tmp_path, not the repository, so nothing can resolve from source.
    probe = subprocess.run(
        [sys.executable, "-c",
         "from registry.loader import Registry\n"
         "import cli\n"
         "r = Registry.from_package()\n"
         "print(sorted(r.models)); print(sorted(r.couplers)); print(cli.__file__)"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(target), "PATH": "/usr/bin:/bin", "HOME": str(tmp_path)},
    )
    assert probe.returncode == 0, probe.stderr
    models, couplers, cli_file = probe.stdout.splitlines()
    assert "M_RAY_OPTILAND" in models and "M_WAVE_CHROMATIX" in models
    assert "C_RAY_TO_WAVE" in couplers and "C_WAVE_TO_RAY" in couplers
    assert str(target) in cli_file, cli_file
