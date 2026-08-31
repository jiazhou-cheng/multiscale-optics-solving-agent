"""The class budget holds, and the counter that says so works.

CHE-171 (R01.1). Same shape as `test_dependency_direction.py`: the real tree is
within budget, and the counter is proved to *detect* growth rather than trusted
to. At R01 both budgets are 0 and both packages are empty, so "within budget" is
almost free information — the detection tests are what make the gate real before
there is anything to count.

The one thing this file deliberately does not test is whether a class is
*justified*. That is a judgement against the five minimality rules and no script
can make it; the budget number is the reviewed artifact instead.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.check_dependencies import LANDED  # noqa: E402
from scripts.class_budget import (  # noqa: E402
    BUDGETS,
    PROJECT_CEILING,
    _classes_in,
    count,
    verify,
)


def test_the_new_tree_is_within_its_class_budget() -> None:
    failures, _, _ = verify()
    assert not failures, "class budget exceeded:\n" + "\n".join(f"  {f}" for f in failures)


def test_the_budget_is_not_structurally_vacuous() -> None:
    _, structural, counts = verify()
    assert not structural, "structural problems:\n" + "\n".join(f"  {s}" for s in structural)
    assert counts, "the gate counted no packages, so it cannot have failed"


def test_the_declared_budgets_stay_under_the_project_ceiling() -> None:
    """The check that stops a local raise from being unbounded.

    Without it, each package could raise its own number and every raise would look
    like a local decision, while the tree drifted back toward the 280 classes the
    reference implementation reached.
    """
    assert sum(BUDGETS.values()) <= PROJECT_CEILING


def test_every_landed_package_has_a_declared_budget() -> None:
    """An unbudgeted package is unbounded growth that no number reports."""
    for package in sorted(LANDED):
        assert package in BUDGETS, (
            f"src/{package}/ is landed with no entry in BUDGETS; declare a number, even 0"
        )


def test_r01_lands_no_production_class() -> None:
    """R01's own contract: 0 classes added.

    The packages exist so the gates have a tree to walk. A base class, protocol or
    placeholder interface created "so the structure is visible" is exactly what the
    architecture document bans, and this is the assertion that notices one.
    """
    for entry in count():
        assert entry.counted == 0, (
            f"src/{entry.package}/ defines {entry.counted} class(es): {list(entry.classes)}. "
            "R01 adds none."
        )


# --- the detection half ---


def test_the_counter_finds_a_class(tmp_path: Path) -> None:
    module = tmp_path / "thing.py"
    module.write_text("class A:\n    pass\n\n\nclass B:\n    pass\n")
    assert _classes_in(module) == ["A", "B"]


def test_the_counter_finds_a_nested_class(tmp_path: Path) -> None:
    """A class hidden inside another still costs a class.

    Counting only module-level `ClassDef`s would let a nested hierarchy grow under
    a single budgeted name, which is the cheapest way to defeat this gate.
    """
    module = tmp_path / "nested.py"
    module.write_text("class Outer:\n    class Inner:\n        pass\n")
    assert _classes_in(module) == ["Outer", "Inner"]


def test_the_counter_ignores_what_is_not_a_class(tmp_path: Path) -> None:
    """The five rules' own alternatives must not read as classes.

    A function, a module constant and a type alias are the sanctioned answers when
    no rule is satisfied. If the counter charged for them the gate would push work
    back toward classes, which is the opposite of its purpose.
    """
    module = tmp_path / "plain.py"
    module.write_text(
        "from typing import Literal, TypedDict\n\n"
        "Precision = Literal['fp32', 'fp64']\n"
        "WAVELENGTH_M = 550e-9\n\n"
        "def propagate(x: float) -> float:\n"
        "    return x\n"
    )
    assert _classes_in(module) == []


def test_a_typeddict_is_counted_and_that_is_deliberate() -> None:
    """Honesty about a limit rather than a silent one.

    `TypedDict` and `Enum` are *sanctioned* alternatives to a class, but both are
    written with `class` syntax, so an AST count charges for them. The gate is a
    count, not a judgement (see `scripts/class_budget.py`), and the budget is
    where that gets resolved: a ticket adding an Enum says so and raises the
    number. This test pins the behaviour so it is a known limit rather than a
    surprise the first time someone adds one.
    """
    source = "from enum import StrEnum\n\nclass Precision(StrEnum):\n    FP32 = 'fp32'\n"
    tree = ast.parse(source)
    names = [n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
    assert names == ["Precision"]


@pytest.mark.parametrize("package", sorted(BUDGETS))
def test_no_budget_is_negative_or_absurd(package: str) -> None:
    budget = BUDGETS[package]
    assert 0 <= budget <= PROJECT_CEILING, (
        f"src/{package}/ has a budget of {budget}, outside [0, {PROJECT_CEILING}]"
    )
