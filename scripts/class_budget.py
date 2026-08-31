"""Production classes per package, against a declared budget.

CHE-171 (R01.1). The reference implementation reached **280 production classes**
under stated principles comparable to the new ones. That is the entire argument
for this script existing in R01 rather than in R15: a principle nothing counts is
a principle that gets restated in every ticket and enforced in none.

The rule this counts against is the project's, quoted so it is checkable rather
than remembered. A class is justified only if:

1. several fields share an invariant enforced together;
2. it is a public serialized / versioned data model;
3. it owns a genuine mutable resource lifecycle;
4. at least two *current* implementations need runtime polymorphism;
5. it is a real plugin boundary used by the runtime or registry.

Otherwise the answer is a function, a module, a frozen dataclass, a TypedDict, a
tuple, a Literal or an Enum.

**What this script can and cannot do, stated plainly.** It counts. It cannot tell
whether a class satisfies one of the five rules -- that is a judgement, and
pretending otherwise would make the gate authoritative about something it does
not know. So the budget is the reviewed artifact: raising a number requires a
ticket to say which rule the new class satisfies, and this script makes an
unjustified raise visible instead of silent. `docs/architecture_principles.md`
labels this a judgement call for exactly that reason.

`PROJECT_CEILING` is the second half. Without it, a package could raise its own
budget indefinitely and every individual raise would look local; the ceiling makes
the sum of the budgets fail against the project's declared target of 22.

Run directly for a report, or through `tests/unit/test_class_budget.py`, which is
what puts it in the default suite.
"""

from __future__ import annotations

import ast
import sys
from dataclasses import dataclass
from pathlib import Path

# The repo root, so `scripts` resolves as a package whether this file is run
# directly (`python scripts/class_budget.py`) or imported by the test that gates
# it. `LANDED` is imported rather than restated: the migration state is one fact
# and a second copy would drift.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.check_dependencies import LANDED

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

#: Production classes allowed per package of the new tree.
#:
#: Both entries are 0 because R01 lands no numerical optics: the packages exist so
#: the gates have a real tree to walk, and `docs/architecture_principles.md` bans
#: a placeholder interface created "so the structure is visible". A budget of 0 is
#: therefore the honest number, and it is also the strictest possible starting
#: point -- the first class added to either package has to be argued for.
BUDGETS: dict[str, int] = {
    "numerics": 0,
    "representations": 0,
}

#: The project's declared target for the whole production tree. The reference
#: implementation had 280; R00's inventory found 66 of them would satisfy a rule
#: at all, which is still three times this number, so the collapse R02-R11 owes is
#: real work and not rounding.
PROJECT_CEILING = 22


@dataclass(frozen=True)
class PackageCount:
    package: str
    counted: int
    budget: int
    classes: tuple[str, ...]

    @property
    def over(self) -> int:
        return max(0, self.counted - self.budget)

    @property
    def headroom(self) -> int:
        return max(0, self.budget - self.counted)


def _classes_in(path: Path) -> list[str]:
    """Top-level and nested class names defined in one module."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]


def _modules_of(package: str) -> list[Path]:
    base = SRC / package
    if not base.is_dir():
        return []
    return sorted(p for p in base.rglob("*.py") if "__pycache__" not in str(p))


def count() -> list[PackageCount]:
    counts: list[PackageCount] = []
    for package in sorted(LANDED):
        found: list[str] = []
        for module in _modules_of(package):
            where = module.relative_to(SRC).as_posix()
            found.extend(f"{where}::{name}" for name in _classes_in(module))
        counts.append(
            PackageCount(
                package=package,
                counted=len(found),
                budget=BUDGETS.get(package, 0),
                classes=tuple(sorted(found)),
            )
        )
    return counts


def verify() -> tuple[list[str], list[str], list[PackageCount]]:
    """Return (budget failures, structural problems, per-package counts)."""
    counts = count()
    failures: list[str] = []
    structural: list[str] = []

    if not LANDED:
        structural.append(
            "no package of the new tree has been landed, so this gate would pass "
            "without counting anything"
        )

    for package in sorted(LANDED):
        if package not in BUDGETS:
            structural.append(
                f"src/{package}/ is a landed package with no entry in BUDGETS, so its "
                "class count is unbudgeted. Declare a number, even if it is 0."
            )
    for package in sorted(BUDGETS.keys() - LANDED):
        structural.append(
            f"BUDGETS declares {package!r}, which is not landed. A budget for a package "
            "that does not exist reads as headroom nobody is using."
        )

    declared = sum(BUDGETS.values())
    if declared > PROJECT_CEILING:
        failures.append(
            f"the budgets sum to {declared}, over the project ceiling of {PROJECT_CEILING}. "
            "Raising one package's budget cannot be a local decision: the target is for the "
            "whole production tree."
        )

    for entry in counts:
        if entry.over:
            failures.append(
                f"src/{entry.package}/ has {entry.counted} production class(es) against a "
                f"budget of {entry.budget} -- {entry.over} over.\n"
                + "\n".join(f"      {name}" for name in entry.classes)
                + "\n    Either collapse them (function, module, frozen dataclass, TypedDict, "
                "tuple, Literal, Enum) or raise the budget in a ticket that names which of the "
                "five minimality rules each new class satisfies."
            )

    return failures, structural, counts


def _report() -> int:
    failures, structural, counts = verify()
    total = sum(entry.counted for entry in counts)
    declared = sum(BUDGETS.values())
    print(f"production classes: {total} / {declared} budgeted (ceiling {PROJECT_CEILING})")
    for entry in counts:
        print(f"  src/{entry.package}/: {entry.counted} / {entry.budget}")
        for name in entry.classes:
            print(f"      {name}")
    if structural:
        print("\nSTRUCTURAL PROBLEM -- this gate cannot be trusted as it stands:")
        for problem in structural:
            print(f"  {problem}")
    if failures:
        print("\nBUDGET EXCEEDED:")
        for failure in failures:
            print(f"  {failure}")
    if not structural and not failures:
        print("\nOK: within budget.")
    return 1 if (structural or failures) else 0


if __name__ == "__main__":
    sys.exit(_report())
