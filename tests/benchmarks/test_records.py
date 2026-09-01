"""What the committed benchmark records claim, and what the scripts may import.

CHE-212 (R06.7) criteria 1 and 5, and CHE-213 (R06.8) criterion 5. This test does
**not** run a benchmark: the scripts cost seconds each where the default gate
costs a fraction of one, and `pyproject.toml`'s `testpaths` excludes
`benchmarks/` for that reason. What it does is make three claims checkable in the
default suite:

1. **The scripts compose the public vocabulary and import no backend.** An AST
   walk, the same rule `tests/solvers/test_chromatix_boundary.py` applies to
   `src/` and `tests/` -- which does not walk `benchmarks/`, so this is the walk
   that covers it. A benchmark that reached past the boundary would be measuring
   the backend rather than this project.
2. **Every committed record claims a clean run**: every gate passed, at least
   one gate is closed-form, every closed-form gate carries a tolerance, and every
   negative control broke the gate it names. A `diagnostic` gate is permitted and
   must be labelled -- a record with *no* closed-form gate would be a benchmark
   certifying itself, which is the risk R06.7 names first.
3. **Exactly one intensity path exists in the tree**, which is R06.8's criterion
   5 and the same rule R11 applies to PSF.

The staleness this creates is intended: a record measures the code at a commit, so
changing the code means re-running the script and committing the regenerated
record in the same change. `benchmarks/README.md` says so.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
BENCHMARKS = ROOT / "benchmarks"
SCRIPTS = sorted(
    path
    for path in (BENCHMARKS / "systems").glob("*.py")
    if path.name != "__init__.py"
)
RECORDS = sorted((BENCHMARKS / "systems" / "records").glob("*.json"))

#: Distributions no benchmark may import. `chromatix` and `jax` are reachable only
#: through `solvers.chromatix`, which is the whole point of the boundary; `numpy`
#: and `scipy` are the oracle vocabulary and are allowed.
BACKEND_IMPORTS = frozenset({"chromatix", "jax", "jaxlib", "optiland", "torch"})


def _top_level_imports(source: str) -> set[str]:
    found: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            found.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            found.add(node.module.split(".")[0])
    return found


def test_the_walk_found_something() -> None:
    """A gate that inspects nothing cannot fail."""
    assert len(SCRIPTS) >= 2, f"expected the two landed system benchmarks, found {SCRIPTS}"
    assert len(RECORDS) >= 4, f"expected two configurations each, found {RECORDS}"


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda path: path.name)
def test_a_benchmark_imports_no_backend(script: Path) -> None:
    """Criterion 1. The composition path is the public one, structurally."""
    imported = _top_level_imports(script.read_text(encoding="utf-8"))
    assert not (BACKEND_IMPORTS & imported), (
        f"{script.relative_to(ROOT)} imports {sorted(BACKEND_IMPORTS & imported)}; a "
        "benchmark composes sources / operators / solvers.chromatix and nothing past them"
    )
    # ...and it really does reach the project through the public packages.
    assert {"sources", "operators"} <= imported


def test_the_detector_would_catch_a_violation() -> None:
    """The rule fires, and the sanctioned route in is not a violation."""
    assert BACKEND_IMPORTS & _top_level_imports("import chromatix.functional as cf\n")
    assert BACKEND_IMPORTS & _top_level_imports("from jax import numpy\n")
    assert not (
        BACKEND_IMPORTS
        & _top_level_imports("from solvers.chromatix import focal_plane_transform\n")
    )


def _load(path: Path) -> dict[str, Any]:
    record: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return record


@pytest.mark.parametrize("record_path", RECORDS, ids=lambda path: path.stem)
def test_a_record_claims_a_clean_run_decided_by_closed_forms(record_path: Path) -> None:
    """Criterion 5 for R06.7, and the "benchmark that certifies itself" risk.

    Every gate passed; at least one gate exists; every *deciding* gate is
    closed-form. A `diagnostic` entry is permitted and is required to be labelled
    -- it is evidence, and a benchmark whose verdict rested on one would be
    repository numerical code judging repository numerical code.
    """
    record = _load(record_path)
    for key in ("benchmark", "ticket", "configuration", "produced_by", "composition", "gates"):
        assert key in record, f"{record_path.name} has no {key!r}"
    assert (BENCHMARKS.parent / record["produced_by"]).is_file()

    gates = record["gates"]
    assert gates, f"{record_path.name} records no gate"
    closed_form = [entry for entry in gates if entry["oracle_kind"] == "closed_form"]
    assert closed_form, f"{record_path.name} has no closed-form gate, so nothing decides"
    for entry in gates:
        assert entry["passed"], f"{record_path.name}: gate {entry['name']!r} did not pass"
        assert entry["oracle"].strip()
        assert entry["tolerance_basis"].strip()
        if entry["oracle_kind"] == "closed_form":
            assert entry["tolerance"] is not None, (
                f"{record_path.name}: closed-form gate {entry['name']!r} has no tolerance"
            )

    controls = record["negative_controls"]
    assert controls, f"{record_path.name} records no negative control"
    for entry in controls:
        assert entry["broke_the_gate"], (
            f"{record_path.name}: control {entry['name']!r} did not break "
            f"{entry['breaks_gate']!r}, so that gate was not measuring what it claimed"
        )

    assert record["not_covered"], (
        f"{record_path.name} does not say what it left out, which reads as coverage"
    )


def test_exactly_one_intensity_path_exists_in_the_tree() -> None:
    """R06.8 criterion 5, the same rule R11 applies to PSF.

    `|U|^2` is a measurement. `measurements/` has not landed, so R06.8 computes it
    locally and its record says so; what is not acceptable is two
    implementations. The count is over `src/` and `benchmarks/`, and it is a count
    of *definitions* -- a call site is fine, a second `def intensity` is not.
    """
    definitions = [
        f"{path.relative_to(ROOT)}::{node.name}"
        for tree in (ROOT / "src", BENCHMARKS)
        for path in sorted(tree.rglob("*.py"))
        if "__pycache__" not in str(path)
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, ast.FunctionDef) and node.name in ("intensity", "to_intensity")
    ]
    assert definitions == ["benchmarks/observables.py::intensity"], (
        "there must be exactly one intensity implementation in the tree, and it is "
        f"benchmarks/observables.py::intensity until measurements/ lands. Found: {definitions}"
    )
