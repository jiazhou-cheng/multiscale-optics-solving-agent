"""The triage table must cover the tree, and the tree must not outgrow it.

CHE-130 (M0.5.1). The inventory decides what survives the deletion of the old
task layer, so the failure mode it is designed against is a file that is in the
tree and not in the table: it gets no verdict, nobody reads it, and it goes out
with the directory around it. A table that is merely *written* does not prevent
that. A table that *fails the gate* when the tree moves does.

The rules below are the ones that make the classification reviewable rather than
decorative:

* every in-scope file is matched by at least one row, and every row matches at
  least one file — a dead row is a classification of something that no longer
  exists, which reads as coverage and is not;
* a glob may not contain ``**`` and may not cross a directory boundary, so a new
  subdirectory can never be silently absorbed by an existing row;
* a bucket-A or bucket-B row must name its destination in the new architecture,
  because "preserve" without a destination is how preservation does not happen;
* a bucket-B row must state a *positive* justification. The ticket's wording is
  that "it already exists" is not one, so the rule here is mechanical: the
  justification opens with ``POSITIVE:``. A row that cannot write that sentence
  is a bucket-C row;
* a file with more than one row must give every row a ``part``, which is what
  made ``benchmarks/physics/L2-PSF-01/run_benchmark.py`` expressible as one
  obsolete wrapper plus one candidate canonical instance. Both of that file's
  rows are gone now -- CHE-116 (M4.1) deleted it once its bucket-B destination
  ``B3-PSF-SINGLET-01`` existed as a committed instance -- which is the split
  verdict working, not the rule going unused: ``benchmarks/probes/
  quadrature_weight.py`` and others still carry one.
"""

from __future__ import annotations

import fnmatch
import sys
from pathlib import Path

import pytest
import yaml

from core.paths import repository_root

ROOT = repository_root()
sys.path.insert(0, str(ROOT))

from scripts.generate_benchmark_inventory import render  # noqa: E402

INVENTORY_PATH = ROOT / "benchmarks/inventory.yaml"
MARKDOWN_PATH = ROOT / "benchmarks/INVENTORY.md"

#: Not artifacts. Build products and editor droppings carry no verdict.
IGNORED_NAMES = {".gitignore", ".DS_Store"}
IGNORED_SUFFIXES = {".pyc", ".pyo"}
IGNORED_DIRS = {"__pycache__", ".ipynb_checkpoints"}


@pytest.fixture(scope="module")
def inventory() -> dict:
    return yaml.safe_load(INVENTORY_PATH.read_text(encoding="utf-8"))


def _in_scope_files(scope: list[str]) -> list[str]:
    found: list[str] = []
    for entry in scope:
        base = ROOT / entry
        assert base.is_dir(), f"inventory scope names {entry}, which is not a directory"
        for path in base.rglob("*"):
            if not path.is_file():
                continue
            if set(path.relative_to(ROOT).parts) & IGNORED_DIRS:
                continue
            if path.name in IGNORED_NAMES or path.suffix in IGNORED_SUFFIXES:
                continue
            found.append(str(path.relative_to(ROOT)))
    return sorted(found)


def _matches(pattern: str, is_glob: bool, candidate: str) -> bool:
    if not is_glob:
        return pattern == candidate
    # Directory-local by construction: the parent must be equal, only the
    # basename is globbed.
    return str(Path(candidate).parent) == str(Path(pattern).parent) and fnmatch.fnmatchcase(
        Path(candidate).name, Path(pattern).name
    )


def test_every_in_scope_file_has_a_verdict(inventory: dict) -> None:
    files = _in_scope_files(inventory["scope"])
    entries = inventory["entries"]
    uncovered = [
        f for f in files if not any(_matches(e["path"], e.get("glob", False), f) for e in entries)
    ]
    assert not uncovered, (
        "these files are in the tree and absent from benchmarks/inventory.yaml, so "
        "nothing has decided whether they are evidence or obsolete:\n  " + "\n  ".join(uncovered)
    )


def test_no_row_classifies_something_that_is_not_there(inventory: dict) -> None:
    files = _in_scope_files(inventory["scope"])
    dead = [
        e["path"]
        for e in inventory["entries"]
        if not any(_matches(e["path"], e.get("glob", False), f) for f in files)
    ]
    assert not dead, (
        "these inventory rows match no file. A row for a path that does not exist "
        "reads as coverage and is not:\n  " + "\n  ".join(dead)
    )


def test_globs_cannot_absorb_a_new_directory(inventory: dict) -> None:
    for entry in inventory["entries"]:
        path = entry["path"]
        if entry.get("glob", False):
            assert "**" not in path, f"{path}: recursive globs are not allowed"
            assert "*" not in str(Path(path).parent), (
                f"{path}: a glob may only vary the basename, so that a new "
                "subdirectory cannot be silently absorbed by an existing row"
            )
        else:
            assert "*" not in path, f"{path}: looks like a glob but is not marked glob: true"


def test_every_row_is_classified_and_preservation_names_a_destination(
    inventory: dict,
) -> None:
    for entry in inventory["entries"]:
        path = entry["path"]
        assert entry["bucket"] in {"A", "B", "C"}, f"{path}: unknown bucket"
        assert entry.get("justification", "").strip(), f"{path}: no justification"
        if entry["bucket"] in {"A", "B"}:
            assert entry.get("destination", "").strip(), (
                f"{path}: bucket {entry['bucket']} with no destination. 'Preserve' "
                "without a named family / oracle / metric / predicate is how "
                "preservation does not happen."
            )


def test_a_candidate_case_states_a_positive_justification(inventory: dict) -> None:
    for entry in inventory["entries"]:
        if entry["bucket"] != "B":
            continue
        assert entry["justification"].strip().startswith("POSITIVE:"), (
            f"{entry['path']}: a bucket-B row must argue *why this physical setup "
            "is worth freezing*, not that it already exists. Write the sentence "
            "starting 'POSITIVE:' or reclassify the row as C."
        )


def test_a_file_with_two_verdicts_names_the_part_each_row_covers(
    inventory: dict,
) -> None:
    by_path: dict[str, list[dict]] = {}
    for entry in inventory["entries"]:
        by_path.setdefault(entry["path"], []).append(entry)
    for path, rows in by_path.items():
        if len(rows) == 1:
            continue
        missing = [r for r in rows if not r.get("part", "").strip()]
        assert not missing, (
            f"{path} carries {len(rows)} verdicts, so every row must name the part "
            "it covers. This is the L2-PSF-01 case: one wrapper to delete, one "
            "prescription to freeze, and they cannot share a row."
        )


def test_pending_edits_name_files_that_exist(inventory: dict) -> None:
    for edit in inventory.get("pending_edits", ()):
        assert (ROOT / edit["path"]).is_file(), f"pending edit names a missing file: {edit['path']}"
        for field in ("section", "required_change", "owner", "why_not_here"):
            assert edit.get(field, "").strip(), f"{edit['path']}: pending edit has no {field}"


def test_the_committed_markdown_still_matches_the_yaml(inventory: dict) -> None:
    assert MARKDOWN_PATH.is_file(), "benchmarks/INVENTORY.md is missing"
    assert MARKDOWN_PATH.read_text(encoding="utf-8") == render(inventory), (
        "benchmarks/INVENTORY.md no longer matches benchmarks/inventory.yaml. "
        "Regenerate it:\n"
        "    ./run.sh python scripts/generate_benchmark_inventory.py\n"
        "and commit the result with the YAML edit."
    )
