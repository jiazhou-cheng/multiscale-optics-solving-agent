"""Fast repository consistency checks that do not import optional physics solvers.

Scope: the **active** tree only. The frozen archives under ``archive/`` and
``docs/archive/`` are preserved exactly as they were, stale paths and all, so
validating them would either report failures nobody may fix or pressure someone
into editing a historical record. Gitignored scratch is excluded for the same
reason it is gitignored -- it is not part of the repository's contract.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import yaml

from core.graph import GraphValidator
from registry.loader import Registry

ROOT = Path(__file__).resolve().parents[1]
MARKDOWN_LINK = re.compile(r"\[[^\]]+\]\((https?://[^)]+)\)")

#: Frozen trees. Not "skip because it fails" -- these are historical records
#: whose contents must not change, which makes checking them meaningless.
EXCLUDED_DIRS = frozenset({".git", "archive", "docs/archive"})


def _is_excluded(path: Path) -> bool:
    relative = path.relative_to(ROOT).as_posix()
    return any(relative == d or relative.startswith(f"{d}/") for d in EXCLUDED_DIRS)


def _ignored_paths() -> frozenset[Path]:
    """Paths git ignores, resolved once by asking git rather than reparsing rules.

    Reimplementing ``.gitignore`` semantics here would drift from the file it is
    meant to mirror; ``check-ignore`` cannot.
    """
    try:
        result = subprocess.run(
            ["git", "-C", str(ROOT), "status", "--porcelain", "--ignored=matching", "-z"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        # Not a git checkout (a wheel, a tarball). Fall back to the directory
        # exclusions above rather than failing the whole validation.
        return frozenset()
    ignored = set()
    for entry in result.stdout.split("\0"):
        if entry.startswith("!! "):
            ignored.add(ROOT / entry[3:])
    return frozenset(ignored)


def active_files(suffix: str) -> list[Path]:
    """Every tracked-or-trackable file with ``suffix`` outside the frozen trees."""
    ignored = _ignored_paths()

    def under_ignored(path: Path) -> bool:
        return any(path == i or i in path.parents for i in ignored)

    return [
        path
        for path in sorted(ROOT.rglob(f"*{suffix}"))
        if not _is_excluded(path) and not under_ignored(path)
    ]


def validate_yaml_files() -> None:
    for path in active_files(".yaml"):
        with path.open("r", encoding="utf-8") as handle:
            yaml.safe_load(handle)


def validate_graph_examples(registry: Registry) -> None:
    validator = GraphValidator(registry)
    for path in (ROOT / "examples" / "graphs").glob("*.yaml"):
        report = validator.validate(Registry.load_graph(path))
        if not report.valid:
            details = "\n".join(f"{item.code}: {item.message}" for item in report.errors)
            raise RuntimeError(f"Invalid example graph {path}:\n{details}")


def validate_markdown_links_are_well_formed() -> None:
    for path in active_files(".md"):
        text = path.read_text(encoding="utf-8")
        for url in MARKDOWN_LINK.findall(text):
            if any(char.isspace() for char in url):
                raise RuntimeError(f"Whitespace in Markdown URL {url!r} at {path}")


def main() -> None:
    registry = Registry.from_package()
    validate_yaml_files()
    validate_graph_examples(registry)
    validate_markdown_links_are_well_formed()
    yaml_count = len(active_files(".yaml"))
    markdown_count = len(active_files(".md"))
    print(
        f"Validated {len(registry.models)} models, {len(registry.couplers)} couplers, "
        f"{yaml_count} active YAML files, {markdown_count} active Markdown files, "
        "and all example graphs."
    )


if __name__ == "__main__":
    main()
