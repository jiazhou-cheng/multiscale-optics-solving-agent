#!/usr/bin/env python3
"""Validate the static Codex/Claude Code context entrypoints.

Enforces four rules (CHE-7 / M0.5):

a. ``AGENTS.md`` exists and is the canonical static context declared by
   ``CONTEXT_MANIFEST.yaml``.
b. ``CLAUDE.md`` points at ``AGENTS.md`` instead of duplicating its content.
c. Every repository path named by ``CONTEXT_MANIFEST.yaml`` exists.
d. ``AGENTS.md`` states the container-only execution rule, and every
   ``./run.sh`` flag it documents is actually implemented by ``run.sh``.

Run from the repository root, inside the container:

    ./run.sh python scripts/check_context_sync.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

MAX_AGENTS_LINES = 200

CONTAINER_RULE_TOKENS = (
    "`./run.sh` is the only supported entry point",
    "Do not run project commands",
    "agent_solver",
)

FORBIDDEN_CLAUDE_IMPORTS = (
    "@PROJECT_PLAN.md",
    "@docs/ARCHITECTURE.md",
    "@docs/BENCHMARK_SPECIFICATION.md",
    "@docs/SOLVER_AND_COUPLER_CATALOG.md",
)

# Manifest entries are prose as well as paths ("relevant source files",
# "selected solver/coupler cards"). Only treat a token as a repository path
# when it has no whitespace and either ends in "/" or carries a known suffix.
PATH_SUFFIXES = (".md", ".yaml", ".yml", ".json", ".py", ".sh", ".toml")

LIVE_REFERENCE_ROOTS = ("src", "tests", "knowledge", "benchmarks", "docker")
LIVE_TEXT_SUFFIXES = {".md", ".py", ".txt", ".yaml", ".yml"}
STALE_REFERENCE_PATTERNS = (
    (re.compile(r"CLAUDE\.md\s+(?:section|rule)\b"), "numbered CLAUDE.md citation"),
    (re.compile(r"CLAUDE\.md's\b"), "historical CLAUDE.md policy citation"),
    (re.compile(r"(?:docs/)?AGENT_KNOWLEDGE_BASE\.md\b"), "removed knowledge-base citation"),
    (
        re.compile(
            r"(?<!@)docs/(?:SOLVER_AND_COUPLER_CATALOG|ARCHITECTURE|"
            r"BENCHMARK_SPECIFICATION)\.md\b"
        ),
        "removed documentation citation",
    ),
)


class ContextSyncError(Exception):
    """A context-synchronization rule was violated."""


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def check_required_files(root: Path) -> list[str]:
    """Rule (a), part 1: the three context files exist."""
    notes = []
    for name in ("AGENTS.md", "CLAUDE.md", "CONTEXT_MANIFEST.yaml"):
        path = root / name
        if not path.is_file():
            raise ContextSyncError(f"missing required context file: {name}")
        notes.append(f"present: {name}")
    return notes


def check_canonical_declaration(root: Path) -> list[str]:
    """Rule (a), part 2: the manifest declares AGENTS.md as canonical."""
    manifest = _read(root / "CONTEXT_MANIFEST.yaml")
    match = re.search(r"^canonical_static_context:\s*(\S+)\s*$", manifest, re.M)
    if match is None:
        raise ContextSyncError(
            "CONTEXT_MANIFEST.yaml does not declare 'canonical_static_context'"
        )
    if match.group(1) != "AGENTS.md":
        raise ContextSyncError(
            f"canonical_static_context is '{match.group(1)}'; expected 'AGENTS.md'"
        )
    return ["canonical_static_context: AGENTS.md"]


def check_claude_entrypoint(root: Path) -> list[str]:
    """Rule (b): CLAUDE.md imports AGENTS.md and duplicates nothing."""
    claude_text = _read(root / "CLAUDE.md").strip()
    if claude_text != "@AGENTS.md":
        raise ContextSyncError(
            "CLAUDE.md must contain exactly '@AGENTS.md' to avoid duplicated static context"
        )
    for token in FORBIDDEN_CLAUDE_IMPORTS:
        if token in claude_text:
            raise ContextSyncError(f"CLAUDE.md imports broad startup context: {token}")
    return ["CLAUDE.md -> @AGENTS.md (no duplicated content)"]


def check_agents_size(root: Path) -> list[str]:
    """Keep the always-loaded context small."""
    line_count = len(_read(root / "AGENTS.md").splitlines())
    if line_count > MAX_AGENTS_LINES:
        raise ContextSyncError(
            f"AGENTS.md has {line_count} lines; keep it at or below {MAX_AGENTS_LINES}"
        )
    return [f"AGENTS.md is {line_count} lines (limit {MAX_AGENTS_LINES})"]


def manifest_declared_paths(manifest_text: str) -> list[str]:
    """Extract the repository paths named by the manifest, ignoring prose."""
    found: list[str] = []
    for line in manifest_text.splitlines():
        line = line.split("#", 1)[0]  # comments are notes, not declarations
        for raw in re.findall(r"[^\s\"']+", line):
            token = raw.strip().rstrip(":")
            if not token:
                continue
            if (token.endswith("/") or token.endswith(PATH_SUFFIXES)) and token not in found:
                found.append(token)
    return found


def check_manifest_paths(root: Path) -> list[str]:
    """Rule (c): every path the manifest names exists in the repository."""
    manifest_text = _read(root / "CONTEXT_MANIFEST.yaml")
    declared = manifest_declared_paths(manifest_text)
    missing = [p for p in declared if not (root / p).exists()]
    if missing:
        raise ContextSyncError(
            "CONTEXT_MANIFEST.yaml names paths that do not exist: " + ", ".join(sorted(missing))
        )
    return [f"manifest paths exist ({len(declared)} checked): " + ", ".join(declared)]


def documented_run_sh_flags(agents_text: str) -> list[str]:
    """Every '--flag' documented next to ./run.sh in AGENTS.md."""
    flags: list[str] = []
    for line in agents_text.splitlines():
        if "./run.sh" not in line:
            continue
        for flag in re.findall(r"(?<![\w-])--[a-z][a-z0-9-]*", line):
            if flag not in flags:
                flags.append(flag)
    return flags


def check_container_rule(root: Path) -> list[str]:
    """Rule (d): container-only execution is stated, and its flags are real."""
    agents_text = _read(root / "AGENTS.md")
    for token in CONTAINER_RULE_TOKENS:
        if token not in agents_text:
            raise ContextSyncError(
                f"AGENTS.md does not state the container-only execution rule (missing {token!r})"
            )

    run_sh = root / "run.sh"
    if not run_sh.is_file():
        raise ContextSyncError("run.sh is missing; the container entry point must exist")
    run_sh_text = _read(run_sh)

    flags = documented_run_sh_flags(agents_text)
    unimplemented = [f for f in flags if f not in run_sh_text]
    if unimplemented:
        raise ContextSyncError(
            "AGENTS.md documents ./run.sh flags that run.sh does not implement: "
            + ", ".join(sorted(unimplemented))
        )
    return [
        "AGENTS.md states the container-only rule",
        f"documented ./run.sh flags implemented by run.sh: {', '.join(flags) or 'none'}",
    ]


def check_stale_documentation_references(root: Path) -> list[str]:
    """Reject citations whose target disappeared during the context migration."""
    stale: list[str] = []
    scanned = 0
    for relative_root in LIVE_REFERENCE_ROOTS:
        search_root = root / relative_root
        if not search_root.exists():
            continue
        for path in search_root.rglob("*"):
            if not path.is_file() or path.suffix not in LIVE_TEXT_SUFFIXES:
                continue
            scanned += 1
            text = _read(path)
            for pattern, label in STALE_REFERENCE_PATTERNS:
                if pattern.search(text):
                    stale.append(f"{path.relative_to(root)} ({label})")

    if stale:
        raise ContextSyncError(
            "live files contain stale documentation references: " + ", ".join(sorted(stale))
        )
    return [f"live documentation references are current ({scanned} files checked)"]


CHECKS = (
    check_required_files,
    check_canonical_declaration,
    check_claude_entrypoint,
    check_agents_size,
    check_manifest_paths,
    check_container_rule,
    check_stale_documentation_references,
)


def run_checks(root: Path) -> list[str]:
    """Run every rule against ``root``; raise ContextSyncError on the first failure."""
    notes: list[str] = []
    for check in CHECKS:
        notes.extend(check(root))
    return notes


def main() -> None:
    root = Path.cwd()
    try:
        notes = run_checks(root)
    except ContextSyncError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    print("Context entrypoints are synchronized.")
    print("Codex entrypoint: AGENTS.md")
    print("Claude Code entrypoint: CLAUDE.md -> @AGENTS.md")
    for note in notes:
        print(f"  - {note}")


if __name__ == "__main__":
    main()
