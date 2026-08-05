#!/usr/bin/env python3
"""Validate the static Codex/Claude Code context entrypoints.

Run from the repository root after copying this script into ``scripts/``.
"""

from __future__ import annotations

from pathlib import Path
import sys

MAX_AGENTS_LINES = 200


def fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def main() -> None:
    root = Path.cwd()
    agents = root / "AGENTS.md"
    claude = root / "CLAUDE.md"
    manifest = root / "CONTEXT_MANIFEST.yaml"

    for path in (agents, claude, manifest):
        if not path.is_file():
            fail(f"missing required context file: {path.name}")

    agents_text = agents.read_text(encoding="utf-8")
    claude_text = claude.read_text(encoding="utf-8").strip()

    if claude_text != "@AGENTS.md":
        fail("CLAUDE.md must contain exactly '@AGENTS.md' to avoid duplicated static context")

    line_count = len(agents_text.splitlines())
    if line_count > MAX_AGENTS_LINES:
        fail(f"AGENTS.md has {line_count} lines; keep it at or below {MAX_AGENTS_LINES}")

    forbidden = (
        "@PROJECT_PLAN.md",
        "@docs/ARCHITECTURE.md",
        "@docs/BENCHMARK_SPECIFICATION.md",
        "@docs/SOLVER_AND_COUPLER_CATALOG.md",
    )
    for token in forbidden:
        if token in claude_text:
            fail(f"CLAUDE.md imports broad startup context: {token}")

    print("Context entrypoints are synchronized.")
    print(f"Canonical file: {agents.name} ({line_count} lines)")
    print("Codex entrypoint: AGENTS.md")
    print("Claude Code entrypoint: CLAUDE.md -> @AGENTS.md")


if __name__ == "__main__":
    main()