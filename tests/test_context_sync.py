"""Tests for scripts/check_context_sync.py (CHE-7 / M0.5).

Each context-synchronization rule is exercised on both branches: the real
repository must pass, and a deliberately broken copy must raise
``ContextSyncError``. The script lives in scripts/ rather than in the installed
package, so it is loaded by path.
"""

from __future__ import annotations

import importlib.util
import re
import shutil
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "check_context_sync.py"

CONTEXT_FILES = ("AGENTS.md", "CLAUDE.md", "CONTEXT_MANIFEST.yaml", "run.sh")


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("check_context_sync", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ccs = _load_module()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A copy of the real context files plus the directories the manifest names."""
    for name in CONTEXT_FILES:
        shutil.copy(ROOT / name, tmp_path / name)
    for declared in ccs.manifest_declared_paths((tmp_path / "CONTEXT_MANIFEST.yaml").read_text()):
        target = tmp_path / declared
        if declared.endswith("/"):
            target.mkdir(parents=True, exist_ok=True)
        elif not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("placeholder\n", encoding="utf-8")
    return tmp_path


# --- pass branch -----------------------------------------------------------


def test_real_repository_passes_every_check() -> None:
    notes = ccs.run_checks(ROOT)
    assert notes
    assert any("CLAUDE.md -> @AGENTS.md" in note for note in notes)


def test_fixture_repository_passes_every_check(repo: Path) -> None:
    assert ccs.run_checks(repo)


# --- rule (a): AGENTS.md exists and is canonical ---------------------------


@pytest.mark.parametrize("name", ["AGENTS.md", "CLAUDE.md", "CONTEXT_MANIFEST.yaml"])
def test_missing_context_file_fails(repo: Path, name: str) -> None:
    (repo / name).unlink()
    with pytest.raises(ccs.ContextSyncError, match="missing required context file"):
        ccs.run_checks(repo)


def test_wrong_canonical_declaration_fails(repo: Path) -> None:
    manifest = repo / "CONTEXT_MANIFEST.yaml"
    manifest.write_text(
        manifest.read_text().replace(
            "canonical_static_context: AGENTS.md",
            "canonical_static_context: CLAUDE.md",
        ),
        encoding="utf-8",
    )
    with pytest.raises(ccs.ContextSyncError, match=re.escape("expected 'AGENTS.md'")):
        ccs.run_checks(repo)


def test_absent_canonical_declaration_fails(repo: Path) -> None:
    manifest = repo / "CONTEXT_MANIFEST.yaml"
    manifest.write_text(
        "\n".join(
            line
            for line in manifest.read_text().splitlines()
            if not line.startswith("canonical_static_context:")
        ),
        encoding="utf-8",
    )
    with pytest.raises(ccs.ContextSyncError, match="canonical_static_context"):
        ccs.run_checks(repo)


# --- rule (b): CLAUDE.md points at AGENTS.md -------------------------------


def test_claude_md_duplicating_agents_content_fails(repo: Path) -> None:
    (repo / "CLAUDE.md").write_text((repo / "AGENTS.md").read_text(), encoding="utf-8")
    with pytest.raises(ccs.ContextSyncError, match=re.escape("must contain exactly '@AGENTS.md'")):
        ccs.run_checks(repo)


def test_claude_md_with_extra_broad_import_fails(repo: Path) -> None:
    (repo / "CLAUDE.md").write_text("@AGENTS.md\n@docs/ARCHITECTURE.md\n", encoding="utf-8")
    with pytest.raises(ccs.ContextSyncError, match=re.escape("must contain exactly '@AGENTS.md'")):
        ccs.run_checks(repo)


# --- always-loaded context stays small -------------------------------------


def test_oversized_agents_md_fails(repo: Path) -> None:
    agents = repo / "AGENTS.md"
    agents.write_text(agents.read_text() + "\nfiller\n" * ccs.MAX_AGENTS_LINES, encoding="utf-8")
    with pytest.raises(ccs.ContextSyncError, match="keep it at or below"):
        ccs.run_checks(repo)


# --- rule (c): manifest paths exist ----------------------------------------


def test_manifest_path_that_does_not_exist_fails(repo: Path) -> None:
    manifest = repo / "CONTEXT_MANIFEST.yaml"
    manifest.write_text(
        manifest.read_text() + "    - docs/does_not_exist/\n",
        encoding="utf-8",
    )
    with pytest.raises(ccs.ContextSyncError, match="docs/does_not_exist/"):
        ccs.run_checks(repo)


def test_manifest_prose_entries_are_not_treated_as_paths() -> None:
    declared = ccs.manifest_declared_paths(
        "loading_policy:\n"
        "  task_linked_only:\n"
        "    - selected solver/coupler cards\n"
        "    - relevant source files\n"
        "    - docs/context/\n"
        "  # docs/commented_out/ is only a note\n"
    )
    assert declared == ["docs/context/"]


# --- rule (d): container-only execution ------------------------------------


@pytest.mark.parametrize("token", ccs.CONTAINER_RULE_TOKENS)
def test_agents_md_without_container_rule_fails(repo: Path, token: str) -> None:
    agents = repo / "AGENTS.md"
    agents.write_text(agents.read_text().replace(token, "REMOVED"), encoding="utf-8")
    with pytest.raises(ccs.ContextSyncError, match="container-only execution rule"):
        ccs.run_checks(repo)


def test_missing_run_sh_fails(repo: Path) -> None:
    (repo / "run.sh").unlink()
    with pytest.raises(ccs.ContextSyncError, match=re.escape("run.sh is missing")):
        ccs.run_checks(repo)


def test_documented_flag_not_implemented_by_run_sh_fails(repo: Path) -> None:
    agents = repo / "AGENTS.md"
    agents.write_text(
        agents.read_text() + "\n```bash\n./run.sh --invented-flag pytest -q\n```\n",
        encoding="utf-8",
    )
    with pytest.raises(ccs.ContextSyncError, match="--invented-flag"):
        ccs.run_checks(repo)


def test_both_documented_build_flags_are_implemented() -> None:
    """AGENTS.md documents --no-build and --rebuild; run.sh must handle both."""
    flags = ccs.documented_run_sh_flags((ROOT / "AGENTS.md").read_text())
    assert "--rebuild" in flags
    assert "--no-build" in flags
    run_sh = (ROOT / "run.sh").read_text()
    for flag in flags:
        assert flag in run_sh


# --- stale live documentation references ----------------------------------


@pytest.mark.parametrize(
    "stale_reference",
    [
        "CLAUDE" + ".md section 7",
        "docs/AGENT_" + "KNOWLEDGE_BASE.md section 2.1",
        "docs/SOLVER_AND_" + "COUPLER_CATALOG.md",
        "docs/ARCHI" + "TECTURE.md",
        "docs/BENCHMARK_" + "SPECIFICATION.md",
    ],
)
def test_stale_live_documentation_reference_fails(
    repo: Path, stale_reference: str
) -> None:
    source = repo / "src" / "stale_reference.py"
    source.parent.mkdir(exist_ok=True)
    source.write_text(f'"""See {stale_reference}."""\n', encoding="utf-8")

    with pytest.raises(ccs.ContextSyncError, match="stale documentation references"):
        ccs.run_checks(repo)
