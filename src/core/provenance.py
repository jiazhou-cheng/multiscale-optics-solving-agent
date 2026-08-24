"""Provenance schema, and the projection that turns a run record into a fingerprint.

A *scientific fingerprint* is the hash of what a run computed, with everything
about *this particular execution* removed. Two runs of the same computation on
the same inputs must produce the same fingerprint on different days, in
different directories, on differently-loaded machines -- otherwise the hash
answers "did I run this twice?" rather than "did the physics change?", and only
the second question is worth asking.

``VOLATILE_KEYS`` and ``strip_volatile`` are that projection. They lived in
``evaluation/m1_bundle.py`` until CHE-88, alongside the gen1 branch-bundle
machinery, and were imported from there by the two Level-2 benchmarks as private
names. Reproducibility is not an M1 concern, so they moved here rather than
being archived with the rest of that module.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "RECORD_PROVENANCE_KEY",
    "VOLATILE_KEYS",
    "RecordVerdict",
    "RunProvenance",
    "environment_fingerprint",
    "loaded_source_files",
    "record_provenance",
    "source_fingerprint",
    "strip_volatile",
    "verify_record_provenance",
]


#: Keys excluded from a scientific fingerprint, and why each is excluded.
#:
#: * ``runtime_seconds``, ``process_wall_seconds``, ``worker_process_seconds``,
#:   ``import_seconds``, ``setup_seconds`` -- timings. They vary with machine
#:   load, so hashing them makes every run unique and the hash worthless.
#: * ``timestamp_utc`` -- when, not what.
#: * ``run_id`` -- identity of the execution, which is the thing being projected
#:   out.
#: * ``output_directory`` -- where the bytes landed. A result that changes when
#:   you write it somewhere else is a bug, not a different result.
#:
#: Deliberately *not* excluded: the git dirty flag, package versions, device and
#: dtype. Those change what was computed, so a fingerprint that ignored them
#: would claim reproducibility across a real change.
VOLATILE_KEYS = (
    "runtime_seconds",
    "process_wall_seconds",
    "worker_process_seconds",
    "import_seconds",
    "setup_seconds",
    "timestamp_utc",
    "run_id",
    "output_directory",
)


def strip_volatile(value: Any) -> Any:
    """Recursively drop execution-identity keys from a nested result structure.

    Applied before hashing a result. Descends dicts and lists and leaves scalars
    alone; the filter is by key name at every depth, because a benchmark's
    nested per-case records carry their own timings.
    """
    if isinstance(value, dict):
        return {
            key: strip_volatile(item) for key, item in value.items() if key not in VOLATILE_KEYS
        }
    if isinstance(value, list):
        return [strip_volatile(item) for item in value]
    return value


class RunProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    created_at: datetime
    task_id: str | None = None
    source_commit: str | None = None
    graph_sha256: str
    environment_lock_sha256: str | None = None
    python_version: str
    packages: dict[str, str] = Field(default_factory=dict)
    hardware: dict[str, Any] = Field(default_factory=dict)
    seed: int | None = None
    prompt_sha256: str | None = None
    disclosed_knowledge: list[str] = Field(default_factory=list)
    repairs: list[dict[str, Any]] = Field(default_factory=list)
    human_interventions: list[dict[str, Any]] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Record provenance: telling "the code changed" from "the environment changed"
# ---------------------------------------------------------------------------
#
# CHE-103 (M0.2). ``strip_volatile`` above answers "did the physics change
# between two runs I have in front of me". It cannot answer the question that
# actually went wrong: three committed probe records stopped reproducing, and
# nothing failed, because the *only* thing asserting on them was a suite that
# read the file instead of re-running the probe. A record is expensive (~6 min
# each) and a test is cheap, so the loop was never closed and the drift sat in
# the repository for weeks.
#
# Re-running every probe in the required gate is not affordable. What *is*
# affordable is refusing to trust a record whose code has moved underneath it.
# So a record carries a fingerprint of the source it was produced by, and the
# gate recomputes that fingerprint from the current tree. The two fingerprints
# are kept separate on purpose:
#
#   code_fingerprint         -- the repository source the probe actually
#                               executed. Changes when we change the physics.
#   environment_fingerprint  -- interpreter and pinned solver versions.
#                               Changes when the image is rebuilt.
#
# A stale record then reports *which* of the two moved, which is exactly the
# distinction CHE-100 could not make and had to establish by hand with a second
# worktree.

RECORD_PROVENANCE_KEY = "record_provenance"

#: Packages whose version changes the numbers a probe produces. Pinned solvers
#: and array libraries only -- not the whole environment, which would make every
#: image rebuild invalidate every record regardless of relevance.
_FINGERPRINTED_PACKAGES = (
    "numpy",
    "scipy",
    "jax",
    "jaxlib",
    "torch",
    "optiland",
    "chromatix",
)


def _normalized_source_digest(path: Path) -> str:
    """A digest of what a source file *does*, with comments and docstrings out.

    Hashing the raw bytes would be simpler and wrong in a way that matters here:
    a probe record would be invalidated by a typo fix in a docstring, the
    six-minute regeneration would be pure ceremony, and the mechanism would be
    routed around within a month. Parsing to an AST and dumping it drops
    comments and formatting; stripping docstring expressions drops the rest of
    the prose. What survives is the code.

    This can still over-trigger -- renaming a local variable changes the dump --
    and that is the correct direction to fail in. It must never under-trigger
    for a semantic change, and it does not.
    """
    source = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source)
    except SyntaxError:
        # Not importable Python; the bytes are the best available statement.
        return hashlib.sha256(source.encode("utf-8")).hexdigest()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        body = node.body
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            node.body = body[1:] or [ast.Pass()]
    return hashlib.sha256(ast.dump(tree).encode("utf-8")).hexdigest()


def source_fingerprint(paths: Iterable[Path], *, root: Path) -> dict[str, Any]:
    """Fingerprint a set of source files, and report the per-file digests.

    The per-file digests are kept, not just the combined hash, so a stale record
    can say *which* module moved rather than only that something did. That is
    the difference between a failure someone can act on and one they re-derive
    by bisection, which is what CHE-100 had to do.
    """
    digests: dict[str, str] = {}
    for path in sorted(set(paths)):
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError:
            continue
        if not path.is_file():
            continue
        digests[relative] = _normalized_source_digest(path)
    combined = hashlib.sha256(
        "\n".join(f"{name}:{digest}" for name, digest in sorted(digests.items())).encode("utf-8")
    ).hexdigest()
    return {"combined_sha256": combined, "files": digests}


#: Directories whose imported modules are part of a record's code fingerprint.
#:
#: ``benchmarks/probes`` is in the list because probes import each other: the
#: first-null study takes the frozen geometry, the trace helper and the
#: perturbation harness from ``psf_oracle_verification``. Scanning only ``src``
#: left that record reporting "reproduces" after an edit to the geometry it was
#: measured on, which is the same hole one level out.
_FINGERPRINTED_TREES = ("src", "benchmarks/probes")


def loaded_source_files(root: Path) -> list[Path]:
    """Every repository source file currently imported, read off ``sys.modules``.

    Read rather than declared, for the CHE-61 reason: a hand-maintained list of
    "the modules this probe depends on" is a second thing to keep in sync, and
    the failure mode of getting it wrong is a record that silently stops being
    checked. Calling this at the *end* of a probe run captures what the probe
    actually imported, including the imports made inside functions that these
    probes use to keep module import cheap.

    The cost of reading rather than declaring is over-capture: anything else in
    the process is swept in too, so a record can be invalidated by a module it
    never used. That is the right direction to be wrong in -- the alternative
    under-captures and reports "reproduces" about a record that does not -- but
    it is why these are expensive records to keep enrolled.
    """
    trees = tuple((root / name).resolve() for name in _FINGERPRINTED_TREES)
    found: list[Path] = []
    for module in list(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if not filename:
            continue
        path = Path(filename).resolve()
        if any(path.is_relative_to(tree) for tree in trees):
            found.append(path)
    return found


def environment_fingerprint() -> dict[str, Any]:
    """Interpreter and pinned solver versions -- what changes speed and numbers."""
    versions: dict[str, str] = {}
    for name in _FINGERPRINTED_PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    payload = {
        "python_version": platform.python_version(),
        "packages": versions,
    }
    payload["combined_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return payload


def _git(root: Path, *args: str) -> str | None:
    try:
        return subprocess.run(
            ["git", *args], cwd=root, capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:  # pragma: no cover - no git in the container is survivable
        return None


def record_provenance(
    *,
    probe: str,
    root: Path,
    extra_sources: Iterable[Path] = (),
    data_inputs: Iterable[Path] = (),
) -> dict[str, Any]:
    """Build the provenance block a probe record carries.

    Call this at the END of a probe, after the physics has run, so that
    :func:`loaded_source_files` sees the modules the run actually imported.

    ``data_inputs`` covers the non-Python files a probe reads *values* out of --
    for these probes, ``benchmarks/protocols/slice_protocol.yaml``, which
    supplies the frozen system geometry and every gate threshold that then
    appears verbatim in the record. Without it, editing a gate value could not
    make a record stale, which would leave the loop open at exactly the place
    the numbers come from. These are hashed as bytes rather than normalized:
    there is no docstring/comment distinction to draw in a data file, and a
    comment beside a threshold is part of what the threshold means.
    """
    root = root.resolve()
    sources = [*loaded_source_files(root), *(Path(p).resolve() for p in extra_sources)]
    commit = _git(root, "rev-parse", "HEAD")
    status = _git(root, "status", "--porcelain")

    data_digests: dict[str, str] = {}
    for raw in sorted({Path(p).resolve() for p in data_inputs}):
        if not raw.is_file():
            continue
        try:
            relative = raw.relative_to(root).as_posix()
        except ValueError:
            continue
        data_digests[relative] = hashlib.sha256(raw.read_bytes()).hexdigest()

    return {
        "schema_version": 2,
        "probe": probe,
        # Named `timestamp_utc` so `strip_volatile` projects it out: a record's
        # scientific fingerprint must not change because it was regenerated on a
        # different day, and `VOLATILE_KEYS` already carries this spelling.
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "source_commit": commit,
        "working_tree_dirty": bool(status) if status is not None else None,
        "code_fingerprint": source_fingerprint(sources, root=root),
        "data_fingerprint": data_digests,
        "environment_fingerprint": environment_fingerprint(),
    }


@dataclass(frozen=True)
class RecordVerdict:
    """Whether a committed record still describes the tree it is read in."""

    record: str
    reproduces: bool
    code_changed: bool
    environment_changed: bool
    data_changed: bool = False
    changed_files: tuple[str, ...] = ()
    removed_files: tuple[str, ...] = ()
    changed_data: tuple[str, ...] = ()
    detail: str = ""

    def explain(self) -> str:
        if self.reproduces:
            return f"{self.record}: code, data and environment fingerprints all match"
        parts = [f"{self.record} is STALE"]
        if self.code_changed:
            changes = []
            if self.changed_files:
                changes.append(f"modified {', '.join(self.changed_files[:6])}")
            if self.removed_files:
                changes.append(f"no longer imported/removed {', '.join(self.removed_files[:6])}")
            parts.append("the CODE changed: " + ("; ".join(changes) or "unknown files"))
        if self.data_changed:
            parts.append(f"the DATA INPUTS changed: {', '.join(self.changed_data[:6])}")
        if self.environment_changed:
            parts.append("the ENVIRONMENT changed (interpreter or a pinned solver version)")
        if self.detail:
            parts.append(self.detail)
        return ". ".join(parts)


def verify_record_provenance(payload: dict[str, Any], *, root: Path, name: str) -> RecordVerdict:
    """Recompute a record's fingerprints from the current tree and compare.

    This is the mechanism that makes a drifted record *fail* instead of being
    read as evidence. It does not re-run the probe -- it establishes that the
    probe, re-run, would produce the same thing, which is the affordable half of
    the guarantee and the half that was missing.
    """
    root = root.resolve()
    block = payload.get(RECORD_PROVENANCE_KEY)
    if not isinstance(block, dict):
        return RecordVerdict(
            record=name,
            reproduces=False,
            code_changed=False,
            environment_changed=False,
            detail=(
                f"carries no {RECORD_PROVENANCE_KEY!r} block, so nothing can say whether it "
                "still describes this tree. Regenerate it through a probe that calls "
                "core.provenance.record_provenance, or declare it in the unenrolled register"
            ),
        )

    recorded_files = dict(block.get("code_fingerprint", {}).get("files", {}))
    current = source_fingerprint((root / name for name in recorded_files), root=root)
    current_files = current["files"]

    changed = tuple(
        sorted(f for f, d in recorded_files.items() if f in current_files and current_files[f] != d)
    )
    removed = tuple(sorted(f for f in recorded_files if f not in current_files))
    code_changed = bool(changed or removed)

    recorded_data = dict(block.get("data_fingerprint", {}))
    changed_data = tuple(
        sorted(
            relative
            for relative, digest in recorded_data.items()
            if not (root / relative).is_file()
            or hashlib.sha256((root / relative).read_bytes()).hexdigest() != digest
        )
    )

    recorded_env = block.get("environment_fingerprint", {}).get("combined_sha256")
    environment_changed = bool(
        recorded_env is not None and recorded_env != environment_fingerprint()["combined_sha256"]
    )

    return RecordVerdict(
        record=name,
        reproduces=not (code_changed or changed_data or environment_changed),
        code_changed=code_changed,
        environment_changed=environment_changed,
        data_changed=bool(changed_data),
        changed_files=changed,
        removed_files=removed,
        changed_data=changed_data,
    )
