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

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["VOLATILE_KEYS", "RunProvenance", "strip_volatile"]


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
            key: strip_volatile(item)
            for key, item in value.items()
            if key not in VOLATILE_KEYS
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
