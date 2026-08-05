"""Minimal provenance schema for reproducible benchmark runs."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


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
