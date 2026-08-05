"""Serializable outputs of numerical, physical, and derivative verification checks."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class CheckStatus(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    INCONCLUSIVE = "inconclusive"
    NOT_APPLICABLE = "not_applicable"


class VerificationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    check_id: str
    kind: str
    target: str
    status: CheckStatus
    value: float | None = None
    tolerance: float | None = None
    units: str | None = None
    diagnostics: dict[str, Any] = Field(default_factory=dict)
    evidence_uris: list[str] = Field(default_factory=list)
    message: str = ""
