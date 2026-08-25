"""Export JSON Schemas used by planners, editors, and external tooling."""

from __future__ import annotations

import json
from pathlib import Path

from core.artifacts import ArtifactRecord
from core.execution_record import ExecutionRecord
from core.provenance import RunProvenance
from core.specs import CouplerSpec, GraphSpec, ModelSpec
from verification.result import VerificationResult

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_DIR = ROOT / "schemas"

SCHEMAS = {
    "artifact.schema.json": ArtifactRecord,
    "coupler.schema.json": CouplerSpec,
    # CHE-113/CHE-132: what an executor emits and what a verifier says about it.
    # These replace `benchmarks/schemas/result.schema.json`, whose `benchmark_id`
    # and `protocol_id` enums hard-coded the retired L1/L2/L3 taxonomy -- a schema
    # that could not describe a result outside a task set that no longer exists.
    "execution_record.schema.json": ExecutionRecord,
    "graph.schema.json": GraphSpec,
    "model.schema.json": ModelSpec,
    "provenance.schema.json": RunProvenance,
    "verification_result.schema.json": VerificationResult,
}


def main() -> None:
    SCHEMA_DIR.mkdir(parents=True, exist_ok=True)
    for name, model in SCHEMAS.items():
        path = SCHEMA_DIR / name
        path.write_text(
            json.dumps(model.model_json_schema(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
