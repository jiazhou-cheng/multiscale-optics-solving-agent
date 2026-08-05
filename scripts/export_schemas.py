"""Export JSON Schemas used by planners, editors, and external tooling."""

from __future__ import annotations

import json
from pathlib import Path

from multiscale_optics_agent.core.artifacts import ArtifactRecord
from multiscale_optics_agent.core.provenance import RunProvenance
from multiscale_optics_agent.core.specs import CouplerSpec, GraphSpec, ModelSpec

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_DIR = ROOT / "schemas"

SCHEMAS = {
    "artifact.schema.json": ArtifactRecord,
    "coupler.schema.json": CouplerSpec,
    "graph.schema.json": GraphSpec,
    "model.schema.json": ModelSpec,
    "provenance.schema.json": RunProvenance,
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
