"""Generated files must still equal what generates them.

`schemas/*.json` are produced from the pydantic models by
`scripts/export_schemas.py` and are the planner-facing contract. Nothing checked
that the committed copies still matched, and at the Phase 0 baseline of CHE-84
they did not: `coupler.schema.json` was missing a `Device` enum plus `devices`,
`dtypes` and `pinned_commit`, and `model.schema.json` was missing
`pinned_commit`. The models had grown those fields and the artifacts were never
regenerated.

Nothing failed in the meantime, which is the point. A stale generated schema
does not break a run -- it silently describes a component shape that no longer
exists, to a reader who has no way to tell.

Regenerated in memory and compared, so the test never writes and cannot
accidentally "fix" a drift it was meant to report.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.artifacts import ArtifactRecord
from core.execution_record import ExecutionRecord
from core.provenance import RunProvenance
from core.specs import CouplerSpec, GraphSpec, ModelSpec
from discovery.api import (
    ComponentDescription,
    ConnectionReport,
    RouteCapability,
    ValidityAnswer,
)
from verification.result import VerificationResult

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_DIR = ROOT / "schemas"

#: Mirrors `scripts/export_schemas.py`'s SCHEMAS. Duplicated deliberately rather
#: than imported: importing the script's table would make this test agree with
#: the script by construction, including about which schemas exist at all.
EXPECTED = {
    "artifact.schema.json": ArtifactRecord,
    "coupler.schema.json": CouplerSpec,
    "execution_record.schema.json": ExecutionRecord,
    "component_description.schema.json": ComponentDescription,
    "connection_report.schema.json": ConnectionReport,
    "graph.schema.json": GraphSpec,
    "model.schema.json": ModelSpec,
    "provenance.schema.json": RunProvenance,
    "route_capability.schema.json": RouteCapability,
    "validity_answer.schema.json": ValidityAnswer,
    "verification_result.schema.json": VerificationResult,
}


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_the_committed_schema_matches_its_model(name: str) -> None:
    model = EXPECTED[name]
    path = SCHEMA_DIR / name
    assert path.is_file(), f"{name} is missing; run scripts/export_schemas.py"

    committed = json.loads(path.read_text(encoding="utf-8"))
    generated = model.model_json_schema()
    assert committed == generated, (
        f"schemas/{name} no longer matches {model.__name__}. Regenerate it:\n"
        "    ./run.sh python scripts/export_schemas.py\n"
        "and commit the result in the same change as the model edit. A generated "
        "artifact that lags its generator describes a shape that does not exist."
    )


def test_every_committed_schema_is_still_generated() -> None:
    """A schema nobody generates is a file nobody maintains.

    Catches the other direction: a model removed from the exporter leaves its
    JSON behind, and that file then looks like a live contract forever.
    """
    committed = {path.name for path in SCHEMA_DIR.glob("*.json")}
    assert committed == set(EXPECTED), (
        f"schemas/ holds {sorted(committed)} but the exporter produces "
        f"{sorted(EXPECTED)}. Delete the orphan, or add the model back to "
        "scripts/export_schemas.py and to this test."
    )


def test_the_exporter_writes_exactly_the_files_this_test_checks() -> None:
    """Guards the duplication above: the two tables must not silently diverge.

    Read from the source rather than imported, because importing the script to
    check the script proves nothing.
    """
    source = (ROOT / "scripts" / "export_schemas.py").read_text(encoding="utf-8")
    for name, model in EXPECTED.items():
        assert f'"{name}": {model.__name__}' in source, (
            f"scripts/export_schemas.py no longer maps {name} to {model.__name__}. "
            "If the mapping changed on purpose, change it here too -- this test "
            "duplicates the table so a drift is a failure rather than an agreement."
        )
