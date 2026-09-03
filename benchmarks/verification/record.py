"""One row of verification evidence, and the provenance every record carries.

CHE-238's record contract: *"git SHA; branch; timestamp; command; CPU/GPU/device;
package + environment versions (including installed `optiland` version);
dtype/precision; numerical parameters; random seed; runtime; peak memory;
metrics; status; failure/refusal reason."*

Separate from `benchmarks/record.py` and deliberately not an extension of it.
That module's vocabulary is `gate` / `control` / `oracle_kind`, and its whole
point is that a *closed form* decides. Nothing in this tree has a closed form:
the comparisons are between this project's operation and the third-party tool it
delegates to, which `AGENTS.md` classifies as evidence that may not gate. Reusing
`gate()` here would produce records that read as decided when they are not, and
`tests/benchmarks/test_records.py` would then have to be told to skip them --
which is the tell that they are a different kind of thing.

So the vocabulary here is `status` from CHE-238's own list, and there is no
`passed` field, no `tolerance`, and no `oracle_kind`. A `PASS` row means the two
implementations agreed to the stated numerical tolerance of the comparison, and
the row says in `note` what that agreement is and is not evidence of.
"""

from __future__ import annotations

import json
import platform
import resource
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

__all__ = [
    "METRE_PER_MM",
    "STATUSES",
    "Row",
    "device_execution",
    "finish",
    "provenance",
]

#: Optiland reports lengths in millimetres and this project's records are SI, so
#: every native length crosses this one constant. Named rather than inlined
#: because a bare `1e-3` in a comparison is the kind of factor that goes missing.
METRE_PER_MM = 1e-3

#: CHE-238's status vocabulary, plus the four CHE-241/242 §4.4 extensions and one
#: of this harness's own. A closed set so an aggregate count cannot be thrown off
#: by a typo.
#:
#: `BASELINE` is the addition, and it exists because `PASS` was being used for two
#: different things. `PASS` means *two implementations were compared and agreed*.
#: A row that runs a catalogued operation on a system for which no upstream golden
#: exists -- every `SOM_SPOT_DIAGRAM` and `SOM_PSF` row in Tier 2, because the
#: tutorial prints nothing those descriptors return -- has no oracle, no expected
#: value and no delta. Counting it as `PASS` inflates the agreement count with
#: rows that compared nothing, which is the "unverified claimed as verified" line
#: in `AGENTS.md`. A `BASELINE` row is a recorded value for a future comparison to
#: be made against, and it decides nothing.
STATUSES: tuple[str, ...] = (
    "PASS",
    "BASELINE",
    "PASS-refused",
    "FAIL",
    "BLOCKED",
    "NOT-COVERED",
    "PASS-native",
    "PASS-graph-only",
    "PASS-transcribed",
    "BLOCKED-no-backend",
    "BLOCKED-untranscribable",
    "BLOCKED-preflight",
)


@dataclass(frozen=True)
class Row:
    """One comparison, one refusal, or one coverage gap.

    A frozen dataclass on minimality rule 2 -- it is the versioned shape the
    record's JSON has, and three drivers write it -- rather than a bare dict,
    because `status` has to be checked against `STATUSES` somewhere and a dict
    has nowhere to put that.
    """

    case: str
    configuration: dict[str, Any]
    descriptor: str
    status: str
    measured: dict[str, Any]
    expected: dict[str, Any]
    deltas: dict[str, float]
    worst_relative_delta: float
    runtime_s: float
    note: str
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise ValueError(f"status={self.status!r} is not one of {list(STATUSES)}")
        if not self.note.strip():
            raise ValueError(
                f"row {self.case!r} carries no note. A number with no statement of what it is "
                "evidence for is what this whole run exists not to produce"
            )
        # The two statuses that make a claim about agreement have to have
        # something to have agreed with. Enforced rather than trusted: the whole
        # failure mode `BASELINE` was added for is a row that compared nothing
        # being counted as a row that compared something.
        if self.status in ("PASS", "FAIL") and not (self.expected or self.deltas):
            raise ValueError(
                f"row {self.case!r} claims {self.status} with no expected value and no delta, "
                "so it compared nothing. A recorded value with no oracle is BASELINE"
            )
        if self.status == "BASELINE" and (self.expected or self.deltas):
            raise ValueError(
                f"row {self.case!r} is BASELINE but carries an expected value or a delta; a row "
                "with an oracle is PASS or FAIL"
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "case": self.case,
            "configuration": self.configuration,
            "descriptor": self.descriptor,
            "status": self.status,
            "measured": self.measured,
            "expected": self.expected,
            "deltas": self.deltas,
            "worst_relative_delta": self.worst_relative_delta,
            "runtime_s": self.runtime_s,
            "note": self.note,
            **({"extra": self.extra} if self.extra else {}),
        }


def device_execution() -> dict[str, str]:
    """The `Execution` every operation in this tree is called with.

    CPU and float64 throughout, and both are recorded on the row rather than
    inherited: the container's jaxlib and torch are CPU-only builds (CHE-238 §3),
    so a device choice here would be a claim the environment cannot honour, and
    `numerics.Precision` requires the caller to name a precision rather than
    default to one.
    """
    return {"device": "cpu", "precision": "fp64"}


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], capture_output=True, text=True, check=False
    ).stdout.strip()


def provenance() -> dict[str, Any]:
    """Everything CHE-238's contract requires that is not a measured number."""
    packages: dict[str, str] = {}
    for name in ("numpy", "scipy", "jax", "jaxlib", "chromatix", "optiland", "torch"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = "not installed"
    return {
        "written_at_utc": datetime.now(UTC).isoformat(),
        "git": {
            "commit": _git("rev-parse", "HEAD"),
            "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(_git("status", "--porcelain")),
        },
        "command": " ".join(sys.argv),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": packages,
            "device": device_execution(),
        },
    }


def finish(record: dict[str, Any], *, path: Path, rows: list[Row]) -> int:
    """Write the record, print the aggregate, and return a process exit code.

    Exit non-zero only on `FAIL`. `BLOCKED`, `NOT-COVERED` and `PASS-refused` are
    *results* of this workstream and not errors of it -- CHE-239 §A.2 is explicit
    that a documented refusal is not a generic failure -- so a run that produces
    nothing but refusals still exits 0 and the report says what it found.
    """
    counts: dict[str, int] = {}
    for row in rows:
        counts[row.status] = counts.get(row.status, 0) + 1
    record["status_counts"] = counts
    record["peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=False) + "\n", encoding="utf-8")

    print("\n" + "-" * 72)
    for status in STATUSES:
        if status in counts:
            print(f"  {status:24s} {counts[status]:4d}")
    print(f"  {'total':24s} {len(rows):4d}")
    print(f"  peak RSS {record['peak_rss_kib'] / 1024:.0f} MiB")
    print(f"  record   {path}")
    return 1 if counts.get("FAIL") else 0
