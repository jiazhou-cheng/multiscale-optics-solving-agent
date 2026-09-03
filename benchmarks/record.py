"""The record a benchmark writes: one JSON file per configuration.

CHE-212 (R06.7) acceptance criterion 5, which asks for "a record of what was run
-- configuration, measured values, tolerances, and which gates are closed-form vs
diagnostic -- written to the repository, in whatever minimal format this ticket
establishes".

The format is a flat JSON document per configuration, and it is deliberately
small. What it holds is the four things a reader needs in order to decide whether
to believe a number: the configuration it came from, the measured value, the
oracle it was compared against, and whether that oracle is **closed form** (may
gate) or **diagnostic** (may not). The reference implementation's family /
instance / verifier machinery is not recreated, because nothing in R06.7 or R06.8
needs a registry of oracles -- there are two benchmarks and each one owns its own.

What is *not* here, on purpose
------------------------------
No git commit and no environment fingerprint beyond the installed package
versions. Execution provenance is R13's (CHE-165) subject and it will want more
than a hash; writing half of it now would mean a second place to change when that
lands. A record here says what was measured and against what, and the reader gets
the code state from the commit that contains the record.

No pass/fail summary field either. `gate` entries carry `passed`, and a benchmark
that failed a gate exits non-zero -- a top-level boolean would be a second copy
of the same fact that could disagree with the entries under it.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

__all__ = [
    "ORACLE_KINDS",
    "control",
    "describe_plan",
    "describe_request",
    "environment",
    "gate",
    "write_record",
]

#: What a comparison is allowed to decide.
#:
#: `closed_form` -- arithmetic, independent of this repository's numerics. It may
#: gate.
#: `diagnostic` -- another implementation's number, including one of this
#: repository's own. It is evidence and it may **not** gate; AGENTS.md's rule is
#: that repository numerical code must not be the sole correctness oracle for the
#: same numerical code, and a differential check between two of our own paths is
#: exactly that.
ORACLE_KINDS: tuple[str, ...] = ("closed_form", "diagnostic")


def gate(
    name: str,
    *,
    oracle: str,
    oracle_kind: str,
    measured: Any,
    expected: Any,
    tolerance: float | None,
    tolerance_basis: str,
    passed: bool,
) -> dict[str, Any]:
    """One comparison, with the oracle it was made against and its standing.

    `tolerance_basis` is required and free text, and it is the field that keeps
    this honest: a tolerance with no stated derivation is a fitted one, and
    AGENTS.md forbids widening a tolerance to make a benchmark pass.
    """
    if oracle_kind not in ORACLE_KINDS:
        raise ValueError(f"oracle_kind={oracle_kind!r} is not one of {list(ORACLE_KINDS)}")
    if not tolerance_basis.strip():
        raise ValueError(
            f"gate {name!r} has no tolerance_basis. A tolerance with no stated derivation "
            "is a fitted one."
        )
    return {
        "name": name,
        "oracle": oracle,
        "oracle_kind": oracle_kind,
        "measured": measured,
        "expected": expected,
        "tolerance": tolerance,
        "tolerance_basis": tolerance_basis,
        "passed": bool(passed),
    }


def control(
    name: str,
    *,
    changed: str,
    breaks_gate: str,
    measured: Any,
    reference: Any,
    broke: bool,
) -> dict[str, Any]:
    """One negative control: what was changed, which gate it must break, and did it.

    A control that *passes* the gate it was supposed to break is a failure of the
    benchmark, not of the control, so `broke` is what a runner checks.
    """
    return {
        "name": name,
        "changed": changed,
        "breaks_gate": breaks_gate,
        "measured": measured,
        "reference": reference,
        "broke_the_gate": bool(broke),
    }


def describe_request(request: Mapping[str, Any]) -> dict[str, Any]:
    """One node's request as JSON: scalars verbatim, bulk arrays as a digest.

    A record has to say *exactly* what was executed, and a plan's per-node requests
    are what says it -- the two focal lengths, the two target surfaces, the
    normalization. Those are scalars and they are written out.

    An amplitude mask is not: it is 192 x 256 float64, and embedding it would make
    the record a data file. It is described by its shape, its dtype and a sha256 of
    its bytes, which is what makes the claim checkable rather than decorative -- two
    runs that filtered through different pupils have different digests, and a reader
    who wants to know *which* mask can recompute one. A shape alone would not
    distinguish an open pupil from a stop.

    Anything else -- a `ReferenceSurface`, an enum -- is `repr`'d. That is lossy and
    is meant to be: the record is provenance, and the code that built the request is
    what a re-run reads.
    """
    return {name: _describe(value) for name, value in sorted(request.items())}


def _describe(value: Any) -> Any:
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, Mapping):
        return {str(key): _describe(item) for key, item in value.items()}
    buffer = getattr(value, "tobytes", None)
    shape = getattr(value, "shape", None)
    if buffer is not None and shape is not None:
        return {
            "array": list(shape),
            "dtype": str(getattr(value, "dtype", "unknown")),
            "sha256": hashlib.sha256(buffer()).hexdigest()[:16],
        }
    if isinstance(value, Sequence):
        return [_describe(item) for item in value]
    return repr(value)


def describe_plan(
    steps: Sequence[tuple[str, Mapping[str, Any]]], *, chain: Sequence[str] = ()
) -> list[dict[str, Any]]:
    """A normalized plan as a list of `{step, operation, produces, request}`.

    `chain` is the semantic type each step produces, from the caller's edge check
    against `planning.capability_graph()`; it is recorded beside the ids so a reader
    can see that the plan is a walk of the capability graph and not an arbitrary
    ordering. Omitted, the field is absent rather than guessed.
    """
    described: list[dict[str, Any]] = []
    for index, (operation_id, request) in enumerate(steps):
        entry: dict[str, Any] = {
            "step": index,
            "operation": operation_id,
            "request": describe_request(request),
        }
        if index < len(chain):
            entry["produces"] = chain[index]
        described.append(entry)
    return described


def environment() -> dict[str, str]:
    """Installed versions of the packages a wave result depends on.

    Read through `importlib.metadata`, which does **not** import them: a record
    writer must not be the thing that pulls JAX into the process, and the
    benchmark's own composition path is the only sanctioned route to the backend.
    """
    versions: dict[str, str] = {}
    for name in ("numpy", "scipy", "jax", "chromatix"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:  # pragma: no cover - both images pin these
            versions[name] = "not installed"
    return versions


def write_record(record: dict[str, Any], *, path: Path) -> Path:
    """Write one record as indented JSON with a trailing newline, and return the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "written_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "environment": environment(),
        **record,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return path
