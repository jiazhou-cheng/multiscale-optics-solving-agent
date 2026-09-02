"""The backend-free loader for the measured capability records in `knowledge/`.

CHE-223 (R03.6). `numerics/precision.py` used to hold `OPTILAND_CAPABILITIES` and
`CHROMATIX_CAPABILITIES` as module constants, with a `COMPONENT_CAPABILITIES` dict
and two name-bound helpers over them. Its own comment called that a bootstrap
anchor: R02.1 needed something real for `negotiate()` to negotiate against, and
neither `solvers/` nor `operations/` existed yet.

Why the rows are data and not solver-package state
--------------------------------------------------
CHE-206 planned to move each row into `solvers/<backend>/`. **That plan is
superseded, and not because it was wrong when written** -- it was written before
either consumer existed. There are now two, on opposite sides of the dependency
graph:

* **solver runtime** -- `solvers/optiland/solver.py` reads the row at roughly
  fifteen refusal sites; `solvers/chromatix/fields.py` negotiates against it;
* **backend-free discovery** -- `operations/` cites a component id, and the whole
  point of that package is that reading it loads no backend.

Solver-local ownership cannot be the single source for both. It would force one of
`operations -> solvers` (which `scripts/check_dependencies.py` names as the edge
that ends the only property `operations/` has), a duplicate copy inside
`operations/`, or a backend-adjacent import behind every capability query.

So the measured rows are **shared data**: one JSON file per component under
`knowledge/capabilities/`, named by component id, so "which file is canonical for
this id" needs no index. `numerics/` keeps the contract -- `ComponentCapabilities`
with all ten of its widening refusals, `negotiate()` and every refusal code -- and
gains this loader, which turns a record into a validated declaration. It keeps no
copy of the data, which is the old `knowledge/README.md` rule ("never two copies")
honoured through its opposite mechanism.

No hard-coded inventory
-----------------------
Nothing here names a backend or a component. `capability_record_ids()` enumerates
the directory, and an unknown id is refused by naming **what is on disk** rather
than by consulting a constant. That is the difference between a loader and a
second registry: adding a measured component is adding a file.

Where the records are, and what happens off a source checkout
-------------------------------------------------------------
`KNOWLEDGE_ROOT` is resolved from this file: `src/numerics/knowledge.py` ->
`<repo>/knowledge`. Stated as a decision rather than left implicit, because the
project is installed with `pip install --no-deps -e .` against a mounted checkout
(`docker/Dockerfile`, `run.sh`) and a real wheel would not ship a repository-root
directory. The alternative -- package data under `src/numerics/` -- was rejected
because `knowledge/` is a **pack**, shared with whatever else comes to cite it, and
burying it inside one package would make `numerics` its owner again. The same
resolution is what `benchmarks/` and the probe-citation test already use.

A missing directory therefore fails with a message that says which path was tried,
rather than reporting "no such component".

Nothing here imports a backend, and neither does loading a record: it is `json`,
`pathlib` and this project's own enums.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from numerics.precision import (
    ArrayNamespace,
    ComponentCapabilities,
    DeviceKind,
    DType,
    Precision,
    refusal,
)

__all__ = [
    "CAPABILITY_DIRECTORY",
    "CAPABILITY_SCHEMA_VERSION",
    "KNOWLEDGE_ROOT",
    "capability_record_ids",
    "capability_rows",
    "load_capabilities",
]

#: The pack root, resolved from this file rather than from the working directory.
#: See the module docstring for why it is the repository root and not package data.
KNOWLEDGE_ROOT: Path = Path(__file__).resolve().parents[2] / "knowledge"

#: Where the measured component rows live. One file per component id.
CAPABILITY_DIRECTORY: Path = KNOWLEDGE_ROOT / "capabilities"

#: The record schema this loader understands.
#:
#: Validated, not decorative: a record declaring any other version is refused, and
#: `tests/knowledge/test_capability_loader.py` proves the refusal. A version field
#: nothing checks is worse than none, because it reads as a compatibility guarantee.
CAPABILITY_SCHEMA_VERSION = 1

#: Exactly the keys a capability record must carry. Both directions are checked:
#: a missing key and an unknown key are each refused, so a typo cannot become a
#: silently defaulted field and a stale key cannot linger unread.
_REQUIRED_KEYS: frozenset[str] = frozenset(
    {
        "schema_version",
        "component",
        "probe",
        "probe_tag",
        "evidence",
        "notes",
        "devices",
        "precisions",
        "minimum_compute_precision",
        "accepted_input_dtypes",
        "native_compute_dtypes",
        "output_dtypes",
        "lossy_input_dtypes",
        "device_namespaces",
    }
)


#: The prefix a duplicate-key `ValueError` from `_no_duplicates` carries, so
#: `load_capabilities` can tell it from a `JSONDecodeError`.
#:
#: A marker string and **not an exception class**, which is a real constraint and
#: not a style choice: `scripts/class_budget.py` counts every `ClassDef` under
#: `src/`, private and exception types included, and `numerics` is at 7/7 with the
#: project's last authorized unit reserved for `runtime.Executor` (CHE-200 / R13.2).
#: A three-line private exception here would spend another ticket's authorization,
#: so the signal travels as a message prefix instead.
_DUPLICATE_KEY_PREFIX = "duplicate capability record key: "


def _no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """`dict(pairs)`, refusing a key that appears twice.

    JSON permits repeated keys and Python's decoder takes the last. So a record
    with `"accepted_input_dtypes"` twice -- once narrow, once wide -- loads as the
    wide one, which is exactly the silent widening this loader exists to refuse, and
    no required-keys check can see it because both spellings are the *same* key.

    Raises a plain `ValueError` carrying `_DUPLICATE_KEY_PREFIX`. `json` does not
    wrap an exception from `object_pairs_hook`, so it arrives at the caller intact.
    """
    seen: dict[str, Any] = {}
    for key, value in pairs:
        if key in seen:
            raise ValueError(f"{_DUPLICATE_KEY_PREFIX}{key}")
        seen[key] = value
    return seen


def _invalid(component: str, message: str, *, path: Path) -> ValueError:
    """The one refusal this module raises for a record it cannot use.

    `INVALID_CAPABILITY_DECLARATION` rather than a new code: a record that cannot
    be parsed and a record that is wider than its probe are the same failure from
    a caller's point of view -- there is no validated capability -- and CHE-223
    adds no refusal vocabulary.
    """
    return refusal(
        code="INVALID_CAPABILITY_DECLARATION",
        component=component,
        message=f"{message} ({path})",
        requested=component,
    )


def capability_record_ids(*, directory: Path | None = None) -> tuple[str, ...]:
    """Every component id with a record on disk, sorted.

    The replacement for the deleted `COMPONENT_CAPABILITIES` keys, and the reason
    it is a replacement rather than a rename: this enumerates the directory, so
    adding a measured component is adding a file and editing no source.
    """
    root = CAPABILITY_DIRECTORY if directory is None else directory
    if not root.is_dir():
        raise refusal(
            code="UNKNOWN_COMPONENT",
            component="knowledge/capabilities",
            message=f"the capability pack directory does not exist at {root}.",
            requested=str(root),
            remedy=(
                "The records are repository-root data resolved from "
                "`numerics.knowledge.KNOWLEDGE_ROOT`, so this path is expected to exist "
                "in a source checkout. See that module's docstring."
            ),
        )
    return tuple(sorted(path.stem for path in root.glob("*.json")))


def _decode_set(
    raw: Any, enum: Any, *, component: str, field: str, path: Path
) -> frozenset[Any]:
    """One list of enum *values* as a frozenset, refusing anything unrecognized.

    Resolved by value through the enum's own constructor, never by `getattr` on the
    class: `getattr(DType, name)` would accept `DType.mro` and any other attribute
    that happens to exist, which is how a hand-edited record becomes a capability
    nobody measured.
    """
    if not isinstance(raw, list) or any(not isinstance(item, str) for item in raw):
        raise _invalid(component, f"`{field}` must be a list of strings", path=path)
    resolved = []
    for item in raw:
        try:
            resolved.append(enum(item))
        except ValueError as exc:
            raise _invalid(
                component,
                f"`{field}` names {item!r}, which is not a declared "
                f"{enum.__name__} ({sorted(member.value for member in enum)})",
                path=path,
            ) from exc
    return frozenset(resolved)


def load_capabilities(
    component: str, *, directory: Path | None = None
) -> ComponentCapabilities:
    """One measured capability record, as a validated `ComponentCapabilities`.

    Constructed through the same `__post_init__` every in-tree declaration went
    through, so all ten widening refusals apply to a record on disk exactly as
    they applied to a module constant. A record wider than its probe is refused
    with `INVALID_CAPABILITY_DECLARATION` at load, not at first use.

    Parameters
    ----------
    component
        The component id, which is also the file stem.
    directory
        The capability directory, for tests. Defaults to `CAPABILITY_DIRECTORY`.
        Present so a synthetic record can be loaded without editing any constant
        in this module -- the property that makes this a loader rather than a
        registry.

    Raises:
        ValueError: with `code='UNKNOWN_COMPONENT'` when no record exists, naming
            the ids that do; with `code='INVALID_CAPABILITY_DECLARATION'` when the
            record is unreadable, structurally wrong, or wider than it measured.
    """
    root = CAPABILITY_DIRECTORY if directory is None else directory
    path = root / f"{component}.json"
    if not path.is_file():
        raise refusal(
            code="UNKNOWN_COMPONENT",
            component=component,
            message="no executable capability declaration exists for this component.",
            requested=component,
            supported=capability_record_ids(directory=root),
            remedy=(
                "Add one with the probe evidence behind it, as "
                f"{root}/{component}.json. A component with no declaration has no "
                "validated device or dtype support, and this project will not guess one."
            ),
        )
    try:
        record = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_no_duplicates)
    except json.JSONDecodeError as exc:
        raise _invalid(component, f"the record is not valid JSON: {exc}", path=path) from exc
    except ValueError as exc:
        # `JSONDecodeError` is itself a `ValueError`, so the clause above runs first
        # and this one only ever sees `_no_duplicates`'s. Anything else is re-raised
        # rather than relabelled as a malformed record.
        if not str(exc).startswith(_DUPLICATE_KEY_PREFIX):
            raise
        raise _invalid(
            component,
            f"the record declares {str(exc).removeprefix(_DUPLICATE_KEY_PREFIX)!r} more "
            "than once; JSON's last-wins rule would silently pick one, which is how a "
            "widened set arrives",
            path=path,
        ) from exc
    if not isinstance(record, dict):
        raise _invalid(component, "the record must be a JSON object", path=path)

    keys = set(record)
    missing = _REQUIRED_KEYS - keys
    unknown = keys - _REQUIRED_KEYS
    if missing or unknown:
        raise _invalid(
            component,
            f"the record's keys are wrong: missing {sorted(missing)}, "
            f"unrecognized {sorted(unknown)}",
            path=path,
        )
    # `is not int` rather than `!=`: JSON `1.0` and `true` both compare equal to 1
    # in Python, so a version check written as equality alone accepts both. Neither
    # is this schema's version.
    if (
        type(record["schema_version"]) is not int
        or record["schema_version"] != CAPABILITY_SCHEMA_VERSION
    ):
        raise _invalid(
            component,
            f"`schema_version` is {record['schema_version']!r} and this loader reads "
            f"{CAPABILITY_SCHEMA_VERSION}",
            path=path,
        )
    # The file name is the identity. A record whose `component` disagrees would be
    # reachable under two ids, which is the two-answers problem the one-file-per-id
    # layout exists to prevent.
    if record["component"] != component:
        raise _invalid(
            component,
            f"the record declares component {record['component']!r} but is filed as "
            f"{component!r}; the file name is the identity",
            path=path,
        )
    for field in ("probe", "probe_tag", "evidence", "notes", "minimum_compute_precision"):
        if not isinstance(record[field], str):
            raise _invalid(component, f"`{field}` must be a string", path=path)

    namespaces = record["device_namespaces"]
    if not isinstance(namespaces, dict):
        raise _invalid(
            component, "`device_namespaces` must be an object keyed by device", path=path
        )
    device_namespaces: dict[DeviceKind, frozenset[ArrayNamespace]] = {}
    for device, value in namespaces.items():
        if not isinstance(device, str):  # pragma: no cover - JSON keys are strings
            raise _invalid(component, "`device_namespaces` keys must be strings", path=path)
        try:
            kind = DeviceKind(device)
        except ValueError as exc:
            raise _invalid(
                component,
                f"`device_namespaces` names device {device!r}, which is not a declared "
                f"DeviceKind ({sorted(member.value for member in DeviceKind)})",
                path=path,
            ) from exc
        device_namespaces[kind] = _decode_set(
            value,
            ArrayNamespace,
            component=component,
            field=f"device_namespaces[{device!r}]",
            path=path,
        )

    try:
        minimum = Precision(record["minimum_compute_precision"])
    except ValueError as exc:
        raise _invalid(
            component,
            f"`minimum_compute_precision` is {record['minimum_compute_precision']!r}, "
            f"which is not a declared Precision "
            f"({sorted(member.value for member in Precision)})",
            path=path,
        ) from exc

    return ComponentCapabilities(
        component=record["component"],
        devices=_decode_set(
            record["devices"], DeviceKind, component=component, field="devices", path=path
        ),
        precisions=_decode_set(
            record["precisions"], Precision, component=component, field="precisions", path=path
        ),
        accepted_input_dtypes=_decode_set(
            record["accepted_input_dtypes"], DType, component=component,
            field="accepted_input_dtypes", path=path,
        ),
        native_compute_dtypes=_decode_set(
            record["native_compute_dtypes"], DType, component=component,
            field="native_compute_dtypes", path=path,
        ),
        output_dtypes=_decode_set(
            record["output_dtypes"], DType, component=component,
            field="output_dtypes", path=path,
        ),
        lossy_input_dtypes=_decode_set(
            record["lossy_input_dtypes"], DType, component=component,
            field="lossy_input_dtypes", path=path,
        ),
        device_namespaces=device_namespaces,
        minimum_compute_precision=minimum,
        probe=record["probe"],
        probe_tag=record["probe_tag"],
        evidence=record["evidence"],
        notes=record["notes"],
    )


def capability_rows(*, directory: Path | None = None) -> list[dict[str, Any]]:
    """The whole capability table, one row per record on disk.

    The name-agnostic replacement for the deleted helper of the same name: that one
    iterated a module-level dict of two hard-coded components, this one iterates the
    pack. Still generated from the declarations rather than written beside them, so
    a documented claim cannot outlive the capability it describes.
    """
    root = CAPABILITY_DIRECTORY if directory is None else directory
    return [
        load_capabilities(component, directory=root).capability_row()
        for component in capability_record_ids(directory=root)
    ]
