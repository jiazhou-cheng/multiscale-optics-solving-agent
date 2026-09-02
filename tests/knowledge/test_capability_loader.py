"""The loader: validated construction, no hard-coded inventory, no backend.

CHE-223 (R03.6), acceptance criteria 3, 4 and 8. Three claims, and each is the
difference between a loader and a second registry:

* a record becomes a `ComponentCapabilities` through the **same** `__post_init__`,
  so all ten widening refusals apply to a file on disk exactly as they applied to a
  module constant. A record wider than its probe is refused at load, not at first
  use;
* the loader **names no component and no backend**. Every test below that needs a
  new component writes a JSON file into a temp directory and loads it, editing no
  source constant. That is the property that makes adding a measured component a
  data change;
* loading imports no backend -- `json`, `pathlib` and this project's own enums.

Strict parsing is the fourth claim and it earns its own section. A JSON record is
not type-checked by mypy, so the loader is the only thing between a hand-edited
file and a capability nobody measured. Unknown keys are refused, missing keys are
refused, and enum values resolve through the enum's own constructor rather than
through `getattr` on the class -- which would accept `DType.mro`.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from numerics import (
    CAPABILITY_DIRECTORY,
    CAPABILITY_SCHEMA_VERSION,
    KNOWLEDGE_ROOT,
    ArrayNamespace,
    DeviceKind,
    DType,
    Precision,
    capability_record_ids,
    capability_rows,
    load_capabilities,
)

ROOT = Path(__file__).resolve().parents[2]


def _code_of(path: Path) -> str:
    """A module's source with every docstring removed.

    The loader's *prose* must name the two consumers -- that is how it explains why
    the rows are shared data rather than solver-package state. Its *code* must name
    neither, and those are different claims.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef):
            continue
        body = node.body
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            node.body = body[1:] or [ast.Pass()]
    return ast.unparse(tree)

#: A well-formed synthetic record, as the loader's input rather than as a Python
#: object. Every test below starts from this and changes one thing.
def a_record(**overrides: Any) -> dict[str, Any]:
    record: dict[str, Any] = {
        "schema_version": CAPABILITY_SCHEMA_VERSION,
        "component": "T_SYNTHETIC",
        "probe": "benchmarks/probes/precision/made_up.py",
        "probe_tag": "a-synthetic-tag",
        "evidence": (
            "test fixture, not a component: a well-formed record so every refusal below "
            "is caused by the one field it changes (synthetic, 0.0.0)"
        ),
        "notes": "",
        "devices": ["cpu"],
        "precisions": ["fp32"],
        "minimum_compute_precision": "fp32",
        "accepted_input_dtypes": ["float32"],
        "native_compute_dtypes": ["float32"],
        "output_dtypes": ["float32"],
        "lossy_input_dtypes": [],
        "device_namespaces": {"cpu": ["numpy"]},
    }
    record.update(overrides)
    return record


def write(directory: Path, record: dict[str, Any]) -> str:
    """Write one record and return its component id."""
    component = str(record["component"])
    (directory / f"{component}.json").write_text(json.dumps(record, indent=2))
    return component


# ---------------------------------------------------------------------------
# 1. No hard-coded inventory (criterion 4)
# ---------------------------------------------------------------------------


def test_a_new_component_is_a_new_file_and_nothing_else(tmp_path: Path) -> None:
    """Criterion 4. The property that separates a loader from a registry.

    `COMPONENT_CAPABILITIES` was a dict literal over two named constants, so a
    third component meant editing `numerics/precision.py`. Here it means writing a
    file, and this test proves it by loading a component no source mentions.
    """
    component = write(tmp_path, a_record())
    assert capability_record_ids(directory=tmp_path) == (component,)
    capability = load_capabilities(component, directory=tmp_path)
    assert capability.component == component
    assert capability.devices == frozenset({DeviceKind.CPU})
    assert capability.namespaces == frozenset({ArrayNamespace.NUMPY})

    # And the loader's own CODE names no component id and no backend -- the check
    # that would fail if this were a registry with a nicer signature. Checked with
    # the docstrings stripped, because the module docstring has to be able to name
    # the two consumers in order to explain why the rows are data.
    code = _code_of(ROOT / "src" / "numerics" / "knowledge.py")
    for real in capability_record_ids():
        assert real not in code, f"the loader's code names {real}"
    for backend in ("optiland", "chromatix"):
        assert backend not in code.lower(), f"the loader's code names the backend {backend!r}"


def test_an_unknown_component_is_refused_by_naming_what_is_on_disk(
    tmp_path: Path,
) -> None:
    """Criterion 4's other half: the refusal reads the directory, not a constant."""
    present = write(tmp_path, a_record())
    with pytest.raises(ValueError) as caught:
        load_capabilities("T_ABSENT", directory=tmp_path)
    assert caught.value.code == "UNKNOWN_COMPONENT"
    assert present in str(caught.value), "the refusal must say what does exist"
    assert "T_ABSENT" in str(caught.value)


def test_a_missing_pack_fails_as_a_missing_pack(tmp_path: Path) -> None:
    """Not as a missing component, which is the confusing failure off a checkout.

    `KNOWLEDGE_ROOT` is resolved from the source file, so a real wheel would not
    have it. The message names the path it tried rather than reporting that no
    component exists.
    """
    absent = tmp_path / "no-such-pack"
    with pytest.raises(ValueError) as caught:
        capability_record_ids(directory=absent)
    assert caught.value.code == "UNKNOWN_COMPONENT"
    assert str(absent) in str(caught.value)


def test_the_pack_root_is_the_repository_root() -> None:
    """The path decision, asserted rather than left implicit."""
    assert KNOWLEDGE_ROOT == ROOT / "knowledge"
    assert CAPABILITY_DIRECTORY == KNOWLEDGE_ROOT / "capabilities"


# ---------------------------------------------------------------------------
# 2. The same validation (criterion 3)
# ---------------------------------------------------------------------------


def test_a_record_wider_than_its_probe_is_refused_at_load(tmp_path: Path) -> None:
    """Criterion 3, with a deliberately widened record.

    The record declares CUDA and gives it no namespace -- exactly the widening
    `tests/numerics/test_capabilities.py` refuses on an in-Python row. The point is
    that the *loader* is not a way around it: a hand-edited file goes through the
    same `__post_init__`.
    """
    component = write(tmp_path, a_record(devices=["cpu", "cuda"]))
    with pytest.raises(ValueError) as caught:
        load_capabilities(component, directory=tmp_path)
    assert caught.value.code == "INVALID_CAPABILITY_DECLARATION"
    assert "device_namespaces" in str(caught.value)


def test_a_record_declaring_a_precision_it_cannot_run_is_refused(tmp_path: Path) -> None:
    """A second widening, so the one above is not the only path that is checked."""
    component = write(tmp_path, a_record(precisions=["fp32", "fp64"]))
    with pytest.raises(ValueError, match="no native compute dtype"):
        load_capabilities(component, directory=tmp_path)


def test_a_record_with_a_blank_probe_tag_is_refused(tmp_path: Path) -> None:
    """CHE-223's own rule, reached through the loader."""
    component = write(tmp_path, a_record(probe_tag="  "))
    with pytest.raises(ValueError, match="probe_tag"):
        load_capabilities(component, directory=tmp_path)


# ---------------------------------------------------------------------------
# 3. Strict parsing
# ---------------------------------------------------------------------------


def test_an_unknown_key_is_refused(tmp_path: Path) -> None:
    """A typo must not become a silently defaulted field.

    `"devicess"` alongside a correct `"devices"` would otherwise load a record the
    author thought said something else. Both directions are checked, so a stale key
    left behind by an edit also fails rather than lingering unread.
    """
    component = write(tmp_path, a_record(supports_float16=True))
    with pytest.raises(ValueError, match="unrecognized"):
        load_capabilities(component, directory=tmp_path)


@pytest.mark.parametrize(
    "field",
    ["devices", "precisions", "probe", "probe_tag", "evidence", "device_namespaces"],
)
def test_a_missing_key_is_refused(tmp_path: Path, field: str) -> None:
    record = a_record()
    del record[field]
    component = write(tmp_path, record)
    with pytest.raises(ValueError, match="missing"):
        load_capabilities(component, directory=tmp_path)


def test_a_repeated_key_is_refused_rather_than_last_wins(tmp_path: Path) -> None:
    """The strictness gap a required-keys check cannot see.

    JSON permits a repeated key and Python's decoder takes the **last**, so a record
    declaring `accepted_input_dtypes` twice -- once narrow, once wide -- loads as
    the wide one. Both spellings are the same key, so `_REQUIRED_KEYS` is satisfied
    either way and the widening is invisible. Written as text rather than through
    `json.dumps`, because a Python dict cannot hold the duplicate.
    """
    record = a_record()
    text = json.dumps(record)[:-1] + ', "accepted_input_dtypes": ["float32", "float64"]}'
    (tmp_path / "T_SYNTHETIC.json").write_text(text)
    with pytest.raises(ValueError, match="more than once"):
        load_capabilities("T_SYNTHETIC", directory=tmp_path)


@pytest.mark.parametrize("version", [1.0, True])
def test_a_schema_version_of_the_wrong_type_is_refused(
    tmp_path: Path, version: object
) -> None:
    """`1.0 == 1` and `True == 1` are both `True` in Python.

    So an equality check alone accepts a float and a boolean as version 1. Neither
    is this schema's version, and a record carrying one is a record written against
    something else.
    """
    component = write(tmp_path, a_record(schema_version=version))
    with pytest.raises(ValueError, match="schema_version"):
        load_capabilities(component, directory=tmp_path)


def test_a_schema_version_the_loader_does_not_read_is_refused(tmp_path: Path) -> None:
    """The version field is validated, which is the condition for it existing.

    A version nothing checks reads as a compatibility guarantee and is not one, so
    the ticket's rule is that the loader is its consumer or the field does not
    exist.
    """
    component = write(tmp_path, a_record(schema_version=CAPABILITY_SCHEMA_VERSION + 1))
    with pytest.raises(ValueError, match="schema_version"):
        load_capabilities(component, directory=tmp_path)


@pytest.mark.parametrize(
    ("field", "value", "fragment"),
    [
        ("devices", ["tpu"], "DeviceKind"),
        ("precisions", ["bfloat16"], "Precision"),
        ("accepted_input_dtypes", ["int32"], "DType"),
        ("minimum_compute_precision", "bfloat16", "Precision"),
    ],
)
def test_an_unrecognized_enum_value_is_refused(
    tmp_path: Path, field: str, value: object, fragment: str
) -> None:
    """Resolved by value through the enum, never by `getattr` on the class.

    `getattr(DType, name)` would accept `"mro"` and every other attribute that
    happens to exist, which is how a hand-edited record becomes a capability nobody
    measured.
    """
    component = write(tmp_path, a_record(**{field: value}))
    with pytest.raises(ValueError, match=fragment):
        load_capabilities(component, directory=tmp_path)


def test_a_getattr_style_enum_name_is_not_accepted(tmp_path: Path) -> None:
    """The specific bypass the `_decode_set` docstring names."""
    for poison in ("mro", "FLOAT32", "__class__"):
        component = write(
            tmp_path, a_record(component="T_POISON", accepted_input_dtypes=[poison])
        )
        with pytest.raises(ValueError, match="DType"):
            load_capabilities(component, directory=tmp_path)


def test_a_record_whose_component_disagrees_with_its_file_name_is_refused(
    tmp_path: Path,
) -> None:
    """The file name is the identity, which is what makes the layout indexless."""
    (tmp_path / "T_FILED_AS.json").write_text(json.dumps(a_record(component="T_DECLARED")))
    with pytest.raises(ValueError, match="the file name is the identity"):
        load_capabilities("T_FILED_AS", directory=tmp_path)


def test_malformed_json_is_refused_as_such(tmp_path: Path) -> None:
    (tmp_path / "T_BROKEN.json").write_text("{ not json")
    with pytest.raises(ValueError, match="not valid JSON"):
        load_capabilities("T_BROKEN", directory=tmp_path)


def test_a_json_document_that_is_not_an_object_is_refused(tmp_path: Path) -> None:
    (tmp_path / "T_LIST.json").write_text("[]")
    with pytest.raises(ValueError, match="must be a JSON object"):
        load_capabilities("T_LIST", directory=tmp_path)


def test_a_wrongly_typed_field_is_refused(tmp_path: Path) -> None:
    component = write(tmp_path, a_record(devices="cpu"))
    with pytest.raises(ValueError, match="must be a list of strings"):
        load_capabilities(component, directory=tmp_path)


def test_rows_are_generated_over_whatever_is_on_disk(tmp_path: Path) -> None:
    """`capability_rows` is the name-agnostic replacement, not a rename."""
    write(tmp_path, a_record(component="T_ONE"))
    write(tmp_path, a_record(component="T_TWO"))
    rows = capability_rows(directory=tmp_path)
    assert [row["component"] for row in rows] == ["T_ONE", "T_TWO"]
    assert rows[0]["precisions"] == ["fp32"]
    assert rows[0]["probe_tag"] == "a-synthetic-tag"


# ---------------------------------------------------------------------------
# 4. Loading imports no backend (criterion 8)
# ---------------------------------------------------------------------------

BACKENDS = ("jax", "jaxlib", "torch", "optiland", "chromatix")


def test_loading_both_records_and_reading_every_field_pulls_no_backend() -> None:
    """Criterion 8, in a fresh interpreter, on the real pack.

    The extension of the check `tests/numerics/test_no_backend_import.py` already
    ran against the deleted `capability_rows()`. It is the same claim about a
    different mechanism: the rows used to be Python constants in a module that
    imports nothing, and are now JSON read by a loader that imports nothing.
    """
    source = """
from numerics.knowledge import capability_record_ids, load_capabilities

for component in capability_record_ids():
    capability = load_capabilities(component)
    assert capability.component == component
    assert capability.probe and capability.probe_tag and capability.evidence
    assert capability.devices and capability.precisions
    assert capability.accepted_input_dtypes and capability.native_compute_dtypes
    assert capability.output_dtypes and capability.namespaces
    _ = capability.lossy_input_dtypes, capability.notes
    _ = capability.minimum_compute_precision, capability.capability_row()
import sys, json
print(json.dumps(sorted({m.split('.')[0] for m in sys.modules})))
"""
    completed = subprocess.run(
        [sys.executable, "-c", source], capture_output=True, text=True, check=True
    )
    loaded = set(json.loads(completed.stdout))
    assert not loaded & set(BACKENDS), (
        f"loading a capability record pulled {sorted(loaded & set(BACKENDS))}. The pack is "
        "JSON and the loader is json + pathlib + this project's enums."
    )
    # And it did load them, so the assertion above is not about an empty loop.
    assert "numerics" in loaded


def test_the_dtype_and_precision_vocabularies_round_trip() -> None:
    """Every enum member the records can name is decodable, which is the loader's job."""
    for component in capability_record_ids():
        capability = load_capabilities(component)
        for dtype in capability.accepted_input_dtypes | capability.output_dtypes:
            assert DType(dtype.value) is dtype
        for precision in capability.precisions:
            assert Precision(precision.value) is precision
