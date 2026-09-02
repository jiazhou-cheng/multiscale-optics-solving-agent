"""The two measured capability records, and the loader that reads them.

CHE-223 (R03.6). This module is the new home of everything
`tests/numerics/test_capabilities.py` asserted **about the two measured records**:
the probe citations, the evidence prose, the `git cat-file` resolution at the
frozen tag, the derived namespace sets, the four pinned measured facts, the
two-rows-not-seven rule, and the compute-floor check across every declared
component.

What stayed in `tests/numerics/test_capabilities.py` is everything about the
`ComponentCapabilities` **contract** -- all ten widening refusals, on synthetic
rows -- because that is a rule about what may be declared rather than a fact about
either backend. The split is the point of the ticket: `numerics/` owns the
contract and knows no backend's name, and the measured rows are data.

Accounting for the move, by name, because splitting a test module is how coverage
gets quietly dropped:

| what | where now |
| -- | -- |
| probe path is a precision probe | here |
| evidence names a version and a tag | here |
| `git cat-file -e <tag>:<probe>` | here |
| namespaces derived, not stored | here |
| Optiland has no float16 path | here |
| Optiland reaches CUDA only through torch | here |
| Chromatix has no complex128 path | here |
| Chromatix's complex128 is lossy, not accepted | here |
| two rows, not seven | here |
| unknown component refused by name | here, and on a temp pack in the loader test |
| the table is generated from the declarations | here |
| no declared component computes below the floor | here |
| the ten `__post_init__` refusals | `tests/numerics/test_capabilities.py` |
| `compute_dtype` and the phase floor | `tests/numerics/test_capabilities.py` |
| the loader's own refusals | `tests/knowledge/test_capability_loader.py` |
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest

from numerics import (
    CAPABILITY_DIRECTORY,
    PHASE_ACCUMULATION_FLOOR,
    ArrayNamespace,
    ArrayState,
    ComponentCapabilities,
    DeviceKind,
    DevicePlacement,
    DType,
    Precision,
    capability_record_ids,
    capability_rows,
    load_capabilities,
)

ROOT = Path(__file__).resolve().parents[2]

#: Every component with a record on disk. Read from the pack rather than listed,
#: which is the property that makes adding a measured component a data change.
COMPONENTS = capability_record_ids()


def _row_of(component: str) -> ComponentCapabilities:
    return load_capabilities(component)


def test_the_pack_is_where_the_loader_says_it_is() -> None:
    """A missing pack has to fail as a missing pack, not as a missing component."""
    assert CAPABILITY_DIRECTORY == ROOT / "knowledge" / "capabilities"
    assert CAPABILITY_DIRECTORY.is_dir()
    assert COMPONENTS, "the pack found no records, so every test below is vacuous"


# --- every record cites the probe that measured it -------------------------


@pytest.mark.parametrize("component", COMPONENTS)
def test_every_record_cites_a_precision_probe(component: str) -> None:
    capability = _row_of(component)
    assert capability.probe.startswith("benchmarks/probes/precision/"), (
        f"{component} cites {capability.probe!r}, which is not a precision probe"
    )
    assert capability.probe.endswith(".py")


@pytest.mark.parametrize("component", COMPONENTS)
def test_every_record_states_a_measurement_and_where_it_ran(component: str) -> None:
    """Evidence must say what was observed, not that support exists.

    The weakest useful check that distinguishes a measurement from a claim: the
    sentence has to name the pinned version it was measured against, and the
    record has to name the frozen tag the probe is reproducible from.

    The tag moved out of the prose and into `probe_tag` -- CHE-223. It used to be
    interpolated into the evidence string from a `PROBE_TAG` module constant that
    `tests/numerics/test_capabilities.py` declared a *second* copy of, so the
    literal existed twice in the repository. It now exists once per record, in the
    field whose job it is, and `test_the_probe_tag_literal_lives_only_in_the_pack`
    holds it there.
    """
    capability = _row_of(component)
    assert len(capability.evidence) > 80, (
        f"{component} evidence is too short to be a measurement"
    )
    assert capability.probe_tag.strip(), f"{component} names no frozen tag"
    assert re.search(r"\d+\.\d+\.\d+", capability.evidence), (
        f"{component} evidence names no pinned version"
    )


def test_the_probe_tag_is_a_record_field_and_not_a_constant_anywhere() -> None:
    """AC 9's "exactly one place", read as "one place it can be *changed*".

    Before CHE-223 the tag was `PROBE_TAG` in `src/numerics/precision.py` **and**
    a second literal declared in `tests/numerics/test_capabilities.py`, so the
    capability check ran against the test's own idea of the tag rather than
    against the measurement's. Now it is a record field.

Two checks. Nothing anywhere may define a `PROBE_TAG`-shaped constant again --
    that is the duplicate itself. And no test fixture may declare the pack's real
    tag on a synthetic row, which is the same confusion from the other side.

    Deliberately *not* "the string appears nowhere else": the tag is a legitimate
    narrative citation in about forty evidence strings and frozen-record comments
    across the tree (`couplers/`, `tests/physics/`, `numerics/arrays.py`). Those
    cite a frozen revision; they do not decide what this loader resolves against,
    and a rule banning the string everywhere would be a rule about prose.
    """
    tags = {_row_of(component).probe_tag for component in COMPONENTS}
    assert len(tags) == 1, f"the records disagree about the tag: {sorted(tags)}"
    tag = tags.pop()

    defined = [
        f"{path.relative_to(ROOT)}:{number}"
        for tree in (ROOT / "src", ROOT / "tests")
        for path in sorted(tree.rglob("*.py"))
        if "__pycache__" not in str(path)
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if re.match(r"\s*PROBE_TAG\s*[:=]", line)
    ]
    assert defined == [], (
        "PROBE_TAG is a constant again, so there is a second place the frozen tag can "
        "be changed:\n  " + "\n  ".join(defined)
    )

    # A *synthetic* row may not cite the pack's real tag. That is the confusion the
    # duplicate created in the other direction: a test fixture whose evidence quoted
    # `pre-rewrite-2026-08-30` read as though it had been measured there. Checkable
    # exactly, unlike "the string appears somewhere" -- the tag is a legitimate
    # narrative citation in about forty evidence strings and frozen-record comments
    # across the tree, and a rule banning it everywhere would be a rule about prose.
    fixtures = [
        f"{path.relative_to(ROOT)}:{number}"
        for path in sorted((ROOT / "tests").rglob("*.py"))
        if "__pycache__" not in str(path)
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if "probe_tag" in line and tag in line
    ]
    assert fixtures == [], (
        f"a test fixture declares probe_tag={tag!r}, the pack's own tag, so a synthetic "
        "row reads as one that was measured there:\n  " + "\n  ".join(fixtures)
    )


def _checkout_has_tags() -> bool:
    """Whether this checkout has any git tags at all.

    The skip condition, and it is deliberately **not** "the record's own tag
    resolves". Keying the skip on the record's tag would let the record decide
    whether it gets checked: a re-measurement that mistyped `probe_tag` would turn
    the citation gate into a silent pass, which is the same defect as a probe path
    that no longer resolves. A tagless checkout (shallow clone, CI fetch without
    `--tags`) is a property of the environment and is the only honest reason to
    skip.
    """
    completed = subprocess.run(
        ["git", "-C", str(ROOT), "tag", "-l"], capture_output=True, text=True
    )
    return completed.returncode == 0 and bool(completed.stdout.strip())


@pytest.mark.parametrize("component", COMPONENTS)
def test_every_cited_probe_resolves_at_its_frozen_tag(component: str) -> None:
    """The citation is a reference, not a sentence that looks like one.

    A probe path that no longer resolves is the same defect as a capability with
    no probe: nothing can be re-run to confirm or falsify the row. Checked with
    `git cat-file` because the file is deliberately not in the working tree, and
    driven by the record's own `probe_tag` rather than by a constant here.

    **An unresolvable tag fails rather than skips.** See `_checkout_has_tags`: the
    skip is keyed on the environment having tags at all, not on the record's tag
    resolving, so a record cannot exempt itself from the gate by naming a revision
    that does not exist.
    """
    if not _checkout_has_tags():
        pytest.skip("this checkout has no git tags, so no citation can be resolved")
    capability = _row_of(component)
    resolved = subprocess.run(
        [
            "git", "-C", str(ROOT), "cat-file", "-e",
            f"{capability.probe_tag}:{capability.probe}",
        ],
        capture_output=True,
    )
    assert resolved.returncode == 0, (
        f"{component} cites {capability.probe} at {capability.probe_tag}, which does not "
        "resolve. Either the tag is wrong or the probe moved: re-run the probe and cite "
        "where it lives now, or drop the record."
    )


# --- the measured facts, unchanged by the move -----------------------------


def test_the_namespace_set_is_derived_and_cannot_disagree_with_the_devices() -> None:
    """One place for the namespace set, so there is one place to widen it."""
    assert _row_of("M_RAY_OPTILAND").namespaces == frozenset(
        {ArrayNamespace.NUMPY, ArrayNamespace.TORCH}
    )
    assert _row_of("M_WAVE_CHROMATIX").namespaces == frozenset({ArrayNamespace.JAX})


def test_optiland_has_no_float16_path() -> None:
    """`set_precision` is `Literal['float32','float64']` and raises otherwise."""
    capability = _row_of("M_RAY_OPTILAND")
    assert capability.precisions == frozenset({Precision.FP32, Precision.FP64})
    assert DType.FLOAT16 not in capability.accepted_input_dtypes
    assert DType.FLOAT16 not in capability.native_compute_dtypes


def test_optiland_reaches_cuda_only_through_torch() -> None:
    """`set_device` raises `BackendCapabilityError` on the numpy backend."""
    capability = _row_of("M_RAY_OPTILAND")
    assert capability.namespaces_for(DeviceKind.CUDA) == frozenset({ArrayNamespace.TORCH})
    assert ArrayNamespace.NUMPY in capability.namespaces_for(DeviceKind.CPU)


def test_chromatix_has_no_complex128_path_at_any_device() -> None:
    """`ScalarField.__init__` is `jnp.asarray(u, dtype=jnp.complex64)`, unconditionally."""
    capability = _row_of("M_WAVE_CHROMATIX")
    assert capability.precisions == frozenset({Precision.FP32})
    for device in capability.devices:
        assert capability.namespaces_for(device) == frozenset({ArrayNamespace.JAX})
    assert capability.output_dtypes == frozenset({DType.COMPLEX64})
    assert DType.COMPLEX128 not in capability.accepted_input_dtypes


def test_chromatix_complex128_is_declared_lossy_rather_than_accepted() -> None:
    """The distinction that makes the truncation happen where something records it."""
    capability = _row_of("M_WAVE_CHROMATIX")
    assert capability.lossy_input_dtypes == frozenset({DType.COMPLEX128})
    state = ArrayState(DType.COMPLEX128, DevicePlacement(DeviceKind.CPU), ArrayNamespace.JAX)
    assert not capability.accepts(state)


def test_the_pack_declares_no_record_for_code_that_does_not_exist() -> None:
    """Two records; a coupler record here would be a claim about nothing.

    The reference implementation's table had seven rows. Five described coupler
    and operator implementations, whose capability is set by what their shared
    implementation is written against -- so they belong to the tickets that
    measure them, with their own evidence. Unchanged by CHE-223: the rule is about
    what may be declared, and moving the rows to data does not make an unmeasured
    one admissible.
    """
    assert set(COMPONENTS) == {"M_RAY_OPTILAND", "M_WAVE_CHROMATIX"}


def test_no_declared_component_computes_below_the_phase_floor() -> None:
    for component in COMPONENTS:
        capability = _row_of(component)
        assert capability.minimum_compute_precision.bits >= PHASE_ACCUMULATION_FLOOR.bits, (
            f"{component} declares a compute floor below float32"
        )
        for dtype in capability.accepted_input_dtypes:
            assert capability.compute_dtype_for(dtype).component_bits >= 32


def test_an_undeclared_component_is_refused_by_naming_the_real_pack() -> None:
    """The refusal names what exists **in the shipped pack**, not just on some path.

    `tests/knowledge/test_capability_loader.py` asserts the same shape against a
    temp directory, which is what proves the refusal reads the directory rather
    than a constant. This one is the assertion the pre-CHE-223
    `test_an_undeclared_component_is_refused_by_name` made -- that a real caller
    asking for a component nobody measured is told `M_RAY_OPTILAND` exists -- and
    it would otherwise have been lost in the split.
    """
    with pytest.raises(ValueError) as caught:
        load_capabilities("C_RAY_TO_WAVE")
    assert caught.value.code == "UNKNOWN_COMPONENT"
    message = str(caught.value)
    assert "M_RAY_OPTILAND" in message
    assert "M_WAVE_CHROMATIX" in message


def test_the_matrix_is_generated_from_the_records() -> None:
    rows = capability_rows()
    assert [row["component"] for row in rows] == sorted(COMPONENTS)
    for row in rows:
        assert row["probe"].startswith("benchmarks/probes/precision/")
        assert row["evidence"]
        assert row["probe_tag"]


# --- the records are the only copy -----------------------------------------


def test_no_capability_data_is_duplicated_under_src() -> None:
    """AC 1 and 2: the foundational layer no longer names a backend's capability.

    The constants that used to live in `numerics/precision.py` are the exact thing
    this ticket removed, so their names are the check. `numerics` may not export
    them either -- a re-export would become the de facto table again, which is
    AC 13.
    """
    import numerics

    removed = (
        "OPTILAND_CAPABILITIES",
        "CHROMATIX_CAPABILITIES",
        "COMPONENT_CAPABILITIES",
        "capabilities_for",
    )
    for name in removed:
        assert not hasattr(numerics, name), f"numerics still exports {name}"
        assert name not in numerics.__all__

    offenders = [
        f"{path.relative_to(ROOT)}: {name}"
        for path in sorted((ROOT / "src" / "numerics").rglob("*.py"))
        if "__pycache__" not in str(path)
        for name in removed
        # The two modules that explain the move may name what they removed; a
        # *definition* is what is forbidden, so the check is on assignment.
        if re.search(rf"^{name}\s*[:=]", path.read_text(encoding="utf-8"), re.MULTILINE)
        or re.search(rf"^def {name}\b", path.read_text(encoding="utf-8"), re.MULTILINE)
    ]
    assert offenders == [], (
        "a concrete capability constant is defined in numerics again:\n  "
        + "\n  ".join(offenders)
    )


def test_the_record_on_disk_carries_every_field_the_contract_has() -> None:
    """No field is silently defaulted: the file and the declaration agree everywhere.

    **This is not the transcription check and cannot be**, which is worth saying
    because it looks like one. It reads the same file through the same parser, so a
    member dropped while transcribing is absent from both sides and this passes.
    What it does catch is a field the loader defaults or drops rather than reading
    -- the class of bug a required-keys check alone would not see, since a key can
    be present and unread.

    The real evidence that nothing was dropped is a parity comparison against the
    pre-move constants in git history, run once on CHE-223 and recorded there:
    all eleven compared fields identical on both records, with the evidence prose
    differing only by the tag prefix that moved into `probe_tag`. That is a one-off
    probe rather than a committed test, because a committed one would have to
    execute a deleted module out of `git show`.
    """
    for component in COMPONENTS:
        record = json.loads((CAPABILITY_DIRECTORY / f"{component}.json").read_text())
        capability = _row_of(component)
        assert record["component"] == capability.component
        assert record["probe"] == capability.probe
        assert record["probe_tag"] == capability.probe_tag
        assert record["evidence"] == capability.evidence
        assert record["notes"] == capability.notes
        assert set(record["devices"]) == {d.value for d in capability.devices}
        assert set(record["precisions"]) == {p.value for p in capability.precisions}
        assert set(record["accepted_input_dtypes"]) == {
            d.value for d in capability.accepted_input_dtypes
        }
        assert set(record["native_compute_dtypes"]) == {
            d.value for d in capability.native_compute_dtypes
        }
        assert set(record["output_dtypes"]) == {d.value for d in capability.output_dtypes}
        assert set(record["lossy_input_dtypes"]) == {
            d.value for d in capability.lossy_input_dtypes
        }
        assert record["minimum_compute_precision"] == (
            capability.minimum_compute_precision.value
        )
        assert {
            key: set(value) for key, value in record["device_namespaces"].items()
        } == {
            device.value: {n.value for n in namespaces}
            for device, namespaces in capability.device_namespaces.items()
        }
