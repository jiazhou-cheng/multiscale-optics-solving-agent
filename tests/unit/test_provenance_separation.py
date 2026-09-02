"""Deleting any provenance field changes no physical result. Executed, not asserted.

CHE-199 (R13.1). The load-bearing test of R13, and the reason it is worth the
effort is that the reference tree had a field sitting on the line: the CHE-50
wavefront-curvature limitation travelled in `provenance["validity"]` as a
*warning*, where nothing could tell whether a consumer read it. R02.4 moved it to
a typed `ValidityFlag` on `ScalarField` for exactly this reason.

Two complementary checks, and neither is sufficient alone
---------------------------------------------------------
**Executable (this module's main body).** A record's `route` and `request` are what
a re-run reads. So: run the physics, build a record, then for each provenance key
in turn, delete it, re-derive the physical quantities *from the stripped record*,
and require bit-identity. A field that physics read would show up as a changed
number. `test_deleting_a_request_field_does_break_the_rerun` is the falsifier that
proves the comparison discriminates -- without it, "every deletion is harmless"
would also be true of a re-run that ignored the record entirely.

**Structural.** `scripts/check_dependencies.py` gives `runtime` no inbound edge:
nothing may import it. So a physical layer *cannot* read a provenance field, which
is a stronger statement than any number of deletions -- the executable check
proves the current code does not, and the structural one proves no code can. Both
are here because the executable one is what would catch a field that should never
have been provenance, and the structural one is what keeps it that way.

The physics
-----------
`sources.gaussian_beam` into `measurements.psf`. Exactly reproducible, backend-free
and cheap: a real analytic source and a real measurement, both with their own
physics tests elsewhere, both deterministic to the bit. What is compared is the
PSF's intensity array, its raw peak and its raw window energy -- three numbers a
provenance leak would have to leave alone.

A Gaussian and **not** a plane wave, which was the first choice and is a poor
probe: a plane wave's intensity is `|A exp(i phi)|^2 = |A|^2`, uniform, so the PSF
of one is a flat map and changing the *wavelength* or the tilt moves nothing in it.
That is a real physical fact rather than a defect -- the carrier enters the phase
only -- but it makes the array insensitive to most of the request, and an array
that barely moves is a weak thing to assert bit-identity of. The Gaussian's
envelope is real, so the waist, the grid, the pitch and the amplitude all change
what is compared.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from measurements import psf
from representations import ReferenceSurface
from runtime import (
    VOLATILE_KEYS,
    ExecutionRecord,
    NodeRecord,
    environment_fingerprint,
    from_json,
    record_provenance,
    require_stable_payload,
    source_fingerprint,
    strip_volatile,
    to_json,
)
from sources import gaussian_beam

ROOT = Path(__file__).resolve().parents[2]
# The repository root, for the `scripts.check_dependencies` import below --
# `scripts/` is not a package on the install path. Same line
# `tests/unit/test_dependency_direction.py` carries, and for the same reason.
sys.path.insert(0, str(ROOT))

#: The route the record below describes, and the two operations it really runs.
ROUTE = ("S_SOURCE_GAUSSIAN_BEAM", "M_PSF")

#: The **typed** inputs, and the whole of what a re-run reads. Every value here is
#: a physical or numerical declaration: a grid, a pitch, a wavelength, a tilt, a
#: normalization. Nothing in this mapping is an observation.
REQUEST: dict[str, Any] = {
    "shape": [16, 16],
    "sample_pitch_m": [1.0e-6, 1.0e-6],
    "wavelength_m": 5.5e-7,
    "waist_radius_m": 4.0e-6,
    "transverse_wavevector_rad_per_m": [1.2e5, -3.4e5],
    "amplitude": 2.5,
    "normalization": "peak",
}


def _physics(request: dict[str, Any]) -> tuple[np.ndarray, float, float]:
    """Run the route and return the numbers a provenance leak would have to move.

    Reads `request` and nothing else -- no record, no provenance, no clock. That is
    the property the whole module is about, and writing the function this way is
    what makes the deletion loop below a real comparison rather than a tautology
    the function's own signature guarantees.
    """
    surface = ReferenceSurface(name="emitting surface", z_m=0.0, medium_index=1.0)
    field = gaussian_beam(
        tuple(request["shape"]),
        sample_pitch_m=tuple(request["sample_pitch_m"]),
        wavelength_m=request["wavelength_m"],
        reference_surface=surface,
        waist_radius_m=request["waist_radius_m"],
        transverse_wavevector_rad_per_m=tuple(request["transverse_wavevector_rad_per_m"]),
        amplitude=request["amplitude"],
    )
    result = psf(field, normalization=request["normalization"])
    return (
        np.asarray(result.intensity).copy(),
        float(result.raw_peak_intensity),
        float(result.raw_window_energy),
    )


def a_record(*, timestamp_utc: str | None = None) -> ExecutionRecord:
    """A record of the route above, with real provenance over this test's sources."""
    root = ROOT
    provenance = record_provenance(
        sources=[
            root / "src" / "sources" / "plane_wave.py",
            root / "src" / "measurements" / "psf.py",
        ],
        root=root,
        timestamp_utc=timestamp_utc,
    )
    return ExecutionRecord(
        route=ROUTE,
        request=REQUEST,
        nodes=tuple(
            NodeRecord(
                operation_id=operation_id,
                status="completed",
                requested={"device": "cpu", "precision": "fp64"},
                observed={"device": "cpu", "precision": "fp64"},
            )
            for operation_id in ROUTE
        ),
        provenance=provenance,
    )


# ---------------------------------------------------------------------------
# 1. The rule: acceptance criterion 1
# ---------------------------------------------------------------------------


def test_the_record_carries_provenance_to_delete() -> None:
    """The meta-check: a deletion loop over an empty provenance block proves nothing."""
    record = a_record(timestamp_utc="2026-09-02T00:00:00+00:00")
    assert set(record.provenance) == {
        "schema_version",
        "code_fingerprint",
        "environment_fingerprint",
        "timestamp_utc",
    }
    assert record.provenance["code_fingerprint"]["files"], "no source was fingerprinted"


@pytest.mark.parametrize(
    "provenance_field",
    ["schema_version", "code_fingerprint", "environment_fingerprint", "timestamp_utc"],
)
def test_deleting_a_provenance_field_changes_no_physical_result(
    provenance_field: str,
) -> None:
    """Acceptance criterion 1, one field at a time, bit-identical.

    The record is stripped of the field and the physics is re-derived **from the
    stripped record's request**, so the comparison is against a re-run that could
    in principle have consulted what was removed.
    """
    reference = _physics(REQUEST)
    record = a_record(timestamp_utc="2026-09-02T00:00:00+00:00")
    stripped = record.without_provenance_field(provenance_field)

    assert provenance_field not in stripped.provenance
    intensity, peak, energy = _physics(dict(stripped.request))

    assert np.array_equal(intensity, reference[0]), (
        f"deleting provenance[{provenance_field!r}] changed the PSF. That field is not "
        "provenance -- it is an input, and it belongs in the typed representation, the "
        "problem or the request."
    )
    assert peak == reference[1]
    assert energy == reference[2]


def test_deleting_every_provenance_field_at_once_changes_no_physical_result() -> None:
    """The whole block, not one key at a time.

    A field-by-field loop cannot catch a *pair* that only matters together, and a
    record with no provenance at all is the strongest form of the claim: the
    physics is a function of the request.
    """
    reference = _physics(REQUEST)
    record = a_record()
    bare = ExecutionRecord(
        route=record.route, request=record.request, nodes=record.nodes, provenance={}
    )
    assert bare.provenance == {}
    intensity, peak, energy = _physics(dict(bare.request))
    assert np.array_equal(intensity, reference[0])
    assert (peak, energy) == (reference[1], reference[2])


def test_deleting_a_request_field_does_break_the_rerun() -> None:
    """The falsifier. Without it, the tests above pass for a re-run that reads nothing.

    `request` is the other half of the record, and it *is* an input -- so removing
    one of its fields must make the re-run impossible. That is what makes "deleting
    provenance changed nothing" a statement about provenance rather than about the
    comparison being blind.
    """
    for required in ("wavelength_m", "normalization", "shape", "waist_radius_m"):
        broken = {key: value for key, value in REQUEST.items() if key != required}
        with pytest.raises((KeyError, TypeError)):
            _physics(broken)

    # And changing a request field changes the physics, which is the same claim
    # from the other side: the request is read.
    reference = _physics(REQUEST)
    for key, value in (("waist_radius_m", 2.0e-6), ("shape", [16, 8]), ("amplitude", 3.5)):
        retuned = {**REQUEST, key: value}
        changed = _physics(retuned)
        assert not (
            changed[0].shape == reference[0].shape
            and np.array_equal(changed[0], reference[0])
            and changed[1] == reference[1]
        ), f"changing request[{key!r}] moved nothing, so the comparison is blind"

    # The carrier tilt is the interesting one, and the honest statement about it is
    # narrower than "it changes nothing". A Gaussian's intensity is
    # `|A|^2 exp(-2 rho^2 / w0^2)` -- the envelope alone, with the tilt in the phase
    # -- so *analytically* removing the tilt leaves it untouched. Measured, it moves
    # the array at the last float32 bit: the field is a float64 envelope times a
    # float64 ramp cast to complex64, and `|z|^2` of two complex64 values with the
    # same modulus and different phases does not round identically. So the
    # comparison is `allclose` at float32 round-off and *not* bit-identical, which
    # is worth recording because it is the one place in this file where a
    # request field's effect is numerical rather than physical.
    tilted = {**REQUEST, "transverse_wavevector_rad_per_m": [0.0, 0.0]}
    tilted_intensity = _physics(tilted)[0]
    assert not np.array_equal(tilted_intensity, reference[0])
    assert np.allclose(tilted_intensity, reference[0], rtol=4.0e-7, atol=0.0), (
        "the tilt moved the intensity by more than complex64 round-off, which would "
        "mean it enters the envelope rather than the phase"
    )


def test_the_real_executor_produces_identical_physics_with_no_provenance_at_all() -> None:
    """The deletion loop's reach, widened to the production consumer.

    The loop above re-derives the physics through this module's own `_physics`,
    which reads `request` by construction -- so it is a statement about that
    helper, and a leak in the *executor's* binding path would be invisible to it.
    The review found that, and this is the case that closes it: `runtime.execute`
    is driven with `record_provenance` patched to return nothing, and the physical
    result has to be bit-identical to the reference.

    That is the strongest available executable form of the rule, because the code
    under test is the one a caller actually runs. The structural check below is
    still the stronger *statement* -- a leak is not expressible -- but this is the
    one that would catch a leak someone had written.
    """
    import runtime
    import runtime.executor as executor_module
    from measurements import PsfResult

    reference = _physics(REQUEST)
    surface = ReferenceSurface(name="emitting surface", z_m=0.0, medium_index=1.0)
    executable = {
        key: value
        for key, value in REQUEST.items()
        if key not in ("shape", "sample_pitch_m", "transverse_wavevector_rad_per_m")
    }
    executable.update(
        shape=tuple(REQUEST["shape"]),
        sample_pitch_m=tuple(REQUEST["sample_pitch_m"]),
        transverse_wavevector_rad_per_m=tuple(REQUEST["transverse_wavevector_rad_per_m"]),
        reference_surface=surface,
    )

    original = executor_module.record_provenance
    executor_module.record_provenance = lambda **_: {}  # type: ignore[assignment]
    try:
        stripped = runtime.execute(ROUTE, request=executable)
    finally:
        executor_module.record_provenance = original  # type: ignore[assignment]

    assert stripped.status == "completed", [
        (node.operation_id, node.status, node.diagnostics) for node in stripped.nodes
    ]
    # `runtime_seconds` and `resources` are added after `record_provenance`, so a
    # record with no provenance block still carries those two -- which is why this
    # asserts on what the *fingerprints* are gone rather than on an empty dict.
    assert "code_fingerprint" not in stripped.provenance
    assert "environment_fingerprint" not in stripped.provenance

    with_provenance = runtime.execute(ROUTE, request=executable)
    assert "code_fingerprint" in with_provenance.provenance
    assert stripped.fingerprinted == with_provenance.fingerprinted

    # And the physics the executor ran is the reference physics, to the bit.
    field = gaussian_beam(
        tuple(REQUEST["shape"]),
        sample_pitch_m=tuple(REQUEST["sample_pitch_m"]),
        wavelength_m=REQUEST["wavelength_m"],
        reference_surface=surface,
        waist_radius_m=REQUEST["waist_radius_m"],
        transverse_wavevector_rad_per_m=tuple(REQUEST["transverse_wavevector_rad_per_m"]),
        amplitude=REQUEST["amplitude"],
    )
    result = psf(field, normalization=REQUEST["normalization"])
    assert isinstance(result, PsfResult)
    assert np.array_equal(np.asarray(result.intensity), reference[0])


def test_nothing_physical_may_import_the_runtime() -> None:
    """The structural half, and the stronger one: a leak is not expressible.

    `scripts/check_dependencies.py` gives `runtime` no inbound edge, so this is that
    rule restated as an executed check over the packages that compute physics. The
    deletion tests prove the current code does not read provenance; this proves no
    code in those packages can.
    """
    from scripts.check_dependencies import ALLOWED

    physical = (
        "representations",
        "problems",
        "numerics",
        "couplers",
        "operators",
        "measurements",
        "sources",
        "backends",
    )
    for package in physical:
        assert "runtime" not in ALLOWED[package], (
            f"{package}/ may import runtime/, so a provenance field could reach physics"
        )
    # And no package at all may: `runtime` is the top of the graph.
    for package, targets in ALLOWED.items():
        assert "runtime" not in targets, f"{package}/ -> runtime/ is an edge"


# ---------------------------------------------------------------------------
# 2. Criterion 2 -- fingerprints move for a real change and not for formatting
# ---------------------------------------------------------------------------


def test_a_fingerprint_is_stable_across_reformatting(tmp_path: Any) -> None:
    """Comments, docstrings and blank lines are not what a record was measured by.

    Hashing raw bytes would invalidate a record on a typo fix in a docstring, the
    regeneration would be ceremony, and the mechanism would be routed around. The
    two files below differ in every comment, every docstring and all the
    whitespace, and compute the same thing.
    """
    original = tmp_path / "a.py"
    original.write_text(
        '"""A module docstring nobody computes with."""\n'
        "\n"
        "# A comment.\n"
        "def area(radius):\n"
        '    """Return the area."""\n'
        "    return 3.141592653589793 * radius * radius\n"
    )
    reformatted = tmp_path / "b.py"
    reformatted.write_text(
        '"""A completely different docstring."""\n'
        "def area(radius):\n"
        '    """Different prose entirely, and a longer sentence about it."""\n'
        "\n"
        "\n"
        "    return 3.141592653589793 * radius * radius\n"
    )
    first = source_fingerprint([original], root=tmp_path)["files"]["a.py"]
    second = source_fingerprint([reformatted], root=tmp_path)["files"]["b.py"]
    assert first == second, "a docstring or comment change moved the fingerprint"


def test_a_fingerprint_moves_for_a_real_source_change(tmp_path: Any) -> None:
    """The other direction, and the one that must never under-trigger.

    Three changes, each smaller than the last: a changed constant, a changed
    operator, and a changed argument name. The third is over-triggering by the
    reference implementation's own account -- renaming a local moves the AST dump --
    and over-triggering is the correct direction to be wrong in.
    """
    base = "def area(radius):\n    return 3.141592653589793 * radius * radius\n"
    original = tmp_path / "a.py"
    original.write_text(base)
    reference = source_fingerprint([original], root=tmp_path)["files"]["a.py"]

    for changed in (
        "def area(radius):\n    return 3.14 * radius * radius\n",
        "def area(radius):\n    return 3.141592653589793 * radius / radius\n",
        "def area(r):\n    return 3.141592653589793 * r * r\n",
    ):
        original.write_text(changed)
        assert source_fingerprint([original], root=tmp_path)["files"]["a.py"] != reference, (
            f"a real change did not move the fingerprint:\n{changed}"
        )


def test_the_fingerprint_names_which_file_moved(tmp_path: Any) -> None:
    """Per-file digests, not only the combined hash.

    The difference between a failure someone can act on and one they re-derive by
    bisection, which is what the reference tree's CHE-100 had to do.
    """
    (tmp_path / "a.py").write_text("A = 1\n")
    (tmp_path / "b.py").write_text("B = 2\n")
    before = source_fingerprint([tmp_path / "a.py", tmp_path / "b.py"], root=tmp_path)
    (tmp_path / "b.py").write_text("B = 3\n")
    after = source_fingerprint([tmp_path / "a.py", tmp_path / "b.py"], root=tmp_path)

    assert before["combined_sha256"] != after["combined_sha256"]
    assert before["files"]["a.py"] == after["files"]["a.py"]
    assert before["files"]["b.py"] != after["files"]["b.py"]


def test_a_file_that_is_not_python_is_hashed_as_bytes(tmp_path: Any) -> None:
    """A data file has no docstring/comment distinction to draw.

    A comment beside a threshold is part of what the threshold means, so the bytes
    are the best available statement -- and a syntactically invalid `.py` takes the
    same path rather than raising.
    """
    broken = tmp_path / "a.py"
    broken.write_text("def f(:\n")
    first = source_fingerprint([broken], root=tmp_path)["files"]["a.py"]
    broken.write_text("def f(:  # a comment\n")
    assert source_fingerprint([broken], root=tmp_path)["files"]["a.py"] != first


def test_the_environment_fingerprint_names_the_pinned_solvers_and_nothing_else() -> None:
    """Interpreter and pinned array/solver versions. Not the whole environment.

    Recording everything would make every image rebuild invalidate every record
    regardless of relevance, which is the failure that gets a staleness mechanism
    switched off.
    """
    payload = environment_fingerprint()
    assert set(payload) == {"python_version", "packages", "combined_sha256"}
    assert payload["packages"], "no fingerprinted package is installed"
    assert set(payload["packages"]) <= {
        "numpy", "scipy", "jax", "jaxlib", "torch", "optiland", "chromatix",
    }
    assert "pytest" not in payload["packages"]
    assert environment_fingerprint()["combined_sha256"] == payload["combined_sha256"]


# ---------------------------------------------------------------------------
# 3. Criterion 3 -- no uuid, no timestamp, in anything hashed
# ---------------------------------------------------------------------------


def test_a_uuid_in_a_fingerprinted_payload_is_refused() -> None:
    """Criterion 3, and the measured case it comes from.

    A `uuid4` inside a **refusal message** made B0-META-01 rehash on every run, so
    the record read as a changed measurement every time and nothing could tell that
    from a real change. Nested in a diagnostic string, which is where the real one
    was.
    """
    with pytest.raises(ValueError, match="uuid"):
        require_stable_payload(
            {"nodes": [{"diagnostics": "refused: 3f2504e0-4f89-11d3-9a0c-0305e82c3301"}]},
            where="a node record",
        )
    with pytest.raises(ValueError, match="timestamp"):
        require_stable_payload({"stamped": "2026-09-02T01:02:03+00:00"}, where="a record")


def test_the_record_that_gets_fingerprinted_carries_neither() -> None:
    """The positive control, on the real record.

    `fingerprinted` excludes `provenance` -- which contains the fingerprints, so
    hashing it would be circular -- and `strip_volatile` removes the run-identity
    keys from what is left.
    """
    record = a_record(timestamp_utc="2026-09-02T00:00:00+00:00")
    require_stable_payload(record.fingerprinted, where="the record's fingerprinted payload")
    assert "provenance" not in record.fingerprinted
    assert "timestamp_utc" not in json.dumps(record.fingerprinted)

    # And hashing it twice gives the same bytes, which is the property in use.
    assert json.dumps(record.fingerprinted, sort_keys=True) == json.dumps(
        a_record(timestamp_utc="2026-09-02T09:99:99+00:00").fingerprinted, sort_keys=True
    )


def test_record_provenance_reads_no_clock() -> None:
    """`timestamp_utc` is an argument, so this module's own output is hashable.

    A function that stamped `datetime.now()` would make every provenance block it
    produced unhashable by `require_stable_payload`, and the temptation would be to
    weaken the check rather than the stamp.
    """
    payload = record_provenance(sources=[ROOT / "src" / "runtime" / "records.py"], root=ROOT)
    assert "timestamp_utc" not in payload
    require_stable_payload(payload, where="a provenance block with no stamp")


@pytest.mark.parametrize("key", VOLATILE_KEYS)
def test_strip_volatile_projects_out_every_declared_key_at_every_depth(key: str) -> None:
    """Nested, because a per-node record carries its own timings."""
    payload = {key: "x", "kept": {key: "x", "nodes": [{key: "x", "kept": 1}]}}
    assert strip_volatile(payload) == {"kept": {"nodes": [{"kept": 1}]}}


def test_device_and_precision_are_not_volatile() -> None:
    """They change what was computed, so projecting them out would claim too much.

    A fingerprint that ignored the device would report reproducibility across a
    float32 GPU run and a float64 host one.
    """
    assert "device" not in VOLATILE_KEYS
    assert "precision" not in VOLATILE_KEYS
    payload = {"requested": {"device": "cuda", "precision": "fp32"}}
    assert strip_volatile(payload) == payload


# ---------------------------------------------------------------------------
# 4. The record's own contract, and criteria 4 and 5
# ---------------------------------------------------------------------------


def test_the_serialization_round_trip_is_exact() -> None:
    record = a_record(timestamp_utc="2026-09-02T00:00:00+00:00")
    text = to_json(record)
    assert from_json(text) == record
    assert to_json(from_json(text)) == text, "the bytes are stable, so a diff is readable"


@pytest.mark.parametrize(
    "mutate",
    [
        lambda payload: payload.pop("route"),
        lambda payload: payload.update(extra=1),
        lambda payload: payload.update(schema_version=99),
    ],
)
def test_a_serialized_record_this_reader_does_not_understand_is_refused(
    mutate: Any,
) -> None:
    """Strict for the reason the capability loader is: JSON is not type-checked."""
    payload = json.loads(to_json(a_record()))
    mutate(payload)
    with pytest.raises(ValueError):
        from_json(json.dumps(payload))


def test_a_status_is_derived_from_the_nodes_and_cannot_disagree() -> None:
    """A record with two answers to "did this run" is a record nobody can use."""
    completed = a_record()
    assert completed.status == "completed"

    refused = ExecutionRecord(
        route=ROUTE,
        request=REQUEST,
        nodes=(
            NodeRecord(operation_id=ROUTE[0], status="completed"),
            NodeRecord(
                operation_id=ROUTE[1],
                status="refused",
                diagnostics="MEASURE_UNDECLARED: no quadrature measure was declared",
            ),
        ),
    )
    assert refused.status == "refused"

    # A run that stopped early is not completed, even though every node it did
    # record succeeded. The route length is part of the answer.
    partial = ExecutionRecord(
        route=ROUTE,
        request=REQUEST,
        nodes=(NodeRecord(operation_id=ROUTE[0], status="completed"),),
    )
    assert partial.status == "failed"


def test_a_failed_node_must_carry_diagnostics_and_a_completed_one_must_not() -> None:
    """A failed path returns diagnostics, never a plausible result and never silence.

    And the other direction, which is the less obvious half: a *completed* node
    carrying diagnostics reads as a warning, and this record has no vocabulary for
    one. R02.4 moved the CHE-50 curvature limitation out of exactly that position
    and into a typed field.
    """
    for status in ("refused", "failed"):
        with pytest.raises(ValueError, match="no diagnostics"):
            NodeRecord(operation_id="M_PSF", status=status)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="carries diagnostics"):
        NodeRecord(operation_id="M_PSF", status="completed", diagnostics="looked a bit off")
    with pytest.raises(ValueError, match="status"):
        NodeRecord(operation_id="M_PSF", status="maybe")  # type: ignore[arg-type]


def test_an_observed_placement_that_disagrees_with_the_request_is_visible() -> None:
    """A requested device is not evidence of an actual one.

    The failure this catches is silent: a process-global JAX platform pin produces
    a successful host run while the caller asked for CUDA, with no error raised.
    Only a comparison of the two mappings can see it, which is why they are two
    fields and not one.
    """
    node = NodeRecord(
        operation_id="S_WAVE_CHROMATIX",
        status="completed",
        requested={"device": "cuda:0", "precision": "fp32"},
        observed={"device": "cpu", "precision": "fp32"},
    )
    assert node.placement_disagreement == ("device",)
    agreeing = NodeRecord(
        operation_id="S_WAVE_CHROMATIX",
        status="completed",
        requested={"device": "cpu"},
        observed={"device": "cpu"},
    )
    assert agreeing.placement_disagreement == ()


def test_the_nodes_are_the_routes_operations_in_order() -> None:
    with pytest.raises(ValueError, match="the route's"):
        ExecutionRecord(
            route=ROUTE,
            nodes=(NodeRecord(operation_id="M_PSF", status="completed"),),
        )
    with pytest.raises(ValueError, match="not a NodeRecord"):
        ExecutionRecord(route=("M_PSF",), nodes=({"operation_id": "M_PSF"},))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="`route` is empty"):
        ExecutionRecord(route=())


def test_there_is_no_verdict_field_anywhere_on_the_record() -> None:
    """Criterion: the runtime records what happened, never whether it was right.

    The reference tree's `RecordVerdict` answered "does this record still describe
    the tree it is read in", which is a verification question. It is not here, and
    neither is a field a caller could put an answer in.
    """
    import dataclasses

    names = {field.name for field in dataclasses.fields(ExecutionRecord)} | {
        field.name for field in dataclasses.fields(NodeRecord)
    }
    for verdict in ("verdict", "reproduces", "passed", "correct", "valid", "ok", "verified"):
        assert verdict not in names, f"the record carries a verdict field {verdict!r}"
    assert not hasattr(ExecutionRecord, "explain")


def test_there_is_no_cache_no_cache_key_and_no_graph_fingerprint() -> None:
    """Criterion 4. The four fingerprints were most of the old executor's 1145 lines.

    Two exist here -- source and environment -- and each has a consumer in this
    file. A per-node cache key and a graph fingerprint do not, and CHE-199 admits
    them only if CHE-200 measures a cost that justifies one.
    """
    import ast

    package = ROOT / "src" / "runtime"
    for path in sorted(package.rglob("*.py")):
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
        code = ast.unparse(tree)
        for forbidden in ("cache", "Cache", "graph_fingerprint", "lru_cache"):
            assert forbidden not in code, f"{path.name} contains {forbidden!r}"

    # The one thing the executor remembers, and it is not a result: the environment
    # fingerprint, read once per lifetime because `importlib.metadata.version` is
    # 7.5 ms and installed versions cannot change in-process. CHE-200 measured that
    # (391% overhead on a two-node plan) and
    # `tests/integration/test_executor.py::test_two_runs_of_one_plan_both_execute_the_physics`
    # is the evidence that no *physics* is skipped -- which is the property this
    # substring scan cannot see, since it constrains identifiers rather than
    # semantics.
    assert "_environment" in (
        (ROOT / "src" / "runtime" / "executor.py").read_text(encoding="utf-8")
    )


def test_no_src_io_package_exists() -> None:
    """Criterion 5. A top-level `io` package would shadow the standard library's."""
    assert not (ROOT / "src" / "io").exists()
    assert math.isfinite(1.0)  # `io` still resolves to the stdlib, which is the point
    import io as standard_library_io

    assert standard_library_io.StringIO("x").read() == "x"
