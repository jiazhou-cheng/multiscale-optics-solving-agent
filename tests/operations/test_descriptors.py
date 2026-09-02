"""One descriptor type, four kinds as metadata, and no second source of capability.

CHE-177 (R03.1). The acceptance criteria this file covers:

1. one descriptor type, four kinds as enum members, zero kind-specific subclasses;
2. capability information in exactly one place -- no YAML mirror, no manifest;
3. `implementation` is a string, and constructing or reading a descriptor imports
   no backend;
4. `derivative` defaults to `forward_only`, and claiming otherwise requires a
   cited finite-difference result.

The absence assertions are here for the same reason `tests/representations/`
carries its own: a budget and a diff record what exists, and neither can record
what was deliberately not built. `SolverDescriptor` would be an entirely
reasonable-looking addition to someone who had not read the ticket.
"""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path

import pytest

from operations import descriptors as descriptors_module
from operations.descriptors import (
    DERIVATIVE_MODES,
    SEMANTIC_TYPES,
    OperationDescriptor,
    OperationKind,
)

SRC = Path(__file__).resolve().parents[2] / "src"


def a_descriptor(**overrides: object) -> OperationDescriptor:
    """A minimal valid descriptor. Every test that needs a bad one starts here."""
    fields: dict[str, object] = {
        "operation_id": "X_DUMMY",
        "kind": OperationKind.COUPLER,
        "inputs": ("ray_bundle",),
        "returns": ("scalar_field",),
        "implementation": "tests.operations.nothing:run",
        "approximation": "none; this record exists to exercise the registry",
        "evidence": ("tests/operations/test_descriptors.py",),
    }
    fields.update(overrides)
    return OperationDescriptor(**fields)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Criterion 1 -- one record, four kinds, no hierarchy
# ---------------------------------------------------------------------------


def test_there_are_exactly_four_kinds() -> None:
    """Still four after CHE-224 (R15.1): `solver` left and `source` arrived.

    The count is the same and the axis is not. `solver` described who executes an
    operation while the other three describe what happens to physical state, so the
    set had one member on a different axis from the rest. `backend` answers the
    execution question now, and it is a field rather than a kind.
    """
    assert [kind.value for kind in OperationKind] == [
        "source",
        "coupler",
        "physical_operator",
        "measurement",
    ]
    assert not hasattr(OperationKind, "SOLVER")


@pytest.mark.parametrize("kind", list(OperationKind))
def test_every_kind_uses_the_same_record(kind: OperationKind) -> None:
    """The point of the enum: four kinds, one type, no per-kind construction path."""
    assert type(a_descriptor(kind=kind)) is OperationDescriptor


def test_a_kind_may_be_given_as_its_string() -> None:
    assert a_descriptor(kind="measurement").kind is OperationKind.MEASUREMENT


def test_an_unknown_kind_is_refused() -> None:
    with pytest.raises(ValueError, match="physical_operator"):
        a_descriptor(kind="propagator")


@pytest.mark.parametrize(
    "absent",
    [
        # The four hierarchies this ticket exists to avoid.
        "SolverDescriptor",
        "CouplerDescriptor",
        "PhysicalOperatorDescriptor",
        "MeasurementDescriptor",
        "OperationBase",
        # The `core/specs.py` classes that became fields on one record.
        "ModelSpec",
        "CouplerSpec",
        "PortSpec",
        "DerivativeSpec",
        "ValiditySpec",
        "CostModelSpec",
        "SourceSpec",
        "InteractionSpec",
        "StrictModel",
        "CouplerRole",
        "ApproximationClass",
        "DerivativeMode",
        "Maturity",
        "ArtifactKind",
    ],
)
def test_the_avoided_names_do_not_exist(absent: str) -> None:
    assert not hasattr(descriptors_module, absent), (
        f"{absent} is back. The four kinds are metadata on one record, and the old "
        "tree's 23 spec classes each existed because one consumer wanted one field group."
    )


def test_the_descriptor_is_frozen() -> None:
    descriptor = a_descriptor()
    with pytest.raises(dataclasses.FrozenInstanceError):
        descriptor.kind = OperationKind.SOURCE  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Criterion 2 -- capability information exists in exactly one place
# ---------------------------------------------------------------------------


def test_capabilities_are_cited_not_copied() -> None:
    """A descriptor names a measured component; it does not restate the measurement.

    The no-copy half is the field-name check below and is unchanged. What CHE-223
    (R03.6) changed is that the citation is no longer resolved *here*: the record
    lives in `knowledge/capabilities/` and nothing in this module loads one, so a
    descriptor is constructible in an interpreter that has never seen the pack.
    `tests/operations/test_capability_references.py` is where the ids are resolved.
    """
    descriptor = a_descriptor(capabilities="M_WAVE_CHROMATIX")
    assert descriptor.capabilities == "M_WAVE_CHROMATIX"
    descriptor_fields = {field.name for field in dataclasses.fields(OperationDescriptor)}
    assert not descriptor_fields & {
        "devices",
        "dtypes",
        "precisions",
        "namespaces",
        "device_namespaces",
        "accepted_input_dtypes",
        "native_compute_dtypes",
        "output_dtypes",
    }, (
        "device/dtype support is measured and lives in knowledge/capabilities/; a copy "
        "here is a second source"
    )


def test_a_capability_citation_is_validated_by_shape_and_not_by_membership() -> None:
    """CHE-223 (R03.6): the last eager coupling, and why removing it mattered.

    `__post_init__` used to check `capabilities` against
    `numerics.COMPONENT_CAPABILITIES`, so constructing *any* descriptor required the
    concrete measured table to be importable. That was an asymmetry inside one
    dataclass -- `implementation` was a lazily resolved string and `capabilities`
    was a string plus an eager global -- and it was the last thing pinning the rows
    into the foundational layer.

    A well-formed id nothing has measured is therefore **accepted here**, and
    refused by `tests/operations/test_capability_references.py`. That is the same
    division `implementation` has always had: shape at construction, resolution when
    someone asks.
    """
    assert a_descriptor(capabilities="M_RAY_INVENTED").capabilities == "M_RAY_INVENTED"
    # An operation-level id has to stay expressible: CHE-223 permits a record
    # measured for one operation rather than a whole component, so the shape must
    # not demand an `M_` prefix.
    assert a_descriptor(capabilities="O_ASM_PROPAGATE").capabilities == "O_ASM_PROPAGATE"


@pytest.mark.parametrize(
    "malformed",
    ["", "  ", "m_ray_optiland", " M_RAY_OPTILAND", "M_RAY_OPTILAND ", "9_LEADING_DIGIT",
     "M-RAY-OPTILAND", "M_RAY OPTILAND"],
)
def test_a_capability_citation_that_is_not_a_component_id_is_refused(
    malformed: str,
) -> None:
    """Shape, and it is not a formality.

    `"m_ray_optiland"` is the failure this catches: a lower-cased id is a citation
    that resolves to no file, and with membership validation gone there would be
    nothing between it and a planner reading it as a measured capability.
    """
    with pytest.raises(ValueError, match="component id"):
        a_descriptor(capabilities=malformed)


def test_no_capability_row_is_allowed_and_means_no_measurement() -> None:
    assert a_descriptor().capabilities is None


def test_there_is_no_yaml_or_manifest_mirror_of_the_registry() -> None:
    """The old tree's second source was 45 KB of YAML with a test that it agreed."""
    mirrors = [
        path
        for pattern in ("*.yaml", "*.yml", "*.json", "*.toml")
        for path in (SRC / "operations").rglob(pattern)
    ]
    assert not mirrors, (
        f"{[str(p) for p in mirrors]} mirrors the descriptors. Capability information "
        "lives in exactly one place; a generated manifest would need a test proving the "
        "generation is current, and there is no consumer asking for one."
    )


# ---------------------------------------------------------------------------
# Criterion 3 -- implementation is a string
# ---------------------------------------------------------------------------


def test_implementation_is_a_string_and_is_not_imported(monkeypatch: pytest.MonkeyPatch) -> None:
    """Constructing a descriptor resolves nothing, asserted as a delta and not a state.

    The last assertion used to be `"chromatix" not in sys.modules`, which is a claim
    about the whole interpreter rather than about this construction, and it held only
    while nothing earlier in the session had imported the backend. CHE-224 (R15.1)
    renamed `tests/solvers/` to `tests/backends/`, which moved those modules ahead of
    this one in collection order, and the assertion started failing on a test that
    had not changed -- so what it was really pinning was the alphabet.

    The before/after difference is the property the test is named for. The absolute
    version of it is a real property and it is checked where it can be:
    `test_registry_imports_no_backend.py` runs a **fresh interpreter** per probe,
    which is the only place `sys.modules` means what this line wanted it to mean.
    """
    import importlib
    import sys

    calls: list[str] = []
    monkeypatch.setattr(
        importlib, "import_module", lambda name, *a, **k: calls.append(name)  # type: ignore[misc]
    )
    before = set(sys.modules)
    descriptor = a_descriptor(implementation="chromatix.functional:transfer_propagate")
    assert isinstance(descriptor.implementation, str)
    assert calls == []
    assert set(sys.modules) - before == set()


@pytest.mark.parametrize("bad", ["chromatix.functional", "", ":run"])
def test_an_unresolvable_implementation_string_is_refused(bad: str) -> None:
    with pytest.raises(ValueError, match=re.escape("module.path:attribute")):
        a_descriptor(implementation=bad)


# ---------------------------------------------------------------------------
# Criterion 4 -- the derivative claim
# ---------------------------------------------------------------------------


def test_derivative_defaults_to_forward_only() -> None:
    assert a_descriptor().derivative == "forward_only"
    assert DERIVATIVE_MODES == ("forward_only", "differentiable")


def test_claiming_a_gradient_without_evidence_is_refused() -> None:
    with pytest.raises(ValueError, match="finite-difference"):
        a_descriptor(derivative="differentiable")


def test_claiming_a_gradient_with_blank_evidence_is_refused() -> None:
    with pytest.raises(ValueError, match="finite-difference"):
        a_descriptor(derivative="differentiable", derivative_evidence="   ")


def test_a_gradient_claim_with_a_cited_validation_is_accepted() -> None:
    descriptor = a_descriptor(
        derivative="differentiable",
        derivative_evidence="benchmarks/probes/gradients/dummy_fd.py: max rel err 3e-6",
    )
    assert descriptor.derivative == "differentiable"


def test_an_undeclared_derivative_mode_is_refused() -> None:
    with pytest.raises(ValueError, match="declared modes"):
        a_descriptor(derivative="native_autodiff")


# ---------------------------------------------------------------------------
# The remaining construction rules
# ---------------------------------------------------------------------------


def test_semantic_types_are_the_boundaries_that_landed() -> None:
    """Two representations from R02, and one measurement result type from R11.1.

    The rule the vocabulary states is that a type joins it **in the change that
    lands the boundary it names** -- so this assertion is the ratchet, and the
    exemplar below had to move off `psf` when `measurements/psf.py` landed. That
    is the vocabulary working, not the test being brittle.
    """
    assert SEMANTIC_TYPES == ("ray_bundle", "scalar_field", "psf")


@pytest.mark.parametrize("field", ["inputs", "returns"])
def test_an_undeclared_semantic_type_is_refused(field: str) -> None:
    """On a port and on the primary result. CHE-222 renamed both fields.

    `returns` is checked on element 0 only, which is where the rule belongs: an
    auxiliary value is diagnostics rather than a boundary, so it is held to the
    opposite requirement -- it must NOT be a semantic type. That is
    `test_an_auxiliary_return_may_not_be_a_semantic_type`.
    """
    with pytest.raises(ValueError, match="declared semantic type"):
        a_descriptor(**{field: ("mueller_matrix",)})


def test_an_empty_approximation_is_refused() -> None:
    with pytest.raises(ValueError, match="approximates"):
        a_descriptor(approximation="  ")


def test_evidence_must_be_written_even_when_it_is_empty() -> None:
    """No default: `evidence=()` is a statement, a missing argument is an omission."""
    with pytest.raises(TypeError):
        OperationDescriptor(  # type: ignore[call-arg]
            operation_id="X_NO_EVIDENCE",
            kind=OperationKind.COUPLER,
            inputs=("ray_bundle",),
            returns=("scalar_field",),
            implementation="tests.operations.nothing:run",
            approximation="none",
        )
    assert a_descriptor(evidence=()).evidence == ()


def test_a_blank_evidence_reference_is_refused() -> None:
    with pytest.raises(ValueError, match="empty reference"):
        a_descriptor(evidence=("",))


def test_an_empty_operation_id_is_refused() -> None:
    with pytest.raises(ValueError, match="operation_id"):
        a_descriptor(operation_id="")


def test_every_problem_is_reported_at_once() -> None:
    """Three faults, one exception -- not three edit-and-rerun cycles."""
    with pytest.raises(ValueError) as caught:
        a_descriptor(operation_id="", inputs=("mueller_matrix",), approximation="")
    message = str(caught.value)
    assert "operation_id" in message
    assert "semantic type" in message
    assert "approximates" in message


def test_lists_are_accepted_and_stored_as_tuples() -> None:
    descriptor = a_descriptor(evidence=["a"], validity=["paraxial"])
    assert descriptor.evidence == ("a",)
    assert descriptor.validity == ("paraxial",)
    assert hash(descriptor) is not None


# ---------------------------------------------------------------------------
# The observable rules. Moved here from `tests/physics/test_psf.py` by CHE-221
# (R03.4): the subject is this schema's port validation rather than the PSF, and
# the ticket confines `OperationDescriptor(...)` construction to this directory.
# R11 criterion 3's catalog-wide half stayed in the PSF tests, where it now runs
# against the shipped catalog instead of an emptied registry.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kind",
    [OperationKind.COUPLER, OperationKind.SOURCE, OperationKind.PHYSICAL_OPERATOR],
)
def test_only_a_measurement_may_produce_an_observable(kind: OperationKind) -> None:
    """Criterion 3 of the parent, as a **construction error** rather than a loop.

    The first version of this test iterated the registered couplers and asserted
    none named `psf`. At the time the registry was empty at import -- no
    registration site had landed anywhere in the tree -- so the loop body had never
    run and could not run. It asserted nothing, while a shared docstring in
    `operations/descriptors.py` cited it as the enforcement. (The catalog is
    populated now, as of CHE-221, and `tests/physics/test_psf.py` does run that
    loop over it; this test is the construction-time half, which is the one that
    binds a record nobody catalogued.)

    Adding `psf` to `SEMANTIC_TYPES` is what made `C_FIELD_TO_PSF` reconstructible,
    so the rule has to live where ports are validated. It does now, and it is the
    general statement rather than one banned id: an observable is derived from
    physical state, so only a measurement produces one.
    """
    with pytest.raises(ValueError) as raised:
        OperationDescriptor(
            operation_id="C_FIELD_TO_PSF",
            kind=kind,
            inputs=("scalar_field",),
            returns=("psf",),
            implementation="couplers.field_to_psf:convert",
            approximation="none",
            evidence=(),
        )
    assert "Only a measurement produces an observable" in str(raised.value)
    assert "CHE-36" in str(raised.value)


@pytest.mark.parametrize("kind", list(OperationKind))
def test_nothing_consumes_an_observable(kind: OperationKind) -> None:
    """The other half. An observable is terminal.

    An operation reading a PSF as its input is either a measurement of a
    measurement, or a physical operation that has mistaken an intensity for a
    state -- and in the second case the representation it should have consumed is
    still sitting upstream, unconsumed. This one binds a `measurement` too.
    """
    with pytest.raises(ValueError, match="observable and not a representation"):
        OperationDescriptor(
            operation_id="X_FROM_PSF",
            kind=kind,
            inputs=("psf",),
            returns=("scalar_field",),
            implementation="nowhere:run",
            approximation="none",
            evidence=(),
        )


# ---------------------------------------------------------------------------
# Graph entry, ports and results. CHE-222 (R03.5).
# ---------------------------------------------------------------------------


def test_a_graph_entry_is_declared_and_only_a_source_stage_may_begin_one() -> None:
    """Acceptance criterion 9. `inputs=()` became expressible, so it needs a rule.

    Only `source`-kind, since CHE-224 (R15.1) -- `solver`-kind before it, which
    contradicted the `ENTRY_KINDS` docstring's own claim that a source is the one
    operation with no input. A source initializes a representation from source
    parameters alone, whether those parameters describe the light or a system to
    launch into. The other three kinds are refused because each would be a claim
    about nothing -- a coupler changing the representation of nothing, an operator
    changing the state of nothing, a measurement observing nothing.
    """
    entry = a_descriptor(kind=OperationKind.SOURCE, inputs=())
    assert entry.is_graph_entry is True
    assert not a_descriptor().is_graph_entry

    for kind in (
        OperationKind.COUPLER,
        OperationKind.PHYSICAL_OPERATOR,
        OperationKind.MEASUREMENT,
    ):
        with pytest.raises(ValueError, match="may begin a graph entry"):
            a_descriptor(kind=kind, inputs=())

    # A composite is admitted on its FIRST stage, not on `kind` -- CHE-225 (R15.2).
    # This is the shape `SO_RAY_LAUNCH_TRACE` needs: it consumes no upstream
    # representation, and it leaves the state somewhere a source could not.
    fused = a_descriptor(
        kind=OperationKind.PHYSICAL_OPERATOR,
        inputs=(),
        composes=(OperationKind.SOURCE, OperationKind.PHYSICAL_OPERATOR),
    )
    assert fused.is_graph_entry is True
    assert fused.entry_stage is OperationKind.SOURCE
    # And a composition that does NOT begin with a source is still refused, so the
    # entry rule is keyed on the stage rather than merely bypassed by `composes`.
    with pytest.raises(ValueError, match="may begin a graph entry"):
        a_descriptor(
            kind=OperationKind.MEASUREMENT,
            inputs=(),
            composes=(OperationKind.COUPLER, OperationKind.MEASUREMENT),
        )


def test_inputs_is_required_so_no_upstream_edge_has_to_be_written() -> None:
    """`inputs=()` is a declaration; a missing argument is an omission.

    The same discipline `evidence` already had. A defaulted `inputs` would make
    "this is a source" and "nobody filled the field in" the same record, which is
    the state CHE-222 found two descriptors in from the other direction -- a fake
    input nobody had noticed was a lie.
    """
    with pytest.raises(TypeError):
        OperationDescriptor(  # type: ignore[call-arg]
            operation_id="X_NO_PORTS",
            kind=OperationKind.SOURCE,
            returns=("ray_bundle",),
            implementation="tests.operations.nothing:run",
            approximation="none",
            evidence=(),
        )


def test_an_empty_returns_is_refused() -> None:
    """Every operation returns something, and element 0 is what a planner routes."""
    with pytest.raises(ValueError, match="`returns` is empty"):
        a_descriptor(returns=())


def test_the_primary_result_is_returns_zero_with_no_operation_id_switch() -> None:
    """Acceptance criterion 5, on the schema rather than on the catalog."""
    descriptor = a_descriptor(returns=("scalar_field", "reconstruction_diagnostics"))
    assert descriptor.primary_output == "scalar_field"
    assert descriptor.returns_auxiliary is True
    assert a_descriptor(returns=("ray_bundle",)).returns_auxiliary is False


def test_an_auxiliary_return_may_not_be_a_semantic_type() -> None:
    """Only `returns[0]` is routable, and the schema says so rather than implying it.

    An auxiliary value is diagnostics a caller reads; a second *representation*
    output would be a second edge a planner could follow, and nothing produces one
    today. Refusing it makes adding one a schema change with a ticket rather than
    a record that quietly means more than the readers assume.
    """
    with pytest.raises(ValueError, match="may be one"):
        a_descriptor(returns=("scalar_field", "ray_bundle"))
    with pytest.raises(ValueError, match="empty name"):
        a_descriptor(returns=("scalar_field", "  "))
    # And a non-semantic auxiliary name is accepted, which is the positive control.
    assert a_descriptor(returns=("scalar_field", "sampling_diagnostics")).returns_auxiliary


def test_an_observable_is_refused_on_every_port_not_just_the_first() -> None:
    """Criterion 11: the `OBSERVABLE_TYPES` rules re-expressed against `inputs`.

    They used to read a single `input`. With a tuple, the rule has to hold for
    every port, or a two-port operation could consume a PSF in its second slot --
    which is the kind of gap a rename introduces silently.
    """
    with pytest.raises(ValueError, match="observable and not a representation"):
        a_descriptor(inputs=("scalar_field", "psf"))


@pytest.mark.parametrize("field", ["requires", "optional"])
def test_a_parameter_name_that_is_a_semantic_type_is_refused(field: str) -> None:
    """`requires`/`optional` name arguments; ports go in `inputs`.

    The confusion is cheap to make and invisible afterwards: a record whose
    `requires` said `"ray_bundle"` would read to a planner as a required argument
    literally named `ray_bundle`, and it would supply one.
    """
    with pytest.raises(ValueError, match="rather than a parameter name"):
        a_descriptor(**{field: ("ray_bundle",)})
    with pytest.raises(ValueError, match="empty parameter name"):
        a_descriptor(**{field: ("",)})


def test_a_parameter_cannot_be_both_required_and_optional() -> None:
    """It either has a default or it does not."""
    with pytest.raises(ValueError, match="both `requires` and `optional`"):
        a_descriptor(requires=("model",), optional=("model",))


def test_the_new_tuples_accept_lists_and_store_tuples() -> None:
    """The same normalization `evidence` and `validity` already had.

    A frozen record with a list on one of its fields is mutable through that field,
    and a transcribed record is what a list arrives as.
    """
    descriptor = a_descriptor(
        inputs=["ray_bundle"],
        returns=["scalar_field", "reconstruction_diagnostics"],
        requires=["grid_shape"],
        optional=["surface"],
    )
    for field in ("inputs", "returns", "requires", "optional"):
        assert isinstance(getattr(descriptor, field), tuple), field
    assert hash(descriptor) is not None


def test_two_records_may_share_an_implementation_and_stay_distinct() -> None:
    """Acceptance criterion 7, at the schema level.

    Nothing here refuses a duplicate `implementation`, and that is deliberate:
    planning identity is the `operation_id`.

    **No landed record relies on it any more.** `S_WAVE_CHROMATIX` and
    `O_ASM_PROPAGATE` were the case that did, and CHE-224 (R15.1) merged them once
    `backend` answered the question the pair was splitting. What this test pins is
    that the *schema* still does not deduplicate by callable, which is a different
    statement from the catalog happening to need it -- and the catalog gate now
    asserts the opposite for the shipped records, one per `implementation`.
    """
    first = a_descriptor(operation_id="X_ONE", kind=OperationKind.SOURCE)
    second = a_descriptor(operation_id="X_TWO", kind=OperationKind.PHYSICAL_OPERATOR)
    assert first.implementation == second.implementation
    assert first != second
    assert len({first, second}) == 2, "two records, not one deduplicated by callable"
