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

from numerics import COMPONENT_CAPABILITIES
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
        "input": "ray_bundle",
        "output": "scalar_field",
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
    assert [kind.value for kind in OperationKind] == [
        "solver",
        "coupler",
        "physical_operator",
        "measurement",
    ]


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
        descriptor.kind = OperationKind.SOLVER  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Criterion 2 -- capability information exists in exactly one place
# ---------------------------------------------------------------------------


def test_capabilities_are_cited_not_copied() -> None:
    """A descriptor names a row of the measured table; it does not restate it."""
    descriptor = a_descriptor(capabilities="M_WAVE_CHROMATIX")
    assert descriptor.capabilities == "M_WAVE_CHROMATIX"
    row = COMPONENT_CAPABILITIES[descriptor.capabilities]
    # The evidence stays with the measurement. Nothing on the descriptor could
    # disagree with the row, because nothing on the descriptor restates it.
    assert row.probe.startswith("benchmarks/probes/")
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
    }, "device/dtype support is measured in numerics; a copy here is a second source"


def test_citing_a_capability_row_that_does_not_exist_is_refused() -> None:
    with pytest.raises(ValueError, match="COMPONENT_CAPABILITIES"):
        a_descriptor(capabilities="M_RAY_INVENTED")


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
    import importlib
    import sys

    calls: list[str] = []
    monkeypatch.setattr(
        importlib, "import_module", lambda name, *a, **k: calls.append(name)  # type: ignore[misc]
    )
    descriptor = a_descriptor(implementation="chromatix.functional:transfer_propagate")
    assert isinstance(descriptor.implementation, str)
    assert calls == []
    assert "chromatix" not in sys.modules


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


@pytest.mark.parametrize("field", ["input", "output"])
def test_an_undeclared_semantic_type_is_refused(field: str) -> None:
    with pytest.raises(ValueError, match="declared semantic type"):
        a_descriptor(**{field: "mueller_matrix"})


def test_an_empty_approximation_is_refused() -> None:
    with pytest.raises(ValueError, match="approximates"):
        a_descriptor(approximation="  ")


def test_evidence_must_be_written_even_when_it_is_empty() -> None:
    """No default: `evidence=()` is a statement, a missing argument is an omission."""
    with pytest.raises(TypeError):
        OperationDescriptor(  # type: ignore[call-arg]
            operation_id="X_NO_EVIDENCE",
            kind=OperationKind.COUPLER,
            input="ray_bundle",
            output="scalar_field",
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
        a_descriptor(operation_id="", input="mueller_matrix", approximation="")
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
    [OperationKind.COUPLER, OperationKind.SOLVER, OperationKind.PHYSICAL_OPERATOR],
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
            input="scalar_field",
            output="psf",
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
            input="psf",
            output="scalar_field",
            implementation="nowhere:run",
            approximation="none",
            evidence=(),
        )
