"""Every catalog record's ports, requirements and results, derived from the code.

CHE-222 (R03.5), acceptance criterion 8 and the mechanical half of 1-6. The
criterion asks that no catalog record declare a representation input its callable
does not accept, "checked … by inspecting the resolved signature, so the check is
against the code and not against a table". The signatures turn out to be fully
machine-readable, so this module does the stronger thing: it **derives all four
tuples** -- `inputs`, `requires`, `optional`, and the arity of `returns` -- from
`inspect.signature` and compares them with the record, for all thirteen.

That makes the descriptor's argument metadata a *checked* restatement rather than a
hand-maintained one, which matters because a second source of truth beside a
signature is exactly the thing that drifts. Renaming a parameter, giving one a
default, removing one, or adding a required argument all fail here.

What this cannot check
----------------------
`approximation`, `validity` and `evidence` are physics prose and no derivation can
verify them -- `tests/operations/test_catalog.py` says the same. And `returns[0]`
is a declared *convention* rather than a derivation for one record: `psf` returns a
`PsfResult`, and that the observable it carries is what a planner routes is a
decision (`measurements/psf.py`), not something the annotation states. That single
mapping lives in `RESULT_TYPES` below, named so it is visible rather than buried in
a comparison.

This module resolves implementations, so it sits beside
`test_catalog_resolution.py` rather than in the no-backend path -- although,
measured, resolving loads no backend either
(`test_registry_imports_no_backend.py::test_resolving_every_operation_still_loads_no_backend`).
"""

from __future__ import annotations

import inspect

import pytest

import operations
from operations import CATALOG

#: How a return annotation names a semantic result, and the one declared
#: convention: `PsfResult` is the record `measurements.psf` returns, and the
#: primary semantic value it carries is the `psf` observable.
RESULT_TYPES: dict[str, str] = {
    "RayBundle": "ray_bundle",
    "ScalarField": "scalar_field",
    "PsfResult": "psf",
}

#: Annotations that name a **representation port** -- the subset an operation can
#: consume. `PsfResult` is absent on purpose: an observable is terminal and nothing
#: takes one, which `OperationDescriptor.__post_init__` refuses independently.
PORT_TYPES: dict[str, str] = {
    "RayBundle": "ray_bundle",
    "ScalarField": "scalar_field",
}


def _split_tuple_members(annotation: str) -> list[str]:
    """The members of a `tuple[...]` annotation, respecting nested brackets.

    A plain `split(",")` reads `tuple[RayBundle, dict[str, Any]]` as three members
    and would report `diffractive_surface` as returning three values. The depth
    counter is the whole of the parsing this file needs.
    """
    inner = annotation[len("tuple[") : -1]
    members: list[str] = []
    depth = 0
    current = ""
    for character in inner:
        if character == "," and depth == 0:
            members.append(current.strip())
            current = ""
            continue
        if character in "[(":
            depth += 1
        elif character in "])":
            depth -= 1
        current += character
    members.append(current.strip())
    return members


def _derive(operation_id: str) -> dict[str, object]:
    """`inputs`, `requires`, `optional` and the return arity, read off the callable.

    A parameter whose annotation names a representation is a **port**; anything
    else is a requirement or an option depending on whether it has a default. That
    classification is the descriptor's own rule, applied to the signature.

    **Port detection fails loudly rather than silently reclassifying.** An exact
    string match is the whole of the port test, so anything it misses -- an
    annotation like `RayBundle | None`, a module-qualified or quoted one, an
    unannotated parameter, `*args`/`**kwargs` -- would fall through and be recorded
    as a *parameter name* in `requires` or `optional`. `__post_init__` accepts
    `"rays"` there quite happily, so a future port could be dropped from `inputs`,
    the record written to match, and every assertion in this module stay green.
    That is the one way this gate can be defeated, so the ambiguous cases raise.
    The return path already behaves this way by construction: `RESULT_TYPES[...]`
    raises `KeyError` on an annotation it does not know.
    """
    signature = inspect.signature(operations.resolve(operation_id))
    inputs: list[str] = []
    requires: list[str] = []
    optional: list[str] = []
    for name, parameter in signature.parameters.items():
        if parameter.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            raise AssertionError(
                f"{operation_id}: {name!r} is *args/**kwargs, which this derivation "
                "cannot classify. A public operation with a variadic parameter needs a "
                "decision about how the descriptor describes it."
            )
        if parameter.annotation is inspect.Parameter.empty:
            raise AssertionError(
                f"{operation_id}: {name!r} carries no annotation, so it cannot be told "
                "from a representation port. An unannotated parameter would be filed as "
                "a requirement whatever it actually is."
            )
        annotation = str(parameter.annotation)
        port = PORT_TYPES.get(annotation)
        if port is None and any(known in annotation for known in PORT_TYPES):
            raise AssertionError(
                f"{operation_id}: {name!r} is annotated {annotation!r}, which mentions a "
                "representation but is not exactly one. Filing it as a parameter name "
                "would hide a port; decide what the descriptor should say and widen "
                "PORT_TYPES deliberately."
            )
        if port is not None:
            inputs.append(port)
        elif parameter.default is not inspect.Parameter.empty:
            optional.append(name)
        else:
            requires.append(name)

    returned = str(signature.return_annotation)
    if returned.startswith("tuple["):
        members = _split_tuple_members(returned)
        primary, arity = members[0], len(members)
    else:
        primary, arity = returned, 1
    return {
        "inputs": tuple(inputs),
        "requires": tuple(requires),
        "optional": tuple(optional),
        "primary_output": RESULT_TYPES[primary],
        "return_arity": arity,
    }


CASES = [(record.operation_id, record) for record in CATALOG]
IDS = [case[0] for case in CASES]


@pytest.mark.parametrize(("operation_id", "record"), CASES, ids=IDS)
def test_the_declared_ports_are_the_ones_the_callable_takes(
    operation_id: str, record: object
) -> None:
    """Criterion 8, and criteria 1 and 2 mechanically.

    This is the test that would have failed before CHE-222: `S_SOURCE_PLANE_WAVE`
    declared `input="scalar_field"` for a function taking no field, and
    `S_RAY_OPTILAND` declared `input="ray_bundle"` for one taking no bundle. Both
    now declare `inputs=()`, and the derivation agrees.
    """
    derived = _derive(operation_id)
    assert record.inputs == derived["inputs"], (  # type: ignore[attr-defined]
        f"{operation_id} declares inputs={record.inputs!r} but its callable takes "  # type: ignore[attr-defined]
        f"{derived['inputs']!r} as representation arguments"
    )


@pytest.mark.parametrize(("operation_id", "record"), CASES, ids=IDS)
def test_the_declared_requirements_are_the_arguments_with_no_default(
    operation_id: str, record: object
) -> None:
    """Criterion 3. Eleven of the thirteen need a value the old schema never mentioned.

    The ticket says nine, counting from a table that predated three of the records;
    the measured figure is twelve -- everything except `O_COMPLEX_TRANSMISSION` and
    `C_SCALAR_TO_RAY`. `test_catalog.py::test_question_3_every_required_value_is_named`
    lists them.
    """
    derived = _derive(operation_id)
    assert record.requires == derived["requires"], (  # type: ignore[attr-defined]
        f"{operation_id} declares requires={record.requires!r} but its callable's "  # type: ignore[attr-defined]
        f"non-representation arguments without a default are {derived['requires']!r}"
    )


@pytest.mark.parametrize(("operation_id", "record"), CASES, ids=IDS)
def test_the_declared_options_are_the_arguments_with_a_default(
    operation_id: str, record: object
) -> None:
    """Criterion 4. Names only -- the defaults deliberately are not mirrored."""
    derived = _derive(operation_id)
    assert record.optional == derived["optional"], (  # type: ignore[attr-defined]
        f"{operation_id} declares optional={record.optional!r} but its callable's "  # type: ignore[attr-defined]
        f"defaulted arguments are {derived['optional']!r}"
    )


@pytest.mark.parametrize(("operation_id", "record"), CASES, ids=IDS)
def test_the_declared_results_match_the_return_annotation(
    operation_id: str, record: object
) -> None:
    """Criteria 5 and 6, against the annotation rather than against a table.

    Arity is what `output` could not carry: `propagate_rays` and
    `diffractive_surface` both had `output="ray_bundle"` while one returns a bundle
    and the other a 2-tuple.
    """
    derived = _derive(operation_id)
    assert record.primary_output == derived["primary_output"], (  # type: ignore[attr-defined]
        f"{operation_id} declares returns[0]={record.primary_output!r} but its callable "  # type: ignore[attr-defined]
        f"returns {derived['primary_output']!r} as its first value"
    )
    assert len(record.returns) == derived["return_arity"], (  # type: ignore[attr-defined]
        f"{operation_id} declares {len(record.returns)} returned value(s) but its "  # type: ignore[attr-defined]
        f"annotation has {derived['return_arity']}"
    )


def test_the_derivation_is_not_vacuous() -> None:
    """The meta-check: the derivation has to distinguish the records it compares.

    Every assertion above is an equality, so a derivation that returned the same
    thing for every operation would pass all fifty-six. These four facts are what
    say it does not.
    """
    assert len(CASES) == 13
    assert _derive("S_SOURCE_PLANE_WAVE")["inputs"] == ()
    assert _derive("O_PROPAGATE_RAYS")["inputs"] == ("ray_bundle",)
    assert _derive("O_PROPAGATE_RAYS")["return_arity"] == 1
    assert _derive("O_DIFFRACTIVE_SURFACE")["return_arity"] == 2, (
        "the bracket-aware split is what makes tuple[RayBundle, dict[str, Any]] two "
        "values rather than three"
    )
    assert _derive("M_PSF")["requires"] == ("normalization",)


def test_the_port_classifier_refuses_what_it_cannot_classify() -> None:
    """The guard in `_derive`, exercised -- because it is what keeps the gate honest.

    An exact annotation match silently reclassifies anything it misses as a
    parameter *name*, which would let a future port be dropped from `inputs` with
    every test still passing. Three shapes are refused: a variadic parameter, an
    unannotated one, and one whose annotation mentions a representation without
    being exactly one.
    """
    import operations as operations_module
    from representations import RayBundle

    def variadic(*rays: RayBundle) -> RayBundle:
        raise NotImplementedError

    def unannotated(rays) -> RayBundle:  # type: ignore[no-untyped-def]
        raise NotImplementedError

    def near_miss(rays: RayBundle | None) -> RayBundle:
        raise NotImplementedError

    for candidate, fragment in (
        (variadic, "cannot classify"),
        (unannotated, "no annotation"),
        (near_miss, "mentions a"),
    ):
        original = operations_module.resolve
        operations_module.resolve = lambda _oid, _fn=candidate: _fn  # type: ignore[assignment]
        try:
            with pytest.raises(AssertionError, match=fragment):
                _derive("X_PROBE")
        finally:
            operations_module.resolve = original


def test_a_fake_input_would_be_caught() -> None:
    """The falsifier for criterion 8, on the exact record that used to carry one.

    Before CHE-222, `S_SOURCE_PLANE_WAVE` declared `input="scalar_field"`. Rebuilt
    here to show the comparison fires rather than being trivially satisfied by two
    empty tuples.
    """
    import dataclasses

    record = next(r for r in CATALOG if r.operation_id == "S_SOURCE_PLANE_WAVE")
    fake = dataclasses.replace(record, inputs=("scalar_field",), kind=record.kind)
    assert fake.inputs != _derive("S_SOURCE_PLANE_WAVE")["inputs"]
