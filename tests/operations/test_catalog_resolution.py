"""Every catalog `implementation` string resolves to the callable it names.

CHE-221 (R03.4), acceptance criteria 3 and 6. The one test module in
`tests/operations/` that **imports backends on purpose**, which is why it is a
separate file: `test_catalog.py` walks the catalog structurally and must stay
runnable without torch or JAX, and the no-backend criterion is asserted in fresh
subprocesses by `test_registry_imports_no_backend.py`, so nothing here can
contaminate it.

Compared **by identity**, not by `callable()`
---------------------------------------------
`callable(operations.resolve(id))` is nearly free to satisfy and catches almost
nothing: it passes for a class, for an unrelated function in the same module, and
for any attribute that happens to be callable. `resolve` already refuses a
non-callable itself. So each assertion below imports the intended attribute the
ordinary way and asserts the resolved object **is** it, which is the check that
catches a record pointing at the wrong name in the right module -- the failure a
rename produces.

Not marked `slow`. It adds no import the default suite does not already pay for:
`tests/physics/` imports both backends, and `-m "not slow"` is the declared
default gate, so a gate that catches a stale implementation string belongs inside
it rather than behind a marker nobody selects.
"""

from __future__ import annotations

import importlib

import pytest

import operations
from operations import CATALOG

#: `(operation_id, module_path, attribute)` for every record, as test ids.
CASES = [
    (record.operation_id, *record.implementation.split(":", 1)) for record in CATALOG
]


@pytest.mark.parametrize(
    ("operation_id", "module_path", "attribute"), CASES, ids=[case[0] for case in CASES]
)
def test_the_implementation_string_resolves_to_the_intended_callable(
    operation_id: str, module_path: str, attribute: str
) -> None:
    """Criteria 3 and 6, per record. A renamed or deleted target fails here."""
    intended = getattr(importlib.import_module(module_path), attribute)
    resolved = operations.resolve(operation_id)
    assert resolved is intended, (
        f"{operation_id} declares {module_path}:{attribute} but resolved to "
        f"{resolved!r}, which is not that attribute"
    )
    assert callable(resolved)


def test_every_record_is_covered_by_the_parametrization() -> None:
    """The meta-check: a parametrization built from an empty catalog proves nothing."""
    assert len(CASES) == len(CATALOG) == 17
    assert {case[0] for case in CASES} == set(operations.registered_ids())


def test_an_unknown_id_is_a_keyerror_naming_the_catalog() -> None:
    """The other half of resolution: what a caller gets for an id nobody declared."""
    with pytest.raises(KeyError, match="not in the catalog"):
        operations.resolve("S_NOT_A_THING")


def test_a_record_pointing_at_a_missing_attribute_is_an_attribute_error() -> None:
    """Criterion 6's mechanism, exercised on a record that is wrong on purpose.

    The real catalog cannot supply this case -- that is what the parametrized test
    above establishes -- so the failure path is driven through `resolve`'s own
    index with a deliberately stale string. This is the error a rename would raise
    before the parametrized test's identity comparison ever ran.
    """
    import dataclasses

    from operations.registry import _BY_ID

    stale = dataclasses.replace(
        _BY_ID["M_PSF"],
        operation_id="M_PSF_STALE",
        implementation="measurements.psf:psf_renamed_away",
    )
    patched = {**_BY_ID, "M_PSF_STALE": stale}
    import operations.registry as registry_module

    original = registry_module._BY_ID
    registry_module._BY_ID = patched
    try:
        with pytest.raises(AttributeError, match="has no attribute"):
            operations.resolve("M_PSF_STALE")
    finally:
        registry_module._BY_ID = original
