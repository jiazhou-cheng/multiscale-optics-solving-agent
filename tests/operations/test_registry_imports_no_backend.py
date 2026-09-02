"""Enumerating the whole catalog loads no backend. The criterion the layer exists for.

CHE-178 (R03.2), acceptance criterion 2, and CHE-155's criterion 3. Checked in a
fresh interpreter against `sys.modules`, the same mechanism
`tests/numerics/test_no_backend_import.py` uses and for the same reason: the
failure is transitive, and reading the imports of the file in front of you will
not show a backend pulled three levels down.

**Why this is now non-vacuous by itself.** It used to be that the registry was
empty at import, so "enumerating it imported nothing" would have been true of a
registry that imported everything -- and each subprocess below therefore
*fabricated* four descriptors naming `optiland`, `chromatix`, `jax` and `torch`
just to give the check something to catch. CHE-221 (R03.4) put the real catalog in
`operations/catalog.py`, and it names `solvers.optiland.solver:trace` and
`solvers.chromatix.solver:propagate` outright. So the check now runs against the
production records, the fabrication is gone, and what is asserted is a property of
the shipped catalog rather than of a test fixture.
"""

from __future__ import annotations

import json
import subprocess
import sys

#: The four the criterion names, plus `jaxlib`, which is what actually appears in
#: `sys.modules` when something imports JAX lazily enough to hide `jax` itself.
BACKENDS = ("jax", "jaxlib", "torch", "optiland", "chromatix")


def _modules_after(statement: str) -> set[str]:
    """Top-level module names loaded by a fresh interpreter running `statement`."""
    source = (
        f"{statement}\n"
        "import sys, json\n"
        "print(json.dumps(sorted({m.split('.')[0] for m in sys.modules})))\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", source], capture_output=True, text=True, check=True
    )
    return set(json.loads(completed.stdout))


def test_importing_operations_pulls_no_backend() -> None:
    """And it is no longer importing an empty package: the catalog is built here.

    `operations/__init__.py` imports `catalog`, which constructs fourteen
    descriptors naming two backends. That construction is what used to be the
    fixture's job, and it happens at import with nothing loaded.
    """
    loaded = _modules_after("import operations")
    assert not loaded & set(BACKENDS), (
        f"importing operations loaded {sorted(loaded & set(BACKENDS))}. The package holds "
        "import paths as strings so that asking what this project can do does not load "
        "what it does it with."
    )


def test_enumerating_the_whole_catalog_pulls_no_backend() -> None:
    """The widest thing a caller can do short of executing something."""
    loaded = _modules_after(
        """
import operations

found = operations.find()
assert len(found) == 14, found
for descriptor in found:
    # Read every field of every descriptor.
    assert descriptor.implementation and descriptor.approximation
    assert descriptor.kind in set(operations.OperationKind)
    assert descriptor.evidence is not None and descriptor.validity is not None
    assert descriptor.derivative and descriptor.input and descriptor.output
    _ = descriptor.capabilities, descriptor.cost, descriptor.derivative_evidence
assert operations.find(input="ray_bundle", kind="solver")
assert operations.find(kind="coupler")
assert len(operations.registered_ids()) == 14
"""
    )
    assert not loaded & set(BACKENDS), (
        f"enumerating the catalog loaded {sorted(loaded & set(BACKENDS))}. "
        "operations.resolve is the only call allowed to import an implementation."
    )


def test_resolving_every_operation_still_loads_no_backend() -> None:
    """Measured, and stronger than the criterion asks for -- so it is recorded here.

    `resolve` is the only call in `operations/` *permitted* to import a backend.
    It turns out that resolving all fourteen imports none, because every adapter
    defers its own backend import into a function body: `solvers/optiland/system.py`
    has `_import_optiland_construction`, and importing the module gives a caller
    the neutral signature without paying for torch.

    Pinned rather than left as a happy accident, because it is a real property of
    the boundary -- resolving is not executing -- and because if it stops being
    true, the change belongs in whichever ticket makes an adapter import its
    backend at module scope, where the cost can be argued about.
    """
    loaded = _modules_after(
        """
import operations

for operation_id in operations.registered_ids():
    assert callable(operations.resolve(operation_id))
"""
    )
    assert not loaded & set(BACKENDS), (
        f"resolving the catalog loaded {sorted(loaded & set(BACKENDS))}. That is not a "
        "violation of the criterion -- resolve is allowed to import -- but it was "
        "measured not to happen, and this is where that measurement lives."
    )


def test_the_check_would_notice_a_backend() -> None:
    """The detection half: prove the subprocess probe sees an import when there is one.

    Necessary because every assertion above is a negative. The subprocess resolves
    a real catalog record and then imports the backend that record names, which is
    what executing it would eventually do -- so the probe is shown to be sensitive
    to exactly the module the assertions above claim is absent.
    """
    loaded = _modules_after(
        """
import operations

descriptor = operations.find(input="ray_bundle", kind="solver")[0]
implementation = operations.resolve(descriptor.operation_id)
assert callable(implementation)
assert descriptor.implementation.startswith("solvers.optiland")

# What executing it would do. Imported explicitly so the probe below is testing
# the probe, not the adapter's laziness.
import optiland  # noqa: F401
"""
    )
    assert "optiland" in loaded, (
        "the subprocess probe did not see a backend the child imported outright, so the "
        "negative assertions above cannot distinguish a lazy catalog from a broken check"
    )
