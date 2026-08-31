"""Enumerating the whole registry loads no backend. The criterion the layer exists for.

CHE-178 (R03.2), acceptance criterion 2, and CHE-155's criterion 3. Checked in a
fresh interpreter against `sys.modules`, the same mechanism
`tests/numerics/test_no_backend_import.py` uses and for the same reason: the
failure is transitive, and reading the imports of the file in front of you will
not show a backend pulled three levels down.

**The non-vacuous part.** With no operation landed the registry is empty, so
"enumerating it imported nothing" would be true of a registry that imported
everything. Each subprocess below therefore *registers descriptors that name real
backends* -- `optiland`, `chromatix`, `jax`, `torch` -- and then enumerates,
queries and reads every field. Nothing resolves, so nothing loads, and the check
has something to catch if that ever stops being true.
"""

from __future__ import annotations

import json
import subprocess
import sys

#: The four the criterion names, plus `jaxlib`, which is what actually appears in
#: `sys.modules` when something imports JAX lazily enough to hide `jax` itself.
BACKENDS = ("jax", "jaxlib", "torch", "optiland", "chromatix")

#: Four descriptors, one per kind, each naming a backend module that exists in the
#: image. They are written as a string because they have to be constructed inside
#: the child process, which is the only interpreter whose `sys.modules` is clean.
REGISTER_BACKEND_OPERATIONS = """
import operations

for index, (kind, implementation, ports) in enumerate([
    ("solver", "optiland.optic:Optic", ("ray_bundle", "ray_bundle")),
    ("coupler", "chromatix.functional:transfer_propagate", ("ray_bundle", "scalar_field")),
    ("physical_operator", "jax.numpy:fft", ("scalar_field", "scalar_field")),
    ("measurement", "torch:as_tensor", ("scalar_field", "scalar_field")),
]):
    operations.register(operations.OperationDescriptor(
        operation_id=f"X_BACKEND_{index}",
        kind=kind,
        input=ports[0],
        output=ports[1],
        implementation=implementation,
        approximation="none; this record exists to give the import check something to catch",
        evidence=("tests/operations/test_registry_imports_no_backend.py",),
    ))
"""


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
    loaded = _modules_after("import operations")
    assert not loaded & set(BACKENDS), (
        f"importing operations loaded {sorted(loaded & set(BACKENDS))}. The package holds "
        "import paths as strings so that asking what this project can do does not load "
        "what it does it with."
    )


def test_enumerating_the_whole_registry_pulls_no_backend() -> None:
    loaded = _modules_after(
        REGISTER_BACKEND_OPERATIONS
        + """
found = operations.find()
assert len(found) == 4, found
for descriptor in found:
    # Read every field of every descriptor, which is the widest thing a caller
    # can do short of executing one.
    assert descriptor.implementation and descriptor.approximation
    assert descriptor.kind in set(operations.OperationKind)
assert operations.find(input="ray_bundle", kind="solver")
assert operations.registered_ids()
"""
    )
    assert not loaded & set(BACKENDS), (
        f"enumerating the registry loaded {sorted(loaded & set(BACKENDS))}. "
        "operations.resolve is the only call allowed to import an implementation."
    )


def test_the_check_would_notice_a_backend() -> None:
    """The detection half: prove the subprocess sees an import when there is one.

    Resolving one of the same descriptors is what a real caller does when it has
    decided to execute, and it is the one call that is *supposed* to load the
    backend -- so this both proves the check works and pins that `resolve` is
    where the import happens.
    """
    loaded = _modules_after(
        REGISTER_BACKEND_OPERATIONS
        + """
implementation = operations.resolve("X_BACKEND_3")
assert callable(implementation)
"""
    )
    assert "torch" in loaded, (
        "resolve() did not import the backend it names, so the subprocess check above "
        "cannot distinguish a lazy registry from a broken one"
    )
