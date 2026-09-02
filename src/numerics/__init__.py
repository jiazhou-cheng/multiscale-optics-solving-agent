"""The lowest layer: precision, device and array-namespace policy.

`numerics/` is the one package that imports nothing else in this project. That is
its definition, not a convention -- `scripts/check_dependencies.py` gives it an
empty allowlist, so an import from here to any project package fails the gate.

It exists because precision and array policy need a home and `src/core/` is
banned: "core" names no domain, and a package that names no domain accumulates
whatever has no other home. That is how the reference implementation reached 110
classes in `core/`. `numerics/` names one job.

What belongs here: dtype ladders and the precision contract, device placement,
the numpy/jax/torch namespace dispatch, the capability *contract* every descriptor
and solver reasons against, and the array-intake rules a representation applies at
construction.

What does not: anything with a physical unit or a physical boundary. A wavelength
is not a numeric policy. That is `representations/`. **And, since CHE-223 (R03.6),
the measured capability rows themselves**: this package no longer knows that
Optiland or Chromatix exist. The rows are data under `knowledge/capabilities/` and
`knowledge.load_capabilities` reads one into a validated `ComponentCapabilities`.
That module's docstring records why solver-local ownership (CHE-206's plan) could
not be the single source once backend-free discovery became a second consumer.

Three modules:

* `precision` -- CHE-173 (R02.1). The vocabulary (`Precision`, `DType`,
  `DeviceKind`, `ArrayNamespace`, `DevicePlacement`, `ArrayState`), the
  `ComponentCapabilities` contract with its ten widening refusals, and
  device/dtype negotiation as functions.
* `knowledge` -- CHE-223 (R03.6). The backend-free loader over
  `knowledge/capabilities/`. It names no component and no backend: it enumerates
  the directory, so adding a measured component is adding a file.
* `arrays` -- CHE-173 (R02.1). Introspection of a real buffer and the conversions
  a negotiation authorized.

Importing this package pulls **no backend**: not JAX, not Torch, not Optiland,
not Chromatix. Neither does loading a capability record -- it is `json`, `pathlib`
and this package's own enums. NumPy is the exception and is not a backend: it is
the array vocabulary the other three are described in, and it is present in every
image. `tests/numerics/test_no_backend_import.py` checks both against
`sys.modules` in a fresh interpreter.
"""

from numerics.arrays import (
    COMPUTE_NAMESPACES,
    array_state,
    device_of,
    dtype_of,
    matmul_precision_kwargs,
    namespace_of,
    numpy_dtype,
    to_host_numpy,
    to_namespace,
    to_state,
    verify_dtype,
    xp_for,
)
from numerics.knowledge import (
    CAPABILITY_DIRECTORY,
    CAPABILITY_SCHEMA_VERSION,
    KNOWLEDGE_ROOT,
    capability_record_ids,
    capability_rows,
    load_capabilities,
)
from numerics.precision import (
    PHASE_ACCUMULATION_FLOOR,
    REFUSAL_CODES,
    ArrayNamespace,
    ArrayState,
    ComponentCapabilities,
    DeviceKind,
    DevicePlacement,
    DType,
    Precision,
    compute_dtype,
    negotiate,
    refusal,
)

__all__ = [
    "CAPABILITY_DIRECTORY",
    "CAPABILITY_SCHEMA_VERSION",
    "COMPUTE_NAMESPACES",
    "KNOWLEDGE_ROOT",
    "PHASE_ACCUMULATION_FLOOR",
    "REFUSAL_CODES",
    "ArrayNamespace",
    "ArrayState",
    "ComponentCapabilities",
    "DType",
    "DeviceKind",
    "DevicePlacement",
    "Precision",
    "array_state",
    "capability_record_ids",
    "capability_rows",
    "compute_dtype",
    "device_of",
    "dtype_of",
    "load_capabilities",
    "matmul_precision_kwargs",
    "namespace_of",
    "negotiate",
    "numpy_dtype",
    "refusal",
    "to_host_numpy",
    "to_namespace",
    "to_state",
    "verify_dtype",
    "xp_for",
]
