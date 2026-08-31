"""The lowest layer: precision, device and array-namespace policy.

`numerics/` is the one package that imports nothing else in this project. That is
its definition, not a convention -- `scripts/check_dependencies.py` gives it an
empty allowlist, so an import from here to any project package fails the gate.

It exists because precision and array policy need a home and `src/core/` is
banned: "core" names no domain, and a package that names no domain accumulates
whatever has no other home. That is how the reference implementation reached 110
classes in `core/`. `numerics/` names one job.

What belongs here: dtype ladders and the precision contract, device placement,
the numpy/jax/torch namespace dispatch, the measured capability table every
descriptor and solver reasons against, and the array-intake rules a
representation applies at construction.

What does not: anything with a physical unit or a physical boundary. A wavelength
is not a numeric policy. That is `representations/`.

Two modules, both landed by CHE-173 (R02.1):

* `precision` -- the vocabulary (`Precision`, `DType`, `DeviceKind`,
  `ArrayNamespace`, `DevicePlacement`, `ArrayState`), the probe-backed
  `ComponentCapabilities` table, and device/dtype negotiation as functions.
* `arrays` -- introspection of a real buffer and the conversions a negotiation
  authorized.

Importing this package pulls **no backend**: not JAX, not Torch, not Optiland,
not Chromatix. NumPy is the exception and is not a backend -- it is the array
vocabulary the other three are described in, and it is present in every image.
`tests/numerics/test_no_backend_import.py` checks this against `sys.modules`.
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
from numerics.precision import (
    CHROMATIX_CAPABILITIES,
    COMPONENT_CAPABILITIES,
    OPTILAND_CAPABILITIES,
    PHASE_ACCUMULATION_FLOOR,
    REFUSAL_CODES,
    ArrayNamespace,
    ArrayState,
    ComponentCapabilities,
    DeviceKind,
    DevicePlacement,
    DType,
    Precision,
    capabilities_for,
    capability_rows,
    compute_dtype,
    negotiate,
    refusal,
)

__all__ = [
    "CHROMATIX_CAPABILITIES",
    "COMPONENT_CAPABILITIES",
    "COMPUTE_NAMESPACES",
    "OPTILAND_CAPABILITIES",
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
    "capabilities_for",
    "capability_rows",
    "compute_dtype",
    "device_of",
    "dtype_of",
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
