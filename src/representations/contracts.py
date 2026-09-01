"""The failure vocabulary and the array-intake rules every representation applies.

CHE-175 (R02.3). Two jobs that belong together because both are "what a
representation does to a declaration before it will hold it", and because
`geometry`, `rays` and `scalar` all need them: a module that only `rays` used
would live in `rays.py`, but `scalar.py` must not import the ray module to get an
exception class, and `geometry.py` cannot import either without a cycle.

`ContractError` is the one exception class in the package
--------------------------------------------------------
It is a class rather than the `ValueError` + `code` attribute that
`numerics/precision.py` uses, and the difference is the caller. A numerics
refusal is read by a person: nothing catches `UNSUPPORTED_DTYPE` and does
something else. A contract failure is read by a *coupler*, which R07 requires to
return an explicit diagnostic instead of an invented field -- `except
ContractError as error: return error.as_diagnostic()`. That needs one catchable
type, and `except ValueError` would also swallow every unrelated arithmetic
error raised inside the same block.

It stays a `ValueError` subclass, so a caller that does not care about the
distinction keeps working and the R02.2 constructions that predate this module
did not change meaning when they moved onto it.

`CONTRACT_CODES` is enumerated for the same reason `REFUSAL_CODES` is: so
`tests/representations/test_contract_codes.py` can prove every declared code has
something that raises it. A code nothing can trigger is a claim about a failure
path that does not exist, and R02.1 found one of those by writing this test.

Array intake
------------
`adopt_array` replaces the reference implementation's `_intake`
(`pre-rewrite-2026-08-30:src/core/boundary.py:278`) and keeps its central
finding: before CHE-61 every boundary field ran through `np.asarray(value,
dtype=np.float64)`, one line that moved data to the host, changed its dtype,
changed its array ecosystem and broke any autograd graph attached to it -- so a
float32 GPU artifact could not exist whatever the producing solver supported. An
input that *is* an array is left in the representation it arrived in; only an
input with no representation of its own (a list, a Python scalar) takes the
historical host float64/complex128 default.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from numerics import (
    ArrayNamespace,
    DType,
    device_of,
    dtype_of,
    namespace_of,
    numpy_dtype,
    xp_for,
)

__all__ = [
    "CONTRACT_CODES",
    "ContractError",
    "adopt_array",
    "require_finite",
    "require_positive_si",
    "require_same_representation",
]

#: Every structured failure a representation, or a consumer of one, can raise.
#:
#: Enumerated, not free-form, because these are the strings a coupler branches on
#: and a diagnostic reports. Each one has a test that triggers it; adding a code
#: without a trigger fails that test.
#:
#: `GRAZING_PHASE_UNREPRESENTABLE` is the first code whose only raiser is a
#: **consumer** rather than a representation's own `__post_init__`. It belongs to
#: `couplers.ray_to_scalar` (CHE-188): whether a mode's constant phase can be
#: represented depends on the compute precision the *consumer* chose, so no
#: representation can decide it at construction. The vocabulary lives here anyway,
#: because a shared failure alphabet is what lets one `except ContractError` handle
#: a boundary end to end -- and because a code invented locally by each consumer is
#: how the free-form provenance string this module replaced came back.
CONTRACT_CODES: tuple[str, ...] = (
    "MISSING_DECLARATION",
    "UNIT_NOT_SI",
    "NON_FINITE",
    "FRAME_MISMATCH",
    "SHAPE_MISMATCH",
    "NON_UNIT_DIRECTION",
    "EMPTY_ENSEMBLE",
    "DTYPE_KIND_MISMATCH",
    "REPRESENTATION_INCONSISTENT",
    "PHASOR_MISMATCH",
    "COHERENT_STATE_INCOMPLETE",
    "OPL_REFERENCE_UNVERIFIED",
    "MEASURE_UNDECLARED",
    "UNKNOWN_MEASURE_KIND",
    "UNKNOWN_VALIDITY_FLAG",
    "PAD_STATE_UNKNOWN",
    "GRAZING_PHASE_UNREPRESENTABLE",
)


class ContractError(ValueError):
    """A boundary declaration is missing, inconsistent, or unusable.

    Carries a machine-readable `code` so a coupler can return a structured
    diagnostic instead of an invented result, and a `declaration` naming the
    field that failed so the diagnostic points at something the caller can fix.
    """

    def __init__(
        self,
        code: str,
        message: str,
        *,
        declaration: str | None = None,
        remedy: str | None = None,
    ) -> None:
        if code not in CONTRACT_CODES:
            raise ValueError(f"{code!r} is not a declared contract code: {list(CONTRACT_CODES)}")
        self.code = code
        self.declaration = declaration
        self.remedy = remedy
        detail = f"[{code}] {message}"
        if declaration:
            detail += f" (declaration: {declaration!r})"
        if remedy:
            detail += f" Remedy: {remedy}"
        super().__init__(detail)

    def as_diagnostic(self) -> dict[str, str | None]:
        """The form a coupler returns instead of a result it could not compute."""
        return {
            "code": self.code,
            "message": str(self),
            "declaration": self.declaration,
            "remedy": self.remedy,
        }


def require_positive_si(value: Any, *, name: str) -> float:
    """A positive, finite SI scalar, as a float. Never substitutes a default."""
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ContractError(
            "UNIT_NOT_SI",
            f"{name} must be a positive finite value in SI units, got {value!r}",
            declaration=name,
        )
    return number


def require_finite(array: Any, *, name: str) -> None:
    """Every element is finite.

    A NaN here is not a rounding artifact: it is a vignetted ray that was kept,
    or a divide-by-zero in a producer, and it propagates through a coherent sum
    to make the whole field NaN rather than a locally wrong one.
    """
    xp = xp_for(namespace_of(array))
    if not bool(xp.all(xp.isfinite(array))):
        raise ContractError(
            "NON_FINITE",
            f"{name} contains non-finite values",
            declaration=name,
        )


def adopt_array(value: Any, *, name: str, complex_: bool, widen_real: bool = False) -> Any:
    """Adopt `value` as boundary data, preserving the representation it arrived in.

    `complex_` says which kind the *field* is, not what the caller passed.

    `widen_real` is the difference between an amplitude and a field, and it is a
    physical distinction rather than a convenience:

    * `RayBundle.amplitude` sets it. A real array there is a phase-free
      amplitude -- `sqrt(w)` of a declared power weight is exactly that -- so it
      is widened to the complex dtype *of the same precision* (`float32 ->
      complex64`, never `-> complex128`, which would fabricate precision the
      producer never had).
    * `ScalarField.u` does not. A real 2-D array on a grid is an intensity map,
      and silently reading `|U|` as `U` discards the phase that makes the field a
      field. It is refused.
    """
    if not hasattr(value, "dtype"):
        # A list, tuple or Python scalar owns no buffer, so there is no
        # representation to preserve and the historical default applies.
        default = DType.COMPLEX128 if complex_ else DType.FLOAT64
        return np.asarray(value, dtype=numpy_dtype(default))

    namespace = namespace_of(value)
    if namespace is ArrayNamespace.TORCH:
        raise ContractError(
            "REPRESENTATION_INCONSISTENT",
            (
                f"{name} is a torch tensor. A representation holds data in a compute "
                "namespace (NumPy or JAX); a torch buffer crosses an explicit bridge "
                "so the conversion, and any autograd graph break it causes, is a "
                "recorded decision rather than an accident of construction."
            ),
            declaration=name,
            remedy="Bridge it with numerics.to_state / numerics.to_namespace first.",
        )

    dtype = dtype_of(value)
    if complex_ and not dtype.is_complex:
        if not widen_real:
            raise ContractError(
                "DTYPE_KIND_MISMATCH",
                (
                    f"{name} is {dtype}, a real array. A real array on a grid is an "
                    "intensity, not an amplitude: |U| has thrown away the phase, and no "
                    "later operation can recover it."
                ),
                declaration=name,
            )
        target = dtype.precision.complex_dtype or DType.COMPLEX64
        return xp_for(namespace).asarray(value, dtype=numpy_dtype(target))
    if not complex_ and dtype.is_complex:
        raise ContractError(
            "DTYPE_KIND_MISMATCH",
            f"{name} must be real-valued; a complex array here is an amplitude, not a magnitude",
            declaration=name,
        )
    return value


def require_same_representation(arrays: dict[str, Any], *, reference: str) -> None:
    """Every array in one artifact lives in the same place, in the same ecosystem.

    CHE-61's rule. One artifact cannot span two devices or two array ecosystems:
    the first operation to touch both would silently move one of them, and the
    move would be attributed to nothing.

    Dtype is deliberately **not** unified. A float32 geometry with a complex64
    amplitude is a legitimate FP32 artifact, and forcing a common dtype would
    reintroduce exactly the hidden conversion this check exists to remove.
    """
    base = arrays[reference]
    base_device, base_namespace = device_of(base), namespace_of(base)
    for name, array in arrays.items():
        if array is None or name == reference:
            continue
        device, namespace = device_of(array), namespace_of(array)
        if namespace is not base_namespace or device != base_device:
            raise ContractError(
                "REPRESENTATION_INCONSISTENT",
                (
                    f"{name} is {namespace}:{device} but {reference} is "
                    f"{base_namespace}:{base_device}. One artifact cannot span two "
                    "devices or two array ecosystems; the first operation to touch both "
                    "would silently move one of them."
                ),
                declaration=name,
                remedy="Bridge the mismatched array explicitly before constructing the artifact.",
            )
