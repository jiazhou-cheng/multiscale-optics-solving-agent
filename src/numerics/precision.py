"""Precision, device and namespace vocabulary, and the capability *contract*.

CHE-173 (R02.1). Three things that a single `config["dtype"] = "float64"` string
used to conflate, separated here because the pinned backends disagree about all
three:

`Precision`
    An execution *policy* -- FP16/FP32/FP64. An intent and an accuracy family,
    not a storage format. FP32 legitimately means `float32` for real data and
    `complex64` for a field.

`DType`
    An *observed* storage property of an actual array. Read off the buffer by
    `numerics.arrays.dtype_of`, never inferred from a requested precision.

`DevicePlacement` / `ArrayNamespace`
    Where the buffer physically is, ordinal included, and which ecosystem owns
    it. Orthogonal: NumPy is host-only, JAX and Torch are either. A requested
    device is not evidence of an actual one, which is why the two are named
    differently everywhere in this package.

The capability contract, and where the rows are not
---------------------------------------------------
`ComponentCapabilities` is the declaration a solver and a descriptor reason
against. A row copied from an API signature would be worse than no row, because it
would be trusted -- so it refuses to be constructed without a probe path, a probe
tag and an evidence sentence, and refuses declarations that are internally wider
than what they state (see `__post_init__`, ten rules).

**The measured rows themselves are not in this module and this module names no
backend.** They are one JSON file per component under `knowledge/capabilities/`,
read by `numerics.knowledge.load_capabilities` -- CHE-223 (R03.6), and the section
lower down says why solver-local ownership could not be the single source. Each
record carries the git tag its probe path resolves against, in `probe_tag`, because
that is a property of the measurement rather than of this module: `benchmarks/` is
not in the working tree, so a path with no revision names nothing, and re-running
the probe is what a *widening* of any record costs.

Failure vocabulary
------------------
No exception classes. A refusal is a `ValueError` carrying a `code` attribute
from `REFUSAL_CODES` and a message that names what was requested, what is
supported and the evidence behind the refusal. The old tree spent two classes
(`CapabilityError`, `BridgeError`) on that, and neither was ever caught by type.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

__all__ = [
    "PHASE_ACCUMULATION_FLOOR",
    "REFUSAL_CODES",
    "ArrayNamespace",
    "ArrayState",
    "ComponentCapabilities",
    "DType",
    "DeviceKind",
    "DevicePlacement",
    "Precision",
    "compute_dtype",
    "negotiate",
    "refusal",
]


#: Every structured refusal this package can raise. Enumerated so
#: `tests/numerics/test_refusals.py` can prove each one is reachable -- a declared
#: failure code nothing can trigger is a claim about a path that does not exist.
REFUSAL_CODES: tuple[str, ...] = (
    "UNKNOWN_PRECISION",
    "UNSUPPORTED_DTYPE",
    "UNSUPPORTED_DEVICE_SPELLING",
    "UNKNOWN_COMPONENT",
    "INVALID_CAPABILITY_DECLARATION",
    "NO_COMPATIBLE_DTYPE_KIND",
    "LOSSY_DOWNCAST_REQUIRED",
    "UNSUPPORTED_DEVICE",
    "DEVICE_TRANSFER_NOT_PERMITTED",
    "NAMESPACE_NOT_A_COMPUTE_NAMESPACE",
    "NUMPY_CANNOT_LEAVE_HOST",
    "SILENT_DTYPE_DOWNCAST",
    "DEVICE_NOT_AVAILABLE",
    "DEVICE_ORDINAL_NOT_AVAILABLE",
)


def refusal(
    *,
    code: str,
    component: str,
    message: str,
    requested: Any = None,
    supported: Iterable[Any] | None = None,
    evidence: str | None = None,
    remedy: str | None = None,
) -> ValueError:
    """Build the one refusal shape this package raises.

    Returned rather than raised so the call site reads `raise refusal(...)` and
    the traceback starts where the decision was made. `code` is attached to the
    instance, so a caller can branch on the failure without parsing prose.
    """
    if code not in REFUSAL_CODES:
        raise ValueError(f"{code!r} is not a declared refusal code: {list(REFUSAL_CODES)}")
    detail = f"[{code}] {component}: {message}"
    if supported is not None:
        detail += f" Supported: {sorted(str(item) for item in supported)}."
    if evidence:
        detail += f" Evidence: {evidence}"
    if remedy:
        detail += f" Remedy: {remedy}"
    error = ValueError(detail)
    error.code = code  # type: ignore[attr-defined]
    error.requested = None if requested is None else str(requested)  # type: ignore[attr-defined]
    return error


# ---------------------------------------------------------------------------
# 1. Precision policy and observed dtype
# ---------------------------------------------------------------------------


class Precision(StrEnum):
    """A requested execution/accuracy family. Deliberately not a dtype."""

    FP16 = "fp16"
    FP32 = "fp32"
    FP64 = "fp64"

    @property
    def bits(self) -> int:
        return _PRECISION_BITS[self]

    @property
    def real_dtype(self) -> DType:
        return _PRECISION_REAL[self]

    @property
    def complex_dtype(self) -> DType | None:
        """The complex dtype of this family, or `None` where none exists.

        FP16 returns `None` on purpose. NumPy, JAX and Torch all lack a
        first-class `complex32`, and inventing one so the table looks
        symmetrical is exactly the "native support that is only an implicit
        cast" this module exists to prevent.
        """
        return _PRECISION_COMPLEX[self]

    @classmethod
    def parse(cls, value: Precision | str) -> Precision:
        """Accept a precision name or a dtype spelling and return the family."""
        if isinstance(value, Precision):
            return value
        text = str(value).strip().lower()
        if text in _PRECISION_ALIASES:
            return _PRECISION_ALIASES[text]
        try:
            return DType(text).precision
        except ValueError:
            pass
        raise refusal(
            code="UNKNOWN_PRECISION",
            component="numerics",
            message=f"{value!r} is neither a precision family nor a known dtype.",
            requested=value,
            supported=sorted(_PRECISION_ALIASES) + [str(d) for d in DType],
        )


class DType(StrEnum):
    """An observed array storage dtype. Read from data, never assumed."""

    FLOAT16 = "float16"
    FLOAT32 = "float32"
    FLOAT64 = "float64"
    COMPLEX64 = "complex64"
    COMPLEX128 = "complex128"

    @property
    def is_complex(self) -> bool:
        return self in (DType.COMPLEX64, DType.COMPLEX128)

    @property
    def precision(self) -> Precision:
        """The accuracy family this dtype belongs to.

        `complex64` is FP32: it stores two float32 components, so its accuracy
        is float32 accuracy. Calling it FP64 because it occupies 64 bits is the
        single most common way this gets stated wrongly.
        """
        return _DTYPE_PRECISION[self]

    @property
    def component_bits(self) -> int:
        """Bits per real component -- the number that governs accuracy."""
        return self.precision.bits

    @classmethod
    def parse(cls, value: DType | str | Any) -> DType:
        if isinstance(value, DType):
            return value
        text = str(getattr(value, "name", value)).strip().lower().removeprefix("torch.")
        text = _DTYPE_ALIASES.get(text, text)
        try:
            return DType(text)
        except ValueError as exc:
            raise refusal(
                code="UNSUPPORTED_DTYPE",
                component="numerics",
                message=(
                    f"dtype {value!r} is outside the project vocabulary. Integer, "
                    "boolean and extended-precision arrays are not scientific field "
                    "data and are not bridged."
                ),
                requested=value,
                supported=[str(d) for d in DType],
            ) from exc


_PRECISION_BITS: dict[Precision, int] = {
    Precision.FP16: 16,
    Precision.FP32: 32,
    Precision.FP64: 64,
}
_PRECISION_REAL: dict[Precision, DType] = {
    Precision.FP16: DType.FLOAT16,
    Precision.FP32: DType.FLOAT32,
    Precision.FP64: DType.FLOAT64,
}
_PRECISION_COMPLEX: dict[Precision, DType | None] = {
    Precision.FP16: None,
    Precision.FP32: DType.COMPLEX64,
    Precision.FP64: DType.COMPLEX128,
}
_DTYPE_PRECISION: dict[DType, Precision] = {
    DType.FLOAT16: Precision.FP16,
    DType.FLOAT32: Precision.FP32,
    DType.COMPLEX64: Precision.FP32,
    DType.FLOAT64: Precision.FP64,
    DType.COMPLEX128: Precision.FP64,
}
_PRECISION_ALIASES: dict[str, Precision] = {
    "fp16": Precision.FP16, "half": Precision.FP16,
    "fp32": Precision.FP32, "single": Precision.FP32,
    "fp64": Precision.FP64, "double": Precision.FP64,
}
_DTYPE_ALIASES: dict[str, str] = {
    "float": "float64", "double": "float64", "single": "float32",
    "half": "float16", "complex": "complex128",
}


#: The floor on the dtype a phase accumulation may compute in.
#:
#: `k * OPL` over an optical path of 1e4 waves needs float32 headroom whatever
#: arrives: at float16 the mantissa runs out inside the product itself, so the
#: accumulated phase is noise before any coupler sees it. Declaring the floor
#: here is what keeps "accepts float16" from being read as "computes in float16".
PHASE_ACCUMULATION_FLOOR: Precision = Precision.FP32


def compute_dtype(dtype: DType, floor: Precision = PHASE_ACCUMULATION_FLOOR) -> DType:
    """The dtype an implementation actually computes in for this input.

    Never below `floor`, and never a change of kind: a real input computes in a
    real dtype and a complex input in a complex one. When the result exceeds the
    input, the input precision was **not** natively executed -- `float16 in ->
    float32 compute` is a promotion, and the caller must report it as one rather
    than as float16 support.
    """
    precision = dtype.precision if dtype.precision.bits >= floor.bits else floor
    if not dtype.is_complex:
        return precision.real_dtype
    # FP16 has no complex counterpart in any of the three namespaces, so a
    # complex quantity at an FP16 floor resolves to the smallest that exists.
    complex_dtype = precision.complex_dtype
    return DType.COMPLEX64 if complex_dtype is None else complex_dtype


# ---------------------------------------------------------------------------
# 2. Device and namespace
# ---------------------------------------------------------------------------


class DeviceKind(StrEnum):
    CPU = "cpu"
    CUDA = "cuda"


class ArrayNamespace(StrEnum):
    """Which array ecosystem owns a buffer. Orthogonal to `DeviceKind`."""

    NUMPY = "numpy"
    JAX = "jax"
    TORCH = "torch"

    @property
    def can_leave_host(self) -> bool:
        return self is not ArrayNamespace.NUMPY

    @property
    def is_differentiable(self) -> bool:
        """Whether the namespace can carry an autograd graph at all."""
        return self is not ArrayNamespace.NUMPY


@dataclass(frozen=True)
class DevicePlacement:
    """A concrete device, with the ordinal a coarse cpu/gpu flag drops.

    Rule 1 -- kind and index are one invariant, not two fields: an index is
    meaningless without a kind, and a host index is a contradiction rather than
    an ordinal, so `__post_init__` refuses it here instead of letting `cpu:3`
    travel to whatever finally tries to honour it.

    `index` is `None` for "this kind of device, ordinal unspecified" -- what a
    *request* usually means -- and an integer for an *observation*, where the
    ordinal is a fact about where an array actually landed.
    """

    kind: DeviceKind = DeviceKind.CPU
    index: int | None = None

    def __post_init__(self) -> None:
        if self.kind is DeviceKind.CPU and self.index is not None:
            raise refusal(
                code="UNSUPPORTED_DEVICE_SPELLING",
                component="numerics",
                message=(
                    f"a host placement has no ordinal, but index={self.index} was given. "
                    "'cpu' is the whole address."
                ),
                requested=f"cpu:{self.index}",
            )
        if self.index is not None and self.index < 0:
            raise refusal(
                code="UNSUPPORTED_DEVICE_SPELLING",
                component="numerics",
                message=f"device ordinal {self.index} is negative.",
                requested=self.index,
            )

    def __str__(self) -> str:
        if self.index is None:
            return str(self.kind.value)
        return f"{self.kind.value}:{self.index}"

    @property
    def is_host(self) -> bool:
        return self.kind is DeviceKind.CPU

    def same_kind(self, other: DevicePlacement) -> bool:
        return self.kind is other.kind

    @classmethod
    def parse(cls, value: DevicePlacement | str | None) -> DevicePlacement:
        if value is None:
            return cls(DeviceKind.CPU)
        if isinstance(value, DevicePlacement):
            return value
        text = str(value).strip().lower()
        head, _, tail = text.partition(":")
        index = int(tail) if tail.isdigit() else None
        if head in ("cpu", "host"):
            # The index is forwarded rather than dropped, so 'cpu:0' meets the
            # same refusal as `DevicePlacement(CPU, 0)` -- one invariant, one place.
            return cls(DeviceKind.CPU, index)
        if head in ("cuda", "gpu"):
            return cls(DeviceKind.CUDA, index)
        raise refusal(
            code="UNSUPPORTED_DEVICE_SPELLING",
            component="numerics",
            message=(
                f"device {value!r} is not 'cpu', 'cuda'/'gpu' or 'cuda:<n>'. TPU and "
                "other accelerators are outside the validated set for this project."
            ),
            requested=value,
            supported=["cpu", "cuda", "cuda:<n>"],
        )


@dataclass(frozen=True)
class ArrayState:
    """What an array *actually is*. Every field observed, never requested.

    Rule 1 -- the three fields are one observation of one buffer. Split apart
    they invite the substitution this package exists to prevent: a namespace
    read from the data and a device read from a config value describe no array
    that exists.

    Built by `numerics.arrays.array_state` from a real buffer. Nothing in this
    project constructs one from a request.
    """

    dtype: DType
    device: DevicePlacement
    namespace: ArrayNamespace

    def __post_init__(self) -> None:
        if self.namespace is ArrayNamespace.NUMPY and self.device.kind is DeviceKind.CUDA:
            raise refusal(
                code="NUMPY_CANNOT_LEAVE_HOST",
                component="numerics",
                message="a NumPy buffer cannot be observed on a CUDA device.",
                requested=f"{self.namespace}@{self.device}",
            )

    def __str__(self) -> str:
        return f"{self.namespace.value}:{self.dtype.value}@{self.device}"

    def as_dict(self) -> dict[str, str]:
        return {
            "dtype": self.dtype.value,
            "device": str(self.device),
            "namespace": self.namespace.value,
        }


# ---------------------------------------------------------------------------
# 3. The measured capability declaration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ComponentCapabilities:
    """What one component can actually execute, with the probe that measured it.

    Rule 2 -- a public, versioned data model: this is the declaration every
    descriptor and solver reasons against, and the thing a probe re-run either
    confirms or falsifies.

    Four dtype sets rather than one, because they are different questions and
    collapsing them is how "supports float16" comes to mean "will not crash if
    handed float16":

    `accepted_input_dtypes`
        What may cross the boundary inward. May be wider than native.
    `native_compute_dtypes`
        What the component actually computes in. The honest answer to "does it
        support precision X".
    `output_dtypes`
        What comes back out. Need not match either of the above.
    `lossy_input_dtypes`
        Dtypes the component will physically ingest, but only by throwing
        precision away. Kept **out** of `accepted_input_dtypes` on purpose, so
        `negotiate` refuses them by default and admits them only under an
        explicit `allow_downcast`. Chromatix's `complex128` is the motivating
        case: `ScalarField.__init__` swallows such an array without complaint
        and returns `complex64`.
    """

    component: str
    devices: frozenset[DeviceKind]
    precisions: frozenset[Precision]
    accepted_input_dtypes: frozenset[DType]
    native_compute_dtypes: frozenset[DType]
    output_dtypes: frozenset[DType]
    #: Which namespaces can drive which device. Optiland reaches CUDA only
    #: through its torch backend -- `set_device` raises `BackendCapabilityError`
    #: on the numpy backend -- and declaring that here rather than as an
    #: `if backend == ...` branch is what keeps the two from drifting.
    device_namespaces: Mapping[DeviceKind, frozenset[ArrayNamespace]]
    #: Where the claims above were executed. All three required: a capability with
    #: no probe is an intention, and an intention that reads like a measurement is
    #: the specific failure these fields exist to prevent.
    probe: str
    #: The git tag the `probe` path resolves against. CHE-223 (R03.6) moved it here
    #: from a `PROBE_TAG` module constant that the tests declared a second copy of:
    #: it belongs to the measurement, not to this module, and a record measured at a
    #: different tag is a different measurement. `tests/knowledge/` drives
    #: `git cat-file -e <probe_tag>:<probe>` from this field.
    probe_tag: str
    evidence: str
    lossy_input_dtypes: frozenset[DType] = frozenset()
    minimum_compute_precision: Precision = PHASE_ACCUMULATION_FLOOR
    notes: str = ""

    def __post_init__(self) -> None:
        """Refuse a declaration wider than what it states it measured.

        Every rule here is a way a row could claim more than its probe covers.
        None of them can tell whether the probe was *run* -- that is what the
        `probe` path is for -- but each catches a row that is inconsistent with
        itself, which is how widening actually arrives: one more dtype in one
        set and nowhere else.
        """
        problems: list[str] = []
        if not self.probe.startswith("benchmarks/probes/"):
            problems.append("`probe` must cite a path under benchmarks/probes/")
        if not self.evidence.strip():
            problems.append("`evidence` is empty; a capability with no measurement is a guess")
        if not self.probe_tag.strip():
            problems.append(
                "`probe_tag` is empty; a probe path with no tag to resolve against is not "
                "a citation"
            )
        for name in ("devices", "precisions", "accepted_input_dtypes",
                     "native_compute_dtypes", "output_dtypes"):
            if not getattr(self, name):
                problems.append(f"`{name}` is empty")
        if set(self.device_namespaces) != set(self.devices):
            problems.append(
                "`device_namespaces` must name exactly the declared devices "
                f"({sorted(str(d) for d in self.devices)}); a device with no namespace is a "
                "device with no way to be reached"
            )
        for device, namespaces in self.device_namespaces.items():
            if not namespaces:
                problems.append(f"`device_namespaces[{device}]` is empty")
            elif device is DeviceKind.CUDA and not any(n.can_leave_host for n in namespaces):
                # NumPy cannot hold device memory, so declaring it as the driver
                # for CUDA claims a path that cannot exist. Refusing it here is
                # what lets `_negotiate_namespace` have no unreachable branch.
                problems.append(
                    f"`device_namespaces[{device}]` names only host-only namespaces "
                    f"{sorted(n.value for n in namespaces)}, which cannot hold device memory"
                )
        if not self.native_compute_dtypes <= self.accepted_input_dtypes:
            problems.append(
                "`native_compute_dtypes` is not a subset of `accepted_input_dtypes`: the "
                "component would compute in a dtype it will not accept"
            )
        if self.lossy_input_dtypes & self.accepted_input_dtypes:
            problems.append(
                "`lossy_input_dtypes` overlaps `accepted_input_dtypes`: a dtype cannot be "
                "both losslessly admissible and known-lossy"
            )
        for precision in self.precisions:
            if not any(d.precision is precision for d in self.native_compute_dtypes):
                problems.append(
                    f"precision {precision} is declared but no native compute dtype belongs "
                    "to it, so there is nothing to execute it in"
                )
        floor = self.minimum_compute_precision
        for precision in self.precisions:
            if precision.bits < floor.bits:
                problems.append(
                    f"precision {precision} is below the declared minimum compute precision "
                    f"{floor}"
                )
        if problems:
            raise refusal(
                code="INVALID_CAPABILITY_DECLARATION",
                component=self.component,
                message="; ".join(problems),
                requested=self.component,
            )

    @property
    def namespaces(self) -> frozenset[ArrayNamespace]:
        """Every namespace this component can be driven through, on any device.

        Derived from `device_namespaces` rather than stored beside it: a second
        field would be a second place for the set to be widened.
        """
        return frozenset().union(*self.device_namespaces.values())

    @property
    def cited_evidence(self) -> str:
        """The evidence sentence with the citation that makes it checkable appended.

        What every refusal that cites this declaration should pass as `evidence=`.
        CHE-223 (R03.6) moved the frozen tag out of the prose and into `probe_tag`,
        which removed the duplicate but also silently shortened the refusal
        messages: an FP16 refusal used to read `(pre-rewrite-2026-08-30,
        agent_solver_gpu, ...)` and would otherwise now read `(agent_solver_gpu,
        ...)`. A refused caller is exactly the person who needs to know which
        revision to re-run the probe from, so the citation is reassembled here
        rather than restored to the prose.
        """
        return f"{self.evidence} [{self.probe_tag}:{self.probe}]"

    def namespaces_for(self, device: DeviceKind) -> frozenset[ArrayNamespace]:
        return self.device_namespaces.get(device, frozenset())

    def accepts(self, state: ArrayState) -> bool:
        """Whether data in exactly this state can enter with no conversion."""
        return (
            state.dtype in self.accepted_input_dtypes
            and state.device.kind in self.devices
            and state.namespace in self.namespaces_for(state.device.kind)
        )

    def accepted_dtypes_like(self, reference: DType) -> list[DType]:
        """Accepted dtypes of the same real/complex kind, narrowest first."""
        return sorted(
            (d for d in self.accepted_input_dtypes if d.is_complex == reference.is_complex),
            key=lambda d: d.component_bits,
        )

    def compute_dtype_for(self, dtype: DType) -> DType:
        """The dtype this component computes in for that input. Never below the floor."""
        return compute_dtype(dtype, self.minimum_compute_precision)

    def capability_row(self) -> dict[str, Any]:
        """One row of the executable capability matrix.

        Generated from the declaration rather than written alongside it, so a
        documented claim cannot outlive the capability it describes.
        """
        return {
            "component": self.component,
            "devices": sorted(d.value for d in self.devices),
            "precisions": sorted(p.value for p in self.precisions),
            "namespaces": sorted(n.value for n in self.namespaces),
            "device_namespaces": {
                k.value: sorted(n.value for n in v) for k, v in self.device_namespaces.items()
            },
            "accepted_input_dtypes": sorted(d.value for d in self.accepted_input_dtypes),
            "native_compute_dtypes": sorted(d.value for d in self.native_compute_dtypes),
            "output_dtypes": sorted(d.value for d in self.output_dtypes),
            "lossy_input_dtypes": sorted(d.value for d in self.lossy_input_dtypes),
            "minimum_compute_precision": self.minimum_compute_precision.value,
            "probe": self.probe,
            "probe_tag": self.probe_tag,
            "evidence": self.evidence,
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# The measured rows are NOT here, and CHE-223 (R03.6) is why
# ---------------------------------------------------------------------------
#
# `OPTILAND_CAPABILITIES`, `CHROMATIX_CAPABILITIES`, `COMPONENT_CAPABILITIES`,
# `capabilities_for` and `PROBE_TAG` used to live at this point in the file. R02.1
# put them here as bootstrap anchors -- `negotiate()` needed something real to
# negotiate against and neither `solvers/` nor `operations/` existed -- and this
# module's own comment called them "not their permanent home".
#
# They are now one JSON file per component under `knowledge/capabilities/`, loaded
# by `numerics.knowledge.load_capabilities`. What stays here is the *contract*:
# `ComponentCapabilities` with all ten of its widening refusals, `negotiate()`,
# and every refusal code. This module keeps no copy of the data, so the foundational
# layer no longer knows a backend's name.
#
# **CHE-206's plan is superseded, and not because it was wrong when written.** It
# proposed moving each row into `solvers/<backend>/`, which was the obvious answer
# before either consumer existed. There are now two, on opposite sides of the
# dependency graph: solver runtime reads the row at roughly fifteen refusal sites,
# and backend-free discovery in `operations/` cites the same component id while
# guaranteeing that reading it loads no backend. Solver-local ownership cannot be
# the single source for both -- it would force `operations -> solvers` (which
# `scripts/check_dependencies.py` names as the edge that ends the only property
# `operations/` has), a duplicate copy inside `operations/`, or a backend-adjacent
# import behind every capability query. Shared *data* is one source usable from
# both sides with no edge in either direction.
#
# CHE-206's steps 3 and 4 are kept verbatim: numerics holds only contracts and the
# pure `negotiate()`, and numerics knows no concrete solver names. So are its three
# constraints that had to survive the move -- the probe-citation test, the widening
# refusals, and "reading a capability imports no backend" -- with its refusal count
# corrected from five to nine.
#
# Why there are two rows and not seven is unchanged and still belongs here, because
# it is a rule about what may be declared rather than a fact about either backend:
# the reference implementation's five coupler and operator rows described
# capability nobody measured. A coupler's or operator's capability is set by what
# its shared implementation is written against, so the ticket that measures one
# declares it with its own evidence. A row for unwritten code is the failure this
# module's docstring names.


# ---------------------------------------------------------------------------
# 4. Negotiation -- functions, not a plan object
# ---------------------------------------------------------------------------


def negotiate(
    source: ArrayState,
    target: ComponentCapabilities,
    *,
    target_device: DevicePlacement | None = None,
    allow_downcast: bool = False,
    allow_device_transfer: bool = False,
) -> ArrayState:
    """The state `source` must be converted into to enter `target`, or a refusal.

    Pure and deterministic: takes no arrays, performs no conversion, so it is
    unit-testable against a capability table alone. `numerics.arrays.to_state`
    executes the result; nothing else may convert.

    The two flags are separate because they are separate questions. A precision
    loss is a physics decision (`allow_downcast`); a host/device copy is a cost
    decision (`allow_device_transfer`). Both default to refusing, so an implicit
    host fallback -- the failure mode where a "GPU run" quietly executes on the
    CPU and reports success -- cannot happen without the caller asking for it.

    Returns `source` unchanged when it is already admissible; a returned state
    equal to the source is the signal that no conversion is required.
    """
    device = _negotiate_device(source, target, target_device, allow_device_transfer)
    namespace = _negotiate_namespace(source, target, device)
    dtype = _negotiate_dtype(source, target, allow_downcast)
    return ArrayState(dtype=dtype, device=device, namespace=namespace)


def _negotiate_dtype(
    source: ArrayState, target: ComponentCapabilities, allow_downcast: bool
) -> DType:
    """Preserve, else promote by the minimum, else refuse."""
    candidates = target.accepted_dtypes_like(source.dtype)
    if not candidates:
        raise refusal(
            code="NO_COMPATIBLE_DTYPE_KIND",
            component=target.component,
            message=(
                f"source is {source.dtype} but the target accepts no "
                f"{'complex' if source.dtype.is_complex else 'real'} dtype at all."
            ),
            requested=source.dtype,
            supported=target.accepted_input_dtypes,
            evidence=target.evidence,
        )

    # Preserve a compatible representation: no float32 -> float64 -> float32
    # round trip for convenience.
    if source.dtype in candidates:
        return source.dtype

    # Minimum necessary safe promotion. float16 -> float32, not float16 -> float64.
    wider = [d for d in candidates if d.component_bits > source.dtype.component_bits]
    if wider:
        return wider[0]

    # No silent precision loss.
    narrowest_loss = candidates[-1]
    if not allow_downcast:
        raise refusal(
            code="LOSSY_DOWNCAST_REQUIRED",
            component=target.component,
            message=(
                f"entering this component requires {source.dtype} -> {narrowest_loss}, "
                "which loses precision, and it was not authorized."
            ),
            requested=source.dtype,
            supported=candidates,
            evidence=target.evidence,
            remedy=(
                "Pass allow_downcast=True to accept the loss deliberately; it is then the "
                "caller's job to record it as lossy provenance."
            ),
        )
    return narrowest_loss


def _negotiate_device(
    source: ArrayState,
    target: ComponentCapabilities,
    requested: DevicePlacement | None,
    allow_device_transfer: bool,
) -> DevicePlacement:
    """Preserve residency; never transfer without being told to."""
    if requested is not None:
        if requested.kind not in target.devices:
            raise refusal(
                code="UNSUPPORTED_DEVICE",
                component=target.component,
                message=f"the target cannot execute on the requested device {requested}.",
                requested=requested,
                supported=target.devices,
                evidence=target.evidence,
            )
        if not requested.same_kind(source.device) and not allow_device_transfer:
            raise refusal(
                code="DEVICE_TRANSFER_NOT_PERMITTED",
                component=target.component,
                message=(
                    f"moving the artifact from {source.device} to {requested} was not "
                    "authorized by the caller."
                ),
                requested=requested,
                remedy="Pass allow_device_transfer=True to authorize the copy explicitly.",
            )
        return requested

    if source.device.kind in target.devices:
        return source.device

    if not allow_device_transfer:
        raise refusal(
            code="UNSUPPORTED_DEVICE",
            component=target.component,
            message=(
                f"the source resides on {source.device} and the target executes only on "
                f"{sorted(d.value for d in target.devices)}."
            ),
            requested=source.device,
            supported=target.devices,
            evidence=target.evidence,
            remedy=(
                "Pass allow_device_transfer=True to authorize an explicit copy. There is "
                "deliberately no implicit host fallback."
            ),
        )
    return DevicePlacement(sorted(target.devices, key=lambda d: d.value)[0])


def _negotiate_namespace(
    source: ArrayState, target: ComponentCapabilities, device: DevicePlacement
) -> ArrayNamespace:
    """A namespace change is a boundary operation, never a detail."""
    # Non-empty and, on CUDA, guaranteed to contain a namespace that can hold
    # device memory: `_negotiate_device` has already established that the target
    # executes on this device, and `ComponentCapabilities.__post_init__` refuses a
    # declaration where that would not follow. So there is no "no namespace"
    # branch here, and therefore no refusal code for one.
    admissible = target.namespaces_for(device.kind)
    if source.namespace in admissible:
        return source.namespace
    if device.kind is DeviceKind.CUDA:
        # Never push a device buffer through the host merely to change ecosystem.
        on_device = sorted((n for n in admissible if n.can_leave_host), key=lambda n: n.value)
        return on_device[0]
    for preferred in (ArrayNamespace.NUMPY, ArrayNamespace.JAX, ArrayNamespace.TORCH):
        if preferred in admissible:
            return preferred
    return sorted(admissible, key=lambda n: n.value)[0]
