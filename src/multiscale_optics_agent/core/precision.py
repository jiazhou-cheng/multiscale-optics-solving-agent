"""Canonical precision / dtype / device vocabulary and the cross-model bridge planner (CHE-61).

Before this module the three concepts were one string. ``config["dtype"]``
meant "float64", ``config["device"]`` meant "cpu", and the pair was compared
against a per-adapter constant with ``!=``. That works exactly as long as every
component supports the same combination, which PB4a measured to be false:

* Optiland's ``set_precision`` is ``Literal['float32', 'float64']`` -- there is
  no float16 path at all;
* Chromatix's ``ScalarField.__init__`` casts unconditionally to ``complex64``,
  including when handed a ``complex128`` array with ``jax_enable_x64`` on
  (measured, ``benchmarks/probes/precision/chromatix_capability.py``) -- there is
  no complex128 path at all;
* a JAX array on ``cuda:0`` and a NumPy array on the host are both "GPU-capable
  float32" under the old vocabulary, and only one of them can enter Chromatix
  without a host round trip.

So three separate things are separated here, and a fourth is added:

``Precision``
    An execution *policy* -- FP16/FP32/FP64. An intent and a computation
    family, not a storage format. ``FP32`` legitimately means ``float32`` for
    real data and ``complex64`` for a field.

``DType``
    An *observed* storage property of an actual array. Never inferred from a
    requested precision; always read off the data.

``DevicePlacement``
    Where the array physically is, including the ordinal (``cuda:0``), which
    the coarse :class:`~multiscale_optics_agent.core.specs.Device` enum cannot
    express.

``ArrayNamespace``
    Which array ecosystem owns the buffer. Orthogonal to device: NumPy is
    CPU-only, JAX and Torch are either. PB4a's klujax hazard is exactly a case
    where the namespace is right and the device is silently wrong, so "GPU" is
    not sufficient information to execute anything.

The separation of responsibility this module implements:

    Backend capabilities determine what a package can execute.
    Artifact state describes what the current data actually is.
    Bridge negotiation determines how that artifact may legally enter the
    next component.

A source backend never decides a destination backend's precision, and no
conversion, transfer, or namespace change happens without appearing in a
:class:`BridgePlan`.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from multiscale_optics_agent.core.errors import UnsupportedCapabilityError
from multiscale_optics_agent.core.specs import Device

__all__ = [
    "ArrayNamespace",
    "ArrayState",
    "BridgeError",
    "BridgePlan",
    "BridgePolicy",
    "CapabilityError",
    "ComponentCapabilities",
    "DType",
    "DeviceKind",
    "DevicePlacement",
    "ExecutionRequest",
    "Precision",
    "ResolvedExecution",
    "plan_bridge",
]


# ---------------------------------------------------------------------------
# 1. Precision policy -- an intent, not a storage format
# ---------------------------------------------------------------------------


class Precision(StrEnum):
    """A requested execution/accuracy family.

    Deliberately *not* a dtype. FP32 maps to ``float32`` for real quantities
    and ``complex64`` for complex ones; asking a component for "FP32" says
    nothing about which of those two it will actually allocate.
    """

    FP16 = "fp16"
    FP32 = "fp32"
    FP64 = "fp64"

    @property
    def real_dtype(self) -> DType:
        return _PRECISION_REAL[self]

    @property
    def complex_dtype(self) -> DType | None:
        """The complex dtype of this family, or ``None`` where none genuinely exists.

        ``FP16`` returns ``None`` on purpose. NumPy, JAX and Torch all lack a
        first-class ``complex32``, and inventing one so the table looks
        symmetrical is precisely the "claim of native precision support that is
        only an implicit cast" the contract forbids.
        """
        return _PRECISION_COMPLEX[self]

    @property
    def bits(self) -> int:
        return {Precision.FP16: 16, Precision.FP32: 32, Precision.FP64: 64}[self]

    @classmethod
    def parse(cls, value: Precision | str) -> Precision:
        """Accept a precision name or a dtype name and return the family.

        Accepting a dtype name keeps existing ``config['dtype'] = 'float64'``
        call sites working: they were always expressing a precision policy
        through a dtype spelling, and this is the one place that conflation is
        resolved rather than propagated.
        """
        if isinstance(value, Precision):
            return value
        text = str(value).strip().lower()
        if text in _PRECISION_ALIASES:
            return _PRECISION_ALIASES[text]
        try:
            return DType(text).precision
        except ValueError:
            pass
        raise CapabilityError(
            code="UNKNOWN_PRECISION",
            component="project",
            message=(
                f"{value!r} is not a precision family or a known dtype. "
                f"Expected one of {sorted(_PRECISION_ALIASES)} "
                f"or a dtype in {[str(d) for d in DType]}."
            ),
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
    def is_real(self) -> bool:
        return not self.is_complex

    @property
    def precision(self) -> Precision:
        """The precision family this dtype belongs to.

        ``complex64`` is FP32: it stores two float32 components, so its
        accuracy is float32 accuracy. Calling it FP64 because it occupies 64
        bits is the single most common way this gets stated wrongly.
        """
        return _DTYPE_PRECISION[self]

    @property
    def component_bits(self) -> int:
        """Bits per real component -- the number that governs accuracy."""
        return self.precision.bits

    def to_kind_of(self, other: DType) -> DType | None:
        """This dtype's counterpart in ``other``'s real/complex kind."""
        if other.is_complex:
            return self.precision.complex_dtype
        return self.precision.real_dtype

    @classmethod
    def parse(cls, value: DType | str | Any) -> DType:
        if isinstance(value, DType):
            return value
        text = str(getattr(value, "name", value)).strip().lower()
        text = text.removeprefix("torch.")
        aliases = {"float": "float64", "double": "float64", "single": "float32",
                   "half": "float16", "complex": "complex128"}
        text = aliases.get(text, text)
        try:
            return DType(text)
        except ValueError as exc:
            raise CapabilityError(
                code="UNSUPPORTED_DTYPE",
                component="project",
                message=(
                    f"dtype {value!r} is outside the project vocabulary "
                    f"{[str(d) for d in DType]}. Integer, boolean and "
                    "extended-precision arrays are not scientific field data "
                    "and are not bridged."
                ),
            ) from exc


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


# ---------------------------------------------------------------------------
# 2. Device -- independent of both concepts above
# ---------------------------------------------------------------------------


class DeviceKind(StrEnum):
    CPU = "cpu"
    CUDA = "cuda"

    def to_spec_device_name(self) -> str:
        """The coarse registry spelling. ``cuda`` is ``gpu`` in ``Device``."""
        return Device.CPU.value if self is DeviceKind.CPU else Device.GPU.value


@dataclass(frozen=True, order=False)
class DevicePlacement:
    """A concrete device, with the ordinal the coarse ``Device`` enum drops.

    ``index`` is ``None`` for "this kind of device, ordinal unspecified" -- what
    a *request* usually means -- and an integer for an *observation*, where the
    ordinal is a fact about where an array actually landed.
    """

    kind: DeviceKind = DeviceKind.CPU
    index: int | None = None

    def __str__(self) -> str:
        if self.kind is DeviceKind.CPU or self.index is None:
            return str(self.kind)
        return f"{self.kind}:{self.index}"

    @property
    def is_host(self) -> bool:
        return self.kind is DeviceKind.CPU

    def same_kind(self, other: DevicePlacement) -> bool:
        return self.kind is other.kind

    def to_spec_device(self) -> Device:
        """Project onto the coarse registry/``ArtifactRecord`` enum."""
        return Device.CPU if self.kind is DeviceKind.CPU else Device.GPU

    @classmethod
    def parse(cls, value: DevicePlacement | Device | str | None) -> DevicePlacement:
        if value is None:
            return cls(DeviceKind.CPU)
        if isinstance(value, DevicePlacement):
            return value
        text = str(value).strip().lower()
        head, _, tail = text.partition(":")
        index = int(tail) if tail.isdigit() else None
        if head in ("cpu", "host"):
            return cls(DeviceKind.CPU, index)
        if head in ("cuda", "gpu"):
            return cls(DeviceKind.CUDA, index)
        raise CapabilityError(
            code="UNSUPPORTED_DEVICE",
            component="project",
            message=(
                f"device {value!r} is not one of 'cpu', 'cuda'/'gpu', or "
                "'cuda:<n>'. TPU and other accelerators are outside the "
                "validated set for this project."
            ),
        )


class ArrayNamespace(StrEnum):
    """Which array ecosystem owns a buffer. Orthogonal to :class:`DeviceKind`."""

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


# ---------------------------------------------------------------------------
# 3. Structured failures
# ---------------------------------------------------------------------------


class CapabilityError(UnsupportedCapabilityError):
    """A request asks a component for something it cannot execute.

    Raised eagerly -- before any solver call -- and carries the requested
    value, the supported set, and the evidence behind the claim, so the caller
    is told what to ask for instead of being handed a framework traceback from
    three layers down.
    """

    def __init__(
        self,
        *,
        code: str,
        component: str,
        message: str,
        requested: Any = None,
        supported: Iterable[Any] | None = None,
        evidence: str | None = None,
        remedy: str | None = None,
    ) -> None:
        self.code = code
        self.component = component
        self.requested = requested
        self.supported = sorted(str(item) for item in supported) if supported else None
        self.evidence = evidence
        self.remedy = remedy
        detail = f"[{code}] {component}: {message}"
        if self.supported is not None:
            detail += f" Supported: {self.supported}."
        if evidence:
            detail += f" Evidence: {evidence}."
        if remedy:
            detail += f" Remedy: {remedy}"
        super().__init__(detail)

    def as_diagnostic(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "component": self.component,
            "message": str(self),
            "requested": None if self.requested is None else str(self.requested),
            "supported": self.supported,
            "evidence": self.evidence,
            "remedy": self.remedy,
        }


class BridgeError(CapabilityError):
    """A source artifact cannot legally enter a target under the stated policy."""


# ---------------------------------------------------------------------------
# 4. Observed state, capability declaration, request/resolution
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArrayState:
    """What an array *actually is*. Every field is observed, never requested.

    Constructed by :func:`multiscale_optics_agent.core.arrays.array_state`
    from a real buffer. Nothing in this project builds one from a config
    value: that is the substitution PB4a's klujax hazard punishes.
    """

    dtype: DType
    device: DevicePlacement
    namespace: ArrayNamespace

    def as_dict(self) -> dict[str, str]:
        return {
            "dtype": str(self.dtype),
            "device": str(self.device),
            "namespace": str(self.namespace),
        }

    def __str__(self) -> str:
        return f"{self.namespace}:{self.dtype}@{self.device}"


@dataclass(frozen=True)
class ComponentCapabilities:
    """The one authoritative statement of what a component can execute.

    Four dtype sets, not one, because they are genuinely different questions
    and collapsing them is how "supports float16" comes to mean "will not crash
    if handed float16":

    ``accepted_input_dtypes``
        What may cross the boundary inward. May be wider than native.
    ``native_compute_dtypes``
        What the component actually computes in. This is the honest answer to
        "does it support precision X".
    ``output_dtypes``
        What comes back out. Need not match either of the above.
    ``promotes_input``
        Whether acceptance of a non-native input dtype is achieved by casting
        it. When true, an accepted-but-not-native dtype is a *promotion*, and
        must never be advertised as native support.

    ``lossy_input_dtypes``
        Dtypes the component will physically ingest, but only by throwing
        precision away. Kept out of ``accepted_input_dtypes`` on purpose, so
        the bridge planner refuses them under SAFE and admits them only under
        an explicit downcast policy. Chromatix's ``complex128`` is the
        motivating case: ``ScalarField.__init__`` swallows such an array
        without complaint and returns ``complex64``.
    """

    component: str
    devices: frozenset[DeviceKind]
    precisions: frozenset[Precision]
    accepted_input_dtypes: frozenset[DType]
    native_compute_dtypes: frozenset[DType]
    output_dtypes: frozenset[DType]
    namespaces: frozenset[ArrayNamespace]
    lossy_input_dtypes: frozenset[DType] = frozenset()
    #: Floor on the dtype the implementation computes in, independent of what
    #: it accepts. A coupler accumulating ``k * OPL`` over 1e4 waves needs
    #: float32 headroom even when handed float16, and saying so here is what
    #: keeps "accepts float16" from being read as "computes in float16".
    minimum_compute_precision: Precision = Precision.FP32
    #: Devices that can only be reached through a particular namespace, e.g.
    #: Optiland reaches CUDA only via its torch backend (``set_device`` raises
    #: ``BackendCapabilityError`` on the numpy backend -- measured).
    device_namespaces: Mapping[DeviceKind, frozenset[ArrayNamespace]] = field(
        default_factory=dict
    )
    promotes_input: bool = False
    #: Where the claims above were measured. Required: a capability with no
    #: evidence is an intention.
    evidence: str = ""
    notes: str = ""

    # -- queries ----------------------------------------------------------

    def namespaces_for(self, device: DeviceKind) -> frozenset[ArrayNamespace]:
        return self.device_namespaces.get(device, self.namespaces)

    def accepts(self, state: ArrayState) -> bool:
        """Whether data in exactly this state can enter with no conversion."""
        return (
            state.dtype in self.accepted_input_dtypes
            and state.device.kind in self.devices
            and state.namespace in self.namespaces_for(state.device.kind)
        )

    def native_dtypes_like(self, reference: DType) -> list[DType]:
        """Native compute dtypes of the same real/complex kind as ``reference``."""
        return sorted(
            (d for d in self.native_compute_dtypes if d.is_complex == reference.is_complex),
            key=lambda d: d.component_bits,
        )

    def accepted_dtypes_like(self, reference: DType) -> list[DType]:
        return sorted(
            (d for d in self.accepted_input_dtypes if d.is_complex == reference.is_complex),
            key=lambda d: d.component_bits,
        )

    def compute_dtype_for(self, dtype: DType) -> DType:
        """The dtype the implementation actually computes in for this input.

        Never below :attr:`minimum_compute_precision`. The result may exceed
        the input dtype, and when it does the input precision is *not* natively
        executed -- ``float16 in -> float32 compute -> float32 out`` is a
        promotion, and the capability table reports it as one.
        """
        floor = self.minimum_compute_precision
        precision = dtype.precision if dtype.precision.bits >= floor.bits else floor
        chosen = precision.complex_dtype if dtype.is_complex else precision.real_dtype
        if chosen is None:  # pragma: no cover - FP16 complex has no representation
            chosen = Precision.FP32.complex_dtype
        assert chosen is not None
        return chosen

    def capability_row(self) -> dict[str, Any]:
        """One row of the executable component capability matrix (PB4b section 15)."""
        return {
            "component": self.component,
            "devices": sorted(str(d) for d in self.devices),
            "precisions": sorted(str(p) for p in self.precisions),
            "namespaces": sorted(str(n) for n in self.namespaces),
            "device_namespaces": {
                str(k): sorted(str(n) for n in v) for k, v in self.device_namespaces.items()
            },
            "accepted_input_dtypes": sorted(str(d) for d in self.accepted_input_dtypes),
            "native_compute_dtypes": sorted(str(d) for d in self.native_compute_dtypes),
            "output_dtypes": sorted(str(d) for d in self.output_dtypes),
            "lossy_input_dtypes": sorted(str(d) for d in self.lossy_input_dtypes),
            "minimum_compute_precision": str(self.minimum_compute_precision),
            "promotes_input": self.promotes_input,
            "evidence": self.evidence,
            "notes": self.notes,
        }

    # -- negotiation ------------------------------------------------------

    def resolve(self, request: ExecutionRequest) -> ResolvedExecution:
        """Validate a request against this declaration or fail structurally.

        Never upgrades, downgrades or attempts an unsupported combination: an
        unsupported request is an error at the point of asking, which is the
        only point at which the caller can still do something about it.
        """
        if request.precision not in self.precisions:
            raise CapabilityError(
                code="UNSUPPORTED_PRECISION",
                component=self.component,
                message=f"precision {request.precision} is not executable here.",
                requested=request.precision,
                supported=self.precisions,
                evidence=self.evidence,
                remedy=(
                    "Request a supported precision, or bridge the artifact into "
                    "one with an explicit policy -- this component will not "
                    "silently substitute a different one."
                ),
            )
        device = request.device
        if device.kind not in self.devices:
            raise CapabilityError(
                code="UNSUPPORTED_DEVICE",
                component=self.component,
                message=f"device {device} is not executable here.",
                requested=device,
                supported=self.devices,
                evidence=self.evidence,
            )

        namespaces = self.namespaces_for(device.kind)
        if request.namespace is not None:
            if request.namespace not in namespaces:
                raise CapabilityError(
                    code="UNSUPPORTED_NAMESPACE_FOR_DEVICE",
                    component=self.component,
                    message=(
                        f"array namespace {request.namespace} cannot drive "
                        f"{device} for this component."
                    ),
                    requested=request.namespace,
                    supported=namespaces,
                    evidence=self.evidence,
                )
            namespace = request.namespace
        elif len(namespaces) == 1:
            namespace = next(iter(namespaces))
        else:
            # Deterministic and stated rather than "whatever iterates first":
            # prefer the namespace that can own the requested device.
            candidates = sorted(namespaces, key=lambda ns: (not ns.can_leave_host, str(ns)))
            on_device = device.kind is DeviceKind.CUDA
            namespace = candidates[0] if on_device else _prefer_host(candidates)

        real = request.precision.real_dtype
        complex_ = request.precision.complex_dtype
        compute = [d for d in (real, complex_) if d is not None and d in self.native_compute_dtypes]
        if not compute:
            raise CapabilityError(
                code="NO_NATIVE_COMPUTE_DTYPE",
                component=self.component,
                message=(
                    f"precision {request.precision} has no native compute dtype here; "
                    "the component computes in "
                    f"{sorted(str(d) for d in self.native_compute_dtypes)}."
                ),
                requested=request.precision,
                supported=self.native_compute_dtypes,
                evidence=self.evidence,
            )

        return ResolvedExecution(
            component=self.component,
            precision=request.precision,
            compute_dtypes=frozenset(compute),
            device=device,
            namespace=namespace,
            bridge_policy=request.bridge_policy,
            evidence=self.evidence,
        )


def _prefer_host(candidates: list[ArrayNamespace]) -> ArrayNamespace:
    for namespace in (ArrayNamespace.NUMPY, ArrayNamespace.JAX, ArrayNamespace.TORCH):
        if namespace in candidates:
            return namespace
    return candidates[0]


@dataclass(frozen=True)
class ExecutionRequest:
    """User intent. Says nothing about what any component can do."""

    component: str
    precision: Precision = Precision.FP64
    device: DevicePlacement = field(default_factory=DevicePlacement)
    namespace: ArrayNamespace | None = None
    bridge_policy: BridgePolicy = None  # type: ignore[assignment]  # set in __post_init__

    def __post_init__(self) -> None:
        if self.bridge_policy is None:
            object.__setattr__(self, "bridge_policy", BridgePolicy.SAFE)

    @classmethod
    def from_config(
        cls,
        component: str,
        config: Mapping[str, Any],
        *,
        default_precision: Precision = Precision.FP64,
    ) -> ExecutionRequest:
        """Build from the existing ``config['device'] / config['dtype']`` surface.

        Backwards compatible on purpose: every current call site spells its
        precision as a dtype, and those spellings keep working while meaning
        what they always meant.

        ``default_precision`` is per-component and not optional in practice,
        because the components do not share a default: Optiland's is FP64 and
        Chromatix's is FP32 (its only precision). A single global default would
        make every existing Chromatix request look like an FP64 request and be
        refused.
        """
        precision_value = config.get("precision", config.get("dtype"))
        precision = (
            default_precision if precision_value is None else Precision.parse(precision_value)
        )
        policy_value = config.get("bridge_policy")
        policy = BridgePolicy.SAFE if policy_value is None else BridgePolicy(str(policy_value))
        namespace_value = config.get("array_namespace")
        return cls(
            component=component,
            precision=precision,
            device=DevicePlacement.parse(config.get("device")),
            namespace=None if namespace_value is None else ArrayNamespace(str(namespace_value)),
            bridge_policy=policy,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "component": self.component,
            "precision": str(self.precision),
            "device": str(self.device),
            "namespace": None if self.namespace is None else str(self.namespace),
            "bridge_policy": str(self.bridge_policy),
        }


@dataclass(frozen=True)
class ResolvedExecution:
    """What capability negotiation selected. Still not an observation."""

    component: str
    precision: Precision
    compute_dtypes: frozenset[DType]
    device: DevicePlacement
    namespace: ArrayNamespace
    bridge_policy: BridgePolicy
    evidence: str = ""

    @property
    def real_dtype(self) -> DType:
        return self.precision.real_dtype

    @property
    def complex_dtype(self) -> DType | None:
        return self.precision.complex_dtype

    def as_dict(self) -> dict[str, Any]:
        return {
            "component": self.component,
            "precision": str(self.precision),
            "compute_dtypes": sorted(str(d) for d in self.compute_dtypes),
            "device": str(self.device),
            "namespace": str(self.namespace),
            "bridge_policy": str(self.bridge_policy),
            "evidence": self.evidence,
        }


# ---------------------------------------------------------------------------
# 5. Bridge policy and planner
# ---------------------------------------------------------------------------


class BridgePolicy(StrEnum):
    """How much conversion a cross-model handoff may perform.

    The default is :attr:`SAFE`: widening is allowed because it cannot lose
    information, and narrowing is not, because it silently can. ``float64 ->
    float32`` on an OPL that spans 1e4 waves is not a rounding difference, it
    is a different wavefront.
    """

    #: Nothing implicit at all. Representation must already be admissible.
    STRICT = "strict"
    #: Non-lossy widening only. The default.
    SAFE = "safe"
    #: Narrowing permitted where the destination requires it, recorded as lossy.
    ALLOW_DOWNCAST = "allow_downcast"


@dataclass(frozen=True)
class BridgePlan:
    """A deterministic, inspectable description of one boundary crossing.

    Produced before anything executes and carried into provenance afterwards,
    so "was a conversion required, was it lossy, and why was it allowed" is
    answerable from the run record rather than from re-reading the code.
    """

    source: ArrayState
    target_dtype: DType
    target_device: DevicePlacement
    target_namespace: ArrayNamespace
    policy: BridgePolicy
    #: Dtype the coupler/backend computes in. May exceed both endpoints when
    #: the implementation needs headroom -- which is NOT native support for the
    #: input dtype, and is reported separately for exactly that reason.
    compute_dtype: DType | None = None
    dtype_conversion: bool = False
    promotion: bool = False
    downcast: bool = False
    lossy: bool = False
    device_transfer: bool = False
    host_transfer: bool = False
    namespace_conversion: bool = False
    graph_break: bool = False
    reason: str = "no conversion required"
    effects: tuple[str, ...] = ()

    @property
    def is_identity(self) -> bool:
        return not (self.dtype_conversion or self.device_transfer or self.namespace_conversion)

    @property
    def target(self) -> ArrayState:
        return ArrayState(self.target_dtype, self.target_device, self.target_namespace)

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source.as_dict(),
            "target": self.target.as_dict(),
            "policy": str(self.policy),
            "compute_dtype": None if self.compute_dtype is None else str(self.compute_dtype),
            "dtype_conversion": self.dtype_conversion,
            "promotion": self.promotion,
            "downcast": self.downcast,
            "lossy": self.lossy,
            "device_transfer": self.device_transfer,
            "host_transfer": self.host_transfer,
            "namespace_conversion": self.namespace_conversion,
            "graph_break": self.graph_break,
            "reason": self.reason,
            "effects": list(self.effects),
        }


def plan_bridge(
    source: ArrayState,
    target: ComponentCapabilities,
    *,
    policy: BridgePolicy = BridgePolicy.SAFE,
    allow_device_transfer: bool = False,
    compute_dtype: DType | None = None,
    target_device: DevicePlacement | None = None,
) -> BridgePlan:
    """Decide how ``source`` may legally enter ``target``. Pure and deterministic.

    Takes no arrays and performs no conversion, so it is unit-testable against
    a capability table alone -- which is the point: policy resolution must not
    be entangled with optical formulas.

    ``allow_device_transfer`` is separate from ``policy`` because a CPU<->GPU
    copy is not a precision question. Leaving it ``False`` (the default) turns
    "target cannot reach this device" into a structured failure instead of the
    implicit host fallback that PB4 removed and PB4a proved can happen without
    anyone noticing.
    """
    effects: list[str] = []
    reasons: list[str] = []

    # --- dtype -----------------------------------------------------------
    target_dtype, dtype_notes = _negotiate_dtype(source, target, policy)
    reasons.extend(dtype_notes)
    dtype_changed = target_dtype is not source.dtype
    promotion = dtype_changed and target_dtype.component_bits > source.dtype.component_bits
    downcast = dtype_changed and target_dtype.component_bits < source.dtype.component_bits
    if dtype_changed:
        effects.append(f"dtype {source.dtype} -> {target_dtype}")

    # --- device ----------------------------------------------------------
    device = _negotiate_device(source, target, policy, allow_device_transfer, target_device)
    device_transfer = not device.same_kind(source.device)
    if device_transfer:
        effects.append(f"device transfer {source.device} -> {device}")
        reasons.append(
            f"target does not execute on {source.device.kind}; transfer explicitly permitted"
        )

    # --- namespace -------------------------------------------------------
    namespace = _negotiate_namespace(source, target, policy, device)
    namespace_conversion = namespace is not source.namespace
    host_transfer = device_transfer and (
        source.device.kind is DeviceKind.CUDA or device.kind is DeviceKind.CUDA
    )
    if namespace_conversion:
        # NumPy cannot hold device memory, so any conversion into or out of it
        # from an accelerator buffer is a host copy whether or not the *device
        # kind* changed. That case is invisible if only the device is checked.
        if namespace is ArrayNamespace.NUMPY and source.device.kind is DeviceKind.CUDA:
            host_transfer = True
        if source.namespace is ArrayNamespace.NUMPY and device.kind is DeviceKind.CUDA:
            host_transfer = True
        effects.append(f"namespace {source.namespace} -> {namespace}")
        reasons.append(f"target owns {namespace} buffers, source is {source.namespace}")
    graph_break = namespace_conversion and (
        source.namespace.is_differentiable and namespace is ArrayNamespace.NUMPY
    )
    if graph_break:
        effects.append("autograd graph break (conversion to NumPy detaches)")
    if host_transfer:
        effects.append("host transfer (device synchronization + copy)")

    if not effects:
        reasons = ["source representation is natively admissible; preserved unchanged"]

    return BridgePlan(
        source=source,
        target_dtype=target_dtype,
        target_device=device,
        target_namespace=namespace,
        policy=policy,
        compute_dtype=compute_dtype,
        dtype_conversion=dtype_changed,
        promotion=promotion,
        downcast=downcast,
        lossy=downcast,
        device_transfer=device_transfer,
        host_transfer=host_transfer,
        namespace_conversion=namespace_conversion,
        graph_break=graph_break,
        reason="; ".join(reasons),
        effects=tuple(effects),
    )


def _negotiate_dtype(
    source: ArrayState, target: ComponentCapabilities, policy: BridgePolicy
) -> tuple[DType, list[str]]:
    """Rules A/B/C of the bridge contract, in that order."""
    candidates = target.accepted_dtypes_like(source.dtype)
    if not candidates:
        raise BridgeError(
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

    # Rule A -- preserve a compatible representation. No float32 -> float64 ->
    # float32 round trip for convenience.
    if source.dtype in candidates:
        return source.dtype, []

    if policy is BridgePolicy.STRICT:
        raise BridgeError(
            code="STRICT_REPRESENTATION_MISMATCH",
            component=target.component,
            message=(
                f"STRICT forbids any implicit conversion, and {source.dtype} is "
                "not directly admissible."
            ),
            requested=source.dtype,
            supported=candidates,
            evidence=target.evidence,
            remedy=(
                "Produce the artifact in an admissible dtype, or opt into "
                f"{BridgePolicy.SAFE} / {BridgePolicy.ALLOW_DOWNCAST} deliberately."
            ),
        )

    # Rule B -- minimum necessary safe promotion. float16 -> float32, not
    # float16 -> float64.
    wider = [d for d in candidates if d.component_bits > source.dtype.component_bits]
    if wider:
        chosen = wider[0]
        return chosen, [
            f"{source.dtype} is below the target's minimum; widened to the "
            f"smallest admissible {chosen} (lossless)"
        ]

    # Rule C -- no silent precision loss.
    narrower = [d for d in candidates if d.component_bits < source.dtype.component_bits]
    if policy is BridgePolicy.SAFE:
        raise BridgeError(
            code="LOSSY_DOWNCAST_REQUIRED",
            component=target.component,
            message=(
                f"entering this component requires {source.dtype} -> "
                f"{narrower[-1]}, which loses precision. SAFE refuses it."
            ),
            requested=source.dtype,
            supported=candidates,
            evidence=target.evidence,
            remedy=(
                f"Set bridge policy {BridgePolicy.ALLOW_DOWNCAST} to accept the "
                "loss deliberately; it will be recorded as lossy provenance."
            ),
        )
    chosen = narrower[-1]
    return chosen, [
        f"{source.dtype} exceeds every dtype the target accepts; downcast to "
        f"{chosen} under {BridgePolicy.ALLOW_DOWNCAST} (LOSSY)"
    ]


def _negotiate_device(
    source: ArrayState,
    target: ComponentCapabilities,
    policy: BridgePolicy,
    allow_device_transfer: bool,
    requested: DevicePlacement | None,
) -> DevicePlacement:
    """Rule D -- preserve residency; never transfer without saying so."""
    if requested is not None:
        if requested.kind not in target.devices:
            raise BridgeError(
                code="UNSUPPORTED_DEVICE",
                component=target.component,
                message=f"target cannot execute on the requested device {requested}.",
                requested=requested,
                supported=target.devices,
                evidence=target.evidence,
            )
        if not requested.same_kind(source.device) and not allow_device_transfer:
            raise BridgeError(
                code="DEVICE_TRANSFER_NOT_PERMITTED",
                component=target.component,
                message=(
                    f"moving the artifact from {source.device} to {requested} was "
                    "not authorized by the caller."
                ),
                requested=requested,
                evidence=target.evidence,
                remedy="Pass allow_device_transfer=True to plan the copy explicitly.",
            )
        return requested

    if source.device.kind in target.devices:
        return source.device

    if policy is BridgePolicy.STRICT or not allow_device_transfer:
        raise BridgeError(
            code="DEVICE_INCOMPATIBLE",
            component=target.component,
            message=(
                f"source resides on {source.device} and the target executes only "
                f"on {sorted(str(d) for d in target.devices)}."
            ),
            requested=source.device,
            supported=target.devices,
            evidence=target.evidence,
            remedy=(
                "Pass allow_device_transfer=True to plan an explicit copy. There "
                "is deliberately no implicit CPU fallback."
            ),
        )
    return DevicePlacement(sorted(target.devices, key=str)[0])


def _negotiate_namespace(
    source: ArrayState,
    target: ComponentCapabilities,
    policy: BridgePolicy,
    device: DevicePlacement,
) -> ArrayNamespace:
    """Rule E -- a namespace change is a boundary operation, never a detail."""
    admissible = target.namespaces_for(device.kind)
    if source.namespace in admissible:
        return source.namespace
    if policy is BridgePolicy.STRICT:
        raise BridgeError(
            code="STRICT_NAMESPACE_MISMATCH",
            component=target.component,
            message=(
                f"STRICT forbids converting {source.namespace} arrays into "
                f"{sorted(str(n) for n in admissible)}; the conversion copies data "
                "and may break an autograd graph."
            ),
            requested=source.namespace,
            supported=admissible,
            evidence=target.evidence,
        )
    if not admissible:
        raise BridgeError(
            code="NO_ADMISSIBLE_NAMESPACE",
            component=target.component,
            message=f"target declares no array namespace able to drive {device}.",
            requested=source.namespace,
            evidence=target.evidence,
        )
    # Prefer a namespace that can stay on the current device, so a GPU source
    # is not pushed through the host merely to change ecosystem.
    if device.kind is DeviceKind.CUDA:
        on_device = [n for n in admissible if n.can_leave_host]
        if on_device:
            return sorted(on_device, key=str)[0]
    return _prefer_host(sorted(admissible, key=str))
