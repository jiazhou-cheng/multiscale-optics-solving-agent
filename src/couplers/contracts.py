"""Typed boundary artifacts for the bidirectional ray-wave coupler (CHE-23).

Before this module, ``RayBundle``, ``WavefrontSamples``, ``ComplexField`` and
``PSF`` existed only as :class:`ArtifactKind` enum members plus prose in
AGENTS.md. The physics lived in per-adapter ``metadata: dict[str, Any]``, so
nothing checked that a coupler input actually declared a phasor sign or a
reference plane.

Three rules shape everything here, and each exists because of something a prior
milestone measured rather than guessed:

1. **A missing declaration is an error, never a default.** M1's exit report
   states that the conventions it pinned *are* the coupler's contract. A
   contract that silently supplies a default for an undeclared phasor sign is
   not a contract.

2. **An unverified quantity may be carried, but never reinterpreted.** Optiland
   emits ``opd_native`` whose sign and reference plane M1 recorded as
   ``unverified``, and an ``intensity`` explicitly marked as not being a complex
   amplitude. Both are preserved here as-is, and both are refused at the point
   where a coupler would need to read them *as* a phase or *as* an amplitude.

3. **Adapter output is not changed to suit the contract.** These types are
   built from, and written back to, exactly the ``ArtifactRecord`` + metadata
   form the Optiland and Chromatix adapters already emit.

The failure codes here are the ones named in the coupler failure guides under
``knowledge/couplers/``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np

from core.arrays import (
    array_state,
    device_of,
    dtype_of,
    namespace_of,
    numpy_dtype,
    to_host_numpy,
    xp_for,
)
from core.artifacts import ArtifactRecord
from core.precision import (
    ArrayNamespace,
    ArrayState,
    DevicePlacement,
    DType,
)
from core.specs import ArtifactKind, Framework

__all__ = [
    "AXIS_ORDER",
    "ORIGIN_RULE",
    "PHASOR",
    "PSF",
    "SPATIAL_FACTOR",
    "ComplexField",
    "ContractCode",
    "ContractError",
    "Frame",
    "RayBundle",
    "ReferencePlane",
    "WavefrontSamples",
]


# --- Frozen project conventions ---------------------------------------------
# Inherited from the M1 baselines. These are string constants rather than
# free-form metadata precisely so that a mismatch is an equality failure with a
# named code, not a silently accepted variant spelling.

PHASOR = "exp(-i omega t)"
SPATIAL_FACTOR = "exp(+i k z)"
AXIS_ORDER = "(y, x)"
ORIGIN_RULE = "array index n//2 is coordinate zero"
HANDEDNESS = "right-handed"
PROPAGATION_AXIS = "+z"

#: Optiland's OPD sign and reference plane were both recorded unverified by M1.
#: A wrong OPL *reference* is a harmless piston; a wrong OPL *sign* conjugates
#: the wavefront and turns a converging beam into a diverging one. Those two are
#: indistinguishable downstream, so the ambiguity is refused rather than
#: defaulted.
UNVERIFIED = "unverified"

#: Historical float64 direction-norm tolerance. Kept verbatim as the FLOOR so
#: the established CPU/float64 behaviour is bit-for-bit unchanged: it is ~4e6
#: times looser than float64 round-off, a legacy allowance that CHE-61 has no
#: business tightening.
_DIRECTION_NORM_TOLERANCE = 1e-9


def _direction_norm_tolerance(dtype: DType) -> float:
    """Unit-norm tolerance appropriate to the dtype the directions are stored in.

    Holding a float32 bundle to a float64 tolerance is not strictness, it is a
    category error: casting an exactly normalized float64 direction to float32
    already perturbs ``|d|`` by about one float32 epsilon (1.2e-7) before the
    norm is even computed, and computing it adds a few more. ``64 * eps`` is
    that round-off with an order of magnitude of headroom, derived rather than
    picked, and it reduces to the historical constant for float64.
    """
    eps = float(np.finfo(numpy_dtype(dtype)).eps)
    return max(_DIRECTION_NORM_TOLERANCE, 64.0 * eps)


class ContractCode(StrEnum):
    """Structured failure codes. Used verbatim in coupler diagnostics."""

    MISSING_DECLARATION = "MISSING_DECLARATION"
    UNIT_NOT_SI = "UNIT_NOT_SI"
    PHASOR_MISMATCH = "PHASOR_MISMATCH"
    AXIS_ORDER_MISMATCH = "AXIS_ORDER_MISMATCH"
    FRAME_MISMATCH = "FRAME_MISMATCH"
    SHAPE_MISMATCH = "SHAPE_MISMATCH"
    NON_FINITE = "NON_FINITE"
    NON_UNIT_DIRECTION = "NON_UNIT_DIRECTION"
    EMPTY_ENSEMBLE = "EMPTY_ENSEMBLE"
    AMPLITUDE_IS_A_WEIGHT = "AMPLITUDE_IS_A_WEIGHT"
    OPL_REFERENCE_UNVERIFIED = "OPL_REFERENCE_UNVERIFIED"
    #: The plane an artifact was produced on is not the plane the consumer
    #: declared. Distinct from FRAME_MISMATCH, which is about axes rather than
    #: position: two planes can share a frame and still be metres apart, and a
    #: silently accepted offset is a defocus, not a piston.
    REFERENCE_PLANE_MISMATCH = "REFERENCE_PLANE_MISMATCH"
    PAD_STATE_UNKNOWN = "PAD_STATE_UNKNOWN"
    NEGATIVE_INTENSITY = "NEGATIVE_INTENSITY"
    ARTIFACT_KIND_MISMATCH = "ARTIFACT_KIND_MISMATCH"
    #: The sample pitch an artifact declares is not the pitch its producer
    #: actually output. Added by CHE-36 (M3.7) for the PSF measurement, which must
    #: take its axes from the propagated field's OUTPUT pitch: a measurement that
    #: silently reads the input pupil pitch instead still produces a plausible
    #: intensity map, and every angular comparison drawn on it is rescaled by a
    #: constant nobody can see. Distinct from SHAPE_MISMATCH, which is about array
    #: extent rather than the physical size of a sample.
    SAMPLE_PITCH_MISMATCH = "SAMPLE_PITCH_MISMATCH"
    #: The record carries no object-space reference for its optical path, and the
    #: field it was traced at makes that omission a TILT rather than a piston.
    #: Added by CHE-41. Optiland seeds its OPD accumulator on a plane
    #: perpendicular to z, so for an off-axis collimated bundle the accumulated
    #: path is measured from a surface that is not a wavefront; the difference is
    #: `n_object * d0 . r_launch`, linear in the launch coordinate. On axis it is
    #: a constant and cancels in the chief-ray subtraction, which is why the
    #: defect survived CHE-30/32/33. Distinct from OPL_REFERENCE_UNVERIFIED: the
    #: reference here is known and stated, and the missing quantity is the
    #: object-space information needed to move it onto a wavefront.
    OBJECT_SPACE_REFERENCE_MISSING = "OBJECT_SPACE_REFERENCE_MISSING"
    #: A per-ray quadrature/area weight was requested from a pupil sampling that
    #: is not a recognized hexapolar ring set -- e.g. a ray's normalized pupil
    #: radius does not land within tolerance of ``j / num_rings`` for any integer
    #: ring ``j``. Added by CHE-47 (M3.9R extension). Distinct from
    #: SHAPE_MISMATCH: the arrays agree in length, but the *geometry* the weight
    #: formula assumes is not the geometry the rays were actually sampled on.
    NON_HEXAPOLAR_SAMPLING = "NON_HEXAPOLAR_SAMPLING"
    #: The arrays inside one artifact do not agree on where they live or which
    #: array ecosystem owns them -- e.g. positions on ``cuda:0`` with a NumPy
    #: amplitude. Added by CHE-61 (PB4b). Before this, every array was force-cast
    #: to host NumPy on construction, so the situation could not arise and
    #: neither could GPU residency. Now that an artifact preserves what its data
    #: actually is, a mixed artifact is refused rather than being silently
    #: unified by the first operation that touches both.
    REPRESENTATION_INCONSISTENT = "REPRESENTATION_INCONSISTENT"


class ContractError(ValueError):
    """A boundary declaration is missing, inconsistent, or unusable.

    Carries a machine-readable code so a coupler can return a structured
    diagnostic instead of an invented result.
    """

    def __init__(
        self,
        code: ContractCode,
        message: str,
        *,
        declaration: str | None = None,
        artifact_id: str | None = None,
        remedy: str | None = None,
    ) -> None:
        self.code = code
        self.declaration = declaration
        self.artifact_id = artifact_id
        self.remedy = remedy
        detail = f"[{code}] {message}"
        if declaration:
            detail += f" (declaration: {declaration!r})"
        if artifact_id:
            detail += f" (artifact: {artifact_id!r})"
        if remedy:
            detail += f" Remedy: {remedy}"
        super().__init__(detail)

    def as_diagnostic(self) -> dict[str, Any]:
        return {
            "code": str(self.code),
            "message": str(self),
            "declaration": self.declaration,
            "artifact_id": self.artifact_id,
            "remedy": self.remedy,
        }


def _require(mapping: dict[str, Any], key: str, *, artifact_id: str | None, what: str) -> Any:
    """Fetch a declaration or fail. Never substitutes a default."""
    if key not in mapping or mapping[key] is None:
        raise ContractError(
            ContractCode.MISSING_DECLARATION,
            f"{what} is not declared",
            declaration=key,
            artifact_id=artifact_id,
            remedy=(
                "The producing adapter must declare this. A coupler may not "
                "assume it, because an assumed convention is indistinguishable "
                "from a verified one once it is downstream."
            ),
        )
    return mapping[key]


def _check_finite(array: Any, name: str) -> None:
    xp = xp_for(namespace_of(array))
    if not bool(xp.all(xp.isfinite(array))):
        raise ContractError(
            ContractCode.NON_FINITE,
            f"{name} contains non-finite values",
            declaration=name,
        )


# --- Representation-preserving array intake ---------------------------------
# Before CHE-61 every field here ran through ``np.asarray(value,
# dtype=np.float64)`` (or ``complex128``). That single line did four things at
# once: it moved data to the host, changed its dtype, changed its array
# ecosystem, and broke any autograd graph attached to it -- so a float32 GPU
# artifact could not exist at all, whatever the producing solver supported.
#
# The replacement keeps the *default* exactly where it was. A Python list or
# scalar still becomes host float64/complex128, because it has no dtype, device
# or namespace of its own and the historical default is the right one. An input
# that IS an array is left in the representation it arrived in.


def _default_dtype(complex_: bool) -> DType:
    """What an input carrying no representation of its own becomes."""
    return DType.COMPLEX128 if complex_ else DType.FLOAT64


def _intake(value: Any, *, name: str, complex_: bool) -> Any:
    """Adopt ``value`` as artifact data, preserving what it already is.

    ``complex_`` says which kind the *field* is, not what the caller passed. A
    real array handed to a complex field is widened to the complex dtype of the
    same precision -- ``float32 -> complex64``, not ``-> complex128`` -- because
    a phase-free amplitude is still an amplitude, while silently promoting its
    precision would be a conversion nobody asked for.
    """
    if not hasattr(value, "dtype"):
        # list / tuple / Python scalar: no representation to preserve.
        return np.asarray(value, dtype=numpy_dtype(_default_dtype(complex_)))

    dtype = dtype_of(value)
    namespace = namespace_of(value)
    if namespace is ArrayNamespace.TORCH:
        raise ContractError(
            ContractCode.REPRESENTATION_INCONSISTENT,
            (
                f"{name} is a torch tensor. Boundary artifacts execute in the "
                "NumPy/JAX compute namespaces; a torch array must cross an "
                "explicit bridge (core.arrays.to_namespace) so the conversion, "
                "and any autograd graph break it causes, is recorded."
            ),
            declaration=name,
            remedy=(
                "Plan the handoff with core.precision.plan_bridge and apply it "
                "with core.arrays.apply_bridge before building the artifact."
            ),
        )

    if complex_ and dtype.is_real:
        target = dtype.precision.complex_dtype
        if target is None:
            # float16 has no complex counterpart anywhere in this stack; widen
            # to the smallest one that exists rather than inventing complex32.
            target = DType.COMPLEX64
        return xp_for(namespace).asarray(value, dtype=numpy_dtype(target))
    if not complex_ and dtype.is_complex:
        raise ContractError(
            ContractCode.MISSING_DECLARATION,
            f"{name} must be real-valued; a complex array here is an amplitude, not a magnitude",
            declaration=name,
        )
    return value


_FRAMEWORK_BY_NAMESPACE = {
    ArrayNamespace.NUMPY: Framework.NUMPY,
    ArrayNamespace.JAX: Framework.JAX,
    ArrayNamespace.TORCH: Framework.PYTORCH,
}


def _framework_for(namespace: ArrayNamespace) -> Framework:
    return _FRAMEWORK_BY_NAMESPACE[namespace]


class _HostView:
    """The explicit execution -> serialization boundary of PB4b section 11.

    ``.npy``/``.npz`` are host formats. A GPU artifact therefore has to be
    copied to the host *to be written*, and the important property is that this
    is the only place it happens: the live artifact stays on its device, the
    copy is made at the moment of writing, and the record says so. A reader can
    then tell a persistence copy apart from a computational fallback, which is
    the distinction that goes missing when ``to_numpy()`` is sprinkled through
    an execution path.
    """

    def __init__(self, artifact: Any, *, reason: str) -> None:
        self._state = artifact.state
        self._reason = reason
        self._copied = False

    def of(self, array: Any) -> np.ndarray:
        if namespace_of(array) is not ArrayNamespace.NUMPY:
            self._copied = True
        return to_host_numpy(array, reason=self._reason)

    def as_metadata(self) -> dict[str, Any]:
        return {
            "boundary": "explicit_persistence",
            "host_copy": self._copied,
            "kind": "serialization" if self._copied else "already_on_host",
            "reason": self._reason,
            "execution_representation_preserved": True,
            "from": self._state.as_dict(),
        }


def _require_same_representation(
    arrays: dict[str, Any], *, reference: str
) -> None:
    """Every array in one artifact must live in the same place, in the same ecosystem.

    Dtype is deliberately NOT unified: a float32 geometry with a complex64
    amplitude is a legitimate FP32 artifact, and forcing a common dtype would
    reintroduce exactly the hidden conversion this class of change removes.
    """
    base = arrays[reference]
    base_device, base_namespace = device_of(base), namespace_of(base)
    for name, array in arrays.items():
        if array is None or name == reference:
            continue
        device, namespace = device_of(array), namespace_of(array)
        if namespace is not base_namespace or device != base_device:
            raise ContractError(
                ContractCode.REPRESENTATION_INCONSISTENT,
                (
                    f"{name} is {namespace}:{device} but {reference} is "
                    f"{base_namespace}:{base_device}. One artifact cannot span two "
                    "devices or two array ecosystems; the first operation to touch "
                    "both would silently move one of them."
                ),
                declaration=name,
                remedy="Bridge the mismatched array explicitly before constructing the artifact.",
            )


@dataclass(frozen=True)
class Frame:
    """The coordinate convention an artifact is expressed in."""

    axis_order: str = AXIS_ORDER
    handedness: str = HANDEDNESS
    origin_rule: str = ORIGIN_RULE
    propagation_axis: str = PROPAGATION_AXIS

    def __post_init__(self) -> None:
        if self.handedness != HANDEDNESS:
            raise ContractError(
                ContractCode.FRAME_MISMATCH,
                f"only a {HANDEDNESS} frame is supported, got {self.handedness!r}",
                declaration="handedness",
            )
        if self.propagation_axis != PROPAGATION_AXIS:
            raise ContractError(
                ContractCode.FRAME_MISMATCH,
                f"propagation must be along {PROPAGATION_AXIS}, got {self.propagation_axis!r}",
                declaration="propagation_axis",
            )

    def require_field_axis_order(self) -> None:
        """Field arrays are ``(y, x)``. A transpose is invisible in any
        rotationally symmetric test case, so it is checked rather than trusted."""
        if self.axis_order != AXIS_ORDER:
            raise ContractError(
                ContractCode.AXIS_ORDER_MISMATCH,
                f"field arrays must be {AXIS_ORDER}, got {self.axis_order!r}",
                declaration="axis_order",
            )

    def as_metadata(self) -> dict[str, str]:
        return {
            "axis_order": self.axis_order,
            "handedness": self.handedness,
            "origin_rule": self.origin_rule,
            "propagation_axis": self.propagation_axis,
        }


@dataclass(frozen=True)
class ReferencePlane:
    """A named plane with an axial coordinate and a unit normal.

    Every artifact crossing a coupler boundary declares one. Without it,
    ``OPL`` has no meaning and the ``<n, d>`` projection factor of main-text
    eq 2 has no normal to project onto.
    """

    name: str
    z_m: float
    normal: tuple[float, float, float] = (0.0, 0.0, 1.0)

    def __post_init__(self) -> None:
        if not self.name:
            raise ContractError(
                ContractCode.MISSING_DECLARATION,
                "reference plane must be named",
                declaration="reference_plane.name",
            )
        if not math.isfinite(self.z_m):
            raise ContractError(
                ContractCode.NON_FINITE,
                "reference plane axial coordinate is not finite",
                declaration="reference_plane.z_m",
            )
        norm = math.sqrt(sum(component * component for component in self.normal))
        if abs(norm - 1.0) > _DIRECTION_NORM_TOLERANCE:
            raise ContractError(
                ContractCode.NON_UNIT_DIRECTION,
                f"reference plane normal must be a unit vector, |n| = {norm!r}",
                declaration="reference_plane.normal",
            )

    def as_metadata(self) -> dict[str, Any]:
        return {"name": self.name, "z_m": self.z_m, "normal": list(self.normal)}


@dataclass(frozen=True)
class RayBundle:
    """Rays as plane wavelets, in SI.

    Two fields are deliberately optional, and their absence is what makes this
    type useful rather than incomplete:

    ``amplitude``
        The complex amplitude ``a`` of main-text eq 2. Optiland supplies only a
        real ``intensity`` weight, explicitly marked as not an amplitude.
        Converting a weight to an amplitude is a modelling decision -- is the
        weight a power, so ``a = sqrt(w)``? a photon count? already an
        amplitude? -- that the caller must declare. See
        :meth:`with_amplitude_from_weight`.

    ``optical_path_length_m``
        The ``OPL`` of main-text eq 2, with a declared reference. Optiland's
        ``opd_native`` does not qualify: M1 recorded its sign and reference as
        unverified. See :meth:`with_declared_optical_path_length`.

    :meth:`require_coherent` is the gate. Carrying an unverified quantity is
    fine; reading it as physics is not.
    """

    positions_m: np.ndarray
    directions: np.ndarray
    wavelength_m: float
    reference_plane: ReferencePlane
    frame: Frame = field(default_factory=Frame)
    amplitude: np.ndarray | None = None
    weight: np.ndarray | None = None
    weight_semantics: str | None = None
    optical_path_length_m: np.ndarray | None = None
    optical_path_length_reference: str | None = None
    phasor: str = PHASOR
    polarization: str = "scalar"
    coherence: str = "fully coherent"
    normalization: str = "none; sum over a given ray ensemble carries no 1/N"
    #: Whether a coherent reconstruction from this bundle must divide by the ray
    #: count. Structured rather than prose because two components have to agree
    #: on it: a bundle sampled from a spectrum is a Monte Carlo estimate and
    #: needs the 1/N of SI eq S5, while a bundle from a physical ray trace is
    #: the ensemble itself and must not be averaged. Getting this wrong scales
    #: the field by the ray count, which is exactly the kind of silent factor
    #: the contract layer exists to prevent.
    reconstruction_normalization: str = "none"
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        positions = _intake(self.positions_m, name="positions_m", complex_=False)
        directions = _intake(self.directions, name="directions", complex_=False)
        object.__setattr__(self, "positions_m", positions)
        object.__setattr__(self, "directions", directions)
        xp = xp_for(namespace_of(positions))

        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ContractError(
                ContractCode.SHAPE_MISMATCH,
                f"positions_m must be (N, 3), got {positions.shape}",
                declaration="positions_m",
            )
        if directions.shape != positions.shape:
            raise ContractError(
                ContractCode.SHAPE_MISMATCH,
                f"directions {directions.shape} must match positions {positions.shape}",
                declaration="directions",
            )
        if positions.shape[0] == 0:
            raise ContractError(
                ContractCode.EMPTY_ENSEMBLE,
                "ray bundle is empty; there is nothing to reconstruct from",
                declaration="positions_m",
            )
        _check_finite(positions, "positions_m")
        _check_finite(directions, "directions")

        norms = xp.linalg.norm(directions, axis=1)
        worst = float(xp.max(xp.abs(norms - 1.0)))
        tolerance = _direction_norm_tolerance(dtype_of(directions))
        if worst > tolerance:
            raise ContractError(
                ContractCode.NON_UNIT_DIRECTION,
                (
                    f"direction vectors must be unit norm; worst deviation {worst:.3e} "
                    f"exceeds {tolerance:.3e} for {dtype_of(directions)}"
                ),
                declaration="directions",
            )

        if not math.isfinite(self.wavelength_m) or self.wavelength_m <= 0.0:
            raise ContractError(
                ContractCode.UNIT_NOT_SI,
                f"wavelength must be a positive value in metres, got {self.wavelength_m!r}",
                declaration="wavelength_m",
            )
        if self.phasor != PHASOR:
            raise ContractError(
                ContractCode.PHASOR_MISMATCH,
                f"phasor must be {PHASOR!r}, got {self.phasor!r}",
                declaration="phasor",
            )

        for name in ("amplitude", "weight", "optical_path_length_m"):
            value = getattr(self, name)
            if value is None:
                continue
            array = _intake(value, name=name, complex_=(name == "amplitude"))
            object.__setattr__(self, name, array)
            if array.shape != (positions.shape[0],):
                raise ContractError(
                    ContractCode.SHAPE_MISMATCH,
                    f"{name} must be ({positions.shape[0]},), got {array.shape}",
                    declaration=name,
                )
            _check_finite(array, name)

        _require_same_representation(
            {
                "positions_m": self.positions_m,
                "directions": self.directions,
                "amplitude": self.amplitude,
                "weight": self.weight,
                "optical_path_length_m": self.optical_path_length_m,
            },
            reference="positions_m",
        )

        if self.optical_path_length_m is not None and not self.optical_path_length_reference:
            raise ContractError(
                ContractCode.MISSING_DECLARATION,
                "an optical path length was supplied without declaring its reference",
                declaration="optical_path_length_reference",
                remedy="State the plane or ray the OPL is measured from.",
            )
        if self.reconstruction_normalization not in {"none", "one_over_n"}:
            raise ContractError(
                ContractCode.MISSING_DECLARATION,
                (
                    "reconstruction_normalization must be 'none' or 'one_over_n', "
                    f"got {self.reconstruction_normalization!r}"
                ),
                declaration="reconstruction_normalization",
            )
        if self.weight is not None and not self.weight_semantics:
            raise ContractError(
                ContractCode.MISSING_DECLARATION,
                "a ray weight was supplied without declaring what it represents",
                declaration="weight_semantics",
                remedy="State whether the weight is a power, a photon count, or an amplitude.",
            )

    @property
    def count(self) -> int:
        return int(self.positions_m.shape[0])

    @property
    def wavenumber(self) -> float:
        """Free-space wavenumber ``k = 2 pi / lambda`` in rad/m."""
        return 2.0 * math.pi / self.wavelength_m

    @property
    def state(self) -> ArrayState:
        """What this bundle's geometry actually is: dtype, device, namespace.

        Read from ``positions_m``, never from a caller-supplied field, so the
        answer cannot contradict the data. ``amplitude`` may legitimately carry
        the complex counterpart of this dtype at the same precision.
        """
        return array_state(self.positions_m)

    @property
    def device(self) -> DevicePlacement:
        return self.state.device

    @property
    def namespace(self) -> ArrayNamespace:
        return self.state.namespace

    @property
    def dtype(self) -> DType:
        return self.state.dtype

    @property
    def xp(self) -> Any:
        """The array module this bundle's data belongs to."""
        return xp_for(self.namespace)

    def require_coherent(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(amplitude, optical_path_length_m)`` or fail structurally.

        Called by ``C_RAY_TO_WAVE`` before it reads either quantity as physics.
        """
        if self.amplitude is None:
            raise ContractError(
                ContractCode.AMPLITUDE_IS_A_WEIGHT,
                (
                    "this bundle carries no complex amplitude"
                    + (
                        f"; it carries a real weight declared as {self.weight_semantics!r}"
                        if self.weight is not None
                        else ""
                    )
                ),
                declaration="amplitude",
                remedy=(
                    "Use with_amplitude_from_weight() and declare the conversion. "
                    "A ray weight is not a complex amplitude, and the coupler "
                    "will not choose the mapping for you."
                ),
            )
        if self.optical_path_length_m is None:
            raise ContractError(
                ContractCode.OPL_REFERENCE_UNVERIFIED,
                "this bundle carries no optical path length with a declared reference",
                declaration="optical_path_length_m",
                remedy=(
                    "Use with_declared_optical_path_length(). Optiland's "
                    "opd_native is not admissible: M1 recorded its sign and "
                    "reference as unverified, and a wrong sign conjugates the "
                    "wavefront."
                ),
            )
        if self.optical_path_length_reference == UNVERIFIED:
            raise ContractError(
                ContractCode.OPL_REFERENCE_UNVERIFIED,
                "optical path length reference is declared 'unverified'",
                declaration="optical_path_length_reference",
                remedy="Characterize the OPL against a known geometry before using it as a phase.",
            )
        return self.amplitude, self.optical_path_length_m

    def with_amplitude_from_weight(
        self, *, mapping: str, amplitude: np.ndarray | None = None
    ) -> RayBundle:
        """Attach a complex amplitude derived from the carried weight.

        ``mapping`` is a free-text declaration of the physical assumption, e.g.
        ``"amplitude = sqrt(weight); weight is a power"``. It is recorded in
        provenance so a reader can see which assumption produced a field.
        """
        if amplitude is None:
            if self.weight is None:
                raise ContractError(
                    ContractCode.MISSING_DECLARATION,
                    "no weight is available to derive an amplitude from",
                    declaration="weight",
                )
            if mapping.startswith("amplitude = sqrt(weight)"):
                xp = self.xp
                if bool(xp.any(self.weight < 0.0)):
                    raise ContractError(
                        ContractCode.NEGATIVE_INTENSITY,
                        "cannot take sqrt of a negative weight",
                        declaration="weight",
                    )
                # The complex counterpart of the weight's OWN precision: a
                # float32 weight yields a complex64 amplitude, not a complex128
                # one. _intake widens real -> complex without touching precision.
                amplitude = xp.sqrt(self.weight)
            else:
                raise ContractError(
                    ContractCode.MISSING_DECLARATION,
                    f"no built-in conversion for mapping {mapping!r}; pass amplitude explicitly",
                    declaration="amplitude",
                )
        return self._replace(
            amplitude=_intake(amplitude, name="amplitude", complex_=True),
            provenance={**self.provenance, "amplitude_mapping": mapping},
        )

    def with_declared_optical_path_length(
        self, optical_path_length_m: np.ndarray, *, reference: str
    ) -> RayBundle:
        if reference == UNVERIFIED or not reference:
            raise ContractError(
                ContractCode.OPL_REFERENCE_UNVERIFIED,
                "an OPL reference must be stated, not declared unverified",
                declaration="optical_path_length_reference",
            )
        return self._replace(
            optical_path_length_m=_intake(
                optical_path_length_m, name="optical_path_length_m", complex_=False
            ),
            optical_path_length_reference=reference,
        )

    def with_provenance(self, **entries: Any) -> RayBundle:
        """Return a copy carrying additional provenance entries.

        Declarations made by a bridge -- which reference plane the OPL was moved
        to, which reference ray its piston was removed against, what the weight
        was assumed to mean -- have to travel with the bundle. Without this they
        would have to be threaded through every call site as loose arguments,
        which is how a declaration and the array it describes drift apart.
        """
        return self._replace(provenance={**self.provenance, **entries})

    def _replace(self, **changes: Any) -> RayBundle:
        current = {
            "positions_m": self.positions_m,
            "directions": self.directions,
            "wavelength_m": self.wavelength_m,
            "reference_plane": self.reference_plane,
            "frame": self.frame,
            "amplitude": self.amplitude,
            "weight": self.weight,
            "weight_semantics": self.weight_semantics,
            "optical_path_length_m": self.optical_path_length_m,
            "optical_path_length_reference": self.optical_path_length_reference,
            "phasor": self.phasor,
            "polarization": self.polarization,
            "coherence": self.coherence,
            "normalization": self.normalization,
            "reconstruction_normalization": self.reconstruction_normalization,
            "provenance": self.provenance,
        }
        current.update(changes)
        return RayBundle(**current)

    # --- ArtifactRecord interoperability ------------------------------------

    @classmethod
    def from_artifact_record(
        cls, record: ArtifactRecord, *, arrays: dict[str, np.ndarray] | None = None
    ) -> RayBundle:
        """Build from the ``rays.npz`` + metadata form the Optiland adapter emits.

        The adapter is not modified. Its unverified declarations are carried
        through as unverified rather than reinterpreted.
        """
        if record.kind is not ArtifactKind.RAY_BUNDLE:
            raise ContractError(
                ContractCode.ARTIFACT_KIND_MISMATCH,
                f"expected {ArtifactKind.RAY_BUNDLE}, got {record.kind}",
                artifact_id=record.id,
            )
        data = arrays if arrays is not None else dict(np.load(record.uri))
        metadata = record.metadata
        conventions = metadata.get("conventions", {})

        if metadata.get("length_unit") != "m":
            raise ContractError(
                ContractCode.UNIT_NOT_SI,
                f"ray positions must be in metres, got {metadata.get('length_unit')!r}",
                declaration="length_unit",
                artifact_id=record.id,
            )
        wavelength_m = float(
            _require(metadata, "wavelength_m", artifact_id=record.id, what="wavelength")
        )
        plane_name = _require(
            conventions, "reference_plane", artifact_id=record.id, what="reference plane"
        )
        plane_z = float(
            _require(
                conventions, "reference_plane_z_m", artifact_id=record.id, what="reference plane z"
            )
        )

        # No .astype(float64): the persisted file already declares a dtype, and
        # widening it back on load would erase the precision the producer chose.
        positions = np.column_stack([data["x_m"], data["y_m"], data["z_m"]])
        directions = np.column_stack([data["L"], data["M"], data["N"]])

        weight = None
        weight_semantics = None
        if "intensity" in data:
            weight = np.asarray(data["intensity"])
            weight_semantics = metadata.get("intensity_is_not_amplitude", "unnamed ray weight")

        # opd_native is carried in provenance, never promoted to an OPL.
        provenance: dict[str, Any] = {
            "source_artifact_id": record.id,
            "source_uri": record.uri,
            "backend": metadata.get("backend"),
        }
        if "opd_native" in data:
            provenance["opd_native"] = np.asarray(data["opd_native"], dtype=np.float64)
            provenance["opd_native_status"] = {
                "reference": conventions.get("opd_reference", UNVERIFIED),
                "sign": conventions.get("opd_sign", UNVERIFIED),
                "note": (
                    "Carried for traceability only. Not usable as an optical "
                    "path length until its sign and reference are characterized."
                ),
            }
        # CHE-41: the object-space term that moves opd_native's reference from the
        # launch PLANE onto a WAVEFRONT of the incoming bundle. Carried, never
        # applied here -- applying it is a declaration, and this classmethod makes
        # none. Absent for a record written before CHE-41 or by a launch geometry
        # the ray adapter declined to characterize; the consumer decides whether
        # that absence is a piston it can ignore or a tilt it must refuse.
        if "object_space_reference_offset_m" in data:
            provenance["object_space_reference_offset_m"] = np.asarray(
                data["object_space_reference_offset_m"], dtype=np.float64
            )
        if isinstance(conventions.get("object_space_reference"), dict):
            provenance["object_space_reference"] = dict(conventions["object_space_reference"])
        # CHE-47: the RAW hexapolar pupil coordinates a per-ray quadrature (area)
        # weight is computed from, carried the same way as the object-space term
        # above -- present when the adapter regenerated an un-vignetted hexapolar
        # fan and matched it row for row against the trace, absent for a record
        # written before CHE-47 or a non-hexapolar sampling. Carried, never turned
        # into a weight or applied here; computing and folding it into the
        # amplitude is a declaration made by the coupler-side handoff, not by this
        # classmethod (which imports no coupler math, only carries adapter data).
        if "pupil_normalized_x" in data:
            provenance["pupil_normalized_x"] = np.asarray(
                data["pupil_normalized_x"], dtype=np.float64
            )
        if "pupil_normalized_y" in data:
            provenance["pupil_normalized_y"] = np.asarray(
                data["pupil_normalized_y"], dtype=np.float64
            )
        if isinstance(conventions.get("quadrature_weight"), dict):
            provenance["quadrature_weight"] = dict(conventions["quadrature_weight"])
        provenance["requested_field"] = {
            "Hx": metadata.get("requested_Hx"),
            "Hy": metadata.get("requested_Hy"),
        }

        return cls(
            positions_m=positions,
            directions=directions,
            wavelength_m=wavelength_m,
            reference_plane=ReferencePlane(name=str(plane_name), z_m=plane_z),
            frame=Frame(axis_order="flat per-ray arrays"),
            weight=weight,
            weight_semantics=weight_semantics,
            polarization=str(conventions.get("polarization", "missing")),
            coherence=str(conventions.get("coherence", "missing")),
            normalization=str(conventions.get("normalization", "unstated")),
            provenance=provenance,
        )

    def to_artifact_record(self, *, artifact_id: str, uri: str | Path) -> ArtifactRecord:
        """Persist the bundle. The host copy taken here is a declared boundary.

        ``.npz`` is a host format, so writing one requires host bytes whatever
        device the bundle lives on. That is a *serialization* requirement, not a
        computational fallback, and the two are distinguished in
        ``metadata['serialization']`` rather than left for a reader to guess.
        Nothing about the live artifact changes: this method returns a record and
        leaves ``self`` on its device.
        """
        path = Path(uri)
        reason = "npz persistence requires host bytes; the live bundle is unchanged"
        host = _HostView(self, reason=reason)
        positions, directions = host.of(self.positions_m), host.of(self.directions)
        arrays: dict[str, np.ndarray] = {
            "x_m": positions[:, 0],
            "y_m": positions[:, 1],
            "z_m": positions[:, 2],
            "L": directions[:, 0],
            "M": directions[:, 1],
            "N": directions[:, 2],
        }
        if self.amplitude is not None:
            arrays["amplitude"] = host.of(self.amplitude)
        if self.weight is not None:
            arrays["intensity"] = host.of(self.weight)
        if self.optical_path_length_m is not None:
            arrays["opl_m"] = host.of(self.optical_path_length_m)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **arrays)

        state = self.state
        return ArtifactRecord(
            id=artifact_id,
            kind=ArtifactKind.RAY_BUNDLE,
            uri=str(path),
            shape=(self.count,),
            dtype=str(state.dtype),
            framework=_framework_for(state.namespace),
            device=state.device.to_spec_device(),
            units="SI: metres, dimensionless unit directions",
            metadata={
                "execution": state.as_dict(),
                "serialization": host.as_metadata(),
                "amplitude_dtype": (
                    None if self.amplitude is None else str(dtype_of(self.amplitude))
                ),
                "length_unit": "m",
                "wavelength_unit": "m",
                "wavelength_m": self.wavelength_m,
                "coordinate_fields": ["x_m", "y_m", "z_m"],
                "direction_fields": ["L", "M", "N"],
                "amplitude_field": "amplitude" if self.amplitude is not None else None,
                "amplitude_is_complex": self.amplitude is not None,
                "intensity_field": "intensity" if self.weight is not None else None,
                "intensity_is_not_amplitude": self.weight_semantics,
                "optical_path_length_field": (
                    "opl_m" if self.optical_path_length_m is not None else None
                ),
                "optical_path_length_reference": self.optical_path_length_reference,
                "phasor": self.phasor,
                "spatial_factor": SPATIAL_FACTOR,
                "polarization": self.polarization,
                "coherence": self.coherence,
                "normalization": self.normalization,
                "conventions": {
                    **self.frame.as_metadata(),
                    "reference_plane": self.reference_plane.name,
                    "reference_plane_z_m": self.reference_plane.z_m,
                    "reference_plane_normal": list(self.reference_plane.normal),
                },
            },
        )


@dataclass(frozen=True)
class WavefrontSamples:
    """Pupil coordinates with phase/OPD and amplitude, before rasterization.

    The intermediate of AGENTS.md's artifact boundary: it has the phase
    information of a wavefront but not yet a grid, pitch, or normalization.
    """

    positions_m: np.ndarray
    optical_path_length_m: np.ndarray
    optical_path_length_reference: str
    wavelength_m: float
    reference_plane: ReferencePlane
    amplitude: np.ndarray | None = None
    pupil_mask: np.ndarray | None = None
    frame: Frame = field(default_factory=lambda: Frame(axis_order="flat per-sample arrays"))
    phasor: str = PHASOR
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        positions = _intake(self.positions_m, name="positions_m", complex_=False)
        object.__setattr__(self, "positions_m", positions)
        if positions.ndim != 2 or positions.shape[1] != 2:
            raise ContractError(
                ContractCode.SHAPE_MISMATCH,
                f"positions_m must be (N, 2) pupil coordinates, got {positions.shape}",
                declaration="positions_m",
            )
        if positions.shape[0] == 0:
            raise ContractError(
                ContractCode.EMPTY_ENSEMBLE,
                "wavefront sample set is empty",
                declaration="positions_m",
            )
        opl = _intake(self.optical_path_length_m, name="optical_path_length_m", complex_=False)
        object.__setattr__(self, "optical_path_length_m", opl)
        if opl.shape != (positions.shape[0],):
            raise ContractError(
                ContractCode.SHAPE_MISMATCH,
                f"optical_path_length_m must be ({positions.shape[0]},), got {opl.shape}",
                declaration="optical_path_length_m",
            )
        _check_finite(positions, "positions_m")
        _check_finite(opl, "optical_path_length_m")

        if not self.optical_path_length_reference:
            raise ContractError(
                ContractCode.MISSING_DECLARATION,
                "optical path length reference is not declared",
                declaration="optical_path_length_reference",
            )
        if self.optical_path_length_reference == UNVERIFIED:
            raise ContractError(
                ContractCode.OPL_REFERENCE_UNVERIFIED,
                "wavefront samples may not be built on an unverified OPL reference",
                declaration="optical_path_length_reference",
                remedy=(
                    "Optiland's opd_native is not admissible here. Characterize "
                    "it against a known geometry, or supply an OPL computed "
                    "from ray geometry with a stated reference."
                ),
            )
        if self.phasor != PHASOR:
            raise ContractError(
                ContractCode.PHASOR_MISMATCH,
                f"phasor must be {PHASOR!r}, got {self.phasor!r}",
                declaration="phasor",
            )
        if not math.isfinite(self.wavelength_m) or self.wavelength_m <= 0.0:
            raise ContractError(
                ContractCode.UNIT_NOT_SI,
                f"wavelength must be a positive value in metres, got {self.wavelength_m!r}",
                declaration="wavelength_m",
            )
        if self.amplitude is not None:
            amplitude = _intake(self.amplitude, name="amplitude", complex_=True)
            object.__setattr__(self, "amplitude", amplitude)
            if amplitude.shape != (positions.shape[0],):
                raise ContractError(
                    ContractCode.SHAPE_MISMATCH,
                    f"amplitude must be ({positions.shape[0]},), got {amplitude.shape}",
                    declaration="amplitude",
                )

        _require_same_representation(
            {
                "positions_m": self.positions_m,
                "optical_path_length_m": self.optical_path_length_m,
                "amplitude": self.amplitude,
                "pupil_mask": self.pupil_mask,
            },
            reference="positions_m",
        )

    @property
    def count(self) -> int:
        return int(self.positions_m.shape[0])

    @property
    def state(self) -> ArrayState:
        """Observed dtype/device/namespace, read from the pupil coordinates."""
        return array_state(self.positions_m)

    @property
    def device(self) -> DevicePlacement:
        return self.state.device

    @property
    def namespace(self) -> ArrayNamespace:
        return self.state.namespace

    @property
    def dtype(self) -> DType:
        return self.state.dtype

    @classmethod
    def from_artifact_record(
        cls, record: ArtifactRecord, *, arrays: dict[str, np.ndarray] | None = None
    ) -> WavefrontSamples:
        """Build from the Optiland adapter's ``wavefront.npz``.

        This *will* fail, by design, on an unmodified Optiland wavefront
        artifact: that artifact's only OPL source is ``RealRays.opd``, whose
        convention the adapter itself documents as not independently verified.
        The failure is the contract working, not a defect.
        """
        if record.kind is not ArtifactKind.WAVEFRONT_SAMPLES:
            raise ContractError(
                ContractCode.ARTIFACT_KIND_MISMATCH,
                f"expected {ArtifactKind.WAVEFRONT_SAMPLES}, got {record.kind}",
                artifact_id=record.id,
            )
        data = arrays if arrays is not None else dict(np.load(record.uri))
        metadata = record.metadata
        if metadata.get("length_unit") != "m":
            raise ContractError(
                ContractCode.UNIT_NOT_SI,
                f"pupil coordinates must be in metres, got {metadata.get('length_unit')!r}",
                declaration="length_unit",
                artifact_id=record.id,
            )
        wavelength_m = float(
            _require(metadata, "wavelength", artifact_id=record.id, what="wavelength")
        )
        reference = metadata.get("optical_path_length_reference")
        if reference is None:
            source = metadata.get("optical_path_length_source", "")
            raise ContractError(
                ContractCode.OPL_REFERENCE_UNVERIFIED,
                (
                    "this wavefront artifact declares no verified OPL reference; "
                    f"its source is {source!r}"
                ),
                declaration="optical_path_length_reference",
                artifact_id=record.id,
                remedy=(
                    "Characterize the OPD convention, then rebuild with "
                    "with_declared_optical_path_length()."
                ),
            )
        return cls(
            positions_m=np.column_stack([data["x_m"], data["y_m"]]),
            optical_path_length_m=np.asarray(data["opl_m"]),
            optical_path_length_reference=str(reference),
            wavelength_m=wavelength_m,
            reference_plane=ReferencePlane(
                name=str(
                    _require(metadata, "reference_plane", artifact_id=record.id, what="plane")
                ),
                z_m=float(
                    _require(metadata, "reference_plane_z_m", artifact_id=record.id, what="plane z")
                ),
            ),
            provenance={"source_artifact_id": record.id},
        )


@dataclass(frozen=True)
class ComplexField:
    """A sampled scalar complex field on a plane, in SI.

    ``u`` is an **amplitude**. Intensity is ``|u|**2``. The pad state is a
    required declaration because M1 measured a 256x256 Chromatix input growing
    to 1756x1756 on output, so an array shape alone does not determine physical
    extent.
    """

    u: np.ndarray
    sample_pitch_m: tuple[float, float]
    wavelength_m: float
    reference_plane: ReferencePlane
    frame: Frame = field(default_factory=Frame)
    phasor: str = PHASOR
    polarization: str = "scalar"
    normalization: str = "u is complex amplitude; discrete power = sum(|u|^2) * dy * dx"
    pad_width: int = 0
    padded: bool = False
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        u = self.u if hasattr(self.u, "dtype") else np.asarray(self.u)
        if not dtype_of(u).is_complex:
            raise ContractError(
                ContractCode.MISSING_DECLARATION,
                "a ComplexField must hold a complex array; a real array is an intensity, "
                "not an amplitude",
                declaration="u",
            )
        # No `.astype(complex128)`. A complex64 field produced on a GPU stays a
        # complex64 field on that GPU; widening it here would both fabricate
        # precision the producer never had and drag the array back to the host.
        u = _intake(u, name="u", complex_=True)
        object.__setattr__(self, "u", u)
        if u.ndim != 2:
            raise ContractError(
                ContractCode.SHAPE_MISMATCH,
                f"field must be a 2-D (y, x) array, got shape {u.shape}",
                declaration="u",
            )
        _check_finite(u, "u")
        self.frame.require_field_axis_order()

        pitch = tuple(float(value) for value in self.sample_pitch_m)
        if len(pitch) != 2 or not all(math.isfinite(v) and v > 0.0 for v in pitch):
            raise ContractError(
                ContractCode.UNIT_NOT_SI,
                f"sample_pitch_m must be a positive (dy, dx) in metres, got {pitch!r}",
                declaration="sample_pitch_m",
            )
        object.__setattr__(self, "sample_pitch_m", pitch)

        if not math.isfinite(self.wavelength_m) or self.wavelength_m <= 0.0:
            raise ContractError(
                ContractCode.UNIT_NOT_SI,
                f"wavelength must be a positive value in metres, got {self.wavelength_m!r}",
                declaration="wavelength_m",
            )
        if self.phasor != PHASOR:
            raise ContractError(
                ContractCode.PHASOR_MISMATCH,
                f"phasor must be {PHASOR!r}, got {self.phasor!r}",
                declaration="phasor",
            )
        if self.padded and self.pad_width <= 0:
            raise ContractError(
                ContractCode.PAD_STATE_UNKNOWN,
                "field is marked padded but declares no pad width",
                declaration="pad_width",
            )

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self.u.shape[0]), int(self.u.shape[1]))

    @property
    def wavenumber(self) -> float:
        return 2.0 * math.pi / self.wavelength_m

    @property
    def state(self) -> ArrayState:
        """Observed dtype/device/namespace of ``u``. Never caller-declared."""
        return array_state(self.u)

    @property
    def device(self) -> DevicePlacement:
        return self.state.device

    @property
    def namespace(self) -> ArrayNamespace:
        return self.state.namespace

    @property
    def dtype(self) -> DType:
        return self.state.dtype

    @property
    def xp(self) -> Any:
        return xp_for(self.namespace)

    @property
    def real_dtype(self) -> DType:
        """The real dtype matching this field's precision -- float32 for complex64."""
        return self.dtype.precision.real_dtype

    def coordinates(self) -> tuple[Any, Any]:
        """Return ``(y, x)`` coordinate vectors in metres.

        Uses the M1-pinned origin rule: array index ``n // 2`` is coordinate
        zero. This is the one place the rule is implemented, so a coupler
        cannot quietly adopt a different centring.

        Built in the field's own namespace, device and real precision, so a
        GPU complex64 field does not silently produce host float64 axes that
        every downstream operation then has to move or demote.
        """
        xp = self.xp
        ny, nx = self.shape
        dy, dx = self.sample_pitch_m
        real = numpy_dtype(self.real_dtype)
        y = (xp.arange(ny, dtype=real) - ny // 2) * dy
        x = (xp.arange(nx, dtype=real) - nx // 2) * dx
        return y, x

    def discrete_power(self) -> float:
        """Total discrete power. A Python float, so this synchronizes a GPU array.

        Called only where the number itself is the product -- diagnostics and
        persisted metadata -- never inside the propagation path.
        """
        xp = self.xp
        dy, dx = self.sample_pitch_m
        return float(xp.sum(xp.abs(self.u) ** 2) * dy * dx)

    @classmethod
    def from_artifact_record(
        cls, record: ArtifactRecord, *, array: np.ndarray | None = None
    ) -> ComplexField:
        """Build from the Chromatix adapter's ``output_field.npy`` + metadata."""
        if record.kind is not ArtifactKind.COMPLEX_FIELD:
            raise ContractError(
                ContractCode.ARTIFACT_KIND_MISMATCH,
                f"expected {ArtifactKind.COMPLEX_FIELD}, got {record.kind}",
                artifact_id=record.id,
            )
        u = array if array is not None else np.load(record.uri)
        metadata = record.metadata

        wavelength_m = float(
            _require(metadata, "wavelength", artifact_id=record.id, what="wavelength")
        )
        pitch = _require(metadata, "sample_pitch", artifact_id=record.id, what="sample pitch")
        pitch_tuple = (
            (float(pitch[0]), float(pitch[1]))
            if isinstance(pitch, (list, tuple))
            else (float(pitch), float(pitch))
        )
        phasor = _require(metadata, "phasor", artifact_id=record.id, what="phasor convention")
        if "pad_width" not in metadata:
            raise ContractError(
                ContractCode.PAD_STATE_UNKNOWN,
                "field arrived without a declared pad width, so its extent cannot be trusted",
                declaration="pad_width",
                artifact_id=record.id,
                remedy=(
                    "M1 measured a 256x256 input growing to 1756x1756; shape alone "
                    "is not extent."
                ),
            )
        pad_width = metadata.get("pad_width") or 0

        return cls(
            # Loaded from a host file, so this artifact IS host NumPy; the
            # producer's own device is recorded in the record's metadata rather
            # than claimed by an object that no longer lives there.
            u=u if hasattr(u, "dtype") else np.asarray(u),
            sample_pitch_m=pitch_tuple,
            wavelength_m=wavelength_m,
            reference_plane=ReferencePlane(
                name=str(metadata.get("propagation_method", "output plane")),
                z_m=float(metadata.get("z_m", 0.0)),
            ),
            phasor=str(phasor),
            polarization=str(metadata.get("polarization", "scalar")),
            normalization=str(
                _require(metadata, "normalization", artifact_id=record.id, what="normalization")
            ),
            pad_width=int(pad_width),
            padded=bool(metadata.get("padded", False)),
            provenance={"source_artifact_id": record.id, "source_uri": record.uri},
        )

    def to_artifact_record(self, *, artifact_id: str, uri: str | Path) -> ArtifactRecord:
        path = Path(uri)
        path.parent.mkdir(parents=True, exist_ok=True)
        host = _HostView(self, reason="npy persistence requires host bytes")
        np.save(path, host.of(self.u))
        state = self.state
        return ArtifactRecord(
            id=artifact_id,
            kind=ArtifactKind.COMPLEX_FIELD,
            uri=str(path),
            shape=self.shape,
            dtype=str(state.dtype),
            framework=_framework_for(state.namespace),
            device=state.device.to_spec_device(),
            units=None,
            metadata={
                "execution": state.as_dict(),
                "serialization": host.as_metadata(),
                "wavelength": self.wavelength_m,
                "sample_pitch": list(self.sample_pitch_m),
                "coordinate_frame": (
                    f"axes={self.frame.axis_order} row-major; {self.frame.handedness} "
                    f"Cartesian; {self.frame.propagation_axis} is the propagation direction"
                ),
                "origin": self.frame.origin_rule,
                "phasor": self.phasor,
                "spatial_factor": SPATIAL_FACTOR,
                "polarization": self.polarization,
                "normalization": self.normalization,
                "z_m": self.reference_plane.z_m,
                "reference_plane": self.reference_plane.name,
                "pad_width": self.pad_width,
                "padded": self.padded,
                "discrete_power": self.discrete_power(),
            },
        )


@dataclass(frozen=True)
class PSF:
    """A point-spread function: a non-negative intensity field with a pitch."""

    intensity: np.ndarray
    sample_pitch_m: tuple[float, float]
    wavelength_m: float
    normalization: str
    coherence_model: str = "fully coherent"
    frame: Frame = field(default_factory=Frame)
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        intensity = _intake(self.intensity, name="intensity", complex_=False)
        object.__setattr__(self, "intensity", intensity)
        xp = xp_for(namespace_of(intensity))
        if intensity.ndim != 2:
            raise ContractError(
                ContractCode.SHAPE_MISMATCH,
                f"PSF must be a 2-D (y, x) array, got shape {intensity.shape}",
                declaration="intensity",
            )
        _check_finite(intensity, "intensity")
        if bool(xp.any(intensity < 0.0)):
            raise ContractError(
                ContractCode.NEGATIVE_INTENSITY,
                "PSF intensity must be non-negative; a negative value means an "
                "amplitude was stored where an intensity was expected",
                declaration="intensity",
            )
        self.frame.require_field_axis_order()
        pitch = tuple(float(value) for value in self.sample_pitch_m)
        if len(pitch) != 2 or not all(math.isfinite(v) and v > 0.0 for v in pitch):
            raise ContractError(
                ContractCode.UNIT_NOT_SI,
                f"sample_pitch_m must be a positive (dy, dx) in metres, got {pitch!r}",
                declaration="sample_pitch_m",
            )
        object.__setattr__(self, "sample_pitch_m", pitch)
        if not self.normalization:
            raise ContractError(
                ContractCode.MISSING_DECLARATION,
                "a PSF must declare its normalization",
                declaration="normalization",
            )

    @classmethod
    def from_complex_field(
        cls,
        field_: ComplexField,
        *,
        normalization: str,
        coherence_model: str | None = None,
    ) -> PSF:
        """Reduce a field to its intensity. The one place ``|u|^2`` is taken.

        ``coherence_model`` was added by CHE-36 (M3.7): the slice is monochromatic
        and fully coherent, and M3.8 compares against oracles that assume exactly
        that, so the caller states it rather than inheriting the class default
        silently. Omitting it keeps the previous behaviour.

        The pitch is taken from the field, so a PSF's axes are only as correct as
        the pitch the producing adapter declared. For a propagated field that must
        be the OUTPUT pitch; see ``evaluation.psf_measurement``, which checks it.
        """
        return cls(
            # |u|^2 in the field's own namespace and precision: a complex64 GPU
            # field yields a float32 GPU PSF, with no host round trip and no
            # fabricated float64 digits.
            intensity=field_.xp.abs(field_.u) ** 2,
            sample_pitch_m=field_.sample_pitch_m,
            wavelength_m=field_.wavelength_m,
            normalization=normalization,
            frame=field_.frame,
            provenance={"from_field": field_.provenance.get("source_artifact_id")},
            **({} if coherence_model is None else {"coherence_model": coherence_model}),
        )

    @property
    def state(self) -> ArrayState:
        """Observed dtype/device/namespace of the intensity map."""
        return array_state(self.intensity)

    @property
    def device(self) -> DevicePlacement:
        return self.state.device

    @property
    def namespace(self) -> ArrayNamespace:
        return self.state.namespace

    @property
    def dtype(self) -> DType:
        return self.state.dtype

    def to_artifact_record(self, *, artifact_id: str, uri: str | Path) -> ArtifactRecord:
        path = Path(uri)
        path.parent.mkdir(parents=True, exist_ok=True)
        host = _HostView(self, reason="npy persistence requires host bytes")
        np.save(path, host.of(self.intensity))
        state = self.state
        return ArtifactRecord(
            id=artifact_id,
            kind=ArtifactKind.PSF,
            uri=str(path),
            shape=(int(self.intensity.shape[0]), int(self.intensity.shape[1])),
            dtype=str(state.dtype),
            framework=_framework_for(state.namespace),
            device=state.device.to_spec_device(),
            units=None,
            metadata={
                "execution": state.as_dict(),
                "serialization": host.as_metadata(),
                "sample_pitch": list(self.sample_pitch_m),
                "wavelength": self.wavelength_m,
                "normalization": self.normalization,
                "coherence_model": self.coherence_model,
                "origin": self.frame.origin_rule,
            },
        )
