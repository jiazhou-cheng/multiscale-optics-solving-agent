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

from multiscale_optics_agent.core.artifacts import ArtifactRecord
from multiscale_optics_agent.core.specs import ArtifactKind, Device, Framework

__all__ = [
    "PHASOR",
    "SPATIAL_FACTOR",
    "AXIS_ORDER",
    "ORIGIN_RULE",
    "ContractCode",
    "ContractError",
    "Frame",
    "ReferencePlane",
    "RayBundle",
    "WavefrontSamples",
    "ComplexField",
    "PSF",
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

_DIRECTION_NORM_TOLERANCE = 1e-9


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
    PAD_STATE_UNKNOWN = "PAD_STATE_UNKNOWN"
    NEGATIVE_INTENSITY = "NEGATIVE_INTENSITY"
    ARTIFACT_KIND_MISMATCH = "ARTIFACT_KIND_MISMATCH"


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


def _check_finite(array: np.ndarray, name: str) -> None:
    if not np.all(np.isfinite(array)):
        raise ContractError(
            ContractCode.NON_FINITE,
            f"{name} contains non-finite values",
            declaration=name,
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
        positions = np.asarray(self.positions_m, dtype=np.float64)
        directions = np.asarray(self.directions, dtype=np.float64)
        object.__setattr__(self, "positions_m", positions)
        object.__setattr__(self, "directions", directions)

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

        norms = np.linalg.norm(directions, axis=1)
        worst = float(np.max(np.abs(norms - 1.0)))
        if worst > _DIRECTION_NORM_TOLERANCE:
            raise ContractError(
                ContractCode.NON_UNIT_DIRECTION,
                f"direction vectors must be unit norm; worst deviation {worst:.3e}",
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
            dtype = np.complex128 if name == "amplitude" else np.float64
            array = np.asarray(value, dtype=dtype)
            object.__setattr__(self, name, array)
            if array.shape != (positions.shape[0],):
                raise ContractError(
                    ContractCode.SHAPE_MISMATCH,
                    f"{name} must be ({positions.shape[0]},), got {array.shape}",
                    declaration=name,
                )
            _check_finite(array, name)

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
                if np.any(self.weight < 0.0):
                    raise ContractError(
                        ContractCode.NEGATIVE_INTENSITY,
                        "cannot take sqrt of a negative weight",
                        declaration="weight",
                    )
                amplitude = np.sqrt(self.weight).astype(np.complex128)
            else:
                raise ContractError(
                    ContractCode.MISSING_DECLARATION,
                    f"no built-in conversion for mapping {mapping!r}; pass amplitude explicitly",
                    declaration="amplitude",
                )
        return self._replace(
            amplitude=np.asarray(amplitude, dtype=np.complex128),
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
            optical_path_length_m=np.asarray(optical_path_length_m, dtype=np.float64),
            optical_path_length_reference=reference,
        )

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

        positions = np.column_stack([data["x_m"], data["y_m"], data["z_m"]]).astype(np.float64)
        directions = np.column_stack([data["L"], data["M"], data["N"]]).astype(np.float64)

        weight = None
        weight_semantics = None
        if "intensity" in data:
            weight = np.asarray(data["intensity"], dtype=np.float64)
            weight_semantics = metadata.get(
                "intensity_is_not_amplitude", "unnamed ray weight"
            )

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
        path = Path(uri)
        arrays: dict[str, np.ndarray] = {
            "x_m": self.positions_m[:, 0],
            "y_m": self.positions_m[:, 1],
            "z_m": self.positions_m[:, 2],
            "L": self.directions[:, 0],
            "M": self.directions[:, 1],
            "N": self.directions[:, 2],
        }
        if self.amplitude is not None:
            arrays["amplitude"] = self.amplitude
        if self.weight is not None:
            arrays["intensity"] = self.weight
        if self.optical_path_length_m is not None:
            arrays["opl_m"] = self.optical_path_length_m
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **arrays)

        return ArtifactRecord(
            id=artifact_id,
            kind=ArtifactKind.RAY_BUNDLE,
            uri=str(path),
            shape=(self.count,),
            dtype="float64",
            framework=Framework.NUMPY,
            device=Device.CPU,
            units="SI: metres, dimensionless unit directions",
            metadata={
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
        positions = np.asarray(self.positions_m, dtype=np.float64)
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
        opl = np.asarray(self.optical_path_length_m, dtype=np.float64)
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
            amplitude = np.asarray(self.amplitude, dtype=np.complex128)
            object.__setattr__(self, "amplitude", amplitude)
            if amplitude.shape != (positions.shape[0],):
                raise ContractError(
                    ContractCode.SHAPE_MISMATCH,
                    f"amplitude must be ({positions.shape[0]},), got {amplitude.shape}",
                    declaration="amplitude",
                )

    @property
    def count(self) -> int:
        return int(self.positions_m.shape[0])

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
            positions_m=np.column_stack([data["x_m"], data["y_m"]]).astype(np.float64),
            optical_path_length_m=np.asarray(data["opl_m"], dtype=np.float64),
            optical_path_length_reference=str(reference),
            wavelength_m=wavelength_m,
            reference_plane=ReferencePlane(
                name=str(_require(metadata, "reference_plane", artifact_id=record.id, what="plane")),
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
        u = np.asarray(self.u)
        if not np.iscomplexobj(u):
            raise ContractError(
                ContractCode.MISSING_DECLARATION,
                "a ComplexField must hold a complex array; a real array is an intensity, "
                "not an amplitude",
                declaration="u",
            )
        u = u.astype(np.complex128, copy=False)
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

    def coordinates(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(y, x)`` coordinate vectors in metres.

        Uses the M1-pinned origin rule: array index ``n // 2`` is coordinate
        zero. This is the one place the rule is implemented, so a coupler
        cannot quietly adopt a different centring.
        """
        ny, nx = self.shape
        dy, dx = self.sample_pitch_m
        y = (np.arange(ny, dtype=np.float64) - ny // 2) * dy
        x = (np.arange(nx, dtype=np.float64) - nx // 2) * dx
        return y, x

    def discrete_power(self) -> float:
        dy, dx = self.sample_pitch_m
        return float(np.sum(np.abs(self.u) ** 2) * dy * dx)

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
                remedy="M1 measured a 256x256 input growing to 1756x1756; shape alone is not extent.",
            )
        pad_width = metadata.get("pad_width") or 0

        return cls(
            u=np.asarray(u),
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
        np.save(path, self.u)
        return ArtifactRecord(
            id=artifact_id,
            kind=ArtifactKind.COMPLEX_FIELD,
            uri=str(path),
            shape=self.shape,
            dtype=str(self.u.dtype),
            framework=Framework.NUMPY,
            device=Device.CPU,
            units=None,
            metadata={
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
        intensity = np.asarray(self.intensity, dtype=np.float64)
        object.__setattr__(self, "intensity", intensity)
        if intensity.ndim != 2:
            raise ContractError(
                ContractCode.SHAPE_MISMATCH,
                f"PSF must be a 2-D (y, x) array, got shape {intensity.shape}",
                declaration="intensity",
            )
        _check_finite(intensity, "intensity")
        if np.any(intensity < 0.0):
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
    def from_complex_field(cls, field_: ComplexField, *, normalization: str) -> PSF:
        return cls(
            intensity=np.abs(field_.u) ** 2,
            sample_pitch_m=field_.sample_pitch_m,
            wavelength_m=field_.wavelength_m,
            normalization=normalization,
            frame=field_.frame,
            provenance={"from_field": field_.provenance.get("source_artifact_id")},
        )

    def to_artifact_record(self, *, artifact_id: str, uri: str | Path) -> ArtifactRecord:
        path = Path(uri)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, self.intensity)
        return ArtifactRecord(
            id=artifact_id,
            kind=ArtifactKind.PSF,
            uri=str(path),
            shape=(int(self.intensity.shape[0]), int(self.intensity.shape[1])),
            dtype=str(self.intensity.dtype),
            framework=Framework.NUMPY,
            device=Device.CPU,
            units=None,
            metadata={
                "sample_pitch": list(self.sample_pitch_m),
                "wavelength": self.wavelength_m,
                "normalization": self.normalization,
                "coherence_model": self.coherence_model,
                "origin": self.frame.origin_rule,
            },
        )
