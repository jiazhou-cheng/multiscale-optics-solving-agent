"""The coherent ray batch that crosses the Optiland boundary (CHE-70, Phases 1-3).

``RayBundle`` is already the project's ray artifact and it already carries the
two quantities that make a bundle coherent -- a complex amplitude and an OPL
with a declared reference. This module adds exactly what sending one *through a
solver* requires and nothing else:

* **ray identity**, so an amplitude can be re-associated with the ray it belongs
  to after a trace, rather than with the ray that happens to sit at the same
  index;
* **a validity mask**, so a clipped ray is excluded from the reconstruction
  instead of contributing whatever its stale state says;
* **one unit-conversion boundary**, in one place, with named constants.

Why the amplitude does not enter Optiland
-----------------------------------------
Optiland's ``RealRays.i`` is a real intensity used for bookkeeping and clipping.
It is *not* a complex amplitude and cannot carry one: there is no complex ray
field in the pinned solver. So the complex amplitude travels as a **sidecar** --
it is never handed to Optiland, never read back from ``RealRays.i``, and is
re-attached after the trace by ray id. ``RealRays.i`` is initialised from
``|a|^2`` only because Optiland wants an intensity, and the value that comes
back is used for exactly one thing: detecting which rays were clipped.

Units, in one place
-------------------
The coupler contracts are SI (metres). Optiland's geometry is millimetres and
its wavelengths are micrometres (``core.optical_system.UNITS``). Rather than
scattering ``1e3`` and ``1e-6`` through the bridge, the two directions are
:func:`metres_to_millimetres` / :func:`millimetres_to_metres` and
:func:`metres_to_micrometres` / :func:`micrometres_to_metres`, and every call
site in the CHE-70 path goes through them. Direction cosines and the validity
mask are dimensionless and are the only quantities that legitimately cross
unconverted.

The OPL contract, inherited and narrowed
----------------------------------------
CHE-30/CHE-41 characterised ``RealRays.opd`` on the *pupil* path, where Optiland
seeds the accumulator itself on a plane nobody downstream declared -- which is
why the adapter refuses to promote ``opd_native`` to an OPL and emits a declared
OPL instead (``ContractCode.OPL_REFERENCE_UNVERIFIED``, hazard H1 of
``knowledge/couplers/ray_to_wave/conventions.md``).

That refusal is not being revisited. It does not apply here for a structural
reason: on this path **the caller constructs the rays**, so ``opd`` starts at
exactly zero on a plane the caller declared, and ``_trace_real`` accumulates
``opd += |t| * n_pre(w)`` -- an index-weighted geometric path in the geometry's
own length unit. The reference is therefore known by construction rather than
inferred, and it is the *same* plane the wave-side spectrum was decomposed on.
:func:`declared_launch_opl_reference` spells that out and is what the emitted
bundle carries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from core.arrays import device_of, dtype_of, namespace_of
from core.boundary import (
    ContractCode,
    ContractError,
    Frame,
    RayBundle,
    ReferencePlane,
)
from core.precision import ArrayNamespace

__all__ = [
    "AMPLITUDE_SIDECAR_RULE",
    "MICROMETRES_PER_METRE",
    "MILLIMETRES_PER_METRE",
    "OPTILAND_INTENSITY_RULE",
    "CoherentRayBatch",
    "declared_launch_opl_reference",
    "metres_to_micrometres",
    "metres_to_millimetres",
    "micrometres_to_metres",
    "millimetres_to_metres",
]

#: Optiland geometry is millimetres; the project's contracts are metres.
MILLIMETRES_PER_METRE = 1.0e3
#: Optiland wavelengths are micrometres (``core.optical_system.UNITS``).
MICROMETRES_PER_METRE = 1.0e6

AMPLITUDE_SIDECAR_RULE = (
    "the complex amplitude is a sidecar: it is never passed to Optiland, never "
    "reconstructed from RealRays.i, and is re-associated after the trace by ray id"
)
OPTILAND_INTENSITY_RULE = (
    "RealRays.i is initialised from |a|^2 for Optiland's own bookkeeping and is "
    "read back only to detect clipped rays; it is never read as an amplitude"
)


def metres_to_millimetres(value: Any) -> Any:
    return value * MILLIMETRES_PER_METRE


def millimetres_to_metres(value: Any) -> Any:
    return value / MILLIMETRES_PER_METRE


def metres_to_micrometres(value: float) -> float:
    return float(value) * MICROMETRES_PER_METRE


def micrometres_to_metres(value: float) -> float:
    return float(value) / MICROMETRES_PER_METRE


def declared_launch_opl_reference(plane: ReferencePlane) -> str:
    """The OPL reference string for a batch whose rays were built at ``plane``.

    Deliberately verbose. It names the plane, states that the zero is a
    construction fact rather than an inference, and names the accumulation rule
    the pinned solver actually applies -- so a reader of a persisted artifact can
    tell this path apart from the pupil path's refused ``opd_native``.
    """
    return (
        f"zero at the emitting plane {plane.name!r} (z = {plane.z_m!r} m), set by "
        "constructing RealRays there with opd = 0; Optiland accumulates "
        "opd += |t| * n_pre(wavelength) per surface, an index-weighted geometric "
        "path in millimetres (optiland 0.6.0 Surface._trace_real)"
    )


@dataclass(frozen=True)
class CoherentRayBatch:
    """A ray bundle plus the identity and validity a solver round trip needs.

    ``bundle`` holds the physics and is an ordinary :class:`RayBundle`, so every
    contract check and every coupler that already exists applies unchanged. The
    two added arrays are bookkeeping:

    ``ray_id``
        A stable integer per ray, unique within the batch. After a trace, the
        traced state is matched to the incoming amplitude through this, not
        through position in the array. Optiland 0.6.0 happens to preserve order
        (it clips by zeroing intensity rather than by removing rows), but that is
        a property of the pinned version, so it is *checked* rather than assumed
        -- see ``tests/test_coherent_batch.py``.
    ``valid``
        Boolean, true for rays that may contribute to a reconstruction. A ray
        clipped by an Optiland aperture is marked false here; its position,
        direction and OPL are then meaningless and must not be summed.

    Both are dimensionless and both live in the same namespace and on the same
    device as the bundle's geometry, so a batch never straddles a device.
    """

    bundle: RayBundle
    ray_id: Any
    valid: Any
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        count = self.bundle.count
        for name, array in (("ray_id", self.ray_id), ("valid", self.valid)):
            shaped = np.asarray(array) if isinstance(array, (list, tuple)) else array
            object.__setattr__(self, name, shaped)
            if shaped.ndim != 1 or int(shaped.shape[0]) != count:
                raise ContractError(
                    ContractCode.SHAPE_MISMATCH,
                    f"{name} must be a 1-D array of length {count}, got shape {shaped.shape}",
                    declaration=name,
                )
        # The two arrays have deliberately different residency rules, because
        # they are used differently. `valid` multiplies device data, so it must
        # live where the geometry lives or the mask would force a transfer.
        # `ray_id` is host NumPy on purpose: the ids come from the seeded host
        # sampler that pins reproducibility, they carry no precision and no
        # physics, and requiring them on the device would buy a transfer and
        # nothing else. `CoherentRayBatch.residency` reports them separately for
        # the same reason, so a device-fallback check does not flag the one host
        # array the design asks for.
        geometry_namespace = namespace_of(self.bundle.positions_m)
        if namespace_of(self.valid) is not geometry_namespace:
            raise ContractError(
                ContractCode.REPRESENTATION_INCONSISTENT,
                (
                    f"valid lives in {namespace_of(self.valid)} while the batch "
                    f"geometry lives in {geometry_namespace}; the mask multiplies "
                    "the amplitude, so a mismatch would force a hidden transfer"
                ),
                declaration="valid",
            )
        if namespace_of(self.ray_id) is not ArrayNamespace.NUMPY:
            raise ContractError(
                ContractCode.REPRESENTATION_INCONSISTENT,
                (
                    f"ray_id lives in {namespace_of(self.ray_id)}; ids are host "
                    "NumPy by design, because they come from the seeded host "
                    "sampler and carry no physics"
                ),
                declaration="ray_id",
            )

        # Ray identity is only identity if it is unique. A duplicated id would
        # let two amplitudes be attributed to one traced ray with no error.
        host_ids = np.asarray(self.ray_id)
        if host_ids.size and np.unique(host_ids).size != host_ids.size:
            raise ContractError(
                ContractCode.MISSING_DECLARATION,
                "ray_id contains duplicates, so it cannot identify a ray",
                declaration="ray_id",
                remedy="Assign ids from a single monotonically increasing counter.",
            )

    # -- convenience -------------------------------------------------------
    @property
    def count(self) -> int:
        return self.bundle.count

    @property
    def valid_count(self) -> int:
        return int(np.asarray(self.valid).sum())

    @property
    def amplitude(self) -> Any:
        amplitude, _ = self.bundle.require_coherent()
        return amplitude

    def residency(self) -> dict[str, dict[str, str]]:
        """Observed dtype/device/namespace of every array in the batch.

        Phase 21's residency log, split into two groups because they answer
        different questions. ``scientific`` holds the physics -- geometry,
        amplitude, OPL -- and every entry must be on the device the run
        requested; a host entry there is a fallback. ``bookkeeping`` holds
        ``ray_id`` and ``valid``, which carry no precision and no physics:
        ``ray_id`` is a host integer array *by design*, because the sampler's
        seed is pinned by ``numpy.random.Generator``, and flagging it as a
        device fallback would make the residency check cry wolf on the one
        transfer the coupler documents as deliberate.

        Read off the arrays, never off a request: a requested device and an
        actual device are different facts, and PB4a measured a configuration
        where they disagreed silently.
        """
        scientific: dict[str, Any] = {
            "positions_m": self.bundle.positions_m,
            "directions": self.bundle.directions,
            "amplitude": self.bundle.amplitude,
            "optical_path_length_m": self.bundle.optical_path_length_m,
        }
        bookkeeping: dict[str, Any] = {"ray_id": self.ray_id, "valid": self.valid}
        return {
            "scientific": {
                name: {
                    "dtype": str(dtype_of(array)),
                    "device": str(device_of(array)),
                    "namespace": str(namespace_of(array)),
                }
                for name, array in scientific.items()
                if array is not None
            },
            "bookkeeping": {
                name: {
                    # Raw dtype, not DType: the project vocabulary is field
                    # dtypes and deliberately excludes integer and boolean.
                    "dtype": str(getattr(array, "dtype", type(array).__name__)),
                    "device": str(device_of(array)),
                    "namespace": str(namespace_of(array)),
                }
                for name, array in bookkeeping.items()
            },
        }

    def with_traced_state(
        self,
        *,
        positions_m: Any,
        directions: Any,
        optical_path_length_m: Any,
        valid: Any,
        plane: ReferencePlane,
        ray_id: Any,
        amplitude: Any = None,
        provenance: dict[str, Any] | None = None,
    ) -> CoherentRayBatch:
        """Re-attach this batch's amplitude to a traced geometry, matched by id.

        The wavelength, normalization and reconstruction convention come from
        ``self``; the geometry and OPL come from the solver. The ids must match
        element-wise -- if the solver reordered or dropped rays this raises rather
        than silently pairing the wrong amplitude with the wrong ray, which is the
        one failure mode this whole type exists to prevent.

        ``amplitude`` defaults to this batch's own, which is the invariant that
        matters: the complex amplitude is never re-derived from anything the
        solver returned. A caller may pass the same amplitude with the invalid
        rays zeroed -- that is a *masking* of rays the solver clipped, not a new
        amplitude, and it is what keeps a clipped ray from contributing.
        """
        incoming = np.asarray(ray_id)
        mine = np.asarray(self.ray_id)
        if incoming.shape != mine.shape or not np.array_equal(incoming, mine):
            raise ContractError(
                ContractCode.SHAPE_MISMATCH,
                (
                    "traced ray ids do not match the batch's ray ids, so the "
                    "complex amplitude cannot be re-associated by identity"
                ),
                declaration="ray_id",
                remedy=(
                    "Recover the valid-ray mapping explicitly: index the batch by "
                    "the returned ids before reconstructing."
                ),
            )
        traced = RayBundle(
            positions_m=positions_m,
            directions=directions,
            wavelength_m=self.bundle.wavelength_m,
            reference_plane=plane,
            frame=Frame(axis_order="flat per-ray arrays"),
            amplitude=self.bundle.amplitude if amplitude is None else amplitude,
            optical_path_length_m=optical_path_length_m,
            optical_path_length_reference=declared_launch_opl_reference(
                self.bundle.reference_plane
            ),
            normalization=self.bundle.normalization,
            reconstruction_normalization=self.bundle.reconstruction_normalization,
            provenance={
                **self.bundle.provenance,
                "traced_through": "M_RAY_OPTILAND",
                "amplitude_handling": AMPLITUDE_SIDECAR_RULE,
                "optiland_intensity_handling": OPTILAND_INTENSITY_RULE,
                **(provenance or {}),
            },
        )
        return CoherentRayBatch(
            bundle=traced,
            ray_id=ray_id,
            valid=valid,
            provenance={**self.provenance, **(provenance or {})},
        )
