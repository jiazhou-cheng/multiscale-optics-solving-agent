"""One ray representation: `RayBundle`, rays as plane wavelets, in SI.

CHE-175 (R02.3). Exactly one public ray type. The reference implementation had
seven names for this idea -- `RayBundle`, `CoherentRayBatch` (340 LOC of second
carrier), `WavefrontSamples`, and the `GeometricRayBundle` / `CoherentRayBundle`
/ `RayBundleBase` / `TrackedRayBundle` family the architecture document bans by
name. R00.2 settled the collapse: `WavefrontSamples` had **zero** production
consumers (`docs/rewrite/reference_inventory.md` §1.1), and the distinction it
existed to express -- pupil-sampled phase before rasterization versus a gridded
field -- is carried by this type plus `ScalarField`.

**Coherence is a contract, not a subtype.** A bundle that can be converted to a
wave is not another class; it is a bundle that passes `require_coherent()`.
Carrying an unverified quantity is fine, reading it as physics is not, and the
gate is the difference.

Three groups of fields, kept apart on purpose
---------------------------------------------
*Geometry* -- `positions_m`, `directions`, `wavelength_m`, and where they are
declared. Always present.

*Coherent state* -- `amplitude` and `optical_path_m` with its reference. Optional
and jointly validated. They are optional because Optiland supplies neither: it
gives a real `intensity` weight, which is **not** an amplitude, and an
`opd_native` whose sign and reference M1 recorded as unverified. Converting a
weight into an amplitude is a modelling decision (is the weight a power, so
`a = sqrt(w)`? a photon count? already an amplitude?) and this type does not make
it -- a producer states it by constructing a bundle that carries the amplitude it
decided on.

*Sampling measure* -- `measure_weight` and `measure_kind`, which are **not** the
amplitude and are never folded into it. `U(r) = sum_i a_i exp[i k (OPL_i +
dr_i(r))]` is a quadrature over the ray ensemble, so each ray stands for a cell
of the pupil, and the size of that cell is a property of how the pupil was
*sampled* rather than of the light. CHE-38 measured what conflating them costs:
treating every ray as an equal-weight sample leaves a sensor-plane residual of
3.84e-3 that a correct area measure collapses to 4.07e-4, and CHE-33 measured the
other half -- without an absolute area element the reconstructed power scales as
(ray count)^2.0024 instead of converging.

The measure lands here rather than in the coupler that consumes it because a
trusted ray-to-wave conversion has to be able to **refuse** an unknown measure,
and that refusal is only expressible if the representation can say "mine is
undeclared". `MEASURE_UNDECLARED` is the default, and it is the useful value.

*Population provenance* -- `ray_splitting`, added by CHE-226 (R16), and the one
field here that is not a physical quantity. It says how the *population* was
produced rather than what any ray is: whether it contains ray-splitting
descendants. It lands on the representation for exactly the reason the measure
did, and the parallel is deliberate down to the three values and the default: a
measurement over ray intersections has to be able to **refuse** a population it
cannot interpret, and a consumer whose only input is a bundle can only be told by
the bundle. `measurements.spot_diagram` is that consumer and
`RAY_SPLITTINGS` is the vocabulary; nothing in this module reads the field,
because whether splitting matters is a property of the consuming measurement and
not of the rays.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from numerics import ArrayState, DType, array_state, dtype_of, numpy_dtype, xp_for
from representations.contracts import (
    ContractError,
    adopt_array,
    require_finite,
    require_positive_si,
    require_same_representation,
)
from representations.geometry import PHASOR, Frame, ReferenceSurface

__all__ = [
    "MEASURE_KINDS",
    "RAY_SPLITTINGS",
    "UNVERIFIED",
    "MeasureKind",
    "RayBundle",
    "RaySplitting",
    "direction_norm_tolerance",
]

#: What a per-ray `measure_weight` means. Three values, and the third is the one
#: that does the work.
#:
#: `quadrature_area_m2`
#:     A declared quadrature: ray `i` stands for a pupil cell of `w_i` square
#:     metres. The reference producer is the hexapolar area weight
#:     (`pre-rewrite-2026-08-30:src/couplers/quadrature.py:127`), whose interior
#:     cells are `pi a^2 / (3 R^2)` with two measured boundary corrections -- the
#:     central ray gets 3/4 of a nominal cell and the rim ring 1/2, because the
#:     rim sits exactly on `rho = a` with no ray beyond it to average with.
#: `importance_weight`
#:     A declared Monte-Carlo importance weight, dimensionless: `1 / p(x_i)` for
#:     the density the rays were drawn from. A reconstruction from these is an
#:     estimator and owes the `1/N` that a physical ray trace must not have.
#: `undeclared`
#:     Nothing is known about the measure. **Not** a synonym for "uniform": a
#:     coupler that treats it as uniform has invented a quadrature. This is the
#:     value that lets R07 refuse instead of guess, and it is the default so that
#:     refusing is what happens when nobody thought about it.
MeasureKind = Literal["quadrature_area_m2", "importance_weight", "undeclared"]

MEASURE_KINDS: tuple[MeasureKind, ...] = (
    "quadrature_area_m2",
    "importance_weight",
    "undeclared",
)

#: Whether this population contains **ray-splitting descendants**: rays produced by
#: one incident ray dividing into several, as at a partially reflecting surface or
#: across the orders of a grating.
#:
#: Three values, and the third is again the one that does the work.
#:
#: `unsplit`
#:     Declared to contain no descendants: each row is one ray of one incident
#:     population, traced along one path. This is what a sequential geometric trace
#:     produces, and `backends/optiland/` declares it on both of its paths.
#: `split_descendants`
#:     Declared to contain them. A geometric measurement over the intersections is
#:     then measuring a superposition of branches it cannot separate -- N rows are
#:     not N rays of one population, so an unweighted first or second moment over
#:     them is a statistic of the branching and not of the spot.
#: `undeclared`
#:     Nothing is known. **Not** a synonym for `unsplit`: a measurement that treats
#:     it as unsplit has assumed the very thing it needed told. The default, so that
#:     refusing is what happens when nobody thought about it.
#:
#: Deliberately **orthogonal to survival**. A ray may be clipped by a surface rim
#: without having split, and a split descendant may survive to the image surface;
#: this project encodes survival by zeroing the amplitude on the supplied-ray path
#: (`backends.optiland.rays.SUPPLIED_RAY_SURVIVAL_RULE`) and by dropping the row on
#: the generated path, and neither of those is a statement about splitting. Nothing
#: derives one from the other, in either direction.
#:
#: **No producer in this tree declares `split_descendants` today**, and the value
#: exists anyway rather than the field being a boolean: `O_DIFFRACTIVE_SURFACE`'s
#: full-field route already re-decomposes a bundle into a different number of rays
#: (`operators/diffractive_surface.py`), and a multi-order model is the first thing
#: that would produce descendants. A two-valued field would force that producer to
#: choose between a false `unsplit` and an `undeclared` that means "nobody said",
#: which is exactly the collapse `measure_kind` avoids.
RaySplitting = Literal["unsplit", "split_descendants", "undeclared"]

RAY_SPLITTINGS: tuple[RaySplitting, ...] = (
    "unsplit",
    "split_descendants",
    "undeclared",
)

#: The one value an optical-path reference may not take.
#:
#: Optiland's OPD sign and reference plane were both recorded unverified by M1. A
#: wrong OPL *reference* is a harmless piston; a wrong OPL *sign* conjugates the
#: wavefront and turns a converging beam into a diverging one, and the two are
#: indistinguishable downstream. So a bundle may *carry* an unverified path --
#: that is how a solver hands one over honestly -- and `require_coherent()`
#: refuses to read it as physics.
UNVERIFIED = "unverified"

#: Historical float64 direction-norm tolerance, kept verbatim as the floor so the
#: established CPU/float64 behaviour is bit-for-bit unchanged. It is ~4e6 times
#: looser than float64 round-off: a legacy allowance, not a derived bound.
_DIRECTION_NORM_FLOOR = 1e-9


def direction_norm_tolerance(dtype: DType) -> float:
    """Unit-norm tolerance appropriate to the dtype the directions are stored in.

    Re-derived from `pre-rewrite-2026-08-30:src/core/boundary.py:102`. R00.2 §6.2
    lists this as one rule with four consumers that belongs to the representation
    in the new tree, which is why it is here and public rather than private to the
    `__post_init__` that first needed it.

    Holding a float32 bundle to a float64 tolerance is not strictness, it is a
    category error: casting an exactly normalized float64 direction to float32
    already perturbs `|d|` by about one float32 epsilon (1.2e-7) before the norm
    is even computed, and computing it adds a few more. `64 * eps` is that
    round-off with an order of magnitude of headroom -- derived rather than
    picked -- and it reduces to the historical constant at float64.
    """
    eps = float(np.finfo(numpy_dtype(dtype)).eps)
    return max(_DIRECTION_NORM_FLOOR, 64.0 * eps)


@dataclass(frozen=True)
class RayBundle:
    """Rays as plane wavelets at a declared reference surface, in SI.

    A class on rules 1 and 2. Rule 1: the geometry, the coherent state and the
    measure are three groups of fields with *joint* invariants -- every per-ray
    array has to have the same length as the positions, live on the same device in
    the same array ecosystem, and an optical path without its reference or a
    measure weight without its kind is a quantity nobody can use. Rule 2: it is
    the public data model a solver produces and a coupler consumes, so its field
    names are an interface rather than an implementation detail.
    """

    #: `(N, 3)` ray origins in metres, in the declared frame.
    positions_m: Any

    #: `(N, 3)` unit direction cosines. Checked against
    #: `direction_norm_tolerance(dtype)`, not a fixed absolute bound.
    directions: Any

    #: Vacuum wavelength in metres. One value: a bundle is monochromatic per
    #: evaluation, and a spectrum is several bundles.
    wavelength_m: float

    #: Where these rays are declared. The optical path, if any, is measured in a
    #: geometry this surface fixes.
    reference_surface: ReferenceSurface

    frame: Frame = field(default_factory=Frame)

    #: `(N,)` complex amplitude `a_i`. An amplitude, never an intensity. A real
    #: array is accepted and widened to the complex dtype of the same precision,
    #: because `sqrt(w)` of a declared power weight is a phase-free amplitude.
    amplitude: Any | None = None

    #: `(N,)` optical path length in metres. Requires `optical_path_reference`.
    optical_path_m: Any | None = None

    #: What the optical path is measured from. Free text because the answer is a
    #: plane or a ray in the caller's system, except for `UNVERIFIED`, which is a
    #: declared absence of knowledge rather than a reference.
    optical_path_reference: str | None = None

    #: `(N,)` sampling measure. Real, finite, non-negative. Never multiplied into
    #: `amplitude` -- see the module docstring.
    measure_weight: Any | None = None

    measure_kind: MeasureKind = "undeclared"

    #: The time convention the amplitude and the optical path are written in.
    phasor: str = PHASOR

    #: How this population was produced, in the one respect a geometric measurement
    #: over the intersections has to know: see `RAY_SPLITTINGS`. Provenance, not
    #: physical state, and independent of survival.
    ray_splitting: RaySplitting = "undeclared"

    def __post_init__(self) -> None:
        positions = adopt_array(self.positions_m, name="positions_m", complex_=False)
        directions = adopt_array(self.directions, name="directions", complex_=False)
        object.__setattr__(self, "positions_m", positions)
        object.__setattr__(self, "directions", directions)

        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ContractError(
                "SHAPE_MISMATCH",
                f"positions_m must be (N, 3), got {tuple(positions.shape)}",
                declaration="positions_m",
            )
        if directions.shape != positions.shape:
            raise ContractError(
                "SHAPE_MISMATCH",
                f"directions {tuple(directions.shape)} must match positions "
                f"{tuple(positions.shape)}",
                declaration="directions",
            )
        count = int(positions.shape[0])
        if count == 0:
            raise ContractError(
                "EMPTY_ENSEMBLE",
                "ray bundle is empty; there is nothing to reconstruct from",
                declaration="positions_m",
            )

        for name, complex_ in (
            ("amplitude", True),
            ("optical_path_m", False),
            ("measure_weight", False),
        ):
            value = getattr(self, name)
            if value is None:
                continue
            array = adopt_array(value, name=name, complex_=complex_, widen_real=complex_)
            object.__setattr__(self, name, array)
            if array.shape != (count,):
                raise ContractError(
                    "SHAPE_MISMATCH",
                    f"{name} must be ({count},), one per ray, got {tuple(array.shape)}",
                    declaration=name,
                )

        # Structural agreement before values: a bundle whose arrays live in two
        # places is not one artifact, and reading its values would be the first
        # operation that silently moved one of them.
        require_same_representation(
            {
                "positions_m": self.positions_m,
                "directions": self.directions,
                "amplitude": self.amplitude,
                "optical_path_m": self.optical_path_m,
                "measure_weight": self.measure_weight,
            },
            reference="positions_m",
        )

        require_finite(positions, name="positions_m")
        require_finite(directions, name="directions")
        for name in ("amplitude", "optical_path_m", "measure_weight"):
            value = getattr(self, name)
            if value is not None:
                require_finite(value, name=name)

        xp = xp_for(array_state(directions).namespace)
        norms = xp.linalg.norm(directions, axis=1)
        worst = float(xp.max(xp.abs(norms - 1.0)))
        tolerance = direction_norm_tolerance(dtype_of(directions))
        if worst > tolerance:
            raise ContractError(
                "NON_UNIT_DIRECTION",
                f"direction vectors must be unit norm; worst deviation {worst:.3e} exceeds "
                f"{tolerance:.3e} for {dtype_of(directions)}",
                declaration="directions",
                remedy="Normalize the directions at the producer; do not widen the tolerance.",
            )

        object.__setattr__(
            self, "wavelength_m", require_positive_si(self.wavelength_m, name="wavelength_m")
        )

        if self.phasor != PHASOR:
            raise ContractError(
                "PHASOR_MISMATCH",
                f"phasor must be {PHASOR!r}, got {self.phasor!r}. The conjugate convention "
                "is invisible in any intensity and reverses the sign of every phase.",
                declaration="phasor",
            )

        if self.optical_path_m is not None and not self.optical_path_reference:
            raise ContractError(
                "MISSING_DECLARATION",
                "an optical path length was supplied without declaring its reference",
                declaration="optical_path_reference",
                remedy=(
                    f"State the plane or ray it is measured from, or declare it "
                    f"{UNVERIFIED!r} so a consumer refuses to read it as a phase."
                ),
            )

        if self.ray_splitting not in RAY_SPLITTINGS:
            raise ContractError(
                "UNKNOWN_RAY_SPLITTING",
                f"ray_splitting must be one of {list(RAY_SPLITTINGS)}, got "
                f"{self.ray_splitting!r}",
                declaration="ray_splitting",
            )

        if self.measure_kind not in MEASURE_KINDS:
            raise ContractError(
                "UNKNOWN_MEASURE_KIND",
                f"measure_kind must be one of {list(MEASURE_KINDS)}, got {self.measure_kind!r}",
                declaration="measure_kind",
            )
        if self.measure_weight is not None and self.measure_kind == "undeclared":
            raise ContractError(
                "MEASURE_UNDECLARED",
                "a per-ray measure weight was supplied with measure_kind 'undeclared', so "
                "nothing states what the numbers are. An area element and an importance "
                "weight differ by the aperture area and by whether the reconstruction owes "
                "a 1/N.",
                declaration="measure_kind",
                remedy=f"Declare one of {[k for k in MEASURE_KINDS if k != 'undeclared']}.",
            )
        if self.measure_weight is None and self.measure_kind != "undeclared":
            raise ContractError(
                "MISSING_DECLARATION",
                f"measure_kind is {self.measure_kind!r} but no measure_weight was supplied",
                declaration="measure_weight",
            )
        if self.measure_weight is not None and bool(xp.any(self.measure_weight < 0.0)):
            raise ContractError(
                "UNIT_NOT_SI",
                "measure_weight has negative entries; a quadrature cell has non-negative "
                "area and an importance weight is a reciprocal density",
                declaration="measure_weight",
            )

    @property
    def count(self) -> int:
        return int(self.positions_m.shape[0])

    @property
    def wavenumber(self) -> float:
        """Free-space wavenumber `k = 2 pi / lambda`, in rad/m."""
        return 2.0 * math.pi / self.wavelength_m

    @property
    def state(self) -> ArrayState:
        """Observed dtype, device and namespace, read from `positions_m`.

        Never caller-declared, so the answer cannot contradict the data. The
        amplitude legitimately carries the complex counterpart of this dtype at
        the same precision, which is why the geometry is what is asked.
        """
        return array_state(self.positions_m)

    @property
    def xp(self) -> Any:
        """The array module this bundle's data belongs to."""
        return xp_for(self.state.namespace)

    def require_coherent(self) -> tuple[Any, Any]:
        """Return `(amplitude, optical_path_m)`, or refuse with everything missing named.

        The gate a ray-to-wave coupler passes through before reading either
        quantity as physics. It names *all* the missing declarations rather than
        the first, because a caller that fixes one and re-runs to discover the
        next is a caller that will start guessing.
        """
        missing: list[str] = []
        if self.amplitude is None:
            detail = "no complex amplitude"
            if self.measure_weight is not None:
                detail += (
                    f" (a measure_weight is present, but it is a {self.measure_kind!r} "
                    "sampling weight, not an amplitude)"
                )
            missing.append(f"amplitude: {detail}")
        if self.optical_path_m is None:
            missing.append("optical_path_m: no optical path length")

        if missing:
            raise ContractError(
                "COHERENT_STATE_INCOMPLETE",
                "this bundle cannot be read as coherent -- " + "; ".join(missing),
                declaration=", ".join(item.split(":")[0] for item in missing),
                remedy=(
                    "A ray weight is not a complex amplitude and this type will not choose "
                    "the mapping for you: the producer declares the amplitude and the "
                    "optical path with its reference, or the conversion does not happen."
                ),
            )
        if self.optical_path_reference == UNVERIFIED:
            raise ContractError(
                "OPL_REFERENCE_UNVERIFIED",
                "the optical path length is carried with its reference declared "
                f"{UNVERIFIED!r}, so its sign is not established. A wrong sign conjugates "
                "the wavefront: a converging beam reconstructs as a diverging one, and no "
                "intensity check can tell the difference.",
                declaration="optical_path_reference",
                remedy="Characterize the path against a known geometry before using it as a phase.",
            )
        return self.amplitude, self.optical_path_m
