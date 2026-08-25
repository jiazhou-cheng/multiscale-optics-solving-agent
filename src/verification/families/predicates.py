"""The validity bounds this repository has already measured, made executable.

CHE-131 (M0.5.2). ``core/specs.py::ValiditySpec`` is three lists of strings and
stays as human documentation. These are its executable counterparts: each one
answers *how far* inside or outside an instance sits, on a normalized signed
scale, so "just outside validity" becomes a reachable sampling target rather
than a hope.

Every predicate here has a prior measurement or derivation behind it. None of
them is new physics; what is new is that the bound can now be evaluated instead
of read.

The normalization convention
----------------------------
For an upper bound ``value <= limit``::

    margin = (limit - value) / limit

so ``+1`` is "value is negligible against the limit", ``0`` is exactly at the
boundary and ``-1`` is "twice the limit". For a boolean condition the margin is
``+1`` or ``-1``: there is no meaningful distance, and pretending otherwise
would let a boundary sampler chase a gradient that does not exist.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from core.capabilities import capabilities_for
from core.precision import CapabilityError, DeviceKind, DType
from verification.families.schema import ValidityBasis, ValidityPredicate

__all__ = [
    "asm_transfer_function_sampling",
    "boolean_margin",
    "capability_intersection_nonempty",
    "declared_planarity",
    "fractional_margin",
    "fresnel_number_regime",
    "hexapolar_ring_membership",
    "paraxial_field_angle",
    "per_axis_nyquist_pitch",
    "si_s3_curvature_bound",
]


def fractional_margin(value: float, limit: float) -> float:
    """``(limit - value) / limit`` for an upper bound, guarded at the edges.

    An infinite limit is unbounded headroom and returns ``+inf``; a zero or
    negative limit is not a bound and raises rather than producing a number
    nobody can interpret.
    """
    if math.isinf(limit):
        return math.inf
    if limit <= 0.0 or not math.isfinite(limit):
        raise ValueError(f"a validity limit must be a positive finite number, got {limit!r}")
    return (limit - float(value)) / limit


def boolean_margin(condition: bool) -> float:
    """``+1`` inside, ``-1`` outside.

    Deliberately not ``+0.5``/``-0.5`` or any other magnitude: a boolean
    predicate has no distance, and a sampler asked to find its boundary must
    fail to find a gradient rather than follow a fabricated one.
    """
    return 1.0 if condition else -1.0


# ---------------------------------------------------------------------------
# SI S3 curvature bound
# ---------------------------------------------------------------------------


def si_s3_curvature_bound(
    *,
    error_key: str = "tangent_plane_error_rad",
    width_key: str = "patch_width_m",
    radius_key: str = "substrate_radius_m",
) -> ValidityPredicate:
    """``eps_curv <= arcsin(D / 2R)`` -- SI eq S9, ACS Photonics 2026 eq 4.

    The bound takes only ``D`` and ``R``, which the paper states and
    ``tests/test_curvature_bound.py`` pins structurally: no DOE design can be
    offered as an argument for relaxing it.

    ``R = inf`` is the planar case; the bound is exactly zero and any nonzero
    tangent-plane error is outside it.
    """

    def margin(params: Mapping[str, Any]) -> float:
        radius = float(params[radius_key])
        width = float(params[width_key])
        observed = float(params[error_key])
        if math.isinf(radius):
            return boolean_margin(observed == 0.0)
        ratio = width / (2.0 * radius)
        if ratio >= 1.0:
            # The patch subtends more than the surface can support. Eq S9 gives
            # no bound here, so this is not "far outside" by some amount -- it
            # is a regime where the tangent-plane picture has no meaning.
            return -1.0
        return fractional_margin(observed, math.asin(ratio))

    return ValidityPredicate(
        predicate_id="SI_S3_CURVATURE",
        statement=(
            "the tangent-plane direction error stays under arcsin(D / 2R) for the "
            "declared patch width and substrate radius"
        ),
        basis=ValidityBasis.SI_S3_CURVATURE,
        margin=margin,
        blind_to=(
            "the phase profile written on the patch -- the bound is design-independent, "
            "so satisfying it says nothing about whether the profile is sampled",
        ),
    )


# ---------------------------------------------------------------------------
# Per-axis Nyquist
# ---------------------------------------------------------------------------


def per_axis_nyquist_pitch(
    *,
    pitch_key: str = "sample_pitch_m",
    wavelength_key: str = "wavelength_m",
    max_direction_cosine_key: str = "max_direction_cosine",
) -> ValidityPredicate:
    """``pitch <= lambda / (2 max|d_axis|)`` from the *marginal* ray angles.

    ``benchmarks/probes/slice_feasibility.py`` derives this and reports the
    binding axis; the limit is per axis, and a study that checks only x can be
    infeasible in y with nothing complaining. The parameter here is the largest
    direction cosine over both axes, so the predicate is the binding one.
    """

    def margin(params: Mapping[str, Any]) -> float:
        d_max = float(params[max_direction_cosine_key])
        if d_max <= 0.0:
            return math.inf  # a collimated on-axis bundle imposes no limit
        limit = float(params[wavelength_key]) / (2.0 * d_max)
        return fractional_margin(float(params[pitch_key]), limit)

    return ValidityPredicate(
        predicate_id="PER_AXIS_NYQUIST",
        statement=(
            "the sample pitch resolves the largest marginal direction cosine on the "
            "binding axis: pitch <= lambda / (2 max|d_axis|)"
        ),
        basis=ValidityBasis.PER_AXIS_NYQUIST,
        margin=margin,
        blind_to=(
            "grid extent -- a pitch fine enough to resolve the angles can still be on "
            "a grid too small to hold the field",
            "the other axis, if the caller passed a per-axis value instead of the max",
        ),
    )


# ---------------------------------------------------------------------------
# ASM sampling
# ---------------------------------------------------------------------------


def asm_transfer_function_sampling(
    *,
    distance_key: str = "propagation_distance_m",
    pitch_key: str = "sample_pitch_m",
    grid_key: str = "grid_n",
    wavelength_key: str = "wavelength_m",
) -> ValidityPredicate:
    """``z <= N pitch^2 / lambda`` -- the angular-spectrum transfer function's
    own sampling limit.

    Past this distance the quadratic phase of the ASM kernel aliases on the
    frequency grid and the propagated field wraps. The failure is not loud: the
    result is a plausible-looking field with energy folded back in from the
    other side, which is exactly the class of silent wrongness B1 exists to
    catch.
    """

    def margin(params: Mapping[str, Any]) -> float:
        pitch = float(params[pitch_key])
        n = float(params[grid_key])
        limit = n * pitch * pitch / float(params[wavelength_key])
        return fractional_margin(abs(float(params[distance_key])), limit)

    return ValidityPredicate(
        predicate_id="ASM_TF_SAMPLING",
        statement=(
            "the propagation distance stays inside the angular-spectrum transfer "
            "function's sampling limit z <= N pitch^2 / lambda"
        ),
        basis=ValidityBasis.ASM_SAMPLING,
        margin=margin,
        blind_to=(
            "the evanescent cut -- this bounds aliasing of the propagating band and "
            "says nothing about power lost past the light cone",
            "the aperture's own bandwidth, which can alias before the kernel does",
        ),
    )


# ---------------------------------------------------------------------------
# Capability intersection
# ---------------------------------------------------------------------------


def capability_intersection_nonempty(
    *,
    component_key: str = "component",
    device_key: str = "device",
    dtype_key: str = "dtype",
) -> ValidityPredicate:
    """The requested (device, dtype) is something the component can execute.

    Reads ``core/capabilities.py``, which is the probe-backed source of truth --
    not the registry, which is a downstream reflection of it. Boolean: a dtype a
    package does not have is not "nearly" available.
    """

    def margin(params: Mapping[str, Any]) -> float:
        try:
            caps = capabilities_for(str(params[component_key]))
        except CapabilityError:
            return -1.0
        device = DeviceKind(params[device_key])
        dtype = DType(params[dtype_key])
        ok = device in caps.devices and dtype in caps.native_compute_dtypes
        return boolean_margin(ok)

    return ValidityPredicate(
        predicate_id="CAPABILITY_INTERSECTION",
        statement=(
            "the component's declared capability table contains the requested device "
            "and can compute natively in the requested dtype"
        ),
        basis=ValidityBasis.CAPABILITY_INTERSECTION,
        margin=margin,
        blind_to=(
            "lossy-but-accepted dtypes -- Chromatix ingests complex128 and truncates "
            "it, which this predicate reports as outside rather than as lossy",
        ),
    )


# ---------------------------------------------------------------------------
# Hexapolar ring membership
# ---------------------------------------------------------------------------


def hexapolar_ring_membership(*, sampling_key: str = "pupil_sampling") -> ValidityPredicate:
    """The pupil sampling is hexapolar, as ``NON_HEXAPOLAR_SAMPLING`` requires.

    Boolean, and the contract code exists because the quadrature weight a
    ray carries is derived from its ring, so a non-hexapolar bundle silently
    carries the wrong weights rather than failing.
    """

    def margin(params: Mapping[str, Any]) -> float:
        return boolean_margin(str(params[sampling_key]).lower() == "hexapolar")

    return ValidityPredicate(
        predicate_id="HEXAPOLAR_RING",
        statement="the pupil bundle is sampled on hexapolar rings",
        basis=ValidityBasis.HEXAPOLAR_RING,
        margin=margin,
        blind_to=(
            "the ring count -- being hexapolar says nothing about whether the rings "
            "resolve the pupil",
        ),
    )


# ---------------------------------------------------------------------------
# Declared planarity
# ---------------------------------------------------------------------------


def declared_planarity(
    *,
    sag_key: str = "surface_sag_m",
    tolerance_key: str = "planarity_tolerance_m",
) -> ValidityPredicate:
    """The surface is planar to within the declared tolerance.

    ``C_PLANAR_DOE_STEP`` and ``C_PATCH_WFT`` both declare planarity and neither
    checks it, so the sag that would invalidate the step is currently a prose
    assumption. Expressing it here is what makes the assumption samplable.
    """

    def margin(params: Mapping[str, Any]) -> float:
        return fractional_margin(abs(float(params[sag_key])), float(params[tolerance_key]))

    return ValidityPredicate(
        predicate_id="DECLARED_PLANARITY",
        statement="the surface sag stays inside the coupler's declared planarity tolerance",
        basis=ValidityBasis.DECLARED_PLANARITY,
        margin=margin,
        blind_to=("local slope -- a small peak-to-valley sag can still have a steep facet",),
    )


# ---------------------------------------------------------------------------
# Fresnel number
# ---------------------------------------------------------------------------


def fresnel_number_regime(
    *,
    aperture_radius_key: str = "aperture_radius_m",
    distance_key: str = "propagation_distance_m",
    wavelength_key: str = "wavelength_m",
    minimum_fresnel_number: float = 1.0,
) -> ValidityPredicate:
    """``N_F = a^2 / (lambda z) >= N_min`` -- the near-field regime the
    angular-spectrum method is the right tool for.

    Below ``N_min`` the field is in the Fraunhofer regime and a single-step ASM
    on a grid sized for the aperture is spending its samples in the wrong place.
    Expressed as a *lower* bound, so the margin is inverted relative to the
    upper-bound convention and normalized the same way.
    """

    def margin(params: Mapping[str, Any]) -> float:
        a = float(params[aperture_radius_key])
        z = abs(float(params[distance_key]))
        lam = float(params[wavelength_key])
        if z == 0.0:
            return math.inf
        n_f = a * a / (lam * z)
        return (n_f - minimum_fresnel_number) / minimum_fresnel_number

    return ValidityPredicate(
        predicate_id="FRESNEL_NUMBER",
        statement=(f"the Fresnel number a^2/(lambda z) stays at or above {minimum_fresnel_number}"),
        basis=ValidityBasis.FRESNEL_NUMBER,
        margin=margin,
        blind_to=("aberration -- the regime test is geometric and knows nothing about phase",),
    )


# ---------------------------------------------------------------------------
# Paraxial regime
# ---------------------------------------------------------------------------


def paraxial_field_angle(
    *,
    angle_key: str = "field_angle_rad",
    max_angle_rad: float = math.radians(5.0),
) -> ValidityPredicate:
    """The field angle stays where ``sin(theta) ~ theta`` holds to the declared
    accuracy.

    The default 5 degrees is where ``sin`` and ``tan`` differ by 0.4%, which is
    the number ``A1-CHX-03``'s tolerance basis already records as the reason
    that check deliberately does *not* separate ``z sin(theta)`` from
    ``z tan(theta)``. A closed form derived paraxially cannot gate outside this.
    """

    def margin(params: Mapping[str, Any]) -> float:
        return fractional_margin(abs(float(params[angle_key])), max_angle_rad)

    return ValidityPredicate(
        predicate_id="PARAXIAL_FIELD_ANGLE",
        statement=f"the field angle stays within {math.degrees(max_angle_rad):.1f} degrees",
        basis=ValidityBasis.PARAXIAL_APPROXIMATION,
        margin=margin,
        blind_to=(
            "aperture -- a paraxial field angle with a fast cone is still outside the "
            "paraxial regime, and this predicate does not see the cone",
        ),
    )
