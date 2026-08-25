"""Closed forms that were verified against the pinned solver, kept as oracles.

CHE-133 (M0.5.4). Five of these shipped inside the retired ``A1-*`` agent task
set, where each one was a ``CheckSpec.expected``. Deleting the task set would
have deleted them, and each was verified against the pinned solver *before* the
task shipped -- which is the expensive part and the part that does not need
repeating.

What is preserved here, and why the shape is what it is
-------------------------------------------------------
A closed form on its own is a formula anybody can look up. What made these
usable as oracles is the three things recorded beside each one:

* the **measured agreement** with the pinned solver, which is what justifies the
  tolerance rather than a house default;
* the **wrong answer the tolerance rejects**, named, so the threshold can be
  argued with;
* and, where relevant, what the tolerance deliberately does *not* separate.
  ``B1-WAVE-TILT``'s does not distinguish ``z sin(theta)`` from ``z tan(theta)``
  at 5 degrees, because they differ by 0.4% there, and the original task said so.

Each entry names the family that will carry it. None of these is wired into a
family yet -- M1.1 and M1.2 do that -- and ``tests/test_analytic_oracles.py``
checks that every closed form still evaluates and that every destination is a
family id the B1 naming scheme admits.

Nothing here calls a solver. These are the *independent* side of the comparison,
and an oracle that imported the thing it judges would not be one.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

__all__ = ["ANALYTIC_ORACLES", "AnalyticOracle", "oracle_for"]


@dataclass(frozen=True)
class AnalyticOracle:
    """A closed form, its measured agreement, and the family it belongs to."""

    oracle_id: str
    #: The family that will carry this as its oracle. Authored by M1.
    destination_family: str
    #: The component the closed form is an oracle *for*.
    component: str
    statement: str
    #: ``params -> expected value``. Pure arithmetic; imports no solver.
    closed_form: Callable[[Mapping[str, Any]], float]
    unit: str
    #: The measured agreement with the pinned solver, recorded when the oracle
    #: was verified. This is the sentence a tolerance is derived from.
    verified_against_pinned_solver: str
    #: Suggested relative tolerance, and what it rejects.
    rtol: float
    rejects: str
    #: What the tolerance deliberately does not separate, where that is a real
    #: limitation rather than an oversight.
    does_not_separate: str = ""

    def __call__(self, params: Mapping[str, Any]) -> float:
        return float(self.closed_form(params))


# ---------------------------------------------------------------------------
# Ray primitives
# ---------------------------------------------------------------------------

EFL_BFL = AnalyticOracle(
    oracle_id="THICK_SINGLET_EFL_BFL",
    destination_family="B1-RAY-EFL",
    component="M_RAY_OPTILAND",
    statement=(
        "for a plano-convex singlet in air the effective focal length is R/(n-1) and "
        "the back focal length from the rear vertex is EFL - t/n"
    ),
    closed_form=lambda p: float(p["radius_mm"]) / (float(p["index"]) - 1.0),
    unit="mm",
    verified_against_pinned_solver=(
        "R/(n-1) is exact for a single refracting surface in air; the pinned solver "
        "reproduces it to 1e-13 relative (measured)"
    ),
    rtol=1e-6,
    rejects=(
        "1e-6 admits only a genuinely different answer. The discriminating wrong "
        "answer is reporting the EFL twice: the thick-lens correction EFL - t/n "
        "differs by 2.64 mm on the reference prescription, so an agent that omits it "
        "fails the BFL check and only the BFL check."
    ),
)


def _back_focal_length(p: Mapping[str, Any]) -> float:
    efl = float(p["radius_mm"]) / (float(p["index"]) - 1.0)
    return efl - float(p["thickness_mm"]) / float(p["index"])


BFL = AnalyticOracle(
    oracle_id="THICK_SINGLET_BFL",
    destination_family="B1-RAY-EFL",
    component="M_RAY_OPTILAND",
    statement="EFL - t/n, exact for a plano rear surface",
    closed_form=_back_focal_length,
    unit="mm",
    verified_against_pinned_solver="exact for a plano rear surface; no residual to record",
    rtol=1e-6,
    rejects="reporting the EFL unchanged, which is 2.64 mm away on the reference lens",
)


PLATE_FOCAL_SHIFT = AnalyticOracle(
    oracle_id="PLANE_PARALLEL_PLATE_FOCAL_SHIFT",
    destination_family="B1-RAY-PLATE",
    component="M_RAY_OPTILAND",
    statement=(
        "a plane-parallel plate of thickness t and index n in a converging beam moves "
        "the focus by t(1 - 1/n), positive AWAY from the plate"
    ),
    closed_form=lambda p: float(p["thickness_mm"]) * (1.0 - 1.0 / float(p["index"])),
    unit="mm",
    verified_against_pinned_solver=(
        "t(1 - 1/n) is the paraxial result; a real trace at h = 0.5 mm into f = 100 mm "
        "gives 3.750048 mm (measured), 1.3e-5 relative from the closed form"
    ),
    rtol=1e-3,
    rejects=(
        "1e-3 covers any sane sampling while still rejecting a sign error (the focus "
        "moves away from the plate, so -3.75 fails) and a t/n answer"
    ),
)


# ---------------------------------------------------------------------------
# Wave primitives
# ---------------------------------------------------------------------------


def _gaussian_radius(p: Mapping[str, Any]) -> float:
    w0 = float(p["waist_um"])
    z = float(p["distance_um"])
    lam = float(p["wavelength_um"])
    z_r = math.pi * w0 * w0 / lam
    return w0 * math.sqrt(1.0 + (z / z_r) ** 2)


GAUSSIAN_SPREADING = AnalyticOracle(
    oracle_id="GAUSSIAN_1_OVER_E2_RADIUS",
    destination_family="B1-WAVE-GAUSS",
    component="M_WAVE_CHROMATIX",
    statement="w(z) = w0 sqrt(1 + (z/zR)^2), exact for a paraxial Gaussian",
    closed_form=_gaussian_radius,
    unit="um",
    verified_against_pinned_solver=(
        "the pinned solver's second moment gives 6.040167 um against 6.039084 analytic "
        "(1.8e-4 relative) for w0 = 5 um, z = 100 um, lambda = 0.532 um"
    ),
    rtol=2e-2,
    rejects=(
        "2e-2 absorbs a different but reasonable grid or radius definition while "
        "rejecting the unpropagated waist -- 5.0 um, 17% low, which is what a run that "
        "propagated zero distance or measured the input returns"
    ),
)


AIRY_FIRST_NULL = AnalyticOracle(
    oracle_id="AIRY_FIRST_NULL_RADIUS",
    destination_family="B1-WAVE-AIRY",
    component="M_WAVE_CHROMATIX",
    statement="0.61 lambda / NA is the exact first zero of the focal-plane intensity",
    closed_form=lambda p: 0.61 * float(p["wavelength_um"]) / float(p["numerical_aperture"]),
    unit="um",
    verified_against_pinned_solver=(
        "the pinned solver's focal-plane pitch is 0.83 um, so the null lands between "
        "samples and the measured value is 6.65 um against 6.4985 analytic -- 2.3%, "
        "which is a sampling limit rather than a physics error"
    ),
    rtol=5e-2,
    rejects=(
        "5e-2 covers the sampling limit and still rejects the 1.22 lambda/NA (2x) and "
        "0.5 lambda/NA confusions"
    ),
    does_not_separate=(
        "a genuinely finer focal-plane grid from a correct interpolation of the coarse "
        "one; both land inside 5e-2, which is why B1-WAVE-AIRY owes a convergence "
        "ladder in the grid rather than a single point"
    ),
)


TILTED_BEAM_WALKOFF = AnalyticOracle(
    oracle_id="TILTED_BEAM_LATERAL_WALKOFF",
    destination_family="B1-WAVE-TILT",
    component="M_WAVE_CHROMATIX",
    statement=(
        "z tan(theta) is exact geometry for a collimated beam; the SIGN is part of the "
        "claim"
    ),
    closed_form=lambda p: float(p["distance_um"]) * math.tan(float(p["tilt_rad"])),
    unit="um",
    verified_against_pinned_solver=(
        "the pinned solver gives +17.5017 um against +17.4977 analytic (2.3e-4 "
        "relative) for z = 200 um, theta = 5 degrees"
    ),
    rtol=2e-2,
    rejects=(
        "a sign error, and the 2*pi confusion in `kykx` -- see B0-UNITS-02, which is "
        "the same hazard as a benchmark in its own right"
    ),
    does_not_separate=(
        "z sin(theta) from z tan(theta). At 5 degrees they differ by 0.4%, well inside "
        "2e-2, and the original task said so rather than claiming a separation it does "
        "not have"
    ),
)


ANALYTIC_ORACLES: tuple[AnalyticOracle, ...] = (
    EFL_BFL,
    BFL,
    PLATE_FOCAL_SHIFT,
    GAUSSIAN_SPREADING,
    AIRY_FIRST_NULL,
    TILTED_BEAM_WALKOFF,
)


def oracle_for(oracle_id: str) -> AnalyticOracle:
    for oracle in ANALYTIC_ORACLES:
        if oracle.oracle_id == oracle_id:
            return oracle
    raise KeyError(
        f"no analytic oracle {oracle_id!r}; have {[o.oracle_id for o in ANALYTIC_ORACLES]}"
    )
