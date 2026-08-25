"""It ran, nothing raised, and the number is wrong.

CHE-133 (M0.5.4). Two measured traps shipped inside the retired ``A1-*`` task
set and they are the most valuable thing in it: both are cases where the code
executes perfectly, every boundary check passes, and the physics is wrong. A
benchmark suite that only measures "did it run" can express neither, and a
contract layer that reports ``ok`` for both is telling the truth about the
contract and nothing about the answer.

Both were measured on the pinned versions, not reasoned about. The wrong numbers
below are what the mistaken code actually returns, which is what makes them
usable as negative controls rather than as illustrations.

Destined for ``B0-UNITS-01`` and ``B0-UNITS-02``. M1.3 authors those families;
``tests/test_measured_hazards.py`` holds the numbers until then.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["MEASURED_HAZARDS", "MeasuredHazard", "hazard_for"]


@dataclass(frozen=True)
class MeasuredHazard:
    """A silent wrong answer, with the number it produces.

    ``contract_status`` is ``"ok"`` on both of these and that is the point: the
    boundary layer has nothing to complain about, so a benchmark that stops at
    contract conformance reports success.
    """

    hazard_id: str
    destination_family: str
    component: str
    #: The API surface where the two meanings meet.
    api: str
    description: str
    #: What the mistaken call returns, measured.
    wrong_value: float
    #: What the correct call returns, measured, so the separation is visible.
    right_value: float
    #: Why nothing catches it.
    why_silent: str
    #: What the correct call is.
    remedy: str
    evidence: tuple[str, ...]
    contract_status: str = "ok"

    @property
    def relative_separation(self) -> float:
        """How far apart the right and wrong answers are, relatively.

        Small here is the alarming case, not the reassuring one: ``B0-UNITS-01``
        separates by 1.7e-5, which is why the coating "working" and the coating
        doing nothing look like the same reflectance.
        """
        return abs(self.wrong_value - self.right_value) / abs(self.right_value)


UNITS_MICROMETRE_NANOMETRE = MeasuredHazard(
    hazard_id="OPTILAND_ADD_LAYER_UM_NM",
    destination_family="B0-UNITS-01",
    component="M_RAY_OPTILAND",
    api="optiland.materials.ThinFilmStack.add_layer",
    description=(
        "`add_layer` takes MICROMETRES while the AR-coating literature and the upstream "
        "tutorial quote a quarter-wave MgF2 layer for 550 nm as 99.64 nanometres. "
        "Passing 99.64 builds a layer 1000x too thick."
    ),
    #: Normal-incidence reflectance with the 1000x-too-thick layer.
    wrong_value=0.04216384,
    #: Bare glass, for comparison. The coating does nothing.
    right_value=0.04216456,
    why_silent=(
        "nothing raises -- 99.64 um is a physically constructible layer -- and the "
        "returned number looks exactly like a reflectance. The coated result differs "
        "from BARE GLASS in the eighth decimal place, so a reader checking that "
        "'the coating gives a small reflectance' sees what they expected."
    ),
    remedy=(
        "pass 0.09964 (micrometres). The correctly coated reflectance is 0.01283544 "
        "against 0.01283537 analytic -- a factor of 3.3 below bare glass, which is "
        "what an AR coating is supposed to look like."
    ),
    evidence=(
        "CHE-57 recorded the same unit hazard on upstream tutorial t07",
        "benchmarks/reports/2026-08/agent_benchmark_v1.md",
    ),
)


KYKX_TWO_PI_AND_SIGN = MeasuredHazard(
    hazard_id="CHROMATIX_KYKX_2PI_AND_SIGN",
    destination_family="B0-UNITS-02",
    component="M_WAVE_CHROMATIX",
    api="chromatix.functional.asm_propagate / chromatix.functional.plane_wave",
    description=(
        "`kykx` means CYCLES per length on `asm_propagate` and RADIANS per length on "
        "`plane_wave` -- the same parameter name, a factor of 2*pi apart -- and the "
        "resulting displacement runs OPPOSITE in sign to the parameter."
    ),
    #: The centroid a run that took the radians-per-length reading produces, for
    #: a 5-degree tilt over 200 um. Both wrong by 2*pi and wrong in sign.
    wrong_value=-200.0 * 0.08748866352592401 / (2.0 * 3.141592653589793),
    #: z tan(5 deg) over 200 um: +17.4977 um.
    right_value=200.0 * 0.08748866352592401,
    why_silent=(
        "neither mistake raises. A 6.28x-too-small displacement is a plausible-looking "
        "beam that has simply not walked very far, and the sign flip is invisible in "
        "any magnitude-only report."
    ),
    remedy=(
        "read the convention off the function being called, not off the parameter "
        "name. knowledge/solvers/chromatix/conventions.md names both hazards."
    ),
    evidence=(
        "CHE-57 finding on upstream example c06",
        "knowledge/solvers/chromatix/conventions.md",
    ),
)


MEASURED_HAZARDS: tuple[MeasuredHazard, ...] = (
    UNITS_MICROMETRE_NANOMETRE,
    KYKX_TWO_PI_AND_SIGN,
)


def hazard_for(hazard_id: str) -> MeasuredHazard:
    for hazard in MEASURED_HAZARDS:
        if hazard.hazard_id == hazard_id:
            return hazard
    raise KeyError(
        f"no measured hazard {hazard_id!r}; have {[h.hazard_id for h in MEASURED_HAZARDS]}"
    )
