"""The scientific evidence that had to survive the deletion of the task layer.

CHE-133 (M0.5.4). The reviewer's job on that change is the inverse of the usual
one -- confirm nothing in the preserve list was lost, not that the deletions
happened -- and this file is the executable half of that.

Five closed forms and two measured traps shipped inside the retired ``A1-*`` task
set. Each closed form had been verified against the pinned solver before the task
shipped, which is the expensive part; each trap is a case where the code runs
perfectly, the contract reports ``ok`` and the physics is wrong, which is the
most valuable shape a benchmark can have. Deleting the tasks without these would
have deleted both.
"""

from __future__ import annotations

import math

import pytest

from verification.analytic import ANALYTIC_ORACLES, AnalyticOracle, oracle_for
from verification.hazards import MEASURED_HAZARDS, MeasuredHazard, hazard_for

# --------------------------------------------------------------------------- #
# The five promoted closed forms
# --------------------------------------------------------------------------- #

#: The families M1.1 and M1.2 owe. Written out rather than derived from the
#: oracles, so that dropping an oracle fails here instead of shrinking the list.
PROMISED_DESTINATIONS = {
    "B1-RAY-EFL",
    "B1-RAY-PLATE",
    "B1-WAVE-GAUSS",
    "B1-WAVE-AIRY",
    "B1-WAVE-TILT",
}


def test_every_promised_destination_still_has_an_oracle() -> None:
    have = {oracle.destination_family for oracle in ANALYTIC_ORACLES}
    assert have >= PROMISED_DESTINATIONS, (
        "a closed form promoted out of the retired task set has no oracle left: "
        f"{sorted(PROMISED_DESTINATIONS - have)}"
    )


@pytest.mark.parametrize("oracle", ANALYTIC_ORACLES, ids=lambda o: o.oracle_id)
def test_every_oracle_records_what_verified_it_and_what_it_rejects(
    oracle: AnalyticOracle,
) -> None:
    """A closed form without its measured agreement is a formula, not an oracle.

    The tolerance is derived from that measurement, so losing it means the next
    person cannot argue with the threshold or re-derive it.
    """
    assert oracle.verified_against_pinned_solver.strip()
    assert oracle.rejects.strip()
    assert oracle.rtol > 0.0


def test_the_thick_singlet_closed_forms_still_evaluate() -> None:
    """R/(n-1) and EFL - t/n on the reference prescription.

    The 2.64 mm separation is the number that makes the second check
    independent of the first: an agent that reports the EFL twice fails the BFL
    check and only the BFL check.
    """
    params = {"radius_mm": 25.0, "index": 1.5168, "thickness_mm": 4.0}
    efl = oracle_for("THICK_SINGLET_EFL_BFL")(params)
    bfl = oracle_for("THICK_SINGLET_BFL")(params)
    assert efl == pytest.approx(25.0 / 0.5168)
    assert bfl == pytest.approx(25.0 / 0.5168 - 4.0 / 1.5168)
    assert efl - bfl == pytest.approx(2.637, abs=1e-3)


def test_the_plate_focal_shift_keeps_its_sign() -> None:
    """A plate in a converging beam moves the focus AWAY from it. Reporting
    -3.75 was a failure in the original task and stays one."""
    shift = oracle_for("PLANE_PARALLEL_PLATE_FOCAL_SHIFT")(
        {"thickness_mm": 10.0, "index": 1.6}
    )
    assert shift == pytest.approx(3.75)
    assert shift > 0.0


def test_the_gaussian_oracle_reproduces_its_measured_agreement() -> None:
    """6.039084 um analytic against the recorded 6.040167 um from the pinned
    solver -- 1.8e-4 relative, which is what the 2e-2 tolerance is sized from."""
    value = oracle_for("GAUSSIAN_1_OVER_E2_RADIUS")(
        {"waist_um": 5.0, "distance_um": 100.0, "wavelength_um": 0.532}
    )
    assert value == pytest.approx(6.039084, abs=1e-5)
    assert abs(6.040167 - value) / value == pytest.approx(1.8e-4, rel=0.1)


def test_the_airy_oracle_reproduces_its_measured_agreement() -> None:
    """6.4985 um analytic against the recorded 6.65 um -- a 2.3% sampling limit,
    not a physics error, and the reason B1-WAVE-AIRY owes a grid ladder."""
    na = 20.0 / math.hypot(20.0, 400.0)
    value = oracle_for("AIRY_FIRST_NULL_RADIUS")(
        {"wavelength_um": 0.532, "numerical_aperture": na}
    )
    assert value == pytest.approx(6.4985, abs=1e-3)
    assert abs(6.65 - value) / value == pytest.approx(2.3e-2, rel=0.15)


def test_the_tilt_oracle_keeps_the_separation_it_does_not_claim() -> None:
    """z tan(theta) against z sin(theta) at 5 degrees differ by 0.4%, inside the
    2e-2 tolerance. The oracle says so rather than claiming a separation it does
    not have."""
    tilt = math.radians(5.0)
    value = oracle_for("TILTED_BEAM_LATERAL_WALKOFF")({"distance_um": 200.0, "tilt_rad": tilt})
    assert value == pytest.approx(17.4977, abs=1e-3)

    sine_answer = 200.0 * math.sin(tilt)
    assert abs(sine_answer - value) / value == pytest.approx(0.0038, abs=5e-4)
    assert oracle_for("TILTED_BEAM_LATERAL_WALKOFF").does_not_separate.strip()


def test_no_oracle_imports_a_solver() -> None:
    """The independent side of a comparison cannot import the thing it judges."""
    from core.paths import repository_root

    source = (repository_root() / "src/verification/analytic.py").read_text(encoding="utf-8")
    for package in ("optiland", "chromatix", "jax", "torch"):
        assert package not in source, (
            f"verification/analytic.py mentions {package}. These are closed forms; an "
            "oracle that reaches for the solver is not one."
        )


# --------------------------------------------------------------------------- #
# The two measured traps
# --------------------------------------------------------------------------- #

PROMISED_HAZARD_DESTINATIONS = {"B0-UNITS-01", "B0-UNITS-02"}


def test_both_measured_traps_survive_with_their_destinations() -> None:
    have = {hazard.destination_family for hazard in MEASURED_HAZARDS}
    assert have == PROMISED_HAZARD_DESTINATIONS


@pytest.mark.parametrize("hazard", MEASURED_HAZARDS, ids=lambda h: h.hazard_id)
def test_a_measured_trap_reports_an_ok_contract(hazard: MeasuredHazard) -> None:
    """Both of these pass every boundary check. That is what makes them the
    archetype: a benchmark that stops at contract conformance calls them fine."""
    assert hazard.contract_status == "ok"
    assert hazard.why_silent.strip()
    assert hazard.remedy.strip()
    assert hazard.evidence


def test_the_micrometre_nanometre_slip_keeps_its_numbers() -> None:
    """0.04216384 coated against bare glass's 0.04216456.

    They differ in the eighth decimal place -- 1.7e-5 relative -- so the coating
    doing nothing and the coating working look like the same reflectance. The
    small separation IS the hazard.
    """
    hazard = hazard_for("OPTILAND_ADD_LAYER_UM_NM")
    assert hazard.wrong_value == 0.04216384
    assert hazard.right_value == 0.04216456
    assert hazard.relative_separation < 1e-4, (
        "if this ever grows, the trap has stopped being silent and the family's "
        "premise needs revisiting"
    )
    # The correctly coated value, quoted in the remedy, is 3.3x below bare glass.
    assert (0.04216456 / 0.01283544) == pytest.approx(3.285, abs=1e-2)  # noqa: SIM300


def test_the_kykx_hazard_keeps_both_the_factor_and_the_sign() -> None:
    """A factor of 2*pi and a sign flip, both measured, neither raising."""
    hazard = hazard_for("CHROMATIX_KYKX_2PI_AND_SIGN")
    right = 200.0 * math.tan(math.radians(5.0))
    assert hazard.right_value == pytest.approx(right, rel=1e-6)
    assert hazard.wrong_value == pytest.approx(-right / (2.0 * math.pi), rel=1e-6)
    assert hazard.wrong_value < 0.0 < hazard.right_value, "the sign flip is half the trap"
    assert abs(hazard.right_value / hazard.wrong_value) == pytest.approx(2.0 * math.pi, rel=1e-6)
