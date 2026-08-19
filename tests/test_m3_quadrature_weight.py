"""CHE-47 (M3.9R extension): the per-ray quadrature weight, in production.

Two kinds of test, as in ``test_m3r_sensor_handoff.py``.

The live ones exercise the pure math (``multiscale_optics_agent.couplers.
quadrature``) and the adapter/handoff wiring on a cheap trace, so they are fast
and need no pre-computed record.

The rest read ``benchmarks/probes/records/m3_quadrature_weight.json``. They pin
what the extension found -- including that the real (residually aberrated)
system does NOT reach the 1e-3 gate the synthetic diagnostic reached, because a
test suite that only pinned the favorable number would let a later change
quietly claim more than CHE-47 actually verified.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
RECORD_PATH = ROOT / "benchmarks" / "probes" / "records" / "m3_quadrature_weight.json"
GATE = 1.0e-3

optiland = pytest.importorskip("optiland")

from multiscale_optics_agent.adapters.base import ModelRunRequest  # noqa: E402
from multiscale_optics_agent.adapters.optiland_adapter import get_adapter  # noqa: E402
from multiscale_optics_agent.couplers.optiland_handoff import (  # noqa: E402
    AMPLITUDE_MAPPING,
    AMPLITUDE_MAPPING_WITH_QUADRATURE_WEIGHT,
    DeclaredHandoffPlane,
    HandoffPerturbation,
    declare_coherent_bundle,
)


@pytest.fixture(scope="module")
def record() -> dict[str, Any]:
    if not RECORD_PATH.exists():
        pytest.skip(f"{RECORD_PATH.relative_to(ROOT)} is missing; run the probe first")
    return json.loads(RECORD_PATH.read_text())


def _trace(tmp_path_factory, num_rays: int):
    out = tmp_path_factory.mktemp(f"che47-{num_rays}")
    result = get_adapter().run(
        ModelRunRequest(
            run_id="che47",
            node_id="lens",
            config={
                "sample": "M3SingletRef",
                "num_rays": num_rays,
                "wavelength": 0.55,
                "Hx": 0.0,
                "Hy": 0.0,
                "handoff_plane": "exit_pupil",
                "output_directory": str(out),
            },
        )
    )
    assert result.status.value == "succeeded", (result.error_type, result.error_message)
    return result.outputs["rays"]


# ---------------------------------------------------------------------------
# Live: the wiring, on a cheap trace
# ---------------------------------------------------------------------------


def test_quadrature_weight_is_applied_by_default_on_an_unvignetted_hexapolar_trace(
    tmp_path_factory,
):
    rays = _trace(tmp_path_factory, 8)
    plane = DeclaredHandoffPlane("exit_pupil", rays.metadata["conventions"]["reference_plane_z_m"])
    handoff = declare_coherent_bundle(rays, declared_plane=plane)

    assert handoff.declarations["quadrature_weight"]["status"] == "applied"
    assert handoff.declarations["amplitude_mapping"] == AMPLITUDE_MAPPING_WITH_QUADRATURE_WEIGHT
    assert handoff.diagnostics["quadrature_weight_applied"] is True
    assert handoff.diagnostics["quadrature_weight_sum_m2"] is not None
    assert handoff.diagnostics["quadrature_weight_sum_m2"] > 0.0

    # The area weight sums to the aperture area, not to something that grows
    # with ray count: pi * a^2 for EPD/2 = a.
    aperture_radius_m = rays.metadata["conventions"]["entrance_pupil_diameter_m"] / 2.0
    expected_area_m2 = math.pi * aperture_radius_m**2
    assert handoff.diagnostics["quadrature_weight_sum_m2"] == pytest.approx(
        expected_area_m2, rel=0.01
    )


def test_apply_quadrature_weight_false_reproduces_the_pre_che47_declaration(tmp_path_factory):
    rays = _trace(tmp_path_factory, 8)
    plane = DeclaredHandoffPlane("exit_pupil", rays.metadata["conventions"]["reference_plane_z_m"])
    legacy = declare_coherent_bundle(
        rays, declared_plane=plane, perturbation=HandoffPerturbation(apply_quadrature_weight=False)
    )

    assert legacy.declarations["quadrature_weight"]["status"] == (
        "available, OMITTED -- negative test only"
    )
    assert legacy.declarations["amplitude_mapping"] == AMPLITUDE_MAPPING
    amplitude, _ = legacy.bundle.require_coherent()
    np.testing.assert_allclose(amplitude.imag, 0.0)
    np.testing.assert_allclose(amplitude.real, np.sqrt(legacy.bundle.weight))


def test_the_perturbation_flag_is_part_of_is_identity_and_describe():
    assert HandoffPerturbation().is_identity
    assert HandoffPerturbation().describe() == "none"
    assert (
        HandoffPerturbation(apply_quadrature_weight=False).describe() == "quadrature_weight_omitted"
    )
    assert not HandoffPerturbation(apply_quadrature_weight=False).is_identity


# ---------------------------------------------------------------------------
# Record: the real-system sensor-plane result
# ---------------------------------------------------------------------------


def test_the_gate_is_not_met_on_the_real_aberrated_system(record):
    """The honest negative half of this ticket's result.

    CHE-38's synthetic aberration-free diagnostic reached 4.07e-4. On the real
    traced M3-SINGLET-REF system this extension reaches ~2.2e-3 to 2.5e-3 --
    real progress (see the improvement-factor test below), but still ABOVE the
    1e-3 gate. Pinned as a failure so a later change cannot quietly claim the
    gate is met here without re-deriving why.
    """
    finest = record["finest_configuration"]
    assert finest["gate_met"] is False
    assert finest["weighted_vs_o2_asm"] > GATE
    assert finest["weighted_vs_o2_asm"] < 5.0e-3


def test_the_weighted_result_improves_on_uniform_weights_vs_o2_diagnostic_only(record):
    """O2 is a custom oracle we wrote; it never decides pass/fail (see
    test_the_weighted_result_does_not_improve_on_uniform_weights_vs_o1 for the
    gate-deciding comparison). This pins the O2 number purely as
    characterization evidence, so a future re-run cannot silently drop it.
    """
    finest = record["finest_configuration"]
    # CHE-38 measured 3.84e-3 at 787969 rays with uniform weights; this run's
    # own uniform baseline must reproduce that order of magnitude...
    assert finest["uniform_vs_o2_asm"] == pytest.approx(3.91e-3, rel=0.1)
    # ...and the weighted result must be a real, substantial improvement over
    # it -- not noise-level, not merely a different tolerance.
    assert finest["improvement_factor_vs_o2_asm_diagnostic_only"] > 1.3
    assert finest["weighted_vs_o2_asm"] < finest["uniform_vs_o2_asm"]


def test_the_weighted_result_does_not_improve_on_uniform_weights_vs_o1(record):
    """The gate-deciding oracle (O1, analytic Airy) shows the OPPOSITE
    ordering from O2: applying the production quadrature weight makes the
    sensor-plane residual worse, not better, on the real aberrated system.
    Pinned so this counterintuitive, already-investigated finding is not lost
    or silently "fixed" by a future change without re-deriving why (see
    test_o1_is_closer_than_o2_which_points_at_the_o2_oracle_not_the_coupler).
    """
    finest = record["finest_configuration"]
    assert finest["improvement_factor_vs_o1"] < 1.0
    assert finest["weighted_vs_o1"] > finest["uniform_vs_o1"]
    assert finest["uniform_vs_o1"] < GATE


def test_o1_is_closer_than_o2_which_points_at_the_o2_oracle_not_the_coupler(record):
    """The argument that the residual is not (mainly) a coupler defect.

    O2 is built from the SAME traced, aberrated wavefront the coupler
    reconstructs; O1 is an aberration-free analytic Airy pattern sharing no
    code or data with the trace. If the leftover residual were a coupler-side
    aberration/quadrature defect, O2 should track the coupler's output more
    closely than O1 does. It does not -- pinned here so that finding is not
    lost to a future re-run's noise.
    """
    finest = record["finest_configuration"]
    assert finest["weighted_vs_o1"] < finest["weighted_vs_o2_asm"]


def test_absolute_power_converges_instead_of_growing_as_n_squared(record):
    absolute_power = record["absolute_power"]
    # CHE-33/CHE-38's uniform-weight finding, reproduced here as the baseline
    # this extension is scored against.
    assert absolute_power["uniform_power_fit"]["exponent"] == pytest.approx(2.0, abs=0.1)
    assert absolute_power["uniform_power_fit"]["r_squared"] > 0.99
    # The resolution: weighted power is flat in ray count, not quadratic.
    assert abs(absolute_power["weighted_power_fit"]["exponent"]) < 0.1
    assert absolute_power["weighted_power_relative_spread_from_1801_rays"] < 0.01
    assert record["verdict"]["absolute_power_converged"] is True


def test_sensor_ladder_ring_counts_match_che38s_own_ladder(record):
    rings = [row["rings"] for row in record["sensor_ladder"]]
    assert rings == [8, 16, 24, 32, 48, 64, 96, 128, 181, 256, 362, 512]
    traced_rays = [row["traced_rays"] for row in record["sensor_ladder"]]
    assert traced_rays[-1] == 787969
    # Every row must have actually applied the weight (an un-vignetted
    # hexapolar fan at every rung of this ladder, same as CHE-38 measured).
    for row in record["sensor_ladder"]:
        assert row["weighted"]["quadrature_weight_status"] == "applied"
        assert row["uniform"]["quadrature_weight_status"] == (
            "available, OMITTED -- negative test only"
        )
