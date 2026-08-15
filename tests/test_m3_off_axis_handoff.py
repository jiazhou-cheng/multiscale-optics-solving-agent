"""CHE-41: the off-axis OPL reference, and the geometric oracle for it.

Three kinds of test, deliberately separated.

**Live, against the real trace.** The refusal path, the perturbation hook, and the
one claim that has to hold bit-exactly rather than approximately: on axis the
declared OPL is *unchanged* by this ticket, because the object-space term is a
constant there and a constant cannot survive the chief-ray subtraction. That is
asserted with ``numpy.array_equal`` on purpose. "Within tolerance" would leave open
exactly the question the M3.4-M3.8 records depend on.

**Against ``knowledge/solvers/optiland/expected/off_axis_opd_reference.json``.**
What Optiland's reference surface actually is, established off axis where the
question is answerable at all.

**Against ``benchmarks/probes/records/m3_off_axis_handoff.json``.** The geometric
oracle: the PSF must land at the traced chief-ray intersection and the fitted
reference sphere must be centred there. Both are geometry; neither is a tolerance.
Running the full slice twice plus a transpose control inside the suite would cost
minutes, so the probe is the evidence and these assertions pin what it found --
including the finding that CHE-37's transpose *metric* is blind for a second
reason, which is pinned as a failure so it cannot go green quietly.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from conftest import load_probe_expected

from multiscale_optics_agent.couplers.contracts import ContractCode, ContractError

ROOT = Path(__file__).resolve().parents[1]
RECORD_PATH = ROOT / "benchmarks" / "probes" / "records" / "m3_off_axis_handoff.json"

WAVELENGTH_UM = 0.55
MM_PER_M = 1e-3
REVERSE_TELEPHOTO_PUPIL_Z_M = 2.1547825721481666 * MM_PER_M

pytest.importorskip("optiland")

from multiscale_optics_agent.adapters.base import ModelRunRequest  # noqa: E402
from multiscale_optics_agent.adapters.optiland_adapter import get_adapter  # noqa: E402
from multiscale_optics_agent.couplers.optiland_handoff import (  # noqa: E402
    OPL_REFERENCE_VERSION,
    SUPERSEDED_OPL_REFERENCE,
    DeclaredHandoffPlane,
    HandoffPerturbation,
    declare_coherent_bundle,
)


@pytest.fixture(scope="module")
def record() -> dict:
    if not RECORD_PATH.exists():
        pytest.skip(f"{RECORD_PATH.relative_to(ROOT)} is missing; run the probe first")
    return json.loads(RECORD_PATH.read_text())


@pytest.fixture(scope="module")
def solver_expected() -> dict:
    return load_probe_expected("optiland", "off_axis_opd_reference")


def _trace(tmp_path_factory, hy: float):
    out = tmp_path_factory.mktemp(f"che41-hy{hy}")
    result = get_adapter().run(
        ModelRunRequest(
            run_id="che41",
            node_id=f"rt-hy{hy}",
            config={
                "sample": "ReverseTelephoto",
                "num_rays": 8,
                "wavelength": WAVELENGTH_UM,
                "Hx": 0.0,
                "Hy": hy,
                "handoff_plane": "exit_pupil",
                "output_directory": str(out),
            },
        )
    )
    assert result.status.value == "succeeded", (result.error_type, result.error_message)
    return result.outputs["rays"]


@pytest.fixture(scope="module")
def off_axis_record(tmp_path_factory):
    return _trace(tmp_path_factory, 0.2)


@pytest.fixture(scope="module")
def on_axis_record(tmp_path_factory):
    return _trace(tmp_path_factory, 0.0)


def _declare(record, **kwargs):
    return declare_coherent_bundle(
        record,
        declared_plane=DeclaredHandoffPlane("exit_pupil", REVERSE_TELEPHOTO_PUPIL_Z_M),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Live: the export, the refusal, and on-axis bit-identity
# ---------------------------------------------------------------------------


def test_the_ray_adapter_exports_the_object_space_reference(off_axis_record) -> None:
    """The term Optic.trace throws away, measured from the regenerated launch state."""
    conventions = off_axis_record.metadata["conventions"]
    declaration = conventions["object_space_reference"]

    assert declaration["available"] is True
    assert declaration["array"] == "object_space_reference_offset_m"
    assert declaration["unit"] == "m"
    assert declaration["launch_geometry"] == (
        "collimated_bundle_launched_on_a_plane_perpendicular_to_z"
    )
    assert declaration["object_space_refractive_index"] == 1.0
    # sin(6 deg): Hy = 0.2 of a 30-degree field.
    assert declaration["launch_direction"][1] == pytest.approx(0.10452846326765348, rel=1e-12)
    assert declaration["launch_direction"][0] == 0.0
    assert declaration["span_m"] > 0.0

    arrays = dict(np.load(off_axis_record.uri))
    assert "object_space_reference_offset_m" in arrays
    assert arrays["object_space_reference_offset_m"].shape == arrays["x_m"].shape
    assert float(np.ptp(arrays["object_space_reference_offset_m"])) == pytest.approx(
        declaration["span_m"], rel=1e-12
    )

    # The reference surface is now stated, which is what CHE-30 left implicit.
    assert "PERPENDICULAR TO Z" in conventions["opd_reference_surface"]
    assert conventions["opd_omits_incoming_wavefront_tilt"] is True
    # And the chief-ray flag is unchanged: it was never the thing that was wrong.
    assert conventions["opd_is_relative_to_chief_ray"] is False


def test_the_traced_ray_hash_does_not_move(off_axis_record, on_axis_record) -> None:
    """`scientific_array_sha256` covers the traced ray set, and CHE-41 adds to neither.

    The object-space arrays travel in the same file under their own hash. Folding
    them into the traced set's fingerprint would move an identity CHE-32 pinned and
    L1-RAY-01 depends on, for a change that alters no traced ray.
    """
    from multiscale_optics_agent.adapters.optiland_adapter import _scientific_array_hash

    for record in (off_axis_record, on_axis_record):
        arrays = dict(np.load(record.uri))
        traced = {
            name: arrays[name]
            for name in (
                "x_m",
                "y_m",
                "z_m",
                "L",
                "M",
                "N",
                "intensity",
                "wavelength_m",
                "opd_native",
                "survived",
            )
        }
        assert _scientific_array_hash(traced) == record.metadata["scientific_array_sha256"]
        object_space = {
            name: arrays[name]
            for name in (
                "object_space_reference_offset_m",
                "launch_x_m",
                "launch_y_m",
                "launch_z_m",
            )
        }
        assert (
            _scientific_array_hash(object_space)
            == record.metadata["conventions"]["object_space_reference"]["sha256"]
        )


def test_on_axis_the_declared_opl_is_bit_identical_to_the_superseded_version(
    on_axis_record,
) -> None:
    """The one claim that must be exact, not close.

    Every number M3.4-M3.8 verified was measured under v1. If v2 moved them even by
    a rounding, they would all need re-verifying rather than asserting -- so the
    coupler does not add a term whose span is zero, and this is the check that the
    span really is zero and the skip really happens.
    """
    v2 = _declare(on_axis_record)
    v1 = _declare(
        on_axis_record,
        perturbation=HandoffPerturbation(reference_incoming_wavefront=False),
    )

    assert v2.diagnostics["object_space_reference_span_m"] == 0.0
    assert v2.diagnostics["object_space_reference_applied"] is False
    assert "pure piston" in v2.diagnostics["object_space_reference_status"]
    assert np.array_equal(
        np.asarray(v1.bundle.optical_path_length_m),
        np.asarray(v2.bundle.optical_path_length_m),
    )


def test_off_axis_the_term_is_applied_and_moves_the_declared_opl(off_axis_record) -> None:
    v2 = _declare(off_axis_record)
    v1 = _declare(
        off_axis_record,
        perturbation=HandoffPerturbation(reference_incoming_wavefront=False),
    )

    assert v2.diagnostics["object_space_reference_applied"] is True
    assert v2.diagnostics["object_space_reference_span_waves"] > 50.0
    assert v2.diagnostics["opl_reference_version"] == OPL_REFERENCE_VERSION
    assert v1.diagnostics["perturbation"] == "incoming_wavefront_reference_omitted"

    difference = np.asarray(v2.bundle.optical_path_length_m) - np.asarray(
        v1.bundle.optical_path_length_m
    )
    assert float(np.ptp(difference)) / v2.bundle.wavelength_m > 50.0
    # The difference is a pure tilt in the launch coordinate, so it is linear in the
    # pupil coordinate to the accuracy of the pupil mapping -- and it is NOT zero,
    # which is the whole finding.
    assert not np.allclose(difference, difference[0])


def test_an_off_axis_record_without_the_term_is_refused(off_axis_record) -> None:
    """A structured refusal, not a silent zero and not a warning.

    The missing quantity is object-space information. A consumer that proceeds
    without it gets a clean converging sphere aimed at the axis, which looks
    healthy at 0.072 waves P-V against its own fitted sphere -- so the failure has
    to be raised at the boundary or it will not be noticed at all.
    """
    arrays = dict(np.load(off_axis_record.uri))
    arrays.pop("object_space_reference_offset_m")

    with pytest.raises(ContractError) as refused:
        _declare(off_axis_record, arrays=arrays)

    assert refused.value.code is ContractCode.OBJECT_SPACE_REFERENCE_MISSING
    assert "off-axis" in str(refused.value)
    assert "cannot be repaired downstream" in (refused.value.remedy or "")


def test_an_on_axis_record_without_the_term_is_still_accepted(on_axis_record) -> None:
    """The refusal is field-conditional, because the omission is a piston on axis.

    Every ray record written before CHE-41 lacks the term. Refusing those outright
    would invalidate the on-axis path for a defect that does not exist there.
    """
    arrays = dict(np.load(on_axis_record.uri))
    arrays.pop("object_space_reference_offset_m")

    handoff = _declare(on_axis_record, arrays=arrays)

    assert handoff.diagnostics["object_space_reference_applied"] is False
    assert "on axis" in handoff.diagnostics["object_space_reference_status"]
    assert np.array_equal(
        np.asarray(handoff.bundle.optical_path_length_m),
        np.asarray(_declare(on_axis_record).bundle.optical_path_length_m),
    )


def test_the_declaration_is_versioned_and_the_superseded_one_is_kept() -> None:
    """A handoff convention is part of the contract, so v1 is recorded, not deleted."""
    assert OPL_REFERENCE_VERSION.startswith("C_RAY_TO_WAVE opl_reference v2")
    assert SUPERSEDED_OPL_REFERENCE["version"].endswith("v1 (CHE-33)")
    assert "0.016999" in SUPERSEDED_OPL_REFERENCE["what_it_got_right"]
    assert "0.13%" in SUPERSEDED_OPL_REFERENCE["what_it_got_wrong"]


# ---------------------------------------------------------------------------
# What Optiland's reference surface is
# ---------------------------------------------------------------------------


def test_the_launch_surface_is_a_plane_perpendicular_to_z(solver_expected: dict) -> None:
    """The measurement that separates a plane from a wavefront.

    A wavefront-seeded launch would spread the launch z by tan(theta) * EPD =
    0.0315 mm across the pupil. The measured spread is exactly zero.
    """
    case = solver_expected["cases"]["launch_surface_off_axis"]

    assert solver_expected["status"] == "passed"
    assert case["launch_z_spread_mm"] == 0.0
    assert max(abs(v) for v in case["launch_direction_spread"]) < 1e-15
    assert case["M_minus_sin_field_angle"] == pytest.approx(0.0, abs=1e-15)
    assert case["verdict"] == "plane perpendicular to z"


def test_the_regenerated_launch_state_is_the_traced_one(solver_expected: dict) -> None:
    """Exactness matters here: a near-miss would misalign a per-ray term."""
    case = solver_expected["cases"]["regenerated_launch_state_reproduces_the_trace"]

    assert case["max_abs_x_difference_mm"] == 0.0
    assert case["max_abs_y_difference_mm"] == 0.0
    assert case["max_abs_opd_difference_mm"] == 0.0
    assert case["ray_count"] == 3169


def test_opd_native_is_absolute_off_axis_not_chief_ray_relative(
    solver_expected: dict,
) -> None:
    """The flag CHE-30 could not test on axis, tested."""
    case = solver_expected["cases"]["opd_is_not_chief_ray_relative"]

    assert case["opd_is_relative_to_chief_ray"] is False
    assert case["chief_ray_opd_native_waves"] > 1.0e4


def test_the_omitted_term_is_a_piston_on_axis_and_a_tilt_off_axis(
    solver_expected: dict,
) -> None:
    piston = solver_expected["cases"]["the_term_is_a_piston_on_axis"]
    tilt = solver_expected["cases"]["the_omitted_term_is_the_convergence_tilt"]

    assert piston["span_is_exactly_zero"] is True
    assert piston["correction_span_mm"] == 0.0

    plane = tilt["variants"]["plane_perpendicular_to_z"]
    wavefront = tilt["variants"]["incoming_wavefront"]
    # The defect reproduced: a plane reference aims the sphere at the AXIS.
    assert plane["fitted_centre_distance_from_the_axis_mm"] < 5.0e-3
    assert plane["pv_waves_against_the_traced_chief_ray_point"] > 50.0
    # The fix: the sphere is aimed at the traced chief-ray intersection instead.
    assert wavefront["fitted_centre_distance_from_the_chief_ray_point_mm"] < 1.0e-3
    assert wavefront["pv_waves_against_the_traced_chief_ray_point"] < 0.25
    assert tilt["improvement_in_the_sphere_centre"] > 100.0


def test_the_admissible_pitch_scan_bounds_the_field_the_frozen_grid_supports(
    solver_expected: dict,
) -> None:
    """The cost of putting the tilt in the pupil, measured over real marginal rays."""
    scan = solver_expected["cases"]["admissible_pitch_versus_field"]
    rows = {row["hy"]: row for row in scan["rows"]}

    assert rows[0.2]["admissible"] is True
    assert rows[0.2]["frozen_pitch_over_admissible"] == pytest.approx(0.946, abs=0.005)
    assert rows[0.25]["admissible"] is False
    assert rows[1.0]["refinement_factor_needed"] == pytest.approx(2.58, abs=0.05)
    # Monotone in field, which is what makes "beyond Hy ~ 0.25" a bound rather than
    # a single measurement.
    cosines = [row["max_transverse_direction_cosine"] for row in scan["rows"]]
    assert cosines == sorted(cosines)


# ---------------------------------------------------------------------------
# The geometric oracle
# ---------------------------------------------------------------------------


def test_the_psf_lands_at_the_traced_chief_ray_intersection(record: dict) -> None:
    """The oracle. Geometry, and nothing the reconstruction had a say in."""
    verdict = record["oracle_verdict"]
    position = record["variants"]["declared_v2_incoming_wavefront"]["psf_position"]

    assert verdict["psf_lands_at_the_traced_chief_ray_intersection"] is True
    # Within one pixel of the prediction, which is the peak index's own resolution.
    assert abs(position["error_pixels_y_x"][0]) <= 1.0
    assert position["error_pixels_y_x"][1] == 0.0
    assert position["error_in_airy_radii"] < 0.5
    assert position["measured_peak_offset_pixels_y_x"][0] > 100
    assert position["peak_is_off_axis_in_y_only"] is True


def test_the_reference_sphere_is_centred_on_the_chief_ray_not_on_the_axis(
    record: dict,
) -> None:
    """The second half of the oracle, and the half M3.8's method note is about.

    ``fit_reference_sphere`` solves for the centre by Gauss-Newton rather than
    subtracting a linear ramp, because a lateral shift of the centre is only
    approximately a ramp and the remainder reads as aberration.
    """
    verdict = record["oracle_verdict"]
    declared = record["variants"]["declared_v2_incoming_wavefront"]

    assert verdict["sphere_centre_distance_from_the_chief_ray_point_m"] < 2.0e-6
    assert verdict["sphere_centre_improvement_over_the_superseded_declaration"] > 100.0
    assert verdict["pv_waves_against_the_chief_ray_point"] < 0.25
    assert verdict["superseded_pv_waves_against_the_chief_ray_point"] > 50.0
    assert declared["pupil_wavefront"]["fitted_centre_distance_from_the_axis_m"] > 1.0e-4


def test_the_slope_excess_over_y_over_r_is_the_spheres_own_nonlinearity(
    record: dict,
) -> None:
    """The one number that could look like a residual error, ruled out by measurement.

    The declared OPL's least-squares slope reads 0.19% above ``y_image / R``, which
    is the quantity CHE-37 compared against. That excess is not a reference error:
    ``y/R`` is the sphere's slope at the pupil CENTRE, and an exact analytic sphere
    aimed at the same point, sampled on the same pupil coordinates, shows the same
    excess.
    """
    check = record["variants"]["declared_v2_incoming_wavefront"][
        "slope_cross_check_against_the_exact_sphere"
    ]

    assert check["ideal_sphere_slope_as_fraction_of_y_over_r"] == pytest.approx(1.0018, abs=5e-4)
    assert check["declared_slope_over_ideal_sphere_slope"] == pytest.approx(1.0, abs=1e-3)
    # What is left between them is the system's own aberration, and it is inside the
    # Rayleigh limit.
    assert check["declared_minus_ideal_sphere_pv_waves"] < 0.25


def test_a_reference_sphere_free_measure_agrees(record: dict) -> None:
    """A geometric spot radius, which fits no sphere and cannot inherit its error."""
    spot = record["variants"]["declared_v2_incoming_wavefront"][
        "geometric_spot_about_the_chief_ray_point"
    ]

    assert spot["rms_radius_in_airy_radii"] < 0.25
    assert spot["max_radius_in_airy_radii"] < 0.5


def test_removing_the_term_moves_the_psf_back_onto_the_axis(record: dict) -> None:
    """The negative control: a fix whose omission changes nothing is not a fix."""
    superseded = record["variants"]["superseded_v1_launch_plane"]
    verdict = record["oracle_verdict"]

    assert superseded["handoff_diagnostics"]["object_space_reference_applied"] is False
    assert abs(superseded["psf_position"]["measured_peak_offset_pixels_y_x"][0]) <= 2
    assert superseded["slope_as_fraction_of_required"] < 0.01
    assert abs(verdict["psf_displacement_between_the_two_declarations_pixels"]) > 100


def test_the_declared_choice_is_stated_with_its_reasoning(record: dict) -> None:
    """The ticket's requirement that the choice not be defaulted into."""
    choice = record["declared_choice"]

    assert choice["reference"] == "the incoming tilted wavefront"
    assert "chief-ray-referenced frame" in choice["alternative_rejected"]
    assert len(choice["reasoning"]) >= 4
    assert record["opl_reference_version"] == OPL_REFERENCE_VERSION
    assert record["superseded_opl_reference"]["version"] == SUPERSEDED_OPL_REFERENCE["version"]


def test_the_frozen_pitch_still_admits_this_field(record: dict) -> None:
    sampling = record["sampling"]

    assert sampling["admissible"] is True
    assert sampling["fraction_of_the_admissible_pitch"] == pytest.approx(0.946, abs=0.005)
    assert sampling["frozen_pitch_m"] == 1.8258157981959995e-06


def test_the_transpose_control_is_detected_only_by_a_non_azimuthal_metric(
    record: dict,
) -> None:
    """Two blind spots, not one, and the second is pinned as still failing.

    CHE-37's transpose margin was 1.0000066 and it was attributed to the off-axis
    PSF forming on axis. Half of that was right. With the PSF genuinely 114 pixels
    off axis, CHE-37's own metric still reads ~1.0 -- an azimuthal average about the
    grid centre cannot distinguish (114, 0) from (0, 114) for any configuration at
    all. That number is asserted to remain ~1.0 so that a later change to the metric
    surfaces here rather than looking like a physics improvement.
    """
    control = record["axis_transpose_control"]

    assert control["status"] == "ran"
    assert control["unperturbed_peak_offset_pixels_y_x"] == [114, 0]
    assert control["transposed_peak_offset_pixels_y_x"] == [0, 114]
    assert control["transpose_moves_the_peak_from_y_to_x"] is True
    assert control["peak_displacement_in_airy_radii"] > 10.0

    che37 = control["che37_metric_azimuthal_profile_about_the_grid_centre"]
    assert che37["detection_margin"] == pytest.approx(1.0, abs=1e-4)
    assert "STILL BLIND" in che37["verdict"]

    # Two repairs, both required to be effective. Re-centring CHE-37's own metric
    # on the traced image point is the smaller of the two and is what the M3.8
    # record now uses; dropping the azimuthal average entirely is the other.
    recentred = control["che37_metric_recentred_on_the_traced_image_point"]
    assert recentred["detection_margin"] > 10.0
    assert recentred["detected"] is True

    window = control["window_metric_no_azimuthal_average"]
    assert window["detection_margin"] > 10.0
    assert window["detected"] is True


def test_the_on_axis_configuration_is_recorded_as_unmoved(record: dict) -> None:
    invariance = record["on_axis_invariance"]

    assert invariance["object_space_reference_span_m"] == 0.0
    assert invariance["declared_opl_is_bit_identical"] is True
    assert invariance["max_abs_difference_waves"] == 0.0
