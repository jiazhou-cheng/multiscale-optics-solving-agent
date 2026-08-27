"""B3-4F-REAL / B4-4F-REAL: the declarations that carry the physics (CHE-145).

What is worth a test here is not "the numbers came out" -- the committed records
under ``benchmarks/systems/records/`` are that -- but the handful of statements
the family makes that would be silently wrong if they drifted:

* the modulation really is B3-4F-IDEAL's, by identity rather than by inspection;
* the aberration law the ``PARAXIAL_LIMIT`` predicate is built on is the
  fourth-power law the authoring probe measured, and the residual-angle law the
  aperture ceiling is built on is its cube;
* the object/shared-plane grid pair really is a discrete Fourier pair, which is
  the single relation every physical quantity in an instance follows from;
* the image-parity convention is the one a physical 4f has, to the sample -- a
  one-sample error there looks like a small distortion;
* every declared instance differs from the instance it is read against in
  exactly one axis, which is CHE-145's own acceptance criterion;
* and the chain composes end to end through the shipping path, with the
  diffractive model named.

The last one runs the real chain at a deliberately tiny grid (2.7 s, 5.8e4
outgoing rays) rather than being skipped as expensive. A composition test that
never composes is the failure mode this file exists to avoid.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

# The drivers live under `benchmarks/`, which is not an installed package -- only
# `src/` is on the path. Added here rather than in the shared pytest config on
# purpose: this is the first test that reaches a driver, and widening the
# repository-wide import path for one file would be a change to test
# infrastructure rather than to this rung.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from benchmarks.systems import b3_4f_ideal, b3_4f_real  # noqa: E402
from couplers.interaction import DiffractiveModel  # noqa: E402
from verification.families.b3_4f_real import (  # noqa: E402
    B3_4F_REAL,
    B4_4F_REAL,
    DIFFRACTIVE_MODEL,
    PARAXIAL_LIMIT_WAVES,
    PRESCRIPTION,
    SPHERICAL_REFERENCE_HEIGHT_MM,
    SPHERICAL_REFERENCE_RESIDUAL_ANGLE_RAD,
    SPHERICAL_REFERENCE_WAVES,
    object_pitch_m,
    peak_wave_aberration_waves,
    residual_ray_angle_rad,
)
from verification.families.schema import (  # noqa: E402
    BenchmarkCategory,
    ValidityState,
)

pytestmark = [pytest.mark.coupler]


# ---------------------------------------------------------------------------
# The modulation is B3-4F-IDEAL's, by identity
# ---------------------------------------------------------------------------


def test_the_modulation_is_the_ideal_relays_own_constructor() -> None:
    """CHE-145's experimental control is that only the RELAY changes.

    Asserted as object identity rather than as agreement: a copied mask
    constructor could drift, and then the two rungs would be comparing two
    modulations while every record claimed one.
    """
    assert b3_4f_real.ideal_mask is b3_4f_ideal._mask
    assert b3_4f_real._order_power_relative_l2 is b3_4f_ideal._order_power_relative_l2
    assert b3_4f_real._order_phase_error_rad is b3_4f_ideal._order_phase_error_rad
    assert b3_4f_real._order_location_error_frac is b3_4f_ideal._order_location_error_frac


def test_the_diffractive_model_is_named_and_is_full_field() -> None:
    assert DiffractiveModel.FULL_FIELD.value == DIFFRACTIVE_MODEL


# ---------------------------------------------------------------------------
# The two aberration laws the predicates are built on
# ---------------------------------------------------------------------------


def test_peak_wave_aberration_is_the_measured_fourth_power_law() -> None:
    at_reference = peak_wave_aberration_waves(
        {"used_semi_aperture_mm": SPHERICAL_REFERENCE_HEIGHT_MM}
    )
    assert at_reference == pytest.approx(SPHERICAL_REFERENCE_WAVES, rel=1e-12)
    # Fourth power, exactly: halving the aperture must divide the aberration by 16.
    half = peak_wave_aberration_waves(
        {"used_semi_aperture_mm": SPHERICAL_REFERENCE_HEIGHT_MM / 2.0}
    )
    assert at_reference / half == pytest.approx(16.0, rel=1e-12)


def test_residual_ray_angle_is_the_measured_cube_law() -> None:
    at_reference = residual_ray_angle_rad(
        {"used_semi_aperture_mm": SPHERICAL_REFERENCE_HEIGHT_MM}
    )
    assert at_reference == pytest.approx(SPHERICAL_REFERENCE_RESIDUAL_ANGLE_RAD, rel=1e-12)
    half = residual_ray_angle_rad({"used_semi_aperture_mm": SPHERICAL_REFERENCE_HEIGHT_MM / 2.0})
    assert at_reference / half == pytest.approx(8.0, rel=1e-12)


def test_the_paraxial_limit_predicate_brackets_the_declared_sweep() -> None:
    """The sweep has to straddle the boundary or it is not a convergence sweep.

    The two wide instances must be outside and the two narrow ones inside; a
    predicate that called all four the same would make the four points a
    repetition rather than a convergence.
    """
    states = {}
    for instance_id in b3_4f_real.declared_instance_ids("B3-4F-REAL"):
        params = b3_4f_real.ALL_PARAMETERS[instance_id]
        states[instance_id] = B3_4F_REAL.evaluate_validity(params)[0]
    assert states["B3-4F-REAL-APERTURE-01"] is ValidityState.FAR_OUTSIDE
    assert states["B3-4F-REAL-APERTURE-02"] is ValidityState.FAR_OUTSIDE
    assert states["B3-4F-REAL-APERTURE-03"] is ValidityState.INSIDE
    assert states["B3-4F-REAL-APERTURE-04"] is ValidityState.INSIDE
    assert states["B3-4F-REAL-FIELD-01"] is ValidityState.INSIDE


def test_the_aperture_ceiling_predicts_the_refusal_it_was_derived_from() -> None:
    """SHARED_PLANE_RAY_ANGLE_CAPACITY was found by a refusal; it must reproduce it.

    ``C_RAY_TO_WAVE`` refused ``used_semi_aperture_mm = 6.0`` reporting
    ``|d|max = 2.435e-3`` against a grid limit of ``1.175e-3``. The predicate's own
    arithmetic has to land on that measured number, not merely on the same side of
    the bound -- otherwise the declared ceiling and the real one could drift apart
    and only the wide instance would ever notice.
    """
    params = b3_4f_real.ALL_PARAMETERS["B4-4F-REAL-APERTURE-WIDE"]
    assert params["used_semi_aperture_mm"] == 6.0
    predicted = b3_4f_real._predicted_shared_plane_angle(params)
    assert predicted == pytest.approx(2.435e-3, rel=0.02)
    state, margins = B4_4F_REAL.evaluate_validity(params)
    assert state is ValidityState.FAR_OUTSIDE
    assert margins["SHARED_PLANE_RAY_ANGLE_CAPACITY"] < 0.0


# ---------------------------------------------------------------------------
# The one geometric relation everything else follows from
# ---------------------------------------------------------------------------


def test_object_and_shared_plane_grids_are_a_discrete_fourier_pair() -> None:
    """``pitch_object * pitch_shared = lambda f / grid_n``, at every aperture.

    This is the relation that makes an M2.9 instance describe the same
    modulation as an M2.8 one: it is what puts diffraction order ``n`` at
    ``n * grid_n / samples_per_period`` sensor samples, which is B3-4F-IDEAL's
    own order-location formula up to the 4f inversion.
    """
    lambda_f = PRESCRIPTION["wavelength_m"] * PRESCRIPTION["effective_focal_length_mm"] * 1e-3
    for aperture_mm in (0.25, 0.7, 1.0, 2.0, 4.0):
        for grid_n in (32, 48, 64):
            params = {"used_semi_aperture_mm": aperture_mm, "grid_n": grid_n}
            pitch_shared = 2.0 * aperture_mm * 1e-3 / grid_n
            assert object_pitch_m(params) * pitch_shared == pytest.approx(
                lambda_f / grid_n, rel=1e-12
            )


def test_order_spacing_matches_the_ideal_relays_own_formula() -> None:
    """The physical order spacing, in sensor samples, is B3-4F-IDEAL's ``grid_n / spp``."""
    for instance_id, params in b3_4f_real.ALL_PARAMETERS.items():
        geom = b3_4f_real.geometry(params)
        expected_px = params["grid_n"] / params["samples_per_period"]
        assert geom["order_spacing_px"] == pytest.approx(expected_px, rel=1e-12), instance_id
        # And in metres, through the grid pair rather than through the pixel count.
        assert geom["order_spacing_m"] == pytest.approx(
            PRESCRIPTION["wavelength_m"]
            * PRESCRIPTION["effective_focal_length_mm"]
            * 1e-3
            / geom["mask_period_m"],
            rel=1e-12,
        ), instance_id


def test_the_4f_inversion_is_a_flip_about_the_origin_sample() -> None:
    """``b[c + j] == a[c - j]`` with ``c = n // 2``, which a plain reversal is not.

    A physical 4f is transform-then-transform and B3-4F-IDEAL's realization is
    transform-then-inverse-transform, so the two images differ by exactly this
    flip. Off by one sample it reads as a small distortion rather than as a
    convention error, which is why it is pinned here.
    """
    n = 8
    centre = n // 2
    for j in range(-centre + 1, centre):
        a = np.zeros((n, n), dtype=np.complex128)
        a[centre + j, centre - j] = 1.0 + 0.0j
        flipped = b3_4f_real._flip_about_origin(a)
        assert flipped[centre - j, centre + j] == 1.0 + 0.0j, j
        assert np.count_nonzero(flipped) == 1
    # A plain reversal would be off by one, so the two must disagree somewhere.
    a = np.zeros((n, n), dtype=np.complex128)
    a[centre + 1, centre] = 1.0
    assert not np.array_equal(b3_4f_real._flip_about_origin(a), a[::-1, ::-1])


# ---------------------------------------------------------------------------
# CHE-145's own acceptance criteria, as executable checks
# ---------------------------------------------------------------------------


def test_every_instance_differs_from_its_reference_in_exactly_one_axis() -> None:
    for instance_id, params in b3_4f_real.ALL_PARAMETERS.items():
        reference_id = b3_4f_real._REFERENCE_INSTANCE[instance_id]
        if instance_id == reference_id:
            continue
        differing = b3_4f_real.differing_axes(params, b3_4f_real.ALL_PARAMETERS[reference_id])
        assert len(differing) == 1, (instance_id, differing)


def test_the_characterization_family_cannot_gate() -> None:
    """CHE-145: no gating tolerance where there is no oracle.

    Category B4 already forbids it structurally, so this checks the thing that
    could still go wrong -- someone adding a tolerance to the shared metric tuple
    and expecting the category to catch it in the *other* family.
    """
    assert B4_4F_REAL.category is BenchmarkCategory.B4
    assert B4_4F_REAL.tolerances == ()
    assert B4_4F_REAL.invariants == ()
    assert not any(t.may_gate for t in B4_4F_REAL.tolerances)
    # And the gated family must still gate, or the split has collapsed.
    assert B3_4F_REAL.category is BenchmarkCategory.B3
    assert any(t.may_gate for t in B3_4F_REAL.tolerances)


def test_the_gated_tolerance_follows_from_the_validity_bound() -> None:
    """The field tolerance is derived, so the derivation must still close.

    ``PARAXIAL_LIMIT`` bounds the peak wave aberration; the measured
    proportionality between that and the field departure is 0.0372 rad^-1; the
    product must sit below the declared threshold. If someone loosens the
    predicate without loosening the tolerance -- or the reverse -- an instance the
    family calls INSIDE could fail the gate it is judged by, which is exactly the
    coupling this asserts.
    """
    measured_ratio = 0.0372  # departure per radian of peak wavefront error
    implied = measured_ratio * 2.0 * math.pi * PARAXIAL_LIMIT_WAVES
    threshold = B3_4F_REAL.tolerance_for("field_relative_l2_vs_ideal_4f").threshold
    assert implied < threshold
    assert threshold / implied == pytest.approx(2.1, abs=0.3)


def test_all_five_declared_controls_are_assigned_to_an_instance() -> None:
    """A declared control nobody runs reads exactly like a control that passed."""
    declared = {control.control_id for control in B3_4F_REAL.negative_controls}
    assigned = {cid for ids in b3_4f_real._CONTROLS_ON.values() for cid in ids}
    assert declared == assigned
    assert len(declared) == 5


# ---------------------------------------------------------------------------
# The composition, actually composed
# ---------------------------------------------------------------------------


@pytest.mark.optiland
@pytest.mark.integration
def test_the_chain_composes_end_to_end_at_a_tiny_grid() -> None:
    """One real pass through the shipping path: ~2.7 s, 5.8e4 outgoing rays.

    Deliberately not a numerical claim -- the committed records are that. What is
    checked is that the composition still holds together and that the two things
    the chain must not do quietly are still not quiet: the model is named in the
    result, and the phase-only modulation conserves the accumulated field's power
    exactly rather than approximately.
    """
    params = b3_4f_real._params(
        used_semi_aperture_mm=1.0,
        grid_n=16,
        object_grid_n=12,
        object_waist_pixels=1.0,
        samples_per_period=4.0,
        sensor_rows=16,
        sensor_cols=16,
    )
    assert B3_4F_REAL.evaluate_validity(params)[0] is ValidityState.INSIDE

    run = b3_4f_real.run_chain(params)
    assert run["model"] == DiffractiveModel.FULL_FIELD.value
    assert run["interaction_diagnostics"]["enumerated"] is True
    assert np.all(np.isfinite(run["sensor_u"]))
    assert run["sensor_u"].dtype == np.complex128
    assert run["outgoing_rays"] > 0
    assert run["invalid_rays_group_1"] == 0
    assert run["invalid_rays_group_2"] == 0

    # The phase-only invariant, exactly: a unit-modulus transmission cannot move
    # the accumulated field's discrete power.
    incident = run["focal_plane_incident_power"]
    transmitted = run["focal_plane_transmitted_power"]
    assert abs(transmitted / incident - 1.0) < 1e-12

    # And the reference arm of the same geometry, which is what supplies the one
    # complex normalization constant, must reproduce the object itself.
    reference = b3_4f_real.run_chain(params, modulated=False)
    measurements, diagnostics = b3_4f_real.measure(run, reference, params)
    assert set(measurements) == {metric.name for metric in B3_4F_REAL.metrics}
    assert diagnostics["diffractive_model"] == DiffractiveModel.FULL_FIELD.value
    assert all(math.isfinite(m.value) for m in measurements.values())
