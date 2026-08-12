"""CHE-27 — the curvature bound must actually bound a measurement.

SI eq S9 gives ``eps_curv <= arcsin(D / 2R)``. A bound that is merely plotted
alongside a measurement is not a bound, so these tests build the sag phase the
derivation describes, measure the direction spread it produces, and require the
analytic expression to sit above it across the regime the paper plots.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from multiscale_optics_agent.couplers.contracts import ContractError
from multiscale_optics_agent.couplers.curvature import (
    check_patch,
    curvature_direction_error_bound,
    curvature_observability_width,
    max_patch_width_for_error,
    measured_tangent_plane_direction_error,
)

WAVELENGTH_M = 1e-6  # the paper's Figure 3c is drawn in units of lambda


# --- The analytic expression ------------------------------------------------------


def test_bound_matches_the_closed_form_of_eq_s9() -> None:
    assert curvature_direction_error_bound(1.0, 10.0) == pytest.approx(math.asin(0.05))
    assert curvature_direction_error_bound(1.0, 1.0) == pytest.approx(math.asin(0.5))
    assert curvature_direction_error_bound(1e-6, 1.0) == pytest.approx(5e-7, rel=1e-9)


def test_bound_is_independent_of_the_phase_profile() -> None:
    """Stated by the paper and worth pinning structurally: the bound takes only
    D and R, so no DOE design can be offered as an argument for relaxing it."""
    import inspect

    parameters = set(inspect.signature(curvature_direction_error_bound).parameters)
    assert parameters == {"patch_width_m", "radius_m"}

    # And empirically: the same D and R give the same bound regardless of what
    # phase profile is later placed on the patch, because the bound never sees
    # one. Measured here against two very different sag magnitudes.
    for radius in (1e-2, 1.0):
        gentle = measured_tangent_plane_direction_error(
            patch_width_m=400 * WAVELENGTH_M, radius_m=radius, wavelength_m=WAVELENGTH_M
        )
        assert gentle <= curvature_direction_error_bound(400 * WAVELENGTH_M, radius)


def test_the_planar_limit_is_exactly_zero() -> None:
    """R -> infinity. Matches SI S2's statement that planar patches have no
    intrinsic upper size bound: any width gives zero curvature error."""
    assert curvature_direction_error_bound(1.0, math.inf) == 0.0
    assert curvature_direction_error_bound(1e6, math.inf) == 0.0
    assert max_patch_width_for_error(1e-3, math.inf) == math.inf


def test_bound_is_monotone_in_patch_width_and_in_curvature() -> None:
    widths = [10e-6, 50e-6, 100e-6, 400e-6]
    bounds = [curvature_direction_error_bound(d, 1e-2) for d in widths]
    assert bounds == sorted(bounds)

    radii = [1e-3, 1e-2, 1e-1]
    tighter = [curvature_direction_error_bound(100e-6, r) for r in radii]
    assert tighter == sorted(tighter, reverse=True)


def test_inverse_recovers_the_width_that_saturates_a_threshold() -> None:
    radius = 5e-3
    threshold = 0.05
    width = max_patch_width_for_error(threshold, radius)
    assert curvature_direction_error_bound(width, radius) == pytest.approx(threshold)


def test_a_patch_wider_than_the_surface_has_no_bound_rather_than_a_large_one() -> None:
    """arcsin is undefined past 1. Reporting a saturated value would be a
    fabricated number where the model has simply stopped applying."""
    with pytest.raises(ContractError, match="no bound"):
        curvature_direction_error_bound(3.0, 1.0)


# --- The bound must bound the measurement -------------------------------------------


@pytest.mark.parametrize("radius_lambda", [1_000, 10_000, 100_000])
@pytest.mark.parametrize("patch_lambda", [50, 100, 200, 400])
def test_measured_direction_error_stays_under_the_analytic_bound(
    radius_lambda: int, patch_lambda: int
) -> None:
    """Reproduces the regime of paper Figure 3c: R = 1k, 10k, 100k wavelengths
    against patch sizes up to 400 lambda."""
    radius_m = radius_lambda * WAVELENGTH_M
    patch_m = patch_lambda * WAVELENGTH_M

    measured = measured_tangent_plane_direction_error(
        patch_width_m=patch_m, radius_m=radius_m, wavelength_m=WAVELENGTH_M
    )
    bound = curvature_direction_error_bound(patch_m, radius_m)

    assert measured <= bound, (
        f"measured {measured:.6e} rad exceeds the eq S9 bound {bound:.6e} rad "
        f"at D = {patch_lambda} lambda, R = {radius_lambda} lambda"
    )


def test_the_bound_is_tight_where_the_effect_is_observable() -> None:
    """A bound of pi/2 would also never be exceeded, so tightness matters --
    but only where there is something to measure.

    Above ``sqrt(2 lambda R)`` the curvature spread exceeds the patch's own
    diffraction limit and the bound tracks the measurement closely.
    """
    radius_m = 10_000 * WAVELENGTH_M
    observable_from = curvature_observability_width(WAVELENGTH_M, radius_m)

    patch_m = 400 * WAVELENGTH_M
    assert patch_m > 2 * observable_from
    measured = measured_tangent_plane_direction_error(
        patch_width_m=patch_m, radius_m=radius_m, wavelength_m=WAVELENGTH_M
    )
    ratio = curvature_direction_error_bound(patch_m, radius_m) / measured
    assert 1.0 <= ratio < 2.0, f"bound/measured = {ratio:.3f}"


def test_below_the_observability_width_the_bound_is_conservative_by_construction() -> None:
    """CHE-27's finding, and not stated in the paper.

    A patch of width D resolves directions no finer than lambda/D, while the
    curvature spread it carries is D/2R. So the effect is only spectrally
    visible when D > sqrt(2 lambda R). Below that the aperture's own diffraction
    limit dominates and the measured error collapses far under the bound --
    which is a property of the aperture, not slack in eq S9.

    Recorded because the natural reading of a 200:1 gap is that the bound is
    useless, when in fact the measurement has nothing to report there.
    """
    radius_m = 10_000 * WAVELENGTH_M
    observable_from = curvature_observability_width(WAVELENGTH_M, radius_m)
    assert observable_from == pytest.approx(math.sqrt(2 * 10_000) * WAVELENGTH_M)

    ratios = {}
    for patch_lambda in (100, 200, 400):
        patch_m = patch_lambda * WAVELENGTH_M
        measured = measured_tangent_plane_direction_error(
            patch_width_m=patch_m, radius_m=radius_m, wavelength_m=WAVELENGTH_M
        )
        ratios[patch_lambda] = curvature_direction_error_bound(patch_m, radius_m) / measured

    # 100 lambda is below sqrt(2 lambda R) = 141 lambda; 400 lambda is well above.
    assert 100 * WAVELENGTH_M < observable_from < 200 * WAVELENGTH_M
    assert ratios[100] > 50
    assert ratios[400] < 2
    # The bound still holds everywhere -- it is conservative, not wrong.
    assert all(ratio >= 1.0 for ratio in ratios.values())


def test_a_flat_surface_has_no_observability_width() -> None:
    assert curvature_observability_width(WAVELENGTH_M, math.inf) == 0.0


def test_a_planar_patch_produces_no_measurable_direction_spread() -> None:
    """The control for the measurement itself. If a flat patch showed spread,
    the measurement would be reporting FFT leakage, not curvature."""
    measured = measured_tangent_plane_direction_error(
        patch_width_m=400 * WAVELENGTH_M, radius_m=math.inf, wavelength_m=WAVELENGTH_M
    )
    assert measured < 1e-3


def test_error_grows_with_patch_size_and_shrinks_with_radius_in_measurement() -> None:
    """Paper Figure 3c's qualitative claim, checked on the measurement rather
    than on the formula that was fitted to it."""
    radius_m = 10_000 * WAVELENGTH_M
    small = measured_tangent_plane_direction_error(
        patch_width_m=50 * WAVELENGTH_M, radius_m=radius_m, wavelength_m=WAVELENGTH_M
    )
    large = measured_tangent_plane_direction_error(
        patch_width_m=400 * WAVELENGTH_M, radius_m=radius_m, wavelength_m=WAVELENGTH_M
    )
    assert large > small

    flatter = measured_tangent_plane_direction_error(
        patch_width_m=400 * WAVELENGTH_M,
        radius_m=100_000 * WAVELENGTH_M,
        wavelength_m=WAVELENGTH_M,
    )
    assert flatter < large


# --- Precondition behaviour -----------------------------------------------------------


def test_exceeding_the_threshold_raises_with_a_usable_remedy() -> None:
    with pytest.raises(ContractError) as excinfo:
        check_patch(patch_width_m=1e-3, radius_m=1e-2, error_threshold_rad=0.01)

    message = str(excinfo.value)
    assert "curvature bound" in message
    # The diagnostic must say what width WOULD work, not merely that this one does not.
    assert "Use a patch no wider than" in message


def test_a_patch_within_budget_passes_and_reports_its_margin() -> None:
    budget = check_patch(patch_width_m=100e-6, radius_m=1.0, error_threshold_rad=0.1)

    assert budget.within_budget is True
    assert budget.error_bound_rad < budget.error_threshold_rad
    assert budget.max_patch_width_m > budget.patch_width_m
    assert budget.thin_patch_assumption_holds is True


def test_the_regime_can_be_entered_deliberately_but_is_still_recorded() -> None:
    """A caller studying the failure regime has to be able to reach it. The
    budget still says the threshold was exceeded, so a number produced there is
    never mistaken for a valid one."""
    budget = check_patch(
        patch_width_m=1e-3, radius_m=1e-2, error_threshold_rad=0.01, enforce=False
    )
    assert budget.within_budget is False
    assert budget.error_bound_rad > budget.error_threshold_rad


def test_the_thin_patch_assumption_is_reported_separately_from_the_budget() -> None:
    """D << R is an assumption of the derivation, not a consequence. A patch can
    sit inside a generous error threshold while violating it, and that has to be
    visible rather than implied by a passing check."""
    budget = check_patch(patch_width_m=0.5, radius_m=1.0, error_threshold_rad=1.0)

    assert budget.within_budget is True
    assert budget.thin_patch_assumption_holds is False
    assert "D << R" in budget.as_dict()["assumptions"]


def test_the_budget_records_the_equation_and_its_assumptions() -> None:
    record = check_patch(
        patch_width_m=10e-6, radius_m=1e-2, error_threshold_rad=0.1
    ).as_dict()

    assert "arcsin(D / 2R)" in record["bound"]
    assert record["independent_of"] == "the DOE phase profile"
    assert len(record["assumptions"]) == 3


def test_non_physical_inputs_are_refused() -> None:
    with pytest.raises(ContractError):
        curvature_direction_error_bound(-1.0, 1.0)
    with pytest.raises(ContractError):
        curvature_direction_error_bound(1e-6, -1.0)
    with pytest.raises(ContractError):
        max_patch_width_for_error(0.0, 1.0)
    with pytest.raises(ContractError):
        measured_tangent_plane_direction_error(
            patch_width_m=1e-6, radius_m=1.0, wavelength_m=WAVELENGTH_M, samples=8
        )
    with pytest.raises(ContractError):
        measured_tangent_plane_direction_error(
            patch_width_m=1e-6,
            radius_m=1.0,
            wavelength_m=WAVELENGTH_M,
            samples=1024,
            windows=1,
        )


def test_a_global_spectrum_measurement_would_report_aperture_diffraction() -> None:
    """Why the measurement is local. Recorded because the global version was
    tried first and made eq S9 look violated at every patch size.

    A flat 50-lambda patch has no curvature error at all, yet its own spectrum
    already reaches ~0.02 rad at the first sinc null -- an order of magnitude
    above the curvature spread of a 100000-lambda radius. That angular content
    is the truncated aperture, not the sag.
    """
    patch_lambda = 50
    aperture_first_null_rad = 1.0 / patch_lambda
    curvature_bound = curvature_direction_error_bound(
        patch_lambda * WAVELENGTH_M, 100_000 * WAVELENGTH_M
    )
    assert aperture_first_null_rad > 50 * curvature_bound

    # The local measurement is unaffected by that, and correctly reports zero
    # for a flat patch of the same size.
    assert (
        measured_tangent_plane_direction_error(
            patch_width_m=patch_lambda * WAVELENGTH_M,
            radius_m=math.inf,
            wavelength_m=WAVELENGTH_M,
        )
        < 1e-9
    )


def test_curvature_module_imports_no_solver_engine() -> None:
    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    tree = ast.parse((root / "src/multiscale_optics_agent/couplers/curvature.py").read_text())
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert not imported & {"optiland", "chromatix"}
