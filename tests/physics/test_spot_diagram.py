"""`measurements.spot_diagram`: the metrics against closed form, and the two gates.

CHE-226 (R16). Two things are tested here and they are not the same kind of claim.

**The metrics, against analytic ray geometry.** A ring of `N` rays of radius `R`
about a known centre has centroid at that centre, RMS radius `R` and geometric
radius `R` -- exactly, for any `N`, by construction rather than by convergence. Two
rays with intensities 1 and 3 have their weighted centroid at the 3/4 point and an
RMS of `sqrt(3)` times the half-separation. Those are the oracles, and they are
independent of this project's code in the way `AGENTS.md` requires: **the pinned
solver's own spot analysis is deliberately not used as the oracle here.** The
cross-path comparison lives in `tests/backends/test_optiland_analysis.py` and is
labelled there as characterization of a coincidence, not as a correctness gate.

**The gates, as refusals.** The no-ray-splitting provenance gate and the
amplitude requirement are behaviour, not physics, and each of the three splitting
states is asserted separately -- accepted, refused as split, refused as
undeclared -- because "refuses something" would pass with the wrong two.

The one test here that is neither is `test_the_sampling_measure_does_not_move_any
_metric`: it pins a *declared exclusion*. `measure_weight` is not applied, and a
change that started applying it would move every number in this file's sister
tests while still looking like a spot diagram.
"""

from __future__ import annotations

import ast
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from measurements import (
    SPOT_METRIC_DEFINITIONS,
    SpotResult,
    spot_diagram,
)
from representations import ContractError, RayBundle, ReferenceSurface

WAVELENGTH_M = 550e-9
IMAGE_SURFACE = ReferenceSurface(name="image_surface", z_m=0.0, medium_index=1.0)

RING_RADIUS_M = 1.0e-3
RING_CENTRE_M = (2.0e-4, -3.0e-4)


def _bundle(
    x: Any,
    y: Any,
    *,
    amplitude: Any = None,
    ray_splitting: str = "unsplit",
    measure_weight: Any = None,
    measure_kind: str = "undeclared",
) -> RayBundle:
    """A bundle of axial rays at `(x, y, 0)`, with everything else declared.

    Axial directions and a plane reference surface, because nothing in this
    measurement reads a direction: it is a statistic of *intersections*, and giving
    the rays a tilt would suggest otherwise.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    count = x.size
    return RayBundle(
        positions_m=np.column_stack([x, y, np.zeros(count)]),
        directions=np.tile(np.array([0.0, 0.0, 1.0]), (count, 1)),
        wavelength_m=WAVELENGTH_M,
        reference_surface=IMAGE_SURFACE,
        amplitude=np.ones(count) if amplitude is None else np.asarray(amplitude),
        measure_weight=measure_weight,
        measure_kind=measure_kind,
        ray_splitting=ray_splitting,
    )


def _ring(count: int = 12, *, radius_m: float = RING_RADIUS_M) -> tuple[Any, Any]:
    """`count` points on a circle of `radius_m` about `RING_CENTRE_M`."""
    angle = np.arange(count, dtype=np.float64) * (2.0 * math.pi / count)
    return (
        RING_CENTRE_M[0] + radius_m * np.cos(angle),
        RING_CENTRE_M[1] + radius_m * np.sin(angle),
    )


@pytest.mark.parametrize("count", [3, 6, 12, 60])
def test_a_uniform_ring_has_the_analytic_centroid_and_both_radii(count: int) -> None:
    """The closed form, for four ring populations. Not a convergence claim.

    Every ray of a ring sits at exactly `R` from the centre, so `sqrt(mean(r^2))`
    and `max(r)` are both `R` for any count, and a uniform ring's first moment is
    its centre because `sum(cos)` and `sum(sin)` over a full turn vanish. The
    tolerances are round-off on that cancellation, not a fitted agreement.
    """
    spot = spot_diagram(_bundle(*_ring(count)))

    assert spot.centroid_m[0] == pytest.approx(RING_CENTRE_M[0], abs=1e-15)
    assert spot.centroid_m[1] == pytest.approx(RING_CENTRE_M[1], abs=1e-15)
    assert spot.rms_radius_m == pytest.approx(RING_RADIUS_M, rel=1e-14)
    assert spot.geometric_radius_m == pytest.approx(RING_RADIUS_M, rel=1e-14)
    assert (spot.ray_count, spot.included_count) == (count, count)


def test_a_ring_plus_its_centre_ray_has_the_analytic_rms() -> None:
    """`R sqrt(N / (N + 1))`, which the ring test alone cannot distinguish from `R`.

    The centre ray contributes zero to the second moment and one to the count, so
    the RMS drops by a factor the geometric radius does not see. That pins the
    denominator: an implementation dividing by the *included* count and one
    dividing by the ring count agree in every uniform-ring case and differ here.
    """
    count = 12
    ring_x, ring_y = _ring(count)
    x = np.append(ring_x, RING_CENTRE_M[0])
    y = np.append(ring_y, RING_CENTRE_M[1])

    spot = spot_diagram(_bundle(x, y))

    expected = RING_RADIUS_M * math.sqrt(count / (count + 1.0))
    assert spot.rms_radius_m == pytest.approx(expected, rel=1e-14)
    assert spot.geometric_radius_m == pytest.approx(RING_RADIUS_M, rel=1e-14)


def test_the_metrics_are_intensity_weighted_and_not_the_unweighted_ones() -> None:
    """Two rays, intensities 1 and 3: the declared weighting, and what it is not.

    `amplitude = sqrt(3)` is intensity 3, so the centroid sits at the 3/4 point
    between the rays rather than halfway. Both numbers are asserted -- the weighted
    one it must be and the unweighted one it must not be -- because a change that
    dropped the weighting would still pass a test that only checked "somewhere
    between the two rays".
    """
    separation_m = 4.0e-3
    spot = spot_diagram(
        _bundle([0.0, separation_m], [0.0, 0.0], amplitude=[1.0, math.sqrt(3.0)])
    )

    weighted = 0.75 * separation_m
    assert spot.centroid_m[0] == pytest.approx(weighted, rel=1e-14)
    assert spot.centroid_m[0] != pytest.approx(0.5 * separation_m, rel=1e-6)

    # sqrt((1 * (3d/4)^2 + 3 * (d/4)^2) / 4) = sqrt(3) d / 4.
    assert spot.rms_radius_m == pytest.approx(math.sqrt(3.0) * separation_m / 4.0, rel=1e-14)
    # A maximum is unweighted: the far ray is 3d/4 from the centroid.
    assert spot.geometric_radius_m == pytest.approx(weighted, rel=1e-14)


def test_a_zero_intensity_ray_is_excluded_from_every_metric_and_counted() -> None:
    """A clipped ray far off axis must not set the geometric radius.

    This is the one metric the inclusion rule has to be applied to explicitly --
    the weighted moments exclude it by arithmetic, a maximum does not -- so it is
    the combination that would look self-consistent and be wrong: an RMS of `R` and
    a geometric radius of `10 R`.
    """
    ring_x, ring_y = _ring(12)
    reference = spot_diagram(_bundle(ring_x, ring_y))

    x = np.append(ring_x, RING_CENTRE_M[0] + 10.0 * RING_RADIUS_M)
    y = np.append(ring_y, RING_CENTRE_M[1])
    spot = spot_diagram(_bundle(x, y, amplitude=np.append(np.ones(12), 0.0)))

    assert spot.rms_radius_m == pytest.approx(reference.rms_radius_m, rel=1e-14)
    assert spot.geometric_radius_m == pytest.approx(reference.geometric_radius_m, rel=1e-14)
    assert spot.centroid_m == pytest.approx(reference.centroid_m, abs=1e-18)
    assert (spot.ray_count, spot.included_count) == (13, 12)
    # And the excluded ray is still in the arrays, at its own position: the result is
    # row-aligned with the bundle, which is what lets a caller see what was dropped.
    assert spot.x_m.shape == (13,)
    assert float(spot.intensity[-1]) == 0.0


def test_the_sampling_measure_does_not_move_any_metric() -> None:
    """A declared exclusion, pinned. `measure_weight` is not a per-ray energy.

    The weights here vary by a factor of 12 across the ring, which would move the
    centroid by a large fraction of `R` if they were being folded in. They must not
    be: a spot diagram is the distribution of intersections, and a quadrature cell
    area is a statement about how the pupil was discretized.
    """
    ring_x, ring_y = _ring(12)
    reference = spot_diagram(_bundle(ring_x, ring_y))

    weights = np.linspace(1.0, 12.0, 12) * 1e-9
    weighted = spot_diagram(
        _bundle(
            ring_x,
            ring_y,
            measure_weight=weights,
            measure_kind="quadrature_area_m2",
        )
    )

    assert weighted.centroid_m == pytest.approx(reference.centroid_m, abs=1e-18)
    assert weighted.rms_radius_m == pytest.approx(reference.rms_radius_m, rel=1e-15)
    assert weighted.geometric_radius_m == pytest.approx(reference.geometric_radius_m, rel=1e-15)


def test_an_unsplit_population_is_accepted_and_says_so() -> None:
    """The first of the three splitting states, and the only admissible one."""
    spot = spot_diagram(_bundle(*_ring(6), ray_splitting="unsplit"))
    assert spot.ray_splitting == "unsplit"


def test_a_population_known_to_contain_split_descendants_is_refused() -> None:
    """The second state. A different code from the third, on purpose."""
    with pytest.raises(ContractError) as error:
        spot_diagram(_bundle(*_ring(6), ray_splitting="split_descendants"))

    assert error.value.code == "SPLIT_RAYS_NOT_MEASURABLE"
    assert error.value.declaration == "ray_splitting"
    assert "branches" in str(error.value)


def test_an_undeclared_population_is_refused_rather_than_assumed_unsplit() -> None:
    """The third state, which is the default, which is the point.

    `"undeclared"` is not a synonym for `"unsplit"`: nothing about the numbers
    distinguishes them, so a measurement that treated the default as admissible
    would have assumed exactly the thing it needed told.
    """
    with pytest.raises(ContractError) as error:
        spot_diagram(_bundle(*_ring(6), ray_splitting="undeclared"))

    assert error.value.code == "RAY_SPLITTING_UNDECLARED"
    assert error.value.code != "SPLIT_RAYS_NOT_MEASURABLE"
    # The remedy points at the producer, because that is who can fix this one.
    assert "producer" in str(error.value) or "Declare it at the producer" in str(error.value)


def test_the_splitting_gate_is_not_derived_from_survival_in_either_direction() -> None:
    """Two independent states, and the cross terms are what proves it.

    A bundle with clipped rays and a declared unsplit population is measured; a
    bundle in which every ray survived but nothing declared its provenance is
    refused. If either state were being read off the other, one of these two would
    come out the other way.
    """
    ring_x, ring_y = _ring(12)
    clipped = np.append(np.ones(11), 0.0)

    with_clipping = spot_diagram(_bundle(ring_x, ring_y, amplitude=clipped))
    assert with_clipping.included_count == 11

    with pytest.raises(ContractError) as error:
        spot_diagram(_bundle(ring_x, ring_y, ray_splitting="undeclared"))
    assert error.value.code == "RAY_SPLITTING_UNDECLARED"


def test_a_bundle_with_no_amplitude_is_refused() -> None:
    """The amplitude is where per-ray intensity and ray survival both live."""
    ring_x, ring_y = _ring(6)
    count = ring_x.size
    bundle = RayBundle(
        positions_m=np.column_stack([ring_x, ring_y, np.zeros(count)]),
        directions=np.tile(np.array([0.0, 0.0, 1.0]), (count, 1)),
        wavelength_m=WAVELENGTH_M,
        reference_surface=IMAGE_SURFACE,
        ray_splitting="unsplit",
    )

    with pytest.raises(ContractError) as error:
        spot_diagram(bundle)
    assert error.value.code == "MISSING_DECLARATION"
    assert error.value.declaration == "amplitude"


def test_a_bundle_in_which_nothing_survived_is_refused_not_reported_as_a_point() -> None:
    """An all-zero amplitude has no centroid; `0/0` would be a spot at the origin."""
    with pytest.raises(ContractError) as error:
        spot_diagram(_bundle(*_ring(6), amplitude=np.zeros(6)))
    assert error.value.code == "EMPTY_ENSEMBLE"


def test_the_result_records_the_plane_the_wavelength_and_the_definitions() -> None:
    """Everything a consumer needs in order not to guess which numbers these are."""
    spot = spot_diagram(_bundle(*_ring(6)))

    assert spot.surface == "image_surface"
    assert spot.wavelength_m == WAVELENGTH_M
    assert set(SPOT_METRIC_DEFINITIONS) == {
        "centroid_m",
        "rms_radius_m",
        "geometric_radius_m",
    }
    for text in (spot.inclusion_rule, spot.reference, spot.weighting):
        assert isinstance(text, str) and text
    # The reference is the centroid and says so, and the weighting names what is
    # excluded. Asserted because these strings are the artifact's only statement of
    # which of the several things called "RMS spot radius" this one is.
    assert "centroid" in spot.reference
    assert "measure_weight is NOT applied" in spot.weighting


def test_every_number_is_available_without_rendering_anything() -> None:
    """The result is numerical, and this module imports no plotting library.

    Checked by AST over the source rather than by `sys.modules`, which under
    `-n 8 --dist loadfile` would be reporting on whatever else the worker imported.
    """
    spot = spot_diagram(_bundle(*_ring(6)))
    for name in ("rms_radius_m", "geometric_radius_m"):
        assert isinstance(getattr(spot, name), float)
    assert isinstance(spot.centroid_m, tuple) and len(spot.centroid_m) == 2
    assert not hasattr(spot, "view")

    source = Path("src/measurements/spot.py")
    imported: set[str] = set()
    for node in ast.walk(ast.parse(source.read_text())):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert not imported & {"matplotlib", "pyplot", "seaborn", "plotly"}
    # And no backend, which is the other half of "this is the neutral path".
    assert not imported & {"optiland", "chromatix", "torch", "jax"}


def test_the_record_refuses_an_amplitude_stored_where_an_intensity_was_expected() -> None:
    """`SpotResult` is public, so its invariants are executed rather than assumed."""
    with pytest.raises(ContractError) as error:
        SpotResult(
            x_m=np.zeros(3),
            y_m=np.zeros(3),
            intensity=np.array([1.0, -1e-30, 1.0]),
            wavelength_m=WAVELENGTH_M,
            surface="image_surface",
            centroid_m=(0.0, 0.0),
            rms_radius_m=0.0,
            geometric_radius_m=0.0,
            ray_count=3,
            included_count=3,
        )
    assert error.value.code == "NEGATIVE_INTENSITY"


def test_the_record_refuses_a_non_finite_radius() -> None:
    """A NaN metric from a caller's own arithmetic is not a spot of infinite size."""
    with pytest.raises(ContractError) as error:
        SpotResult(
            x_m=np.zeros(3),
            y_m=np.zeros(3),
            intensity=np.ones(3),
            wavelength_m=WAVELENGTH_M,
            surface="image_surface",
            centroid_m=(0.0, 0.0),
            rms_radius_m=float("nan"),
            geometric_radius_m=0.0,
            ray_count=3,
            included_count=3,
        )
    assert error.value.code == "UNIT_NOT_SI"
