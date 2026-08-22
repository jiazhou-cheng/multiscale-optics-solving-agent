"""CHE-47 (M3.9R extension): the hexapolar quadrature-weight math, in isolation.

Pure functions over ring indices, so these are cheap and need no engine.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from couplers.contracts import ContractCode, ContractError
from couplers.quadrature import (
    hexapolar_area_weight_m2,
    hexapolar_ring_index,
)

pytestmark = pytest.mark.coupler


def _hexapolar(num_rings: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A reference implementation, independent of the one under test.

    Mirrors ``optiland.distribution.HexagonalDistribution.generate_points`` (ring
    ``j`` at radius ``j / num_rings``, ``6j`` points), which is the geometry
    :func:`hexapolar_ring_index` assumes.
    """
    x = [0.0]
    y = [0.0]
    ring = [0]
    r = np.linspace(0.0, 1.0, num_rings + 1)
    for j in range(1, num_rings + 1):
        theta = np.linspace(0.0, 2.0 * math.pi, 6 * j, endpoint=False)
        x.extend((r[j] * np.cos(theta)).tolist())
        y.extend((r[j] * np.sin(theta)).tolist())
        ring.extend([j] * (6 * j))
    return np.asarray(x), np.asarray(y), np.asarray(ring)


@pytest.mark.parametrize("num_rings", [1, 2, 8, 16, 64])
def test_ring_index_recovers_the_generating_ring(num_rings: int) -> None:
    px, py, expected_ring = _hexapolar(num_rings)
    ring = hexapolar_ring_index(px, py, num_rings)
    assert np.array_equal(ring, expected_ring)


def test_ring_index_matches_expected_hexapolar_count() -> None:
    num_rings = 32
    px, _py, _ = _hexapolar(num_rings)
    assert px.size == 1 + 3 * num_rings * (num_rings + 1)


def test_ring_index_refuses_a_ray_off_any_ring() -> None:
    # 0.37 does not equal j/8 for any integer j in [0, 8].
    with pytest.raises(ContractError) as excinfo:
        hexapolar_ring_index(np.array([0.37]), np.array([0.0]), 8)
    assert excinfo.value.code == ContractCode.NON_HEXAPOLAR_SAMPLING


def test_ring_index_rejects_num_rings_below_one() -> None:
    with pytest.raises(ContractError):
        hexapolar_ring_index(np.array([0.0]), np.array([0.0]), 0)


@pytest.mark.parametrize("num_rings", [1, 4, 8, 64, 512])
def test_area_weight_sums_to_aperture_area_in_the_limit(num_rings: int) -> None:
    aperture_radius_m = 5.0e-3
    _, _, ring = _hexapolar(num_rings)
    weight = hexapolar_area_weight_m2(ring, num_rings, aperture_radius_m)

    true_area = math.pi * aperture_radius_m**2
    total = float(np.sum(weight))
    # Exact, worked from the ring counts 6j: sum = pi a^2 (1 + 1/(4 n^2)).
    expected_total = true_area * (1.0 + 0.25 / num_rings**2)
    assert total == pytest.approx(expected_total, rel=1e-9)
    # And therefore converges to the true aperture area as num_rings grows.
    assert abs(total - true_area) / true_area == pytest.approx(0.25 / num_rings**2, rel=1e-6)


def test_area_weight_boundary_corrections() -> None:
    num_rings = 16
    aperture_radius_m = 1.0e-3
    _, _, ring = _hexapolar(num_rings)
    weight = hexapolar_area_weight_m2(ring, num_rings, aperture_radius_m)
    nominal = math.pi * aperture_radius_m**2 / (3.0 * num_rings**2)

    assert weight[ring == 0] == pytest.approx(0.75 * nominal)
    assert np.all(weight[ring == num_rings] == pytest.approx(0.5 * nominal))
    interior = (ring > 0) & (ring < num_rings)
    assert np.all(weight[interior] == pytest.approx(nominal))


def test_area_weight_rejects_nonpositive_aperture() -> None:
    with pytest.raises(ContractError):
        hexapolar_area_weight_m2(np.array([0]), 4, 0.0)
    with pytest.raises(ContractError):
        hexapolar_area_weight_m2(np.array([0]), 4, float("nan"))
