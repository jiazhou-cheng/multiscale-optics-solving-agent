"""R07.2: the k-space route, and the error budget between it and the direct sum.

CHE-186. The two routes evaluate the same sum. They are the same *operator* only
where every ray's transverse wavevector lands on a k-grid node; everywhere else
the k-space route carries a bilinear interpolation the direct route does not
have, and the whole job of this file is to say where that boundary is with
numbers rather than to assert agreement in the comfortable regime and call it
equivalence.

The direct route is the oracle here, and that is a deliberate and limited claim.
It is exact per ray, and it is separately anchored against three analytic oracles
in `tests/physics/test_ray_to_scalar.py`, so using it as the reference for the
fast route is not this tree's numerics grading itself -- the analytic anchor is
one link away and the first test below re-establishes it in this file.
"""

from __future__ import annotations

import dataclasses
import math
import time

import numpy as np
import pytest
from ray_support import (
    WAVELENGTH_M,
    a_surface,
    collimated_bundle,
    mode_bundle,
    shifted_inverse_dft,
)

from couplers import DEFAULT_KSPACE_OVERSAMPLE, Reconstruction, ray_to_scalar
from representations import ContractError, RayBundle

SHAPE = (16, 16)
PITCH_M = (0.5e-6, 0.5e-6)


def peak_relative_residual(u, reference) -> float:
    return float(
        np.max(np.abs(np.asarray(u) - np.asarray(reference)))
        / np.max(np.abs(np.asarray(reference)))
    )


def an_enumerated_spectrum(seed: int = 20260831):
    """The same random field the direct route's round-trip gate uses."""
    rng = np.random.default_rng(seed)
    source = rng.standard_normal(SHAPE) + 1j * rng.standard_normal(SHAPE)
    rays, retained, spectrum = mode_bundle(source, sample_pitch_m=PITCH_M)
    return rays, shifted_inverse_dft(np.where(retained, spectrum, 0.0))


def scattered_directions(count: int = 2000, *, half_angle: float = 0.15, seed: int = 11):
    """Rays whose directions land nowhere in particular, which is the off-node case.

    An *enumerated* spectrum lands on k-grid nodes at every integer oversampling,
    so measuring interpolation error on one measures nothing. These directions are
    drawn from a cone and are off-node at every oversampling.
    """
    rng = np.random.default_rng(seed)
    polar = half_angle * np.sqrt(rng.random(count))
    azimuth = 2.0 * np.pi * rng.random(count)
    directions = np.column_stack(
        [np.sin(polar) * np.cos(azimuth), np.sin(polar) * np.sin(azimuth), np.cos(polar)]
    )
    return RayBundle(
        positions_m=np.zeros((count, 3)),
        directions=directions,
        wavelength_m=WAVELENGTH_M,
        reference_surface=a_surface(),
        amplitude=rng.standard_normal(count) + 1j * rng.standard_normal(count),
        optical_path_m=np.zeros(count),
        optical_path_reference="the surface itself",
        measure_weight=np.full(count, 1.0e-12),
        measure_kind="quadrature_area_m2",
    )


# ---------------------------------------------------------------------------
# 1. Where the two routes agree, and where they do not
# ---------------------------------------------------------------------------


def test_on_node_the_two_routes_are_the_same_arithmetic() -> None:
    """Criterion 1, the agreeing regime.

    With `kspace_grid_shape` equal to the grid the modes were enumerated on, every
    ray's `(k_x, k_y)` is a node, the bilinear weights collapse to `(1, 0)`, and
    both routes reproduce the analytic inverse DFT. Measured 1.6e-15 of peak
    between them, and both within 1.3e-15 of the analytic reference.
    """
    rays, reference = an_enumerated_spectrum()
    direct, direct_record = ray_to_scalar(rays, grid_shape=SHAPE, sample_pitch_m=PITCH_M)
    kspace, kspace_record = ray_to_scalar(
        rays,
        grid_shape=SHAPE,
        sample_pitch_m=PITCH_M,
        reconstruction=Reconstruction.KSPACE,
        kspace_grid_shape=SHAPE,
    )

    assert peak_relative_residual(direct.u, reference) < 1e-13
    assert peak_relative_residual(kspace.u, reference) < 1e-13
    assert peak_relative_residual(kspace.u, direct.u) < 1e-13

    assert direct_record.kspace is None
    assert kspace_record.kspace is not None
    assert kspace_record.kspace["on_node_fraction"] == 1.0
    assert kspace_record.kspace["dropped_fraction"] == 0.0


@pytest.mark.parametrize("oversample", [2.0, 4.0])
def test_an_integer_oversampling_of_an_enumeration_is_still_on_node(oversample: float) -> None:
    """The exactness condition is about the k-grid *period*, not about its size.

    Doubling the k-grid keeps every mode of a 16x16 enumeration on a node -- they
    land on the even indices -- so exactness survives. This is the half of the
    trap that is easy to miss: oversampling does not by itself take a mode
    off-node, and a caller who concluded that "more oversampling is safer" from
    this case would be wrong for the general one below.
    """
    rays, reference = an_enumerated_spectrum()
    kspace, record = ray_to_scalar(
        rays,
        grid_shape=SHAPE,
        sample_pitch_m=PITCH_M,
        reconstruction=Reconstruction.KSPACE,
        kspace_oversample=oversample,
    )
    assert record.kspace["on_node_fraction"] == 1.0
    assert peak_relative_residual(kspace.u, reference) < 1e-13


def test_the_default_oversampling_puts_an_enumeration_off_node() -> None:
    """Criterion 1, the *disagreeing* regime, and the reason the default is not enough.

    `ceil(1.5 * 16) = 24` is not a multiple of 16, so three quarters of the modes
    fall between nodes and the reconstruction misses the analytic reference by
    3.6e-1 of peak. Neither route is at fault: the caller owns an enumeration and
    did not say so. This is why `kspace_grid_shape` exists and why the module
    docstring tells an enumerating caller to pass it.
    """
    rays, reference = an_enumerated_spectrum()
    direct, _ = ray_to_scalar(rays, grid_shape=SHAPE, sample_pitch_m=PITCH_M)
    kspace, record = ray_to_scalar(
        rays, grid_shape=SHAPE, sample_pitch_m=PITCH_M, reconstruction=Reconstruction.KSPACE
    )

    assert record.kspace["kspace_grid_shape"] == [24, 24]
    assert record.kspace["on_node_fraction"] == pytest.approx(0.25)
    assert peak_relative_residual(kspace.u, direct.u) == pytest.approx(3.6e-1, rel=0.15)
    assert peak_relative_residual(kspace.u, reference) > 1e-2


def test_the_off_node_error_is_the_interpolations_and_falls_as_oversample_squared() -> None:
    """Criterion 1, the declared error budget: what the disagreement *is*.

    Measured on 2 000 random directions in a 0.15 rad cone onto a 64x64 grid, as a
    fraction of peak amplitude against the direct route:

    | oversample | 1.0 | 1.5 | 2 | 4 | 8 | 16 |
    | -- | -- | -- | -- | -- | -- | -- |
    | vs direct | 7.5e-1 | 4.0e-1 | 2.8e-1 | 7.3e-2 | 1.8e-2 | 4.9e-3 |

    Every doubling divides the error by about four. That second-order rate is
    bilinear interpolation's own, and it is what identifies the disagreement as
    the interpolation rather than as a defect in either route. A test that only
    asserted "the error is small at high oversampling" would pass on a route with
    a first-order bug in it.
    """
    rays = scattered_directions()
    shape, pitch = (64, 64), PITCH_M
    direct, _ = ray_to_scalar(rays, grid_shape=shape, sample_pitch_m=pitch)

    errors = {}
    for oversample in (2.0, 4.0, 8.0, 16.0):
        kspace, record = ray_to_scalar(
            rays,
            grid_shape=shape,
            sample_pitch_m=pitch,
            reconstruction=Reconstruction.KSPACE,
            kspace_oversample=oversample,
        )
        assert record.kspace["on_node_fraction"] == 0.0
        errors[oversample] = peak_relative_residual(kspace.u, direct.u)

    assert errors[2.0] > 1e-1
    assert errors[16.0] < 1e-2
    for coarse, fine in ((2.0, 4.0), (4.0, 8.0), (8.0, 16.0)):
        assert errors[coarse] / errors[fine] == pytest.approx(4.0, rel=0.25), errors


def test_the_default_oversampling_is_not_a_tolerance() -> None:
    """The budget stated as the thing a caller must not assume.

    At `DEFAULT_KSPACE_OVERSAMPLE` the two routes disagree by tens of percent on a
    generic ensemble. The constant is characterized, not tuned, and this pins that
    it is not quietly safe.
    """
    rays = scattered_directions(count=500)
    shape = (32, 32)
    direct, _ = ray_to_scalar(rays, grid_shape=shape, sample_pitch_m=PITCH_M)
    kspace, _ = ray_to_scalar(
        rays,
        grid_shape=shape,
        sample_pitch_m=PITCH_M,
        reconstruction=Reconstruction.KSPACE,
        kspace_oversample=DEFAULT_KSPACE_OVERSAMPLE,
    )
    assert peak_relative_residual(kspace.u, direct.u) > 1e-1


# ---------------------------------------------------------------------------
# 2. What the k-grid cannot hold
# ---------------------------------------------------------------------------


def test_a_mode_past_the_output_grid_nyquist_limit_is_refused_before_either_route() -> None:
    """Criterion 2, first half: the declared rule for the *output* grid is refusal.

    It is checked before the route is chosen, so neither realization can alias.
    """
    theta = 0.5
    rays, _, _ = collimated_bundle(
        shape=(8, 8),
        sample_pitch_m=(2.0e-6, 2.0e-6),
        direction=(math.sin(theta), 0.0, math.cos(theta)),
    )
    for route in (Reconstruction.DIRECT, Reconstruction.KSPACE):
        with pytest.raises(ContractError) as raised:
            ray_to_scalar(
                rays,
                grid_shape=(8, 8),
                sample_pitch_m=(2.0e-6, 2.0e-6),
                reconstruction=route,
            )
        assert raised.value.code == "SHAPE_MISMATCH"


def test_a_mode_the_k_grid_cannot_hold_is_counted_not_silently_dropped() -> None:
    """Criterion 2, second half: the top band edge, where the two routes differ.

    A ray's fractional k-index is `d_u K dx / lambda + K // 2`, so the band is
    `[-lambda / (2 dx), lambda / (2 dx) (1 - 2 / K)]` -- one bin short of the
    output grid's Nyquist limit on the positive side, at any oversampling. A mode
    exactly on `+lambda / (2 dx)` is inside the output grid's limit, is evaluated
    by the direct route, and is out of the k-grid's band. It is dropped, **and the
    record says so**, which is the difference between a declared rule and a silent
    one.
    """
    pitch = (0.5e-6, 0.5e-6)
    limit = WAVELENGTH_M / (2.0 * pitch[1])
    rays, _, _ = collimated_bundle(
        shape=(8, 8),
        sample_pitch_m=pitch,
        direction=(limit, 0.0, np.sqrt(1.0 - limit**2)),
    )
    direct, _ = ray_to_scalar(rays, grid_shape=(8, 8), sample_pitch_m=pitch)
    kspace, record = ray_to_scalar(
        rays, grid_shape=(8, 8), sample_pitch_m=pitch, reconstruction=Reconstruction.KSPACE
    )

    assert record.kspace["dropped_fraction"] == 1.0
    assert record.kspace["rays_dropped_out_of_band"] == rays.count
    assert record.kspace["note"].startswith("no ray was representable")
    assert float(np.max(np.abs(np.asarray(kspace.u)))) == 0.0
    assert float(np.max(np.abs(np.asarray(direct.u)))) > 0.0

    # ...and the negative twin: a mode just inside the band is kept and exact.
    inside = limit * (1.0 - 4.0 / 8.0)
    rays_inside, _, _ = collimated_bundle(
        shape=(8, 8),
        sample_pitch_m=pitch,
        direction=(inside, 0.0, np.sqrt(1.0 - inside**2)),
    )
    direct_inside, _ = ray_to_scalar(rays_inside, grid_shape=(8, 8), sample_pitch_m=pitch)
    kspace_inside, inside_record = ray_to_scalar(
        rays_inside,
        grid_shape=(8, 8),
        sample_pitch_m=pitch,
        reconstruction=Reconstruction.KSPACE,
        kspace_grid_shape=(8, 8),
    )
    assert inside_record.kspace["dropped_fraction"] == 0.0
    assert peak_relative_residual(kspace_inside.u, direct_inside.u) < 1e-12


def test_a_partial_drop_is_reported_in_both_rays_and_launch_power() -> None:
    """The arithmetic that only a *partial* drop exercises.

    `on_node_fraction` is a fraction of the *splatted* rays, not of all of them,
    and `dropped_launch_power_fraction` is not `dropped_fraction` unless the
    amplitudes are uniform. Both denominators are invisible at 0 % and 100 %
    dropped, so this mixes one out-of-band ray with in-band ones and gives the
    out-of-band one a much larger amplitude than the rest.
    """
    pitch = (0.5e-6, 0.5e-6)
    limit = WAVELENGTH_M / (2.0 * pitch[1])
    tilts = np.array([limit, 0.0, 0.1 * limit, 0.2 * limit])
    directions = np.column_stack(
        [tilts, np.zeros(4), np.sqrt(1.0 - tilts**2)]
    )
    amplitude = np.array([3.0, 1.0, 1.0, 1.0], dtype=complex)
    rays = RayBundle(
        positions_m=np.zeros((4, 3)),
        directions=directions,
        wavelength_m=WAVELENGTH_M,
        reference_surface=a_surface(),
        amplitude=amplitude,
        optical_path_m=np.zeros(4),
        optical_path_reference="the surface itself",
        measure_weight=np.full(4, 1.0e-12),
        measure_kind="quadrature_area_m2",
    )
    _, record = ray_to_scalar(
        rays, grid_shape=(8, 8), sample_pitch_m=pitch, reconstruction=Reconstruction.KSPACE
    )
    kspace = record.kspace
    assert kspace["rays_dropped_out_of_band"] == 1
    assert kspace["dropped_fraction"] == pytest.approx(0.25)
    # 3^2 / (3^2 + 1 + 1 + 1) -- three times the ray fraction, which is the whole
    # reason the power number is reported beside the count.
    assert kspace["dropped_launch_power_fraction"] == pytest.approx(9.0 / 12.0)
    assert 0.0 < kspace["on_node_fraction"] <= 1.0


def test_the_jax_scatter_add_branch_agrees_with_the_numpy_one() -> None:
    """`.at[].add` is namespace-specific production code, so it gets its own gate.

    Scatter-add is outside the array-API surface, so it is the one operation in
    the module that names its namespace -- which means it is also the one that can
    silently diverge between the host and a device. Both branches must accumulate
    in the declared complex dtype, not in a wider one, so the check is agreement
    with NumPy rather than agreement with an oracle.
    """
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    rays, reference = an_enumerated_spectrum()
    on_host, _ = ray_to_scalar(
        rays,
        grid_shape=SHAPE,
        sample_pitch_m=PITCH_M,
        reconstruction=Reconstruction.KSPACE,
        kspace_grid_shape=SHAPE,
    )
    on_jax_rays = dataclasses.replace(
        rays,
        positions_m=jnp.asarray(np.asarray(rays.positions_m)),
        directions=jnp.asarray(np.asarray(rays.directions)),
        amplitude=jnp.asarray(np.asarray(rays.amplitude)),
        optical_path_m=jnp.asarray(np.asarray(rays.optical_path_m)),
        measure_weight=jnp.asarray(np.asarray(rays.measure_weight)),
    )
    on_jax, record = ray_to_scalar(
        on_jax_rays,
        grid_shape=SHAPE,
        sample_pitch_m=PITCH_M,
        reconstruction=Reconstruction.KSPACE,
        kspace_grid_shape=SHAPE,
    )

    assert record.output_state["namespace"] == "jax"
    assert record.kspace["on_node_fraction"] == 1.0
    assert peak_relative_residual(np.asarray(on_jax.u), np.asarray(on_host.u)) < 1e-13
    assert peak_relative_residual(np.asarray(on_jax.u), reference) < 1e-13


def test_a_k_grid_smaller_than_the_output_grid_is_refused() -> None:
    """The field cannot be cropped out of a k-grid it does not fit in."""
    rays, _ = an_enumerated_spectrum()
    with pytest.raises(ContractError) as raised:
        ray_to_scalar(
            rays,
            grid_shape=SHAPE,
            sample_pitch_m=PITCH_M,
            reconstruction=Reconstruction.KSPACE,
            kspace_grid_shape=(8, 8),
        )
    assert raised.value.code == "SHAPE_MISMATCH"
    assert "smaller than the output grid" in str(raised.value)


# ---------------------------------------------------------------------------
# 3. The record, the origin, and the cost
# ---------------------------------------------------------------------------


def test_the_route_travels_with_the_field() -> None:
    """Criterion 3: a downstream consumer can tell which realization produced it."""
    rays, _reference = an_enumerated_spectrum()
    _, direct = ray_to_scalar(rays, grid_shape=SHAPE, sample_pitch_m=PITCH_M)
    _, kspace = ray_to_scalar(
        rays, grid_shape=SHAPE, sample_pitch_m=PITCH_M, reconstruction=Reconstruction.KSPACE
    )
    assert direct.reconstruction == "direct"
    assert kspace.reconstruction == "kspace"
    assert direct.as_dict()["kspace"] is None
    assert kspace.as_dict()["kspace"]["kspace_grid_shape"] == [24, 24]


@pytest.mark.parametrize("shape", [(8, 8), (8, 9), (9, 8), (9, 9)])
@pytest.mark.parametrize("oversample", [1.0, 2.0])
def test_the_crop_keeps_the_origin_at_index_n_over_two(
    shape: tuple[int, int], oversample: float
) -> None:
    """The crop offset is `K // 2 - n // 2`, checked over every parity combination.

    `(K - n) // 2` agrees whenever `K` and `n` share parity and differs by one
    sample otherwise, which puts the reconstructed origin one pixel off the
    coordinate origin this repository pins -- a half-pixel tilt rather than a
    visible failure, so only a parity sweep finds it.
    """
    rays, _, _ = collimated_bundle(
        shape=shape, sample_pitch_m=PITCH_M, direction=(0.0, 0.0, 1.0)
    )
    direct, _ = ray_to_scalar(rays, grid_shape=shape, sample_pitch_m=PITCH_M)
    kspace, _ = ray_to_scalar(
        rays,
        grid_shape=shape,
        sample_pitch_m=PITCH_M,
        reconstruction=Reconstruction.KSPACE,
        kspace_oversample=oversample,
    )
    assert peak_relative_residual(kspace.u, direct.u) < 1e-12


@pytest.mark.slow
def test_the_fast_route_is_faster_and_the_configuration_is_stated() -> None:
    """Criterion 4. A performance claim with no configuration is not a claim.

    Measured on the pinned CPU container, NumPy on the host: 60 000 rays onto a
    256x256 grid, 1.275 s direct against 0.021 s k-space, a 61x speedup. The gate
    asserts 10x rather than 61x because the number is a property of this host and
    the *asymptotics* are the claim -- direct is `O(N_rays x ny x nx)` and k-space
    is `O(N_rays + K log K)`, so the ratio grows with the pixel count and shrinks
    on a faster machine only if both shrink together.
    """
    rays = scattered_directions(count=60_000, half_angle=0.02, seed=7)
    shape, pitch = (256, 256), PITCH_M

    started = time.perf_counter()
    ray_to_scalar(rays, grid_shape=shape, sample_pitch_m=pitch)
    direct_seconds = time.perf_counter() - started

    started = time.perf_counter()
    ray_to_scalar(
        rays, grid_shape=shape, sample_pitch_m=pitch, reconstruction=Reconstruction.KSPACE
    )
    kspace_seconds = time.perf_counter() - started

    assert direct_seconds / kspace_seconds > 10.0, (direct_seconds, kspace_seconds)
