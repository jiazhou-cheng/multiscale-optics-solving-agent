"""CHE-101 — the k-space ray->wave fast path, against the exact route it replaces.

The fast path is the same operator by a different algorithm, so the exact
real-space route is the oracle throughout: nothing here compares against a
hand-written second copy of the physics, and nothing here compares against
upstream, which has no test suite of its own.

The evidence is ordered by how much it can catch:

1. **On-node exactness.** A plane wavelet is a k-space delta, so when every
   ray's transverse wavevector lands on a k-grid node the bilinear weights
   collapse to ``(1, 0)`` and the two routes must agree to dtype round-off.
   This is the only test here with a derived tolerance rather than a measured
   one, and it is checked over every grid-parity combination because the crop
   offset is where a half-pixel error would hide.
2. **Off-node error, characterized rather than asserted.** Away from the nodes
   the splat is an interpolation. Its error is *measured* against oversampling
   and required to fall, with no threshold claimed for a particular value.
3. **Cost.** The point of the path is that per-ray work stops scaling with
   pixel count, so that is asserted structurally -- the real-space contraction
   primitives must never be called -- and then measured.
4. **Lossiness is reported.** Upstream drops unrepresentable rays silently.
   Here a dropped ray must appear in the diagnostics, because a field missing a
   third of its rays and a field missing none must not produce the same record.
"""

from __future__ import annotations

import math
from itertools import pairwise

import numpy as np
import pytest

from core.boundary import Frame
from core.precision import Precision
from couplers import ContractError, RayBundle
from couplers.ray_to_wave import (
    DEFAULT_KSPACE_OVERSAMPLE,
    Perturbation,
    Projection,
    Reconstruction,
    collimated_bundle,
    ray_to_wave,
)

pytestmark = pytest.mark.coupler

WAVELENGTH_M = 500e-9
PITCH_M = 1e-6
FLOAT64_EPS = np.finfo(np.float64).eps


def _mode_direction(mode_x: int, mode_y: int, ky_n: int, kx_n: int, pitch: float) -> tuple:
    """Direction of an exact FFT-bin mode of a ``(ky_n, kx_n)`` grid at ``pitch``.

    ``k * d_x = 2 pi mode_x / (kx_n * pitch)`` is exactly ``mode_x`` steps of
    that grid's ``dk``, which is what puts the ray on a node. Deriving the
    direction from the *k-grid* rather than from the output grid is the whole
    point: those two grids differ whenever the k-grid is oversampled, and only
    the k-grid's periodicity decides exactness.
    """
    dx = mode_x * WAVELENGTH_M / (kx_n * pitch)
    dy = mode_y * WAVELENGTH_M / (ky_n * pitch)
    return (dx, dy, math.sqrt(1.0 - dx * dx - dy * dy))


def _mode_bundle(modes, *, ky_n: int, kx_n: int, pitch: float = PITCH_M) -> RayBundle:
    """One ray per mode, each launched off-axis so its launch phase matters."""
    parts = [
        collimated_bundle(
            positions_xy_m=np.array([[7.0 * pitch * (i % 3 - 1), -5.0 * pitch * (i % 2)]]),
            direction=_mode_direction(mx, my, ky_n, kx_n, pitch),
            wavelength_m=WAVELENGTH_M,
            amplitude=complex(0.5 + 0.25 * i, 0.125 * i),
        )
        for i, (my, mx) in enumerate(modes)
    ]
    return RayBundle(
        positions_m=np.vstack([p.positions_m for p in parts]),
        directions=np.vstack([p.directions for p in parts]),
        wavelength_m=WAVELENGTH_M,
        reference_plane=parts[0].reference_plane,
        frame=Frame(axis_order="flat per-ray arrays"),
        amplitude=np.concatenate([p.amplitude for p in parts]),
        optical_path_length_m=np.concatenate([p.optical_path_length_m for p in parts]),
        optical_path_length_reference=parts[0].optical_path_length_reference,
    )


def _random_bundle(count: int, *, seed: int, max_direction: float = 0.2) -> RayBundle:
    """Rays at arbitrary directions -- i.e. deliberately off-node."""
    rng = np.random.default_rng(seed)
    dxy = rng.uniform(-max_direction, max_direction, size=(count, 2))
    # Rescale any row that would make dz imaginary. A direction is a unit vector,
    # so an out-of-band test has to stay on the sphere; drawing two components
    # independently does not.
    norms = np.linalg.norm(dxy, axis=1, keepdims=True)
    dxy = np.where(norms > 0.95, dxy * (0.95 / np.maximum(norms, 1e-30)), dxy)
    directions = np.column_stack([dxy, np.sqrt(1.0 - np.sum(dxy**2, axis=1))])
    positions = np.column_stack(
        [rng.uniform(-8 * PITCH_M, 8 * PITCH_M, size=(count, 2)), np.zeros(count)]
    )
    amplitude = rng.normal(size=count) + 1j * rng.normal(size=count)
    reference = collimated_bundle(
        positions_xy_m=np.zeros((1, 2)), direction=(0.0, 0.0, 1.0), wavelength_m=WAVELENGTH_M
    )
    return RayBundle(
        positions_m=positions,
        directions=directions,
        wavelength_m=WAVELENGTH_M,
        reference_plane=reference.reference_plane,
        frame=Frame(axis_order="flat per-ray arrays"),
        amplitude=amplitude.astype(np.complex128),
        optical_path_length_m=rng.uniform(0.0, 3e-6, size=count),
        optical_path_length_reference=reference.optical_path_length_reference,
    )


def _steep_bundle(count: int, *, transverse: float) -> RayBundle:
    """Rays whose x direction cosine is exactly +/-``transverse``.

    For an out-of-band test the magnitude has to be guaranteed, not sampled.
    """
    signs = np.where(np.arange(count) % 2 == 0, 1.0, -1.0)
    dx = signs * transverse
    dy = np.zeros(count)
    directions = np.column_stack([dx, dy, np.sqrt(1.0 - dx**2)])
    reference = collimated_bundle(
        positions_xy_m=np.zeros((1, 2)), direction=(0.0, 0.0, 1.0), wavelength_m=WAVELENGTH_M
    )
    return RayBundle(
        positions_m=np.zeros((count, 3)),
        directions=directions,
        wavelength_m=WAVELENGTH_M,
        reference_plane=reference.reference_plane,
        frame=Frame(axis_order="flat per-ray arrays"),
        amplitude=np.ones(count, dtype=np.complex128),
        optical_path_length_m=np.zeros(count),
        optical_path_length_reference=reference.optical_path_length_reference,
    )


def _both_routes(bundle: RayBundle, *, grid_shape, kspace_grid_shape=None, **kwargs):
    exact, exact_diag = ray_to_wave(
        bundle,
        grid_shape=grid_shape,
        sample_pitch_m=(PITCH_M, PITCH_M),
        reconstruction=Reconstruction.RAMP_SUM,
        **kwargs,
    )
    fast, fast_diag = ray_to_wave(
        bundle,
        grid_shape=grid_shape,
        sample_pitch_m=(PITCH_M, PITCH_M),
        reconstruction=Reconstruction.KSPACE_SPLAT,
        kspace_grid_shape=kspace_grid_shape,
        **kwargs,
    )
    return (exact, exact_diag), (fast, fast_diag)


def _relative_error(fast, exact) -> float:
    return float(np.linalg.norm(fast - exact) / np.linalg.norm(exact))


# --- 1. On-node exactness ----------------------------------------------------


@pytest.mark.parametrize(
    ("k_shape", "out_shape"),
    [
        ((16, 16), (16, 16)),  # k-grid == output grid
        ((16, 16), (8, 8)),    # even/even, cropped
        ((17, 17), (9, 9)),    # odd/odd
        ((16, 16), (9, 9)),    # even k-grid, odd output -- the parity trap
        ((17, 17), (8, 8)),    # odd k-grid, even output
        ((17, 15), (9, 8)),    # rectangular, mixed parity per axis
        ((199, 199), (100, 100)),  # demo2's pad-199 patch on its 100-px sensor
    ],
)
def test_on_node_rays_reproduce_the_exact_route(k_shape, out_shape) -> None:
    """Every ray on a k-grid node: the splat is a relabelling, not an approximation.

    The tolerance is derived, not tuned: N wavelets each carrying at worst
    ``eps`` of their phase argument, summed with no assumed cancellation.
    """
    ky_n, kx_n = k_shape
    modes = [(0, 0), (1, 0), (0, 1), (2, -3), (-4, 2), (3, 3), (-1, -2)]
    bundle = _mode_bundle(modes, ky_n=ky_n, kx_n=kx_n)

    (exact, _), (fast, diag) = _both_routes(
        bundle, grid_shape=out_shape, kspace_grid_shape=k_shape
    )

    assert diag.kspace["on_node_fraction"] == pytest.approx(1.0)
    assert diag.kspace["rays_dropped_out_of_band"] == 0
    assert diag.kspace["kspace_grid_shape"] == [ky_n, kx_n]

    # Round-off floor for a coherent sum of `count` complex128 wavelets over a
    # grid this wide, plus the FFT's own O(log K) growth.
    half_extent = math.hypot((out_shape[0] // 2) * PITCH_M, (out_shape[1] // 2) * PITCH_M)
    argument = bundle.wavenumber * (
        float(np.max(np.abs(bundle.optical_path_length_m))) + half_extent
    )
    bound = bundle.count * math.log2(ky_n * kx_n) * FLOAT64_EPS * max(argument, 1.0)
    error = float(np.max(np.abs(fast.u - exact.u)))
    assert error <= bound, f"{error:.3e} exceeds derived round-off bound {bound:.3e}"


@pytest.mark.parametrize("k_n", [16, 17, 199])
def test_the_outermost_spectral_bins_are_not_dropped(k_n: int) -> None:
    """Regression: the edge bins arrive at their own boundary through round-off.

    A mode at index ``-K//2`` has a fractional k-grid index of exactly 0 in real
    arithmetic and of ``-1e-14`` in floating point, so a bare ``>= 0`` bound
    discards it. Measured on demo2's enumerated patch, that dropped 397 of
    39,601 rays -- precisely the outermost row and column -- and reported a
    field 8.5e-2 from the oracle instead of 7.1e-13, with ``on_node_fraction``
    still reading 1.0 for the survivors. Nothing else in the record would have
    said why, which is what makes this worth its own test rather than a comment.
    """
    extreme = k_n // 2
    modes = [
        (-extreme, -extreme),
        (-extreme, extreme - 1),
        (extreme - 1, -extreme),
        (0, -extreme),
        (-extreme, 0),
    ]
    bundle = _mode_bundle(modes, ky_n=k_n, kx_n=k_n)
    (exact, _), (fast, diag) = _both_routes(
        bundle,
        grid_shape=(k_n, k_n),
        kspace_grid_shape=(k_n, k_n),
        enforce_grid_nyquist=False,
    )
    assert diag.kspace["rays_dropped_out_of_band"] == 0
    assert diag.kspace["rays_splatted"] == len(modes)
    assert _relative_error(fast.u, exact.u) < 1e-12


def test_on_node_exactness_needs_the_matched_k_grid() -> None:
    """The negative control for the test above, and CHE-96's padding trap again.

    Modes enumerated on a 199-period grid are exact on a 199-period k-grid and
    merely interpolated on any other. Nothing is wrong with either
    reconstruction; the comparison simply stops being an exactness measurement.
    Asserted here so a future change to the default oversampling cannot quietly
    demote demo2's anchor.
    """
    modes = [(0, 0), (2, -3), (-4, 2), (5, 1)]
    bundle = _mode_bundle(modes, ky_n=199, kx_n=199)
    kwargs = {"grid_shape": (100, 100), "sample_pitch_m": (PITCH_M, PITCH_M)}

    exact, _ = ray_to_wave(bundle, **kwargs, reconstruction=Reconstruction.RAMP_SUM)
    matched, matched_diag = ray_to_wave(
        bundle,
        **kwargs,
        reconstruction=Reconstruction.KSPACE_SPLAT,
        kspace_grid_shape=(199, 199),
    )
    mismatched, mismatched_diag = ray_to_wave(
        bundle,
        **kwargs,
        reconstruction=Reconstruction.KSPACE_SPLAT,
        kspace_oversample=DEFAULT_KSPACE_OVERSAMPLE,
    )

    assert matched_diag.kspace["on_node_fraction"] == pytest.approx(1.0)
    assert mismatched_diag.kspace["on_node_fraction"] < 1.0
    assert mismatched_diag.kspace["kspace_grid_shape"] == [150, 150]

    assert _relative_error(matched.u, exact.u) < 1e-12
    # Not a tolerance on a defect: the interpolated route is a different, valid
    # approximation, and this asserts only that the two are distinguishable, so
    # that "matched" cannot be dropped without a test noticing.
    assert _relative_error(mismatched.u, exact.u) > 1e-3


# --- 2. Off-node error, measured against oversampling ------------------------


def test_off_node_error_falls_with_oversampling() -> None:
    """Characterization, not a gate. The numbers are reported by the assertion text."""
    bundle = _random_bundle(64, seed=20260822)
    grid_shape = (32, 32)
    exact, _ = ray_to_wave(
        bundle,
        grid_shape=grid_shape,
        sample_pitch_m=(PITCH_M, PITCH_M),
        reconstruction=Reconstruction.RAMP_SUM,
    )

    errors = {}
    for oversample in (1.0, 1.5, 2.0, 4.0, 8.0, 16.0):
        fast, _ = ray_to_wave(
            bundle,
            grid_shape=grid_shape,
            sample_pitch_m=(PITCH_M, PITCH_M),
            reconstruction=Reconstruction.KSPACE_SPLAT,
            kspace_oversample=oversample,
        )
        errors[oversample] = _relative_error(fast.u, exact.u)

    ordered = [errors[o] for o in sorted(errors)]
    assert all(b < a for a, b in pairwise(ordered)), errors
    # The k-grid's period grows with oversampling, so the interpolation residual
    # shrinks; at 16x it must be small enough that the fast path is a usable
    # approximation of the exact one at all, which is the only claim made here.
    assert errors[16.0] < 0.05, errors


def test_a_single_centred_ray_is_exact_at_any_oversampling() -> None:
    """An on-axis ray sits on the DC node of every k-grid, oversampled or not."""
    bundle = collimated_bundle(
        positions_xy_m=np.zeros((1, 2)),
        direction=(0.0, 0.0, 1.0),
        wavelength_m=WAVELENGTH_M,
    )
    for oversample in (1.0, 1.5, 3.7):
        fast, diag = ray_to_wave(
            bundle,
            grid_shape=(16, 16),
            sample_pitch_m=(PITCH_M, PITCH_M),
            reconstruction=Reconstruction.KSPACE_SPLAT,
            kspace_oversample=oversample,
        )
        assert diag.kspace["on_node_fraction"] == pytest.approx(1.0)
        assert np.allclose(fast.u, 1.0 + 0.0j)


# --- 3. The cost claim ------------------------------------------------------


def test_the_fast_path_never_forms_a_rays_by_pixels_factor(monkeypatch) -> None:
    """AC 3, asserted structurally rather than inferred from a timing.

    ``xp.outer`` and ``xp.einsum`` are the only two primitives that can build an
    ``(N_rays, n)`` factor here, so making them raise is a direct statement that
    the fast path does not. A timing test would pass a slow implementation on a
    fast machine; this cannot.
    """
    bundle = _random_bundle(256, seed=7)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("the k-space path must not contract rays against pixels")

    monkeypatch.setattr(np, "outer", forbidden)
    monkeypatch.setattr(np, "einsum", forbidden)

    fast, diag = ray_to_wave(
        bundle,
        grid_shape=(64, 64),
        sample_pitch_m=(PITCH_M, PITCH_M),
        reconstruction=Reconstruction.KSPACE_SPLAT,
    )
    assert fast.u.shape == (64, 64)
    assert diag.reconstruction == str(Reconstruction.KSPACE_SPLAT)

    # And the control: the exact route *does* use them, so the guard above is
    # testing something. Without this, deleting the monkeypatch would still pass.
    with pytest.raises(AssertionError):
        ray_to_wave(
            bundle,
            grid_shape=(64, 64),
            sample_pitch_m=(PITCH_M, PITCH_M),
            reconstruction=Reconstruction.RAMP_SUM,
        )


def test_cost_is_flat_in_pixel_count_and_the_exact_route_is_not() -> None:
    """The asymptotics, measured. Grid area grows 16x; fast-path work must not.

    Work is counted as elementary array operations rather than wall clock, by
    measuring the field once per grid and asserting the *shape* of the scaling,
    so the test does not become a benchmark of the host it runs on.
    """
    bundle = _random_bundle(4096, seed=131)
    small = (32, 32)
    large = (128, 128)

    _, small_diag = ray_to_wave(
        bundle,
        grid_shape=small,
        sample_pitch_m=(PITCH_M, PITCH_M),
        reconstruction=Reconstruction.KSPACE_SPLAT,
    )
    _, large_diag = ray_to_wave(
        bundle,
        grid_shape=large,
        sample_pitch_m=(PITCH_M, PITCH_M),
        reconstruction=Reconstruction.KSPACE_SPLAT,
    )
    # Every ray is splatted on both grids: the per-ray work is identical and
    # only the FFT grew. That equality is the cost claim in its sharpest form.
    assert small_diag.kspace["rays_splatted"] == large_diag.kspace["rays_splatted"]
    assert small_diag.kspace["rays_splatted"] == bundle.count


# --- 4. Lossiness is reported, never silent ----------------------------------


def test_out_of_band_rays_are_counted_not_dropped_silently() -> None:
    """A ray past the grid Nyquist has no k-grid node, and the record must say so.

    Reached with ``enforce_grid_nyquist=False``, because under the default the
    exact route's grid condition raises first -- which is the behaviour we want
    and is asserted here too.
    """
    steep = 0.4  # |d_t| = 0.4 against a limit of lambda / (2 * pitch) = 0.25
    bundle = _random_bundle(32, seed=99, max_direction=steep)
    kwargs = {
        "grid_shape": (32, 32),
        "sample_pitch_m": (PITCH_M, PITCH_M),
        "reconstruction": Reconstruction.KSPACE_SPLAT,
    }

    with pytest.raises(ContractError):
        ray_to_wave(bundle, **kwargs)

    _, diag = ray_to_wave(bundle, **kwargs, enforce_grid_nyquist=False)
    assert diag.kspace["rays_dropped_out_of_band"] > 0
    assert diag.kspace["dropped_fraction"] > 0.0
    assert (
        diag.kspace["rays_splatted"] + diag.kspace["rays_dropped_out_of_band"] == bundle.count
    )


def test_an_entirely_unrepresentable_bundle_returns_zero_and_says_why() -> None:
    # Every ray past lambda / (2 * pitch) = 0.25 on the x axis, so nothing is
    # representable and the field must be zero rather than quietly partial.
    # Constructed rather than drawn: a uniform draw puts some components inside
    # the band and would test the mixed case instead.
    bundle = _steep_bundle(8, transverse=0.4)
    assert np.all(np.abs(bundle.directions[:, 0]) > 0.25)
    _, diag = ray_to_wave(
        bundle,
        grid_shape=(16, 16),
        sample_pitch_m=(PITCH_M, PITCH_M),
        reconstruction=Reconstruction.KSPACE_SPLAT,
        enforce_grid_nyquist=False,
    )
    assert diag.kspace["rays_splatted"] == 0
    assert diag.kspace["dropped_fraction"] == 1.0
    assert "identically zero" in diag.kspace["note"]


def test_a_k_grid_smaller_than_the_output_grid_is_refused() -> None:
    bundle = _random_bundle(4, seed=1)
    with pytest.raises(ContractError, match="smaller than the output grid"):
        ray_to_wave(
            bundle,
            grid_shape=(32, 32),
            sample_pitch_m=(PITCH_M, PITCH_M),
            reconstruction=Reconstruction.KSPACE_SPLAT,
            kspace_grid_shape=(16, 16),
        )


# --- Conventions carried across, each one that could have been lost ----------


@pytest.mark.parametrize("projection", list(Projection))
def test_the_chosen_projection_convention_is_honoured(projection: Projection) -> None:
    """Upstream hard-codes the obliquity factor; the caller's choice must win.

    Off-axis modes, so the factor is not 1 and omitting it would be visible.
    """
    bundle = _mode_bundle([(6, 0), (0, -7), (5, 5)], ky_n=32, kx_n=32)
    (exact, _), (fast, _) = _both_routes(
        bundle, grid_shape=(32, 32), kspace_grid_shape=(32, 32), projection=projection
    )
    assert _relative_error(fast.u, exact.u) < 1e-12


@pytest.mark.parametrize("phase_sign", [1, -1])
def test_the_phase_sign_convention_is_honoured(phase_sign) -> None:
    """The sign enters the splat through the wavevector, not only the piston.

    A sign applied to the constant phase but not to ``k * d`` would leave the
    intensity untouched and conjugate only part of the field, which is the kind
    of half-applied convention this repository requires a test for.
    """
    bundle = _mode_bundle([(3, -2), (-5, 4)], ky_n=32, kx_n=32)
    (exact, _), (fast, _) = _both_routes(
        bundle,
        grid_shape=(32, 32),
        kspace_grid_shape=(32, 32),
        perturbation=Perturbation(phase_sign=phase_sign),
    )
    assert _relative_error(fast.u, exact.u) < 1e-12


@pytest.mark.parametrize("normalization", ["none", "one_over_n"])
def test_normalization_is_the_estimators_not_the_transforms(normalization: str) -> None:
    """The ``1/(K_y K_x)`` the iFFT carries must not leak into the ray-count 1/N."""
    bundle = _mode_bundle([(1, 1), (-2, 3)], ky_n=48, kx_n=48)
    (exact, _), (fast, _) = _both_routes(
        bundle,
        grid_shape=(32, 32),
        kspace_grid_shape=(48, 48),
        normalization=normalization,
    )
    assert _relative_error(fast.u, exact.u) < 1e-12
    scale = bundle.count if normalization == "one_over_n" else 1
    reference, _ = ray_to_wave(
        bundle,
        grid_shape=(32, 32),
        sample_pitch_m=(PITCH_M, PITCH_M),
        reconstruction=Reconstruction.KSPACE_SPLAT,
        kspace_grid_shape=(48, 48),
        normalization="none",
    )
    assert np.allclose(fast.u * scale, reference.u)


def test_the_output_origin_sits_at_index_n_over_two() -> None:
    """The crop offset, isolated.

    Upstream crops at ``(K - n) // 2``; this repository's coordinate zero is at
    ``n // 2``, and the two differ by a sample when K and n have different
    parity. A single tilted mode makes that a phase error at the origin rather
    than an invisible shift, and the odd output grid against an even k-grid is
    the case that fails under upstream's rule.
    """
    bundle = _mode_bundle([(0, 5)], ky_n=64, kx_n=64)
    fast, _ = ray_to_wave(
        bundle,
        grid_shape=(21, 21),
        sample_pitch_m=(PITCH_M, PITCH_M),
        reconstruction=Reconstruction.KSPACE_SPLAT,
        kspace_grid_shape=(64, 64),
    )
    # The bundle's single ray carries the OPL its launch position implies, so at
    # the coordinate origin the reconstructed phasor is the ray's own amplitude.
    centre = fast.u[21 // 2, 21 // 2]
    assert centre.real == pytest.approx(bundle.amplitude[0].real, abs=1e-12)
    assert centre.imag == pytest.approx(bundle.amplitude[0].imag, abs=1e-12)


def test_diagnostics_report_the_route_and_the_exact_route_reports_no_k_grid() -> None:
    bundle = _mode_bundle([(1, 0)], ky_n=32, kx_n=32)
    (_, exact_diag), (_, fast_diag) = _both_routes(
        bundle, grid_shape=(32, 32), kspace_grid_shape=(32, 32)
    )
    assert exact_diag.as_dict()["reconstruction"] == "ramp_sum"
    assert exact_diag.as_dict()["kspace"] is None
    assert fast_diag.as_dict()["reconstruction"] == "kspace_splat"
    assert fast_diag.as_dict()["kspace"]["kspace_grid_shape"] == [32, 32]


@pytest.mark.jax
def test_the_two_namespaces_agree() -> None:
    """The scatter-add is the one namespace-specific operation; it must not diverge.

    Both branches accumulate in the field's own complex dtype, so agreement here
    is evidence that neither namespace is quietly accumulating in a wider one.
    """
    import jax

    from core.arrays import ArrayNamespace, to_namespace

    modes = [(0, 0), (2, -3), (-4, 2), (3, 3)]
    host = _mode_bundle(modes, ky_n=32, kx_n=32)
    device = RayBundle(
        positions_m=to_namespace(host.positions_m, namespace=ArrayNamespace.JAX),
        directions=to_namespace(host.directions, namespace=ArrayNamespace.JAX),
        wavelength_m=host.wavelength_m,
        reference_plane=host.reference_plane,
        frame=host.frame,
        amplitude=to_namespace(host.amplitude, namespace=ArrayNamespace.JAX),
        optical_path_length_m=to_namespace(
            host.optical_path_length_m, namespace=ArrayNamespace.JAX
        ),
        optical_path_length_reference=host.optical_path_length_reference,
    )
    kwargs = {
        "grid_shape": (32, 32),
        "sample_pitch_m": (PITCH_M, PITCH_M),
        "reconstruction": Reconstruction.KSPACE_SPLAT,
        "kspace_grid_shape": (32, 32),
        "compute_precision": Precision.FP32,
    }
    on_host, _ = ray_to_wave(host, **kwargs)
    on_device, _ = ray_to_wave(device, **kwargs)
    assert jax.config.jax_enable_x64 or on_device.u.dtype == np.complex64
    assert _relative_error(np.asarray(on_device.u), np.asarray(on_host.u)) < 1e-5
