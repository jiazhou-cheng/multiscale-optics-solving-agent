"""Workstream B: the seven wave kernel checks, measured against recorded expectations.

CHE-240. The expected values below come from the catalog -- `MODELS` and the
`_carrier_removed_propagator` docstring in `backends/chromatix/solver.py`, and
`representations.VALIDITY_NOTES['paraxial']` -- and they are **expectations to
reproduce, not tolerances to tune**. Nothing here widens one, and nothing here
redesigns a propagator.

Every **oracle** grid below is built from the declared pitch and shape in numpy,
not read off the implementation: an oracle that takes its frequency grid from the
code it is judging has borrowed part of the answer.
`tests/backends/test_chromatix_boundary.py` scans `src/` and `tests/` and not this
tree, so the no-chromatix-symbol rule is not enforced here -- and sweep 1
deliberately does reach for `native.f_grid` and `compute_transfer_propagator`,
because its claim is about the propagator arrays themselves and cannot be made
without them. Nothing else in this file imports chromatix.

"101's grid"
------------
Four of the seven checks are on it, and it is fully specified by
`VALIDITY_NOTES['paraxial']`: **512 x 512 samples at dx = 0.3 um, lambda_0 =
0.532 um, n = 1.33, z = 50 um**. Not re-chosen here; it is the grid the recorded
2.3e-1 and 4.9e-6 were measured on, and re-choosing it would make the comparison
against them meaningless.

Which oracle decides what
-------------------------
Three kinds of comparison appear below and they do not have equal standing.

* **Closed form.** The paraxial residual `n k0 z (1 - cos - sin^2/2)`, the
  walk-off `z sin(theta)`, the focus at `f sin(theta)`, the Fourier pitch
  `lambda f / (n N dx)`. Arithmetic, independent of this repository. These may
  decide.
* **Kernel identity.** Sweep 1 is an algebraic claim about two of this project's
  kernels -- substitute one factor and they coincide -- and "exactly 0.0" is a
  statement about the pair, not about either being right.
* **Diagnostic.** Sweeps 2, 3 and 4 compare Fresnel against this repository's own
  angular spectrum. `AGENTS.md` forbids that as a correctness gate, so those rows
  reproduce a *recorded measurement* and are labelled diagnostic. What they
  establish is that the recorded number still holds, not that either kernel is
  right.

Sweep 4 is a declared bound with no enforcement
-----------------------------------------------
The ticket asks for "refusal at the boundary" on `z` versus `N pitch^2 / lambda`.
The bound **is** declared: `src/operations/catalog.py` carries
`"z <= N pitch^2 / lambda, the transfer function's own sampling bound"` as a
`validity` entry on both `O_ASM_PROPAGATE` and `O_FRESNEL_PROPAGATE`. What does
not exist is any **runtime refusal** of it. Searched: `MODELS`,
`_require_model`/`_require_pad_target_crop`, `ScalarField.__post_init__`,
`CONTRACT_CODES` (22 codes) and `numerics.REFUSAL_CODES` -- none mentions a
transfer-function sampling bound, and neither propagation raises on either side of
it.

"Declared domain, unenforced at runtime" is a materially different gap from
"absent", so that is what the rows say. Recorded as `NOT-COVERED` and
**quantified**: the sweep runs both sides of the boundary, shows that neither
refuses, and measures how far the unpadded result has drifted where the declared
criterion says it should not be trusted.

Note that the declared criterion omits the medium index -- in-medium it would be
`n N dx^2 / lambda_0`, 115 um rather than 86.6 um on this grid. That is the tree's
own wording, reproduced faithfully rather than corrected here, and both probed
distances fall on the same side of the bound either way.
"""

from __future__ import annotations

import argparse
import math
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from backends.chromatix import focal_plane_transform, fresnel_propagate, propagate
from backends.chromatix.fields import fourier_plane_pitch_m
from benchmarks.verification.record import Row, finish, provenance
from representations import ReferenceSurface, ScalarField

# --- 101's grid, and the constants every sweep shares ----------------------

#: `VALIDITY_NOTES['paraxial']`'s grid, restated as data so a row can carry it.
GRID_101: dict[str, Any] = {
    "shape": (512, 512),
    "pitch_m": (0.3e-6, 0.3e-6),
    "wavelength_m": 0.532e-6,
    "refractive_index": 1.33,
    "distance_m": 50e-6,
    "dtype": "complex64",
}

#: The recorded expectations, so a row's `expected` is a citation rather than a
#: number retyped from a ticket.
EXPECTED: dict[str, Any] = {
    "kernel_identity_max_difference": 0.0,
    "hard_edge_intensity_difference_over_peak": 2.3e-1,
    "soft_edge_intensity_difference_over_peak": 4.9e-6,
    "paraxial_bound_phase_error_rad": math.pi / 4.0,
    "nyquist_phase_error_rad": 25.5,
    "corner_phase_error_rad": 175.0,
}


def _field(
    u: np.ndarray,
    pitch_m: tuple[float, float],
    *,
    wavelength_m: float,
    medium_index: float,
) -> ScalarField:
    return ScalarField(
        u=u.astype(np.complex64),
        sample_pitch_m=pitch_m,
        wavelength_m=wavelength_m,
        reference_surface=ReferenceSurface(name="source", z_m=0.0, medium_index=medium_index),
    )


def _exact(field: ScalarField, distance_m: float, *, pad_width: int) -> ScalarField:
    return propagate(
        field,
        distance_m=distance_m,
        model={
            "method": "asm_carrier_removed",
            "pad_width": pad_width,
            "target_surface": "target",
        },
    )


def _paraxial(field: ScalarField, distance_m: float, *, pad_width: int) -> ScalarField:
    return fresnel_propagate(
        field,
        distance_m=distance_m,
        model={"pad_width": pad_width, "target_surface": "target"},
    )


def _coordinates(shape: tuple[int, int], pitch_m: tuple[float, float]) -> tuple[Any, Any]:
    """`(y, x)` sample coordinates, index `n // 2` at the origin.

    Built here in numpy rather than read off a field, so the grid a prediction is
    evaluated on does not come from the code being measured.
    """
    y = (np.arange(shape[0], dtype=np.float64) - shape[0] // 2) * pitch_m[0]
    x = (np.arange(shape[1], dtype=np.float64) - shape[1] // 2) * pitch_m[1]
    return y, x


def _intensity_difference_over_peak(a: ScalarField, b: ScalarField) -> float:
    """`max| |A|^2 - |B|^2 | / max|B|^2`, the quantity `VALIDITY_NOTES` records."""
    intensity_a = np.abs(np.asarray(a.u)).astype(np.float64) ** 2
    intensity_b = np.abs(np.asarray(b.u)).astype(np.float64) ** 2
    return float(np.max(np.abs(intensity_a - intensity_b)) / np.max(intensity_b))


def _uniform_advance_rad(out: ScalarField, source: ScalarField) -> float:
    """`arg(mean(U_out / U_in))` for a field that advances uniformly, in radians.

    Averaged before the angle is taken so the estimate is the grid mean rather
    than one sample's, which matters at 175 rad where a single sample's float32
    phase is the least reliable thing on the grid.
    """
    ratio = np.asarray(out.u).astype(np.complex128) / np.asarray(source.u).astype(np.complex128)
    return float(np.angle(np.mean(ratio)))


def _wrapped(radians: float) -> float:
    return (radians + math.pi) % (2.0 * math.pi) - math.pi


def _plane_wave_at_bins(
    bins: tuple[int, int], *, shape, pitch_m, wavelength_m, refractive_index
) -> ScalarField:
    """A unit plane wave on the `(ny, nx)` DFT bin pair `bins`.

    Parameterized by **bin index** and not by a target direction cosine, so the
    grid represents the wave exactly rather than aliasing it: bin `n` on an axis is
    `n / (N dx)` cycles per metre, and `sin(theta) = lambda_0 |f| / n_medium`. Built
    from the declared pitch, not from a backend frequency grid.
    """
    y, x = _coordinates(shape, pitch_m)
    frequency_y = bins[0] / (shape[0] * pitch_m[0])
    frequency_x = bins[1] / (shape[1] * pitch_m[1])
    phase = 2.0 * np.pi * (frequency_y * y[:, None] + frequency_x * x[None, :])
    return _field(
        np.exp(1j * phase).astype(np.complex64),
        pitch_m,
        wavelength_m=wavelength_m,
        medium_index=refractive_index,
    )


# --- 1. the kernel identity ------------------------------------------------


def sweep_kernel_identity() -> list[Row]:
    """`delay + 1.0 -> 2.0` turns the carrier-removed ASM kernel into the Fresnel one.

    The recorded claim is a **maximum difference of exactly 0.0 in float32 over a
    512^2 grid**, and the substitution it names is inside
    `_carrier_removed_propagator`. So that is what is substituted: the private
    factory is replaced for the duration of one call with a version that uses
    `2.0`, and the comparison is then between two *public* propagations of the same
    field -- `propagate(method='asm_carrier_removed')` against `fresnel_propagate`.

    **And the propagator arrays are compared too**, which is the quantity the
    record actually names -- see `kernel_array_difference`. An earlier draft of this
    sweep skipped that comparison on the grounds that it "would have needed the
    backend's own `f_grid`", which was not a reason: the monkeypatch below already
    uses `native.f_grid`. Skipping it meant the sweep could not adjudicate the
    recorded claim at all, and it reported a clean result by measuring something
    else.

    The patch is restored in a `finally`, and the row records the unpatched
    difference too. Without that second number the check is unfalsifiable: a patch
    that silently failed to apply would report 0.0 for the wrong reason.
    """
    from backends.chromatix import solver

    shape, pitch = GRID_101["shape"], GRID_101["pitch_m"]
    wavelength, index = GRID_101["wavelength_m"], GRID_101["refractive_index"]
    distance = GRID_101["distance_m"]

    # A field with structure at every spatial frequency the grid represents, so the
    # comparison is over the whole kernel and not over the few bins a smooth field
    # occupies. A fixed seed, recorded, because "random" and "reproducible" have to
    # both be true.
    generator = np.random.default_rng(20260903)
    noise = generator.standard_normal(shape) + 1j * generator.standard_normal(shape)
    field = _field(noise, pitch, wavelength_m=wavelength, medium_index=index)

    def difference() -> float:
        exact = np.asarray(_exact(field, distance, pad_width=0).u)
        approximate = np.asarray(_paraxial(field, distance, pad_width=0).u)
        return float(np.max(np.abs(exact - approximate)))

    unpatched = difference()

    #: The bins the per-bin phase probe uses. Spread across the propagating cone,
    #: away from both the axis (where every propagator ever written gives 0) and
    #: the Nyquist bin (degenerate).
    phase_bins = (4, 32, 115, 200, 255)

    def phase_difference() -> float:
        """Max |phase(ASM) - phase(Fresnel)| over `phase_bins`, in radians.

        This is the measurement that separates *kernel* from *plumbing*. A
        broadband field's complex difference mixes the two: a float32 FFT over
        512^2 accumulates ~1e-5 in amplitude whatever the kernel is. A single plane
        wave on an exactly-represented bin advances uniformly, so its phase is read
        off directly and a kernel mismatch would show as a systematic function of
        frequency rather than as noise.
        """
        worst = 0.0
        for bin_index in phase_bins:
            source = _plane_wave_at_bins(
                (0, bin_index),
                shape=shape,
                pitch_m=pitch,
                wavelength_m=wavelength,
                refractive_index=index,
            )
            asm = _uniform_advance_rad(_exact(source, distance, pad_width=0), source)
            fresnel = _uniform_advance_rad(_paraxial(source, distance, pad_width=0), source)
            worst = max(worst, abs(_wrapped(asm - fresnel)))
        return worst

    unpatched_phase = phase_difference()

    def kernel_array_difference() -> dict[str, float]:
        """The quantity the record actually names: the two propagator ARRAYS.

        The only place in this file that reaches past the public surface, and it has
        to: the recorded claim is that substituting one factor "reproduces the
        Fresnel phase with a maximum difference of exactly 0.0 in float32 over a
        512^2 grid", and "the Fresnel phase" is
        `chromatix.functional.compute_transfer_propagator`. Comparing propagated
        *fields* instead -- which is what this sweep did first -- measures a
        different quantity and cannot adjudicate the claim at all.
        """
        import chromatix.functional as cf
        from chromatix.utils import l2_sq_norm

        _, jnp, _ = solver.import_backend()
        native, _requested = solver.to_native(field)
        wavelength_native = native.broadcasted_wavelength
        frequency_squared = l2_sq_norm(native.f_grid)
        substituted_phase = (
            -2.0
            * jnp.pi
            * jnp.abs(distance)
            * (wavelength_native / index)
            * frequency_squared
            / 2.0
        )
        ours = jnp.fft.ifftshift(jnp.exp(1j * substituted_phase), axes=native.spatial_dims)
        theirs = cf.compute_transfer_propagator(native, distance, index)
        return {
            "max_abs_difference": float(jnp.max(jnp.abs(ours - theirs))),
            "max_phase_difference_rad": float(
                jnp.max(jnp.abs(jnp.angle(ours * jnp.conj(theirs))))
            ),
        }

    kernel_arrays = kernel_array_difference()

    # The float32 phase floor this comparison cannot go below, predicted rather
    # than fitted. The largest probe bin's carrier-removed phase argument is
    # `2 pi |z| (lambda/n) f^2 / (delay + 1)`, about 198 rad on this grid, and one
    # float32 epsilon on that is 2.4e-5 rad. Two *different* code paths --
    # `propagate` and `fresnel_propagate` -- build their kernels from separately
    # computed frequency grids, so their round-off does not cancel and the residual
    # sits at this floor rather than at zero. A factor of 4 of headroom, stated: it
    # is a floor on one operation and there are a handful in the chain.
    largest_bin = max(phase_bins)
    largest_frequency = largest_bin / (shape[1] * pitch[1])
    largest_sine = (wavelength / index) * largest_frequency
    largest_phase_rad = (
        2.0
        * math.pi
        * distance
        * (wavelength / index)
        * largest_frequency**2
        / (math.sqrt(1.0 - largest_sine**2) + 1.0)
    )
    float32_phase_floor = 4.0 * float(np.finfo(np.float32).eps) * largest_phase_rad

    original = solver._carrier_removed_propagator
    started = time.perf_counter()
    try:

        def substituted(native: Any, *, distance_m: float, refractive_index: float) -> Any:
            """`_carrier_removed_propagator`, line for line, with one change.

            `/ (delay + 1.0)` becomes `/ 2.0`. Everything else -- the same
            `native.f_grid`, the same `l2_sq_norm`, the same `jnp.where` for
            negative `z`, the same `ifftshift` over `native.spatial_dims` -- is
            copied from the original rather than rewritten, because the claim is
            that *one factor* is the whole of the Fresnel approximation and a
            paraphrase would test the paraphrase.

            The first attempt did rewrite the tail, hardcoding `axes=(1, 2)`
            instead of `native.spatial_dims`, and raised `IndexError: tuple index
            out of range`. Recorded because it is the argument for copying.
            """
            _, jnp, _ = solver.import_backend()
            from chromatix.utils import l2_sq_norm

            wavelength_native = native.broadcasted_wavelength
            frequency_squared = l2_sq_norm(native.f_grid)
            relative_phase = (
                -2.0
                * jnp.pi
                * jnp.abs(distance_m)
                * (wavelength_native / refractive_index)
                * frequency_squared
                / 2.0
            )
            kernel = jnp.exp(1j * relative_phase)
            kernel = jnp.where(distance_m >= 0, kernel, jnp.conj(kernel))
            return jnp.fft.ifftshift(kernel, axes=native.spatial_dims)

        solver._carrier_removed_propagator = substituted
        patched = difference()
        patched_phase = phase_difference()
    finally:
        solver._carrier_removed_propagator = original

    return [
        Row(
            case="1-kernel-identity",
            configuration={
                **GRID_101,
                "pad_width": 0,
                "seed": 20260903,
                "substitution": "delay + 1.0 -> 2.0 in _carrier_removed_propagator",
                "compared": "propagate(asm_carrier_removed) vs fresnel_propagate, complex output",
            },
            descriptor="O_ASM_PROPAGATE vs O_FRESNEL_PROPAGATE",
            # The recorded figure is "exactly 0.0" on the arrays and it does NOT
            # reproduce: they differ by ~6e-5. So the row fails against the record
            # as written, and what it establishes instead -- that the difference sits
            # at the float32 floor and the substitution is otherwise exact -- is
            # carried in `extra`. Passing this would have been the harness deciding
            # the record meant something looser than it says.
            status="FAIL" if kernel_arrays["max_abs_difference"] != 0.0 else "PASS",
            measured={
                "per_bin_phase_difference_rad_patched": patched_phase,
                "per_bin_phase_difference_rad_unpatched": unpatched_phase,
                "field_max_abs_difference_patched": patched,
                "field_max_abs_difference_unpatched": unpatched,
                "kernel_array_max_abs_difference": kernel_arrays["max_abs_difference"],
                "kernel_array_max_phase_difference_rad": kernel_arrays[
                    "max_phase_difference_rad"
                ],
                "probe_bins": list(phase_bins),
                "float32_phase_floor_rad": float32_phase_floor,
                "largest_probe_phase_argument_rad": largest_phase_rad,
                "collapse_factor_from_the_substitution": unpatched_phase / patched_phase,
            },
            expected={
                "recorded_claim": EXPECTED["kernel_identity_max_difference"],
                "recorded_claim_source": (
                    "src/backends/chromatix/solver.py module docstring, repeated in "
                    "src/operations/catalog.py: 'reproduces the Fresnel phase with a maximum "
                    "difference of exactly 0.0 in float32 over a 512^2 grid'"
                ),
            },
            deltas={
                "kernel_array_max_abs_difference": kernel_arrays["max_abs_difference"],
                "per_bin_phase_difference_rad_patched": patched_phase,
                "field_max_abs_difference_patched": patched,
            },
            worst_relative_delta=kernel_arrays["max_abs_difference"],
            runtime_s=time.perf_counter() - started,
            note=(
                "kernel identity, not a physics gate: an algebraic claim about two of this "
                "project's own kernels. Three quantities are measured because they are three "
                "different things -- the propagator ARRAYS, which is what the record names; "
                "the per-bin PHASE of exactly-represented plane waves through the two public "
                "operations; and the complex FIELD difference on a broadband input, which "
                "additionally carries float32 FFT round-off. Every patched figure has its "
                "unpatched twin beside it, so a patch that failed to apply cannot report "
                "agreement for the wrong reason"
            ),
            extra={
                "classification": "stale-record",
                "why": (
                    "The record's 'exactly 0.0' does not reproduce in the quantity it names. "
                    "Measured on the propagator arrays themselves -- the patched kernel against "
                    "chromatix's own compute_transfer_propagator, same grid, same float32 -- "
                    "the maximum difference is ~6.1e-5, not 0.0. The two are not the same "
                    "expression: chromatix computes "
                    "'-pi * wavelength/n * z * l2_sq_norm(f_grid)' where the patched kernel "
                    "computes '-2 pi |z| (wavelength/n) f_sq / 2.0', a different multiply order "
                    "in float32. So the substitution is algebraically exact and the 'one "
                    "substitution apart' claim holds -- 6.1e-5 is below this row's own predicted "
                    "float32 floor of 9.4e-5 -- but 'exactly 0.0' is overstated and should read "
                    "'to float32 round-off, ~6e-5 rad'."
                ),
                "affects": [
                    "src/backends/chromatix/solver.py (module docstring)",
                    "src/operations/catalog.py (O_FRESNEL_PROPAGATE approximation)",
                ],
                "not_fixed_here": (
                    "CHE-240's non-goals forbid converting a stale recorded expectation into a "
                    "new baseline without a separate ticket"
                ),
            },
        )
    ]


# --- 2 and 3. Fresnel vs exact ASM on 101's grid ---------------------------


def _edge_case(*, hard: bool, pad_widths: tuple[int, ...]) -> list[Row]:
    """The recorded 2.3e-1 / 4.9e-6, over a padding sweep.

    The hard-edged case is a square aperture of half the grid; the soft-edged one
    is the same aperture with a super-Gaussian edge, which is what
    `VALIDITY_NOTES` means by "a soft-edged field on the identical grid". The
    padding sweep is what tests the recorded claim's *second* half -- that the
    difference is pad-independent, i.e. a kernel effect rather than a wraparound
    artifact.
    """
    shape, pitch = GRID_101["shape"], GRID_101["pitch_m"]
    wavelength, index = GRID_101["wavelength_m"], GRID_101["refractive_index"]
    distance = GRID_101["distance_m"]

    y, x = _coordinates(shape, pitch)
    grid_y, grid_x = np.meshgrid(y, x, indexing="ij")
    half_width = 0.25 * shape[1] * pitch[1]
    if hard:
        aperture = (
            (np.abs(grid_x) <= half_width) & (np.abs(grid_y) <= half_width)
        ).astype(np.complex64)
        key, expected = "hard_edge_intensity_difference_over_peak", EXPECTED[
            "hard_edge_intensity_difference_over_peak"
        ]
    else:
        # A **circular** super-Gaussian, and the geometry is a finding rather than
        # a default. The record says "a soft-edged field on the identical grid" -- a
        # field, not "the same aperture softened" -- and the two readings do not give
        # the same number: measured on this grid, softening the *square* aperture
        # gives 1.8e-4 while a radially symmetric soft field gives 6e-7, against a
        # recorded 4.9e-6 that only the second family brackets.
        # `sweep_soft_edge_profile_sensitivity` measures both families, which is what
        # turns that from a guess into an attribution.
        #
        # On this grid the 8th power's 0.9 -> 1.1 transition spans about 26 samples
        # (7.7 um), a genuinely soft edge rather than the "few samples" an earlier
        # version of this comment claimed.
        radial = np.sqrt(grid_x**2 + grid_y**2) / half_width
        aperture = np.exp(-(radial**8)).astype(np.complex64)
        key, expected = "soft_edge_intensity_difference_over_peak", EXPECTED[
            "soft_edge_intensity_difference_over_peak"
        ]

    field = _field(aperture, pitch, wavelength_m=wavelength, medium_index=index)

    rows: list[Row] = []
    measured_by_pad: dict[int, float] = {}
    for pad_width in pad_widths:
        started = time.perf_counter()
        difference = _intensity_difference_over_peak(
            _paraxial(field, distance, pad_width=pad_width),
            _exact(field, distance, pad_width=pad_width),
        )
        measured_by_pad[pad_width] = difference
        # A factor-of-two band, and its basis is *setup underspecification* rather
        # than significant figures -- an earlier comment claimed the record states
        # one significant figure, which is arithmetically wrong; it states two. What
        # the record does not state is the aperture size or the edge profile, and
        # the two sensitivity sweeps measure how far each moves the number: the
        # aperture size alone spans a factor of a few and the edge profile three
        # orders of magnitude. So a point figure was never reproducible to better
        # than a factor, and the band is that fact rather than a tolerance.
        within = 0.5 * expected <= difference <= 2.0 * expected
        rows.append(
            Row(
                case=f"{'2-hard-edge' if hard else '3-soft-edge'}",
                configuration={
                    **GRID_101,
                    "pad_width": pad_width,
                    "edge": "hard" if hard else "circular_super_gaussian_8",
                },
                descriptor="O_FRESNEL_PROPAGATE vs O_ASM_PROPAGATE",
                status="PASS" if within else "FAIL",
                measured={key: difference},
                expected={key: expected, "band": "within a factor of 2 of the recorded figure"},
                deltas={f"{key}_ratio_to_recorded": difference / expected},
                worst_relative_delta=abs(difference / expected - 1.0),
                runtime_s=time.perf_counter() - started,
                note=(
                    "DIAGNOSTIC. Fresnel against this repository's own angular spectrum, which "
                    "AGENTS.md forbids as a correctness gate. What this reproduces is the "
                    "figure recorded in representations.VALIDITY_NOTES['paraxial'], not a claim "
                    "that either kernel is right"
                ),
                extra=(
                    {}
                    if hard
                    else {
                        "classification": "setup",
                        "why": (
                            "the record does not name its soft edge, and the profile alone "
                            "spans three orders of magnitude on this grid with the recorded "
                            "4.9e-6 inside it -- see 3-soft-edge-profile-sensitivity. A FAIL "
                            "here is 'this profile does not reproduce the point figure', not "
                            "'the record is stale' and not 'the kernel changed'"
                        ),
                    }
                ),
            )
        )

    spread = max(measured_by_pad.values()) / min(measured_by_pad.values())
    # Whether pad-independence is even a meaningful question at this magnitude.
    # `|U|^2` in complex64 has a relative floor of order 1e-7, so a difference of
    # 6e-7 of peak is a handful of epsilons and its pad-to-pad variation is
    # arithmetic noise rather than a boundary effect. Asserting independence there
    # would be asserting something about round-off. Measured: the hard edge sits at
    # 2e-1 and is flat to 1e-6 across four paddings; the soft edge sits at 6e-7 and
    # varies by 18%, which is what a few epsilons look like.
    floor = 100.0 * float(np.finfo(np.float32).eps)
    meaningful = min(measured_by_pad.values()) > floor
    rows.append(
        Row(
            case=f"{'2-hard-edge' if hard else '3-soft-edge'}-pad-independence",
            configuration={**GRID_101, "pad_widths": list(pad_widths), "complex64_floor": floor},
            descriptor="O_FRESNEL_PROPAGATE vs O_ASM_PROPAGATE",
            status="PASS" if (spread < 1.1 or not meaningful) else "FAIL",
            measured={
                "by_pad_width": {str(key): value for key, value in measured_by_pad.items()},
                "max_over_min": spread,
                "above_the_complex64_floor": meaningful,
            },
            expected={"max_over_min": "< 1.1 -- the record calls the difference pad-independent"},
            deltas={"max_over_min": spread},
            worst_relative_delta=abs(spread - 1.0),
            runtime_s=0.0,
            note=(
                "the second half of the recorded claim: pad-independence is what says the "
                "difference is the kernel and not wraparound. Only asked where the difference "
                "is above the complex64 intensity floor -- below it, the pad-to-pad variation "
                "is round-off and independence would be a statement about arithmetic. "
                "Diagnostic, same as the rows above"
            ),
        )
    )
    return rows


def sweep_hard_edge_aperture_sensitivity() -> list[Row]:
    """How much of the recorded 2.3e-1 is the aperture size the record does not name.

    The companion to `sweep_soft_edge_profile_sensitivity`, and it exists because
    the hard-edge row reproduces the recorded figure only to +39% (0.32 against
    0.23). Before that counts as "reproduces", the question is whether a point
    figure was ever reproducible: the record says "a hard-edged square aperture" on
    a 512^2 grid and does not say how large it is. This measures the span.

    Not a gate -- a classification aid, and the premise the soft-edge
    classification rests on, so it is measured rather than asserted.
    """
    shape, pitch = GRID_101["shape"], GRID_101["pitch_m"]
    wavelength, index = GRID_101["wavelength_m"], GRID_101["refractive_index"]
    distance = GRID_101["distance_m"]
    recorded = EXPECTED["hard_edge_intensity_difference_over_peak"]

    y, x = _coordinates(shape, pitch)
    grid_y, grid_x = np.meshgrid(y, x, indexing="ij")
    extent = shape[1] * pitch[1]
    fractions = (0.125, 0.25, 0.375, 0.5)

    started = time.perf_counter()
    measured: dict[str, float] = {}
    for fraction in fractions:
        half_width = 0.5 * fraction * extent
        aperture = (
            (np.abs(grid_x) <= half_width) & (np.abs(grid_y) <= half_width)
        ).astype(np.complex64)
        field = _field(aperture, pitch, wavelength_m=wavelength, medium_index=index)
        measured[f"width={fraction:g}_of_extent"] = _intensity_difference_over_peak(
            _paraxial(field, distance, pad_width=256),
            _exact(field, distance, pad_width=256),
        )

    low, high = min(measured.values()), max(measured.values())
    brackets = low <= recorded <= high
    return [
        Row(
            case="2-hard-edge-aperture-sensitivity",
            configuration={
                **GRID_101,
                "pad_width": 256,
                "widths_as_fraction_of_extent": list(fractions),
            },
            descriptor="O_FRESNEL_PROPAGATE vs O_ASM_PROPAGATE",
            status="PASS" if brackets else "FAIL",
            measured={
                "by_aperture_width": measured,
                "span_low": low,
                "span_high": high,
                "span_ratio": high / low,
                "recorded_figure_is_inside_the_span": brackets,
            },
            expected={
                "hard_edge_intensity_difference_over_peak": recorded,
                "claim": "the recorded figure lies inside the span the aperture size spans",
            },
            deltas={"span_ratio": high / low},
            worst_relative_delta=0.0,
            runtime_s=time.perf_counter() - started,
            note=(
                "CLASSIFICATION, not a gate. The hard-edge row reproduces the recorded 2.3e-1 "
                "only to +39%, and the record does not name its aperture size. This measures "
                "how far the size alone moves the number, which is what decides whether +39% "
                "is a setup difference or something else"
            ),
        )
    ]


def sweep_soft_edge_profile_sensitivity() -> list[Row]:
    """How much of the recorded 4.9e-6 is the edge profile the record does not name.

    The record says "a soft-edged field on the identical grid differs by 4.9e-6"
    and does not say what the field is -- neither the edge softness nor the
    aperture *geometry*. This measures ten configurations, two families of five,
    which is what decides whether a non-reproducing figure is a **setup**
    difference or something else -- the classification CHE-240 asks for. Nothing
    here is a gate; the row is a classification aid.

    The two families do not agree, and that is the finding: softening the hard
    case's square aperture spans 1.8e-4 to 1.2e-3 and does **not** reach the
    recorded 4.9e-6, while a radially symmetric soft field spans 4.8e-7 to 5.6e-4
    and brackets it. So the record's soft field is radially symmetric rather than a
    softened square, and the aperture geometry turns out to be a larger term than
    the edge softness.

    The hard-edged case reproducing 2.3e-1 to within the aperture-size span (see
    `sweep_hard_edge_aperture_sensitivity`) is what makes this attributable: the
    grid, the propagation path and the metric are shown right by it on the *same*
    square aperture, so what is left for a soft-edge mismatch is the profile.
    """
    shape, pitch = GRID_101["shape"], GRID_101["pitch_m"]
    wavelength, index = GRID_101["wavelength_m"], GRID_101["refractive_index"]
    distance = GRID_101["distance_m"]
    recorded = EXPECTED["soft_edge_intensity_difference_over_peak"]

    y, x = _coordinates(shape, pitch)
    grid_y, grid_x = np.meshgrid(y, x, indexing="ij")
    half_width = 0.25 * shape[1] * pitch[1]
    # **Both** families, because the record's wording admits both and they do not
    # agree. `max(|x|, |y|)` is the hard case's square aperture softened;
    # `sqrt(x^2 + y^2)` is a radially symmetric soft field of the same half-width.
    # Measuring one family alone would have attributed the mismatch to the edge
    # softness when the aperture *geometry* is the larger term.
    square = np.maximum(np.abs(grid_x), np.abs(grid_y)) / half_width
    circular = np.sqrt(grid_x**2 + grid_y**2) / half_width

    profiles: dict[str, np.ndarray] = {
        f"{name}_super_gaussian_{power}": np.exp(-(coordinate**power))
        for name, coordinate in (("square", square), ("circular", circular))
        for power in (2, 4, 8, 16, 32)
    }

    started = time.perf_counter()
    measured: dict[str, float] = {}
    for name, profile in profiles.items():
        field = _field(
            profile.astype(np.complex64), pitch, wavelength_m=wavelength, medium_index=index
        )
        measured[name] = _intensity_difference_over_peak(
            _paraxial(field, distance, pad_width=256),
            _exact(field, distance, pad_width=256),
        )

    by_family = {
        family: {name: value for name, value in measured.items() if name.startswith(family)}
        for family in ("square", "circular")
    }
    spans = {
        family: (min(values.values()), max(values.values()))
        for family, values in by_family.items()
    }
    brackets_in = [family for family, (low, high) in spans.items() if low <= recorded <= high]
    low, high = min(measured.values()), max(measured.values())
    brackets = bool(brackets_in)
    return [
        Row(
            case="3-soft-edge-profile-sensitivity",
            configuration={**GRID_101, "pad_width": 256, "profiles": list(profiles)},
            descriptor="O_FRESNEL_PROPAGATE vs O_ASM_PROPAGATE",
            status="PASS" if brackets else "FAIL",
            measured={
                "by_profile": measured,
                "span_by_family": {
                    family: {"low": low_value, "high": high_value}
                    for family, (low_value, high_value) in spans.items()
                },
                "families_whose_span_contains_the_recorded_figure": brackets_in,
                "span_low": low,
                "span_high": high,
                "span_ratio": high / low,
                "recorded_figure_is_inside_the_span": brackets,
            },
            expected={
                "soft_edge_intensity_difference_over_peak": recorded,
                "claim": (
                    "the recorded figure lies inside the span of at least one aperture family, "
                    "which is what makes a non-reproducing point figure a setup difference"
                ),
            },
            deltas={"span_ratio": high / low},
            worst_relative_delta=0.0,
            runtime_s=time.perf_counter() - started,
            note=(
                "CLASSIFICATION, not a gate. The record does not name its soft edge, so this "
                "measures how far the profile alone moves the number. If the recorded figure "
                "is inside the span, a mismatch from any one profile is a SETUP difference and "
                "not a stale record or an implementation change"
            ),
        )
    ]


def sweep_hard_edge() -> list[Row]:
    return _edge_case(hard=True, pad_widths=(0, 128, 256, 512))


def sweep_soft_edge() -> list[Row]:
    return _edge_case(hard=False, pad_widths=(0, 128, 256, 512))


# --- 4. the sampling bound that has no refusal -----------------------------


def sweep_sampling_bound() -> list[Row]:
    """`z` against `N pitch^2 / lambda`, on both sides -- and no refusal either side.

    The transfer-function sampling criterion for a convolutional propagation is
    `z <= N dx^2 / lambda`. On 101's grid that is
    `512 * (0.3 um)^2 / 0.532 um = 86.6 um`, so the grid's own `z = 50 um` is
    inside it and `z = 200 um` is well outside.

    The ticket expects a **refusal** at the boundary. There is none: searched
    `MODELS`, `_require_model`, `_require_pad_target_crop`,
    `ScalarField.__post_init__`, the 22 `CONTRACT_CODES` and
    `numerics.REFUSAL_CODES`, and no sampling bound appears in any of them. Both
    rows below therefore come back with a field and no diagnostic, which is the
    finding.

    Both records are preserved, as the ticket requires, and the gap is quantified
    rather than only named: each row carries how far the unpadded result sits from
    a heavily padded one at the same `z`. Beyond the bound the transfer function
    aliases, and the padded and unpadded answers stop agreeing.
    """
    shape, pitch = GRID_101["shape"], GRID_101["pitch_m"]
    wavelength, index = GRID_101["wavelength_m"], GRID_101["refractive_index"]
    bound_m = shape[1] * pitch[1] ** 2 / wavelength

    y, x = _coordinates(shape, pitch)
    grid_y, grid_x = np.meshgrid(y, x, indexing="ij")
    half_width = 0.25 * shape[1] * pitch[1]
    radial = np.sqrt(grid_x**2 + grid_y**2) / half_width
    field = _field(
        np.exp(-(radial**8)).astype(np.complex64),
        pitch,
        wavelength_m=wavelength,
        medium_index=index,
    )

    rows: list[Row] = []
    for label, distance in (("inside", GRID_101["distance_m"]), ("outside", 200e-6)):
        started = time.perf_counter()
        refused: str | None = None
        drift = math.nan
        try:
            unpadded = _exact(field, distance, pad_width=0)
            padded = _exact(field, distance, pad_width=512)
            drift = _intensity_difference_over_peak(unpadded, padded)
        except Exception as error:
            refused = f"{type(error).__name__}: {error}"

        rows.append(
            Row(
                case=f"4-sampling-bound-{label}",
                configuration={
                    **GRID_101,
                    "distance_m": distance,
                    "bound_m": bound_m,
                    "z_over_bound": distance / bound_m,
                    "pad_widths_compared": [0, 512],
                },
                descriptor="O_ASM_PROPAGATE",
                status="NOT-COVERED" if refused is None else "PASS",
                measured={
                    "refused": refused,
                    "unpadded_vs_padded_intensity_difference_over_peak": drift,
                },
                expected={},
                deltas={},
                worst_relative_delta=0.0,
                runtime_s=time.perf_counter() - started,
                note=(
                    "the ticket expects a refusal at this boundary and NONE EXISTS in the tree: "
                    "no sampling bound appears in MODELS, the model validators, "
                    "ScalarField.__post_init__, CONTRACT_CODES or numerics.REFUSAL_CODES. The "
                    "operation returns a field on both sides of the bound with no diagnostic. "
                    "The drift figure quantifies the gap and is DIAGNOSTIC -- it compares two "
                    "runs of the same implementation"
                ),
            )
        )
    return rows


# --- 5. the paraxial bound -------------------------------------------------


def sweep_paraxial_bound() -> list[Row]:
    """The three direction cosines the record names, against the closed-form residual.

    `sin(theta_max) <= (lambda_0 / (n z))^(1/4)` is the bound, and the record says
    the leading error reaches `pi/4` at it. On 101's grid that bound is 0.2988
    (17.4 deg); the per-axis Nyquist frequency `1/(2 dx)` corresponds to 0.6667 and
    the grid corner to `sqrt(2)` times that, 0.9428.

    The oracle is the closed form `n k0 z (1 - cos(theta) - sin^2(theta)/2)`,
    which is arithmetic and may decide. What is *measured* is the phase difference
    between the two public propagations of the same plane wave -- and it is
    measured **wrapped**, because 175 rad is 27 turns and a complex64 field cannot
    hold more. So two numbers appear on each row: the unwrapped closed-form
    prediction, which is what the record's figures are, and the wrapped residual
    between measurement and prediction, which is what says the measurement agrees.
    """
    shape, pitch = GRID_101["shape"], GRID_101["pitch_m"]
    wavelength, index = GRID_101["wavelength_m"], GRID_101["refractive_index"]
    distance = GRID_101["distance_m"]

    # Each case is a **DFT bin pair**, not a target direction cosine, and the sine
    # is computed from the bins. Not a refinement -- the difference between
    # measuring the kernel and measuring an aliasing artifact.
    #
    # The record quotes its three figures at the cosines 0.299, 0.667 and 0.943;
    # evaluated in float64 the bound is 0.2990697562442441.
    # The first two are single-axis frequencies this grid represents; the third is
    # not. 0.943 is `sqrt(2)` times the per-axis Nyquist, so it exists only as the
    # *corner* of the 2-D frequency grid -- `fx = fy = Nyquist` -- and a single-axis
    # ramp at that frequency is above Nyquist and aliases. Measured: the first
    # version of this sweep did exactly that and came back with a 2.41 rad residual
    # against the closed form, which was the alias and not a disagreement.
    #
    # Bin `N//2 - 1` rather than `N//2`, because the Nyquist bin is the degenerate
    # `+/-` bin where a ramp advances by exactly pi per sample and the ratio
    # estimate is ill-conditioned. So the cosines probed sit just below the
    # record's, and every row carries both: the closed form at the cosine actually
    # probed, which the measurement is held to, and the same closed form at the
    # cosine the record quotes, which is what reproduces the record's figure.
    bound = (wavelength / (index * distance)) ** 0.25
    bound_bin = round(index * bound / wavelength * shape[1] * pitch[1])
    edge_bin = shape[1] // 2 - 1
    nyquist_sine = wavelength / (2.0 * pitch[1]) / index

    cases = (
        ("paraxial_bound", (0, bound_bin), EXPECTED["paraxial_bound_phase_error_rad"], bound),
        ("per_axis_nyquist", (0, edge_bin), EXPECTED["nyquist_phase_error_rad"], nyquist_sine),
        (
            "grid_corner",
            (edge_bin, edge_bin),
            EXPECTED["corner_phase_error_rad"],
            math.sqrt(2.0) * nyquist_sine,
        ),
    )

    rows: list[Row] = []
    for label, bins, recorded, ideal_sine in cases:
        started = time.perf_counter()
        frequency_y = bins[0] / (shape[0] * pitch[0])
        frequency_x = bins[1] / (shape[1] * pitch[1])
        sine = (wavelength / index) * math.hypot(frequency_y, frequency_x)
        source = _plane_wave_at_bins(
            bins,
            shape=shape,
            pitch_m=pitch,
            wavelength_m=wavelength,
            refractive_index=index,
        )
        approximate = _uniform_advance_rad(_paraxial(source, distance, pad_width=0), source)
        exact = _uniform_advance_rad(_exact(source, distance, pad_width=0), source)

        def _closed_form(value: float) -> float:
            """`n k0 z (1 - cos(theta) - sin^2(theta)/2)`, the exact residual."""
            return (
                2.0
                * math.pi
                * index
                * distance
                / wavelength
                * (1.0 - math.sqrt(1.0 - value**2) - 0.5 * value**2)
            )

        def _leading_term(value: float) -> float:
            """`n k0 z sin^4(theta)/8`, the first term of the same series.

            Carried because the record's `pi/4` is the **leading term** at the
            bound and not the full residual -- `VALIDITY_NOTES` says so literally,
            "at which the leading error reaches pi/4" -- and at the bound the
            leading term is `pi/4` **exactly**, by construction: `n k0 z sin^4/8` at
            `sin = (lambda_0/(n z))^(1/4)` is `n k0 z (lambda_0/(n z))/8 = 2 pi/8`.
            Measured agreement 3.3e-16 relative. The full residual at the same
            cosine is 0.82262, 4.7% above it, so reporting only the residual would
            make a correctly-stated record look 5% wrong. The record's other two
            figures, 25.5 and 175, are the full residual (25.4640 and 174.533, 0.14%
            and 0.27%) and not the leading term (19.393 and 77.570) -- an asymmetry
            that is the record's, reproduced rather than smoothed over.
            """
            return 2.0 * math.pi * index * distance / wavelength * value**4 / 8.0

        predicted = _closed_form(sine)
        recorded_sine_prediction = _closed_form(ideal_sine)
        recorded_sine_leading = _leading_term(ideal_sine)
        residual = abs(_wrapped(approximate - exact - predicted))
        rows.append(
            Row(
                case=f"5-paraxial-bound-{label}",
                configuration={
                    **GRID_101,
                    "pad_width": 0,
                    "dft_bins_yx": list(bins),
                    "sin_theta_probed": sine,
                    "sin_theta_quoted_in_the_record": ideal_sine,
                    "theta_deg": math.degrees(math.asin(min(sine, 1.0))),
                },
                descriptor="O_FRESNEL_PROPAGATE vs O_ASM_PROPAGATE",
                status=(
                    "PASS"
                    if residual < 5e-3
                    and min(
                        abs(recorded_sine_prediction / recorded - 1.0),
                        abs(recorded_sine_leading / recorded - 1.0),
                    )
                    < 0.05
                    else "FAIL"
                ),
                measured={
                    "closed_form_phase_error_rad_at_probed_sine": predicted,
                    "closed_form_phase_error_rad_at_recorded_sine": recorded_sine_prediction,
                    "leading_term_rad_at_recorded_sine": recorded_sine_leading,
                    "wrapped_residual_rad": residual,
                    "wrapped_measured_difference_rad": _wrapped(approximate - exact),
                },
                expected={"recorded_phase_error_rad": recorded},
                deltas={
                    "recorded_figure_relative_full_residual": abs(
                        recorded_sine_prediction / recorded - 1.0
                    ),
                    "recorded_figure_relative_leading_term": abs(
                        recorded_sine_leading / recorded - 1.0
                    ),
                    "wrapped_residual_rad": residual,
                },
                worst_relative_delta=max(
                    residual, abs(recorded_sine_prediction / recorded - 1.0)
                ),
                runtime_s=time.perf_counter() - started,
                note=(
                    "the closed form n k0 z (1 - cos - sin^2/2) is the oracle and it is "
                    "arithmetic, so it may decide. The measured difference is WRAPPED -- 175 rad "
                    "is 27 turns and complex64 holds none of them -- so what gates is the "
                    "wrapped residual against the closed form at the cosine ACTUALLY probed, "
                    "and the record's figure is reproduced separately by evaluating the same "
                    "closed form at the cosine the record quotes"
                ),
            )
        )
    return rows


# --- 6. the tilted beam ----------------------------------------------------

#: Inherited from `tests/physics/test_fresnel_propagation.py` rather than
#: re-chosen: at 200 um the beam walks 34.7 um against a 64 um half-extent, and
#: widening either the angle or the distance walks it off the window, which would
#: replace a physics measurement with a truncation artifact.
WALKOFF: dict[str, Any] = {
    "shape": (256, 256),
    "pitch_m": (0.5e-6, 0.5e-6),
    "distance_m": 200e-6,
    "waist_m": 12e-6,
    "wavelength_m": 0.532e-6,
    "refractive_index": 1.0,
    "pad_width": 256,
}


def sweep_tilted_beam() -> list[Row]:
    """`z sin(theta)` against `z tan(theta)`, and the group delay's linearity.

    Two claims in the ticket and they are separate measurements. The landing point
    distinguishes the two candidate oracles only where they separate: at 2 deg they
    are 0.01 samples apart on this grid and at 10 deg they are 1.07, so the
    separation is reported on every row and the discrimination is only claimed
    where it exceeds the tolerance. Asserting it at 2 deg would be claiming a
    resolution the geometry does not have.

    The group delay `lambda_0 z f / n` being *linear in spatial frequency* is the
    reason `z sin(theta)` rather than `z tan(theta)` is where the beam lands, so it
    is checked as a linearity: three angles, and the ratio
    `landing / sin(theta)` constant across them.
    """
    shape, pitch = WALKOFF["shape"], WALKOFF["pitch_m"]
    wavelength, index = WALKOFF["wavelength_m"], WALKOFF["refractive_index"]
    distance, pad_width = WALKOFF["distance_m"], WALKOFF["pad_width"]
    wavenumber = 2.0 * math.pi / wavelength

    y, x = _coordinates(shape, pitch)
    grid_y, grid_x = np.meshgrid(y, x, indexing="ij")
    envelope = np.exp(-(grid_x**2 + grid_y**2) / WALKOFF["waist_m"] ** 2)

    rows: list[Row] = []
    ratios: list[float] = []
    for theta_deg in (2.0, 5.0, 10.0):
        started = time.perf_counter()
        theta = math.radians(theta_deg)
        ramp = np.exp(1j * wavenumber * math.sin(theta) * grid_x)
        field = _field(
            (envelope * ramp).astype(np.complex64),
            pitch,
            wavelength_m=wavelength,
            medium_index=index,
        )
        out = _paraxial(field, distance, pad_width=pad_width)
        intensity = np.abs(np.asarray(out.u)).astype(np.float64) ** 2
        _, out_x = _coordinates(shape, out.sample_pitch_m)
        landing = float((intensity * out_x[None, :]).sum() / intensity.sum())

        sin_prediction = distance * math.sin(theta)
        tan_prediction = distance * math.tan(theta)
        separation_samples = abs(tan_prediction - sin_prediction) / pitch[1]
        error_samples = abs(landing - sin_prediction) / pitch[1]
        distinguishes = separation_samples > 10.0 * 0.05
        rejected = abs(landing - tan_prediction) / pitch[1] > separation_samples / 2.0
        ratios.append(landing / math.sin(theta))

        rows.append(
            Row(
                case=f"6-tilted-beam-{theta_deg:g}deg",
                configuration={**WALKOFF, "theta_deg": theta_deg},
                descriptor="O_FRESNEL_PROPAGATE",
                status=(
                    "PASS"
                    if error_samples < 0.05 and (not distinguishes or rejected)
                    else "FAIL"
                ),
                measured={
                    "landing_x_m": landing,
                    "error_samples": error_samples,
                    "separation_samples": separation_samples,
                    "distinguishes_the_two_oracles": distinguishes,
                    "z_tan_theta_rejected": rejected if distinguishes else None,
                },
                expected={
                    "z_sin_theta_m": sin_prediction,
                    "z_tan_theta_m_rejected": tan_prediction,
                },
                deltas={"error_samples": error_samples},
                worst_relative_delta=error_samples,
                runtime_s=time.perf_counter() - started,
                note=(
                    "closed form: the Fresnel kernel's group delay is lambda_0 z f / n, so the "
                    "beam lands at z sin(theta) where the exact angular spectrum lands it at "
                    "z tan(theta). Both are arithmetic. `distinguishes_the_two_oracles` says "
                    "whether this angle can tell them apart at all -- at 2 deg it cannot, and "
                    "claiming otherwise would claim a resolution the grid does not have"
                ),
            )
        )

    spread = max(ratios) / min(ratios)
    rows.append(
        Row(
            case="6-tilted-beam-group-delay-linearity",
            configuration={**WALKOFF, "theta_deg": [2.0, 5.0, 10.0]},
            descriptor="O_FRESNEL_PROPAGATE",
            status="PASS" if abs(spread - 1.0) < 1e-3 else "FAIL",
            measured={"landing_over_sin_theta_m": ratios, "max_over_min": spread},
            expected={"landing_over_sin_theta_m": distance, "max_over_min": 1.0},
            deltas={"max_over_min_minus_one": abs(spread - 1.0)},
            worst_relative_delta=abs(spread - 1.0),
            runtime_s=0.0,
            note=(
                "the group delay is linear in spatial frequency, so landing / sin(theta) is the "
                "constant z at every angle. Three angles, because one cannot show a relation is "
                "linear rather than fitted at a point"
            ),
        )
    )
    return rows


# --- 7. the focal-plane transform ------------------------------------------

FOCAL: dict[str, Any] = {
    "shape": (128, 128),
    "pitch_m": (0.5e-6, 0.5e-6),
    "focal_length_m": 20e-3,
    "wavelength_m": 0.532e-6,
    "refractive_index": 1.0,
    "theta_deg": 20.0,
}


def sweep_focal_plane_transform() -> list[Row]:
    """Focus at `f sin(theta)`, and the output grid at `lambda f / (n N dx)` per axis.

    Both claims are closed forms and both are checked. The pitch is checked on an
    **asymmetric** grid -- different sample counts *and* different pitches per axis
    -- because on a square grid a transposed `(y, x)` passes.

    20 degrees is the angle: `f sin = 6.8404 mm` against `f tan = 7.2794 mm`, and
    the output pitch is 166.3 um, so the two candidates are 2.6 samples apart and
    the measurement can choose. The envelope is a super-Gaussian rather than a hard
    window because a step edge puts power past the light cone, which would broaden
    the focus for a reason unrelated to the claim.
    """
    rows: list[Row] = []

    # 7a. the output pitch, on an asymmetric grid.
    started = time.perf_counter()
    shape, pitch = (48, 64), (0.30e-6, 0.25e-6)
    u = np.zeros(shape)
    u[shape[0] // 2, shape[1] // 2] = 1.0
    out = focal_plane_transform(
        _field(
            u,
            pitch,
            wavelength_m=FOCAL["wavelength_m"],
            medium_index=FOCAL["refractive_index"],
        ),
        focal_length_m=FOCAL["focal_length_m"],
        model={"target_surface": "back_focal"},
    )
    analytic = tuple(
        FOCAL["wavelength_m"]
        * FOCAL["focal_length_m"]
        / (FOCAL["refractive_index"] * count * step)
        for count, step in zip(shape, pitch, strict=True)
    )
    pitch_error = max(
        abs(measured / expected - 1.0)
        for measured, expected in zip(out.sample_pitch_m, analytic, strict=True)
    )
    rows.append(
        Row(
            case="7-focal-plane-output-pitch",
            configuration={
                "shape": shape,
                "pitch_m": pitch,
                "focal_length_m": FOCAL["focal_length_m"],
                "wavelength_m": FOCAL["wavelength_m"],
                "refractive_index": FOCAL["refractive_index"],
            },
            descriptor="O_FOCAL_PLANE_TRANSFORM",
            status="PASS" if pitch_error == 0.0 else "FAIL",
            measured={
                "sample_pitch_m": list(out.sample_pitch_m),
                "axes_differ": out.sample_pitch_m[0] != out.sample_pitch_m[1],
                "helper_agrees": out.sample_pitch_m
                == fourier_plane_pitch_m(
                    pitch,
                    shape,
                    wavelength_m=FOCAL["wavelength_m"],
                    focal_length_m=FOCAL["focal_length_m"],
                    medium_index=FOCAL["refractive_index"],
                ),
            },
            expected={"sample_pitch_m": list(analytic)},
            deltas={"max_relative_pitch_error": pitch_error},
            worst_relative_delta=pitch_error,
            runtime_s=time.perf_counter() - started,
            note=(
                "closed form lambda f / (n N dx) per axis, computed in float64 from the declared "
                "pitch and shape. Asymmetric in both count and pitch, so a transposed (y, x) "
                "cannot pass"
            ),
        )
    )

    # 7b. the focus position.
    started = time.perf_counter()
    grid = FOCAL["shape"][0]
    step = FOCAL["pitch_m"][0]
    theta = math.radians(FOCAL["theta_deg"])
    coordinate = (np.arange(grid) - grid // 2) * step
    y, x = np.meshgrid(coordinate, coordinate, indexing="ij")
    envelope = np.exp(-(((x**2 + y**2) / (0.4 * grid * step) ** 2) ** 4))

    measurements: list[dict[str, Any]] = []
    worst = 0.0
    for sign in (1.0, -1.0):
        transverse_wavenumber = sign * 2.0 * math.pi * math.sin(theta) / FOCAL["wavelength_m"]
        out = focal_plane_transform(
            _field(
                envelope * np.exp(1j * transverse_wavenumber * x),
                (step, step),
                wavelength_m=FOCAL["wavelength_m"],
                medium_index=FOCAL["refractive_index"],
            ),
            focal_length_m=FOCAL["focal_length_m"],
            model={"target_surface": "back_focal"},
        )
        intensity = np.abs(np.asarray(out.u)).astype(np.float64) ** 2
        row = intensity[grid // 2, :]
        peak = int(np.argmax(row))
        window = slice(max(0, peak - 6), min(grid, peak + 7))
        indices = np.arange(grid)[window]
        centroid = float((row[window] * indices).sum() / row[window].sum())
        landing = (centroid - grid // 2) * out.sample_pitch_m[1]

        sin_prediction = sign * FOCAL["focal_length_m"] * math.sin(theta)
        tan_prediction = sign * FOCAL["focal_length_m"] * math.tan(theta)
        relative = abs(landing / sin_prediction - 1.0)
        worst = max(worst, relative)
        measurements.append(
            {
                "sign": sign,
                "landing_x_m": landing,
                "f_sin_theta_m": sin_prediction,
                "f_tan_theta_m_rejected": tan_prediction,
                "relative_error": relative,
                "separation_samples": abs(tan_prediction - sin_prediction)
                / out.sample_pitch_m[1],
                "tan_rejected": abs(landing - tan_prediction) > 2.0 * out.sample_pitch_m[1],
                "focus_stayed_on_the_y_axis": int(np.argmax(intensity[:, peak])) == grid // 2,
            }
        )
    rows.append(
        Row(
            case="7-focal-plane-f-sin-theta",
            configuration={**FOCAL},
            descriptor="O_FOCAL_PLANE_TRANSFORM",
            status=(
                "PASS"
                if worst < 1e-4
                and all(
                    entry["tan_rejected"] and entry["focus_stayed_on_the_y_axis"]
                    for entry in measurements
                )
                else "FAIL"
            ),
            measured={"per_sign": measurements},
            expected={"landing_x_m": "f sin(theta), and NOT f tan(theta)"},
            deltas={"worst_relative_error": worst},
            worst_relative_delta=worst,
            runtime_s=time.perf_counter() - started,
            note=(
                "closed form. Both signs, because a sign error is invisible on one. The "
                "y-axis check is what a transposition would break and is invisible in a "
                "rotationally symmetric case"
            ),
        )
    )
    return rows


SWEEPS: dict[str, Callable[[], list[Row]]] = {
    "1-kernel-identity": sweep_kernel_identity,
    "2-hard-edge": sweep_hard_edge,
    "2-hard-edge-aperture-sensitivity": sweep_hard_edge_aperture_sensitivity,
    "3-soft-edge": sweep_soft_edge,
    "3-soft-edge-profile-sensitivity": sweep_soft_edge_profile_sensitivity,
    "4-sampling-bound": sweep_sampling_bound,
    "5-paraxial-bound": sweep_paraxial_bound,
    "6-tilted-beam": sweep_tilted_beam,
    "7-focal-plane-transform": sweep_focal_plane_transform,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", action="append", default=None, help="run only these sweeps")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/che-238-overnight/workstream-b/kernel_sweep.json"),
    )
    arguments = parser.parse_args(argv)

    rows: list[Row] = []
    for name in arguments.only or list(SWEEPS):
        started = time.perf_counter()
        try:
            produced = SWEEPS[name]()
        except Exception as error:
            produced = [
                Row(
                    case=name,
                    configuration={},
                    descriptor="",
                    status="FAIL",
                    measured={"exception": type(error).__name__, "message": str(error)},
                    expected={"outcome": "the sweep runs"},
                    deltas={},
                    worst_relative_delta=math.inf,
                    runtime_s=time.perf_counter() - started,
                    note="the sweep raised before it could produce a measurement",
                )
            ]
        rows.extend(produced)
        statuses = ", ".join(sorted({row.status for row in produced}))
        print(
            f"{name:26s} {len(produced):2d} row(s) [{statuses}]"
            f"  {time.perf_counter() - started:6.1f}s",
            flush=True,
        )

    record = {
        "workstream": "B",
        "ticket": "CHE-240",
        "produced_by": "benchmarks/verification/wave_kernel_sweep.py",
        "grid_101": GRID_101,
        "recorded_expectations": EXPECTED,
        **provenance(),
        "rows": [row.as_dict() for row in rows],
    }
    return finish(record, path=arguments.out, rows=rows)


if __name__ == "__main__":
    raise SystemExit(main())
