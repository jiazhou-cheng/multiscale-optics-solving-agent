#!/usr/bin/env python3
"""demo3 estimator variance — where it comes from, and what can be done (CHE-120).

M5.1 and M5.2 moved the constant: the trace and the emitter got faster and the
field did not change by one bit. This probe is about the exponent. demo3's
seed-to-seed NCC ladder says the estimator is noise-limited and extrapolates to
1.7e9-2.4e9 rays for NCC 0.9, against the paper's own 2.6e9. Throughput cannot
touch that number. Only variance can.

Three measurements, and they answer three different questions.

``--stage decomposition`` -- **where the variance is.**
    The estimator draws twice, so its variance has exactly two terms:

        V(P, S) = A / P + B / (P S)

    ``A`` is the part that falls only with ``P`` and ``B`` the part that falls
    with ``P S``. That is a statement about how ``V`` responds to the two knobs
    and **not** a split into a reducible and an irreducible share: an importance
    density on positions multiplies whole patches, so it scales both terms.
    Multiplying through, ``P V = A + B / S``
    is *linear in 1/S*, so a sweep in ``S`` at fixed ``P`` separates them by a
    straight-line fit rather than by an argument. This is the deliverable's
    "variance decomposition", and it is also what decides the allocation below,
    because the cost is ``c(P, S) = P (a + b S)`` -- the padded transform is per
    *patch* and the draw, trace and reconstruction are per *ray* -- and
    minimizing ``V`` at fixed ``c`` gives

        S* = sqrt( (a B) / (b A) ).

``--stage candidates`` -- **whether the variance can be reduced.**
    Position densities and stratified draws from `couplers.patch_positions`, at
    matched ray count, each with the anti-bias test below. The direction axis is
    deliberately *not* a candidate: with ``p = |U~| / ||U~||_1`` the per-ray
    weight has constant modulus within a patch, which is the Cauchy-Schwarz
    optimum, so there is nothing to win there and a candidate claiming otherwise
    would be measuring noise.

``--stage allocation`` -- **whether the split is right.**
    ``(P, S)`` pairs at fixed ``P x S``. Fixed ray count is *not* fixed cost
    here, which is the point: both are reported, and the smallest ``P x S``
    meeting a declared variance target is the answer rather than the largest run
    that finished.

Metric. Seed-to-seed intensity NCC is the coordinate the committed records use
and is reported for continuity, with its noise floor ``3 / sqrt(N_px)`` in every
record. But NCC saturates, is nonlinear in the noise, and at low k-space
oversampling stops tracking the estimator at all, so the *variance* work is
scored on the absolute field variance

    V = sum_px Var_r[F_r(px)],

estimated over ``R >= 3`` seeds with a leave-one-out error bar. Absolute rather
than signal-normalized because every arm estimates the same field with the same
``one_over_n`` normalization, so ``V`` is already comparable, its ratios are
exact, and it has neither a floor nor a saturation. See `variance_statistics`
for what normalizing would have cost.

Anti-bias, on the CHE-101 pattern and one step past it. CHE-101 checked that a
fast path had not moved the seed-to-seed noise. That is necessary and it is not
sufficient: an estimator can keep its noise and shift its mean. So each arm is
also compared to the baseline through

    bias_ratio = ||F_bar_A - F_bar_B||^2 / (V_A / R_A + V_B / R_B),

which is 1 when both estimators have the same mean and grows without bound when
they do not. Its own fluctuation is ~1/sqrt(N_px) because it pools every pixel,
so it is a sharp test rather than a formality -- and `tests/test_patch_positions.py`
holds the exact version of the same question, against an enumerated oracle, at a
size where one exists.

Run sequentially in a dedicated GPU session (GPU 6 or 7), one stage per command:
    MOA_GPUS=device=6 ./run.sh --gpu python \\
        benchmarks/probes/ray_wave/demo3_variance.py --stage decomposition
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _demo_support import (
    build_doe,
    device_memory_stats,
    enable_x64_if_needed,
    environment,
    ncc,
    write_record,
)
from demo3_hologram_lens import (
    GRID_N,
    PITCH_M,
    PRESETS,
    demo3_system,
    run_route,
)

from core.precision import DeviceKind, DevicePlacement, Precision
from couplers.patch_positions import (
    PositionDensity,
    PositionDraw,
    candidate_index_grid,
    plan_positions,
    predicted_variance_ratio,
    spectral_l1_map,
    window_energy_map,
    window_sample_count_map,
)
from solvers.optiland.builder import build_optiland_system
from solvers.optiland.coherent_trace import configure_optiland_execution

#: ``S`` values for the decomposition fit, at fixed ``P``. Four points rather
#: than two: ``P V = A + B / S`` is a two-parameter model and a two-point fit
#: returns it exactly whatever the data are, which is the mistake the earlier
#: demo3 ladder made in the other direction (see `demo3_convergence.LADDER`).
SECONDARY_LADDER = (2_500, 5_000, 10_000, 20_000)

#: ``(P, S)`` pairs at fixed ``P x S = 2e7``. Spanning a factor of 12 in ``P``,
#: which is what makes the per-patch and per-ray cost terms separable in the
#: same runs.
ALLOCATION = ((400, 50_000), (1_000, 20_000), (2_500, 8_000), (5_000, 4_000))

#: The candidate estimators. ``(label, density, draw)``. The first is the
#: shipped one and is bitwise it, so the comparison has a real control arm.
CANDIDATES: tuple[tuple[str, PositionDensity, PositionDraw], ...] = (
    ("uniform_iid", PositionDensity.UNIFORM, PositionDraw.IID),
    ("uniform_jittered_grid", PositionDensity.UNIFORM, PositionDraw.JITTERED_GRID),
    ("sqrt_energy_iid", PositionDensity.SQRT_WINDOW_ENERGY, PositionDraw.IID),
    (
        "sqrt_energy_jittered_grid",
        PositionDensity.SQRT_WINDOW_ENERGY,
        PositionDraw.JITTERED_GRID,
    ),
    (
        "sqrt_energy_stratified_cdf",
        PositionDensity.SQRT_WINDOW_ENERGY,
        PositionDraw.STRATIFIED_CDF,
    ),
)


def variance_statistics(fields: list[np.ndarray]) -> dict[str, Any]:
    """The variance summary for one arm, from ``R`` independent realizations.

    Scored on the **absolute** ``V = sum_px Var_r[F_r(px)]`` rather than on a
    signal-normalized version of it. ``V`` needs no signal estimate at all:
    every arm and every rung is an unbiased estimator of the same field on the
    same grid with the same ``one_over_n`` normalization, so ``V`` is already in
    comparable units, the ratios of it are exact, and it neither saturates nor
    has a noise floor.

    Normalizing would mean dividing by ``sum|F_bar|^2 - V/R`` -- the mean field's
    own power is biased upward by exactly one over ``R`` of the variance. That
    estimate is *usable* at these budgets and it is not free: at ``P = 1000``,
    ``S = 20000``, ``R = 6`` it reads ``301.7 - 245.7 = 56.0``, a 3.7%
    per-realization signal fraction known to a few percent. So the reason for
    not using it is that it would add a few percent of its own error to every
    ratio for no gain, which is a smaller and truer claim than the one this
    docstring made first (it said the denominator carried a ~100% error, which
    the records do not support).

    ``jackknife_relative_error`` is the honest error bar on ``V`` itself:
    leave-one-seed-out over the realizations. Pixels are not independent -- one
    ray sample writes the whole sensor -- so an analytic chi-squared dof would be
    wrong, and a claimed improvement has to clear this rather than a formula.
    """
    stack = np.stack([np.asarray(f, dtype=np.complex128) for f in fields])
    replicates = stack.shape[0]
    mean_field = stack.mean(axis=0)

    def total_variance(sample: np.ndarray) -> float:
        centred = sample - sample.mean(axis=0)
        return float((np.abs(centred) ** 2).sum() / max(sample.shape[0] - 1, 1))

    variance = total_variance(stack)
    if replicates >= 3:
        leave_one_out = np.array(
            [
                total_variance(np.delete(stack, drop, axis=0))
                for drop in range(replicates)
            ]
        )
        jackknife = float(
            np.sqrt((replicates - 1) / replicates * ((leave_one_out - leave_one_out.mean()) ** 2).sum())
        )
    else:
        jackknife = float("nan")

    intensities = [np.abs(f) ** 2 for f in stack]
    pairs = [
        ncc(intensities[i], intensities[j])
        for i in range(replicates)
        for j in range(i + 1, replicates)
    ]
    return {
        "replicates": replicates,
        "field_variance_sum": variance,
        "jackknife_standard_error": jackknife,
        "jackknife_relative_error": jackknife / variance if variance > 0 else float("nan"),
        "mean_field_power": float((np.abs(mean_field) ** 2).sum()),
        "seed_to_seed_ncc_mean": float(np.mean(pairs)),
        "seed_to_seed_ncc_min": float(np.min(pairs)),
        "seed_to_seed_ncc_max": float(np.max(pairs)),
    }


def bias_ratio(a: list[np.ndarray], b: list[np.ndarray]) -> dict[str, Any]:
    """``||F_bar_A - F_bar_B||^2`` over what the two arms' own noise predicts.

    1.0 when both estimators have the same mean. Pools every pixel, so its own
    fluctuation is about ``1 / sqrt(N_px)`` -- reported next to it, because a
    ratio of 1.03 means something very different on 1.76e5 pixels than on 100.
    """
    stack_a = np.stack([np.asarray(f, dtype=np.complex128) for f in a])
    stack_b = np.stack([np.asarray(f, dtype=np.complex128) for f in b])
    mean_a, mean_b = stack_a.mean(axis=0), stack_b.mean(axis=0)
    var_a = float((np.abs(stack_a - mean_a) ** 2).sum() / max(stack_a.shape[0] - 1, 1))
    var_b = float((np.abs(stack_b - mean_b) ** 2).sum() / max(stack_b.shape[0] - 1, 1))
    expected = var_a / stack_a.shape[0] + var_b / stack_b.shape[0]
    observed = float((np.abs(mean_a - mean_b) ** 2).sum())
    pixels = mean_a.size
    # Two terms, and the second is the larger one. The numerator's own scatter:
    # |Delta_k|^2 for COMPLEX noise is exponential, so its variance equals its
    # mean squared and the relative sd of the sum over N pixels is 1/sqrt(N) --
    # not sqrt(2/N), which is the real-Gaussian figure and was what this
    # reported first. The denominator is estimated from the same few seeds, so it
    # carries the arms' jackknife errors, and at R = 6 that dominates 1/sqrt(N)
    # by a factor of three.
    numerator_scatter = float(1.0 / np.sqrt(pixels))
    denominator_scatter = float(
        np.hypot(
            _jackknife_relative(stack_a) * (var_a / stack_a.shape[0]),
            _jackknife_relative(stack_b) * (var_b / stack_b.shape[0]),
        )
        / expected
        if expected > 0
        else float("nan")
    )
    return {
        "observed_mean_field_separation": observed,
        "predicted_from_own_noise": expected,
        "bias_ratio": observed / expected if expected > 0 else float("nan"),
        "fluctuation_from_pixel_count": numerator_scatter,
        "fluctuation_from_variance_estimate": denominator_scatter,
        "expected_fluctuation": float(
            np.hypot(numerator_scatter, denominator_scatter)
        ),
        "reading": (
            "1.0 means the two arms estimate the same field. Read a departure "
            "against `expected_fluctuation`, which combines the pixel-count "
            "scatter with the error on the variance estimate that forms the "
            "denominator -- the second is the larger term at these seed counts, "
            "and omitting it would make a 1.008 look like a 2.5-sigma "
            "detection when it is inside one. Pixel noise in speckle is also "
            "not independent, which pushes the true scale up further, so this "
            "is a lower bound on the fluctuation and not a tight threshold. The "
            "exact unbiasedness gate is the enumerated-oracle test in "
            "tests/test_patch_positions.py, not this."
        ),
    }


def _jackknife_relative(stack: np.ndarray) -> float:
    """Leave-one-out relative error on ``sum_px Var_r``, for one arm."""
    replicates = stack.shape[0]
    if replicates < 3:
        return float("nan")

    def total(sample: np.ndarray) -> float:
        centred = sample - sample.mean(axis=0)
        return float((np.abs(centred) ** 2).sum() / max(sample.shape[0] - 1, 1))

    full = total(stack)
    loo = np.array([total(np.delete(stack, drop, axis=0)) for drop in range(replicates)])
    error = float(
        np.sqrt((replicates - 1) / replicates * ((loo - loo.mean()) ** 2).sum())
    )
    return error / full if full > 0 else float("nan")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=(
            "decomposition",
            "candidates",
            "allocation",
            "ladder",
            "ladderfit",
            "l1map",
        ),
        required=True,
    )
    parser.add_argument("--backend", choices=("numpy", "jax"), default="jax")
    parser.add_argument("--patches", type=int, default=1_000)
    parser.add_argument("--secondary-count", type=int, default=20_000)
    parser.add_argument("--replicates", type=int, default=6)
    parser.add_argument("--seed0", type=int, default=20260824)
    parser.add_argument("--rays-per-chunk", type=float, default=1e6)
    parser.add_argument(
        "--reconstruction", choices=("ramp_sum", "kspace_splat"), default="kspace_splat"
    )
    parser.add_argument("--kspace-oversample", type=float, default=1.5)
    parser.add_argument(
        "--arms",
        default=None,
        help=(
            "comma-separated candidate labels, for splitting one stage across "
            "commands. Every arm still writes its own record and the baseline "
            "arm must be in the same command as anything compared to it."
        ),
    )
    parser.add_argument("--l1-shards", type=int, default=4)
    parser.add_argument(
        "--ladder-patches",
        default="2000,4000,8000",
        help="P values for the ladder stage, at --secondary-count each",
    )
    parser.add_argument("--target-ncc", type=float, default=0.9)
    parser.add_argument(
        "--merge",
        default=None,
        help=(
            "comma-separated existing ladder record names whose rungs are "
            "pooled before fitting. Used to extend a ladder's lever arm across "
            "commands rather than putting a 10-minute run in one."
        ),
    )
    parser.add_argument("--output-name", default=None)
    args = parser.parse_args()

    preset = PRESETS["characterization"]
    settings = dict(preset["rw_p"])
    doe = build_doe("demo3_smile_phase_profile.npy", pitch_m=PITCH_M)
    patch_px = settings["patch_px"] or (GRID_N - 1)

    if args.stage == "l1map":
        return _l1map(args, doe=doe, patch_px=patch_px, settings=settings)
    if args.stage == "ladderfit":
        return _ladderfit(args, sensor_px=preset["sensor_px"])

    enable_x64_if_needed(backend=args.backend, precisions=[settings["precision"]])
    device = DevicePlacement(
        kind=DeviceKind.CUDA if args.backend == "jax" else DeviceKind.CPU, index=0
    )
    configure_optiland_execution(
        device=device,
        precision=Precision.FP32 if settings["precision"] == "fp32" else Precision.FP64,
        enable_grad=False,
    )
    lens = build_optiland_system(demo3_system())
    sensor_px = preset["sensor_px"]

    def realize(
        *, patches: int, secondary: int, seed: int, density, draw
    ) -> tuple[np.ndarray, float, dict[str, Any]]:
        """One realization, and the positions it was drawn from."""
        started = time.perf_counter()
        if density is PositionDensity.UNIFORM and draw is PositionDraw.IID:
            # The shipped path, untouched: `run_route` draws the centres itself
            # inside `plan_patches` and no weight is applied anywhere. Routing
            # the control arm through the override would be a different rng
            # stream and would stop reproducing the committed records.
            override = None
            weights = None
            position_info: dict[str, Any] = {
                "density_kind": str(PositionDensity.UNIFORM),
                "draw_kind": str(PositionDraw.IID),
                "path": "plan_patches internal draw (shipped)",
            }
        else:
            plan = plan_positions(
                doe_field=doe.transmission,
                patch_px=patch_px,
                sample_pitch_m=doe.pitch_m,
                count=patches,
                density=density,
                draw=draw,
                rng=np.random.default_rng(seed),
            )
            override = plan.centers_xy_m
            weights = plan.center_weights
            position_info = plan.as_dict()
            position_info["path"] = "patch_positions.plan_positions"
        run = run_route(
            doe,
            lens,
            sensor_shape=(sensor_px, sensor_px),
            sensor_pitch_m=preset["sensor_pitch_m"],
            patch_px=patch_px,
            pad_factor=settings["pad_factor"],
            pad_width=settings["pad_width"],
            patch_count=patches,
            secondary_count=secondary,
            batches=max(1, patches // 50),
            seed=seed,
            backend=args.backend,
            precision=settings["precision"],
            secondary_chunks=max(
                1, int(np.ceil(50 * secondary / args.rays_per_chunk))
            ),
            route=settings["route"],
            reconstruction_route=args.reconstruction,
            kspace_oversample=args.kspace_oversample,
            emitters_override=override,
            center_weights_override=weights,
        )
        return run["field"], time.perf_counter() - started, position_info

    def arm(
        *, patches: int, secondary: int, density, draw, label: str
    ) -> dict[str, Any]:
        fields: list[np.ndarray] = []
        timings: list[float] = []
        info: dict[str, Any] = {}
        for replicate in range(args.replicates):
            field, seconds, info = realize(
                patches=patches,
                secondary=secondary,
                seed=args.seed0 + replicate,
                density=density,
                draw=draw,
            )
            fields.append(np.asarray(field))
            timings.append(seconds)
        statistics = variance_statistics(fields)
        # The MEDIAN, not the mean. JAX compiles on the first call into a fresh
        # shape, so the first realization of the first arm carries a few seconds
        # of compilation that has nothing to do with the estimator -- and the
        # first version of this probe read that as the control arm being 15%
        # slower than every candidate, which would have made a cost ratio out of
        # a warm-up.
        realized_patches = int(info.get("patch_count", patches))
        record = {
            "label": label,
            "patches": patches,
            "realized_patches": realized_patches,
            "secondary_per_patch": secondary,
            "total_rays_requested": patches * secondary,
            "total_rays_realized": realized_patches * secondary,
            "wall_clock_s_per_realization": float(np.median(timings)),
            "wall_clock_s_first_realization": timings[0],
            "wall_clock_s_all": timings,
            # V scales as 1 / N for an unbiased estimator, so this is the
            # ray-count-invariant figure and the one a stratified scheme that
            # skipped an empty stratum has to be compared on.
            "variance_times_rays": statistics["field_variance_sum"]
            * realized_patches
            * secondary,
            "positions": info,
            **statistics,
        }
        print(
            f"  {label:<24} P={patches:<6} S={secondary:<6} "
            f"V {statistics['field_variance_sum']:.6e} "
            f"(+-{statistics['jackknife_relative_error']:.1%})  "
            f"NCC {statistics['seed_to_seed_ncc_mean']:.5f}  "
            f"{np.median(timings):6.1f} s/run"
        )
        return record, fields

    noise_floor = float(3.0 / np.sqrt(sensor_px * sensor_px))
    payload: dict[str, Any] = {
        "probe": "demo3_variance",
        "issue": "CHE-120",
        "route": "rw_p",
        "stage": args.stage,
        "environment": environment(),
        "configuration": {
            "sensor_grid": [sensor_px, sensor_px],
            "sensor_pitch_m": preset["sensor_pitch_m"],
            "precision": settings["precision"],
            "backend": args.backend,
            "reconstruction": args.reconstruction,
            "kspace_oversample": (
                args.kspace_oversample if args.reconstruction != "ramp_sum" else None
            ),
            "replicates": args.replicates,
            "seeds": [args.seed0 + i for i in range(args.replicates)],
            "patch_px": patch_px,
        },
        "noise_floor": {
            "seed_to_seed_ncc": noise_floor,
            "note": (
                "3 / sqrt(N_px). Seed-to-seed NCC below this is zero with an "
                "error bar; the earlier demo3 ladder fitted through such points "
                "and returned a slope of 5.7. The relative_variance metric has "
                "no such floor, which is why the variance work is scored on it."
            ),
        },
        "status_of_this_evidence": (
            "CHARACTERIZATION, not validation. demo3 has no conventional "
            "reference; every number here is a property of the estimator, "
            "measured against itself across seeds or against another arm of the "
            "same probe. The unbiasedness of each arm is gated against an "
            "enumerated oracle in tests/test_patch_positions.py, at the size "
            "where such an oracle exists."
        ),
    }

    # One throwaway realization so nothing below pays for JAX's compilation.
    # Deliberately at the smallest shape the stage will use, and its field is
    # discarded rather than folded into an arm.
    realize(
        patches=64,
        secondary=256,
        seed=args.seed0 - 1,
        density=PositionDensity.UNIFORM,
        draw=PositionDraw.IID,
    )

    if args.stage == "decomposition":
        wanted = (
            [c for c in CANDIDATES if c[0] in set(args.arms.split(","))]
            if args.arms
            else [CANDIDATES[0]]
        )
        by_arm = []
        for label, density, draw in wanted:
            rungs = []
            for secondary in SECONDARY_LADDER:
                record, _ = arm(
                    patches=args.patches,
                    secondary=secondary,
                    density=density,
                    draw=draw,
                    label=f"{label} S={secondary}",
                )
                rungs.append(record)
            by_arm.append(
                {
                    "label": label,
                    "rungs": rungs,
                    "decomposition": _fit_decomposition(
                        rungs, patches=args.patches
                    ),
                }
            )
        payload["arms"] = by_arm
        if len(by_arm) > 1:
            base = by_arm[0]["decomposition"]
            payload["term_by_term_ratio_vs_first_arm"] = [
                {
                    "label": entry["label"],
                    "position_term_A_ratio": base["position_term_A"]
                    / entry["decomposition"]["position_term_A"],
                    "direction_term_B_ratio": base["direction_term_B"]
                    / entry["decomposition"]["direction_term_B"],
                    "reading": (
                        "A position density scales both terms. These two ratios "
                        "are the measurement of that, and they are what makes "
                        "the total reduction larger than the direction share "
                        "would allow if B were fixed."
                    ),
                }
                for entry in by_arm[1:]
            ]

    elif args.stage == "candidates":
        wanted = (
            [c for c in CANDIDATES if c[0] in set(args.arms.split(","))]
            if args.arms
            else list(CANDIDATES)
        )
        if not wanted or wanted[0][0] != CANDIDATES[0][0]:
            raise SystemExit(
                "the uniform_iid control arm must be present and first: every "
                "ratio and every bias test here is against it"
            )
        arms = []
        fields_by_label: dict[str, list[np.ndarray]] = {}
        for label, density, draw in wanted:
            record, fields = arm(
                patches=args.patches,
                secondary=args.secondary_count,
                density=density,
                draw=draw,
                label=label,
            )
            arms.append(record)
            fields_by_label[label] = fields
        control = arms[0]
        for record in arms:
            record["variance_ratio_vs_control"] = (
                control["variance_times_rays"] / record["variance_times_rays"]
            )
            # Two jackknife errors in quadrature, propagated through the ratio.
            # Reported because a 1.4x reduction measured to +-30% and one
            # measured to +-3% are different claims.
            record["variance_ratio_relative_error"] = float(
                np.hypot(
                    control["jackknife_relative_error"],
                    record["jackknife_relative_error"],
                )
            )
            # Read with the spread in `wall_clock_s_all`: the arms differ by up
            # to 9% in median wall clock and the within-arm spread is 3-7%, so
            # the difference is probably real and is NOT attributed here. Treat
            # the fixed-cost column as the fixed-ray column plus a few percent
            # of unexplained timing, not as a second independent result.
            record["seconds_per_realization_ratio_vs_control"] = (
                record["wall_clock_s_per_realization"]
                / control["wall_clock_s_per_realization"]
            )
            record["variance_ratio_at_fixed_cost"] = (
                record["variance_ratio_vs_control"]
                / record["seconds_per_realization_ratio_vs_control"]
            )
            record["anti_bias"] = (
                {
                    "control": True,
                    "note": (
                        "the control arm is what every other arm is tested "
                        "against; comparing it to itself would report a "
                        "separation of exactly zero and mean nothing"
                    ),
                }
                if record is control
                else bias_ratio(
                    fields_by_label[control["label"]], fields_by_label[record["label"]]
                )
            )
        payload["arms"] = arms

    elif args.stage == "ladder":
        wanted = (
            [c for c in CANDIDATES if c[0] in set(args.arms.split(","))]
            if args.arms
            else [CANDIDATES[0]]
        )
        ladder = [int(p) for p in args.ladder_patches.split(",")]
        by_arm = []
        for label, density, draw in wanted:
            rungs = []
            for patches in ladder:
                record, _ = arm(
                    patches=patches,
                    secondary=args.secondary_count,
                    density=density,
                    draw=draw,
                    label=f"{label} P={patches}",
                )
                rungs.append(record)
            by_arm.append(
                {
                    "label": label,
                    "rungs": rungs,
                    "trend": _fit_ladder(
                        rungs, noise_floor=noise_floor, target=args.target_ncc
                    ),
                }
            )
        payload["ladders"] = by_arm

    else:
        cells = []
        for patches, secondary in ALLOCATION:
            record, _ = arm(
                patches=patches,
                secondary=secondary,
                density=PositionDensity.UNIFORM,
                draw=PositionDraw.IID,
                label=f"P={patches}xS={secondary}",
            )
            cells.append(record)
        payload["cells"] = cells
        payload["cost_model"] = _fit_cost_model(cells)

    payload["device_memory"] = device_memory_stats()
    payload["reference_power_note"] = (
        "Every `field_variance_sum` is absolute, in the units of "
        "`mean_field_power` on the same rows. Divide by a `mean_field_power` to "
        "read a variance as a fraction -- but the fits and the ratios use the "
        "absolute values, because a signal-power denominator is not measurable "
        "at these budgets. See `variance_statistics`."
    )
    path = write_record(
        args.output_name or f"demo3_variance_{args.stage}", payload
    )
    print(f"wrote {path}")
    return 0


def _fit_decomposition(rungs: list[dict[str, Any]], *, patches: int) -> dict[str, Any]:
    """``P V = A + B / S`` by least squares, with the residual reported.

    The residual is the model check. If the two-term decomposition were wrong --
    if the draws were correlated, or if the reconstruction contributed a term
    that scaled with neither -- the four points would not lie on a line in
    ``1/S`` and this fit would say so instead of averaging it away.
    """
    inverse_s = np.array([1.0 / r["secondary_per_patch"] for r in rungs])
    scaled = np.array([patches * r["field_variance_sum"] for r in rungs])
    slope, intercept = np.polyfit(inverse_s, scaled, 1)
    predicted = intercept + slope * inverse_s
    residual = float(
        np.sqrt(np.mean((scaled - predicted) ** 2)) / max(np.mean(scaled), 1e-300)
    )
    return {
        "model": "P * V = A + B / S, V the absolute field variance sum",
        "position_term_A": float(intercept),
        "direction_term_B": float(slope),
        "relative_rms_residual": residual,
        "direction_share_at_shipped_S": float(
            (slope / 20_000.0) / (intercept + slope / 20_000.0)
        ),
        "reading": (
            "A is the part of the variance that falls only with P; B is the part "
            "that falls with P*S. Their ratio, with the cost model's per-patch "
            "and per-ray constants, is what sets the optimal S -- that is what "
            "this split is FOR. It is not a split into reducible and "
            "irreducible: a position density scales BOTH terms, because the "
            "per-ray weight modulus ||U~_c||_1 is a per-patch quantity and the "
            "importance weight multiplies the whole patch. Read the share as "
            "'how V responds to P versus S', never as a cap on what reweighting "
            "can achieve -- run this stage per arm to get each arm's own A and B."
        ),
    }


def _ladderfit(args, *, sensor_px: int) -> int:
    """Pool the rungs of several ladder records and refit, changing no number.

    The 10-minute command limit is a real constraint on a ladder whose top rung
    is a two-minute run: three seeds of P = 16000 do not fit alongside the rungs
    below them. Rather than shortening the lever arm -- which is what produced
    the flattened slope this issue is correcting -- the rungs are measured in
    separate commands and pooled here. Nothing is recomputed: the rungs are
    carried across verbatim and only the fit is redone, and the source records
    are named in the output so the pooling is auditable.
    """
    from _demo_support import RECORDS

    if not args.merge:
        raise SystemExit("ladderfit needs --merge with at least one record name")
    noise_floor = float(3.0 / np.sqrt(sensor_px * sensor_px))
    sources = [name.strip() for name in args.merge.split(",") if name.strip()]
    pooled: dict[str, list[dict[str, Any]]] = {}
    configurations = []
    for name in sources:
        source = json.loads((RECORDS / f"{name}.json").read_text())
        configurations.append({name: source["configuration"]})
        for ladder in source["ladders"]:
            pooled.setdefault(ladder["label"], []).extend(ladder["rungs"])

    ladders = []
    for label, rungs in pooled.items():
        rungs = sorted(rungs, key=lambda r: r["total_rays_realized"])
        seen = {r["total_rays_realized"] for r in rungs}
        if len(seen) != len(rungs):
            raise SystemExit(
                f"arm {label!r} has two rungs at the same ray count; pooling "
                "would double-weight it in the fit"
            )
        ladders.append(
            {
                "label": label,
                "rungs": rungs,
                "trend": _fit_ladder(
                    rungs, noise_floor=noise_floor, target=args.target_ncc
                ),
            }
        )

    payload = {
        "probe": "demo3_variance",
        "issue": "CHE-120",
        "stage": "ladderfit",
        "environment": environment(),
        "pooled_from": sources,
        "source_configurations": configurations,
        "configuration_note": (
            "Every pooled record must share reconstruction, oversampling, "
            "precision and sensor grid. They are listed above rather than "
            "asserted equal, because a pooled ladder across two reconstructions "
            "would fit a slope through two different estimators."
        ),
        "ladders": ladders,
        "status_of_this_evidence": (
            "A refit of measurements made elsewhere. No physics runs here."
        ),
    }
    path = write_record(args.output_name or "demo3_variance_ladderfit", payload)
    print(f"wrote {path}")
    for ladder in ladders:
        trend = ladder["trend"]
        print(
            f"  {ladder['label']:<28} {len(ladder['rungs'])} rungs, "
            f"V exponent {trend['variance_scaling_exponent']:.4f}"
        )
        if "power_law" in trend:
            print(
                f"      power law slope {trend['power_law']['log_log_slope']:.3f} "
                f"-> {trend['power_law']['rays_required']:.3e} rays; "
                f"saturating p {trend['saturating_law']['exponent_p']:.3f} "
                f"-> {trend['saturating_law']['rays_required']:.3e} rays"
            )
    return 0


def _fit_ladder(
    rungs: list[dict[str, Any]], *, noise_floor: float, target: float
) -> dict[str, Any]:
    """Two extrapolations of the same rungs, and the variance check that beats both.

    The committed cost ceiling extrapolates the seed-to-seed NCC on a **power
    law anchored at slope 1**, on the argument that an unbiased estimator's
    signal fraction grows linearly in the ray count while it is small. That
    argument is one power out. The shared part of two realizations is the
    deterministic ``|mu|^2``; the part that differs is the noise, whose spatial
    variance goes like ``(E|n|^2)^2`` once the noise dominates. So

        NCC ~ Var_px(|mu|^2) / V^2  ~  N^2,

    and the saturating form of the same statement is ``NCC = 1 / (1 + c N^-p)``,
    which is linear in ``log N`` after a logit and is fitted here with ``p``
    free. Both are reported, along with the power law the earlier record used,
    because the spread between them IS the uncertainty on the answer and at a
    two-decade extrapolation it is the dominant term.

    ``variance_scaling_exponent`` is the honest version of the same question.
    ``V`` is linear in the estimator's own variance with no saturation and no
    spatial statistics in the way, so ``V ~ 1/N`` is the direct test of
    "noise-limited, more rays fix it" -- and it is the one to trust when it and
    the NCC fits disagree.
    """
    usable = [r for r in rungs if r["seed_to_seed_ncc_mean"] > noise_floor]
    rays = np.array([r["total_rays_realized"] for r in rungs], dtype=np.float64)
    variance = np.array([r["field_variance_sum"] for r in rungs])
    # One rung is a measurement, not a trend. A single-rung ladder is a normal
    # way to extend an existing one across commands (see `_ladderfit`), so this
    # returns nan rather than a poorly conditioned fit through one point.
    variance_slope = (
        np.polyfit(np.log(rays), np.log(variance), 1)[0]
        if rays.size >= 2
        else float("nan")
    )

    out: dict[str, Any] = {
        "rungs_above_noise_floor": len(usable),
        "noise_floor_ncc": noise_floor,
        "variance_scaling_exponent": float(variance_slope),
        "variance_scaling_reading": (
            "-1 is a noise-limited unbiased estimator: variance falls as 1/N and "
            "more rays fix it. A shallower exponent would mean a floor no budget "
            "removes. This is the primary convergence evidence; the NCC fits "
            "below are the same statement in a coordinate that saturates."
        ),
        "target_seed_to_seed_ncc": target,
    }
    if len(usable) < 3:
        out["refused"] = (
            f"only {len(usable)} of {len(rungs)} rungs are above the noise floor "
            f"{noise_floor:.5f}; a two-parameter fit through them would return "
            "an exponent whatever the data are"
        )
        return out

    good_rays = np.array([r["total_rays_realized"] for r in usable], dtype=np.float64)
    good_ncc = np.array([r["seed_to_seed_ncc_mean"] for r in usable])

    power_slope, power_intercept = np.polyfit(np.log(good_rays), np.log(good_ncc), 1)
    power_required = float(
        np.exp((np.log(target) - power_intercept) / power_slope)
    )
    # NCC = 1 / (1 + c N^-p)  =>  log((1 - NCC) / NCC) = log c - p log N
    logit = np.log((1.0 - good_ncc) / good_ncc)
    minus_p, log_c = np.polyfit(np.log(good_rays), logit, 1)
    saturating_p = float(-minus_p)
    saturating_required = float(
        np.exp((log_c - np.log((1.0 - target) / target)) / saturating_p)
    )
    residual = float(
        np.sqrt(np.mean((logit - (log_c + minus_p * np.log(good_rays))) ** 2))
    )
    seconds_per_ray = float(
        np.mean(
            [
                r["wall_clock_s_per_realization"] / r["total_rays_realized"]
                for r in usable
            ]
        )
    )
    # The two models are fitted on the same points, so the fit residual cannot
    # choose between them -- both have two parameters and three or four points.
    # Refitting on all but the top rung and predicting it is a real test, and it
    # is computed here rather than by hand in a report.
    out["out_of_sample_top_rung"] = _out_of_sample(good_rays, good_ncc)

    out.update(
        {
            "power_law": {
                "log_log_slope": float(power_slope),
                "rays_required": power_required,
                "note": (
                    "the coordinate the committed record used. Its slope was "
                    "anchored at 1 rather than fitted; this one is fitted."
                ),
            },
            "saturating_law": {
                "exponent_p": saturating_p,
                "rays_required": saturating_required,
                "logit_rms_residual": residual,
                "note": (
                    "NCC = 1 / (1 + c N^-p). Respects NCC <= 1, which the power "
                    "law does not, so it is the one to prefer for a target as "
                    "high as 0.9 -- and the gap to the power law is the model "
                    "uncertainty, not a rounding."
                ),
            },
            "cost_ceiling": {
                "measured_seconds_per_ray": seconds_per_ray,
                "hours_per_run_power_law": power_required * seconds_per_ray / 3600.0,
                "hours_per_run_saturating": (
                    saturating_required * seconds_per_ray / 3600.0
                ),
                "extrapolation_basis": (
                    "Two fits on the same rungs, both reported: a fitted power "
                    "law and a saturating law that cannot exceed NCC 1. The "
                    "spread between them is the uncertainty on this number, and "
                    "at this extrapolation it is larger than the measurement "
                    "error on any rung."
                ),
                "paper_table_s2_rays": {"rw_f": 5.28e9, "rw_p": 2.6e9},
            },
        }
    )
    return out


def _out_of_sample(rays: np.ndarray, ncc: np.ndarray) -> dict[str, Any]:
    """Fit both laws on all but the highest rung; predict it; report the error.

    With fewer than four rungs there is nothing left to fit on after holding one
    out, and this says so rather than fitting a two-parameter model to two
    points and reporting the exact interpolation as a validation.
    """
    if rays.size < 4:
        return {
            "refused": (
                f"{rays.size} usable rungs: holding one out leaves a "
                "two-parameter fit with too few points to be a test"
            )
        }
    train_rays, train_ncc = rays[:-1], ncc[:-1]
    held_rays, held_ncc = float(rays[-1]), float(ncc[-1])

    power_slope, power_intercept = np.polyfit(np.log(train_rays), np.log(train_ncc), 1)
    power_prediction = float(np.exp(power_intercept + power_slope * np.log(held_rays)))

    logit = np.log((1.0 - train_ncc) / train_ncc)
    minus_p, log_c = np.polyfit(np.log(train_rays), logit, 1)
    held_logit = log_c + minus_p * np.log(held_rays)
    saturating_prediction = float(1.0 / (1.0 + np.exp(held_logit)))

    return {
        "trained_on_rays": [float(r) for r in train_rays],
        "held_out_rays": held_rays,
        "measured_ncc": held_ncc,
        "power_law_prediction": power_prediction,
        "power_law_relative_error": abs(power_prediction - held_ncc) / held_ncc,
        "saturating_prediction": saturating_prediction,
        "saturating_relative_error": abs(saturating_prediction - held_ncc) / held_ncc,
        "reading": (
            "The model with the smaller relative error is the one the "
            "extrapolation to the target should use. The power law cannot be "
            "right all the way to NCC 0.9 in any case -- it exceeds 1 -- so a "
            "win for the saturating law here is a confirmation rather than a "
            "surprise, and a win for the power law would be the thing needing "
            "explanation."
        ),
    }


def _fit_cost_model(cells: list[dict[str, Any]]) -> dict[str, Any]:
    """``c(P, S) = P (a + b S)`` from the allocation cells, then ``S*``.

    Fitted on the same runs that measured the variance, so the optimum is not
    assembled out of two different machines' timings.
    """
    design = np.array(
        [[c["patches"], c["patches"] * c["secondary_per_patch"]] for c in cells],
        dtype=np.float64,
    )
    seconds = np.array([c["wall_clock_s_per_realization"] for c in cells])
    (per_patch, per_ray), *_ = np.linalg.lstsq(design, seconds, rcond=None)
    predicted = design @ np.array([per_patch, per_ray])
    return {
        "model": "seconds = P * (a + b * S)",
        "seconds_per_patch_a": float(per_patch),
        "seconds_per_ray_b": float(per_ray),
        "relative_rms_residual": float(
            np.sqrt(np.mean((seconds - predicted) ** 2)) / np.mean(seconds)
        ),
        "note": (
            "a is the padded transform, which is per patch whatever S is; b is "
            "the draw, the trace and the reconstruction, which are per ray. That "
            "fixed P x S is NOT fixed cost is the whole reason the allocation "
            "question is not answered by the ray count alone."
        ),
    }


def _l1map(args, *, doe, patch_px: int, settings) -> int:
    """``||U~_c||_1`` over every candidate position: the exact optimal density.

    CPU only, and the expensive one -- one padded transform per candidate
    position. Sharded inside a single process so the whole thing stays inside one
    command rather than becoming four records to reconcile.
    """
    from couplers.patch import resolve_pad_px

    rows, cols = candidate_index_grid(grid_shape=doe.grid_shape, patch_px=patch_px)
    total = rows.size * cols.size
    pad_px = resolve_pad_px(
        grid_n=max(doe.grid_shape),
        patch_px=patch_px,
        pad_factor=settings["pad_factor"],
        max_center_px=float(max(doe.grid_shape) // 2 + patch_px // 2),
    )
    print(f"{total} candidate positions, pad {pad_px}, {args.l1_shards} shards")

    edges = np.linspace(0, total, args.l1_shards + 1).astype(int)
    pieces = []
    started = time.perf_counter()
    for shard, (lo, hi) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        pieces.append(
            spectral_l1_map(
                doe.transmission,
                patch_px=patch_px,
                pad_px=pad_px,
                start=int(lo),
                stop=int(hi),
            )
        )
        print(
            f"  shard {shard + 1}/{args.l1_shards}: {hi - lo} positions, "
            f"{time.perf_counter() - started:.1f} s elapsed"
        )
    exact = np.concatenate(pieces)
    energy = window_energy_map(doe.transmission, patch_px=patch_px).ravel()
    count = window_sample_count_map(doe.transmission, patch_px=patch_px).ravel()
    proxy = np.sqrt(energy)
    support = exact > 0

    out = Path(__file__).resolve().parents[1] / "records" / "ray_wave"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "demo3_position_spectral_l1.npz",
        spectral_l1=exact,
        window_energy=energy,
        window_sample_count=count,
        rows=rows,
        cols=cols,
    )

    payload = {
        "probe": "demo3_variance",
        "issue": "CHE-120",
        "stage": "l1map",
        "environment": environment(),
        "configuration": {
            "patch_px": patch_px,
            "pad_px": pad_px,
            "doe_grid": list(doe.grid_shape),
            "candidate_positions": total,
            "shards": args.l1_shards,
        },
        "map": {
            "empty_positions": int((count == 0).sum()),
            "empty_fraction": float((count == 0).mean()),
            "mean_window_fill": float((energy / (patch_px * patch_px)).mean()),
            "support_agrees_with_window_count": bool(
                np.array_equal(support, count > 0)
            ),
            "proxy_correlation": float(
                np.corrcoef(exact[support], proxy[support])[0, 1]
            ),
            "wall_clock_s": float(time.perf_counter() - started),
        },
        "predicted_variance_ratio_vs_uniform": {
            "note": (
                "All three are scored against the SAME weight model -- the exact "
                "||U~_c||_1 -- so they answer 'how much of the available "
                "reduction does this density capture', not 'how much does each "
                "predict for itself'. The exact row is the ceiling on this axis."
            ),
            "uniform": predicted_variance_ratio(np.ones_like(exact), exact),
            "sqrt_window_energy": predicted_variance_ratio(proxy, exact),
            "window_energy": predicted_variance_ratio(energy, exact),
            "spectral_l1_exact": predicted_variance_ratio(exact, exact),
        },
        "status_of_this_evidence": (
            "A prediction, computed from the DOE alone. The measured ratio is "
            "the candidates stage; these two are reported together precisely so "
            "the model can be wrong visibly."
        ),
    }
    path = write_record("demo3_position_spectral_l1", payload)
    print(f"wrote {path} and demo3_position_spectral_l1.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
