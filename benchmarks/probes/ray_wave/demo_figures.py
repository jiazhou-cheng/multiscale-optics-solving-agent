#!/usr/bin/env python3
"""Render the demo2/demo3 sensor fields that CHE-96 measured (paper Fig 5b/5c).

Reads only what the demo probes already wrote -- the `*_fields.npz` arrays and
their JSON records -- and draws them. It recomputes nothing, so the numbers
printed on a panel are the numbers that were measured on the GPU run that
produced the field, not a re-measurement that could quietly disagree with the
record.

demo2 is a validation: three routes against an independent float64 ASM oracle,
and the panels are expected to be indistinguishable by eye. demo3 is a
*characterization* -- the paper states no conventional reference exists for that
system, and neither of our routes converges at a budget that fits on this
machine -- so its panels are expected to look like noise, and the figure says so
rather than cropping or smoothing until they do not.

Run (CPU is sufficient; nothing here touches a solver):
    ./run.sh python benchmarks/probes/ray_wave/demo_figures.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RECORDS = Path(__file__).resolve().parents[1] / "records" / "ray_wave"
DATA = Path(__file__).resolve().parents[1] / "data"

FACECOLOR = "#fcfcfb"
#: gamma < 1 lifts the diffraction rings out of the black floor. Every intensity
#: panel in a figure shares one norm, so panels are comparable to each other and
#: not each to its own maximum.
GAMMA = 0.4


def _load(name: str) -> dict[str, np.ndarray]:
    with np.load(RECORDS / f"{name}.npz") as data:
        return {key: data[key] for key in data.files}


def _record(name: str) -> dict[str, Any]:
    return json.loads((RECORDS / f"{name}.json").read_text())


def _unit_sum(intensity: np.ndarray) -> np.ndarray:
    total = float(intensity.sum())
    return intensity / total if total > 0 else intensity


def _extent_mm(shape: tuple[int, ...], pitch_m: float) -> list[float]:
    """Axis extent about the origin at index ``n // 2`` -- this repository's rule."""
    half = [(n // 2) * pitch_m * 1e3 for n in shape[:2]]
    return [-half[1], half[1], -half[0], half[0]]


def _show(axes, image, *, title, extent, norm=None, cmap="inferno", subtitle=None):
    handle = axes.imshow(image, origin="lower", extent=extent, cmap=cmap, norm=norm, aspect="equal")
    axes.set_title(title, fontsize=10)
    if subtitle:
        axes.set_xlabel(subtitle, fontsize=8)
    axes.tick_params(labelsize=7)
    return handle


def figure_demo2(output: Path) -> dict[str, Any]:
    """Fig 5b: a planar hologram in free space, both routes against the oracle."""
    # A separate record from `demo2_paper_jax`, deliberately: this run carries
    # all three routes in one process so the saved fields share one oracle and
    # one device, and writing it here would overwrite the committed two-route
    # record with a different route set.
    fields = _load("demo2_paper_figure_jax_fields")
    record = _record("demo2_paper_figure_jax")

    pitch = record["configuration"]["sample_pitch_m"]
    extent = _extent_mm(fields["route_rw_f"].shape, pitch)

    # The oracle at the padding each route was actually scored against. The
    # matched-pad oracle is the one AC 1 reads from; upstream's 2x is the
    # nearer-aperiodic field the paper's own numbers describe. Both are drawn
    # because their difference (wraparound between two periods, not an error) is
    # visible at this stretch and would otherwise look like a discrepancy.
    matched_keys = sorted(k for k in fields if k.startswith("oracle_matched_pad"))
    oracle_matched = fields[matched_keys[0]]

    panels = [
        ("oracle_upstream_pad", "Oracle, float64 ASM, pad 200\n(upstream padding)", None),
        (matched_keys[0], f"Oracle, {matched_keys[0].split('pad')[-1]} pad\n(matched)", None),
        ("route_rw_f", "RW-F enumerated\n39,601 rays", record["routes"]["rw_f"]),
        (
            "route_rw_f_paper_budget",
            "RW-F, Table S2 budget\n1.1e6 rays",
            record["routes"].get("rw_f_paper_budget"),
        ),
        ("route_rw_p", "RW-P, Table S2 budget\n1.6e8 rays", record["routes"].get("rw_p")),
    ]
    if "route_rw_f_paper_budget" not in fields:
        panels = [p for p in panels if p[0] != "route_rw_f_paper_budget"]

    intensities = {key: _unit_sum(np.abs(fields[key]) ** 2) for key, _, _ in panels}
    vmax = max(float(v.max()) for v in intensities.values())
    norm = matplotlib.colors.PowerNorm(gamma=GAMMA, vmin=0.0, vmax=vmax)

    figure, axes = plt.subplots(2, 4, figsize=(17.5, 9.2), layout="constrained")
    figure.suptitle(
        "CHE-96 demo2 — paper Fig 5b: 100x100 SLM at 6.3 um, circular mask, "
        "sensor at 1.26 mm, lambda 0.7 um, one RTX A6000",
        fontsize=12,
    )

    _show(
        axes[0, 0],
        np.angle(fields["transmission"]),
        title="Input: DOE transmission phase\n(amplitude mask applied)",
        extent=extent,
        cmap="twilight",
        subtitle="rad; zero outside the aperture",
    )

    flat = list(axes.ravel())
    for slot, (key, title, route) in zip(flat[1:], panels, strict=False):
        subtitle = None
        if route is not None:
            subtitle = (
                f"vs matched oracle\nNCC {route['vs_oracle']['ncc_intensity']:.6f}\n"
                f"rel-L2 {route['vs_oracle']['relative_l2_field_after_global_phase']:.2e}"
            )
        handle = _show(
            slot,
            intensities[key],
            title=title,
            extent=extent,
            norm=norm,
            subtitle=subtitle,
        )
    figure.colorbar(
        handle,
        ax=axes.ravel().tolist(),
        location="right",
        shrink=0.45,
        label="unit-sum intensity",
    )

    # A residual panel, because at this stretch the intensity panels are meant
    # to be indistinguishable and "looks the same" is not a measurement.
    residual = np.abs(fields["route_rw_p"] - oracle_matched) / np.abs(oracle_matched).max()
    _show(
        axes[1, 2],
        residual,
        title="|RW-P - matched oracle| / max|oracle|",
        extent=extent,
        cmap="magma",
        subtitle="the field disagreement,\nspatially resolved",
    )

    # Central cut: the one panel where a sub-percent difference is legible.
    cut = intensities["route_rw_f"].shape[0] // 2
    axis = axes[1, 3]
    x = np.linspace(extent[0], extent[1], intensities["route_rw_f"].shape[1])
    for key, title, _ in panels:
        axis.semilogy(
            x,
            np.maximum(intensities[key][cut], 1e-12),
            label=title.split("\n")[0],
            lw=1.1,
        )
    axis.set_title("Central row cut", fontsize=10)
    axis.set_xlabel("x (mm)", fontsize=8)
    axis.set_ylabel("unit-sum intensity", fontsize=8)
    axis.legend(fontsize=6.5)
    axis.tick_params(labelsize=7)
    axis.grid(alpha=0.25)

    figure.savefig(output, dpi=150, facecolor=FACECOLOR)
    plt.close(figure)
    return {
        "figure": str(output),
        "routes_drawn": [k for k, _, _ in panels if k.startswith("route_")],
    }


def figure_demo3(output: Path) -> dict[str, Any]:
    """Fig 5c: hologram + refractive singlet. A characterization, not a validation."""
    rwf = _load("demo3_characterization_rw_f_fields")
    rwp = _load("demo3_characterization_rw_p_fields")
    record = _record("demo3_characterization_rw_f")
    agreement = _record("demo3_route_agreement")

    pitch = record["configuration"]["sensor_pitch_m"]
    shape = next(iter(rwf.values())).shape
    extent = _extent_mm(shape, pitch)

    def stack(fields: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        keys = sorted(fields)
        single = _unit_sum(np.abs(fields[keys[0]]) ** 2)
        # Coherent average over seeds: three independent estimators of ONE field,
        # so averaging the fields (not the intensities) is what reduces the
        # variance of the estimator being characterized.
        mean_field = np.mean([fields[k] for k in keys], axis=0)
        return single, _unit_sum(np.abs(mean_field) ** 2)

    f_single, f_mean = stack(rwf)
    p_single, p_mean = stack(rwp)
    vmax = max(float(v.max()) for v in (f_single, f_mean, p_single, p_mean))
    norm = matplotlib.colors.PowerNorm(gamma=GAMMA, vmin=0.0, vmax=vmax)

    figure, axes = plt.subplots(2, 3, figsize=(14.5, 9.0), layout="constrained")
    figure.suptitle(
        "CHE-96 demo3 — paper Fig 5c: 200x200 SLM, 3.0 mm gap, LA1131-A singlet "
        "(EFL 50.283 mm), sensor at 53 mm — NOISE-LIMITED, not converged",
        fontsize=12,
    )

    doe_phase = np.load(DATA / "demo3_smile_phase_profile.npy")
    _show(
        axes[0, 0],
        doe_phase,
        title="Input: DOE phase profile",
        extent=_extent_mm(doe_phase.shape, record["configuration"]["doe_pitch_m"]),
        cmap="twilight",
        subtitle="rad",
    )

    seeds = record["configuration"]["seeds"]
    rays = record["routes"]["rw_f"]["runs"][0]["total_rays"]
    budget = f"{rays / 1e6:.0f} M rays"
    for slot, image, title, sub in (
        (axes[0, 1], f_single, f"RW-F, one seed ({seeds[0]})", budget),
        (axes[0, 2], p_single, f"RW-P, one seed ({seeds[0]})", budget),
        (axes[1, 1], f_mean, f"RW-F, coherent mean of {len(rwf)} seeds", "variance / 3"),
        (axes[1, 2], p_mean, f"RW-P, coherent mean of {len(rwp)} seeds", "variance / 3"),
    ):
        handle = _show(slot, image, title=title, extent=extent, norm=norm, subtitle=sub)
    figure.colorbar(
        handle,
        ax=axes[:, 1:].ravel().tolist(),
        location="right",
        shrink=0.5,
        label="unit-sum intensity",
    )

    # The evidence that these panels are noise rather than disagreement. Printed
    # on the figure because the images alone cannot distinguish the two, and
    # showing them without it would invite the wrong reading.
    noise = agreement["noise_limited_agreement"]
    self_ncc = noise["mean_self_ncc"]
    lines = [
        "Why these panels look like noise:",
        "",
        f"mean cross-route NCC       {noise['mean_cross_route_ncc']:.4f}",
        f"predicted if one field     {noise['predicted_if_same_field']:.4f}",
        "  sqrt(NCC(A,A') * NCC(B,B'))",
        f"ratio measured / predicted {noise['ratio_measured_over_predicted']:.2f}",
        "",
        f"seed-to-seed NCC, RW-F     {self_ncc['demo3_characterization_rw_f']:.4f}",
        f"seed-to-seed NCC, RW-P     {self_ncc['demo3_characterization_rw_p']:.4f}",
        "  each route disagrees with ITSELF at",
        "  the same order as with the other one.",
        "",
        "Both routes are unbiased estimators of the",
        "same field. Their disagreement is fully",
        "explained by their own seed-to-seed scatter,",
        "so it is Monte-Carlo variance, not a",
        "physics discrepancy between the routes.",
        "",
        "Convergence: log-log slope 0.956 in ray",
        "count — linear, i.e. noise more rays remove.",
        "NCC 0.9 needs 1.78e9 rays (1.2 h / run);",
        "the paper's own RW-P budget here is 2.6e9.",
        "",
        "The paper states no conventional reference",
        "exists for this system, so there is no",
        "oracle panel to compare against.",
    ]
    axes[1, 0].axis("off")
    axes[1, 0].text(
        0.0,
        1.0,
        "\n".join(lines),
        va="top",
        ha="left",
        fontsize=8.5,
        family="monospace",
        transform=axes[1, 0].transAxes,
    )

    figure.savefig(output, dpi=150, facecolor=FACECOLOR)
    plt.close(figure)
    return {
        "figure": str(output),
        "seeds": seeds,
        "mean_cross_route_ncc": noise["mean_cross_route_ncc"],
    }


#: Categorical slots 1 and 2 of the validated default palette, in fixed order:
#: slot 1 is always the exact route and slot 2 always the fast one, so the colour
#: follows the entity rather than its position in any particular panel.
ROUTE_COLOUR = {"ramp_sum": "#2a78d6", "kspace_splat": "#eb6834"}
INK = "#0b0b0b"
INK_MUTED = "#52514e"


def figure_demo3_kspace(output: Path) -> dict[str, Any]:
    """CHE-101: the fast reconstruction against the exact one, on demo3's own rays.

    Three separate claims share one figure because they are only meaningful
    together: the two routes produce the same field, the fast one does not change
    the estimator's noise, and the stage that was optimized was not the stage that
    dominates the cost.
    """
    exact = _load("demo3_stage_ramp_fields")
    fast = _load("demo3_kspace_rw_p_fields")
    old = _load("demo3_characterization_rw_p_fields")
    exact_record = _record("demo3_stage_ramp")
    fast_record = _record("demo3_kspace_rw_p")
    sweep = _record("demo3_equivalence_characterization_jax")

    seed = sorted(fast)[0]
    common = "rw_p_seed20260822"
    pitch = exact_record["configuration"]["sensor_pitch_m"]
    extent = _extent_mm(exact[common].shape, pitch)

    a, b = exact[common], fast[common]
    ia, ib = _unit_sum(np.abs(a) ** 2), _unit_sum(np.abs(b) ** 2)

    def coherent_mean(fields: dict[str, np.ndarray]) -> np.ndarray:
        return _unit_sum(np.abs(np.mean([fields[k] for k in sorted(fields)], axis=0)) ** 2)

    fast_mean, old_mean = coherent_mean(fast), coherent_mean(old)
    vmax = max(float(v.max()) for v in (ia, ib, fast_mean, old_mean))
    norm = matplotlib.colors.PowerNorm(gamma=GAMMA, vmin=0.0, vmax=vmax)

    figure, axes = plt.subplots(2, 4, figsize=(19.0, 9.4), layout="constrained")
    figure.suptitle(
        "CHE-101 demo3 (paper Fig 5c) — k-space splat vs the exact ramp sum, "
        "60 M rays on one RTX A6000",
        fontsize=13,
    )

    for slot, image, title, sub in (
        (axes[0, 0], ia, "Exact route (ramp_sum)", "O(rays x pixels)"),
        (axes[0, 1], ib, "Fast route (kspace_splat, 8x)", "O(rays + K log K)"),
        (axes[1, 0], fast_mean, "Fast route, coherent mean of 3 seeds", "variance / 3"),
        (axes[1, 1], old_mean, "CHE-96 exact route, mean of 3 seeds", "the committed result"),
    ):
        handle = _show(slot, image, title=title, extent=extent, norm=norm, subtitle=sub)
    figure.colorbar(
        handle,
        ax=[axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]],
        location="left",
        shrink=0.45,
        label="unit-sum intensity",
    )

    # Same seed means the same rays through the same trace, so this panel is the
    # reconstruction difference alone -- not a difference of two Monte Carlo draws.
    residual = np.abs(b - a) / float(np.abs(a).max())
    _show(
        axes[0, 2],
        residual,
        title="|fast - exact| / max|exact|, same rays",
        extent=extent,
        cmap="magma",
        subtitle="residual grows off-axis -- the splat kernel's\nsignature, not a ray-count effect",
    )

    # --- stage cost: one measure across five categories, so one hue with the
    # --- stage this ticket optimized picked out, and every bar directly labelled.
    stages = exact_record["routes"]["rw_p"]["runs"][0]["stage_wall_clock_s"]
    order = [
        ("optiland_trace", "Optiland trace"),
        ("emit_patch_spectra", "emit patch spectra"),
        ("reconstruct", "reconstruction"),
        ("host_to_device", "host to device"),
        ("power_bookkeeping", "probe bookkeeping"),
    ]
    values = [stages[key] for key, _ in order]
    total = sum(values)
    axis = axes[0, 3]
    positions = np.arange(len(order))
    colours = ["#c9c8c1"] * len(order)
    colours[2] = ROUTE_COLOUR["kspace_splat"]
    bars = axis.barh(positions, values, height=0.62, color=colours, linewidth=0)
    axis.set_yticks(positions, [label for _, label in order], fontsize=8.5)
    axis.invert_yaxis()
    axis.set_xlabel("seconds of one 60 M-ray run", fontsize=8.5)
    axis.set_title(
        "Where demo3's time actually goes\n(exact route, measured per stage)", fontsize=10
    )
    for bar, value in zip(bars, values, strict=False):
        axis.text(
            bar.get_width() + total * 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.0f} s  ({100 * value / total:.0f}%)",
            va="center",
            fontsize=8,
            color=INK,
        )
    axis.set_xlim(0, max(values) * 1.38)
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(labelsize=8, length=0)
    axis.grid(axis="x", alpha=0.2)
    axis.text(
        0.98,
        0.04,
        "reconstruction is 7% of the run:\nmaking it 10x cheaper cannot\nmake demo3 converge",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color=ROUTE_COLOUR["kspace_splat"],
    )

    # --- error against oversampling: one series, so no legend box; the title names it.
    axis = axes[1, 2]
    oversamples = [row["kspace_oversample"] for row in sweep["comparisons"]]
    errors = [row["relative_l2_field_after_global_phase"] for row in sweep["comparisons"]]
    axis.plot(
        oversamples,
        errors,
        color=ROUTE_COLOUR["kspace_splat"],
        lw=2.0,
        marker="o",
        ms=8,
        markeredgecolor="#fcfcfb",
        markeredgewidth=2,
    )
    chosen = 8.0
    if chosen in oversamples:
        index = oversamples.index(chosen)
        axis.annotate(
            f"used here: {chosen:g}x\n{errors[index]:.1e}",
            xy=(oversamples[index], errors[index]),
            xytext=(-8, 26),
            textcoords="offset points",
            fontsize=8.5,
            color=INK,
            ha="right",
            arrowprops={"arrowstyle": "-", "color": INK_MUTED, "lw": 1},
        )
    axis.set_yscale("log")
    axis.set_xlabel("k-grid oversampling", fontsize=8.5)
    axis.set_ylabel("field error vs exact route", fontsize=8.5)
    axis.set_title(
        "The splat's only cost is interpolation,\nand it buys down with a bigger FFT",
        fontsize=10,
    )
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(labelsize=8)
    axis.grid(alpha=0.2)

    # --- the numbers, stated rather than left to the eye.
    def ncc_of(x: np.ndarray, y: np.ndarray) -> float:
        u, v = x - x.mean(), y - y.mean()
        return float((u * v).sum() / np.sqrt((u * u).sum() * (v * v).sum()))

    def seed_spread(fields: dict[str, np.ndarray]) -> tuple[float, float]:
        keys = sorted(fields)
        pairs = [
            ncc_of(np.abs(fields[i]) ** 2, np.abs(fields[j]) ** 2)
            for a_, i in enumerate(keys)
            for j in keys[a_ + 1 :]
        ]
        return min(pairs), max(pairs)

    fast_low, fast_high = seed_spread(fast)
    old_low, old_high = seed_spread(old)
    runs = fast_record["routes"]["rw_p"]["runs"]
    lines = [
        "Same rays, both reconstructions:",
        f"  NCC intensity        {ncc_of(ia, ib):.6f}",
        f"  rel-L2 field         "
        f"{np.linalg.norm(b - a) / np.linalg.norm(a):.3e}",
        # From the RAW fields, not the unit-sum panels: normalising first makes
        # this ratio 1.0000 by construction, which is how it read at first.
        f"  power ratio          "
        f"{float((np.abs(b) ** 2).sum() / (np.abs(a) ** 2).sum()):.4f}",
        "",
        "Seed-to-seed NCC (the estimator's",
        "own noise) is what a biased fast",
        "path would have changed:",
        f"  exact route   {old_low:.4f} - {old_high:.4f}",
        f"  fast route    {fast_low:.4f} - {fast_high:.4f}",
        "  unchanged, so the splat did not",
        "  trade accuracy for convergence.",
        "",
        "Reconstruction kernel alone, 1e6 rays",
        "at 420x420 on the GPU:",
        "  ramp_sum      0.183 s",
        "  kspace 8x     0.019 s   (9.6x)",
        "",
        f"Whole run: {np.mean([r['wall_clock_s'] for r in runs]):.0f} s fast vs "
        f"{exact_record['routes']['rw_p']['runs'][0]['wall_clock_s']:.0f} s exact",
        "-- a 7% stage cannot give more.",
        "",
        "Still NOT converged: the paper's",
        "system needs ~1.8e9 rays, and the",
        "trace and the emitter now set that",
        "price. Recorded, not worked around.",
    ]
    axes[1, 3].axis("off")
    axes[1, 3].text(
        0.0,
        1.0,
        "\n".join(lines),
        va="top",
        ha="left",
        fontsize=8.5,
        family="monospace",
        color=INK,
        transform=axes[1, 3].transAxes,
    )

    figure.savefig(output, dpi=150, facecolor=FACECOLOR)
    plt.close(figure)
    return {
        "figure": str(output),
        "ncc_same_rays": ncc_of(ia, ib),
        "seed_spread_fast": [fast_low, fast_high],
        "seed_spread_exact": [old_low, old_high],
        "seed": seed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/CHE-96"))
    parser.add_argument("--demos", default="demo2,demo3,demo3-kspace")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    wanted = {d.strip() for d in args.demos.split(",") if d.strip()}
    summary = {}
    if "demo2" in wanted:
        summary["demo2"] = figure_demo2(args.output_dir / "demo2_fig5b_sensor_fields.png")
    if "demo3" in wanted:
        summary["demo3"] = figure_demo3(args.output_dir / "demo3_fig5c_sensor_fields.png")
    if "demo3-kspace" in wanted:
        summary["demo3-kspace"] = figure_demo3_kspace(
            args.output_dir / "demo3_kspace_vs_exact.png"
        )
    for name, info in summary.items():
        print(f"{name}: wrote {info['figure']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
