#!/usr/bin/env python3
"""One summary figure for the M3 exit report (CHE-39/M3.10).

Not a gate, not a new measurement: every number plotted here is already
recorded in benchmarks/reports/2026-08/sensor_handoff_convergence.md,
benchmarks/probes/records/m3_quadrature_weight.json, and
outputs/M3/L2-PSF-01/result.json. This script only renders the milestone's
own residual/gate trajectory across its four successive configurations.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

GATE = 1.0e-3
GOOD = "#0ca30c"
CRITICAL = "#d03b3b"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#d8d6cf"

ROWS = [
    (
        "M3.8 (CHE-37)\nexit-pupil, hard-support\nreconstruction",
        0.015087255918918473,
        "out of contract\n(FFT/Fraunhofer oracle)",
    ),
    (
        "M3.9R (CHE-38)\nsensor plane, uniform\nray weight, 787,969 rays",
        0.0038397829891449045,
        "real traced system\n(vs O2/ASM)",
    ),
    (
        "CHE-47 extension\nsensor plane, weighted,\n787,969 rays (production)",
        0.0024822583978904795,
        "real traced system\n(vs O2/ASM)",
    ),
    (
        "CHE-47 diagnostic\nsensor plane, weighted,\nsynthetic aberration-free",
        4.0734e-4,
        "no traced aberration\n(vs Rayleigh-Sommerfeld)",
    ),
]

labels = [row[0] for row in ROWS]
ratios = [row[1] / GATE for row in ROWS]
notes = [row[2] for row in ROWS]
colors = [CRITICAL if r >= 1.0 else GOOD for r in ratios]

fig, ax = plt.subplots(figsize=(10.5, 5.6))
y = range(len(ROWS))
bars = ax.barh(y, ratios, color=colors, height=0.58, zorder=3)

ax.axvline(1.0, color=TEXT_PRIMARY, linewidth=1.5, linestyle="--", zorder=2)
ax.text(
    1.0, len(ROWS) - 0.35, " 1.0e-3 gate\n (fft_oracle_intensity_relative_l2)",
    color=TEXT_PRIMARY, fontsize=8.5, va="bottom", ha="left",
)

for yi, (bar, ratio) in enumerate(zip(bars, ratios, strict=True)):
    ax.text(
        bar.get_width() * 1.08, yi, f"{ratio:.2f}x gate", va="center",
        ha="left", fontsize=9, color=TEXT_PRIMARY, fontweight="bold",
    )

ax.set_yticks(list(y))
ax.set_yticklabels(
    [f"{label}\n{note}" for label, note in zip(labels, notes, strict=True)],
    fontsize=8.5, color=TEXT_PRIMARY,
)
ax.invert_yaxis()
ax.set_xscale("log")
ax.set_xlim(0.1, 30)
ax.set_xlabel("measured relative-L2 residual / frozen 1.0e-3 gate (log scale)", color=TEXT_PRIMARY)
ax.set_title(
    "M3: the sensor-plane handoff correction, in one number\n"
    "each bar is a real measurement already in the record -- no gate was widened",
    fontsize=11, color=TEXT_PRIMARY, pad=14,
)
ax.grid(True, axis="x", which="both", color=GRID, linewidth=0.8, zorder=1)
ax.set_axisbelow(True)
for spine in ("top", "right", "left"):
    ax.spines[spine].set_visible(False)
ax.spines["bottom"].set_color(GRID)
ax.tick_params(axis="y", length=0)

legend_handles = [
    plt.Rectangle((0, 0), 1, 1, color=GOOD, label="gate met"),
    plt.Rectangle((0, 0), 1, 1, color=CRITICAL, label="gate not met"),
]
ax.legend(handles=legend_handles, loc="lower right", frameon=False, fontsize=9)

fig.tight_layout()
out = Path(__file__).resolve().parents[2] / "outputs" / "M3" / "M3_EXIT_SUMMARY.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=150, facecolor="#fcfcfb")
print(f"wrote {out}")
