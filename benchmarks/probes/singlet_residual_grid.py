"""Is the singlet residual a physics finding or a sampling one?

CHE-117 (M4.2). ``B3-PSF-SINGLET``'s gate is unmet: the production (weighted)
configuration measures ``2.21e-3`` against O1, the analytic Airy oracle, on a
``1.0e-3`` gate. The uniform configuration measures ``9.21e-4`` and clears it.
Two things about that are unexplained and this probe is aimed at both:

1. the ORDERING. The production weight is independently required -- it is what
   fixes CHE-33's ``N^2.0024`` absolute-power divergence -- and it makes the
   agreement with O1 *worse*. That is the negative control reading ``0.42``
   against a ``1.2`` floor: it fires backwards.
2. whether either number is converged. CHE-103 found the *pupil-to-focus* grid
   puts 2.44 pixels across the Airy radius and is not converged for radius-like
   quantities. The sensor grid this residual is measured on is finer -- 0.5 um
   pitch against a 6.486 um Airy radius, so about 13 px -- but nobody has swept
   it, and "finer than the one that was wrong" is not an argument.

The experiment
--------------
Hold the physical window fixed at 128 um and refine the sensor pitch, at two ray
counts. If the weighted/uniform ordering against O1 is a sampling artefact it
should move; if it is physics it should not. Either answer attributes something.

**This probe decides nothing against O2.** Our own float64 ASM/RS propagator is
reported beside O1 as characterization -- it is exactly the oracle L2-PSF-01 had
to retire from the gate as circular -- and the ordering question is asked of O1
alone.

    ./run.sh python benchmarks/probes/singlet_residual_grid.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from core.paths import repository_root  # noqa: E402
from core.provenance import RECORD_PROVENANCE_KEY, record_provenance  # noqa: E402

ROOT = repository_root()
RECORD = ROOT / "benchmarks/probes/records/singlet_residual_grid.json"

#: The physical window every configuration holds fixed. Refining the pitch while
#: holding the window means the gate disc always covers the same physical area,
#: so the comparison is about sampling and not about how much of the field is in
#: the frame.
WINDOW_M = 128.0e-6

#: Sensor pitches, in metres. 0.5e-6 is the frozen configuration; the sweep
#: brackets it on both sides so a monotone trend is distinguishable from a
#: coincidence at one point.
PITCH_SWEEP_M = (1.0e-6, 0.5e-6, 0.25e-6, 0.125e-6)

#: Two ray counts. 256 rings is 197,377 traced rays and is cheap enough to sweep;
#: 512 rings is 787,969 and is the count every committed number was measured at.
RING_SWEEP = (256, 512)


def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _reconstruct(
    sensor: Any,
    quadrature: Any,
    rings: int,
    workdir: Path,
    *,
    weighted: bool,
    grid_n: int,
    pitch_m: float,
) -> dict[str, Any]:
    """One reconstruction at a declared grid. Reuses the frozen probe's own path."""
    rays = sensor._trace(rings, workdir / f"trace_{rings}_{weighted}")
    bundle = quadrature._weighted_bundle(sensor, rays, apply_quadrature_weight=weighted)
    bundle, _ = sensor._advance_bundle_to_z(bundle, sensor.SENSOR_Z_M)
    field, _ = sensor._reconstruct_core(bundle, grid_n=grid_n, pitch_m=pitch_m)
    return {
        "traced_rays": bundle.count,
        "intensity": np.abs(field.u) ** 2,
        "discrete_power": field.discrete_power(),
    }


def _encircled_radius_m(
    intensity: np.ndarray, pitch_m: float, fraction: float
) -> float:
    """Radius containing ``fraction`` of the windowed energy.

    A width statistic that does not depend on locating a null, so it stays
    meaningful at the coarse end of the sweep where the first null falls between
    samples. That is the CHE-103 failure mode, avoided rather than repeated.
    """
    n_y, n_x = intensity.shape
    y = (np.arange(n_y) - n_y // 2) * pitch_m
    x = (np.arange(n_x) - n_x // 2) * pitch_m
    r = np.hypot(y[:, None], x[None, :]).ravel()
    order = np.argsort(r)
    cumulative = np.cumsum(intensity.ravel()[order])
    total = cumulative[-1]
    if total <= 0:
        return float("nan")
    index = int(np.searchsorted(cumulative, fraction * total))
    return float(r[order][min(index, r.size - 1)])


def characterize() -> dict[str, Any]:
    sensor = _load("sensor_probe", ROOT / "benchmarks/probes/sensor_handoff_convergence.py")
    quadrature = _load("quadrature_probe", ROOT / "benchmarks/probes/quadrature_weight.py")

    airy_radius_m = sensor._airy_radius_m()
    rows: list[dict[str, Any]] = []

    with tempfile.TemporaryDirectory() as directory:
        work = Path(directory)
        for pitch_m in PITCH_SWEEP_M:
            grid_n = int(round(WINDOW_M / pitch_m))
            o1 = sensor._o1_analytic_airy(grid_n=grid_n, pitch=pitch_m)
            gate_disc = sensor._disc_mask(
                (grid_n, grid_n), pitch_m, sensor.GATE_AIRY_RADII * airy_radius_m
            )
            o1_ee50 = _encircled_radius_m(o1, pitch_m, 0.5)

            for rings in RING_SWEEP:
                began = time.perf_counter()
                weighted = _reconstruct(
                    sensor, quadrature, rings, work,
                    weighted=True, grid_n=grid_n, pitch_m=pitch_m,
                )
                uniform = _reconstruct(
                    sensor, quadrature, rings, work,
                    weighted=False, grid_n=grid_n, pitch_m=pitch_m,
                )
                weighted_vs_o1 = sensor._relative_l2(weighted["intensity"], o1, gate_disc)
                uniform_vs_o1 = sensor._relative_l2(uniform["intensity"], o1, gate_disc)
                rows.append(
                    {
                        "sensor_pitch_m": pitch_m,
                        "grid_n": grid_n,
                        "pixels_per_airy_radius": airy_radius_m / pitch_m,
                        "rings": rings,
                        "traced_rays": weighted["traced_rays"],
                        "weighted_vs_o1": weighted_vs_o1,
                        "uniform_vs_o1": uniform_vs_o1,
                        # The negative control, as CHE-47 defines it: how much
                        # ADDING the production weight improves agreement with O1.
                        # Above 1.0 the weight helps; 0.42 is what the frozen
                        # configuration reads, and below 1.2 the control is
                        # recorded as not firing.
                        "improvement_factor_vs_o1": (
                            uniform_vs_o1 / weighted_vs_o1 if weighted_vs_o1 else float("nan")
                        ),
                        "weighted_ee50_m": _encircled_radius_m(
                            weighted["intensity"], pitch_m, 0.5
                        ),
                        "uniform_ee50_m": _encircled_radius_m(
                            uniform["intensity"], pitch_m, 0.5
                        ),
                        "o1_ee50_m": o1_ee50,
                        "weighted_discrete_power": weighted["discrete_power"],
                        "uniform_discrete_power": uniform["discrete_power"],
                        "wall_seconds": time.perf_counter() - began,
                    }
                )
                print(
                    f"pitch {pitch_m * 1e6:.3f} um  grid {grid_n:5d}  rings {rings:4d}  "
                    f"weighted {weighted_vs_o1:.4e}  uniform {uniform_vs_o1:.4e}  "
                    f"factor {rows[-1]['improvement_factor_vs_o1']:.3f}",
                    flush=True,
                )

    return {
        "probe": "singlet_residual_grid",
        "issue": "CHE-117",
        "question": (
            "does the weighted/uniform ordering against O1 survive grid refinement, "
            "or is the unmet singlet gate a sampling artefact?"
        ),
        "oracle": (
            "O1, the analytic Airy pattern. O2 is deliberately not consulted: it is "
            "our own ASM/RS propagator and L2-PSF-01 already had to retire it from "
            "this gate as circular."
        ),
        "window_m": WINDOW_M,
        "airy_radius_m": airy_radius_m,
        "gate_airy_radii": sensor.GATE_AIRY_RADII,
        "frozen_configuration": {
            "sensor_pitch_m": sensor.SENSOR_PITCH_M,
            "grid_n": sensor.SENSOR_GRID_N,
            "committed_weighted_vs_o1": 2.2072391812867093e-3,
            "committed_improvement_factor_vs_o1": 0.4173375512174577,
        },
        "rows": rows,
    }


def _verdict(record: dict[str, Any]) -> dict[str, Any]:
    """What the sweep says, stated so a reader cannot over-read it."""
    rows = record["rows"]
    by_rings: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_rings.setdefault(row["rings"], []).append(row)

    findings: dict[str, Any] = {}
    for rings, group in sorted(by_rings.items()):
        group = sorted(group, key=lambda r: -r["sensor_pitch_m"])
        factors = [r["improvement_factor_vs_o1"] for r in group]
        weighted = [r["weighted_vs_o1"] for r in group]
        uniform = [r["uniform_vs_o1"] for r in group]
        findings[f"rings_{rings}"] = {
            "pixels_per_airy_radius": [r["pixels_per_airy_radius"] for r in group],
            "weighted_vs_o1": weighted,
            "uniform_vs_o1": uniform,
            "improvement_factor_vs_o1": factors,
            "ordering_flips_under_refinement": bool(
                min(factors) < 1.0 < max(factors)
            ),
            "weighted_residual_converges": bool(
                abs(weighted[-1] - weighted[-2]) < 0.1 * abs(weighted[-1])
            )
            if len(weighted) >= 2
            else None,
            "control_fires_anywhere": bool(max(factors) >= 1.2),
        }

    findings["statement"] = (
        "The ordering against O1 is reported at four sensor samplings and two ray "
        "counts. A flip under refinement would attribute the unmet gate to sampling; "
        "a stable ordering would attribute it to the physics of the weighted "
        "reconstruction and leave the negative control genuinely backwards."
    )
    findings["not_claimed"] = (
        "that either configuration is CORRECT. O1 is an aberration-free paraxial "
        "Airy pattern and M3-SINGLET-REF is a real traced singlet, so a nonzero "
        "residual is expected from aberration alone and this probe does not "
        "separate that term. It answers the ordering question only."
    )
    return findings


def main() -> None:
    record = characterize()
    record["verdict"] = _verdict(record)
    record[RECORD_PROVENANCE_KEY] = record_provenance(
        probe="benchmarks/probes/singlet_residual_grid.py", root=ROOT
    )
    RECORD.parent.mkdir(parents=True, exist_ok=True)
    RECORD.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(record["verdict"], indent=1))
    print(f"wrote {RECORD.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
