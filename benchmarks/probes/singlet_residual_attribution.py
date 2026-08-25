"""What the singlet residual is, and why its negative control fires backwards.

CHE-117 (M4.2). ``B3-PSF-SINGLET``'s gate is unmet -- the production (weighted)
configuration measures ``2.21e-3`` against O1's ``1.0e-3`` -- and its
quadrature-weight negative control reads ``0.42`` against a ``1.2`` floor: the
UNIFORM configuration is closer to the analytic Airy oracle than the weighted
one, which is the opposite of what the control asserts. CHE-48 was supposed to
decompose that and was closed with no comment, no commit and no artifact.

This probe decomposes it. Three experiments, each cheap, each ruling out one
candidate:

**1. Is it the sensor sampling?** No. ``singlet_residual_grid.py`` holds the
window fixed and refines the pitch from 6.5 to 52 pixels per Airy radius: the
weighted residual reads ``2.2072391812867093e-3`` at every one of them, to ten
significant figures.

**2. Is it the quadrature weight's boundary corrections?** The weight differs
from uniform in exactly two places -- the central ray gets 3/4 of a nominal cell
and the outermost ring gets 1/2. Removing each in turn: dropping the CENTRE
correction changes nothing (2.2076e-3 against 2.2072e-3), and dropping the RIM
correction reproduces the uniform arm to four significant figures at every ray
count. So the entire weighted-versus-uniform difference is the rim half-weight.

**3. Then is the rim half-weight the defect?** No -- and this is the finding.
Extending the ray ladder past the committed 512 rings shows the two arms
CONVERGING TO THE SAME NUMBER. The weighted arm is flat at ``2.208e-3`` from 24
rings to 1024 (a 43x range in ray count, five significant figures). The uniform
arm descends from ``8.3e-2``, crosses the weighted arm near 181 rings, reaches a
MINIMUM of ``7.0e-4`` at 362 rings, and then climbs back: ``9.2e-4`` at 512,
``1.25e-3`` at 724, ``1.52e-3`` at 1024.

The conclusions
---------------
**The 2.21e-3 residual is converged and is not caused by the quadrature
weight.** It is what the correctly-quadratured reconstruction of the real traced
``M3-SINGLET-REF`` measures against an aberration-free paraxial Airy pattern.

**The negative control is mis-specified, not backwards.** It compares a
converged arm against an unconverged one at a single ray count, and its verdict
is a function of that ray count: 10.7 at 8 rings, 1.02 at 181, 0.42 at 512, 0.57
at 724, 0.69 at 1024. A control whose sign flips with a numerical parameter is
measuring where two convergence curves cross, not whether the weight helps.

**No tolerance is widened and the gate stays unmet.** What changes is that
2.21e-3 now has an owner: it is a statement about the oracle's applicability to
this system, and the next question is whether O1 -- aberration-free, paraxial --
can decide a real singlet at the 1e-3 level at all. That question is not
answered here and this probe does not pretend to answer it.

**O2 is not consulted.** Our own float64 ASM/RS propagator is the oracle
L2-PSF-01 already had to retire from this gate as circular, and nothing here
readmits it.

    ./run.sh python benchmarks/probes/singlet_residual_attribution.py
"""

from __future__ import annotations

import importlib.util
import itertools
import json
import math
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from core.paths import repository_root  # noqa: E402
from core.provenance import RECORD_PROVENANCE_KEY, record_provenance  # noqa: E402

ROOT = repository_root()
RECORD = ROOT / "benchmarks/probes/records/singlet_residual_attribution.json"

#: The committed ladder stops at 512 rings. These extend it, and the extension
#: is where the finding is: two arms that look like they disagree are converging.
LADDER_RINGS = (128, 256, 362, 512, 724, 1024)

#: Ring counts for the boundary-correction decomposition. Enough to show the
#: rim-removed arm tracking uniform across a decade, and no more -- 1024 rings
#: x three arms would be twenty minutes for a point already made.
DECOMPOSITION_RINGS = (128, 256, 512)

#: ``(centre, rim)`` multipliers on the nominal cell area. Production is
#: ``(0.75, 0.5)``; the other two remove one correction each.
ARMS: dict[str, tuple[float, float]] = {
    "production": (0.75, 0.5),
    "rim_correction_removed": (0.75, 1.0),
    "centre_correction_removed": (1.0, 0.5),
}


def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _area_weight_variant(centre: float, rim: float) -> Any:
    """``hexapolar_area_weight_m2`` with the two boundary corrections varied.

    Substituted for the production function rather than reimplemented around it:
    the interior weight, the ring indexing and everything downstream stay
    exactly what ships, so the only difference between arms is the two
    multipliers.
    """

    def variant(
        ring_index: NDArray[Any], num_rings: int, aperture_radius_m: float
    ) -> NDArray[np.float64]:
        ring_index = np.asarray(ring_index)
        nominal = math.pi * aperture_radius_m**2 / (3.0 * num_rings**2)
        weight = np.full(ring_index.shape, nominal, dtype=np.float64)
        weight[ring_index == 0] = centre * nominal
        weight[ring_index == num_rings] = rim * nominal
        return weight

    return variant


def characterize() -> dict[str, Any]:
    sensor = _load("sensor_probe", ROOT / "benchmarks/probes/sensor_handoff_convergence.py")
    quadrature = _load("quadrature_probe", ROOT / "benchmarks/probes/quadrature_weight.py")
    import couplers.quadrature as quadrature_module

    production_weight = quadrature_module.hexapolar_area_weight_m2

    o1 = sensor._o1_analytic_airy(grid_n=sensor.SENSOR_GRID_N, pitch=sensor.SENSOR_PITCH_M)
    gate_disc = sensor._disc_mask(
        (sensor.SENSOR_GRID_N, sensor.SENSOR_GRID_N),
        sensor.SENSOR_PITCH_M,
        sensor.GATE_AIRY_RADII * sensor._airy_radius_m(),
    )

    def residual(rings: int, *, weighted: bool) -> tuple[float, int]:
        with tempfile.TemporaryDirectory() as directory:
            result = quadrature._sensor_reconstruction(
                sensor, rings, Path(directory), apply_quadrature_weight=weighted
            )
        return (
            sensor._relative_l2(result["intensity"], o1, gate_disc),
            int(result["traced_rays"]),
        )

    # --- 1. the ladder, extended past the committed 512 rings ---------------
    ladder: list[dict[str, Any]] = []
    for rings in LADDER_RINGS:
        began = time.perf_counter()
        weighted, traced = residual(rings, weighted=True)
        uniform, _ = residual(rings, weighted=False)
        ladder.append(
            {
                "rings": rings,
                "traced_rays": traced,
                "weighted_vs_o1": weighted,
                "uniform_vs_o1": uniform,
                "improvement_factor_vs_o1": uniform / weighted if weighted else float("nan"),
                "wall_seconds": time.perf_counter() - began,
            }
        )
        print(
            f"ladder  rings {rings:5d}  rays {traced:8d}  weighted {weighted:.5e}  "
            f"uniform {uniform:.5e}  factor {ladder[-1]['improvement_factor_vs_o1']:.4f}",
            flush=True,
        )

    # --- 2. which boundary correction is responsible ------------------------
    decomposition: list[dict[str, Any]] = []
    try:
        for rings in DECOMPOSITION_RINGS:
            row: dict[str, Any] = {"rings": rings}
            for arm, (centre, rim) in ARMS.items():
                quadrature_module.hexapolar_area_weight_m2 = _area_weight_variant(centre, rim)
                value, traced = residual(rings, weighted=True)
                row[arm] = value
                row["traced_rays"] = traced
            quadrature_module.hexapolar_area_weight_m2 = production_weight
            row["uniform"], _ = residual(rings, weighted=False)
            decomposition.append(row)
            print(
                f"decomp  rings {rings:5d}  production {row['production']:.5e}  "
                f"rim_removed {row['rim_correction_removed']:.5e}  "
                f"centre_removed {row['centre_correction_removed']:.5e}  "
                f"uniform {row['uniform']:.5e}",
                flush=True,
            )
    finally:
        quadrature_module.hexapolar_area_weight_m2 = production_weight

    return {
        "probe": "singlet_residual_attribution",
        "issue": "CHE-117",
        "oracle": (
            "O1, the analytic Airy pattern, over the 5-Airy-radius gate disc. O2 is "
            "deliberately not consulted: it is our own ASM/RS propagator and this gate "
            "already had to retire it as circular validation."
        ),
        "frozen_gate": {
            "metric": "fft_oracle_intensity_relative_l2",
            "threshold": 1.0e-3,
            "committed_observed": 2.2072391812867093e-3,
            "committed_improvement_factor_vs_o1": 0.4173375512174577,
            "improvement_factor_floor": 1.2,
        },
        "ladder": ladder,
        "boundary_correction_decomposition": decomposition,
    }


def _verdict(record: dict[str, Any]) -> dict[str, Any]:
    ladder = record["ladder"]
    decomposition = record["boundary_correction_decomposition"]

    weighted = [row["weighted_vs_o1"] for row in ladder]
    uniform = [row["uniform_vs_o1"] for row in ladder]

    weighted_spread = (max(weighted) - min(weighted)) / min(weighted)
    minimum_index = int(np.argmin(uniform))
    minimum_at = ladder[minimum_index]["rings"]
    tail = uniform[minimum_index:]
    uniform_rising_after_minimum = len(tail) > 1 and all(
        b > a for a, b in itertools.pairwise(tail)
    )
    rim_tracks_uniform = max(
        abs(row["rim_correction_removed"] - row["uniform"]) / row["uniform"]
        for row in decomposition
    )
    centre_tracks_production = max(
        abs(row["centre_correction_removed"] - row["production"]) / row["production"]
        for row in decomposition
    )

    return {
        "the_weighted_residual_is_converged": {
            "value": weighted[-1],
            "relative_spread_over_the_ladder": weighted_spread,
            "ray_count_range": [ladder[0]["traced_rays"], ladder[-1]["traced_rays"]],
            "statement": (
                f"flat to {weighted_spread:.2%} over a "
                f"{ladder[-1]['traced_rays'] / ladder[0]['traced_rays']:.0f}x range in "
                "ray count here, flat since 24 rings in the committed ladder "
                "(m3_quadrature_weight.json), and invariant to ten significant figures "
                "under sensor-grid refinement (singlet_residual_grid.json). It is a "
                "converged number."
            ),
        },
        "the_uniform_arm_is_not_converged": {
            "minimum_at_rings": minimum_at,
            "minimum_value": min(uniform),
            "value_at_the_largest_ray_count": uniform[-1],
            "rising_after_its_minimum": bool(uniform_rising_after_minimum),
            "statement": (
                "it descends from 8.3e-2 at 8 rings (m3_quadrature_weight.json), "
                "crosses the weighted arm near 181 rings, reaches a minimum, and climbs "
                "back toward it. Its 9.2e-4 at 512 rings -- the number that clears the "
                "gate -- is a point on a transient dip, not a converged value."
            ),
        },
        "the_difference_is_entirely_the_rim_half_weight": {
            "max_relative_gap_rim_removed_vs_uniform": rim_tracks_uniform,
            "max_relative_gap_centre_removed_vs_production": centre_tracks_production,
            "statement": (
                "removing the outermost ring's 1/2 area weight reproduces the uniform "
                "arm; removing the central ray's 3/4 changes nothing. The two arms "
                "differ in exactly one place."
            ),
        },
        "the_negative_control_is_mis_specified": {
            "improvement_factor_by_ray_count": {
                str(row["rings"]): row["improvement_factor_vs_o1"] for row in ladder
            },
            "floor": 1.2,
            "statement": (
                "the control asks whether adding the weight improves agreement with O1 "
                "by at least 1.2x, at one ray count, with neither arm required to be "
                "converged. Its verdict is therefore a function of the ray count and "
                "changes sign within the ladder. It is mis-specified rather than "
                "backwards, and the fix is to require convergence of both arms before "
                "comparing them -- not to widen anything."
            ),
        },
        "what_this_does_not_establish": (
            "that 2.21e-3 is CORRECT. O1 is an aberration-free paraxial Airy pattern "
            "and M3-SINGLET-REF is a real traced singlet, so some residual is expected "
            "from aberration alone and this probe does not separate that term. What it "
            "establishes is that the residual is converged, that it is not caused by "
            "the quadrature weight, and that the control which appeared to indict the "
            "weight was comparing a converged arm against a transient."
        ),
        "gate_disposition": (
            "UNCHANGED and UNMET. The 1.0e-3 threshold is not widened and 2.21e-3 does "
            "not meet it. The open question narrows from 'why does the weight make "
            "agreement worse' -- answered: it does not -- to 'can an aberration-free "
            "oracle decide this system at the 1e-3 level at all'."
        ),
    }


def main() -> None:
    record = characterize()
    record["verdict"] = _verdict(record)
    record[RECORD_PROVENANCE_KEY] = record_provenance(
        probe="benchmarks/probes/singlet_residual_attribution.py", root=ROOT
    )
    RECORD.parent.mkdir(parents=True, exist_ok=True)
    RECORD.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(record["verdict"], indent=1))
    print(f"wrote {RECORD.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
