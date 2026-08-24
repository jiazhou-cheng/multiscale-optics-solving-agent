"""Is the frozen configuration's first-null radius a converged measurement?

CHE-103 (M0.2). Regenerating ``m3_psf_verification.json`` against the current
tree moved ``diffraction_limited.vs_analytic_airy.first_null.
ratio_measured_over_analytic`` from ``0.9963`` to ``1.0345``, breaking a test
that required it within 1% of unity. The obvious reading is that CHE-47's
per-ray quadrature weight -- which halves the rim ring's area element and takes
the centre ray to 3/4 -- apodizes the pupil edge and broadens the core.

That reading is wrong, and the difference matters. Widening the tolerance would
have recorded a real defect as an accepted one; attributing it to the quadrature
weight would have put a correct piece of physics on trial. What this probe
measures is that **the frozen configuration does not sample the PSF finely
enough to answer the question at all**.

Two sweeps, because one cannot separate the candidates:

* ``rings`` -- refine the ray count, hold the grid. If the deviation were a
  discretization error of the rim cell it would fall as ``O(1/rings)``.
* ``grid`` -- refine the grid, hold the ray count. If the deviation were an
  estimator artifact of marginal sampling it would fall with pitch.

The frozen configuration puts 2.44 pixels across the Airy radius, and the
deviation in question is 0.09 pixels. That is the whole story: the ring sweep
shows the deviation is flat under a 9x change in ray count, so it is not ray
sampling; the grid sweep shows it collapses once the grid resolves the pattern,
so it is the grid.

Consequence for what the repository claims: the frozen M3 first-null number is
not a converged measurement, and the 1% agreement the old record showed was the
uniform-weight convention landing near unity rather than the measurement being
sound. The metric is reported, and pinned, as not-converged. The error budget
this belongs to is M2.1 (CHE-109); this probe establishes only that the frozen
grid is the binding term, which is what CHE-103 needed in order not to widen a
tolerance around it.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any

from core.provenance import RECORD_PROVENANCE_KEY, record_provenance

ROOT = Path(__file__).resolve().parents[2]
RECORD_PATH = Path(__file__).resolve().parent / "records" / "m3_first_null_grid_convergence.json"

sys.path.insert(0, str(Path(__file__).resolve().parent))

#: Ring counts for the ray-refinement arm. A hexapolar fan of ``n`` rings traces
#: ``3n(n+1)+1`` rays, so 16..96 spans 817..27937 -- a factor of 34.
RING_SWEEP = (16, 24, 32, 48, 64, 96)

#: Grid refinement factors for the sampling arm, at a fixed ring count. The
#: frozen configuration is ``1``; ``2`` is the first that clears 4 pixels per
#: Airy radius.
GRID_SWEEP = (1, 2, 4)

#: Ring count held fixed while the grid is refined. 48 rings (7057 rays) is past
#: the knee of the ring sweep, so the grid arm is not confounded by ray
#: starvation.
GRID_ARM_RINGS = 48


def _ratio(geometry: dict[str, Any], rays: Any, directory: Path, *, weighted: bool) -> Any:
    """First-null radius, measured over analytic, on one configuration."""
    import psf_oracle_verification as probe

    from couplers.handoff import HandoffPerturbation
    from verification.psf_oracles import first_null_comparison

    result = probe._perturbed_psf(
        rays,
        geometry,
        directory,
        handoff_perturbation=HandoffPerturbation(apply_quadrature_weight=weighted),
    )
    if result["status"] != "succeeded":
        return {"status": result["status"]}
    measurement = result["measurement"]
    comparison = first_null_comparison(
        measurement.intensity,
        sample_pitch_m=measurement.sample_pitch_m,
        wavelength_m=probe.WAVELENGTH_M,
        numerical_aperture=geometry["na_frozen"],
    )
    return comparison["ratio_measured_over_analytic"]


def _ring_arm() -> list[dict[str, Any]]:
    """Refine the rays, hold the grid frozen."""
    import psf_oracle_verification as probe

    rows: list[dict[str, Any]] = []
    for rings in RING_SWEEP:
        probe.NUM_RAYS = rings
        with tempfile.TemporaryDirectory() as workdir:
            root = Path(workdir)
            rays = probe._trace(probe.SINGLET, root / "rays")
            rows.append(
                {
                    "rings": rings,
                    "traced_rays": 3 * rings * (rings + 1) + 1,
                    "weighted": _ratio(probe.SINGLET, rays, root / "w", weighted=True),
                    "uniform": _ratio(probe.SINGLET, rays, root / "u", weighted=False),
                }
            )
    return rows


def _grid_arm() -> list[dict[str, Any]]:
    """Refine the grid, hold the ray count fixed."""
    import psf_oracle_verification as probe

    base = dict(probe.SINGLET)
    frozen_pixels_per_airy_radius = 2.439339871593739
    rows: list[dict[str, Any]] = []
    for refine in GRID_SWEEP:
        geometry = dict(base)
        geometry["pitch_m"] = base["pitch_m"] / refine
        geometry["grid_n"] = base["grid_n"] * refine
        geometry["pad_width"] = base["pad_width"] * refine
        probe.NUM_RAYS = GRID_ARM_RINGS
        with tempfile.TemporaryDirectory() as workdir:
            root = Path(workdir)
            rays = probe._trace(geometry, root / "rays")
            rows.append(
                {
                    "grid_refine": refine,
                    "grid_n": geometry["grid_n"],
                    "sample_pitch_m": geometry["pitch_m"],
                    "pixels_per_airy_radius": frozen_pixels_per_airy_radius * refine,
                    "weighted": _ratio(geometry, rays, root / "w", weighted=True),
                    "uniform": _ratio(geometry, rays, root / "u", weighted=False),
                }
            )
    return rows


def _spread(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [r[key] for r in rows if isinstance(r.get(key), float)]
    return float(max(values) - min(values)) if len(values) > 1 else None


def characterize() -> dict[str, Any]:
    ring_rows = _ring_arm()
    grid_rows = _grid_arm()

    weighted_flat_over_rings = _spread([r for r in ring_rows if r["rings"] >= 32], "weighted")
    frozen = next(r for r in grid_rows if r["grid_refine"] == 1)
    resolved = [r for r in grid_rows if r["grid_refine"] > 1]

    frozen_gap = abs(frozen["weighted"] - frozen["uniform"])
    resolved_gap = max(abs(r["weighted"] - r["uniform"]) for r in resolved)

    return {
        "probe": "m3_first_null_grid_convergence",
        "issue": "CHE-103 (M0.2)",
        "question": (
            "the frozen configuration reports first_null ratio_measured_over_analytic "
            "= 1.0345 with CHE-47's quadrature weight and 0.9963 without it. Is that "
            "3.4% a physical consequence of the weight, a ray-sampling error, or an "
            "artifact of the frozen grid?"
        ),
        "ray_refinement_holding_the_grid": {
            "rows": ring_rows,
            "weighted_spread_over_32_to_96_rings": weighted_flat_over_rings,
            "reading": (
                "The weighted ratio is flat at ~1.034 across a 9x change in ray count "
                "(3169 -> 27937 rays), and the UNIFORM ratio drifts UP toward it "
                "(0.9963 -> 1.0189) over the same sweep. Both conventions are heading "
                "for the same ~1.03 on this grid, which is the diagnostic: a limit "
                "that both conventions share is a property of the grid, not of the "
                "amplitude. Note what this arm does NOT show -- the DIFFERENCE "
                "between the conventions does fall with ring count (0.0382 at 32 "
                "rings to 0.0150 at 96, roughly O(1/rings)), exactly as a rim-cell "
                "quadrature correction should. So the rim cell is behaving "
                "correctly; it is the common offset that needs explaining, and the "
                "grid arm explains it."
            ),
        },
        "grid_refinement_holding_the_rays": {
            "rows": grid_rows,
            "rings": GRID_ARM_RINGS,
            "frozen_weighted_minus_uniform": frozen["weighted"] - frozen["uniform"],
            "resolved_weighted_minus_uniform_max": resolved_gap,
            "reading": (
                "the gap between the two conventions collapses from "
                f"{frozen_gap:.4f} at the frozen 2.44 pixels per Airy radius to at "
                f"most {resolved_gap:.4f} once the grid is refined. The deviation is "
                "the GRID, not the quadrature weight."
            ),
        },
        "verdict": {
            "deviation_is_a_grid_artifact": bool(resolved_gap < frozen_gap / 3.0),
            "frozen_configuration_is_first_null_converged": False,
            "pixels_per_airy_radius_frozen": frozen["pixels_per_airy_radius"],
            "statement": (
                "The frozen M3 configuration does not sample the PSF finely enough to "
                "measure a first-null radius to 1%. It puts 2.44 pixels across the "
                "Airy radius and the disputed shift is 0.09 pixels. The old record's "
                "0.9963 was the uniform-weight convention landing near unity at an "
                "unconverged sampling, not evidence that the measurement was sound. "
                "The supported statement about CHE-47's quadrature weight is the "
                "narrow one: the 3.6% is not a converged physical shift attributable "
                "to the weight. The weight is a necessary condition for the frozen "
                "gap -- without it there is no gap -- but at 4.88 pixels per Airy "
                "radius the two conventions agree to well under 1%, so what the "
                "frozen number measures is the sampling."
            ),
            "not_claimed": (
                "that either convention's first-null radius is correct in absolute "
                "terms, and not that the WEIGHTED arm is itself converged: it reads "
                "1.0354, 1.0039, 1.0093 over the grid sweep, moving away at the "
                "finest point, and the 0.0087 gap there is only just inside the 1% "
                "the retired assertion required. Both arms are non-monotone at the "
                "~0.5% level, which is the estimator's own floor. The rings=24 "
                "weighted point is a 5.4% excursion (1.0889) and is excluded from "
                "the flatness statistic, which is quoted over 32-96 rings only. "
                "Establishing an absolute first-null accuracy is M2.1 (CHE-109)'s "
                "error budget, not this probe's; this probe answers only which term "
                "is binding."
            ),
        },
    }


def main() -> None:
    record = characterize()
    record[RECORD_PROVENANCE_KEY] = record_provenance(
        probe="m3_first_null_grid_convergence",
        root=ROOT,
        extra_sources=[Path(__file__)],
        data_inputs=[ROOT / "benchmarks" / "protocols" / "slice_protocol.yaml"],
    )
    RECORD_PATH.parent.mkdir(parents=True, exist_ok=True)
    RECORD_PATH.write_text(json.dumps(record, indent=1, sort_keys=True, default=str) + "\n")
    print(f"wrote {RECORD_PATH.relative_to(ROOT)}")
    print(json.dumps(record["verdict"], indent=1, default=str))


if __name__ == "__main__":
    main()
