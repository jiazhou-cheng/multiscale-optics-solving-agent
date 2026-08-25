"""Record the performance baselines M5 will set its targets against.

CHE-105 (M0.4). One subcommand per baseline, because the expensive ones are
minutes each and a single script that runs all of them is a script nobody can
re-run one piece of.

    ./run.sh python benchmarks/perf/run_baselines.py overhead
    ./run.sh python benchmarks/perf/run_baselines.py scaling
    ./run.sh python benchmarks/perf/run_baselines.py estimate
    ./run.sh python benchmarks/perf/run_baselines.py suite
    ./run.sh python benchmarks/perf/run_baselines.py l2-psf-01
    ./run.sh python benchmarks/perf/run_baselines.py compare <baseline> <candidate>
    MOA_GPUS=device=6 ./run.sh --gpu python benchmarks/perf/run_baselines.py demo3 ...

The demo2 and demo3 subcommands are implemented and NOT run by CHE-105; their
baselines are deferred. Nothing about the harness is untested as a result -- the
whole-command path is exercised by the `suite` and `l2-psf-01` baselines, which
go through the same `_timed_command`.

Repeats
-------
Cheap baselines take the protocol's warmup + 3 repeats. The demo baselines take
one timed run and no warmup, and their records say so: three repeats of demo3 at
the 60 M-ray configuration is ten minutes of a shared GPU to turn a number that
varies by a few percent into a slightly better number. That is the wrong trade,
and stating it is better than quietly running one repeat and calling the median
of a single sample a median.

Every record is written under `benchmarks/perf/records/` and validated against
`benchmarks/schemas/performance.schema.json` before it is written, so a record
that does not fit the schema fails here rather than in whatever reads it later.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from core.performance import (  # noqa: E402
    PerformanceRecord,
    StageTimer,
    Workload,
    environment_fingerprint,
    fit_scaling,
    measure,
)

RECORDS = Path(__file__).resolve().parent / "records"
SCHEMA_PATH = ROOT / "benchmarks" / "schemas" / "performance.schema.json"


def _write(name: str, payload: dict[str, Any]) -> Path:
    RECORDS.mkdir(parents=True, exist_ok=True)
    path = RECORDS / f"{name}.json"
    # No `default=str`. It would silently stringify a numpy scalar into a
    # provenance record -- `"0.451"` where a reader expects a number -- rather
    # than failing here, which is the wrong direction for a file whose whole
    # purpose is that a later reader can trust what it says.
    path.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    print(f"wrote {path.relative_to(ROOT)}")
    return path


@contextmanager
def scan_limit_forced_off() -> Iterator[None]:
    """Run the reconstruction with the O(N^2) ray-density diagnostic disabled.

    Reaching into `couplers.ray_to_wave._NEAREST_NEIGHBOUR_SCAN_LIMIT`, a private
    module global, from a benchmark. Deliberate and narrow: it is the only way to
    separate the diagnostic from the reconstruction on ONE configuration rather
    than inferring the split from the threshold crossing, which would compare two
    different ray counts. Forcing the limit to 0 makes `_ray_density_diagnostic`
    take its `count > limit` early return, which changes the returned diagnostic
    triple and leaves the reconstructed field untouched -- so the two arms really
    are the same reconstruction.

    One helper rather than two hand-rolled try/finally blocks, because the next
    copy is the one that forgets the `finally`.
    """
    import couplers.ray_to_wave as rtw

    previous = rtw._NEAREST_NEIGHBOUR_SCAN_LIMIT
    rtw._NEAREST_NEIGHBOUR_SCAN_LIMIT = 0
    try:
        yield
    finally:
        rtw._NEAREST_NEIGHBOUR_SCAN_LIMIT = previous


#: `workload.route` for the diagnostic-disabled arm, and for the shipping arm
#: whenever the ray count is above the scan limit so the diagnostic does not run
#: anyway. Those two really are the same computation and comparing them is fair.
ROUTE_DIAGNOSTIC_OFF = "ramp_sum"
#: `workload.route` for a shipping call in which the O(N^2) diagnostic actually
#: runs. It goes in `route`, not only in `detail`, so that
#: `core.performance.compare` refuses to divide it by the diagnostic-disabled
#: call: with the marker only in `detail` the two records were byte-identical in
#: unit and route, and compare() returned an 11x "speedup" between two different
#: computations, from committed artifacts, with no refusal.
ROUTE_AS_SHIPPED = "ramp_sum+density_diagnostic"


def _r2(fit: Any) -> str:
    """`r_squared` for prose. `None` means the costs were all equal, not r^2 = 1."""
    return "n/a (no variance to explain)" if fit.r_squared is None else f"{fit.r_squared:.4f}"


#: Repeats per point in the ray-axis scaling sweep.
#:
#: Seven, not the protocol's three, and the number is measured rather than
#: chosen. At three repeats, four realizations of this baseline returned
#: ray-density-diagnostic exponents of 2.071, 2.020, 1.609 and 2.040 -- a spread
#: of 0.46 on the exponent that is quoted as evidence that a pairwise scan is
#: quadratic. The outlier came from the 817-ray row, whose call is ~0.04 s on a
#: shared 80-core box, where a 3-repeat median is not a median of anything
#: stable. The whole sweep is seconds, so buying the exponent back is nearly
#: free. What was already stable at three repeats and stays stable: the 3169-ray
#: diagnostic share (90.6-91.1%) and the reconstruction exponent (1.05-1.13).
_SCALING_REPEATS = 7


def shipping_route(ray_count: float) -> str:
    """Which route string a shipping `ray_to_wave` call is, at this ray count."""
    import couplers.ray_to_wave as rtw

    return (
        ROUTE_AS_SHIPPED
        if ray_count <= rtw._NEAREST_NEIGHBOUR_SCAN_LIMIT
        else ROUTE_DIAGNOSTIC_OFF
    )


def _validate(record: PerformanceRecord) -> dict[str, Any]:
    from jsonschema import Draft202012Validator

    payload = record.as_dict()
    Draft202012Validator(json.loads(SCHEMA_PATH.read_text())).validate(payload)
    return payload


# ---------------------------------------------------------------------------
# Framework overhead: the same physics through the abstraction and directly
# ---------------------------------------------------------------------------


def _singlet_request(directory: Path, num_rays: int = 32) -> Any:
    from solvers.base import ModelRunRequest

    return ModelRunRequest(
        run_id="perf",
        node_id="lens",
        config={
            "sample": "M3SingletRef",
            "num_rays": num_rays,
            "wavelength": 0.55,
            "Hx": 0.0,
            "Hy": 0.0,
            "handoff_plane": "exit_pupil",
            "output_directory": str(directory),
        },
    )


def baseline_overhead() -> None:
    """S5's number, on one solver node and one coupler node.

    "Framework overhead" is ambiguous unless you say what the denominator is, so
    both halves are defined here rather than assumed:

    * **Solver** -- `get_adapter().run(...)`, which validates the request,
      negotiates precision and device, builds the prescription, traces, writes
      an `.npy`, and assembles an artifact record with provenance -- against
      `build_optiland_system(...)` plus `optic.trace(...)`, which is the physics
      alone. The gap is everything the graph layer buys.
    * **Coupler** -- `RayToWaveCoupler().transform(...)`, the graph node, against
      `couplers.ray_to_wave.ray_to_wave(...)`, the pure function it calls. CHE-34
      pinned these two as bit-identical in output, which is what makes the timing
      difference pure overhead rather than a different computation.
    """
    import numpy as np

    from core.boundary import ComplexField  # noqa: F401  (import cost is part of the path)
    from couplers.base import CouplerRunRequest
    from couplers.handoff import DeclaredHandoffPlane, declare_coherent_bundle
    from couplers.node import RayToWaveCoupler
    from couplers.ray_to_wave import ray_to_wave
    from registry.prescriptions import resolve_prescription
    from solvers.optiland.adapter import get_adapter
    from solvers.optiland.builder import build_optiland_system

    results: dict[str, Any] = {
        "probe": "perf_framework_overhead",
        "issue": "CHE-105 (M0.4)",
        "environment": environment_fingerprint().as_dict(),
        "what_overhead_means_here": (
            "adapter/node wall time divided by the wall time of the physics call it "
            "wraps, on the same configuration. Above 1.0 by definition; the interesting "
            "quantity is how far above."
        ),
    }

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        adapter = get_adapter()

        # -- solver: through the adapter ------------------------------------
        def through_adapter(timer: StageTimer) -> Any:
            with timer.stage("adapter_run"):
                return adapter.run(_singlet_request(work / "adapter"))

        adapter_record, adapter_result = measure(
            through_adapter,
            label="M_RAY_OPTILAND via adapter.run",
            workload=Workload(size=32, unit="ray_fan_ring", route="adapter"),
            repeats=3,
            warmup=1,
        )

        # -- solver: the physics alone --------------------------------------
        spec = resolve_prescription("M3SingletRef")

        def direct_trace(timer: StageTimer) -> Any:
            with timer.stage("build_optic"):
                optic = build_optiland_system(spec)
            with timer.stage("trace"):
                optic.trace(Hx=0.0, Hy=0.0, wavelength=0.55, num_rays=32, distribution="hexapolar")
            return optic

        direct_record, _ = measure(
            direct_trace,
            label="M_RAY_OPTILAND direct build + trace",
            workload=Workload(size=32, unit="ray_fan_ring", route="direct"),
            repeats=3,
            warmup=1,
        )

        results["solver"] = {
            "framework_s": adapter_record.measurement.median_s,
            "direct_s": direct_record.measurement.median_s,
            "overhead_ratio": (
                adapter_record.measurement.median_s / direct_record.measurement.median_s
            ),
            "overhead_s": (
                adapter_record.measurement.median_s - direct_record.measurement.median_s
            ),
            "framework_record": _validate(adapter_record),
            "direct_record": _validate(direct_record),
            "note": (
                "The direct arm rebuilds the Optic each repeat, which the adapter also "
                "does, so the comparison is like for like. What the adapter adds on top "
                "is request validation, the precision/device bridge, the .npy write and "
                "the artifact record."
            ),
        }

        # -- coupler: node vs pure function ---------------------------------
        rays = adapter_result.outputs["rays"]
        pupil_z_m = 0.06814345991561233e-3
        grid_n, pitch_m = 188, 2.6587352810843895e-06
        bundle = declare_coherent_bundle(
            rays, declared_plane=DeclaredHandoffPlane("exit_pupil", pupil_z_m)
        ).bundle
        ray_count = int(np.asarray(bundle.positions_m).shape[0])

        node = RayToWaveCoupler()

        def through_node(timer: StageTimer) -> Any:
            with timer.stage("node_transform"):
                return node.transform(
                    CouplerRunRequest(
                        run_id="perf",
                        edge_id="pupil",
                        source=rays,
                        config={
                            "handoff_plane": "exit_pupil",
                            "handoff_plane_z_m": pupil_z_m,
                            "grid_n": grid_n,
                            "target_sample_pitch_m": pitch_m,
                            "output_dir": str(work / "node"),
                        },
                    )
                )

        node_record, _ = measure(
            through_node,
            label="C_RAY_TO_WAVE via node.transform",
            workload=Workload(
                size=ray_count * grid_n * grid_n,
                unit="ray_pixel_product",
                route="node",
                detail={"rays": ray_count, "grid_n": grid_n},
            ),
            repeats=3,
            warmup=1,
        )

        def direct_kernel(timer: StageTimer) -> Any:
            with timer.stage("ray_to_wave"):
                return ray_to_wave(
                    bundle,
                    grid_shape=(grid_n, grid_n),
                    sample_pitch_m=(pitch_m, pitch_m),
                )

        kernel_record, _ = measure(
            direct_kernel,
            label="C_RAY_TO_WAVE direct kernel call",
            workload=Workload(
                size=ray_count * grid_n * grid_n,
                unit="ray_pixel_product",
                route="direct",
                detail={"rays": ray_count, "grid_n": grid_n},
            ),
            repeats=3,
            warmup=1,
        )

        results["coupler"] = {
            "framework_s": node_record.measurement.median_s,
            "direct_s": kernel_record.measurement.median_s,
            "overhead_ratio": (
                node_record.measurement.median_s / kernel_record.measurement.median_s
            ),
            "overhead_s": (
                node_record.measurement.median_s - kernel_record.measurement.median_s
            ),
            "framework_record": _validate(node_record),
            "direct_record": _validate(kernel_record),
            "note": (
                "CHE-34 pinned the node's output as bit-identical to the direct call, "
                "so the difference is overhead and not a different computation. The "
                "node adds config validation, the handoff declaration the direct arm "
                "was handed pre-built, and the artifact write."
            ),
        }

    _write("framework_overhead", results)


# ---------------------------------------------------------------------------
# Scaling: a fitted exponent on the ray axis
# ---------------------------------------------------------------------------


def baseline_scaling() -> None:
    """Fit the coupler's cost exponent on the ray axis, holding the grid.

    Two fits, not one, because the first sweep found the curve is not a power
    law at all: cost RISES superlinearly to 3169 rays, then DROPS by 4x at 4921
    and becomes linear. That is a code path, not noise.
    ``_ray_density_diagnostic`` runs an O(N^2) pairwise nearest-neighbour scan
    below ``_NEAREST_NEIGHBOUR_SCAN_LIMIT`` (4096 rays) and is skipped above it,
    so the two sides of that threshold are different computations and fitting
    one exponent across them would describe neither.

    Measured share of the call taken by that diagnostic: 70% at 817 rays, 85% at
    1801, **91% at 3169** -- which is the frozen M3-SINGLET-REF configuration.
    So at the ray count this repository actually uses, nine tenths of
    C_RAY_TO_WAVE's wall time is a sampling diagnostic and one tenth is the
    physics. This is CHE-101's lesson one layer down: the stage a reader would
    optimize is not the stage that costs.
    """
    import numpy as np

    import couplers.ray_to_wave as rtw
    from couplers.handoff import DeclaredHandoffPlane, declare_coherent_bundle
    from solvers.base import ModelRunRequest
    from solvers.optiland.adapter import get_adapter

    grid_n, pitch_m = 188, 2.6587352810843895e-06
    pupil_z_m = 0.06814345991561233e-3
    scan_limit = rtw._NEAREST_NEIGHBOUR_SCAN_LIMIT
    rows: list[dict[str, Any]] = []

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        adapter = get_adapter()
        for rings in (16, 24, 32, 40, 48, 64):
            traced = adapter.run(
                ModelRunRequest(
                    run_id="perf",
                    node_id="lens",
                    config={
                        "sample": "M3SingletRef",
                        "num_rays": rings,
                        "wavelength": 0.55,
                        "Hx": 0.0,
                        "Hy": 0.0,
                        "handoff_plane": "exit_pupil",
                        "output_directory": str(work / f"r{rings}"),
                    },
                )
            ).outputs["rays"]
            bundle = declare_coherent_bundle(
                traced, declared_plane=DeclaredHandoffPlane("exit_pupil", pupil_z_m)
            ).bundle
            ray_count = int(np.asarray(bundle.positions_m).shape[0])

            def kernel(timer: StageTimer, _b: Any = bundle) -> Any:
                with timer.stage("ray_to_wave"):
                    return rtw.ray_to_wave(
                        _b, grid_shape=(grid_n, grid_n), sample_pitch_m=(pitch_m, pitch_m)
                    )

            diagnostic_runs = ray_count <= scan_limit
            full, _ = measure(
                kernel,
                label=f"C_RAY_TO_WAVE full call rings={rings}",
                workload=Workload(
                    size=ray_count, unit="ray", route=shipping_route(ray_count),
                    detail={"rings": rings, "grid_n": grid_n, "diagnostic": "as shipped"},
                ),
                repeats=_SCALING_REPEATS,
                warmup=1,
            )

            # The same call with the diagnostic forced off, to separate the two
            # costs on ONE configuration rather than inferring the split from the
            # threshold crossing.
            with scan_limit_forced_off():
                reconstruction, _ = measure(
                    kernel,
                    label=f"C_RAY_TO_WAVE reconstruction only rings={rings}",
                    workload=Workload(
                        size=ray_count, unit="ray", route=ROUTE_DIAGNOSTIC_OFF,
                        detail={"rings": rings, "grid_n": grid_n, "diagnostic": "forced off"},
                    ),
                    repeats=_SCALING_REPEATS,
                    warmup=1,
                )

            # Only a difference between two DIFFERENT computations is a
            # diagnostic cost. Above the scan limit the diagnostic does not run in
            # either arm, so the difference is run-to-run noise -- and publishing
            # noise under the name `diagnostic_share_of_call` produced committed
            # rows reading -10.8% and -2.5%, which a downstream tool would read as
            # a share. Reported as null instead, with the noise kept under its own
            # name so the ~10% spread at 3 repeats stays visible.
            delta_s = full.measurement.median_s - reconstruction.measurement.median_s
            rows.append(
                {
                    "rings": rings,
                    "rays": ray_count,
                    "diagnostic_runs": diagnostic_runs,
                    "full_call_s": full.measurement.median_s,
                    "reconstruction_only_s": reconstruction.measurement.median_s,
                    "diagnostic_s": delta_s if diagnostic_runs else None,
                    "diagnostic_share_of_call": (
                        delta_s / full.measurement.median_s
                        if diagnostic_runs and full.measurement.median_s
                        else None
                    ),
                    "arm_difference_s": delta_s,
                    "arm_difference_note": (
                        "the diagnostic's cost"
                        if diagnostic_runs
                        else "run-to-run noise between two identical computations, "
                        "which is the noise floor the shares above should be read against"
                    ),
                    "seconds_per_ray_full": full.cost_per_unit,
                    "seconds_per_ray_reconstruction": reconstruction.cost_per_unit,
                    "full_record": _validate(full),
                    "reconstruction_record": _validate(reconstruction),
                }
            )

    reconstruction_fit = fit_scaling(
        [(r["rays"], r["reconstruction_only_s"]) for r in rows], axis="rays"
    )
    diagnostic_rows = [
        r for r in rows if r["diagnostic_runs"] and (r["diagnostic_s"] or 0) > 0
    ]
    diagnostic_fit = (
        fit_scaling([(r["rays"], r["diagnostic_s"]) for r in diagnostic_rows], axis="rays")
        if len(diagnostic_rows) >= 3
        else None
    )
    as_shipped_fit = fit_scaling([(r["rays"], r["full_call_s"]) for r in rows], axis="rays")

    _write(
        "scaling_ray_axis",
        {
            "probe": "perf_scaling_ray_axis",
            "issue": "CHE-105 (M0.4)",
            "environment": environment_fingerprint().as_dict(),
            "held_fixed": {
                "grid_n": grid_n,
                "sample_pitch_m": pitch_m,
                # The reconstruction is `ramp_sum` on every row. `workload.route`
                # on the individual records is NOT always that string: below the
                # scan limit the shipping arm is marked
                # `ramp_sum+density_diagnostic`, so that `compare` refuses to
                # divide it by the diagnostic-disabled arm. Same reconstruction,
                # different work.
                "reconstruction": "ramp_sum",
            },
            "nearest_neighbour_scan_limit_rays": scan_limit,
            "fits": {
                "reconstruction_only": reconstruction_fit.as_dict(),
                "ray_density_diagnostic": (
                    diagnostic_fit.as_dict() if diagnostic_fit else None
                ),
                "as_shipped_do_not_use": {
                    **as_shipped_fit.as_dict(),
                    "why_not": (
                        "fitted across the scan-limit threshold, so it averages two "
                        "different computations. Reported to show that a single "
                        "exponent here is meaningless -- its r^2 is the evidence."
                    ),
                },
            },
            "rows": rows,
            "finding": (
                f"The reconstruction scales as rays^{reconstruction_fit.exponent:.3f} "
                f"(r^2 {_r2(reconstruction_fit)}) at fixed grid -- linear in "
                "rays, consistent with the O(rays x pixels) product model and NOT with "
                "the registry's O(rays + pixels). The ray-density diagnostic scales as "
                + (
                    f"rays^{diagnostic_fit.exponent:.3f} "
                    f"(r^2 {_r2(diagnostic_fit)}), i.e. quadratic as its "
                    "pairwise scan implies"
                    if diagnostic_fit
                    else "quadratic by construction (pairwise scan)"
                )
                + ", and is skipped entirely above "
                f"{scan_limit} rays. At the frozen M3-SINGLET-REF configuration of "
                "3169 rays the diagnostic is ~91% of the call."
            ),
            "consequence": (
                "At the frozen M3-SINGLET-REF configuration (3169 rays) the "
                "reconstruction is ~9% of the call, so optimizing the kernel there "
                "moves ~9% of the wall time and the diagnostic is the target. The "
                "share is NOT ~9% across the whole sub-4096-ray range -- it rises as "
                "the ray count falls, reaching ~30% at 817 rays, because the "
                "diagnostic is quadratic and the reconstruction is linear. Read the "
                "per-row `diagnostic_share_of_call` for a given ray count rather than "
                "carrying one number across the range. The diagnostic is not physics: "
                "it is a sampling check that could be sampled rather than computed "
                "exhaustively, or cached. Handed to M5.2 (CHE-119) as a measured "
                "target rather than a guess."
            ),
        },
    )


# ---------------------------------------------------------------------------
# estimate() vs measured
# ---------------------------------------------------------------------------


def baseline_estimate() -> None:
    """Score the cost model M3's executor and the agent will both plan against.

    An estimator nobody has scored is a guess with a type signature. Reported
    per component, including the components whose honest answer is `None` --
    Optiland's `estimate()` returns no wall time at all and says why, which is a
    better failure than a number with no basis and should be recorded as such
    rather than skipped.
    """

    from couplers.base import CouplerRunRequest
    from couplers.node import RayToWaveCoupler
    from solvers.optiland.adapter import get_adapter

    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        adapter = get_adapter()
        request = _singlet_request(work / "est")

        predicted = adapter.estimate(request)

        def run_adapter(timer: StageTimer) -> Any:
            with timer.stage("run"):
                return adapter.run(_singlet_request(work / "measured"))

        record, result = measure(
            run_adapter,
            label="M_RAY_OPTILAND",
            workload=Workload(size=32, unit="ray_fan_ring", route="adapter"),
            repeats=3,
            warmup=1,
        )
        rows.append(
            {
                "component": "M_RAY_OPTILAND",
                "predicted_wall_time_s": predicted.wall_time_s,
                "measured_median_s": record.measurement.median_s,
                "ratio_predicted_over_measured": (
                    predicted.wall_time_s / record.measurement.median_s
                    if predicted.wall_time_s and record.measurement.median_s
                    else None
                ),
                "confidence": predicted.confidence,
                # Derived from the estimate rather than written down, because the
                # answer now depends on the host: CHE-118 gave `estimate()` a
                # measured cost model that is bound to the environment fingerprint
                # it was calibrated on, so the same code predicts on the GPU box it
                # was fitted on and refuses elsewhere. Hardcoded prose here would be
                # wrong on one of the two.
                "verdict": (
                    (
                        f"PREDICTS {predicted.wall_time_s:.4f} s at confidence "
                        f"'{predicted.confidence}', against a measured "
                        f"{record.measurement.median_s:.4f} s for the whole adapter "
                        "call. The prediction covers the TRACE only -- the call also "
                        "builds the system, generates its rays and writes artifacts, "
                        "none of which is calibrated -- so a ratio below 1 here is "
                        "expected and is not an underestimate of what was modelled."
                    )
                    if predicted.wall_time_s is not None
                    else (
                        "NO PREDICTION on this host, and the reason is now specific: "
                        "the cost model in solvers.optiland.cost_model is bound to "
                        "the environment fingerprint it was calibrated on (CHE-118) "
                        "and refuses to extrapolate. It is not the old refusal -- the "
                        "surface count and the traced ray count ARE known now, and "
                        "are reported in the notes. Re-run the calibration sweep on "
                        "this host to get a prediction here."
                    )
                ),
                "notes": predicted.notes,
            }
        )

        rays = result.outputs["rays"]
        pupil_z_m = 0.06814345991561233e-3
        grid_n, pitch_m = 188, 2.6587352810843895e-06
        config = {
            "handoff_plane": "exit_pupil",
            "handoff_plane_z_m": pupil_z_m,
            "grid_n": grid_n,
            "target_sample_pitch_m": pitch_m,
            "output_dir": str(work / "coupler"),
        }
        coupler_request = CouplerRunRequest(
            run_id="perf", edge_id="pupil", source=rays, config=config
        )
        node = RayToWaveCoupler()
        coupler_predicted = node.estimate(coupler_request)

        def run_node(timer: StageTimer) -> Any:
            with timer.stage("transform"):
                return node.transform(
                    CouplerRunRequest(
                        run_id="perf", edge_id="pupil", source=rays, config=config
                    )
                )

        ray_count = float(rays.shape[0]) if rays.shape else 1.0
        coupler_record, _ = measure(
            run_node,
            label="C_RAY_TO_WAVE",
            workload=Workload(
                size=ray_count,
                unit="ray",
                # Marked, so `compare` refuses to divide this by the arm below.
                # Unmarked, these two were byte-identical in unit and route and
                # compare() returned an 11x "speedup" between two different
                # computations.
                route=shipping_route(ray_count),
                detail={"diagnostic": "as shipped"},
            ),
            repeats=3,
            warmup=1,
        )
        # Scored against BOTH denominators, because the estimator models the
        # ray x pixel reconstruction and the shipping call is dominated by
        # something else. Reporting one ratio would pick which error to show.
        with scan_limit_forced_off():
            reconstruction_record, _ = measure(
                run_node,
                label="C_RAY_TO_WAVE reconstruction only",
                workload=Workload(
                    size=ray_count,
                    unit="ray",
                    route=ROUTE_DIAGNOSTIC_OFF,
                    detail={"diagnostic": "forced off"},
                ),
                repeats=3,
                warmup=1,
            )

        predicted_s = coupler_predicted.wall_time_s
        full_s = coupler_record.measurement.median_s
        recon_s = reconstruction_record.measurement.median_s
        rows.append(
            {
                "component": "C_RAY_TO_WAVE",
                "predicted_wall_time_s": predicted_s,
                "predicted_peak_memory_bytes": coupler_predicted.peak_memory_bytes,
                "measured_median_s": full_s,
                "measured_reconstruction_only_s": recon_s,
                "measured_peak_host_rss_bytes": coupler_record.peak_host_rss_bytes,
                "ratio_predicted_over_measured": predicted_s / full_s if predicted_s else None,
                "ratio_predicted_over_reconstruction": (
                    predicted_s / recon_s if predicted_s else None
                ),
                "confidence": coupler_predicted.confidence,
                "verdict": (
                    "WRONG IN BOTH DIRECTIONS, and the two errors have different causes. "
                    f"Against the shipping call ({full_s:.3f} s) it under-predicts by "
                    f"{full_s / predicted_s:.1f}x, because it models the ray x pixel "
                    "reconstruction and the call is dominated by the O(N^2) ray-density "
                    f"diagnostic. Against the reconstruction alone ({recon_s:.3f} s) it "
                    f"over-predicts by {predicted_s / recon_s:.1f}x, because its "
                    "_RAY_PIXEL_PRODUCTS_PER_SECOND constant is not calibrated to this "
                    "host. A planner ordering work by this number would be wrong about "
                    "which node is expensive, which is the failure that matters."
                ),
                "notes": coupler_predicted.notes,
                "reconstruction_record": _validate(reconstruction_record),
            }
        )

    _write(
        "estimate_accuracy",
        {
            "probe": "perf_estimate_accuracy",
            "issue": "CHE-105 (M0.4)",
            "environment": environment_fingerprint().as_dict(),
            "rows": rows,
            "why_this_matters": (
                "M3's executor and M6's planner both order work by CostEstimate. An "
                "estimator that has never been scored against a measurement is a guess, "
                "and one that returns None is at least a guess that says so."
            ),
        },
    )


# ---------------------------------------------------------------------------
# Whole-command baselines
# ---------------------------------------------------------------------------


#: How often the memory guard looks at a running subprocess. AGENTS.md asks for
#: swap to be watched *during* a substantial run, and `measure` only checks
#: between repeats -- which, for a one-repeat 200-second command, is after it has
#: already finished. Half a second is fine against a workload whose memory grows
#: over seconds.
_CHILD_POLL_S = 0.5


def _timed_command(
    argv: list[str], *, label: str, workload: Workload, notes: list[str]
) -> dict[str, Any]:
    """Run a subprocess once and record it. One repeat, and the record says so.

    The child is polled rather than waited on, so a memory guard breach
    **terminates it** instead of being discovered in the report afterwards. For
    the demo baselines this is the difference between a stop condition and a
    post-mortem: they are single-repeat commands of minutes, and `measure` can
    only check its own watchdog between repeats.
    """
    from core.performance import MemoryGuardBreached, SwapGrowthAbort
    from core.resources import MemoryWatchdog

    captured: dict[str, Any] = {}

    def run(timer: StageTimer) -> int:
        import resource

        # `ru_maxrss` over waited-for children, which is the only figure here that
        # describes the CHILD. `record.peak_host_rss_bytes` is this process's own
        # RSS and says nothing about the workload -- it read 0.32 GB when the
        # parent imported torch and 0.03 GB once it stopped, neither of which is
        # the demo's memory. Sampled as a delta because the counter is a
        # high-water mark across every child this process has ever reaped.
        rss_before = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss

        # Output to files, not pipes: a chunked demo prints steadily and a
        # subprocess that fills a pipe nobody is draining deadlocks, which would
        # look like a slow benchmark.
        with timer.stage("subprocess"), tempfile.TemporaryDirectory() as logs:
            out_path, err_path = Path(logs) / "stdout", Path(logs) / "stderr"
            guard = MemoryWatchdog(interval_s=_CHILD_POLL_S).start()
            try:
                with out_path.open("w") as out, err_path.open("w") as err:
                    proc = subprocess.Popen(argv, cwd=ROOT, stdout=out, stderr=err, text=True)
                    while True:
                        try:
                            returncode = proc.wait(timeout=_CHILD_POLL_S)
                            break
                        except subprocess.TimeoutExpired:
                            pass
                        growth = guard.swap_growth_bytes
                        breach = guard.verdict.breached
                        if growth or breach:
                            proc.terminate()
                            try:
                                proc.wait(timeout=30)
                            except subprocess.TimeoutExpired:  # pragma: no cover
                                proc.kill()
                                proc.wait()
                            report = guard.report()
                            if growth:
                                raise SwapGrowthAbort(growth, report)
                            raise MemoryGuardBreached(guard.verdict, report)
            finally:
                guard.stop()
            captured["returncode"] = returncode
            captured["tail"] = out_path.read_text().strip().splitlines()[-8:]
            captured["stderr_tail"] = err_path.read_text().strip().splitlines()[-12:]
            after = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
            captured["peak_child_rss_bytes"] = (
                # ru_maxrss is kilobytes on Linux.
                int(after) * 1024 if after > rss_before else None
            )
        return returncode

    record, _ = measure(
        run,
        label=label,
        workload=workload,
        repeats=1,
        warmup=0,
        notes=[
            *notes,
            "Measured across a process boundary, so this process does not touch a "
            "device: initializing JAX here would preallocate ~78% of the card and "
            "leave the child to OOM on an allocation it has room for alone. The "
            "process boundary is the barrier, and `cuda` is null because a parent's "
            "allocator snapshot is not the child's peak -- read the demo's own "
            "record for its device memory.",
        ],
        touch_devices=False,
    )

    # A failed run is not a baseline. Timing how long something took to crash and
    # committing it as a performance record is worse than having no record: the
    # number is real, the label is a lie, and the next reader compares against it.
    # This check exists because it caught exactly that -- L2-PSF-01 exited 1 in
    # 0.5 s and was written out as a 0.5-second bundle baseline.
    if captured["returncode"] != 0:
        raise RuntimeError(
            f"{label!r} exited {captured['returncode']} after "
            f"{record.measurement.median_s:.2f} s, so there is nothing to baseline. "
            "stderr tail:\n  " + "\n  ".join(captured["stderr_tail"])
        )

    payload = _validate(record)
    payload["subprocess"] = {"argv": argv, **captured}
    return payload


def baseline_suite() -> None:
    """The PR gate itself. Its growth is the thing worth watching."""
    payload = _timed_command(
        [sys.executable, "-m", "pytest", "-q", "tests"],
        label="default test suite",
        workload=Workload(size=1.0, unit="suite_run", route="pytest -q tests"),
        notes=[
            "One timed run, no warmup. Repeating a 3.5-minute suite three times to "
            "median it would cost ten minutes to sharpen a number whose useful "
            "resolution is 'did this grow by tens of seconds'.",
            "Run INSIDE the already-containerized process, so this measures the suite "
            "and not docker startup.",
        ],
    )
    _write(f"suite_default_{_device_suffix()}", payload)


def baseline_l2_psf_01() -> None:
    payload = _timed_command(
        [sys.executable, str(ROOT / "benchmarks" / "physics" / "L2-PSF-01" / "run_benchmark.py")],
        label="L2-PSF-01 bundle",
        workload=Workload(size=1.0, unit="bundle_run", route="M3-SLICE-CPU-V1"),
        notes=["One timed run, no warmup, for the same reason as the suite baseline."],
    )
    _write(f"l2_psf_01_{_device_suffix()}", payload)


def _device_suffix() -> str:
    """`cuda` or `cpu`, from what is actually visible.

    In the record NAME, not only in the fingerprint. The fingerprint is what
    makes two records incomparable; the filename is what stops the second run
    from silently overwriting the first, which it did once here -- a GPU demo2
    run landed on top of the CPU one and the only evidence left was that the
    number had changed.
    """
    fingerprint = environment_fingerprint()
    return "cuda" if fingerprint.gpu_count else "cpu"


#: Where the demo probes write their own scientific records. A timing run must
#: not land on top of a committed one -- CHE-105 did exactly that once, a GPU
#: demo2 run overwriting the CPU record, and the only evidence left was that a
#: number had changed. So every timing run is given an explicit `--output-name`
#: under the `perf_` prefix, which no committed record uses.
DEMO_RECORDS = ROOT / "benchmarks" / "probes" / "records" / "ray_wave"


def _demo(script: str, name: str, extra: list[str], *, declared_rays: float, unit: str,
          route: str, detail: dict[str, Any], label_suffix: str | None = None) -> None:
    """Time a demo end to end, and take its ray count and stage split from its record.

    Two things the earlier version of this got wrong and that matter for the
    numbers M5 will be set against:

    * **The workload size was declared, not read.** `--rays` came from whoever
      typed the command and was written straight into `workload.size`, so
      `seconds_per_ray` was partly an operator's arithmetic. The demo record says
      how many rays it actually emitted; that is the number used, and the typed
      one is kept beside it as `declared_rays` so a mismatch is visible rather
      than silently resolved.
    * **The demo overwrote its own committed record as a side effect.** Redirected
      with `--output-name` to a `perf_`-prefixed name, so a timing run never
      touches a scientific record.
    """
    # `label_suffix` before the device suffix, so a re-measurement of the same
    # configuration lands BESIDE the committed one instead of on top of it. Without
    # it, a before/after on one configuration -- which is what any optimization
    # ticket needs -- destroys its own baseline on the second run, and the only
    # evidence left is that a number changed. That already happened once here with
    # demo2 (see `_device_suffix`), and the device suffix only separates records
    # that differ by device.
    if label_suffix:
        name = f"{name}_{label_suffix}"
    name = f"{name}_{_device_suffix()}"
    probe_record_name = f"perf_{name}"
    payload = _timed_command(
        [
            sys.executable,
            str(ROOT / "benchmarks" / "probes" / "ray_wave" / script),
            *extra,
            "--output-name",
            probe_record_name,
        ],
        label=name,
        # Provisional: replaced below by the ray count the demo actually emitted.
        workload=Workload(size=declared_rays, unit=unit, route=route, detail=detail),
        notes=[
            "One timed run, no warmup. Three repeats of this configuration is minutes "
            "of a shared GPU to improve a number that varies by a few percent.",
            "seconds_per_ray is route-specific: ramp_sum is O(rays x pixels) and "
            "kspace_splat is O(rays) + one FFT, so the two are not one number. "
            "core.performance.compare refuses to divide them.",
            f"The demo's own record is at benchmarks/probes/records/ray_wave/"
            f"{probe_record_name}.json -- written under a perf_ name so this timing "
            "run does not overwrite a committed scientific record.",
        ],
    )

    measured = _demo_workload(probe_record_name)
    if measured is not None:
        rays, stages = measured
        payload["workload"]["size"] = rays
        payload["workload"]["detail"] = {
            **detail,
            "declared_rays": declared_rays,
            "rays_read_from": f"{probe_record_name}.json",
            **({"label_suffix": label_suffix} if label_suffix else {}),
        }
        payload["cost_per_unit"] = (
            payload["measurement"]["median_s"] / rays if rays else None
        )
        if stages:
            # The five-stage breakdown CHE-129 asks for. It comes from the demo's
            # own in-process timers, not from this harness's StageTimer, because
            # the harness times a subprocess and cannot see inside it. Kept under
            # its own key so nobody mistakes it for a StageTimer breakdown.
            payload["stages"] = {
                "seconds": stages,
                "accounted_s": round(sum(stages.values()), 4),
                "fraction_of_total": {
                    k: round(v / payload["measurement"]["median_s"], 4)
                    for k, v in stages.items()
                },
                "unaccounted_s": round(
                    payload["measurement"]["median_s"] - sum(stages.values()), 4
                ),
                "total_s": round(payload["measurement"]["median_s"], 4),
                "source": (
                    f"{probe_record_name}.json stage_wall_clock_s -- the demo's own "
                    "per-stage timers, summed over chunks. The denominator is the "
                    "whole command, which includes interpreter start, JAX "
                    "compilation, the plan setup and the record write, so the "
                    "unaccounted remainder is real and named rather than hidden."
                ),
            }
        if abs(rays - declared_rays) > 0.005 * max(rays, declared_rays):
            print(
                f"NOTE: declared {declared_rays:,.0f} rays, the demo emitted "
                f"{rays:,.0f}. seconds_per_ray uses the emitted count."
            )
    else:
        payload["workload"]["detail"] = {
            **detail,
            "declared_rays": declared_rays,
            "rays_read_from": None,
            "size_is_declared_not_measured": (
                "the demo record could not be read, so workload.size is the operator's "
                "typed --rays and seconds_per_ray inherits whatever that was"
            ),
        }
    _validate_payload(payload)
    _write(name, payload)


def _demo_workload(record_name: str) -> tuple[float, dict[str, float]] | None:
    """The rays the demo actually emitted, and its per-stage wall clock.

    Summed across routes and seeds, because the timed command is the whole demo:
    a per-ray cost computed against one route's rays while the clock covers both
    would be wrong by the ratio between them.

    The two demos shape their records differently -- demo3 nests per-seed `runs`
    and carries `stage_wall_clock_s`, demo2 puts one run per route inline and has
    no stage split at all, since it is a bare SLM with no ray trace to time. Both
    are read; a demo with no stages returns an empty dict rather than a fabricated
    one.
    """
    path = DEMO_RECORDS / f"{record_name}.json"
    if not path.exists():
        return None
    record = json.loads(path.read_text())
    rays = 0.0
    stages: dict[str, float] = {}

    def absorb(run: dict[str, Any]) -> None:
        nonlocal rays
        # `rays_emitted_here` is what THIS process traced; `total_rays` can name a
        # larger enumeration carried by several processes, and a per-second rate
        # against rays another process carried is not a rate.
        rays += float(run.get("rays_emitted_here") or run.get("total_rays") or 0.0)
        for key, value in (run.get("stage_wall_clock_s") or {}).items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue  # the stage dict carries a prose `note` alongside the numbers
            stages[key] = stages.get(key, 0.0) + float(value)

    for route in (record.get("routes") or {}).values():
        if not isinstance(route, dict):
            continue
        runs = route.get("runs")
        if isinstance(runs, list) and runs:
            for run in runs:
                if isinstance(run, dict):
                    absorb(run)
        else:
            absorb(route)
    return (rays, stages) if rays > 0 else None


def _validate_payload(payload: dict[str, Any]) -> None:
    from jsonschema import Draft202012Validator

    Draft202012Validator(json.loads(SCHEMA_PATH.read_text())).validate(payload)


def baseline_demo2(args: argparse.Namespace) -> None:
    _demo(
        "demo2_hologram.py",
        f"demo2_{args.preset}_{args.routes}_{args.reconstruction}",
        ["--preset", args.preset, "--routes", args.routes,
         "--reconstruction", args.reconstruction, "--backend", args.backend],
        declared_rays=float(args.rays),
        unit="ray",
        route=args.reconstruction,
        detail={"preset": args.preset, "routes": args.routes, "backend": args.backend},
        label_suffix=args.label_suffix,
    )


def baseline_demo3(args: argparse.Namespace) -> None:
    _demo(
        "demo3_hologram_lens.py",
        f"demo3_{args.preset}_{args.routes}_{args.reconstruction}",
        ["--preset", args.preset, "--routes", args.routes,
         "--reconstruction", args.reconstruction, "--backend", args.backend],
        declared_rays=float(args.rays),
        unit="ray",
        route=args.reconstruction,
        detail={"preset": args.preset, "routes": args.routes, "backend": args.backend},
        label_suffix=args.label_suffix,
    )


def compare_records(args: argparse.Namespace) -> None:
    """Compare two committed records, or explain why they cannot be compared.

    The user-facing half of the acceptance criterion. `core.performance.compare`
    raises; this prints the refusal, because a person running this at a terminal
    wants the reason and not a traceback.
    """
    from core.performance import Incomparable, compare

    left = json.loads((RECORDS / f"{args.baseline}.json").read_text())
    right = json.loads((RECORDS / f"{args.candidate}.json").read_text())
    try:
        print(json.dumps(compare(left, right), indent=1))
    except Incomparable as refusal:
        print("REFUSED TO COMPARE")
        print(f"  {refusal}")
        raise SystemExit(2) from None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    cmp_parser = sub.add_parser("compare", help="compare two committed records")
    cmp_parser.add_argument("baseline")
    cmp_parser.add_argument("candidate")
    sub.add_parser("overhead")
    sub.add_parser("scaling")
    sub.add_parser("estimate")
    sub.add_parser("suite")
    sub.add_parser("l2-psf-01")

    d2 = sub.add_parser("demo2")
    d2.add_argument("--preset", default="paper")
    d2.add_argument("--routes", default="rw_p")
    d2.add_argument("--reconstruction", default="ramp_sum")
    d2.add_argument("--backend", default="jax")
    d2.add_argument("--rays", type=float, required=True, help="ray budget, for cost per ray")
    d2.add_argument(
        "--label-suffix",
        default=None,
        help=(
            "distinguish this measurement from an earlier one of the SAME "
            "configuration, e.g. --label-suffix che118_after. Without it the second "
            "run overwrites the first and the baseline is gone."
        ),
    )

    d3 = sub.add_parser("demo3")
    d3.add_argument("--preset", default="characterization")
    d3.add_argument("--routes", default="rw_p")
    d3.add_argument("--reconstruction", default="ramp_sum")
    d3.add_argument("--backend", default="jax")
    d3.add_argument("--rays", type=float, required=True)
    d3.add_argument(
        "--label-suffix",
        default=None,
        help=(
            "distinguish this measurement from an earlier one of the SAME "
            "configuration, e.g. --label-suffix che118_after. Without it the second "
            "run overwrites the first and the baseline is gone."
        ),
    )

    args = parser.parse_args()
    started = time.perf_counter()
    if args.command == "compare":
        compare_records(args)
        return 0
    if args.command == "overhead":
        baseline_overhead()
    elif args.command == "scaling":
        baseline_scaling()
    elif args.command == "estimate":
        baseline_estimate()
    elif args.command == "suite":
        baseline_suite()
    elif args.command == "l2-psf-01":
        baseline_l2_psf_01()
    elif args.command == "demo2":
        baseline_demo2(args)
    elif args.command == "demo3":
        baseline_demo3(args)
    print(f"{args.command} finished in {time.perf_counter() - started:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
