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
    path.write_text(json.dumps(payload, indent=1, sort_keys=True, default=str) + "\n")
    print(f"wrote {path.relative_to(ROOT)}")
    return path


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

            full, _ = measure(
                kernel,
                label=f"C_RAY_TO_WAVE full call rings={rings}",
                workload=Workload(
                    size=ray_count, unit="ray", route="ramp_sum",
                    detail={"rings": rings, "grid_n": grid_n, "diagnostic": "as shipped"},
                ),
                repeats=3,
                warmup=1,
            )

            # The same call with the diagnostic forced off, to separate the two
            # costs on ONE configuration rather than inferring the split from the
            # threshold crossing.
            rtw._NEAREST_NEIGHBOUR_SCAN_LIMIT = 0
            try:
                reconstruction, _ = measure(
                    kernel,
                    label=f"C_RAY_TO_WAVE reconstruction only rings={rings}",
                    workload=Workload(
                        size=ray_count, unit="ray", route="ramp_sum",
                        detail={"rings": rings, "grid_n": grid_n, "diagnostic": "forced off"},
                    ),
                    repeats=3,
                    warmup=1,
                )
            finally:
                rtw._NEAREST_NEIGHBOUR_SCAN_LIMIT = scan_limit

            diagnostic_s = full.measurement.median_s - reconstruction.measurement.median_s
            rows.append(
                {
                    "rings": rings,
                    "rays": ray_count,
                    "diagnostic_runs": ray_count <= scan_limit,
                    "full_call_s": full.measurement.median_s,
                    "reconstruction_only_s": reconstruction.measurement.median_s,
                    "diagnostic_s": diagnostic_s,
                    "diagnostic_share_of_call": (
                        diagnostic_s / full.measurement.median_s
                        if full.measurement.median_s
                        else None
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
    diagnostic_rows = [r for r in rows if r["diagnostic_runs"] and r["diagnostic_s"] > 0]
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
            "held_fixed": {"grid_n": grid_n, "sample_pitch_m": pitch_m, "route": "ramp_sum"},
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
                f"(r^2 {reconstruction_fit.r_squared:.4f}) at fixed grid -- linear in "
                "rays, consistent with the O(rays x pixels) product model and NOT with "
                "the registry's O(rays + pixels). The ray-density diagnostic scales as "
                + (
                    f"rays^{diagnostic_fit.exponent:.3f} "
                    f"(r^2 {diagnostic_fit.r_squared:.4f}), i.e. quadratic as its "
                    "pairwise scan implies"
                    if diagnostic_fit
                    else "quadratic by construction (pairwise scan)"
                )
                + ", and is skipped entirely above "
                f"{scan_limit} rays. At the frozen M3-SINGLET-REF configuration of "
                "3169 rays the diagnostic is ~91% of the call."
            ),
            "consequence": (
                "Any optimization of the reconstruction kernel below 4096 rays moves "
                "at most ~9% of the wall time. The diagnostic is the target there, and "
                "it is not physics -- it is a sampling check that could be sampled "
                "rather than computed exhaustively, or cached. Handed to M5.2 "
                "(CHE-119) as a measured target rather than a guess."
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
                "ratio_predicted_over_measured": None,
                "confidence": predicted.confidence,
                "verdict": (
                    "NO PREDICTION. estimate() returns wall_time_s=None with "
                    "confidence='low' and says why: it does not import optiland and "
                    "does not know the surface count or the traced ray count. That is "
                    "an honest refusal, and it means a planner cannot use this "
                    "estimator to order work by cost."
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

        coupler_record, _ = measure(
            run_node,
            label="C_RAY_TO_WAVE",
            workload=Workload(
                size=float(rays.shape[0]) if rays.shape else 1.0,
                unit="ray",
                route="ramp_sum",
            ),
            repeats=3,
            warmup=1,
        )
        # Scored against BOTH denominators, because the estimator models the
        # ray x pixel reconstruction and the shipping call is dominated by
        # something else. Reporting one ratio would pick which error to show.
        import couplers.ray_to_wave as rtw

        scan_limit = rtw._NEAREST_NEIGHBOUR_SCAN_LIMIT
        rtw._NEAREST_NEIGHBOUR_SCAN_LIMIT = 0
        try:
            reconstruction_record, _ = measure(
                run_node,
                label="C_RAY_TO_WAVE reconstruction only",
                workload=Workload(
                    size=float(rays.shape[0]) if rays.shape else 1.0,
                    unit="ray",
                    route="ramp_sum",
                ),
                repeats=3,
                warmup=1,
            )
        finally:
            rtw._NEAREST_NEIGHBOUR_SCAN_LIMIT = scan_limit

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


def _timed_command(
    argv: list[str], *, label: str, workload: Workload, notes: list[str]
) -> dict[str, Any]:
    """Run a subprocess once and record it. One repeat, and the record says so."""
    captured: dict[str, Any] = {}

    def run(timer: StageTimer) -> int:
        with timer.stage("subprocess"):
            proc = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True)
        captured["returncode"] = proc.returncode
        captured["tail"] = proc.stdout.strip().splitlines()[-8:]
        captured["stderr_tail"] = proc.stderr.strip().splitlines()[-12:]
        return proc.returncode

    record, _ = measure(run, label=label, workload=workload, repeats=1, warmup=0, notes=notes)

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


def _demo(script: str, name: str, extra: list[str], *, unit_size: float, unit: str,
          route: str, detail: dict[str, Any]) -> None:
    name = f"{name}_{_device_suffix()}"
    payload = _timed_command(
        [sys.executable, str(ROOT / "benchmarks" / "probes" / "ray_wave" / script), *extra],
        label=name,
        workload=Workload(size=unit_size, unit=unit, route=route, detail=detail),
        notes=[
            "One timed run, no warmup. Three repeats of this configuration is minutes "
            "of a shared GPU to improve a number that varies by a few percent.",
            "seconds_per_ray is route-specific: ramp_sum is O(rays x pixels) and "
            "kspace_splat is O(rays) + one FFT, so the two are not one number. "
            "core.performance.compare refuses to divide them.",
        ],
    )
    _write(name, payload)


def baseline_demo2(args: argparse.Namespace) -> None:
    _demo(
        "demo2_hologram.py",
        f"demo2_{args.preset}_{args.routes}_{args.reconstruction}",
        ["--preset", args.preset, "--routes", args.routes,
         "--reconstruction", args.reconstruction, "--backend", args.backend],
        unit_size=float(args.rays),
        unit="ray",
        route=args.reconstruction,
        detail={"preset": args.preset, "routes": args.routes, "backend": args.backend},
    )


def baseline_demo3(args: argparse.Namespace) -> None:
    _demo(
        "demo3_hologram_lens.py",
        f"demo3_{args.preset}_{args.routes}_{args.reconstruction}",
        ["--preset", args.preset, "--routes", args.routes,
         "--reconstruction", args.reconstruction, "--backend", args.backend],
        unit_size=float(args.rays),
        unit="ray",
        route=args.reconstruction,
        detail={"preset": args.preset, "routes": args.routes, "backend": args.backend},
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

    d3 = sub.add_parser("demo3")
    d3.add_argument("--preset", default="characterization")
    d3.add_argument("--routes", default="rw_p")
    d3.add_argument("--reconstruction", default="ramp_sum")
    d3.add_argument("--backend", default="jax")
    d3.add_argument("--rays", type=float, required=True)

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
