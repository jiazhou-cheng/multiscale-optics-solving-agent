#!/usr/bin/env python3
"""B3 energy accounting at the handoff plane, and the runtime/memory envelopes.

CHE-116 (M4.1), two acceptance criteria answered by one set of runs.

The first is the criterion that says::

    Intermediate invariants are checked, not just the final result: energy
    accounting at the handoff plane, and the declared-versus-actual reference
    plane. A correct final image can hide an incorrect intermediate convention.

All three B3 families declare an energy-accounting intermediate and each one
gates on it:

===================  ==============================  =========  =========
family               invariant / metric              threshold  may_gate
===================  ==============================  =========  =========
``B3-PSF-SINGLET``   ``HANDOFF_ENERGY_CLOSES`` /     1e-3       True
                     ``handoff_power_ratio``
``B3-DEMO2``         ``PATCH_ENERGY_CLOSES`` /       1e-3       True
                     ``patch_handoff_power_ratio``
``B3-DUALROUTE``     ``route_power_ratio`` (the      1e-2       True
                     family's whole gate)
===================  ==============================  =========  =========

**None of the three is measurable from the shipping surface, and the reason is
one convention, not three bugs.** This probe measures it rather than asserting
it, on two real runs plus one committed record.

What the shipping surface exposes
---------------------------------
``couplers.ray_to_wave.ReconstructionReport`` carries exactly two power figures:

* ``reconstructed_discrete_power`` -- ``ComplexField.discrete_power()``, which is
  ``sum(|u|^2) * dy * dx``: an integral over the output plane, so it carries an
  area;
* ``incident_amplitude_power_sum`` -- ``sum(|amplitude|^2)`` over rays, a bare
  sum with no area element in it.

And ``couplers/handoff.py`` declares the amplitude convention as
``amplitude = sqrt(weight) * quadrature_weight_m2``: the per-ray area element
multiplies the *field*, because the wavelet sum approximates a surface integral
and a quadrature weight is the area element that integral discretizes with. So
``|amplitude|^2 = weight * quadrature_weight_m2^2``, and

    reconstructed_discrete_power / incident_amplitude_power_sum

is not a dimensionless closure ratio: the numerator carries an area the
denominator does not, so the two are incommensurable and reading their quotient
against a ``1e-3`` gate compares different physical quantities. Measured:
``2.8e-5`` on the singlet, ``1.2e-8`` on the Cooke triplet, ``2.1e-5`` on
demo2's first streaming chunk.

The **ray-count scaling arm** -- the same Cooke triplet configuration at 32 and
64 hexapolar rings -- is here to make the dimensional claim a measurement rather
than an argument, and it does: ``sum(|amplitude|)`` is invariant to 0.02% across
the two rungs while ``sum(|amplitude|^2)`` falls by very nearly the ray-count
factor. That is the squared area element in the denominator, seen directly.

The arm also shows something the probe reports and deliberately does not build
on: on this configuration the quotient sits within 0.5% of the plain grid area
``N_pix * dy * dx`` at 32 rings, because the reconstruction is dominated by a
near-uniform floor rather than by the focal spot -- every ray splats a full-grid
ramp, and the measured border energy fraction sits near the value a flat field
would give. **An earlier draft of this probe read the 10.9% drift between the
rungs as ruling out a missing calibration constant. That inference was not
supported by these measurements** -- two points on a floor-dominated grid cannot
decide it, and the drift is consistent with the focal spot's share rising above a
floor falling as 1/N -- **it was caught in independent review, and it is
withdrawn.** The dimensional finding does not need it.

Why this probe does not fix it
------------------------------
Forming a real closure needs the incident power ``sum(weight_i *
quadrature_weight_m2_i)`` -- intensity times area, once -- and then an argument
about what the coherent wavelet sum's cross terms do to it. That is a
conservation claim across a representation boundary: it needs a derivation, an
oracle that is not the coupler, and independent review. M4.1's scope is three
B3 families with decidable oracles and the deletion of the last bespoke entry
point; deriving the ray-to-wave absolute-power normalization is not in it, and
inventing a normalization that makes the number read 1.0 would be the
fabrication the whole verification layer exists to prevent.

``benchmarks/instances/b3_psf_singlet.py`` reached the same conclusion for the
singlet alone and recorded it in the measurement note ("nothing here measures
the traced bundle's power, so it is reported without a tolerance rather than
gated on the wrong quantity"). This probe establishes that it is a property of
the boundary shared by all three families, not a gap in one driver.

The second criterion
--------------------
    Runtime and memory envelopes recorded per family, and every case fits one
    GPU per the shared-host policy.

Each case below is timed and watched by ``core.resources.MemoryWatchdog``, and
the observed wall time and peak RSS are compared against the family's own
``ExecutionPolicy.max_wall_seconds`` / ``max_peak_memory_gib``. ``B3-DEMO2``'s
envelope is read off its committed GPU runs rather than re-run: it is a 95-second
1.6e8-ray job on one A6000 whose numbers are already committed evidence, and
re-running it here would restamp CHE-96/CHE-101 records this issue has no reason
to move.

Run it::

    ./run.sh python benchmarks/probes/b3_energy_accounting.py --write
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from core.execution import RunStatus
from core.paths import repository_root
from core.provenance import record_provenance
from core.resources import MemoryWatchdog

ROOT = repository_root()
RECORD_PATH = ROOT / "benchmarks" / "probes" / "records" / "b3_energy_accounting.json"

#: The committed demo2 record whose envelope and power figures are read rather
#: than re-measured. Its ``rw_p`` route is the Table S2 budget the B3-DEMO2
#: family's gate disposition quotes.
DEMO2_RECORD = ROOT / "benchmarks" / "probes" / "records" / "ray_wave" / "demo2_paper_jax.json"
DEMO2_PERF_RECORD = (
    ROOT / "benchmarks" / "perf" / "records" / "demo2_paper_rw_p_ramp_sum_cuda.json"
)

#: The Cooke triplet field the FIXED-V1 instance ``B3-DUALROUTE-01`` declares:
#: 20 degrees, which is ``Hy = 1.0`` on this prescription (max field 20 deg).
DUALROUTE_HY = 1.0
#: 64 hexapolar rings -- the ``pupil_rings`` default the family declares and the
#: value ``B3-DUALROUTE-01`` freezes. Not PB7's 128: this probe measures a power
#: bookkeeping identity, which does not move with ray count, and the cheaper
#: rung keeps the run inside the CPU envelope.
DUALROUTE_RINGS = 64
#: The second rung of the ray-count scaling arm. A quadrature closure is
#: invariant under refining the quadrature, so the same configuration at two ring
#: counts is what turns "the number is not 1.0" into "the number is not a
#: closure". 32 and 64 rings are 0.7 s each on CPU.
DUALROUTE_SCALING_RINGS = 32


def _dimensional_reading(recon_power: float, incident_sum: float) -> dict[str, Any]:
    """The two power figures the shipping report exposes, and their quotient.

    No derived length or corrected closure is reported. The quotient is stated as
    what it is -- the number a caller reading these two fields as a conservation
    ratio would get -- together with why it is not one. Claiming a specific
    dimensional form would be deriving the normalization this probe explicitly
    declines to derive.
    """
    ratio = recon_power / incident_sum if incident_sum else float("nan")
    return {
        "reconstructed_discrete_power": recon_power,
        "incident_amplitude_power_sum": incident_sum,
        "ratio": ratio,
        "closure_residual_if_read_as_a_ratio": abs(1.0 - ratio) if np.isfinite(ratio) else None,
        "verdict": (
            "NOT a closure. reconstructed_discrete_power is "
            "ComplexField.discrete_power() = sum(|u|^2) * dy * dx, an integral over "
            "the output plane; incident_amplitude_power_sum is sum(|amplitude|^2) "
            "over rays, a bare sum, with amplitude = sqrt(weight) * "
            "quadrature_weight_m2 (couplers/handoff.py) putting the per-ray area "
            "element inside the field. The two are incommensurable, so their "
            "quotient is not the dimensionless ratio the declared tolerance is "
            "written against."
        ),
    }


# ---------------------------------------------------------------------------
# B3-PSF-SINGLET -- the frozen graph, through the executor
# ---------------------------------------------------------------------------


def measure_singlet() -> dict[str, Any]:
    """The committed graph document, run by the executor, powers read off the edge.

    The same graph and the same driver entry point the family's canonical
    instance uses -- ``benchmarks/instances/b3_psf_singlet.py`` -- so this is the
    envelope of the configuration the gate is defined on, not of a cheaper proxy.
    """
    from couplers.ray_to_wave import ray_to_wave
    from verification.families.b3_composed import B3_PSF_SINGLET

    sys.path.insert(0, str(ROOT / "benchmarks" / "instances"))
    import b3_psf_singlet as singlet

    reports: list[Any] = []
    original = ray_to_wave

    def capture(*args: Any, **kwargs: Any) -> Any:
        field, report = original(*args, **kwargs)
        reports.append(report)
        return field, report

    # The powers are on the ReconstructionReport, which the executor consumes and
    # does not put on the record. Wrapping the shipping call is how this probe
    # reads them without editing the coupler to expose them for one probe's
    # benefit -- an ExecutionRecord field for this belongs to whoever settles the
    # closure question, not here.
    import couplers.node as coupler_node

    patched = hasattr(coupler_node, "ray_to_wave")
    if patched:
        coupler_node.ray_to_wave = capture  # type: ignore[assignment]
    try:
        watchdog = MemoryWatchdog(interval_s=0.25).start()
        started = time.perf_counter()
        record = singlet.execute(singlet.load_graph(), singlet.canonical_instance(), seed=1)
        wall_s = time.perf_counter() - started
        watchdog.stop()
    finally:
        if patched:
            coupler_node.ray_to_wave = original  # type: ignore[assignment]

    payload: dict[str, Any] = {
        "family_id": "B3-PSF-SINGLET",
        "instance_id": "B3-PSF-SINGLET-01",
        "invariant": "HANDOFF_ENERGY_CLOSES",
        "metric": "handoff_power_ratio",
        "declared_tolerance": B3_PSF_SINGLET.invariants[0].tolerance.threshold,
        "how_it_ran": "GraphExecutor over examples/graphs/psf_singlet_sensor.yaml",
        "status": record.status.value,
        "envelope": _envelope(B3_PSF_SINGLET, wall_s, watchdog, device="cpu"),
    }
    if record.status is not RunStatus.SUCCEEDED:
        payload["error"] = next((n.error_message for n in record.nodes if n.error_message), None)
        return payload
    if not reports:
        payload["energy_accounting"] = {
            "verdict": (
                "NOT MEASURED: the reconstruction report was not observable through "
                "couplers.node, so this probe declines to report a number rather than "
                "estimating one from the artifacts."
            )
        }
        return payload
    report = reports[-1]
    payload["ray_count"] = int(report.ray_count)
    payload["energy_accounting"] = _dimensional_reading(
        float(report.reconstructed_discrete_power),
        float(report.incident_amplitude_power_sum),
    )
    payload["sample_pitch_m"] = [float(v) for v in report.sample_pitch_m]
    return payload


# ---------------------------------------------------------------------------
# B3-DUALROUTE -- the Cooke triplet at 20 degrees, the route the family gates on
# ---------------------------------------------------------------------------


def measure_dualroute(rings: int = DUALROUTE_RINGS) -> dict[str, Any]:
    """``route = ray_to_wave`` on the Cooke triplet at 20 degrees.

    The two Optiland routes are not run: their PSFs are peak-normalized arrays
    with no absolute scale (``benchmarks/probes/cooke_triplet_psf_routes.py``
    records their native peak as "UNCALIBRATED ... not comparable"), so a power
    ratio for them would be a normalization artefact rather than a measurement.
    The route the family's ``route`` parameter defaults to is the one with an
    absolute power figure at all, and it is the one measured here.
    """
    import importlib.util

    from couplers.handoff import DeclaredHandoffPlane, declare_coherent_bundle
    from couplers.ray_to_wave import ray_to_wave
    from solvers.base import ModelRunRequest
    from solvers.optiland.adapter import get_adapter as optiland
    from verification.families.b3_composed import B3_DUALROUTE

    spec = importlib.util.spec_from_file_location(
        "che116_pb7", ROOT / "benchmarks" / "probes" / "cooke_triplet_psf_routes.py"
    )
    assert spec is not None and spec.loader is not None
    pb7 = importlib.util.module_from_spec(spec)
    sys.modules["che116_pb7"] = pb7
    spec.loader.exec_module(pb7)

    watchdog = MemoryWatchdog(interval_s=0.25).start()
    started = time.perf_counter()
    with tempfile.TemporaryDirectory() as directory:
        workdir = Path(directory)
        trace = optiland().run(
            ModelRunRequest(
                run_id="che116",
                node_id="lens",
                config={
                    "prescription": pb7.cooke_triplet_spec(),
                    "num_rays": rings,
                    "wavelength": pb7.WAVELENGTH_UM,
                    "Hx": 0.0,
                    "Hy": DUALROUTE_HY,
                    "handoff_plane": "exit_pupil",
                    "output_directory": str(workdir / "rays"),
                },
            )
        )
        if trace.status.value != "succeeded":
            wall_s = time.perf_counter() - started
            watchdog.stop()
            return {
                "family_id": "B3-DUALROUTE",
                "instance_id": "B3-DUALROUTE-01",
                "status": trace.status.value,
                "error": f"{trace.error_type}: {trace.error_message}",
                "envelope": _envelope(B3_DUALROUTE, wall_s, watchdog, device="cpu"),
            }
        rays = trace.outputs["rays"]
        conventions = (rays.metadata or {}).get("conventions", {})
        pupil_z_m = float(conventions["exit_pupil"]["z_m"])
        handoff = declare_coherent_bundle(
            rays, declared_plane=DeclaredHandoffPlane("exit_pupil", pupil_z_m)
        )
        at_sensor, _ = pb7.advance_bundle_to_z(handoff.bundle, pb7.IMAGE_Z_MM * 1e-3)
        positions = np.asarray(at_sensor.positions_m)
        # Centre on the bundle's own centroid rather than on a separately traced
        # chief ray: this probe measures a power identity, and the centring only
        # decides which part of the grid the spot lands on.
        recentred = pb7.translate_bundle_transverse(
            at_sensor, float(np.mean(positions[:, 1])), float(np.mean(positions[:, 0]))
        )
        field, report = ray_to_wave(
            recentred,
            grid_shape=(pb7.SENSOR_GRID_N, pb7.SENSOR_GRID_N),
            sample_pitch_m=(pb7.SENSOR_PITCH_M, pb7.SENSOR_PITCH_M),
        )
        # sum(|amplitude|) beside sum(|amplitude|^2). The pair is what separates
        # "the denominator squares an area element" -- the abs sum is invariant
        # under refinement, the squared sum falls with the ray count -- from any
        # statement about the reconstruction's fidelity.
        amplitudes = np.abs(np.asarray(recentred.amplitude, dtype=np.complex128))
        incident_abs_sum = float(np.sum(amplitudes))
        intensity = np.abs(np.asarray(field.u, dtype=np.complex128)) ** 2
        total = float(intensity.sum())
        border = float(
            intensity[0, :].sum()
            + intensity[-1, :].sum()
            + intensity[:, 0].sum()
            + intensity[:, -1].sum()
        )
    wall_s = time.perf_counter() - started
    watchdog.stop()
    grid_area = pb7.SENSOR_GRID_N**2 * pb7.SENSOR_PITCH_M**2

    return {
        "family_id": "B3-DUALROUTE",
        "instance_id": "B3-DUALROUTE-01",
        "invariant": "route_power_ratio (the family's gating metric)",
        "metric": "route_power_ratio",
        "declared_tolerance": B3_DUALROUTE.tolerance_for("route_power_ratio").threshold,
        "how_it_ran": (
            "shipping calls only: Optiland adapter -> declare_coherent_bundle -> "
            "ray_to_wave, the same sequence benchmarks/probes/"
            "cooke_triplet_psf_routes.py::method_c_ray_to_wave uses"
        ),
        "status": "succeeded",
        "field_angle_deg": 20.0,
        "hexapolar_rings": rings,
        "ray_count": int(report.ray_count),
        "quadrature_weight_status": handoff.diagnostics.get("quadrature_weight_status"),
        "sample_pitch_m": [float(v) for v in report.sample_pitch_m],
        "incident_amplitude_abs_sum": incident_abs_sum,
        "grid_area_m2": grid_area,
        "ratio_over_grid_area": (
            float(report.reconstructed_discrete_power)
            / float(report.incident_amplitude_power_sum)
            / grid_area
        ),
        # A flat field over an N x N grid puts (4N - 4) / N^2 of its power on the
        # border. Measuring the border fraction is how "the reconstruction is
        # floor-dominated here" becomes a number instead of an assertion.
        "border_energy_fraction": (border / total if total else None),
        "border_energy_fraction_if_field_were_flat": (
            (4 * pb7.SENSOR_GRID_N - 4) / pb7.SENSOR_GRID_N**2
        ),
        "ray_density_status": report.ray_density_status,
        "grid_nyquist_satisfied": bool(report.grid_nyquist_satisfied),
        "max_adjacent_ray_phase_rad": report.max_adjacent_ray_phase_rad,
        "energy_accounting": _dimensional_reading(
            float(report.reconstructed_discrete_power),
            float(report.incident_amplitude_power_sum),
        ),
        "envelope": _envelope(B3_DUALROUTE, wall_s, watchdog, device="cpu"),
    }


# ---------------------------------------------------------------------------
# B3-DEMO2 -- read off the committed GPU runs, not re-run
# ---------------------------------------------------------------------------


def read_demo2() -> dict[str, Any]:
    from verification.families.b3_composed import B3_DEMO2

    payload: dict[str, Any] = {
        "family_id": "B3-DEMO2",
        "instance_id": "B3-DEMO2-01",
        "invariant": "PATCH_ENERGY_CLOSES",
        "metric": "patch_handoff_power_ratio",
        "declared_tolerance": B3_DEMO2.invariants[0].tolerance.threshold,
        "how_it_ran": (
            "NOT re-run. Read off the committed GPU records named below: a 1.6e8-ray "
            "job on one A6000 whose numbers are committed CHE-96/CHE-101 evidence, "
            "and re-running it here would restamp records this issue has no reason "
            "to move."
        ),
        "source_records": [
            DEMO2_RECORD.relative_to(ROOT).as_posix(),
            DEMO2_PERF_RECORD.relative_to(ROOT).as_posix(),
        ],
    }
    if not DEMO2_RECORD.is_file():
        payload["status"] = "record_absent"
        return payload
    science = json.loads(DEMO2_RECORD.read_text())
    route = science["routes"]["rw_p"]
    chunk = route["streaming"]["first_chunk_reconstruction"]
    payload["status"] = "read_from_record"
    payload["ray_count"] = int(route["total_rays"])
    payload["chunk_count"] = int(route["streaming"]["chunk_count"])
    payload["sample_pitch_m"] = [float(v) for v in chunk["sample_pitch_m"]]
    payload["energy_accounting"] = _dimensional_reading(
        float(chunk["reconstructed_discrete_power"]),
        float(chunk["incident_amplitude_power_sum"]),
    )
    payload["energy_accounting"]["scope"] = (
        "the FIRST of 40 streaming chunks, which is the only reconstruction report "
        "the committed record carries. The accumulator's total is not reported, so "
        "even a corrected closure would need the probe to emit it -- which is a "
        "second reason this invariant has never been checked."
    )
    envelope: dict[str, Any] = {
        "declared_max_wall_seconds": B3_DEMO2.execution_policy.max_wall_seconds,
        "declared_max_peak_memory_gib": B3_DEMO2.execution_policy.max_peak_memory_gib,
        "observed_wall_seconds": float(route["wall_clock_s"]),
        "observed_device": route["actual"]["field_device"],
        "observed_dtype": route["actual"]["field_dtype"],
        "gpu_count_used": 1,
    }
    if DEMO2_PERF_RECORD.is_file():
        perf = json.loads(DEMO2_PERF_RECORD.read_text())
        memory = perf.get("memory_report", {})
        envelope["harness_wall_seconds"] = perf["measurement"]["median_s"]
        # `memory_report.peak_rss_bytes` is the HARNESS process, which times a
        # subprocess and therefore never samples the run: it reads 0.027 GiB.
        # Feeding that to the envelope check would manufacture a passing verdict
        # from a number the record itself declares irrelevant. The subprocess peak
        # is what this case cost, and it is in the same file.
        child = (perf.get("subprocess") or {}).get("peak_child_rss_bytes")
        envelope["observed_peak_rss_gib"] = _gib(child)
        envelope["observed_peak_rss_source"] = (
            "subprocess.peak_child_rss_bytes -- the sampled peak of the child that "
            "actually ran demo2"
            if child is not None
            else "UNMEASURED: this record carries no subprocess peak, and "
            "memory_report.peak_rss_bytes is the harness process (0.027 GiB), not "
            "the run. Reported as unmeasured rather than as a passing verdict."
        )
        envelope["harness_process_peak_rss_gib"] = _gib(memory.get("peak_rss_bytes"))
        envelope["peak_cgroup_swap_bytes"] = memory.get("peak_cgroup_swap_bytes")
        envelope["gpu_name"] = perf.get("environment", {}).get("gpu_name")
        envelope["peak_rss_caveat"] = (
            "HOST RSS, not device memory. The declared 40 GiB envelope was sized "
            "against the 40-chunk plan's working set; nothing in either committed "
            "record reports CUDA peak bytes for this run (the perf record's `cuda` "
            "field is null), so the device side of this envelope is UNMEASURED. The "
            "29.213 GB and 8.086 GB figures in the science record are the PAPER's "
            "Table S2 numbers for its own implementation, not ours, and are not "
            "used here."
        )
    envelope |= _fits(
        B3_DEMO2.execution_policy.max_wall_seconds,
        envelope["observed_wall_seconds"],
        B3_DEMO2.execution_policy.max_peak_memory_gib,
        envelope.get("observed_peak_rss_gib"),
        gpu_count=int(envelope["gpu_count_used"]),
    )
    payload["envelope"] = envelope
    return payload


# ---------------------------------------------------------------------------
# Envelope bookkeeping
# ---------------------------------------------------------------------------


def _gib(value: Any) -> float | None:
    return None if value is None else round(float(value) / 2**30, 6)


def _fits(
    max_wall_s: float | None,
    wall_s: float | None,
    max_gib: float | None,
    observed_gib: float | None,
    *,
    gpu_count: int,
) -> dict[str, Any]:
    """Declared against observed, as a verdict rather than two numbers side by side."""
    wall_ok = None if (max_wall_s is None or wall_s is None) else wall_s <= max_wall_s
    memory_ok = None if (max_gib is None or observed_gib is None) else observed_gib <= max_gib
    return {
        "wall_inside_declared_envelope": wall_ok,
        # ``None``, not ``False`` and certainly not ``True``, when the observed
        # value is missing. A verdict computed from an absent measurement is the
        # failure mode this whole record exists to avoid.
        "memory_inside_declared_envelope": memory_ok,
        # Read off ``gpu_count_used``, which each case sets from what it actually
        # ran on. Hardcoding True here would have made this key say nothing.
        "devices_used": gpu_count,
        "fits_one_gpu": gpu_count <= 1,
        "fits_one_gpu_basis": (
            "the two CPU cases use no GPU at all, and demo2 ran on one A6000 with "
            "its 1.6e8 rays deliberately chunked 40 ways. No case in this family "
            "set requires a second device."
        ),
    }


def _envelope(
    family: Any, wall_s: float, watchdog: MemoryWatchdog, *, device: str
) -> dict[str, Any]:
    report = watchdog.report()
    observed_gib = _gib(report.get("peak_rss_bytes"))
    policy = family.execution_policy
    envelope: dict[str, Any] = {
        "declared_max_wall_seconds": policy.max_wall_seconds,
        "declared_max_peak_memory_gib": policy.max_peak_memory_gib,
        "observed_wall_seconds": wall_s,
        "observed_peak_rss_gib": observed_gib,
        "observed_device": device,
        "peak_cgroup_swap_bytes": report.get("peak_cgroup_swap_bytes"),
        "swap_delta_peak_bytes": report.get("swap_delta_peak_bytes"),
        "min_mem_available_bytes": report.get("min_mem_available_bytes"),
        "watchdog_verdict": report.get("verdict"),
        "samples": report.get("samples"),
        "gpu_count_used": 0,
    }
    envelope |= _fits(
        policy.max_wall_seconds,
        wall_s,
        policy.max_peak_memory_gib,
        observed_gib,
        gpu_count=0,
    )
    return envelope


def ray_count_scaling_arm() -> dict[str, Any]:
    """The same Cooke triplet configuration at two hexapolar ring counts.

    A quadrature closure does not move when the quadrature is refined. This arm
    is what makes the finding decisive rather than suggestive: "the quotient is
    2.8e-5 instead of 1.0" is consistent with a missing constant factor, and a
    constant factor could be absorbed. A quotient that TRACKS the ray count
    cannot be, because no fixed normalization would make both rungs read 1.0.

    Reported as a measurement with no fitted exponent. Two points do not
    establish a power law and this probe does not claim one.
    """
    rungs = []
    for rings in (DUALROUTE_SCALING_RINGS, DUALROUTE_RINGS):
        case = measure_dualroute(rings)
        energy = case.get("energy_accounting", {})
        rungs.append(
            {
                "hexapolar_rings": rings,
                "ray_count": case.get("ray_count"),
                "ratio": energy.get("ratio"),
                "reconstructed_discrete_power": energy.get("reconstructed_discrete_power"),
                "incident_amplitude_power_sum": energy.get("incident_amplitude_power_sum"),
                "incident_amplitude_abs_sum": case.get("incident_amplitude_abs_sum"),
                "grid_area_m2": case.get("grid_area_m2"),
                "ratio_over_grid_area": case.get("ratio_over_grid_area"),
                "border_energy_fraction": case.get("border_energy_fraction"),
                "border_energy_fraction_if_field_were_flat": case.get(
                    "border_energy_fraction_if_field_were_flat"
                ),
                "ray_density_status": case.get("ray_density_status"),
                "grid_nyquist_satisfied": case.get("grid_nyquist_satisfied"),
                "max_adjacent_ray_phase_rad": case.get("max_adjacent_ray_phase_rad"),
                "status": case.get("status"),
            }
        )
    ratios = [r["ratio"] for r in rungs if r["ratio"] is not None]
    counts = [r["ray_count"] for r in rungs if r["ray_count"] is not None]
    abs_sums = [r["incident_amplitude_abs_sum"] for r in rungs if r["incident_amplitude_abs_sum"]]
    payload: dict[str, Any] = {
        "configuration": (
            "B3-DUALROUTE-01's system and field angle, route = ray_to_wave, at two "
            "hexapolar ring counts. Nothing else changes."
        ),
        "rungs": rungs,
    }
    if len(ratios) == 2 and len(counts) == 2 and ratios[0]:
        payload["ratio_change_factor"] = ratios[1] / ratios[0]
        payload["ray_count_change_factor"] = counts[1] / counts[0]
        payload["squared_sum_change_factor"] = (
            rungs[1]["incident_amplitude_power_sum"] / rungs[0]["incident_amplitude_power_sum"]
        )
        if len(abs_sums) == 2:
            payload["abs_sum_change_factor"] = abs_sums[1] / abs_sums[0]
        payload["verdict"] = (
            "WHAT THIS ARM SHOWS, AND WHAT IT DOES NOT -- read before quoting it. "
            "SHOWS: sum(|amplitude|) is essentially INVARIANT under refinement "
            "while sum(|amplitude|^2) falls by roughly the ray-count factor "
            "(abs_sum_change_factor against squared_sum_change_factor). That is "
            "the squared area element in the denominator and nothing else, and it "
            "is the dimensional defect, measured. SHOWS ALSO: on THIS "
            "configuration the quotient is close to the plain grid area "
            "N_pix * dy * dx -- see ratio_over_grid_area -- because the "
            "reconstruction is dominated by a near-uniform floor rather than by "
            "the focal spot: every ray splats a full-grid ramp, and "
            "border_energy_fraction sits near what a flat field would give. "
            "DOES NOT SHOW: that no per-configuration normalization could close "
            "the quantity. Two points on a floor-dominated grid cannot decide "
            "that, and the drift between the rungs is consistent with the focal "
            "spot's share rising above a floor that falls as 1/N. An earlier draft "
            "of this record read the drift as ruling out a calibration constant. "
            "That inference was not supported by these measurements, was caught in "
            "independent review, and is withdrawn. The dimensional finding stands "
            "on its own and does not need it."
        )
    return payload


def characterize() -> dict[str, Any]:
    cases = [measure_singlet(), measure_dualroute(), read_demo2()]
    scaling = ray_count_scaling_arm()
    ratios = {
        case["family_id"]: case.get("energy_accounting", {}).get("ratio")
        for case in cases
        if "energy_accounting" in case
    }
    measured = [r for r in ratios.values() if r is not None and np.isfinite(r)]
    return {
        "probe": "b3_energy_accounting",
        "issue": "CHE-116 (M4.1)",
        "question": (
            "do the three B3 families' declared energy-accounting intermediates "
            "close, and do the three cases fit their declared runtime and memory "
            "envelopes on one GPU?"
        ),
        "cases": cases,
        "ray_count_scaling_arm": scaling,
        "finding": {
            "ratios_by_family": ratios,
            "all_far_from_unity": bool(measured) and all(r < 1e-3 for r in measured),
            "statement": (
                "The three declared energy-accounting intermediates share one root "
                "cause and none of them is a closure as written. "
                "reconstructed_discrete_power is an integral over the output plane "
                "(sum(|u|^2) * dy * dx); incident_amplitude_power_sum is a bare sum "
                "over rays of |amplitude|^2, and couplers/handoff.py puts the per-ray "
                "area element INSIDE the amplitude (amplitude = sqrt(weight) * "
                "quadrature_weight_m2). The two are incommensurable, so their "
                "quotient is not the dimensionless ratio the tolerances are written "
                "against: it reads 1.2e-8 to 2.8e-5 against gates of 1e-2 and 1e-3. "
                "The ray-count scaling arm shows the dimensional defect directly: "
                "sum(|amplitude|) is invariant to 0.02% across a 3.94x refinement "
                "while sum(|amplitude|^2) falls by nearly the same factor. Whether "
                "some per-configuration normalization would nonetheless close the "
                "quantity is UNTESTED and this record does not claim otherwise -- on "
                "the arm's configuration the quotient sits within 0.5% of the plain "
                "grid area, so the reconstruction is floor-dominated there and two "
                "such points decide nothing about a constant. Forming the right "
                "quantity needs the "
                "incident power sum(weight_i * quadrature_weight_m2_i) plus an "
                "argument about what the coherent wavelet sum's cross terms do to it: "
                "a conservation claim across a representation boundary, which owes a "
                "derivation, an oracle that is not the coupler, and independent "
                "review. This probe reports the open gate rather than widening it, "
                "re-deriving it, or quietly dropping may_gate to False."
            ),
            "consequence_for_the_families": (
                "HANDOFF_ENERGY_CLOSES, PATCH_ENERGY_CLOSES and B3-DUALROUTE's whole "
                "route_power_ratio gate stay UNMEASURED, declared, and un-widened. "
                "B3-DUALROUTE's gate_disposition therefore remains NOT_MEASURED -- the "
                "reason has moved from 'PB7 did not measure it' to 'the quantity is "
                "not formable from the shipping surface', which is a sharper open "
                "question and a worse thing to have left implicit."
            ),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write", action="store_true", help=f"write {RECORD_PATH.relative_to(ROOT)}"
    )
    args = parser.parse_args(argv)

    record = characterize()
    for case in record["cases"]:
        energy = case.get("energy_accounting", {})
        envelope = case.get("envelope", {})
        print(f"== {case['family_id']}  ({case.get('status')})")
        if "ratio" in energy:
            print(
                f"   {case['metric']}: ratio {energy['ratio']:.6e} against a "
                f"declared {case['declared_tolerance']:.1e}"
            )
        print(
            f"   wall {envelope.get('observed_wall_seconds')} s / "
            f"{envelope.get('declared_max_wall_seconds')} s declared -> "
            f"{envelope.get('wall_inside_declared_envelope')};  "
            f"peak rss {envelope.get('observed_peak_rss_gib')} GiB / "
            f"{envelope.get('declared_max_peak_memory_gib')} GiB declared -> "
            f"{envelope.get('memory_inside_declared_envelope')}"
        )
    arm = record["ray_count_scaling_arm"]
    print("== ray-count scaling arm (B3-DUALROUTE's system, route = ray_to_wave)")
    for rung in arm["rungs"]:
        print(
            f"   {rung['hexapolar_rings']:>4} rings, {rung['ray_count']} rays: "
            f"ratio {rung['ratio']:.6e}"
        )
    if "ratio_change_factor" in arm:
        print(
            f"   rays x{arm['ray_count_change_factor']:.3f}: "
            f"sum|a| x{arm.get('abs_sum_change_factor', float('nan')):.5f}, "
            f"sum|a|^2 x{arm['squared_sum_change_factor']:.4f}, "
            f"quotient x{arm['ratio_change_factor']:.4f}"
        )
        for rung in arm["rungs"]:
            over = rung.get("ratio_over_grid_area")
            if over is not None:
                print(
                    f"     {rung['hexapolar_rings']:>4} rings: quotient / grid area "
                    f"= {over:.4f};  border energy {rung['border_energy_fraction']:.2e} "
                    f"(flat field would give "
                    f"{rung.get('border_energy_fraction_if_field_were_flat', 0.0):.2e})"
                )
        print(f"   {arm['verdict']}")
    print()
    print(record["finding"]["statement"])

    if args.write:
        record["record_provenance"] = record_provenance(
            probe="probes/b3_energy_accounting",
            root=ROOT,
            data_inputs=[DEMO2_RECORD, DEMO2_PERF_RECORD],
        )
        RECORD_PATH.parent.mkdir(parents=True, exist_ok=True)
        RECORD_PATH.write_text(
            json.dumps(record, indent=2, sort_keys=True, default=float) + "\n"
        )
        print(f"wrote {RECORD_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
