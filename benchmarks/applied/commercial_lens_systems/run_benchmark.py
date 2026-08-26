#!/usr/bin/env python
"""Build and characterize commercial catalog lens systems (CHE-139 / M1.1.5).

    MOA_GPUS=device=6 ./run.sh --gpu python \
        benchmarks/applied/commercial_lens_systems/run_benchmark.py run \
        --device cuda --backend torch --dtype float64 --tag gpu

    ./run.sh python benchmarks/applied/commercial_lens_systems/run_benchmark.py run \
        --device cpu --backend numpy --dtype float64 --tag cpu \
        --systems S4_PAC052_KBX058_TANDEM

    ./run.sh python benchmarks/applied/commercial_lens_systems/run_benchmark.py compare \
        --primary records/S4_PAC052_KBX058_TANDEM.gpu.json \
        --secondary records/S4_PAC052_KBX058_TANDEM.cpu.json

What this script is allowed to do
--------------------------------
It reads catalog components from ``catalog_sources``, installs them per
``benchmark_systems``, asks ``core.optical_assembly`` for one
``OpticalSystemSpec``, and hands that spec to the **shipping** Optiland adapter
through an ordinary ``ModelRunRequest``. It constructs no ``Optic`` of its own
except through ``solvers.optiland.builder.build_optiland_system`` -- the same
single builder every prescription in this repository goes through -- and it
contains no optical physics: every number it reports is either a manufacturer
value, an Optiland output, or an explicitly named arithmetic combination of ray
coordinates the adapter already exported.

Four questions, kept apart
--------------------------
Each record has four blocks answering the four questions of the ticket
separately, because collapsing them is how a benchmark ends up reporting
"passed" for a system that built wrong and traced fine:

``construction``  A -- could the components be built from published data?
``assembly``      B -- did ordering, orientation and spacing reach the built system?
``execution``     C -- did the trace run, where, in what precision, losing what?
``characterization``  D -- does the result behave plausibly, against what?

Nothing here decides a pass. There is no tolerance and no verdict: this is an
applied characterization benchmark, and the comparisons it reports (catalog EFL
and BFL, the reversed-achromat control, GPU against CPU) are stated with their
bases so a reader judges them. The one thing it does refuse is executing at all
when the requested device is unavailable -- see :func:`_require_execution`.

Paraxial quantities are measured on the numpy backend at float64 regardless of
what the traces run on, because they are properties of the *prescription* rather
than of the trace, and mixing the two would make a GPU record's catalog
comparison depend on the GPU. Optiland's backend is process-global and not
thread-safe, so the order matters and is fixed: per system, all paraxial reads
first, then all traces.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
for _path in (str(_ROOT / "src"), str(_HERE)):
    if _path not in sys.path:  # pragma: no cover - import-path bootstrap
        sys.path.insert(0, _path)

import numpy as np  # noqa: E402
from benchmark_systems import (  # noqa: E402
    GPU_CPU_COMPARISON_METRICS,
    SYSTEM_KEYS,
    ImagePlaneRule,
    SystemDefinition,
    SystemRole,
    resolve_system,
)
from catalog_sources import catalog_components, require_supported  # noqa: E402

from core.errors import (  # noqa: E402
    AdapterDependencyError,
    UnsupportedCapabilityError,
)
from core.optical_assembly import assemble_optical_system, surface_table  # noqa: E402
from core.optical_system import (  # noqa: E402
    OpticalSystemSpec,
    PrescriptionError,
)
from core.performance import environment_fingerprint  # noqa: E402
from solvers.base import ModelRunRequest, RunStatus  # noqa: E402
from solvers.optiland.adapter import OptilandAdapter  # noqa: E402
from solvers.optiland.builder import build_optiland_system  # noqa: E402
from solvers.optiland.cost_model import hexapolar_ray_count  # noqa: E402

RECORDS = _HERE / "records"
FIGURES = _HERE / "figures"

#: mm per metre. The adapter exports positions in SI; every length this
#: benchmark reports is in the schema's millimetre unit, so the conversion
#: happens once, here, with its factor named.
_MM_PER_M = 1.0e3

#: The two placeholder image distances used to prove that the paraxial back
#: focus of an assembly does not depend on where the image plane was parked
#: while it was measured. Deliberately far apart and deliberately not near any
#: answer.
_PARAXIAL_PROBE_DISTANCES_MM = (10.0, 200.0)

#: How closely the two probes must agree, in mm. This is not a physics tolerance:
#: the paraxial back focus is `placeholder + F2` and F2 is signed from the image
#: surface, so the two probes must agree to floating-point round-off on a ~1e2 mm
#: quantity or the relation this benchmark relies on is not the relation Optiland
#: implements.
_PARAXIAL_PROBE_AGREEMENT_MM = 1.0e-9


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------


def _scalar(value: Any) -> float:
    """A backend array or tensor as one host float."""
    import optiland.backend as be

    return float(np.asarray(be.utils.to_numpy(value)).ravel()[0])


def _finite_or_none(value: Any) -> float | None:
    """``None`` for a non-finite solver reading, so JSON stays real JSON."""
    try:
        number = _scalar(value)
    except Exception:  # pragma: no cover - defensive around solver internals
        return None
    return number if math.isfinite(number) else None


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _fingerprint(payload: Any) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _use_numpy_backend() -> None:
    """Pin Optiland's process-global backend to numpy/float64.

    Called before every paraxial read. ``set_backend``/``set_precision`` are
    global and not thread-safe (Optiland's own documentation), and the adapter
    re-pins them on every ``run()``, so neither this function nor the adapter may
    rely on inheriting the other's choice.
    """
    import optiland.backend as be

    be.set_backend("numpy")
    be.set_precision("float64")


# --------------------------------------------------------------------------
# A. construction -- what the manufacturer published, and what got built
# --------------------------------------------------------------------------


def construction_record() -> dict[str, Any]:
    """Every considered catalog component: built, or refused with a reason."""
    components = catalog_components()
    return {
        "question": (
            "A. Could the commercial components be constructed from published data?"
        ),
        "components_considered": len(components),
        "components_constructed": sum(1 for c in components if c.supported),
        "components_refused": sum(1 for c in components if not c.supported),
        "components": [component.as_record() for component in components],
    }


# --------------------------------------------------------------------------
# B. assembly -- did the built system get what the assembly asked for?
# --------------------------------------------------------------------------


def _built_surface_readback(lens: Any, primary_wavelength_um: float) -> list[dict[str, Any]]:
    """The assembled system read back off the ``Optic`` Optiland actually built.

    Deliberately read from the solver object rather than from the prescription
    that produced it: comparing this against
    ``core.optical_assembly.surface_table`` is what makes "ordering, orientation
    and separation survived into the built system" a measurement instead of an
    assertion. The object surface (index 0, at ``z = -inf`` for an infinite
    object) and the image plane (the last) are reported too, since they are part
    of what the builder placed.
    """
    rows: list[dict[str, Any]] = []
    for index, surface in enumerate(lens.surfaces.surfaces):
        rows.append(
            {
                "built_index": index,
                "radius_mm": _finite_or_none(surface.geometry.radius),
                # A plane geometry class carries no `k` attribute at all in the
                # pinned solver, and a plane's conic constant is 0 -- not
                # unknown. Defaulting here keeps that distinction rather than
                # reporting a null the comparison would have to special-case.
                "conic": _finite_or_none(getattr(surface.geometry, "k", 0.0)),
                "vertex_z_mm": _finite_or_none(surface.geometry.cs.z),
                "is_stop": bool(surface.is_stop),
                "surface_type": str(getattr(surface, "surface_type", "")),
                "refractive_index_after": _finite_or_none(
                    surface.material_post.n(primary_wavelength_um)
                ),
                "comment": str(getattr(surface, "comment", "")),
            }
        )
    return rows


def _compare_spec_against_build(
    spec: OpticalSystemSpec, built: list[dict[str, Any]]
) -> dict[str, Any]:
    """Line up the prescription's surfaces with the built ones and diff them.

    The built list carries one leading object surface and one trailing image
    plane that the prescription does not list (``OpticalSystemSpec`` fixes both
    rather than parameterizing them), so the optical surfaces are
    ``built[1:-1]``. Any length mismatch is itself the finding.
    """
    spec_rows = surface_table(spec)
    optical = built[1:-1]
    mismatches: list[str] = []
    if len(optical) != len(spec_rows):
        mismatches.append(
            f"built system has {len(optical)} optical surfaces, prescription has "
            f"{len(spec_rows)}"
        )
    paired: list[dict[str, Any]] = []
    for spec_row, built_row in zip(spec_rows, optical, strict=False):
        radius_delta: float | None = None
        if spec_row["radius_mm"] is None or built_row["radius_mm"] is None:
            if (spec_row["radius_mm"] is None) != (built_row["radius_mm"] is None):
                mismatches.append(
                    f"surface {spec_row['index']}: plane/curved disagreement "
                    f"(prescription {spec_row['radius_mm']!r}, built "
                    f"{built_row['radius_mm']!r})"
                )
        else:
            radius_delta = abs(spec_row["radius_mm"] - built_row["radius_mm"])
            if radius_delta != 0.0:
                mismatches.append(
                    f"surface {spec_row['index']}: radius {spec_row['radius_mm']} "
                    f"built as {built_row['radius_mm']}"
                )
        vertex_delta = None
        if built_row["vertex_z_mm"] is not None:
            vertex_delta = abs(spec_row["vertex_z_mm"] - built_row["vertex_z_mm"])
            # The builder places vertices by accumulating the same thicknesses the
            # prescription lists, so this is exact-in-float-arithmetic rather than
            # approximate; anything above summation round-off is a real defect.
            if vertex_delta > 1.0e-9:
                mismatches.append(
                    f"surface {spec_row['index']}: vertex z {spec_row['vertex_z_mm']} "
                    f"built at {built_row['vertex_z_mm']}"
                )
        conic_delta = None
        if built_row["conic"] is not None:
            conic_delta = abs(float(spec_row["conic"]) - built_row["conic"])
            if conic_delta != 0.0:
                mismatches.append(
                    f"surface {spec_row['index']}: conic {spec_row['conic']} built as "
                    f"{built_row['conic']}"
                )
        if spec_row["is_stop"] != built_row["is_stop"]:
            mismatches.append(
                f"surface {spec_row['index']}: is_stop {spec_row['is_stop']} built as "
                f"{built_row['is_stop']}"
            )
        paired.append(
            {
                "prescription": dict(spec_row),
                "built": built_row,
                "radius_abs_delta_mm": radius_delta,
                "vertex_z_abs_delta_mm": vertex_delta,
                "conic_abs_delta": conic_delta,
            }
        )
    return {
        "surfaces": paired,
        "mismatches": mismatches,
        "ordering_orientation_and_spacing_survived": not mismatches,
    }


# --------------------------------------------------------------------------
# paraxial readout -- prescription-level, always numpy/float64
# --------------------------------------------------------------------------


def _paraxial_readout(spec: OpticalSystemSpec) -> dict[str, Any]:
    """Optiland's paraxial analysis of the built prescription, plus the readback.

    ``back_focal_length_mm`` is the declared image distance plus ``F2``, which
    Optiland reports signed and measured from the image surface. That relation is
    verified rather than assumed: on the KPX094 build, ``image_solve()`` moves the
    last thickness to 96.98866 mm, and 96.97 + F2 = 96.97 + 0.018657 = 96.98866 --
    the same number, so ``last_thickness + F2`` is the distance from the last
    vertex to the paraxial focus.
    """
    _use_numpy_backend()
    lens = build_optiland_system(spec)
    paraxial = lens.paraxial
    trailing = float(spec.surfaces[-1].thickness_mm)
    f2_from_image = _finite_or_none(paraxial.F2())
    readback = _built_surface_readback(lens, spec.primary_wavelength_um)
    return {
        "measured_on": {
            "backend": "numpy",
            "dtype": "float64",
            "why": (
                "paraxial quantities are properties of the prescription, not of the "
                "trace; measuring them on the host at float64 keeps a GPU record's "
                "catalog comparison independent of the GPU"
            ),
        },
        "effective_focal_length_mm": _finite_or_none(paraxial.f2()),
        "front_focal_length_mm": _finite_or_none(paraxial.f1()),
        "rear_focal_point_from_image_surface_mm": f2_from_image,
        "back_focal_length_mm": (
            None if f2_from_image is None else trailing + f2_from_image
        ),
        "declared_image_distance_from_last_vertex_mm": trailing,
        "principal_plane_2_mm": _finite_or_none(paraxial.P2()),
        "entrance_pupil_diameter_mm": _finite_or_none(paraxial.EPD()),
        "entrance_pupil_location_mm": _finite_or_none(paraxial.EPL()),
        "exit_pupil_diameter_mm": _finite_or_none(paraxial.XPD()),
        "exit_pupil_location_from_image_surface_mm": _finite_or_none(paraxial.XPL()),
        "f_number_on_entrance_pupil": _finite_or_none(paraxial.FNO()),
        "image_plane_z_mm": readback[-1]["vertex_z_mm"],
        "built_surface_readback": readback,
    }


def _resolve_image_distance(definition: SystemDefinition) -> dict[str, Any]:
    """The distance from the last vertex to the image plane, and where it came from.

    ``CATALOG_BFL`` takes the manufacturer's published number verbatim.
    ``PARAXIAL_FOCUS`` measures the assembly's own paraxial back focus, and
    measures it **twice** from two far-apart placeholder image distances: the
    quantity must not depend on where the image plane was parked while it was
    being read, and a benchmark that relied on that without checking would not
    notice if it did.
    """
    if definition.image_plane_rule is ImagePlaneRule.CATALOG_BFL:
        comparison = definition.catalog_value("bfl_mm")
        assert comparison is not None  # enforced by SystemDefinition
        return {
            "rule": definition.image_plane_rule.value,
            "image_distance_mm": comparison.catalog_value,
            "provenance": {
                "source": "catalog",
                "url": comparison.catalog_source_url,
                "basis": comparison.basis,
            },
        }

    probes: list[dict[str, Any]] = []
    for placeholder in _PARAXIAL_PROBE_DISTANCES_MM:
        probe_spec = definition.assemble(placeholder)
        readout = _paraxial_readout(probe_spec)
        probes.append(
            {
                "placeholder_image_distance_mm": placeholder,
                "rear_focal_point_from_image_surface_mm": readout[
                    "rear_focal_point_from_image_surface_mm"
                ],
                "back_focal_length_mm": readout["back_focal_length_mm"],
                "effective_focal_length_mm": readout["effective_focal_length_mm"],
            }
        )
    values = [probe["back_focal_length_mm"] for probe in probes]
    if any(value is None for value in values):
        raise PrescriptionError(
            "SYSTEM_PARAXIAL_FOCUS_UNAVAILABLE",
            f"{definition.key!r} has no finite paraxial back focus; an afocal or "
            "virtual-image assembly cannot have its image plane placed this way",
            path="image_plane_rule",
            expected="an assembly with a real paraxial focus",
        )
    spread = max(values) - min(values)  # type: ignore[type-var]
    if spread > _PARAXIAL_PROBE_AGREEMENT_MM:
        raise PrescriptionError(
            "SYSTEM_PARAXIAL_FOCUS_PLACEHOLDER_DEPENDENT",
            f"{definition.key!r} paraxial back focus moved by {spread:.6g} mm between "
            f"placeholder image distances {_PARAXIAL_PROBE_DISTANCES_MM}, which means "
            "`last_thickness + F2` is not the placeholder-independent quantity this "
            "benchmark reads it as",
            path="image_plane_rule",
            expected=f"agreement within {_PARAXIAL_PROBE_AGREEMENT_MM} mm",
        )
    return {
        "rule": definition.image_plane_rule.value,
        # The first probe's value, not a mean: an average of two numbers that must
        # be identical is a way of hiding that they were not.
        "image_distance_mm": float(values[0]),  # type: ignore[arg-type]
        "provenance": {
            "source": "derived",
            "url": None,
            "basis": (
                "paraxial back focus of the assembled prescription "
                "(last-surface thickness + Optiland Paraxial.F2), measured from two "
                "far-apart placeholder image distances and required to agree; a "
                "closed-form paraxial solve, not a spot-size search"
            ),
        },
        "placeholder_independence_check": {
            "probes": probes,
            "spread_mm": spread,
            "tolerance_mm": _PARAXIAL_PROBE_AGREEMENT_MM,
            "independent": True,
        },
    }


# --------------------------------------------------------------------------
# C. execution -- the shipping adapter, and what it actually did
# --------------------------------------------------------------------------


class ExecutionRefused(RuntimeError):
    """The requested execution environment is unavailable. No fallback happens."""

    def __init__(self, diagnostic: dict[str, Any]) -> None:
        super().__init__(diagnostic["message"])
        self.diagnostic = diagnostic


def _trace_config(
    spec: OpticalSystemSpec,
    *,
    backend: str,
    device: str,
    dtype: str,
    wavelength_um: float,
    hy: float,
    rings: int,
    output_directory: Path,
) -> dict[str, Any]:
    return {
        # The canonical prescription itself, through the adapter's documented
        # inline-prescription port. No `sample` name, no Optiland object.
        "prescription": spec,
        "backend": backend,
        "device": device,
        "dtype": dtype,
        "wavelength": wavelength_um,
        "Hx": 0.0,
        "Hy": hy,
        "num_rays": rings,
        "output_directory": str(output_directory),
    }


def _require_execution(
    adapter: OptilandAdapter,
    spec: OpticalSystemSpec,
    *,
    backend: str,
    device: str,
    dtype: str,
    rings: int,
    probe_directory: Path,
) -> dict[str, Any]:
    """Refuse an unavailable execution environment before any trace runs.

    The adapter's own capability gate is the authority here, and it is
    deliberately consulted *without* running anything: ``validate_request``
    reports the same structured codes ``run()`` would raise, including
    ``OPTILAND_CUDA_UNAVAILABLE`` for a CUDA request in a container with no
    device. This function turns that into a refusal record and stops. It never
    substitutes another device -- a silent GPU-to-CPU fallback would make every
    downstream number a claim about hardware that was not used.
    """
    request = ModelRunRequest(
        run_id="che139-execution-probe",
        node_id="probe",
        config=_trace_config(
            spec,
            backend=backend,
            device=device,
            dtype=dtype,
            wavelength_um=spec.primary_wavelength_um,
            hy=0.0,
            rings=rings,
            output_directory=probe_directory,
        ),
    )
    report = adapter.validate_request(request)
    errors = [
        {"code": issue.code, "message": issue.message, "location": issue.location}
        for issue in report.errors
    ]
    if errors:
        raise ExecutionRefused(
            {
                "refused": True,
                "code": "CHE139_REQUESTED_EXECUTION_UNAVAILABLE",
                "message": (
                    f"the requested execution environment (backend={backend!r}, "
                    f"device={device!r}, dtype={dtype!r}) is not available here; the "
                    "benchmark stops rather than falling back to another device"
                ),
                "requested": {"backend": backend, "device": device, "dtype": dtype},
                "adapter_validation_errors": errors,
                "fallback_performed": False,
                "remedy": (
                    "run the GPU image with a device attached: "
                    "`MOA_GPUS=device=6 ./run.sh --gpu python "
                    "benchmarks/applied/commercial_lens_systems/run_benchmark.py run "
                    "--device cuda --backend torch --dtype float64 --tag gpu` "
                    "(docs/testing/gpu_environment.md)"
                ),
            }
        )
    return {
        "requested": {"backend": backend, "device": device, "dtype": dtype},
        "adapter_validation": [issue.code for issue in report.issues],
    }


def _spot_metrics(rays_npz: Path) -> dict[str, Any]:
    """Scalar spot characterization from the adapter's exported ray set.

    Every quantity here is plain arithmetic on coordinates the adapter already
    wrote: this benchmark implements no ray tracing and no optics. ``rms`` is
    about the *centroid* of the traced set, not about the chief ray or the
    paraxial image point, so an off-axis value is a blur size and not a blur plus
    a pointing error -- the pointing error is reported separately as the
    centroid position.
    """
    with np.load(rays_npz) as data:
        x_mm = np.asarray(data["x_m"], dtype=np.float64) * _MM_PER_M
        y_mm = np.asarray(data["y_m"], dtype=np.float64) * _MM_PER_M
        z_mm = np.asarray(data["z_m"], dtype=np.float64) * _MM_PER_M
        intensity = np.asarray(data["intensity"], dtype=np.float64)
    centroid_x = float(x_mm.mean())
    centroid_y = float(y_mm.mean())
    radial = np.hypot(x_mm - centroid_x, y_mm - centroid_y)
    return {
        "surviving_ray_count": int(x_mm.size),
        "centroid_x_mm": centroid_x,
        "centroid_y_mm": centroid_y,
        "rms_spot_radius_mm": float(np.sqrt(np.mean(radial**2))),
        "max_spot_radius_mm": float(radial.max()),
        "spot_radius_definition": (
            "radial distance from the centroid of the traced intercepts on the image "
            "surface; RMS is sqrt(mean(r^2)), unweighted by intensity and unweighted "
            "by pupil area. What it is blind to: because hexapolar samples are equally "
            "weighted but not equal-area, this RMS depends on the ring count -- the "
            "KPX094 on-axis spot measures 97.5 um at 8 rings and 90.1 um at the 16 "
            "rings this benchmark uses. It is a comparable metric ACROSS systems at a "
            "fixed ring count, which is how it is used here, and it is not an "
            "energy-weighted RMS spot size."
        ),
        "image_surface_z_mm": float(z_mm.mean()),
        "image_surface_z_spread_mm": float(z_mm.max() - z_mm.min()),
        "surviving_intensity_sum": float(intensity.sum()),
        "surviving_intensity_min": float(intensity.min()),
        "surviving_intensity_max": float(intensity.max()),
    }


def _survival_block(launched: int, metrics: dict[str, Any]) -> dict[str, Any]:
    """Ray survival, clipping, and exactly how much of it is inferred.

    ``Optic.trace`` returns survivors only and exposes no rejected-candidate
    count, so the clipped count is ``launched - surviving`` where ``launched`` is
    the hexapolar sampler's own ``1 + 3N(N+1)``. The power figure is a *proxy* and
    labelled one: it assumes each launched ray carried unit intensity, and
    ``unit_launch_intensity_evidenced`` reports whether every survivor actually
    carries 1.0, which is the only part of that assumption this benchmark can
    check.
    """
    surviving = int(metrics["surviving_ray_count"])
    unit_launch = (
        metrics["surviving_intensity_min"] == 1.0
        and metrics["surviving_intensity_max"] == 1.0
    )
    return {
        "launched_ray_count": launched,
        "launched_ray_count_basis": (
            "Optiland's hexapolar pupil sampler produces 1 + 3N(N+1) rays for N "
            "rings (solvers.optiland.cost_model.hexapolar_ray_count)"
        ),
        "surviving_ray_count": surviving,
        "clipped_ray_count": launched - surviving,
        "clipped_ray_fraction": (launched - surviving) / launched if launched else None,
        "clipped_power_proxy": launched - metrics["surviving_intensity_sum"],
        "clipped_power_proxy_caveat": (
            "a proxy, not a measurement: it assumes every launched ray carried unit "
            "intensity, which Optic.trace does not report for rays it dropped. "
            "optical-system-spec/1 also has no per-surface aperture, so nothing here "
            "is rim vignetting -- a non-zero clipped count means a ray missed a "
            "surface or was totally internally reflected."
        ),
        "unit_launch_intensity_evidenced": bool(unit_launch),
    }


def _run_one_trace(
    adapter: OptilandAdapter,
    spec: OpticalSystemSpec,
    definition: SystemDefinition,
    *,
    field_deg: float,
    wavelength_um: float,
    backend: str,
    device: str,
    dtype: str,
    trace_root: Path,
    label: str,
) -> dict[str, Any]:
    """One adapter run, plus the scalars derived from what it exported."""
    rings = int(definition.pupil_rings.value)
    hy = definition.normalized_field(field_deg)
    directory = trace_root / label
    directory.mkdir(parents=True, exist_ok=True)
    request = ModelRunRequest(
        run_id=f"che139-{definition.key}",
        node_id=label,
        config=_trace_config(
            spec,
            backend=backend,
            device=device,
            dtype=dtype,
            wavelength_um=wavelength_um,
            hy=hy,
            rings=rings,
            output_directory=directory,
        ),
    )
    started = time.perf_counter()
    result = adapter.run(request)
    elapsed = time.perf_counter() - started
    if result.status is not RunStatus.SUCCEEDED:
        return {
            "label": label,
            "field_angle_deg": field_deg,
            "normalized_field_hy": hy,
            "wavelength_um": wavelength_um,
            "status": result.status.value,
            "error_type": result.error_type,
            "error_message": result.error_message,
            "diagnostics": {
                key: value
                for key, value in result.diagnostics.items()
                if key not in {"objective_tensor", "design_parameter_tensors"}
            },
        }

    artifact = result.outputs["rays"]
    metrics = _spot_metrics(Path(artifact.uri))
    launched = hexapolar_ray_count(rings)
    execution = result.diagnostics["execution"]
    return {
        "label": label,
        "field_angle_deg": field_deg,
        "normalized_field_hy": hy,
        "wavelength_um": wavelength_um,
        "status": result.status.value,
        "wall_time_s": elapsed,
        # C -- where it actually ran, read off the traced arrays by the adapter
        # rather than echoed back from the request.
        "actual_execution": {
            "requested": execution["requested"],
            "resolved": execution["resolved"],
            "applied_to_optiland": execution["applied_to_optiland"],
            "actual": execution["actual"],
            "mismatches": execution["mismatches"],
            "artifact_framework": artifact.framework.value,
            "artifact_device": artifact.device.value,
            "artifact_dtype": artifact.dtype,
        },
        "ray_survival": _survival_block(launched, metrics),
        "spot": metrics,
        "provenance": {
            "prescription_fingerprint": result.diagnostics["prescription_fingerprint"],
            "prescription_source": result.diagnostics["prescription_source"],
            "prescription_spec_version": result.diagnostics["prescription_spec_version"],
            "scientific_array_sha256": result.diagnostics["scientific_array_sha256"],
            "rays_npz": str(Path(artifact.uri).relative_to(trace_root.parent)),
            "rays_npz_sha256": artifact.sha256,
            "optiland_version": result.diagnostics["package_version"],
            "seed": result.diagnostics["seed"],
        },
        "adapter_warnings": list(result.warnings),
    }


# --------------------------------------------------------------------------
# aperture clearance -- the one thing the schema cannot enforce, measured
# --------------------------------------------------------------------------


def _aperture_clearance(
    adapter: OptilandAdapter,
    definition: SystemDefinition,
    image_distance_mm: float,
    *,
    backend: str,
    device: str,
    dtype: str,
    trace_root: Path,
) -> dict[str, Any]:
    """Measure the ray footprint where each downstream component's first surface sits.

    ``optical-system-spec/1`` has no per-surface aperture, so no component rim
    exists in the built system and nothing vignettes. That gap is closed by
    measurement rather than by assumption: for each downstream component, this
    builds a *truncated* prescription ending exactly at that component's first
    vertex -- the same components, the same declared air gap, one fewer element --
    traces it through the same shipping adapter, and reports the largest radial
    intercept against the component's published clear-aperture semi-diameter.

    A negative margin does not fail anything here. It is reported, and it means
    the reported spot of the full system is optimistic by however much light the
    real rim would have removed.
    """
    if len(definition.placements) < 2:
        return {
            "checked": False,
            "reason": "single-component system: no downstream component to clear",
        }
    full = definition.placements_with_image_distance(image_distance_mm)
    full_spec = definition.assemble(image_distance_mm)
    checks: list[dict[str, Any]] = []
    for boundary in range(1, len(definition.placements)):
        downstream_plan = definition.placements[boundary]
        downstream = require_supported(downstream_plan.part_number)
        assert downstream.component is not None
        # `placements_with_image_distance` overrides only the LAST placement's gap,
        # so for any earlier boundary the retained gap is the declared inter-lens
        # air gap -- which puts the truncated build's image plane exactly on the
        # downstream component's first vertex. Re-assembled from the placements
        # rather than trimmed off the full prescription: the assembly function is
        # the thing under test, and a hand-trimmed surface list would bypass it.
        truncated = full[:boundary]
        truncated_spec = assemble_optical_system(
            name=f"{definition.key}__clearance_at_{downstream_plan.part_number}",
            description=(
                f"Truncated build of {definition.key}: components up to index "
                f"{boundary - 1}, image plane placed exactly at the first vertex of "
                f"{downstream_plan.part_number} by the declared "
                f"{full[boundary - 1].air_gap_after_mm} mm air gap. Exists only to "
                "measure the ray footprint arriving at that component."
            ),
            placements=truncated,
            aperture=full_spec.aperture,
            fields=full_spec.fields,
            wavelengths=full_spec.wavelengths,
            object_distance_mm=None,
            stop_component_index=definition.stop_component_index,
            stop_surface_index=definition.stop_surface_index,
        )
        semi_aperture = downstream.component.clear_aperture_mm
        per_field: list[dict[str, Any]] = []
        for field_deg in definition.field_angles_deg:
            label = (
                f"clearance_{downstream_plan.part_number}_f{field_deg:g}"
                f"_w{definition.primary_wavelength_um:g}"
            )
            trace = _run_one_trace(
                adapter,
                truncated_spec,
                definition,
                field_deg=field_deg,
                wavelength_um=definition.primary_wavelength_um,
                backend=backend,
                device=device,
                dtype=dtype,
                trace_root=trace_root,
                label=label,
            )
            if trace["status"] != RunStatus.SUCCEEDED.value:
                per_field.append(
                    {"field_angle_deg": field_deg, "status": trace["status"],
                     "error_message": trace.get("error_message")}
                )
                continue
            with np.load(
                trace_root / label / "rays.npz"
            ) as data:
                x_mm = np.asarray(data["x_m"], dtype=np.float64) * _MM_PER_M
                y_mm = np.asarray(data["y_m"], dtype=np.float64) * _MM_PER_M
            max_radius = float(np.hypot(x_mm, y_mm).max())
            per_field.append(
                {
                    "field_angle_deg": field_deg,
                    "status": trace["status"],
                    "max_ray_radius_mm": max_radius,
                    "clear_aperture_semi_diameter_mm": (
                        None if semi_aperture is None else semi_aperture / 2.0
                    ),
                    "margin_mm": (
                        None if semi_aperture is None else semi_aperture / 2.0 - max_radius
                    ),
                    "inside_clear_aperture": (
                        None if semi_aperture is None else max_radius <= semi_aperture / 2.0
                    ),
                }
            )
        checks.append(
            {
                "downstream_part_number": downstream_plan.part_number,
                "declared_air_gap_mm": full[boundary - 1].air_gap_after_mm,
                "truncated_prescription_fingerprint": truncated_spec.fingerprint(),
                "clear_aperture_mm": semi_aperture,
                "clear_aperture_provenance": (
                    "published clear aperture of the downstream component; see the "
                    "construction block for its verbatim source text"
                ),
                "per_field": per_field,
            }
        )
    return {
        "checked": True,
        "why": (
            "optical-system-spec/1 cannot express a per-surface aperture, so the built "
            "system has no rims and clipping by one cannot occur. The constraint is "
            "therefore verified by measuring the footprint at each downstream "
            "component's first vertex instead of being assumed."
        ),
        "checks": checks,
    }


# --------------------------------------------------------------------------
# D. characterization -- plausibility, against a stated reference
# --------------------------------------------------------------------------


def _catalog_comparisons(
    definition: SystemDefinition, paraxial: dict[str, Any]
) -> list[dict[str, Any]]:
    """Simulated against published, with the absolute and relative difference."""
    simulated_by_quantity = {
        "efl_mm": paraxial["effective_focal_length_mm"],
        "bfl_mm": paraxial["back_focal_length_mm"],
    }
    rows: list[dict[str, Any]] = []
    for comparison in definition.catalog_comparisons:
        simulated = simulated_by_quantity[comparison.quantity]
        absolute = None if simulated is None else simulated - comparison.catalog_value
        rows.append(
            {
                "quantity": comparison.quantity,
                "catalog_value": comparison.catalog_value,
                "catalog_source_url": comparison.catalog_source_url,
                "simulated_value": simulated,
                "absolute_difference": absolute,
                "relative_difference": (
                    None
                    if absolute is None or comparison.catalog_value == 0.0
                    else absolute / comparison.catalog_value
                ),
                "comparison_basis": comparison.basis,
            }
        )
    return rows


def _field_dependence(
    paraxial: dict[str, Any], traces: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Spot size and chief-ray landing versus field, against the paraxial image height.

    The paraxial image height is ``EFL * tan(theta)``, and the difference between
    it and the traced centroid is a chief-ray displacement -- a distortion proxy,
    named as a proxy because it is measured against the paraxial height rather
    than against a reference-field linearization.
    """
    efl = paraxial["effective_focal_length_mm"]
    rows: list[dict[str, Any]] = []
    for trace in traces:
        if trace["status"] != RunStatus.SUCCEEDED.value:
            continue
        angle = float(trace["field_angle_deg"])
        paraxial_height = (
            None if efl is None else efl * math.tan(math.radians(angle))
        )
        centroid_y = trace["spot"]["centroid_y_mm"]
        rows.append(
            {
                "field_angle_deg": angle,
                "wavelength_um": trace["wavelength_um"],
                "rms_spot_radius_mm": trace["spot"]["rms_spot_radius_mm"],
                "max_spot_radius_mm": trace["spot"]["max_spot_radius_mm"],
                "centroid_y_mm": centroid_y,
                "paraxial_image_height_mm": paraxial_height,
                "chief_ray_displacement_mm": (
                    None if paraxial_height is None else centroid_y - paraxial_height
                ),
                "relative_displacement": (
                    None
                    if paraxial_height is None or paraxial_height == 0.0
                    else (centroid_y - paraxial_height) / paraxial_height
                ),
            }
        )
    return rows


def _chromatic_spread(traces: list[dict[str, Any]]) -> dict[str, Any] | None:
    """On-axis RMS spot and focus shift across wavelength, when more than one ran."""
    on_axis = [
        trace
        for trace in traces
        if trace["status"] == RunStatus.SUCCEEDED.value
        and float(trace["field_angle_deg"]) == 0.0
    ]
    if len(on_axis) < 2:
        return None
    by_wavelength = sorted(on_axis, key=lambda trace: trace["wavelength_um"])
    return {
        "why": (
            "an achromat is sold on holding focus across wavelength; this reports the "
            "on-axis RMS spot at each traced line at ONE fixed image plane, so a "
            "spread here is residual chromatic blur plus defocus at that plane, not a "
            "per-wavelength best focus"
        ),
        "per_wavelength": [
            {
                "wavelength_um": trace["wavelength_um"],
                "rms_spot_radius_mm": trace["spot"]["rms_spot_radius_mm"],
                "max_spot_radius_mm": trace["spot"]["max_spot_radius_mm"],
            }
            for trace in by_wavelength
        ],
        "rms_spot_radius_spread_mm": (
            max(t["spot"]["rms_spot_radius_mm"] for t in on_axis)
            - min(t["spot"]["rms_spot_radius_mm"] for t in on_axis)
        ),
    }


# --------------------------------------------------------------------------
# spot diagrams
# --------------------------------------------------------------------------


def _spot_diagram(
    definition: SystemDefinition,
    traces: list[dict[str, Any]],
    trace_root: Path,
    figure_path: Path,
) -> dict[str, Any] | None:
    """One PNG per system: intercepts about their centroid, per field and wavelength.

    Written with ``metadata={'Software': None}`` so the PNG carries no generator
    string and no timestamp, which is what lets the file be byte-identical across
    re-runs and its sha256 be part of the reproducibility claim.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    successful = [t for t in traces if t["status"] == RunStatus.SUCCEEDED.value]
    if not successful:
        return None
    fields = sorted({float(t["field_angle_deg"]) for t in successful})
    wavelengths = sorted({float(t["wavelength_um"]) for t in successful})
    # constrained layout rather than tight_layout: it accounts for the two-line
    # suptitle and the per-axis titles together, where a hand-tuned tight_layout
    # rect clipped the bottom row's x label.
    figure, axes = plt.subplots(
        len(wavelengths),
        len(fields),
        figsize=(2.9 * len(fields) + 1.2, 3.0 * len(wavelengths) + 1.1),
        squeeze=False,
        layout="constrained",
    )
    for row, wavelength in enumerate(wavelengths):
        for column, field_deg in enumerate(fields):
            axis = axes[row][column]
            match = [
                t
                for t in successful
                if float(t["field_angle_deg"]) == field_deg
                and float(t["wavelength_um"]) == wavelength
            ]
            if not match:
                axis.set_axis_off()
                continue
            trace = match[0]
            with np.load(trace_root / trace["label"] / "rays.npz") as data:
                x_um = np.asarray(data["x_m"], dtype=np.float64) * 1.0e6
                y_um = np.asarray(data["y_m"], dtype=np.float64) * 1.0e6
            x_um = x_um - x_um.mean()
            y_um = y_um - y_um.mean()
            rms_um = trace["spot"]["rms_spot_radius_mm"] * 1.0e3
            axis.scatter(x_um, y_um, s=1.5, color="#1f4e79", linewidths=0)
            axis.add_patch(
                plt.Circle(
                    (0.0, 0.0), rms_um, fill=False, color="#c0392b", linewidth=1.0
                )
            )
            extent = 1.15 * max(float(np.abs(x_um).max()), float(np.abs(y_um).max()), rms_um)
            axis.set_xlim(-extent, extent)
            axis.set_ylim(-extent, extent)
            axis.set_aspect("equal")
            axis.tick_params(labelsize=7)
            # ASCII-only text: a glyph the bundled font lacks renders as a
            # fallback box, and the fallback choice is not something this
            # benchmark's byte-level determinism claim should depend on.
            axis.set_title(
                f"field {field_deg:g} deg, {wavelength * 1e3:.1f} nm\n"
                f"RMS {rms_um:.2f} um, n={trace['spot']['surviving_ray_count']}",
                fontsize=8,
            )
            if column == 0:
                axis.set_ylabel("y - y_centroid [um]", fontsize=8)
            if row == len(wavelengths) - 1:
                axis.set_xlabel("x - x_centroid [um]", fontsize=8)
    figure.suptitle(
        f"{definition.key} - {definition.title}\n"
        "geometric spot diagrams at the image surface; red circle = RMS spot radius",
        fontsize=10,
    )
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(figure_path, dpi=150, metadata={"Software": None})
    plt.close(figure)
    return {
        "path": str(figure_path.relative_to(_ROOT)),
        "sha256": hashlib.sha256(figure_path.read_bytes()).hexdigest(),
        "determinism": (
            "PNG metadata is suppressed (Software=None), so no generator string or "
            "timestamp enters the file and the sha256 is stable across re-runs"
        ),
    }


# --------------------------------------------------------------------------
# one system, end to end
# --------------------------------------------------------------------------


def run_system(
    adapter: OptilandAdapter,
    definition: SystemDefinition,
    *,
    backend: str,
    device: str,
    dtype: str,
    tag: str,
    trace_root: Path,
) -> dict[str, Any]:
    image_plane = _resolve_image_distance(definition)
    spec = definition.assemble(image_plane["image_distance_mm"])
    paraxial = _paraxial_readout(spec)
    assembly_check = _compare_spec_against_build(spec, paraxial["built_surface_readback"])

    system_trace_root = trace_root / definition.key
    _require_execution(
        adapter,
        spec,
        backend=backend,
        device=device,
        dtype=dtype,
        rings=int(definition.pupil_rings.value),
        probe_directory=system_trace_root / "_probe",
    )

    traces: list[dict[str, Any]] = []
    for wavelength in definition.wavelengths_um:
        for field_deg in definition.field_angles_deg:
            label = f"f{field_deg:g}_w{wavelength:g}"
            traces.append(
                _run_one_trace(
                    adapter,
                    spec,
                    definition,
                    field_deg=field_deg,
                    wavelength_um=wavelength,
                    backend=backend,
                    device=device,
                    dtype=dtype,
                    trace_root=system_trace_root,
                    label=label,
                )
            )

    clearance = _aperture_clearance(
        adapter,
        definition,
        image_plane["image_distance_mm"],
        backend=backend,
        device=device,
        dtype=dtype,
        trace_root=system_trace_root,
    )

    figure = _spot_diagram(
        definition,
        traces,
        system_trace_root,
        FIGURES / f"{definition.key}.{tag}.png",
    )

    successful = [t for t in traces if t["status"] == RunStatus.SUCCEEDED.value]
    on_axis_primary = [
        t
        for t in successful
        if float(t["field_angle_deg"]) == 0.0
        and float(t["wavelength_um"]) == definition.primary_wavelength_um
    ]

    # The reproducibility claim: everything scientific, nothing about this host,
    # this clock or these paths.
    scientific = {
        "prescription_fingerprint": spec.fingerprint(),
        "image_distance_mm": image_plane["image_distance_mm"],
        "paraxial": {
            key: paraxial[key]
            for key in (
                "effective_focal_length_mm",
                "back_focal_length_mm",
                "rear_focal_point_from_image_surface_mm",
                "entrance_pupil_diameter_mm",
                "exit_pupil_diameter_mm",
                "f_number_on_entrance_pupil",
                "image_plane_z_mm",
            )
        },
        "traces": [
            {
                "label": trace["label"],
                "field_angle_deg": trace["field_angle_deg"],
                "wavelength_um": trace["wavelength_um"],
                "scientific_array_sha256": trace["provenance"]["scientific_array_sha256"],
                "surviving_ray_count": trace["spot"]["surviving_ray_count"],
                "centroid_x_mm": trace["spot"]["centroid_x_mm"],
                "centroid_y_mm": trace["spot"]["centroid_y_mm"],
                "rms_spot_radius_mm": trace["spot"]["rms_spot_radius_mm"],
                "max_spot_radius_mm": trace["spot"]["max_spot_radius_mm"],
            }
            for trace in successful
        ],
    }

    return {
        "benchmark": "CHE-139 / M1.1.5 commercial catalog lens systems",
        "system_key": definition.key,
        "title": definition.title,
        "question": definition.question,
        "description": definition.description,
        "role": definition.role.value,
        "control_of": definition.control_of,
        "tag": tag,
        # --- A -----------------------------------------------------------
        "construction": {
            "question": "A. Could the commercial components be constructed?",
            "part_numbers": list(definition.part_numbers),
            "source_urls": sorted(
                {
                    url
                    for part in definition.part_numbers
                    for url in require_supported(part).source_urls()
                }
            ),
            "all_components_constructed": True,
            "note": (
                "the full catalog provenance for every component, including the one "
                "this benchmark refused to build, is in components.json beside this "
                "record"
            ),
        },
        # --- B -----------------------------------------------------------
        "assembly": {
            "question": "B. Could multiple components be assembled into a real optical system?",
            "component_count": len(definition.placements),
            "is_multi_component": len(definition.placements) > 1,
            "definition": definition.assembly_record(image_plane["image_distance_mm"]),
            "image_plane": image_plane,
            "assembled_prescription": {
                "spec_version": spec.spec_version,
                "fingerprint": spec.fingerprint(),
                "surface_count": len(spec.surfaces),
                "surface_table": [dict(row) for row in surface_table(spec)],
            },
            "built_system_readback": assembly_check,
            "aperture_clearance": clearance,
        },
        # --- C -----------------------------------------------------------
        "execution": {
            "question": "C. Did the ray trace execute correctly?",
            "requested": {"backend": backend, "device": device, "dtype": dtype},
            "traces_attempted": len(traces),
            "traces_succeeded": len(successful),
            "silent_fallback_performed": False,
            "traces": traces,
        },
        # --- D -----------------------------------------------------------
        "characterization": {
            "question": "D. Does the resulting optical system behave plausibly?",
            "paraxial": paraxial,
            "catalog_comparisons": _catalog_comparisons(definition, paraxial),
            "on_axis_primary": (
                {
                    "wavelength_um": on_axis_primary[0]["wavelength_um"],
                    "rms_spot_radius_mm": on_axis_primary[0]["spot"]["rms_spot_radius_mm"],
                    "max_spot_radius_mm": on_axis_primary[0]["spot"]["max_spot_radius_mm"],
                    "surviving_ray_count": on_axis_primary[0]["spot"][
                        "surviving_ray_count"
                    ],
                }
                if on_axis_primary
                else None
            ),
            "field_dependence": _field_dependence(paraxial, traces),
            "chromatic": _chromatic_spread(traces),
            "spot_diagram": figure,
        },
        "reproducibility": {
            "scientific_result": scientific,
            "result_fingerprint": _fingerprint(scientific),
            "basis": (
                "sha256 over the canonical JSON of the assembled prescription "
                "fingerprint, the paraxial readout and every trace's scientific array "
                "hash and derived scalars. Host, clock, wall time and filesystem paths "
                "are excluded. A re-run with the same --device/--backend/--dtype must "
                "reproduce this exactly; a different precision or device legitimately "
                "will not, because the traced arrays themselves differ."
            ),
        },
    }


# --------------------------------------------------------------------------
# subcommands
# --------------------------------------------------------------------------


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False, allow_nan=False) + "\n")
    return path



def _observed_execution_environment() -> dict[str, Any]:
    """What the runtime says about itself, read rather than declared.

    ``core.performance.environment_fingerprint`` reports ``container_image`` from
    the ``MOA_IMAGE`` environment variable, which ``run.sh --gpu`` does not set --
    so a GPU run's fingerprint says ``agent_solver`` even though it executed in
    ``agent_solver_gpu``. That is a labelling gap in the shared fingerprint, not
    something this benchmark should paper over or reach outside its scope to fix,
    so the GPU claim is carried by observed facts instead: the torch build string
    (``+cu126`` only exists in the CUDA image), whether a CUDA device is actually
    reachable, and its name. Every trace additionally records the device its own
    arrays came back on.
    """
    observed: dict[str, Any] = {
        "container_image_label_caveat": (
            "environment.container_image comes from MOA_IMAGE, which run.sh --gpu "
            "does not set; trust the torch build and per-trace actual device below"
        )
    }
    try:
        import torch
    except Exception as exc:  # pragma: no cover - torch is pinned in both images
        observed["torch"] = f"unimportable: {type(exc).__name__}: {exc}"
        return observed
    observed["torch_version"] = torch.__version__
    observed["torch_cuda_build"] = torch.version.cuda
    observed["torch_cuda_available"] = bool(torch.cuda.is_available())
    observed["visible_cuda_device_count"] = (
        int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    )
    observed["cuda_device_names"] = (
        [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
        if torch.cuda.is_available()
        else []
    )
    return observed


def _print_system_summary(record: dict[str, Any]) -> None:
    """One console line per system. Formats nothing that may be ``None``."""
    paraxial = record["characterization"]["paraxial"]
    summary = record["characterization"]["on_axis_primary"]
    assembly = record["assembly"]
    execution = record["execution"]
    survived = assembly["built_system_readback"][
        "ordering_orientation_and_spacing_survived"
    ]
    parts = [
        f"  surfaces {assembly['assembled_prescription']['surface_count']}",
        f"assembly_ok {survived}",
        f"traces {execution['traces_succeeded']}/{execution['traces_attempted']}",
    ]
    for label, value in (
        ("EFL", paraxial["effective_focal_length_mm"]),
        ("BFL", paraxial["back_focal_length_mm"]),
    ):
        parts.append(f"{label} {value:.4f} mm" if value is not None else f"{label} n/a")
    if summary is not None:
        parts.append(f"on-axis RMS {summary['rms_spot_radius_mm'] * 1e3:.3f} um")
        parts.append(f"rays {summary['surviving_ray_count']}")
    print("[che139]" + "  ".join(parts), flush=True)


def command_run(args: argparse.Namespace) -> int:
    keys = SYSTEM_KEYS if args.systems == "all" else tuple(args.systems.split(","))
    definitions = [resolve_system(key) for key in keys]
    trace_root = Path(args.trace_output).resolve() / args.tag
    adapter = OptilandAdapter()

    environment = environment_fingerprint()
    header = {
        "benchmark": "CHE-139 / M1.1.5 commercial catalog lens systems",
        "tag": args.tag,
        "requested_execution": {
            "backend": args.backend,
            "device": args.device,
            "dtype": args.dtype,
        },
        "environment": environment.as_dict()
        if hasattr(environment, "as_dict")
        else json.loads(json.dumps(environment, default=lambda o: o.__dict__)),
        "observed_execution_environment": _observed_execution_environment(),
        "systems": list(keys),
    }

    _write_json(RECORDS / "components.json", construction_record())

    records: list[dict[str, Any]] = []
    for definition in definitions:
        print(f"[che139] {definition.key} ({args.tag}) ...", flush=True)
        try:
            record = run_system(
                adapter,
                definition,
                backend=args.backend,
                device=args.device,
                dtype=args.dtype,
                tag=args.tag,
                trace_root=trace_root,
            )
        except ExecutionRefused as refusal:
            payload = {**header, "system_key": definition.key, "refusal": refusal.diagnostic}
            path = _write_json(RECORDS / f"execution_refusal.{args.tag}.json", payload)
            print(
                f"[che139] REFUSED: {refusal.diagnostic['message']}\n"
                f"[che139] structured diagnostic written to {path}",
                file=sys.stderr,
            )
            return 2
        except (AdapterDependencyError, UnsupportedCapabilityError, PrescriptionError) as exc:
            payload = {
                **header,
                "system_key": definition.key,
                "refusal": {
                    "refused": True,
                    "code": type(exc).__name__,
                    "message": str(exc),
                    "fallback_performed": False,
                },
            }
            path = _write_json(RECORDS / f"execution_refusal.{args.tag}.json", payload)
            print(f"[che139] REFUSED: {exc}\n[che139] written to {path}", file=sys.stderr)
            return 2
        record["environment"] = header["environment"]
        _write_json(RECORDS / f"{definition.key}.{args.tag}.json", record)
        records.append(record)
        _print_system_summary(record)

    control_comparisons = _control_comparisons(records)
    _write_json(
        RECORDS / f"summary.{args.tag}.json",
        {
            **header,
            "systems_run": [
                {
                    "system_key": record["system_key"],
                    "role": record["role"],
                    "component_count": record["assembly"]["component_count"],
                    "is_multi_component": record["assembly"]["is_multi_component"],
                    "assembly_survived": record["assembly"]["built_system_readback"][
                        "ordering_orientation_and_spacing_survived"
                    ],
                    "traces_succeeded": record["execution"]["traces_succeeded"],
                    "traces_attempted": record["execution"]["traces_attempted"],
                    "actual_execution": (
                        record["execution"]["traces"][0]["actual_execution"]["actual"]
                        if record["execution"]["traces"]
                        and "actual_execution" in record["execution"]["traces"][0]
                        else None
                    ),
                    "effective_focal_length_mm": record["characterization"]["paraxial"][
                        "effective_focal_length_mm"
                    ],
                    "back_focal_length_mm": record["characterization"]["paraxial"][
                        "back_focal_length_mm"
                    ],
                    "catalog_comparisons": record["characterization"]["catalog_comparisons"],
                    "on_axis_primary": record["characterization"]["on_axis_primary"],
                    "result_fingerprint": record["reproducibility"]["result_fingerprint"],
                }
                for record in records
            ],
            "negative_controls": control_comparisons,
        },
    )
    return 0


def _control_comparisons(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Every negative control against the case it controls, on the shared field grid.

    A control that does not fire is reported as a control that did not fire. The
    alternative -- quietly not reporting it -- is how a benchmark ends up trusting
    machinery it never tested.
    """
    by_key = {record["system_key"]: record for record in records}
    rows: list[dict[str, Any]] = []
    for record in records:
        if record["role"] != SystemRole.NEGATIVE_CONTROL.value:
            continue
        target = by_key.get(record["control_of"])
        if target is None:
            rows.append(
                {
                    "control": record["system_key"],
                    "control_of": record["control_of"],
                    "compared": False,
                    "reason": "the controlled system was not part of this run",
                }
            )
            continue

        def indexed(entry: dict[str, Any]) -> dict[tuple[float, float], dict[str, Any]]:
            return {
                (float(row["field_angle_deg"]), float(row["wavelength_um"])): row
                for row in entry["characterization"]["field_dependence"]
            }

        control_rows = indexed(record)
        target_rows = indexed(target)
        shared = sorted(set(control_rows) & set(target_rows))
        per_point = [
            {
                "field_angle_deg": key[0],
                "wavelength_um": key[1],
                "case_rms_spot_radius_mm": target_rows[key]["rms_spot_radius_mm"],
                "control_rms_spot_radius_mm": control_rows[key]["rms_spot_radius_mm"],
                "ratio_control_over_case": (
                    control_rows[key]["rms_spot_radius_mm"]
                    / target_rows[key]["rms_spot_radius_mm"]
                    if target_rows[key]["rms_spot_radius_mm"]
                    else None
                ),
            }
            for key in shared
        ]
        ratios = [
            row["ratio_control_over_case"]
            for row in per_point
            if row["ratio_control_over_case"] is not None
        ]
        rows.append(
            {
                "control": record["system_key"],
                "control_of": record["control_of"],
                "compared": True,
                "expectation": (
                    "installing the achromat against its manufacturer-stated "
                    "orientation must make the RMS spot LARGER at every field; a ratio "
                    "at or below 1 would mean the orientation never reached the built "
                    "system"
                ),
                "shared_points": len(per_point),
                "per_point": per_point,
                "min_ratio": min(ratios) if ratios else None,
                "max_ratio": max(ratios) if ratios else None,
                "control_fired_at_every_point": bool(ratios) and all(r > 1.0 for r in ratios),
            }
        )
    return rows


def command_compare(args: argparse.Namespace) -> int:
    """GPU vs CPU behaviour equivalence on the declared scalar set."""
    primary = json.loads(Path(args.primary).read_text())
    secondary = json.loads(Path(args.secondary).read_text())
    if primary["system_key"] != secondary["system_key"]:
        print(
            f"[che139] refusing to compare {primary['system_key']} against "
            f"{secondary['system_key']}: different optical systems",
            file=sys.stderr,
        )
        return 2
    if (
        primary["assembly"]["assembled_prescription"]["fingerprint"]
        != secondary["assembly"]["assembled_prescription"]["fingerprint"]
    ):
        print(
            "[che139] refusing to compare: the two records built different "
            "prescriptions (fingerprints differ)",
            file=sys.stderr,
        )
        return 2

    #: Below this magnitude a length is float64 cancellation noise rather than a
    #: value: the on-axis centroid of a symmetric ray set comes out at ~1e-17 mm
    #: on both backends, and dividing one such residue by another produces a
    #: ratio of order 1 that says nothing about either backend. 1e-9 mm is a
    #: picometre, twelve orders below the ~50 um spots being compared, so no real
    #: quantity in this benchmark is suppressed by it.
    negligible_mm = 1.0e-9
    #: Which metrics are lengths in mm, and so subject to the rule above. A count
    #: is exact on both backends and needs no floor.
    length_metrics = {
        "centroid_x_mm",
        "centroid_y_mm",
        "rms_spot_radius_mm",
        "max_spot_radius_mm",
        "focal_plane_z_mm",
    }

    def traces(record: dict[str, Any]) -> dict[str, dict[str, Any]]:
        return {
            trace["label"]: trace
            for trace in record["execution"]["traces"]
            if trace["status"] == RunStatus.SUCCEEDED.value
        }

    primary_traces = traces(primary)
    secondary_traces = traces(secondary)
    shared = sorted(set(primary_traces) & set(secondary_traces))
    rows: list[dict[str, Any]] = []
    for label in shared:
        left, right = primary_traces[label], secondary_traces[label]
        metrics: dict[str, Any] = {}
        for metric in GPU_CPU_COMPARISON_METRICS:
            if metric == "surviving_ray_count":
                a = left["spot"]["surviving_ray_count"]
                b = right["spot"]["surviving_ray_count"]
            elif metric == "focal_plane_z_mm":
                a = left["spot"]["image_surface_z_mm"]
                b = right["spot"]["image_surface_z_mm"]
            else:
                a = left["spot"][metric]
                b = right["spot"][metric]
            absolute = a - b
            suppressed = (
                metric in length_metrics and abs(b) <= negligible_mm
            )
            metrics[metric] = {
                "primary": a,
                "secondary": b,
                "absolute_difference": absolute,
                "relative_difference": (
                    None if suppressed or not b else absolute / b
                ),
                "relative_difference_suppressed_reason": (
                    (
                        f"|reference| = {abs(b):.3g} mm is at or below the "
                        f"{negligible_mm:g} mm float64-cancellation floor, so this "
                        "quantity is zero on both backends and a ratio of the two "
                        "residues would be noise divided by noise. Judge it on "
                        "absolute_difference."
                    )
                    if suppressed
                    else None
                ),
            }
        rows.append(
            {
                "label": label,
                "field_angle_deg": left["field_angle_deg"],
                "wavelength_um": left["wavelength_um"],
                "primary_execution": left["actual_execution"]["actual"],
                "secondary_execution": right["actual_execution"]["actual"],
                "scientific_array_sha256_match": (
                    left["provenance"]["scientific_array_sha256"]
                    == right["provenance"]["scientific_array_sha256"]
                ),
                "metrics": metrics,
            }
        )
    payload = {
        "benchmark": "CHE-139 / M1.1.5 commercial catalog lens systems",
        "comparison": "GPU vs CPU behaviour equivalence",
        "system_key": primary["system_key"],
        "prescription_fingerprint": primary["assembly"]["assembled_prescription"][
            "fingerprint"
        ],
        "declared_metrics": list(GPU_CPU_COMPARISON_METRICS),
        "declared_before_running": (
            "GPU_CPU_COMPARISON_METRICS in benchmark_systems.py; fixed in source so "
            "the compared set cannot be chosen after seeing the differences"
        ),
        "primary": {
            "tag": primary["tag"],
            "requested": primary["execution"]["requested"],
            "result_fingerprint": primary["reproducibility"]["result_fingerprint"],
        },
        "secondary": {
            "tag": secondary["tag"],
            "requested": secondary["execution"]["requested"],
            "result_fingerprint": secondary["reproducibility"]["result_fingerprint"],
        },
        "paraxial_identical": (
            primary["characterization"]["paraxial"]["effective_focal_length_mm"]
            == secondary["characterization"]["paraxial"]["effective_focal_length_mm"]
        ),
        "paraxial_note": (
            "the paraxial block is measured on numpy/float64 in both records by "
            "construction, so its agreement is a consistency check on the harness, "
            "not evidence about the backends"
        ),
        "shared_traces": len(rows),
        "per_trace": rows,
        "worst_relative_difference": max(
            (
                abs(entry["metrics"][metric]["relative_difference"])
                for entry in rows
                for metric in GPU_CPU_COMPARISON_METRICS
                if entry["metrics"][metric]["relative_difference"] is not None
            ),
            default=None,
        ),
        "worst_absolute_difference_mm": max(
            (
                abs(entry["metrics"][metric]["absolute_difference"])
                for entry in rows
                for metric in GPU_CPU_COMPARISON_METRICS
                if metric in length_metrics
            ),
            default=None,
        ),
        "relative_difference_floor_mm": negligible_mm,
        "scientific_array_hashes_expected_to_differ": (
            "the per-trace scientific_array_sha256 is a hash of the exported ray "
            "arrays, and a float64 CUDA trace does not reproduce a float64 host "
            "trace bit-for-bit. A False match is the expected result here and is not "
            "a disagreement; the declared scalars above are where equivalence is "
            "judged."
        ),
    }
    path = _write_json(
        RECORDS / f"gpu_cpu_comparison.{primary['system_key']}.json", payload
    )
    print(f"[che139] wrote {path}")
    worst = payload["worst_relative_difference"]
    print(
        "[che139] worst relative difference across "
        f"{len(rows)} shared traces and {len(GPU_CPU_COMPARISON_METRICS)} metrics: "
        f"{worst!r}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="build, trace and characterize the systems")
    run.add_argument("--backend", choices=("numpy", "torch"), default="torch")
    run.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    run.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    run.add_argument(
        "--systems",
        default="all",
        help=f"'all' or a comma-separated subset of {','.join(SYSTEM_KEYS)}",
    )
    run.add_argument("--tag", required=True, help="record/figure suffix, e.g. 'gpu' or 'cpu'")
    run.add_argument(
        "--trace-output",
        default=str(_ROOT / "outputs" / "che139_commercial_lens_systems"),
        help="where the raw per-trace rays.npz go (gitignored; records keep their hashes)",
    )
    run.set_defaults(handler=command_run)

    compare = sub.add_parser("compare", help="compare two records on the declared scalars")
    compare.add_argument("--primary", required=True)
    compare.add_argument("--secondary", required=True)
    compare.set_defaults(handler=command_compare)

    args = parser.parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
