"""Tier 1 of workstream A: the 11 gallery notebooks whose headline number has a descriptor.

CHE-239 §A.4. Each of these notebooks prints or plots a quantity that one of the
project's four Optiland-touching descriptors returns, so the notebook's own
output can be diffed against ours. Nothing here reproduces a *plot*; what is
compared is the number under it.

What a Tier-1 diff actually establishes
---------------------------------------
Both sides of every comparison in this file run the same Optiland code. That is
not an oversight -- it is the point, and it bounds the claim:

* **`SOM_SPOT_DIAGRAM` and `SOM_PSF` delegate.** They construct a lens from
  `problems.OpticalSetup` and hand it to `optiland.analysis.SpotDiagram` or one
  of the three PSF classes. So a Tier-1 diff is a **plumbing regression**: it
  says the setup extracted from the notebook's lens rebuilds *that lens*, that
  the field and wavelength reached the right arguments, and that the unit
  conversion on the way out is right. `AGENTS.md`'s rule that repository code may
  not be its own correctness oracle is why this is not called validation: a zero
  diff here is silent about whether Optiland's spot is the right spot.
* A **nonzero** diff is the informative outcome, and every one of them is
  reported with its cause attributed as far as the evidence goes.

The three traps CHE-239 §A.3 names, and where each is handled
-------------------------------------------------------------
1. *Native spot is not `measurements.spot_diagram`.* Not a trap here: Tier 1 is
   native-vs-native throughout, and `M_SPOT_DIAGRAM` is never called. The
   `reference="chief_ray"` default is passed identically on both sides.
2. *PSF normalization is Optiland Strehl-percent.* Also native-vs-native, so both
   sides carry it and no rescale is needed. It is recorded on every PSF row
   anyway, because a reader comparing these numbers to a `measurements.psf`
   result would need it.
3. *The three PSF methods disagree at coarse sampling.* No method is ever used as
   an oracle for another. Each notebook's method is compared only against itself.

The multi-field / multi-wavelength decomposition
------------------------------------------------
Four of these notebooks call an analysis over `fields="all"` and
`wavelengths="all"`, and `problems` cannot express that: `SourceSpec` carries one
wavelength and refuses a sequence, `OpticalSetup` carries no field list, and
`build_lens` declares exactly one of each. CHE-239 §A.2 calls this `PASS-refused`
and it is recorded that way -- with the actual refusal, raised and caught, rather
than asserted in prose.

But refusing the *aggregate* call is not the same as being unable to compute the
numbers, and reporting only the refusal would understate what the tree does. So
each such notebook is also decomposed into one single-(field, wavelength) run per
cell of the native grid, and every cell is diffed. A 3x3 notebook therefore
contributes one refusal row and nine comparison rows.
"""

from __future__ import annotations

import argparse
import math
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from backends.optiland.analysis import psf as native_psf
from backends.optiland.analysis import spot_diagram as native_spot_diagram
from benchmarks.verification.optiland_prescription import (
    Unexpressible,
    extract_setup,
    sample_fields_deg,
    sample_wavelengths_um,
)
from benchmarks.verification.record import (
    METRE_PER_MM,
    Row,
    device_execution,
    finish,
    provenance,
)
from problems import SourceSpec

#: Optiland's own defaults for `SpotDiagram`, restated so both sides of a diff
#: pass the same thing rather than inheriting it. A delegation that quietly
#: changed one of these would answer a different question than the notebook.
SPOT_DEFAULTS: dict[str, Any] = {
    "num_rings": 6,
    "distribution": "hexapolar",
    "coordinates": "local",
    "reference": "chief_ray",
}

#: Pupil sampling for the PSF cases. The notebooks that pass `num_rays`
#: explicitly are honoured; where a notebook takes the class default this records
#: what that default was, because `fft` reduces the request itself
#: (`NativePsfAnalysis.num_rays` is the count actually used, not the one asked
#: for) and a row that named the request would name a sampling that never ran.
PSF_DEFAULT_NUM_RAYS = 128


def _relative(measured: float, reference: float) -> float:
    """`|measured - reference| / |reference|`, and the absolute difference at zero.

    A centroid is legitimately zero on axis, and a relative error against zero is
    either `inf` or a lie. Returning the absolute difference there keeps the
    column readable and is stated on every row that uses it.
    """
    if reference == 0.0:
        return abs(measured - reference)
    return abs(measured - reference) / abs(reference)


def _spot_comparison(
    *,
    case: str,
    lens: Any,
    setup: Any,
    field_deg: tuple[float, float],
    wavelength_um: float,
    defocus_mm: float = 0.0,
) -> Row:
    """One (field, wavelength) cell: native `SpotDiagram` against `SOM_SPOT_DIAGRAM`.

    The native side is asked for exactly the cell being compared -- one field, one
    wavelength -- because the project side can only produce one, and diffing a
    single value against the first cell of a 3x3 grid would silently compare the
    wrong pair whenever the ordering changed.

    `defocus_mm` shifts the image plane, which is how `ThroughFocusSpotDiagram`
    decomposes: on the native side by moving `image_surface.geometry.cs.z`, which
    is what that class itself does, and on ours by lengthening the last surface's
    `thickness_mm`, which is where `OpticalSetup` puts the image plane. Those are
    two spellings of one geometric change and the row exists partly to show they
    agree.
    """
    from optiland.analysis import SpotDiagram  # local: see the package docstring

    max_field = float(lens.fields.max_field)
    normalized = (
        0.0 if max_field == 0.0 else field_deg[0] / max_field,
        0.0 if max_field == 0.0 else field_deg[1] / max_field,
    )

    def _native_at(wavelength: float) -> tuple[tuple[float, float], float, float]:
        analysis = SpotDiagram(
            lens, fields=[normalized], wavelengths=[wavelength], **SPOT_DEFAULTS
        )
        centroid = analysis.centroid()[0]
        return (
            (float(centroid[0]), float(centroid[1])),
            float(analysis.rms_spot_radius()[0][0]),
            float(analysis.geometric_spot_radius()[0][0]),
        )

    nominal_z = float(lens.image_surface.geometry.cs.z)
    if defocus_mm:
        lens.image_surface.geometry.cs.z = nominal_z + defocus_mm
    try:
        native_centroid, native_rms, native_geometric = _native_at(wavelength_um)
        # The same analysis at the *setup's reference* wavelength rather than the
        # source's. Computed on every row and not only on the ones that disagree,
        # because that is what makes a row self-diagnosing: CHE-238's own
        # discrepancy rule asks for the deviation to be attributed, and an
        # attribution that has to be recomputed later against a lens that has since
        # been mutated is not evidence. See `_wavelength_attribution`.
        reference_um = float(setup.reference_wavelength_um)
        at_reference = (
            (native_centroid, native_rms, native_geometric)
            if reference_um == wavelength_um
            else _native_at(reference_um)
        )
    finally:
        lens.image_surface.geometry.cs.z = nominal_z

    if defocus_mm:
        last = setup.surfaces[-1]
        shifted = (
            *setup.surfaces[:-1],
            type(last)(
                thickness_mm=last.thickness_mm + defocus_mm,
                radius_mm=last.radius_mm,
                curvature_per_mm=last.curvature_per_mm,
                conic=last.conic,
                aspheric_coefficients=last.aspheric_coefficients,
                clear_semi_diameter_mm=last.clear_semi_diameter_mm,
                material=last.material,
                comment=last.comment,
            ),
        )
        setup = type(setup)(
            name=f"{setup.name}@{defocus_mm:+.3f}mm",
            surfaces=shifted,
            entrance_pupil_diameter_mm=setup.entrance_pupil_diameter_mm,
            stop_index=setup.stop_index,
            reference_wavelength_um=setup.reference_wavelength_um,
            description=setup.description,
        )

    source = SourceSpec(wavelength_um=wavelength_um, field_angle_deg=field_deg)
    started = time.perf_counter()
    ours = native_spot_diagram(
        setup,
        source,
        num_rings=SPOT_DEFAULTS["num_rings"],
        execution=device_execution(),
        distribution=SPOT_DEFAULTS["distribution"],
        coordinates=SPOT_DEFAULTS["coordinates"],
        reference=SPOT_DEFAULTS["reference"],
    )
    runtime_s = time.perf_counter() - started

    measured = {
        "centroid_x_m": ours.centroid_m[0],
        "centroid_y_m": ours.centroid_m[1],
        "rms_radius_m": ours.rms_radius_m,
        "geometric_radius_m": ours.geometric_radius_m,
    }
    expected = {
        "centroid_x_m": float(native_centroid[0]) * METRE_PER_MM,
        "centroid_y_m": float(native_centroid[1]) * METRE_PER_MM,
        "rms_radius_m": native_rms * METRE_PER_MM,
        "geometric_radius_m": native_geometric * METRE_PER_MM,
    }
    deltas = {
        key: _relative(measured[key], expected[key]) for key in measured
    }
    worst = max(deltas.values())

    at_reference_expected = {
        "centroid_x_m": at_reference[0][0] * METRE_PER_MM,
        "centroid_y_m": at_reference[0][1] * METRE_PER_MM,
        "rms_radius_m": at_reference[1] * METRE_PER_MM,
        "geometric_radius_m": at_reference[2] * METRE_PER_MM,
    }
    attribution = _wavelength_attribution(
        measured=measured,
        at_source=expected,
        at_reference=at_reference_expected,
        source_um=wavelength_um,
        reference_um=float(setup.reference_wavelength_um),
    )

    return Row(
        case=case,
        configuration={
            "field_deg": list(field_deg),
            "normalized_field": list(normalized),
            "wavelength_um": wavelength_um,
            "setup_reference_wavelength_um": float(setup.reference_wavelength_um),
            "defocus_mm": defocus_mm,
            **SPOT_DEFAULTS,
        },
        descriptor="SOM_SPOT_DIAGRAM",
        status="PASS" if worst <= 1e-9 else "FAIL",
        measured=measured,
        expected=expected,
        deltas=deltas,
        worst_relative_delta=worst,
        runtime_s=runtime_s,
        note=(
            "native optiland.analysis.SpotDiagram vs backends.optiland.analysis:spot_diagram on "
            "the same single (field, wavelength). Relative difference, except a zero reference "
            "where the absolute difference is reported. Plumbing regression, not validation: "
            "both sides run the same Optiland code"
        ),
        extra={
            "native_at_setup_reference_wavelength": at_reference_expected,
            "our_numbers_match": attribution,
        },
    )


#: What a row's numbers turned out to be the spot *of*, when the source
#: wavelength and the setup's reference wavelength differ.
#:
#: This exists because of a confirmed discrepancy rather than as a precaution.
#: `backends.optiland.analysis:spot_diagram` builds the lens with `build_lens`,
#: which declares exactly one wavelength -- `setup.reference_wavelength_um` -- and
#: then asks `SpotDiagram` for `wavelengths="all"`. "All" is therefore the
#: *reference* wavelength, while the returned `NativeSpotAnalysis.wavelength_m`
#: reports `source.wavelength_um`. Measured on `CookeTriplet` on axis with
#: `num_rings=6`: at a 0.55 um reference, a source at 0.48 um returns an RMS
#: radius of 4.293689564257647e-3 mm, which is the 0.55 um answer to sixteen
#: digits, where the 0.48 um answer is 3.7913354614484123e-3 mm. Moving the
#: reference to 0.48 um returns the 0.48 um answer for a 0.55 um source. So the
#: source wavelength does not reach the analysis at all.
#:
#: Not fixed here. CHE-238's code-change policy is no production change by
#: default, and this is a behaviour change to a solver adapter that would move
#: frozen numbers -- independent review and its own ticket, per `AGENTS.md`.
#: `psf` is unaffected: it passes `source.wavelength_um` to the PSF class
#: explicitly, and the Tier-1 PSF rows exercise a source that differs from the
#: reference for exactly that reason.
_ATTRIBUTIONS = (
    "source_wavelength",
    "setup_reference_wavelength",
    "neither",
    "indistinguishable",
)


def _wavelength_attribution(
    *,
    measured: dict[str, float],
    at_source: dict[str, float],
    at_reference: dict[str, float],
    source_um: float,
    reference_um: float,
) -> str:
    """Which wavelength our numbers are actually the answer for."""
    if source_um == reference_um:
        return "indistinguishable"
    tolerance = 1e-9
    matches_source = all(
        _relative(measured[key], at_source[key]) <= tolerance for key in measured
    )
    matches_reference = all(
        _relative(measured[key], at_reference[key]) <= tolerance for key in measured
    )
    if matches_source and not matches_reference:
        return "source_wavelength"
    if matches_reference and not matches_source:
        return "setup_reference_wavelength"
    if matches_source and matches_reference:
        return "indistinguishable"
    return "neither"


def _psf_comparison(
    *,
    case: str,
    lens: Any,
    setup: Any,
    method: str,
    field_deg: tuple[float, float],
    wavelength_um: float,
    num_rays: int,
) -> Row:
    """One PSF cell: the notebook's own PSF class against `SOM_PSF` with the same method.

    Diffed on all three of what CHE-239 §A.4 asks for -- the peak value, the peak
    location, and the full intensity grid -- because a PSF that agrees at its peak
    and disagrees three samples out is a different PSF, and a scalar comparison
    would not see it.
    """
    from optiland.psf import FFTPSF, MMDFTPSF, HuygensPSF  # local: package docstring

    classes = {"fft": FFTPSF, "mmdft": MMDFTPSF, "huygens": HuygensPSF}
    max_field = float(lens.fields.max_field)
    normalized = (
        0.0 if max_field == 0.0 else field_deg[0] / max_field,
        0.0 if max_field == 0.0 else field_deg[1] / max_field,
    )
    native = classes[method](lens, normalized, wavelength_um, num_rays=num_rays)
    native_map = np.asarray(native.psf, dtype=np.float64)

    source = SourceSpec(wavelength_um=wavelength_um, field_angle_deg=field_deg)
    started = time.perf_counter()
    ours = native_psf(
        setup,
        source,
        method=method,
        num_rays=num_rays,
        execution=device_execution(),
    )
    runtime_s = time.perf_counter() - started
    our_map = np.asarray(ours.intensity, dtype=np.float64)

    if our_map.shape != native_map.shape:
        return Row(
            case=case,
            configuration={
                "method": method,
                "field_deg": list(field_deg),
                "wavelength_um": wavelength_um,
                "num_rays_requested": num_rays,
            },
            descriptor="SOM_PSF",
            status="FAIL",
            measured={"shape": list(our_map.shape)},
            expected={"shape": list(native_map.shape)},
            deltas={},
            worst_relative_delta=math.inf,
            runtime_s=runtime_s,
            note="the two maps are not the same shape, so no element-wise diff is defined",
        )

    scale = float(np.max(np.abs(native_map))) or 1.0
    grid_max_abs = float(np.max(np.abs(our_map - native_map)))
    grid_relative_l2 = float(
        np.linalg.norm(our_map - native_map) / (np.linalg.norm(native_map) or 1.0)
    )
    native_peak_index = np.unravel_index(int(np.argmax(native_map)), native_map.shape)

    measured = {
        "peak_intensity": ours.peak_intensity,
        "peak_index": list(ours.peak_index),
        "strehl_ratio": ours.strehl_ratio,
        "num_rays_used": ours.num_rays,
        "image_shape": list(ours.image_shape),
        "pixel_pitch_m": ours.pixel_pitch_m,
        "normalization": ours.normalization,
    }
    expected = {
        "peak_intensity": float(np.max(native_map)),
        "peak_index": [int(native_peak_index[0]), int(native_peak_index[1])],
        "strehl_ratio": float(native.strehl_ratio()),
        "num_rays_used": int(native.num_rays),
        "image_shape": list(native_map.shape),
        "normalization": "strehl_percent",
    }
    deltas = {
        "peak_intensity_relative": _relative(
            measured["peak_intensity"], expected["peak_intensity"]
        ),
        "grid_max_abs_over_peak": grid_max_abs / scale,
        "grid_relative_l2": grid_relative_l2,
        "peak_index_shift": float(
            max(
                abs(a - b)
                for a, b in zip(measured["peak_index"], expected["peak_index"], strict=True)
            )
        ),
    }
    worst = max(deltas["peak_intensity_relative"], deltas["grid_relative_l2"])

    return Row(
        case=case,
        configuration={
            "method": method,
            "field_deg": list(field_deg),
            "normalized_field": list(normalized),
            "wavelength_um": wavelength_um,
            "num_rays_requested": num_rays,
        },
        descriptor="SOM_PSF",
        status="PASS" if worst <= 1e-12 and deltas["peak_index_shift"] == 0.0 else "FAIL",
        measured=measured,
        expected=expected,
        deltas=deltas,
        worst_relative_delta=worst,
        runtime_s=runtime_s,
        note=(
            "the notebook's own PSF class vs backends.optiland.analysis:psf at the same method, "
            "field, wavelength and pupil sampling. Both sides are Optiland's Strehl-percent "
            "normalization, so no rescale is applied and none is needed; a comparison against "
            "measurements.psf would need one. Plumbing regression, not validation"
        ),
    )


def _refusal_row(*, case: str, descriptor: str, what: str, error: BaseException) -> Row:
    """A refusal the tree raised, recorded as a result rather than as prose.

    CHE-239 §A.2: a documented refusal is `PASS-refused`, not a failure. The
    exception type and its message are carried verbatim so a reader can tell a
    designed refusal from an accident.
    """
    return Row(
        case=case,
        configuration={"attempted": what},
        descriptor=descriptor,
        status="PASS-refused",
        measured={"exception": type(error).__name__, "message": str(error)},
        expected={"refusal": "documented"},
        deltas={},
        worst_relative_delta=0.0,
        runtime_s=0.0,
        note="the tree refused this configuration by design; the raised exception is the evidence",
    )


def _blocked_row(*, case: str, descriptor: str, error: Unexpressible) -> Row:
    """A system `problems.OpticalSetup` cannot express."""
    return Row(
        case=case,
        configuration={},
        descriptor=descriptor,
        status="BLOCKED",
        measured={"category": error.category, "detail": error.detail},
        expected={},
        deltas={},
        worst_relative_delta=0.0,
        runtime_s=0.0,
        note=(
            "the setup schema cannot express this system. CHE-239 §A.2: determine "
            "expressibility first and do not invent a representation during this run"
        ),
    )


# --- the eleven cases -------------------------------------------------------


def case_spot() -> list[Row]:
    """`gallery/analysis/spot.ipynb` -- CookeTriplet, `SpotDiagram(lens)`.

    The notebook runs it twice: once at `fields=[(0, 1)]`, the maximum field, and
    once over all three fields at all three wavelengths.
    """
    from optiland.samples.objectives import CookeTriplet

    lens = CookeTriplet()
    setup, _ = extract_setup(lens, name="CookeTriplet")
    fields = sample_fields_deg(lens)
    wavelengths = sample_wavelengths_um(lens)

    rows: list[Row] = []
    try:
        SourceSpec(wavelength_um=wavelengths, field_angle_deg=fields)  # type: ignore[arg-type]
    except ValueError as error:
        rows.append(
            _refusal_row(
                case="spot",
                descriptor="SOM_SPOT_DIAGRAM",
                what=(
                    f"one source declaring {len(fields)} fields and "
                    f"{len(wavelengths)} wavelengths"
                ),
                error=error,
            )
        )
    for field in fields:
        for wavelength in wavelengths:
            rows.append(
                _spot_comparison(
                    case="spot",
                    lens=lens,
                    setup=setup,
                    field_deg=field,
                    wavelength_um=wavelength,
                )
            )
    return rows


def case_rms_spot_size_vs_field() -> list[Row]:
    """`gallery/analysis/rms_spot_size_vs_field.ipynb` -- TessarLens, RMS vs field.

    `RmsSpotSizeVsField` sweeps the field and reports one RMS radius per (field,
    wavelength). The curve has no descriptor; each of its points does. The sweep
    is subsampled to its endpoints and midpoint, because the notebook's headline
    is the shape of the curve and 20 field points at three wavelengths is sixty
    traces for three that answer the same question -- see the ticket's cheap-probe
    rule.
    """
    from optiland.analysis import RmsSpotSizeVsField
    from optiland.samples.objectives import TessarLens

    lens = TessarLens()
    setup, _ = extract_setup(lens, name="TessarLens")
    sweep = RmsSpotSizeVsField(lens)
    max_field = float(lens.fields.max_field)
    normalized_fields = list(sweep.fields)
    # `RmsSpotSizeVsField.fields` is a list of `FieldPoint(coord=(Hx, Hy), weight=)`
    # rather than the bare pairs `SpotDiagram(fields=...)` takes -- measured, and
    # the reason this unpacks `.coord` instead of the element.
    indices = (0, len(normalized_fields) // 2, len(normalized_fields) - 1)
    wavelengths = sample_wavelengths_um(lens)

    rows: list[Row] = []
    for index in indices:
        hx, hy = normalized_fields[index].coord
        field_deg = (float(hx) * max_field, float(hy) * max_field)
        for wavelength in wavelengths:
            rows.append(
                _spot_comparison(
                    case="rms_spot_size_vs_field",
                    lens=lens,
                    setup=setup,
                    field_deg=field_deg,
                    wavelength_um=wavelength,
                )
            )
    return rows


def case_through_focus_spot_diagram() -> list[Row]:
    """`gallery/analysis/through_focus_spot_diagram.ipynb` -- CookeTriplet, 5 focus steps.

    `num_steps=5, delta_focus=0.1` puts the image plane at
    `nominal + (i - 2) * 0.1` mm, read out of the pinned class rather than
    assumed. CHE-239 §A.4 asks for one graph per step; here that is one setup per
    step with the last surface's thickness lengthened by the offset.

    The full notebook is 5 steps x 3 fields x 3 wavelengths = 45 cells. The
    on-axis field at the primary wavelength is run at all five steps, and the
    remaining two fields at the central and both extreme steps: the defocus
    behaviour is what this notebook is about, and 45 spot analyses to establish
    that the plane moved is the sweep the ticket's cheap-probe rule excludes.
    """
    from optiland.samples.objectives import CookeTriplet

    lens = CookeTriplet()
    setup, _ = extract_setup(lens, name="CookeTriplet")
    fields = sample_fields_deg(lens)
    primary = next(
        float(wavelength.value)
        for wavelength in lens.wavelengths.wavelengths
        if bool(wavelength.is_primary)
    )
    num_steps, delta_focus = 5, 0.1
    offsets = [(index - num_steps // 2) * delta_focus for index in range(num_steps)]

    rows: list[Row] = []
    for offset in offsets:
        rows.append(
            _spot_comparison(
                case="through_focus_spot_diagram",
                lens=lens,
                setup=setup,
                field_deg=fields[0],
                wavelength_um=primary,
                defocus_mm=offset,
            )
        )
    for field in fields[1:]:
        for offset in (offsets[0], offsets[len(offsets) // 2], offsets[-1]):
            rows.append(
                _spot_comparison(
                    case="through_focus_spot_diagram",
                    lens=lens,
                    setup=setup,
                    field_deg=field,
                    wavelength_um=primary,
                    defocus_mm=offset,
                )
            )
    return rows


def case_cylindrical_lens() -> list[Row]:
    """`gallery/basic_lenses/cylindrical_lens.ipynb` -- a toroidal surface, built from scratch.

    The notebook's surface takes `radius_x=30, radius_y=40` on
    `surface_type="toroidal"`. `SurfaceSpec` carries one radius, so this is a
    geometry the schema cannot express, and CHE-239 forbids inventing one during
    this run. The lens is still constructed natively so the refusal is measured
    against the real object rather than against the notebook source.
    """
    import optiland.backend as be
    from optiland.optic import Optic

    lens = Optic()
    lens.surfaces.add(index=0, radius=be.inf, thickness=be.inf)
    lens.surfaces.add(
        index=1,
        thickness=7,
        radius_x=30,
        radius_y=40,
        is_stop=True,
        material="N-BK7",
        surface_type="toroidal",
        conic=0.0,
        toroidal_coeffs_poly_y=[],
    )
    lens.surfaces.add(index=2, thickness=65)
    lens.surfaces.add(index=3)
    lens.set_aperture(aperture_type="EPD", value=20.0)
    lens.fields.set_type(field_type="angle")
    lens.fields.add(y=0)
    lens.wavelengths.add(value=0.587, is_primary=True)

    try:
        extract_setup(lens, name="cylindrical_lens")
    except Unexpressible as error:
        return [_blocked_row(case="cylindrical_lens", descriptor="SOM_SPOT_DIAGRAM", error=error)]
    raise AssertionError(
        "the toroidal surface extracted without refusing; SurfaceSpec has one radius and this "
        "notebook declares two, so either the extractor or the schema changed"
    )


def case_nurbs_parabolic_mirror() -> list[Row]:
    """`gallery/freeform/nurbs_parabolic_mirror.ipynb` -- a NURBS freeform mirror.

    Two independent reasons the schema cannot express it: the geometry is a NURBS
    patch, and the surface is reflective. Whichever the extractor reports first is
    sufficient; the row carries the category so the aggregate count is right.
    """
    import optiland.backend as be
    from optiland.coordinate_system import CoordinateSystem
    from optiland.geometries import NurbsGeometry
    from optiland.materials import IdealMaterial
    from optiland.optic import Optic
    from optiland.surfaces import Surface

    cs = CoordinateSystem(x=0, y=0, z=0, rx=0, ry=0, rz=0, reference_cs=None)
    geometry = NurbsGeometry(
        coordinate_system=cs, radius=-100.0, conic=-1, n_points_u=7, n_points_v=7
    )
    # The notebook (upstream `master`, the pinned commit) writes
    # `Surface(geometry=..., material_pre=..., material_post=..., is_stop=True)`.
    # The **installed** optiland 0.6.0 signature is
    # `Surface(previous_surface, material_post, geometry, is_stop=..., ...)` with
    # no `material_pre` at all, so the notebook cell raises `TypeError` here.
    # CHE-238 §3.1 classifies exactly this as an *environment finding* -- pin
    # versus release drift -- and not a physics failure, and the case still has to
    # reach its real answer, so the surface is built with the installed signature
    # and the drift is carried on the row.
    surface = Surface(
        previous_surface=None,
        material_post=IdealMaterial(n=1.0),
        geometry=geometry,
        is_stop=True,
    )
    surface.interaction_model.is_reflective = True

    lens = Optic()
    lens.surfaces.add(index=0, radius=be.inf, thickness=be.inf)
    lens.surfaces.add(index=1, new_surface=surface, thickness=-50)
    lens.surfaces.add(index=2)
    lens.set_aperture(aperture_type="EPD", value=20.0)
    lens.fields.set_type(field_type="angle")
    lens.fields.add(y=0)
    lens.wavelengths.add(value=0.55, is_primary=True)
    lens.update_paraxial()

    try:
        extract_setup(lens, name="nurbs_parabolic_mirror")
    except Unexpressible as error:
        row = _blocked_row(
            case="nurbs_parabolic_mirror", descriptor="SOM_SPOT_DIAGRAM", error=error
        )
        return [
            Row(
                case=row.case,
                configuration=row.configuration,
                descriptor=row.descriptor,
                status=row.status,
                measured=row.measured,
                expected=row.expected,
                deltas=row.deltas,
                worst_relative_delta=row.worst_relative_delta,
                runtime_s=row.runtime_s,
                note=row.note,
                extra={
                    "environment_finding": (
                        "the notebook at the pinned upstream commit calls "
                        "Surface(geometry=..., material_pre=..., material_post=..., is_stop=...); "
                        "installed optiland 0.6.0 takes "
                        "Surface(previous_surface, material_post, geometry, ...) and has no "
                        "material_pre, so the notebook cell raises TypeError as written. Built "
                        "with the installed signature instead; CHE-238 §3.1 classifies this as "
                        "pin-vs-release drift, not a physics failure"
                    )
                },
            )
        ]
    raise AssertionError(
        "a reflective NURBS surface extracted without refusing; the schema has neither a "
        "freeform sag nor a reflective flag, so the extractor changed"
    )


def _psf_case(
    *, case: str, sample: str, module: str, method: str, field_hy: float, num_rays: int
) -> list[Row]:
    """The six PSF notebooks, which differ only in sample, method, field and sampling."""
    import importlib

    lens = getattr(importlib.import_module(module), sample)()
    setup, _ = extract_setup(lens, name=sample)
    max_field = float(lens.fields.max_field)
    field_deg = (0.0, field_hy * max_field)
    return [
        _psf_comparison(
            case=case,
            lens=lens,
            setup=setup,
            method=method,
            field_deg=field_deg,
            wavelength_um=0.55,
            num_rays=num_rays,
        )
    ]


#: The eleven, in the order CHE-239 §A.4 tables them. Each entry is a thunk so a
#: failure in one case does not prevent the others from running.
CASES: dict[str, Callable[[], list[Row]]] = {
    "rms_spot_size_vs_field": case_rms_spot_size_vs_field,
    "spot": case_spot,
    "through_focus_spot_diagram": case_through_focus_spot_diagram,
    "cylindrical_lens": case_cylindrical_lens,
    "nurbs_parabolic_mirror": case_nurbs_parabolic_mirror,
    "fft_psf_2d": lambda: _psf_case(
        case="fft_psf_2d",
        sample="CookeTriplet",
        module="optiland.samples.objectives",
        method="fft",
        field_hy=0.0,
        num_rays=PSF_DEFAULT_NUM_RAYS,
    ),
    "fft_psf_3d": lambda: _psf_case(
        case="fft_psf_3d",
        sample="TessarLens",
        module="optiland.samples.objectives",
        method="fft",
        field_hy=1.0,
        num_rays=PSF_DEFAULT_NUM_RAYS,
    ),
    "huygens_psf_2d": lambda: _psf_case(
        case="huygens_psf_2d",
        sample="DoubleGauss",
        module="optiland.samples.objectives",
        method="huygens",
        field_hy=0.0,
        num_rays=PSF_DEFAULT_NUM_RAYS,
    ),
    "huygens_psf_3d": lambda: _psf_case(
        case="huygens_psf_3d",
        sample="DoubleGauss",
        module="optiland.samples.objectives",
        method="huygens",
        field_hy=0.0,
        num_rays=PSF_DEFAULT_NUM_RAYS,
    ),
    "mmdft_psf_2d": lambda: _psf_case(
        case="mmdft_psf_2d",
        sample="CookeTriplet",
        module="optiland.samples.objectives",
        method="mmdft",
        field_hy=0.0,
        num_rays=512,
    ),
    "mmdft_psf_3d": lambda: _psf_case(
        case="mmdft_psf_3d",
        sample="DoubleGauss",
        module="optiland.samples.objectives",
        method="mmdft",
        field_hy=0.0,
        num_rays=512,
    ),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only", action="append", default=None, help="run only these cases (repeatable)"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/che-238-overnight/workstream-a/tier1.json"),
        help="where the record goes; the default is gitignored on purpose",
    )
    arguments = parser.parse_args(argv)

    selected = arguments.only or list(CASES)
    rows: list[Row] = []
    for name in selected:
        started = time.perf_counter()
        try:
            produced = CASES[name]()
        except Exception as error:
            produced = [
                Row(
                    case=name,
                    configuration={},
                    descriptor="",
                    status="FAIL",
                    measured={"exception": type(error).__name__, "message": str(error)},
                    expected={},
                    deltas={},
                    worst_relative_delta=math.inf,
                    runtime_s=time.perf_counter() - started,
                    note="the case raised before it could produce a comparison",
                )
            ]
        rows.extend(produced)
        elapsed = time.perf_counter() - started
        statuses = ", ".join(sorted({row.status for row in produced}))
        print(f"{name:32s} {len(produced):3d} row(s)  [{statuses}]  {elapsed:6.1f}s", flush=True)

    record = {
        "workstream": "A-tier1",
        "ticket": "CHE-239",
        "produced_by": "benchmarks/verification/ray_tier1.py",
        "upstream_pin": "optiland/optiland@00c0837fbee5d66019a24a1735ff91cd4f9b2646",
        **provenance(),
        "rows": [row.as_dict() for row in rows],
    }
    return finish(record, path=arguments.out, rows=rows)


if __name__ == "__main__":
    raise SystemExit(main())
