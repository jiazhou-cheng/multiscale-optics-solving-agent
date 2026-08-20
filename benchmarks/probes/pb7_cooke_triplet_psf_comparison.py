"""PB7 (CHE-58): three PSF routes on the same Cooke Triplet, side by side.

What this is
------------
A **standalone diagnostic benchmark**. It runs three independent diffraction
paths on one optical system at one wavelength and two fields, puts them into a
single declared coordinate frame, and reports what they agree and disagree
about. It sets no tolerance, gates nothing, is not imported by any test, and
changes no coupler, adapter, or measurement physics.

    Method A   Optiland ``psf.FFTPSF``       -- FFT of the exit-pupil function
    Method B   Optiland ``psf.HuygensPSF``   -- direct Huygens-Fresnel summation
    Method C   Optiland trace -> C_RAY_TO_WAVE -> Chromatix ASM -> |U|^2

A and B are two independent implementations inside one package; C is this
repository's own ray-to-wave route. So the A-vs-B residual is the *floor* this
comparison can resolve, and C is only interpretable against it. That is why all
three appear in every pairwise number below rather than C being compared to a
single "reference".

The optical system: one prescription, verified identical
-------------------------------------------------------
``optiland.samples.objectives.CookeTriplet`` is a bundled sample class, and the
repository's Optiland adapter deliberately does not construct systems from
bundled samples (CHE-56): it builds from a canonical ``OpticalSystemSpec``. So
this benchmark transcribes the Cooke Triplet into that schema
(:func:`cooke_triplet_spec`) and then **checks** the transcription rather than
trusting it -- surface positions, paraxial f, EPD/EPL/XPD/XPL/FNO, and every
traced ray array at both fields must be bit-identical to the bundled class
(:func:`verify_system_equivalence`). Methods A and B run on the bundled object,
Method C on the built one. If that check ever fails the run aborts, because a
three-way PSF comparison across two different lenses is not a comparison.

The one wavelength and the two fields
-------------------------------------
0.55 um, the system's own primary wavelength. Fields ``(0, 0)`` and ``(0, 1.0)``
in normalized field coordinates; the Cooke Triplet's maximum field is 20 deg, so
the off-axis case is a **20 deg** field whose chief ray lands 18.136 mm off axis.
That is not a small perturbation of the on-axis case: the working F-number moves
from 4.978 to 5.480, and the geometric spot grows from 1.9 to ~6 Airy radii.

Coordinate frame -- the single most important declaration here
-------------------------------------------------------------
Everything is reported in the Optiland global frame at the image surface
(z = 60.17675 mm), in micrometres, **relative to the traced chief-ray
intersection** for that field. The chief ray is
``lens.trace_generic(Hx, Hy, Px=0, Py=0)``, which is exactly the point Optiland's
own ``strategy="chief_ray"`` centres its reference sphere on, so this origin is
the packages' convention rather than one invented here.

Each method reaches that frame differently, and none of the three is shifted to
match another:

* **A (FFT)** has no intrinsic coordinates. Its array centre *is* the reference
  sphere centre, i.e. the chief-ray point, and its pixel scale is Optiland's own
  ``dx = lambda * FNO_working / Q`` with ``Q = grid_size / (num_rays - 1)`` --
  the formula ``ScalarFFTPSF._get_psf_units`` uses to label its own plots. Row
  index maps to +y and column index to +x, which is asserted by this benchmark
  rather than assumed: the row-flipped residual against B is reported next to
  the unflipped one, and a wrong orientation would show up there.
* **B (Huygens)** carries absolute coordinates. Its grid is
  ``linspace(c - e, c + e, image_size)`` about ``(cx, cy)``, reproduced here from
  the object's own ``pixel_pitch``/``image_size``. Note ``(cx, cy)`` is the
  **mean intersection of a hexapolar-6 ray fan**, i.e. a centroid, *not* the
  chief ray -- off axis the two differ by 0.9 um, and that offset is reported
  rather than removed. Note also that Optiland stores
  ``pixel_pitch = 2e / image_size`` while its coordinates step by
  ``2e / (image_size - 1)``; the linspace value is the true sample spacing and is
  the one used here.
* **C (ray->wave)** is reconstructed on a grid centred on the chief-ray point.
  The recentring is an exact translation, not an interpolation: the
  ``C_RAY_TO_WAVE`` kernel
  ``U(r) = sum_i a_i exp(i k (OPL_i + d_i . (r - r_i)))`` depends on ray
  positions only through ``d_i . r_i``, so evaluating it on a grid centred at
  ``c`` is identically evaluating it at the origin with positions ``r_i - c``.
  That is what :func:`translate_bundle_transverse` does, and it is why the grid
  can sit 18 mm off axis at 0.2 um pitch.

Method C, step by step, through shipping calls only
---------------------------------------------------
1. ``optiland_adapter.run(...)`` with ``handoff_plane="exit_pupil"`` -- the only
   plane the adapter resolves -- at ``num_rays`` hexapolar rings.
2. ``declare_coherent_bundle(...)`` -- CHE-33/40/41/47: piston-removed relative
   OPL, the object-space launch-tilt term the off-axis field needs, the
   ``sqrt(intensity)`` amplitude mapping, and the CHE-47 per-ray hexapolar
   quadrature weight. Whether each was actually applied is read off the
   handoff's own diagnostics and recorded.
3. :func:`advance_bundle_to_z` -- ray-domain advance from the exit pupil to the
   declared sensor plane through image-space air, with ``n`` read off the
   adapter. This is CHE-38's selected handoff (the sensor plane itself), so no
   physics is moved into the wave domain that the ray model can do exactly.
4. :func:`translate_bundle_transverse` -- the exact recentring above.
5. ``ray_to_wave(...)`` -- the coupler core, unmodified, at the declared grid.
6. Chromatix ``asm_carrier_removed`` over a **zero** propagation distance. The
   ticket asks for the wave-propagation leg, and CHE-38 selected a handoff that
   leaves no distance to propagate; running it at ``z = 0`` exercises the
   shipping wave path and its identity residual is reported (~3e-7 relative)
   rather than the leg being skipped silently.
7. ``measure_psf_from_record(..., normalization=PEAK)`` -- the M3.7/CHE-36 frozen
   sensor-plane measurement semantics, with the output pitch checked against the
   pitch the propagation reported.

Normalization, resampling, cropping -- all of it, stated
-------------------------------------------------------
* **Intensity.** A and B are Optiland's own "relative intensity in percent",
  normalized so a diffraction-limited system peaks at 100; ``strehl_ratio()`` is
  that array's *centre pixel* over 100. C is ``|U|^2`` in uncalibrated units --
  the ray amplitudes descend from Optiland's per-ray intensity weights and no
  step converts them to watts. There is therefore **no common absolute scale**,
  and every pairwise comparison below is on peak-normalized intensity. The
  native peaks and Strehl ratios are reported separately and are *not* used to
  scale anything.
* **Peak normalization is applied twice, for two different purposes.** Once on
  each native array (for the native-grid figures), and again on the common
  comparison window (so a pairwise residual is not sensitive to structure
  outside the window). Both are recorded.
* **Cropping.** The 2048^2 FFT array is cropped to :data:`NATIVE_CROP_HALF_WIDTH_M`
  about its centre index for figures and resampling. The crop is symmetric about
  index ``n // 2``, so the coordinate mapping is unchanged; nothing is shifted.
* **The common comparison grid** is one declared grid per field:
  :data:`COMMON_PITCH_M` pitch, centred on the chief-ray point, with a half-width
  chosen by a rule fixed before the run -- the largest whole micrometre that fits
  inside *every* method's own window, so no method is ever extrapolated. In
  practice B's automatic extent always binds. All three are resampled onto it by
  the same linear ``RegularGridInterpolator``; C is not privileged by being the
  grid's owner.
* **Peak-aligned residuals** are reported *in addition to* the in-frame ones, as
  a diagnostic that separates "these have different shapes" from "these sit in
  different places". The integer shift applied is recorded. The in-frame number
  is the primary one.

What this benchmark does not do
------------------------------
No tolerance is set, no pass/fail is declared, no wavelength or field sweep is
run, and no convergence study beyond a single two-ray-count stability check on
Method C whose only job is to say whether the picture would move if the ray
count changed. Nothing here is fitted to the result. Discrepancies are
characterized and attributed to candidate causes; none is fixed.

Run it:

    ./run.sh python benchmarks/probes/pb7_cooke_triplet_psf_comparison.py

Writes ``outputs/PB7/`` -- six figures and one JSON record.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field as dataclass_field
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "outputs" / "PB7"

# ---------------------------------------------------------------------------
# DECLARED BEFORE THE RUN. Nothing below is chosen after looking at a result.
# ---------------------------------------------------------------------------

#: The one wavelength, in the Optiland native unit and in SI. It is the Cooke
#: Triplet's own primary wavelength, so no method needs a wavelength override.
WAVELENGTH_UM = 0.55
WAVELENGTH_M = WAVELENGTH_UM * 1.0e-6

#: The two fields, in normalized field coordinates. The Cooke Triplet's maximum
#: field is 20 deg, so Hy = 1.0 is a 20 deg field, not a small excursion.
FIELDS: tuple[tuple[str, float, float], ...] = (
    ("on_axis", 0.0, 0.0),
    ("full_field", 0.0, 1.0),
)

#: The image surface, read from the prescription's own surface positions and
#: asserted against this value at run time rather than trusted.
IMAGE_Z_MM = 60.17675
#: Exit pupil: image_z + XPL(), signed and measured from the image surface. Also
#: asserted against the adapter's own reading.
EXIT_PUPIL_Z_MM = IMAGE_Z_MM - 50.961347703805274

#: Method A. `grid_size` is given explicitly, which switches off Optiland's
#: OpticStudio-emulating auto-sampling and makes both Q and the pixel scale
#: declared quantities. num_rays sets the window (lambda * FNO * (num_rays - 1)
#: = 348 um on axis); grid_size sets the pitch inside it (0.170 um on axis, i.e.
#: ~20 samples per Airy first-null radius).
FFT_NUM_RAYS = 128
FFT_GRID_SIZE = 2048
#: Optiland's defaults, restated so they are declarations and not inheritances.
#: "chief_ray" is what makes the array centre the chief-ray intersection.
FFT_STRATEGY = "chief_ray"
FFT_REMOVE_TILT = False

#: Method B. Same pupil sampling as A so the two differ in algorithm and not in
#: pupil density. image_size 256 (over Optiland's automatic extent) gives
#: 0.13 um on axis and 0.18 um off axis.
HUYGENS_NUM_RAYS = 128
HUYGENS_IMAGE_SIZE = 256
HUYGENS_STRATEGY = "chief_ray"
HUYGENS_REMOVE_TILT = False

#: Method C. 64 hexapolar rings = 12481 rays. The grid: 0.2 um satisfies the
#: coupler's per-axis Nyquist limit at both fields with a wide margin (the
#: steepest off-axis ramp is |d_y| = 0.408 against a limit of 1.375) and gives
#: ~17 samples per Airy first-null radius; 512 samples then span 102.4 um, which
#: holds the off-axis geometric spot (~24 um) with room for its wings.
RAY_TO_WAVE_RINGS = 64
#: The second ray count, for the stability check only. Not a convergence study:
#: two points cannot establish a rate, and none is claimed.
RAY_TO_WAVE_STABILITY_RINGS = 32
SENSOR_PITCH_M = 0.2e-6
SENSOR_GRID_N = 512

#: How much of each native array is kept for figures and resampling. Symmetric
#: about the centre index, so no coordinate is shifted by cropping.
NATIVE_CROP_HALF_WIDTH_M = 40.0e-6

#: The common comparison grid: pitch, and the rule that fixes its half-width.
COMMON_PITCH_M = 0.2e-6
COMMON_HALF_WIDTH_RULE = (
    "the largest whole micrometre strictly inside every method's own native "
    "half-window, so that no method is ever extrapolated. Applied per field."
)

#: Encircled-energy fractions reported. Window-dependent by construction.
ENCIRCLED_ENERGY_FRACTIONS = (0.5, 0.8)

#: Log-scale floor for the log panels, in units of each map's own peak.
LOG_FLOOR = 1.0e-5


# ---------------------------------------------------------------------------
# 1. The optical system: one prescription, checked against the bundled sample
# ---------------------------------------------------------------------------
def cooke_triplet_spec():
    """The bundled ``CookeTriplet`` transcribed into the canonical schema.

    Every radius, thickness, glass, aperture, field and wavelength is the
    bundled class's own value. The object and image surfaces are added by
    ``optiland_builder``, which is why only the six optical surfaces appear.

    ``expected_catalog_file`` is not decoration. Optiland resolves a bare glass
    name by substring filter plus Levenshtein ranking, and bare ``"SK16"``
    selects HIKARI's ``SK16`` rather than SCHOTT's ``N-SK16`` -- exactly what the
    bundled sample gets, and recorded here so a future catalog change is an
    error instead of a quietly different lens.
    """
    from multiscale_optics_agent.core.optical_system import (
        ApertureSpec,
        CatalogMaterialSpec,
        FieldSpec,
        OpticalSystemSpec,
        SphericalGeometrySpec,
        SurfaceSpec,
        WavelengthSpec,
    )

    def surface(radius_mm: float, thickness_mm: float, material=None, is_stop: bool = False):
        extra = {"material": material} if material is not None else {}
        return SurfaceSpec(
            geometry=SphericalGeometrySpec(radius_mm=radius_mm),
            thickness_mm=thickness_mm,
            is_stop=is_stop,
            **extra,
        )

    sk16 = CatalogMaterialSpec(name="SK16", expected_catalog_file="glass/hikari/SK16.yml")
    f2 = CatalogMaterialSpec(
        name="F2", catalog="schott", expected_catalog_file="glass/schott/F2.yml"
    )
    return OpticalSystemSpec(
        name="CookeTriplet",
        description=(
            "optiland.samples.objectives.CookeTriplet transcribed into the canonical "
            "prescription schema for PB7 (CHE-58). Verified bit-identical to the "
            "bundled class at run time."
        ),
        object_distance_mm=None,  # object at infinity
        surfaces=(
            surface(22.01359, 3.25896, sk16),
            surface(-435.76044, 6.00755),
            surface(-22.21328, 0.99997, f2),
            surface(20.29192, 4.75041, is_stop=True),
            surface(79.68360, 2.95208, sk16),
            surface(-18.39533, 42.20778),
        ),
        aperture=ApertureSpec(value_mm=10.0),
        fields=(FieldSpec(y_deg=0.0), FieldSpec(y_deg=14.0), FieldSpec(y_deg=20.0)),
        wavelengths=(
            WavelengthSpec(value_um=0.48),
            WavelengthSpec(value_um=0.55, is_primary=True),
            WavelengthSpec(value_um=0.65),
        ),
    )


_PARAXIAL_KEYS = ("f2", "EPD", "EPL", "XPD", "XPL", "FNO")
_RAY_ARRAYS = ("x", "y", "z", "L", "M", "N", "i", "opd")


def verify_system_equivalence(built: Any, bundled: Any) -> dict[str, Any]:
    """The built prescription and the bundled sample must be the same lens.

    Three levels, because any one of them alone can pass on a different system:
    surface positions (geometry), the paraxial set (first-order behaviour), and
    every traced ray array at both benchmark fields (the actual numbers each
    method consumes). Equality is exact -- ``array_equal``, not ``allclose`` --
    because the construction paths differ only in bookkeeping and any real
    numerical difference here would be a defect, not a tolerance question.
    """
    report: dict[str, Any] = {"levels": {}}

    positions_built = [float(np.asarray(z).ravel()[0]) for z in built.surfaces.positions]
    positions_bundled = [float(np.asarray(z).ravel()[0]) for z in bundled.surfaces.positions]
    report["levels"]["surface_positions_mm"] = {
        "built": positions_built,
        "bundled": positions_bundled,
        "identical": positions_built == positions_bundled,
    }

    paraxial: dict[str, Any] = {}
    for key in _PARAXIAL_KEYS:
        a = float(np.asarray(getattr(built.paraxial, key)()).ravel()[0])
        b = float(np.asarray(getattr(bundled.paraxial, key)()).ravel()[0])
        paraxial[key] = {"built": a, "bundled": b, "identical": a == b}
    report["levels"]["paraxial"] = paraxial

    traces: dict[str, Any] = {}
    for name, hx, hy in FIELDS:
        rays_built = built.trace(Hx=hx, Hy=hy, wavelength=WAVELENGTH_UM, num_rays=16)
        rays_bundled = bundled.trace(Hx=hx, Hy=hy, wavelength=WAVELENGTH_UM, num_rays=16)
        per_field: dict[str, Any] = {}
        for attr in _RAY_ARRAYS:
            u = np.asarray(getattr(rays_built, attr), dtype=np.float64)
            v = np.asarray(getattr(rays_bundled, attr), dtype=np.float64)
            per_field[attr] = {
                "identical": bool(np.array_equal(u, v)),
                "max_abs_difference": float(np.max(np.abs(u - v))) if u.shape == v.shape else None,
                "count": int(u.size),
            }
        traces[name] = per_field
    report["levels"]["traced_rays"] = traces

    report["all_identical"] = bool(
        report["levels"]["surface_positions_mm"]["identical"]
        and all(entry["identical"] for entry in paraxial.values())
        and all(
            entry["identical"]
            for per_field in traces.values()
            for entry in per_field.values()
        )
    )
    report["why_this_check_exists"] = (
        "Methods A and B run on the bundled sample class; Method C runs on the "
        "canonical prescription the adapter builds (CHE-56: the adapter does not "
        "construct from bundled samples). Without this check a three-way PSF "
        "comparison could silently be a comparison across two lenses."
    )
    return report


def chief_ray_image_point(lens: Any, hx: float, hy: float) -> dict[str, float]:
    """The Px = Py = 0 ray's intersection with the image surface, in metres.

    This is the origin of every coordinate this benchmark reports, and it is
    Optiland's own choice: ``ChiefRayStrategy._create_spherical_ref`` centres the
    reference sphere on exactly this point, so Methods A and B are already
    referred to it.
    """
    ray = lens.trace_generic(hx, hy, Px=0.0, Py=0.0, wavelength=WAVELENGTH_UM)
    return {
        "x_m": float(np.asarray(ray.x).ravel()[0]) * 1.0e-3,
        "y_m": float(np.asarray(ray.y).ravel()[0]) * 1.0e-3,
        "z_m": float(np.asarray(ray.z).ravel()[0]) * 1.0e-3,
    }


# ---------------------------------------------------------------------------
# 2. A PSF map: an intensity array plus the coordinates it actually lives on
# ---------------------------------------------------------------------------
@dataclass
class PsfMap:
    """One method's result, on its own grid, in one declared frame.

    ``y_rel_m``/``x_rel_m`` are metres from the chief-ray intersection, row-major
    ``(y, x)``, increasing index = increasing coordinate on both axes.
    ``intensity`` is peak-normalized on this array; ``native_peak`` is what that
    normalization removed.
    """

    method: str
    label: str
    intensity: np.ndarray
    y_rel_m: np.ndarray
    x_rel_m: np.ndarray
    pitch_y_m: float
    pitch_x_m: float
    native_peak: float
    native_peak_units: str
    provenance: dict[str, Any] = dataclass_field(default_factory=dict)

    @property
    def half_window_m(self) -> float:
        """Largest radius about the chief ray fully inside this array."""
        return float(
            min(
                abs(self.y_rel_m[0]),
                abs(self.y_rel_m[-1]),
                abs(self.x_rel_m[0]),
                abs(self.x_rel_m[-1]),
            )
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "label": self.label,
            "shape": [int(n) for n in self.intensity.shape],
            "sample_pitch_m": [self.pitch_y_m, self.pitch_x_m],
            "half_window_m": self.half_window_m,
            "native_peak": self.native_peak,
            "native_peak_units": self.native_peak_units,
            "y_rel_range_m": [float(self.y_rel_m[0]), float(self.y_rel_m[-1])],
            "x_rel_range_m": [float(self.x_rel_m[0]), float(self.x_rel_m[-1])],
            **self.provenance,
        }


def _crop_symmetric(array: np.ndarray, half_width_m: float, pitch_m: float) -> tuple[np.ndarray, int]:
    """Crop about index ``n // 2``, keeping that index's coordinate meaning.

    Returns the cropped array and the index of the old centre inside it, so the
    caller can rebuild coordinates without re-deriving the offset.
    """
    n = array.shape[0]
    centre = n // 2
    reach = int(np.floor(half_width_m / pitch_m))
    lo = max(0, centre - reach)
    hi = min(n, centre + reach + 1)
    return array[lo:hi, lo:hi], centre - lo


# ---------------------------------------------------------------------------
# 3. Method A -- Optiland FFT PSF
# ---------------------------------------------------------------------------
def method_a_fft(lens: Any, hx: float, hy: float, chief: dict[str, float]) -> PsfMap:
    """``psf.FFTPSF`` with an explicitly declared grid, placed in the frame.

    Pixel scale is Optiland's own: ``dx = lambda * FNO_working / Q`` with
    ``Q = grid_size / (num_rays - 1)``, taken from
    ``ScalarFFTPSF._get_psf_units``. The array centre index ``n // 2`` is the
    reference-sphere centre, which under ``strategy="chief_ray"`` is the
    chief-ray intersection -- so the frame offset is exactly zero and no shift is
    applied.
    """
    from optiland import psf as optiland_psf
    from optiland.utils import get_working_FNO

    started = time.time()
    obj = optiland_psf.FFTPSF(
        lens,
        field=(hx, hy),
        wavelength=WAVELENGTH_UM,
        num_rays=FFT_NUM_RAYS,
        grid_size=FFT_GRID_SIZE,
        strategy=FFT_STRATEGY,
        remove_tilt=FFT_REMOVE_TILT,
    )
    elapsed = time.time() - started

    raw = np.asarray(obj.psf, dtype=np.float64)
    working_fno = float(np.asarray(get_working_FNO(lens, (hx, hy), WAVELENGTH_UM)).ravel()[0])
    # Cross-check against the object's own private accessor: the pixel scale is
    # the whole coordinate system for this method, so it is read twice.
    working_fno_internal = float(np.asarray(obj._get_working_FNO()).ravel()[0])
    q_factor = obj.grid_size / (obj.num_rays - 1)
    pitch_m = WAVELENGTH_UM * working_fno / q_factor * 1.0e-6

    strehl = float(obj.strehl_ratio())
    native_peak = float(raw.max())
    centre_value = float(raw[raw.shape[0] // 2, raw.shape[1] // 2])

    cropped, centre_index = _crop_symmetric(raw, NATIVE_CROP_HALF_WIDTH_M, pitch_m)
    axis = (np.arange(cropped.shape[0], dtype=np.float64) - centre_index) * pitch_m

    return PsfMap(
        method="A",
        label="Optiland FFT PSF",
        intensity=cropped / cropped.max(),
        y_rel_m=axis,
        x_rel_m=axis.copy(),
        pitch_y_m=pitch_m,
        pitch_x_m=pitch_m,
        native_peak=native_peak,
        native_peak_units=(
            "Optiland relative intensity in percent; 100 = the peak of the "
            "diffraction-limited system with the same pupil amplitude"
        ),
        provenance={
            "engine": "optiland.psf.FFTPSF (ScalarFFTPSF)",
            "num_rays_across_pupil_diameter": int(obj.num_rays),
            "fft_grid_size": int(obj.grid_size),
            "strategy": FFT_STRATEGY,
            "remove_tilt": FFT_REMOVE_TILT,
            "working_fno": working_fno,
            "working_fno_from_object": working_fno_internal,
            "working_fno_agrees": bool(np.isclose(working_fno, working_fno_internal, rtol=0.0, atol=0.0)),
            "q_factor": float(q_factor),
            "pixel_scale_formula": "dx = lambda * FNO_working / Q, Q = grid_size / (num_rays - 1)",
            "native_full_window_m": float(raw.shape[0] * pitch_m),
            "native_shape": [int(n) for n in raw.shape],
            "strehl_ratio": strehl,
            "centre_pixel_value": centre_value,
            "peak_over_centre": native_peak / centre_value if centre_value > 0 else None,
            "frame_offset_applied_m": [0.0, 0.0],
            "frame_offset_justification": (
                "the array centre is the reference-sphere centre, which under "
                "strategy='chief_ray' is the chief-ray image intersection -- the "
                "origin of this benchmark's frame. No shift is applied."
            ),
            "cropped_from_native": True,
            "crop_half_width_m": NATIVE_CROP_HALF_WIDTH_M,
            "seconds": elapsed,
        },
    )


# ---------------------------------------------------------------------------
# 4. Method B -- Optiland Huygens PSF
# ---------------------------------------------------------------------------
def method_b_huygens(lens: Any, hx: float, hy: float, chief: dict[str, float]) -> PsfMap:
    """``psf.HuygensPSF``, with its absolute grid rebuilt exactly.

    ``ScalarHuygensPSF._get_image_coordinates`` builds
    ``linspace(c - e, c + e, image_size)`` on each axis, with
    ``e = pixel_pitch * image_size / 2`` and ``(cx, cy)`` the mean image-surface
    intersection of a hexapolar ``num_rays=6`` fan restricted to rays with
    positive intensity. Both are read back off the object here, so this is the
    grid the summation actually ran on and not a reconstruction of it.

    Two things are recorded rather than smoothed over:

    * ``(cx, cy)`` is a **centroid**, not the chief ray, despite
      ``strategy="chief_ray"`` governing the reference sphere. Off axis the two
      differ, and the offset is reported.
    * ``pixel_pitch`` (``2e / image_size``) is not the coordinate step
      (``2e / (image_size - 1)``). At image_size 256 they differ by 0.4%. The
      linspace step is the true sample spacing and the one used.
    """
    from optiland import psf as optiland_psf

    started = time.time()
    obj = optiland_psf.HuygensPSF(
        lens,
        field=(hx, hy),
        wavelength=WAVELENGTH_UM,
        num_rays=HUYGENS_NUM_RAYS,
        image_size=HUYGENS_IMAGE_SIZE,
        strategy=HUYGENS_STRATEGY,
        remove_tilt=HUYGENS_REMOVE_TILT,
    )
    elapsed = time.time() - started

    raw = np.asarray(obj.psf, dtype=np.float64)
    size = int(obj.image_size)
    stored_pitch_mm = float(np.asarray(obj.pixel_pitch).ravel()[0])
    half_extent_mm = stored_pitch_mm * size / 2.0
    cx_mm = float(np.asarray(obj.cx).ravel()[0])
    cy_mm = float(np.asarray(obj.cy).ravel()[0])

    x_abs_m = np.linspace(cx_mm - half_extent_mm, cx_mm + half_extent_mm, size) * 1.0e-3
    y_abs_m = np.linspace(cy_mm - half_extent_mm, cy_mm + half_extent_mm, size) * 1.0e-3
    coordinate_step_m = float(x_abs_m[1] - x_abs_m[0])

    strehl = float(obj.strehl_ratio())
    native_peak = float(raw.max())
    centre_value = float(raw[size // 2, size // 2])

    return PsfMap(
        method="B",
        label="Optiland Huygens PSF",
        intensity=raw / native_peak,
        y_rel_m=y_abs_m - chief["y_m"],
        x_rel_m=x_abs_m - chief["x_m"],
        pitch_y_m=coordinate_step_m,
        pitch_x_m=coordinate_step_m,
        native_peak=native_peak,
        native_peak_units=(
            "Optiland relative intensity in percent; 100 = the peak of the "
            "diffraction-limited Huygens sum over the same pupil"
        ),
        provenance={
            "engine": "optiland.psf.HuygensPSF (ScalarHuygensPSF, Numba summation)",
            "num_rays_across_pupil_diameter": int(obj.num_rays),
            "image_size": size,
            "strategy": HUYGENS_STRATEGY,
            "remove_tilt": HUYGENS_REMOVE_TILT,
            "extent_mode": "automatic (max of geometric footprint and 5 Airy radii)",
            "stored_pixel_pitch_m": stored_pitch_mm * 1.0e-3,
            "coordinate_step_m": coordinate_step_m,
            "stored_pitch_over_coordinate_step": stored_pitch_mm * 1.0e-3 / coordinate_step_m,
            "pitch_discrepancy_note": (
                "Optiland stores pixel_pitch = 2e / image_size but steps its "
                "coordinates by 2e / (image_size - 1). The linspace step is the "
                "true sample spacing and is what is used here; the stored value is "
                "what its own view() uses to label axes."
            ),
            "half_extent_m": half_extent_mm * 1.0e-3,
            "grid_centre_abs_m": [cy_mm * 1.0e-3, cx_mm * 1.0e-3],
            "grid_centre_minus_chief_ray_m": [
                cy_mm * 1.0e-3 - chief["y_m"],
                cx_mm * 1.0e-3 - chief["x_m"],
            ],
            "grid_centre_semantics": (
                "mean image-surface intersection of a hexapolar num_rays=6 fan with "
                "intensity > 0 -- a centroid, NOT the chief ray. The offset above is "
                "reported, not removed."
            ),
            "strehl_ratio": strehl,
            "centre_pixel_value": centre_value,
            "peak_over_centre": native_peak / centre_value if centre_value > 0 else None,
            "frame_offset_applied_m": [-chief["y_m"], -chief["x_m"]],
            "frame_offset_justification": (
                "this grid is absolute in the Optiland global frame, so reaching the "
                "chief-ray-referred frame is a subtraction of the chief-ray point "
                "from the axis vectors. The intensity array is untouched."
            ),
            "cropped_from_native": False,
            "seconds": elapsed,
        },
    )


# ---------------------------------------------------------------------------
# 5. Method C -- Optiland trace -> C_RAY_TO_WAVE -> Chromatix ASM -> |U|^2
# ---------------------------------------------------------------------------
def advance_bundle_to_z(bundle, z_m: float):
    """Ray-domain advance to a declared plane in image space.

    Each ray moves along its own direction by ``s = (z - z0) / d_z`` and its
    optical path grows by ``n s`` with ``n = 1`` (image space is air for this
    system; the index is read off the adapter, not assumed). Directions are
    unchanged, which is what makes this a propagation of the ray *state* rather
    than a change of the ray model, and it is exact: the resulting per-ray
    constant phase differs from the original by the phase an exact plane wave
    accumulates over the plane offset.

    Same operation as ``benchmarks/probes/m3r_sensor_handoff.py``'s
    ``_advance_bundle_to_z``; reproduced rather than imported because that probe
    is a 126 kB study module whose import runs its own declarations.
    """
    from multiscale_optics_agent.couplers.contracts import Frame, RayBundle, ReferencePlane

    positions = np.asarray(bundle.positions_m, dtype=np.float64)
    directions = np.asarray(bundle.directions, dtype=np.float64)
    amplitude, optical_path_length = bundle.require_coherent()
    if np.any(directions[:, 2] == 0.0):
        raise ValueError("a ray with d_z = 0 never reaches the declared plane")
    step = (float(z_m) - positions[:, 2]) / directions[:, 2]
    advanced = RayBundle(
        positions_m=positions + step[:, None] * directions,
        directions=directions.copy(),
        wavelength_m=bundle.wavelength_m,
        reference_plane=ReferencePlane(name="image_space_observation_plane", z_m=float(z_m)),
        frame=Frame(),
        amplitude=np.asarray(amplitude).copy(),
        optical_path_length_m=np.asarray(optical_path_length) + step,
        optical_path_length_reference=(
            f"{bundle.optical_path_length_reference}, then advanced along each ray to "
            f"z = {float(z_m)!r} m through image-space air (n = 1)"
        ),
    )
    return advanced, step


def translate_bundle_transverse(bundle, dx_m: float, dy_m: float):
    """Recentre the reconstruction window, exactly, by translating the rays.

    ``ray_to_wave`` puts coordinate zero at index ``n // 2``, so a PSF 18 mm off
    axis cannot be reached by the grid at a 0.2 um pitch. The kernel is

        U(r) = sum_i a_i exp(i k (OPL_i + d_i . (r - r_i)))

    which depends on ray positions only through ``d_i . r_i``. Hence

        U(r + c) = sum_i a_i exp(i k (OPL_i + d_i . (r - (r_i - c))))

    identically -- evaluating the field on a grid centred at ``c`` *is*
    evaluating it at the origin with positions ``r_i - c``. Nothing is
    interpolated, resampled or approximated, and no optical path is altered:
    only the transverse components of ``positions_m`` change.
    """
    from multiscale_optics_agent.couplers.contracts import Frame, RayBundle

    positions = np.asarray(bundle.positions_m, dtype=np.float64).copy()
    positions[:, 0] -= float(dx_m)
    positions[:, 1] -= float(dy_m)
    amplitude, optical_path_length = bundle.require_coherent()
    return RayBundle(
        positions_m=positions,
        directions=np.asarray(bundle.directions, dtype=np.float64).copy(),
        wavelength_m=bundle.wavelength_m,
        reference_plane=bundle.reference_plane,
        frame=Frame(),
        amplitude=np.asarray(amplitude).copy(),
        optical_path_length_m=np.asarray(optical_path_length).copy(),
        optical_path_length_reference=bundle.optical_path_length_reference,
    )


def method_c_ray_to_wave(
    spec: Any,
    hx: float,
    hy: float,
    chief: dict[str, float],
    *,
    rings: int,
    workdir: Path,
) -> tuple[PsfMap, dict[str, Any]]:
    """The repository's ray-to-wave route, through shipping calls only.

    Returns the map plus a diagnostics dict. The wave-propagation leg is run at
    ``z = 0`` -- CHE-38 selected the sensor plane itself as the handoff, so there
    is no distance left to propagate -- and its residual against the coupler
    output is reported, so "the wave leg was exercised and is the identity here"
    is a measurement rather than an assertion.
    """
    from multiscale_optics_agent.adapters.base import ModelRunRequest
    from multiscale_optics_agent.adapters.chromatix_adapter import get_adapter as chromatix
    from multiscale_optics_agent.adapters.optiland_adapter import get_adapter as optiland
    from multiscale_optics_agent.couplers.optiland_handoff import (
        DeclaredHandoffPlane,
        declare_coherent_bundle,
    )
    from multiscale_optics_agent.couplers.ray_to_wave import ray_to_wave
    from multiscale_optics_agent.evaluation.psf_measurement import (
        M3_ORACLE_NORMALIZATION,
        measure_psf,
        measure_psf_from_record,
    )

    workdir.mkdir(parents=True, exist_ok=True)
    diagnostics: dict[str, Any] = {"rings": rings}

    # --- 1. trace, exported at the exit pupil (the only plane the adapter resolves)
    started = time.time()
    trace = optiland().run(
        ModelRunRequest(
            run_id="pb7",
            node_id="lens",
            config={
                "prescription": spec,
                "num_rays": rings,
                "wavelength": WAVELENGTH_UM,
                "Hx": hx,
                "Hy": hy,
                "handoff_plane": "exit_pupil",
                "output_directory": str(workdir / "rays"),
            },
        )
    )
    diagnostics["trace_seconds"] = time.time() - started
    if trace.status.value != "succeeded":
        raise RuntimeError(f"Optiland adapter failed: {trace.error_type}: {trace.error_message}")
    rays = trace.outputs["rays"]
    conventions = (rays.metadata or {}).get("conventions", {})
    adapter_pupil_z_m = float(conventions["exit_pupil"]["z_m"])
    image_space_index = float(conventions["image_space_refractive_index"])
    diagnostics["adapter_exit_pupil_z_m"] = adapter_pupil_z_m
    diagnostics["image_space_refractive_index"] = image_space_index
    if not np.isclose(adapter_pupil_z_m, EXIT_PUPIL_Z_MM * 1e-3, rtol=0.0, atol=1e-12):
        raise RuntimeError(
            f"declared exit pupil {EXIT_PUPIL_Z_MM * 1e-3!r} m disagrees with the "
            f"adapter's own reading {adapter_pupil_z_m!r} m"
        )
    if image_space_index != 1.0:
        raise RuntimeError(
            f"the ray-domain advance assumes image-space air; the adapter reports "
            f"n = {image_space_index!r}"
        )

    # --- 2. declare the coherent bundle (CHE-33/40/41/47)
    handoff = declare_coherent_bundle(
        rays, declared_plane=DeclaredHandoffPlane("exit_pupil", adapter_pupil_z_m)
    )
    diagnostics["handoff"] = {
        key: handoff.diagnostics[key]
        for key in (
            "ray_count",
            "opl_reference_version",
            "object_space_reference_applied",
            "object_space_reference_status",
            "object_space_reference_span_waves",
            "quadrature_weight_applied",
            "quadrature_weight_status",
            "removed_reference_opl_waves",
            "relative_opl_span_waves",
            "chief_ray_radius_m",
            "pupil_semi_extent_m",
        )
        if key in handoff.diagnostics
    }
    diagnostics["amplitude_mapping"] = handoff.declarations.get("amplitude_mapping")

    # --- 3. ray-domain advance to the declared sensor plane
    at_sensor, step = advance_bundle_to_z(handoff.bundle, IMAGE_Z_MM * 1e-3)
    diagnostics["advance_step_m"] = [float(np.min(step)), float(np.max(step))]

    # Geometry of the ray bundle where the PSF is measured, about the chief ray:
    # the purely geometric statement the diffraction result has to be read against.
    positions = np.asarray(at_sensor.positions_m)
    radial = np.hypot(positions[:, 0] - chief["x_m"], positions[:, 1] - chief["y_m"])
    diagnostics["geometric_spot_about_chief_ray"] = {
        "rms_radius_m": float(np.sqrt(np.mean(radial**2))),
        "max_radius_m": float(np.max(radial)),
        "centroid_minus_chief_ray_m": [
            float(np.mean(positions[:, 1]) - chief["y_m"]),
            float(np.mean(positions[:, 0]) - chief["x_m"]),
        ],
    }
    diagnostics["direction_space_extent_at_the_sensor"] = axis_numerical_aperture(
        np.asarray(at_sensor.directions)
    )

    # --- 4. exact transverse recentring on the chief-ray point
    recentred = translate_bundle_transverse(at_sensor, chief["x_m"], chief["y_m"])
    diagnostics["recentring_translation_m"] = [chief["y_m"], chief["x_m"]]

    # --- 5. C_RAY_TO_WAVE, unmodified
    started = time.time()
    complex_field, reconstruction = ray_to_wave(
        recentred,
        grid_shape=(SENSOR_GRID_N, SENSOR_GRID_N),
        sample_pitch_m=(SENSOR_PITCH_M, SENSOR_PITCH_M),
    )
    diagnostics["coupler_seconds"] = time.time() - started
    diagnostics["coupler"] = {
        "grid_shape": list(reconstruction.grid_shape),
        "sample_pitch_m": list(reconstruction.sample_pitch_m),
        "projection": reconstruction.projection,
        "normalization": reconstruction.normalization,
        "max_transverse_direction": reconstruction.max_transverse_direction,
        "grid_nyquist_direction_limit": reconstruction.grid_nyquist_direction_limit,
        "grid_nyquist_satisfied": reconstruction.grid_nyquist_satisfied,
        "nyquist_utilisation": (
            reconstruction.max_transverse_direction / reconstruction.grid_nyquist_direction_limit
        ),
        "ray_spacing_estimate_m": reconstruction.ray_spacing_estimate_m,
        "max_adjacent_ray_phase_rad": reconstruction.max_adjacent_ray_phase_rad,
        "ray_density_status": reconstruction.ray_density_status,
        "reconstructed_discrete_power": reconstruction.reconstructed_discrete_power,
    }

    coupler_measurement = measure_psf(complex_field, normalization=M3_ORACLE_NORMALIZATION)

    # --- 6. the shipping wave leg, over zero distance
    record = complex_field.to_artifact_record(
        artifact_id=f"field:pb7:{rings}", uri=workdir / "sensor_field.npy"
    )
    record.metadata["z_m"] = IMAGE_Z_MM * 1e-3
    record.metadata["reference_plane"] = complex_field.reference_plane.name
    started = time.time()
    propagated = chromatix().run(
        ModelRunRequest(
            run_id="pb7",
            node_id="wave",
            inputs={"input_field": record},
            config={
                "propagation": "angular_spectrum",
                "propagation_method": "asm_carrier_removed",
                "target_plane_z_m": IMAGE_Z_MM * 1e-3,
                "pad_width": 0,
                "output_dir": str(workdir / "wave"),
            },
        )
    )
    diagnostics["wave_leg_seconds"] = time.time() - started

    if propagated.status.value == "succeeded":
        reported = propagated.diagnostics["output_sample_pitch_m"]
        # --- 7. the frozen sensor-plane measurement (CHE-36 / M3.7)
        measurement = measure_psf_from_record(
            propagated.outputs["output_field"],
            normalization=M3_ORACLE_NORMALIZATION,
            expected_output_sample_pitch_m=(float(reported[0]), float(reported[1])),
        )
        a = np.asarray(measurement.intensity, dtype=np.float64)
        b = np.asarray(coupler_measurement.intensity, dtype=np.float64)
        diagnostics["wave_leg"] = {
            "engine": "Chromatix adapter, asm_carrier_removed",
            "propagation_distance_m": 0.0,
            "pad_width": 0,
            "why_zero": (
                "CHE-38 selected the sensor plane itself as the handoff, so no "
                "propagation remains after C_RAY_TO_WAVE. The leg is still run so the "
                "shipping wave path is exercised and its identity is measured."
            ),
            "output_sample_pitch_m": [float(reported[0]), float(reported[1])],
            "identity_relative_l2_vs_coupler_output": float(
                np.linalg.norm(a - b) / np.linalg.norm(b)
            ),
            "identity_max_abs_difference": float(np.max(np.abs(a - b))),
            "status": "succeeded",
        }
        source = "chromatix zero-distance ASM leg, then measure_psf_from_record"
    else:
        measurement = coupler_measurement
        diagnostics["wave_leg"] = {
            "engine": "Chromatix adapter, asm_carrier_removed",
            "status": "failed",
            "error_type": str(propagated.error_type),
            "error_message": propagated.error_message,
            "consequence": (
                "the PSF below is measured directly on the coupler output, which is "
                "CHE-38's shipping sensor path; the zero-distance wave leg is reported "
                "as failed rather than silently omitted."
            ),
        }
        source = "coupler output directly (wave leg failed)"

    intensity = np.asarray(measurement.intensity, dtype=np.float64)
    pitch_y, pitch_x = (float(v) for v in measurement.sample_pitch_m)
    ny, nx = intensity.shape
    y_rel = (np.arange(ny, dtype=np.float64) - ny // 2) * pitch_y
    x_rel = (np.arange(nx, dtype=np.float64) - nx // 2) * pitch_x

    cropped, centre_index = _crop_symmetric(intensity, NATIVE_CROP_HALF_WIDTH_M, pitch_y)
    axis = (np.arange(cropped.shape[0], dtype=np.float64) - centre_index) * pitch_y

    psf_map = PsfMap(
        method="C",
        label=f"ray -> C_RAY_TO_WAVE -> sensor ({rings} rings)",
        intensity=cropped / cropped.max(),
        y_rel_m=axis,
        x_rel_m=axis.copy(),
        pitch_y_m=pitch_y,
        pitch_x_m=pitch_x,
        native_peak=measurement.raw_peak_intensity,
        native_peak_units=(
            "|U|^2 in the field's own amplitude units squared. UNCALIBRATED: the ray "
            "amplitudes descend from Optiland per-ray intensity weights and no step "
            "converts them to watts, so this number is not comparable to A or B."
        ),
        provenance={
            "engine": "Optiland adapter -> C_RAY_TO_WAVE -> Chromatix ASM -> |U|^2",
            "measured_via": source,
            "hexapolar_rings": rings,
            "traced_rays": int(recentred.count),
            "handoff_plane": "nominal sensor (the declared paraxial image plane)",
            "handoff_plane_z_m": IMAGE_Z_MM * 1e-3,
            "psf_normalization": str(measurement.normalization),
            "psf_normalization_declaration": measurement.psf.normalization,
            "coherence_model": measurement.psf.coherence_model,
            "raw_window_energy": measurement.raw_window_energy,
            "border_energy_fraction": measurement.border_energy_fraction,
            "peak_index_native": list(measurement.peak_index),
            "peak_position_native_m": list(measurement.peak_position_m),
            "frame_offset_applied_m": [0.0, 0.0],
            "frame_offset_justification": (
                "the reconstruction grid was centred on the chief-ray point by an exact "
                "translation of the ray positions (see translate_bundle_transverse), so "
                "index n // 2 already is the frame origin. Nothing is shifted after the "
                "fact."
            ),
            "cropped_from_native": True,
            "crop_half_width_m": NATIVE_CROP_HALF_WIDTH_M,
            "native_shape": [int(ny), int(nx)],
            "native_full_window_m": float(ny * pitch_y),
        },
    )
    return psf_map, diagnostics


# ---------------------------------------------------------------------------
# 6. The common comparison grid, and the numbers taken on it
# ---------------------------------------------------------------------------
def common_grid(maps: list[PsfMap]) -> dict[str, Any]:
    """One grid per field, by the rule declared in :data:`COMMON_HALF_WIDTH_RULE`."""
    binding = min(maps, key=lambda m: m.half_window_m)
    half_width_m = np.floor(binding.half_window_m * 1e6) * 1e-6
    reach = int(round(half_width_m / COMMON_PITCH_M))
    axis = (np.arange(-reach, reach + 1, dtype=np.float64)) * COMMON_PITCH_M
    return {
        "axis_m": axis,
        "pitch_m": COMMON_PITCH_M,
        "half_width_m": float(half_width_m),
        "points_per_axis": int(axis.size),
        "half_width_set_by": binding.method,
        "half_width_set_by_label": binding.label,
        "native_half_windows_m": {m.method: m.half_window_m for m in maps},
        "rule": COMMON_HALF_WIDTH_RULE,
        "interpolation": (
            "scipy.interpolate.RegularGridInterpolator, method='linear', "
            "bounds_error=False, fill_value=0.0. The same call for all three "
            "methods; none is privileged by owning the grid."
        ),
    }


def resample(psf_map: PsfMap, grid: dict[str, Any]) -> np.ndarray:
    """Linear resampling onto the common grid, then peak normalization on it."""
    from scipy.interpolate import RegularGridInterpolator

    interpolator = RegularGridInterpolator(
        (psf_map.y_rel_m, psf_map.x_rel_m),
        psf_map.intensity,
        method="linear",
        bounds_error=False,
        fill_value=0.0,
    )
    axis = grid["axis_m"]
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    values = interpolator(np.stack([yy.ravel(), xx.ravel()], axis=-1)).reshape(yy.shape)
    peak = float(values.max())
    if peak <= 0.0:
        raise RuntimeError(f"method {psf_map.method} has no energy on the common grid")
    return values / peak


def _subpixel_peak(intensity: np.ndarray, axis: np.ndarray) -> dict[str, Any]:
    """Integer argmax plus a 3-point parabolic refinement on each axis."""
    iy, ix = (int(v) for v in np.unravel_index(int(np.argmax(intensity)), intensity.shape))
    pitch = float(axis[1] - axis[0])

    def refine(values: np.ndarray, index: int) -> float:
        if index <= 0 or index >= values.size - 1:
            return 0.0
        left, centre, right = values[index - 1], values[index], values[index + 1]
        denominator = left - 2.0 * centre + right
        if denominator == 0.0:
            return 0.0
        return float(np.clip(0.5 * (left - right) / denominator, -1.0, 1.0))

    dy = refine(intensity[:, ix], iy)
    dx = refine(intensity[iy, :], ix)
    return {
        "index_y_x": [iy, ix],
        "position_m_y_x": [float(axis[iy] + dy * pitch), float(axis[ix] + dx * pitch)],
        "subpixel_shift_pixels_y_x": [dy, dx],
        "note": (
            "a parabolic fit through the three samples straddling the argmax. Sub-pixel "
            f"here means a fraction of {pitch * 1e6:.3f} um."
        ),
    }


def _centroid(intensity: np.ndarray, axis: np.ndarray) -> list[float]:
    total = float(intensity.sum())
    if total <= 0.0:
        return [float("nan"), float("nan")]
    weight_y = intensity.sum(axis=1)
    weight_x = intensity.sum(axis=0)
    return [
        float((weight_y * axis).sum() / total),
        float((weight_x * axis).sum() / total),
    ]


def _fwhm(profile: np.ndarray, axis: np.ndarray, peak_index: int) -> float | None:
    """Full width at half the profile's own maximum, by linear interpolation.

    Walks outward from ``peak_index`` to the first crossing on each side, so a
    secondary lobe above half maximum does not widen the number. ``None`` when a
    crossing is not inside the window.
    """
    peak = float(profile[peak_index])
    if peak <= 0.0:
        return None
    half = 0.5 * peak

    def crossing(direction: int) -> float | None:
        index = peak_index
        while 0 <= index + direction < profile.size:
            nxt = index + direction
            if profile[nxt] <= half <= profile[index]:
                span = profile[index] - profile[nxt]
                if span == 0.0:
                    return float(axis[nxt])
                fraction = (profile[index] - half) / span
                return float(axis[index] + fraction * (axis[nxt] - axis[index]))
            index = nxt
        return None

    left, right = crossing(-1), crossing(+1)
    if left is None or right is None:
        return None
    return abs(right - left)


def _encircled_energy_radii(
    intensity: np.ndarray, axis: np.ndarray, centre_m: tuple[float, float]
) -> dict[str, float | None]:
    """Radii about ``centre_m`` holding declared fractions of the window energy.

    Window-dependent by construction: the denominator is the energy inside the
    common comparison window, not the total energy in the image.
    """
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    radius = np.hypot(yy - centre_m[0], xx - centre_m[1]).ravel()
    values = intensity.ravel()
    order = np.argsort(radius)
    cumulative = np.cumsum(values[order])
    total = float(cumulative[-1])
    out: dict[str, float | None] = {}
    for fraction in ENCIRCLED_ENERGY_FRACTIONS:
        if total <= 0.0:
            out[f"ee{int(fraction * 100)}_radius_m"] = None
            continue
        hit = np.searchsorted(cumulative, fraction * total)
        out[f"ee{int(fraction * 100)}_radius_m"] = (
            float(radius[order][min(hit, radius.size - 1)]) if hit < radius.size else None
        )
    return out


def map_metrics(intensity: np.ndarray, axis: np.ndarray) -> dict[str, Any]:
    """Everything measured on one method's resampled, peak-normalized map."""
    peak = _subpixel_peak(intensity, axis)
    iy, ix = peak["index_y_x"]
    centroid = _centroid(intensity, axis)
    return {
        "peak": peak,
        "centroid_m_y_x": centroid,
        "peak_minus_centroid_m_y_x": [
            peak["position_m_y_x"][0] - centroid[0],
            peak["position_m_y_x"][1] - centroid[1],
        ],
        "fwhm_y_m": _fwhm(intensity[:, ix], axis, iy),
        "fwhm_x_m": _fwhm(intensity[iy, :], axis, ix),
        "fwhm_cut_note": (
            "cuts pass through this map's OWN peak pixel, not a shared point, so a "
            "displaced PSF is still measured at its own core."
        ),
        **_encircled_energy_radii(intensity, axis, (peak["position_m_y_x"][0], peak["position_m_y_x"][1])),
        "energy_in_window": float(intensity.sum()),
        "energy_on_window_border_fraction": _border_fraction(intensity),
    }


def _border_fraction(intensity: np.ndarray) -> float:
    total = float(intensity.sum())
    if total <= 0.0 or min(intensity.shape) < 3:
        return 0.0
    border = float(
        intensity[0, :].sum()
        + intensity[-1, :].sum()
        + intensity[1:-1, 0].sum()
        + intensity[1:-1, -1].sum()
    )
    return border / total


def _relative_l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def axis_numerical_aperture(directions: np.ndarray) -> dict[str, Any]:
    """Per-axis direction-space semi-extent of the bundle, and the F/# it implies.

    Why this is measured at all: Optiland scales its FFT PSF by **one scalar**
    working F-number (``utils.get_working_FNO``, an RMS over four marginal rays),
    but the pixel scale of a Fourier-transform PSF is set per axis by that axis'
    direction-space semi-extent. On axis the two axes are equal and the scalar is
    exact. Off axis the image-space pupil is foreshortened in the meridional
    plane, the two are not equal, and no single scalar can be right for both.

    This is the geometric quantity that hypothesis rests on, taken from Method C's
    own traced bundle at the sensor plane -- so it is measured on the same rays
    the PSF came from, not derived from a paraxial model.
    """
    dx = np.asarray(directions, dtype=np.float64)[:, 0]
    dy = np.asarray(directions, dtype=np.float64)[:, 1]
    na_x = 0.5 * float(np.max(dx) - np.min(dx))
    na_y = 0.5 * float(np.max(dy) - np.min(dy))
    return {
        "direction_x_range": [float(np.min(dx)), float(np.max(dx))],
        "direction_y_range": [float(np.min(dy)), float(np.max(dy))],
        "numerical_aperture_x": na_x,
        "numerical_aperture_y": na_y,
        "f_number_x": 1.0 / (2.0 * na_x) if na_x > 0 else None,
        "f_number_y": 1.0 / (2.0 * na_y) if na_y > 0 else None,
        "anisotropy_f_number_y_over_x": (na_x / na_y) if na_y > 0 else None,
        "definition": (
            "NA_axis = half the range of that axis' direction cosine over the traced "
            "bundle at the sensor plane; F/#_axis = 1 / (2 NA_axis)."
        ),
    }


#: Coarse then fine scale search. Deterministic, no optimizer, no random restarts.
_SCALE_COARSE = (0.86, 1.14, 0.004)
_SCALE_FINE_SPAN = 0.004
_SCALE_FINE_STEP = 0.00025


def anisotropic_scale_diagnostic(
    subject: np.ndarray,
    reference: np.ndarray,
    axis: np.ndarray,
    *,
    predicted: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Fit two scale factors that map ``subject``'s coordinates onto ``reference``.

    DIAGNOSTIC ONLY. Nothing reported anywhere else in this benchmark is scaled by
    the result; the fitted numbers exist so that "the two disagree" can be
    separated into "they disagree about the coordinate scale" and "they disagree
    about the physics". The residual left *after* the best scale is the part the
    scale hypothesis does not explain, and it is reported next to the residual
    before.

    ``subject`` is resampled at ``(y / s_y, x / s_x)`` about the common origin, so
    ``s > 1`` means the subject's features have to be **stretched** to reach the
    reference -- i.e. the subject's own pixel scale was too small.

    Two-stage deterministic grid search: 0.004 steps over [0.86, 1.14], then
    0.00025 steps over +-0.004 of the winner. No optimizer, so the answer does
    not depend on a starting guess.
    """
    from scipy.ndimage import map_coordinates

    n = subject.shape[0]
    centre = n // 2
    index = np.arange(n, dtype=np.float64) - centre
    reference_norm = float(np.linalg.norm(reference))

    def residual(scale_y: float, scale_x: float) -> tuple[float, np.ndarray]:
        rows = centre + index / scale_y
        cols = centre + index / scale_x
        grid_rows, grid_cols = np.meshgrid(rows, cols, indexing="ij")
        sampled = map_coordinates(
            subject, np.stack([grid_rows, grid_cols]), order=1, mode="constant", cval=0.0
        )
        peak = float(sampled.max())
        if peak > 0.0:
            sampled = sampled / peak
        return float(np.linalg.norm(sampled - reference) / reference_norm), sampled

    lo, hi, step = _SCALE_COARSE
    candidates = np.arange(lo, hi + 0.5 * step, step)
    best = (float("inf"), 1.0, 1.0)
    for scale_y in candidates:
        for scale_x in candidates:
            value, _ = residual(float(scale_y), float(scale_x))
            if value < best[0]:
                best = (value, float(scale_y), float(scale_x))
    fine_y = np.arange(best[1] - _SCALE_FINE_SPAN, best[1] + _SCALE_FINE_SPAN + 1e-12, _SCALE_FINE_STEP)
    fine_x = np.arange(best[2] - _SCALE_FINE_SPAN, best[2] + _SCALE_FINE_SPAN + 1e-12, _SCALE_FINE_STEP)
    for scale_y in fine_y:
        for scale_x in fine_x:
            value, _ = residual(float(scale_y), float(scale_x))
            if value < best[0]:
                best = (value, float(scale_y), float(scale_x))

    before, _ = residual(1.0, 1.0)
    out: dict[str, Any] = {
        "status": "diagnostic only -- no reported PSF is scaled by this",
        "relative_l2_at_unit_scale": before,
        "fitted_scale_y": best[1],
        "fitted_scale_x": best[2],
        "relative_l2_at_fitted_scale": best[0],
        "fraction_of_residual_explained_by_scale": (
            1.0 - best[0] / before if before > 0 else None
        ),
        "search": {
            "coarse_range_step": list(_SCALE_COARSE),
            "fine_span": _SCALE_FINE_SPAN,
            "fine_step": _SCALE_FINE_STEP,
            "deterministic": True,
        },
        "sign_convention": (
            "subject is resampled at (y / s_y, x / s_x), so s > 1 means the subject's "
            "features are stretched outward to meet the reference, i.e. the subject's "
            "own pixel scale was too SMALL."
        ),
    }
    if predicted is not None:
        out["predicted_from_per_axis_f_number"] = predicted
        out["fitted_over_predicted_y"] = (
            best[1] / predicted["scale_y"] if predicted.get("scale_y") else None
        )
        out["fitted_over_predicted_x"] = (
            best[2] / predicted["scale_x"] if predicted.get("scale_x") else None
        )
    return out


def _shift_to(source: np.ndarray, shift: tuple[int, int]) -> np.ndarray:
    """Integer-pixel roll with zero fill -- for the peak-aligned diagnostic only."""
    out = np.zeros_like(source)
    dy, dx = shift
    ny, nx = source.shape
    ys = slice(max(0, dy), min(ny, ny + dy))
    xs = slice(max(0, dx), min(nx, nx + dx))
    yd = slice(max(0, -dy), min(ny, ny - dy))
    xd = slice(max(0, -dx), min(nx, nx - dx))
    out[ys, xs] = source[yd, xd]
    return out


def pairwise(
    resampled: dict[str, np.ndarray], metrics: dict[str, Any], axis: np.ndarray
) -> dict[str, Any]:
    """Every pair, in the declared frame and again peak-aligned.

    ``relative_l2`` is ``||a - b|| / ||b||``, so the pair label states which map
    is the denominator. The peak-aligned variant rolls ``a`` onto ``b``'s peak
    pixel and is a *diagnostic*: it separates a shape disagreement from a
    position disagreement. It is never the primary number.
    """
    pitch = float(axis[1] - axis[0])
    out: dict[str, Any] = {}
    for first, second in (("A", "B"), ("A", "C"), ("B", "C")):
        a, b = resampled[first], resampled[second]
        pa = metrics[first]["peak"]["index_y_x"]
        pb = metrics[second]["peak"]["index_y_x"]
        shift = (pb[0] - pa[0], pb[1] - pa[1])
        aligned = _shift_to(a, shift)
        separation = np.hypot(
            metrics[first]["peak"]["position_m_y_x"][0] - metrics[second]["peak"]["position_m_y_x"][0],
            metrics[first]["peak"]["position_m_y_x"][1] - metrics[second]["peak"]["position_m_y_x"][1],
        )
        out[f"{first}_vs_{second}"] = {
            "numerator": first,
            "denominator": second,
            "relative_l2": _relative_l2(a, b),
            "relative_l2_symmetric": float(
                np.linalg.norm(a - b) / np.sqrt(np.linalg.norm(a) * np.linalg.norm(b))
            ),
            "max_abs_difference": float(np.max(np.abs(a - b))),
            "peak_separation_m": float(separation),
            "peak_separation_pixels": float(separation / pitch),
            "relative_l2_peak_aligned": _relative_l2(aligned, b),
            "peak_alignment_shift_pixels_y_x": [int(shift[0]), int(shift[1])],
            "relative_l2_with_A_rows_flipped": (
                _relative_l2(a[::-1, :], b) if first == "A" else None
            ),
        }
    out["orientation_check_note"] = (
        "relative_l2_with_A_rows_flipped exists because the FFT PSF has no intrinsic "
        "coordinates: its row axis is asserted to be +y. If the flipped residual were "
        "the smaller one, that assertion would be wrong and every A coordinate here "
        "would be mirrored."
    )
    return out


# ---------------------------------------------------------------------------
# 7. Figures
# ---------------------------------------------------------------------------
def _figure_paths(name: str) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR / name


def figure_native(field_name: str, maps: list[PsfMap], airy_radius_m: float) -> str:
    """Each method on its own grid: the side-by-side the ticket asks for.

    Nothing is resampled here. All three panels share axis limits and a colour
    scale so morphology is directly comparable, and each panel's own pitch and
    window are printed on it so the reader can see what differs.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    limit_um = min(m.half_window_m for m in maps) * 1e6
    figure, axes = plt.subplots(2, 3, figsize=(15.0, 9.2))
    for column, psf_map in enumerate(maps):
        extent = [
            psf_map.x_rel_m[0] * 1e6,
            psf_map.x_rel_m[-1] * 1e6,
            psf_map.y_rel_m[0] * 1e6,
            psf_map.y_rel_m[-1] * 1e6,
        ]
        for row, (log, floor) in enumerate(((False, None), (True, LOG_FLOOR))):
            axis = axes[row][column]
            data = psf_map.intensity
            if log:
                data = np.maximum(data, floor)
                image = axis.imshow(
                    data,
                    origin="lower",
                    extent=extent,
                    cmap="inferno",
                    norm=matplotlib.colors.LogNorm(vmin=floor, vmax=1.0),
                )
            else:
                image = axis.imshow(
                    data, origin="lower", extent=extent, cmap="inferno", vmin=0.0, vmax=1.0
                )
            axis.set_xlim(-limit_um, limit_um)
            axis.set_ylim(-limit_um, limit_um)
            axis.plot(0.0, 0.0, "+", color="#7fdcff", markersize=9, markeredgewidth=1.3)
            circle = plt.Circle(
                (0.0, 0.0), airy_radius_m * 1e6, fill=False, color="#7fdcff",
                linewidth=0.9, linestyle="--",
            )
            axis.add_patch(circle)
            axis.set_xlabel("x - x(chief ray)  [um]")
            axis.set_ylabel("y - y(chief ray)  [um]")
            title = f"{psf_map.method}. {psf_map.label}"
            if row == 0:
                title += (
                    f"\npitch {psf_map.pitch_y_m * 1e6:.3f} um, "
                    f"half-window {psf_map.half_window_m * 1e6:.1f} um"
                )
            else:
                title += "\nlog10, floor 1e-5 of own peak"
            axis.set_title(title, fontsize=9.5)
            figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.suptitle(
        f"PB7 / CHE-58 -- Cooke Triplet PSF, {field_name}, {WAVELENGTH_UM} um. "
        "Each method on its OWN grid; peak-normalized; dashed circle = Airy first-null "
        f"radius ({airy_radius_m * 1e6:.2f} um); + = traced chief ray.",
        fontsize=11,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    path = _figure_paths(f"pb7_{field_name}_three_psf_native.png")
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return str(path.relative_to(ROOT))


def figure_common(
    field_name: str,
    resampled: dict[str, np.ndarray],
    labels: dict[str, str],
    grid: dict[str, Any],
    airy_radius_m: float,
) -> str:
    """The three maps on one grid, and their three differences."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    axis_um = grid["axis_m"] * 1e6
    extent = [axis_um[0], axis_um[-1], axis_um[0], axis_um[-1]]
    figure, axes = plt.subplots(2, 3, figsize=(15.0, 9.4))

    for column, key in enumerate(("A", "B", "C")):
        ax = axes[0][column]
        image = ax.imshow(
            resampled[key], origin="lower", extent=extent, cmap="inferno", vmin=0.0, vmax=1.0
        )
        ax.plot(0.0, 0.0, "+", color="#7fdcff", markersize=9, markeredgewidth=1.3)
        ax.add_patch(
            plt.Circle(
                (0.0, 0.0), airy_radius_m * 1e6, fill=False, color="#7fdcff",
                linewidth=0.9, linestyle="--",
            )
        )
        ax.set_title(f"{key}. {labels[key]}", fontsize=9.5)
        ax.set_xlabel("x - x(chief ray)  [um]")
        ax.set_ylabel("y - y(chief ray)  [um]")
        figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    for column, (first, second) in enumerate((("A", "B"), ("A", "C"), ("B", "C"))):
        ax = axes[1][column]
        difference = resampled[first] - resampled[second]
        span = float(np.max(np.abs(difference))) or 1.0
        image = ax.imshow(
            difference, origin="lower", extent=extent, cmap="RdBu_r", vmin=-span, vmax=span
        )
        ax.set_title(
            f"{first} - {second}   (max |d| = {span:.3f} of peak)\n"
            f"relative L2 = {_relative_l2(resampled[first], resampled[second]):.4f}",
            fontsize=9.5,
        )
        ax.set_xlabel("x - x(chief ray)  [um]")
        ax.set_ylabel("y - y(chief ray)  [um]")
        figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    figure.suptitle(
        f"PB7 / CHE-58 -- {field_name}: one common grid "
        f"({grid['pitch_m'] * 1e6:.2f} um pitch, +-{grid['half_width_m'] * 1e6:.0f} um, "
        f"half-width set by method {grid['half_width_set_by']}), linear interpolation, "
        "peak-normalized on the window.",
        fontsize=11,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    path = _figure_paths(f"pb7_{field_name}_three_psf_common_grid.png")
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return str(path.relative_to(ROOT))


def figure_profiles(
    field_name: str,
    resampled: dict[str, np.ndarray],
    metrics: dict[str, Any],
    labels: dict[str, str],
    grid: dict[str, Any],
    airy_radius_m: float,
) -> str:
    """Normalized line cuts and radial encircled energy.

    Two cut conventions are drawn, because they answer different questions and
    disagree off axis: through each method's own peak (shape), and through the
    common origin at the chief ray (shape *and* placement).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    axis_um = grid["axis_m"] * 1e6
    colours = {"A": "#1f77b4", "B": "#2f7d4f", "C": "#a8442a"}
    figure, axes = plt.subplots(2, 2, figsize=(14.0, 9.0))

    for key in ("A", "B", "C"):
        iy, ix = metrics[key]["peak"]["index_y_x"]
        centre = grid["axis_m"].size // 2
        axes[0][0].plot(axis_um, resampled[key][iy, :], color=colours[key], label=f"{key}. {labels[key]}")
        axes[0][1].plot(axis_um, resampled[key][:, ix], color=colours[key], label=f"{key}. {labels[key]}")
        axes[1][0].plot(axis_um, resampled[key][centre, :], color=colours[key], label=f"{key} (x cut at y=0)")
        axes[1][0].plot(
            axis_um, resampled[key][:, centre], color=colours[key], linestyle="--",
            label=f"{key} (y cut at x=0)",
        )

    for ax, title, note in (
        (axes[0][0], "x cut through each method's own peak", "y fixed at that method's peak row"),
        (axes[0][1], "y cut through each method's own peak", "x fixed at that method's peak column"),
        (axes[1][0], "cuts through the chief ray (common origin)", "solid = x cut, dashed = y cut"),
    ):
        ax.set_title(f"{title}\n{note}", fontsize=9.5)
        ax.set_xlabel("offset from chief ray  [um]")
        ax.set_ylabel("intensity / own peak")
        ax.axvline(0.0, color="0.7", linewidth=0.8)
        ax.axhline(0.5, color="0.85", linewidth=0.8, linestyle=":")
        for sign in (-1.0, 1.0):
            ax.axvline(sign * airy_radius_m * 1e6, color="0.85", linewidth=0.8, linestyle="--")
        ax.set_xlim(axis_um[0], axis_um[-1])
        ax.legend(fontsize=7.5)

    # Radial encircled energy about each method's own peak, over the window.
    for key in ("A", "B", "C"):
        peak = metrics[key]["peak"]["position_m_y_x"]
        yy, xx = np.meshgrid(grid["axis_m"], grid["axis_m"], indexing="ij")
        radius = np.hypot(yy - peak[0], xx - peak[1]).ravel()
        values = resampled[key].ravel()
        order = np.argsort(radius)
        cumulative = np.cumsum(values[order])
        axes[1][1].plot(
            radius[order] * 1e6,
            cumulative / cumulative[-1],
            color=colours[key],
            label=f"{key}. {labels[key]}",
        )
    axes[1][1].set_title(
        "encircled energy about each method's own peak\n"
        "denominator is the energy inside the common window, not the whole image",
        fontsize=9.5,
    )
    axes[1][1].set_xlabel("radius  [um]")
    axes[1][1].set_ylabel("enclosed fraction of window energy")
    axes[1][1].set_xlim(0.0, grid["half_width_m"] * 1e6)
    axes[1][1].axvline(airy_radius_m * 1e6, color="0.85", linewidth=0.8, linestyle="--")
    for fraction in ENCIRCLED_ENERGY_FRACTIONS:
        axes[1][1].axhline(fraction, color="0.9", linewidth=0.8, linestyle=":")
    axes[1][1].legend(fontsize=7.5)

    figure.suptitle(
        f"PB7 / CHE-58 -- {field_name}: normalized profiles on the common grid. "
        f"Dashed vertical = Airy first-null radius ({airy_radius_m * 1e6:.2f} um).",
        fontsize=11,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    path = _figure_paths(f"pb7_{field_name}_profiles.png")
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return str(path.relative_to(ROOT))


# ---------------------------------------------------------------------------
# 8. Driver
# ---------------------------------------------------------------------------
def _environment() -> dict[str, Any]:
    from importlib.metadata import PackageNotFoundError, version

    packages: dict[str, str | None] = {}
    for name in ("optiland", "chromatix", "jax", "numpy", "scipy", "matplotlib"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        commit = None
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": packages,
        "git_commit": commit,
        "device": "cpu (agent_solver container is CPU-only by construction)",
    }


def characterize() -> dict[str, Any]:
    from optiland.samples.objectives import CookeTriplet
    from optiland.utils import get_working_FNO

    from multiscale_optics_agent.adapters.optiland_builder import build_optiland_system

    spec = cooke_triplet_spec()
    built = build_optiland_system(spec)
    bundled = CookeTriplet()

    equivalence = verify_system_equivalence(built, bundled)
    if not equivalence["all_identical"]:
        raise RuntimeError(
            "the canonical CookeTriplet prescription is not identical to the bundled "
            "sample; a three-way PSF comparison across two lenses is meaningless. "
            f"Detail: {json.dumps(equivalence['levels'], default=str)[:2000]}"
        )

    image_z_mm = float(np.asarray(bundled.surfaces.positions[-1]).ravel()[0])
    if not np.isclose(image_z_mm, IMAGE_Z_MM, rtol=0.0, atol=1e-9):
        raise RuntimeError(f"declared image plane {IMAGE_Z_MM} mm, prescription says {image_z_mm}")
    xpl_mm = float(np.asarray(bundled.paraxial.XPL()).ravel()[0])
    if not np.isclose(image_z_mm + xpl_mm, EXIT_PUPIL_Z_MM, rtol=0.0, atol=1e-9):
        raise RuntimeError("declared exit pupil disagrees with image_z + XPL()")

    record: dict[str, Any] = {
        "benchmark": "PB7 / CHE-58 -- Cooke Triplet: FFT vs Huygens vs ray->wave PSF",
        "status": "diagnostic benchmark; no tolerance, no pass/fail, no gate",
        "environment": _environment(),
        "declared_configuration": {
            "optical_system": "optiland CookeTriplet (bundled sample, and the canonical prescription verified identical to it)",
            "prescription_fingerprint": spec.fingerprint(),
            "wavelength_um": WAVELENGTH_UM,
            "wavelength_is_the_systems_primary": True,
            "fields": [
                {"name": name, "Hx": hx, "Hy": hy, "field_angle_deg": hy * float(bundled.fields.max_field)}
                for name, hx, hy in FIELDS
            ],
            "max_field_deg": float(bundled.fields.max_field),
            "image_plane_z_mm": IMAGE_Z_MM,
            "exit_pupil_z_mm": EXIT_PUPIL_Z_MM,
            "paraxial_fno": float(np.asarray(bundled.paraxial.FNO()).ravel()[0]),
            "method_a": {
                "num_rays": FFT_NUM_RAYS,
                "grid_size": FFT_GRID_SIZE,
                "strategy": FFT_STRATEGY,
                "remove_tilt": FFT_REMOVE_TILT,
            },
            "method_b": {
                "num_rays": HUYGENS_NUM_RAYS,
                "image_size": HUYGENS_IMAGE_SIZE,
                "strategy": HUYGENS_STRATEGY,
                "remove_tilt": HUYGENS_REMOVE_TILT,
            },
            "method_c": {
                "hexapolar_rings": RAY_TO_WAVE_RINGS,
                "stability_rings": RAY_TO_WAVE_STABILITY_RINGS,
                "sensor_pitch_m": SENSOR_PITCH_M,
                "sensor_grid_n": SENSOR_GRID_N,
                "handoff_plane": "nominal sensor (declared paraxial image plane)",
                "post_handoff_propagation_m": 0.0,
                "psf_normalization": "peak (M3_ORACLE_NORMALIZATION, CHE-36)",
            },
            "common_grid_pitch_m": COMMON_PITCH_M,
            "common_grid_half_width_rule": COMMON_HALF_WIDTH_RULE,
            "native_crop_half_width_m": NATIVE_CROP_HALF_WIDTH_M,
        },
        "system_equivalence": equivalence,
        "conventions": {
            "frame": (
                "Optiland global frame at the image surface z = 60.17675 mm. Reported "
                "coordinates are metres (printed as um) from the traced chief-ray "
                "intersection for that field, row-major (y, x), increasing index = "
                "increasing coordinate."
            ),
            "origin_definition": (
                "lens.trace_generic(Hx, Hy, Px=0, Py=0) at the image surface -- the same "
                "point Optiland's strategy='chief_ray' centres its reference sphere on."
            ),
            "units": "SI internally; micrometres in every figure axis and printed table",
            "intensity": "|U|^2 for method C; Optiland relative-intensity percent for A and B",
            "comparison_normalization": (
                "every map is peak-normalized on the common comparison window before any "
                "pairwise number is taken. There is no common absolute scale: A and B are "
                "normalized to a diffraction-limited reference, C is uncalibrated."
            ),
            "coherence_model": "monochromatic, fully coherent, scalar, single wavelength",
            "polarization": "none; all three paths are scalar (optic.polarization_state is None)",
            "gradient_claim": "none. Forward only.",
            "resampling_artefact_note": (
                "the difference maps carry a faint square-pixel texture at the core. "
                "That is the linear-interpolation footprint of resampling three "
                "different native pitches (0.13-0.20 um) onto the 0.20 um common grid, "
                "not structure in any PSF. It sets the floor of the residuals reported "
                "here at the few-1e-3 level, which is why nothing below is read as "
                "meaningful at that scale."
            ),
        },
        "fields": {},
    }

    workroot = Path(tempfile.mkdtemp(prefix="pb7_"))
    figures: list[str] = []

    for field_name, hx, hy in FIELDS:
        chief = chief_ray_image_point(bundled, hx, hy)
        working_fno = float(np.asarray(get_working_FNO(bundled, (hx, hy), WAVELENGTH_UM)).ravel()[0])
        numerical_aperture = 1.0 / (2.0 * working_fno)
        airy_radius_m = 0.6098349456 * WAVELENGTH_M / numerical_aperture

        print(f"\n=== {field_name}: Hx={hx}, Hy={hy} "
              f"({hy * float(bundled.fields.max_field):g} deg)  ".ljust(78, "="))
        print(f"    chief ray at image plane: x={chief['x_m'] * 1e3:.6f} mm, "
              f"y={chief['y_m'] * 1e3:.6f} mm")
        print(f"    working F/# = {working_fno:.4f}  ->  NA = {numerical_aperture:.5f}, "
              f"Airy first-null radius = {airy_radius_m * 1e6:.4f} um")

        map_a = method_a_fft(bundled, hx, hy, chief)
        print(f"    A  FFT      {map_a.provenance['seconds']:.2f} s, "
              f"pitch {map_a.pitch_y_m * 1e6:.4f} um, Strehl {map_a.provenance['strehl_ratio']:.5f}")
        map_b = method_b_huygens(bundled, hx, hy, chief)
        print(f"    B  Huygens  {map_b.provenance['seconds']:.2f} s, "
              f"pitch {map_b.pitch_y_m * 1e6:.4f} um, Strehl {map_b.provenance['strehl_ratio']:.5f}")
        map_c, diagnostics_c = method_c_ray_to_wave(
            spec, hx, hy, chief, rings=RAY_TO_WAVE_RINGS,
            workdir=workroot / field_name / f"r{RAY_TO_WAVE_RINGS}",
        )
        print(f"    C  ray->wave {diagnostics_c['coupler_seconds']:.2f} s coupler, "
              f"{diagnostics_c['traced_rays'] if 'traced_rays' in diagnostics_c else map_c.provenance['traced_rays']} rays, "
              f"pitch {map_c.pitch_y_m * 1e6:.4f} um")
        print(f"       Nyquist utilisation {diagnostics_c['coupler']['nyquist_utilisation']:.3f}, "
              f"ray density: {diagnostics_c['coupler']['ray_density_status']}")
        print(f"       wave leg: {diagnostics_c['wave_leg']['status']}, identity relative L2 = "
              f"{diagnostics_c['wave_leg'].get('identity_relative_l2_vs_coupler_output')}")

        # Stability of method C under a ray-count change. Two points; no rate claimed.
        map_c_alt, diagnostics_c_alt = method_c_ray_to_wave(
            spec, hx, hy, chief, rings=RAY_TO_WAVE_STABILITY_RINGS,
            workdir=workroot / field_name / f"r{RAY_TO_WAVE_STABILITY_RINGS}",
        )

        maps = [map_a, map_b, map_c]
        grid = common_grid(maps + [map_c_alt])
        resampled = {"A": resample(map_a, grid), "B": resample(map_b, grid), "C": resample(map_c, grid)}
        resampled_alt = resample(map_c_alt, grid)
        metrics = {key: map_metrics(resampled[key], grid["axis_m"]) for key in ("A", "B", "C")}
        comparisons = pairwise(resampled, metrics, grid["axis_m"])

        # Why the FFT PSF might be on the wrong coordinate scale off axis, tested.
        per_axis = diagnostics_c["direction_space_extent_at_the_sensor"]
        predicted = {
            "isotropic_working_f_number_used_by_method_a": working_fno,
            "per_axis_f_number_y": per_axis["f_number_y"],
            "per_axis_f_number_x": per_axis["f_number_x"],
            "scale_y": per_axis["f_number_y"] / working_fno,
            "scale_x": per_axis["f_number_x"] / working_fno,
            "reasoning": (
                "the FFT PSF's pixel scale is dx = lambda * F/# / Q with ONE scalar "
                "F/#. If the correct per-axis F/# differs from it, the axis is "
                "mis-scaled by exactly F/#_axis / F/#_scalar. That ratio is the "
                "prediction; the fit below is the measurement."
            ),
        }
        scale_diagnostic = {
            "hypothesis": (
                "Method A uses one isotropic working F-number where the off-axis "
                "image-space pupil has anisotropic direction-space extent, so its pixel "
                "scale is wrong by a different factor on each axis."
            ),
            "A_onto_B": anisotropic_scale_diagnostic(
                resampled["A"], resampled["B"], grid["axis_m"], predicted=predicted
            ),
            "A_onto_C": anisotropic_scale_diagnostic(
                resampled["A"], resampled["C"], grid["axis_m"], predicted=predicted
            ),
            "B_onto_C": anisotropic_scale_diagnostic(
                resampled["B"], resampled["C"], grid["axis_m"]
            ),
            "control": (
                "B_onto_C carries no prediction and is the control: two methods that "
                "already agree must fit scales of ~1.000 and gain almost nothing from "
                "the extra freedom. If they did not, the fit would be absorbing "
                "something other than a coordinate scale."
            ),
        }

        stability = {
            "rings_primary": RAY_TO_WAVE_RINGS,
            "rings_alternate": RAY_TO_WAVE_STABILITY_RINGS,
            "traced_rays_primary": map_c.provenance["traced_rays"],
            "traced_rays_alternate": map_c_alt.provenance["traced_rays"],
            "ray_density_status_primary": diagnostics_c["coupler"]["ray_density_status"],
            "ray_density_status_alternate": diagnostics_c_alt["coupler"]["ray_density_status"],
            "max_adjacent_ray_phase_rad_alternate": (
                diagnostics_c_alt["coupler"]["max_adjacent_ray_phase_rad"]
            ),
            "ray_density_note": (
                "the coupler's nearest-neighbour phase-step diagnostic is skipped above "
                "an O(N^2) cost limit, which the primary ray count exceeds. The "
                "alternate (coarser) ray count is below the limit and its measured worst "
                "adjacent-ray phase step is reported here: it bounds the primary run from "
                "ABOVE, because refining the fan can only shrink the spacing between "
                "neighbouring rays. Below pi the local wavelet picture holds."
            ),
            "relative_l2_between_ray_counts": _relative_l2(resampled_alt, resampled["C"]),
            "peak_separation_m": float(
                np.hypot(
                    _subpixel_peak(resampled_alt, grid["axis_m"])["position_m_y_x"][0]
                    - metrics["C"]["peak"]["position_m_y_x"][0],
                    _subpixel_peak(resampled_alt, grid["axis_m"])["position_m_y_x"][1]
                    - metrics["C"]["peak"]["position_m_y_x"][1],
                )
            ),
            "interpretation_limit": (
                "two ray counts. This says whether the picture would move if the ray "
                "count changed; it establishes no convergence rate and none is claimed. "
                "Read it against the A-vs-B residual: a ray-count sensitivity well below "
                "that residual means the ray sampling is not what separates C from A/B."
            ),
        }

        labels = {"A": map_a.label, "B": map_b.label, "C": map_c.label}
        figures.append(figure_native(field_name, maps, airy_radius_m))
        figures.append(figure_common(field_name, resampled, labels, grid, airy_radius_m))
        figures.append(figure_profiles(field_name, resampled, metrics, labels, grid, airy_radius_m))

        record["fields"][field_name] = {
            "field": {"Hx": hx, "Hy": hy, "field_angle_deg": hy * float(bundled.fields.max_field)},
            "chief_ray_image_point_m": chief,
            "working_fno": working_fno,
            "image_space_numerical_aperture": numerical_aperture,
            "airy_first_null_radius_m": airy_radius_m,
            "airy_radius_in_common_pixels": airy_radius_m / COMMON_PITCH_M,
            "maps": {"A": map_a.as_dict(), "B": map_b.as_dict(), "C": map_c.as_dict()},
            "method_c_diagnostics": diagnostics_c,
            "method_c_stability": stability,
            "common_grid": {
                key: value for key, value in grid.items() if key != "axis_m"
            },
            "metrics_on_common_grid": metrics,
            "pairwise": comparisons,
            "per_axis_direction_space_extent": per_axis,
            "anisotropic_scale_diagnostic": scale_diagnostic,
            "strehl_ratios_reported_by_optiland": {
                "A_fft": map_a.provenance["strehl_ratio"],
                "B_huygens": map_b.provenance["strehl_ratio"],
                "note": (
                    "strehl_ratio() is the CENTRE PIXEL over 100, i.e. the relative "
                    "intensity at the chief-ray point -- not the peak. Off axis the peak "
                    "is elsewhere, so peak_over_centre below is the ratio that says how "
                    "far. Method C has no absolute Strehl: its scale is uncalibrated."
                ),
                "A_peak_over_centre": map_a.provenance["peak_over_centre"],
                "B_peak_over_centre": map_b.provenance["peak_over_centre"],
                "A_peak_relative_intensity_over_100": map_a.native_peak / 100.0,
                "B_peak_relative_intensity_over_100": map_b.native_peak / 100.0,
            },
        }

        print("\n    -- on the common grid "
              f"({grid['pitch_m'] * 1e6:.2f} um, +-{grid['half_width_m'] * 1e6:.0f} um, "
              f"{grid['points_per_axis']}^2, half-width set by {grid['half_width_set_by']}) --")
        print(f"    {'':<10}{'peak y':>10}{'peak x':>10}{'cen y':>10}{'cen x':>10}"
              f"{'FWHM y':>10}{'FWHM x':>10}{'EE50 r':>10}{'EE80 r':>10}   [um]")
        for key in ("A", "B", "C"):
            m = metrics[key]
            def fmt(value: float | None) -> str:
                return "     n/a" if value is None else f"{value * 1e6:10.3f}"
            print(
                f"    {key:<10}{m['peak']['position_m_y_x'][0] * 1e6:10.3f}"
                f"{m['peak']['position_m_y_x'][1] * 1e6:10.3f}"
                f"{m['centroid_m_y_x'][0] * 1e6:10.3f}{m['centroid_m_y_x'][1] * 1e6:10.3f}"
                f"{fmt(m['fwhm_y_m'])}{fmt(m['fwhm_x_m'])}"
                f"{fmt(m['ee50_radius_m'])}{fmt(m['ee80_radius_m'])}"
            )
        print(f"    {'pair':<10}{'rel L2':>12}{'rel L2 aligned':>18}{'peak sep um':>14}"
              f"{'rel L2 (A rows flipped)':>26}")
        for pair in ("A_vs_B", "A_vs_C", "B_vs_C"):
            c = comparisons[pair]
            flipped = c["relative_l2_with_A_rows_flipped"]
            print(
                f"    {pair:<10}{c['relative_l2']:12.4f}{c['relative_l2_peak_aligned']:18.4f}"
                f"{c['peak_separation_m'] * 1e6:14.3f}"
                f"{('%.4f' % flipped) if flipped is not None else 'n/a':>26}"
            )
        print(f"    per-axis image-space F/#: y = {per_axis['f_number_y']:.4f}, "
              f"x = {per_axis['f_number_x']:.4f}; the single scalar method A used = "
              f"{working_fno:.4f}")
        print(f"    {'scale fit':<12}{'s_y':>9}{'s_x':>9}{'pred s_y':>10}{'pred s_x':>10}"
              f"{'L2 before':>12}{'L2 after':>11}")
        for pair_name in ("A_onto_B", "A_onto_C", "B_onto_C"):
            entry = scale_diagnostic[pair_name]
            predicted_entry = entry.get("predicted_from_per_axis_f_number")
            py = f"{predicted_entry['scale_y']:10.4f}" if predicted_entry else "       n/a"
            px = f"{predicted_entry['scale_x']:10.4f}" if predicted_entry else "       n/a"
            print(
                f"    {pair_name:<12}{entry['fitted_scale_y']:9.4f}{entry['fitted_scale_x']:9.4f}"
                f"{py}{px}{entry['relative_l2_at_unit_scale']:12.4f}"
                f"{entry['relative_l2_at_fitted_scale']:11.4f}"
            )
        print(f"    method C ray-count stability ({RAY_TO_WAVE_STABILITY_RINGS} vs "
              f"{RAY_TO_WAVE_RINGS} rings): relative L2 = "
              f"{stability['relative_l2_between_ray_counts']:.4f}, peak moves "
              f"{stability['peak_separation_m'] * 1e6:.3f} um")

    record["figures"] = figures
    record["open_issues_this_benchmark_touches"] = {
        "CHE-50 (missing wavefront-curvature term in C_RAY_TO_WAVE)": (
            "NOT ACTIVE in the measured intensity here. The reconstructed field is a sum "
            "of plane waves in the transverse coordinate and carries no "
            "exp(i k r^2 / 2R) term, which is invisible in |U|^2 at the handoff plane. "
            "Because CHE-38's selected handoff is the sensor plane itself, there is zero "
            "post-handoff propagation for that missing term to be seen by. It would "
            "become active the moment a caller propagated the method-C field further."
        ),
        "CHE-47 / CHE-48 (per-ray quadrature weight, unattributed sensor residual)": (
            "POTENTIALLY ACTIVE. The per-ray hexapolar area weight is applied when the "
            "adapter supplies the pupil coordinates; whether it was is recorded per field "
            "under method_c_diagnostics.handoff.quadrature_weight_status. CHE-38 measured "
            "the unweighted sum's residual as an effective-NA overshoot of about half a "
            "ring spacing, which is a WIDTH error -- so it is the first thing to look at "
            "if C's FWHM or first-ring radius differs from A and B."
        ),
        "CHE-51 (Fresnel number vs NA in the sensor residual)": (
            "NOT SEPARATED HERE. This benchmark runs one system at one wavelength, so N_f "
            "and NA are not independently varied and nothing here can distinguish them. "
            "Named so the absence is explicit."
        ),
        "not an issue but the dominant caveat": (
            "A and B are two implementations inside one package and share the same "
            "Wavefront/OPD front end (same reference sphere, same launch-tilt removal, "
            "same pupil sampling). They are therefore NOT independent in the way an "
            "analytic oracle would be, and the A-vs-B residual understates the true "
            "uncertainty of the Optiland pair."
        ),
    }
    return record


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def main() -> None:
    started = time.time()
    record = characterize()
    record["wall_clock_seconds"] = time.time() - started
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / "pb7_cooke_triplet_psf_comparison.json"
    path.write_text(json.dumps(_json_ready(record), indent=2, sort_keys=False) + "\n")
    print(f"\nwrote {path.relative_to(ROOT)}")
    for name in record["figures"]:
        print(f"wrote {name}")
    print(f"total {record['wall_clock_seconds']:.1f} s")


if __name__ == "__main__":
    main()
