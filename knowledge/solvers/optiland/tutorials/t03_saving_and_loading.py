"""Beginner / "Saving and Loading Files" -- https://www.optiland.org/tutorials/saving-and-loading

Repo-owned reproduction of the JSON serialization tutorial:
``save_optiland_file`` / ``load_optiland_file`` on the bundled
``UVReflectingMicroscope``.

Upstream validates by eye (draw the system, save, reload, draw again). This
reproduction replaces that with a machine-checkable round trip, which is the
property the tutorial actually claims:

* ``Optic.to_dict()`` of the reloaded system equals that of the original.
* A real ray trace through the reloaded system is **element-wise identical** to
  the original, coordinates and accumulated OPL alike -- serialization does not
  perturb the numerics.
* Paraxial EFL survives the round trip exactly.
* The on-disk artifact is valid JSON containing the surface count.

The system is a reflecting (mirror) microscope, so this also exercises
serialization of a non-refractive prescription, not just a glass singlet.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from _harness import TutorialMeta, TutorialResult, standalone_main

TUTORIAL = TutorialMeta(
    slug="t03_saving_and_loading",
    title="Saving and Loading Files",
    level="beginner",
    url="https://www.optiland.org/tutorials/saving-and-loading",
    demonstrates=(
        "optiland.fileio.save_optiland_file / load_optiland_file JSON round trip, "
        "and Optic.to_dict as the serialized form. Exercised on a reflecting "
        "(mirror) sample system."
    ),
)

WAVELENGTH_UM = 0.25


def run() -> TutorialResult:
    from optiland.fileio import load_optiland_file, save_optiland_file
    from optiland.samples.microscopes import UVReflectingMicroscope

    result = TutorialResult()
    system = UVReflectingMicroscope()

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "uv_reflecting_microscope.json"
        save_optiland_file(system, str(path))
        size_bytes = path.stat().st_size
        raw = json.loads(path.read_text())
        new_system = load_optiland_file(str(path))

    result.record(
        artifact_bytes=size_bytes,
        artifact_top_level_keys=sorted(raw.keys()),
        num_surfaces=len(system.surfaces.surfaces),
        reloaded_num_surfaces=len(new_system.surfaces.surfaces),
    )
    result.check_true(
        "artifact_is_valid_json_with_content",
        "invariant",
        size_bytes > 0 and isinstance(raw, dict) and bool(raw),
        f"{size_bytes} bytes, top-level keys {sorted(raw.keys())}",
    )
    result.check_true(
        "surface_count_survives_round_trip",
        "invariant",
        len(system.surfaces.surfaces) == len(new_system.surfaces.surfaces),
        f"{len(system.surfaces.surfaces)} == {len(new_system.surfaces.surfaces)}",
    )
    result.check_true(
        "to_dict_survives_round_trip",
        "invariant",
        system.to_dict() == new_system.to_dict(),
        "Optic.to_dict() of original and reloaded system compare equal",
    )

    wl = float(np.asarray(system.primary_wavelength).ravel()[0])
    result.record(primary_wavelength_um=wl)
    before = system.trace(Hx=0.0, Hy=0.0, wavelength=wl, num_rays=12)
    after = new_system.trace(Hx=0.0, Hy=0.0, wavelength=wl, num_rays=12)
    identical = {}
    for attr in ("x", "y", "z", "L", "M", "N", "opd", "i"):
        a = np.asarray(getattr(before, attr), dtype=float)
        b = np.asarray(getattr(after, attr), dtype=float)
        identical[attr] = bool(a.shape == b.shape and np.array_equal(a, b))
    result.record(
        num_traced_rays=int(np.asarray(before.x).size),
        trace_arrays_identical=identical,
    )
    result.check_true(
        "trace_is_element_wise_identical_after_round_trip",
        "invariant",
        all(identical.values()),
        f"element-wise equal for {sorted(identical)}",
    )

    efl_before = float(np.asarray(system.paraxial.f2()).ravel()[0])
    efl_after = float(np.asarray(new_system.paraxial.f2()).ravel()[0])
    result.record(efl_before_mm=efl_before, efl_after_mm=efl_after)
    result.check_close("paraxial_efl_survives_round_trip", "invariant", efl_after, efl_before, rel=0.0, abs_=0.0)
    return result


if __name__ == "__main__":
    raise SystemExit(standalone_main(TUTORIAL, run))
