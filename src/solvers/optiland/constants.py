"""Pinned values and declarations shared across the Optiland integration.

Extracted by CHE-91 so `requests.py`, `execution.py`, `pupil.py`,
`provenance.py` and `adapter.py` can each import what they need without
importing each other. Every entry is a *declaration* -- a unit conversion, a
supported set, a default, a warning text -- and none of it executes anything.
"""

from __future__ import annotations

import re

from registry.prescriptions import (
    prescription_names,
)

MODEL_ID = "M_RAY_OPTILAND"

_SUPPORTED_BACKENDS = ("numpy", "torch")

# Since CHE-56 (PB5) every supported system -- bundled or adapter-owned -- is a
# canonical prescription in registry/prescriptions.py, built by the single
# generic builder in optiland_builder.py. There is no longer one construction
# path for bundled samples and another for adapter-owned ones, and the name
# `sample` survives only as the lookup key for a canonical prescription.
#
# CHE-32 (M3.3)'s reason for owning a prescription at all still holds: no
# bundled system qualifies as M3.2's diffraction-limited reference.
# tmp_probes/optiland_exit_pupil_probe.py measured every system in
# optiland.samples.objectives on axis at 550 nm and the best is WideAngle100FOV
# at 0.36 waves peak-to-valley, against Rayleigh's 0.25.
#
# What CHE-56 changes is that an unnamed prescription is no longer refused: a
# caller may pass a canonical `OpticalSystemSpec` (or its serialized mapping)
# through `config['prescription']`. What is still refused is an arbitrary
# Python Optiland object through the `system` input port -- that is not a
# validated contract, and the canonical schema exists precisely so it does not
# need to be.
_SUPPORTED_SAMPLES = prescription_names()

# M3-SINGLET-REF's numbers (frozen by M3.2 in benchmarks/slice_protocol.yaml)
# moved to the prescription itself, registry/prescriptions.py, so there is one
# definition rather than a copy here and a construction site there.

#: The DEFAULT device and precision, no longer the only supported ones (CHE-61).
#: Optiland's own API is `set_precision(Literal['float32','float64'])` and
#: `set_device(str)` (torch backend only, `BackendCapabilityError` otherwise), so
#: what this adapter may execute is declared once in
#: `core/capabilities.py::OPTILAND_CAPABILITIES` and validated from there. These
#: two constants remain the defaults, which is what keeps every existing request
#: -- and L1-RAY-01's recorded fingerprint -- byte-identical.
_DEFAULT_DEVICE = "cpu"
_DEFAULT_DTYPE = "float64"
_SUPPORTED_DEVICE = _DEFAULT_DEVICE
_SUPPORTED_DTYPE = _DEFAULT_DTYPE

#: float64 direction-norm bound, unchanged. A float32 trace cannot meet it and
#: must not be asked to: see `_direction_norm_tolerance`.
_DIRECTION_NORM_TOLERANCE = 1e-12
_GEOMETRY_M_PER_MM = 1e-3
_WAVELENGTH_M_PER_UM = 1e-6
_BASELINE_SEED = 20260811

# The only design-parameter path characterized by
# knowledge/solvers/optiland/probes/gradient_probe.py.
_VALIDATED_DESIGN_PARAMETER_PATTERN = re.compile(r"^surfaces\.surfaces\[(\d+)\]\.geometry\.radius$")

_DEFAULT_WAVELENGTH = 0.55  # micrometres; verified by CHE-12
_DEFAULT_NUM_RAYS = 16
_DEFAULT_HX = 0.0
_DEFAULT_HY = 0.0

_MISSING_WAVEFRONT_METADATA = ["amplitude", "polarization", "pupil_mask"]

# CHE-32: which plane the exported rays are referenced to. The default stays
# "image_surface" so that L1-RAY-01's recorded scientific fingerprint
# (43dab1ee...) reproduces bit-identically -- M3.3 adds a plane, it does not move
# the existing one.
_DEFAULT_SAMPLE = "ReverseTelephoto"
_SUPPORTED_HANDOFF_PLANES = ("image_surface", "exit_pupil")
_DEFAULT_HANDOFF_PLANE = "image_surface"

_OPD_WARNING = (
    "RealRays.opd is preserved in Optiland-native values. CHE-30 established the "
    "convention -- absolute accumulated optical path in the geometry unit (mm), "
    "index-weighted, referenced to the ray launch state -- but for an infinite "
    "object that launch plane is aperture-dependent, so the exported value is a "
    "piston of order 1e4 waves plus the wavefront. It must not be read as a phase "
    "without subtracting a declared reference; see conventions.opd_reference."
)
