"""Pinned values and declared conventions for the Chromatix integration.

Extracted by CHE-91 so the sibling modules can import what they need without
importing each other. Everything here is a declaration -- a supported set, a
tolerance, a pinned commit, a phasor convention -- and none of it executes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass



MODEL_ID = "M_WAVE_CHROMATIX"

_SUPPORTED_PROPAGATION = "angular_spectrum"
_EXPECTED_PHASOR = "exp(-i omega t)"
_SUPPORTED_DTYPES = {"complex64", "complex128"}

# CHE-35 (M3.6). Chromatix declares no time convention, so M1 recorded the
# phasor as forwarded-but-unchecked and the adapter warned rather than acted.
# The convention is now established by measurement rather than by reading source:
# a converging spherical wave written under this project's declaration --
# exp(-i omega t) in time with exp(+i k z) in space, i.e. pupil field
# exp(-i k sqrt(rho^2 + R^2)) -- focuses under asm_propagate, reaching 0.990 of
# the analytic Airy peak (pi a^2 / (lambda R))^2, while its complex conjugate
# does not (peak ratio 1008x, and off axis). See
# knowledge/solvers/chromatix/probes/m3_pupil_to_focus.py.
#
# Because the sign is now known, a mismatched input phasor is refused rather than
# forwarded: for a converging pupil field it is the difference between focusing
# and defocusing, and nothing downstream could tell the two apart.
_CHROMATIX_SPATIAL_FACTOR = "exp(+i k_z z) for z > 0"
_PHASOR_ESTABLISHED_BY = (
    "CHE-35 (M3.6), knowledge/solvers/chromatix/expected/m3_pupil_to_focus.json"
)

_PROPAGATION_METHODS = ("asm_propagate", "asm_carrier_removed")
_DEFAULT_PROPAGATION_METHOD = "asm_propagate"

# Absolute tolerance, in metres, on agreement between a declared target plane and
# the propagation distance. 1 pm: both come from the same float64 protocol
# literals, so any real disagreement is a modelling error, not round-off.
_PLANE_TOLERANCE_M = 1.0e-12

# Above this fraction of |u|^2 on the one-pixel border, the sampled window is
# truncating the field and any power or second-moment metric on it is
# window-limited. Reported as a diagnostic, never as a pass/fail gate on its own:
# CHE-35 measured it moving by only 2x between a run carrying 1.4e-1 relative
# intensity error from wraparound and a correctly padded one, so it is a weak
# wraparound indicator and the padding decision is made against a float64
# reference instead.
_EDGE_ENERGY_REPORTING_THRESHOLD = 0.05

# ---------------------------------------------------------------------------
# CHE-14 standalone wave baseline constants
# ---------------------------------------------------------------------------
_BASELINE_SEED = 20260811
_BASELINE_DEVICE = "cpu"
_BASELINE_DTYPE = "complex64"
_BASELINE_FIELD_KIND = "scalar"
_PINNED_COMMIT = "d24bdf0022835bb8ce1cdcc6aeafbc7fcb39daee"
_PINNED_VERSION = "0.6.0"
_PADDING_POLICIES = ("explicit", "auto_transfer", "none")
_OUTPUT_MODES = ("full", "same")
# 4096**2 complex64 = 128 MiB for the output array alone, and asm_propagate
# holds several arrays of that size simultaneously during the FFT pair. Above
# this the baseline returns a structured resource diagnostic instead of
# attempting the run: compute_padding_transfer is a worst-case (full-bandwidth)
# estimator and routinely proposes grids two orders of magnitude larger than a
# band-limited input actually needs (see conventions.md).
_DEFAULT_MAX_OUTPUT_PIXELS = 16_777_216
