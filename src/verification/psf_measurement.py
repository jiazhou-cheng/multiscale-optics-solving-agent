"""CHE-36 (M3.7): the PSF as a **measurement** on the terminal simulated field.

Why this is not a coupler
-------------------------
A coupler performs a physically meaningful change of representation.
``C_RAY_TO_WAVE`` is one: ``RayBundle -> ComplexField`` carries assumptions about
the OPL reference, the phase sign, the amplitude weighting, the sampling and the
handoff plane, and it refuses when they are not declared. Getting any of them
wrong produces a field that is wrong in a way no downstream check can name.

``ComplexField -> |u|^2`` is not that. It changes no representation, consults no
convention it does not already hold, and cannot be gotten wrong in more than one
way. It is an **observable** of the terminal state. ``C_FIELD_TO_PSF`` was
removed from ``registry/couplers.yaml`` for that reason, and the M3 graph now
terminates at the propagated ``ComplexField`` -- the last physical state the slice
evolves. This module runs after the graph.

The distinction is worth the paragraph because the registry is where the
architecture states what a coupler *is*. A trivial observable sitting in that
list, complete with a ``framework`` and a ``derivative.mode`` it never had
numerics for, made the category unfalsifiable.

What is frozen here, and why M3.8 needs it frozen
-------------------------------------------------
M3.8 (CHE-37) compares this measurement against an analytic Airy pattern and an
independent FFT/Fraunhofer oracle. Both oracles are defined only up to a
multiplicative scale, so the comparison is meaningless unless the normalization
is a stated choice rather than an accident:

* **Intensity is** ``|u|^2``. Taken in exactly one place,
  :meth:`contracts.PSF.from_complex_field`.
* **Normalization is required**, never defaulted, and recorded on the artifact.
  :data:`M3_ORACLE_NORMALIZATION` is the frozen choice for the M3 oracles and
  :data:`NORMALIZATION_RATIONALE` says why, including what it hides.
* **Axes come from the propagated field's output pitch.** See
  :func:`measure_psf_from_record`, which will refuse a record whose declared
  pitch is not the pitch the propagation reported.
* **Coherence model is stated**: :data:`COHERENCE_MODEL`. Monochromatic, fully
  coherent, scalar. Every oracle in M3.8 assumes it.

Absolute phase is deliberately not required
-------------------------------------------
``|u|^2`` is invariant under a global phase, so a field from CHE-40's
carrier-removed propagation path -- whose absolute phase is explicitly not
physical -- is admissible here, and ``benchmarks/slice_protocol.yaml`` requires
that path for any phase-insensitive M3 PSF. The measurement records which path it
consumed rather than inferring that the phase was meaningful.

Failure is the measurement contract's
-------------------------------------
Every refusal below is a :class:`ContractError` from the ``PSF`` contract or from
this module. The two invariants the retired registry entry declared,
``nonnegative_intensity`` and ``declared_psf_normalization``, survive under the
same names as executed checks in ``PSF.__post_init__``; they are now enforced by
the artifact rather than asserted by an edge.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from dataclasses import field as dataclass_field
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np

from core.artifacts import ArtifactRecord
from core.boundary import (
    ORIGIN_RULE,
    PSF,
    ComplexField,
    ContractCode,
    ContractError,
)
from core.specs import ArtifactKind

__all__ = [
    "COHERENCE_MODEL",
    "M3_ORACLE_NORMALIZATION",
    "NORMALIZATION_RATIONALE",
    "PSF_INVARIANTS",
    "PsfMeasurement",
    "PsfNormalization",
    "measure_psf",
    "measure_psf_from_record",
]

#: The slice's coherence model, stated on every PSF this module produces. M3.8's
#: analytic Airy oracle and its FFT/Fraunhofer oracle both assume it; an
#: incoherent or polychromatic aggregation would need a different comparison and
#: is out of M3 scope.
COHERENCE_MODEL = "monochromatic, fully coherent, scalar (single wavelength)"

#: The invariant names the retired C_FIELD_TO_PSF entry declared, kept verbatim so
#: M3.10's claim audit can trace them from the registry it removed to the contract
#: that now enforces them.
PSF_INVARIANTS = ("nonnegative_intensity", "declared_psf_normalization")

_ZERO_ENERGY_REMEDY = (
    "A dark field has no peak and no total to divide by. Normalizing anyway would "
    "yield NaN and be rejected one layer later as a non-finite intensity, which "
    "names the symptom rather than the cause. Check that the propagation ran and "
    "that the input field was not all zeros."
)


class PsfNormalization(StrEnum):
    """How the measured intensity is scaled. Always an explicit choice."""

    #: ``|u|^2`` in the field's own amplitude units, unscaled.
    RAW = "raw"
    #: ``|u|^2 / max(|u|^2)``. Dimensionless; the peak is 1 by construction.
    PEAK = "peak"
    #: ``|u|^2 / (sum(|u|^2) * dy * dx)``. Integrates to 1 over the sampled window.
    ENERGY = "energy"


_DECLARATIONS: dict[PsfNormalization, str] = {
    PsfNormalization.RAW: (
        "raw: intensity = |u|^2 in the field's own amplitude units. Not calibrated "
        "radiometric irradiance: the traced ray amplitudes carry Optiland's "
        "intensity weights, which have no SI calibration."
    ),
    PsfNormalization.PEAK: (
        "peak: intensity = |u|^2 / max(|u|^2) over the sampled window. "
        "Dimensionless; max == 1 by construction."
    ),
    PsfNormalization.ENERGY: (
        "energy: intensity = |u|^2 / (sum(|u|^2) * dy * dx) over the sampled "
        "window. Integrates to 1 over that window, so the value depends on the "
        "window: energy that left the grid is not in the denominator."
    ),
}

#: Frozen for M3.8's oracle comparisons.
M3_ORACLE_NORMALIZATION = PsfNormalization.PEAK

NORMALIZATION_RATIONALE = """\
M3.8 compares this PSF against an analytic Airy pattern and an independent
FFT/Fraunhofer PSF. Peak normalization is the frozen choice, for one reason and
with one cost.

The reason: neither oracle fixes an absolute scale that this slice could be held
to. The Airy formula gives a shape whose absolute irradiance depends on total
flux and aperture area, and the flux here descends from Optiland's per-ray
intensity weights, which are uncalibrated -- no step between the trace and the
propagated field converts them to watts. A raw comparison would therefore be
testing an arbitrary constant, and it would fail for a reason that has nothing to
do with whether the diffraction physics is right. Peak normalization removes
exactly that one degree of freedom and leaves the quantities M3.8 actually wants
to check -- peak position, first-null radius, profile shape -- untouched.

The cost, stated because M3.8 has to design around it: peak normalization is
blind to a constant multiplicative error. Any defect that scales the whole
intensity map by a constant -- an omitted per-ray area weight under uniform
sampling is the M2 example, and it was an exact constant -- survives it
unchanged. So the scale is not allowed to simply disappear:
``PsfMeasurement`` records ``raw_peak_intensity`` and ``raw_window_energy``
before scaling, and M3.8's energy ledger, not the normalized profile, is what
constrains the multiplicative factor. Energy normalization would not fix this
either: it hides the same constant, and it additionally makes the result depend
on how much energy left the observation window.
"""


@dataclass(frozen=True)
class PsfMeasurement:
    """A measured PSF plus everything the normalization removed.

    The raw scale is kept deliberately. A normalized PSF alone cannot answer
    whether the slice conserved energy, and M3.8's ledger needs an answer.
    """

    psf: PSF
    normalization: PsfNormalization
    #: The factor the raw ``|u|^2`` was multiplied by. 1.0 for ``RAW``.
    scale_factor: float
    #: ``max(|u|^2)`` before scaling, in the field's own amplitude units.
    raw_peak_intensity: float
    #: ``sum(|u|^2) * dy * dx`` before scaling, over the sampled window only.
    raw_window_energy: float
    #: ``(iy, ix)`` of the maximum. First occurrence in row-major order if tied.
    peak_index: tuple[int, int]
    #: ``(y, x)`` of the maximum in metres, under the pinned origin rule.
    peak_position_m: tuple[float, float]
    #: Fraction of PSF energy on the 1-pixel border of the window. A finite-window
    #: indicator for M3.8's energy ledger, not a correctness claim.
    border_energy_fraction: float
    provenance: dict[str, Any] = dataclass_field(default_factory=dict)

    @property
    def intensity(self) -> np.ndarray[Any, Any]:
        return self.psf.intensity

    @property
    def sample_pitch_m(self) -> tuple[float, float]:
        """The measurement's axes: the pitch of the field it measured."""
        return self.psf.sample_pitch_m

    def coordinates(self) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """``(y, x)`` coordinate vectors in metres, under the pinned origin rule."""
        ny, nx = self.psf.intensity.shape
        dy, dx = self.psf.sample_pitch_m
        y = (np.arange(ny, dtype=np.float64) - ny // 2) * dy
        x = (np.arange(nx, dtype=np.float64) - nx // 2) * dx
        return y, x

    def as_dict(self) -> dict[str, Any]:
        """Serializable measurement record, for a benchmark bundle's provenance."""
        return {
            "measurement": "psf",
            "is_a_graph_edge": False,
            "intensity_definition": "|u|^2",
            "normalization": str(self.normalization),
            "normalization_declaration": self.psf.normalization,
            "scale_factor": self.scale_factor,
            "raw_peak_intensity": self.raw_peak_intensity,
            "raw_window_energy": self.raw_window_energy,
            "raw_energy_units": (
                "the field's own amplitude units squared, times m^2. Not watts: the "
                "ray amplitudes are uncalibrated intensity weights."
            ),
            "peak_index": list(self.peak_index),
            "peak_position_m": list(self.peak_position_m),
            "sample_pitch_m": list(self.psf.sample_pitch_m),
            "pitch_source": "propagated field output pitch (dx_out)",
            "border_energy_fraction": self.border_energy_fraction,
            "coherence_model": self.psf.coherence_model,
            "wavelength_m": self.psf.wavelength_m,
            "shape": [int(n) for n in self.psf.intensity.shape],
            "origin_rule": self.psf.frame.origin_rule,
            "axis_order": self.psf.frame.axis_order,
            "invariants_enforced": list(PSF_INVARIANTS),
            "invariants_enforced_by": "contracts.PSF.__post_init__",
            "gradient_claim": "none. M3 is forward-only; no gradient is claimed.",
            **self.provenance,
        }

    def to_artifact_record(self, *, artifact_id: str, uri: str | Path) -> ArtifactRecord:
        record = self.psf.to_artifact_record(artifact_id=artifact_id, uri=uri)
        record.metadata.update(
            {
                "measurement": self.as_dict(),
                # Restated at the top level because a consumer reading pitch off an
                # artifact should not have to know this measurement's schema.
                "raw_peak_intensity": self.raw_peak_intensity,
                "raw_window_energy": self.raw_window_energy,
            }
        )
        return record


def measure_psf(
    field: ComplexField,
    *,
    normalization: PsfNormalization,
) -> PsfMeasurement:
    """Measure ``|u|^2`` on a field. No propagation, no resampling, no new physics.

    ``normalization`` is required. There is no default, because M3.8's oracle
    comparison is only meaningful against a stated scale and an implicitly
    normalized PSF entering an oracle is the specific failure this signature
    exists to prevent.

    The axes are the field's own ``sample_pitch_m``. For a propagated field that
    must be the output pitch; :func:`measure_psf_from_record` is the entry point
    that can check it, and it is the one a benchmark driver should call.
    """
    normalization = PsfNormalization(normalization)

    if field.frame.origin_rule != ORIGIN_RULE:
        raise ContractError(
            ContractCode.FRAME_MISMATCH,
            f"peak position is reported under {ORIGIN_RULE!r}; this field declares "
            f"{field.frame.origin_rule!r}",
            declaration="frame.origin_rule",
            remedy=(
                "A different centring shifts every reported coordinate by up to half "
                "a pixel, which is a large fraction of an Airy radius at M3 sampling."
            ),
        )

    raw = np.abs(field.u) ** 2
    dy, dx = field.sample_pitch_m
    raw_peak = float(raw.max())
    raw_energy = float(raw.sum() * dy * dx)

    if normalization is PsfNormalization.RAW:
        scale = 1.0
    elif normalization is PsfNormalization.PEAK:
        if raw_peak <= 0.0:
            raise ContractError(
                ContractCode.EMPTY_ENSEMBLE,
                "the field carries no energy, so it has no peak to normalize to",
                declaration="normalization",
                remedy=_ZERO_ENERGY_REMEDY,
            )
        scale = 1.0 / raw_peak
    else:
        if raw_energy <= 0.0:
            raise ContractError(
                ContractCode.EMPTY_ENSEMBLE,
                "the field carries no energy, so it has no total to normalize to",
                declaration="normalization",
                remedy=_ZERO_ENERGY_REMEDY,
            )
        scale = 1.0 / raw_energy

    # |u|^2 is taken in exactly one place -- the PSF contract -- and the scale is
    # then applied to the intensity itself. Not by rescaling the amplitude and
    # squaring again: that round trip would put a sqrt and a square between the
    # field and the number M3.8 compares to an oracle, for no reason.
    #
    # dataclasses.replace re-runs PSF.__post_init__, so the invariants (finite,
    # non-negative, declared normalization) are enforced on the array that is
    # actually returned. The refusal belongs to the measurement contract, not to
    # this function.
    psf = PSF.from_complex_field(
        field,
        normalization=_DECLARATIONS[normalization],
        coherence_model=COHERENCE_MODEL,
    )
    if scale != 1.0:
        psf = replace(psf, intensity=psf.intensity * scale)

    peak_index = np.unravel_index(int(np.argmax(psf.intensity)), psf.intensity.shape)
    iy, ix = int(peak_index[0]), int(peak_index[1])
    ny, nx = psf.intensity.shape

    return PsfMeasurement(
        psf=psf,
        normalization=normalization,
        scale_factor=scale,
        raw_peak_intensity=raw_peak,
        raw_window_energy=raw_energy,
        peak_index=(iy, ix),
        peak_position_m=((iy - ny // 2) * dy, (ix - nx // 2) * dx),
        border_energy_fraction=_border_energy_fraction(raw),
        provenance={
            "measured_from_field": field.provenance.get("source_artifact_id"),
            "field_reference_plane": field.reference_plane.name,
            "field_reference_plane_z_m": field.reference_plane.z_m,
            "field_pad_width": field.pad_width,
            "field_padded": field.padded,
            "absolute_phase_required": False,
            "absolute_phase_note": (
                "|u|^2 is invariant under a global phase, so a carrier-removed field "
                "(CHE-40) is admissible here and the slice protocol requires that "
                "path for a phase-insensitive PSF."
            ),
        },
    )


def measure_psf_from_record(
    record: ArtifactRecord,
    *,
    normalization: PsfNormalization,
    expected_output_sample_pitch_m: tuple[float, float] | None = None,
    pitch_rtol: float = 1e-9,
) -> PsfMeasurement:
    """Measure the PSF of a propagated-field artifact, checking its axes.

    ``expected_output_sample_pitch_m`` is the pitch the propagation itself
    reported (``output_sample_pitch_m`` / ``dx_out`` on the Chromatix result). Pass
    it. The pitch on the record is what sets every coordinate this measurement
    reports, and the Chromatix adapter carries both an input and an output pitch:
    its graph path writes ``dx_out`` onto the output artifact, but its baseline
    summary's ``field_metadata.sample_pitch_m`` is the *input* pitch. Reading the
    input pupil pitch here would rescale every angular comparison in M3.8 by a
    constant while leaving the intensity map entirely plausible.

    Angular-spectrum propagation happens to preserve the pitch, which makes the
    check cheap rather than vacuous: it is the same assertion for a method that
    does not, and the failure it guards against is silent.
    """
    if record.kind is not ArtifactKind.COMPLEX_FIELD:
        raise ContractError(
            ContractCode.ARTIFACT_KIND_MISMATCH,
            f"a PSF is measured on {ArtifactKind.COMPLEX_FIELD.value!r}, got {record.kind.value!r}",
            artifact_id=record.id,
            remedy=(
                "The measurement consumes the terminal field of the graph. It is not "
                "an edge and does not accept a ray bundle or a wavefront sample set."
            ),
        )

    field = ComplexField.from_artifact_record(record)

    if expected_output_sample_pitch_m is not None:
        expected = (
            float(expected_output_sample_pitch_m[0]),
            float(expected_output_sample_pitch_m[1]),
        )
        declared = field.sample_pitch_m
        if not all(
            np.isclose(got, want, rtol=pitch_rtol, atol=0.0)
            for got, want in zip(declared, expected, strict=True)
        ):
            raise ContractError(
                ContractCode.SAMPLE_PITCH_MISMATCH,
                f"the field declares sample_pitch_m={declared!r} but the propagation "
                f"reported an output pitch of {expected!r}",
                declaration="metadata.sample_pitch",
                artifact_id=record.id,
                remedy=(
                    "A PSF's axes must come from the propagated field's OUTPUT pitch. "
                    "If the input pupil pitch was written here instead, every distance "
                    "this measurement reports -- peak position, first-null radius -- is "
                    "scaled by the ratio, and the intensity map looks correct anyway."
                ),
            )

    measurement = measure_psf(field, normalization=normalization)
    measurement.provenance.update(
        {
            "measured_from_artifact": record.id,
            "measured_from_uri": record.uri,
            "propagation_method": record.metadata.get("propagation_method"),
            "output_pitch_checked_against_propagation": (
                expected_output_sample_pitch_m is not None
            ),
            "absolute_phase_is_physical": record.metadata.get("absolute_phase_is_physical"),
        }
    )
    return measurement


def _border_energy_fraction(intensity: np.ndarray[Any, Any]) -> float:
    """Fraction of energy on the 1-pixel border of the sampled window.

    The same indicator the Chromatix adapter reports for a field, restated for the
    measured intensity so M3.8 can attribute a ledger deficit to the finite
    observation window. CHE-35 measured it moving by only ~2x across a run that
    carried 1.4e-1 relative intensity error from wraparound, so it notices window
    truncation and does not certify padding.
    """
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
