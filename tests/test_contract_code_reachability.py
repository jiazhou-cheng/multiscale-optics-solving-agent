"""Every declared ``ContractCode``, shown reachable by the smallest real trigger.

CHE-108 (M1.3), part B0.2. ``tests/test_b0_families.py`` already proves every
code has a catalogue entry, which is a claim about the catalogue. This file
proves every code is *emitted*, which is a claim about the implementation, and
the two are not the same: a code nothing has ever raised cannot be trusted to
raise when it matters, and a catalogue entry for it is documentation of a
promise rather than of a behaviour.

The precedent is deliberate. The retired V1 agent suite carried
``TestOutcomeCodesAreReachable`` on exactly this argument for its eight outcome
codes; this applies the same standard to the nineteen contract codes.

What each case asserts
----------------------
* the shipping code raises ``ContractError`` -- not a bare ``ValueError``, not a
  ``KeyError`` from three layers down;
* the code is **the** code, compared by identity against the enum member;
* a reason exists and is not empty;
* a remedy is available, from the raise site or from the catalogue, and the case
  records which -- a refusal with no remedy is a dead end dressed as a
  diagnostic;
* nothing was returned. A refusal that also produced a partially-populated
  artifact would let a downstream node consume it, which is the fabrication the
  contract layer exists to prevent.

The triggers are the smallest thing that reaches each code on purpose. A trigger
that needs a 500-ray trace to reach ``NON_UNIT_DIRECTION`` is testing the trace.

Two looked as though they would need a real Optiland run -- the off-axis
object-space term and the reference-plane mismatch -- and neither does, because
what the coupler refuses is a *missing declaration* rather than a wrong number.
Both are reachable from a record that carries every other declaration the coupler
needs and is missing exactly one, which is the sharper test anyway: it isolates
which declaration the code is about. The whole suite therefore runs in the
default gate in well under a second.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from core.artifacts import ArtifactRecord
from core.boundary import (
    PSF,
    ComplexField,
    ContractCode,
    ContractError,
    Frame,
    RayBundle,
    ReferencePlane,
)
from core.specs import ArtifactKind, Device
from verification.refusals import REFUSAL_CATALOGUE

# --------------------------------------------------------------------------- #
# Shared minimal fixtures. Small on purpose: see the module docstring.
# --------------------------------------------------------------------------- #

_PLANE = ReferencePlane(name="test plane", z_m=0.0)
_PITCH = (1e-6, 1e-6)
_WAVELENGTH_M = 5.5e-7


def _field(**overrides):
    kwargs = {
        "u": np.ones((4, 4), dtype=np.complex128),
        "sample_pitch_m": _PITCH,
        "wavelength_m": _WAVELENGTH_M,
        "reference_plane": _PLANE,
    }
    kwargs.update(overrides)
    return ComplexField(**kwargs)


def _bundle(**overrides):
    kwargs = {
        "positions_m": np.zeros((3, 3)),
        "directions": np.tile(np.array([0.0, 0.0, 1.0]), (3, 1)),
        "wavelength_m": _WAVELENGTH_M,
        "reference_plane": _PLANE,
    }
    kwargs.update(overrides)
    return RayBundle(**kwargs)


def _complex_field_record(metadata: dict, *, uri: Path) -> ArtifactRecord:
    """A ``complex_field`` record on disk, so the file-loading path is real."""
    np.save(uri, np.ones((4, 4), dtype=np.complex64))
    return ArtifactRecord(
        id="reachability-field",
        kind=ArtifactKind.COMPLEX_FIELD,
        uri=str(uri),
        dtype="complex64",
        shape=(4, 4),
        metadata=metadata,
    )


_FIELD_METADATA = {
    "wavelength": _WAVELENGTH_M,
    "sample_pitch": [1e-6, 1e-6],
    "phasor": "exp(-i omega t)",
    "normalization": "none",
    "pad_width": 0,
    "coordinate_frame": "right-handed",
}


# --------------------------------------------------------------------------- #
# The triggers
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Trigger:
    """The smallest real call that reaches one code."""

    code: ContractCode
    #: Which shipping surface raises it. Recorded so a reader can tell a
    #: boundary-artifact refusal from a coupler refusal from a measurement one.
    surface: str
    #: What is being asked for, in one line.
    request: str
    call: Callable[[Path], object]
    #: Marks a trigger that needs a real solver run.
    needs_optiland: bool = False


def _off_ring_quadrature(_tmp: Path) -> object:
    """A per-ray weight asked of pupil coordinates that are not on any ring."""
    from couplers.quadrature import hexapolar_ring_index

    # Radii deliberately between rings: 0.5/3 is not j/3 for any integer j.
    return hexapolar_ring_index(
        np.array([0.5 / 3.0]), np.array([0.0]), 3
    )


def _zero_spectrum_sampling(_tmp: Path) -> object:
    """Magnitude-proportional sampling of a field whose spectrum is zero.

    ``p_mag`` is undefined there, and no amount of ``1/p`` reweighting recovers
    it, so the estimator refuses rather than dividing by zero and emitting rays
    with infinite amplitudes.
    """
    from couplers.wave_to_ray import SamplingDensity, wave_to_ray

    return wave_to_ray(
        _field(u=np.zeros((8, 8), dtype=np.complex128)),
        count=4,
        density_kind=SamplingDensity.MAGNITUDE,
        rng=np.random.default_rng(0),
    )


def _pitch_mismatch(tmp: Path) -> object:
    """A PSF measured against a pitch the propagation did not report."""
    from verification.psf_measurement import PsfNormalization, measure_psf_from_record

    record = _complex_field_record(dict(_FIELD_METADATA), uri=tmp / "pitch.npy")
    return measure_psf_from_record(
        record,
        normalization=PsfNormalization.PEAK,
        expected_output_sample_pitch_m=(2e-6, 2e-6),
    )


def _full_field_on_a_conformal_substrate(_tmp: Path) -> object:
    """The wrong diffractive model for the declared surface (CHE-142).

    ``FULL_FIELD``'s central step is one coherent accumulation onto the single
    common plane every incident ray crosses. A conformal substrate has no such
    plane (SI S10), and the accumulation would nevertheless *run* -- folding rays
    from different tangent frames into one field and returning something that
    looks like a diffraction pattern. Hence a refusal, and hence
    ``could_have_proceeded``.

    The smallest thing that reaches it: a 4x4 surface and a one-ray bundle. The
    refusal happens before any transform, so nothing here has to be physical.
    """
    from couplers.interaction import (
        DiffractiveModel,
        DiffractiveSurface,
        FullFieldParameters,
        diffractive_interaction,
    )
    from couplers.patch import Substrate

    return diffractive_interaction(
        RayBundle(
            positions_m=np.zeros((1, 3)),
            directions=np.array([[0.0, 0.0, 1.0]]),
            wavelength_m=_WAVELENGTH_M,
            reference_plane=_PLANE,
        ),
        DiffractiveSurface(
            transmission=np.ones((4, 4), dtype=np.complex128),
            sample_pitch_m=_PITCH,
            plane=_PLANE,
            substrate=Substrate.CONFORMAL,
            radius_m=1e-3,
        ),
        model=DiffractiveModel.FULL_FIELD,
        parameters=FullFieldParameters(launch_positions_xy_m=np.zeros((1, 2))),
    )


def _declared_gpu_over_host_data(tmp: Path) -> object:
    """A record that says ``gpu`` over an array that is on the host.

    Reachable because CHE-108 added the comparison. Before that,
    ``from_artifact_record`` compared nothing between the declared device and the
    array it was handed, so one line assigning the *requested* device to the
    record produced a reported accelerator run that happened on the CPU.
    """
    record = ArtifactRecord(
        id="declared-gpu",
        kind=ArtifactKind.COMPLEX_FIELD,
        uri=str(tmp / "unused.npy"),
        dtype="complex64",
        shape=(4, 4),
        device=Device.GPU,
        metadata=dict(_FIELD_METADATA),
    )
    return ComplexField.from_artifact_record(
        record, array=np.ones((4, 4), dtype=np.complex64)
    )


def _plane_mismatch(_tmp: Path) -> object:
    """A consumer declaring a plane the producer did not export at."""
    from couplers.handoff import DeclaredHandoffPlane, declare_coherent_bundle

    record = ArtifactRecord(
        id="plane-mismatch",
        kind=ArtifactKind.RAY_BUNDLE,
        uri="memory://plane-mismatch",
        metadata={
            "length_unit": "m",
            "wavelength_m": _WAVELENGTH_M,
            "conventions": {
                "handoff_plane": "image_surface",
                "reference_plane": "image_surface",
                "reference_plane_z_m": 0.0,
            },
        },
    )
    arrays = {
        "x_m": np.zeros(3),
        "y_m": np.zeros(3),
        "z_m": np.zeros(3),
        "L": np.zeros(3),
        "M": np.zeros(3),
        "N": np.ones(3),
        "intensity": np.ones(3),
        "opd_native": np.zeros(3),
    }
    return declare_coherent_bundle(
        record,
        declared_plane=DeclaredHandoffPlane(handoff_plane="exit_pupil", z_m=1e-3),
        arrays=arrays,
    )


def _off_axis_without_the_object_space_term(_tmp: Path) -> object:
    """An off-axis record with no ``object_space_reference_offset_m``.

    The term is ``n_object * (d0 . r_launch)``, linear in the launch coordinate,
    so on axis it is a constant the chief-ray subtraction removes exactly -- and
    that is why the coupler accepts an on-axis record without it and refuses an
    off-axis one. The field angle is what makes this reachable.
    """
    from couplers.handoff import DeclaredHandoffPlane, declare_coherent_bundle

    record = ArtifactRecord(
        id="off-axis-no-reference",
        kind=ArtifactKind.RAY_BUNDLE,
        uri="memory://off-axis",
        metadata={
            "length_unit": "m",
            "wavelength_m": _WAVELENGTH_M,
            "Hx": 0.0,
            "Hy": 0.2,
            "conventions": {
                # Matching, so the refusal that fires is the object-space one
                # rather than the plane check standing in front of it.
                "handoff_plane": "exit_pupil",
                "reference_plane": "exit_pupil",
                "reference_plane_z_m": 0.0,
                # Both of these are declared so the refusal that fires is the
                # off-axis one and not a declaration check standing in front of
                # it. The point of the trigger is which code the OFF-AXIS FIELD
                # produces once everything else the coupler needs is present.
                "image_space_refractive_index": 1.0,
                "exit_pupil": {"location_from_image_m": 0.0},
            },
        },
    )
    arrays = {
        "x_m": np.array([0.0, 1e-4, -1e-4]),
        "y_m": np.array([0.0, 1e-4, -1e-4]),
        "z_m": np.zeros(3),
        "L": np.zeros(3),
        "M": np.zeros(3),
        "N": np.ones(3),
        "intensity": np.ones(3),
        "opd_native": np.zeros(3),
    }
    return declare_coherent_bundle(
        record,
        declared_plane=DeclaredHandoffPlane(handoff_plane="exit_pupil", z_m=0.0),
        arrays=arrays,
    )


TRIGGERS: tuple[Trigger, ...] = (
    Trigger(
        ContractCode.MISSING_DECLARATION,
        surface="core.boundary.PSF",
        request="a PSF with no declared normalization",
        call=lambda _t: PSF(
            intensity=np.ones((4, 4)),
            sample_pitch_m=_PITCH,
            wavelength_m=_WAVELENGTH_M,
            normalization="",
        ),
    ),
    Trigger(
        ContractCode.UNIT_NOT_SI,
        surface="core.boundary.ComplexField",
        request="a sample pitch that is not a positive length in metres",
        call=lambda _t: _field(sample_pitch_m=(0.0, 1e-6)),
    ),
    Trigger(
        ContractCode.PHASOR_MISMATCH,
        surface="core.boundary.ComplexField",
        request="the opposite phasor sign convention",
        call=lambda _t: _field(phasor="exp(+i omega t)"),
    ),
    Trigger(
        ContractCode.AXIS_ORDER_MISMATCH,
        surface="core.boundary.Frame.require_field_axis_order",
        request="a field declared (x, y) rather than (y, x)",
        call=lambda _t: _field(frame=Frame(axis_order="(x, y)")),
    ),
    Trigger(
        ContractCode.FRAME_MISMATCH,
        surface="core.boundary.Frame",
        request="a left-handed frame",
        call=lambda _t: Frame(handedness="left-handed"),
    ),
    Trigger(
        ContractCode.SHAPE_MISMATCH,
        surface="core.boundary.ComplexField",
        request="a 1-D array as a field",
        call=lambda _t: _field(u=np.ones(4, dtype=np.complex128)),
    ),
    Trigger(
        ContractCode.NON_FINITE,
        surface="core.boundary.ComplexField",
        request="a field containing a NaN",
        call=lambda _t: _field(
            u=np.array([[np.nan, 1.0], [1.0, 1.0]], dtype=np.complex128)
        ),
    ),
    Trigger(
        ContractCode.NON_UNIT_DIRECTION,
        surface="core.boundary.RayBundle",
        request="direction cosines whose norm is not one",
        call=lambda _t: _bundle(directions=np.tile(np.array([1.0, 1.0, 1.0]), (3, 1))),
    ),
    Trigger(
        ContractCode.EMPTY_ENSEMBLE,
        surface="couplers.wave_to_ray",
        request="magnitude-proportional sampling of an identically zero spectrum",
        call=_zero_spectrum_sampling,
    ),
    Trigger(
        ContractCode.AMPLITUDE_IS_A_WEIGHT,
        surface="core.boundary.RayBundle.require_coherent",
        request="a real intensity weight read as a complex amplitude",
        call=lambda _t: _bundle(
            weight=np.ones(3), weight_semantics="optiland intensity"
        ).require_coherent(),
    ),
    Trigger(
        ContractCode.OPL_REFERENCE_UNVERIFIED,
        surface="core.boundary.RayBundle.require_coherent",
        request="a coherent bundle with an amplitude and no declared optical path",
        call=lambda _t: _bundle(
            amplitude=np.ones(3, dtype=np.complex128)
        ).require_coherent(),
    ),
    Trigger(
        ContractCode.REFERENCE_PLANE_MISMATCH,
        surface="couplers.handoff.declare_coherent_bundle",
        request="a consumer declaring the exit pupil for a record exported at the image",
        call=_plane_mismatch,
    ),
    Trigger(
        ContractCode.PAD_STATE_UNKNOWN,
        surface="core.boundary.ComplexField.from_artifact_record",
        request="a field record that does not declare its pad width",
        call=lambda t: ComplexField.from_artifact_record(
            _complex_field_record(
                {k: v for k, v in _FIELD_METADATA.items() if k != "pad_width"},
                uri=t / "nopad.npy",
            )
        ),
    ),
    Trigger(
        ContractCode.NEGATIVE_INTENSITY,
        surface="core.boundary.PSF",
        request="a negative value in an intensity array",
        call=lambda _t: PSF(
            intensity=np.array([[-1.0, 1.0], [1.0, 1.0]]),
            sample_pitch_m=_PITCH,
            wavelength_m=_WAVELENGTH_M,
            normalization="peak",
        ),
    ),
    Trigger(
        ContractCode.ARTIFACT_KIND_MISMATCH,
        surface="core.boundary.ComplexField.from_artifact_record",
        request="a ray-bundle record where a complex field is required",
        call=lambda _t: ComplexField.from_artifact_record(
            ArtifactRecord(
                id="wrong-kind",
                kind=ArtifactKind.RAY_BUNDLE,
                uri="memory://wrong-kind",
            )
        ),
    ),
    Trigger(
        ContractCode.SAMPLE_PITCH_MISMATCH,
        surface="verification.psf_measurement.measure_psf_from_record",
        request="a PSF measured against a pitch the propagation did not report",
        call=_pitch_mismatch,
    ),
    Trigger(
        ContractCode.OBJECT_SPACE_REFERENCE_MISSING,
        surface="couplers.handoff.declare_coherent_bundle",
        request="an off-axis record with no object-space reference term",
        call=_off_axis_without_the_object_space_term,
    ),
    Trigger(
        ContractCode.NON_HEXAPOLAR_SAMPLING,
        surface="couplers.quadrature.hexapolar_ring_index",
        request="a per-ray area weight for a pupil coordinate between rings",
        call=_off_ring_quadrature,
    ),
    Trigger(
        ContractCode.REPRESENTATION_INCONSISTENT,
        surface="core.boundary.ComplexField.from_artifact_record",
        request="a record declaring gpu over an array that is on the host",
        call=_declared_gpu_over_host_data,
    ),
    Trigger(
        ContractCode.MODEL_NOT_APPLICABLE,
        surface="couplers.interaction.diffractive_interaction",
        request="diffractive model 'full_field' for a conformal substrate",
        call=_full_field_on_a_conformal_substrate,
    ),
)


# --------------------------------------------------------------------------- #
# Completeness
# --------------------------------------------------------------------------- #


def test_every_declared_code_has_a_trigger() -> None:
    """A code with no trigger is a code nobody has shown can fire.

    M1.3's criterion is "shown reachable, or deleted as dead". This is the half
    that fails when a new code arrives without one.
    """
    triggered = {t.code for t in TRIGGERS}
    missing = sorted(c.value for c in ContractCode if c not in triggered)
    assert not missing, (
        f"these declared codes have no reachability trigger: {missing}. Either add "
        "the smallest real call that emits it, or delete the code as dead -- a "
        "catalogue entry is not evidence that anything raises it."
    )


def test_no_trigger_names_a_code_that_no_longer_exists() -> None:
    for trigger in TRIGGERS:
        assert trigger.code in set(ContractCode)


def test_the_triggers_are_distinct() -> None:
    """One trigger per code, so a case cannot cover for a missing one."""
    codes = [t.code for t in TRIGGERS]
    assert len(codes) == len(set(codes))


# --------------------------------------------------------------------------- #
# Reachability, one code at a time
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("trigger", TRIGGERS, ids=lambda t: t.code.value)
def test_the_code_is_reachable_and_actionable(trigger: Trigger, tmp_path: Path) -> None:
    """The whole B0.2 assertion, per code."""
    with pytest.raises(ContractError) as excinfo:
        result = trigger.call(tmp_path)
        # Reached only if nothing raised. Naming what came back makes a silent
        # acceptance a legible failure rather than a bare "DID NOT RAISE".
        pytest.fail(
            f"{trigger.code.value} was not raised by {trigger.surface} for "
            f"{trigger.request!r}; it returned {type(result).__name__}. Either the "
            "trigger no longer reaches the code, or the code has become dead."
        )

    error = excinfo.value
    assert error.code is trigger.code, (
        f"{trigger.surface} raised {error.code} for {trigger.request!r}, not "
        f"{trigger.code}. A wrong specific code is worse than a general one, "
        "because the caller acts on it."
    )

    # A reason. Not merely a code: the message has to name what was wrong.
    message = str(error)
    assert message.strip()
    assert trigger.code.value in message

    # A remedy, from the raise site or from the catalogue. Both are actionable;
    # having neither is a dead end.
    catalogue = REFUSAL_CATALOGUE[trigger.code.value]
    remedy = error.remedy or catalogue.remedy
    assert remedy and remedy.strip(), (
        f"{trigger.code.value} carries no remedy at the raise site and none in the "
        "catalogue"
    )
    assert catalogue.trigger.strip()


@pytest.mark.parametrize("trigger", TRIGGERS, ids=lambda t: t.code.value)
def test_no_refusal_returns_a_usable_artifact(trigger: Trigger, tmp_path: Path) -> None:
    """Nothing comes back. Asserted rather than assumed.

    An `AGENTS.md` non-negotiable, mechanically checked: a refusal that also
    produced a partially populated artifact would let a downstream node consume
    it, and every number computed from it would be a number about nothing.
    """
    returned: object | None = None
    try:
        returned = trigger.call(tmp_path)
    except ContractError:
        returned = None
    assert returned is None, (
        f"{trigger.code.value}: {trigger.surface} returned "
        f"{type(returned).__name__} instead of refusing"
    )


@pytest.mark.parametrize("trigger", TRIGGERS, ids=lambda t: t.code.value)
def test_the_catalogue_and_the_raise_site_agree_on_the_outcome(
    trigger: Trigger, tmp_path: Path
) -> None:
    """The five negative outcomes stay distinguishable from the diagnostic alone.

    This is the assertion that caught the collapse CHE-108 fixed: the catalogue
    classified ``OPL_REFERENCE_UNVERIFIED`` as ``blocked`` while the record path
    reported ``invalid_configuration``, so the same code meant two things
    depending on which way a caller read it.
    """
    from verification.status import VerificationStatus

    entry = REFUSAL_CATALOGUE[trigger.code.value]
    assert entry.status is not VerificationStatus.OK
    # `could_have_proceeded` is what separates `blocked` from the rest, and the
    # catalogue is the only place that records it.
    if entry.status is VerificationStatus.BLOCKED:
        assert entry.could_have_proceeded, (
            f"{trigger.code.value} is classified blocked, which means the component "
            "could have proceeded and chose not to. If it could not have, it is "
            "invalid_configuration or unsupported instead."
        )


# --------------------------------------------------------------------------- #
# The triggers themselves are minimal, and that is a property worth keeping
# --------------------------------------------------------------------------- #


def test_no_trigger_needs_a_solver_it_does_not_declare() -> None:
    """None of the nineteen needs a real trace, which is the useful outcome.

    Two candidates did look like they would -- the off-axis object-space term and
    the reference-plane mismatch -- and both turned out to be reachable from a
    hand-built record carrying the *declarations* under test, because what the
    coupler refuses is a missing declaration rather than a wrong number. That
    keeps the whole suite in the default gate at negligible cost.
    """
    assert not [t.code.value for t in TRIGGERS if t.needs_optiland]


def test_the_suite_covers_every_shipping_surface_that_refuses() -> None:
    """Reported rather than asserted exhaustive.

    The point is that the codes are not all raised by one file: a suite whose
    triggers all went through ``core.boundary`` would prove the artifact
    validators work and say nothing about the couplers or the measurement.
    """
    surfaces = {t.surface.split(".")[0] + "." + t.surface.split(".")[1] for t in TRIGGERS}
    assert {"core.boundary", "couplers.handoff", "couplers.quadrature"} <= surfaces
    assert len(surfaces) >= 5, sorted(surfaces)


def test_a_trigger_that_stops_working_is_a_failure_not_a_skip() -> None:
    """The guard on the guard.

    A reachability suite whose cases silently degrade to conditional skips
    reports full coverage and has none. This asserts there are none: every
    trigger runs in the default suite, unconditionally.

    The marker name is assembled rather than written literally, because a test
    that searches its own source for a string cannot also contain that string in
    prose -- which is how the first version of this test failed on its own
    docstring.
    """
    source = Path(__file__).read_text(encoding="utf-8")
    conditional_skip = "skip" + "if"
    assert conditional_skip not in source
    # And no conditional imports either: every one of the nineteen triggers
    # reaches its code through a dependency this image is required to have, so
    # there is nothing here that can quietly stop running.
    conditional_import = "importor" + "skip"
    assert conditional_import not in source
