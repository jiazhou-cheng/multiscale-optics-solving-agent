"""CHE-36 (M3.7): the PSF measurement, and the retirement of C_FIELD_TO_PSF.

Two things are being pinned here.

The first is architectural. ``ComplexField -> |u|^2`` is an observable of the
terminal simulated field, not a cross-representation handoff, so it is not a
coupler. The registry is where this project states what a coupler is, and the
retired entry -- ``framework: jax``, ``derivative.mode: native_autodiff``, no
numerics -- made that statement unfalsifiable. ``test_graph_validation`` covers the
graph-level half; the checks here cover the source tree and the invariants the
entry declared.

The second is the measurement semantics M3.8 consumes. Those tests are written
against the failure they exist to prevent rather than against the happy path: an
implicitly normalized PSF reaching an oracle, and a PSF whose axes came from the
input pupil pitch instead of the propagated field's output pitch. Both produce a
completely plausible intensity map, which is why neither can be left to review.
"""

from __future__ import annotations

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
    ReferencePlane,
)
from core.specs import ArtifactKind, Device, Framework
from verification.psf_measurement import (
    COHERENCE_MODEL,
    M3_ORACLE_NORMALIZATION,
    NORMALIZATION_RATIONALE,
    PSF_INVARIANTS,
    PsfNormalization,
    measure_psf,
    measure_psf_from_record,
)

pytestmark = [pytest.mark.optiland, pytest.mark.chromatix]

ROOT = Path(__file__).resolve().parents[1]

# Deliberately non-square, and deliberately not centred. A square pitch makes an
# axis transpose invisible, and a peak at the origin makes the origin rule
# invisible; M2 lost two negative controls to exactly that kind of symmetry.
PITCH_YX_M = (1.0e-6, 3.0e-6)
PEAK_INDEX = (5, 6)
GRID = (9, 11)


def _field(
    *,
    u: np.ndarray | None = None,
    pitch: tuple[float, float] = PITCH_YX_M,
) -> ComplexField:
    if u is None:
        u = np.zeros(GRID, dtype=np.complex128)
        u[PEAK_INDEX] = 2.0 + 1.0j  # |u|^2 = 5
        u[0, 0] = 0.5  # a border sample, so the window indicator is not trivially 0
        u[2, 3] = 1.0 - 1.0j
    return ComplexField(
        u=u,
        sample_pitch_m=pitch,
        wavelength_m=5.5e-7,
        reference_plane=ReferencePlane(name="focus", z_m=4.837461300309598e-3),
    )


# ---------------------------------------------------------------------------
# AC1: the coupler abstraction is retired, and cannot drift back
# ---------------------------------------------------------------------------


def test_no_field_to_psf_coupler_is_implemented_or_registered() -> None:
    """No implementation under src/, and no entry in the registry file.

    The graph-level check lives in ``test_graph_validation``. This is the other
    half: a module could define a ``FieldToPSFCoupler`` and be wired up by hand
    without the registry ever mentioning it, and the M3 graph would then terminate
    in a coupler again with nothing asserting otherwise.

    Prose is deliberately allowed. Explaining *why* the id was retired is how the
    decision survives; the retired name appearing in a comment is not the failure
    mode, a class implementing it is.
    """
    implementations: list[str] = []
    for path in sorted((ROOT / "src").rglob("*.py")):
        text = path.read_text()
        for needle in ("FieldToPSF", "FieldToPsf"):
            if needle in text:
                implementations.append(f"{path.relative_to(ROOT)}: {needle}")
    assert not implementations, (
        "PSF extraction is a measurement, not a coupler; these would reintroduce "
        f"it as one: {implementations}"
    )

    registry = (ROOT / "src" / "registry" / "couplers.yaml").read_text()
    declarations = [
        line for line in registry.splitlines() if line.strip() == "- id: C_FIELD_TO_PSF"
    ]
    assert not declarations, declarations


def test_the_retired_registry_invariants_are_still_enforced_somewhere() -> None:
    """The entry went away; its two invariants did not.

    ``nonnegative_intensity`` and ``declared_psf_normalization`` were declared by
    C_FIELD_TO_PSF and enforced by nothing. They are kept under the same names so
    M3.10's claim audit can trace them from the removed registry entry to the
    contract that now executes them -- which is the opposite of dropping a claim
    to make an audit pass.
    """
    assert PSF_INVARIANTS == ("nonnegative_intensity", "declared_psf_normalization")

    # nonnegative_intensity
    with pytest.raises(ContractError) as negative:
        PSF(
            intensity=np.array([[1.0, -1e-30]]),
            sample_pitch_m=PITCH_YX_M,
            wavelength_m=5.5e-7,
            normalization="raw",
        )
    assert negative.value.code is ContractCode.NEGATIVE_INTENSITY

    # declared_psf_normalization
    with pytest.raises(ContractError) as undeclared:
        PSF(
            intensity=np.array([[1.0]]),
            sample_pitch_m=PITCH_YX_M,
            wavelength_m=5.5e-7,
            normalization="",
        )
    assert undeclared.value.code is ContractCode.MISSING_DECLARATION


# ---------------------------------------------------------------------------
# AC2: a measurement over the existing primitive, with no new physics
# ---------------------------------------------------------------------------


def test_raw_intensity_is_exactly_u_squared_bitwise() -> None:
    """No scaling, no resampling, no propagation. ``|u|^2`` and nothing else."""
    field = _field()
    measurement = measure_psf(field, normalization=PsfNormalization.RAW)

    assert np.array_equal(measurement.intensity, np.abs(field.u) ** 2)
    assert measurement.scale_factor == 1.0
    assert measurement.intensity.shape == field.shape
    assert measurement.sample_pitch_m == field.sample_pitch_m


def test_a_global_phase_does_not_change_the_measurement() -> None:
    """Why a carrier-removed field is admissible here.

    CHE-40's propagation path removes an absolute ``exp(i k z)`` and records that
    the field's absolute phase is no longer physical, and the slice protocol
    requires that path for a phase-insensitive PSF. This is the executable form of
    "phase-insensitive": the measurement cannot see a global phase.
    """
    field = _field()
    rotated = _field(u=field.u * np.exp(1j * 0.7913))

    a = measure_psf(field, normalization=M3_ORACLE_NORMALIZATION)
    b = measure_psf(rotated, normalization=M3_ORACLE_NORMALIZATION)

    assert np.allclose(a.intensity, b.intensity, rtol=1e-12, atol=0.0)
    assert a.peak_index == b.peak_index
    assert a.provenance["absolute_phase_required"] is False


# ---------------------------------------------------------------------------
# AC3: normalization is explicit, recorded, and honest about what it hides
# ---------------------------------------------------------------------------


def test_normalization_cannot_be_omitted() -> None:
    """There is no default, because the default would be the silent failure."""
    with pytest.raises(TypeError):
        measure_psf(_field())  # type: ignore[call-arg]


@pytest.mark.parametrize("normalization", list(PsfNormalization))
def test_every_normalization_records_its_own_declaration(normalization: PsfNormalization) -> None:
    """The artifact says which scaling it carries, in words, not by convention."""
    measurement = measure_psf(_field(), normalization=normalization)

    assert measurement.psf.normalization.startswith(f"{normalization}:")
    assert measurement.as_dict()["normalization"] == str(normalization)
    assert measurement.as_dict()["intensity_definition"] == "|u|^2"


def test_peak_normalization_puts_the_peak_at_exactly_one() -> None:
    measurement = measure_psf(_field(), normalization=PsfNormalization.PEAK)

    assert float(measurement.intensity.max()) == pytest.approx(1.0, rel=1e-15)
    assert measurement.raw_peak_intensity == pytest.approx(5.0, rel=1e-15)
    assert measurement.scale_factor == pytest.approx(1.0 / 5.0, rel=1e-15)


def test_energy_normalization_integrates_to_one_over_the_window() -> None:
    measurement = measure_psf(_field(), normalization=PsfNormalization.ENERGY)
    dy, dx = measurement.sample_pitch_m

    integral = float(measurement.intensity.sum()) * dy * dx
    assert integral == pytest.approx(1.0, rel=1e-12)


def test_normalization_scales_but_does_not_reshape() -> None:
    """A normalization that changed the profile would invalidate every oracle."""
    raw = measure_psf(_field(), normalization=PsfNormalization.RAW).intensity
    peak = measure_psf(_field(), normalization=PsfNormalization.PEAK).intensity

    assert np.allclose(peak * float(raw.max()), raw, rtol=1e-15, atol=0.0)


def test_peak_normalization_hides_a_constant_scale_and_the_raw_record_catches_it() -> None:
    """The declared blind spot, demonstrated rather than asserted in prose.

    M2's amplitude-weight omission was an exact constant scale, and this is the
    normalization that cannot see one. So the raw scale is not allowed to
    disappear: the measurement records it, and M3.8's energy ledger -- not the
    normalized profile -- is what constrains the multiplicative factor.
    """
    field = _field()
    scaled = _field(u=field.u * np.sqrt(1000.0))  # intensity x1000 everywhere

    a = measure_psf(field, normalization=PsfNormalization.PEAK)
    b = measure_psf(scaled, normalization=PsfNormalization.PEAK)

    # Indistinguishable after normalization -- that is the blind spot.
    assert np.allclose(a.intensity, b.intensity, rtol=1e-12, atol=0.0)

    # And fully visible in the retained raw scale -- that is the mitigation.
    assert b.raw_peak_intensity / a.raw_peak_intensity == pytest.approx(1000.0, rel=1e-12)
    assert b.raw_window_energy / a.raw_window_energy == pytest.approx(1000.0, rel=1e-12)
    assert "constant multiplicative error" in NORMALIZATION_RATIONALE


def test_the_frozen_oracle_normalization_is_peak() -> None:
    """M3.8's comparison scale, frozen here with its reasoning in the module."""
    assert M3_ORACLE_NORMALIZATION is PsfNormalization.PEAK
    assert "uncalibrated" in NORMALIZATION_RATIONALE


# ---------------------------------------------------------------------------
# AC3 (cont.): the axes come from the propagated field's OUTPUT pitch
# ---------------------------------------------------------------------------


def test_axes_and_peak_position_come_from_the_field_pitch_per_axis() -> None:
    """Non-square pitch, off-centre peak: a transpose or a re-centring must show.

    ``PSF.from_complex_field`` reads ``field.sample_pitch_m``, so this is the
    assertion that the measurement inherits the field's scale per axis rather than
    a single number, and that the pinned origin rule (index ``n//2`` is zero) is
    the one used to turn an index into a coordinate.
    """
    measurement = measure_psf(_field(), normalization=M3_ORACLE_NORMALIZATION)
    ny, nx = GRID
    dy, dx = PITCH_YX_M

    assert measurement.peak_index == PEAK_INDEX
    assert measurement.peak_position_m == (
        (PEAK_INDEX[0] - ny // 2) * dy,
        (PEAK_INDEX[1] - nx // 2) * dx,
    )
    # Transposing the pitch must move the reported position, or this test proves
    # nothing about which axis is which.
    transposed = measure_psf(_field(pitch=(dx, dy)), normalization=M3_ORACLE_NORMALIZATION)
    assert transposed.peak_position_m != measurement.peak_position_m

    y, x = measurement.coordinates()
    assert y[ny // 2] == 0.0 and x[nx // 2] == 0.0
    assert y[1] - y[0] == pytest.approx(dy, rel=1e-15)
    assert x[1] - x[0] == pytest.approx(dx, rel=1e-15)


def test_a_pitch_that_is_not_the_propagations_output_pitch_is_refused(tmp_path: Path) -> None:
    """The specific silent failure this guard exists for.

    The Chromatix adapter carries both pitches: its graph path writes ``dx_out``
    onto the output artifact, and its baseline summary's
    ``field_metadata.sample_pitch_m`` is the *input* pitch. A measurement that took
    the input pupil pitch would rescale every distance M3.8 reports -- peak
    position, first-null radius -- by a constant, and the intensity map would look
    exactly as expected. ASM preserves the pitch, so in the shipping slice the two
    agree; that makes this check cheap, not vacuous.
    """
    record = _field().to_artifact_record(artifact_id="wave:output_field", uri=tmp_path / "f.npy")
    input_pupil_pitch = PITCH_YX_M
    propagation_reported = (input_pupil_pitch[0] * 2.0, input_pupil_pitch[1] * 2.0)

    with pytest.raises(ContractError) as refused:
        measure_psf_from_record(
            record,
            normalization=M3_ORACLE_NORMALIZATION,
            expected_output_sample_pitch_m=propagation_reported,
        )

    assert refused.value.code is ContractCode.SAMPLE_PITCH_MISMATCH
    assert "OUTPUT pitch" in str(refused.value)


def test_the_matching_output_pitch_is_accepted_and_recorded(tmp_path: Path) -> None:
    record = _field().to_artifact_record(artifact_id="wave:output_field", uri=tmp_path / "f.npy")

    measurement = measure_psf_from_record(
        record,
        normalization=M3_ORACLE_NORMALIZATION,
        expected_output_sample_pitch_m=PITCH_YX_M,
    )

    assert measurement.sample_pitch_m == PITCH_YX_M
    assert measurement.provenance["output_pitch_checked_against_propagation"] is True
    assert measurement.as_dict()["pitch_source"] == "propagated field output pitch (dx_out)"


def test_the_measurement_refuses_an_artifact_that_is_not_a_field(tmp_path: Path) -> None:
    """It consumes the terminal state of the graph, not an arbitrary artifact."""
    path = tmp_path / "rays.npy"
    np.save(path, np.zeros((4, 3)))
    record = ArtifactRecord(
        id="lens:rays",
        kind=ArtifactKind.RAY_BUNDLE,
        uri=str(path),
        shape=(4, 3),
        dtype="float64",
        framework=Framework.NUMPY,
        device=Device.CPU,
        metadata={},
    )

    with pytest.raises(ContractError) as refused:
        measure_psf_from_record(record, normalization=M3_ORACLE_NORMALIZATION)
    assert refused.value.code is ContractCode.ARTIFACT_KIND_MISMATCH


# ---------------------------------------------------------------------------
# AC3 (cont.): the coherence model is stated, not inherited
# ---------------------------------------------------------------------------


def test_the_coherence_model_is_stated_by_the_measurement(tmp_path: Path) -> None:
    """Monochromatic and fully coherent, on the artifact, in every normalization."""
    for normalization in PsfNormalization:
        measurement = measure_psf(_field(), normalization=normalization)
        assert measurement.psf.coherence_model == COHERENCE_MODEL
        assert "monochromatic" in measurement.psf.coherence_model
        assert "fully coherent" in measurement.psf.coherence_model

    record = measure_psf(_field(), normalization=M3_ORACLE_NORMALIZATION).to_artifact_record(
        artifact_id="measurement:psf", uri=tmp_path / "psf.npy"
    )
    assert record.metadata["coherence_model"] == COHERENCE_MODEL
    assert record.metadata["measurement"]["is_a_graph_edge"] is False
    assert record.metadata["measurement"]["raw_peak_intensity"] == pytest.approx(5.0)


def test_from_complex_field_keeps_its_previous_default() -> None:
    """The new keyword is additive: omitting it changes nothing.

    CHE-36 added ``coherence_model`` to ``PSF.from_complex_field``. M2's evidence
    binds on the old behaviour, so the default must be untouched.
    """
    psf = PSF.from_complex_field(_field(), normalization="raw")
    assert psf.coherence_model == "fully coherent"


# ---------------------------------------------------------------------------
# AC4: the invariants are executable, and the refusal is the contract's
# ---------------------------------------------------------------------------


def test_a_non_finite_intensity_is_refused() -> None:
    with pytest.raises(ContractError) as refused:
        PSF(
            intensity=np.array([[1.0, np.nan]]),
            sample_pitch_m=PITCH_YX_M,
            wavelength_m=5.5e-7,
            normalization="raw",
        )
    assert refused.value.code is ContractCode.NON_FINITE


def test_a_dark_field_is_measurable_raw_but_not_normalizable() -> None:
    """Refusing beats returning NaN, and beats returning zeros.

    Normalizing a dark field would divide by zero and be caught one layer later as
    a non-finite intensity, which names the symptom instead of the cause. The raw
    measurement is still well defined, so it is still allowed.
    """
    dark = _field(u=np.zeros(GRID, dtype=np.complex128))

    raw = measure_psf(dark, normalization=PsfNormalization.RAW)
    assert float(raw.intensity.max()) == 0.0
    assert raw.raw_window_energy == 0.0

    for normalization in (PsfNormalization.PEAK, PsfNormalization.ENERGY):
        with pytest.raises(ContractError) as refused:
            measure_psf(dark, normalization=normalization)
        assert refused.value.code is ContractCode.EMPTY_ENSEMBLE
        assert "no energy" in str(refused.value)


def test_an_unpinned_origin_rule_is_refused() -> None:
    """Peak position is reported in metres, so the centring cannot be assumed."""
    field = ComplexField(
        u=np.ones(GRID, dtype=np.complex128),
        sample_pitch_m=PITCH_YX_M,
        wavelength_m=5.5e-7,
        reference_plane=ReferencePlane(name="focus", z_m=1e-3),
        frame=Frame(origin_rule="index 0 is coordinate zero"),
    )

    with pytest.raises(ContractError) as refused:
        measure_psf(field, normalization=M3_ORACLE_NORMALIZATION)
    assert refused.value.code is ContractCode.FRAME_MISMATCH


def test_the_window_indicator_reports_border_energy() -> None:
    """A finite-window indicator for M3.8's ledger, not a correctness claim."""
    on_border = measure_psf(_field(), normalization=PsfNormalization.RAW)
    assert on_border.border_energy_fraction > 0.0

    interior = np.zeros(GRID, dtype=np.complex128)
    interior[PEAK_INDEX] = 1.0
    assert (
        measure_psf(_field(u=interior), normalization=PsfNormalization.RAW).border_energy_fraction
        == 0.0
    )


# ---------------------------------------------------------------------------
# AC3 (cont.): the protocol freeze and the implementation must not drift apart
# ---------------------------------------------------------------------------


def test_the_protocol_freeze_matches_the_implementation() -> None:
    """``slice_protocol.yaml`` is what M3.8 reads; the module is what runs.

    Freezing semantics in a document that the code does not have to agree with is
    how a later ticket makes its oracle pass by adjusting the measurement. The
    frozen normalization, the coherence string and the invariant names are
    compared against the module, not restated.
    """
    import yaml

    protocol = yaml.safe_load((ROOT / "benchmarks" / "protocols" / "slice_protocol.yaml").read_text())
    frozen = protocol["psf_measurement"]

    assert frozen["is_a_coupler"] is False
    assert frozen["category"] == "observable"
    assert frozen["intensity"] == "abs(u)**2"
    assert frozen["normalization"]["frozen_for_m3_oracles"] == str(M3_ORACLE_NORMALIZATION)
    assert frozen["normalization"]["required"] is True
    assert frozen["coherence_model"] == COHERENCE_MODEL
    assert frozen["absolute_phase_required"] is False
    assert tuple(entry["name"] for entry in frozen["invariants"])[:2] == PSF_INVARIANTS
    assert (ROOT / frozen["implementation"]).is_file()

    # The graph the protocol declares must stop at the wave node: the measurement
    # is not a fourth step.
    assert protocol["scope"]["graph"] == [
        "M_RAY_OPTILAND",
        "C_RAY_TO_WAVE",
        "M_WAVE_CHROMATIX",
    ]
    assert protocol["scope"]["terminal_state"] == "complex_field"
    assert protocol["scope"]["terminal_measurement"]["is_a_graph_edge"] is False


def test_the_protocol_records_the_normalization_blind_spot() -> None:
    """A freeze that only states the choice would hide the cost of the choice."""
    import yaml

    protocol = yaml.safe_load((ROOT / "benchmarks" / "protocols" / "slice_protocol.yaml").read_text())
    normalization = protocol["psf_measurement"]["normalization"]

    assert "constant multiplicative error" in normalization["what_it_hides"]
    assert "energy ledger" in normalization["mitigation"]


# ---------------------------------------------------------------------------
# AC2 (cont.): the same measurement on a real propagated field
# ---------------------------------------------------------------------------

pytest.importorskip("optiland")
pytest.importorskip("chromatix")

from couplers.base import CouplerRunRequest  # noqa: E402
from couplers.node import RayToWaveCoupler  # noqa: E402
from solvers.base import ModelRunRequest, RunStatus  # noqa: E402
from solvers.chromatix.adapter import (  # noqa: E402
    get_adapter as get_wave_adapter,
)
from solvers.optiland.adapter import (  # noqa: E402
    get_adapter as get_ray_adapter,
)

# The frozen M3-SLICE-CPU-V1 geometry, as M3.6 used it.
PUPIL_Z_M = 0.06814345991561233e-3
IMAGE_Z_M = 4.90560476022521e-3
PITCH_M = 2.6587352810843895e-06
GRID_N = 188
PAD_WIDTH = 566


@pytest.fixture(scope="module")
def propagated_field(tmp_path_factory):
    """A real terminal field: Optiland trace -> C_RAY_TO_WAVE -> Chromatix ASM."""
    out = tmp_path_factory.mktemp("m37")
    rays = (
        get_ray_adapter()
        .run(
            ModelRunRequest(
                run_id="che36",
                node_id="lens",
                config={
                    "sample": "M3SingletRef",
                    "num_rays": 8,
                    "wavelength": 0.55,
                    "handoff_plane": "exit_pupil",
                    "output_directory": str(out / "rays"),
                },
            )
        )
        .outputs["rays"]
    )
    coupled = RayToWaveCoupler().transform(
        CouplerRunRequest(
            run_id="che36",
            edge_id="pupil",
            source=rays,
            config={
                "handoff_plane": "exit_pupil",
                "handoff_plane_z_m": PUPIL_Z_M,
                "grid_n": GRID_N,
                "target_sample_pitch_m": PITCH_M,
                "output_dir": str(out / "field"),
            },
        )
    )
    assert coupled.status is RunStatus.SUCCEEDED, coupled.error_message

    result = get_wave_adapter().run(
        ModelRunRequest(
            run_id="che36",
            node_id="wave",
            inputs={"input_field": coupled.target},
            config={
                "propagation": "angular_spectrum",
                "propagation_method": "asm_carrier_removed",
                "target_plane_z_m": IMAGE_Z_M,
                "pad_width": PAD_WIDTH,
                "output_dir": str(out / "wave"),
            },
        )
    )
    assert result.status is RunStatus.SUCCEEDED, result.error_message
    return result


def test_the_real_terminal_field_is_measured_at_the_pitch_the_propagation_reported(
    propagated_field,
) -> None:
    """The end-to-end form of the axes check, against the adapter's own report.

    The expected pitch is read from the propagation's own diagnostics, not from
    this file's constants and not from the artifact being measured: a test that
    read the pitch off the same metadata it is checking would agree with itself
    however wrong that metadata was.

    Those diagnostics are new (CHE-36). The graph path wrote ``dx_out`` onto the
    artifact and reported it nowhere else, so there had been nothing independent to
    compare against.
    """
    record = propagated_field.outputs["output_field"]
    reported = propagated_field.diagnostics["output_sample_pitch_m"]

    measurement = measure_psf_from_record(
        record,
        normalization=M3_ORACLE_NORMALIZATION,
        expected_output_sample_pitch_m=(float(reported[0]), float(reported[1])),
    )

    assert measurement.sample_pitch_m == (float(reported[0]), float(reported[1]))
    assert measurement.psf.coherence_model == COHERENCE_MODEL
    assert float(measurement.intensity.max()) == pytest.approx(1.0, rel=1e-12)
    assert np.all(measurement.intensity >= 0.0)
    assert np.all(np.isfinite(measurement.intensity))
    assert measurement.raw_peak_intensity > 0.0
    assert measurement.provenance["propagation_method"] == "asm_carrier_removed"
    # The carrier-removed path is admissible precisely because this measurement
    # does not read absolute phase.
    assert measurement.provenance["absolute_phase_is_physical"] is False


def test_the_real_measurement_concentrates_energy_on_axis(propagated_field) -> None:
    """A diffraction-limited system focuses. Not an oracle -- that is M3.8.

    Weak on purpose: it is here so that the measurement is known to be running on
    a field with a focus in it, rather than on something that happens to satisfy
    every contract check while being physically meaningless.
    """
    record = propagated_field.outputs["output_field"]
    measurement = measure_psf_from_record(record, normalization=M3_ORACLE_NORMALIZATION)

    ny, nx = measurement.intensity.shape
    iy, ix = measurement.peak_index
    assert abs(iy - ny // 2) <= 2 and abs(ix - nx // 2) <= 2, (
        f"peak at {(iy, ix)} is not near the on-axis centre {(ny // 2, nx // 2)}"
    )
