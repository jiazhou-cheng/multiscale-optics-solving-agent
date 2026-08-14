"""CHE-33 (M3.4): the declared handoff from a real Optiland trace.

Two declarations are made here and both are load-bearing:

* the **optical path length**, in SI, with CHE-30's sign and reference and
  CHE-32's plane, conditioned as CHE-40 requires;
* the **intensity-to-amplitude map**, which is a modelling decision rather than
  a ``sqrt``.

What makes this more than plumbing is that the first one is checked against an
oracle that owes nothing to this repository -- a diffraction-limited system
delivers equal optical path to its focus, so the pupil OPL must be
``R - sqrt(rho^2 + R^2)`` -- and against Optiland's own wavefront machinery,
which uses the opposite sign convention and a different reference construction.
Three routes, one number.

Every falsifier is exercised. A declaration that cannot be made to fail is not
evidence, and M2's carried-forward lesson is that a shared convention error
cancels in a round trip: the negative tests here therefore look at a *downstream
observable*, not at the declared array.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from conftest import load_coupler_probe_expected

from multiscale_optics_agent.adapters.base import ModelRunRequest
from multiscale_optics_agent.couplers import ContractCode, ContractError, RayBundle
from multiscale_optics_agent.couplers.optiland_handoff import (
    AMPLITUDE_MAPPING,
    DeclaredHandoffPlane,
    HandoffPerturbation,
    declare_coherent_bundle,
    reconstruct_hashed_arrays,
)
from multiscale_optics_agent.couplers.ray_to_wave import Projection, ray_to_wave

pytest.importorskip("optiland")

from multiscale_optics_agent.adapters.optiland_adapter import (
    _scientific_array_hash,
    get_adapter,
)

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = load_coupler_probe_expected("ray_to_wave", "coherent_handoff")

WAVELENGTH_UM = 0.55
WAVELENGTH_M = 5.5e-7
MM_PER_M = 1e-3

# Frozen by M3.2 in benchmarks/slice_protocol.yaml, restated here as the plane a
# consumer declares. Read as literals on purpose: the point of the check is that
# the consumer states the plane independently of the producer.
SINGLET_PUPIL_Z_M = 0.06814345991561233 * MM_PER_M
SINGLET_PITCH_M = 2.6587352810843895e-06

# Reconstruction grid for the downstream-observable tests. Smaller than the
# protocol's 188 because these tests are about whether an observable MOVES, not
# about its accuracy; the per-axis Nyquist limit (lambda / 2 pitch = 0.1034) still
# comfortably clears this system's 0.0517 marginal direction cosine.
GRID = (64, 64)


def _trace(tmp_path_factory, sample: str, num_rays: int, plane: str = "exit_pupil"):
    out = tmp_path_factory.mktemp(f"{sample}-{num_rays}-{plane}")
    result = get_adapter().run(
        ModelRunRequest(
            run_id="che33",
            node_id=f"{sample}-{num_rays}-{plane}",
            config={
                "sample": sample,
                "num_rays": num_rays,
                "wavelength": WAVELENGTH_UM,
                "handoff_plane": plane,
                "output_directory": str(out),
            },
        )
    )
    assert result.status.value == "succeeded", (result.error_type, result.error_message)
    return result.outputs["rays"]


@pytest.fixture(scope="module")
def singlet_record(tmp_path_factory):
    """One 817-ray M3SingletRef trace at the declared exit pupil, reused throughout."""
    return _trace(tmp_path_factory, "M3SingletRef", 16)


@pytest.fixture(scope="module")
def small_record(tmp_path_factory):
    """A 217-ray trace, for the tests that run a field reconstruction."""
    return _trace(tmp_path_factory, "M3SingletRef", 8)


@pytest.fixture(scope="module")
def singlet_plane() -> DeclaredHandoffPlane:
    return DeclaredHandoffPlane(handoff_plane="exit_pupil", z_m=SINGLET_PUPIL_Z_M)


def _reconstruct(bundle, projection: Projection = Projection.ASM_CONSISTENT):
    return ray_to_wave(
        bundle,
        grid_shape=GRID,
        sample_pitch_m=(SINGLET_PITCH_M, SINGLET_PITCH_M),
        projection=projection,
    )


# --- The gate stays shut for anything undeclared -----------------------------


def test_a_real_trace_is_still_refused_until_the_convention_is_declared(singlet_record):
    """The adapter record on its own must not be coherent.

    This is the check the whole ticket is built around: `from_artifact_record`
    carries `opd_native` in provenance and refuses to promote it, so a caller who
    skips the declaration gets a named failure rather than a plausible field.
    """
    raw = RayBundle.from_artifact_record(singlet_record)
    assert raw.optical_path_length_m is None
    assert raw.provenance["opd_native"] is not None

    with pytest.raises(ContractError) as excinfo:
        raw.require_coherent()
    error = excinfo.value
    # Amplitude is refused first; that is the other half of the same gate.
    assert error.code is ContractCode.AMPLITUDE_IS_A_WEIGHT

    with pytest.raises(ContractError) as excinfo:
        raw.with_amplitude_from_weight(mapping=AMPLITUDE_MAPPING).require_coherent()
    error = excinfo.value
    assert error.code is ContractCode.OPL_REFERENCE_UNVERIFIED
    assert error.remedy and "with_declared_optical_path_length" in error.remedy
    assert "opd_native is not admissible" in error.remedy


def test_an_explicitly_unverified_reference_is_refused_by_name():
    """`with_declared_optical_path_length` must not accept the word 'unverified'."""
    positions = np.zeros((2, 3))
    directions = np.tile(np.array([0.0, 0.0, 1.0]), (2, 1))
    bundle = RayBundle(
        positions_m=positions,
        directions=directions,
        wavelength_m=WAVELENGTH_M,
        reference_plane=__import__(
            "multiscale_optics_agent.couplers", fromlist=["ReferencePlane"]
        ).ReferencePlane(name="p", z_m=0.0),
    )
    with pytest.raises(ContractError) as excinfo:
        bundle.with_declared_optical_path_length(np.zeros(2), reference="unverified")
    assert excinfo.value.code is ContractCode.OPL_REFERENCE_UNVERIFIED


# --- The declaration itself ---------------------------------------------------


def test_declared_bundle_is_coherent_and_carries_both_declarations(singlet_record, singlet_plane):
    handoff = declare_coherent_bundle(singlet_record, declared_plane=singlet_plane)
    amplitude, optical_path_m = handoff.bundle.require_coherent()

    assert handoff.bundle.count == 817
    assert amplitude.shape == optical_path_m.shape == (817,)
    # Optiland supplies no phase in `i`, so the amplitude must be real and
    # non-negative; any imaginary part would mean phase leaked in from the weight.
    assert np.all(amplitude.imag == 0.0)
    assert np.all(amplitude.real >= 0.0)
    assert handoff.bundle.provenance["amplitude_mapping"] == AMPLITUDE_MAPPING

    reference = handoff.bundle.optical_path_length_reference
    assert reference and reference != "unverified"
    assert "ray minus chief" in reference
    assert "EPD" in reference  # CHE-30: the zero is aperture-dependent

    # The removed piston is retained in float64 and never folded back.
    assert handoff.diagnostics["removed_reference_opl_waves"] == pytest.approx(
        EXPECTED["systems"]["M3SingletRef"]["removed_reference_opl_waves"], rel=1e-12
    )
    assert "retained_as_metadata_not_reapplied" in handoff.declarations["global_phase_policy"]


def test_the_declaration_moves_nothing_the_adapter_hashed(singlet_record, singlet_plane):
    """Round trip: adapter record in, typed bundle out, same scientific arrays.

    Hashed with the adapter's own function over arrays rebuilt from the *bundle*,
    not re-read from the file -- re-reading would only prove the file was not
    edited, which was never in doubt.
    """
    handoff = declare_coherent_bundle(singlet_record, declared_plane=singlet_plane)
    rebuilt = _scientific_array_hash(reconstruct_hashed_arrays(handoff.bundle))
    assert rebuilt == singlet_record.metadata["scientific_array_sha256"]
    assert rebuilt == EXPECTED["systems"]["M3SingletRef"]["scientific_array_sha256"]


def test_image_space_index_is_read_from_the_prescription_not_assumed(singlet_record, singlet_plane):
    assert singlet_record.metadata["conventions"]["image_space_refractive_index"] == 1.0
    stripped = singlet_record.model_copy(deep=True)
    stripped.metadata["conventions"].pop("image_space_refractive_index")
    with pytest.raises(ContractError) as excinfo:
        declare_coherent_bundle(stripped, declared_plane=singlet_plane)
    assert excinfo.value.code is ContractCode.MISSING_DECLARATION
    assert "image-space refractive index" in str(excinfo.value)


# --- Verified against two independent oracles ---------------------------------


@pytest.mark.parametrize("sample", ["M3SingletRef", "ReverseTelephoto"])
def test_declared_optical_path_is_a_converging_wavefront(tmp_path_factory, sample):
    """The analytic oracle: equal optical path to the focus.

    ``OPL(rho) - OPL(0) = R - sqrt(rho^2 + R^2)`` is exact for a perfect system,
    so the residual is the wavefront aberration and nothing else. On
    ``M3SingletRef`` it must land on M3.2's independently frozen 0.01700 waves.
    """
    expected = EXPECTED["systems"][sample]
    record = _trace(tmp_path_factory, sample, 16)
    plane = DeclaredHandoffPlane(handoff_plane="exit_pupil", z_m=expected["exit_pupil_z_m"])
    handoff = declare_coherent_bundle(record, declared_plane=plane)
    _, optical_path_m = handoff.bundle.require_coherent()

    radius_m = np.hypot(handoff.bundle.positions_m[:, 0], handoff.bundle.positions_m[:, 1])
    distance_m = expected["pupil_to_focus_distance"]["measured_from_record_m"]
    oracle_m = distance_m - np.hypot(radius_m, distance_m)
    residual_pv_waves = float(np.ptp((optical_path_m - oracle_m) / WAVELENGTH_M))

    assert residual_pv_waves == pytest.approx(
        expected["analytic_oracle"]["declared_residual_pv_waves"], rel=1e-9
    )
    # Optiland's own wavefront code, opposite sign convention, own reference
    # sphere. Agreement here is what turns "plausible" into "verified".
    assert residual_pv_waves == pytest.approx(
        expected["optiland_wavefront_cross_check"]["peak_to_valley_waves"], abs=2e-6
    )


def test_singlet_residual_reproduces_the_frozen_diffraction_limited_evidence():
    """M3.2 measured 0.01700 waves by a different route; this must be that number."""
    oracle = EXPECTED["systems"]["M3SingletRef"]["analytic_oracle"]
    assert oracle["declared_residual_pv_waves"] == pytest.approx(
        oracle["frozen_peak_to_valley_waves"], abs=5e-6
    )


# --- Falsifiers, on a downstream observable -----------------------------------


@pytest.mark.parametrize(
    "perturbation",
    [
        HandoffPerturbation(opl_sign=-1),
        HandoffPerturbation(transfer_opl_to_plane=False),
    ],
    ids=["opl_sign_flipped", "plane_transfer_omitted"],
)
def test_a_wrong_declaration_changes_the_reconstructed_field(
    small_record, singlet_plane, perturbation
):
    """Mandatory negative test: the error must reach an observable.

    M2's evidence is that a convention error shared by both sides of a round trip
    cancels, so checking the declared array against itself would prove nothing.
    The observable here is the reconstructed pupil intensity, one coupler call
    downstream. A flipped OPL sign is *not* a global conjugation of the field --
    the oblique ramp keeps its sign -- so the intensity genuinely moves.
    """
    correct, _ = _reconstruct(
        declare_coherent_bundle(small_record, declared_plane=singlet_plane).bundle
    )
    wrong, _ = _reconstruct(
        declare_coherent_bundle(
            small_record, declared_plane=singlet_plane, perturbation=perturbation
        ).bundle
    )

    correct_intensity = np.abs(correct.u) ** 2
    wrong_intensity = np.abs(wrong.u) ** 2
    relative_l2 = float(
        np.linalg.norm(wrong_intensity - correct_intensity) / np.linalg.norm(correct_intensity)
    )
    # An order-unity change, not a tolerance-scale one. The gates in
    # benchmarks/slice_protocol.yaml sit at 1e-3, so this is >100x any of them.
    assert relative_l2 > 0.5, relative_l2


def test_the_perturbation_default_is_the_correct_physics():
    """Guard against a defect defaulting on, which would silently ship it."""
    assert HandoffPerturbation().is_identity
    assert HandoffPerturbation().describe() == "none"
    assert HandoffPerturbation(opl_sign=-1).describe() == "opl_sign_flipped"


# --- Plane agreement ----------------------------------------------------------


def test_a_plane_of_the_wrong_kind_is_a_structured_refusal(singlet_record):
    with pytest.raises(ContractError) as excinfo:
        declare_coherent_bundle(
            singlet_record,
            declared_plane=DeclaredHandoffPlane(
                handoff_plane="image_surface", z_m=SINGLET_PUPIL_Z_M
            ),
        )
    error = excinfo.value
    assert error.code is ContractCode.REFERENCE_PLANE_MISMATCH
    assert error.remedy and "defocus" in error.remedy


def test_a_plane_at_the_wrong_place_is_a_structured_refusal(singlet_record, singlet_plane):
    """A micron of axial offset must be refused, not absorbed as a piston."""
    offset_m = 1e-6
    with pytest.raises(ContractError) as excinfo:
        declare_coherent_bundle(
            singlet_record,
            declared_plane=DeclaredHandoffPlane(
                handoff_plane="exit_pupil", z_m=singlet_plane.z_m + offset_m
            ),
        )
    error = excinfo.value
    assert error.code is ContractCode.REFERENCE_PLANE_MISMATCH
    assert error.remedy and "do not widen the tolerance" in error.remedy


def test_the_declared_plane_arrives_on_the_bundle(singlet_record, singlet_plane):
    handoff = declare_coherent_bundle(singlet_record, declared_plane=singlet_plane)
    assert handoff.bundle.reference_plane.z_m == pytest.approx(singlet_plane.z_m, abs=1e-15)
    assert "exit pupil" in handoff.bundle.reference_plane.name
    assert np.allclose(handoff.bundle.positions_m[:, 2], singlet_plane.z_m)


# --- Projection convention ----------------------------------------------------


def test_reconstructed_field_provenance_records_the_projection_convention(
    small_record, singlet_plane
):
    """M2's F1: only ASM_CONSISTENT preserves a field; the other is a sensor model.

    The choice has to be legible in the artifact, because the two differ by
    ``<n, d>`` and a few percent off-axis -- which looks like nothing until it is
    compared against an oracle.
    """
    bundle = declare_coherent_bundle(small_record, declared_plane=singlet_plane).bundle
    asm_field, asm_diag = _reconstruct(bundle, Projection.ASM_CONSISTENT)
    sensor_field, _ = _reconstruct(bundle, Projection.SENSOR_OBLIQUITY)

    assert asm_field.provenance["projection"] == "asm_consistent"
    assert "no obliquity factor" in asm_field.provenance["equation"]
    assert sensor_field.provenance["projection"] == "sensor_obliquity"
    assert "projection = asm_consistent" in asm_field.normalization

    # Not cosmetic: report the size of the difference this real trace actually
    # sees, and check it is consistent with the smallest cos(theta) in the bundle.
    difference = float(np.max(np.abs(sensor_field.u - asm_field.u)) / np.max(np.abs(asm_field.u)))
    assert difference > 0.0
    assert difference < 1.0 - asm_diag.min_projection_factor + 1e-9


# --- The recorded evidence stays in step --------------------------------------


def test_probe_fixture_records_the_protocol_geometry_disagreement():
    """CHE-33 found `M3-SINGLET-REF.propagation_distance_mm` to be the back focal
    length rather than the exit-pupil-to-image distance. The fixture must keep
    saying so until amendment A2's correction is what the protocol carries."""
    import yaml

    protocol = yaml.safe_load((ROOT / "benchmarks" / "slice_protocol.yaml").read_text())
    frozen = {system["id"]: system["derived"] for system in protocol["systems"]}
    for sample, system_id in (
        ("M3SingletRef", "M3-SINGLET-REF"),
        ("ReverseTelephoto", "M3-REVERSE-TELEPHOTO"),
    ):
        measured = EXPECTED["systems"][sample]["pupil_to_focus_distance"]["measured_from_record_m"]
        derived = frozen[system_id]
        assert measured == pytest.approx(
            (derived["image_plane_z_mm"] - derived["exit_pupil_z_mm"]) * MM_PER_M, rel=1e-12
        )
        assert derived["propagation_distance_mm"] * MM_PER_M == pytest.approx(measured, rel=1e-12)
