"""CHE-23 — typed boundary artifacts for the bidirectional coupler.

The point of these types is not convenience. It is that a missing declaration
becomes an error instead of a default, and that the two quantities M1 recorded
as unverified -- Optiland's OPD sign/reference, and its ray `intensity` -- can
be carried without being reinterpreted as a phase or an amplitude.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from multiscale_optics_agent.core.artifacts import ArtifactRecord
from multiscale_optics_agent.core.specs import ArtifactKind
from multiscale_optics_agent.couplers import (
    AXIS_ORDER,
    ORIGIN_RULE,
    PHASOR,
    PSF,
    ComplexField,
    ContractCode,
    ContractError,
    CouplerRunRequest,
    Frame,
    RayBundle,
    ReferencePlane,
    WavefrontSamples,
)

pytestmark = pytest.mark.coupler

WAVELENGTH_M = 632.8e-9
PLANE = ReferencePlane(name="test plane", z_m=0.0)


def _directions(n: int, *, theta: float = 0.0) -> np.ndarray:
    d = np.zeros((n, 3), dtype=np.float64)
    d[:, 0] = math.sin(theta)
    d[:, 2] = math.cos(theta)
    return d


def _ray_bundle(n: int = 4, **overrides) -> RayBundle:
    kwargs = {
        "positions_m": np.zeros((n, 3), dtype=np.float64),
        "directions": _directions(n),
        "wavelength_m": WAVELENGTH_M,
        "reference_plane": PLANE,
        "frame": Frame(axis_order="flat per-ray arrays"),
    }
    kwargs.update(overrides)
    return RayBundle(**kwargs)


def _complex_field(ny: int = 8, nx: int = 8, **overrides) -> ComplexField:
    kwargs = {
        "u": np.ones((ny, nx), dtype=np.complex128),
        "sample_pitch_m": (1e-6, 1e-6),
        "wavelength_m": WAVELENGTH_M,
        "reference_plane": PLANE,
    }
    kwargs.update(overrides)
    return ComplexField(**kwargs)


# --- The two unverified quantities M1 handed forward -------------------------


def test_ray_bundle_carries_a_weight_but_refuses_to_call_it_an_amplitude() -> None:
    """Optiland emits `intensity` with an explicit intensity_is_not_amplitude
    marker. Carrying it is fine; reading it as `a` in eq 2 is not."""
    bundle = _ray_bundle(
        weight=np.array([1.0, 4.0, 9.0, 16.0]),
        weight_semantics="RealRays.i is a per-ray intensity, not a complex amplitude",
    )

    assert bundle.weight is not None
    assert bundle.amplitude is None

    with pytest.raises(ContractError) as excinfo:
        bundle.require_coherent()
    assert excinfo.value.code is ContractCode.AMPLITUDE_IS_A_WEIGHT
    assert "not a complex amplitude" in str(excinfo.value)


def test_weight_to_amplitude_conversion_must_be_declared_by_the_caller() -> None:
    bundle = _ray_bundle(
        weight=np.array([1.0, 4.0, 9.0, 16.0]),
        weight_semantics="per-ray power",
    )
    converted = bundle.with_amplitude_from_weight(
        mapping="amplitude = sqrt(weight); weight is a power"
    ).with_declared_optical_path_length(np.zeros(4), reference="entrance pupil")

    amplitude, _ = converted.require_coherent()
    np.testing.assert_allclose(amplitude.real, [1.0, 2.0, 3.0, 4.0])
    # The assumption that produced the field is recorded, not implied.
    assert "sqrt(weight)" in converted.provenance["amplitude_mapping"]

    # An undeclared mapping is refused rather than guessed.
    with pytest.raises(ContractError) as excinfo:
        bundle.with_amplitude_from_weight(mapping="something clever")
    assert excinfo.value.code is ContractCode.MISSING_DECLARATION


def test_unverified_opl_reference_is_refused_not_defaulted() -> None:
    """A wrong OPL *reference* is a harmless piston. A wrong OPL *sign*
    conjugates the wavefront. Those are indistinguishable downstream, so an
    'unverified' marker may not pass through."""
    bundle = _ray_bundle(
        amplitude=np.ones(4, dtype=np.complex128),
        optical_path_length_m=np.zeros(4),
        optical_path_length_reference="unverified",
    )
    with pytest.raises(ContractError) as excinfo:
        bundle.require_coherent()
    assert excinfo.value.code is ContractCode.OPL_REFERENCE_UNVERIFIED

    with pytest.raises(ContractError):
        bundle.with_declared_optical_path_length(np.zeros(4), reference="unverified")


def test_an_opl_without_a_reference_cannot_be_constructed_at_all() -> None:
    with pytest.raises(ContractError) as excinfo:
        _ray_bundle(optical_path_length_m=np.zeros(4))
    assert excinfo.value.code is ContractCode.MISSING_DECLARATION
    assert excinfo.value.declaration == "optical_path_length_reference"


def test_a_weight_without_semantics_cannot_be_constructed_at_all() -> None:
    with pytest.raises(ContractError) as excinfo:
        _ray_bundle(weight=np.ones(4))
    assert excinfo.value.declaration == "weight_semantics"


# --- Convention negative tests -----------------------------------------------


def test_phasor_sign_flip_is_rejected_on_every_contract() -> None:
    wrong = "exp(+i omega t)"
    for build in (
        lambda: _ray_bundle(phasor=wrong),
        lambda: _complex_field(phasor=wrong),
    ):
        with pytest.raises(ContractError) as excinfo:
            build()
        assert excinfo.value.code is ContractCode.PHASOR_MISMATCH


def test_axis_transpose_is_rejected_for_field_arrays() -> None:
    """A transpose is invisible in any rotationally symmetric test case, so it
    has to be checked at the boundary rather than discovered numerically."""
    with pytest.raises(ContractError) as excinfo:
        _complex_field(frame=Frame(axis_order="(x, y)"))
    assert excinfo.value.code is ContractCode.AXIS_ORDER_MISMATCH


def test_left_handed_frame_and_minus_z_propagation_are_rejected() -> None:
    with pytest.raises(ContractError) as excinfo:
        Frame(handedness="left-handed")
    assert excinfo.value.code is ContractCode.FRAME_MISMATCH

    with pytest.raises(ContractError):
        Frame(propagation_axis="-z")


def test_millimetre_for_metre_is_caught_as_a_non_si_wavelength() -> None:
    with pytest.raises(ContractError) as excinfo:
        _ray_bundle(wavelength_m=-1.0)
    assert excinfo.value.code is ContractCode.UNIT_NOT_SI


def test_non_unit_directions_are_rejected() -> None:
    directions = _directions(4)
    directions[2] *= 1.05
    with pytest.raises(ContractError) as excinfo:
        _ray_bundle(directions=directions)
    assert excinfo.value.code is ContractCode.NON_UNIT_DIRECTION


def test_empty_ensemble_is_a_named_failure_not_an_empty_field() -> None:
    with pytest.raises(ContractError) as excinfo:
        RayBundle(
            positions_m=np.zeros((0, 3)),
            directions=np.zeros((0, 3)),
            wavelength_m=WAVELENGTH_M,
            reference_plane=PLANE,
            frame=Frame(axis_order="flat per-ray arrays"),
        )
    assert excinfo.value.code is ContractCode.EMPTY_ENSEMBLE


def test_a_real_array_is_not_a_complex_field() -> None:
    with pytest.raises(ContractError) as excinfo:
        _complex_field(u=np.ones((4, 4), dtype=np.float64))
    assert excinfo.value.code is ContractCode.MISSING_DECLARATION
    assert "intensity" in str(excinfo.value)


def test_negative_psf_intensity_is_rejected() -> None:
    with pytest.raises(ContractError) as excinfo:
        PSF(
            intensity=np.array([[-1.0, 0.0], [0.0, 1.0]]),
            sample_pitch_m=(1e-6, 1e-6),
            wavelength_m=WAVELENGTH_M,
            normalization="peak = 1",
        )
    assert excinfo.value.code is ContractCode.NEGATIVE_INTENSITY


def test_psf_requires_a_declared_normalization() -> None:
    with pytest.raises(ContractError) as excinfo:
        PSF(
            intensity=np.ones((2, 2)),
            sample_pitch_m=(1e-6, 1e-6),
            wavelength_m=WAVELENGTH_M,
            normalization="",
        )
    assert excinfo.value.declaration == "normalization"


# --- Grid centring and power -------------------------------------------------


def test_origin_is_index_n_over_2_on_both_even_and_odd_grids() -> None:
    """M1 pinned index n//2 as coordinate zero. Implemented in exactly one
    place so a coupler cannot quietly adopt a different centring."""
    even = _complex_field(ny=8, nx=8, sample_pitch_m=(2e-6, 1e-6))
    y, x = even.coordinates()
    assert y[8 // 2] == 0.0
    assert x[8 // 2] == 0.0
    np.testing.assert_allclose(np.diff(y), 2e-6)
    np.testing.assert_allclose(np.diff(x), 1e-6)

    odd = _complex_field(ny=7, nx=5)
    y_odd, x_odd = odd.coordinates()
    assert y_odd[7 // 2] == 0.0
    assert x_odd[5 // 2] == 0.0


def test_discrete_power_uses_amplitude_squared_times_pixel_area() -> None:
    field_ = _complex_field(ny=4, nx=4, sample_pitch_m=(2e-6, 3e-6))
    assert field_.discrete_power() == pytest.approx(16 * 1.0 * 2e-6 * 3e-6)


def test_psf_from_field_is_the_squared_modulus() -> None:
    field_ = _complex_field(ny=3, nx=3, u=np.full((3, 3), 2.0 + 0.0j))
    psf = PSF.from_complex_field(field_, normalization="raw |u|^2")
    np.testing.assert_allclose(psf.intensity, 4.0)


# --- Round trip with the artifacts the adapters already emit -----------------


def _optiland_ray_record(tmp_path, **metadata_overrides) -> tuple[ArtifactRecord, dict]:
    """Reproduces the metadata shape emitted by optiland_adapter._build_ray_bundle_artifact."""
    arrays = {
        "x_m": np.array([0.0, 1e-3]),
        "y_m": np.array([0.0, 0.0]),
        "z_m": np.array([0.1, 0.1]),
        "L": np.array([0.0, 0.0]),
        "M": np.array([0.0, 0.0]),
        "N": np.array([1.0, 1.0]),
        "intensity": np.array([1.0, 0.5]),
        "opd_native": np.array([12.0, 12.1]),
    }
    metadata = {
        "length_unit": "m",
        "native_length_unit": "mm",
        "wavelength_unit": "m",
        "wavelength_m": WAVELENGTH_M,
        "intensity_is_not_amplitude": (
            "RealRays.i is a real-valued per-ray intensity, not a complex amplitude"
        ),
        "backend": "numpy",
        "conventions": {
            "axes": "x,y,z right-handed Cartesian; propagation is +z",
            "handedness": "right-handed",
            "reference_plane": "final traced image surface, surface index 14",
            "reference_plane_z_m": 0.1,
            "opd_field": "opd_native",
            "opd_reference": "unverified",
            "opd_sign": "unverified",
            "polarization": "missing; RealRays provides no polarization state",
            "coherence": "missing; sequential rays are not a coherent complex field",
            "normalization": "raw Optiland ray intensity/weight; not normalized",
        },
    }
    metadata.update(metadata_overrides)
    record = ArtifactRecord(
        id="node-rays-abc123",
        kind=ArtifactKind.RAY_BUNDLE,
        uri=str(tmp_path / "rays.npz"),
        metadata=metadata,
    )
    return record, arrays


def test_ray_bundle_builds_from_an_unmodified_optiland_artifact(tmp_path) -> None:
    record, arrays = _optiland_ray_record(tmp_path)
    bundle = RayBundle.from_artifact_record(record, arrays=arrays)

    assert bundle.count == 2
    assert bundle.wavelength_m == WAVELENGTH_M
    assert bundle.reference_plane.z_m == 0.1
    assert bundle.polarization.startswith("missing")

    # opd_native rides along for traceability, marked unverified, and is NOT
    # promoted into optical_path_length_m.
    assert bundle.optical_path_length_m is None
    assert bundle.provenance["opd_native_status"]["sign"] == "unverified"
    np.testing.assert_allclose(bundle.provenance["opd_native"], [12.0, 12.1])

    with pytest.raises(ContractError) as excinfo:
        bundle.require_coherent()
    assert excinfo.value.code is ContractCode.AMPLITUDE_IS_A_WEIGHT


def test_ray_bundle_rejects_a_non_si_optiland_artifact(tmp_path) -> None:
    record, arrays = _optiland_ray_record(tmp_path, length_unit="mm")
    with pytest.raises(ContractError) as excinfo:
        RayBundle.from_artifact_record(record, arrays=arrays)
    assert excinfo.value.code is ContractCode.UNIT_NOT_SI


def test_ray_bundle_names_the_missing_declaration(tmp_path) -> None:
    record, arrays = _optiland_ray_record(tmp_path)
    del record.metadata["wavelength_m"]
    with pytest.raises(ContractError) as excinfo:
        RayBundle.from_artifact_record(record, arrays=arrays)
    assert excinfo.value.code is ContractCode.MISSING_DECLARATION
    assert excinfo.value.declaration == "wavelength_m"


def test_ray_bundle_round_trips_through_an_artifact_record(tmp_path) -> None:
    original = _ray_bundle(
        positions_m=np.array([[0.0, 0.0, 0.0], [1e-3, 2e-3, 0.0]]),
        directions=_directions(2, theta=0.1),
        amplitude=np.array([1.0 + 0.0j, 0.0 + 2.0j]),
        optical_path_length_m=np.array([0.0, 1e-6]),
        optical_path_length_reference="entrance pupil",
    )
    record = original.to_artifact_record(artifact_id="rt", uri=tmp_path / "rt.npz")
    assert record.kind is ArtifactKind.RAY_BUNDLE
    assert record.metadata["phasor"] == PHASOR
    assert record.metadata["optical_path_length_reference"] == "entrance pupil"

    arrays = dict(np.load(record.uri))
    np.testing.assert_allclose(arrays["amplitude"], original.amplitude)
    np.testing.assert_allclose(arrays["opl_m"], original.optical_path_length_m)


def _chromatix_field_record(tmp_path, **metadata_overrides) -> tuple[ArtifactRecord, np.ndarray]:
    """Reproduces the metadata shape emitted by chromatix_adapter._run_asm_propagate."""
    u = np.ones((8, 8), dtype=np.complex64)
    metadata = {
        "wavelength": WAVELENGTH_M,
        "sample_pitch": (1e-6, 1e-6),
        "coordinate_frame": "axes=(y, x) row-major; right-handed Cartesian; +z is propagation",
        "phasor": PHASOR,
        "polarization": "scalar (chromatix ScalarField; no polarization state tracked)",
        "normalization": "u stores complex field amplitude, not intensity",
        "propagation_method": "asm_propagate",
        "z_m": 1e-3,
        "pad_width": 4,
        "padded": True,
        "input_shape": (4, 4),
    }
    metadata.update(metadata_overrides)
    record = ArtifactRecord(
        id="node:output_field",
        kind=ArtifactKind.COMPLEX_FIELD,
        uri=str(tmp_path / "output_field.npy"),
        metadata=metadata,
    )
    return record, u


def test_complex_field_builds_from_an_unmodified_chromatix_artifact(tmp_path) -> None:
    record, u = _chromatix_field_record(tmp_path)
    field_ = ComplexField.from_artifact_record(record, array=u)

    assert field_.shape == (8, 8)
    assert field_.sample_pitch_m == (1e-6, 1e-6)
    assert field_.padded is True
    assert field_.pad_width == 4
    # complex64 from the engine is widened for the reference core; the source
    # dtype stays visible through provenance rather than being lost.
    assert field_.u.dtype == np.complex128


def test_complex_field_refuses_a_field_with_an_undeclared_pad_state(tmp_path) -> None:
    """M1 measured a 256x256 input growing to 1756x1756. An array shape alone
    does not determine physical extent."""
    record, u = _chromatix_field_record(tmp_path)
    del record.metadata["pad_width"]
    with pytest.raises(ContractError) as excinfo:
        ComplexField.from_artifact_record(record, array=u)
    assert excinfo.value.code is ContractCode.PAD_STATE_UNKNOWN


def test_complex_field_refuses_an_undeclared_normalization(tmp_path) -> None:
    record, u = _chromatix_field_record(tmp_path)
    del record.metadata["normalization"]
    with pytest.raises(ContractError) as excinfo:
        ComplexField.from_artifact_record(record, array=u)
    assert excinfo.value.declaration == "normalization"


def test_complex_field_round_trips_and_declares_its_conventions(tmp_path) -> None:
    original = _complex_field(ny=4, nx=6, sample_pitch_m=(2e-6, 3e-6))
    record = original.to_artifact_record(artifact_id="f", uri=tmp_path / "f.npy")

    assert record.metadata["phasor"] == PHASOR
    assert record.metadata["origin"] == ORIGIN_RULE
    assert AXIS_ORDER in record.metadata["coordinate_frame"]
    assert record.metadata["sample_pitch"] == [2e-6, 3e-6]

    restored = ComplexField.from_artifact_record(record)
    np.testing.assert_array_equal(restored.u, original.u)
    assert restored.sample_pitch_m == original.sample_pitch_m


def test_wavefront_samples_reject_the_unverified_optiland_opd(tmp_path) -> None:
    """The Optiland wavefront artifact's only OPL source is RealRays.opd, whose
    convention the adapter itself documents as unverified. Failing here is the
    contract working."""
    record = ArtifactRecord(
        id="node-wavefront-1",
        kind=ArtifactKind.WAVEFRONT_SAMPLES,
        uri=str(tmp_path / "wavefront.npz"),
        metadata={
            "length_unit": "m",
            "wavelength_unit": "m",
            "wavelength": WAVELENGTH_M,
            "coordinate_fields": ["x_m", "y_m"],
            "optical_path_length_source": "RealRays.opd -- convention not independently verified",
            "missing_declared_metadata": ["amplitude", "polarization", "pupil_mask"],
        },
    )
    with pytest.raises(ContractError) as excinfo:
        WavefrontSamples.from_artifact_record(
            record, arrays={"x_m": np.zeros(2), "y_m": np.zeros(2), "opl_m": np.zeros(2)}
        )
    assert excinfo.value.code is ContractCode.OPL_REFERENCE_UNVERIFIED


def test_wavefront_samples_accept_a_declared_reference() -> None:
    samples = WavefrontSamples(
        positions_m=np.array([[0.0, 0.0], [1e-3, 0.0]]),
        optical_path_length_m=np.array([0.0, 1e-7]),
        optical_path_length_reference="exit pupil reference sphere",
        wavelength_m=WAVELENGTH_M,
        reference_plane=PLANE,
    )
    assert samples.count == 2


# --- Request contract widening ------------------------------------------------


def test_single_source_requests_still_work_and_are_mirrored_into_ports(tmp_path) -> None:
    record, _ = _chromatix_field_record(tmp_path)
    request = CouplerRunRequest(run_id="r", edge_id="e", source=record)

    assert request.require_source() is record
    assert request.sources["source"] is record


def test_named_source_ports_are_supported(tmp_path) -> None:
    field_record, _ = _chromatix_field_record(tmp_path)
    ray_record, _ = _optiland_ray_record(tmp_path)
    request = CouplerRunRequest(
        run_id="r",
        edge_id="e",
        sources={"incident_rays": ray_record, "doe_transmission": field_record},
    )

    assert request.require_source("incident_rays") is ray_record
    with pytest.raises(KeyError) as excinfo:
        request.require_source("missing_port")
    assert "doe_transmission" in str(excinfo.value)


def test_a_request_with_no_source_at_all_is_rejected() -> None:
    with pytest.raises(ValueError, match="must supply"):
        CouplerRunRequest(run_id="r", edge_id="e")
