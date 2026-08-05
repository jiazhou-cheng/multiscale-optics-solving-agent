from multiscale_optics_agent.core.artifacts import ArtifactRecord
from multiscale_optics_agent.core.specs import ArtifactKind


def test_artifact_record_preserves_semantic_metadata() -> None:
    artifact = ArtifactRecord(
        id="field-1",
        kind=ArtifactKind.COMPLEX_FIELD,
        uri="runs/example/field.zarr",
        shape=(128, 128),
        dtype="complex64",
        metadata={
            "wavelength": 5.5e-7,
            "sample_pitch": 2.0e-6,
            "phasor": "exp(-i omega t)",
        },
    )
    assert artifact.metadata["phasor"] == "exp(-i omega t)"
