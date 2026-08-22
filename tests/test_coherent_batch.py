"""The CoherentRayBatch contract: identity, validity, residency (CHE-70, Phase 1).

Three failure modes this type exists to make impossible, each tested for:

* an amplitude paired with the wrong ray after a trace (``ray_id`` mismatch);
* a duplicated id, which would let two amplitudes claim one ray with no error;
* a batch straddling two array ecosystems, where the first operation touching
  both would silently unify them.

The residency split is tested as a *contract*, not an implementation detail:
``valid`` must live with the geometry because it multiplies device data, and
``ray_id`` must be host NumPy because it comes from the seeded host sampler. Both
directions are checked, so neither rule can be relaxed by accident.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from couplers.coherent_batch import (
    AMPLITUDE_SIDECAR_RULE,
    MICROMETRES_PER_METRE,
    MILLIMETRES_PER_METRE,
    OPTILAND_INTENSITY_RULE,
    CoherentRayBatch,
    declared_launch_opl_reference,
)
from couplers.contracts import (
    ContractError,
    Frame,
    RayBundle,
    ReferencePlane,
)

pytestmark = [pytest.mark.coupler]

PLANE = ReferencePlane(name="launch", z_m=0.0)
SENSOR = ReferencePlane(name="sensor", z_m=5e-5)


def _bundle(count: int = 6) -> RayBundle:
    rng = np.random.default_rng(0)
    transverse = rng.uniform(-0.2, 0.2, (count, 2))
    axial = np.sqrt(1.0 - (transverse**2).sum(axis=1))
    return RayBundle(
        positions_m=np.column_stack(
            [rng.uniform(-1e-5, 1e-5, count), rng.uniform(-1e-5, 1e-5, count), np.zeros(count)]
        ),
        directions=np.column_stack([transverse, axial]),
        wavelength_m=500e-9,
        reference_plane=PLANE,
        frame=Frame(axis_order="flat per-ray arrays"),
        amplitude=rng.normal(size=count) + 1j * rng.normal(size=count),
        optical_path_length_m=np.zeros(count),
        optical_path_length_reference=declared_launch_opl_reference(PLANE),
        reconstruction_normalization="one_over_n",
    )


def _batch(count: int = 6) -> CoherentRayBatch:
    bundle = _bundle(count)
    return CoherentRayBatch(
        bundle=bundle,
        ray_id=np.arange(count, dtype=np.int64),
        valid=np.ones(count, dtype=bool),
    )


class TestConstruction:
    def test_a_well_formed_batch_reports_its_counts(self):
        batch = _batch(6)
        assert batch.count == 6
        assert batch.valid_count == 6
        assert batch.amplitude.shape == (6,)

    def test_a_wrong_length_id_array_is_refused(self):
        bundle = _bundle(6)
        with pytest.raises(ContractError) as raised:
            CoherentRayBatch(
                bundle=bundle, ray_id=np.arange(5), valid=np.ones(6, dtype=bool)
            )
        assert raised.value.code == "SHAPE_MISMATCH"

    def test_a_two_dimensional_mask_is_refused(self):
        bundle = _bundle(6)
        with pytest.raises(ContractError) as raised:
            CoherentRayBatch(
                bundle=bundle,
                ray_id=np.arange(6),
                valid=np.ones((6, 1), dtype=bool),
            )
        assert raised.value.code == "SHAPE_MISMATCH"

    def test_a_duplicated_id_is_refused(self):
        """An id that repeats is not an identity."""
        bundle = _bundle(4)
        with pytest.raises(ContractError) as raised:
            CoherentRayBatch(
                bundle=bundle,
                ray_id=np.array([0, 1, 1, 3]),
                valid=np.ones(4, dtype=bool),
            )
        assert raised.value.code == "MISSING_DECLARATION"
        assert "duplicates" in str(raised.value)

    def test_ids_need_not_be_contiguous_or_start_at_zero(self):
        """A chunk's ids are a slice of a global counter, not a local range."""
        bundle = _bundle(3)
        batch = CoherentRayBatch(
            bundle=bundle,
            ray_id=np.array([1_000_000, 1_000_001, 1_000_002]),
            valid=np.ones(3, dtype=bool),
        )
        assert batch.count == 3

    def test_a_partly_invalid_batch_counts_only_the_valid_rays(self):
        bundle = _bundle(8)
        valid = np.ones(8, dtype=bool)
        valid[::2] = False
        batch = CoherentRayBatch(
            bundle=bundle, ray_id=np.arange(8), valid=valid
        )
        assert batch.count == 8
        assert batch.valid_count == 4


class TestResidencyRules:
    def test_the_residency_report_separates_physics_from_bookkeeping(self):
        report = _batch().residency()
        assert set(report) == {"scientific", "bookkeeping"}
        assert set(report["scientific"]) == {
            "positions_m", "directions", "amplitude", "optical_path_length_m"
        }
        assert set(report["bookkeeping"]) == {"ray_id", "valid"}
        assert report["bookkeeping"]["ray_id"]["dtype"] == "int64"
        assert report["bookkeeping"]["valid"]["dtype"] == "bool"

    def test_the_residency_is_read_off_the_arrays(self):
        report = _batch().residency()
        assert report["scientific"]["positions_m"]["dtype"] == "float64"
        assert report["scientific"]["amplitude"]["dtype"] == "complex128"
        assert report["scientific"]["positions_m"]["device"] == "cpu"

    def test_a_device_mask_with_host_geometry_is_refused(self):
        """The mask multiplies the amplitude, so a mismatch is a hidden transfer."""
        jax = pytest.importorskip("jax")
        import jax.numpy as jnp

        del jax
        bundle = _bundle(4)
        with pytest.raises(ContractError) as raised:
            CoherentRayBatch(
                bundle=bundle,
                ray_id=np.arange(4),
                valid=jnp.ones(4, dtype=bool),
            )
        assert raised.value.code == "REPRESENTATION_INCONSISTENT"
        assert "valid" in str(raised.value)

    def test_a_device_id_array_is_refused_because_ids_are_host_by_design(self):
        jax = pytest.importorskip("jax")
        import jax.numpy as jnp

        del jax
        bundle = _bundle(4)
        with pytest.raises(ContractError) as raised:
            CoherentRayBatch(
                bundle=bundle,
                ray_id=jnp.arange(4),
                valid=np.ones(4, dtype=bool),
            )
        assert raised.value.code == "REPRESENTATION_INCONSISTENT"
        assert "host" in str(raised.value)


class TestTracedStateHandoff:
    def test_the_amplitude_is_carried_over_not_recomputed(self):
        batch = _batch(5)
        traced = batch.with_traced_state(
            positions_m=np.zeros((5, 3)),
            directions=np.tile(np.array([0.0, 0.0, 1.0]), (5, 1)),
            optical_path_length_m=np.full(5, 5e-5),
            valid=np.ones(5, dtype=bool),
            plane=SENSOR,
            ray_id=batch.ray_id,
        )
        assert np.array_equal(
            np.asarray(traced.bundle.amplitude), np.asarray(batch.bundle.amplitude)
        )
        assert traced.bundle.reference_plane is SENSOR
        assert traced.bundle.provenance["amplitude_handling"] == AMPLITUDE_SIDECAR_RULE
        assert (
            traced.bundle.provenance["optiland_intensity_handling"]
            == OPTILAND_INTENSITY_RULE
        )

    def test_a_masked_amplitude_may_be_supplied_but_a_different_one_is_still_the_batchs(self):
        """Masking is permitted; substituting a re-derived amplitude is not the API."""
        batch = _batch(4)
        masked = np.asarray(batch.bundle.amplitude).copy()
        masked[1] = 0.0
        traced = batch.with_traced_state(
            positions_m=np.zeros((4, 3)),
            directions=np.tile(np.array([0.0, 0.0, 1.0]), (4, 1)),
            optical_path_length_m=np.zeros(4),
            valid=np.array([True, False, True, True]),
            plane=SENSOR,
            ray_id=batch.ray_id,
            amplitude=masked,
        )
        assert np.asarray(traced.bundle.amplitude)[1] == 0.0
        assert traced.valid_count == 3

    def test_reordered_ids_are_refused_rather_than_paired_by_position(self):
        """The single failure mode the type exists for."""
        batch = _batch(5)
        shuffled = np.asarray(batch.ray_id)[::-1].copy()
        with pytest.raises(ContractError) as raised:
            batch.with_traced_state(
                positions_m=np.zeros((5, 3)),
                directions=np.tile(np.array([0.0, 0.0, 1.0]), (5, 1)),
                optical_path_length_m=np.zeros(5),
                valid=np.ones(5, dtype=bool),
                plane=SENSOR,
                ray_id=shuffled,
            )
        assert raised.value.code == "SHAPE_MISMATCH"
        assert "identity" in str(raised.value)

    def test_a_dropped_ray_is_refused_rather_than_silently_shortening_the_batch(self):
        batch = _batch(5)
        with pytest.raises(ContractError):
            batch.with_traced_state(
                positions_m=np.zeros((4, 3)),
                directions=np.tile(np.array([0.0, 0.0, 1.0]), (4, 1)),
                optical_path_length_m=np.zeros(4),
                valid=np.ones(4, dtype=bool),
                plane=SENSOR,
                ray_id=np.asarray(batch.ray_id)[:4],
            )

    def test_the_traced_bundle_declares_the_launch_plane_as_its_opl_reference(self):
        batch = _batch(3)
        traced = batch.with_traced_state(
            positions_m=np.zeros((3, 3)),
            directions=np.tile(np.array([0.0, 0.0, 1.0]), (3, 1)),
            optical_path_length_m=np.zeros(3),
            valid=np.ones(3, dtype=bool),
            plane=SENSOR,
            ray_id=batch.ray_id,
        )
        reference = traced.bundle.optical_path_length_reference
        assert "launch" in reference
        assert "index-weighted geometric path" in reference
        # The OPL is admissible physics on this path, unlike opd_native.
        amplitude, opl = traced.bundle.require_coherent()
        assert amplitude.shape == opl.shape == (3,)


class TestUnitConstants:
    def test_the_constants_are_the_solver_units_the_schema_declares(self):
        from core.optical_system import UNITS

        assert UNITS["thickness"] == "mm"
        assert UNITS["wavelength"] == "um"
        assert MILLIMETRES_PER_METRE == 1e3
        assert MICROMETRES_PER_METRE == 1e6

    def test_the_opl_reference_names_the_plane_and_its_position(self):
        text = declared_launch_opl_reference(ReferencePlane(name="metalens_exit", z_m=0.0))
        assert "metalens_exit" in text
        assert "opd = 0" in text
        assert "millimetres" in text

    def test_the_reference_string_distinguishes_this_path_from_opd_native(self):
        """The pupil path's refusal is not being reopened; this is a different path."""
        text = declared_launch_opl_reference(PLANE)
        assert "constructing RealRays there" in text


class TestFrozen:
    def test_the_batch_is_immutable(self):
        batch = _batch()
        with pytest.raises(dataclasses.FrozenInstanceError):
            batch.ray_id = np.arange(6)  # type: ignore[misc]
