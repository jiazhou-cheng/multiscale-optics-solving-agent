"""The CHE-70 bridge on a real CUDA device (Phase 21, requirements 11 and 18).

    ./run.sh --gpu pytest -q -m gpu

Needs a dedicated session, like every other ``gpu`` test: enabling the GPU means
JAX computing on the GPU for the whole process, which changes
what every other test in the session computes on.

What "runs on the GPU" is allowed to mean here
----------------------------------------------
A kernel ran, and the arrays a real computation produced are on the device.
Never that a config value said ``cuda``. So every assertion below reads dtype and
device **off an array** that the pipeline actually produced, at each of the
boundaries Phase 21 names.

Requirement 18, and how it is actually checked
----------------------------------------------
"No complete large ray bundle is transferred to CPU" is checked two ways, because
neither alone is convincing:

*Structurally* -- ``core.arrays.to_host_numpy`` is the declared serialization
boundary, so it is instrumented and every array that crosses it must be at most
the 100x100 grid. A ray-sized transfer through the front door fails.

*Behaviourally* -- a candidate with a ray population two orders of magnitude
larger than any chunk must not grow the process's RSS by anything like the size
of that population. This catches a transfer that went around the front door.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("optiland")

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.optiland,
    pytest.mark.coupler,
    pytest.mark.integration,
    pytest.mark.torch,
    pytest.mark.jax,
]


@pytest.fixture(scope="module")
def cuda_available() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("FAIL_ENVIRONMENT_NO_CUDA: no CUDA device attached to this container")


@pytest.fixture(scope="module")
def small_run(cuda_available):
    """One real CUDA candidate, shared by the residency assertions."""
    from multiscale_optics_agent.benchmarks.metalens_candidate import (
        CandidateRequest,
        run_candidate,
    )

    record = run_candidate(
        CandidateRequest(
            config="METALENS-AIR-100",
            launch_count=8,
            samples_per_launch=512,
            seed=1,
            density="p_mag",
            chunk_size=1024,
            device="cuda",
            precision="fp32",
            label="gpu_residency",
        )
    )
    assert record["status"] == "PASS_RUN", record.get("failure")
    return record


class TestDeviceResidency:
    """Requirement 11 / Phase 21, boundary by boundary."""

    def test_every_scientific_array_is_on_the_device_at_every_boundary(self, small_run):
        residency = small_run["residency"]
        assert set(residency) >= {"wave_to_ray_output", "optiland_output", "sensor_field"}
        for boundary in ("wave_to_ray_output", "optiland_output"):
            arrays = residency[boundary]["scientific"]
            assert set(arrays) == {
                "positions_m", "directions", "amplitude", "optical_path_length_m"
            }
            for name, state in arrays.items():
                assert state["device"].startswith("cuda"), f"{boundary}.{name}"
                assert state["namespace"] == "jax", f"{boundary}.{name}"
        accumulator = residency["sensor_field"]["sensor_accumulator"]
        assert accumulator["device"].startswith("cuda")
        assert accumulator["dtype"] == "complex64"

    def test_the_input_field_lands_on_the_device_at_the_requested_precision(self, small_run):
        cast = small_run["input_field_residency"]
        assert cast["requested"] == "complex64"
        assert cast["actual"] == "complex64", (
            "requested and actual must be separately observable; JAX drops a 64-bit "
            "request in silence with x64 disabled"
        )
        assert cast["device"].startswith("cuda")

    def test_the_ray_geometry_is_float32_and_the_amplitude_complex64(self, small_run):
        arrays = small_run["residency"]["optiland_output"]["scientific"]
        assert arrays["positions_m"]["dtype"] == "float32"
        assert arrays["directions"]["dtype"] == "float32"
        assert arrays["optical_path_length_m"]["dtype"] == "float32"
        assert arrays["amplitude"]["dtype"] == "complex64"

    def test_the_ray_ids_are_the_one_deliberate_host_array(self, small_run):
        """Declared, not incidental: the seeded sampler is what pins reproducibility."""
        book = small_run["residency"]["wave_to_ray_output"]["bookkeeping"]
        assert book["ray_id"]["device"] == "cpu"
        assert book["ray_id"]["namespace"] == "numpy"
        assert book["valid"]["device"].startswith("cuda")

    def test_optiland_reports_the_torch_backend_on_cuda_in_float32(self, small_run):
        execution = small_run["optiland_execution"]
        assert execution["requested"]["backend"] == "torch"
        assert execution["observed"]["device"] == "cuda"
        assert execution["observed"]["precision"] == "float32"
        assert execution["cuda_device_name"]
        assert "cu12" in execution["torch_version"]

    def test_autograd_is_off_for_the_production_path(self, small_run):
        """Phase 13, read back from the solver rather than assumed."""
        assert small_run["optiland_execution"]["observed"]["grad_enabled"] is False

    def test_the_bridge_is_a_namespace_change_and_nothing_else(self, small_run):
        """DLPack on CUDA: no host transfer, no device transfer, no dtype change."""
        for direction in ("into_optiland", "out_of_optiland"):
            plan = small_run["bridge_plans"][direction]
            assert plan["namespace_conversion"] is True
            assert plan["host_transfer"] is False, direction
            assert plan["device_transfer"] is False, direction
            assert plan["dtype_conversion"] is False, direction
            assert plan["lossy"] is False, direction
        assert small_run["bridge_plans"]["into_optiland"]["target"]["namespace"] == "torch"
        assert small_run["bridge_plans"]["out_of_optiland"]["target"]["namespace"] == "jax"

    def test_jax_is_actually_on_the_gpu_and_not_pinned_to_the_host(self, small_run):
        """A platform pin produces exactly a successful run on the wrong device."""
        environment = small_run["environment"]
        assert environment["jax_backend"] == "gpu"
        assert any(device.startswith("cuda") for device in environment["jax_devices"])
        assert environment["jax_enable_x64"] is False

    def test_no_ray_was_clipped_or_lost(self, small_run):
        assert small_run["first_chunk_trace"]["invalid_rays"] == 0
        assert small_run["streaming"]["valid_rays"] == small_run["streaming"]["total_rays"]


class TestPhysicsOnTheDevice:
    def test_the_gpu_result_meets_the_gate_against_the_analytic_oracle(self, small_run):
        assert small_run["metrics"]["ncc"] > 0.99

    def test_float32_on_the_gpu_agrees_with_float64_on_the_host(self, cuda_available):
        """The precision claim, measured rather than asserted from the dtype label.

        Same seed, same population, same estimator; only the representation and
        the device differ. A disagreement here would mean the GPU path's answer is
        set by its arithmetic rather than by its sampling -- which is exactly the
        TF32 hazard ``matmul_precision_kwargs`` exists to prevent.
        """
        from multiscale_optics_agent.benchmarks.metalens_candidate import (
            CandidateRequest,
            run_candidate,
        )

        common = {
            "config": "METALENS-AIR-100",
            "launch_count": 8,
            "samples_per_launch": 512,
            "seed": 1,
            "density": "p_mag",
            "chunk_size": 2048,
        }
        gpu = run_candidate(
            CandidateRequest(**common, device="cuda", precision="fp32", label="gpu")
        )
        host = run_candidate(
            CandidateRequest(**common, device="cpu", precision="fp64", label="host")
        )
        assert gpu["status"] == "PASS_RUN", gpu.get("failure")
        assert host["status"] == "PASS_RUN", host.get("failure")
        assert gpu["metrics"]["ncc"] == pytest.approx(host["metrics"]["ncc"], abs=1e-6)
        assert gpu["metrics"]["relative_power_error"] == pytest.approx(
            host["metrics"]["relative_power_error"], rel=1e-4
        )

    def test_the_slab_configuration_also_meets_the_gate_on_the_device(self, cuda_available):
        """Real refraction at two interfaces, and an index-weighted OPL."""
        from multiscale_optics_agent.benchmarks.metalens_candidate import (
            CandidateRequest,
            run_candidate,
        )

        record = run_candidate(
            CandidateRequest(
                config="METALENS-SLAB-100",
                launch_count=32,
                samples_per_launch=4096,
                seed=1,
                density="p_mag",
                chunk_size=65536,
                device="cuda",
                precision="fp32",
                label="gpu_slab",
            )
        )
        assert record["status"] == "PASS_RUN", record.get("failure")
        assert record["metrics"]["ncc"] > 0.99
        assert len(record["configuration"]["layers"]) == 2


class TestNoHostRayTransfer:
    """Requirement 18, structurally and behaviourally."""

    def test_the_only_declared_host_transfers_are_the_final_grid_sized_arrays(
        self, cuda_available, monkeypatch
    ):
        import multiscale_optics_agent.core.arrays as arrays_module
        from multiscale_optics_agent.benchmarks.metalens_candidate import (
            CandidateRequest,
            run_candidate,
        )

        transferred: list[tuple[int, str]] = []
        original = arrays_module.to_host_numpy

        def spy(value, *, reason):
            result = original(value, reason=reason)
            transferred.append((int(np.asarray(result).size), reason))
            return result

        monkeypatch.setattr(arrays_module, "to_host_numpy", spy)
        record = run_candidate(
            CandidateRequest(
                config="METALENS-AIR-100",
                launch_count=64,
                samples_per_launch=8192,
                seed=1,
                density="p_mag",
                chunk_size=8192,
                device="cuda",
                precision="fp32",
                label="no_host_transfer",
            )
        )
        assert record["status"] == "PASS_RUN", record.get("failure")
        assert transferred, "the final field must cross the declared boundary"
        for size, reason in transferred:
            assert size <= 100 * 100, (
                f"a {size}-element array crossed the serialization boundary for "
                f"{reason!r}; the ray population is {record['request']['total_rays']}"
            )
            assert "final" in reason

    def test_the_process_rss_does_not_grow_with_the_ray_population(self, cuda_available):
        """A transfer that went around the front door would show up here.

        524288 rays at the ~40 bytes/ray a float32 batch needs on the wave side is
        ~21 MB per full copy, and the reconstruction's per-ray ramps are 100x that.
        Holding the population would therefore be plainly visible against a chunk
        of 8192.
        """
        from multiscale_optics_agent.benchmarks.metalens_candidate import (
            CandidateRequest,
            run_candidate,
        )

        small = run_candidate(
            CandidateRequest(
                config="METALENS-AIR-100",
                launch_count=4,
                samples_per_launch=8192,
                seed=1,
                density="p_mag",
                chunk_size=8192,
                device="cuda",
                precision="fp32",
                label="rss_small",
            )
        )
        large = run_candidate(
            CandidateRequest(
                config="METALENS-AIR-100",
                launch_count=256,
                samples_per_launch=8192,
                seed=1,
                density="p_mag",
                chunk_size=8192,
                device="cuda",
                precision="fp32",
                label="rss_large",
            )
        )
        assert small["status"] == "PASS_RUN" and large["status"] == "PASS_RUN"
        assert large["request"]["total_rays"] == 64 * small["request"]["total_rays"]
        growth = (
            large["memory"]["peak_rss_bytes"] - small["memory"]["peak_rss_bytes"]
        )
        # A generous bound: the 64x population increase must not cost anything
        # like 64x the memory. Set well above allocator noise and well below any
        # per-ray host residency.
        assert growth < 512 * 1024**2, (
            f"peak RSS grew by {growth} B for a 64x larger ray population; "
            "something is retaining per-ray host state"
        )
        assert small["memory"]["swap_delta_peak_bytes"] == 0
        assert large["memory"]["swap_delta_peak_bytes"] == 0

    def test_the_gpu_peak_is_set_by_the_chunk_and_not_by_the_total(self, cuda_available):
        """Phase 32's claim, as a test rather than only as a plot."""
        from multiscale_optics_agent.benchmarks.metalens_candidate import (
            CandidateRequest,
            run_candidate,
        )

        peaks = {}
        for launches in (8, 128):
            record = run_candidate(
                CandidateRequest(
                    config="METALENS-AIR-100",
                    launch_count=launches,
                    samples_per_launch=8192,
                    seed=1,
                    density="p_mag",
                    chunk_size=8192,
                    device="cuda",
                    precision="fp32",
                    label=f"peak_{launches}",
                )
            )
            assert record["status"] == "PASS_RUN", record.get("failure")
            peaks[launches] = record["gpu_memory_after"]["peak_allocated_bytes"]
            assert record["gpu_peak_growth_bytes"] == 0, (
                "the GPU peak grew across chunks; a chunk is being retained"
            )
        assert peaks[128] < 2 * peaks[8], (
            f"a 16x larger population raised the GPU peak from {peaks[8]} to "
            f"{peaks[128]} B; peak memory must follow the chunk, not the total"
        )
