"""Shared scaffolding for the CHE-96 Fig 5 reproductions.

Two probes (`demo2_hologram.py`, `demo3_hologram_lens.py`) run the same four
things -- build a DOE from a vendored phase mask, emit patch secondary rays,
reconstruct them in chunks on whichever device is available, and score the
result -- so those live here rather than being written twice and drifting.

The scoring functions are the part worth reading. NCC and MSE are the paper's
metrics and neither is well defined until you say what was normalized, so both
say so in their own names and both are recorded alongside the raw values.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from core.boundary import ReferencePlane
from core.precision import ArrayNamespace, DType
from couplers.patch import patch_secondary_rays, plan_patches
from couplers.ray_to_wave import Projection, Reconstruction
from couplers.streaming import StreamingReconstruction

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "benchmarks" / "probes" / "data"
RECORDS = REPO / "benchmarks" / "probes" / "records" / "ray_wave"


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def ncc(a: np.ndarray, b: np.ndarray) -> float:
    """Zero-mean normalized cross-correlation of two real images.

    The paper's headline metric. Mean-subtracted (Pearson), which is the usual
    reading of "NCC" for image similarity and the only one that is insensitive
    to an additive pedestal. The un-centred variant is reported next to it,
    because the two differ noticeably on a speckle field with a bright DC lobe
    and a reader cannot tell from the number alone which was used.
    """
    x = np.asarray(a, dtype=np.float64).ravel()
    y = np.asarray(b, dtype=np.float64).ravel()
    x = x - x.mean()
    y = y - y.mean()
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    return float(x @ y / denominator) if denominator > 0.0 else float("nan")


def ncc_uncentred(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float64).ravel()
    y = np.asarray(b, dtype=np.float64).ravel()
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    return float(x @ y / denominator) if denominator > 0.0 else float("nan")


def mse_unit_sum(a: np.ndarray, b: np.ndarray) -> float:
    """MSE after normalizing each intensity image to unit total power.

    Declared rather than assumed: an MSE between two intensities is meaningless
    until their scales are tied together, and the patch route's absolute scale
    depends on the coverage correction and the ray budget. Unit *sum* rather
    than unit *max*, because a single hot speckle pixel would otherwise set the
    scale for the whole comparison.
    """
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    x = x / x.sum()
    y = y / y.sum()
    return float(np.mean((x - y) ** 2))


def relative_l2(a: np.ndarray, b: np.ndarray) -> float:
    """Complex field error, ``||a - b|| / ||b||``. No normalization applied."""
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)) / np.linalg.norm(np.asarray(b)))


# ---------------------------------------------------------------------------
# The DOE
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Doe:
    transmission: np.ndarray
    grid_shape: tuple[int, int]
    pitch_m: tuple[float, float]
    radius_m: float
    #: True when every sample the full-aperture odd patch does NOT cover is
    #: identically zero, so nothing is lost by being one sample short.
    #:
    #: Checked, not argued. `patch_px` must be odd and the paper's grids are
    #: even, so the widest legal patch is `n - 1`; centred at index `n // 2` it
    #: covers rows and columns `1 .. n-1`, dropping only index 0. With the
    #: origin at index `n // 2` that row sits at `y = -(n/2) * pitch`, exactly
    #: on the circular mask's radius, and the strict inequality excludes it.
    #: The asymmetry is real -- index `n-1` is at `+(n/2 - 1) * pitch`, inside
    #: the mask -- so it is index 0 and only index 0 that has to be empty.
    dropped_border_is_empty: bool


def build_doe(mask_name: str, *, pitch_m: float, flip: bool = False) -> Doe:
    """Circular-aperture phase DOE from a vendored mask.

    ``flip`` reproduces the notebook's ``flip(phase, dims=[0, 1])``. That flip
    compensates DeepLens's ``Ray.flip_xy`` and this pipeline has no such thing,
    so the default is **unflipped** -- but the probes score both orientations
    once rather than settling it by argument.
    """
    phase = np.load(DATA / mask_name).astype(np.float64)
    if flip:
        phase = phase[::-1, ::-1]
    ny, nx = phase.shape
    radius_m = (ny // 2) * pitch_m
    y = (np.arange(ny) - ny // 2) * pitch_m
    x = (np.arange(nx) - nx // 2) * pitch_m
    yy, xx = np.meshgrid(y, x, indexing="ij")
    aperture = (xx**2 + yy**2) < radius_m**2
    transmission = np.where(aperture, np.exp(1j * phase), 0.0).astype(np.complex128)
    dropped_border_is_empty = bool(
        np.all(transmission[0, :] == 0) and np.all(transmission[:, 0] == 0)
    )
    return Doe(
        transmission=transmission,
        grid_shape=(ny, nx),
        pitch_m=(pitch_m, pitch_m),
        radius_m=radius_m,
        dropped_border_is_empty=dropped_border_is_empty,
    )


# ---------------------------------------------------------------------------
# The route
# ---------------------------------------------------------------------------

class _Batch:
    """The two attributes `StreamingReconstruction.add_chunk` reads.

    `valid_count == count`: nothing in demo2 clips a secondary ray before the
    sensor. demo3's Optiland leg does clip, and reports its own survival count
    rather than reusing this.
    """

    __slots__ = ("bundle", "count", "valid_count")

    def __init__(self, bundle: Any, valid_count: int | None = None) -> None:
        self.bundle = bundle
        self.count = int(bundle.count)
        self.valid_count = int(valid_count if valid_count is not None else bundle.count)


class _DualAccumulator(StreamingReconstruction):
    """Coherent field **and** the literal-notebook intensity sum, in one pass.

    The notebook binds `huygens_psf`'s 4th return value -- `psf = |field|^2` --
    accumulates that across batches, and then squares the total again. SI eq S5
    and the Figure S2 text both require coherent *field* accumulation, so the
    correct version is what this repository computes; the literal variant is
    computed alongside so the difference is visible rather than argued.

    Overriding `add_chunk` rather than calling `ray_to_wave` twice, because at
    the paper's budgets a second reconstruction pass doubles the run. What is
    duplicated is three lines of accumulation; what is inherited untouched is
    `finalize`, which owns the single `1/N_total` and the check that the chunks
    carried exactly the ray count the estimator was normalized for. The
    dangerous arithmetic is the inherited part.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.notebook_intensity = np.zeros(self.grid_shape, dtype=np.float64)

    def add_chunk(self, batch: Any) -> dict[str, Any]:
        from couplers.ray_to_wave import ray_to_wave

        chunk_field, diagnostics = ray_to_wave(
            batch.bundle,
            grid_shape=self.grid_shape,
            sample_pitch_m=self.sample_pitch_m,
            plane=self.plane,
            normalization="none",
            projection=self.projection,
            reconstruction=self.reconstruction,
            kspace_oversample=self.kspace_oversample,
            kspace_grid_shape=self.kspace_grid_shape,
        )
        self._accumulator = self._accumulator + chunk_field.u
        self.notebook_intensity += np.abs(np.asarray(chunk_field.u)) ** 2
        self.valid_rays += batch.valid_count
        self.chunk_sizes.append(batch.count)
        record = diagnostics.as_dict()
        if self._first_diagnostics is None:
            self._first_diagnostics = record
        return record


def enable_x64_if_needed(*, backend: str, precisions: list[str]) -> dict[str, Any]:
    """Turn on JAX's 64-bit mode *before* the first array, or say fp64 is off.

    JAX silently truncates a requested float64 to float32 and emits a
    `UserWarning` -- which is exactly the failure mode this repository has a
    policy against, because the array then reports the dtype it got rather than
    the one that was asked for and every downstream record says fp32 without
    anyone deciding that. `jax_enable_x64` must be set before the first array is
    created, so it happens here rather than lazily.

    Returns what was actually done, for the record.
    """
    if backend != "jax":
        return {"x64_requested": False, "x64_enabled": False}
    if "fp64" not in precisions:
        return {"x64_requested": False, "x64_enabled": False}
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    enabled = jnp.zeros(1, dtype=jnp.float64).dtype == np.float64
    return {"x64_requested": True, "x64_enabled": bool(enabled)}


def to_namespace(bundle: Any, *, backend: str, precision: str) -> Any:
    """Move a bundle's arrays onto the compute device, keeping the contract.

    `ray_to_wave` takes its array module from the bundle, so this is the whole
    of "run it on the GPU" -- there is no device argument anywhere downstream.
    The dtype is set here too, and the pair is reported off the arrays
    afterwards rather than off this request.
    """
    import dataclasses

    if backend == "numpy":
        real = np.float64 if precision == "fp64" else np.float32
        complex_ = np.complex128 if precision == "fp64" else np.complex64
        convert = np.asarray
    else:
        import jax.numpy as jnp

        real = jnp.float64 if precision == "fp64" else jnp.float32
        complex_ = jnp.complex128 if precision == "fp64" else jnp.complex64
        convert = jnp.asarray

    moved = dataclasses.replace(
        bundle,
        positions_m=convert(np.asarray(bundle.positions_m), dtype=real),
        directions=convert(np.asarray(bundle.directions), dtype=real),
        amplitude=convert(np.asarray(bundle.amplitude), dtype=complex_),
        optical_path_length_m=convert(
            np.asarray(bundle.optical_path_length_m), dtype=real
        ),
    )
    # Read off the array, never off the request. JAX truncates a requested
    # float64 to float32 with a warning when x64 is off, and a probe that
    # trusted its own argument would record fp64 for a run that computed in
    # fp32.
    actual = str(moved.directions.dtype)
    expected = "float64" if precision == "fp64" else "float32"
    if actual != expected:
        raise RuntimeError(
            f"asked for {expected} and got {actual}. On JAX this means "
            "jax_enable_x64 was not set before the first array was created; "
            "call enable_x64_if_needed() at the top of the run. Refusing rather "
            "than recording a precision that did not execute."
        )
    return moved


def matched_kspace_grid(
    *,
    pad_px: int,
    doe_pitch_m: float,
    sensor_pitch_m: float,
    min_shape: tuple[int, int],
    max_grid: int = 8192,
) -> dict[str, Any]:
    """Smallest k-grid on which this patch's spectral bins land exactly on nodes.

    `patch.py` draws secondary directions from `fftshift(fftfreq(pad, d=pitch))`,
    so every ray's transverse wavevector is an exact integer multiple of
    ``dk_mode = 2 pi / (pad_px * doe_pitch)``. The reconstruction's k-grid has
    spacing ``dk_grid = 2 pi / (K * sensor_pitch)``. The splat is exact -- not
    interpolated -- exactly when ``dk_mode / dk_grid = K * sensor_pitch /
    (pad_px * doe_pitch)`` is an integer.

    So the k-grid is not a free tuning knob for a route whose rays came from an
    enumeration or from a draw over that enumeration: there is a *right* answer,
    and picking an oversampling factor instead converts an exactness measurement
    into an interpolation error. This is the CHE-96 oracle-padding lesson in a
    different coordinate.

    Note the DOE and sensor pitches need not match -- demo3 samples at 4.2 um a
    field whose modes were enumerated at 6.3 um -- which is why the ratio is
    carried explicitly rather than assumed to be one.

    Returns the grid and the residual, so a caller can record how exact
    "exact" was rather than trusting this function.
    """
    period = pad_px * doe_pitch_m
    ratio = period / sensor_pitch_m
    need = max(int(min_shape[0]), int(min_shape[1]))
    best: dict[str, Any] | None = None
    for multiple in range(1, max_grid):
        k = multiple * ratio
        rounded = round(k)
        if rounded > max_grid:
            break
        residual = abs(k - rounded) / max(k, 1.0)
        if residual > 1e-12 or rounded < need:
            continue
        best = {
            "kspace_grid_shape": [rounded, rounded],
            "spectral_periods_per_kgrid": multiple,
            "integrality_residual": residual,
            "basis": (
                f"pad_px={pad_px} at {doe_pitch_m:.3e} m reconstructed at "
                f"{sensor_pitch_m:.3e} m; K * sensor_pitch is {multiple} whole "
                "spectral periods, so every drawn bin lands on a node"
            ),
        }
        break
    if best is None:
        raise ValueError(
            f"no k-grid <= {max_grid} makes pad_px={pad_px} at {doe_pitch_m} m "
            f"commensurate with a {sensor_pitch_m} m sensor grid of at least {need}; "
            "the fast path would be interpolating and must be reported as such"
        )
    return best


def patch_route(
    doe: Doe,
    *,
    wavelength_m: float,
    sensor_z_m: float,
    sensor_shape: tuple[int, int],
    patch_px: int,
    pad_factor: int,
    patch_count: int | None,
    secondary_count: int | None,
    batches: int,
    seed: int,
    backend: str,
    precision: str,
    advance: bool = True,
    reconstruction: str = "ramp_sum",
    kspace_oversample: float | None = None,
    kspace_grid_shape: tuple[int, int] | str | None = None,
) -> dict[str, Any]:
    """One full patch-route run, chunked so the ray budget never lands at once.

    Chunking is over **patches**, not over rays, which matters for the
    normalization: `StreamingReconstruction` divides once by the total ray count
    at the end, so a chunk boundary cannot change the estimator. Splitting a
    single patch's secondary sample across chunks would be equally valid
    arithmetically, but splitting on patches keeps each chunk a whole number of
    independent draws, which is what makes a per-chunk diagnostic readable.
    """
    plane = ReferencePlane(name="doe", z_m=0.0)
    sensor = ReferencePlane(name="sensor", z_m=sensor_z_m)
    rng = np.random.default_rng(seed)

    plan = plan_patches(
        grid_shape=doe.grid_shape,
        sample_pitch_m=doe.pitch_m,
        patch_px=patch_px,
        pad_factor=pad_factor,
        patch_count=patch_count,
        rng=rng,
    )
    centers = np.asarray(plan.centers_xy_m)
    n_patches = centers.shape[0]

    # "matched" can only be resolved here: the k-grid that makes the splat exact
    # is a function of plan.pad_px, which plan_patches derives (and may dilate)
    # rather than the caller choosing. Resolving it in the caller would pin a pad
    # the plan did not use.
    kspace_basis: dict[str, Any] | None = None
    if kspace_grid_shape == "matched":
        kspace_basis = matched_kspace_grid(
            pad_px=plan.pad_px,
            doe_pitch_m=doe.pitch_m[0],
            sensor_pitch_m=doe.pitch_m[0],
            min_shape=sensor_shape,
        )
        kspace_grid_shape = tuple(kspace_basis["kspace_grid_shape"])
    groups = np.array_split(np.arange(n_patches), max(1, min(batches, n_patches)))

    # The total must be known before the first chunk: the 1/N is the
    # estimator's, and it is applied once at the end.
    if secondary_count is None:
        probe_plan = _single_patch_plan(plan, centers[:1])
        probe, probe_diagnostics = patch_secondary_rays(
            doe.transmission,
            plan=probe_plan,
            sample_pitch_m=doe.pitch_m,
            wavelength_m=wavelength_m,
            plane=plane,
        )
        per_patch = int(probe.count)
        del probe, probe_diagnostics
    else:
        per_patch = int(secondary_count)
    total_rays = n_patches * per_patch

    accumulator = _DualAccumulator(
        grid_shape=sensor_shape,
        sample_pitch_m=doe.pitch_m,
        plane=sensor,
        wavelength_m=wavelength_m,
        namespace=ArrayNamespace.NUMPY if backend == "numpy" else ArrayNamespace.JAX,
        complex_dtype=DType.COMPLEX128 if precision == "fp64" else DType.COMPLEX64,
        total_rays=total_rays,
        projection=Projection.ASM_CONSISTENT,
        reconstruction=Reconstruction(reconstruction),
        **(
            {"kspace_oversample": kspace_oversample}
            if kspace_oversample is not None
            else {}
        ),
        kspace_grid_shape=tuple(kspace_grid_shape) if kspace_grid_shape else None,
    )
    started = time.perf_counter()
    diagnostics: dict[str, Any] = {}
    for group in groups:
        if group.size == 0:
            continue
        chunk_plan = _single_patch_plan(plan, centers[group])
        bundle, patch_diagnostics = patch_secondary_rays(
            doe.transmission,
            plan=chunk_plan,
            sample_pitch_m=doe.pitch_m,
            wavelength_m=wavelength_m,
            plane=plane,
            secondary_count=secondary_count,
            rng=rng,
        )
        diagnostics = patch_diagnostics.as_dict()
        if advance:
            from couplers.patch import advance_bundle_to_plane

            bundle = advance_bundle_to_plane(bundle, target=sensor)
        else:
            import dataclasses

            bundle = dataclasses.replace(bundle, reference_plane=sensor)
        device_bundle = to_namespace(bundle, backend=backend, precision=precision)
        accumulator.add_chunk(_Batch(device_bundle))
        del bundle, device_bundle

    result = accumulator.finalize(provenance={"probe": "patch_route"})
    wall_clock_s = time.perf_counter() - started
    field = np.asarray(result.field.u)

    return {
        "field": field,
        "notebook_variant_intensity": accumulator.notebook_intensity**2,
        "plan": {
            "patch_px": plan.patch_px,
            "pad_px": plan.pad_px,
            "pad_requested": patch_px * pad_factor,
            "coverage": plan.coverage,
            "dilation_px": plan.dilation_px,
            "curvature_bound_rad": plan.curvature_bound_rad,
            "patch_count": n_patches,
        },
        "patch_diagnostics": diagnostics,
        "reconstruction": {
            "route": reconstruction,
            "requested_kspace_grid_shape": list(kspace_grid_shape) if kspace_grid_shape else None,
            "requested_kspace_oversample": kspace_oversample,
            "matched_grid_basis": kspace_basis,
            # Read off the first chunk's diagnostics rather than restated from
            # the request: on_node_fraction is what says whether this run was an
            # exactness measurement or an interpolation, and dropped_fraction is
            # what stops a lossy run from reading like a clean one.
            "measured": (accumulator._first_diagnostics or {}).get("kspace"),
        },
        "streaming": result.as_dict(),
        "total_rays": total_rays,
        "secondary_per_patch": per_patch,
        "wall_clock_s": wall_clock_s,
        "actual": {
            "field_dtype": str(result.field.u.dtype),
            "field_device": str(getattr(result.field.u, "device", "host")),
        },
    }


def _single_patch_plan(plan: Any, centers: np.ndarray) -> Any:
    import dataclasses

    return dataclasses.replace(plan, centers_xy_m=centers)


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

def device_memory_stats() -> dict[str, Any]:
    """Peak device bytes, read from the runtime rather than estimated.

    SI Table S2 reports peak memory on the same GPU class this box has, which
    makes it one of the few directly comparable columns -- but only if the
    number comes from the allocator. `peak_bytes_in_use` is what JAX's BFC
    allocator actually reserved, so it includes the arena's own slack; that is
    the same quantity `nvidia-smi` would show and the same one the paper's
    column means.
    """
    try:
        import jax

        device = jax.devices()[0]
        stats = device.memory_stats() or {}
    except Exception:
        return {"available": False}
    peak = stats.get("peak_bytes_in_use")
    return {
        "available": True,
        "device": str(jax.devices()[0]),
        "peak_bytes_in_use": peak,
        "peak_gb": round(peak / 1e9, 3) if peak else None,
        "bytes_limit": stats.get("bytes_limit"),
    }


def environment() -> dict[str, Any]:
    """Commit, image and device, read rather than declared."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=REPO,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
    except Exception:  # pragma: no cover - git absent in the container is fine
        commit, dirty = "unknown", True

    devices: list[str] = []
    try:
        import jax

        devices = [str(d) for d in jax.devices()]
    except Exception:
        devices = ["jax unavailable"]

    return {
        "commit": commit,
        "working_tree_dirty": dirty,
        "container_image": os.environ.get("MOA_IMAGE", "agent_solver"),
        "jax_devices": devices,
    }


def write_record(name: str, payload: dict[str, Any]) -> Path:
    """Write a demo record, stamped with the source and environment it ran under.

    CHE-129 (M0.4 follow-up). The stamp is the enrollment: `REGISTER.yaml`
    deferred the whole `ray_wave/*.json` corpus because regenerating it is hours
    of GPU compute, and `tests/test_provenance_fingerprint.py` therefore could not
    tell whether any of these records still described the code that produced them.
    Stamping here rather than in each demo means a demo cannot be added that
    writes an unstamped record.

    Called with `sys.modules` already holding everything the run imported, since
    every caller writes its record after the physics.
    """
    from core.provenance import RECORD_PROVENANCE_KEY, record_provenance

    RECORDS.mkdir(parents=True, exist_ok=True)
    path = RECORDS / f"{name}.json"
    stamped = {
        **payload,
        RECORD_PROVENANCE_KEY: record_provenance(
            probe=name,
            root=REPO,
            extra_sources=[Path(__file__)],
        ),
    }
    path.write_text(json.dumps(stamped, indent=2, sort_keys=True) + "\n")
    return path
