"""Narrow Optiland operations used only by the L1-RAY-01 accuracy suite.

The analytic oracles live in the benchmark evaluator, not here. This module
contains the external-solver imports and exposes solver-native arrays so the
evaluator cannot accidentally use Optiland to compute its own expectations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from multiscale_optics_agent.core.errors import AdapterDependencyError


@dataclass(frozen=True)
class OptilandTrace:
    inputs: dict[str, np.ndarray]
    outputs: dict[str, np.ndarray]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class OptilandScalingRequest:
    """Fixed Optiland-only input contract for the CHE-16 scaling capability."""

    requested_sampling: int
    wavelength_um: float = 0.55
    field_hx: float = 0.0
    field_hy: float = 0.0
    backend: str = "numpy"
    device: str = "cpu"
    dtype: str = "float64"

    def __post_init__(self) -> None:
        if self.requested_sampling <= 0:
            raise ValueError("requested_sampling must be positive")
        if self.wavelength_um <= 0:
            raise ValueError("wavelength_um must be positive")
        if not -1.0 <= self.field_hx <= 1.0 or not -1.0 <= self.field_hy <= 1.0:
            raise ValueError("field coordinates must lie in [-1, 1]")
        if (self.backend, self.device, self.dtype) != ("numpy", "cpu", "float64"):
            raise ValueError(
                "CHE-16 supports only backend='numpy', device='cpu', dtype='float64'"
            )


@dataclass(frozen=True)
class OptilandScalingDependencies:
    """Opaque holder that keeps all external imports inside this adapter module."""

    backend: Any
    backend_utils: Any
    reverse_telephoto_type: Any


@dataclass(frozen=True)
class OptilandScalingTrace:
    """One measured trace with canonical NumPy arrays and count diagnostics."""

    arrays: dict[str, np.ndarray]
    generated_ray_count: int
    traced_ray_count: int
    surviving_ray_count: int
    invalid_ray_count: int
    vignetted_ray_count: int


@dataclass
class OptilandScalingSession:
    """Prepared fixed prescription whose :meth:`trace` call excludes setup."""

    request: OptilandScalingRequest
    _lens: Any
    _backend_utils: Any
    surface_count: int

    def trace(self) -> OptilandScalingTrace:
        rays = self._lens.trace(
            Hx=self.request.field_hx,
            Hy=self.request.field_hy,
            wavelength=self.request.wavelength_um,
            num_rays=self.request.requested_sampling,
        )
        native = {
            "x_m": np.asarray(self._backend_utils.to_numpy(rays.x), dtype=np.float64) * 1e-3,
            "y_m": np.asarray(self._backend_utils.to_numpy(rays.y), dtype=np.float64) * 1e-3,
            "z_m": np.asarray(self._backend_utils.to_numpy(rays.z), dtype=np.float64) * 1e-3,
            "L": np.asarray(self._backend_utils.to_numpy(rays.L), dtype=np.float64),
            "M": np.asarray(self._backend_utils.to_numpy(rays.M), dtype=np.float64),
            "N": np.asarray(self._backend_utils.to_numpy(rays.N), dtype=np.float64),
            "intensity": np.asarray(
                self._backend_utils.to_numpy(rays.i), dtype=np.float64
            ),
            "wavelength_m": np.asarray(
                self._backend_utils.to_numpy(rays.w), dtype=np.float64
            )
            * 1e-6,
            "opd_native": np.asarray(
                self._backend_utils.to_numpy(rays.opd), dtype=np.float64
            ),
        }
        shapes = {name: value.shape for name, value in native.items()}
        if not native["x_m"].size or len(set(shapes.values())) != 1:
            raise ValueError(f"Optiland scaling arrays must be non-empty and aligned: {shapes}")
        invalid = np.zeros(native["x_m"].shape, dtype=np.bool_)
        for value in native.values():
            invalid |= ~np.isfinite(value)
        vignetted = native["intensity"] <= 0.0
        surviving = ~invalid & ~vignetted
        returned_count = int(native["x_m"].size)
        return OptilandScalingTrace(
            arrays=native,
            # In Optiland 0.6.0 RealRayTracer.trace expands the generated pupil
            # points, traces that same RealRays object in place, and does not
            # filter its rows before returning. With this single fixed field,
            # generated and traced counts therefore equal the returned length.
            generated_ray_count=returned_count,
            traced_ray_count=returned_count,
            surviving_ray_count=int(np.count_nonzero(surviving)),
            invalid_ray_count=int(np.count_nonzero(invalid)),
            vignetted_ray_count=int(np.count_nonzero(vignetted)),
        )


def import_optiland_scaling_dependencies() -> OptilandScalingDependencies:
    """Import the pinned CHE-16 dependency set without constructing a lens."""

    try:
        import optiland.backend as be
        import optiland.backend.utils as be_utils
        from optiland.samples.objectives import ReverseTelephoto
    except Exception as exc:
        raise AdapterDependencyError(
            f"CHE-16 requires optiland==0.6.0: {type(exc).__name__}: {exc}"
        ) from exc
    be.set_backend("numpy")
    return OptilandScalingDependencies(
        backend=be,
        backend_utils=be_utils,
        reverse_telephoto_type=ReverseTelephoto,
    )


def prepare_optiland_scaling_session(
    dependencies: OptilandScalingDependencies,
    request: OptilandScalingRequest,
) -> OptilandScalingSession:
    """Construct the one fixed ReverseTelephoto prescription used by CHE-16."""

    lens = dependencies.reverse_telephoto_type()
    return OptilandScalingSession(
        request=request,
        _lens=lens,
        _backend_utils=dependencies.backend_utils,
        surface_count=len(lens.surfaces.surfaces),
    )


@dataclass(frozen=True)
class Edmund45362Prescription:
    """Complete catalog geometry shared by tracing, validation, and plotting."""

    manufacturer: str
    part_number: str
    radius_1_mm: float
    center_thickness_mm: float
    clear_aperture_mm: float
    material: str
    catalog_efl_mm: float
    catalog_bfl_mm: float

    @property
    def clear_aperture_radius_mm(self) -> float:
        return self.clear_aperture_mm / 2.0

    @property
    def front_vertex_z_mm(self) -> float:
        return 0.0

    @property
    def back_vertex_z_mm(self) -> float:
        return self.center_thickness_mm

    @property
    def image_reference_plane_z_mm(self) -> float:
        return self.center_thickness_mm + self.catalog_bfl_mm

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> Edmund45362Prescription:
        return cls(
            manufacturer=str(value["manufacturer"]),
            part_number=str(value["part_number"]),
            radius_1_mm=float(value["surface_1_radius_mm"]),
            center_thickness_mm=float(value["center_thickness_mm"]),
            clear_aperture_mm=float(value["clear_aperture_mm"]),
            material=str(value["material"]),
            catalog_efl_mm=float(value["catalog_efl_mm"]),
            catalog_bfl_mm=float(value["catalog_bfl_mm"]),
        )


def _imports() -> tuple[Any, Any, Any, Any]:
    try:
        import optiland.backend as be
        from optiland.optic import Optic
        from optiland.physical_apertures import RadialAperture
        from optiland.rays import RealRays
    except Exception as exc:
        raise AdapterDependencyError(
            f"L1-RAY-01 requires optiland==0.6.0: {type(exc).__name__}: {exc}"
        ) from exc
    be.set_backend("numpy")
    return be, Optic, RadialAperture, RealRays


def _copy_inputs(**arrays: np.ndarray) -> dict[str, np.ndarray]:
    return {name: np.asarray(value, dtype=np.float64).copy() for name, value in arrays.items()}


def _outputs(rays: Any) -> dict[str, np.ndarray]:
    return {
        "x_mm": np.asarray(rays.x, dtype=np.float64),
        "y_mm": np.asarray(rays.y, dtype=np.float64),
        "z_mm": np.asarray(rays.z, dtype=np.float64),
        "L": np.asarray(rays.L, dtype=np.float64),
        "M": np.asarray(rays.M, dtype=np.float64),
        "N": np.asarray(rays.N, dtype=np.float64),
        "intensity": np.asarray(rays.i, dtype=np.float64),
        "wavelength_um": np.asarray(rays.w, dtype=np.float64),
        "opd_native_mm": np.asarray(rays.opd, dtype=np.float64),
    }


def trace_free_space(
    *,
    x_mm: np.ndarray,
    y_mm: np.ndarray,
    z_mm: np.ndarray,
    L: np.ndarray,
    M: np.ndarray,
    N: np.ndarray,
    distance_mm: float,
    wavelength_um: float,
) -> OptilandTrace:
    """Trace manufactured rays to one plane through homogeneous air."""
    be, Optic, _, RealRays = _imports()
    count = np.asarray(x_mm).size
    inputs = _copy_inputs(
        x_mm=x_mm,
        y_mm=y_mm,
        z_mm=z_mm,
        L=L,
        M=M,
        N=N,
        intensity=np.ones(count, dtype=np.float64),
        wavelength_um=np.full(count, wavelength_um, dtype=np.float64),
    )
    # RealRays uses x/y/z argument names, so construct explicitly after keeping
    # an immutable copy for the benchmark artifact.
    rays = RealRays(
        inputs["x_mm"],
        inputs["y_mm"],
        inputs["z_mm"],
        inputs["L"],
        inputs["M"],
        inputs["N"],
        inputs["intensity"],
        inputs["wavelength_um"],
    )
    optic = Optic("L1-RAY-01 free space")
    optic.surfaces.add(index=0, radius=be.inf, thickness=0.0)
    optic.surfaces.add(index=1, radius=be.inf, z=distance_mm)
    optic.surfaces[1].trace(rays)
    return OptilandTrace(
        inputs=inputs,
        outputs=_outputs(rays),
        metadata={
            "distance_mm": distance_mm,
            "wavelength_um": wavelength_um,
            "medium": "air ideal n=1",
            "reference_plane_z_mm": distance_mm,
        },
    )


def trace_paraxial_thin_lens(
    *,
    focal_length_mm: float,
    pupil_heights_mm: np.ndarray,
    launch_slopes_rad: np.ndarray,
    wavelength_um: float,
) -> OptilandTrace:
    """Trace manufactured rays through Optiland's ideal paraxial surface."""
    be, Optic, _, RealRays = _imports()
    slope_grid, height_grid = np.meshgrid(launch_slopes_rad, pupil_heights_mm, indexing="ij")
    angles = slope_grid.ravel()
    x = height_grid.ravel()
    L = np.sin(angles)
    N = np.cos(angles)
    zeros = np.zeros_like(x)
    inputs = _copy_inputs(
        x_mm=x,
        y_mm=zeros,
        z_mm=zeros,
        L=L,
        M=zeros,
        N=N,
        launch_slope_rad=angles,
        pupil_height_mm=x,
        intensity=np.ones_like(x),
        wavelength_um=np.full_like(x, wavelength_um),
    )
    rays = RealRays(
        inputs["x_mm"],
        inputs["y_mm"],
        inputs["z_mm"],
        inputs["L"],
        inputs["M"],
        inputs["N"],
        inputs["intensity"],
        inputs["wavelength_um"],
    )
    optic = Optic("L1-RAY-01 paraxial thin lens")
    optic.surfaces.add(index=0, radius=be.inf, thickness=0.0)
    optic.surfaces.add(
        index=1,
        surface_type="paraxial",
        f=focal_length_mm,
        thickness=focal_length_mm,
        is_stop=True,
    )
    optic.surfaces.add(index=2, radius=be.inf, thickness=0.0)
    optic.surfaces.trace(rays, skip=1)
    return OptilandTrace(
        inputs=inputs,
        outputs=_outputs(rays),
        metadata={
            "focal_length_mm": focal_length_mm,
            "wavelength_um": wavelength_um,
            "aperture_radius_mm": float(np.max(np.abs(pupil_heights_mm))),
            "reference_plane_z_mm": focal_length_mm,
            "surface_type": "paraxial thin lens",
        },
    )


def build_edmund_45362_optic(prescription: Edmund45362Prescription) -> Any:
    """Build the catalog optic from the benchmark's single prescription object."""
    be, Optic, RadialAperture, _ = _imports()
    optic = Optic(f"{prescription.manufacturer} {prescription.part_number}")
    aperture = RadialAperture(prescription.clear_aperture_radius_mm)
    optic.surfaces.add(index=0, radius=be.inf, thickness=0.0, comment="object plane")
    optic.surfaces.add(
        index=1,
        radius=prescription.radius_1_mm,
        thickness=prescription.center_thickness_mm,
        material=prescription.material,
        aperture=aperture,
        comment="convex spherical front surface",
    )
    optic.surfaces.add(
        index=2,
        radius=be.inf,
        thickness=prescription.catalog_bfl_mm,
        material="air",
        aperture=RadialAperture(prescription.clear_aperture_radius_mm),
        comment="plano rear surface",
    )
    optic.surfaces.add(
        index=3,
        radius=be.inf,
        thickness=0.0,
        comment="catalog BFL image/reference plane",
    )
    return optic


def trace_edmund_45362(
    *,
    prescription: Edmund45362Prescription,
    pupil_heights_mm: np.ndarray,
    launch_slopes_rad: np.ndarray,
    object_plane_z_mm: float,
    wavelength_um: float,
) -> OptilandTrace:
    """Trace the published Edmund Optics TECHSPEC stock #45-362 prescription."""
    _, _, _, RealRays = _imports()

    slope_grid, height_grid = np.meshgrid(launch_slopes_rad, pupil_heights_mm, indexing="ij")
    angles = slope_grid.ravel()
    x = height_grid.ravel()
    L = np.sin(angles)
    N = np.cos(angles)
    zeros = np.zeros_like(x)
    inputs = _copy_inputs(
        x_mm=x,
        y_mm=zeros,
        z_mm=np.full_like(x, object_plane_z_mm),
        L=L,
        M=zeros,
        N=N,
        launch_slope_rad=angles,
        pupil_height_mm=x,
        intensity=np.ones_like(x),
        wavelength_um=np.full_like(x, wavelength_um),
    )
    rays = RealRays(
        inputs["x_mm"],
        inputs["y_mm"],
        inputs["z_mm"],
        inputs["L"],
        inputs["M"],
        inputs["N"],
        inputs["intensity"],
        inputs["wavelength_um"],
    )
    optic = build_edmund_45362_optic(prescription)
    history_x = [np.asarray(rays.x, dtype=np.float64).copy()]
    history_y = [np.asarray(rays.y, dtype=np.float64).copy()]
    history_z = [np.asarray(rays.z, dtype=np.float64).copy()]
    geometric_path_mm = np.zeros_like(x)
    for surface in optic.surfaces.surfaces[1:]:
        before = np.stack((rays.x, rays.y, rays.z), axis=0).astype(np.float64)
        surface.trace(rays)
        after = np.stack((rays.x, rays.y, rays.z), axis=0).astype(np.float64)
        geometric_path_mm += np.linalg.norm(after - before, axis=0)
        history_x.append(np.asarray(rays.x, dtype=np.float64).copy())
        history_y.append(np.asarray(rays.y, dtype=np.float64).copy())
        history_z.append(np.asarray(rays.z, dtype=np.float64).copy())
    outputs = _outputs(rays)
    outputs["geometric_path_mm"] = geometric_path_mm
    outputs["history_x_mm"] = np.stack(history_x)
    outputs["history_y_mm"] = np.stack(history_y)
    outputs["history_z_mm"] = np.stack(history_z)
    n_d = float(np.asarray(optic.surfaces[1].material_post.n(np.array([wavelength_um])))[0])
    return OptilandTrace(
        inputs=inputs,
        outputs=outputs,
        metadata={
            "manufacturer": prescription.manufacturer,
            "part_number": prescription.part_number,
            "radius_1_mm": prescription.radius_1_mm,
            "radius_2_mm": "infinity",
            "center_thickness_mm": prescription.center_thickness_mm,
            "material": prescription.material,
            "clear_aperture_mm": prescription.clear_aperture_mm,
            "clear_aperture_radius_mm": prescription.clear_aperture_radius_mm,
            "front_vertex_z_mm": prescription.front_vertex_z_mm,
            "back_vertex_z_mm": prescription.back_vertex_z_mm,
            "catalog_efl_mm": prescription.catalog_efl_mm,
            "catalog_bfl_mm": prescription.catalog_bfl_mm,
            "reference_plane_z_mm": prescription.image_reference_plane_z_mm,
            "wavelength_um": wavelength_um,
            "optiland_refractive_index": n_d,
        },
    )
