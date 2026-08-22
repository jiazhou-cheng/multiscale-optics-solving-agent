"""GPU leg of the reduced-precision measurement table (CHE-61 / PB4b step 9)."""
import json

import jax
import numpy as np

from core.precision import ArrayNamespace, Precision
from couplers.contracts import (
    ComplexField,
    Frame,
    RayBundle,
    ReferencePlane,
)
from couplers.ray_to_wave import collimated_bundle, ray_to_wave
from couplers.wave_to_ray import wave_to_ray

WL, PITCH, N = 500e-9, 1.0e-6, 16
GRID, PIT = (N, N), (PITCH, PITCH)
D = (0.10, -0.05, float(np.sqrt(1 - 0.01 - 0.0025)))
axis = (np.arange(N) - N // 2) * PITCH
xx, yy = np.meshgrid(axis, axis, indexing="xy")
pos = np.column_stack([xx.ravel(), yy.ravel()])

gpu = next(d for d in jax.devices() if d.platform == "gpu")
out = {"jax_backend": jax.default_backend(), "device": str(gpu)}

def analytic():
    k = 2 * np.pi / WL
    a = (np.arange(N, dtype=np.float64) - N // 2) * PITCH
    X, Y = np.meshgrid(a, a, indexing="xy")
    return np.exp(1j * k * (D[0] * X + D[1] * Y)) * (N * N)

oracle = analytic()
scale = float(np.max(np.abs(oracle)))

host = collimated_bundle(positions_xy_m=pos, direction=D, wavelength_m=WL,
                         precision=Precision.FP32, namespace=ArrayNamespace.JAX)
b = RayBundle(
    positions_m=jax.device_put(host.positions_m, gpu),
    directions=jax.device_put(host.directions, gpu),
    wavelength_m=WL, reference_plane=host.reference_plane, frame=host.frame,
    amplitude=jax.device_put(host.amplitude, gpu),
    optical_path_length_m=jax.device_put(host.optical_path_length_m, gpu),
    optical_path_length_reference=host.optical_path_length_reference,
)
field, _ = ray_to_wave(b, grid_shape=GRID, sample_pitch_m=PIT)
out["ray_to_wave_gpu_fp32"] = {
    "input_device": str(b.device), "output_device": str(field.device),
    "output_dtype": str(field.dtype),
    "max_abs_error_rel_peak_vs_analytic":
        float(np.max(np.abs(np.asarray(field.u, np.complex128) - oracle)) / scale),
}

# round trip on GPU
rng = np.random.default_rng(20260819)
base = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
spec = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(base)))
f = np.fft.fftshift(np.fft.fftfreq(N, d=PITCH)) * WL
dv, du = np.meshgrid(f, f, indexing="ij")
spec = np.where(du**2 + dv**2 < 1.0, spec, 0.0)
u0 = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(spec))).astype(np.complex64)
fld = ComplexField(u=jax.device_put(u0, gpu), sample_pitch_m=PIT, wavelength_m=WL,
                   reference_plane=ReferencePlane(name="p", z_m=0.0), frame=Frame())
bun, spectrum, _ = wave_to_ray(fld)
back, _ = ray_to_wave(bun, grid_shape=GRID, sample_pitch_m=PIT)
a, bb = np.asarray(back.u, np.complex128), u0.astype(np.complex128)
out["round_trip_gpu_c64"] = {
    "bundle_device": str(bun.device), "output_device": str(back.device),
    "propagating_modes": spectrum.propagating_count,
    "relative_l2_error": float(np.linalg.norm(a - bb) / np.linalg.norm(bb)),
    "max_abs_error_rel_peak": float(np.max(np.abs(a - bb)) / np.max(np.abs(bb))),
}
print(json.dumps(out, indent=2))
