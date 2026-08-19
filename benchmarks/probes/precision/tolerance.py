"""Measure reduced-precision coupler error so tolerances are derived, not chosen."""
import json

import numpy as np

from multiscale_optics_agent.core.precision import ArrayNamespace, Precision
from multiscale_optics_agent.couplers import wave_to_ray as w2r
from multiscale_optics_agent.couplers.contracts import ComplexField, Frame, ReferencePlane
from multiscale_optics_agent.couplers.ray_to_wave import collimated_bundle, ray_to_wave

WL = 500e-9
N = 16
PITCH = 1.0e-6
grid = (N, N)
pitch = (PITCH, PITCH)
xs = (np.arange(N) - N // 2) * PITCH
X, Y = np.meshgrid(xs, xs, indexing="xy")
pos = np.column_stack([X.ravel(), Y.ravel()])
direction = (0.10, -0.05, np.sqrt(1 - 0.01 - 0.0025))

out = {}

def analytic(dtype):
    """exp(+i k d.r) on the grid -- exact oracle, evaluated in float64."""
    k = 2 * np.pi / WL
    y = (np.arange(N, dtype=np.float64) - N // 2) * PITCH
    x = (np.arange(N, dtype=np.float64) - N // 2) * PITCH
    xx, yy = np.meshgrid(x, y, indexing="xy")
    return np.exp(1j * k * (direction[0] * xx + direction[1] * yy)) * pos.shape[0]

ref = None
for label, precision, namespace in [
    ("numpy_fp64", Precision.FP64, None),
    ("numpy_fp32", Precision.FP32, None),
    ("jax_fp32", Precision.FP32, ArrayNamespace.JAX),
]:
    bundle = collimated_bundle(
        positions_xy_m=pos, direction=direction, wavelength_m=WL,
        precision=precision, namespace=namespace,
    )
    field, diag = ray_to_wave(bundle, grid_shape=grid, sample_pitch_m=pitch)
    u = np.asarray(field.u, dtype=np.complex128)
    oracle = analytic(None)
    scale = float(np.max(np.abs(oracle)))
    rel = float(np.max(np.abs(u - oracle)) / scale)
    out[label] = {
        "field_dtype": str(field.dtype), "namespace": str(field.namespace),
        "device": str(field.device),
        "max_abs_error_rel_peak_vs_analytic": rel,
        "compute_precision": diag.as_dict()["normalization"],
    }
    if label == "numpy_fp64":
        ref = u
    else:
        out[label]["max_abs_error_rel_peak_vs_fp64"] = float(
            np.max(np.abs(u - ref)) / scale
        )

# wave_to_ray round trip at each precision
rng = np.random.default_rng(7)
base = (rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N)))
for label, cdtype in [("c128", np.complex128), ("c64", np.complex64)]:
    # band-limit so every mode propagates
    spec = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(base)))
    fu = np.fft.fftshift(np.fft.fftfreq(N, d=PITCH)) * WL
    dv, du = np.meshgrid(fu, fu, indexing="ij")
    spec = np.where(du**2 + dv**2 < 1.0, spec, 0.0)
    u0 = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(spec))).astype(cdtype)
    field0 = ComplexField(u=u0, sample_pitch_m=pitch, wavelength_m=WL,
                          reference_plane=ReferencePlane(name="p", z_m=0.0), frame=Frame())
    bundle, spectrum, density = w2r.wave_to_ray(field0)
    back, _ = ray_to_wave(bundle, grid_shape=grid, sample_pitch_m=pitch)
    a, b = np.asarray(back.u, np.complex128), np.asarray(u0, np.complex128)
    out[f"round_trip_{label}"] = {
        "bundle_dtype": str(bundle.dtype),
        "amplitude_dtype": str(np.asarray(bundle.amplitude).dtype),
        "field_dtype": str(back.dtype),
        "propagating_modes": spectrum.propagating_count,
        "relative_l2_error": float(np.linalg.norm(a - b) / np.linalg.norm(b)),
        "max_abs_error_rel_peak": float(np.max(np.abs(a - b)) / np.max(np.abs(b))),
    }
print(json.dumps(out, indent=2))
