"""Is a complex64 einsum on this GPU actually complex64, or TF32? (CHE-61)

XLA:GPU's default precision for an f32/c64 dot on Ampere is TF32 -- a 10-bit
mantissa rather than 24 -- so `dtype=complex64` on the array says nothing about
the accuracy of a matmul over it. This measures the coupler's own contraction,
`einsum("n,ny,nx->yx", ...)` over 256 complex64 wavelets, against a complex128
reference, and shows that `precision="highest"` restores genuine complex64
accuracy.

The FFT is measured alongside it as a control: it is unaffected, which is why the
fix belongs on the dot products and not on the transform.

    ./run.sh --gpu python benchmarks/probes/precision/gpu_matmul.py
"""
import json

import jax
import jax.numpy as jnp
import numpy as np

out = {"backend": jax.default_backend(),
       "default_matmul_precision": str(jax.config.jax_default_matmul_precision)}

rng = np.random.default_rng(0)
n, m = 256, 16
coeff = (rng.standard_normal(n) + 1j*rng.standard_normal(n)).astype(np.complex64)
ry = (rng.standard_normal((n, m)) + 1j*rng.standard_normal((n, m))).astype(np.complex64)
rx = (rng.standard_normal((n, m)) + 1j*rng.standard_normal((n, m))).astype(np.complex64)

exact = np.einsum("n,ny,nx->yx", coeff.astype(np.complex128),
                  ry.astype(np.complex128), rx.astype(np.complex128), optimize=True)
scale = np.max(np.abs(exact))

cpu = np.einsum("n,ny,nx->yx", coeff, ry, rx, optimize=True)
out["numpy_complex64"] = float(np.max(np.abs(cpu - exact)) / scale)

jc, jy, jx = jnp.asarray(coeff), jnp.asarray(ry), jnp.asarray(rx)
out["jax_device"] = str(jc.devices())

g = np.asarray(jnp.einsum("n,ny,nx->yx", jc, jy, jx, optimize=True))
out["jax_default"] = float(np.max(np.abs(g - exact)) / scale)

for prec in ("highest", "float32", "high", "default"):
    try:
        h = np.asarray(jnp.einsum("n,ny,nx->yx", jc, jy, jx, optimize=True, precision=prec))
        out[f"jax_precision_{prec}"] = float(np.max(np.abs(h - exact)) / scale)
    except Exception as e:
        out[f"jax_precision_{prec}"] = repr(e)

# fft accuracy alone
f = (rng.standard_normal((16,16)) + 1j*rng.standard_normal((16,16))).astype(np.complex64)
fe = np.fft.fft2(f.astype(np.complex128))
out["fft_numpy_c64"] = float(np.max(np.abs(np.fft.fft2(f) - fe)) / np.max(np.abs(fe)))
gpu_fft = np.asarray(jnp.fft.fft2(jnp.asarray(f)))
out["fft_jax_gpu_c64"] = float(np.max(np.abs(gpu_fft - fe)) / np.max(np.abs(fe)))
print(json.dumps(out, indent=2))
