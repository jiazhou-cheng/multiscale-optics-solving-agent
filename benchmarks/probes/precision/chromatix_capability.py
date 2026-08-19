"""Chromatix's real precision and device behaviour, plus the torch->JAX hop (CHE-61).

Three findings this probe establishes, each of which a capability declaration
depends on:

1. **Chromatix has no complex128 path.** ``ScalarField.__init__`` is
   ``jnp.asarray(u, dtype=jnp.complex64)`` unconditionally, and
   ``Field.build`` handed a ``complex128`` array *with* ``jax_enable_x64``
   enabled still returns ``complex64``. So an FP64 request has nothing to
   execute, at any device.
2. **``asm_propagate`` runs on the GPU and its output lands there**, which is
   what the device-aware adapter path reports rather than asserting.
3. **A torch CUDA tensor crosses into JAX via DLPack without leaving the
   device**, which is what makes the Optiland -> coupler hop a namespace change
   rather than a host round trip.

    ./run.sh --gpu python benchmarks/probes/precision/chromatix_capability.py
"""

import inspect
import json

import chromatix.functional as cf
import jax
import jax.numpy as jnp
import numpy as np
import torch
from chromatix.core.field import ScalarField

WAVELENGTH_M = 0.55e-6
PITCH_M = 1e-6


def probe_jax() -> dict[str, object]:
    field = jnp.ones((8, 8), dtype=jnp.complex64)
    # A real kernel, not just enumeration: PB4a measured an image where
    # jax.devices() listed a CUDA device and the first jitted call could not
    # compile.
    transformed = jnp.fft.fft2(field).block_until_ready()
    out: dict[str, object] = {
        "version": jax.__version__,
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "array_device": str(next(iter(field.devices()))),
        "fft_device": str(next(iter(transformed.devices()))),
    }
    jax.config.update("jax_enable_x64", True)
    out["complex128_dtype_under_x64"] = str(jnp.ones((4,), dtype=jnp.complex128).dtype)
    jax.config.update("jax_enable_x64", False)
    out["complex128_dtype_without_x64"] = str(jnp.ones((4,), dtype=jnp.complex128).dtype)
    return out


def probe_chromatix() -> dict[str, object]:
    pitch = jnp.asarray([[PITCH_M, PITCH_M]])
    field = cf.Field.build(jnp.ones((16, 16), dtype=jnp.complex64), pitch, WAVELENGTH_M)
    propagated = cf.asm_propagate(field, z=1e-3, n=1.0, pad_width=4)

    out: dict[str, object] = {
        "scalarfield_source_mentions_complex64": "complex64" in inspect.getsource(ScalarField),
        "field_u_dtype": str(field.u.dtype),
        "field_u_device": str(next(iter(field.u.devices()))),
        "asm_output_dtype": str(propagated.u.dtype),
        "asm_output_device": str(next(iter(propagated.u.devices()))),
        "asm_output_platform": next(iter(propagated.u.devices())).platform,
    }

    # The decisive test: a genuine complex128 input, with x64 enabled so JAX
    # itself would preserve it. Chromatix does not.
    jax.config.update("jax_enable_x64", True)
    try:
        from_c128 = cf.Field.build(
            np.ones((8, 8), dtype=np.complex128), pitch, WAVELENGTH_M
        )
        out["build_from_complex128_returns"] = str(from_c128.u.dtype)
    except Exception as exc:
        out["build_from_complex128_returns"] = f"{type(exc).__name__}: {exc}"
    finally:
        jax.config.update("jax_enable_x64", False)
    return out


def probe_dlpack() -> dict[str, object]:
    out: dict[str, object] = {}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tensor = torch.arange(6, dtype=torch.float32, device=device).reshape(2, 3)
    try:
        converted = jnp.from_dlpack(tensor)
        out["torch_to_jax"] = {
            "dtype": str(converted.dtype),
            "source_device": str(tensor.device),
            "target_device": str(next(iter(converted.devices()))),
            "values_match": bool(
                np.allclose(np.asarray(converted), tensor.cpu().numpy())
            ),
        }
    except Exception as exc:
        out["torch_to_jax"] = f"{type(exc).__name__}: {exc}"
    try:
        back = torch.from_dlpack(jnp.ones((2, 3), dtype=jnp.float32))
        out["jax_to_torch_device"] = str(back.device)
    except Exception as exc:
        out["jax_to_torch_device"] = f"{type(exc).__name__}: {exc}"
    return out


def main() -> None:
    print(
        json.dumps(
            {
                "jax": probe_jax(),
                "chromatix": probe_chromatix(),
                "dlpack": probe_dlpack(),
            },
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
