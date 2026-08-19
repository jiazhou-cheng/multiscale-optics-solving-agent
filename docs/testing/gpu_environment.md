# GPU execution environment (CHE-60 / PB4a, Phase 0)

Status: **complete and validated** on the target host, 2026-08-19.

This document records how GPU execution is provisioned, the evidence that it
actually works, and two non-obvious traps that silently disable it. It is the
prerequisite for every other phase of CHE-60 — Optiland GPU tracing, Chromatix
GPU propagation, and GPU-resident ray-wave couplers all assume what is
established here.

## Why a separate image

The default `agent_solver` image cannot reach a GPU, and not by accident:

| Component | Default `agent_solver` | `agent_solver_gpu` |
| --- | --- | --- |
| torch | `2.13.0+cpu` (from `https://download.pytorch.org/whl/cpu`) | `2.13.0+cu126` |
| jaxlib | CPU-only (plain `jax==0.6.2`) | + `jax-cuda12-plugin`/`pjrt` `0.6.2` |
| PTX compiler | n/a | `nvidia-cuda-nvcc-cu12==12.6.85` |
| `--gpus` | never passed | passed by `run.sh --gpu` |
| Image size | 3.57 GB | 9.98 GB |

CHE-60 keeps these as two images rather than swapping CUDA wheels into the
existing one, because the issue requires that "existing CPU/high-precision
behavior does not regress." Two images make that true by construction: nothing
about the CPU path changes, and CPU-only runs (all of Tier A) do not pay the
~6.4 GB size increase.

`docker/Dockerfile.gpu` installs `docker/requirements.txt` first and applies
`docker/requirements-gpu.txt` as an *overlay*, so no solver version is
duplicated between the two images and a version bump cannot silently skip the
GPU image.

## Usage

```bash
./run.sh --gpu --rebuild pytest -q -m gpu          # first time (builds the image)
./run.sh --gpu pytest -q -m gpu                    # subsequently
MOA_GPUS=device=3 ./run.sh --gpu python probe.py   # choose a device
MOA_GPUS=device=0,1 ./run.sh --gpu pytest -q -m gpu
```

`MOA_GPUS` defaults to `device=0`. Per the GPU server resource policy in
`AGENTS.md`, `run.sh` **rejects** `MOA_GPUS=all` and any selection naming more
than 2 devices (exit code 2) rather than silently clamping, and
`test_gpu_visibility_honors_the_two_device_project_cap` asserts the cap from
inside the container so a raw `docker run --gpus all` is caught too.

## Verified evidence

Recorded by `tests/test_gpu_environment.py::test_record_actual_devices_used` via
`record_property`, so every GPU validation run carries its own provenance
(`--junitxml`) instead of a hand-copied `nvidia-smi` line:

```
torch_version:      2.13.0+cu126
torch_cuda_version: 12.6
torch_devices:      NVIDIA RTX A6000
torch_capability:   (8, 6)            # sm_86
jax_version:        0.6.2
jax_backend:        gpu
jax_devices:        cuda:0
```

Host: 8x NVIDIA RTX A6000 (49 GB each), driver 550.163.01 (CUDA 12.4), docker
`nvidia` runtime present. Results:

- `./run.sh --gpu pytest -q -m gpu` — **8 passed**
- `./run.sh pytest -q -m "not slow and not benchmark and not fmmax and not fdtdx and not sax"`
  (Tier A, CPU image) — **561 passed, 27 skipped**; the 6 non-`sax` GPU tests
  auto-skip, matching the 561 passed measured before this change.

### Driver/CUDA compatibility

torch 2.13.0 publishes no cu124 wheel, so cu126 is the lowest available 12.x
build and it runs on this 12.4 driver through CUDA minor-version compatibility
(any CUDA 12.x build requires driver >= 525.60.13; `jax-cuda12-plugin` has the
same floor). Both are satisfied at 550.163.01. If the host driver is ever
downgraded below 525.60.13 this image stops working and the CPU image must be
used instead.

## Trap 1: a visible GPU that cannot execute anything

`jax-cuda12-plugin` alone is **not sufficient**, and the failure is badly
misleading. With the plugin installed but no PTX compiler:

```
jax.devices()  ->  [CudaDevice(id=0)]      # looks fine
jax.jit(jnp.sum)(x)  ->  XlaRuntimeError: UNAVAILABLE: No PTX compilation
    provider is available. Neither ptxas/nvlink nor nvjtlink is available.
```

Device *enumeration* needs only the driver and the CUDA runtime libraries, which
torch's cu126 wheels already provide — so a GPU-availability probe that stops at
`jax.devices()` reports success on an image that cannot compile a single kernel.
This is why `tests/test_gpu_environment.py` asserts on real kernel execution
(torch matmul and a jitted JAX reduction) rather than device presence.

The canonical fix is the `jax-cuda12-plugin[with-cuda]` extra (equivalently
`jax[cuda12]`), deliberately **not** used here: it pulls its own `nvidia-*`
CUDA library set and fights torch's pinned 12.6.x versions, with pip silently
picking one winner for libraries both frameworks load. Since torch already
installs cublas/cufft/nvrtc/nvjitlink at 12.6.x, only the compiler binaries were
missing, so `docker/requirements-gpu.txt` pins just `nvidia-cuda-nvcc-cu12`
(matching torch's 12.6.85 family). Its wheel unpacks to
`site-packages/nvidia/cuda_nvcc/bin/`, already one of the paths XLA searches, so
no `PATH` or env var is needed.

## Trap 2: importing SAX silently disables the GPU for the whole process

`klujax.py:47` executes, at **import** time:

```python
jax.config.update(name="jax_platform_name", val="cpu")
```

klujax is a hard dependency of SAX. So importing SAX — an out-of-milestone
solver that a graph or a pytest session may load for unrelated reasons — forces
every subsequent JAX computation in that process onto the CPU. It emits no
warning and never sets `JAX_PLATFORMS`; the only symptom is
`jax.default_backend() == 'cpu'` on a machine with a working GPU.

This was found the hard way: `pytest -m gpu` skipped everything (pytest imports
`tests/test_sax_adapter.py` during collection even though `-m gpu` deselects it)
while running `tests/test_gpu_environment.py` alone passed.

The pin is reversible, **but only before JAX initializes its backend** — the
backend is built once and cached, so a reset after the first `jax.devices()`
call has no effect. Both directions are pinned down by
`test_jax_platform_pin_is_reversible_only_before_backend_init`. This is why the
repair in `tests/conftest.py::undo_third_party_jax_platform_pin` runs inside
`pytest_collection_modifyitems` — after collection has done the offending
import, before any test touches JAX — and cannot be a fixture, which would run
too late. When the repair fires it prints

```
jax platform repair: undid klujax's jax_platform_name='cpu' pin (imported via SAX)
```

in the terminal summary, so mutating another package's global config is stated
out loud in the run it affects rather than applied invisibly.

### Why GPU tests require a dedicated session

Because that repair is process-global, it changes the backend *every* test in the
session computes on. So `gpu`-marked tests run only when the selection contains
nothing else, and skip otherwise. This is not caution in the abstract — both
failure modes were measured on the GPU image before the guard existed:

| Selection | Effect of repairing the pin |
| --- | --- |
| `-m sax` | `test_mzi_circuit_matches_analytic_oracle_and_probe_evidence` failed — klujax needs JAX on the CPU |
| Tier A | Chromatix moved onto the GPU; 2 tolerance-sensitive tests in `test_m3_pupil_to_focus.py` failed |

Skipping the GPU tests is the safe direction in both cases: the established CPU
results stay authoritative, and the GPU tests remain fully available on their
own. Measured consequence — Tier A is now byte-identical on both images
(561 passed, 29 skipped), and `-m gpu` passes 8/8 in the GPU image.

Note the second row is *not* itself evidence of a bug in Chromatix or in those
two tests: they were written against CPU float32 results and their tolerances
have never been re-derived for a GPU backend. Establishing dtype-appropriate
GPU tolerances is exactly CHE-60 Phase 5's job, and those two tests are the first
concrete candidates for it.

**This remains an open hazard outside the test harness.** Any production process
that imports SAX before Chromatix inherits the same silent CPU downgrade, and
nothing in the adapter layer currently detects it. CHE-60 Phase 4 must therefore
have the Chromatix adapter report the device its output **actually** landed on,
never the device that was requested — a requested-vs-actual mismatch is exactly
what this trap produces. `test_importing_sax_silently_pins_jax_to_cpu`
characterizes the upstream behavior so that if a future klujax drops the pin,
the test fails and the workaround can be deleted rather than lingering forever.

## Disk footprint

The GPU image is 9.98 GB. Building it took the host filesystem from 46 GB to
36 GB free (97% used). That is enough headroom for the current image but not for
many more large images; check `df -h /` before rebuilding, and note that
`docker system df` reported ~46 GB reclaimable in unused images at the time of
writing (not pruned here — this is a shared server).
