# GPU execution environment (CHE-60 / PB4a, Phase 0)

Status: **complete and validated** on the target host. Established 2026-08-19
(CHE-60); revalidated 2026-08-20 from a rebuilt image by CHE-72/CHE-73, after
the SAX removal took Trap 2 out at the source.

This document records how GPU execution is provisioned, the evidence that it
actually works, and two non-obvious traps that silently disable it (the second
of which is now historical). It is the
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

### Trap 3: one faulted GPU blocks every GPU container on the host

Observed 2026-08-22, and the reason `--gpu` grew a second code path.

```
docker: Error response from daemon: ... error running prestart hook #0:
nvidia-container-cli: detection error: nvml error: unknown error
```

Nothing about the container is wrong. GPU 5 (`0000:B2:00.0`) on this host is in
a fault state at the NVML level — `nvidia-smi -L` lists the other seven and
fails on that one, and `nvidia-container-cli info` fails outright.
`nvidia-container-cli`'s prestart hook calls `nvmlInit` and enumerates **every**
GPU before deciding which to expose, so a single bad device fails the hook for
all of them, however healthy the one you asked for is. `MOA_GPUS=device=0`,
selecting by UUID, and `--gpus '"device=<uuid>"'` all fail identically, because
the failure precedes device selection.

Repairing it needs `nvidia-smi -r` or a reboot — root on a shared host, and
outside what this project may do.

So `run.sh --gpu` falls back. When `nvidia-container-cli info` fails, it binds
the requested `/dev/nvidia<N>` nodes plus `/dev/nvidiactl`, `/dev/nvidia-uvm`
and `/dev/nvidia-uvm-tools`, and bind-mounts the host's userspace driver
libraries, skipping enumeration entirely. Three properties make this safe rather
than a workaround that hides a problem:

* It is **narrower** than `--gpus`, not wider — only the devices `MOA_GPUS`
  names are visible, and the two-device cap is applied before it.
* It changes no host state, no driver, and no daemon configuration.
* It works unprivileged. The container runs as the invoking user, so it cannot
  create symlinks or run `ldconfig` the way the real hook does; each library is
  therefore bind-mounted **directly onto its SONAME**
  (`libcuda.so.$VERSION` → `libcuda.so.1`, `libnvidia-nvvm.so.$VERSION` →
  `libnvidia-nvvm.so.4`, and so on), with the version read from
  `/sys/module/nvidia/version` rather than hard-coded.

Control it with `MOA_GPU_PASSTHROUGH`: `auto` (default) detects,
`1` forces the fallback, `0` forbids it and keeps the stock `--gpus` path. Once
the host is repaired, nothing has to be un-done — auto-detection simply stops
choosing the fallback.

Evidence on device 0 through the fallback, 2026-08-22, driver 550.163.01:
`nvidia-smi -L` names the A6000; `jax.devices()` returns `[CudaDevice(id=0)]`;
a 2048² matmul returns the exact expected `8589934592.0`;
`torch.cuda.is_available()` is `True` with `device_count() == 1`; and
`./run.sh --gpu pytest -q -m gpu` is **48 passed, 769 deselected in 69.68 s**,
matching the CHE-72/CHE-73 figure of 48 passed in 70 s measured through the
stock path. The suite asserts real kernel execution, not device enumeration, so
that equality is a statement about compute, not about visibility.

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

- `./run.sh --gpu pytest -q -m gpu` — **8 passed** (2026-08-19, the 8 tests in
  `test_gpu_environment.py` as it then stood)
- `./run.sh pytest -q -m "not slow and not benchmark and not fmmax and not fdtdx and not sax"`
  (Tier A, CPU image, as run on 2026-08-19) — **561 passed, 27 skipped**; the 6
  non-`sax` GPU tests auto-skip, matching the 561 passed measured before that
  change. The `sax` marker no longer exists (CHE-72); the command still runs
  because an unknown name evaluates false in a `-m` expression.

### Driver/CUDA compatibility

torch 2.13.0 publishes no cu124 wheel, so cu126 is the lowest available 12.x
build and it runs on this 12.4 driver through CUDA minor-version compatibility
(any CUDA 12.x build requires driver >= 525.60.13; `jax-cuda12-plugin` has the
same floor). Both are satisfied at 550.163.01. If the host driver is ever
downgraded below 525.60.13 this image stops working and the CPU image must be
used instead.

## Revalidation from a rebuilt image (CHE-72 / CHE-73, 2026-08-20)

Both images were rebuilt from changed `docker/requirements.txt` (the SAX removal
dropped `sax`, `klujax`, `jaxellip`, `lark`, `natsort`, `orjson` and `scikit-rf`),
so this is a clean-dependency-state result rather than a re-run against the
image CHE-60 built.

| Check | Command | Result |
| --- | --- | --- |
| Dedicated GPU suite | `./run.sh --gpu pytest -q -m gpu` | **48 passed, 769 deselected, 69.6 s** on 1x RTX A6000 |
| Default suite, CPU image | `./run.sh pytest -q` | **769 passed, 48 skipped, 182 s** |
| Fast subset | `./run.sh pytest -q -m "not slow"` | 751 passed, 48 skipped, 18 deselected, 37 s |
| Legacy Tier A expression (still names the deleted `sax` marker) | `-m "not slow and not benchmark and not fmmax and not fdtdx and not sax"` | 799/817 collected — the marker is gone but the expression still parses |
| Agent benchmark gate | `./run.sh pytest -q benchmarks_agent` | 52 passed, 7.6 s |

Against the CHE-60 baseline of 770 passed / 48 skipped, the default suite moved by
exactly two accounted-for deltas and nothing else:

- **−1 passed**: `test_adapter_registry.py::test_discovered_adapter_spec_matches_model_id[M_CIRCUIT_SAX]`.
  That test is parametrized over discovered adapter modules, so deleting
  `sax_adapter.py` removes one case.
- **48 skipped, unchanged**: two SAX pin-characterization tests were deleted and
  two new `gpu` tests added (`test_a_clean_interpreter_reaches_the_gpu_with_no_platform_repair`,
  `test_one_coherent_cuda_dependency_family`).

### Clean-interpreter evidence

`benchmarks/probes/gpu/che72_gpu_revalidation.py`, run in a process pytest never
touches, because the old harness repair meant "works under pytest" and "works in a
clean interpreter" were different claims:

```json
{
  "sax_family_installed": [],
  "sax_importable": false,
  "klujax_importable": false,
  "jax_platform_name": "<unset>",
  "jax_devices": ["cuda:0"],
  "jax_default_backend": "gpu",
  "jax_jit_result": 1000000.0,
  "jax_jit_result_exact": true,
  "jax_result_device": "cuda:0",
  "torch_version": "2.13.0+cu126",
  "torch_cuda_version": "12.6",
  "torch_result_device": "cuda"
}
```

`jax_platform_name` reads `<unset>`: nothing pins it and nothing repairs it.

### Resolved CUDA family

One coherent stack, not two. Asserted by
`test_one_coherent_cuda_dependency_family`, which is the guard CHE-73 added
because nothing previously checked this:

| Package | Version |
| --- | --- |
| `nvidia-cuda-nvcc-cu12` | 12.6.85 |
| `nvidia-cuda-nvrtc-cu12` | 12.6.85 |
| `nvidia-nvjitlink-cu12` | 12.6.85 |
| `nvidia-cuda-runtime-cu12` | 12.6.77 |
| `nvidia-cuda-cupti-cu12` | 12.6.80 |
| `nvidia-nvtx-cu12` | 12.6.77 |
| `nvidia-cublas-cu12` | 12.6.4.1 |
| `torch` | 2.13.0+cu126 (`torch.version.cuda == "12.6"`) |

The compiler/JIT trio (`nvcc`/`nvrtc`/`nvjitlink`) are byte-identical at 12.6.85,
which is the specific coherence that matters for the PTX stage. The math libraries
(`cufft` 11.3.0.4, `cusolver` 11.7.1.2, `cusparse` 12.5.4.2) version on their own
schedules and are deliberately *not* asserted against 12.6 — doing so would be a
false claim. `nvidia-cuda-nvcc-cu12` remains the only CUDA package added for JAX;
`jax[cuda12]` / `jax-cuda12-plugin[with-cuda]` are still not used, and the minimal
strategy was not shown to be insufficient.

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

## Trap 2 (resolved at the source): a process-global JAX platform pin

**Status: no longer live.** CHE-72 removed the dependency that caused it, and the
test harness no longer repairs anything. Kept here because the *state* is still
reachable by other means and is genuinely hard to diagnose.

`klujax.py:47` executed, at **import** time:

```python
jax.config.update(name="jax_platform_name", val="cpu")
```

klujax was a hard dependency of SAX, so importing SAX — an out-of-milestone
solver that a graph or a pytest session could load for unrelated reasons — forced
every subsequent JAX computation in that process onto the CPU. It emitted no
warning and never set `JAX_PLATFORMS`; the only symptom was
`jax.default_backend() == 'cpu'` on a machine with a working GPU.

This was found the hard way: `pytest -m gpu` skipped everything (pytest imported
`tests/test_sax_adapter.py` during collection even though `-m gpu` deselects it)
while running `tests/test_gpu_environment.py` alone passed.

CHE-72 deleted the SAX integration outright — adapter, tests, knowledge pack,
registry entry, and the `klujax`/`sax` pins — rather than keeping the harness
workaround that undid the pin during collection. The pin was reversible, but only
before JAX initialized its backend (the backend is built once and cached), which
forced the repair into `pytest_collection_modifyitems` and made it impossible to
express as a fixture. Removing the cause removed all of that.

What replaced it:

- `tests/test_gpu_environment.py::test_a_clean_interpreter_reaches_the_gpu_with_no_platform_repair`
  asserts the property directly, in a subprocess that imports the project
  package: `jax_platform_name` is not pinned, the default backend is the GPU, and
  a jitted kernel's *output array* lands on a GPU device. A future dependency
  with the same import-time habit fails this test.
- The Chromatix adapter still names a `jax_platform_name` pin in
  `CHROMATIX_CUDA_UNAVAILABLE` when it finds one, because an env var or another
  package can reach the same state. It no longer attributes it to SAX.
- The requested-vs-actual device rule (`core/arrays.py`, PB4b) is unchanged. It
  was motivated by this trap but never depended on it: a requested device is not
  evidence of an actual one regardless of what caused the divergence.

### Why GPU tests still require a dedicated session

The reason changed with CHE-72, so read this rather than assuming the old one.

It is no longer "the harness mutates global JAX state." Nothing is mutated now.
On the GPU image JAX simply computes on the GPU for the whole process — there is
no per-test backend — so a mixed selection would silently move the non-GPU tests
onto a backend their tolerances were never derived for. Measured on the GPU image
before the guard existed: running Tier A there moved Chromatix onto the GPU and
broke two tolerance-sensitive tests in `test_m3_pupil_to_focus.py` (archived by
CHE-67; the measurement stands).

That is *not* itself evidence of a bug in Chromatix or in those two tests: they
were written against CPU float32 results and their tolerances have never been
re-derived for a GPU backend. Establishing dtype-appropriate GPU tolerances
remains open work. Until then the established CPU results stay authoritative, so
`gpu`-marked tests run only when the selection contains nothing else and skip
otherwise — which is also what keeps every documented tier command green
unchanged on both images.

## Disk footprint

The GPU image is 9.98 GB. Building it took the host filesystem from 46 GB to
36 GB free (97% used). That is enough headroom for the current image but not for
many more large images; check `df -h /` before rebuilding, and note that
`docker system df` reported ~46 GB reclaimable in unused images at the time of
writing (not pruned here — this is a shared server).
