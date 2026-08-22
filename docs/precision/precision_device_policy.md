# Precision, dtype and device policy

CHE-61 (PB4b). Successor to CHE-55/PB4's rejection gates and CHE-60/PB4a's GPU
environment. Everything below is executable: the capability table is generated
from `core/capabilities.py`, and every number is measured on the pinned installs
in `agent_solver` / `agent_solver_gpu` (host: 8× RTX A6000, torch 2.13.0+cu126,
jax 0.6.2).

## Four separate concepts

Before PB4b these were one string, `config["dtype"] == "float64"` alongside
`config["device"] == "cpu"`, compared against a per-adapter constant. That works
only while every component supports the same combination, which PB4a measured to
be false.

| Concept | What it is | Where it lives |
| --- | --- | --- |
| **Precision** | An execution/accuracy *policy*: FP16 / FP32 / FP64. An intent, not a storage format. | `Precision` |
| **DType** | An *observed* storage property of a real array. Read off the buffer, never inferred from a request. | `DType` |
| **Device** | Where the array physically is, including the ordinal (`cuda:0`). | `DevicePlacement` |
| **Namespace** | Which array ecosystem owns the buffer: NumPy / JAX / Torch. Orthogonal to device. | `ArrayNamespace` |

`precision` and `dtype` are not interchangeable. FP32 means `float32` for real
quantities *and* `complex64` for a field. `complex64` is FP32, not FP64: it
stores two float32 components, so its accuracy is float32 accuracy.

FP16 has **no complex dtype** in this vocabulary and that is deliberate. NumPy,
JAX and Torch all lack a first-class `complex32`; inventing one so the table
looks symmetrical would be exactly the kind of claim this document exists to
prevent.

Namespace matters separately from device because "GPU" is not enough information
to execute anything. NumPy cannot leave the host; JAX and Torch can be on
either. PB4a's platform-pin hazard is precisely the case where the namespace is
right and the device is silently wrong.

## Separation of responsibility

> **Backend capabilities** determine what a package can execute.
> **Artifact state** describes what the current data actually is.
> **Bridge negotiation** determines how that artifact may legally enter the next
> component.

A source backend never decides a destination backend's precision. No conversion,
transfer or namespace change happens without appearing in a `BridgePlan`.

```
ExecutionRequest  ->  ComponentCapabilities.resolve  ->  ResolvedExecution
                                                              |
Source artifact (observed dtype/device/namespace) -------------+
                                |
                          plan_bridge(source, target, policy)
                                |
                            BridgePlan   dtype? compute dtype? device transfer?
                                |        namespace? lossy? reason?
                          Coupler / target backend
                                |
                        Observed output (read off the array)
                                |
                     Provenance: requested vs resolved vs actual
```

## Component capability table

Generated from `core/capabilities.py` by `benchmarks/probes/precision/capability_table.py`.
`tests/test_registry_matches_capabilities.py` fails if `registry/models.yaml` or
`registry/couplers.yaml` disagrees with it.

| Component        | Devices   | Precisions | Input accepted                          | Native compute                          | Output                                  | Ingestible but lossy | Namespaces   | Compute floor |
|------------------|-----------|------------|-----------------------------------------|-----------------------------------------|-----------------------------------------|----------------------|--------------|---------------|
| M_RAY_OPTILAND   | cpu, cuda | fp32, fp64 | float32, float64                        | float32, float64                        | float32, float64                        | --                   | numpy, torch | fp32          |
| C_RAY_TO_WAVE    | cpu, cuda | fp32, fp64 | complex128, complex64, float32, float64 | complex128, complex64, float32, float64 | complex128, complex64                   | --                   | jax, numpy   | fp32          |
| M_WAVE_CHROMATIX | cpu, cuda | fp32       | complex64                               | complex64                               | complex64                               | complex128           | jax          | fp32          |
| C_WAVE_TO_RAY    | cpu, cuda | fp32, fp64 | complex128, complex64                   | complex128, complex64, float32, float64 | complex128, complex64, float32, float64 | --                   | jax, numpy   | fp32          |
| C_PLANAR_DOE_STEP | cpu, cuda | fp32, fp64 | complex128, complex64, float32, float64 | complex128, complex64, float32, float64 | complex128, complex64, float32, float64 | --                   | jax, numpy   | fp32          |

`C_PLANAR_DOE_STEP` is the one composed row, and it is worth reading as such:
its capability is `C_RAY_TO_WAVE`'s and `C_WAVE_TO_RAY`'s **intersected**, not an
independent measurement. The step accumulates through the first and resamples
through the second, so anything either refuses is refused here, and the
intermediate DOE multiply introduces no cast of its own. A composed component
whose row was *wider* than its parts' would be the thing to distrust.

Four dtype columns rather than one, because they are different questions and
collapsing them is how "supports float16" comes to mean "will not crash if
handed float16":

- **Input accepted** — what may cross the boundary inward.
- **Native compute** — what the component actually computes in. This is the
  honest answer to "does it support precision X".
- **Output** — what comes back out. Need not match either.
- **Ingestible but lossy** — dtypes the component will physically swallow only by
  throwing precision away. Kept *out* of the accepted set on purpose, so the
  bridge refuses them under SAFE.

Device-to-namespace constraints:

| Component | CPU | CUDA |
| --- | --- | --- |
| M_RAY_OPTILAND | numpy, torch | **torch only** — `set_device` raises `BackendCapabilityError` on the numpy backend |
| C_RAY_TO_WAVE | numpy, jax | **jax only** — NumPy cannot hold device memory |
| M_WAVE_CHROMATIX | jax | jax |
| C_WAVE_TO_RAY | numpy, jax | **jax only** |

## What each package actually supports, and how we know

### Optiland 0.6.0

`set_precision` is typed `Literal['float32','float64']` and raises
`ValueError("Precision must be 'float32' or 'float64'.")` otherwise. `set_device`
raises `BackendCapabilityError` on the numpy backend. With
`set_backend('torch'); set_device('cuda')`, arrays land on `cuda:0` at the
selected precision. Evidence: `benchmarks/probes/precision/optiland_capability.py`.

**There is no float16 path.** FP16 is refused, not promoted. Geometry, OPL and
direction cosines all accumulate; float16 carries ~3 decimal digits and an OPL
here spans ~1e4 waves.

**The torch backend defaults to float32; the numpy backend defaults to float64.**
`be.get_precision()` returns 32 after `set_backend('torch')` and 64 after
`set_backend('numpy')` (`benchmarks/probes/precision/default_precision.py`). Before CHE-61
the adapter never called `set_precision` while reporting `dtype: 'float64'`, so
**every torch-backend run traced in float32 under a float64 label**. The recorded
gradient probe in `benchmarks/probes/records/optiland/gradient_probe.json` is
therefore the float32 path; `config['dtype']='float32'` reproduces it
bit-identically, and the float64 default now genuinely runs float64, differing by
1.3e-05 relative on the objective and 2.3e-06 on the gradient
(`benchmarks/probes/precision/grad_precision.py`).

### Chromatix 0.6.0 @ d24bdf0

`ScalarField.__init__` is `jnp.asarray(u, dtype=jnp.complex64)`
**unconditionally**. Handing `Field.build` a `complex128` array *with*
`jax_enable_x64=True` still yields `field.u.dtype == complex64`
(`benchmarks/probes/precision/chromatix_capability.py`).

**There is no complex128 path at any device**, so the project does not claim one.
An FP64 request is refused on precision grounds, not device grounds. A
`complex128` input array is ingested through an explicitly recorded lossy bridge
(the adapter's own input port defaults to `ALLOW_DOWNCAST`; see below).

`asm_propagate` runs on `cuda:0` and returns `complex64` on `cuda:0`.

**Consequence for the reference path:** there is no uniform FP64 reference for
the whole stack. Optiland and both couplers reach float64/complex128; the
Chromatix leg's accepted reference is `complex64` on the CPU. Asserting a
uniform FP64 reference would assert something the stack cannot provide.

## Bridge policy

`plan_bridge(source_state, target_capabilities, policy=...)` is pure,
deterministic and unit-tested without any solver
(`tests/test_precision_contract.py`).

| Policy | Widening | Narrowing |
| --- | --- | --- |
| `STRICT` | rejected | rejected |
| `SAFE` (default) | allowed, minimum necessary | **rejected** |
| `ALLOW_DOWNCAST` | allowed | allowed, recorded `lossy=true` |

Rules, in the order they apply:

- **A — preserve a compatible representation.** No `float32 -> float64 ->
  float32` and no `GPU -> CPU -> GPU` for convenience.
- **B — minimum necessary safe promotion.** `float16 -> float32`, not
  `float16 -> float64`.
- **C — no silent precision loss.** `float64` into a float32-only target is
  refused under SAFE.
- **D — no silent device transfer.** Residency is preserved when both sides
  support the device. A transfer requires `allow_device_transfer=True`, which is
  a *separate* argument from `policy` because a CPU↔GPU copy is not a precision
  question. Without it, "target cannot reach this device" is a structured
  failure, never an implicit host fallback.
- **E — no silent namespace conversion.** JAX/Torch/NumPy conversion is a
  boundary operation. Host transfer, graph break, copy and device
  synchronization are each recorded in the plan.

### The one documented policy exception

The Chromatix adapter's own input port defaults to `ALLOW_DOWNCAST` rather than
the project-wide `SAFE`. This is narrow and has history behind it: CHE-35
established that a `complex128` input is truncated *inside*
`ScalarField.__init__` whatever the adapter does, and chose to **measure** the
loss rather than pretend it did not happen. Refusing the input would not prevent
any truncation, only remove the measurement. CHE-61 adds the recorded
`BridgePlan` with `lossy=true`; `config['bridge_policy']='safe'` makes it refuse
instead.

## Serialization is not a fallback

Execution representation and persisted `ArtifactRecord` representation are not
the same thing. `.npy`/`.npz` are host formats, so a GPU artifact must be copied
to the host **to be written**. That copy:

- happens at the moment of writing, not before;
- leaves the live artifact on its device;
- is declared in `metadata['serialization']` with
  `kind: "serialization"` (vs `"already_on_host"`), the reason, and — for
  Optiland — the mechanism, because `to_numpy` is a host transfer *and* a
  `.detach()` and a reader has to know both happened.

`metadata['execution']` alongside it records where the computation actually
occurred.

**Known limitation, stated rather than papered over:** the graph runtime passes
`ArtifactRecord`s, and a record is a file. So the record-mediated path
host-copies once per boundary regardless of device. Removing those copies needs
an in-memory artifact-passing mechanism for the graph runtime, which PB4b does
not attempt. What PB4b does hold is the *live* boundary — `bundle ->
C_RAY_TO_WAVE -> field -> Chromatix's representation` — which keeps GPU residency
end to end (`tests/test_precision_gpu_pipeline.py::TestLiveBoundaryAndEndToEnd::test_the_live_coupler_boundary_never_leaves_the_device`).

## Requested vs resolved vs actual

Three different facts, never conflated. `metadata['execution']` and adapter
diagnostics carry all three; the third is always read off the array that was
produced.

PB4a's hazard is why. A process-global `jax_platform_name='cpu'` silently moves
every later JAX computation onto the host, with no warning and `JAX_PLATFORMS`
never set, so any code writing `record.device = requested_device` would report GPU
execution that never happened. The Chromatix adapter reads `field_out.u`'s own
placement and sets `metadata['execution']['device_mismatch']`, warning explicitly
and naming the pin as the likely cause.

PB4a found this through a dependency that set the pin at *import* time
(`klujax.py:47`, a SAX dependency). CHE-72 removed SAX and klujax entirely, so
that particular route is gone — but the policy is unchanged and not contingent on
it: an env var, another package, or a missing CUDA plugin reaches the same state,
and "requested" was never evidence of "actual" in the first place.

The GPU availability probe compiles and runs a kernel rather than trusting
`jax.devices()`, because PB4a measured an image where enumeration returned
`[CudaDevice(id=0)]` while the first jitted call died with "No PTX compilation
provider is available".

## Two silent precision losses found by measurement

Both were found by holding the GPU to the CPU-derived tolerance and refusing to
loosen it. Both are fixed rather than tolerated.

### 1. JAX drops 64-bit requests without erroring

With `jax_enable_x64` disabled — the state the Chromatix adapter *enforces* on
every call, and the process-global default — `astype(float64)` returns
`float32` with only a `UserWarning`. `core.arrays.verify_dtype` checks the dtype
that came back rather than the one that was asked for, and raises
`SILENT_DTYPE_DOWNCAST` naming the config option.

### 2. XLA:GPU computes complex64 matmuls in TF32

On Ampere, XLA's default precision for an `f32`/`c64` dot is **TF32**: a 10-bit
mantissa rather than 24. Measured for the coupler's
`einsum("n,ny,nx->yx", ...)` over 256 complex64 wavelets, against a complex128
reference (`benchmarks/probes/precision/gpu_matmul.py`):

| path | relative error |
| --- | --- |
| NumPy complex64, host | 2.6e-07 |
| JAX complex64, GPU default | **3.5e-04** |
| JAX complex64, GPU, `precision="highest"` | 2.3e-07 |

So "complex64 on the GPU" was 1500× less accurate than complex64, silently, with
the array still reporting `dtype=complex64`. Before the fix the GPU round trip
missed the CPU-derived bound by 170×; a looser tolerance would have "passed" it
and buried the loss in a constant. `core.arrays.matmul_precision_kwargs` requests
`highest` for JAX dot products and returns nothing for NumPy, which needs no such
knob — one operation, one "and mean it" flag that differs by namespace.

## Measured tolerances

Never chosen. Sources: `benchmarks/probes/precision/tolerance.py` (host),
`benchmarks/probes/precision/tolerance_gpu.py` (device). 16×16 grid, 500 nm, 1 µm pitch,
errors relative to peak against a float64 reference.

| path | max error | reference |
| --- | --- | --- |
| NumPy float64 → complex128 | 3.5e-15 | analytic `exp(i k d·r)` |
| NumPy float32 → complex64 | 2.1e-06 | analytic |
| JAX cpu float32 → complex64 | 4.0e-06 | analytic |
| JAX gpu float32 → complex64 | 1.9e-06 | analytic |
| round trip complex128, host | 1.6e-15 (L2) | the input field |
| round trip complex64, host | 8.3e-07 (L2) | the input field |
| round trip complex64, gpu | 8.3e-07 (L2) | the input field |

The enforced FP32 bound is **2e-5**, five times the worst measured value —
headroom for a different BLAS or FFT ordering, not for a precision bug. Note the
GPU column shows no penalty: the same bound holds on host and device.

### Tolerances derived from dtype rather than fixed

Three bounds that were float64 absolutes are now scaled to the precision of the
data they judge. Each keeps its historical value for float64 exactly, so nothing
established regresses:

| Check | Old | New | Why |
| --- | --- | --- | --- |
| `RayBundle` direction unit norm | `1e-9` absolute | `max(1e-9, 64·eps(dtype))` | Casting an exactly-normalized float64 direction to float32 perturbs `\|d\|` by ~1 float32 eps *before* the norm is computed. |
| Optiland export direction norm | `1e-12` absolute | `max(1e-12, 64·eps(dtype))` | Optiland normalizes in float32 on that path; the rays arrive already off by a few eps. |
| Handoff plane agreement | `1e-12` m absolute | `max(1e-12, 64·eps(dtype)·\|z\|)` | See below. |

The plane bound is the one that needed a physical argument, not just an
arithmetic one. Its declared premise was "both come from the same float64
protocol literals, so any real disagreement is a modelling error, not
round-off" — exactly true for float64 and exactly false for float32, where the
exit pupil comes out of `Paraxial.XPL()` evaluated in float32. Measured on the M3
singlet: the float32 pupil lands 9.2e-11 m from the float64 value, ~11 float32
epsilons on a 6.8e-5 m coordinate.

The quantity being bounded is a **defocus**, with wavefront error
`~ (2π/λ)·dz·NA²/2`. At 550 nm and the M3 singlet's NA, 5.2e-10 m of offset is of
order 1e-4 rad — four orders below the 2π that would matter, five below the
Rayleigh quarter wave. Meanwhile a genuine plane mismatch is a pupil-to-focus
distance, i.e. millimetres. Nothing the widened bound admits is anywhere near
what the check exists to catch, and refusing it would refuse float32 tracing
altogether.

## Per-subsystem verdict

| Quantity | float16 | float32 / complex64 | float64 / complex128 |
| --- | --- | --- | --- |
| Ray positions, directions | **refused** (no Optiland path) | validated, 1e-7 relative | validated, reference |
| OPL accumulation | **refused** | validated; dominates the reduced-precision end-to-end difference | validated, reference |
| Object-space OPL reference | n/a | **deliberately float64 regardless of trace precision** — the correction is ~1e4 waves, so computing it in float32 would inject more error than the wavefront it corrects | reference |
| Hexapolar ring-index assignment | n/a | **deliberately float64** — a tolerance test on a ratio | reference |
| Coupler phase accumulation | promoted to float32 | validated, 2e-6 | validated, 3.5e-15 |
| Complex field / FFT | n/a | validated on host and GPU | validated (couplers only) |
| Chromatix propagation | n/a | **the only option** | **does not exist** |
| PSF intensity | n/a | float32, follows the field | float64, follows the field |

The objective is not lowest-possible precision everywhere. It is lower precision
where it buys speed or memory without unacceptable degradation, stated per
subsystem, with the two float64-by-declaration exceptions above called out at
their call sites rather than inherited silently.

## Failure codes

Every unsupported combination fails early with a structured, actionable reason,
before any solver call. Never a late framework traceback where the answer was
knowable in advance.

| Code | Meaning |
| --- | --- |
| `UNSUPPORTED_PRECISION` | The component has no native compute dtype for this precision family. |
| `UNSUPPORTED_DEVICE` | The component does not execute on this device kind. |
| `UNSUPPORTED_NAMESPACE_FOR_DEVICE` | The namespace cannot drive that device for this component (Optiland numpy + CUDA). |
| `OPTILAND_CUDA_UNAVAILABLE` | Well-formed CUDA request, no device in *this container*. Distinct code from a malformed one. |
| `CHROMATIX_CUDA_UNAVAILABLE` | Same, plus a `jax_platform_name` pin named when it is the cause. |
| `LOSSY_DOWNCAST_REQUIRED` | SAFE refuses a narrowing conversion. Names `allow_downcast` as the deliberate opt-in. |
| `STRICT_REPRESENTATION_MISMATCH` | STRICT refuses any implicit dtype conversion. |
| `STRICT_NAMESPACE_MISMATCH` | STRICT refuses a namespace conversion. |
| `DEVICE_INCOMPATIBLE` | GPU source, CPU-only target, transfer not authorized. |
| `DEVICE_TRANSFER_NOT_PERMITTED` | A transfer was needed and `allow_device_transfer` was not set. |
| `NO_COMPATIBLE_DTYPE_KIND` | Real data into a complex-only target, or vice versa. Not a precision question. |
| `SILENT_DTYPE_DOWNCAST` | A cast did not land; the requested precision is unavailable here. |
| `REPRESENTATION_INCONSISTENT` | One artifact spanning two devices or two array ecosystems. |
| `NUMPY_CANNOT_LEAVE_HOST` | A CUDA placement requested for a NumPy target. |

## Verification commands

```bash
./run.sh pytest -q -m optiland
./run.sh pytest -q -m chromatix
./run.sh pytest -q -m coupler
./run.sh --gpu pytest -q -m gpu          # dedicated session; see below
./run.sh pytest -q -m "not slow and not benchmark and not fmmax and not fdtdx and not sax"
```

GPU tests need a dedicated session because on the GPU image every computation in
the process runs on the GPU, and the non-GPU tests' tolerances were derived on the
CPU. `conftest.py` skips `gpu`-marked tests whenever anything else is selected
with them, which is what keeps every other tier command green unchanged. See
`docs/testing/gpu_environment.md`. (Before CHE-72 this was justified by having to
undo klujax's platform pin; the harness no longer repairs anything.)

## Non-goals of PB4b

No new optical physics; no change to the ray-wave coupling equations; no separate
CPU and GPU physics implementations; no fake FP16 in Optiland; no fake
complex128 in Chromatix; no mixed-precision performance heuristics beyond the
explicit bridge/compute policy needed for correctness; no per-kernel GPU tuning;
no in-memory artifact passing for the graph runtime.
