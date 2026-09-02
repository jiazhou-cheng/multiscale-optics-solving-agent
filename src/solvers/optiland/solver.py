"""The public trace entry points, and the one genuinely stateful thing in this package.

CHE-181 (R05.3). `trace(problem, sampling=..., execution=...) -> RayBundle`: one
neutral problem in, one neutral bundle out, and no facade class in between.

Two entry points, and why the second is not a parameter on the first
--------------------------------------------------------------------
CHE-217 (R05.6) added `trace_rays(problem, rays, *, execution=...) -> RayBundle`,
which traces a `RayBundle` the project already holds. The two differ in what the
rays *are*, not in how the trace runs: `trace` builds them inside the solver from
a normalized field coordinate and a hexapolar ring count, and `trace_rays`
consumes a representation declared at a boundary. Folding them into one function
behind an optional `rays=` would make `sampling=` mean nothing on half its calls
and would put a `sqrt(intensity)` and a regenerated pupil quadrature on the same
code path as a caller's own coefficients -- which is exactly the corruption
`rays.AMPLITUDE_SIDECAR_RULE` and `rays.SUPPLIED_RAY_SURVIVAL_RULE` exist to name.
`trace` is numerically untouched by the addition; its frozen ray parity holds.

What is not here
----------------
No adapter object reached through a `get_adapter()` singleton, no execution-state
class, no trace-plan class, and no request / result / failure triple. There is one
function; the request is its arguments and the result is its return value. If a
resource lifecycle ever genuinely needs an owner, that owner is R13's executor,
not a second one here.

Process-global solver state, and why it is set on every call
-----------------------------------------------------------
`optiland.backend.set_backend`, `set_precision` and `set_device` all mutate
module-level state and none is thread-safe. That is not an implementation detail:
process-global solver state was a **measured** source of nondeterminism. The
failure was a trace that every artifact described as NumPy while it actually
executed in torch, because one component in the process had selected the torch
backend and nothing at the boundary checked. A contiguous chunk of the traced
geometry came back at roughly 2^-34 absolute accuracy instead of float64,
nondeterministically and in a different chunk each time, and an exactness gate
failed about a third of its runs.

So `configure_execution` sets all three explicitly on every call, never inherits
what a previous call left behind, and **reads the resulting state back off the
solver** rather than echoing the request. A request and an observation are
different facts.

The capability gate runs before any solver call
-----------------------------------------------
A precision or device the measured capability table does not admit is refused
before optiland or torch is imported, not discovered inside a trace. The table is
`numerics.OPTILAND_CAPABILITIES`, which is probe-backed: `set_precision` accepts
only `float32` / `float64`, and `set_device` raises `BackendCapabilityError` on
the numpy backend, so CUDA is reachable **only** through torch. The one check that
cannot be import-free is CUDA availability, because importing torch is how that
question is answered; it still happens before the first solver call.

Gradients
---------
None are claimed. `DERIVATIVE` is `forward_only` and there is no knob to ask for
anything else: torch autograd is switched *off* explicitly on every configure, so
a caller cannot inherit a live graph from something else in the process. The one
design-parameter path the reference implementation characterized was a surface
radius under the torch backend, and re-exposing it would be a gradient claim
across a framework boundary that this rewrite has not validated.
"""

from __future__ import annotations

import math
from typing import Any, Literal, TypedDict

from numerics import (
    OPTILAND_CAPABILITIES,
    ArrayNamespace,
    DeviceKind,
    DevicePlacement,
    Precision,
    refusal,
)
from problems import RayTraceProblem
from representations import RayBundle
from solvers.optiland.rays import (
    NATIVE_WAVELENGTH_M,
    REFERENCE_SURFACES,
    SKIP_OBJECT_SURFACE,
    hexapolar_ray_count,
    require_launch_surface,
    to_native_rays,
    to_ray_bundle,
    to_traced_ray_bundle,
)
from solvers.optiland.system import build_lens

__all__ = [
    "CAPABILITIES",
    "DERIVATIVE",
    "Execution",
    "Sampling",
    "configure_execution",
    "trace",
    "trace_rays",
]

#: The row of `numerics.COMPONENT_CAPABILITIES` this solver executes within.
#: Cited by name rather than restated, so the declaration stays with the probe
#: that measured it and there is one place to widen.
CAPABILITIES = OPTILAND_CAPABILITIES.component

#: What may be claimed about differentiating through a trace. See the module
#: docstring: `forward_only`, with no argument that changes it.
DERIVATIVE = "forward_only"

#: Which array namespace drives Optiland for a given device, and the backend name
#: that selects it. One mapping, so the pairing cannot drift.
#:
#: The FP64 host runs in **numpy**, not torch, and that is the measured choice
#: rather than a preference: the project's representations hold NumPy or JAX
#: buffers, so selecting torch on the host meant Optiland converted every array on
#: entry and executed in a namespace no caller had chosen. Torch is selected only
#: where it is the sole namespace that can serve the request -- see
#: `_resolve_namespace`.
_NAMESPACE_BACKEND: dict[ArrayNamespace, str] = {
    ArrayNamespace.NUMPY: "numpy",
    ArrayNamespace.TORCH: "torch",
}


class Sampling(TypedDict, total=False):
    """How the pupil and the source are sampled for one trace.

    A `TypedDict` rather than a class: the fields share no invariant that has to
    be enforced across them, nothing subclasses it, and the only runtime check it
    needs -- refusing a key the solver does not recognize -- is a function,
    because a `TypedDict` is a static annotation that disappears at run time.
    Refusing unknown keys matters here for the same reason it matters in
    `problems.ray_trace`: the pinned solver silently discards keyword arguments it
    does not recognize, which turns a misspelling into a different trace.

    `num_rings`
        Required. The hexapolar ring count. This is a sampling **density**, not an
        output ray count: a fan of `n` rings is `1 + 3n(n + 1)` rays
        (`rays.hexapolar_ray_count`), and the solver spells the same argument
        `num_rays`.
    `reference_surface`
        Required, and deliberately without a default: `"image_surface"` or
        `"exit_pupil"`. The difference between them is the whole pupil-to-focus
        distance, so a caller states which one the bundle is declared on.
    `field_deg`
        The field angle as `(x_deg, y_deg)`, in the problem's own units. Defaults
        to on-axis. Converted to the solver's normalized field coordinates here,
        against the maximum field the constructed system declares -- which is the
        one place that normalization convention belongs.
    `wavelength_um`
        The wavelength to trace at, in micrometres. Defaults to the problem's
        primary wavelength. A free value rather than an index into
        `problem.wavelengths_um`, because the solver evaluates dispersion at
        whatever wavelength it is handed and tracing a prescription at a
        wavelength outside its declared set is a legitimate probe -- the frozen
        ray records do exactly that.
    """

    num_rings: int
    reference_surface: Literal["image_surface", "exit_pupil"]
    field_deg: tuple[float, float]
    wavelength_um: float


class Execution(TypedDict, total=False):
    """Where and in what precision the trace runs. Both required, neither inherited.

    A `TypedDict` for the same reasons as `Sampling`. Both keys are required
    because this is the process-global state the module docstring is about: a
    default here would be a value the caller did not choose, silently competing
    with whatever another component in the process last set.

    `device`
        `"cpu"`, `"cuda"` or `"cuda:<n>"`.
    `precision`
        `"fp32"` or `"fp64"` (a dtype spelling such as `"float64"` is also
        accepted -- `numerics.Precision.parse` owns that vocabulary).
    """

    device: str
    precision: str


#: The keys each argument accepts, and which are required. Checked rather than
#: trusted, because a `TypedDict` is erased at run time and
#: `{"num_rings": 16, "reference_surfce": "exit_pupil"}` is a perfectly good dict.
_ARGUMENT_KEYS: dict[str, tuple[frozenset[str], frozenset[str]]] = {
    # name: (required, optional)
    "sampling": (
        frozenset({"num_rings", "reference_surface"}),
        frozenset({"field_deg", "wavelength_um"}),
    ),
    "execution": (frozenset({"device", "precision"}), frozenset()),
}


def _require_keys(mapping: Any, *, name: str) -> None:
    """Refuse an argument that is misspelled, incomplete, or over-specified."""
    if not isinstance(mapping, dict):
        raise TypeError(f"{name}= must be a mapping, got {type(mapping).__name__}")
    required, optional = _ARGUMENT_KEYS[name]
    keys = set(mapping)
    missing = required - keys
    unknown = keys - required - optional
    if missing or unknown:
        detail = []
        if unknown:
            detail.append(
                f"does not take {sorted(unknown)} -- an unrecognized key is silently "
                "discarded by the pinned solver, which yields a different trace and no error"
            )
        if missing:
            detail.append(f"needs {sorted(missing)}")
        raise ValueError(f"{name}= " + "; and ".join(detail))


def _resolve_execution(execution: Execution) -> tuple[DevicePlacement, Precision, ArrayNamespace]:
    """Refuse an inadmissible request against the measured table, before any solver call.

    Every refusal here is a capability the probe did not measure, and every one is
    raised before optiland or torch enters the process -- except the CUDA
    availability check, which needs torch to answer at all and still precedes the
    first solver call.
    """
    device = DevicePlacement.parse(execution["device"])
    precision = Precision.parse(execution["precision"])

    if device.kind not in OPTILAND_CAPABILITIES.devices:
        raise refusal(
            code="UNSUPPORTED_DEVICE",
            component=CAPABILITIES,
            message=f"device {device} is outside the measured capability table.",
            requested=device,
            supported=sorted(str(kind) for kind in OPTILAND_CAPABILITIES.devices),
            evidence=OPTILAND_CAPABILITIES.evidence,
        )
    if precision not in OPTILAND_CAPABILITIES.precisions:
        raise refusal(
            code="UNSUPPORTED_DTYPE",
            component=CAPABILITIES,
            message=(
                f"precision {precision} has no measured execution path: "
                f"set_precision accepts only the real dtypes below, and there is no "
                f"{precision.real_dtype} mode to execute."
            ),
            requested=precision,
            supported=sorted(str(p) for p in OPTILAND_CAPABILITIES.precisions),
            evidence=OPTILAND_CAPABILITIES.evidence,
        )

    namespace = _resolve_namespace(device, precision)
    if device.kind is DeviceKind.CUDA:
        _require_cuda(device)
    return device, precision, namespace


def _resolve_namespace(device: DevicePlacement, precision: Precision) -> ArrayNamespace:
    """Which array namespace must drive the solver for this request.

    Two measured facts decide it, and neither is a preference:

    * **CUDA is torch-only.** `set_device` raises `BackendCapabilityError` on the
      numpy backend, so there is no other namespace that can reach a device. The
      capability table declares this already and it is read rather than re-derived.
    * **FP32 is torch-only too, on every device.** The numpy backend honours
      `set_precision('float32')` for its *own array constructor* while `Optic.trace`
      still returns float64. Measured on the reference singlet, independently
      re-measured for this rewrite: numpy/float32 returns float64 arrays whose
      direction norms are off by 9.19e-11 -- float32-scale error travelling inside
      float64 buffers, which is worse than either honest precision because nothing
      downstream can see it. Torch honours float32 end to end (2.12e-7, which is
      float32 round-off and inside `direction_norm_tolerance(float32)`).
      Recorded at `pre-rewrite-2026-08-30`, under
      `benchmarks/probes/records/optiland/b1_ray_device_precision.json`, row
      `backend=numpy, dtype=float32`.

    Everything else -- which is the default path, host FP64 -- runs in numpy.

    This coupling is a property of the solver rather than of the project's
    precision vocabulary, so it lives here and not in `numerics`. When CHE-206
    moves the concrete capability rows into `solvers/<backend>/`, this is the fact
    that row has to carry.
    """
    admissible = OPTILAND_CAPABILITIES.namespaces_for(device.kind)
    if ArrayNamespace.NUMPY in admissible and precision is not Precision.FP32:
        return ArrayNamespace.NUMPY
    if ArrayNamespace.TORCH not in admissible:  # pragma: no cover - the table admits it
        raise refusal(
            code="NAMESPACE_NOT_A_COMPUTE_NAMESPACE",
            component=CAPABILITIES,
            message=(
                f"{precision} on {device} needs the torch backend, which the capability "
                "table does not admit here."
            ),
            requested=device,
            supported=sorted(str(n) for n in admissible),
        )
    return ArrayNamespace.TORCH


def _require_cuda(device: DevicePlacement) -> None:
    """Refuse a CUDA request this install cannot serve. There is no host fallback.

    The default `agent_solver` image installs torch from the CPU-only wheel index,
    where `set_device('cuda')` either raises from inside torch or -- on a
    half-provisioned image -- succeeds and fails at the first kernel. Both are
    worse than a named refusal, and a silent fallback to the host would be worst
    of all: it reports success for a run that did not happen where the caller
    asked.
    """
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - torch is pinned in both images
        raise refusal(
            code="DEVICE_NOT_AVAILABLE",
            component=CAPABILITIES,
            message=f"a CUDA trace needs the torch backend, and torch is not importable ({exc}).",
            requested=device,
            supported=["cpu"],
        ) from exc
    if torch.version.cuda is None or "+cpu" in torch.__version__:
        raise refusal(
            code="DEVICE_NOT_AVAILABLE",
            component=CAPABILITIES,
            message=(
                f"torch is a CPU-only build ({torch.__version__}), so no CUDA device is "
                "reachable however many are attached to the host."
            ),
            requested=device,
            supported=["cpu"],
            remedy="Run in the CUDA image: `MOA_GPUS=device=6 ./run.sh --gpu ...`.",
        )
    if not torch.cuda.is_available():
        raise refusal(
            code="DEVICE_NOT_AVAILABLE",
            component=CAPABILITIES,
            message=(
                f"torch.cuda.is_available() is False (torch {torch.__version__}, "
                f"torch.version.cuda={torch.version.cuda!r}): no CUDA device is attached to "
                "this container."
            ),
            requested=device,
            supported=["cpu"],
            remedy="Attach one with `./run.sh --gpu`; there is deliberately no host fallback.",
        )
    if device.index is not None and device.index >= torch.cuda.device_count():
        raise refusal(
            code="DEVICE_ORDINAL_NOT_AVAILABLE",
            component=CAPABILITIES,
            message=(
                f"device ordinal {device.index} was requested but this container sees "
                f"{torch.cuda.device_count()} CUDA device(s). Note that the solver's own "
                "set_device takes 'cpu' or 'cuda' with no ordinal, so which physical GPU a "
                "trace lands on is decided by the container's visible device set -- an "
                "ordinal is only checkable, never selectable, from here."
            ),
            requested=device,
            supported=[f"cuda:{index}" for index in range(torch.cuda.device_count())],
        )


def configure_execution(
    *, device: DevicePlacement, precision: Precision, namespace: ArrayNamespace
) -> dict[str, Any]:
    """Drive the solver's global execution controls, and report the observed state.

    All three setters run on every call, in this order, so a previous call's
    choices in this process cannot leak in. Idempotent: calling it twice with the
    same arguments leaves the solver in the same state and returns the same
    report, which is what makes two consecutive traces bit-identical.

    The returned mapping keeps `requested` and `observed` apart. Nothing echoes:
    every `observed` value is read back from the solver.
    """
    import optiland.backend as be

    backend = _NAMESPACE_BACKEND[namespace]
    be.set_backend(backend)
    # The solver spells precision as a dtype name; this project spells it as a
    # policy. This is the one place the two vocabularies meet.
    be.set_precision(str(precision.real_dtype))

    if namespace is ArrayNamespace.TORCH:
        # `set_device` takes exactly "cpu" or "cuda" and has no ordinal -- it uses
        # whichever device torch considers current -- so the *kind* is what is set
        # and the physical GPU comes from the container's visible devices.
        be.set_device(str(device.kind))
        observed_device = str(be.get_device())
        # No gradient is claimed across this boundary, so autograd is switched off
        # rather than left at whatever another component in the process set. On the
        # numpy backend `be.grad_mode` itself raises: NumPy has no autodiff to
        # switch, which is an absence rather than a flag that happens to be off.
        be.grad_mode.disable()
    else:
        # `set_device` raises BackendCapabilityError on the numpy backend, so it is
        # not called at all rather than called and caught.
        observed_device = "cpu (the numpy backend has no device concept)"

    # `get_precision` returns an int width on the pinned install while
    # `set_precision` takes a dtype name. Normalized here so the report reads in
    # one vocabulary, and both spellings are accepted because that asymmetry is
    # exactly the kind of thing a minor release changes.
    observed_precision = be.get_precision()
    return {
        "requested": {
            "backend": backend,
            "device": str(device),
            "precision": str(precision.real_dtype),
        },
        "observed": {
            "backend": str(be.get_backend()),
            "device": observed_device,
            "precision": (
                f"float{int(observed_precision)}"
                if isinstance(observed_precision, int)
                else str(observed_precision)
            ),
        },
        "derivative": DERIVATIVE,
        "capabilities": CAPABILITIES,
    }


def _normalized_field(lens: Any, field_deg: tuple[float, float]) -> tuple[float, float]:
    """The solver's normalized field coordinates for a field angle in degrees.

    The solver normalizes an angular field by the largest field the system
    declares, so the conversion needs the constructed lens rather than the problem
    alone -- and it is done here, once, instead of at every call site.

    A system that declares only the axis has a maximum field of zero and no
    non-zero field is expressible on it. That is refused rather than silently
    normalized to the axis, which would trace a different field than the caller
    asked for and report success.
    """
    max_field = float(lens.fields.max_field)
    x_deg, y_deg = (float(value) for value in field_deg)
    if not (math.isfinite(x_deg) and math.isfinite(y_deg)):
        raise ValueError(f"sampling['field_deg']={field_deg!r} is not a finite (x, y) pair")
    if max_field == 0.0:
        if x_deg != 0.0 or y_deg != 0.0:
            raise ValueError(
                f"sampling['field_deg']={field_deg!r} was requested, but this problem "
                "declares only the on-axis field, so its maximum field angle is 0 deg and "
                "no other field is expressible. Add the field to the problem's "
                "`field_angles_deg` -- normalizing it to the axis would trace a different "
                "field than the one asked for."
            )
        return (0.0, 0.0)
    normalized = (x_deg / max_field, y_deg / max_field)
    if max(abs(value) for value in normalized) > 1.0:
        raise ValueError(
            f"sampling['field_deg']={field_deg!r} lies outside the largest field this "
            f"problem declares ({max_field} deg), so the solver's normalized field "
            f"coordinate would exceed 1: {normalized!r}"
        )
    return normalized


def trace(
    problem: RayTraceProblem, *, sampling: Sampling, execution: Execution
) -> RayBundle:
    """Trace `problem` and return the rays as a neutral `RayBundle`.

    The one entry point. Everything Optiland is happens between these lines: the
    argument is a `RayTraceProblem`, the return value is a `RayBundle` carrying
    `optical_path_m` with a declared reference, `amplitude`, `measure_weight` and a
    declared `measure_kind`, and no native ray state, native unit or solver type
    is observable in either.

    Raises:
        ValueError: an argument key is missing or unrecognized, a field is not
            expressible on this problem, or the requested precision or device is
            outside the measured capability table (the capability refusals carry a
            `code`). Raised before the solver is imported.
        ContractError: the trace ran but the result cannot be declared -- an
            unresolvable exit pupil, an unreadable image-space index, an
            undeclarable optical path, or every ray clipped.
        ImportError: optiland is not installed.
    """
    _require_keys(sampling, name="sampling")
    _require_keys(execution, name="execution")

    num_rings = int(sampling["num_rings"])
    if num_rings < 1:
        raise ValueError(
            f"sampling['num_rings']={num_rings!r} must be at least 1; it is a hexapolar "
            f"ring count, and n rings is {hexapolar_ray_count(1)} rays at n = 1"
        )
    reference_surface = str(sampling["reference_surface"])
    if reference_surface not in REFERENCE_SURFACES:
        raise ValueError(
            f"sampling['reference_surface']={reference_surface!r} is not one of "
            f"{list(REFERENCE_SURFACES)}"
        )
    wavelength_um = float(sampling.get("wavelength_um", problem.primary_wavelength_um))
    if not math.isfinite(wavelength_um) or wavelength_um <= 0.0:
        raise ValueError(
            f"sampling['wavelength_um']={wavelength_um!r} is not a finite positive "
            "wavelength in micrometres"
        )

    # The whole capability gate, before the solver enters the process.
    device, precision, namespace = _resolve_execution(execution)

    configure_execution(device=device, precision=precision, namespace=namespace)
    lens = build_lens(problem)
    requested_field = sampling.get("field_deg", (0.0, 0.0))
    field = _normalized_field(lens, (float(requested_field[0]), float(requested_field[1])))

    # `num_rays` is the solver's name for what this project calls a ring count.
    traced = lens.trace(
        Hx=field[0], Hy=field[1], wavelength=wavelength_um, num_rays=num_rings
    )
    bundle, _ = to_ray_bundle(
        lens,
        traced,
        field=field,
        wavelength_um=wavelength_um,
        num_rings=num_rings,
        reference_surface=reference_surface,
    )
    return bundle


def trace_rays(
    problem: RayTraceProblem, rays: RayBundle, *, execution: Execution
) -> RayBundle:
    """Trace an **externally supplied** `RayBundle` through `problem`'s system.

    The second entry point, and the one that makes a composed workflow expressible:
    `couplers.scalar_to_ray` produces a fully declared bundle, and this is what
    carries it through an optical system so `couplers.ray_to_scalar` can
    reconstruct on the other side. `trace` is unchanged and unaffected -- it
    *generates* its rays inside the solver from a field coordinate and a ring
    count, which is the right shape for a pupil benchmark and cannot express an
    importance-weighted ensemble drawn from a scalar grid's angular spectrum.

    There is deliberately no `sampling=` argument. Every quantity it carries is
    already a property of the bundle: the rays *are* the sampling, the wavelength
    is `rays.wavelength_m`, and no field angle, object distance or pupil-sampling
    parameter is constructed anywhere in this call path. A trace **consumes** a
    representation; it does not regenerate one from a higher-level source
    specification.

    What this operation may and may not do to the bundle
    ---------------------------------------------------
    It evolves the geometry and it extends the optical path. It does **not**
    restate the amplitude or the quadrature:

    * the complex amplitude is a sidecar. `|a|^2` crosses into the solver so the
      clipping bookkeeping is meaningful, and what comes back is read for exactly
      one purpose -- deciding which rays survived. `rays.AMPLITUDE_SIDECAR_RULE`;
    * `measure_weight` and `measure_kind` pass through untouched, for the reason
      `operators.propagate_rays` already states for propagation: the quadrature
      that fixed each plane wavelet's coefficient was taken at the surface where
      the rays were declared, and passing through a surface does not restate it;
    * the optical path is `incoming + accumulator`, carrying a reference that says
      so. `rays.compose_optical_path_m`;
    * survival keeps the row and zeroes the amplitude, so the output aligns with
      the caller's own arrays row for row. `rays.SUPPLIED_RAY_SURVIVAL_RULE`.

    The bundle must be **coherent** -- `RayBundle.require_coherent()` is the gate,
    and it settles the three cases that would otherwise need new vocabulary here:
    a bundle with no amplitude and a bundle with no optical path are both refused
    with `COHERENT_STATE_INCOMPLETE`, and one whose reference is `"unverified"` is
    refused with `OPL_REFERENCE_UNVERIFIED`. The last of those is the one worth
    naming: composing onto an unverified zero and then labelling the sum with this
    package's own version prefix would *launder* an unverified path into an
    admissible one, which is worse than either half.

    Parameters
    ----------
    problem
        The system to trace through. Its `field_angles_deg`, `object_distance_mm`
        and `entrance_pupil_diameter_mm` still describe the prescription -- the
        first two place the object surface this trace skips, and the third is the
        solver's pupil for ray *generation*, which this path does not use. No
        aperture is clipped by anything but the surface geometry: a physical clear
        aperture is R05.9's.
    rays
        The bundle to trace. It must declare itself on the first surface after the
        object surface, in that surface's medium; both are checked
        (`rays.require_launch_surface`).
    execution
        Device and precision, on `trace`'s contract and with the same capability
        gate, applied before the solver enters the process.

    Returns
    -------
    A `RayBundle` on the traced image surface, with the same row count, the same
    `wavelength_m`, `frame`, `measure_weight` and `measure_kind`, the caller's
    amplitude re-associated row for row, and a composed optical path.

    Raises
    ------
    TypeError, ValueError
        `execution=` is misspelled or incomplete, `rays` is not a `RayBundle`, or
        the requested precision or device is outside the measured capability
        table. Raised before the solver is imported.
    ContractError
        The bundle is not coherent, is not declared where the trace starts or in
        the medium it starts in, or no supplied ray survived.
    ImportError
        optiland is not installed.
    """
    if not isinstance(rays, RayBundle):
        raise TypeError(
            f"trace_rays takes a RayBundle as its second argument, got "
            f"{type(rays).__name__}. This entry point consumes a representation the "
            "project already holds; `trace` is the one that generates its own rays."
        )
    _require_keys(execution, name="execution")
    # Before anything else, and before the solver is imported: a bundle that
    # cannot be read as coherent has no amplitude to carry across and no path to
    # compose onto, and the refusal names every missing declaration at once.
    rays.require_coherent()

    # The whole capability gate, before the solver enters the process.
    device, precision, namespace = _resolve_execution(execution)

    configure_execution(device=device, precision=precision, namespace=namespace)
    lens = build_lens(problem)

    wavelength_um = rays.wavelength_m / NATIVE_WAVELENGTH_M
    # The launch surface and its medium, checked against the constructed system
    # rather than assumed from the skip count: getting this wrong is not a crash,
    # it is a silently different optical system.
    object_space_index = require_launch_surface(
        lens,
        rays.reference_surface,
        skip=SKIP_OBJECT_SURFACE,
        wavelength_um=wavelength_um,
    )

    native = to_native_rays(rays, namespace=namespace, device=device, precision=precision)
    traced = lens.surfaces.trace(native, skip=SKIP_OBJECT_SURFACE)
    bundle, _ = to_traced_ray_bundle(
        lens,
        traced,
        rays,
        launch_surface_z_m=rays.reference_surface.z_m,
        object_space_index=object_space_index,
        wavelength_um=wavelength_um,
    )
    return bundle
