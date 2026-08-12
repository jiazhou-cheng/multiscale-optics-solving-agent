"""Adapter for ``M_RCWA_FMMAX`` — FMMAX (Fourier Modal Method / RCWA), pinned ``fmmax==1.7.1``.

Scope of this adapter (deliberately narrow; see ``knowledge/solvers/fmmax/``):

* A single periodic unit cell: ambient / (optional patterned interior layers) /
  substrate, one wavelength, **normal incidence only**
  (``polar_angle_rad == azimuthal_angle_rad == 0``).
* Only the fundamental-order, index-0 excitation mode (documented below as
  "te" in the request metadata) is used as the incident wave. Requesting any
  other polarization label raises ``UnsupportedCapabilityError`` because the
  mapping from FMMAX's mode-basis ordering to a physical TM label has not
  been independently verified in this repository (see
  ``knowledge/solvers/fmmax/conventions.md``); only the index-0
  (order ``(0, 0)``) mode has been checked against the analytic Fresnel
  oracle (``knowledge/solvers/fmmax/probes/fresnel_oracle_probe.py``).
* Only ``material_convention == "permittivity"`` (real or complex relative
  permittivity, isotropic) is supported. Anisotropic media are out of scope.
* Only ``truncation == "circular"`` and ``formulation == "fft"`` are
  supported — these are exactly what the pinned oracle/gradient probes used;
  changing either is an unverified fidelity change and is rejected eagerly.
* Length quantities (``period``, ``wavelength_m``, layer ``thickness_m``)
  must be given in **meters** (repository scientific conventions, SI units). FMMAX itself
  is scale-invariant (``c = eps0 = mu0 = 1``, see
  ``knowledge/solvers/fmmax/conventions.md``) and only requires that all
  length-like inputs share one consistent scale; using meters everywhere
  satisfies that requirement without any implicit unit assumption.

Non-standard scattering-matrix labeling (repository scientific-contract requirements)
---------------------------------------------------------------------
Per ``inspect.getdoc(fmmax.ScatteringMatrix)`` (echoed in
``knowledge/solvers/fmmax/conventions.md``), following [1999 Whittaker]::

    a_N = s11 @ a_0 + s12 @ b_N
    b_0 = s21 @ a_0 + s22 @ b_N

so ``s11`` is **transmission** (forward-going at start -> forward-going at
end) and ``s21`` is **reflection** (forward-going at start -> backward-going
at start). This is the *opposite* of the common RF/photonic-circuit
convention. The swap is applied explicitly below, at the point where
``a_N``/``b_0`` are formed, with an inline comment citing this docstring
section — never assume ``s11`` means reflection.

Energy-conservation / power normalization
------------------------------------------
Raw scattering-matrix amplitudes are **not** physical power (a bare-interface
check with a naive ``|s11|^2`` term gave ``> 1``, see
``knowledge/solvers/fmmax/failure_guide.md``). This adapter instead uses
``fmmax.directional_poynting_flux`` on the physical amplitude vectors
(``a_0``, ``b_0`` at the start layer; ``a_N``, ``b_N`` at the end layer) to
obtain time-averaged Poynting flux, and forms::

    R = -sum(bwd_start.real) / fwd_start[0]
    T =  sum(fwd_end.real)   / fwd_start[0]

``fwd_start[0]`` is the incident flux (the only nonzero entry, since the
incident vector ``a_0`` is a one-hot excitation of mode index 0); the sums
run over *all* diffraction-order/polarization modes because a real grating
scatters power into more than one channel. This was independently checked
against two cases while building this adapter (not previously exercised in
the knowledge pack): a bare interface (``R + T`` closes to
``0.99999976``, i.e. ~2.4e-7 residual) and a small binary lamellar grating
with 9 Fourier orders / 18 modes (``R + T`` closes to ``0.9999998584...``,
~1.4e-7 residual). Only the *magnitude* (power) balance has been checked;
the complex reflection amplitude's *sign/phase* does not match the textbook
Fresnel convention (see ``knowledge/solvers/fmmax/conventions.md``) and is
NOT reconciled by this adapter.

Failure-reporting convention (repository scientific-contract requirements / ``core/errors.py``)
-------------------------------------------------------------------------------
* ``AdapterDependencyError`` — raised by :func:`_import_fmmax` if ``fmmax``
  or ``jax`` cannot be imported.
* ``UnsupportedCapabilityError`` — raised eagerly by :meth:`FmmaxAdapter.run`
  *before* any solver call, whenever :meth:`FmmaxAdapter.validate_request`
  reports a structural error (missing metadata, unsupported polarization,
  oblique incidence, anisotropic media, non-"fft"/"circular" numerics, a
  patterned ambient/substrate layer, etc.).
* Genuine solver-execution-time failures (e.g. a degenerate/non-convergent
  Fourier truncation that yields zero modes and cannot support an incident
  excitation vector, or any other exception raised while calling into
  ``fmmax``/``jax``) are caught at the ``run()`` boundary and reported as
  ``ModelRunResult(status=RunStatus.FAILED, error_type=..., error_message=...)``
  rather than as a raised ``SolverExecutionError``, per the option offered in
  ``adapters/base.py`` and the repository failure contract.

Derivative semantics and the "no silent detach" boundary (repository gradient policy)
---------------------------------------------------------------------------------
The physics core of this adapter (lattice/expansion setup, eigensolve,
scattering-matrix cascade, Poynting-flux reduction) is pure JAX and is
reused unchanged whether or not gradients are requested — no separate
"gradient path" is substituted (repository scientific-contract requirements). The only difference is
at the **output-construction boundary**:

* If ``request.require_gradients`` is ``False`` (the default), scalar
  physical outputs (``reflectance_total``, ``transmittance_total``,
  per-order amplitudes) are concretized with ``float(...)``/``.tolist()``
  and a ``sha256`` of the underlying array bytes is recorded. This
  concretization is a derivative boundary and is recorded explicitly in
  ``ModelRunResult.diagnostics["derivative_boundary"]``.
* If ``request.require_gradients`` is ``True``, those concretization calls
  are skipped and the scalar outputs remain live JAX values (potentially
  tracers) inside ``ArtifactRecord.metadata`` so that an outer
  ``jax.grad``/``jax.vjp`` wrapped around a call to :meth:`FmmaxAdapter.run`
  can differentiate through them. No ``sha256`` is computed in this mode
  (computing one would require concretizing a possibly-traced value, which
  is exactly the silent host-copy this project forbids) and
  ``diagnostics["derivative_boundary"]`` records that fact instead.

``derivative.verified`` for ``M_RCWA_FMMAX`` MUST remain ``false`` in the
registry: only one narrow directional-derivative probe (substrate
permittivity -> bare-interface reflectance, relative error 1.24e-4 vs. one
centered finite difference) has been run, which is not the full five-part
gradient-verification bundle.
"""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path
from typing import Any

from multiscale_optics_agent.adapters.base import (
    CostEstimate,
    ModelRunRequest,
    ModelRunResult,
    RunStatus,
)
from multiscale_optics_agent.core.artifacts import ArtifactRecord
from multiscale_optics_agent.core.errors import AdapterDependencyError, UnsupportedCapabilityError
from multiscale_optics_agent.core.graph import Severity, ValidationIssue, ValidationReport
from multiscale_optics_agent.core.specs import ArtifactKind, Device, Framework, ModelSpec
from multiscale_optics_agent.registry.loader import Registry

MODEL_ID = "M_RCWA_FMMAX"

_SUPPORTED_POLARIZATION = "te"
_SUPPORTED_MATERIAL_CONVENTION = "permittivity"
_SUPPORTED_TRUNCATION = "circular"
_SUPPORTED_FORMULATION = "fft"
_REQUIRED_STRUCTURE_METADATA = (
    "period",
    "wavelength_m",
    "incidence",
    "polarization",
    "material_convention",
    "layers",
)


def _import_fmmax() -> tuple[Any, Any, Any]:
    """Lazily import ``fmmax``/``jax``; never called at module import time.

    Returns ``(fmmax, jax, jax.numpy)``. Raises ``AdapterDependencyError`` if
    either package (or a sub-dependency of either) is not importable in the
    current environment.
    """

    try:
        import fmmax
        import jax
        import jax.numpy as jnp
    except ImportError as exc:  # pragma: no cover - exercised via monkeypatch in tests
        raise AdapterDependencyError(
            "M_RCWA_FMMAX requires the 'fmmax' and 'jax' packages, which could not be "
            f"imported in this environment: {exc}. Install the 'wave' extras "
            "(see repository container-only execution policy) or run inside the "
            "agent_solver Docker image."
        ) from exc
    return fmmax, jax, jnp


def _issue(
    severity: Severity, code: str, message: str, location: str | None = None
) -> ValidationIssue:
    return ValidationIssue(severity=severity, code=code, message=message, location=location)


class FmmaxAdapter:
    """``ModelAdapter`` implementation for ``M_RCWA_FMMAX`` (FMMAX RCWA solver)."""

    def __init__(self) -> None:
        self._spec: ModelSpec | None = None

    @property
    def spec(self) -> ModelSpec:
        if self._spec is None:
            self._spec = Registry.from_package().models[MODEL_ID]
        return self._spec

    def estimate(self, request: ModelRunRequest) -> CostEstimate:
        """Heuristic-only cost estimate; no solver call is made.

        Follows the registry ``cost_model``
        (``O(batch * harmonics^3 * layers)``) without measuring wall time.
        """

        num_terms = request.config.get("approximate_num_terms", 1)
        n_layers = None
        structure = request.inputs.get("structure")
        if structure is not None:
            layers = structure.metadata.get("layers")
            if isinstance(layers, list):
                n_layers = len(layers)
        notes = [
            "Heuristic only (no solver call performed): registry cost_model is "
            "O(batch * harmonics^3 * layers) for the eigensolve/scattering-matrix "
            f"recursion. approximate_num_terms={num_terms!r}, n_layers={n_layers!r}.",
        ]
        return CostEstimate(solver_calls=1, confidence="heuristic", notes=notes)

    def validate_request(self, request: ModelRunRequest) -> ValidationReport:
        """Structural/scope validation. Never imports or calls fmmax/jax.

        Deliberately does NOT range-check ``approximate_num_terms`` (e.g.
        reject ``<= 0``): a degenerate/non-convergent truncation order is a
        genuine solver-execution-time failure (fmmax happily returns a
        zero-mode expansion; this adapter's own incident-vector construction
        then fails), not a structural scope violation, and is reported via
        ``ModelRunResult(status=FAILED, ...)`` rather than rejected eagerly.
        """

        issues: list[ValidationIssue] = []
        structure = request.inputs.get("structure")
        if structure is None:
            issues.append(
                _issue(
                    Severity.ERROR,
                    "MISSING_STRUCTURE_INPUT",
                    "M_RCWA_FMMAX requires an input artifact named 'structure'.",
                    "inputs.structure",
                )
            )
            return ValidationReport(issues=issues)
        if structure.kind != ArtifactKind.PERIODIC_STRUCTURE:
            issues.append(
                _issue(
                    Severity.ERROR,
                    "WRONG_ARTIFACT_KIND",
                    f"'structure' input must be kind={ArtifactKind.PERIODIC_STRUCTURE.value}, "
                    f"got {structure.kind.value}.",
                    "inputs.structure",
                )
            )

        md = structure.metadata
        missing = [key for key in _REQUIRED_STRUCTURE_METADATA if key not in md]
        if missing:
            issues.append(
                _issue(
                    Severity.ERROR,
                    "MISSING_STRUCTURE_METADATA",
                    f"'structure' metadata is missing required keys: {sorted(missing)!r}.",
                    "inputs.structure.metadata",
                )
            )
            return ValidationReport(issues=issues)

        if md["polarization"] != _SUPPORTED_POLARIZATION:
            issues.append(
                _issue(
                    Severity.ERROR,
                    "UNSUPPORTED_POLARIZATION",
                    f"Only polarization={_SUPPORTED_POLARIZATION!r} (fundamental-order mode "
                    f"index 0, verified against the analytic Fresnel oracle) is supported; "
                    f"got {md['polarization']!r}. See fmmax_adapter module docstring.",
                    "inputs.structure.metadata.polarization",
                )
            )
        if md["material_convention"] != _SUPPORTED_MATERIAL_CONVENTION:
            issues.append(
                _issue(
                    Severity.ERROR,
                    "UNSUPPORTED_MATERIAL_CONVENTION",
                    f"Only material_convention={_SUPPORTED_MATERIAL_CONVENTION!r} (isotropic "
                    f"relative permittivity) is supported; got {md['material_convention']!r}.",
                    "inputs.structure.metadata.material_convention",
                )
            )

        incidence = md.get("incidence") or {}
        polar = incidence.get("polar_angle_rad", 0.0)
        azimuthal = incidence.get("azimuthal_angle_rad", 0.0)
        if polar != 0.0 or azimuthal != 0.0:
            issues.append(
                _issue(
                    Severity.ERROR,
                    "UNSUPPORTED_INCIDENCE",
                    "Only normal incidence (polar_angle_rad == azimuthal_angle_rad == 0.0) is "
                    f"supported; got polar={polar!r}, azimuthal={azimuthal!r}.",
                    "inputs.structure.metadata.incidence",
                )
            )

        period = md.get("period") or {}
        if "u_m" not in period or "v_m" not in period:
            issues.append(
                _issue(
                    Severity.ERROR,
                    "INVALID_PERIOD",
                    "'period' metadata must provide 'u_m' and 'v_m' primitive lattice "
                    "vectors in meters.",
                    "inputs.structure.metadata.period",
                )
            )

        layers = md.get("layers")
        if not isinstance(layers, list) or len(layers) < 2:
            issues.append(
                _issue(
                    Severity.ERROR,
                    "INVALID_LAYER_STACK",
                    "'layers' must be a list of at least 2 entries (ambient + substrate).",
                    "inputs.structure.metadata.layers",
                )
            )
        else:
            for index, layer in enumerate(layers):
                if not isinstance(layer, dict) or "permittivity" not in layer:
                    issues.append(
                        _issue(
                            Severity.ERROR,
                            "INVALID_LAYER_ENTRY",
                            f"layers[{index}] must be a dict with a 'permittivity' key.",
                            f"inputs.structure.metadata.layers[{index}]",
                        )
                    )
                    continue
                is_patterned = isinstance(layer["permittivity"], list)
                if is_patterned and index in (0, len(layers) - 1):
                    issues.append(
                        _issue(
                            Severity.ERROR,
                            "PATTERNED_SEMI_INFINITE_LAYER",
                            f"layers[{index}] is the ambient/substrate (semi-infinite) layer "
                            "and must be a homogeneous scalar permittivity, not a patterned grid.",
                            f"inputs.structure.metadata.layers[{index}]",
                        )
                    )

        truncation = request.config.get("truncation", _SUPPORTED_TRUNCATION)
        if truncation != _SUPPORTED_TRUNCATION:
            issues.append(
                _issue(
                    Severity.ERROR,
                    "UNSUPPORTED_TRUNCATION",
                    f"Only truncation={_SUPPORTED_TRUNCATION!r} (matches the validated "
                    f"oracle probes) is supported; got {truncation!r}.",
                    "config.truncation",
                )
            )
        formulation = request.config.get("formulation", _SUPPORTED_FORMULATION)
        if formulation != _SUPPORTED_FORMULATION:
            issues.append(
                _issue(
                    Severity.ERROR,
                    "UNSUPPORTED_FORMULATION",
                    f"Only formulation={_SUPPORTED_FORMULATION!r} (matches the validated "
                    f"oracle probes) is supported; got {formulation!r}.",
                    "config.formulation",
                )
            )
        num_terms = request.config.get("approximate_num_terms", 1)
        if not isinstance(num_terms, int):
            issues.append(
                _issue(
                    Severity.ERROR,
                    "INVALID_NUM_TERMS_TYPE",
                    f"config.approximate_num_terms must be an int; got {type(num_terms).__name__}.",
                    "config.approximate_num_terms",
                )
            )

        if not issues:
            issues.append(
                _issue(
                    Severity.INFO,
                    "VALID",
                    "Request is within M_RCWA_FMMAX's supported scope "
                    "(normal incidence, fundamental-order excitation, isotropic permittivity, "
                    "fft/circular numerics).",
                )
            )
        return ValidationReport(issues=issues)

    def run(self, request: ModelRunRequest) -> ModelRunResult:
        fmmax, jax, jnp = _import_fmmax()

        report = self.validate_request(request)
        if not report.valid:
            raise UnsupportedCapabilityError(
                "M_RCWA_FMMAX request is outside this adapter's supported scope: "
                + "; ".join(issue.message for issue in report.errors)
            )

        structure = request.inputs["structure"]
        md = structure.metadata

        try:
            outputs, diagnostics, warnings = _simulate(
                fmmax, jax, jnp, md, request.config, request.design_parameters, request
            )
        except Exception as exc:  # genuine solver-execution-time failure, see module docstring
            return ModelRunResult(
                status=RunStatus.FAILED,
                error_type=type(exc).__name__,
                error_message=str(exc),
                diagnostics={
                    "stage": "fmmax_rcwa_solve",
                    "approximate_num_terms": request.config.get("approximate_num_terms", 1),
                },
            )

        return ModelRunResult(
            status=RunStatus.SUCCEEDED,
            outputs=outputs,
            diagnostics=diagnostics,
            warnings=warnings,
        )


def get_adapter() -> FmmaxAdapter:
    return FmmaxAdapter()


def _resolve_layer_permittivity(
    jnp: Any, index: int, layer_cfg: dict[str, Any], design_parameters: dict[str, Any]
) -> Any:
    """Return a permittivity array shaped for ``fmmax.eigensolve_isotropic_media``.

    A scalar (Python number or a 0-d JAX value, including a traced value
    under ``jax.grad``) is broadcast to FMMAX's homogeneous-layer grid shape
    ``(1, 1)`` exactly as the validated oracle probes do
    (``jnp.asarray(n**2)[..., None, None]``); a 2D nested list is treated as
    a patterned real-space permittivity grid.
    """

    key = f"layers[{index}].permittivity"
    value = design_parameters.get(key, layer_cfg["permittivity"])
    arr = jnp.asarray(value)
    if arr.ndim == 0:
        return arr[..., None, None]
    if arr.ndim == 2:
        return arr
    raise ValueError(
        f"layers[{index}].permittivity must resolve to a scalar or a 2D grid; "
        f"got shape {arr.shape}."
    )


def _simulate(
    fmmax: Any,
    jax: Any,
    jnp: Any,
    structure_metadata: dict[str, Any],
    config: dict[str, Any],
    design_parameters: dict[str, Any],
    request: ModelRunRequest,
) -> tuple[dict[str, ArtifactRecord], dict[str, Any], list[str]]:
    md = structure_metadata
    wavelength_m = float(md["wavelength_m"])
    period = md["period"]
    layers_cfg: list[dict[str, Any]] = md["layers"]
    num_layers = len(layers_cfg)

    primitive_lattice_vectors = fmmax.LatticeVectors(
        u=jnp.asarray(period["u_m"]),
        v=jnp.asarray(period["v_m"]),
    )

    ambient_value = design_parameters.get("layers[0].permittivity", layers_cfg[0]["permittivity"])
    ambient_scalar = jnp.asarray(ambient_value)
    in_plane_wavevector = fmmax.plane_wave_in_plane_wavevector(
        wavelength=jnp.asarray(wavelength_m),
        polar_angle=jnp.asarray(0.0),
        azimuthal_angle=jnp.asarray(0.0),
        permittivity=ambient_scalar,
    )

    approximate_num_terms = config.get("approximate_num_terms", 1)
    expansion = fmmax.generate_expansion(
        primitive_lattice_vectors=primitive_lattice_vectors,
        approximate_num_terms=approximate_num_terms,
        truncation=fmmax.Truncation.CIRCULAR,
    )

    permittivities = [
        _resolve_layer_permittivity(jnp, index, layer_cfg, design_parameters)
        for index, layer_cfg in enumerate(layers_cfg)
    ]
    # formulation is pinned to FFT (not the fmmax default JONES_DIRECT_FOURIER) because FFT is
    # what knowledge/solvers/fmmax/probes/fresnel_oracle_probe.py and gradient_probe.py used to
    # obtain the independently-verified reflectance oracle match; switching formulations without
    # re-verifying would be a silent fidelity change (repository scientific-contract requirements).
    layer_solve_results = [
        fmmax.eigensolve_isotropic_media(
            wavelength=jnp.asarray(wavelength_m),
            in_plane_wavevector=in_plane_wavevector,
            primitive_lattice_vectors=primitive_lattice_vectors,
            permittivity=permittivity,
            expansion=expansion,
            formulation=fmmax.Formulation.FFT,
        )
        for permittivity in permittivities
    ]

    thicknesses = []
    for index, layer_cfg in enumerate(layers_cfg):
        if index in (0, num_layers - 1):
            thicknesses.append(jnp.asarray(0.0))  # semi-infinite ambient/substrate
        else:
            thicknesses.append(jnp.asarray(float(layer_cfg.get("thickness_m", 0.0))))

    s_matrix = fmmax.stack_s_matrix(layer_solve_results, thicknesses)

    n_modes = s_matrix.s11.shape[-1]
    if n_modes == 0:
        raise ValueError(
            "fmmax produced a zero-mode expansion (approximate_num_terms too small / "
            "non-convergent truncation order); cannot construct an incident excitation vector."
        )

    a_0 = jnp.zeros((n_modes, 1), dtype=s_matrix.s11.dtype).at[0, 0].set(1.0)
    b_N = jnp.zeros_like(a_0)

    # --- Explicit s11/s21 convention swap -----------------------------------------------------
    # Per inspect.getdoc(fmmax.ScatteringMatrix) (see module docstring above and
    # knowledge/solvers/fmmax/conventions.md): s11 relates forward-at-start to forward-at-end,
    # i.e. s11 IS TRANSMISSION; s21 relates forward-at-start to backward-at-start, i.e. s21 IS
    # REFLECTION. This is the opposite of common RF/photonic-circuit S-parameter naming, where
    # S11 usually denotes a reflection coefficient. Do not swap these back without re-deriving
    # the convention from the installed fmmax version's own docstring.
    a_N = s_matrix.s11 @ a_0 + s_matrix.s12 @ b_N  # transmitted forward amplitude at end layer
    b_0 = s_matrix.s21 @ a_0 + s_matrix.s22 @ b_N  # reflected backward amplitude at start layer
    # -------------------------------------------------------------------------------------------

    # Physical power via time-averaged Poynting flux, NOT raw |amplitude|^2 (see module
    # docstring / knowledge/solvers/fmmax/failure_guide.md: a naive |s11|^2 * index-ratio
    # attempt gave >1 and did not close energy conservation).
    fwd_start, bwd_start = fmmax.directional_poynting_flux(
        a_0, b_0, s_matrix.start_layer_solve_result
    )
    fwd_end, _bwd_end = fmmax.directional_poynting_flux(a_N, b_N, s_matrix.end_layer_solve_result)

    incident_power = fwd_start[0, 0].real
    reflectance_total = jnp.sum(-bwd_start.real) / incident_power
    transmittance_total = jnp.sum(fwd_end.real) / incident_power
    energy_residual = jnp.abs(reflectance_total + transmittance_total - 1.0)

    order_basis = expansion.basis_coefficients.tolist()  # static (geometry-only), never traced

    require_gradients = request.require_gradients
    metadata: dict[str, Any] = {
        "wavelength_m": wavelength_m,
        "wavelength": wavelength_m,
        "incidence": {"polar_angle_rad": 0.0, "azimuthal_angle_rad": 0.0},
        "polarization": _SUPPORTED_POLARIZATION,
        "order_basis": order_basis,
        "power_normalization": (
            "time_averaged_poynting_flux via fmmax.directional_poynting_flux; "
            "R = -sum(bwd_start.real)/fwd_start[0], T = sum(fwd_end.real)/fwd_start[0]"
        ),
        "reference_plane": {
            "reflection": "start_layer interface (ambient side, z=0)",
            "transmission": "end_layer interface (substrate side)",
        },
        "s_parameter_convention": (
            "fmmax.ScatteringMatrix.s11 is TRANSMISSION and s21 is REFLECTION "
            "(non-standard vs. RF/photonic-circuit convention); swap applied explicitly."
        ),
        "incident_mode_index": 0,
        "energy_residual": (float(energy_residual) if not require_gradients else energy_residual),
    }

    if require_gradients:
        metadata["reflectance_total"] = reflectance_total
        metadata["transmittance_total"] = transmittance_total
        derivative_boundary = (
            "require_gradients=True: reflectance_total/transmittance_total/energy_residual are "
            "left as live JAX values (not concretized with float()/tolist()) so an outer "
            "jax.grad wrapped around FmmaxAdapter.run stays traceable through this ArtifactRecord. "
            "No sha256 was computed (would require concretizing a possibly-traced value)."
        )
        sha256 = None
        uri = f"trace://fmmax/{request.run_id}/{request.node_id}"
    else:
        metadata["reflectance_total"] = float(reflectance_total)
        metadata["transmittance_total"] = float(transmittance_total)
        derivative_boundary = (
            "require_gradients=False: reflectance_total/transmittance_total/energy_residual were "
            "concretized with float() at the ModelRunResult output boundary (this "
            "host/Python-scalar extraction is a derivative boundary and is recorded "
            "here, not performed silently)."
        )
        digest_source = repr(
            (float(reflectance_total), float(transmittance_total), order_basis, wavelength_m)
        ).encode("utf-8")
        sha256 = hashlib.sha256(digest_source).hexdigest()
        run_dir = Path(tempfile.gettempdir()) / "multiscale_optics_agent_runs" / request.run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        output_path = run_dir / f"{request.node_id}_diffraction.npz"
        import numpy as np

        np.savez(
            output_path,
            reflectance_total=float(reflectance_total),
            transmittance_total=float(transmittance_total),
            order_basis=np.asarray(order_basis),
            reflection_amplitude=np.asarray(b_0),
            transmission_amplitude=np.asarray(a_N),
        )
        uri = f"file://{output_path}"

    diffraction_record = ArtifactRecord(
        id=f"{request.node_id}.diffraction",
        kind=ArtifactKind.DIFFRACTION_CHANNELS,
        uri=uri,
        sha256=sha256,
        shape=(),
        dtype=str(s_matrix.s11.dtype),
        framework=Framework.JAX,
        device=Device.CPU,
        units="dimensionless",
        metadata=metadata,
    )

    diagnostics: dict[str, Any] = {
        "approximate_num_terms_requested": approximate_num_terms,
        "approximate_num_terms_actual": expansion.num_terms,
        "n_modes": n_modes,
        "truncation": _SUPPORTED_TRUNCATION,
        "formulation": _SUPPORTED_FORMULATION,
        "s_parameter_swap_applied": True,
        "derivative_boundary": derivative_boundary,
        "derivative_mode": "native_autodiff",
        "derivative_verified": False,
    }
    if not require_gradients:
        diagnostics["energy_residual"] = float(energy_residual)

    warnings = [
        "FMMAX's complex reflection/transmission amplitude sign/phase does not match the "
        "textbook Fresnel convention (verified only for |r|^2 magnitude to ~3e-7 relative "
        "error against a bare-interface oracle; phase NOT reconciled). Do not assume this "
        "adapter's phase matches another solver's convention without an explicit, tested "
        "coupler-level reconciliation. See knowledge/solvers/fmmax/conventions.md.",
        "A unit-cell result does not by itself validate a finite or rapidly varying metasurface "
        "(registry validity.warnings).",
    ]

    return {"diffraction": diffraction_record}, diagnostics, warnings
