# Roadmap — intent, explicitly non-executable

Everything on this page is a **plan**, not a capability. Nothing here has an
adapter, a coupler implementation, a registry entry, or a runnable benchmark,
and nothing on this page may be cited as evidence of anything.

It exists because CHE-87 removed three solver integrations and thirteen
speculative registry entries, and deleting an intention without recording it is
how a project loses the reason it had the intention. The alternative — keeping
the entries in the registry so the plan stays visible — was rejected: a registry
entry is a statement that a graph may address the component, and thirteen of
them said so about components that cannot execute.

`benchmarks/manifest.yaml` now lists implemented benchmarks only, so the two
files answer different questions and neither has to be read for the other's.

## Removed by CHE-87 — solver integrations

Each had an adapter, a knowledge pack, a registry entry, an example graph, and a
pytest marker. All six surfaces were removed in one commit, because a partial
retirement is the worst state: code deleted while a registry entry survives
reads to a planner as a supported capability.

The implementation is in git history at `10904f8` and earlier. Reintroducing one
means a fresh, scoped integration with a Linear issue behind it — the same rule
CHE-72 set for SAX — not a revert.

| Component | Package | Why it went | Worth knowing before restoring |
| -- | -- | -- | -- |
| `M_RCWA_FMMAX` | `fmmax` 1.7.1 | out of scope per AGENTS.md; tests archived by CHE-67; no capability declaration, so `capabilities_for` already raised `UNKNOWN_COMPONENT` | Energy conservation *did* close through `fmmax.directional_poynting_flux` (R+T = 0.99999976 bare interface, 0.9999998584 on a 9-order lamellar grating). What never closed is the **phase/sign convention**: reflectance magnitude matched a Fresnel oracle to ~1e-7 while the complex amplitude's sign did not match the textbook convention. That is unresolved and is the first thing to settle before coupling FMMAX to anything with its own phase reference. |
| `M_EM_FDTDX` | `fdtdx` 0.6.2 | out of scope; tests archived by CHE-67 | Two gradient paths failed outright when probed: `jax.grad` w.r.t. source wavelength returned exactly `0.0` against a large nonzero finite difference (suspected internal `round()`), and `jax.grad` w.r.t. background permittivity raised `ConcretizationTypeError` because `fdtdx.place_objects` does concrete Python introspection. Differentiable parameters must flow through `ParameterContainer`/`apply_params`, never construction-time arguments. |
| `M_THERMAL_JAX_FEM` | `jax-fem` 0.0.12 | out of scope, and it never had an adapter at all | Two blockers, neither about optics. `jax_fem.solver` — the only FEM solve entry point — unconditionally imports `petsc4py`, which has no prebuilt wheel; mesh and `Problem` construction worked, no solve was ever executable. And a **licence mismatch**: PyPI metadata claims BSD, the repository's actual `LICENSE` is GPLv3 (verified directly, 2026-07-30). This project is MIT. Get explicit sign-off before shipping an adapter that depends on it. |

## Removed by CHE-87 — registry entries with no implementation

Two internal models:

* `M_SENSOR_IDEAL` — pixel integration and noise on a PSF. Deliberately excluded
  from M3, and the exclusion is recorded at length in
  `benchmarks/protocols/slice_protocol.yaml`: it is why the M3 graph terminates at the
  propagated `ComplexField` and PSF extraction is a measurement rather than an
  edge. Restoring it means deciding a radiometric normalization, which is the
  actual work.
* `M_SCATTERING_ASSEMBLER_INTERNAL` — assembles modal excitation/response data
  into an S-parameter object. Had no consumer once SAX was removed by CHE-72.

Eight couplers, none with an implementation and none with a graph node:
`C_EIKONAL_TO_WAVE`, `C_NEAR_TO_FAR`, `C_CELL_TO_SURFACE`,
`C_ABSORPTION_TO_HEAT`, `C_TEMPERATURE_TO_MATERIAL`, `C_FIELD_TO_MODE`,
`C_SMATRIX_TO_CIRCUIT`, `C_GENERALIZED_SNELL`.

`C_FIELD_TO_PSF` is **not** on this list and is not coming back. It was retired
by CHE-36 on a definitional argument rather than a scheduling one: extracting
`|U|²` from the terminal simulated field is a measurement, not a
cross-representation handoff, so it is not a coupler.
`tests/test_graph_validation.py::test_field_to_psf_is_not_a_registered_coupler`
pins that.

## Planned benchmarks, moved off `manifest.yaml`

Level 1 — one-model simulation. Two of the four name a model id that has never
existed anywhere in this repository, which is worth stating plainly: writing an
id into a plan does not create a component.

| Task | Model | Status |
| -- | -- | -- |
| `L1-TMM-01` — multilayer Bragg mirror | `M_TMM_JAXLAYERLUMOS` | **dangling id**: no adapter, no registry entry, no package chosen |
| `L1-RCWA-01` — binary grating diffraction | `M_RCWA_FMMAX` | integration removed above |
| `L1-EM-01` — full-wave waveguide component | `M_EM_FDTDX` | integration removed above |
| `L1-PC-01` — photonic-crystal band structure | `M_PC_LEGUME` | **dangling id**: same |

Level 2 — two-to-three-model orchestration. `L2-PSF-01` is implemented and stays
in the manifest; `L2-COUPLER-01` is implemented and archived with gen1.

| Task | Graph | Status |
| -- | -- | -- |
| `L2-META-01` — unit-cell-to-system meta-optic imaging | `M_RCWA_FMMAX → C_CELL_TO_SURFACE → M_WAVE_CHROMATIX` | every non-Chromatix component removed |
| `L2-GRATING-01` — near-field to far-field / fiber overlap | `M_EM_FDTDX → C_NEAR_TO_FAR` | both removed |
| `L2-THERMO-01` — optical absorption to thermo-optic response | `M_EM_FDTDX → C_ABSORPTION_TO_HEAT → M_THERMAL_JAX_FEM → C_TEMPERATURE_TO_MATERIAL` | all four removed |

Level 3 — hard inverse design. Three candidate paper demonstrations, none
implemented and none with a protocol:

1. `L3-HYBRID-01` — joint refractive-diffractive wide-field microscope. The
   nearest to reachable, because it exercises the ray-wave bridge that exists
   rather than a solver that does not.
2. `L3-ACHROMATIC-01` — achromatic meta-optic computational imager with
   RCWA-to-system coupling.
3. `L3-ROBUST-01` — fabrication- and temperature-robust grating coupler with
   adaptive fidelity.

The standard these were written against, which still applies to whatever
replaces them: independent final validation, multiple initializations, held-out
perturbations, equal compute budgets, gradient checks, and a complete accounting
of solver calls and human interventions. A Level 2 task passes only when the
**coupling boundary** is independently tested; passing each model separately is
not sufficient and never was.
