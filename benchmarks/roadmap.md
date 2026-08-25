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

## The planned L1/L2/L3 benchmarks are gone (CHE-133, M0.5.4)

Ten rows lived here: four Level 1 one-model tasks, three Level 2 orchestrations
and three Level 3 inverse-design demonstrations. They were task definitions in a
retired taxonomy, and four of them named model ids — `M_TMM_JAXLAYERLUMOS`,
`M_PC_LEGUME`, and the two dangling Level 1 entries — that have never existed
anywhere in this repository. Writing an id into a plan does not create a
component, and keeping the rows meant the plan looked like coverage.

Two things in them were worth keeping and are kept:

* **The standard the Level 3 tasks were written against.** Independent final
  validation, multiple initializations, held-out perturbations, equal compute
  budgets, gradient checks, and a complete accounting of solver calls and human
  interventions. A composed task passes only when the **coupling boundary** is
  independently tested; passing each model separately is not sufficient and never
  was. That standard now has an executable form: `BenchmarkFamily` refuses to let
  an oracle that shares code with the thing under test decide anything, and a
  composed family that cannot name an independent decider is category B4.
* **The observation behind the hybrid-microscope row**: that a joint refractive–diffractive wide-field
  microscope is the nearest to reachable of the three, because it exercises the
  ray–wave bridge that exists rather than a solver that does not. That is the
  demo2/demo3 line of work, which is real and is where it went.

The replacement is the B0–B4 family architecture: `src/verification/families/`
and `benchmarks/inventory.yaml`. What each retired row's content became, if
anything, is recorded there.

