# Repository Cleanup and Reorganization Plan

**Status:** Proposed  
**Scope:** Repository-wide cleanup and reorganization  
**Basis:** Read-only audit of the current working tree, imports, registries, tests,
entry points, knowledge dependencies, and execution paths  
**Non-goal:** This document does not authorize deleting, moving, renaming, or
rewriting code. Each implementation phase requires its own scoped Linear issue.

## 1. Executive recommendation

The repository should be organized around two explicitly separated planes:

1. **Agent and control infrastructure** — context selection, task prompts, agent
   benchmarking, execution policy, and provenance.
2. **Optics and data infrastructure** — typed artifacts, Optiland and Chromatix
   integrations, ray-wave couplers, graph contracts, observables, and scientific
   verification.

These planes are currently blurred. The Python package is primarily an optics
simulation and validation library; it is not currently an executable multi-agent
runtime. The `agents/` package is empty, while the actual agent-facing behavior
lives in `AGENTS.md`, `CONTEXT_MANIFEST.yaml`, `knowledge/`, and the agent benchmark
harness.

Until an agent runtime exists, the repository should describe itself precisely as
**agent-facing optics infrastructure**. Empty namespaces should not imply
functionality that has not been implemented.

## 2. Audit summary

The current working tree contains approximately 24,000 lines of production Python
across 55 Python files. The default test surface collects 817 tests. The principal
cleanup issue is not a lack of implementation; it is that production code,
historical benchmarks, future catalog entries, executable evidence, agent context,
and research records share overlapping directories and names.

The most important structural findings are:

- The packaged registry lists substantially more components than can be executed.
- Adapter discovery imports files that are benchmark helpers rather than adapters.
- The Optiland and Chromatix adapter modules combine too many responsibilities.
- Adapters and couplers have reverse dependencies through shared execution types.
- `knowledge/` mixes concise agent guidance with probes, tests, recorded output,
  tutorials, source catalogs, and large binary papers.
- `benchmarks/` mixes active suites, superseded suites, probes, protocols, records,
  narrative reports, and speculative future tasks.
- Ticket history is embedded in active explanations to the point that many files
  describe how the code evolved instead of explaining its current behavior.

## 3. Runtime registry versus planning catalog

The packaged registry currently exposes seven models and ten couplers. That is not
the same as the executable surface:

- Four model adapters are dynamically discoverable: Optiland, Chromatix, FMMAX,
  and FDTDX.
- Four components have authoritative executable capability declarations:
  Optiland, Chromatix, ray-to-wave, and wave-to-ray.
- Only `C_RAY_TO_WAVE` has a graph-facing executable coupler node.
- The CLI validates graph structure but does not execute a graph.

This makes `list-models` and `list-couplers` ambiguous: they appear to list
runnable components, while many entries are plans or contracts without an
implementation.

The registry should be split into:

- **Runtime registry:** components invocable through a declared, tested protocol.
- **Planning catalog:** possible future components, explicitly non-executable and
  excluded from runtime discovery.

The initial runtime registry should contain Optiland, Chromatix, and
`C_RAY_TO_WAVE`. `C_WAVE_TO_RAY` should be labeled as a library API until it gains
a graph node, or be promoted deliberately by implementing and testing that node.

CLI output should distinguish at least these states:

- specified;
- implemented;
- capability-validated;
- graph-executable;
- current-scope or planning-only.

Speculative production registry entries should not be retained merely to provide
fixtures for graph-validation tests. Those tests can construct local fixture
registries instead.

## 4. Adapter disposition

The similarly named files in `adapters/` have different roles and should not be
treated alike.

| Current file | Actual role | Recommended disposition |
|---|---|---|
| `chromatix_carrier_removed.py` | Live carrier-removed ASM used by the production adapter, active tests, and probes | Keep; rename to `carrier_removed_asm.py` and place under the Chromatix integration package |
| `chromatix_benchmark_adapter.py` | Helper used only by Gen1 `L1-WAVE-01` | Move or archive with the Gen1 benchmark suite |
| `chromatix_scaling_adapter.py` | Helper used only by the Gen1 scaling runner | Move or archive with the Gen1 benchmark suite |
| `optiland_benchmark_adapter.py` | Helper used only by Gen1 ray benchmarks | Move or archive with the Gen1 benchmark suite |
| `optiland_builder.py` | Live conversion from the canonical optical-system contract to Optiland | Keep under an Optiland integration package |
| `optiland_ray_trace.py` | Live caller-supplied coherent ray tracing used by recent bridge work | Keep, but remove its dependency on coupler implementation modules |
| `fmmax_adapter.py` | Dynamically discoverable, but direct behavioral tests are archived | Retire from the active runtime in one dedicated change |
| `fdtdx_adapter.py` | Dynamically discoverable, but direct behavioral tests are archived | Retire from the active runtime in one dedicated change |

The three benchmark helper files end in `_adapter.py`, so package scanning imports
them even though they register no model. Replace implicit package scanning with an
explicit runtime registration map. Registration should be intentional and should
fail on duplicate identifiers.

## 5. Split oversized solver integrations by capability

`optiland_adapter.py` and `chromatix_adapter.py` each combine several concerns:

- graph-facing adapter protocol;
- standalone baseline contracts;
- dependency and provenance discovery;
- capability negotiation;
- request validation;
- scientific execution;
- persistence and hashing;
- historical benchmark behavior.

They should be split only after characterization tests preserve current behavior.
A suitable target is:

```text
integrations/
  optiland/
    adapter.py          # Graph-facing ModelAdapter only
    requests.py         # Typed request/result/failure contracts
    builder.py          # OpticalSystemSpec -> Optiland object
    trace.py            # Ordinary model trace
    coherent_trace.py   # Caller-supplied coherent ray population
    provenance.py
  chromatix/
    adapter.py
    requests.py
    propagation.py
    carrier_removed_asm.py
    provenance.py
```

Standalone M1 baseline persistence and reporting should move to benchmark code.
It should not dominate the current solver adapter API.

## 6. Correct the dependency direction

The current layer direction is not clean:

- `couplers/base.py` imports `CostEstimate` and `RunStatus` from
  `adapters/base.py`.
- `optiland_ray_trace.py` imports the coupler bridge, coherent batch, and coupler
  contracts.

Shared execution concepts should move to a neutral package. General boundary
artifacts should not be owned by a coupler implementation package merely because
the first consumer was a coupler.

The desired dependency direction is:

```text
domain contracts
      |
      v
execution and graph protocols
      |
      v
integrations       couplers
       \             /
        orchestration
```

Adapters and couplers should not import one another's implementation packages.
An automated dependency-direction test should enforce this.

## 7. Proposed package structure

```text
src/multiscale_optics_agent/
  domain/
    artifacts/
      ray.py
      wavefront.py
      field.py
      psf.py
      frames.py
    optical_system.py
    serialization.py

  execution/
    protocols.py
    arrays.py
    precision.py
    capabilities.py
    resources.py
    errors.py

  graph/
    specs.py
    registry.py
    validation.py

  integrations/
    optiland/
      adapter.py
      requests.py
      builder.py
      trace.py
      coherent_trace.py
    chromatix/
      adapter.py
      requests.py
      propagation.py
      carrier_removed_asm.py

  couplers/
    ray_wave/
      ray_to_wave.py
      wave_to_ray.py
      handoff.py
      quadrature.py
      curvature.py
      gradient.py
      streaming.py
      cascade.py
      node.py

  observables/
    psf.py

  verification/
    asm.py
    psf.py

  agent_eval/
    suite.py
    grading.py
    participants.py

  cli.py
```

This is a target layout, not a recommendation for one large mechanical move.
Compatibility imports should preserve public paths during a staged migration.

## 8. Knowledge reorganization

`knowledge/` should contain only material intended to be disclosed to an agent.
It should not also be a test tree, probe tree, evidence database, tutorial suite,
paper archive, and broad future-solver catalog.

The concise target is:

```text
knowledge/
  README.md
  solvers/
    optiland/
      card.yaml
      conventions.md
      examples.md
      failures.md
    chromatix/
      card.yaml
      conventions.md
      examples.md
      failures.md
  couplers/
    ray_to_wave/
      card.yaml
      conventions.md
      theory.md
      failures.md
    wave_to_ray/
      card.yaml
      conventions.md
      theory.md
      failures.md
```

Relocate the remaining material:

- `tutorials/` -> `tests_tutorial/cases/<solver>/`
- `probes/` -> `verification/probes/<solver-or-coupler>/`
- `expected/` -> `verification/records/`
- paper PDFs -> `references/papers/` or DOI/artifact storage
- broad solver source catalog -> archive or remove
- inactive solver packs -> retire with their corresponding integrations

The flat `knowledge/solver_cards/<name>.yaml` files and nested
`knowledge/solvers/<name>/solver_card.yaml` files should collapse into one card
per active component. The two levels have already drifted and should not remain
parallel sources of truth.

Each card should provide concise routing information and link to conventions,
examples, failures, executable evidence, and the authoritative capability
declaration. It should not restate device and dtype tables owned by executable
code.

## 9. Remove ticket history from active explanations

Active code and documentation should explain current behavior without requiring
access to old tickets.

Replace statements such as "CHE-38 showed" with self-contained evidence:

- what was measured;
- solver version or commit;
- input configuration;
- observed value and tolerance;
- evidence path;
- measurement date;
- consequence for current behavior.

Ticket identifiers remain appropriate in archived reports, Git history, Linear,
and optional structured provenance fields. They should not be needed to understand
a scientific convention, failure mode, or code invariant.

The same rule applies to source comments. Comments should explain why an invariant
exists and cite a stable test or evidence record, not narrate the ticket sequence
that produced the implementation.

## 10. Evaluation and benchmark reorganization

The current `evaluation/` package mixes measurements, independent oracles,
benchmark utilities, and benchmark-specific metalens fixtures.

Recommended disposition:

- `psf_measurement.py` -> `observables/psf.py`
- `asm_oracle.py` and `psf_oracles.py` -> `verification/`
- `m1_bundle.py` -> Gen1 benchmark archive
- `metalens.py` -> the metalens benchmark package
- unused `evaluation/checks.py` -> remove unless a current consumer is introduced

The top-level `benchmarks/` directory should distinguish executable suites from
historical evidence:

```text
benchmarks/
  physics/
    current/
    archive/gen1/
  agents/
    v1/
  probes/
  records/
  protocols/
```

The active benchmark manifest should list only implemented benchmarks. Planned
TMM, RCWA, EM, thermal, and Level-3 tasks belong in a roadmap rather than an
executable benchmark registry.

Ticket-named reports should move into a dated report archive and receive
descriptive filenames. Historical identifiers may remain inside those archived
reports as provenance.

## 11. Removal and retention recommendations

### Remove from the active tree after dedicated verification

- the empty `agents/` package;
- unused `evaluation/checks.py`;
- FMMAX and FDTDX adapters, runtime registry entries, smoke graphs, default-image
  dependencies, and knowledge packs;
- JAX-FEM knowledge and speculative model entries;
- unimplemented sensor, scattering, and coupler entries from the runtime registry;
- Gen1 benchmark helpers from production `src/`;
- stale active audits that describe an implementation state that no longer exists;
- ignored `tmp_probes/`, `__pycache__`, and `.DS_Store` workspace residue.

Git history plus one concise archival disposition document is preferable to
copying obsolete source code into another large archive tree.

### Keep and clarify

- carrier-removed ASM;
- Optiland builder and caller-supplied ray tracing;
- ray-to-wave implementation and graph node;
- wave-to-ray and coherent bridge code, which have strong active test coverage;
- precision, dtype, device, and array bridge machinery;
- resource guards;
- typed physical artifacts;
- scientific measurements and independent oracles.

Wave-to-ray should be documented as a supported research extension outside the
canonical forward milestone. Its active tests and current consumers show that it
is not dead code.

## 12. Command and validation surface

The Make targets currently invoke bare `pytest` and `python`, while repository
policy describes the targets as equivalents of `./run.sh ...` commands. The
interface should be made unambiguous:

- host-facing Make targets should invoke `./run.sh`; or
- Make should be explicitly container-internal, with separate documented
  host-facing commands.

Repository validation should scan manifest-defined active paths. Archives and
ignored scratch probes should not affect the active package validator or import
audit.

The import audit should also understand dynamic registration explicitly rather
than treating package scanning as ordinary static use.

## 13. Recommended migration sequence

1. **Stabilize the current working tree.** Land or separate the existing staged,
   unstaged, and untracked work before structural cleanup begins.
2. **Establish truth-bearing registries.** Split runtime registration from the
   planning catalog and make implementation status visible in the CLI.
3. **Retire out-of-scope integrations coherently.** Remove FMMAX, FDTDX, JAX-FEM,
   speculative entries, related dependencies, graphs, and knowledge in one scoped
   change.
4. **Move historical benchmark support out of production.** Relocate the three
   benchmark-only adapter helpers and `evaluation/m1_bundle.py` with Gen1.
5. **Introduce solver subpackages.** Move Optiland and Chromatix files with
   compatibility imports before splitting large modules.
6. **Correct layer dependencies.** Extract shared execution protocols and boundary
   artifacts so integrations and couplers no longer import one another.
7. **Slim `knowledge/`.** Consolidate cards, move probes and tutorials, replace
   ticket narratives, and keep only active solver/coupler guidance.
8. **Reorganize evaluation and benchmarks.** Separate observables, verification,
   current benchmarks, and historical records.
9. **Archive or remove stale documentation.** Retain a concise architecture
   overview, current operational guidance, scientific conventions, and dated
   historical reports.
10. **Add architectural enforcement.** Make future drift fail tests rather than
    relying on another manual repository audit.

Each phase should be its own reviewable issue and pull request. Do not combine
dependency retirement, package moves, contract changes, and scientific behavior
changes in one migration.

## 14. Enforcement to add

The final structure should be protected by automated checks:

- the runtime registry equals the executable implementation registry;
- capability claims equal authoritative measured declarations;
- production dependency direction has no cycles or reverse-layer imports;
- benchmark-only files cannot match runtime adapter discovery conventions;
- active knowledge contains no executable probes, tutorials, or recorded arrays;
- active knowledge uses ticket IDs only in structured provenance;
- generated schemas and inventories reproduce without drift;
- repository audits exclude ignored scratch paths and archives;
- every active solver and coupler has one canonical knowledge card;
- CLI output distinguishes plans from executable components.

## 15. Verification gates for implementation

Every cleanup phase should run checks proportional to its scope:

```bash
./run.sh pytest -q
./run.sh python scripts/check_context_sync.py
./run.sh python scripts/validate_package.py
./run.sh python scripts/export_schemas.py
```

In addition:

- run the narrow affected subsystem tests after every change;
- run the tutorial suite when solver integration or tutorial paths change;
- run the deterministic agent benchmark gate when knowledge disclosure paths or
  grading code change;
- run the dedicated GPU suite when device logic, dependencies, or adapter imports
  change, following the shared-server resource policy;
- review the final diff for unrelated changes and compatibility-shim omissions.

## 16. Risks and intentional non-goals

Primary risks:

- breaking imports used by probes and historical benchmark runners;
- accidentally changing scientific behavior during mechanical moves;
- losing executable evidence while slimming agent-facing knowledge;
- presenting planned components as supported after the registry split;
- changing process-global JAX or Optiland state through import-order changes;
- deleting unguarded integrations without removing their registry and dependency
  claims at the same time.

Intentional non-goals of the cleanup:

- no new solver integrations;
- no new graph executor unless separately approved;
- no new cross-framework derivative claim;
- no scientific tolerance changes;
- no restoration of SAX;
- no redesign of the verified ray-to-wave mathematics;
- no universal artifact type system beyond the established boundary artifacts.

The cleanup is successful when a new contributor or agent can answer, from the
directory structure alone:

1. what is executable now;
2. what is only planned;
3. where each solver integration lives;
4. where ray-wave boundary contracts and physics live;
5. which documents are concise agent context;
6. where executable scientific evidence is stored;
7. which benchmarks are maintained and which are historical.
