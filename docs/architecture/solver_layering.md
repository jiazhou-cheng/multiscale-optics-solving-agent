# The solver tier — Physics Solver, Knowledge, Adapter

**Status: the canonical architecture document for this repository.** Promoted by
CHE-93. `README.md` orients a new reader; `AGENTS.md` states the rules; this
states how the pieces fit and which file wins when two disagree.

**Date:** 2026-08-20, with the gap list updated as CHE-84's phases closed each one
**Stance:** normative for the solver tier, with a gap list (§8)
**Scope:** the solver tier only. Couplers, artifacts, the registry schema and the
graph get a pointer in §9, not a description.

## 0. What this report is, and is not

This is the normative statement of how an external physics solver enters this
project: what each layer of the solver stack is *for*, which file holds it, and
which file wins when two disagree. Every claim cites the file that holds it, and
every quoted string is verbatim.

It is **not** a replacement for `AGENTS.md`, which remains the rule source for
scope, execution and workflow. It is **not** startup context: read it when adding
or auditing a solver, the way `docs/context/` is read. It does **not** describe
couplers — a coupler carries physical assumptions belonging to neither solver it
joins, which is why it gets its own tier and its own report.

The abstraction it makes explicit:

```
Physics Solver                      the pinned external package
  ↳ Knowledge                       capabilities, conventions, constraints, usage guidance
      ↳ API / Adapter               standardized access to the solver
```

## 1. The three layers, defined

| Layer | Question it answers | Concrete artifact | Who may change it | What enforces it |
|---|---|---|---|---|
| **Physics Solver** | what actually computes the physics | a pinned external package (`optiland==0.6.0`, `chromatix @ d24bdf0`) | upstream only; we pin | the version pin + an import probe |
| **Knowledge** | what may we ask it, and how do we know | `knowledge/solvers/<name>/` (prose + YAML) **and** `core/capabilities.py` (executable) | us, but only after a probe passes | `probes/*.py` vs `expected/*.json`; `tests/test_registry_matches_capabilities.py` |
| **API / Adapter** | how does this project call it | `adapters/<solver>_adapter.py` behind `ModelAdapter` | us | `tests/test_<solver>_adapter.py`, `tests/test_adapter_registry.py` |

### The invariant: claims flow up, evidence flows down

The adapter may expose only what Knowledge declares. Knowledge may declare only
what a probe measured against the pinned solver. **No layer may widen a lower
layer's claim.** `core/capabilities.py:8-11` states the direction of authority
for the machine-readable half:

> This module is the source of truth. `registry/models.yaml` and
> `registry/couplers.yaml` are downstream reflections of it, updated only after
> the executable tests pass -- never the other way round -- and
> `tests/test_registry_matches_capabilities.py` fails if the two disagree.

The corollary, and the reason the layering is worth writing down: a solver's
*absence* from a layer is itself a statement. A package with a knowledge pack but
no capability declaration is planning-only, and `capabilities_for()` says so by
raising rather than defaulting (`core/capabilities.py:202-218`).

## 2. Layer 1 — Physics Solver

**Definition.** The pinned external package. The only thing in the system that
computes physics. Treated as an unmodifiable dependency: we never patch it, and
we never describe it from its documentation when we can measure it.

**The two in scope**, pinned in `docker/requirements.txt`:

```
chromatix @ git+https://github.com/chromatix-team/chromatix.git@d24bdf0022835bb8ce1cdcc6aeafbc7fcb39daee   # :57
optiland==0.6.0                                                                                            # :104
```

Three rules, each with its enforcement.

**(a) Never imported outside an adapter module.** `AGENTS.md` ("Keep external
solver imports inside adapter modules"). For the coupler core this is not a
convention but an assertion: `tests/test_ray_to_wave.py` contains both
`test_coupler_core_imports_no_solver_engine` (an AST walk of the module) and
`test_coupler_core_loads_no_engine_at_runtime` (a `sys.modules` check). The
reason is stated in `knowledge/couplers/ray_to_wave/coupler_card.yaml`:

> The coupler core is the physics under test, so it imports neither optiland
> nor chromatix. […] if the core could import one, a coupler defect could be
> misattributed to engine behaviour.

**(b) Imported lazily, never at module import time.** `adapters/__init__.py`:

> the external solver package must be imported lazily, inside a private
> `_import_<solver>()` helper called from `run()`/`estimate()`, never at module
> import time -- importing this package (or any of its siblings)
> must never require any heavy optional dependency to be installed.

Real instances: `solvers/optiland/adapter.py:511` (`_import_optiland`, returning
`(optiland.backend, optiland.backend.utils, torch-or-None)`) and
`solvers/chromatix/adapter.py:220` (`_do_import_chromatix`, deliberately split
from its wrapper at `:252` "so tests can force an `ImportError`").

**(c) A pin is a physics claim, not hygiene.** Two measured reasons, one per
solver:

- Chromatix — `knowledge/solvers/chromatix/solver_card.yaml:15-20`:
  > The PyPI package literally named "chromatix" (0.0.1, published 2022-07-10,
  > 2.6 kB wheel) is an unrelated namesquat with no optics code. `pip install
  > chromatix` alone silently installs the wrong package.
- Optiland — `torch` is not a declared dependency even though the project
  advertises PyTorch differentiability, so a bare install is silently
  non-differentiable (`knowledge/solvers/optiland/solver_card.yaml:15-21`,
  confirmed via installed wheel metadata). `docker/Dockerfile:43` therefore
  installs `torch==2.13.0` separately from the CPU-only wheel index.

Both facts live at the Knowledge layer because neither is visible from the code.

## 3. Layer 2 — Knowledge: four kinds of statement

This is the layer the rest of the report is really about, and the one where the
vocabulary needs fixing. Each subsection below gives: *definition → the question
it answers → the authoritative file → one real excerpt → how it is checked*.

### 3.1 Capabilities — the word names two different things

**This is the central ambiguity.** "Capabilities" in this repo refers to two
artifacts with different authority, different consumers and different
enforcement. The test that separates them is one question: **can a program act on
it?**

#### 3.1.a Capability *declaration* — executable, authoritative

**Definition.** A machine-readable statement of what the package can *execute*:
devices, precisions, dtypes, array namespaces. Every entry is measured, not read
off documentation.

**File.** `src/core/capabilities.py` — exactly four
instances of `ComponentCapabilities` (`M_RAY_OPTILAND`, `M_WAVE_CHROMATIX`,
`C_RAY_TO_WAVE`, `C_WAVE_TO_RAY`), keyed in `COMPONENT_CAPABILITIES` at `:191`.
The dataclass is `core/precision.py:399-451`:

```python
component: str
devices: frozenset[DeviceKind]
precisions: frozenset[Precision]
accepted_input_dtypes: frozenset[DType]
native_compute_dtypes: frozenset[DType]
output_dtypes: frozenset[DType]
namespaces: frozenset[ArrayNamespace]
lossy_input_dtypes: frozenset[DType] = frozenset()
minimum_compute_precision: Precision = Precision.FP32
device_namespaces: Mapping[DeviceKind, frozenset[ArrayNamespace]] = ...
promotes_input: bool = False
evidence: str = ""
notes: str = ""
```

**Why four dtype sets and not one** — `core/precision.py:401-405`:

> Four dtype sets, not one, because they are genuinely different questions and
> collapsing them is how "supports float16" comes to mean "will not crash if
> handed float16"

So `accepted_input_dtypes` is what may cross inward, `native_compute_dtypes` is
"the honest answer to 'does it support precision X'", `output_dtypes` need match
neither, and `promotes_input` records whether acceptance was achieved by casting
— because "an accepted-but-not-native dtype is a *promotion*, and must never be
advertised as native support."

**Worked example — `CHROMATIX_CAPABILITIES` (`capabilities.py:100-135`)**, which
exercises every interesting field at once:

- `precisions=frozenset({Precision.FP32})`, with the comment: "Not a policy
  choice -- there is no complex128 storage in the package, so an FP64 request has
  nothing to execute."
- `lossy_input_dtypes=frozenset({DType.COMPLEX128})` — the field that only makes
  sense once you accept that a solver can lie by succeeding:
  > complex128 is physically ingestible and silently truncated by Chromatix
  > itself. Keeping it out of `accepted_input_dtypes` is what makes the bridge
  > refuse it under SAFE and record it as lossy under ALLOW_DOWNCAST, instead of
  > letting the loss happen inside ScalarField where nothing measures it.
- `evidence=` names the mechanism, the probe, the image and the device:
  "`ScalarField.__init__` is `jnp.asarray(u, dtype=jnp.complex64)`;
  `Field.build(complex128 array)` returns complex64 even under
  `jax_enable_x64=True` […] (`benchmarks/probes/precision/chromatix_capability.py`,
  `agent_solver_gpu`, jax 0.6.2 backend gpu)".

**Second example — `device_namespaces` (`capabilities.py:81-84`).** Optiland
reaches CUDA only through its torch backend, because `set_device` raises
`BackendCapabilityError` on the numpy backend. The declaration encodes that as
data rather than control flow:

> Declaring it here rather than as an `if backend == ...` branch is what keeps
> the two from drifting.

**Two hard rules.**

1. Evidence is mandatory. `core/precision.py:448-450`: "Where the claims above
   were measured. **Required: a capability with no evidence is an intention.**"
2. An undeclared component is an error, never a default
   (`capabilities.py:207-218`):
   > Add one to core/capabilities.py with the probe evidence behind it. A
   > component with no declaration has no validated device or dtype support, and
   > this project will not guess one.

**How it is checked.** Two layers, and the distinction matters. Whether the
declaration is *true* is tested by `tests/test_precision_execution_matrix.py`
(host) and `tests/test_precision_gpu_pipeline.py` (device), which execute the
claims. Whether the declaration and the registry *agree* is
`tests/test_registry_matches_capabilities.py`, which is an equality check per
component, in both directions:

> registry wider than the capability model = a claim nothing has executed […];
> registry narrower = a validated capability the graph planner will refuse to
> use, so the work of validating it was wasted.

#### 3.1.b Capability *guidance* — prose, advisory

**Definition.** Human- and agent-readable statements about what the solver is
*good for* and what must not be assumed of it. Not machine-actionable, and
deliberately so: most of it cannot be expressed as a dtype set.

**Files.** `knowledge/solvers/<name>/capability_notes.md`, sections
`## Use Optiland for` (`:6`), `## Do not assume (per repository
scientific-contract requirements)` (`:54`), `## Not yet exercised in this
repository` (`:78`), `## Confirmed NOT trustworthy in the pinned version`
(`:144`); plus the flat routing card `knowledge/solver_cards/<name>.yaml` with
`agent_should_use_for` / `agent_should_not_assume`.

**The example worth citing**, `knowledge/solver_cards/optiland.yaml:41`:

```yaml
agent_should_not_assume:
  - exported ray weight is already a coherent field amplitude
```

No dtype table can express that, and it is exactly the confusion
`RayBundle.require_coherent()` later refuses with
`ContractCode.AMPLITUDE_IS_A_WEIGHT`. A second, from the same file: "the ray
count returned by `trace(num_rays=N)` equals `N`" — observed 16 requested → 817
returned.

#### 3.1.c The resolution rule

**When the two disagree, `core/capabilities.py` wins and the prose is the bug.**
This report recommends the repo adopt the two terms explicitly — *capability
declaration* and *capability guidance* — because "capabilities" alone currently
resolves to whichever file the reader opened first. See gap 1 (§8).

### 3.2 Conventions

**Definition.** The interpretation of the numbers the solver returns: units,
axes, frame, handedness, phasor sign, normalization, reference plane. Not
negotiable per call. A mistake here does not raise; it produces a plausible wrong
answer.

**Files.** `knowledge/solvers/<name>/conventions.md` (Optiland's is 534 lines /
21 sections; Chromatix's 361 / 16) plus the machine-readable mirror in
`solver_card.yaml` for the conventions a consumer must act on.

**Worked example — `RealRays.opd`**, the best one in the repo, established by
CHE-30 and extended by CHE-41
(`knowledge/solvers/optiland/solver_card.yaml:206+`):

```yaml
opd_convention:
  status: verified
  field: RealRays.opd
  quantity: absolute_optical_path_length
  is_relative_to_chief_ray: false
  index_weighted: true
  unit: mm
  weighting_medium: material_pre     # the medium BEFORE each surface weights that segment
```

The card states what the name would otherwise imply: "**NOT** an OPD relative to
a chief ray, which is what the name suggests" — tested off axis, where at
`Hy = 0.2` the chief ray's own OPL is 11051.3 waves rather than zero.

What makes this architecture rather than trivia is the reference plane
(`:236-242`). For an infinite object the accumulator is seeded at
`positions[1] - (EPD - min(positions[1:-1]))`, so:

```yaml
  depends_on_aperture: true
  piston_is_aperture_dependent: true
```

> changing the aperture moves the OPL zero, so an absolute Optiland OPL is only
> meaningful at a declared EPD.

CHE-41 then measured the plane's *orientation*: `is_perpendicular_to_z: true`,
`is_a_wavefront_of_the_incoming_bundle: false`, omitted term
`n_object * (d0 . r_launch)`, which is

> linear in the LAUNCH coordinate, so a piston on axis (exactly constant, span
> 0.0) and the entire convergence tilt off it.

A consumer that removes a piston is therefore correct on axis and wrong by the
whole tilt off it — measured as a 209 µm error in where the reconstructed wave
converges. That is a convention fact with a downstream numerical consequence, and
it is why conventions get their own layer rather than living in a docstring.

**Where conventions become project constants.** Some conventions are not per
solver but frozen for the whole system, in `core/boundary.py:80-85`:

```python
PHASOR = "exp(-i omega t)"
SPATIAL_FACTOR = "exp(+i k z)"
AXIS_ORDER = "(y, x)"
ORIGIN_RULE = "array index n//2 is coordinate zero"
HANDEDNESS = "right-handed"
PROPAGATION_AXIS = "+z"
```

> These are string constants rather than free-form metadata precisely so that a
> mismatch is an equality failure with a named code, not a silently accepted
> variant spelling.

And where a convention is genuinely unknown, it is refused rather than defaulted
(`contracts.py:87-92`): `UNVERIFIED = "unverified"`, because "A wrong OPL
*reference* is a harmless piston; a wrong OPL *sign* conjugates the wavefront
[…]. Those two are indistinguishable downstream."

**The sign trap, in one line.** Optiland's own wavefront code reports
`opd_ref - opd` (chief-minus-ray), the reverse of this repo's evaluator
convention: "A consumer that mixes the two conjugates the wavefront"
(`solver_card.yaml::opd_convention.wavefront_sign_note`).

### 3.3 Constraints

**Definition.** The boundary of the validated envelope. Two distinct things, and
the distinction is load-bearing: what is **known false or hazardous**, and what
has simply **never been tried**. Collapsing them turns an untested path into an
implied guarantee.

Fields in `solver_card.yaml`, each with a real example:

- **`package_reported_unverified_regimes`** — the package claims it, we have not
  seen it. Optiland: "the NumPy backend on any device other than 'cpu' --
  `set_device` raises `BackendCapabilityError` there"; "float16 anywhere.
  Refused, not promoted -- geometry, OPL and direction cosines all accumulate and
  an OPL here spans ~1e4 waves". Chromatix: "TPU execution. `jax.devices()`
  reports no TPU on this host and none has ever been attempted; **the package
  claims one and this project does not**."
- **`not_yet_probed`** (`optiland/solver_card.yaml:473`) — the honest unknowns:
  "TPU. Never attempted"; "an off-axis field in x, or in both axes at once
  (CHE-41 used y only and did so deliberately, because a one-axis field is what
  makes the transpose control non-vacuous)".
- **`construction_hazards`** — correct-looking calls that build something else:
  "`GeometryFactory.create` silently DISCARDS kwargs that are not fields of the
  selected geometry config"; "`grating_period` shares the wavelength's unit (um),
  not the geometry's (mm) -- a 1000x trap"; "`Material('SK1')` returns SK16".
- **`known_defective`** (`chromatix/solver_card.yaml:145`) — present in the
  package, wrong in the pinned commit. `high_na_ff_lens` is not
  sampling-independent: refining only the pupil sampling moved the `|E_z|` ring
  radius from 246 to 2536 nm against a Richards-Wolf oracle converged to 2e-14,
  root-caused to `s_z` being derived from `field.f_grid * lambda / n` instead of
  the pupil position grid.
- **`inadmissible_sources`** — the strongest form, an outright refusal.
  `surface_type: paraxial` is not an admissible OPL source: the interaction model
  subtracts `(x^2+y^2)/(2f)` but leaves the direction un-normalized, so the
  intended cancellation never happens and OPL to focus is not flat — measured at
  0.36 mm, ≈655 waves at 550 nm. "Use real refractive surfaces for any wavefront
  handed to a coupler."
- **`derivative.verified: false` *with a passing probe*** — the single best
  illustration of how this layer is meant to be written. One directional-
  derivative probe passed at 1.11e-03; CHE-57 root-caused that number as float32
  finite-difference cancellation noise and showed the O(eps²) convergence a
  correct reverse-mode gradient must have (6.24e-05 → 6.24e-07 → 6.28e-09 under
  `set_precision('float64')`). `verified` still stays `false`:
  > verified stays false because this is still NOT the full repository gradient
  > test […] and it covers only this one parameter/objective pair -- not
  > wavefront/OPD export, not every surface type, not GPU.
- **`optimizer_hazard`** — a success flag that is not evidence:
  `OptimizerGeneric` returned `success=True` with message "CONVERGENCE: RELATIVE
  REDUCTION OF F <= FACTR*EPSMCH" at a point **1.97× worse than its start**.
  "`res.success` is not evidence: record `problem.sum_squared()` before and
  after."

### 3.4 Usage guidance

**Definition.** How to call it correctly, and what to do when it breaks.
Advisory by construction, and cheap to get wrong: a bad example costs an hour, a
bad convention corrupts a result.

- **`api_minimal_examples.md`** — numbered sections (Import → minimal forward →
  batched → gradient → serialization → error signatures → modern construction API
  → full-fidelity reproductions). What makes it trustworthy is the evidence rule
  in its header: every snippet was executed inside the `agent_solver` container
  and "Output values shown are real, captured on 2026-07-30, not illustrative."
  A real captured line, from the Optiland file `:31`:
  ```
  # rays.x.shape == (817,)  -- NOT 16; aperture/pupil sampling changes the count
  ```
  and from the Chromatix file `:38`:
  ```
  # asm.u.shape == (1056, 1056) for this pad; dx unchanged; power ~= 0.999997
  ```
- **`failure_guide.md`** — symptom-keyed, one section per *observed* failure, not
  per imagined one: "`Optic.draw3D()` never returns" (it hangs headlessly), "The
  torch backend silently runs in float32", "`use_czt=True` gives an amplitude 14x
  different from `use_czt=False`", "`pip install chromatix` installs the wrong
  package".
- **`probes/*.py` + `expected/*.json`** — the executable substrate the card
  cites; eight pairs for Optiland (`import_probe`, `raytrace_probe`,
  `gradient_probe`, `opd_convention_probe`, `exit_pupil_handoff`,
  `off_axis_opd_reference`, `standalone_baseline`, `system_construction_probe`).
  The rule: a card claim should name the probe that produced it, and
  `validated_probe_ids` is where that list lives.
- **`tutorials/`** — 41 Optiland + 16 Chromatix repo-owned reproductions of the
  frozen upstream tutorial scope, gated by `tests_tutorial/` (~33 min, opt-in by
  location). These test the *pinned dependency*, not this repo's physics, which
  is why they are not in the default suite.

### 3.5 The validation ladder

`knowledge/README.md:11-21` defines three values, and they are narrower than
"scientifically valid":

- `unvalidated` — planning only.
- `environment_verified` — "the exact package source/version, import, minimal CPU
  forward path, and recorded conventions passed in the supported container. This
  does not imply an analytic benchmark or verified gradient."
- `scientifically_validated` — the issue-specific analytic or independent oracle
  and required convergence checks also passed.

Plus the operating rule that makes `not_yet_probed` load-bearing rather than
decorative:

> Before unattended execution, require at least `environment_verified` and check
> the card's explicit `not_yet_probed` list against the intended task.

Current values: Optiland `environment_verified`
(`solvers/optiland/solver_card.yaml:497`); Chromatix `environment_verified` on
the flat card but `analytically_validated_scalar_asm_cpu` on the deep card
(`solvers/chromatix/solver_card.yaml:258`) — a value not on the ladder. See gaps
3 and 6 (§8).

## 4. Layer 3 — API / Adapter

**Definition.** The only code permitted to import the solver. It converts a typed
project request into solver calls, converts solver output into typed artifacts
with declared metadata, and fails structurally. It adds no physics.

**The convention that defines the layer** (`adapters/__init__.py`), which is also
what makes discovery possible without a hand-maintained list:

- file name `<solver>_adapter.py` (must end in `_adapter`)
- a module-level `MODEL_ID: str` naming the registered `ModelSpec.id`
- a module-level `get_adapter() -> ModelAdapter` factory
- lazy solver import inside `_import_<solver>()`

**The protocol** (`solvers/base.py:53`) is four methods and nothing else:

```python
class ModelAdapter(Protocol):
    """Adapter protocol; implementations may remain solver-native internally."""
    @property
    def spec(self) -> ModelSpec: ...
    def estimate(self, request: ModelRunRequest) -> CostEstimate: ...
    def validate_request(self, request: ModelRunRequest) -> ValidationReport: ...
    def run(self, request: ModelRunRequest) -> ModelRunResult: ...
```

Note the docstring's concession: "implementations may remain solver-native
internally." The adapter is a boundary, not a rewrite.

**Discovery** is by convention at `solvers/registry.py:22` — `_discover()` walks
`pkgutil.iter_modules`, keeps modules ending in `_adapter`, imports them and maps
`MODEL_ID → module`; `get_adapter_for_model()` raises `AdapterNotFoundError`.

**Typed contracts.** Both in-scope adapters carry a standalone typed
request/result pair with `ConfigDict(extra="forbid")`:
`OptilandRayRequest`/`OptilandRayResult`/`OptilandRayFailure`
(`optiland_adapter.py:280,307,298`) and
`ChromatixWaveRequest`/`ChromatixWaveResult`/`ChromatixWaveFailure`
(`chromatix_adapter.py:441,504,495`).

Worth noticing about `ChromatixWaveRequest`: `phasor`, `coordinate_frame`,
`origin`, `reference_plane` and `normalization` are **request fields**, not
assumptions. The conventions of §3.2 are asserted at the boundary and checked
against the frozen constants, so a mismatch is a named refusal rather than a
silent reinterpretation.

**Structured failure — three channels, deliberately different.**

| Channel | Used for | Form |
|---|---|---|
| raise | capability and dependency problems, decided *before* any solver call | `UnsupportedCapabilityError`, `AdapterDependencyError` (`core/errors.py`) |
| `ModelRunResult(status=FAILED, …)` | request validation and solver failure on the graph-facing path | `diagnostics={"code": …, "stage": …}` |
| typed failure object | the standalone baseline path, which never raises | `OptilandRayFailure` / `ChromatixWaveFailure` with `code`/`message`/`stage` |

`chromatix_adapter.py`'s module docstring states the split for its own two entry
points: `run` — "Capability and dependency failures *raise*; solver failures come
back as `ModelRunResult(status=FAILED, ...)`"; `run_standalone` — "It never
raises: every rejected capability, invalid convention/sampling value, unreadable
input, resource-estimate overrun, and solver failure is returned as a structured
`ChromatixWaveFailure`."

**The refusal example.** `HandoffPlaneError` (`optiland_adapter.py:550`):

> Carried as an exception rather than a sentinel so the caller cannot mistake an
> unresolved plane for one at z = 0. `run()` converts it to a structured failure;
> it is never allowed to reach the export.

and the diagnostic code behind it,
`OPTILAND_EXIT_PUPIL_UNRESOLVED`: `RunStatus.FAILED`,

> never a silent fallback to the image surface, which would be wrong by the whole
> pupil-to-focus distance with nothing to notice it by.

**What an adapter must not do.** Invent a field; promote a weight to an
amplitude; report a *requested* device or precision as the actual one
(`AGENTS.md`: "Never write a requested device or precision into an artifact: read
it off the array").

The exemplary case is honest *under*-population.
`optiland_adapter.py:2140` builds the wavefront artifact and declares what it
does not have (`:2185-2192`):

```python
"optical_path_length_source": (
    "RealRays.opd -- convention not independently verified "
    "(absolute optical path length vs. OPD relative to a "
    "chief/reference ray); see conventions.md."
),
"missing_declared_metadata": _MISSING_WAVEFRONT_METADATA,   # ["amplitude", "polarization", "pupil_mask"]
```

The consequence is intended: `WavefrontSamples.from_artifact_record` refuses that
artifact, and its docstring says so — "This *will* fail, by design, on an
unmodified Optiland wavefront artifact… The failure is the contract working, not
a defect."

**Adapter definition of done**, from `AGENTS.md`, annotated with the Optiland
file that satisfies each item:

| Requirement | Satisfied by |
|---|---|
| typed request and result contract | `optiland_adapter.py:280-330` |
| pinned version, one import probe, one minimal forward probe | `docker/requirements.txt:104`; `knowledge/solvers/optiland/probes/{import_probe,raytrace_probe}.py` |
| explicit conventions and supported devices/dtypes | `knowledge/solvers/optiland/conventions.md`; `core/capabilities.py:70-97` |
| one analytic or independently reviewed validation case | L1-RAY-01 free-space + paraxial oracles (CHE-17); `opd_convention_probe` against closed-form geometries |
| structured failure behavior | `OptilandRayFailure`, `HandoffPlaneError`, `UnsupportedCapabilityError` |
| a gradient test only when differentiability is claimed | not claimed: `derivative.verified: false` |

## 5. Worked example A — Optiland, file by file

`M_RAY_OPTILAND`. Every row is a real file; the last column is a real line from
it.

| Path | Layer | What it holds | One real line |
|---|---|---|---|
| `docker/requirements.txt:104` | Solver | the pin | `optiland==0.6.0` |
| `docker/Dockerfile:43` | Solver | the opt-in differentiability dependency | `pip install --index-url .../whl/cpu torch==2.13.0` |
| `knowledge/solver_cards/optiland.yaml` | Knowledge (routing) | 45-line card an agent reads *first*: role, install hazard, `agent_should_not_assume`, `required_probes`, pointer to the deep pack | `- exported ray weight is already a coherent field amplitude` |
| `knowledge/solvers/optiland/solver_card.yaml` | Knowledge (facts) | 511 lines of routing-critical, machine-readable fact: `supported_regimes`, `opd_convention`, `exit_pupil_handoff`, `derivative`, `not_yet_probed`, `validation_status` | `piston_is_aperture_dependent: true` |
| `knowledge/solvers/optiland/conventions.md` | Knowledge (conventions) | 534 lines / 21 sections: backend abstraction, units, torch float32 default, `RealRays.opd` | `## Torch backend precision defaults to float32 (CHE-57)` |
| `knowledge/solvers/optiland/capability_notes.md` | Knowledge (guidance) | "Use Optiland for" / "Do not assume" / "Not yet exercised" / "Confirmed NOT trustworthy" | `## Do not assume (per repository scientific-contract requirements)` |
| `knowledge/solvers/optiland/api_minimal_examples.md` | Knowledge (guidance) | 12 executed snippets with captured outputs | `# rays.x.shape == (817,)  -- NOT 16` |
| `knowledge/solvers/optiland/failure_guide.md` | Knowledge (guidance) | 448 lines, symptom-keyed, observed failures only | `## Optic.draw3D() never returns` |
| `knowledge/solvers/optiland/probes/` + `expected/` | Knowledge (evidence) | 8 probe/JSON pairs — the executable substrate every card claim cites | `opd_convention_probe.py` → `opd_convention_probe.json` |
| `knowledge/solvers/optiland/tutorials/` | Knowledge (dependency gate) | 41 reproductions, 459 declared checks; run by `tests_tutorial/` | `t10_differentiable_ray_tracing.py` |
| `core/capabilities.py:70-97` | Capability declaration | the authoritative device/dtype/namespace claim + its evidence string | `device_namespaces={DeviceKind.CUDA: frozenset({ArrayNamespace.TORCH})}` |
| `registry/models.yaml:53-54` | Registry (reflection) | the graph planner's view, held equal to the above by test | `devices: [cpu, gpu]` / `dtypes: [float32, float64]` |
| `registry/prescriptions.py` | Project data | canonical named optical systems, so "adding a system" is data not code | `M3-SINGLET-REF` |
| `solvers/optiland/adapter.py` | Adapter | `MODEL_ID` (`:204`), typed contracts, `run`/`run_standalone`, artifact export, structured failure | `MODEL_ID = "M_RAY_OPTILAND"` |
| `solvers/optiland/builder.py` | Adapter (support) | one function, `build_optiland_system`, from a prescription | `import optiland.backend` (in-function, `:76`) |
| `solvers/optiland/coherent_trace.py` | Adapter (support) | traces a *caller-supplied* ray population (CHE-70) | — |
| `tests/test_optiland_adapter.py` | Guard | 23 tests over the adapter contract | `test_cuda_rejected_eagerly_on_the_numpy_backend` |
| `tests/test_optiland_opd_convention.py` | Guard | the convention itself, with its falsifiers | — |
| `tests/test_optiland_canonical_prescriptions.py` | Guard | builder + prescription registry | — |

## 6. Worked example B — Chromatix, file by file

`M_WAVE_CHROMATIX`. Same shape; the interesting differences are that the pin is a
git commit, the capability is single-precision, and the package ships a component
we have measured to be defective.

| Path | Layer | What it holds | One real line |
|---|---|---|---|
| `docker/requirements.txt:57` | Solver | the pin — a commit, not a version, because of the namesquat | `chromatix @ git+…@d24bdf0022835bb8ce1cdcc6aeafbc7fcb39daee` |
| `knowledge/solver_cards/chromatix.yaml` | Knowledge (routing) | flat card; `pypi_name_is_a_namesquat: true`, `required_probes` with done/not-done | `- Airy-pattern analytic comparison  # not yet done` |
| `knowledge/solvers/chromatix/solver_card.yaml` | Knowledge (facts) | 275 lines: `install_hazard`, `precision_verdict_che61`, `known_defective`, `m3_pupil_to_focus`, `not_yet_probed` | `install_hazard: … an unrelated namesquat with no optics code` |
| `knowledge/solvers/chromatix/conventions.md` | Knowledge (conventions) | 361 lines / 16 sections; note the two entries that exist only because a name lies | `## kykx means two different things (CHE-57)` |
| `knowledge/solvers/chromatix/capability_notes.md` | Knowledge (guidance) | plus a dedicated defect section | `## Known defective: high_na_ff_lens (do not use for quantitative work)` |
| `knowledge/solvers/chromatix/api_minimal_examples.md` | Knowledge (guidance) | 12 executed sections incl. vector fields and the full-wave solver | `# component order is (E_z, E_y, E_x) -- the REVERSE of this project's convention` |
| `knowledge/solvers/chromatix/failure_guide.md` | Knowledge (guidance) | 363 lines; the first entry is the install trap | `## pip install chromatix installs the wrong package` |
| `knowledge/solvers/chromatix/probes/` + `expected/` | Knowledge (evidence) | incl. `m3_pupil_to_focus.py` — whose *test* was archived by CHE-67, so the probe still runs but nothing guards the convention | `tests: … ARCHIVED by CHE-67 and not runnable` |
| `knowledge/solvers/chromatix/tutorials/` | Knowledge (dependency gate) | 16 reproductions, 205 declared checks | — |
| `core/capabilities.py:100-135` | Capability declaration | FP32-only, `lossy_input_dtypes={COMPLEX128}` — the FP32 floor of the whole stack | `precisions=frozenset({Precision.FP32})` |
| `registry/models.yaml:146-147` | Registry (reflection) | `tpu` deliberately absent: "nothing has executed there" | `devices: [cpu, gpu]` / `dtypes: [complex64]` |
| `solvers/chromatix/adapter.py` | Adapter | `MODEL_ID` (`:159`), lazy import (`:220`), conventions as request fields, `WaveHandoffError` | `MODEL_ID = "M_WAVE_CHROMATIX"` |
| `solvers/chromatix/carrier_removed_asm.py` | Adapter (support) | carrier-removed exact ASM over Chromatix machinery (CHE-40) | — |
| `tests/test_chromatix_adapter.py` | Guard | 20 tests over the adapter contract | — |
| `tests/test_carrier_removed_asm.py` | Guard | the carrier-removed path | — |

**Files that look like adapters and are not.**
`adapters/optiland_benchmark_adapter.py`, `adapters/chromatix_benchmark_adapter.py`
and `adapters/chromatix_scaling_adapter.py` match the `*_adapter.py` filename
pattern and are therefore *imported* by `_discover()`, but they declare no
`MODEL_ID`, so they register nothing. They are benchmark harnesses. Renaming them
would make the convention self-evident; that is a cleanup, not a defect.

## 7. Adding a new solver — the layering as a procedure

Ordered. Each step's output is the next step's input, and the registry entry is
**last**, never first.

1. **Pin it** in `docker/requirements.txt` (version *or* commit — commit if the
   name is ambiguous anywhere), and record any install hazard.
2. **Import probe** → `knowledge/solvers/<name>/probes/import_probe.py` +
   `expected/import_probe.json`.
3. **Flat routing card** `knowledge/solver_cards/<name>.yaml` at
   `validation_status: unvalidated`, with `agent_should_not_assume` and
   `required_probes`. The solver is now planning-only, and nothing may execute it
   unattended.
4. **Conventions probe** — units, axes, sign, normalization, reference plane —
   then write `conventions.md` and the machine-readable mirror in
   `solver_card.yaml`. Unknowns are recorded as unknown, not defaulted.
5. **Minimal forward probe** + captured `expected/*.json`, then
   `api_minimal_examples.md` with the real outputs pasted in.
6. **Capability probe** under `benchmarks/probes/precision/` — devices,
   precisions, and what the package *computes* in as opposed to *accepts*.
7. **`ComponentCapabilities`** in `core/capabilities.py`, with a non-empty
   `evidence` naming the probe, image and device.
8. **`registry/models.yaml`** — only now, and only once
   `./run.sh pytest -q tests/test_registry_matches_capabilities.py` passes.
9. **Adapter** `adapters/<name>_adapter.py`: `MODEL_ID`, `get_adapter()`, lazy
   import, typed request/result, structured failure, artifact metadata read off
   the arrays.
10. **Adapter test** `tests/test_<name>_adapter.py`, including the negative paths
    (missing dependency, refused capability, invalid request).
11. **`failure_guide.md`** from what actually broke while doing steps 2-10.
12. **Promote `validation_status`** to `environment_verified`, and only to
    `scientifically_validated` once an analytic or independent oracle passes.

## 8. Gaps and recommendations

Each item: what, the evidence, and the options. All were verified while writing
this report, and none was fixed *by* it. Five of the six have since been closed
by CHE-84's phases; the status line on each says which.

| Gap | Status | Closed by |
| -- | -- | -- |
| 1 — "capabilities" names two layers | open | CHE-92 renames `capability_notes.md` |
| 2 — device/dtype claims in three places | **partially resolved** | CHE-87 closed the registry half; the card half is CHE-92 |
| 3 — two knowledge tiers, already drifted | open | CHE-92 collapses them to one card |
| 4 — a stale `not_yet_probed` entry | open | CHE-92 |
| 5 — the three tiers disagree on scope | **resolved** | CHE-87 |
| 6 — `validation_status` has no executable meaning | open | CHE-92 constrains it to the ladder |

The pattern in what closed and what did not is worth naming: the gaps that were
about **executable declarations** were closable by deleting or enforcing, and
the four still open are all about **prose that restates an executable
declaration**. That is one problem with four faces, and CHE-92 addresses it by
removing the restatement rather than by policing it.

### Gap 1 — "capabilities" is one word for two layers

**What.** `core/capabilities.py` (executable, authoritative) and
`knowledge/solvers/*/capability_notes.md` (prose, advisory) share a name and have
no cross-reference beyond a comment.

**Evidence.** `capabilities.py` never mentions the knowledge packs;
`capability_notes.md` never mentions `capabilities.py`. The only link is a comment
inside `solver_card.yaml`.

**Options.** (a) Adopt *capability declaration* vs *capability guidance* as
project vocabulary and add one pointer line to each file — cheap, no code change.
(b) Rename `capability_notes.md` to `usage_notes.md`. (a) is enough.

### Gap 2 — device/dtype claims live in three places; only two are enforced — **PARTIALLY RESOLVED (CHE-87)**

CHE-87 closed the half of this that was a *scope* problem: there is no longer a
registry entry without a capability declaration, and the equality check now
covers every entry rather than an `_OWNED` subset. What remains is the third
place — `knowledge/solvers/*/solver_card.yaml`'s restated `devices_tested` /
`dtypes_validated_for_m1` / `precision_verdict_che61` tables, still unchecked.
CHE-92 demotes them to pointers, which removes the drift surface instead of
policing it.

**What.** The same claim is written in `core/capabilities.py`, in
`registry/*.yaml`, and in `knowledge/solvers/*/solver_card.yaml`. Only the first
pair is enforced.

**Evidence.** `tests/test_registry_matches_capabilities.py` asserts equality
between `capabilities.py` and the registry, per component. Nothing checks
`solver_card.yaml`'s `devices_tested`, `dtypes_validated_for_m1` or
`precision_verdict_che61` — even though the YAML itself concedes authority
(`optiland/solver_card.yaml:176-179`: "the authoritative machine-readable table is
`core/capabilities.py`, which `tests/test_registry_matches_capabilities.py` holds
the registry to").

**Options.** (a) Demote the YAML fields to pointers, deleting the restated lists.
(b) Extend the test to cover the cards. (a) is smaller and removes the drift
surface rather than policing it.

### Gap 3 — two knowledge tiers, no consistency check, and they have already drifted

**What.** Every solver has a flat routing card *and* a deep pack card. The
duplication is deliberate ("This supplements, and does not replace,
`knowledge/solver_cards/optiland.yaml`") but unguarded, and three disagreements
exist today.

**Evidence.**

1. `knowledge/solver_cards/optiland.yaml:48` lists
   `- wavefront/OPD export convention  # not done`, while
   `knowledge/solvers/optiland/solver_card.yaml:206` records
   `opd_convention: {status: verified, verified_by: CHE-30}`. These may be
   different claims — the *export* path versus the accumulator convention — which
   is exactly why one disambiguating line is needed.
2. `knowledge/solvers/optiland/capability_notes.md:75` still says: "That
   `RealRays.opd` is absolute OPL or piston-removed OPD. Its reference and sign
   remain unverified" — contradicted by CHE-30, which established
   `quantity: absolute_optical_path_length` and the sign.
3. Chromatix's `validation_status` is `environment_verified` on the flat card and
   `analytically_validated_scalar_asm_cpu` on the deep card.

Note the asymmetry: `tests/test_coupler_knowledge_pack.py` guards evidence honesty
for *couplers*. There is no solver equivalent.

**Options.** (a) Add `tests/test_solver_knowledge_pack.py` mirroring the coupler
test. (b) Collapse the flat cards into a generated index. (a) first — the drift is
already real, and it is in the direction of *understating* what has been verified,
which wastes work.

### Gap 4 — a stale `not_yet_probed` entry, which is a live defect

**What.** Optiland's card still lists the GPU suite as outstanding.

**Evidence.** `knowledge/solvers/optiland/solver_card.yaml:477`: "the gpu-marked
suite since CHE-60. It is expected to pass and has not been re-run; a dedicated
`./run.sh --gpu pytest -q -m gpu` pass is outstanding." `AGENTS.md` records
CHE-72/CHE-73 revalidating it on 2026-08-20 — 48 passed in 70 s on one RTX A6000
— explicitly "cleared CHE-60's outstanding dedicated pass".

**Why it matters.** `knowledge/README.md` makes `not_yet_probed` a gate on
unattended execution. A stale entry there is not cosmetic: it either blocks work
that is in fact validated, or trains readers to discount the list.

**Options.** Update the entry and add "check the affected cards' `not_yet_probed`"
to the closing checklist of any issue that validates a capability.

### Gap 5 — the three tiers disagree on what is in scope — **RESOLVED (CHE-87)**

**What it was.** Out-of-scope solvers were absent from the capability layer,
present at the Knowledge layer, and still *reachable* at the adapter layer.
`adapters/fmmax_adapter.py` (`MODEL_ID = "M_RCWA_FMMAX"`) and
`adapters/fdtdx_adapter.py` (`M_EM_FDTDX`) were discovered by
`solvers/registry.py`'s filename scan, while their tests had been archived by
CHE-67 and `capabilities_for("M_RCWA_FMMAX")` already raised
`CapabilityError(code="UNKNOWN_COMPONENT")`. `registry/models.yaml` declared
`devices: [cpu, gpu, tpu]` for `M_RCWA_FMMAX`, `M_EM_FDTDX` and
`M_SENSOR_IDEAL` — precisely the "claim nothing has executed" that
`tests/test_registry_matches_capabilities.py` exists to stop, and which escaped
it only because `_OWNED = sorted(COMPONENT_CAPABILITIES)` scoped the test to the
four declared components.

**How it was resolved.** Neither of the recorded options. Option (a) would have
built a mechanism to describe components that do not exist, and option (b) would
have corrected a false device claim while leaving the entry that made the claim.
CHE-87 deleted instead, atomically across all six surfaces in one commit —
adapter, registry entry, example graph, knowledge pack, pytest marker,
dependency pin — because a partial retirement is worse than none: code removed
while a registry claim survives reads to a planner as a supported capability.

Three things now hold the resolution rather than a convention holding it:

* `solvers/registry.py` is an **explicit map**, so a filename can no longer
  imply a registration and a duplicated `MODEL_ID` raises instead of being
  resolved by directory order.
* `test_registry_declares_no_component_without_a_capability` drops the `_OWNED`
  exemption and asserts the registry contains **no** entry outside
  `COMPONENT_CAPABILITIES`. The remedy it names is asymmetric on purpose: add a
  declaration backed by an executable probe, or delete the entry — never a
  placeholder.
* `test_no_registry_entry_is_still_experimental` makes `maturity` carry
  information. All 17 entries were `experimental`, which is a constant, not a
  classification. The four survivors are `characterized`.

The intent behind the deleted components is preserved, marked non-executable, in
`benchmarks/roadmap.md`, together with the two findings worth keeping: FMMAX's
unresolved phase/sign convention (energy closed to ~1e-7, the complex
amplitude's sign did not match the Fresnel convention), and JAX-FEM's GPLv3
licence against this project's MIT.

### Gap 6 — `validation_status` has no executable meaning for solvers

**What.** It is a free string. Two of the eight cards carry values that are not on
the ladder at all: `analytically_validated_scalar_asm_cpu`
(`solvers/chromatix/solver_card.yaml:258`) and, on the coupler side,
`characterized_stochastic_scalar_planar_cpu`
(`knowledge/couplers/wave_to_ray/coupler_card.yaml:358`).

**Evidence.** `knowledge/README.md:11-18` defines exactly three values. Nothing
validates the field for solvers.

**Options.** (a) Constrain it to the three values and move the qualifier into a
separate `validation_scope` field — the qualifiers are genuinely informative
("scalar ASM, CPU") and should not be lost, just separated from the ladder value.
(b) Declare the field advisory. (a) preserves the information and restores the
ladder's meaning.

## 9. What sits above this tier

One tier up, the system stops being about packages and starts being about
handoffs. Named here only so this report's boundary is explicit; each is a
follow-up:

- **Artifacts** — `core/boundary.py` holds `RayBundle`, `WavefrontSamples`,
  `ComplexField` and `PSF` as frozen dataclasses with required declarations,
  replacing the per-adapter `metadata: dict[str, Any]` that preceded them. Its
  three rules ("A missing declaration is an error, never a default"; "An
  unverified quantity may be carried, but never reinterpreted"; "Adapter output is
  not changed to suit the contract") are what make §4's honest under-population
  work.
- **Couplers** — a parallel tier with the same layering and their own knowledge
  packs (`knowledge/couplers/<direction>/`), documented separately "because a
  coupler changes *representation*, so it carries physical assumptions that belong
  to neither solver it joins."
- **Bridge negotiation** — `core/precision.py::plan_bridge` is the pure function
  that decides how one component's output may legally enter another's
  `ComponentCapabilities`. It is the consumer that makes §3.1.a's four dtype sets
  pay for themselves.
- **The graph** — `registry/*.yaml` + `core/specs.py` + `core/graph.py`.

Planned follow-up: `docs/architecture/coupler_layering.md`.
