# `knowledge/` — versioned measured evidence

Shared **data**, cited by code that must not own it. One rule:

> A record here is the single source for the measurement it carries. No package
> under `src/` keeps a copy.

Landed by CHE-223 (R03.6) with one kind of record.

## `capabilities/` — measured component capability

One JSON file per component, named by its component id, so "which file is
canonical for this id" needs no index. `numerics.knowledge.load_capabilities`
reads one into a validated `numerics.ComponentCapabilities`, through the same
`__post_init__` — all ten widening refusals — that an in-tree declaration went
through. A record wider than its probe is refused at load with
`INVALID_CAPABILITY_DECLARATION`.

Present today:

| file | what it measures |
| -- | -- |
| `M_RAY_OPTILAND.json` | optiland 0.6.0: CPU + CUDA, FP32/FP64, real dtypes, CUDA only through the torch backend |
| `M_WAVE_CHROMATIX.json` | chromatix 0.6.0: CPU + CUDA, FP32 only, `complex64`; `complex128` ingestible but lossy |

Every field is required and every unknown field is refused, so a typo cannot
become a silently defaulted value and a stale key cannot linger unread.
`schema_version` is validated rather than decorative: the loader refuses a version
it does not read, and a test proves the refusal.

`probe` cites a path under `benchmarks/probes/` and `probe_tag` is the git tag it
resolves against. `benchmarks/probes/precision/` is not in the working tree — the
greenfield rewrite deleted it — and both records resolve at
`pre-rewrite-2026-08-30`, which `tests/knowledge/test_capability_pack.py` checks
with `git cat-file -e`. **Widening a row costs a probe re-run against the pinned
image, not a re-reading of the packages' documentation.**

### Two records, not seven

The reference implementation declared seven. Five of them described couplers and
operators whose capability nobody had measured — a coupler's capability is set by
what its shared implementation is written against, so the ticket that measures one
declares it with its own evidence. A record for unwritten code is the failure this
pack exists to prevent.

### Component-level, and why the id shape allows otherwise

Both records are **component-level**, because that is what the probes measured:
they exercised the packages' device and dtype behaviour, not any one semantic
operation. An `operations.OperationDescriptor` cites the component id its
implementation executes within, and several descriptors may cite one record —
`S_RAY_OPTILAND` and `S_RAY_OPTILAND_BUNDLE` both cite `M_RAY_OPTILAND`.

Operation-level records are permitted **when independently measured**, which is why
`operations.descriptors._COMPONENT_ID` constrains the shape without demanding an
`M_` prefix. Do not duplicate a component row per descriptor merely because several
cite it: that is the second source this pack removes.

## The card rule this pack replaces

At `pre-rewrite-2026-08-30` this pack was `knowledge/<kind>/<name>/card.yaml` plus
prose, and its README said:

> A card does **not** restate a device or dtype table. `core/capabilities.py` owns
> those … a third copy in prose could only ever drift.

**That rule is superseded and the reason is worth keeping**, because its *intent*
is exactly what this directory preserves. The old rule assumed the measured table
lived in code, so a card restating it would have been a second copy. The direction
is now reversed: `knowledge/capabilities/` holds the rows, `numerics/` holds only
the contract and keeps none of the data, and there is still exactly one copy.

What has not changed: **a card carries consequences, not measurements.** These
files are data read by a loader, not prose. Neither component has a card, and
adding one that restated a dtype table would recreate the drift the old rule
named.

## `capabilities/` is not YAML, deliberately

PyYAML left `pyproject.toml` at R02.1, with the four other dependencies nothing
under `src/` imported; CHE-223 removed the `types-PyYAML` stub that outlived it.
`json` is stdlib and is what `benchmarks/systems/records/*.json` already uses.

## Format, and where it is resolved from

`numerics.knowledge.KNOWLEDGE_ROOT` resolves this directory from
`src/numerics/knowledge.py`, i.e. relative to the repository root. That is a
decision, not an accident: the project is installed `pip install --no-deps -e .`
against a mounted checkout, and a real wheel would not ship a repository-root
directory. Package data under `src/numerics/` was rejected because that would make
one package the owner of a shared pack again. A missing directory fails naming the
path it tried.
