# `benchmarks/` — system-level runs and their records

Established by **CHE-212 (R06.7)**, the first entry in the new tree. This is the
minimal layout that ticket needs; it is *not* a revival of the reference
implementation's family / instance / verifier machinery, and R14.3 (CHE-203) owns
the disposition of the old `benchmarks/` tree.

## What a benchmark is here

A **runnable script** that composes several of this project's public primitives
into a complete optical system, checks the result against **closed-form** optics,
demonstrates that a set of **negative controls** each break a named gate, and
writes a **record** of what it measured.

Three properties, in decreasing obviousness:

1. **It uses the public vocabulary and calls no backend directly.** `sources`,
   `operators`, `solvers.chromatix` — never `chromatix.functional`. A benchmark
   that reaches past the boundary is testing the backend, not this project.
2. **The composition happens in the script, not in `src/`.** The dependency
   allowlist forbids `operators/ -> solvers/`, so no production package can hold
   a graph that uses both. A benchmark is therefore four public calls in order.
   Do **not** add a `systems/` package, a `Pipeline`, or a composite operator to
   make it read better: the composition layer is R12/R13's design question
   (CHE-164 / CHE-165), and the current awkwardness is evidence for it.
3. **Only a closed form may gate.** Every comparison carries an `oracle_kind` —
   `closed_form` or `diagnostic` — and a `diagnostic` entry is evidence that may
   not decide anything. AGENTS.md's rule is that repository numerical code must
   not be the sole correctness oracle for the same numerical code, and a
   differential check between two of our own numerical paths is exactly that.

## Layout

```
benchmarks/
  README.md                          this file
  record.py                          gate(), control(), write_record()
  systems/
    b4f_ideal.py                     CHE-212 (R06.7) -- the ideal coherent 4f relay
    b_fourier_ptychography.py        CHE-213 (R06.8) -- the FP forward model
    records/
      B-4F-IDEAL-<configuration>.json
      B-FP-FORWARD-<configuration>.json
```

One module per system, one record per configuration, records beside the script
that writes them. `benchmarks/observables.py` holds the project's single
`intensity()` path — see "The intensity path" below.

## How it is invoked

```
./run.sh python -m benchmarks.systems.b4f_ideal
./run.sh python -m benchmarks.systems.b_fourier_ptychography
make benchmarks          # both of the above, in order
```

`python -m` and not a path, because the scripts import `benchmarks.record`. Each
exits non-zero if any gate fails or any negative control fails to break the gate
it names, and rewrites its records in place.

**Benchmarks are not part of the pytest suite.** `pyproject.toml` sets
`testpaths = ["tests"]`, so `pytest` does not collect them, and they cost minutes
where the default gate costs seconds. What *is* in the default suite is
`tests/benchmarks/test_records.py`: a structural check that the committed records
claim what they should — every gate passed, at least one gate is closed-form,
every closed-form gate carries a tolerance, and every negative control broke the
gate it names — and that the scripts import no backend directly. A `diagnostic`
gate is permitted and must be labelled; what is not permitted is a record with no
closed-form gate at all, which would be a benchmark certifying itself. That test
reads JSON and parses two ASTs; it does not re-run anything.

**A record therefore goes stale when the code it measured changes.** Re-run the
script and commit the regenerated record in the same change. The `written_at_utc`
field moves on every run; that churn is deliberate, because a record whose
timestamp never changes is a record nobody re-ran.

## The record format

A flat JSON document per configuration:

| field | meaning |
| --- | --- |
| `written_at_utc`, `environment` | when, and the installed versions of numpy / scipy / jax / chromatix |
| `benchmark`, `ticket`, `configuration`, `produced_by` | which run this is |
| `composition` | the public calls, in order — the claim that criterion 1 makes |
| `parameters` | everything a re-run needs, including the derived sampling at each plane |
| `not_covered` | what this benchmark does **not** measure, stated rather than implied |
| `gates[]` | `name`, `oracle`, `oracle_kind`, `measured`, `expected`, `tolerance`, `tolerance_basis`, `passed` |
| `negative_controls[]` | `name`, `changed`, `breaks_gate`, `measured`, `reference`, `broke_the_gate` |

`tolerance_basis` is required and is the field that keeps this honest: a
tolerance with no stated derivation is a fitted one, and AGENTS.md forbids
widening one to make a benchmark pass.

There is deliberately **no** top-level pass/fail summary (the entries carry it,
and a second copy could disagree), **no** git commit, and **no** environment
fingerprint beyond package versions. Execution provenance is R13's subject and
half of it now would be a second place to change later.

## The intensity path

`benchmarks/observables.py::intensity` is the **only** `|U|^2` in the tree.
R11 (CHE-163) has not landed `measurements/`, so CHE-213's forward model computes
its observable locally and says so — which is what that ticket permits. When
`measurements/` lands, this function moves there and this file is deleted; what
must not happen is two production intensity implementations, which is the same
rule R11 applies to PSF. `tests/benchmarks/test_records.py` counts the
implementations.

## What is not here

No agentic or planned execution path. No reconstruction, optimization or
inverse problem. No family/instance registry, no verifier stack, no plotting.
No GPU requirement: both benchmarks are CPU and take about three seconds each. If a case
needs a grid that makes that false, shrink the case.
