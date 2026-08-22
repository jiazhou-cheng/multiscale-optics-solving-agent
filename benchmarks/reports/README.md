# Milestone reports — 2026-08

The narrative record of what each milestone measured, in the state it was
written. Their numbers stand and are not revised; where a later issue changed
the conclusion, the later report says so rather than the earlier one being
edited.

| Report | What it records |
| -- | -- |
| `2026-08/ray_and_wave_baselines.md` | the M1 ray and wave baselines, their independence check, and the limitations carried into M2 |
| `2026-08/coupler_characterization.md` | the M2 bidirectional coupler protocol and its measured tolerances |
| `2026-08/ray_to_wave_slice.md` | the M3 end-to-end slice: Optiland → `C_RAY_TO_WAVE` → Chromatix → PSF |
| `2026-08/ray_to_wave_slice_exit.md` | the M3.5 exit review, including which gates were *not* met |
| `2026-08/sensor_handoff_convergence.md` | the sensor-side handoff, measured against ray count |
| `2026-08/slice_cleanup_disposition.md` | which M3/M3.5 findings were acted on, and which stand open |
| `2026-08/metalens_bridge.md` | the GPU-only coherent wave→ray→wave round trip at scale |
| `2026-08/agent_benchmark_v1.md` | the V1 agent benchmark's design and its graded baseline |
| `2026-08/cooke_triplet_psf_routes.md` | comparing Optiland's PSF routes, and why two of them are not two oracles |

Filenames say what the report is about rather than which ticket commissioned it.
Ticket IDs remain *inside* these documents, which is where they belong: a
historical record should say who established what.

## Where the evidence actually is

Three different things get called evidence here, and only two of them are in
this repository.

**Committed records** — `benchmarks/probes/records/*.json`. Small canonical
measurements: metrics, hashes, tolerances. These resolve for every clone, and
active tests read them.

**The numbers in the reports themselves.** Every claim a report makes is stated
in the report. You do not need the artifacts to read the result.

**`outputs/` is local-only.** It is gitignored, roughly 510 MB on the machine
that produced it, and **does not exist in any other clone**. Every `outputs/…`
path in these reports names an artifact on one filesystem — an `arrays.npz`, a
`plot.png`, a per-run `provenance.json`. Treat such a path as *"this is where it
was written"*, never as a file you can open.

To regenerate one, run the command the report gives. Where the command names a
suite under `archive/benchmarks/gen1/`, that suite was archived by CHE-88 and is
**not runnable** — the command is preserved as the record of what was run, not
as an instruction. `archive/benchmarks/gen1/README.md` says what each archived
suite guarded and what is unguarded now.

Large arrays and figures stay out of git deliberately. What should never happen
again is a bare `outputs/` path implying an artifact the reader can open, which
is why this section exists rather than a note per citation.
