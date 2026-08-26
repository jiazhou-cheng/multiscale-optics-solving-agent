# Commercial catalog lens systems — applied builder + ray-trace benchmark

CHE-139 (M1.1.5). An applied benchmark for one end-user workflow:

> given real commercial optical components and their manufacturer documentation,
> can this repository construct a meaningful optical system through the shipping
> builder, ray trace it, and produce physically meaningful characterization —
> without hand-written benchmark-specific modeling code?

It is **not** an optimization task, a coupler task, or an optical-design
benchmark. It is a sanity benchmark for the optical-system builder plus the
ray-tracing execution path, and it decides nothing: there is no tolerance, no
gate and no verdict anywhere in it. Every comparison it reports comes with the
basis on which it is comparable, so a reader judges it.

The chain under test, end to end:

```
Newport product page + vendor Zemax file        catalog_sources.py
  -> normalized ComponentSpec                   core/optical_assembly.py
    -> ordering / orientation / air gaps        benchmark_systems.py
      -> ONE OpticalSystemSpec                  core/optical_assembly.py
        -> build_optiland_system                solvers/optiland/builder.py   (shipping)
          -> OptilandAdapter.run                solvers/optiland/adapter.py   (shipping)
            -> characterization + records        run_benchmark.py
```

No step is bypassed. The benchmark constructs no `Optic` of its own, contains no
ray tracing and no optical physics, and reaches the solver only through an
ordinary `ModelRunRequest` carrying a canonical prescription in
`config['prescription']`.

## Reproducing it

```bash
# primary records, GPU (Optiland torch backend on CUDA)
MOA_GPUS=device=6 ./run.sh --gpu python \
  benchmarks/applied/commercial_lens_systems/run_benchmark.py run \
  --backend torch --device cuda --dtype float64 --tag gpu

# lightweight CPU backend check on the representative multi-lens case
./run.sh python benchmarks/applied/commercial_lens_systems/run_benchmark.py run \
  --backend numpy --device cpu --dtype float64 --tag cpu \
  --systems S4_PAC052_KBX058_TANDEM

# GPU vs CPU behaviour equivalence on the declared scalar set
./run.sh python benchmarks/applied/commercial_lens_systems/run_benchmark.py compare \
  --primary  benchmarks/applied/commercial_lens_systems/records/S4_PAC052_KBX058_TANDEM.gpu.json \
  --secondary benchmarks/applied/commercial_lens_systems/records/S4_PAC052_KBX058_TANDEM.cpu.json
```

Machine-readable records land in `records/`, spot diagrams in `figures/`, and the
raw per-trace `rays.npz` under `outputs/che139_commercial_lens_systems/<tag>/`
(gitignored; each record carries the file's sha256 and the adapter's
`scientific_array_sha256`).

Requesting `--device cuda` outside the GPU image is refused, not downgraded:

```
$ ./run.sh python .../run_benchmark.py run --backend torch --device cuda --tag x
[che139] REFUSED: the requested execution environment (backend='torch', device='cuda',
         dtype='float64') is not available here; the benchmark stops rather than
         falling back to another device
exit 2, records/execution_refusal.x.json written
```

The refusal quotes the adapter's own `OPTILAND_CUDA_UNAVAILABLE` diagnostic and
sets `"fallback_performed": false`. There is no code path from a CUDA request to
a CPU trace.

## The components

Newport (MKS Instruments) was chosen because every part below publishes a
complete prescription **twice**: once in the product page's specification table
and once in the vendor's own Zemax design file, linked from the same page. The
second source independently confirms the radii, thicknesses, glasses, surface
*order* and entrance-pupil diameter — and where the two disagree, the
disagreement is recorded rather than resolved silently.

| Part | Type | Ø | EFL | BFL | Prescription | Source |
| --- | --- | --- | --- | --- | --- | --- |
| [KPX094](https://www.newport.com/p/KPX094) | plano-convex singlet, N-BK7 | 25.4 mm | 100 mm | 96.97 mm | R1 = +51.680, plane; t = 4.585 | page + [Zemax](https://api.p1.mks.com/medias/sys_master/images/images/hb8/h05/8797167124510/KPX094-ZEMAX.zip) |
| [KBX058](https://www.newport.com/p/KBX058) | equiconvex singlet, N-BK7 | 25.4 mm | 75.6 mm | 73.89 mm | R1 = +77.265, R2 = −77.265; t = 5.102 | page + [Zemax](https://api.p1.mks.com/medias/sys_master/images/images/h2a/hcb/8797131145246/KBX058-ZEMAX.zip) |
| [PAC052](https://www.newport.com/p/PAC052) | cemented achromat, N-BK7/N-SF5 | 25.4 mm | 100 mm | 96.5 mm | R = +60.741, −44.710, −133.104; t = 5.0 (crown) + 2.17 (flint) | page + [Zemax](https://api.p1.mks.com/medias/sys_master/images/images/h56/h66/8797231284254/PAC052-ZEMAX.zip) |
| [M-10X](https://www.newport.com/p/M-10X) | 10×/0.25 NA microscope objective | — | 16.5 mm | — | **not published — refused** | page only |

Every published value is stored with the verbatim source text it was read from,
which source document it came from, and — when the stored value is not literally
the published one — the derivation. See `records/components.json`.

### The transcription is checked, not trusted

`published` (the manufacturer's words) and `component` (the model) are written
separately in `catalog_sources.py`. Each supported component also declares
`radius_keys` / `thickness_keys` naming which published value every built radius
and internal thickness was transcribed from, and `CatalogComponent`
cross-checks them **at import time by exact float equality**. A digit
transposed between the two literals would otherwise have produced a component
that traces perfectly and represents a lens nobody sells, with the correct value
three lines above it.

### The refused component

M-10X publishes magnification, NA, EFL, working distance, tube length and clear
aperture — enough to fabricate something that hits those numbers, and nothing at
all about how many elements it has, where their surfaces are, how curved they
are, what they are made of, or how far apart they sit. Its product page also
links no optical-design file, unlike every other part here. It is recorded with
a structured `CATALOG_PRESCRIPTION_NOT_PUBLISHED` refusal listing every missing
parameter, and it is kept in the benchmark on purpose: a benchmark containing
only the components that worked cannot demonstrate that it would have refused
one that did not.

### Two real source disagreements

1. **Legacy glass naming.** The product pages say `N-BK7` / `N-SF5`; the vendor
   Zemax files say `BK7` / `SF5`. Those are genuinely different melts. The
   benchmark takes the product page as the vendor's current statement about the
   part on sale. Measured on the pinned Optiland 0.6.0 catalog, both spellings
   resolve to the same `glass/schott/*.yml` file at similarity 0 — the catalog
   carries no legacy entry — so the choice does not change the built material
   *here*. The disagreement is recorded because on a catalog that did carry both,
   the two readings would diverge.
2. **Image-distance reference.** The pages quote BFL at the parts' stated 589 nm
   design wavelength; the Zemax files' last air distance is at their own 546.1 nm
   primary (`PWAV`), and is correspondingly shorter — the expected direction for
   a shorter wavelength in a normally dispersive glass. The benchmark uses the
   page BFL and traces the page's design wavelength, so its image plane and its
   trace wavelength come from one document.

## The systems

| Key | Components | Fields | λ (µm) | Image plane |
| --- | --- | --- | --- | --- |
| `S1_KPX094_SINGLET` | KPX094 | 0°, 1°, 2° | 0.589 | catalog BFL |
| `S2_PAC052_ACHROMAT` | PAC052 | 0°, 1°, 2° | 0.4861, 0.589, 0.6563 | catalog BFL |
| `S3_KBX058_BICONVEX` | KBX058 | 0°, 1° | 0.589 | catalog BFL |
| `S4_PAC052_KBX058_TANDEM` | **PAC052 + 50.0 mm air + KBX058** | 0°, 1°, 2°, 3° | 0.589 | paraxial focus of the assembly |
| `S5_TANDEM_REVERSED_ACHROMAT` | S4 with the achromat **backwards** | 0°, 1°, 2°, 3° | 0.589 | paraxial focus of the assembly |

Every parameter of every system carries a `ParameterSource`:

- **`catalog`** — radii, thicknesses, glasses, design wavelengths, per-component
  entrance pupils, published BFLs. None of these is chosen here.
- **`assembly_choice`** — the 50.0 mm air gap, the field angles, the 16-ring
  pupil sampling. Nobody publishes these; each states its basis.
- **`derived`** — only the multi-component image distance, from the solver's own
  paraxial analysis of the assembled prescription. It is measured from two
  far-apart placeholder image distances (10 mm and 200 mm) and required to agree
  to 1e-9 mm, because the whole relation `image_distance + Paraxial.F2` rests on
  the answer being independent of where the image plane was parked while it was
  read. It is a closed-form paraxial solve, **not** a spot-size search.

The 50.0 mm air gap was fixed before any trace ran, as a round number roughly
half the achromat's back focal length, and was not tuned against any result.
This is a forward-modelling benchmark.

## Results (GPU, float64, RTX A6000)

### Catalog agreement

| System | EFL catalog → simulated | Δ | BFL catalog → simulated | Δ |
| --- | --- | --- | --- | --- |
| S1 KPX094 | 100.000 → 100.0116 mm | +1.16e-4 rel | 96.970 → 96.9887 mm | +1.92e-4 rel |
| S2 PAC052 | 100.000 → 100.0244 mm | +2.44e-4 rel | 96.500 → 96.5548 mm | +5.68e-4 rel |
| S3 KBX058 | 75.600 → 75.6125 mm | +1.65e-4 rel | 73.890 → 73.9114 mm | +2.90e-4 rel |

All six agree to better than 6e-4 relative — comfortably inside Newport's own
published focal-length tolerance (±1% for the singlets, ±2% for the achromat),
which is the only tolerance in sight and is the manufacturer's, not this
benchmark's. The simulated EFL is Optiland's paraxial `f2` and the simulated BFL
is `image_distance + Paraxial.F2`; both are read at the same 589 nm the catalog
quotes, which is what makes them the same quantities. That `F2` relation is not
assumed: on the KPX094 build, Optiland's own `image_solve()` moves the last
thickness to 96.98866 mm and `96.97 + F2 = 96.98866` — the same number.

Note that the vendor's `F/#` is `EFL / full diameter` while the simulated
F-number is on the *entrance pupil*, so those two are deliberately **not**
compared as if equal; the record states the basis.

### Spot size and field dependence (RMS spot radius, µm, at 589 nm)

| System | 0° | 1° | 2° | 3° |
| --- | --- | --- | --- | --- |
| S1 KPX094 plano-convex, f/4.4 | 90.08 | 93.35 | 103.31 | — |
| S2 PAC052 achromat, f/4.4 | **2.35** | 2.00 | 12.27 | — |
| S3 KBX058 equiconvex, f/6.6 | 27.16 | 28.92 | — | — |
| S4 tandem | 53.22 | 55.33 | 61.65 | 72.19 |
| S5 tandem, achromat reversed | 192.43 | 194.59 | 201.02 | 211.66 |

The achromat is 38× tighter on axis than the singlet at the same f-number and
the same aperture — which is what an achromat is sold for, obtained here with no
tuning of any kind. Its field behaviour is **non-monotonic** (2.35 → 2.00 → 12.27
µm) and that is reported as measured rather than smoothed: the spot diagrams show
why, with the 1° spot an astigmatic figure that partly balances the residual
on-axis spherical aberration before astigmatism dominates at 2°.

### Chromatic behaviour of the achromat (on axis, one fixed image plane)

| λ | 486.1 nm | 589.0 nm (design) | 656.3 nm |
| --- | --- | --- | --- |
| RMS spot radius | 9.18 µm | 2.35 µm | 5.62 µm |

Minimum at the design wavelength, rising on both sides. This is residual
chromatic blur *plus* defocus at a single fixed plane, not a per-wavelength best
focus, and the record says so.

### Ray survival

817 rays launched, 817 surviving, 0 clipped, in every trace of every system.
That is consistent rather than convenient — see *aperture* below.

### The negative control fired

S5 is S4 with the PAC052 installed against the orientation its product page
instructs ("Steepest convex surface should face the infinite conjugate"), and
with the mechanical layout, air gap, pupil, field set, wavelength, ray count and
image-plane rule held identical. Its own paraxial focus is used, so the control
is defocus-free and isolates orientation alone.

RMS spot ratio control/case: **2.93× to 3.62×**, larger at every one of the four
field points. The orientation this benchmark claims to represent therefore
demonstrably reaches the built system. Had the ratio come out at or below 1 the
summary would have said `control_fired_at_every_point: false`, and the
orientation machinery would have been reported as inert.

### GPU vs CPU

The representative multi-lens case S4, on `torch`/`cuda:0`/float64 against
`numpy`/`cpu`/float64, compared on the scalar set declared in
`benchmark_systems.GPU_CPU_COMPARISON_METRICS` (fixed in source so it cannot be
chosen after seeing the differences):

- worst relative difference **5.7e-15**, worst absolute difference **8.9e-16 mm**,
  across 4 shared traces × 5 metrics;
- surviving ray count identical (817) at every field;
- the per-trace `scientific_array_sha256` values do **not** match, and are not
  expected to: a float64 CUDA trace does not reproduce a float64 host trace
  bit-for-bit. Equivalence is judged on the declared scalars.

The on-axis `centroid_y` relative difference is suppressed rather than reported:
both backends put it at ~1e-17 mm, so it is zero on both and a ratio of the two
residues would be noise over noise. The record says that in place of the number.

### GPU execution is read, not claimed

Every trace records the adapter's requested / resolved / applied-to-Optiland /
**actual** execution, where `actual` is read off the traced arrays themselves:

```
applied: {set_backend: torch, set_precision: float64, set_device: cuda, get_device: cuda}
actual:  {namespace: torch, device: cuda:0, dtype: float64}   mismatches: []
artifact: framework=pytorch  device=gpu  dtype=float64
```

One known labelling gap, stated rather than papered over:
`core.performance.environment_fingerprint` reads `container_image` from
`MOA_IMAGE`, which `run.sh --gpu` does not set, so a GPU record's fingerprint
says `agent_solver`. Fixing that is outside this ticket, so the GPU claim is
carried by observed facts instead — `observed_execution_environment` records
`torch 2.13.0+cu126` (the `+cu126` build exists only in the CUDA image),
`torch.cuda.is_available() == True`, one visible device, `NVIDIA RTX A6000` —
plus the per-trace actual device above.

Per shared-server policy the run used a single GPU (`MOA_GPUS=device=6`), and
`gpu_count: 1` in the fingerprint is the observable consequence of that.

### Reproducibility

Each record carries a `result_fingerprint`: sha256 over the canonical JSON of the
assembled prescription fingerprint, the paraxial readout, and every trace's
scientific array hash and derived scalars — with host, clock, wall time and
filesystem paths excluded. Verified by re-running:

- GPU, **all five systems**: identical `result_fingerprint` *and* identical
  spot-diagram PNG sha256 across two consecutive full-matrix runs;
- CPU, S4: likewise identical across two runs.

Incidentally, S4's GPU and CPU spot-diagram PNGs are byte-identical to each
other — the two backends agree to below the plot's pixel resolution, so the
rasters coincide. That is a consequence of the agreement reported above, not an
independent check of it.

PNGs are byte-stable because they are written with `metadata={"Software": None}`,
so no generator string or timestamp enters the file, and every label is ASCII so
a font fallback cannot change the raster.

A different `--device`/`--dtype` legitimately produces a different fingerprint,
because the traced arrays themselves differ.

## What this benchmark does not model

Stated here rather than discovered later. Each is recorded per component or per
record with its consequence.

- **Physical apertures.** `optical-system-spec/1` has no per-surface aperture
  field, so the built systems have no rims and **nothing is vignetted by one**.
  Adding the field would change the canonical normalization and move every
  recorded prescription fingerprint in the repository, which is far outside this
  ticket. Consequences: (a) the reported clipped-ray count reflects trace failure
  only — a missed surface or total internal reflection — never rim vignetting,
  which is why 0 clipped rays everywhere is not evidence that the beams fit;
  (b) the fit is therefore *measured* instead. For each multi-component system
  the benchmark builds a **truncated** prescription — same components, same
  declared air gap, one fewer element — whose image plane lands exactly on the
  downstream component's first vertex, traces it through the same shipping
  adapter, and reports the largest radial intercept against that component's
  published clear-aperture semi-diameter. For S4/S5 the footprint at KBX058 runs
  5.36 mm (on axis) to 8.16 mm (3°) against an 11.43 mm semi-aperture, so the
  margin is 3.27 mm at worst and the zero clipped-ray count is consistent with
  the geometry rather than with the absence of rims.
- **`clipped_power` is a proxy, not a measurement.** `Optic.trace` returns
  survivors only and reports nothing about rays it dropped, so the figure assumes
  each launched ray carried unit intensity. Whether that holds for the survivors
  is checked and reported as `unit_launch_intensity_evidenced`.
- **RMS spot radius is sampling-dependent.** Hexapolar samples are equally
  weighted but not equal-area, so this unweighted RMS depends on the ring count
  (KPX094 on axis: 97.5 µm at 8 rings, 90.1 µm at the 16 rings used here). It is
  comparable *across systems at a fixed ring count*, which is how it is used, and
  it is not an energy-weighted RMS spot size.
- **Coatings, transmission, Fresnel losses, scatter, surface irregularity,
  centration and every manufacturing tolerance.** Published for these parts,
  outside a geometric ray trace of a nominal prescription.
- **Wave optics, diffraction, MTF, ray-to-wave coupling.** Out of scope by
  ticket.
- **Element decentre and tilt.** Not expressible in `optical-system-spec/1`; all
  systems are perfectly centred and untilted.

## Files

| Path | What it is |
| --- | --- |
| `catalog_sources.py` | manufacturer sources, published values with verbatim text, normalized `ComponentSpec`s, refusals, source disagreements, import-time transcription check |
| `benchmark_systems.py` | the five systems: ordering, orientation, air gaps, pupil, fields, wavelengths, image-plane rule, catalog comparisons, the declared GPU/CPU metric set |
| `run_benchmark.py` | execution through the shipping adapter, characterization, records, spot diagrams |
| `records/components.json` | question A for every component, including the refused one |
| `records/<SYSTEM>.<tag>.json` | one record per system per execution tag, in four blocks (A construction, B assembly, C execution, D characterization) plus reproducibility |
| `records/summary.<tag>.json` | cross-system summary and the negative-control comparison |
| `records/gpu_cpu_comparison.<SYSTEM>.json` | the declared scalar comparison |
| `figures/<SYSTEM>.<tag>.png` | spot diagrams, one panel per (field, wavelength) |

The reusable part of this work is **not** in this directory:
`src/core/optical_assembly.py` holds `ComponentSpec` / `ComponentPlacement` /
`assemble_optical_system`, is solver-free, and is covered by
`tests/test_optical_assembly.py`. `core/optical_system.py` and
`solvers/optiland/` are unchanged.
