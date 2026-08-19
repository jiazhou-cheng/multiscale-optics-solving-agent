# Optiland tutorial coverage (CHE-57 / PB6)

Repo-owned executable reproductions of the **frozen 41-tutorial scope** of the
official Optiland Tutorials & Recipes index (https://www.optiland.org/tutorials),
run against the pinned `optiland==0.6.0`. Every one is reimplemented here rather
than executed as an upstream notebook, and every one carries declared validation
whose recorded output lives in `expected/<slug>.json`.

## How to run

```bash
# one reproduction, printing its evidence as JSON
./run.sh python knowledge/solvers/optiland/tutorials/t04_material_database.py

# all 41 sequentially (never concurrently: the Optiland backend is global state)
./run.sh python knowledge/solvers/optiland/tutorials/run_all.py

# re-record the evidence after an intentional change
./run.sh python knowledge/solvers/optiland/tutorials/run_all.py --write-expected

# as a regression gate
./run.sh pytest -q tests/test_optiland_tutorials.py -m "not slow"   # 13 tests, ~6 s
./run.sh pytest -q tests/test_optiland_tutorials.py                 # 42 tests, ~10.5 min
```

The 29 `slow`-marked reproductions are excluded from the Tier A gate by design
(`AGENTS.md` "Test Command Surface"); the 13 remaining ones run in 6 seconds and
are Tier-A eligible. `t10_differentiable_ray_tracing` additionally carries the
`torch` marker.

## Validation strength

Each check declares its own strength, following the CHE-57 priority order:

| Kind | Meaning | Count |
|---|---|---|
| `reference` | compared against a value published by the upstream tutorial | 34 |
| `analytic` | compared against a closed form or independently computable expectation | 218 |
| `invariant` | structural or physical invariant (shape, conservation, symmetry, direction of change) | 189 |
| `qualitative` | inherently visual upstream example; never the only check for a tutorial | 18 |

**459 checks across 41 reproductions.** 19 tutorials are validated against a
quantitative upstream reference, 21 against an analytic expectation, 1 against
structural invariants; **none** rests on qualitative evidence alone.

## Coverage inventory
### Beginner

| # | Upstream tutorial | Reproduction | Checks (ref/ana/inv/qual) | Outcome |
|---|---|---|---|---|
| 01 | [Singlet Lens](https://www.optiland.org/tutorials/your-first-optical-system) | `t01_singlet_lens.py` | 9 (0/1/7/1) | **validated (analytic)** |
| 02 | [Determining Lens Properties](https://www.optiland.org/tutorials/lens-properties) | `t02_lens_properties.py` | 14 (0/11/3/0) | **validated (analytic)** |
| 03 | [Saving and Loading Files](https://www.optiland.org/tutorials/saving-and-loading) | `t03_saving_and_loading.py` | 5 (0/0/5/0) | **validated (invariant)** |
| 04 | [Material Database](https://www.optiland.org/tutorials/material-database) | `t04_material_database.py` | 18 (1/12/5/0) | **validated (reference)** |
| 05 | [Tracing and Analyzing Rays](https://www.optiland.org/tutorials/tracing-and-analyzing-rays) | `t05_tracing_and_analyzing_rays.py` | 23 (0/6/17/0) | **validated (analytic)** |
| 06 | [Tilting & De-centering Components](https://www.optiland.org/tutorials/tilting-and-decentering) | `t06_tilting_and_decentering.py` | 11 (0/9/2/0) | **validated (analytic)** |
| 07 | [Anti-Reflective Coating](https://www.optiland.org/tutorials/anti-reflective-coating) | `t07_anti_reflective_coating.py` _(slow)_ | 18 (1/10/7/0) | **validated (reference)** |
| 08 | [Edmund Optics Catalogue](https://www.optiland.org/tutorials/catalogue-edmund-optics) | `t08_edmund_optics_catalogue.py` | 12 (2/4/5/1) | **validated (reference)** |

### Intermediate

| # | Upstream tutorial | Reproduction | Checks (ref/ana/inv/qual) | Outcome |
|---|---|---|---|---|
| 09 | [Non-Rotationally Symmetric Systems](https://www.optiland.org/tutorials/non-rotationally-symmetric) | `t09_non_rotationally_symmetric.py` _(slow)_ | 8 (0/3/5/0) | **validated (analytic)** |
| 10 | [Differentiable Ray Tracing](https://www.optiland.org/tutorials/differentiable-ray-tracing) | `t10_differentiable_ray_tracing.py` _(slow, torch)_ | 12 (1/4/7/0) | **validated (reference)** |
| 11 | [Monte Carlo Raytracing Methods](https://www.optiland.org/tutorials/monte-carlo-ray-tracing) | `t11_monte_carlo_raytracing.py` _(slow)_ | 10 (0/5/5/0) | **validated (analytic)** |
| 12 | [Raytracing Aspheres](https://www.optiland.org/tutorials/raytracing-aspheres) | `t12_raytracing_aspheres.py` | 8 (0/5/3/0) | **validated (analytic)** |
| 13 | [Common Aberration Analyses](https://www.optiland.org/tutorials/common-aberration-analyses) | `t13_common_aberration_analyses.py` _(slow)_ | 17 (0/5/12/0) | **validated (analytic)** |
| 14 | [1st & 3rd Order Aberrations](https://www.optiland.org/tutorials/first-third-order-aberrations) | `t14_first_third_order_aberrations.py` | 13 (0/10/3/0) | **validated (analytic)** |
| 15 | [Chromatic Aberrations](https://www.optiland.org/tutorials/chromatic-aberrations) | `t15_chromatic_aberrations.py` | 8 (2/4/0/2) | **validated (reference)** |
| 16 | [OPD Calculations](https://www.optiland.org/tutorials/opd-calculations) | `t16_opd_calculations.py` _(slow)_ | 15 (0/10/4/1) | **validated (analytic)** |
| 17 | [Simple Optimization](https://www.optiland.org/tutorials/simple-optimization) | `t17_simple_optimization.py` _(slow)_ | 8 (2/2/3/1) | **validated (reference)** |
| 18 | [Introduction to Coatings](https://www.optiland.org/tutorials/introduction-to-coatings) | `t18_introduction_to_coatings.py` _(slow)_ | 10 (0/3/7/0) | **validated (analytic)** |
| 19 | [Multilayer Stack](https://www.optiland.org/tutorials/multilayer-stack) | `t19_multilayer_stack.py` _(slow)_ | 10 (0/6/3/1) | **validated (analytic)** |
| 20 | [Color Analysis for Thin-Films](https://www.optiland.org/tutorials/color-analysis-thin-film) | `t20_color_analysis_thin_film.py` _(slow)_ | 7 (0/4/2/1) | **validated (analytic)** |
| 21 | [Surface Roughness & Scattering](https://www.optiland.org/tutorials/surface-roughness-scattering) | `t21_surface_roughness_scattering.py` _(slow)_ | 15 (0/7/7/1) | **validated (analytic)** |
| 22 | [Thin Film Tolerance Analysis](https://www.optiland.org/tutorials/thin-film-tolerance-analysis) | `t22_thin_film_tolerance_analysis.py` _(slow)_ | 8 (2/3/3/0) | **validated (reference)** |
| 23 | [Tolerancing, Sensitivity Analyses](https://www.optiland.org/tutorials/tolerancing-sensitivity) | `t23_tolerancing_sensitivity.py` _(slow)_ | 9 (0/6/3/0) | **validated (analytic)** |
| 24 | [Thorlabs Catalogue](https://www.optiland.org/tutorials/catalogue-thorlabs) | `t24_thorlabs_catalogue.py` _(slow)_ | 6 (0/3/3/0) | **validated (analytic)** |

### Advanced

| # | Upstream tutorial | Reproduction | Checks (ref/ana/inv/qual) | Outcome |
|---|---|---|---|---|
| 25 | [PSF and MTF Calculation](https://www.optiland.org/tutorials/psf-and-mtf) | `t25_psf_and_mtf.py` _(slow)_ | 17 (0/10/7/0) | **validated (analytic)** |
| 26 | [Zernike Decomposition](https://www.optiland.org/tutorials/zernike-decomposition) | `t26_zernike_decomposition.py` _(slow)_ | 14 (0/6/8/0) | **validated (analytic)** |
| 27 | [Advanced Optimization](https://www.optiland.org/tutorials/advanced-optimization) | `t27_advanced_optimization.py` _(slow)_ | 10 (3/1/5/1) | **validated (reference)** |
| 28 | [Optimization Case Study](https://www.optiland.org/tutorials/optimization-case-study) | `t28_optimization_case_study.py` _(slow)_ | 10 (2/3/5/0) | **validated (reference)** |
| 29 | [Custom Optimization Operands](https://www.optiland.org/tutorials/custom-optimization-operands) | `t29_custom_optimization_operands.py` _(slow)_ | 10 (1/2/6/1) | **validated (reference)** |
| 30 | [Introduction to Polarization](https://www.optiland.org/tutorials/introduction-to-polarization) | `t30_introduction_to_polarization.py` _(slow)_ | 11 (2/7/2/0) | **validated (reference)** |
| 31 | [Dichroic Mirror Optimization for Polarization Separation](https://www.optiland.org/tutorials/thin-film-optimization) | `t31_dichroic_mirror_optimization.py` _(slow)_ | 8 (1/3/4/0) | **validated (reference)** |
| 32 | [Needle Synthesis for Thin Film Design](https://www.optiland.org/tutorials/needle-synthesis) | `t32_needle_synthesis.py` _(slow)_ | 13 (3/6/4/0) | **validated (reference)** |
| 33 | [Lithographic Projection System](https://www.optiland.org/tutorials/lithographic-projection-system) | `t33_lithographic_projection_system.py` _(slow)_ | 12 (2/5/4/1) | **validated (reference)** |
| 34 | [Freeform Surfaces](https://www.optiland.org/tutorials/freeform-surfaces) | `t34_freeform_surfaces.py` _(slow)_ | 7 (1/3/2/1) | **validated (reference)** |
| 35 | [Three-Mirror Anastigmat](https://www.optiland.org/tutorials/three-mirror-anastigmat) | `t35_three_mirror_anastigmat.py` _(slow)_ | 9 (2/4/2/1) | **validated (reference)** |
| 36 | [Glass Expert](https://www.optiland.org/tutorials/glass-expert) | `t36_glass_expert.py` _(slow)_ | 10 (1/2/7/0) | **validated (reference)** |
| 37 | [Multi-Configuration Zoom Lens](https://www.optiland.org/tutorials/multi-configuration-zoom-lens) | `t37_multi_configuration_zoom_lens.py` _(slow)_ | 14 (4/6/3/1) | **validated (reference)** |
| 38 | [Tolerancing, Monte Carlo](https://www.optiland.org/tutorials/tolerancing-monte-carlo) | `t38_tolerancing_monte_carlo.py` _(slow)_ | 8 (0/3/4/1) | **validated (analytic)** |
| 39 | [Custom Surface Types](https://www.optiland.org/tutorials/extending-surfaces) | `t39_custom_surface_types.py` | 10 (0/9/0/1) | **validated (analytic)** |
| 40 | [Custom Coating Types](https://www.optiland.org/tutorials/extending-coatings) | `t40_custom_coating_types.py` | 12 (0/11/1/0) | **validated (analytic)** |
| 41 | [Custom Optimization Algorithms](https://www.optiland.org/tutorials/extending-optimization) | `t41_custom_optimization_algorithms.py` _(slow)_ | 10 (1/4/4/1) | **validated (reference)** |

## Outcome classification

All 41 in-scope tutorials are **reproduced and validated**. None is
"blocked by pinned-environment/upstream issue" at the tutorial level, but five
carry recorded upstream defects or unreachable claims (see below), and three
required a substitution for an unavailable external artifact.

### Upstream defects found (asserted, not worked around)

| Tutorial | Finding |
|---|---|
| 39 Custom Surface Types | The published `_surface_normal` is **mathematically wrong**: `d(a*r)/dx = a*x/r` but the tutorial writes `a*x/r2`. Disagreement against a central difference of its own `sag()` is 0.52 in direction cosine; correcting only that term gives 1.3e-9. Both versions normalise to unit vectors and both trace without raising, so it is a silent physics error. |
| 35 Three-Mirror Anastigmat | `OptimizerGeneric` returns `success=True` with `CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH` at a point **1.97x worse** than its start (2.750 -> 5.417). `maxiter` 60/200/500 all stop there and the point is inside the bounds. `res.success` is not evidence. |
| 07 Anti-Reflective Coating | `ThinFilmCoating(air, glass, <ThinFilmStack>)` raises `TypeError`; the pinned signature wants a layer list **in nanometres** while `ThinFilmStack.add_layer` takes micrometres, so transcribing the tutorial's numbers builds a 1000x-too-thin stack silently. Sharing one coating across all four doublet surfaces gives `rays.i` ~ 3.6, i.e. transmittance above unity. |
| 28 Optimization Case Study | Upstream's "RMS spot size of ~20 um or less for all wavelengths and fields" is **not reproducible** from the published recipe: 22 um at the 0.7 field to 50 um at the 20-degree field, after three extra optimizer restarts. |
| 31 Dichroic Mirror | The operand's declared `min_val=0.99` is not met: L-BFGS-B converges to 0.9719 in 47 of 200 allowed iterations. `add_operand(min_val=...)` states a goal, not a guarantee. |
| 41 Custom Optimization Algorithms | Upstream's "converges within about 250 iterations" holds under no criterion: 356 steps to reach within 10x of the final value, 889 to reach 1%. |
| 04 Material Database | `AbbeMaterialE` recovers only 0.57-0.83x of its requested Abbe number over V_e = 20..80 and errs by 1.4e-2 in index against real N-BK7, 98x worse than `AbbeMaterial(model='buchdahl')`. The 0.6.0 *default* `polynomial` model also misses its own defining numbers. |

### Environment blockers (recorded, worked around)

| Tutorial | Blocker | Substitution |
|---|---|---|
| 01, 09, 39 | `Optic.draw3D()` **hangs indefinitely** headlessly (VTK finds no X server, no EGL, no OSMesa) | not called; `draw()` exercised instead |
| 08 Edmund Optics Catalogue | the tutorial's `.zmx` is a vendor website download and Optiland 0.6.0 ships no `.zmx` fixture | `save_zemax_file` -> `load_zemax_file` on the repo-owned Edmund #45-362 prescription, whose datasheet EFL/BFL give a *stronger* offline oracle |
| 24 Thorlabs Catalogue | `load_zemax_file(url)` raises `ValueError: Failed to read Zemax file.` -- thorlabs.com answers the documented URL with a 1313-byte HTML page, even though the container **does** have outbound network access | the analysis workflow reproduced on a repo-owned finite-conjugate doublet |

### Scope reductions (all recorded in the metrics)

| Tutorial | Upstream | Here | Why |
|---|---|---|---|
| 11 Monte Carlo | 1000 systems | 200 | test budget; seeded and replayable |
| 21 Surface Roughness | 1,000,000 random rays | 10921 hexapolar | test budget, and `distribution="random"` is unseeded |
| 22 Thin Film Tolerance | 500 MC trials | 150 | test budget |
| 23 Tolerancing Sensitivity | 128 steps/perturbation | 33 | test budget |
| 27 Advanced Optimization | `maxiter=256`, `workers=-1` | `maxiter=40`, `workers=1` | **`workers=-1` forks one process per CPU; AGENTS.md forbids parallel solver processes on this shared machine** |
| 32 Needle Synthesis | dichroic `max_iterations=8` | 5 | test budget; the AR half uses upstream's settings verbatim because that is where the quantitative claim lives |
| 33 Lithographic Projection | uncapped `tol=1e-9` | `maxiter=30` | 42-variable finite-difference gradient at ~3 s/step |
| 35 Three-Mirror Anastigmat | uncapped `tol=1e-9` | `maxiter=60` | irrelevant: the optimizer stops after 4 iterations regardless |
| 36 Glass Expert | `num_neighbours=7, maxiter=100` | `2, 3` | combinatorial search over 564 glasses; already 66,390x merit reduction |

## Reproducibility notes

Three different RNG paths behave differently, and this matters for what can be
recorded as evidence:

| Source | Seedable? | How |
|---|---|---|
| `distribution.RandomDistribution` | yes | `RandomDistribution(seed=...)`; `Optic.trace(distribution="random")` constructs an **unseeded** one |
| `optimization.DifferentialEvolution` | yes, indirectly | no `seed` parameter, but SciPy falls back to NumPy's global RNG, so `np.random.seed(...)` works |
| `tolerancing.perturbation.DistributionSampler` | yes, **only** explicitly | builds its own `be.default_rng(seed)`; `seed=None` ignores `np.random.seed` |
| `optiland.scatter` BSDFs | **no** | numba-compiled with an RNG unreachable from NumPy; two identical calls differ by ~1% |

`t21_surface_roughness_scattering` therefore declares `metric_rtol=0.35` in its
`TutorialMeta` and records only statistically stable quantities.
