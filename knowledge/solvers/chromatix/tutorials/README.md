# Chromatix example coverage (CHE-57 / PB6)

Repo-owned executable reproductions of **Chromatix 101 plus all 15 documented
examples**, run against the pinned `chromatix==0.6.0` (commit
`d24bdf0022835bb8ce1cdcc6aeafbc7fcb39daee`) with `jax_enable_x64` pinned to
`False`. Every one is reimplemented here rather than executed as an upstream
notebook, and every one carries declared validation whose recorded output lives in
`expected/<slug>.json`.

## Scope resolution

The CHE-57 ticket enumerates 15 examples by title. Seven of those titles do not
match their live URL slugs, so the set had to be resolved before any work started
(the docs landing page lists only four of them in its visible Examples section;
the rest were found in its `href`s):

| ticket title | live slug |
|---|---|
| Scalable Angular Spectrum | `sas` |
| Bandlimited Angular Spectrum (BLAS) | `bandlimited_angular_spectrum` |
| Scaled and Shifted Free-Space Propagation | `rescaled_propagation` |
| CGH using a Digital Micromirror Device | `dmd` |
| Scattering through 3D birefringent samples | `polarized_multislice` |
| High-NA vectorial PSF generation | `highNA_PSF` |
| Modified Born Series | `modified_born` |

All 15 were located; none is out of scope. Four docs pages truncate before their
optimization cells, so `c01`, `c02`, `c03` and `c09` were written against the
notebooks in `github.com/chromatix-team/chromatix/docs/examples/*.ipynb`, which is
also where their published output values come from.

## How to run

```bash
# one reproduction, printing its evidence as JSON
./run.sh python knowledge/solvers/chromatix/tutorials/c05_scalable_angular_spectrum.py

# all 16 sequentially (never concurrently: jax_enable_x64 is process-global)
./run.sh python knowledge/solvers/chromatix/tutorials/run_all.py

# re-record the evidence after an intentional change
./run.sh python knowledge/solvers/chromatix/tutorials/run_all.py --write-expected

# as a regression gate
./run.sh pytest -q tests/test_chromatix_tutorials.py -m "not slow"   # 2 tests, <1 s
./run.sh pytest -q tests/test_chromatix_tutorials.py                 # 18 tests, ~17 min
```

Every reproduction is `slow`-marked and so excluded from the Tier A gate by design
(`AGENTS.md` "Test Command Surface"): the cheapest is 2 s and the most expensive
(`c04`, `c09`, `c10`) are ~5 min each, dominated by 400-1000 Adam steps on
1152x1152 or 1920x1920 grids. All carry the `chromatix`, `jax` and `integration`
markers.

## Validation strength

| Kind | Meaning | Count |
|---|---|---|
| `reference` | compared against a value published by the upstream page or notebook | 46 |
| `analytic` | compared against a closed form or independently computable expectation | 84 |
| `invariant` | structural or physical invariant (shape, conservation, symmetry, direction of change) | 75 |
| `qualitative` | inherently visual upstream example with no machine-checkable oracle | 0 |

**205 checks across 16 reproductions.** 10 units are validated against a
quantitative upstream reference and 6 against analytic expectations; **none** rests
on qualitative evidence alone, and no reproduction contains a single qualitative
check.

## Coverage inventory

| # | Upstream page | Reproduction | Checks (ref/ana/inv/qual) | Outcome |
|---|---|---|---|---|
| 00 | [Chromatix 101](https://chromatix.readthedocs.io/en/latest/101/) | `c00_chromatix_101.py` _(slow)_ | 16 (5/5/6/0) | **validated (reference)** |
| 01 | [Fourier Ptychography](https://chromatix.readthedocs.io/en/latest/examples/fourier_ptychography/) | `c01_fourier_ptychography.py` _(slow)_ | 10 (1/5/4/0) | **validated (reference)** |
| 02 | [Holoscope](https://chromatix.readthedocs.io/en/latest/examples/holoscope/) | `c02_holoscope.py` _(slow)_ | 18 (4/10/4/0) | **validated (reference)** |
| 03 | [Computer Generated Holography](https://chromatix.readthedocs.io/en/latest/examples/cgh/) | `c03_computer_generated_holography.py` _(slow)_ | 13 (5/5/3/0) | **validated (reference)** |
| 04 | [Aberration Phase Retrieval (Zernike Fitting)](https://chromatix.readthedocs.io/en/latest/examples/zernike_fitting/) | `c04_zernike_fitting.py` _(slow)_ | 13 (5/3/5/0) | **validated (reference)** |
| 05 | [Scalable Angular Spectrum](https://chromatix.readthedocs.io/en/latest/examples/sas/) | `c05_scalable_angular_spectrum.py` _(slow)_ | 15 (0/11/4/0) | **validated (analytic)** |
| 06 | [Off-Axis Propagation](https://chromatix.readthedocs.io/en/latest/examples/off_axis_propagation/) | `c06_off_axis_propagation.py` _(slow)_ | 20 (5/8/7/0) | **validated (reference)** |
| 07 | [Bandlimited Angular Spectrum (BLAS)](https://chromatix.readthedocs.io/en/latest/examples/bandlimited_angular_spectrum/) | `c07_bandlimited_angular_spectrum.py` _(slow)_ | 12 (3/5/4/0) | **validated (reference)** |
| 08 | [Scaled and Shifted Free-Space Propagation](https://chromatix.readthedocs.io/en/latest/examples/rescaled_propagation/) | `c08_rescaled_propagation.py` _(slow)_ | 15 (11/1/3/0) | **validated (reference)** |
| 09 | [Computer Generated Holography using a Digital Micromirror Device](https://chromatix.readthedocs.io/en/latest/examples/dmd/) | `c09_dmd_cgh.py` _(slow)_ | 12 (0/5/7/0) | **validated (analytic)** |
| 10 | [Seidel Fitting](https://chromatix.readthedocs.io/en/latest/examples/seidel_fitting/) | `c10_seidel_fitting.py` _(slow)_ | 12 (6/4/2/0) | **validated (reference)** |
| 11 | [Scattering through 3D birefringent samples](https://chromatix.readthedocs.io/en/latest/examples/polarized_multislice/) | `c11_polarized_multislice.py` _(slow)_ | 11 (1/5/5/0) | **validated (reference)** |
| 12 | [High-NA vectorial PSF generation](https://chromatix.readthedocs.io/en/latest/examples/highNA_PSF/) | `c12_high_na_psf.py` _(slow)_ | 7 (0/2/5/0) | **validated (analytic)** |
| 13 | [Pollen grain phantom data generator](https://chromatix.readthedocs.io/en/latest/examples/pollen/) | `c13_pollen_phantom.py` _(slow)_ | 10 (0/5/5/0) | **validated (analytic)** |
| 14 | [Filaments phantom data generator](https://chromatix.readthedocs.io/en/latest/examples/filaments/) | `c14_filaments_phantom.py` _(slow)_ | 8 (0/3/5/0) | **validated (analytic)** |
| 15 | [Modified Born Series](https://chromatix.readthedocs.io/en/latest/examples/modified_born/) | `c15_modified_born_series.py` _(slow)_ | 13 (0/7/6/0) | **validated (analytic)** |

## Outcome classification

All 16 in-scope units are **reproduced and validated**. Two carry unreproducible
upstream claims and two required a substitution for an unavailable dependency; all
four are asserted rather than skipped.

### Upstream findings (asserted, not worked around)

| Unit | Finding |
|---|---|
| 05 | `transform_propagate` places a tilted beam at `z*sin(theta)`, not the geometric `z*tan(theta)` -- its output coordinate is the direction-cosine (Fourier) mapping `x' = lambda*z*f_x`. Measured 350.000 against `z*sin` = 350.229 and `z*tan` = 372.706: a **6.1% position error at 20 degrees**. SAS and ASM both give the geometric position and agree with each other to 0.3%. |
| 06 | `asm_propagate`'s `kykx` is a **spatial frequency in cycles per length** (`sin theta = lambda*kykx`) while `plane_wave`'s `kykx` is an **angular wavenumber in radians per length** (`sin theta = kykx/k0`). Same parameter name, factor of `2*pi` apart, and the displacement is opposite in sign to the parameter. Established by sweeping three `kykx` values against unclipped displacements. |
| 06, 08 | The modified-kernel and CZT shifted-propagation paths **disagree in amplitude**: r = 0.998 in structure but a 14.13x norm difference at 4x zoom. Upstream's own `c08` cell prints `3.1434343` and `44.420246` and then compares only after normalising each by its own norm, so this is upstream-known and documented nowhere else. |
| 04, 10 | Both aberration-fitting examples' `update()` returns only `(model, metrics)` while rebinding `opt_state` internally, so the loop passes the **initial** Adam state every iteration -- every step is a fresh bias-corrected Adam step, i.e. sign descent at a fixed step of `lr`. Honouring that faithfully is what makes `c10` reproduce its published numbers; threading the state through stalls it in a local minimum (loss 9.39 vs 0.78). `c03` and `c09` rebind correctly, so the inconsistency is between the examples. |
| 04 | `c04`'s published convergence does **not** reproduce under either convention (0.137 and 0.151 against the published 0.00174). The diagnosis is specific: `c10` projects its coefficients onto the non-negative orthant after every step and `c04` does not, and that projection is what breaks the sign/twin ambiguity of single-intensity phase retrieval. |
| 12 | `high_na_ff_lens` is **not sampling-independent**, re-confirming CHE-18: refining only the pupil sampling (128 -> 192 -> 256 px) with everything else fixed moves `Iz/Ix` from 0.011431 to 0.003619. The check is written to FAIL if the bug is fixed upstream. |
| 15 | `solve()` returns a **component-last** `(*spatial, 3)` array, contradicting its own docstring ("the first (left-most) axis the polarization vector"), and its input current density must be component-last too. `Source` takes a current density, not a field. `add_absorbing_bc` pads the sample and records `Sample.ROI`. |
| 13 | `pollen_3d` returns a **real** `float64` array, so upstream's `np.angle`-based colouriser gives an information-free hue axis; it also contains subnormal doubles down to 4.9e-324 with 52% of voxels strictly non-zero, so `count_nonzero` measures the numerical floor. Its `radius` parameter is counter-intuitive: reducing it from 0.8 to 0.25 **fills** the volume. |
| 03 | Upstream's CGH target is **circularly shifted by half the kernel width**: `fftn(kernel, s=sample.shape)` puts the ball's origin at index 0 rather than centring it, so every blob lands 12 voxels from its seeded coordinate. The seeded voxels hold ~0. |
| 00 | The published `Field.power` of `1.0000002` is not reproduced digit-for-digit (`1.0000118`); both are float32 normalisation residue, so the docs page was built from a different commit. `Spectrum` density weights do **not** scale `Field.power` (which stays 1 per wavelength); they enter through `Field.intensity`. |
| 02 | `defocused_ramps` hard-requires exactly six `delta` entries (`delta=[x]*3` raises `IndexError`), so the six-view geometry is structural. Tracking the brightest lobe is **not** a usable depth code (r = 0.76 against z, because the argmax hops between views); the z-reflection asymmetry is (126% of RMS engineered vs 0.0006% unmasked). |

### Environment blockers (recorded, worked around)

| Unit | Blocker | Substitution |
|---|---|---|
| 01, 09 | **`scikit-image` is not installed in the pinned environment** (`ModuleNotFoundError: No module named 'skimage'`). `c01` needs `camera()`/`moon()`, `c09` needs `cat()`. Installing it would change the pinned environment; fetching the images over the network would make a repository test depend on a third-party server. | Deterministic `chromatix.utils.siemens_star`-based targets of the same shape. Upstream's published loss/correlation trajectories are properties of those photographs and are recorded for reference rather than asserted; every behavioural claim (surrogate gradient, band-limit, resolution gain, saturation shape, design-distance sharpness) is target-independent and **is** asserted. |
| 15 | Upstream's page truncates after the sample-construction cell, so no `solve()` call or source construction is shown. | Written from the pinned signature; the source construction is this repository's choice and is labelled as such. |

### Scope reductions (all recorded in the metrics)

| Unit | Upstream | Here | Why |
|---|---|---|---|
| 05 | -- | `pad_width=2048` ASM reference | kept as published; the 4608x4608 grid is affordable |
| 08 | -- | full 4096x4096 oversampled reference | kept as published |
| 02 | -- | full 1920x1920 x 40 planes | kept as published (~70 s) |

No reproduction reduces an iteration count or grid below what upstream specifies.

## Reproducibility notes

| Source | Seedable? | How |
|---|---|---|
| `chromatix.ops.shot_noise` | yes | takes a `PRNGKey`; bit-reproducible (c04) |
| `chromatix.utils.filaments_3d` | yes | `seed=` (c02 uses `972920147`); the docs' *unseeded* call is measured in c14 |
| `chromatix.utils.pollen_3d` | n/a | deterministic, no randomness (c13) |
| `jax.nn.initializers.uniform` | yes | takes a `PRNGKey` (c09) |

`jax_enable_x64` is pinned to `False` by the harness at import, and
`tests/test_chromatix_tutorials.py::test_jax_x64_is_pinned_off` asserts the pin
held: `sax.saxtypes.core` sets it to `True` as an import side effect and the
adapter registry imports every adapter eagerly, so collection order could
otherwise flip it process-wide and change every recorded number
(`conventions.md`, "Numerical dtype").
