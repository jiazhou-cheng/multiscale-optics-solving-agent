# C_WAVE_TO_RAY — theory

Source: Cheng et al., ACS Photonics 2026, DOI `10.1021/acsphotonics.6c00818`,
main text eq 1 and Figure 1c; SI S2 (eqs S1–S5), SI S3 (eqs S6–S9), SI S4
(Algorithm S1), SI S7 (Algorithm S2). Equation labels are the paper's.

## The physical claim

A complex field on a plane is exactly a superposition of plane-wave modes. Each
propagating mode is a direction plus a complex weight — which is precisely what
a ray-as-wavelet carries. So a field can be *converted into rays* with no
approximation beyond truncating the mode set, and the truncation is a Monte
Carlo sampling error with a known rate rather than a modelling assumption.

This is the direction this repository has never claimed. It is also the
stochastic one.

## Step 1 — angular spectrum

For a field patch `U(u, v)` in a local tangent frame `(û, v̂, n̂)` (SI eq S1),

```
Ũ(k_u, k_v) = ∬ U(u, v) e^{−i(k_u u + k_v v)} du dv
```

## Step 2 — evanescent cut

Only modes satisfying `k_u² + k_v² ≤ k²` propagate. Modes with
`k_u² + k_v² > k²` are evanescent and are **discarded** (SI S2). The retained
normal component is

```
k_n = √( k² − k_u² − k_v² )   ∈ ℝ
```

Discarding is physically correct for a far-field/ray representation — an
evanescent mode has no propagation direction to give a ray — but it is a real
loss of power. This repository requires that loss to be reported as a named
term, never absorbed silently, because a large evanescent fraction is the
signature of a field that should not be turned into rays at all.

## Step 3 — the exact target

The ASM gives the exact diffracted field at an observation point
`r = (x, y, d)` (SI eq S2):

```
U_obs(r) = ∬ Ũ(k_u, k_v) e^{i(k_u x + k_v y + k_n d)} dk_u dk_v
```

Everything below is an estimator *of this integral*. That is the key framing:
`C_WAVE_TO_RAY` is not an approximation of the physics, it is a quadrature
scheme for an integral whose exact value is known. This is why the exactness
limit (enumerate every bin) is a meaningful and mandatory check.

## Step 4 — Monte Carlo sampling into rays

Draw `N` spectral components `{(k_u⁽ⁱ⁾, k_v⁽ⁱ⁾)}` from a probability density
`p(k_u, k_v)` over the propagating modes. Each draw becomes a secondary ray:

```
direction   d̂⁽ⁱ⁾ = ( k_u⁽ⁱ⁾, k_v⁽ⁱ⁾, k_n⁽ⁱ⁾ ) / k
amplitude   a⁽ⁱ⁾ = Ũ(k_u⁽ⁱ⁾, k_v⁽ⁱ⁾) / p(k_u⁽ⁱ⁾, k_v⁽ⁱ⁾)        (eq 1 / eq S4)
OPL         OPL⁽ⁱ⁾ = d̂⁽ⁱ⁾ · r                                     (SI S2)
```

and the reconstructed field is the coherent sum (eq S5)

```
U_obs(r) ≈ (1/N) Σ_i a⁽ⁱ⁾ e^{i k · OPL⁽ⁱ⁾}
```

### Why the `1/p` weight is not optional

It is what makes the estimator **unbiased** under non-uniform sampling
(SI S2, citing importance sampling). Rays drawn from high-density regions are
drawn more often, so each carries proportionally less weight. Omitting `1/p`
while sampling non-uniformly produces a confidently wrong answer that still
looks like a field — which is why the omission is a required negative test
rather than a code-review item.

In the limit `N → ∞` the secondary-ray ensemble converges to `Ũ(k_u, k_v)`
itself, so the representation is faithful, not merely adequate (SI S2).

### Choice of `p`

| Density | Definition | Behaviour |
|---|---|---|
| `p_uni` | uniform over propagating bins | Robust; no assumption about the spectrum |
| `p_mag` | `∝ \|Ũ(k_u, k_v)\|` | Faster for spectra concentrated in one lobe (paper Figure 4a, metalens); comparable for multilobed spectra (Figure 4b, Siemens star) |

The paper's summary is that magnitude-proportional sampling is "a robust and
generally effective strategy," with the caveat that the advantage depends on
spectral concentration. Both must be reported at matched `N`.

## Step 5 — launch position and relative phase

Rays are launched from chosen positions on the plane, not only from its centre.
A ray launched at `(x_p, y_p)` carries the relative phase (SI Algorithm S1,
line 9)

```
φ⁽ᵖˢ⁾ = k_x⁽ᵖˢ⁾ x_p + k_y⁽ᵖˢ⁾ y_p
```

and is initialized with `OPL = 0`, because its accumulated path restarts at the
plane. Omitting `φ` destroys the phase relation between launch positions — the
same error, seen from the other side, as omitting `Δr` in
`../ray_to_wave/theory.md`.

## Cascade: Algorithm S1

Naively, propagating sampled secondary rays into the *next* DOE and resampling
makes the ray count grow multiplicatively per surface. For **planar** surfaces
there is an exact way out, because all rays cross a common Cartesian plane:

```
1  U_in  ← RayToField(R_in, Ω)          coherently accumulate ALL incident rays
2  U_out ← U_in · U_DOE                 apply the complex DOE transmission once
3  Ũ_out ← F{U_out}                     one global FFT
4  p     ← f(Ũ_out)                     density over propagating modes
5  {(x_p, y_p)} ← P launch positions
6  for p = 1..P, s = 1..S:
       sample (k_x, k_y) ~ p
       φ   = k_x x_p + k_y y_p
       o   = (x_p, y_p, z_DOE)
       d̂   = (k_x, k_y, √(k² − k_x² − k_y²)) / k
       OPL = 0
       a   = Ũ_out(k_x, k_y) / p(k_x, k_y).detach() · e^{iφ}
```

The outgoing ray count is the **budget** `P·S`, set by the caller, not the
product of incoming count and secondary count. Coherent interference at the
plane is preserved because the accumulation in line 1 happens before the
transmission in line 2.

This does **not** apply to conformal (curved) DOEs: rays there strike different
local tangent planes with position-dependent frames and normals, so there is no
common plane to accumulate onto. The paper retains the direct per-patch
implementation for that case, and the patch size is then bounded by the
curvature error below.

## Curvature bound (SI S3)

A patch of width `D` on a surface of local radius `R` carries an extra
quadratic sag phase (eq S6), giving a local spatial frequency linear in
position (eq S7), a frequency spread at the patch edge (eq S8), and hence a
bound on the direction error extracted from the local angular spectrum (eq S9):

```
φ_curv(x) = (1/2)(k₀/R) x²
f(x)      = (1/2π) dφ/dx = k₀ x / (2πR)
|Δf|      ≤ k₀ D / (4πR)
ε_curv    ≤ arcsin( λ · k₀ D / (4πR) ) = arcsin( D / (2R) )
```

Two properties make this useful as an executable precondition: it is
**independent of the DOE phase profile**, and it is monotone in `D/R`. The
planar limit `R → ∞` gives `ε_curv → 0`, consistent with SI S2's statement that
planar patches have no intrinsic upper size bound.

Assumptions to record as validity limits: one principal curvature direction at
a time (the 2-D result follows by applying the argument along both principal
axes), quadratic sag, and `D ≪ R`.

## Differentiability (SI S7.2, Algorithm S2)

The draw in step 4 is discrete and non-differentiable. The paper's estimator
holds the **sampled wavevector and the resulting ray direction fixed** during
backpropagation and detaches `p`, so gradients flow only through `Ũ` in the
amplitude:

```
d̂⁽ˢ⁾  = d̂_in + (k_u, k_v, √(k² − k_u² − k_v²))/k     ← no gradient
a⁽ˢ⁾  = Ũ(k_u, k_v) / p(k_u, k_v).detach()            ← gradient flows here
```

The paper is explicit that this "neglects gradients associated with changes in
the sampled secondary-ray directions" while still providing low-variance
gradients sufficient for stable DOE optimization. So there are two separate
quantities, and M2 must not conflate them:

1. the derivative of the **fixed-direction surrogate** — the estimator should
   compute this exactly;
2. the derivative of the **true objective** — the estimator is biased with
   respect to this, by construction.

### The rejected alternative

SI S7.3 describes a Gumbel–Softmax reparameterization (Algorithms S3, eqs
S11–S15): perturb log-probabilities with Gumbel noise, take a hard argmax
forward and a softmax backward via `ȳ = y_hard + y_soft − y_soft.detach()`.
The forward sampling statistics are provably unchanged (eq S12), so it
introduces no forward bias; the backward pass is still "an approximate,
generally biased surrogate." It was **not** used, on memory grounds: it must
retain an `N_t × qD × qD` tensor, and both factors must be large for accurate
simulation. Temperature `τ = 3` was used where it was evaluated.

M2 records this as a documented alternative and does not implement it.
