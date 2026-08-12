# Ray–wave framework paper

Primary scientific source for the M2 bidirectional ray–wave coupler.

| Item | Value |
|---|---|
| Title | A Differentiable Ray–Wave Framework for Hybrid Refractive–Diffractive System Modeling and Optimization |
| Authors | Jiazhou Cheng, Margaret Gao, Yixuan Shao, Chenkai Mao, Tom D. Milster, Jonathan A. Fan |
| Venue | ACS Photonics, 2026 |
| DOI | [10.1021/acsphotonics.6c00818](https://doi.org/10.1021/acsphotonics.6c00818) |
| Preprint | [arXiv:2605.15418](https://arxiv.org/abs/2605.15418) |
| Reference implementation | <https://github.com/jiazhou-cheng/raywave-tracing> — **not vendored, not pinned, and not executed by this repository** |
| Files | `paper.pdf`, `supporting_information.pdf` (23 pp, 8 figures, 3 tables) |

Stored in full because the repository owner is the first author. This is the
documented exception to the `knowledge/README.md` rule against storing
copyrighted full papers; see that file for the policy.

## What each section supplies to M2

| Section | Supplies |
|---|---|
| Main text, Figure 1 | The ray↔DOE interaction picture: local patch, angular spectrum, Monte Carlo secondary-ray sampling, autodiff path |
| Main text, eq 1 | Secondary-ray complex amplitude `a = Ũ/p` |
| Main text, eq 2 | **C_RAY_TO_WAVE**: coherent wavelet sum at a plane, with the `⟨n̂, d̂⟩` projection factor |
| Main text, eq 3 | NCC figure of merit used for image-plane comparisons |
| Main text, eq 4 | Curvature error bound `ε_curv ≤ arcsin(D/2R)` |
| Main text, Figure 2 | Planar full-field patch rule: zero-pad to 2×, one FFT, off-centre rays as linear phase ramps |
| Main text, Figure 3 | Patch size / incident-ray count / secondary-ray count trade-off; curvature error vs patch size |
| Main text, Figure 4 | Monte Carlo convergence, `p_uni` vs `p_mag` |
| Main text, Figure 5 | Three benchmark systems: grating–lens, hologram, hologram–lens |
| SI S1 | Comparison against nine prior ray–wave frameworks |
| SI S2 | **Derivation** of secondary-ray sampling as an unbiased MC estimator of the per-patch ASM integral (eqs S1–S5) |
| SI S3 | **Derivation** of the curvature bound (eqs S6–S9) |
| SI S4, Algorithm S1 | **C_WAVE_TO_RAY** + cascade: accumulate, transmit, one FFT, resample a fixed `P×S` budget |
| SI S5 | Peak GPU memory for the sampling sweep |
| SI S6 | Monte Carlo ensemble convergence and undersampling artifacts |
| SI S7, Algorithms S2/S3 | Fixed-direction gradient estimator, and the rejected Gumbel–Softmax alternative |
| SI S8 | Computational cost tables |
| SI S9 | Inverse-design settings and loss trajectories |

## How M2 uses it

M2 authors `knowledge/couplers/ray_to_wave/` and
`knowledge/couplers/wave_to_ray/` **from this paper**, not from the reference
implementation. The reference implementation is neither vendored nor executed
here, so no claim in this repository may cite it as evidence. Where the paper
leaves a convention implicit, the coupler packs record it as an open question
and resolve it with an executable probe rather than a guess.
