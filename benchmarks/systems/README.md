# Layer-C systems

CHE-141 (M2.5). The home for **layer-C** benchmark specs, drivers and records —
physically meaningful end-to-end optical systems, as opposed to the primitive
qualification (layer A) and numerical-realization characterization (layer B)
that support them. See [`docs/benchmark_design.md`](../../docs/benchmark_design.md)
for the layer axis and its three consistency rules.

A layer-C family declares a `topology` of at least three stages and at least two
distinct observables. Both are refused at construction if absent, because a
system claim whose chain is not written down cannot be checked against the
system it claims to model, and because no system collapses to a single
threshold.

## What is here

`B3-4F-IDEAL` (CHE-144, M2.8) is the first rung of the system ladder: an ideal,
aberration-free FFT-based 4f relay, checked against a hand-derived analytic
Fourier-optics reference (Jacobi-Anger for a sinusoidal phase grating, the
two-level Fourier series for a binary phase grating, the trivial single-term
series for a pure carrier). Its family lives in
[`src/verification/families/b3_4f_ideal.py`](../../src/verification/families/b3_4f_ideal.py),
its driver at [`b3_4f_ideal.py`](b3_4f_ideal.py) in this directory, and its nine
canonical instances' records in [`records/`](records/). Run it::

    ./run.sh python benchmarks/systems/b3_4f_ideal.py --write

`B3-4F-REAL` and `B4-4F-REAL` (CHE-145, M2.9) are the second rung, and the
first that genuinely requires hybrid ray–wave modelling: M2.8's relay with the
ideal lenses replaced by two real Newport-KBX058-geometry N-BK7 singlets, the
modulation held **unchanged** (the same `_mask` constructor, imported from the
M2.8 driver rather than copied). The chain is `object field → C_WAVE_TO_RAY →
Optiland → DiffractiveInteraction(model=full_field) → Optiland →
C_RAY_TO_WAVE`. Both families live in one module,
[`src/verification/families/b3_4f_real.py`](../../src/verification/families/b3_4f_real.py),
and share one driver, [`b3_4f_real.py`](b3_4f_real.py).

The split is the ticket's own: there is no oracle for an aberrated 4f relay
carrying a high-frequency modulation, so `B3-4F-REAL` gates only the *paraxial
limit*, where the 4F-1 answer is the reference and third-order spherical
aberration's fourth-power law fixes the rate, while `B4-4F-REAL` is category B4
— structurally incapable of gating — and reports the measured departure away
from that limit as data. Run them separately; each is a few minutes::

    ./run.sh python benchmarks/systems/b3_4f_real.py --family B3-4F-REAL --write
    ./run.sh python benchmarks/systems/b3_4f_real.py --family B4-4F-REAL --write

Three sampling walls are declared as validity predicates rather than left in
comments, because each one caps a parameter range the ticket asks for: the
reachable field angle obeys `theta_max * R = (object_grid_n / 2 - 4 *
object_waist_pixels) * lambda / 2`, the aperture ceiling is where the
aberration's own residual ray angle stops fitting the shared plane's grid
(the margin crosses zero at 4.19 mm for `grid_n = 48`, with the 6.0 mm instance
recorded as the structured
refusal it produces), and the order copies have to stay separated at the
sensor — which is what puts M2.8's best-behaved `samples_per_period = 16` out of
reach here at a `grid_n ** 4` ray count.

The remaining rungs are authored M2.10 onward — the canonical refractive→wave
boundary, the terminal-DOE hybrid, the embedded diffractive system, the
conformal metasurface, and the SLM relay.

## Existing layer-C evidence is re-homed by classification, not moved on disk

`B3-PSF-SINGLET`, `B3-DEMO2` and `B4-DEMO3` are layer C today. Their families
live in `src/verification/families/`, their drivers in `benchmarks/instances/`
and their records in `benchmarks/instances/records/`, and they **stay there**.

That is a deliberate decision, not an omission. Those records carry committed
scientific fingerprints; relocating 58 of them to express a taxonomy change
would invalidate every one for no scientific gain, and the classification is
already expressed by a field on the family. The generated layer view in
[`../INVENTORY.md`](../INVENTORY.md) and
[`../validation/coverage_matrix.md`](../validation/coverage_matrix.md) is what
makes "which evidence supports this system claim" answerable without reading
source.

So this directory is where *new* layer-C artifacts go. It is not a place that
existing evidence is migrating to.
