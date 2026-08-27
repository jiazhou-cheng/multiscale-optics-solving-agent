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

`B3-DOE-INLINE` and `B4-DOE-INLINE` (M2.12) are the **embedded** diffractive
system: the DOE sits *inside* the refractive train, so the interaction has to
hand back rays and downstream refractive optics has to keep working. The chain is
`ray → DiffractiveInteraction(model=generalized_snell) → ray → Optiland →
C_RAY_TO_WAVE`, with no reconstruction between the interaction and the trace —
which is what makes this the only rung where the interaction's *ray output* is
load-bearing. Two topologies run: a collimated bundle into a linear phase ramp in
front of a real singlet (the one system with a conventional reference — the
textbook `x = f tan(arcsin(m λ / Λ))`), and a converging bundle out of one
singlet through the ramp into a second that relays the intermediate image at
−1.914×, where the incident optical path and the incident direction are per-ray
rather than constant. Both families live in
[`src/verification/families/b3_doe_inline.py`](../../src/verification/families/b3_doe_inline.py)
and share one driver, [`b3_doe_inline.py`](b3_doe_inline.py). Each family is
about 30 s::

    ./run.sh python benchmarks/systems/b3_doe_inline.py --family B3-DOE-INLINE --write
    ./run.sh python benchmarks/systems/b3_doe_inline.py --family B4-DOE-INLINE --write

The split is the same as the 4f rung's and for the same reason. `B3-DOE-INLINE`
is a **convention** claim and gates: the grating equation to 1e-14 in direction
cosine, `opl_in + m φ / k0` reproduced from the declaration, `|a_in| |t|` for the
amplitude, ray power conserved exactly, and the zero-phase limit *bitwise*
identical to the plain refractive train — plus the analytic order position over
three grating periods, whose tolerance is derived from a validity bound rather
than read off a run. `B4-DOE-INLINE` is category B4 and measures the interference
structure an aberrated singlet puts in a diffracted order against the
diffraction-blurred ray density, whose fringe contrast is **exactly zero** at
every aperture measured while the coherent field's is 0.99 at R = 1 mm.

This rung is also what found the order/OPL inconsistency in
[`src/couplers/generalized_snell.py`](../../src/couplers/generalized_snell.py):
the momentum equation carried `m grad(φ)` while the optical path carried `φ`
rather than `m φ`, so for `|m| ≠ 1` the rays were deflected as if the phase were
one thing and given the optical path of another. `order=1` is bitwise unaffected;
`tests/test_diffractive_interaction.py` pins the two identities that decide the
correct form.

The remaining rungs are authored elsewhere — the canonical refractive→wave
boundary, the terminal-DOE hybrid, the conformal metasurface, and the SLM relay.

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
