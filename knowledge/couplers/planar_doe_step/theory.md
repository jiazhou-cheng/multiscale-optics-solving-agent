# C_PLANAR_DOE_STEP — SI Algorithm S1, and why the composition is exact in the limit

**Read first:** `knowledge/couplers/ray_to_wave/theory.md` and
`knowledge/couplers/wave_to_ray/theory.md`. This step is those two operators back
to back. Nothing here re-derives either of them; what follows is only what the
adjacency creates.

## The algorithm

One planar DOE interaction, in four moves:

1. **Accumulate.** Every incident ray is summed coherently onto one common
   Cartesian plane, giving a complex field `U_in(x,y)`. This is `C_RAY_TO_WAVE`
   unchanged, and its governing equation is the paper's main-text eq 2.
2. **Transmit.** Multiply once by the complex transmission `t(x,y)`, sampled on
   the same grid. `U_out = t * U_in`.
3. **Transform.** Take the angular spectrum `U~(k_x,k_y)` of `U_out`, optionally
   over a zero-padded aperture.
4. **Resample.** Draw a fixed budget of `P x S` outgoing rays — `P` launch
   positions on the plane, `S` directions per position — with directions drawn
   from the propagating bins. This is `C_WAVE_TO_RAY` unchanged.

Ray bundle in, ray bundle out. The field exists only between moves 1 and 4.

## Why this is a coupler and not a model

The source and target artifacts are both `ray_bundle`, which reads at first like
a model. It is a coupler because it *changes representation and back*, and
carries assumptions belonging to neither side:

* the accumulation onto a common plane is valid **only** because the surface is
  planar, and
* the interference that survives that accumulation is the entire reason the step
  exists. A model would leave the representation alone; this one destroys the
  per-ray identity of its input and rebuilds a different population.

## Why the outgoing count is the budget

Treating a DOE per-incident-ray multiplies: `N` incident rays each diffracting
into `M` orders gives `N*M`, and two stacked DOEs give `N*M^2`. The whole point
of Algorithm S1 is that the accumulation **erases the incident population**. The
field on the plane is a single object regardless of how many rays built it, so
the outgoing count is whatever the caller asks for and is independent of `N`.

That is the property the operator exists for, and it is asserted directly rather
than argued:
`tests/test_planar_doe_step.py::test_the_outgoing_count_does_not_depend_on_the_incident_count`
and `::test_two_stacked_does_keep_the_outgoing_count_at_the_budget`.

## The exactness limit, which is the only oracle this composition has

With `secondary_count=None` the step **enumerates** every propagating bin instead
of sampling it. There is then no sampling error, and the outgoing bundle is a
complete, deterministic representation of the transmitted field. Reconstructing
from it must return the transmitted field to dtype round-off.

That is the composition's oracle, and it is a `deterministic_limit` — not an
analytic closed form and not an independent implementation. It is genuinely
independent of the sampled path (it removes the estimator entirely rather than
comparing two configurations of it), which is why it can gate. It is also the
*only* thing that can: there is no analytic solution for an arbitrary DOE, and
the repository's own ASM propagator is not admissible as a decider for the
repository's own coupler.

Evidence:
`tests/test_planar_doe_step.py::test_full_enumeration_still_reproduces_the_transmitted_field`.

## What the exactness limit does not establish

It is one of the four kinds of evidence a stochastic operator needs, and the
other three are absent for the sampled step: **unbiasedness**, **convergence
exponent**, and **variance**. A sampled run has no stated error. Reporting an
accuracy from a single realization would be inventing one, and the ledger records
this gap rather than rounding it off.

## Where the phase reference goes

Because the incident optical path is already inside `U_in`'s phase, the outgoing
rays are given `OPL = 0`. This is not a convenience: carrying the incident path
forward would double-count it against the phase already accumulated.

The consequence is that **the step rebases the phase reference to its own plane**.
Downstream OPL is measured from here. Two stacked steps rebase twice. A caller
comparing an OPL across a step is comparing two different origins, and nothing in
the shapes or the intensities will say so — which is why the rebase is emitted in
the diagnostics on every run.

## Where the amplitude goes

The outgoing amplitude is `U~[m]/p[m]`: a spectral amplitude divided by the
probability with which that bin was drawn. The division is what keeps the
estimator unbiased, and it is why the amplitude is *not* a transformed incident
weight and carries no per-ray correspondence to the input.

`importance_weight_applied` is the invariant that guards it. Omitting the
division does not change the shape of anything; it changes the answer by the
sampling density, which on a peaked spectrum is a large factor concentrated
exactly where the signal is.

## Evanescent power

Bins outside the propagating circle carry no power to the far field. They are
accounted for explicitly — `evanescent_power_accounted` — rather than silently
dropped, so that a power ledger across the step balances and a caller can see how
much of the transmitted field never left the plane.
