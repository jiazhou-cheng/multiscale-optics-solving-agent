"""B3-4F-IDEAL: the ideal 4f relay, the first rung of the M2 system ladder.

CHE-144 (M2.8). The topology is ``input field -> ideal FT -> Fourier-plane
modulation -> ideal inverse FT -> image field``. Ideal is the point: no real
lens, no aberration, and therefore a system that is wave-solvable in closed
form. What is checked is whether the repository's own discrete realization of
that idealization -- ``fft2``/``ifft2`` sandwiching a Fourier-plane mask
multiply -- reproduces the *continuous* analytic Fourier-optics prediction for
a periodic phase modulation, and where finite-grid sampling stops matching it.

What the modulation does, physically
-------------------------------------
A periodic mask ``M(fx)`` sitting at the Fourier plane is, by the convolution
theorem, equivalent to convolving the object with ``IFFT[M]`` in the image
plane. A mask periodic in ``fx`` with period ``Lf`` decomposes into a Fourier
series ``M(fx) = sum_n c_n exp(i 2 pi n fx / Lf)``, and each term is a pure
shift theorem: it places a copy of the object at ``x0_n = -n / Lf``, weighted
by ``c_n``. That sign -- order ``n`` lands at ``-n/Lf``, not ``+n/Lf`` -- was
derived and confirmed to 1e-12 relative against ``scipy.special.jv`` before
this family was written; getting it backwards silently relabels order ``n`` as
order ``-n``, which for the sinusoidal grating is invisible in POWER (``J_n``
and ``J_-n`` have equal magnitude) and only shows up as a phase error, which is
exactly why this family measures phase as an independent observable rather
than power alone.

The closed forms this family checks against
---------------------------------------------
``grating_order_coefficient`` is the independent side: a hand-derived Fourier
series, evaluated with ``scipy.special.jv`` for the sinusoidal case and plain
arithmetic for the binary and carrier cases. It shares no code with the
driver's ``fft2``/``ifft2`` realization in
``benchmarks/systems/b3_4f_ideal.py``, which is what makes it an admissible B3
decider rather than a second route through the same implementation.

Why the validity envelope is keyed to the FUNDAMENTAL order only
------------------------------------------------------------------
``samples_per_period`` is how many frequency-grid samples span one period of
the mask. Order ``n``'s own Nyquist limit is ``samples_per_period >= 2n``, so
order 3 aliases at ``samples_per_period = 6``, long before the fundamental
(order 1) aliases at ``samples_per_period = 2``. The declared
``FFT_GRID_NYQUIST`` predicate uses the fundamental's limit -- the latest one
to break -- and says so in its ``blind_to``: an instance can report INSIDE
while a higher order it still checks has already aliased, and that is a
property of the measurement, not a defect in the predicate.

What this family does not attempt
-----------------------------------
No real lens, no ray propagation, no partial coherence, no polarization --
those are M2.9 and later rungs, or out of scope entirely. No adapter or
registry change: there is no ``M_WAVE_CHROMATIX`` capability for an ideal
lens's Fourier-transforming property, and adding one is not required by this
family's acceptance criteria, so the ideal transform is realized directly with
``numpy.fft`` in the driver rather than by extending a shared adapter.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from scipy.special import jv  # type: ignore[import-untyped]

from core.precision import ArrayNamespace, DeviceKind, DType
from verification.families.predicates import fractional_margin
from verification.families.registry import register
from verification.families.schema import (
    BenchmarkCategory,
    BenchmarkFamily,
    BenchmarkLayer,
    ClaimKind,
    ExecutionParameter,
    ExecutionPolicy,
    FamilyOracle,
    GateDisposition,
    GateStatus,
    Invariant,
    Metric,
    NegativeControl,
    NumericalParameter,
    Oracle,
    OracleIndependence,
    PhysicalParameter,
    SamplerAbsentReason,
    StochasticPolicy,
    Tolerance,
    ToleranceBasis,
    ValidityBasis,
    ValidityPredicate,
)
from verification.status import VerificationStatus

__all__ = [
    "B3_4F_IDEAL",
    "CHECKED_ORDERS",
    "grating_order_coefficient",
    "order_coefficients",
]

#: The diffraction orders every instance checks: the fundamental and DC, which
#: is also exactly what CHE-144's own independent-reference section names --
#: "three orders with Bessel/J_n weights". Fixed at three rather than swept
#: wider, so the checked set's own Nyquist limit (order n aliases at
#: samples_per_period = 2n) coincides with the declared FFT_GRID_NYQUIST
#: predicate's fundamental-only bound instead of aliasing earlier than the
#: predicate can see.
CHECKED_ORDERS: tuple[int, ...] = (-1, 0, 1)


def grating_order_coefficient(modulation_type: str, n: int, phase_depth_rad: float) -> complex:
    """The Fourier-plane mask's own Fourier-series coefficient at harmonic ``n``.

    Independent of the FFT: derived by hand from the mask's definition, not
    computed by transforming it. ``sinusoidal_phase`` is the Jacobi-Anger
    expansion of ``exp(i m sin(theta))``; ``binary_phase`` is the standard
    50%-duty-cycle two-level phase grating's Fourier series (derived directly:
    ``c_0 = (1 + exp(i phi0)) / 2``, even ``n != 0`` vanish, odd ``n`` carry
    ``(exp(i phi0) - 1) / (i pi n)``); ``pure_carrier`` is a single exponential
    term and therefore, by definition, has exactly one nonzero order.
    """
    if modulation_type == "sinusoidal_phase":
        return complex(jv(n, phase_depth_rad))
    if modulation_type == "binary_phase":
        phi0 = phase_depth_rad
        top = complex(math.cos(phi0), math.sin(phi0))
        if n == 0:
            return (1.0 + top) / 2.0
        if n % 2 == 0:
            return 0j
        return (top - 1.0) / (1j * math.pi * n)
    if modulation_type == "pure_carrier":
        return 1.0 + 0j if n == 1 else 0j
    raise ValueError(
        f"unknown modulation_type {modulation_type!r}; have "
        "sinusoidal_phase, binary_phase, pure_carrier"
    )


def order_coefficients(params: Mapping[str, Any]) -> dict[int, complex]:
    """The oracle: every checked order's analytic coefficient for this instance."""
    modulation_type = str(params["modulation_type"])
    phase_depth_rad = float(params["phase_depth_rad"])
    return {
        n: grating_order_coefficient(modulation_type, n, phase_depth_rad) for n in CHECKED_ORDERS
    }


def _fft_grid_nyquist_margin(params: Mapping[str, Any]) -> float:
    return fractional_margin(2.0, float(params["samples_per_period"]))


FFT_GRID_NYQUIST_PREDICATE = ValidityPredicate(
    predicate_id="FFT_GRID_NYQUIST",
    statement=(
        "the Fourier-plane modulation's own period stays resolved by the frequency "
        "grid: samples_per_period >= 2, the Nyquist limit of the fundamental harmonic"
    ),
    basis=ValidityBasis.FFT_GRID_NYQUIST,
    margin=_fft_grid_nyquist_margin,
    blind_to=(
        "the actual, modulation-depth-dependent onset of aliasing, which arrives "
        "well before this bound. Confirmed while authoring this family: at "
        "samples_per_period=8 -- margin 0.75, deep INSIDE by this predicate -- "
        "the checked fundamental already carries a 2.4e-5 aliasing contamination "
        "from the mask's own Fourier tail (J_{1+8k}(phase_depth_rad) folding back "
        "for integer k), growing to 0.179 relative by samples_per_period=4. The "
        "true resolving requirement is samples_per_period large enough that the "
        "aliased tail is negligible, which depends on phase_depth_rad and is not "
        "expressible as a fixed Nyquist number the way this predicate is",
    ),
)


IDEAL_4F_EXECUTION = ExecutionPolicy(
    devices=frozenset({DeviceKind.CPU}),
    dtypes=frozenset({DType.COMPLEX128}),
    namespaces=frozenset({ArrayNamespace.NUMPY}),
    max_wall_seconds=60.0,
    max_peak_memory_gib=1.0,
    notes=(
        "plain numpy fft2/ifft2 over a grid no larger than 1024x1024; no solver "
        "adapter is invoked, because the ideal Fourier-transforming property of a "
        "lens is not an M_WAVE_CHROMATIX capability and adding one is not required "
        "by this family -- that is deliberately left to a real lens in M2.9"
    ),
)

DETERMINISTIC = StochasticPolicy(
    is_stochastic=False,
    determinism_reason=(
        "a deterministic FFT of a deterministically built object field and mask; "
        "nothing here draws a sample"
    ),
)


_METRIC_ORDER_POWER = Metric(
    name="order_power_relative_l2",
    description=(
        "relative L2, over the checked order set, between each order's measured "
        "power -- image-plane intensity at its predicted location, normalized by "
        "the object's own peak intensity -- and the analytic |c_n|^2"
    ),
    unit=None,
    blind_to=(
        "phase entirely -- two coefficients with equal modulus and opposite sign "
        "score identically here, which is exactly the order_phase_error_rad hazard "
        "this family's own sign derivation found while being authored",
        "which order carries the discrepancy: this is a norm over the checked set, "
        "not a per-order map",
    ),
)

_METRIC_ORDER_PHASE = Metric(
    name="order_phase_error_rad",
    description=(
        "RMS phase error, in radians, between each order's measured complex "
        "coefficient and its analytic c_n, restricted to orders whose analytic "
        "power exceeds a 1e-6 floor -- a near-zero-power order's phase is "
        "undefined and excluded rather than reported as a spurious large error"
    ),
    unit="rad",
    blind_to=(
        "any order with negligible analytic power, excluded by construction",
        "amplitude -- a coefficient with the right phase and the wrong modulus "
        "scores zero here",
    ),
)

_METRIC_ORDER_LOCATION = Metric(
    name="order_location_error_frac",
    description=(
        "RMS, over the checked order set, of the peak-search offset from the "
        "analytically predicted shift x0_n = -n / Lf, as a fraction of the "
        "fundamental shift x0_1 = 1 / Lf"
    ),
    unit=None,
    blind_to=(
        "power and phase -- a peak found at exactly the right place with the "
        "wrong height or phase scores zero here",
    ),
)

_METRIC_POWER_CONSERVATION = Metric(
    name="total_power_relative_error",
    description=(
        "|sum|image|^2 / sum|object|^2 - 1| -- how far total power departs from "
        "exact conservation across the relay"
    ),
    unit=None,
    blind_to=(
        "where the power went if it did not conserve -- reported beside the "
        "order-resolved metrics rather than instead of them",
    ),
)


_ORDER_POWER_BASIS = (
    "B3-4F-IDEAL-SIN-01 (sinusoidal_phase, samples_per_period=16, grid_n=512) "
    "measures 2.99e-14, and B3-4F-IDEAL-CARRIER-SNAPPED measures 4.8e-33; the "
    "residual at coarser sampling is not float64 round-off but the mask's own "
    "Fourier tail aliasing back onto the checked orders, and it is exactly "
    "accounted for: at samples_per_period=8 the checked order n=1 measures "
    "0.5579120236816821, and J_1(1.5) + sum_k J_{1+8k}(1.5) over the neighbouring "
    "aliases reproduces it to 2.2e-16, while J_1(1.5) alone is off by 2.4e-5 -- "
    "the same aliasing mechanism this predicate's blind_to names, confirmed "
    "against scipy.special.jv while authoring this family. 1e-6 sits between the "
    "round-off floor and the smallest aliasing effect this sweep resolves, so it "
    "rejects a wrong closed form or a real aliasing fold rather than absorbing "
    "one. It is NOT met at every declared-INSIDE instance -- B3-4F-IDEAL-SIN-02 "
    "(samples_per_period=8) measures 7.6e-5 and B3-4F-IDEAL-SIN-03 "
    "(samples_per_period=4) measures 0.179, both comfortably inside the "
    "fundamental-only FFT_GRID_NYQUIST bound -- which is the family's own "
    "observed answer to CHE-144's question of where sampling stops matching the "
    "continuous theory: considerably sooner than the naive Nyquist wall, and "
    "depending on the modulation depth. See the family's gate_disposition"
)

_ORDER_PHASE_BASIS = (
    "the same basis as order_power_relative_l2, in radians. "
    "B3-4F-IDEAL-SIN-01 measures 1.42e-16 rad; B3-4F-IDEAL-BIN-01 (the only "
    "canonical instance with genuinely complex analytic coefficients) measures "
    "0.160 rad, entirely attributable to the same finite-samples_per_period "
    "aliasing tail as the power metric, not to a phase-sign defect -- confirmed "
    "by the phasor-sign-flip control firing cleanly on that same instance. 1e-6 "
    "rad rejects the sign confusion this family's own derivation found while "
    "being authored: mislabeling order n as order -n leaves J_n and J_-n's "
    "magnitudes indistinguishable but is off in phase by up to pi"
)

_ORDER_LOCATION_BASIS = (
    "inside validity every checked order lands on an exact integer pixel offset "
    "by construction (samples_per_period chosen so the grid is commensurate with "
    "the mask period), so the peak search finds it exactly: measured 0.0 on "
    "every declared instance except B3-4F-IDEAL-SIN-04 (samples_per_period=2, "
    "declared NEAR_BOUNDARY, measures 6.4e-3 as the fundamental order's own peak "
    "starts to smear at Nyquist itself). 1e-6 of the fundamental shift rejects a "
    "real aliasing fold, a transposed axis, or a 2*pi frequency-scale error, each "
    "of which moves a peak by an integer number of pixels comparable to the "
    "shift itself"
)

_POWER_CONSERVATION_BASIS = (
    "every declared modulation is phase-only (unit modulus everywhere on the "
    "Fourier plane), so Parseval's theorem makes total power exact across an "
    "orthonormal FFT pair regardless of validity -- an aliased order moves energy "
    "to the wrong place, it does not create or destroy it. 1e-9 is the float64 "
    "round-off floor for a sum over a grid no larger than 1024x1024"
)


B3_4F_IDEAL = register(
    BenchmarkFamily(
        family_id="B3-4F-IDEAL",
        family_version="1.0.0",
        category=BenchmarkCategory.B3,
        layer=BenchmarkLayer.SYSTEM,
        topology=(
            "input field",
            "ideal Fourier transform",
            "Fourier-plane modulation",
            "ideal inverse Fourier transform",
            "image field",
        ),
        question=(
            "does an ideal, aberration-free FFT-based 4f relay -- FT, Fourier-plane "
            "modulation, inverse FT -- reproduce the analytic Fourier-optics "
            "diffraction-order prediction for a periodic phase modulation, and where "
            "does finite-grid sampling stop matching the continuous analytic theory?"
        ),
        components=("M_WAVE_CHROMATIX",),
        claim_kind=ClaimKind.FORWARD_ACCURACY,
        parameters=(
            PhysicalParameter(
                "modulation_type",
                "which Fourier-plane mask: sinusoidal phase grating, 50%-duty binary "
                "phase grating, or a pure single-order carrier",
                domain=("sinusoidal_phase", "binary_phase", "pure_carrier"),
                default="sinusoidal_phase",
            ),
            PhysicalParameter(
                "samples_per_period",
                "how many frequency-grid samples span one period of the Fourier-plane "
                "mask. The frequency axis the ticket asks this family to sweep, "
                "expressed as a Nyquist-normalized number rather than a physical "
                "frequency so the envelope reads the same at any grid size",
                domain=(1.0, 64.0),
                default=16.0,
            ),
            PhysicalParameter(
                "phase_depth_rad",
                "the modulation depth: m for the sinusoidal grating, phi0 for the "
                "binary grating. Unused (fixed at 0) for pure_carrier, which has no "
                "depth parameter of its own",
                unit="rad",
                domain=(0.0, 6.0),
                default=1.5,
            ),
            NumericalParameter(
                "grid_n",
                "the square FFT grid's side length. Larger reduces the object's own "
                "edge-truncation error; it does not move the analytic answer",
                domain=(64, 1024),
                default=512,
                refines_toward=1,
            ),
            NumericalParameter(
                "object_waist_pixels",
                "the narrow object's 1/e amplitude half-width, in pixels. Smaller "
                "keeps the shifted order copies from overlapping; it does not move "
                "the analytic answer",
                domain=(1.0, 20.0),
                default=3.0,
                refines_toward=-1,
            ),
            ExecutionParameter(
                "device",
                "cpu only, per this family's execution policy",
                domain=("cpu",),
                default="cpu",
            ),
        ),
        validity=(FFT_GRID_NYQUIST_PREDICATE,),
        oracle=FamilyOracle(
            kind=Oracle.ANALYTIC,
            independence=OracleIndependence.INDEPENDENT,
            description=(
                "the Fourier-plane mask's own Fourier-series coefficients: Jacobi-Anger "
                "for the sinusoidal grating, the standard two-level Fourier series for "
                "the binary grating, and the trivial single-term series for a pure "
                "carrier -- hand-derived, sharing no code with the driver's fft2/ifft2"
            ),
            callable=order_coefficients,
            reference="src/verification/families/b3_4f_ideal.py::grating_order_coefficient",
        ),
        metrics=(
            _METRIC_ORDER_POWER,
            _METRIC_ORDER_PHASE,
            _METRIC_ORDER_LOCATION,
            _METRIC_POWER_CONSERVATION,
        ),
        invariants=(
            Invariant(
                invariant_id="TOTAL_POWER_CONSERVED",
                statement=(
                    "a phase-only Fourier-plane mask conserves total power exactly, "
                    "regardless of whether the checked orders are resolved correctly"
                ),
                metric="total_power_relative_error",
                tolerance=Tolerance(
                    metric="total_power_relative_error",
                    threshold=1e-9,
                    basis=_POWER_CONSERVATION_BASIS,
                    basis_kind=ToleranceBasis.CONSERVATION_LAW,
                    may_gate=True,
                    rejects=(
                        "a mask that is not actually unit-modulus, or a border "
                        "row/column dropped in bookkeeping"
                    ),
                ),
            ),
        ),
        tolerances=(
            Tolerance(
                metric="order_power_relative_l2",
                threshold=1e-6,
                basis=_ORDER_POWER_BASIS,
                basis_kind=ToleranceBasis.ANALYTIC_DERIVATION,
                may_gate=True,
                rejects=(
                    "a wrong closed form, a mislabeled order, or a real aliasing "
                    "fold past the declared envelope"
                ),
            ),
            Tolerance(
                metric="order_phase_error_rad",
                threshold=1e-6,
                basis=_ORDER_PHASE_BASIS,
                basis_kind=ToleranceBasis.ANALYTIC_DERIVATION,
                may_gate=True,
                rejects=(
                    "the order-n/-n sign confusion this family's own derivation "
                    "found, and any other phasor-sign error"
                ),
            ),
            Tolerance(
                metric="order_location_error_frac",
                threshold=1e-6,
                basis=_ORDER_LOCATION_BASIS,
                basis_kind=ToleranceBasis.ANALYTIC_DERIVATION,
                may_gate=True,
                rejects=(
                    "an aliasing fold, a transposed axis, or a 2*pi "
                    "frequency-scale error, each of which moves a peak by an "
                    "integer number of pixels"
                ),
            ),
        ),
        negative_controls=(
            NegativeControl(
                control_id="phasor-sign-flip",
                description=(
                    "negate the measured order phase before comparing to the "
                    "analytic phase"
                ),
                mutation=(
                    "measured order phase -> -measured order phase, then "
                    "recompute order_phase_error_rad"
                ),
                target_metric="order_phase_error_rad",
            ),
            NegativeControl(
                control_id="axis-transpose",
                description="transpose the measured image before reading off order power",
                mutation="image -> image.T, then recompute order_power_relative_l2",
                target_metric="order_power_relative_l2",
            ),
            NegativeControl(
                control_id="modulation-in-image-plane",
                description=(
                    "apply the Fourier-plane mask directly to the object field "
                    "instead of at the Fourier plane"
                ),
                mutation=(
                    "image -> object * mask (no fft2/ifft2 sandwich), then "
                    "recompute order_power_relative_l2"
                ),
                target_metric="order_power_relative_l2",
            ),
            NegativeControl(
                control_id="frequency-grid-two-pi",
                description="scale the mask period by 2*pi before predicting order locations",
                mutation=(
                    "Lf -> Lf * 2*pi in the predicted-location formula only, "
                    "then recompute order_location_error_frac"
                ),
                target_metric="order_location_error_frac",
            ),
            NegativeControl(
                control_id="grid-snapped-vs-continuous-carrier",
                description=(
                    "compare a grid-commensurate carrier instance against an "
                    "otherwise-identical off-grid one, which cannot be exactly "
                    "represented by the discrete FFT and must show spectral leakage"
                ),
                mutation=(
                    "samples_per_period moved off an integer divisor of "
                    "grid_n, holding everything else fixed"
                ),
                target_metric="order_power_relative_l2",
            ),
        ),
        failure_semantics=(
            VerificationStatus.OUT_OF_VALIDITY,
            VerificationStatus.LOSSY_BUT_ALLOWED,
        ),
        execution_policy=IDEAL_4F_EXECUTION,
        stochastic_policy=DETERMINISTIC,
        gate_disposition=GateDisposition(
            status=GateStatus.NOT_MET,
            metric="order_power_relative_l2",
            observed=0.179240407985276,
            evidence=(
                "benchmarks/systems/records/B3-4F-IDEAL-SIN-01.json",
                "benchmarks/systems/records/B3-4F-IDEAL-SIN-02.json",
                "benchmarks/systems/records/B3-4F-IDEAL-SIN-03.json",
                "benchmarks/systems/records/B3-4F-IDEAL-BIN-01.json",
                "benchmarks/systems/records/B3-4F-IDEAL-CARRIER-SNAPPED.json",
            ),
            note=(
                "carried forward unwidened, not hidden. The deepest-inside instances "
                "(B3-4F-IDEAL-SIN-01 at samples_per_period=16, "
                "B3-4F-IDEAL-CARRIER-SNAPPED) meet 1e-6 to the float64 floor. Two "
                "instances this family itself declares INSIDE by FFT_GRID_NYQUIST do "
                "not: B3-4F-IDEAL-SIN-02 (samples_per_period=8) measures 7.6e-5 and "
                "B3-4F-IDEAL-SIN-03 (samples_per_period=4) measures 0.179 -- both fully "
                "attributed to the mask's own Fourier tail aliasing back onto the "
                "checked orders (confirmed against scipy.special.jv; see "
                "FFT_GRID_NYQUIST's blind_to), not to a defect. B3-4F-IDEAL-BIN-01 "
                "(binary_phase, also declared INSIDE) measures 1.05e-2, because a "
                "discontinuous mask's Fourier series decays as O(1/n) rather than "
                "the sinusoidal grating's superexponential Bessel decay, so its "
                "aliasing tail is far larger at the same samples_per_period and does "
                "not fall under 1e-6 anywhere in this family's declared domain "
                "(checked up to samples_per_period=256; still 1.3e-3 there). This IS "
                "CHE-144's own requested answer to 'where does the system stop being "
                "modelled correctly' -- the real boundary is modulation-dependent and "
                "considerably tighter than the naive Nyquist wall this family's single "
                "validity predicate declares."
            ),
        ),
        sampler=None,
        sampler_absent_reason=SamplerAbsentReason.STABLE_FINGERPRINT_VALUABLE,
        sampler_absent_note=(
            "this is the baseline every later system-ladder rung (M2.9 onward) is "
            "read against; the canonical frequency sweep is the thing being compared "
            "across rungs, so it stays fixed rather than generated"
        ),
        evidence=("benchmarks/systems/README.md",),
        notes=(
            "Scope, deliberately: this family measures order power, order phase, "
            "order location and total power. It does NOT reconstruct a full "
            "image-plane field from a truncated order series and compare it by "
            "relative_l2_field, and it does not report a residual-spectrum plot -- "
            "both are named as candidate observables in CHE-144 but are not required "
            "by its acceptance criteria, and adding them is left as follow-up rather "
            "than expanding this family's scope."
        ),
    )
)
