"""One definition per metric, and what each one cannot see.

CHE-112 (M2.4), promoted from an M2 deliverable to a substrate one because every
family in B0-B4 references it.

The argument for centralizing is the current state. NCC, relative L2 on field,
relative L2 on intensity, power ratio and MSE-unit-sum are each computed in more
than one place -- ``benchmarks/probes/ray_wave/_demo_support.py``,
``verification/psf_measurement.py``, and inline in several probes -- so a
benchmark and its own probe can disagree about what a number means. They have
already produced wrong conclusions elsewhere: relative L2 on *intensity* and on
*field* are different numbers, and confusing them is easy because the name is
the same.

Every metric here states what it is **blind to**. That is not decoration; it is
the field a reader needs most, and each entry below names something that has
actually gone wrong or could:

* NCC is blind to absolute scale, so it cannot see the power loss that the
  k-space route's 0.9832 ratio shows. A route can lose 1.7% of the energy and
  correlate at 0.9999.
* A centred metric can be blind to an off-axis defect. That is CHE-44's concern,
  which was never audited, and it is why every metric here declares whether it is
  evaluated over the whole array or over a centred region.
* A round-trip metric is blind to any error that is its own inverse. The backward
  pass undoes whatever the forward pass did, so a wrong ``k_z`` magnitude
  round-trips perfectly.

What this module is not
-----------------------
It is not a place to put physics. Every function here is arithmetic over two
arrays that somebody else computed. A metric that had to know what a pupil was
would be a measurement, and measurements live in ``psf_measurement.py``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "METRICS",
    "MetricDefinition",
    "central_relative_l2_intensity",
    "metric",
    "mse_unit_sum",
    "ncc",
    "ncc_uncentred",
    "power_ratio",
    "relative_l2_field",
    "relative_l2_intensity",
    "relative_rms",
]

_Array = np.ndarray[Any, Any]


def _flat(value: Any) -> _Array:
    return np.asarray(value).ravel()


# ---------------------------------------------------------------------------
# The definitions
# ---------------------------------------------------------------------------


def ncc(a: Any, b: Any) -> float:
    """Zero-mean normalized cross-correlation of two real images.

    Mean-subtracted (Pearson), which is the usual reading of "NCC" for image
    similarity and the only one insensitive to an additive pedestal. Moved here
    from ``_demo_support.py`` unchanged: it is the paper's headline metric and
    every demo number was produced with this definition.
    """
    x = _flat(a).astype(np.float64)
    y = _flat(b).astype(np.float64)
    x = x - x.mean()
    y = y - y.mean()
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    return float(x @ y / denominator) if denominator > 0.0 else float("nan")


def ncc_uncentred(a: Any, b: Any) -> float:
    """NCC without mean subtraction.

    Reported beside :func:`ncc` rather than instead of it, because the two
    differ noticeably on a speckle field with a bright DC lobe and a reader
    cannot tell from the number alone which was used.
    """
    x = _flat(a).astype(np.float64)
    y = _flat(b).astype(np.float64)
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    return float(x @ y / denominator) if denominator > 0.0 else float("nan")


def relative_l2_field(a: Any, b: Any) -> float:
    """``||a - b|| / ||b||`` on the COMPLEX field. Sensitive to phase."""
    reference = np.asarray(b)
    denominator = float(np.linalg.norm(reference))
    if denominator == 0.0:
        return float("nan")
    return float(np.linalg.norm(np.asarray(a) - reference) / denominator)


def relative_l2_intensity(a: Any, b: Any) -> float:
    """``|| |a|^2 - |b|^2 || / || |b|^2 ||``. A different number from the field one.

    Both are called "relative L2" in the literature and in this repository's own
    history, and they answer different questions: a wavefront with the right
    modulus and the wrong curvature scores zero here and badly on the field
    metric.
    """
    x = np.abs(np.asarray(a)) ** 2
    y = np.abs(np.asarray(b)) ** 2
    denominator = float(np.linalg.norm(y))
    if denominator == 0.0:
        return float("nan")
    return float(np.linalg.norm(x - y) / denominator)


def relative_rms(a: Any, b: Any) -> float:
    """RMS difference over RMS reference. The round-trip metric.

    Distinct from :func:`relative_l2_field` only by a factor of sqrt(N) that
    cancels -- they are numerically equal for equally-sized arrays -- and named
    separately because the round-trip literature reports RMS and a reader
    matching a number against a paper should not have to work that out.
    """
    return relative_l2_field(a, b)


def power_ratio(a: Any, b: Any) -> float:
    """Total power in ``a`` over total power in ``b``.

    The metric NCC cannot see. 1.0 is conservation; the k-space route on demo3
    at 8x oversampling reads 0.9832.
    """
    numerator = float(np.sum(np.abs(np.asarray(a)) ** 2))
    denominator = float(np.sum(np.abs(np.asarray(b)) ** 2))
    return numerator / denominator if denominator > 0.0 else float("nan")


def mse_unit_sum(a: Any, b: Any) -> float:
    """MSE after normalizing each intensity image to unit total power.

    Declared rather than assumed: an MSE between two intensities is meaningless
    until their scales are tied together, and the patch route's absolute scale
    depends on the coverage correction and the ray budget. Unit *sum* rather
    than unit *max*, because a single hot speckle pixel would otherwise set the
    scale for the whole comparison.
    """
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    x_sum, y_sum = float(x.sum()), float(y.sum())
    if x_sum == 0.0 or y_sum == 0.0:
        return float("nan")
    return float(np.mean((x / x_sum - y / y_sum) ** 2))


def central_relative_l2_intensity(a: Any, b: Any, *, fraction: float = 0.5) -> float:
    """:func:`relative_l2_intensity` over a centred sub-window.

    CHE-44's concern, made explicit rather than implicit. A gate disc around the
    axis is the right window for an on-axis PSF and the wrong one for anything
    off axis, and the difference between this and the full-array metric is the
    measurement of that blindness. Reporting both is how an off-axis defect stops
    being invisible.
    """
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    x = np.asarray(a)
    y = np.asarray(b)
    if x.ndim != 2 or x.shape != y.shape:
        raise ValueError("a centred window needs two arrays of the same 2-D shape")
    n_y, n_x = x.shape
    half_y = max(1, round(n_y * fraction / 2))
    half_x = max(1, round(n_x * fraction / 2))
    cy, cx = n_y // 2, n_x // 2
    window = (slice(cy - half_y, cy + half_y), slice(cx - half_x, cx + half_x))
    return relative_l2_intensity(x[window], y[window])


# ---------------------------------------------------------------------------
# The register
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MetricDefinition:
    """One metric: what it computes, what question it answers, what it misses."""

    name: str
    fn: Callable[..., float]
    #: The physical question this number answers, in a sentence.
    answers: str
    #: What it cannot see. Required, and the most-read field here.
    blind_to: tuple[str, ...]
    unit: str | None = None
    #: Whether the metric is evaluated over the whole array or a centred window.
    #: CHE-44: a centred metric cannot see an off-axis error, and which one a
    #: number came from must not be something a reader has to guess.
    support: str = "whole_array"
    #: The value a perfect agreement takes, so a reader knows which direction is
    #: better without inferring it from the name.
    ideal: float = 0.0

    def __post_init__(self) -> None:
        if not self.blind_to:
            raise ValueError(f"{self.name}: state what this metric cannot see")
        if not self.answers.strip():
            raise ValueError(f"{self.name}: state which physical question it answers")

    def __call__(self, a: Any, b: Any, **kwargs: Any) -> float:
        return float(self.fn(a, b, **kwargs))


METRICS: Mapping[str, MetricDefinition] = {
    definition.name: definition
    for definition in (
        MetricDefinition(
            name="ncc",
            fn=ncc,
            answers="do these two intensity patterns have the same structure?",
            blind_to=(
                "absolute scale, by construction. A route that loses 1.7% of the power "
                "correlates at 0.9999, which is exactly the k-space route on demo3",
                "an additive pedestal, which is the point of the mean subtraction and "
                "also means a raised background is invisible",
                "a translation of a few samples, which correlates well and is a "
                "different field",
            ),
            ideal=1.0,
        ),
        MetricDefinition(
            name="ncc_uncentred",
            fn=ncc_uncentred,
            answers="the same question, without discarding the DC component",
            blind_to=(
                "absolute scale, as the centred version is",
                "nothing about the pedestal -- which is why it differs noticeably from "
                "ncc on a speckle field with a bright DC lobe, and why both are reported",
            ),
            ideal=1.0,
        ),
        MetricDefinition(
            name="relative_l2_field",
            fn=relative_l2_field,
            answers="how far is this complex field from the reference, amplitude and phase?",
            blind_to=(
                "nothing about phase -- this is the metric that sees it, which is why "
                "it must not be confused with the intensity version",
                "where the error is: a large local error and a small global one can "
                "produce the same norm",
            ),
        ),
        MetricDefinition(
            name="relative_l2_intensity",
            fn=relative_l2_intensity,
            answers="how far is this intensity pattern from the reference?",
            blind_to=(
                "phase entirely. A wavefront with the right modulus and the wrong "
                "curvature scores zero here",
                "a global scale factor is NOT normalized away, unlike NCC -- so this "
                "number moves when the power does, which is usually what is wanted",
            ),
        ),
        MetricDefinition(
            name="relative_rms",
            fn=relative_rms,
            answers="the round-trip question: did the field come back?",
            blind_to=(
                "any error that is its OWN INVERSE. The backward pass undoes whatever "
                "the forward pass did, so a wrong k_z magnitude round-trips perfectly. "
                "This is the reason a round trip needs a deliberately broken twin: one "
                "that cannot be made to fail has proved nothing",
                "evanescent content, which is removed on the way out and cannot return",
            ),
        ),
        MetricDefinition(
            name="power_ratio",
            fn=power_ratio,
            answers="was energy conserved across this representation change?",
            blind_to=(
                "where the power went. A ratio of 1.0 with the energy in the wrong "
                "place is invisible here, which is why it is reported beside an "
                "accuracy metric and never instead of one",
                "a loss exactly compensated by a gain elsewhere in the same stage",
            ),
            ideal=1.0,
        ),
        MetricDefinition(
            name="mse_unit_sum",
            fn=mse_unit_sum,
            answers=(
                "how far apart are these two intensity patterns once their total "
                "power is tied together?"
            ),
            blind_to=(
                "absolute scale, by construction -- that is what the normalization is "
                "for, and it means this cannot see a power loss either",
                "the same thing every mean does: a few large errors and many small "
                "ones look alike",
            ),
        ),
        MetricDefinition(
            name="central_relative_l2_intensity",
            fn=central_relative_l2_intensity,
            answers=(
                "how far is the intensity from the reference in the region the "
                "measurement is about?"
            ),
            blind_to=(
                "EVERYTHING OUTSIDE THE WINDOW. CHE-44's concern: a metric evaluated on "
                "axis cannot see an off-axis error, and an off-axis defect that moves "
                "energy out of the window can even improve this number",
                "phase, as the full-array intensity metric is",
            ),
            support="centred_window",
        ),
    )
}


def metric(name: str) -> MetricDefinition:
    try:
        return METRICS[name]
    except KeyError:
        raise KeyError(f"no metric {name!r}; defined: {sorted(METRICS)}") from None
