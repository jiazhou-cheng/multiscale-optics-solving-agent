"""Every way a component can say no, with what would fix it.

CHE-108 (M1.3). An agent will spend a meaningful fraction of its time asking
components to do things they cannot do. What happens then is a correctness
property in its own right, and it is the property that decides whether the agent
can recover. The two failure modes it must never see are a fabricated number and
an unstructured traceback.

The repository already refuses well. ``ContractCode`` has nineteen members with
long explanatory comments, ``CapabilityError`` carries requested/supported/
evidence/remedy, and the couplers' ``diagnose()`` returns a *list* of contract
errors rather than raising on the first. What was missing is a single place that
says, for each code: what triggers it, what a caller should do about it, and --
the field that decides recovery -- **which of the five negative outcomes it is**.

The five, and why they may not collapse
---------------------------------------
An agent that cannot tell "this route has no executable precision" from "this
configuration is malformed" from "this approximation does not apply here" cannot
recover from any of them:

``unsupported``
    The component cannot do this, ever. Change the component or the request.
``invalid_configuration``
    The request is malformed or internally inconsistent. Fix the request.
``out_of_validity``
    It would run, and the answer would be wrong. Change the physics, or accept
    a characterization instead of a validation.
``lossy_but_allowed``
    It runs, and something is lost. Read the measured loss and decide.
``blocked``
    A guard stopped it. Nothing is wrong with the request.

The catalogue below is keyed by code and asserted complete: a code with no entry
fails ``tests/test_b0_families.py``, and a catalogue entry for a code that no
longer exists fails too. That is what stops the catalogue from becoming a
list somebody stopped updating.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from core.boundary import ContractCode
from verification.status import VerificationStatus

__all__ = ["REFUSAL_CATALOGUE", "RefusalEntry", "refusal_for", "statuses_covered"]


@dataclass(frozen=True)
class RefusalEntry:
    """One refusal code, and everything a caller needs to act on it."""

    code: str
    #: Which of the five negative outcomes this is. The field that decides what
    #: a caller should do next.
    status: VerificationStatus
    #: What produces it, concretely enough to reproduce.
    trigger: str
    #: What would fix it. A refusal without one is a dead end dressed as a
    #: diagnostic.
    remedy: str
    #: Whether refusing is the right answer, or whether the component *could*
    #: proceed and is choosing not to. Both are legitimate; conflating them is
    #: not.
    could_have_proceeded: bool = False

    def __post_init__(self) -> None:
        if not self.trigger.strip() or not self.remedy.strip():
            raise ValueError(f"{self.code}: a refusal needs a trigger and a remedy")
        if self.status is VerificationStatus.OK:
            raise ValueError(f"{self.code}: OK is not a refusal")


def _entry(
    code: ContractCode,
    status: VerificationStatus,
    trigger: str,
    remedy: str,
    *,
    could_have_proceeded: bool = False,
) -> RefusalEntry:
    return RefusalEntry(
        code=code.value,
        status=status,
        trigger=trigger,
        remedy=remedy,
        could_have_proceeded=could_have_proceeded,
    )


REFUSAL_CATALOGUE: Mapping[str, RefusalEntry] = {
    entry.code: entry
    for entry in (
        _entry(
            ContractCode.MISSING_DECLARATION,
            VerificationStatus.INVALID_CONFIGURATION,
            "an artifact or edge omits a declaration the consumer requires",
            "declare it. The consumer will not default a convention it cannot verify",
        ),
        _entry(
            ContractCode.UNIT_NOT_SI,
            VerificationStatus.INVALID_CONFIGURATION,
            "a length, angle or wavelength arrives in a non-SI unit",
            "convert to SI at the boundary. The repository is SI internally and the "
            "adapters convert at their own edges",
        ),
        _entry(
            ContractCode.PHASOR_MISMATCH,
            VerificationStatus.INVALID_CONFIGURATION,
            "producer and consumer declare opposite phasor sign conventions",
            "state the convention the producer actually used. A mismatched sign "
            "scrambles wavefront curvature entirely -- the round trip reads 1.40 "
            "against a correct 1.32e-15",
        ),
        _entry(
            ContractCode.AXIS_ORDER_MISMATCH,
            VerificationStatus.INVALID_CONFIGURATION,
            "producer and consumer disagree about (y, x) versus (x, y)",
            "declare the producer's order. A transpose is invisible on a rotationally "
            "symmetric field and total off axis",
        ),
        _entry(
            ContractCode.FRAME_MISMATCH,
            VerificationStatus.INVALID_CONFIGURATION,
            "the coordinate frame or origin rule differs between producer and consumer",
            "align the frames. A different centring shifts every reported coordinate "
            "by up to half a pixel, which is a large fraction of an Airy radius",
        ),
        _entry(
            ContractCode.SHAPE_MISMATCH,
            VerificationStatus.INVALID_CONFIGURATION,
            "array shapes are incompatible, or a patch exceeds twice the substrate radius",
            "resize the grid, or reduce the patch width below 2R -- and well below it "
            "for the eq S9 bound to be useful",
        ),
        _entry(
            ContractCode.NON_FINITE,
            VerificationStatus.INVALID_CONFIGURATION,
            "a NaN or an infinity appears in an artifact",
            "find the upstream division or overflow. A non-finite value propagates "
            "silently through a sum and poisons everything downstream of it",
        ),
        _entry(
            ContractCode.NON_UNIT_DIRECTION,
            VerificationStatus.INVALID_CONFIGURATION,
            "a direction cosine triple does not have unit norm to the dtype's floor",
            "normalize at the source. A drifting norm is a sign that the refraction "
            "step lost a factor rather than that the tolerance is tight",
        ),
        _entry(
            ContractCode.EMPTY_ENSEMBLE,
            VerificationStatus.INVALID_CONFIGURATION,
            "a ray bundle or field ensemble has no members",
            "check the upstream trace: total vignetting produces this, and so does a "
            "field angle outside the system's stop",
        ),
        _entry(
            ContractCode.AMPLITUDE_IS_A_WEIGHT,
            VerificationStatus.INVALID_CONFIGURATION,
            "a quantity declared as a complex amplitude is a real weight",
            "declare it as a weight. Complex fields are amplitudes, not intensities, "
            "and the distinction decides whether a sum is coherent",
        ),
        _entry(
            ContractCode.OPL_REFERENCE_UNVERIFIED,
            VerificationStatus.BLOCKED,
            "a bare Optiland opd_native arrives with no declared handoff plane, so it "
            "is an absolute accumulated path whose zero moves with the aperture",
            "declare handoff_plane and handoff_plane_z_m on the edge. Stating the "
            "plane is what promotes an absolute opd into an optical path length with "
            "a reference",
            could_have_proceeded=True,
        ),
        _entry(
            ContractCode.REFERENCE_PLANE_MISMATCH,
            VerificationStatus.INVALID_CONFIGURATION,
            "the record was exported at one plane and the consumer declared another",
            "re-run the ray model with config['handoff_plane'] set to the declared "
            "plane, or declare the plane the trace was actually exported at. On the "
            "singlet these two planes are a pupil-to-focus distance apart, so "
            "accepting the mismatch would defocus the reconstruction rather than "
            "piston it",
        ),
        _entry(
            ContractCode.PAD_STATE_UNKNOWN,
            VerificationStatus.BLOCKED,
            "a field's pad state is not declared, so the consumer cannot tell the "
            "physical window from the padding",
            "declare pad_width. Guessing it would rescale every coordinate the "
            "measurement reports",
            could_have_proceeded=True,
        ),
        _entry(
            ContractCode.NEGATIVE_INTENSITY,
            VerificationStatus.INVALID_CONFIGURATION,
            "an intensity array contains a negative value",
            "check for an amplitude that was subtracted rather than a power that was "
            "summed",
        ),
        _entry(
            ContractCode.ARTIFACT_KIND_MISMATCH,
            VerificationStatus.INVALID_CONFIGURATION,
            "a consumer is handed the wrong artifact kind -- a ray bundle where a "
            "complex field is required",
            "connect the port the graph validator expects. A PSF is measured on a "
            "COMPLEX_FIELD and the measurement does not accept a ray bundle",
        ),
        _entry(
            ContractCode.SAMPLE_PITCH_MISMATCH,
            VerificationStatus.INVALID_CONFIGURATION,
            "the pitch on a record disagrees with the pitch the propagation reported",
            "pass the propagation's own output pitch. Reading an input pupil pitch "
            "instead rescales every angular comparison by a constant while leaving the "
            "intensity map entirely plausible",
        ),
        _entry(
            ContractCode.OBJECT_SPACE_REFERENCE_MISSING,
            VerificationStatus.BLOCKED,
            "an off-axis record lacks the object-space term n_object * (d0 . r_launch)",
            "supply the object-space reference. Without it only 0.13% of the required "
            "convergence tilt survives, and the reconstruction converges CLEANLY 209 um "
            "from where the rays actually go -- the defect is invisible on axis and "
            "total off it",
            could_have_proceeded=True,
        ),
        _entry(
            ContractCode.NON_HEXAPOLAR_SAMPLING,
            VerificationStatus.OUT_OF_VALIDITY,
            "a per-ray quadrature weight is requested on a bundle that is not sampled "
            "on hexapolar rings",
            "use hexapolar sampling, or drop the quadrature weight. The weight a ray "
            "carries is derived from its ring, so applying it to another sampling "
            "assigns weights that mean nothing",
            could_have_proceeded=True,
        ),
        _entry(
            ContractCode.REPRESENTATION_INCONSISTENT,
            VerificationStatus.INVALID_CONFIGURATION,
            "a declared property disagrees with the data -- declared device cuda with "
            "actual cpu placement being the case that motivated it",
            "read the property off the array rather than off the request. A "
            "process-global JAX platform pin produces a successful run on the host "
            "while the caller asked for CUDA, with no error raised",
        ),
    )
}


def refusal_for(code: str) -> RefusalEntry:
    try:
        return REFUSAL_CATALOGUE[code]
    except KeyError:
        raise KeyError(
            f"no catalogue entry for {code!r}. A refusal a caller cannot look up is a "
            "traceback with extra steps."
        ) from None


def statuses_covered() -> frozenset[VerificationStatus]:
    """Which of the five negative outcomes the catalogue can actually produce.

    Reported rather than asserted complete: ``unsupported`` and
    ``lossy_but_allowed`` come from ``CapabilityError`` and the precision bridge
    rather than from a ``ContractCode``, so a catalogue built only from contract
    codes legitimately does not cover them. Saying which are missing is more
    useful than claiming coverage this does not have.
    """
    return frozenset(entry.status for entry in REFUSAL_CATALOGUE.values())
