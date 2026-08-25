"""The seven outcomes a verification can have, kept apart on purpose.

CHE-132 (M0.5.3) needs these and CHE-131 (M0.5.2) needs them too -- a family
declares which of them its failure paths are expected to produce -- so they live
in their own module rather than in either consumer.

The reason there are seven and not two is that this repository has already been
bitten by the collapse. A benchmark that reports ``pass: false`` cannot
distinguish "the solver refuses this dtype" from "the approximation does not
apply here" from "it ran and the number is wrong", and those three have nothing
in common: the first is a capability fact, the second is a physics fact about
the instance, and only the third is evidence about the implementation.
"""

from __future__ import annotations

from enum import StrEnum

__all__ = ["VerificationStatus"]


class VerificationStatus(StrEnum):
    """What kind of outcome a verification had.

    Not a verdict. ``OK`` means the run produced evidence the verifier could
    measure -- it does **not** mean the physics was right; that is what the
    per-metric ``met`` flags say.
    """

    #: The run executed inside its declared validity and produced measurable
    #: evidence. Says nothing about whether the metrics met their tolerances.
    OK = "ok"

    #: The component cannot execute what was asked at all -- no device, no
    #: dtype, no such capability. A capability fact, not a physics result.
    UNSUPPORTED = "unsupported"

    #: The parameters are internally inconsistent or violate a declared
    #: precondition. The instance is malformed; nothing was measured.
    INVALID_CONFIGURATION = "invalid_configuration"

    #: The instance is outside the family's declared validity domain. The code
    #: would run; the approximation does not apply. Distinct from
    #: ``INVALID_CONFIGURATION`` because the configuration is well formed and
    #: distinct from ``UNSUPPORTED`` because the capability exists.
    OUT_OF_VALIDITY = "out_of_validity"

    #: Executed, but a declared precision or representation loss was taken on
    #: the way. Chromatix's unconditional ``complex64`` cast is the live case:
    #: the run succeeds and the loss is real, so it is neither ``OK`` nor a
    #: failure, and the loss must be reported as a number.
    LOSSY_BUT_ALLOWED = "lossy_but_allowed"

    #: A guard stopped the run before it executed -- a resource envelope, an
    #: unverified-gradient requirement, a policy refusal. Nothing was measured
    #: and nothing is wrong with the instance.
    BLOCKED = "blocked"

    #: It ran, and the convergence study says the number is not yet converged
    #: in its declared refinement dimension. Reporting the value as an accuracy
    #: result would be claiming a number the ladder does not support.
    UNCONVERGED = "unconverged"
