"""Claims, projected from families rather than maintained beside them.

CHE-131 (M0.5.2). ``claim_ledger.CLAIMS`` was a hand-maintained table of what
this project has established. A family declares the same facts -- the component,
the oracle and its independence, the tolerance and its basis, where the gate
stands, what evidence resolves -- so keeping both by hand would be two places
for one fact and therefore one place for them to disagree.

The direction is: **the family registry is authoritative wherever it has
content, and the ledger projects it.** Claims for components whose families have
not been authored yet stay hand-written in ``claim_ledger._LEGACY_CLAIMS``, and
``tests/test_claim_ledger.py`` fails if a legacy claim and a projected claim
occupy the same ``(component, kind)`` cell. That is the anti-drift mechanism:
when M1/M2/M4 land a family, the legacy row it replaces must go, because leaving
it produces a collision rather than a silent duplicate.
"""

from __future__ import annotations

from verification.claim_ledger import (
    Claim,
    GateStatus,
    StochasticEvidence,
)
from verification.families.registry import FAMILIES
from verification.families.schema import (
    BenchmarkCategory,
    BenchmarkFamily,
    StochasticEvidenceKind,
)

__all__ = ["claims_from_families", "claims_from_family"]


_EVIDENCE_FIELD = {
    StochasticEvidenceKind.EXACTNESS_LIMIT: "exactness_limit",
    StochasticEvidenceKind.UNBIASEDNESS: "unbiasedness",
    StochasticEvidenceKind.CONVERGENCE_EXPONENT: "convergence_exponent",
    StochasticEvidenceKind.VARIANCE_CHARACTERIZATION: "variance_characterization",
}


def _stochastic(fam: BenchmarkFamily) -> StochasticEvidence | None:
    """The four evidence kinds, as the ledger's four separate facts.

    A family declares which kinds it *requires*; the ledger field records which
    are *established*. Only the required ones can be established, so an
    unrequired kind stays ``None`` -- which is what ``missing`` reports on.
    """
    policy = fam.stochastic_policy
    if not policy.is_stochastic:
        return None
    # Deliberately empty. A family declares which kinds it REQUIRES; this field
    # records which are ESTABLISHED, and filling it from the requirement would
    # turn an obligation into a citation -- the ledger would then report evidence
    # that does not exist, which is the one thing it must never do. What the
    # family owes stays on the family, and StochasticEvidence.missing reports all
    # four until something measures them.
    return StochasticEvidence()


def claims_from_family(fam: BenchmarkFamily) -> tuple[Claim, ...]:
    """One claim per component this family speaks about.

    The tolerance carried into the claim is the *gating* one, because that is
    what the ledger's ``tolerance``/``tolerance_basis`` pair has always meant. A
    family with several gating tolerances projects the first, and the rest stay
    visible on the family -- the ledger is a coverage view, not a second copy of
    the measurements.
    """
    gating = fam.gating_tolerances
    disposition = fam.gate_disposition

    if disposition is not None:
        status = disposition.status
        observed = disposition.observed
        evidence = disposition.evidence or fam.evidence
    else:
        status = GateStatus.CHARACTERIZED_NO_GATE
        observed = None
        evidence = fam.evidence

    # A threshold with nothing measured against it is a claim with no content --
    # ``tests/test_claim_ledger.py`` has enforced that since M0.3. So the ledger
    # cell carries the tolerance only once something has been measured against
    # it; until then the declared threshold stays on the family, which is the
    # source of truth for it, and appears below as a caveat so it is still
    # visible to a reader of the coverage view.
    tol = gating[0] if (gating and observed is not None) else None
    # The metric NAME is what the family measures and is known whether or not a
    # number exists yet; only the threshold waits for a measurement.
    metric_name = next(
        (
            name
            for name in (
                gating[0].metric if gating else None,
                disposition.metric if disposition is not None else None,
                # A B4 family has no gating tolerance by construction, and a cell
                # with no metric name is a cell nothing can be looked up by. Its
                # first declared metric is the headline one.
                fam.metrics[0].name if fam.metrics else None,
            )
            if name is not None
        ),
        None,
    )

    devices = sorted(str(d) for d in fam.execution_policy.devices)
    dtypes = sorted(str(d) for d in fam.execution_policy.dtypes)

    caveats: list[str] = []
    if fam.category is BenchmarkCategory.B4:
        caveats.append(
            "B4 characterization: reports measurements and carries no gating tolerance "
            "by construction. Not a validation claim."
        )
    if not fam.oracle.may_decide_correctness:
        caveats.append(
            f"oracle {fam.oracle.kind.value} / {fam.oracle.independence.value} cannot "
            "decide correctness, so nothing here gates."
        )
    if gating and observed is None:
        thresholds = ", ".join(f"{t.metric} <= {t.threshold:g}" for t in gating)
        caveats.append(
            f"gating tolerance declared and nothing measured against it yet: {thresholds}. "
            f"The threshold and its basis live on {fam.family_id}."
        )
    if disposition is not None and disposition.note.strip():
        # The disposition note is where a family records WHY its gate stands
        # where it does -- including, for B3-PSF-SINGLET, the rule that it must
        # not be closed against another Optiland PSF route. Dropping it in the
        # projection would leave the coverage view with the verdict and not the
        # constraint on how the verdict may change.
        caveats.append(disposition.note)
    for control in fam.negative_controls:
        if control.expectation.value != "must_fail":
            caveats.append(
                f"negative control {control.control_id}: {control.expectation.value} "
                f"-- {control.caveat}"
            )

    return tuple(
        Claim(
            component=component,
            kind=fam.claim_kind,
            claim=fam.question,
            oracle=fam.oracle.kind,
            oracle_independence=fam.oracle.independence,
            evidence=tuple(evidence),
            metric=metric_name,
            tolerance=tol.threshold if tol else None,
            tolerance_basis=tol.basis if tol else None,
            observed=observed,
            gate_status=status,
            device=devices[0] if len(devices) == 1 else "+".join(devices),
            dtype=dtypes[0] if len(dtypes) == 1 else "+".join(dtypes),
            stochastic=_stochastic(fam),
            caveats=tuple(caveats),
        )
        for component in fam.components
    )


def claims_from_families() -> tuple[Claim, ...]:
    """Every registered family's claims, in family-id order."""
    return tuple(claim for fam in FAMILIES.values() for claim in claims_from_family(fam))
