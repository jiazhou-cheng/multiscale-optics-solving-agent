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
    filled = {
        _EVIDENCE_FIELD[kind]: f"required by {fam.family_id}" for kind in policy.required_evidence
    }
    return StochasticEvidence(**filled)


def claims_from_family(fam: BenchmarkFamily) -> tuple[Claim, ...]:
    """One claim per component this family speaks about.

    The tolerance carried into the claim is the *gating* one, because that is
    what the ledger's ``tolerance``/``tolerance_basis`` pair has always meant. A
    family with several gating tolerances projects the first, and the rest stay
    visible on the family -- the ledger is a coverage view, not a second copy of
    the measurements.
    """
    gating = fam.gating_tolerances
    tol = gating[0] if gating else None
    disposition = fam.gate_disposition

    if disposition is not None:
        status = disposition.status
        observed = disposition.observed
        evidence = disposition.evidence or fam.evidence
    else:
        status = GateStatus.CHARACTERIZED_NO_GATE
        observed = None
        evidence = fam.evidence

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
            metric=tol.metric if tol else (disposition.metric if disposition else None),
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
