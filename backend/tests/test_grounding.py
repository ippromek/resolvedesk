"""grounding checks."""

from __future__ import annotations

import pytest

from app.actions import (
    validate_proposals,
)
from app.config import ROOT
from app.guards import check_grounding
from app.kb import get_kb
from app.schemas import Draft, Findings, InboundEmail, PolicySnippet, ProposedAction, Triage

_WARRANTY = PolicySnippet(
    id="WAR-1",
    title="Standard warranty",
    text="We review warranty claims within 10 working days.",
    score=1.0,
)
_EMAIL = "The dealer charged me AED 8,400 for the battery."
_POLICY = PolicySnippet(
    id="SER-1", title="Service visits", text="We aim to return the vehicle on the promised day.", score=1.0
)
_BILLING = PolicySnippet(id="BIL-1", title="Invoice disputes", text="We reply within 5 working days.", score=1.0)


def _draft(body: str) -> Draft:
    return Draft(body=body, cited_policy_ids=["SER-1"])


def test_grounding_catches_invented_commitment() -> None:
    snippets = [PolicySnippet(id="BIL-1", title="Invoice disputes", text="We reply within 5 working days.", score=1)]
    bad = Draft(body="We will refund AED 2,000 today.", cited_policy_ids=["BIL-1", "XYZ-9"])
    g = check_grounding(bad, snippets)
    assert not g.ok
    assert any("XYZ-9" in i for i in g.issues)
    assert any("AED" in i for i in g.issues)


def test_quoting_a_customer_amount_without_a_promise_is_grounded() -> None:
    draft = Draft(
        body="You mentioned a repair of AED 8,400. We cannot confirm that figure from the policy.",
        cited_policy_ids=["WAR-1"],
    )
    result = check_grounding(draft, [_WARRANTY], _EMAIL)
    assert result.ok


def test_promising_the_customer_amount_is_an_issue() -> None:
    draft = Draft(body="We will refund AED 8,400.", cited_policy_ids=["WAR-1"])
    result = check_grounding(draft, [_WARRANTY], _EMAIL)
    assert not result.ok
    assert any("AED 8,400" in issue for issue in result.issues)


def test_amount_missing_from_email_and_policy_is_an_issue() -> None:
    draft = Draft(body="The inspection fee is AED 500.", cited_policy_ids=["WAR-1"])
    result = check_grounding(draft, [_WARRANTY], "Is my battery covered?")
    assert not result.ok
    assert any("AED 500" in issue for issue in result.issues)


@pytest.mark.parametrize(
    "body",
    [
        "Please accept this voucher for your next visit.",
        "We have applied a discount to the invoice.",
        "The next service is 20% off.",
        "This is a goodwill payment toward the repair.",
        "Collection will be complimentary.",
        "We have booked a free service.",
    ],
)
def test_commitment_vocabulary_must_be_in_the_cited_policy(body: str) -> None:
    draft = Draft(body=body, cited_policy_ids=["WAR-1"])
    result = check_grounding(draft, [_WARRANTY])
    assert not result.ok
    assert any("commitment" in issue for issue in result.issues)


def test_thin_retrieval_includes_complaint_handling_policy() -> None:
    kb = get_kb(ROOT / "kb")
    query = (
        "Small suggestion about the waiting area Just feedback: the coffee machine "
        "in the waiting area was broken during my last two visits. Minor thing, staff were lovely otherwise."
    )
    hits = kb.search(query, category="service_experience", k=3)
    assert "CMP-1" in [hit.id for hit in hits]
    assert sum(1 for hit in hits if hit.id != "CMP-1" and hit.score > 9) < 2


def test_two_strong_hits_do_not_add_the_fallback() -> None:
    kb = get_kb(ROOT / "kb")
    hits = kb.search(
        "Warranty claim refused - gearbox Your workshop says my gearbox repair is not covered "
        "under warranty because I serviced the car at an independent garage. The car is 3 years "
        "old with 70,000 km. I have all invoices. This repair is AED 8,400.",
        category="warranty",
        k=3,
    )
    assert len(hits) == 3
    assert "CMP-1" not in [hit.id for hit in hits]
    assert sum(1 for hit in hits if hit.score > 9) >= 2


def test_a_refusal_that_names_a_voucher_is_grounded() -> None:
    draft = Draft(
        body="Please note that we are unable to offer a discount voucher, as our policies do not provide for one.",
        cited_policy_ids=["SER-1"],
    )
    result = check_grounding(draft, [_POLICY])
    assert result.ok


def test_an_offered_discount_voucher_is_an_issue() -> None:
    draft = Draft(body="We are pleased to offer you a 50% discount voucher.", cited_policy_ids=["SER-1"])
    result = check_grounding(draft, [_POLICY])
    assert not result.ok
    assert any("discount" in issue or "voucher" in issue or "50%" in issue for issue in result.issues)


def test_a_refund_promise_is_an_issue() -> None:
    draft = Draft(body="We will refund AED 500.", cited_policy_ids=["SER-1"])
    result = check_grounding(draft, [_POLICY])
    assert not result.ok
    assert any("AED 500" in issue or "refund" in issue for issue in result.issues)


def test_mixed_evidence_keeps_only_the_policy_id() -> None:
    triage = Triage(
        category="billing",
        severity="medium",
        language="en",
        summary="The investigator thinks the branch should recheck the work.",
        rationale="A charge is disputed.",
    )
    email = InboundEmail(sender="p@example.com", subject="Invoice", body="The invoice looks wrong.")
    findings = Findings(
        summary="The branch should be asked to recheck the work from the service visit.", facts=[], nothing_found=True
    )
    raw = [
        ProposedAction(
            type="open_refund_review",
            args={"reason": "Review the charge."},
            rationale="A person should review the invoice.",
            evidence=["BIL-1", "The branch should be asked to recheck the work from the service visit."],
        )
    ]
    kept, drops = validate_proposals(raw, triage, email, [_BILLING], findings)
    assert len(kept) == 1
    assert kept[0]["evidence"] == ["BIL-1"]
    assert any(item.startswith("evidence removed:") for item in drops)


def test_a_negation_does_not_excuse_a_later_clause() -> None:
    refund = _draft("We are unable to repair it today, but we will refund AED 2,000 next week.")
    voucher = _draft("We cannot wait to see you again, so here is a 50% discount voucher.")
    refunded = check_grounding(refund, [_POLICY])
    offered = check_grounding(voucher, [_POLICY])
    assert not refunded.ok
    assert any("AED 2,000" in issue or "refund" in issue for issue in refunded.issues)
    assert not offered.ok
    assert any("50%" in issue or "discount" in issue or "voucher" in issue for issue in offered.issues)


def test_a_refusal_in_the_same_clause_stays_grounded() -> None:
    draft = _draft("I'm not able to offer a discount voucher, as our service standards do not provide for one.")
    assert check_grounding(draft, [_POLICY]).ok


def test_an_unapproved_action_claim_is_an_issue() -> None:
    draft = _draft("We are alerting the branch so this is treated as a repeat brake concern.")
    result = check_grounding(draft, [_POLICY])
    assert not result.ok
    assert any("we are alerting" in issue for issue in result.issues)
    policy = _draft("Roadside collection is available under our safety policy.")
    assert check_grounding(policy, [_POLICY]).ok
