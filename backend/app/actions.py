"""Closed catalogue of operational actions.

The model may only propose these types. Executors run later, and only for the
ids a reviewer approved. A reject path never reaches them.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Protocol

from app.completion import Completion
from app.config import ROOT
from app.schemas import ACTION_TYPES, ActionType, Draft, Findings, InboundEmail, PolicySnippet, ProposedAction, Triage

MAX_PROPOSALS = 3
_MONEY = re.compile(r"(AED\s?\d|\d[\d,]*\s?(AED|dirham)|\b\d+\s?%)", re.IGNORECASE)

_ALLOWED_ARGS: dict[str, set[str]] = {
    "schedule_roadside_pickup": {"branch"},
    "book_safety_inspection": {"branch", "priority"},
    "register_recall_repair": {"recall_id"},
    "open_refund_review": {"invoice_ref", "reason"},
    "escalate_to_branch_manager": {"reason"},
    "offer_courtesy_car": {"branch"},
    "forward_to_data_protection": {"request_type"},
}
_REQUIRED_ARGS: dict[str, set[str]] = {
    "schedule_roadside_pickup": {"branch"},
    "book_safety_inspection": {"branch", "priority"},
    "register_recall_repair": {"recall_id"},
    "open_refund_review": {"reason"},
    "escalate_to_branch_manager": {"reason"},
    "offer_courtesy_car": {"branch"},
    "forward_to_data_protection": {"request_type"},
}
_PRIORITY = {"same_day", "next_day"}
_REQUEST = {"access", "delete", "consent"}
_RANK = {name: index for index, name in enumerate(ACTION_TYPES)}
_REF_PREFIX = {
    "schedule_roadside_pickup": "PICKUP",
    "book_safety_inspection": "INSP",
    "register_recall_repair": "RECALL",
    "open_refund_review": "REFUND",
    "escalate_to_branch_manager": "ESC",
    "offer_courtesy_car": "COURTESY",
    "forward_to_data_protection": "DPO",
}


def reference_for(case_id: str, action_id: str, action_type: str) -> str:
    digest = hashlib.sha256(f"{case_id}:{action_id}".encode()).hexdigest()
    prefix = _REF_PREFIX.get(action_type, "ACT")
    return f"{prefix}-{digest[:6].upper()}"


def _staff_conduct(email: InboundEmail) -> bool:
    text = f"{email.subject}\n{email.body}".lower()
    return any(word in text for word in ("rude", "attitude", "dismissive", "shouted"))


def _recall_ids(findings: Findings) -> set[str]:
    """Only ids that a check_recalls lookup actually returned. The summary is not evidence."""
    blob = " ".join(fact.text for fact in findings.facts if fact.source_tool == "check_recalls")
    return set(re.findall(r"\b[A-Z]{1,4}-\d+\b", blob))


def _eligible(
    action_type: ActionType,
    triage: Triage,
    email: InboundEmail,
    snippets: list[PolicySnippet],
    findings: Findings,
    args: dict[str, str],
) -> str | None:
    category = triage.category
    severity = triage.severity
    if action_type == "schedule_roadside_pickup":
        if category == "vehicle_fault" and severity == "critical":
            return None
        return "only for a critical vehicle fault"
    if action_type == "book_safety_inspection":
        if category == "vehicle_fault":
            return None
        return "only for a vehicle fault"
    if action_type == "register_recall_repair":
        if args.get("recall_id") in _recall_ids(findings):
            return None
        return "no matching open recall in the findings"
    if action_type == "open_refund_review":
        if category == "billing":
            return None
        return "only for a billing complaint"
    if action_type == "escalate_to_branch_manager":
        if severity in {"high", "critical"} or _staff_conduct(email):
            return None
        return "severity is not high or critical and this is not a staff-conduct complaint"
    if action_type == "offer_courtesy_car":
        held = category == "vehicle_fault" and severity in {"critical", "high"}
        if held and any(snippet.id == "SAF-2" for snippet in snippets):
            return None
        return "the car is not being held for a safety inspection under SAF-2"
    if action_type == "forward_to_data_protection":
        if category == "data_privacy":
            return None
        return "only for a data-privacy complaint"
    unexpected: str = action_type
    return f"unknown type {unexpected}"


def _check_args(action_type: str, raw: dict[str, Any]) -> tuple[dict[str, str] | None, str | None]:
    allowed = _ALLOWED_ARGS[action_type]
    unknown = sorted(set(raw) - allowed)
    if unknown:
        return None, f"unexpected argument {', '.join(unknown)}"
    args = {key: str(value).strip() for key, value in raw.items() if value is not None and str(value).strip()}
    missing = sorted(name for name in _REQUIRED_ARGS[action_type] if not args.get(name))
    if missing:
        return None, f"missing {', '.join(missing)}"
    if action_type == "book_safety_inspection" and args.get("priority") not in _PRIORITY:
        return None, "priority must be same_day or next_day"
    if action_type == "forward_to_data_protection" and args.get("request_type") not in _REQUEST:
        return None, "request_type must be access, delete, or consent"
    if any(_MONEY.search(value) for value in args.values()):
        return None, "argument contains a money amount"
    branch = args.get("branch")
    if branch is not None and branch not in known_branches():
        return None, f"unknown branch {branch}"
    return args, None


def known_branches(directory: Path | None = None) -> frozenset[str]:
    """Branch names that appear on a customer profile or a service visit."""
    root = directory or (ROOT / "data" / "enterprise")
    customers = json.loads((root / "customers.json").read_text(encoding="utf-8"))
    history = json.loads((root / "service_history.json").read_text(encoding="utf-8"))
    names: set[str] = set()
    for row in customers.values():
        preferred = str(row.get("preferred_branch") or "").strip()
        if preferred:
            names.add(preferred)
    for visits in history.values():
        for item in visits:
            visited = str(item.get("branch") or "").strip()
            if visited:
                names.add(visited)
    return frozenset(names)


def _evidence_parts(
    evidence: list[str], snippets: list[PolicySnippet], findings: Findings
) -> tuple[list[str], list[str]]:
    """Keep a policy id or a fact, item by item. The summary is not evidence."""
    ids = {snippet.id for snippet in snippets}
    facts = [fact.text for fact in findings.facts]
    kept: list[str] = []
    removed: list[str] = []
    for item in evidence:
        text = item.strip()
        if not text:
            continue
        is_policy = text in ids
        is_fact = any(text in fact or fact in text for fact in facts)
        if is_policy or is_fact:
            kept.append(text)
        else:
            removed.append(text)
    return kept, removed


def validate_proposals(
    raw: list[ProposedAction],
    triage: Triage,
    email: InboundEmail,
    snippets: list[PolicySnippet],
    findings: Findings,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Return kept proposals and drop reasons. Kept items are capped at 3."""
    kept: list[dict[str, Any]] = []
    drops: list[str] = []
    ranked = sorted(raw, key=lambda item: _RANK.get(item.type, 99))
    for proposal in ranked:
        action_type = proposal.type.strip()
        if action_type not in ACTION_TYPES:
            drops.append(f"dropped {action_type or 'blank'}: not in the catalogue")
            continue
        if _MONEY.search(proposal.rationale or ""):
            drops.append(f"dropped {action_type}: rationale contains a money amount")
            continue
        args, arg_error = _check_args(action_type, dict(proposal.args))
        if arg_error or args is None:
            drops.append(f"dropped {action_type}: {arg_error}")
            continue
        reason = _eligible(action_type, triage, email, snippets, findings, args)  # type: ignore[arg-type]
        if reason:
            drops.append(f"dropped {action_type}: {reason}")
            continue
        kept_evidence, removed_evidence = _evidence_parts(proposal.evidence, snippets, findings)
        for text in removed_evidence:
            drops.append(f"evidence removed: {text}")
        if not kept_evidence:
            drops.append(f"dropped {action_type}: no evidence from a retrieved policy or a finding")
            continue
        if len(kept) >= MAX_PROPOSALS:
            drops.append(f"dropped {action_type}: more than {MAX_PROPOSALS} proposals")
            continue
        kept.append(
            {
                "id": f"a{len(kept) + 1}",
                "type": action_type,
                "args": args,
                "rationale": proposal.rationale.strip(),
                "evidence": kept_evidence,
            }
        )
    return kept, drops


def offline_proposals(
    triage: Triage, email: InboundEmail, snippets: list[PolicySnippet], findings: Findings
) -> list[ProposedAction]:
    """Deterministic candidates. The validator still drops anything ineligible."""
    branch = "Al Quoz"
    for fact in findings.facts:
        if "Preferred branch" in fact.text:
            branch = fact.text.split("Preferred branch", 1)[1].strip().strip(".")
            break
    evidence = [snippet.id for snippet in snippets[:2]] or [fact.text for fact in findings.facts[:1]]
    if not evidence:
        evidence = ["CMP-1"] if any(snippet.id == "CMP-1" for snippet in snippets) else []
    proposals: list[ProposedAction] = []
    if triage.category == "vehicle_fault" and triage.severity == "critical":
        proposals.append(
            ProposedAction(
                type="schedule_roadside_pickup",
                args={"branch": branch},
                rationale="The complaint describes a possible safety risk, so the vehicle should be collected.",
                evidence=evidence,
            )
        )
    if triage.category == "vehicle_fault":
        priority = "same_day" if triage.severity == "critical" else "next_day"
        proposals.append(
            ProposedAction(
                type="book_safety_inspection",
                args={"branch": branch, "priority": priority},
                rationale="A vehicle fault should be inspected before the customer keeps driving.",
                evidence=evidence,
            )
        )
    for recall_id in sorted(_recall_ids(findings)):
        proposals.append(
            ProposedAction(
                type="register_recall_repair",
                args={"recall_id": recall_id},
                rationale=f"The findings include open recall {recall_id}.",
                evidence=[fact.text for fact in findings.facts if recall_id in fact.text] or evidence,
            )
        )
    if triage.category == "billing":
        proposals.append(
            ProposedAction(
                type="open_refund_review",
                args={"reason": "The customer disputes a charge and a person needs to review it."},
                rationale="Billing complaints are reviewed by a person. No amount is decided here.",
                evidence=evidence,
            )
        )
    if triage.category == "data_privacy":
        request = (
            "delete" if "delete" in email.body.lower() else "consent" if "consent" in email.body.lower() else "access"
        )
        proposals.append(
            ProposedAction(
                type="forward_to_data_protection",
                args={"request_type": request},
                rationale="This is a personal-data request and should go to the data protection contact.",
                evidence=evidence,
            )
        )
    if triage.severity in {"high", "critical"} or _staff_conduct(email):
        proposals.append(
            ProposedAction(
                type="escalate_to_branch_manager",
                args={"reason": "The severity or the staff-conduct complaint needs a branch manager."},
                rationale="A manager should see this before it is closed.",
                evidence=evidence,
            )
        )
    if (
        triage.category == "vehicle_fault"
        and triage.severity in {"critical", "high"}
        and any(s.id == "SAF-2" for s in snippets)
    ):
        proposals.append(
            ProposedAction(
                type="offer_courtesy_car",
                args={"branch": branch},
                rationale="The vehicle will be held for inspection, and the courtesy-car policy was retrieved.",
                evidence=["SAF-2"] if any(s.id == "SAF-2" for s in snippets) else evidence,
            )
        )
    return proposals


class Proposer(Protocol):
    def propose(
        self,
        email: InboundEmail,
        triage: Triage,
        snippets: list[PolicySnippet],
        findings: Findings,
        draft: Draft,
    ) -> list[ProposedAction] | Completion[list[ProposedAction]]: ...


class OfflineProposer:
    """Deterministic proposals. Used offline, and by tests that script only triage and draft."""

    def propose(
        self,
        email: InboundEmail,
        triage: Triage,
        snippets: list[PolicySnippet],
        findings: Findings,
        draft: Draft,
    ) -> list[ProposedAction]:
        del draft
        return offline_proposals(triage, email, snippets, findings)


_CUSTOMER_ACTIONS = (
    "schedule_roadside_pickup",
    "book_safety_inspection",
    "register_recall_repair",
    "open_refund_review",
    "offer_courtesy_car",
)


def _priority_phrase(priority: str, language: str) -> str:
    if language == "ar":
        return "في اليوم نفسه" if priority == "same_day" else "في اليوم التالي"
    return "same day" if priority == "same_day" else "next day"


def customer_sentence(action_type: str, args: dict[str, str], ref: str, language: str) -> str | None:
    """One customer-facing sentence, or None for an internal action."""
    branch = args.get("branch") or ""
    if action_type not in _CUSTOMER_ACTIONS:
        return None
    if language == "ar":
        if action_type == "schedule_roadside_pickup":
            return f"استلام السيارة من موقعكم عبر المساعدة على الطريق، المرجع {ref}."
        if action_type == "book_safety_inspection":
            when = _priority_phrase(args.get("priority") or "", language)
            place = f"في فرع {branch}" if branch else "في الفرع"
            return f"فحص سلامة {place}، {when}، المرجع {ref}."
        if action_type == "register_recall_repair":
            return f"إصلاح حملة الاستدعاء {args.get('recall_id', '')}، المرجع {ref}."
        if action_type == "open_refund_review":
            return f"مراجعة للمبلغ المدفوع، المرجع {ref}."
        if action_type == "offer_courtesy_car":
            return f"سيارة بديلة من فرع {branch}، المرجع {ref}." if branch else f"سيارة بديلة، المرجع {ref}."
        return None
    if action_type == "schedule_roadside_pickup":
        return f"Roadside pickup from your location, reference {ref}."
    if action_type == "book_safety_inspection":
        when = _priority_phrase(args.get("priority") or "", language)
        place = f"at our {branch} branch" if branch else "at our branch"
        return f"Safety inspection {place}, {when}, reference {ref}."
    if action_type == "register_recall_repair":
        return f"Recall repair {args.get('recall_id', '')}, reference {ref}."
    if action_type == "open_refund_review":
        return f"A review of the charge, reference {ref}."
    if action_type == "offer_courtesy_car":
        return (
            f"A courtesy car from our {branch} branch, reference {ref}."
            if branch
            else f"A courtesy car, reference {ref}."
        )
    return None


_SIGN_OFFS = ("Meridian Motors Customer Care", "خدمة عملاء ميريديان موتورز")
_EN_CLOSINGS = ("kind regards", "best regards", "yours sincerely")
_AR_CLOSINGS = ("مع أطيب التحيات", "مع التحية")


def _is_closing_line(line: str) -> bool:
    stripped = line.strip()
    lowered = stripped.lower()
    return lowered.startswith(_EN_CLOSINGS) or stripped.startswith(_AR_CLOSINGS)


def place_arranged(body: str, section: str) -> str:
    """Put the arranged section before the sign-off. Append it when there is none."""
    if not section:
        return body
    lines = body.splitlines()
    sign: int | None = None
    for index in range(len(lines) - 1, -1, -1):
        if any(mark in lines[index] for mark in _SIGN_OFFS):
            sign = index
            break
    insert_at = sign
    if sign is not None:
        cursor = sign - 1
        while cursor >= 0 and not lines[cursor].strip():
            cursor -= 1
        if cursor >= 0 and _is_closing_line(lines[cursor]):
            insert_at = cursor
    if insert_at is None:
        return body.rstrip() + "\n\n" + section
    head = "\n".join(lines[:insert_at]).rstrip()
    tail = "\n".join(lines[insert_at:])
    if head:
        return f"{head}\n\n{section}\n\n{tail}"
    return f"{section}\n\n{tail}"


def compose_final_body(
    body: str,
    proposals: list[dict[str, Any]],
    approved_ids: list[str],
    language: str,
) -> tuple[str, str]:
    """The reply a reviewer would send for these ids, and the arranged paragraph. Does not record anything."""
    approved = set(approved_ids)
    results: list[dict[str, Any]] = []
    for proposal in proposals:
        if proposal.get("id") not in approved:
            continue
        ref = str(proposal.get("ref") or "")
        results.append(
            {
                "action_id": proposal["id"],
                "type": proposal["type"],
                "status": "executed",
                "ref": ref,
                "args": proposal.get("args") or {},
            }
        )
    section = arranged_section(results, language)
    return place_arranged(body, section), section


def arranged_section(results: list[dict[str, Any]], language: str) -> str:
    """Text appended after approval. Built only from actions that actually ran."""
    sentences = [
        line
        for item in results
        if item.get("status") == "executed" and item.get("ref")
        for line in [customer_sentence(str(item.get("type")), dict(item.get("args") or {}), str(item["ref"]), language)]
        if line
    ]
    if not sentences:
        return ""
    heading = "ما رتبناه:" if language == "ar" else "What we have arranged:"
    return heading + " " + " ".join(sentences)
