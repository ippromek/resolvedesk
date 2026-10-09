"""The complaint pipeline as a LangGraph state machine.

    guard -> triage -> (spam? -> close_no_reply)
          -> retrieve -> investigate -> draft -> ground_check
          -> (issues and revisions < 2 -> revise -> ground_check)
          -> propose_actions -> human_review [interrupt]
          -> (approve/edit -> execute_actions -> send | reject -> reject)

The human review step is a real `interrupt()`: the graph stops, its state is
checkpointed in SQLite, and only `Command(resume=...)` with a reviewer's decision
can move it on. There is no code path from draft to send that skips it.
Action executors run only in `execute_actions`, and only for ids the reviewer approved.
"""

from __future__ import annotations

import operator
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Annotated, Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Checkpointer, interrupt

from app.actions import Proposer, compose_final_body, reference_for, validate_proposals
from app.brain import Brain
from app.completion import Completion
from app.db import Store
from app.failures import log_failure
from app.guards import check_grounding, screen_input
from app.investigation import InvestigationFailed, Investigator, apply_tool_facts, trace_detail
from app.kb import KnowledgeBase
from app.schemas import Draft, Findings, InboundEmail, PolicySnippet, ReviewInput, Triage

MAX_REVISIONS = 2


def _parts(result: object) -> tuple[object, dict[str, int] | None]:
    if isinstance(result, Completion):
        return result.value, result.usage.as_mapping()
    return result, None


def validate_review(state: Mapping[str, Any], review: ReviewInput) -> list[str]:
    """Checks shared by the review route, the human_review node, and the snapshot builder."""
    errors: list[str] = []
    guard = state.get("guard") or {}
    if (
        review.decision in {"approve", "edit"}
        and guard.get("injection_suspected")
        and not review.injection_acknowledged
    ):
        errors.append("injection warning must be acknowledged")
    if review.decision == "edit" and not (review.body or "").strip():
        errors.append("an edit needs the edited reply body")
    if review.decision != "reject" and review.approved_action_ids:
        proposals = list(state.get("proposals") or [])
        known = {str(item["id"]) for item in proposals}
        unknown = [action_id for action_id in review.approved_action_ids if action_id not in known]
        if unknown:
            errors.append("unknown action id: " + ", ".join(unknown))
    return errors


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def event(
    step: str,
    detail: str,
    *,
    actor: str = "agent",
    usage: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {"at": _now(), "step": step, "actor": actor, "detail": detail}
    for key in ("latency_ms", "input_tokens", "output_tokens"):
        if usage and usage.get(key) is not None:
            row[key] = usage[key]
    return row


class CaseState(TypedDict, total=False):
    case_id: str
    email: dict[str, Any]
    guard: dict[str, Any]
    triage: dict[str, Any]
    snippets: list[dict[str, Any]]
    findings: dict[str, Any]
    tool_calls: list[dict[str, Any]]
    draft: dict[str, Any]
    draft_history: list[dict[str, Any]]
    revision_count: int
    grounding: dict[str, Any]
    proposals: list[dict[str, Any]]
    action_results: list[dict[str, Any]]
    review: dict[str, Any]
    final_body: str | None
    status: str
    model: str
    events: Annotated[list[dict[str, Any]], operator.add]


def build_graph(
    brain: Brain,
    kb: KnowledgeBase,
    investigator: Investigator,
    proposer: Proposer,
    checkpointer: Checkpointer = None,
    *,
    store: Store,
):

    def guard(state: CaseState) -> CaseState:
        result = screen_input(InboundEmail(**state["email"]))
        detail = (
            "suspected prompt injection: " + "; ".join(result.reasons)
            if result.injection_suspected
            else "no injection patterns found"
        )
        return {
            "guard": result.model_dump(),
            "status": "processing",
            "model": brain.name,
            "events": [event("guard", detail)],
        }

    def triage(state: CaseState) -> CaseState:
        value, usage = _parts(brain.triage(InboundEmail(**state["email"])))
        if not isinstance(value, Triage):
            raise TypeError("triage did not return a Triage")
        t = value
        return {
            "triage": t.model_dump(),
            "events": [
                event(
                    "triage",
                    f"{t.category} / {t.severity} ({t.language}) - {t.rationale}",
                    usage=usage,
                )
            ],
        }

    def route_after_triage(state: CaseState) -> Literal["retrieve", "close_no_reply"]:
        return "close_no_reply" if state["triage"]["category"] == "spam" else "retrieve"

    def close_no_reply(state: CaseState) -> CaseState:
        return {"status": "closed_no_reply", "events": [event("close", "classified as spam; no reply drafted")]}

    def retrieve(state: CaseState) -> CaseState:
        email = InboundEmail(**state["email"])
        t = Triage(**state["triage"])
        hits = kb.search(f"{email.subject} {email.body} {t.summary}", category=t.category, k=3)
        return {
            "snippets": [h.model_dump() for h in hits],
            "events": [event("retrieve", "found " + (", ".join(h.id for h in hits) or "nothing"))],
        }

    def investigate(state: CaseState) -> CaseState:
        email = InboundEmail(**state["email"])
        t = Triage(**state["triage"])
        failed_kind: str | None = None
        try:
            outcome = investigator.investigate(email, t, state.get("case_id"))
        except InvestigationFailed as exc:
            findings, traces, failed_kind = exc.findings, exc.traces, exc.kind
            meta = None
        except Exception as exc:  # noqa: BLE001  # a broken lookup must not stop the case reaching review
            log_failure(str(state.get("case_id") or "-"), "investigate", exc)
            findings = Findings(summary="Investigation failed before any lookup.", facts=[], nothing_found=True)
            traces = []
            failed_kind = type(exc).__name__
            meta = None
        else:
            if isinstance(outcome, Completion):
                findings, traces = outcome.value
                meta = outcome.usage.as_mapping()
            else:
                findings, traces = outcome
                meta = None
        findings, removed = apply_tool_facts(findings, traces)
        events = [event("investigate", trace_detail(trace), usage=trace.as_event_extra()) for trace in traces]
        events.extend(event("investigate", note) for note in removed)
        if failed_kind:
            events.append(event("investigate", f"investigation failed ({failed_kind})", usage=meta or None))
        else:
            events.append(event("investigate", f"investigation finished: {len(traces)} lookups", usage=meta or None))
        calls = [
            {
                "tool": trace.tool,
                "args": trace.args,
                "result": trace.result,
                **trace.as_event_extra(),
            }
            for trace in traces
        ]
        return {"findings": findings.model_dump(), "tool_calls": calls, "events": events}

    def draft(state: CaseState) -> CaseState:
        snippets = [PolicySnippet(**s) for s in state.get("snippets", [])]
        findings = Findings(**state.get("findings", {"summary": "", "facts": [], "nothing_found": True}))
        value, usage = _parts(
            brain.draft(InboundEmail(**state["email"]), Triage(**state["triage"]), snippets, findings)
        )
        if not isinstance(value, Draft):
            raise TypeError("draft did not return a Draft")
        d = value
        return {
            "draft": d.model_dump(),
            "revision_count": state.get("revision_count", 0),
            "events": [event("draft", f"drafted reply citing {d.cited_policy_ids or 'nothing'}", usage=usage)],
        }

    def ground_check(state: CaseState) -> CaseState:
        snippets = [PolicySnippet(**s) for s in state.get("snippets", [])]
        d = Draft(**state["draft"])
        g = check_grounding(d, snippets, state["email"].get("body", ""))
        history = list(state.get("draft_history") or [])
        history.append(
            {
                "version": len(history) + 1,
                "body": d.body,
                "cited_policy_ids": list(d.cited_policy_ids),
                "issues": list(g.issues),
            }
        )
        detail = "grounded" if g.ok else "; ".join(g.issues)
        return {
            "grounding": g.model_dump(),
            "draft_history": history,
            "events": [event("ground_check", detail)],
        }

    def route_after_ground(state: CaseState) -> Literal["revise", "propose_actions"]:
        issues = (state.get("grounding") or {}).get("issues") or []
        if issues and state.get("revision_count", 0) < MAX_REVISIONS:
            return "revise"
        return "propose_actions"

    def revise(state: CaseState) -> CaseState:
        snippets = [PolicySnippet(**s) for s in state.get("snippets", [])]
        findings = Findings(**state.get("findings", {"summary": "", "facts": [], "nothing_found": True}))
        issues = list((state.get("grounding") or {}).get("issues") or [])
        attempt = state.get("revision_count", 0) + 1
        value, usage = _parts(
            brain.revise(
                InboundEmail(**state["email"]),
                Triage(**state["triage"]),
                snippets,
                findings,
                Draft(**state["draft"]),
                issues,
            )
        )
        if not isinstance(value, Draft):
            raise TypeError("revise did not return a Draft")
        revised = value
        return {
            "draft": revised.model_dump(),
            "revision_count": attempt,
            "events": [
                event(
                    "revise",
                    f"revised draft (attempt {attempt}): fixed {'; '.join(issues)}",
                    usage=usage,
                )
            ],
        }

    def propose_actions(state: CaseState) -> CaseState:
        snippets = [PolicySnippet(**s) for s in state.get("snippets", [])]
        findings = Findings(**state.get("findings", {"summary": "", "facts": [], "nothing_found": True}))
        email = InboundEmail(**state["email"])
        triage = Triage(**state["triage"])
        value, meta = _parts(proposer.propose(email, triage, snippets, findings, Draft(**state["draft"])))
        if not isinstance(value, list):
            raise TypeError("propose did not return a list")
        raw = value
        kept, drops = validate_proposals(raw, triage, email, snippets, findings)
        for item in kept:
            item["ref"] = reference_for(state["case_id"], item["id"], item["type"])
        events = [event("propose_actions", drop) for drop in drops]
        if kept:
            names = ", ".join(item["type"] for item in kept)
            events.append(event("propose_actions", f"proposed {names}", usage=meta or None))
        else:
            events.append(event("propose_actions", "proposed nothing", usage=meta or None))
        return {"proposals": kept, "status": "awaiting_review", "events": events}

    def human_review(state: CaseState) -> CaseState:
        payload = {
            "case_id": state["case_id"],
            "draft": state["draft"],
            "triage": state["triage"],
            "grounding": state["grounding"],
            "guard": state["guard"],
            "proposals": state.get("proposals") or [],
            "findings": state.get("findings"),
        }
        # A raise here would replay the same resume forever. Ask again instead.
        review = ReviewInput(**interrupt(payload))
        while errors := validate_review(state, review):
            review = ReviewInput(**interrupt({**payload, "errors": errors}))
        final = None
        if review.decision == "approve":
            final = state["draft"]["body"]
        elif review.decision == "edit":
            final = review.body
        note = f" - {review.note}" if review.note else ""
        acknowledged = " — injection warning acknowledged" if review.injection_acknowledged else ""
        return {
            "review": review.model_dump(),
            "final_body": final,
            "events": [event("human_review", f"{review.decision}{acknowledged}{note}", actor=review.reviewer)],
        }

    def route_after_review(state: CaseState) -> Literal["execute_actions", "reject"]:
        return "reject" if state["review"]["decision"] == "reject" else "execute_actions"

    def execute_actions(state: CaseState) -> CaseState:
        review = state["review"]
        reviewer = review["reviewer"]
        approved = set(review.get("approved_action_ids") or [])
        events: list[dict[str, Any]] = []
        results: list[dict[str, Any]] = []
        for proposal in state.get("proposals") or []:
            if proposal["id"] not in approved:
                events.append(event("execute_actions", f"{proposal['type']} not approved", actor=reviewer))
                results.append(
                    {
                        "action_id": proposal["id"],
                        "type": proposal["type"],
                        "status": "not_approved",
                        "ref": None,
                        "args": proposal["args"],
                    }
                )
                continue
            ref = str(proposal.get("ref") or "")
            store.record_action(
                state["case_id"],
                proposal["id"],
                proposal["type"],
                proposal["args"],
                reviewer,
                _now(),
                "executed",
            )
            events.append(
                event("execute_actions", f"{proposal['type']} executed ({ref}), approved by {reviewer}", actor=reviewer)
            )
            results.append(
                {
                    "action_id": proposal["id"],
                    "type": proposal["type"],
                    "status": "executed",
                    "ref": ref,
                    "args": proposal["args"],
                }
            )
        update: CaseState = {"action_results": results, "events": events}
        language = str((state.get("triage") or {}).get("language") or "en")
        final = state.get("final_body")
        if final:
            update["final_body"], _arranged = compose_final_body(
                final,
                list(state.get("proposals") or []),
                list(review.get("approved_action_ids") or []),
                language,
            )
        return update

    def send(state: CaseState) -> CaseState:
        reviewer = str((state.get("review") or {}).get("reviewer") or "")
        store.queue_outbound(
            state["case_id"],
            state["email"]["sender"],
            str(state.get("final_body") or ""),
            reviewer,
            _now(),
        )
        return {"status": "sent", "events": [event("send", f"reply queued to {state['email']['sender']}")]}

    def reject(state: CaseState) -> CaseState:
        return {"status": "rejected", "events": [event("close", "draft rejected; case handed back to a person")]}

    g = StateGraph(CaseState)
    g.add_node("guard", guard)
    g.add_node("triage", triage)
    g.add_node("close_no_reply", close_no_reply)
    g.add_node("retrieve", retrieve)
    g.add_node("investigate", investigate)
    g.add_node("draft", draft)
    g.add_node("ground_check", ground_check)
    g.add_node("revise", revise)
    g.add_node("propose_actions", propose_actions)
    g.add_node("human_review", human_review)
    g.add_node("execute_actions", execute_actions)
    g.add_node("send", send)
    g.add_node("reject", reject)

    g.add_edge(START, "guard")
    g.add_edge("guard", "triage")
    g.add_conditional_edges("triage", route_after_triage)
    g.add_edge("close_no_reply", END)
    g.add_edge("retrieve", "investigate")
    g.add_edge("investigate", "draft")
    g.add_edge("draft", "ground_check")
    g.add_conditional_edges("ground_check", route_after_ground)
    g.add_edge("revise", "ground_check")
    g.add_edge("propose_actions", "human_review")
    g.add_conditional_edges("human_review", route_after_review)
    g.add_edge("execute_actions", "send")
    g.add_edge("send", END)
    g.add_edge("reject", END)
    return g.compile(checkpointer=checkpointer)
