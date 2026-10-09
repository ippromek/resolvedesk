"""A read-only assistant for complaint handlers.

Built with `langchain.agents.create_agent`. Its safety property is structural:
every tool below only reads. Asking it to approve, send or close a case cannot
work however the request is worded, because no tool that does that exists.
"""

from __future__ import annotations

import json
from typing import Any

from langchain.agents.middleware.types import InputAgentState
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from app.brain import make_chat_model
from app.config import Settings
from app.db import Store
from app.guards import strip_delimiters
from app.kb import KnowledgeBase
from app.prompts import ASSISTANT_SYSTEM
from app.schemas import AssistantAnswer

READ_ONLY_TOOLS = ("get_case", "search_cases", "search_policies", "case_metrics")


def make_tools(store: Store, kb: KnowledgeBase) -> list[Any]:
    @tool
    def get_case(case_id: str) -> str:
        """Get one case: the email, triage, retrieved policies, draft, review and timeline."""
        case = store.get_case(case_id)
        if not case:
            return f"No case with id {case_id}."
        s = case["state"]
        raw_email = s.get("email") if isinstance(s.get("email"), dict) else {}
        subject = strip_delimiters(str(raw_email.get("subject") or ""))
        body = strip_delimiters(str(raw_email.get("body") or ""))
        sender = str(raw_email.get("sender") or "")
        draft = strip_delimiters(str((s.get("draft") or {}).get("body") or ""))
        events = [
            {"step": item.get("step"), "actor": item.get("actor"), "at": item.get("at")}
            for item in (s.get("events") or [])
            if isinstance(item, dict)
        ]
        return json.dumps(
            {
                "id": case["id"],
                "status": case["status"],
                "email": {
                    "sender": sender,
                    "subject": subject,
                    "body": f"<customer_email>\nSubject: {subject}\n\n{body}\n</customer_email>",
                },
                "triage": s.get("triage"),
                "guard": s.get("guard"),
                "policies": [p["id"] for p in s.get("snippets", [])],
                "draft": f"<draft>\n{draft}\n</draft>",
                "review": s.get("review"),
                "events": events,
            },
            ensure_ascii=False,
        )

    @tool
    def search_cases(
        category: str | None = None,
        severity: str | None = None,
        status: str | None = None,
        text: str | None = None,
    ) -> str:
        """Find cases by category, severity, status or free text. Returns up to 20 summaries."""
        rows = store.list_cases(status=status, category=category, severity=severity, text=text, limit=20)
        return json.dumps(
            [
                {
                    "id": r["id"],
                    "subject": strip_delimiters(str(r["subject"])),
                    "status": r["status"],
                    "category": r["category"],
                    "severity": r["severity"],
                    "created_at": r["created_at"],
                }
                for r in rows
            ]
        )

    @tool
    def search_policies(query: str) -> str:
        """Search Meridian Motors customer policies. Returns matching sections with their ids."""
        return json.dumps([h.model_dump() for h in kb.search(query, k=3)], ensure_ascii=False)

    @tool
    def case_metrics() -> str:
        """Counts of cases by status, category, severity and review decision."""
        return json.dumps(store.metrics())

    return [get_case, search_cases, search_policies, case_metrics]


class Assistant:
    def __init__(self, settings: Settings, store: Store, kb: KnowledgeBase) -> None:
        self._store = store
        self._kb = kb
        self._offline = settings.llm_provider == "offline"
        self._tools = make_tools(store, kb)
        if not self._offline:
            from langchain.agents import create_agent

            model = make_chat_model(settings, temperature=0)
            self._agent = create_agent(model, self._tools, system_prompt=ASSISTANT_SYSTEM)

    def ask(self, case_id: str, question: str) -> AssistantAnswer:
        if self._offline:
            return self._offline_answer(case_id, question)
        state: InputAgentState = {
            "messages": [HumanMessage(content=f"Open case id: {case_id}\n\nQuestion: {question}")]
        }
        config: RunnableConfig = {"recursion_limit": 12}
        result = self._agent.invoke(state, config)
        messages = result["messages"]
        used = [m.name for m in messages if isinstance(m, ToolMessage) and m.name]
        final = next((m for m in reversed(messages) if isinstance(m, AIMessage) and m.content), None)
        text = final.text if final is not None else "No answer."
        return AssistantAnswer(answer=text, tools_used=list(dict.fromkeys(used)))

    def _offline_answer(self, case_id: str, question: str) -> AssistantAnswer:
        case = self._store.get_case(case_id)
        if not case:
            return AssistantAnswer(answer=f"No case with id {case_id}.", tools_used=["get_case"])
        q = question.lower()
        if any(w in q for w in ("approve", "send", "close", "escalate", "delete")):
            return AssistantAnswer(
                answer="I can only read cases. I have no tool that approves, sends or closes anything. "
                "Use the review buttons on the case.",
                tools_used=[],
            )
        t = case["state"].get("triage") or {}
        similar = self._store.list_cases(category=t.get("category"), limit=6)
        others = [c["id"][:8] for c in similar if c["id"] != case_id][:5]
        return AssistantAnswer(
            answer=(
                f"Case {case_id[:8]} is '{case['subject']}', status {case['status']}, "
                f"triaged {t.get('category')} / {t.get('severity')}. "
                f"Other cases in the same category: {', '.join(others) or 'none'}. "
                "(Offline mode: set LLM_PROVIDER to use the LLM agent.)"
            ),
            tools_used=["get_case", "search_cases"],
        )
