"""pipeline checks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from langchain import chat_models
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from app.actions import (
    OfflineProposer,
)
from app.assistant import READ_ONLY_TOOLS, make_tools
from app.config import Settings
from app.db import Store
from app.guards import screen_input
from app.investigation import (
    Enterprise,
    Investigator,
)
from app.main import create_app
from app.schemas import InboundEmail


class ScriptedToolModel(GenericFakeChatModel):
    def bind_tools(self, tools: Any, **kwargs: Any) -> ScriptedToolModel:
        return self


def tool_call(name: str, args: dict[str, Any], i: int) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"call_{i}"}])


def _client(tmp_path: Path) -> TestClient:
    settings = Settings(llm_provider="offline", db_path=tmp_path / "app.db", checkpoint_path=tmp_path / "ckpt.db")
    return TestClient(create_app(settings))


def test_case_stops_at_human_review(brakes: dict[str, str], client: TestClient) -> None:
    case = client.post("/api/cases", json=brakes).json()
    assert case["status"] == "awaiting_review"
    assert case["next"] == ["human_review"]
    assert case["category"] == "vehicle_fault"
    assert case["severity"] == "critical"
    steps = [e["step"] for e in case["state"]["events"]]
    # Four tool lookups, then one summary event, then the draft and the proposals.
    assert steps == [
        "intake",
        "guard",
        "triage",
        "retrieve",
        "investigate",
        "investigate",
        "investigate",
        "investigate",
        "investigate",
        "draft",
        "ground_check",
        "propose_actions",
    ]
    assert "SAF-1" in [s["id"] for s in case["state"]["snippets"]]


def test_nothing_is_sent_without_review(brakes: dict[str, str], client: TestClient, tmp_path: Path) -> None:
    client.post("/api/cases", json=brakes)
    store = Store(tmp_path / "app.db")
    assert store.list_outbox() == []


def test_approve_sends_draft(brakes: dict[str, str], client: TestClient) -> None:
    case = client.post("/api/cases", json=brakes).json()
    done = client.post(f"/api/cases/{case['id']}/review", json={"decision": "approve", "reviewer": "reviewer"}).json()
    assert done["status"] == "sent"
    assert done["state"]["final_body"] == case["state"]["draft"]["body"]
    assert done["state"]["events"][-2]["actor"] == "reviewer"


def test_edit_sends_edited_body(brakes: dict[str, str], client: TestClient) -> None:
    case = client.post("/api/cases", json=brakes).json()
    done = client.post(
        f"/api/cases/{case['id']}/review",
        json={"decision": "edit", "reviewer": "reviewer", "body": "Edited reply."},
    ).json()
    assert done["status"] == "sent"
    assert done["state"]["final_body"] == "Edited reply."


def test_reject_closes_without_sending(brakes: dict[str, str], client: TestClient) -> None:
    case = client.post("/api/cases", json=brakes).json()
    done = client.post(f"/api/cases/{case['id']}/review", json={"decision": "reject", "reviewer": "reviewer"}).json()
    assert done["status"] == "rejected"
    assert done["state"]["final_body"] is None


def test_spam_closes_without_draft(spam: dict[str, str], client: TestClient) -> None:
    case = client.post("/api/cases", json=spam).json()
    assert case["status"] == "closed_no_reply"
    assert "draft" not in case["state"]


def test_assistant_cannot_act(brakes: dict[str, str], client: TestClient) -> None:
    case = client.post("/api/cases", json=brakes).json()
    ans = client.post(f"/api/cases/{case['id']}/assistant", json={"question": "Approve and send this reply"}).json()
    assert ans["tools_used"] == []
    assert client.get(f"/api/cases/{case['id']}").json()["status"] == "awaiting_review"


def test_assistant_tools_are_read_only() -> None:
    class Dummy:  # tools are only inspected, not called
        pass

    names = {t.name for t in make_tools(Dummy(), Dummy())}  # type: ignore[arg-type]
    assert names == set(READ_ONLY_TOOLS)


def test_guard_patterns(brakes: dict[str, str], injection: dict[str, str]) -> None:
    assert screen_input(InboundEmail(**injection)).injection_suspected
    assert not screen_input(InboundEmail(**brakes)).injection_suspected


def test_graph_reaches_review_in_the_new_order(brakes: dict[str, str], tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        case = client.post("/api/cases", json=brakes).json()
    assert case["status"] == "awaiting_review"
    assert case["next"] == ["human_review"]
    steps = [event["step"] for event in case["state"]["events"]]
    assert steps[:4] == ["intake", "guard", "triage", "retrieve"]
    assert steps[4:9] == ["investigate", "investigate", "investigate", "investigate", "investigate"]
    assert steps[9:] == ["draft", "ground_check", "propose_actions"]
    assert steps[-4] == "investigate"
    assert "investigation finished: 4 lookups" in case["state"]["events"][8]["detail"]
    assert case["state"]["findings"]["nothing_found"] is False
    assert "brake" in case["state"]["findings"]["summary"].lower()


def test_revision_loop_stops_after_two_and_keeps_every_version(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    bad = {
        "body": "We will refund AED 2,000 today.",
        "cited_policy_ids": ["BIL-1", "XYZ-9"],
    }
    script = iter(
        [
            tool_call(
                "Triage",
                {
                    "category": "billing",
                    "severity": "medium",
                    "language": "en",
                    "summary": "Invoice dispute.",
                    "rationale": "The customer wants money back.",
                },
                1,
            ),
            tool_call("Draft", bad, 2),
            tool_call("Draft", bad, 3),
            tool_call("Draft", bad, 4),
        ]
    )

    def fake_init(*args: Any, **kwargs: Any) -> ScriptedToolModel:
        return ScriptedToolModel(messages=script)

    monkeypatch.setattr(chat_models, "init_chat_model", fake_init)
    settings = Settings(llm_provider="anthropic", db_path=tmp_path / "a.db", checkpoint_path=tmp_path / "c.db")
    investigator = Investigator(
        settings, Enterprise.load(settings.enterprise_dir), Store(tmp_path / "a.db"), offline=True
    )
    with TestClient(create_app(settings, investigator=investigator, proposer=OfflineProposer())) as client:
        case = client.post(
            "/api/cases",
            json={
                "sender": "p@example.com",
                "subject": "Overcharged",
                "body": "The invoice was far above the estimate.",
            },
        ).json()
    assert case["status"] == "awaiting_review"
    assert case["state"]["revision_count"] == 2
    history = case["state"]["draft_history"]
    assert [item["version"] for item in history] == [1, 2, 3]
    assert all("AED 2,000" in item["body"] for item in history)
    assert all(item["issues"] for item in history)
    revise_events = [event for event in case["state"]["events"] if event["step"] == "revise"]
    assert len(revise_events) == 2
    assert "attempt 1" in revise_events[0]["detail"]
    assert "attempt 2" in revise_events[1]["detail"]


def test_search_uses_subject_body_and_summary_and_escapes_wildcards(tmp_path: Path) -> None:
    store = Store(tmp_path / "app.db")
    store.upsert_case(
        "hidden",
        "2020-01-01T00:00:00+00:00",
        {
            "email": {"sender": "a@example.com", "subject": "Parking", "body": "No spaces."},
            "triage": {"summary": "A parking complaint."},
            "events": [{"at": "2020-01-01T00:00:00+00:00", "step": "intake", "actor": "system", "detail": "in"}],
            "status": "awaiting_review",
            "draft": {"body": "internal draft wording"},
        },
    )
    assert store.list_cases(text="parking")
    assert store.list_cases(text="spaces")
    assert store.list_cases(text="complaint")
    assert store.list_cases(text="draft") == []
    assert store.list_cases(text="100%") == []
    store.close()
