"""llm wiring checks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from langchain import chat_models
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from app.actions import (
    OfflineProposer,
)
from app.config import Settings, get_settings
from app.db import Store
from app.investigation import (
    Enterprise,
    Investigator,
)
from app.main import ASSISTANT_MODEL_FAILURE, CASE_MODEL_FAILURE, create_app
from app.prompts import ASSISTANT_SYSTEM
from eval.run_eval import parse_args


class ScriptedToolModel(GenericFakeChatModel):
    def bind_tools(self, tools: Any, **kwargs: Any) -> ScriptedToolModel:
        return self


def tool_call(name: str, args: dict[str, Any], i: int) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"call_{i}"}])


class RaisingModel(GenericFakeChatModel):
    def bind_tools(self, tools: Any, **kwargs: Any) -> RaisingModel:
        return self

    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        raise TimeoutError("timed out")


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    return Settings(llm_provider="anthropic", db_path=tmp_path / "a.db", checkpoint_path=tmp_path / "c.db")


def test_missing_provider_key_refuses_to_start(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("LLM_MODEL", "claude-sonnet-5-5")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match=r"ANTHROPIC_API_KEY"):
        get_settings()


def test_chat_model_receives_timeout_and_retries(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    seen: dict[str, Any] = {}

    def fake_init(*args: Any, **kwargs: Any) -> RaisingModel:
        seen.update(kwargs)
        return RaisingModel(messages=iter([]))

    monkeypatch.setattr(chat_models, "init_chat_model", fake_init)
    settings = Settings(
        llm_provider="anthropic",
        llm_model="claude-sonnet-5-5",
        llm_timeout_s=12,
        llm_max_retries=4,
        db_path=tmp_path / "a.db",
        checkpoint_path=tmp_path / "c.db",
    )
    with TestClient(create_app(settings)) as client:
        health = client.get("/api/health").json()
    assert health["model"] == "anthropic:claude-sonnet-5-5"
    assert seen["timeout"] == 12
    assert seen["max_retries"] == 4
    assert seen["model_provider"] == "anthropic"


def test_model_failure_returns_503_and_creates_no_case(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def fake_init(*args: Any, **kwargs: Any) -> RaisingModel:
        return RaisingModel(messages=iter([]))

    monkeypatch.setattr(chat_models, "init_chat_model", fake_init)
    settings = Settings(llm_provider="anthropic", db_path=tmp_path / "a.db", checkpoint_path=tmp_path / "c.db")
    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/api/cases",
            json={
                "sender": "p@example.com",
                "subject": "Overcharged",
                "body": "The invoice was far above the estimate.",
            },
        )
        assert response.status_code == 503
        assert response.json()["detail"] == CASE_MODEL_FAILURE
        assert client.get("/api/cases").json() == []
        store = client.app.state.svc.store
        cases = len(store.list_cases())
        outbox = len(store.list_outbox())
        assert cases == 0
        assert outbox == 0


def test_assistant_failure_returns_503_and_leaves_the_case(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    script = iter(
        [
            tool_call(
                "Triage",
                {
                    "category": "billing",
                    "severity": "medium",
                    "language": "en",
                    "summary": "Invoice above estimate.",
                    "rationale": "Extra work not approved.",
                },
                1,
            ),
            tool_call(
                "Draft",
                {"body": "We will review your invoice within 5 working days.", "cited_policy_ids": ["BIL-1"]},
                2,
            ),
        ]
    )

    def fake_init(*args: Any, **kwargs: Any) -> ScriptedToolModel:
        return ScriptedToolModel(messages=script)

    monkeypatch.setattr(chat_models, "init_chat_model", fake_init)
    settings = Settings(llm_provider="anthropic", db_path=tmp_path / "a.db", checkpoint_path=tmp_path / "c.db")

    class Boom:
        def invoke(self, *args: Any, **kwargs: Any) -> Any:
            raise TimeoutError("timed out")

    investigator = Investigator(
        settings, Enterprise.load(settings.enterprise_dir), Store(settings.db_path), offline=True
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
        client.app.state.svc.assistant._agent = Boom()
        response = client.post(f"/api/cases/{case['id']}/assistant", json={"question": "What is this about?"})
        assert response.status_code == 503
        assert response.json()["detail"] == ASSISTANT_MODEL_FAILURE
        again = client.get(f"/api/cases/{case['id']}").json()
        assert again["status"] == case["status"]
        assert again["state"]["draft"]["body"] == case["state"]["draft"]["body"]


def test_eval_repeat_argument_parses_without_a_model() -> None:
    assert parse_args([]).repeat == 1
    assert parse_args(["--repeat", "3"]).repeat == 3
    with pytest.raises(SystemExit):
        parse_args(["--repeat", "0"])


def test_llm_brain_and_assistant(monkeypatch: pytest.MonkeyPatch, settings: Settings) -> None:
    script = iter(
        [
            tool_call(
                "Triage",
                {
                    "category": "billing",
                    "severity": "medium",
                    "language": "en",
                    "summary": "Invoice above estimate.",
                    "rationale": "Extra work not approved.",
                },
                1,
            ),
            tool_call(
                "Draft",
                {"body": "We will review your invoice within 5 working days.", "cited_policy_ids": ["BIL-1"]},
                2,
            ),
        ]
    )

    def fake_init(*args: Any, **kwargs: Any) -> ScriptedToolModel:
        return ScriptedToolModel(messages=script)

    monkeypatch.setattr(chat_models, "init_chat_model", fake_init)

    investigator = Investigator(
        settings, Enterprise.load(settings.enterprise_dir), Store(settings.db_path), offline=True
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
        assert case["category"] == "billing"
        assert case["state"]["draft"]["cited_policy_ids"] == ["BIL-1"]
        assert case["state"]["grounding"]["ok"] is True

        # Assistant: one tool call, then an answer.
        svc = client.app.state.svc
        agent_model = ScriptedToolModel(
            messages=iter(
                [
                    tool_call("get_case", {"case_id": case["id"]}, 3),
                    AIMessage(content="The customer disputes an invoice; policy BIL-1 applies."),
                ]
            )
        )
        svc.assistant._agent = create_agent(agent_model, svc.assistant._tools, system_prompt=ASSISTANT_SYSTEM)
        ans = client.post(f"/api/cases/{case['id']}/assistant", json={"question": "What is this about?"}).json()
        assert ans["tools_used"] == ["get_case"]
        assert "BIL-1" in ans["answer"]
