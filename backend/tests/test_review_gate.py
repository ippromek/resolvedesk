"""review gate checks."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from fastapi.testclient import TestClient
from langgraph.types import Command

from app.config import Settings
from app.investigation import (
    Investigator,
)
from app.main import create_app


def _client(tmp_path: Path, investigator: Investigator | None = None) -> TestClient:
    settings = Settings(llm_provider="offline", db_path=tmp_path / "app.db", checkpoint_path=tmp_path / "ckpt.db")
    return TestClient(create_app(settings, investigator=investigator))


_REVIEW = {"decision": "approve", "reviewer": "reviewer", "approved_action_ids": []}


def test_edit_without_body_is_refused(brakes: dict[str, str], client: TestClient) -> None:
    case = client.post("/api/cases", json=brakes).json()
    r = client.post(f"/api/cases/{case['id']}/review", json={"decision": "edit", "reviewer": "reviewer"})
    assert r.status_code == 422
    assert client.get(f"/api/cases/{case['id']}").json()["status"] == "awaiting_review"


def test_cannot_review_twice(brakes: dict[str, str], client: TestClient) -> None:
    case = client.post("/api/cases", json=brakes).json()
    client.post(f"/api/cases/{case['id']}/review", json={"decision": "approve", "reviewer": "reviewer"})
    r = client.post(f"/api/cases/{case['id']}/review", json={"decision": "approve", "reviewer": "reviewer"})
    assert r.status_code == 409


def test_injection_is_flagged_and_still_needs_review(injection: dict[str, str], client: TestClient) -> None:
    case = client.post("/api/cases", json=injection).json()
    assert case["injection_suspected"] is True
    assert case["status"] == "awaiting_review"


def test_injection_approval_requires_acknowledgement(injection: dict[str, str], tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        case = client.post("/api/cases", json=injection).json()
        case_id = case["id"]
        blocked = client.post(
            f"/api/cases/{case_id}/review",
            json={"decision": "approve", "reviewer": "reviewer"},
        )
        assert blocked.status_code == 422
        edited = client.post(
            f"/api/cases/{case_id}/review",
            json={"decision": "edit", "reviewer": "reviewer", "body": "A short reply with no offer."},
        )
        assert edited.status_code == 422
        still = client.get(f"/api/cases/{case_id}").json()
        assert still["status"] == "awaiting_review"
        rejected = client.post(
            f"/api/cases/{case_id}/review",
            json={"decision": "reject", "reviewer": "reviewer", "note": "Not sending this."},
        )
        assert rejected.status_code == 200
        assert rejected.json()["status"] == "rejected"


def test_acknowledged_injection_is_stored_on_the_review(injection: dict[str, str], tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        case = client.post("/api/cases", json=injection).json()
        done = client.post(
            f"/api/cases/{case['id']}/review",
            json={"decision": "approve", "reviewer": "reviewer", "injection_acknowledged": True},
        ).json()
    assert done["status"] == "sent"
    assert done["state"]["review"]["injection_acknowledged"] is True
    review_event = next(item for item in done["state"]["events"] if item["step"] == "human_review")
    assert review_event["detail"] == "approve — injection warning acknowledged"


def test_invalid_direct_resume_then_valid_review_sends(
    make_settings: Callable[..., Settings], injection: dict[str, str], tmp_path: Path
) -> None:
    with TestClient(create_app(make_settings(tmp_path))) as client:
        case = client.post("/api/cases", json=injection).json()
        graph = client.app.state.svc.graph
        config = {"configurable": {"thread_id": case["id"]}}
        graph.invoke(Command(resume={**_REVIEW, "injection_acknowledged": False}), config)
        assert tuple(graph.get_state(config).next) == ("human_review",)
        approved = client.post(
            f"/api/cases/{case['id']}/review",
            json={"decision": "approve", "reviewer": "reviewer", "injection_acknowledged": True},
        )
        assert approved.status_code == 200
        assert approved.json()["status"] == "sent"
