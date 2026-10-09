"""logging checks."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel, ValidationError

from app.brain import OfflineBrain
from app.config import Settings
from app.failures import log_failure
from app.main import create_app
from app.schemas import Draft, Findings, InboundEmail, PolicySnippet, Triage


class _DraftRaises(OfflineBrain):
    def __init__(self, exc: BaseException) -> None:
        self._exc = exc

    def draft(self, email: InboundEmail, triage: Triage, snippets: list[PolicySnippet], findings: Findings) -> Draft:
        raise self._exc


def test_offline_events_omit_usage_and_metrics_are_null(brakes: dict[str, str], tmp_path: Path) -> None:
    settings = Settings(llm_provider="offline", db_path=tmp_path / "a.db", checkpoint_path=tmp_path / "c.db")
    with TestClient(create_app(settings)) as client:
        case = client.post(
            "/api/cases",
            json={
                "sender": "r.haddad@example.com",
                "subject": "Brakes grinding after service",
                "body": "Since the service the brakes grind and the pedal feels soft. Is it safe?",
            },
        ).json()
        for ev in case["state"]["events"]:
            assert "latency_ms" not in ev
            assert "input_tokens" not in ev
            assert "output_tokens" not in ev
        metrics = client.get("/api/metrics").json()
        assert metrics["avg_latency_ms"] == {"triage": None, "draft": None, "investigate": None}
        assert metrics["total_tokens"] is None


def test_failure_text_follows_the_error_class(
    make_settings: Callable[..., Settings],
    wait_for_status: Callable[..., dict[str, object]],
    brakes: dict[str, str],
    tmp_path: Path,
) -> None:
    provider_dir = tmp_path / "provider"
    other_dir = tmp_path / "other"
    provider_dir.mkdir()
    other_dir.mkdir()
    provider = TestClient(create_app(make_settings(provider_dir), brain=_DraftRaises(TimeoutError("timed out"))))
    other = TestClient(create_app(make_settings(other_dir), brain=_DraftRaises(RuntimeError("database is locked"))))
    with provider, other:
        provider_id = provider.post("/api/cases/async", json=brakes).json()["id"]
        other_id = other.post("/api/cases/async", json=brakes).json()["id"]
        provider_case = wait_for_status(provider, provider_id)
        other_case = wait_for_status(other, other_id)
    assert provider_case["status"] == "failed"
    assert other_case["status"] == "failed"
    provider_detail = provider_case["state"]["events"][-1]["detail"]
    other_detail = other_case["state"]["events"][-1]["detail"]
    assert provider_detail.endswith("The model did not respond. Try again.")
    assert "timed out" not in provider_detail
    assert other_detail.endswith("This step failed. Try again.")
    assert "database is locked" not in other_detail
    assert provider_case["state"]["events"][-1]["failed_step"] == "draft"
    assert other_case["state"]["events"][-1]["failed_step"] == "draft"


def test_validation_error_input_stays_out_of_the_log(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Marked(BaseModel):
        note: str

    monkeypatch.delenv("LOG_TRACEBACKS", raising=False)
    with pytest.raises(ValidationError) as caught:
        Marked.model_validate({"note": ["MARKER-EMAIL-SECRET"]})
    with caplog.at_level(logging.ERROR, logger="app.failures"):
        log_failure("case-1", "investigate", caught.value)
    assert "MARKER-EMAIL-SECRET" not in caplog.text
    assert "ValidationError" in caplog.text
    assert "Traceback" not in caplog.text
    assert "\\" not in caplog.text
    assert "/" not in caplog.text
