"""async runs checks."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi.testclient import TestClient

from app.actions import (
    OfflineProposer,
)
from app.brain import OfflineBrain
from app.completion import Completion, Usage
from app.config import Settings
from app.main import create_app
from app.schemas import Draft, Findings, InboundEmail, PolicySnippet, ProposedAction, Triage


def _steps(case: dict) -> list[str]:
    return [event["step"] for event in case["state"]["events"]]


class _DraftFailsOnce(OfflineBrain):
    def __init__(self) -> None:
        self.drafts = 0

    def draft(self, email: InboundEmail, triage: Triage, snippets: list[PolicySnippet], findings: Findings) -> Draft:
        self.drafts += 1
        if self.drafts == 1:
            raise RuntimeError("draft unavailable")
        return super().draft(email, triage, snippets, findings)


class _SlowProposer(OfflineProposer):
    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()

    def propose(
        self,
        email: InboundEmail,
        triage: Triage,
        snippets: list[PolicySnippet],
        findings: Findings,
        draft: Draft,
    ) -> list[ProposedAction]:
        self.entered.set()
        if not self.release.wait(8):
            raise TimeoutError("the scripted proposer was not released")
        return super().propose(email, triage, snippets, findings, draft)


class _BlockDraft(OfflineBrain):
    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()

    def draft(self, email: InboundEmail, triage: Triage, snippets: list[PolicySnippet], findings: Findings) -> Draft:
        self.entered.set()
        self.release.wait()
        return super().draft(email, triage, snippets, findings)


class _SleepingTriage(OfflineBrain):
    def __init__(self) -> None:
        self.by_sender: dict[str, int] = {}
        self._lock = threading.Lock()

    def triage(self, email: InboundEmail) -> Completion[Triage]:
        started = time.perf_counter()
        time.sleep(0.25)
        latency = round((time.perf_counter() - started) * 1000)
        with self._lock:
            self.by_sender[email.sender] = latency
        return Completion(super().triage(email), Usage(latency_ms=latency))


def test_async_create_matches_the_sync_step_order(
    make_settings: Callable[..., Settings],
    wait_for_status: Callable[..., dict[str, object]],
    brakes: dict[str, str],
    tmp_path: Path,
) -> None:
    with TestClient(create_app(make_settings(tmp_path))) as client:
        sync = client.post("/api/cases", json=brakes)
        assert sync.status_code == 201
        started = client.post("/api/cases/async", json=brakes)
        assert started.status_code == 202
        case_id = started.json()["id"]
        async_case = wait_for_status(client, case_id)
    assert async_case["status"] == "awaiting_review"
    assert _steps(async_case) == _steps(sync.json())


def test_a_draft_failure_can_be_retried_without_repeating_triage(
    make_settings: Callable[..., Settings],
    wait_for_status: Callable[..., dict[str, object]],
    brakes: dict[str, str],
    tmp_path: Path,
) -> None:
    brain = _DraftFailsOnce()
    with TestClient(create_app(make_settings(tmp_path), brain=brain)) as client:
        started = client.post("/api/cases/async", json=brakes)
        assert started.status_code == 202
        case_id = started.json()["id"]
        failed = wait_for_status(client, case_id)
        assert failed["status"] == "failed"
        assert any(
            event["step"] == "failed" and event["detail"].startswith("draft failed:")
            for event in failed["state"]["events"]
        )
        assert client.post(f"/api/cases/{case_id}/retry").status_code == 202
        done = wait_for_status(client, case_id)
    assert done["status"] == "awaiting_review"
    assert _steps(done).count("triage") == 1


def test_status_stays_processing_while_actions_are_still_pending(
    make_settings: Callable[..., Settings],
    wait_for_status: Callable[..., dict[str, object]],
    brakes: dict[str, str],
    tmp_path: Path,
) -> None:
    proposer = _SlowProposer()
    with TestClient(create_app(make_settings(tmp_path), proposer=proposer)) as client:
        started = client.post("/api/cases/async", json=brakes)
        assert started.status_code == 202
        case_id = started.json()["id"]
        assert proposer.entered.wait(8)
        mid = client.get(f"/api/cases/{case_id}").json()
        assert mid["status"] == "processing"
        steps = [event["step"] for event in mid["state"]["events"]]
        assert "ground_check" in steps
        assert "propose_actions" not in steps
        proposer.release.set()
        done = wait_for_status(client, case_id)
    assert done["status"] == "awaiting_review"
    assert done["next"] == ["human_review"]


def test_a_processing_case_is_failed_on_startup_and_retry_completes_it(
    make_settings: Callable[..., Settings],
    wait_for_status: Callable[..., dict[str, object]],
    brakes: dict[str, str],
    tmp_path: Path,
) -> None:
    settings = make_settings(tmp_path)
    blocker = _BlockDraft()
    with TestClient(create_app(settings, brain=blocker)) as client:
        started = client.post("/api/cases/async", json=brakes)
        assert started.status_code == 202
        case_id = started.json()["id"]
        assert blocker.entered.wait(8)
        mid = client.get(f"/api/cases/{case_id}").json()
        assert mid["status"] == "processing"
    with TestClient(create_app(settings)) as client:
        failed = client.get(f"/api/cases/{case_id}").json()
        assert failed["status"] == "failed"
        assert any(
            event["step"] == "failed" and event["detail"] == "interrupted by a server restart"
            for event in failed["state"]["events"]
        )
        assert client.post(f"/api/cases/{case_id}/retry").status_code == 202
        done = wait_for_status(client, case_id)
    assert done["status"] == "awaiting_review"
    assert [event["step"] for event in done["state"]["events"]].count("triage") == 1


def test_two_retries_at_once_one_runs(
    make_settings: Callable[..., Settings],
    wait_for_status: Callable[..., dict[str, object]],
    brakes: dict[str, str],
    tmp_path: Path,
) -> None:
    brain = _DraftFailsOnce()
    with TestClient(create_app(make_settings(tmp_path), brain=brain)) as client:
        started = client.post("/api/cases/async", json=brakes)
        case_id = started.json()["id"]
        assert wait_for_status(client, case_id)["status"] == "failed"
        svc = client.app.state.svc
        assert svc.begin_inflight(case_id)
        held = client.post(f"/api/cases/{case_id}/retry")
        assert held.status_code == 409
        assert held.json()["detail"] == "case is still running"
        svc.end_inflight(case_id)

        def hit() -> int:
            return client.post(f"/api/cases/{case_id}/retry").status_code

        with ThreadPoolExecutor(max_workers=2) as pool:
            codes = sorted(future.result() for future in (pool.submit(hit), pool.submit(hit)))
        assert codes == [202, 409]
        assert wait_for_status(client, case_id)["status"] == "awaiting_review"
    assert brain.drafts == 2


def test_concurrent_cases_keep_their_own_latency(
    make_settings: Callable[..., Settings],
    wait_for_status: Callable[..., dict[str, object]],
    brakes: dict[str, str],
    tmp_path: Path,
) -> None:
    brain = _SleepingTriage()
    with TestClient(create_app(make_settings(tmp_path), brain=brain)) as client:
        first = client.post("/api/cases/async", json=brakes)
        second = client.post(
            "/api/cases/async",
            json={
                "sender": "i.novak@example.com",
                "subject": "Airbag warning light",
                "body": "The airbag light is on.",
            },
        )
        assert first.status_code == 202
        assert second.status_code == 202
        done = [wait_for_status(client, first.json()["id"]), wait_for_status(client, second.json()["id"], timeout=20)]
        deadline = time.monotonic() + 5
        while client.app.state.svc.any_inflight() and time.monotonic() < deadline:
            time.sleep(0.05)
    for case in done:
        assert case["status"] == "awaiting_review"
        triage = next(event for event in case["state"]["events"] if event["step"] == "triage")
        assert triage["latency_ms"] == brain.by_sender[case["sender"]]
        assert triage["latency_ms"] >= 200
