"""actions checks."""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Callable
from pathlib import Path

from fastapi.testclient import TestClient

from app.actions import (
    compose_final_body,
    place_arranged,
    reference_for,
    validate_proposals,
)
from app.config import Settings
from app.db import Store
from app.investigation import (
    Investigator,
)
from app.main import create_app
from app.schemas import Findings, InboundEmail, PolicySnippet, ProposedAction, Triage


def _client(tmp_path: Path) -> TestClient:
    settings = Settings(llm_provider="offline", db_path=tmp_path / "app.db", checkpoint_path=tmp_path / "ckpt.db")
    return TestClient(create_app(settings))


def _action_count(store: Store) -> int:
    return len(store.list_actions())


def _client_m24(tmp_path: Path, investigator: Investigator | None = None) -> TestClient:
    settings = Settings(llm_provider="offline", db_path=tmp_path / "app.db", checkpoint_path=tmp_path / "ckpt.db")
    return TestClient(create_app(settings, investigator=investigator))


def _pickup_ids(case: dict) -> list[str]:
    return [item["id"] for item in case["state"]["proposals"] if item["type"] == "schedule_roadside_pickup"]


def _outbox_rows(path: Path) -> list[dict[str, object]]:
    store = Store(path)
    try:
        return store.list_outbox()
    finally:
        store.close()


def test_validator_drops_ineligible_unknown_and_unevidenced_proposals() -> None:
    triage = Triage(
        category="billing",
        severity="medium",
        language="en",
        summary="Invoice dispute.",
        rationale="A charge is disputed.",
    )
    email = InboundEmail(sender="p@example.com", subject="Invoice", body="The invoice looks wrong.")
    snippets = [PolicySnippet(id="BIL-1", title="Invoice disputes", text="We reply within 5 working days.", score=1)]
    findings = Findings(summary="Nothing relevant found.", facts=[], nothing_found=True)
    raw = [
        ProposedAction(
            type="schedule_roadside_pickup", args={"branch": "Al Quoz"}, rationale="Collect it.", evidence=["BIL-1"]
        ),
        ProposedAction(type="send_cash", args={}, rationale="Pay them.", evidence=["BIL-1"]),
        ProposedAction(
            type="open_refund_review",
            args={"reason": "Review the charge."},
            rationale="A person should review it.",
            evidence=[],
        ),
    ]
    kept, drops = validate_proposals(raw, triage, email, snippets, findings)
    assert kept == []
    assert any("only for a critical vehicle fault" in item for item in drops)
    assert any("not in the catalogue" in item for item in drops)
    assert any("no evidence" in item for item in drops)


def test_a_branch_that_is_not_on_file_is_dropped() -> None:
    triage = Triage(
        category="vehicle_fault",
        severity="critical",
        language="en",
        summary="The brakes may be unsafe.",
        rationale="A safety risk.",
    )
    email = InboundEmail(sender="r.haddad@example.com", subject="Brakes", body="The brakes grind.")
    snippets = [PolicySnippet(id="SAF-1", title="Safety", text="We collect unsafe vehicles.", score=1.0)]
    findings = Findings(summary="Nothing else.", facts=[], nothing_found=True)
    raw = [
        ProposedAction(
            type="schedule_roadside_pickup",
            args={"branch": "the customer's driveway"},
            rationale="Collect it.",
            evidence=["SAF-1"],
        )
    ]
    kept, drops = validate_proposals(raw, triage, email, snippets, findings)
    assert kept == []
    assert any("unknown branch" in item for item in drops)


def test_actions_do_not_run_before_review_or_on_reject_or_twice(brakes: dict[str, str], tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        case = client.post("/api/cases", json=brakes).json()
        store = client.app.state.svc.store
        assert _action_count(store) == 0
        proposals = case["state"]["proposals"]
        assert len(proposals) >= 2
        second = proposals[1]["id"]

        empty = client.post(
            f"/api/cases/{case['id']}/review",
            json={"decision": "approve", "reviewer": "reviewer", "approved_action_ids": []},
        ).json()
        assert empty["status"] == "sent"
        assert _action_count(store) == 0
        assert all(item["status"] == "not_approved" for item in empty["state"]["action_results"])

        second_case = client.post("/api/cases", json=brakes).json()
        chosen = second_case["state"]["proposals"][0]
        other = second_case["state"]["proposals"][1]
        done = client.post(
            f"/api/cases/{second_case['id']}/review",
            json={"decision": "approve", "reviewer": "reviewer", "approved_action_ids": [chosen["id"]]},
        ).json()
        rows = store.list_actions(second_case["id"])
        assert len(rows) == 1
        assert rows[0]["type"] == chosen["type"]
        assert rows[0]["approved_by"] == "reviewer"
        executed = [item for item in done["state"]["action_results"] if item["status"] == "executed"]
        skipped = [item for item in done["state"]["action_results"] if item["status"] == "not_approved"]
        assert len(executed) == 1 and executed[0]["ref"]
        assert any(item["action_id"] == other["id"] for item in skipped)

        rejected = client.post("/api/cases", json=brakes).json()
        ids = [item["id"] for item in rejected["state"]["proposals"]]
        closed = client.post(
            f"/api/cases/{rejected['id']}/review",
            json={"decision": "reject", "reviewer": "reviewer", "approved_action_ids": ids},
        ).json()
        assert closed["status"] == "rejected"
        assert store.list_actions(rejected["id"]) == []

        unknown = client.post("/api/cases", json=brakes).json()
        bad = client.post(
            f"/api/cases/{unknown['id']}/review",
            json={"decision": "approve", "reviewer": "reviewer", "approved_action_ids": ["no-such-action"]},
        )
        assert bad.status_code == 422
        assert client.get(f"/api/cases/{unknown['id']}").json()["status"] == "awaiting_review"
        assert store.list_actions(unknown["id"]) == []

        again = client.post(
            f"/api/cases/{second_case['id']}/review",
            json={"decision": "approve", "reviewer": "reviewer", "approved_action_ids": [chosen["id"], second]},
        )
        assert again.status_code == 409
        assert len(store.list_actions(second_case["id"])) == 1


def test_approved_pickup_is_named_in_the_final_reply(brakes: dict[str, str], tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        case = client.post("/api/cases", json=brakes).json()
        pickup = next(item for item in case["state"]["proposals"] if item["type"] == "schedule_roadside_pickup")
        done = client.post(
            f"/api/cases/{case['id']}/review",
            json={"decision": "approve", "reviewer": "reviewer", "approved_action_ids": [pickup["id"]]},
        ).json()
    assert done["status"] == "sent"
    body = done["state"]["final_body"]
    executed = next(item for item in done["state"]["action_results"] if item["status"] == "executed")
    assert "What we have arranged:" in body
    assert executed["ref"] in body
    assert "Roadside pickup from your location" in body


def test_escalation_alone_adds_no_arranged_section(brakes: dict[str, str], tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        case = client.post("/api/cases", json=brakes).json()
        escalation = next(item for item in case["state"]["proposals"] if item["type"] == "escalate_to_branch_manager")
        done = client.post(
            f"/api/cases/{case['id']}/review",
            json={"decision": "approve", "reviewer": "reviewer", "approved_action_ids": [escalation["id"]]},
        ).json()
    assert "What we have arranged" not in done["state"]["final_body"]
    assert done["state"]["final_body"] == case["state"]["draft"]["body"]


def test_reject_has_no_final_body(brakes: dict[str, str], tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        case = client.post("/api/cases", json=brakes).json()
        done = client.post(
            f"/api/cases/{case['id']}/review",
            json={
                "decision": "reject",
                "reviewer": "reviewer",
                "approved_action_ids": [item["id"] for item in case["state"]["proposals"]],
            },
        ).json()
    assert done["status"] == "rejected"
    assert done["state"]["final_body"] is None


def test_preview_matches_the_body_sent_for_the_same_actions(brakes: dict[str, str], tmp_path: Path) -> None:
    with _client_m24(tmp_path) as client:
        case = client.post("/api/cases", json=brakes).json()
        ids = [
            item["id"]
            for item in case["state"]["proposals"]
            if item["type"] in {"schedule_roadside_pickup", "book_safety_inspection"}
        ]
        preview = client.post(
            f"/api/cases/{case['id']}/preview",
            json={"approved_action_ids": ids},
        ).json()
        unchanged = client.get(f"/api/cases/{case['id']}").json()
        assert unchanged["status"] == "awaiting_review"
        assert "human_review" in unchanged["next"]
        done = client.post(
            f"/api/cases/{case['id']}/review",
            json={"decision": "approve", "reviewer": "reviewer", "approved_action_ids": ids},
        ).json()
    body = done["state"]["final_body"]
    assert body == preview["final_body"]
    assert body.index("What we have arranged:") < body.index("Meridian Motors Customer Care")


def test_arabic_case_inserts_the_section_before_the_sign_off() -> None:
    body = "عزيزنا العميل،\n\nشكراً لتواصلك معنا.\n\nمع التحية،\nخدمة عملاء ميريديان موتورز"
    proposals = [
        {
            "id": "a1",
            "type": "schedule_roadside_pickup",
            "args": {"branch": "Al Quoz"},
            "ref": "PICKUP-0001",
        }
    ]
    placed, _section = compose_final_body(body, proposals, ["a1"], "ar")
    assert placed.index("ما رتبناه:") < placed.index("مع التحية،")
    assert placed.rstrip().endswith("خدمة عملاء ميريديان موتورز")
    fallback = place_arranged("No sign-off in this reply.", "What we have arranged: Roadside pickup.")
    assert fallback.endswith("What we have arranged: Roadside pickup.")


def test_concurrent_reviews_record_each_action_once(brakes: dict[str, str], tmp_path: Path) -> None:
    settings = Settings(llm_provider="offline", db_path=tmp_path / "app.db", checkpoint_path=tmp_path / "ckpt.db")
    with TestClient(create_app(settings)) as client:
        case = client.post("/api/cases", json=brakes).json()
        pickup = next(item for item in case["state"]["proposals"] if item["type"] == "schedule_roadside_pickup")
        barrier = threading.Barrier(2)
        codes: list[int] = []
        codes_lock = threading.Lock()

        def approve() -> None:
            barrier.wait()
            response = client.post(
                f"/api/cases/{case['id']}/review",
                json={"decision": "approve", "reviewer": "reviewer", "approved_action_ids": [pickup["id"]]},
            )
            with codes_lock:
                codes.append(response.status_code)

        threads = [threading.Thread(target=approve) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)
    assert sorted(codes) == [200, 409]
    recorded = [row for row in Store(settings.db_path).list_actions(case["id"]) if row["action_id"] == pickup["id"]]
    assert len(recorded) == 1


def test_record_action_failure_then_retry_executes_once(
    make_settings: Callable[..., Settings],
    wait_for_status: Callable[..., dict[str, object]],
    brakes: dict[str, str],
    tmp_path: Path,
) -> None:
    settings = make_settings(tmp_path)
    with TestClient(create_app(settings)) as client:
        case = client.post("/api/cases", json=brakes).json()
        chosen = _pickup_ids(case)
        assert chosen
        store = client.app.state.svc.store
        original = store.record_action
        calls = {"n": 0}

        def boom(
            case_id: str,
            action_id: str,
            action_type: str,
            args: dict,
            approved_by: str,
            at: str,
            status: str,
        ) -> None:
            calls["n"] += 1
            if calls["n"] == 1:
                raise sqlite3.OperationalError("database is locked")
            original(case_id, action_id, action_type, args, approved_by, at, status)

        store.record_action = boom
        failed = client.post(
            f"/api/cases/{case['id']}/review",
            json={"decision": "approve", "reviewer": "reviewer", "approved_action_ids": chosen},
        )
        assert failed.status_code == 503
        assert client.get(f"/api/cases/{case['id']}").json()["status"] == "failed"
        assert store.list_actions(case["id"]) == []
        retried = client.post(f"/api/cases/{case['id']}/retry")
        assert retried.status_code == 202
        done = wait_for_status(client, case["id"])
        assert done["status"] == "sent"
        executed = [row["action_id"] for row in store.list_actions(case["id"])]
        assert executed == chosen
        rows = _outbox_rows(settings.db_path)
        assert len(rows) == 1
        assert rows[0]["approved_by"] == "reviewer"
        assert rows[0]["recipient"] == brakes["sender"]


def test_second_action_insert_and_send_failure_retry_once_each(
    make_settings: Callable[..., Settings],
    wait_for_status: Callable[..., dict[str, object]],
    brakes: dict[str, str],
    tmp_path: Path,
) -> None:
    settings = make_settings(tmp_path)
    with TestClient(create_app(settings)) as client:
        case = client.post("/api/cases", json=brakes).json()
        chosen = [
            item["id"]
            for item in case["state"]["proposals"]
            if item["type"] in {"schedule_roadside_pickup", "book_safety_inspection"}
        ]
        assert len(chosen) == 2
        store = client.app.state.svc.store
        original = store.record_action
        calls = {"n": 0}

        def boom(
            case_id: str,
            action_id: str,
            action_type: str,
            args: dict,
            approved_by: str,
            at: str,
            status: str,
        ) -> None:
            calls["n"] += 1
            if calls["n"] == 2:
                raise sqlite3.OperationalError("database is locked")
            original(case_id, action_id, action_type, args, approved_by, at, status)

        store.record_action = boom
        failed = client.post(
            f"/api/cases/{case['id']}/review",
            json={"decision": "approve", "reviewer": "reviewer", "approved_action_ids": chosen},
        )
        assert failed.status_code == 503
        assert [row["action_id"] for row in store.list_actions(case["id"])] == [chosen[0]]
        store.record_action = original
        assert client.post(f"/api/cases/{case['id']}/retry").status_code == 202
        done = wait_for_status(client, case["id"])
        assert done["status"] == "sent"
        assert [row["action_id"] for row in store.list_actions(case["id"])] == chosen

        other = client.post("/api/cases", json=brakes).json()
        pickup = [item["id"] for item in other["state"]["proposals"] if item["type"] == "schedule_roadside_pickup"]
        preview = client.post(
            f"/api/cases/{other['id']}/preview",
            json={"approved_action_ids": pickup},
        ).json()
        original_send = store.queue_outbound
        sent = {"n": 0}

        def boom_send(case_id: str, recipient: str, body: str, approved_by: str, at: str) -> None:
            sent["n"] += 1
            if sent["n"] == 1:
                raise sqlite3.OperationalError("database is locked")
            original_send(case_id, recipient, body, approved_by, at)

        store.queue_outbound = boom_send
        send_failed = client.post(
            f"/api/cases/{other['id']}/review",
            json={"decision": "approve", "reviewer": "reviewer", "approved_action_ids": pickup},
        )
        assert send_failed.status_code == 503
        assert len(store.list_actions(other["id"])) == 1
        assert len(_outbox_rows(settings.db_path)) == 1
        mid = client.get(f"/api/cases/{other['id']}").json()
        assert mid["state"]["events"][-1]["failed_step"] == "send"
        store.queue_outbound = original_send
        assert client.post(f"/api/cases/{other['id']}/retry").status_code == 202
        finished = wait_for_status(client, other["id"])
        assert finished["status"] == "sent"
        assert len(store.list_actions(other["id"])) == 1
        assert len(_outbox_rows(settings.db_path)) == 2
        assert preview["arranged_section"]
        assert preview["arranged_section"] in finished["state"]["final_body"]


def test_approve_outbox_has_the_arranged_section_and_reject_and_spam_have_none(
    make_settings: Callable[..., Settings], brakes: dict[str, str], spam: dict[str, str], tmp_path: Path
) -> None:
    settings = make_settings(tmp_path)
    with TestClient(create_app(settings)) as client:
        case = client.post("/api/cases", json=brakes).json()
        chosen = _pickup_ids(case)
        assert chosen
        approved = client.post(
            f"/api/cases/{case['id']}/review",
            json={"decision": "approve", "reviewer": "reviewer", "approved_action_ids": chosen},
        )
        assert approved.status_code == 200
        rows = _outbox_rows(settings.db_path)
        assert len(rows) == 1
        assert rows[0]["approved_by"] == "reviewer"
        assert rows[0]["recipient"] == brakes["sender"]
        assert "What we have arranged:" in rows[0]["body"]

        rejected = client.post("/api/cases", json=brakes).json()
        reject = client.post(
            f"/api/cases/{rejected['id']}/review",
            json={"decision": "reject", "reviewer": "reviewer"},
        )
        assert reject.status_code == 200
        client.post("/api/cases", json=spam)
        assert len(_outbox_rows(settings.db_path)) == 1


def test_references_use_six_hex_characters() -> None:
    ref = reference_for("case-1", "a1", "schedule_roadside_pickup")
    suffix = ref.split("-", 1)[1]
    assert ref.startswith("PICKUP-")
    assert len(suffix) == 6
    assert suffix == suffix.upper()
    assert int(suffix, 16) >= 0
