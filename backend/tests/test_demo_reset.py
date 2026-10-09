"""demo reset checks."""

from __future__ import annotations

import json
import shutil
import sqlite3
import threading
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi.testclient import TestClient

from app.brain import OfflineBrain
from app.config import Settings
from app.db import Store
from app.demo_clock import shift_restored_clock
from app.main import create_app
from app.schemas import Draft, Findings, InboundEmail, PolicySnippet, Triage


def _action_rows(path: Path) -> list[tuple[str, str, str, str]]:
    scratch = path.with_name(path.name + ".rows")
    shutil.copyfile(path, scratch)
    store = Store(scratch)
    try:
        rows = store.list_actions()
    finally:
        store.close()
        scratch.unlink(missing_ok=True)
    return [(str(row["case_id"]), str(row["action_id"]), str(row["type"]), str(row["status"])) for row in rows]


def _outbox_count(path: Path) -> int:
    store = Store(path)
    try:
        return len(store.list_outbox())
    finally:
        store.close()


def _duration(case: dict) -> timedelta:
    moments = [datetime.fromisoformat(event["at"]) for event in case["state"]["events"]]
    return max(moments) - min(moments)


def _newest(case: dict) -> datetime:
    return max(datetime.fromisoformat(event["at"]) for event in case["state"]["events"])


class _HoldDraft(OfflineBrain):
    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()

    def draft(self, email: InboundEmail, triage: Triage, snippets: list[PolicySnippet], findings: Findings) -> Draft:
        self.started.set()
        assert self.release.wait(timeout=10)
        return super().draft(email, triage, snippets, findings)


def test_reset_restores_the_snapshot_and_does_not_touch_it(
    make_settings: Callable[..., Settings], brakes: dict[str, str], tmp_path: Path
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    built = Settings(llm_provider="offline", db_path=source / "app.db", checkpoint_path=source / "ckpt.db")
    with TestClient(create_app(built)) as client:
        created = client.post("/api/cases", json=brakes).json()
    snap_dir = tmp_path / "demo"
    snap_dir.mkdir()
    shutil.copyfile(built.db_path, snap_dir / "snapshot.db")
    shutil.copyfile(built.checkpoint_path, snap_dir / "snapshot-ckpt.db")
    before_db = (snap_dir / "snapshot.db").read_bytes()
    before_ckpt = (snap_dir / "snapshot-ckpt.db").read_bytes()
    snapshot_actions = _action_rows(snap_dir / "snapshot.db")

    live = make_settings(tmp_path / "live", demo_mode=True, demo_dir=snap_dir)
    with TestClient(create_app(live)) as client:
        blocked = client.post("/api/demo/reset")
        assert blocked.status_code == 200
        restored = client.get("/api/cases").json()
        assert len(restored) == 1
        assert restored[0]["id"] == created["id"]
        assert restored[0]["status"] == "awaiting_review"
        extra = client.post(
            "/api/cases",
            json={"sender": "other@example.com", "subject": "Extra", "body": "A later complaint about the brakes."},
        ).json()
        client.post(
            f"/api/cases/{created['id']}/review",
            json={
                "decision": "approve",
                "reviewer": "reviewer",
                "approved_action_ids": [
                    item["id"]
                    for item in client.get(f"/api/cases/{created['id']}").json()["state"]["proposals"]
                    if item["type"] == "schedule_roadside_pickup"
                ],
            },
        )
        assert _outbox_count(live.db_path) == 1
        again = client.post("/api/demo/reset")
        assert again.status_code == 200
        assert again.json() == {"cases": 1}
        after = client.get("/api/cases").json()
        assert [item["id"] for item in after] == [created["id"]]
        assert after[0]["status"] == "awaiting_review"
        assert client.get(f"/api/cases/{extra['id']}").status_code == 404
    assert (snap_dir / "snapshot.db").read_bytes() == before_db
    assert (snap_dir / "snapshot-ckpt.db").read_bytes() == before_ckpt
    assert _action_rows(live.db_path) == snapshot_actions
    assert _outbox_count(live.db_path) == 0


def test_reset_is_404_when_demo_mode_is_off(make_settings: Callable[..., Settings], tmp_path: Path) -> None:
    with TestClient(create_app(make_settings(tmp_path, demo_mode=False))) as client:
        assert client.post("/api/demo/reset").status_code == 404


def test_reset_shifts_timestamps_without_changing_durations(
    make_settings: Callable[..., Settings], brakes: dict[str, str], tmp_path: Path
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    built = make_settings(source)
    other = {
        "sender": "p.santos@example.com",
        "subject": "Overcharged on my invoice",
        "body": "I was charged twice for the same service.",
    }
    with TestClient(create_app(built)) as client:
        first = client.post("/api/cases", json=brakes).json()
        second = client.post("/api/cases", json=other).json()
    durations = {first["id"]: _duration(first), second["id"]: _duration(second)}
    newest = max(_newest(first), _newest(second))
    shift_restored_clock(
        built.db_path,
        built.checkpoint_path,
        now=newest - timedelta(days=10) + timedelta(minutes=2),
    )
    snap_dir = tmp_path / "demo"
    snap_dir.mkdir()
    shutil.copyfile(built.db_path, snap_dir / "snapshot.db")
    shutil.copyfile(built.checkpoint_path, snap_dir / "snapshot-ckpt.db")
    before_db = (snap_dir / "snapshot.db").read_bytes()
    before_ckpt = (snap_dir / "snapshot-ckpt.db").read_bytes()

    live = make_settings(tmp_path / "live", demo_mode=True, demo_dir=snap_dir)
    with TestClient(create_app(live)) as client:
        assert client.post("/api/demo/reset").status_code == 200
        restored = {
            item["id"]: client.get(f"/api/cases/{item['id']}").json() for item in client.get("/api/cases").json()
        }
        for case_id, duration in durations.items():
            assert _duration(restored[case_id]) == duration
        newest_after = max(_newest(item) for item in restored.values())
        assert abs(datetime.now(timezone.utc) - newest_after) <= timedelta(minutes=5)
        intake_at = restored[first["id"]]["state"]["events"][0]["at"]
        approved = client.post(
            f"/api/cases/{first['id']}/review",
            json={"decision": "approve", "reviewer": "Demo", "approved_action_ids": []},
        )
        assert approved.status_code == 200
        assert approved.json()["state"]["events"][0]["at"] == intake_at
    assert (snap_dir / "snapshot.db").read_bytes() == before_db
    assert (snap_dir / "snapshot-ckpt.db").read_bytes() == before_ckpt


def test_reset_while_a_case_is_running_returns_409(
    make_settings: Callable[..., Settings],
    wait_for_status: Callable[..., dict[str, object]],
    brakes: dict[str, str],
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    built = Settings(llm_provider="offline", db_path=source / "app.db", checkpoint_path=source / "ckpt.db")
    with TestClient(create_app(built)) as client:
        client.post("/api/cases", json=brakes)
    snap_dir = tmp_path / "demo"
    snap_dir.mkdir()
    (snap_dir / "snapshot.db").write_bytes((source / "app.db").read_bytes())
    (snap_dir / "snapshot-ckpt.db").write_bytes((source / "ckpt.db").read_bytes())

    brain = _HoldDraft()
    live = make_settings(tmp_path / "live", demo_mode=True, demo_dir=snap_dir)
    with TestClient(create_app(live, brain=brain)) as client:
        client.post("/api/demo/reset")
        before = [item["id"] for item in client.get("/api/cases").json()]
        started = client.post("/api/cases/async", json=brakes)
        case_id = started.json()["id"]
        assert brain.started.wait(timeout=10)
        blocked = client.post("/api/demo/reset")
        assert blocked.status_code == 409
        brain.release.set()
        assert wait_for_status(client, case_id)["status"] == "awaiting_review"
        after = [item["id"] for item in client.get("/api/cases").json()]
    assert case_id in after
    assert set(before).issubset(set(after))
    assert len(after) == len(before) + 1


def test_reset_during_sync_create_or_assistant_returns_409(
    make_settings: Callable[..., Settings], brakes: dict[str, str], tmp_path: Path
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    built = Settings(llm_provider="offline", db_path=source / "app.db", checkpoint_path=source / "ckpt.db")
    with TestClient(create_app(built)) as client:
        client.post("/api/cases", json=brakes)
    snap_dir = tmp_path / "demo"
    snap_dir.mkdir()
    (snap_dir / "snapshot.db").write_bytes((source / "app.db").read_bytes())
    (snap_dir / "snapshot-ckpt.db").write_bytes((source / "ckpt.db").read_bytes())

    brain = _HoldDraft()
    live = make_settings(tmp_path / "live", demo_mode=True, demo_dir=snap_dir)
    with TestClient(create_app(live, brain=brain)) as client:
        client.post("/api/demo/reset")

        def create() -> int:
            return client.post("/api/cases", json=brakes).status_code

        worker = threading.Thread(target=create)
        worker.start()
        assert brain.started.wait(timeout=10)
        blocked = client.post("/api/demo/reset")
        assert blocked.status_code == 409
        brain.release.set()
        worker.join(timeout=10)
        assert not worker.is_alive()

        ready = client.post("/api/cases", json=brakes).json()
        held = threading.Event()
        release = threading.Event()
        assistant = client.app.state.svc.assistant
        original = assistant.ask

        def ask(case_id: str, question: str) -> object:
            held.set()
            assert release.wait(timeout=10)
            return original(case_id, question)

        client.app.state.svc.assistant.ask = ask  # type: ignore[method-assign]

        def call_assistant() -> None:
            client.post(f"/api/cases/{ready['id']}/assistant", json={"question": "What is this about?"})

        assistant_thread = threading.Thread(target=call_assistant)
        assistant_thread.start()
        assert held.wait(timeout=10)
        during_assistant = client.post("/api/demo/reset")
        assert during_assistant.status_code == 409
        release.set()
        assistant_thread.join(timeout=10)


def test_reset_during_a_case_list_returns_409(
    make_settings: Callable[..., Settings], brakes: dict[str, str], tmp_path: Path
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    built = Settings(llm_provider="offline", db_path=source / "app.db", checkpoint_path=source / "ckpt.db")
    with TestClient(create_app(built)) as client:
        client.post("/api/cases", json=brakes)
    snap_dir = tmp_path / "demo"
    snap_dir.mkdir()
    (snap_dir / "snapshot.db").write_bytes((source / "app.db").read_bytes())
    (snap_dir / "snapshot-ckpt.db").write_bytes((source / "ckpt.db").read_bytes())

    live = make_settings(tmp_path / "live", demo_mode=True, demo_dir=snap_dir)
    with TestClient(create_app(live)) as client:
        client.post("/api/demo/reset")
        store = client.app.state.svc.store
        original = store.list_cases
        held = threading.Event()
        release = threading.Event()

        def slow(*args: object, **kwargs: object) -> list[dict[str, object]]:
            held.set()
            assert release.wait(timeout=10)
            return original(*args, **kwargs)  # type: ignore[arg-type]

        store.list_cases = slow  # type: ignore[method-assign]

        def listing() -> None:
            client.get("/api/cases")

        thread = threading.Thread(target=listing)
        thread.start()
        assert held.wait(timeout=10)
        blocked = client.post("/api/demo/reset")
        assert blocked.status_code == 409
        release.set()
        thread.join(timeout=10)
        assert not thread.is_alive()
        assert client.get("/api/cases").status_code == 200


def test_a_shifted_timestamp_equal_to_another_original_is_shifted_once(tmp_path: Path) -> None:
    earlier = "2020-01-01T00:00:00+00:00"
    later = "2020-01-01T00:05:00+00:00"
    db_path = tmp_path / "cases.db"
    ckpt_path = tmp_path / "ckpt.db"
    store = Store(db_path)
    store.upsert_case(
        "case-1",
        earlier,
        {
            "email": {"sender": "a@example.com", "subject": "One", "body": "Body"},
            "events": [
                {"at": earlier, "step": "intake", "actor": "system", "detail": "in"},
                {"at": later, "step": "triage", "actor": "agent", "detail": "done"},
            ],
            "status": "awaiting_review",
        },
    )
    store.close()
    connection = sqlite3.connect(ckpt_path)
    connection.execute(
        """CREATE TABLE checkpoints (
            thread_id TEXT, checkpoint_ns TEXT, checkpoint_id TEXT, checkpoint BLOB, metadata BLOB
        )"""
    )
    connection.execute(
        "INSERT INTO checkpoints VALUES (?,?,?,?,?)",
        ("case-1", "", "1", f"{earlier}|{later}".encode(), b"{}"),
    )
    connection.commit()
    connection.close()
    shift_restored_clock(db_path, ckpt_path, now=datetime.fromisoformat("2020-01-01T00:12:00+00:00"))
    shifted = sqlite3.connect(ckpt_path).execute("SELECT checkpoint FROM checkpoints").fetchone()[0]
    text = shifted.decode()
    assert text == "2020-01-01T00:05:00+00:00|2020-01-01T00:10:00+00:00"


def test_health_reports_the_snapshot_case_count(
    make_settings: Callable[..., Settings], spam: dict[str, str], tmp_path: Path
) -> None:
    demo = tmp_path / "demo"
    demo.mkdir()
    (demo / "snapshot.json").write_text(
        json.dumps(
            {
                "cases": [
                    {"sample": "S01-brake-noise", "id": "s01-id"},
                    {"sample": "S18-seo-spam", "id": "s18-id"},
                ]
            }
        ),
        encoding="utf-8",
    )
    settings = make_settings(tmp_path, demo_mode=True, demo_dir=demo)
    with TestClient(create_app(settings)) as client:
        health = client.get("/api/health").json()
    assert health["snapshot_cases"] == 2
    assert health["s01_case_id"] == "s01-id"
