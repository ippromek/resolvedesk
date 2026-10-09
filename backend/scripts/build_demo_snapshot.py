"""Build the demo snapshot from real graph runs.

Refuses offline mode unless --allow-offline is passed. An offline snapshot
would show rule-based replies in a demo that is supposed to be the model.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from app.brain import build_brain, build_proposer
from app.config import ROOT, Settings, get_settings
from app.db import Store
from app.graph import build_graph, event, validate_review
from app.investigation import Enterprise, Investigator
from app.kb import get_kb
from app.prompts import INVESTIGATE_PROMPT_VERSION, PROMPT_VERSION
from app.schemas import InboundEmail, ReviewInput

SAMPLES = ROOT / "data" / "sample_emails.jsonl"
HOLDOUT = ROOT / "data" / "holdout_emails.jsonl"
DEMO_DIR = ROOT / "data" / "demo"

# sample id, expected status, action type to approve (sent cases only)
CASE_PLAN: tuple[tuple[str, str, str | None], ...] = (
    ("S01-brake-noise", "awaiting_review", None),
    ("S24-inject-refund", "awaiting_review", None),
    ("S21-ar-brakes", "awaiting_review", None),
    ("S30-recall", "awaiting_review", None),
    ("S09-overcharge", "sent", "open_refund_review"),
    ("S16-data-delete", "sent", "forward_to_data_protection"),
    ("S18-seo-spam", "closed_no_reply", None),
    ("H13-subtle-inject", "awaiting_review", None),
)


def _load_rows(path: Path) -> dict[str, dict[str, Any]]:
    found: dict[str, dict[str, Any]] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        print(f"Could not read {path}: {exc}")
        raise SystemExit(1) from exc
    for line in lines:
        if not line.strip():
            continue
        row = json.loads(line)
        found[str(row["id"])] = {
            "sender": row["sender"],
            "subject": row["subject"],
            "body": row["body"],
        }
    return found


def _emails() -> dict[str, dict[str, str]]:
    rows = _load_rows(SAMPLES)
    rows.update(_load_rows(HOLDOUT))
    missing = [sample_id for sample_id, _status, _action in CASE_PLAN if sample_id not in rows]
    if missing:
        print("Missing emails: " + ", ".join(missing))
        raise SystemExit(1)
    return rows


def _approve(graph: Any, store: Store, case_id: str, action_type: str) -> None:
    config = {"configurable": {"thread_id": case_id}}
    snapshot = graph.get_state(config)
    proposals = list(snapshot.values.get("proposals") or [])
    chosen = [item["id"] for item in proposals if item.get("type") == action_type]
    if not chosen:
        names = ", ".join(str(item.get("type")) for item in proposals) or "none"
        print(f"{case_id} did not propose {action_type} (got {names})")
        raise SystemExit(1)
    review = ReviewInput(
        decision="approve",
        reviewer="Demo",
        body=None,
        note=None,
        approved_action_ids=chosen,
        injection_acknowledged=False,
    )
    errors = validate_review(snapshot.values, review)
    if errors:
        print(f"{case_id} cannot be approved: {errors[0]}")
        raise SystemExit(1)
    graph.invoke(Command(resume=review.model_dump()), config)
    state = dict(graph.get_state(config).values)
    store.upsert_case(case_id, state["events"][0]["at"], state)


def build(settings: Settings, dest: Path) -> dict[str, Any]:
    emails = _emails()
    work = Path(tempfile.mkdtemp(prefix="demo-snapshot-"))
    live = Settings(
        llm_provider=settings.llm_provider,
        llm_model=settings.llm_model,
        llm_temperature=settings.llm_temperature,
        llm_timeout_s=settings.llm_timeout_s,
        llm_max_retries=settings.llm_max_retries,
        db_path=work / "cases.db",
        checkpoint_path=work / "ckpt.db",
        kb_dir=settings.kb_dir,
        enterprise_dir=settings.enterprise_dir,
        demo_mode=False,
        demo_dir=settings.demo_dir,
    )
    brain = build_brain(live)
    store = Store(live.db_path)
    investigator = Investigator(live, Enterprise.load(live.enterprise_dir), store)
    proposer = build_proposer(live)
    connection = sqlite3.connect(live.checkpoint_path, check_same_thread=False)
    graph = build_graph(
        brain,
        get_kb(live.kb_dir),
        investigator,
        proposer,
        SqliteSaver(connection),
        store=store,
    )
    built: list[dict[str, str]] = []
    try:
        for sample_id, expected, action_type in CASE_PLAN:
            email = InboundEmail(**emails[sample_id])
            case_id = str(uuid.uuid4())
            print(f"running {sample_id}")
            graph.invoke(
                {
                    "case_id": case_id,
                    "email": email.model_dump(),
                    "events": [event("intake", f"email received from {email.sender}", actor="system")],
                },
                {"configurable": {"thread_id": case_id}},
            )
            state = dict(graph.get_state({"configurable": {"thread_id": case_id}}).values)
            store.upsert_case(case_id, state["events"][0]["at"], state)
            if action_type:
                _approve(graph, store, case_id, action_type)
                state = dict(graph.get_state({"configurable": {"thread_id": case_id}}).values)
            status = str(state.get("status"))
            if status != expected:
                print(f"{sample_id} ended as {status}, expected {expected}")
                raise SystemExit(1)
            built.append(
                {
                    "sample": sample_id,
                    "id": case_id,
                    "sender": email.sender,
                    "subject": email.subject,
                    "status": status,
                }
            )
            print(f"  {status}")
    finally:
        try:
            connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        except sqlite3.Error as exc:
            print(f"Could not checkpoint the build database: {exc}")
        connection.close()
        store.close()
        if not built or len(built) != len(CASE_PLAN):
            shutil.rmtree(work, ignore_errors=True)

    dest.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(live.db_path, dest / "snapshot.db")
    shutil.copyfile(live.checkpoint_path, dest / "snapshot-ckpt.db")
    manifest = {
        "model": brain.name,
        "prompt_version": PROMPT_VERSION,
        "investigate_prompt_version": INVESTIGATE_PROMPT_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "cases": built,
    }
    (dest / "snapshot.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    shutil.rmtree(work, ignore_errors=True)
    print(f"wrote {dest / 'snapshot.json'}")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the ResolveDesk demo snapshot.")
    parser.add_argument("--allow-offline", action="store_true", help="Permit an offline snapshot.")
    args = parser.parse_args(argv)
    try:
        settings = get_settings()
    except RuntimeError as exc:
        print(exc)
        return 1
    if settings.llm_provider == "offline" and not args.allow_offline:
        print("Refusing to build a demo snapshot in offline mode. Pass --allow-offline to override.")
        return 1
    try:
        build(settings, DEMO_DIR)
    except SystemExit as exc:
        code = exc.code
        return code if isinstance(code, int) else 1
    except Exception as exc:  # noqa: BLE001  # a CLI failure prints the type only, never the message
        print(f"Snapshot build failed: {type(exc).__name__}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
