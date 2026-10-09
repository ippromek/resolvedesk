"""SQLite read model for the UI.

The LangGraph checkpointer is the source of truth for a case's workflow state.
This table is a projection of it, written after every graph run, so the inbox
can list and filter cases without replaying checkpoints.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS cases (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    sender TEXT NOT NULL,
    subject TEXT NOT NULL,
    status TEXT NOT NULL,
    category TEXT,
    severity TEXT,
    injection_suspected INTEGER NOT NULL DEFAULT 0,
    review_decision TEXT,
    state_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS outbox (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id TEXT NOT NULL REFERENCES cases(id),
    recipient TEXT NOT NULL,
    body TEXT NOT NULL,
    approved_by TEXT NOT NULL,
    queued_at TEXT NOT NULL,
    -- A reply can only be queued with a named approver, and only once per case.
    CHECK (length(approved_by) > 0),
    UNIQUE(case_id)
);
CREATE TABLE IF NOT EXISTS actions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id TEXT NOT NULL,
    action_id TEXT NOT NULL,
    type TEXT NOT NULL,
    args_json TEXT NOT NULL,
    approved_by TEXT NOT NULL,
    executed_at TEXT NOT NULL,
    status TEXT NOT NULL,
    CHECK (length(approved_by) > 0),
    UNIQUE(case_id, action_id)
);
"""


class Store:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._conn.executescript(SCHEMA)

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            except sqlite3.Error as exc:
                logger.error("could not close the case database (%s)", type(exc).__name__)
            self._conn.close()

    def upsert_case(self, case_id: str, created_at: str, state: dict[str, Any]) -> None:
        triage = state.get("triage") or {}
        review = state.get("review") or {}
        email = state["email"]
        events = state.get("events") or []
        updated = events[-1]["at"] if events else created_at
        with self._lock, self._conn:
            self._conn.execute(
                """INSERT INTO cases (id, created_at, updated_at, sender, subject, status, category,
                       severity, injection_suspected, review_decision, state_json)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(id) DO UPDATE SET updated_at=excluded.updated_at, status=excluded.status,
                       category=excluded.category, severity=excluded.severity,
                       injection_suspected=excluded.injection_suspected,
                       review_decision=excluded.review_decision, state_json=excluded.state_json""",
                (
                    case_id,
                    created_at,
                    updated,
                    email["sender"],
                    email["subject"],
                    state.get("status", "processing"),
                    triage.get("category"),
                    triage.get("severity"),
                    int(bool((state.get("guard") or {}).get("injection_suspected"))),
                    review.get("decision"),
                    json.dumps(state, ensure_ascii=False),
                ),
            )

    def record_action(
        self,
        case_id: str,
        action_id: str,
        action_type: str,
        args: dict[str, Any],
        approved_by: str,
        at: str,
        status: str,
    ) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                """INSERT OR IGNORE INTO actions
                       (case_id, action_id, type, args_json, approved_by, executed_at, status)
                   VALUES (?,?,?,?,?,?,?)""",
                (case_id, action_id, action_type, json.dumps(args, ensure_ascii=False), approved_by, at, status),
            )

    def list_actions(self, case_id: str | None = None) -> list[dict[str, Any]]:
        sql = "SELECT * FROM actions"
        args: tuple[Any, ...] = ()
        if case_id is not None:
            sql += " WHERE case_id = ?"
            args = (case_id,)
        sql += " ORDER BY id"
        with self._lock:
            rows = self._conn.execute(sql, args).fetchall()
        found: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["args"] = json.loads(item.pop("args_json"))
            found.append(item)
        return found

    def list_outbox(self, case_id: str | None = None) -> list[dict[str, Any]]:
        sql = "SELECT id, case_id, recipient, body, approved_by, queued_at FROM outbox"
        args: tuple[Any, ...] = ()
        if case_id is not None:
            sql += " WHERE case_id = ?"
            args = (case_id,)
        sql += " ORDER BY id"
        with self._lock:
            rows = self._conn.execute(sql, args).fetchall()
        return [dict(row) for row in rows]

    def queue_outbound(self, case_id: str, recipient: str, body: str, approved_by: str, at: str) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                """INSERT OR IGNORE INTO outbox (case_id, recipient, body, approved_by, queued_at)
                   VALUES (?,?,?,?,?)""",
                (case_id, recipient, body, approved_by, at),
            )

    def get_case(self, case_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM cases WHERE id = ?", (case_id,)).fetchone()
        return _row(row) if row else None

    def list_cases(
        self,
        status: str | None = None,
        category: str | None = None,
        severity: str | None = None,
        text: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        sql = "SELECT * FROM cases WHERE 1=1"
        args: list[Any] = []
        for col, val in (("status", status), ("category", category), ("severity", severity)):
            if val:
                sql += f" AND {col} = ?"
                args.append(val)
        if text:
            escaped = text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            like = f"%{escaped}%"
            sql += (
                " AND (subject LIKE ? ESCAPE '\\'"
                " OR json_extract(state_json, '$.email.body') LIKE ? ESCAPE '\\'"
                " OR json_extract(state_json, '$.triage.summary') LIKE ? ESCAPE '\\'"
                " OR json_extract(state_json, '$.findings.summary') LIKE ? ESCAPE '\\')"
            )
            args += [like, like, like, like]
        sql += " ORDER BY created_at DESC LIMIT ?"
        args.append(limit)
        with self._lock:
            rows = self._conn.execute(sql, args).fetchall()
        return [_row(r) for r in rows]

    def metrics(self) -> dict[str, Any]:
        with self._lock:
            by_status = dict(self._conn.execute("SELECT status, count(*) FROM cases GROUP BY status").fetchall())
            by_cat = dict(
                self._conn.execute(
                    "SELECT category, count(*) FROM cases WHERE category IS NOT NULL GROUP BY category"
                ).fetchall()
            )
            by_sev = dict(
                self._conn.execute(
                    "SELECT severity, count(*) FROM cases WHERE severity IS NOT NULL GROUP BY severity"
                ).fetchall()
            )
            decisions = dict(
                self._conn.execute(
                    "SELECT review_decision, count(*) FROM cases WHERE review_decision IS NOT NULL GROUP BY review_decision"
                ).fetchall()
            )
            injections = self._conn.execute("SELECT count(*) FROM cases WHERE injection_suspected = 1").fetchone()[0]
            total = self._conn.execute("SELECT count(*) FROM cases").fetchone()[0]
        reviewed = sum(decisions.values())
        return {
            "total": total,
            "by_status": by_status,
            "by_category": by_cat,
            "by_severity": by_sev,
            "review_decisions": decisions,
            # The share of drafts a human accepted unchanged is the closest thing this
            # system has to a quality signal in production.
            "approved_unchanged_rate": round(decisions.get("approve", 0) / reviewed, 2) if reviewed else None,
            "injection_flags": injections,
            **_model_usage(self),
            "actions": _action_counts(self),
        }


def _model_usage(store: Store) -> dict[str, Any]:
    """Average triage/draft latency and total tokens, from events that recorded them."""
    triage_ms: list[int] = []
    draft_ms: list[int] = []
    investigate_ms: list[int] = []
    total_tokens = 0
    saw_tokens = False
    with store._lock:
        rows = store._conn.execute("SELECT state_json FROM cases").fetchall()
    for (state_json,) in rows:
        try:
            state = json.loads(state_json)
        except json.JSONDecodeError:
            continue
        for ev in state.get("events") or []:
            if not isinstance(ev, dict):
                continue
            step = ev.get("step")
            detail = str(ev.get("detail") or "")
            latency = ev.get("latency_ms")
            summary = step == "investigate" and detail.startswith("investigation finished")
            if isinstance(latency, (int, float)) and step == "triage":
                triage_ms.append(int(latency))
            elif isinstance(latency, (int, float)) and step == "draft":
                draft_ms.append(int(latency))
            elif isinstance(latency, (int, float)) and summary:
                investigate_ms.append(int(latency))
            # Per-tool investigate rows keep their own token counts for the audit line.
            # The summary event holds the total, so counting both would double it.
            if step == "investigate" and not summary:
                continue
            input_tokens = ev.get("input_tokens")
            output_tokens = ev.get("output_tokens")
            if input_tokens is not None or output_tokens is not None:
                saw_tokens = True
                total_tokens += int(input_tokens or 0) + int(output_tokens or 0)

    def _avg(values: list[int]) -> int | None:
        if not values:
            return None
        return round(sum(values) / len(values))

    return {
        "avg_latency_ms": {"triage": _avg(triage_ms), "draft": _avg(draft_ms), "investigate": _avg(investigate_ms)},
        "total_tokens": total_tokens if saw_tokens else None,
    }


def _action_counts(store: Store) -> dict[str, dict[str, int]]:
    """Proposed and dropped come from case events. Approved comes from the actions table."""
    proposed: dict[str, int] = {}
    dropped: dict[str, int] = {}
    approved: dict[str, int] = {}
    with store._lock:
        rows = store._conn.execute("SELECT state_json FROM cases").fetchall()
        executed = store._conn.execute(
            "SELECT type, count(*) FROM actions WHERE status = 'executed' GROUP BY type"
        ).fetchall()
    for action_type, count in executed:
        approved[str(action_type)] = int(count)
    for (state_json,) in rows:
        try:
            state = json.loads(state_json)
        except json.JSONDecodeError:
            continue
        for proposal in state.get("proposals") or []:
            if isinstance(proposal, dict) and proposal.get("type"):
                name = str(proposal["type"])
                proposed[name] = proposed.get(name, 0) + 1
        for ev in state.get("events") or []:
            if not isinstance(ev, dict) or ev.get("step") != "propose_actions":
                continue
            detail = str(ev.get("detail") or "")
            if not detail.startswith("dropped "):
                continue
            name = detail.split(":", 1)[0].removeprefix("dropped ").strip()
            if name:
                dropped[name] = dropped.get(name, 0) + 1
    return {"proposed": proposed, "approved": approved, "dropped": dropped}


def _row(row: sqlite3.Row) -> dict[str, Any]:
    data = dict(row)
    data["state"] = json.loads(data.pop("state_json"))
    data["injection_suspected"] = bool(data["injection_suspected"])
    return data
