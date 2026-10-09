"""Move timestamps on a restored demo database so the inbox is not two days old.

The snapshot files are never opened here. Callers pass the live copies only.
Every stored event moves by the same offset, so the gap between the first and
last event of a case stays the same.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

TARGET_AGE = timedelta(minutes=2)


def _parse(value: str) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _format(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).isoformat(timespec="seconds")


def shift_restored_clock(db_path: Path, checkpoint_path: Path, *, now: datetime | None = None) -> None:
    """Put the newest stored event about two minutes before `now`."""
    moment = now or datetime.now(timezone.utc)
    try:
        connection = sqlite3.connect(db_path)
    except sqlite3.Error as exc:
        logger.error("could not open the case database to shift timestamps (%s)", type(exc).__name__)
        return
    mapping: dict[str, str] = {}
    try:
        rows = connection.execute("SELECT id, created_at, updated_at, state_json FROM cases").fetchall()
        newest: datetime | None = None
        loaded: list[tuple[str, str, str, dict]] = []
        for case_id, created_at, updated_at, state_json in rows:
            try:
                state = json.loads(state_json)
            except json.JSONDecodeError as exc:
                logger.error("could not read timestamps for case %s (%s)", case_id, type(exc).__name__)
                continue
            if not isinstance(state, dict):
                logger.error("could not read timestamps for case %s (state is not an object)", case_id)
                continue
            for event in state.get("events") or []:
                if not isinstance(event, dict):
                    continue
                parsed = _parse(str(event.get("at") or ""))
                if parsed is not None and (newest is None or parsed > newest):
                    newest = parsed
            loaded.append((str(case_id), str(created_at), str(updated_at), state))
        if newest is None:
            return
        offset = (moment - TARGET_AGE) - newest

        def mapped(value: str) -> str:
            cached = mapping.get(value)
            if cached is not None:
                return cached
            parsed = _parse(value)
            if parsed is None:
                return value
            updated = _format(parsed + offset)
            mapping[value] = updated
            return updated

        for case_id, created_at, updated_at, state in loaded:
            for event in state.get("events") or []:
                if isinstance(event, dict) and isinstance(event.get("at"), str):
                    event["at"] = mapped(event["at"])
            connection.execute(
                "UPDATE cases SET created_at = ?, updated_at = ?, state_json = ? WHERE id = ?",
                (mapped(created_at), mapped(updated_at), json.dumps(state, ensure_ascii=False), case_id),
            )
        connection.commit()
    except sqlite3.Error as exc:
        logger.error("could not write shifted case timestamps (%s)", type(exc).__name__)
        return
    finally:
        connection.close()
    _shift_checkpoint(checkpoint_path, mapping)


def _shift_checkpoint(path: Path, mapping: dict[str, str]) -> None:
    if not mapping or not path.exists():
        return
    pairs: list[tuple[bytes, bytes]] = []
    for old, new in mapping.items():
        old_bytes = old.encode("utf-8")
        new_bytes = new.encode("utf-8")
        if len(old_bytes) != len(new_bytes):
            logger.error("skipped a timestamp whose shifted form changed length")
            continue
        pairs.append((old_bytes, new_bytes))
    if not pairs:
        return
    replacements = dict(pairs)
    try:
        connection = sqlite3.connect(path)
    except sqlite3.Error as exc:
        logger.error("could not open the checkpoint database to shift timestamps (%s)", type(exc).__name__)
        return
    try:
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        if "checkpoints" in tables:
            stored = connection.execute(
                "SELECT thread_id, checkpoint_ns, checkpoint_id, checkpoint, metadata FROM checkpoints"
            ).fetchall()
            for thread_id, checkpoint_ns, checkpoint_id, checkpoint, metadata in stored:
                connection.execute(
                    """UPDATE checkpoints SET checkpoint = ?, metadata = ?
                       WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id = ?""",
                    (
                        _replace(checkpoint, replacements),
                        _replace(metadata, replacements),
                        thread_id,
                        checkpoint_ns,
                        checkpoint_id,
                    ),
                )
        if "writes" in tables:
            stored = connection.execute(
                "SELECT thread_id, checkpoint_ns, checkpoint_id, task_id, idx, value FROM writes"
            ).fetchall()
            for thread_id, checkpoint_ns, checkpoint_id, task_id, idx, value in stored:
                connection.execute(
                    """UPDATE writes SET value = ?
                       WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id = ? AND task_id = ? AND idx = ?""",
                    (_replace(value, replacements), thread_id, checkpoint_ns, checkpoint_id, task_id, idx),
                )
        connection.commit()
    except sqlite3.Error as exc:
        logger.error("could not write shifted checkpoint timestamps (%s)", type(exc).__name__)
    finally:
        connection.close()


def _replace(blob: bytes | None, mapping: dict[bytes, bytes]) -> bytes | None:
    """Replace each original timestamp once. A shifted value is not shifted again."""
    if not blob or not mapping:
        return blob
    pattern = re.compile(b"|".join(re.escape(old) for old in sorted(mapping, key=len, reverse=True)))
    return pattern.sub(lambda match: mapping[match.group(0)], blob)
