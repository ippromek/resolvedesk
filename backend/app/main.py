from __future__ import annotations

import json
import logging
import shutil
import sqlite3
import threading
import uuid
from collections.abc import Iterator
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from app.actions import Proposer, compose_final_body
from app.assistant import Assistant
from app.brain import Brain, build_brain, build_proposer
from app.config import ROOT, Settings, get_settings
from app.db import Store
from app.demo_clock import shift_restored_clock
from app.failures import log_failure, step_failure_message
from app.graph import build_graph, event, validate_review
from app.investigation import Enterprise, Investigator
from app.kb import get_kb
from app.prompts import INVESTIGATE_PROMPT_VERSION, PROMPT_VERSION
from app.schemas import AssistantAnswer, AssistantQuestion, InboundEmail, PreviewInput, ReviewInput

SAMPLES = ROOT / "data" / "sample_emails.jsonl"
_REVIEW_LOCKS: dict[str, threading.Lock] = {}
_REVIEW_LOCKS_GUARD = threading.Lock()


def _case_lock(case_id: str) -> threading.Lock:
    with _REVIEW_LOCKS_GUARD:
        lock = _REVIEW_LOCKS.get(case_id)
        if lock is None:
            lock = threading.Lock()
            _REVIEW_LOCKS[case_id] = lock
        return lock


CASE_MODEL_FAILURE = "The model did not respond; the case was not created. Try again."
ASSISTANT_MODEL_FAILURE = "The model did not respond. Try again."
RESTART_FAILURE = "interrupted by a server restart"
logger = logging.getLogger(__name__)


class Services:
    def __init__(
        self,
        settings: Settings,
        *,
        investigator: Investigator | None = None,
        proposer: Proposer | None = None,
        brain: Brain | None = None,
    ) -> None:
        self.settings = settings
        self.store = Store(settings.db_path)
        self.kb = get_kb(settings.kb_dir)
        self.brain = brain or build_brain(settings)
        # Offline versus the live model is decided here, from settings only.
        # Tests that script triage and draft pass their own offline collaborators.
        enterprise = Enterprise.load(settings.enterprise_dir)
        self.investigator = investigator or Investigator(settings, enterprise, self.store)
        self.proposer = proposer or build_proposer(settings)
        settings.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        self._ckpt_conn = sqlite3.connect(settings.checkpoint_path, check_same_thread=False)
        self.graph = build_graph(
            self.brain,
            self.kb,
            self.investigator,
            self.proposer,
            SqliteSaver(self._ckpt_conn),
            store=self.store,
        )
        self.assistant = Assistant(settings, self.store, self.kb)
        self._flight_lock = threading.Lock()
        self.in_flight: set[str] = set()
        self._gate = threading.Lock()
        self._busy = 0
        self._resetting = False

    def acquire_worker(self) -> bool:
        """Count one active request. False while a reset is in progress."""
        with self._gate:
            if self._resetting:
                return False
            self._busy += 1
            return True

    def release_worker(self) -> None:
        with self._gate:
            self._busy -= 1

    def acquire_reset(self) -> bool:
        """False while any request or background run is active."""
        with self._gate:
            if self._busy > 0 or self.in_flight:
                return False
            self._resetting = True
            return True

    def release_reset(self) -> None:
        with self._gate:
            self._resetting = False

    def begin_inflight(self, case_id: str) -> bool:
        """Record a running case. The caller holds that case's lock."""
        with self._flight_lock:
            if case_id in self.in_flight:
                return False
            self.in_flight.add(case_id)
            return True

    def end_inflight(self, case_id: str) -> None:
        """Drop a running case. The caller holds that case's lock."""
        with self._flight_lock:
            self.in_flight.discard(case_id)

    def any_inflight(self) -> bool:
        with self._flight_lock:
            return bool(self.in_flight)

    def close(self) -> None:
        try:
            self._ckpt_conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        except sqlite3.Error as exc:
            logger.error("could not close the checkpoint database (%s)", type(exc).__name__)
        self._ckpt_conn.close()
        self.store.close()


def _drop_sidecars(path: Path) -> None:
    for suffix in ("-wal", "-shm"):
        sidecar = Path(str(path) + suffix)
        if sidecar.exists():
            sidecar.unlink()


def _install_db(source: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    _drop_sidecars(dest)
    shutil.copyfile(source, dest)


def _fail_interrupted(store: Store) -> None:
    """A daemon run dies with the process. Offer Retry instead of leaving the row spinning."""
    try:
        rows = store.list_cases(status="processing", limit=500)
    except Exception as exc:  # noqa: BLE001  # listing interrupted cases can fail if the database is already closed
        logger.error("could not list interrupted cases (%s)", type(exc).__name__)
        return
    for row in rows:
        state = row.get("state")
        if not isinstance(state, dict) or "email" not in state:
            logger.error("skipped interrupted case %s (missing email)", row.get("id"))
            continue
        events = list(state.get("events") or [])
        events.append(event("failed", RESTART_FAILURE, actor="system"))
        events[-1]["failed_step"] = "agent"
        updated = dict(state)
        updated["events"] = events
        updated["status"] = "failed"
        created = events[0]["at"] if events else str(row["created_at"])
        try:
            store.upsert_case(str(row["id"]), created, updated)
        except Exception as exc:  # noqa: BLE001  # marking one row failed can fail while the database is closing
            logger.error("could not mark interrupted case %s failed (%s)", row.get("id"), type(exc).__name__)


def create_app(
    settings: Settings | None = None,
    *,
    investigator: Investigator | None = None,
    proposer: Proposer | None = None,
    brain: Brain | None = None,
) -> FastAPI:
    if not logging.getLogger().handlers:
        logging.basicConfig(format="%(asctime)s %(levelname)s %(name)s %(message)s")
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.svc = Services(settings, investigator=investigator, proposer=proposer, brain=brain)
        _fail_interrupted(app.state.svc.store)
        yield
        app.state.svc.close()

    app = FastAPI(title="ResolveDesk", version="0.1.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5180", "http://127.0.0.1:5180"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def svc() -> Services:
        return app.state.svc

    @contextmanager
    def _hold_worker() -> Iterator[None]:
        """Refuse the request while a reset is replacing the database."""
        if not svc().acquire_worker():
            raise HTTPException(409, "a case is still running")
        try:
            yield
        finally:
            svc().release_worker()

    def cfg(case_id: str) -> dict[str, Any]:
        return {"configurable": {"thread_id": case_id}}

    def sync_case(case_id: str) -> dict[str, Any]:
        s = svc()
        snapshot = s.graph.get_state(cfg(case_id))
        state = dict(snapshot.values)
        created = state["events"][0]["at"]
        s.store.upsert_case(case_id, created, state)
        case = s.store.get_case(case_id)
        assert case is not None
        case["next"] = list(snapshot.next)
        return case

    def snapshot_info() -> tuple[int, str | None]:
        manifest = settings.demo_dir / "snapshot.json"
        if not manifest.exists():
            return 0, None
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.error("could not read the demo snapshot (%s)", type(exc).__name__)
            return 0, None
        cases = data.get("cases") if isinstance(data, dict) else None
        if not isinstance(cases, list):
            return 0, None
        s01 = next(
            (
                str(item["id"])
                for item in cases
                if isinstance(item, dict) and item.get("sample") == "S01-brake-noise" and item.get("id")
            ),
            None,
        )
        return len(cases), s01

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        with _hold_worker():
            count, s01 = snapshot_info()
            return {
                "ok": True,
                "model": svc().brain.name,
                "prompt_version": PROMPT_VERSION,
                "investigate_prompt_version": INVESTIGATE_PROMPT_VERSION,
                "demo_mode": settings.demo_mode,
                "snapshot_cases": count,
                "s01_case_id": s01,
            }

    @app.get("/api/graph")
    def graph_diagram() -> dict[str, str]:
        with _hold_worker():
            return {"mermaid": svc().graph.get_graph().draw_mermaid()}

    @app.get("/api/samples")
    def samples() -> list[dict[str, Any]]:
        with _hold_worker():
            if not SAMPLES.exists():
                return []
            rows = [json.loads(line) for line in SAMPLES.read_text(encoding="utf-8").splitlines() if line.strip()]
            return [{k: r[k] for k in ("id", "sender", "subject", "body")} for r in rows]

    @app.get("/api/metrics")
    def metrics() -> dict[str, Any]:
        with _hold_worker():
            return svc().store.metrics()

    @app.get("/api/cases")
    def list_cases(
        status: str | None = None, category: str | None = None, severity: str | None = None, q: str | None = None
    ) -> list[dict[str, Any]]:
        with _hold_worker():
            rows = svc().store.list_cases(status=status, category=category, severity=severity, text=q)
            for r in rows:
                r.pop("state", None)
            return rows

    @app.post("/api/cases", status_code=201)
    def create_case(email: InboundEmail) -> dict[str, Any]:
        case_id = str(uuid.uuid4())
        if not svc().acquire_worker():
            raise HTTPException(409, "a case is still running")
        try:
            svc().graph.invoke(
                {
                    "case_id": case_id,
                    "email": email.model_dump(),
                    "events": [event("intake", f"email received from {email.sender}", actor="system")],
                },
                cfg(case_id),
            )
        except HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001  # logged, then the caller sees the safe failure
            log_failure(case_id, "create", exc)
            raise HTTPException(503, CASE_MODEL_FAILURE) from None
        else:
            return sync_case(case_id)
        finally:
            svc().release_worker()

    def seed_case(case_id: str, email: InboundEmail) -> dict[str, Any]:
        intake = event("intake", f"email received from {email.sender}", actor="system")
        state: dict[str, Any] = {
            "case_id": case_id,
            "email": email.model_dump(),
            "events": [intake],
            "status": "processing",
        }
        svc().store.upsert_case(case_id, intake["at"], state)
        return state

    def mark_failed(case_id: str, step: str, exc: BaseException) -> None:
        case = svc().store.get_case(case_id)
        if not case:
            return
        state = dict(case["state"])
        events = list(state.get("events") or [])
        events.append(event("failed", f"{step} failed: {step_failure_message(exc)}", actor="system"))
        events[-1]["failed_step"] = step
        state["events"] = events
        state["status"] = "failed"
        created = events[0]["at"]
        svc().store.upsert_case(case_id, created, state)

    def failed_step(case_id: str) -> str:
        try:
            nxt = tuple(svc().graph.get_state(cfg(case_id)).next)
        except Exception as exc:  # noqa: BLE001  # the checkpoint may be missing after a reset; fall back to step "agent"
            logger.error("could not read the checkpoint for case %s (%s)", case_id, type(exc).__name__)
            return "agent"
        return str(nxt[0]) if nxt else "agent"

    def run_stream(case_id: str, graph_input: dict[str, Any] | None) -> None:
        try:
            for _chunk in svc().graph.stream(graph_input, cfg(case_id), stream_mode="values", durability="sync"):
                sync_case(case_id)
            sync_case(case_id)
        except Exception as exc:  # noqa: BLE001  # logged, then the caller sees the safe failure
            step = failed_step(case_id)
            log_failure(case_id, step, exc)
            try:
                mark_failed(case_id, step, exc)
            except Exception as inner:  # noqa: BLE001  # the process may already be closing the database
                log_failure(case_id, step, inner)
        finally:
            with _case_lock(case_id):
                svc().end_inflight(case_id)
            svc().release_worker()

    def start_stream(case_id: str, graph_input: dict[str, Any] | None) -> bool:
        """The caller already holds a worker slot. The background thread releases it."""
        with _case_lock(case_id):
            if not svc().begin_inflight(case_id):
                svc().release_worker()
                return False
        threading.Thread(target=run_stream, args=(case_id, graph_input), daemon=True).start()
        return True

    def run_resume(case_id: str) -> None:
        try:
            svc().graph.invoke(None, cfg(case_id))
            sync_case(case_id)
        except Exception as exc:  # noqa: BLE001  # logged, then the caller sees the safe failure
            step = failed_step(case_id)
            log_failure(case_id, step, exc)
            try:
                mark_failed(case_id, step, exc)
            except Exception as inner:  # noqa: BLE001  # the process may already be closing the database
                log_failure(case_id, step, inner)
        finally:
            with _case_lock(case_id):
                svc().end_inflight(case_id)
            svc().release_worker()

    @app.post("/api/cases/async", status_code=202)
    def create_case_async(email: InboundEmail) -> JSONResponse:
        if not svc().acquire_worker():
            raise HTTPException(409, "a case is still running")
        case_id = str(uuid.uuid4())
        state = seed_case(case_id, email)
        if not start_stream(case_id, state):
            raise HTTPException(409, "a case is still running")
        return JSONResponse({"id": case_id}, status_code=202)

    @app.post("/api/cases/{case_id}/retry", status_code=202)
    def retry_case(case_id: str) -> JSONResponse:
        if not svc().store.get_case(case_id):
            raise HTTPException(404, "case not found")
        with _case_lock(case_id):
            case = svc().store.get_case(case_id)
            if not case:
                raise HTTPException(404, "case not found")
            if case["status"] != "failed":
                raise HTTPException(409, "case is not failed")
            if not svc().acquire_worker():
                raise HTTPException(409, "a case is still running")
            if not svc().begin_inflight(case_id):
                svc().release_worker()
                raise HTTPException(409, "case is still running")
            state = dict(case["state"])
            state["status"] = "processing"
            svc().store.upsert_case(case_id, state["events"][0]["at"], state)
        threading.Thread(target=run_resume, args=(case_id,), daemon=True).start()
        return JSONResponse({"id": case_id}, status_code=202)

    @app.post("/api/demo/reset")
    def reset_demo() -> dict[str, int]:
        if not settings.demo_mode:
            raise HTTPException(404, "not found")
        snapshot_db = settings.demo_dir / "snapshot.db"
        snapshot_ckpt = settings.demo_dir / "snapshot-ckpt.db"
        if not snapshot_db.exists() or not snapshot_ckpt.exists():
            raise HTTPException(404, "demo snapshot not found")
        current = svc()
        if not current.acquire_reset():
            raise HTTPException(409, "a case is still running")
        try:
            current.close()
            try:
                _install_db(snapshot_db, settings.db_path)
                _install_db(snapshot_ckpt, settings.checkpoint_path)
            except OSError as exc:
                logger.error("could not copy the demo snapshot (%s)", type(exc).__name__)
                app.state.svc = Services(settings, investigator=investigator, proposer=proposer, brain=brain)
                raise HTTPException(500, "could not restore the demo snapshot") from None
            try:
                shift_restored_clock(settings.db_path, settings.checkpoint_path)
            except Exception as exc:  # noqa: BLE001  # an unexpected checkpoint shape must not undo the restored cases
                logger.error("could not shift restored timestamps (%s)", type(exc).__name__)
            app.state.svc = Services(settings, investigator=investigator, proposer=proposer, brain=brain)
            return {"cases": int(app.state.svc.store.metrics()["total"])}
        finally:
            if app.state.svc is current:
                current.release_reset()

    @app.get("/api/cases/{case_id}")
    def get_case(case_id: str) -> dict[str, Any]:
        with _hold_worker():
            case = svc().store.get_case(case_id)
            if not case:
                raise HTTPException(404, "case not found")
            try:
                case["next"] = list(svc().graph.get_state(cfg(case_id)).next)
            except Exception as exc:  # noqa: BLE001  # a missing checkpoint still returns the case, without a next step
                logger.error("could not read the checkpoint for case %s (%s)", case_id, type(exc).__name__)
                case["next"] = []
            return case

    @app.post("/api/cases/{case_id}/preview")
    def preview(case_id: str, body: PreviewInput) -> dict[str, str]:
        s = svc()
        if not s.store.get_case(case_id):
            raise HTTPException(404, "case not found")
        if not s.acquire_worker():
            raise HTTPException(409, "a case is still running")
        try:
            state = s.graph.get_state(cfg(case_id)).values
            proposals = list(state.get("proposals") or [])
            known = {item["id"] for item in proposals}
            unknown = [action_id for action_id in body.approved_action_ids if action_id not in known]
            if unknown:
                raise HTTPException(422, "unknown action id: " + ", ".join(unknown))
            text = body.body if body.body is not None else str((state.get("draft") or {}).get("body") or "")
            language = str((state.get("triage") or {}).get("language") or "en")
            final_body, arranged = compose_final_body(text, proposals, body.approved_action_ids, language)
            return {"final_body": final_body, "arranged_section": arranged}
        finally:
            s.release_worker()

    @app.post("/api/cases/{case_id}/review")
    def review(case_id: str, body: ReviewInput) -> dict[str, Any]:
        s = svc()
        if not s.store.get_case(case_id):
            raise HTTPException(404, "case not found")
        if not s.acquire_worker():
            raise HTTPException(409, "a case is still running")
        try:
            with _case_lock(case_id):
                snapshot = s.graph.get_state(cfg(case_id))
                if tuple(snapshot.next) != ("human_review",):
                    raise HTTPException(409, "case is not awaiting review")
                errors = validate_review(snapshot.values, body)
                if errors:
                    raise HTTPException(422, errors[0])
                if not s.begin_inflight(case_id):
                    raise HTTPException(409, "case is still running")
                try:
                    s.graph.invoke(Command(resume=body.model_dump()), cfg(case_id))
                    return sync_case(case_id)
                except HTTPException:
                    raise
                except Exception as exc:  # noqa: BLE001  # logged, then the caller sees the safe failure
                    step = failed_step(case_id)
                    log_failure(case_id, step, exc)
                    mark_failed(case_id, step, exc)
                    raise HTTPException(503, step_failure_message(exc)) from None
                finally:
                    s.end_inflight(case_id)
        finally:
            s.release_worker()

    @app.post("/api/cases/{case_id}/assistant")
    def ask(case_id: str, body: AssistantQuestion) -> AssistantAnswer:
        if not svc().store.get_case(case_id):
            raise HTTPException(404, "case not found")
        if not svc().acquire_worker():
            raise HTTPException(409, "a case is still running")
        try:
            return svc().assistant.ask(case_id, body.question)
        except HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001  # logged, then the caller sees the safe failure
            log_failure(case_id, "assistant", exc)
            raise HTTPException(503, ASSISTANT_MODEL_FAILURE) from None
        finally:
            svc().release_worker()

    return app


app = create_app()
