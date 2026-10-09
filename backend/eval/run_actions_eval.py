"""Check proposed actions against the reviewer-written expectations.

    python -m eval.run_actions_eval --repeat 3

`must_include_any` passes when the list is empty or at least one of those types
was proposed. This does not send a reply. The graph stops at human review.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
import uuid
from pathlib import Path
from typing import Any

from langgraph.checkpoint.sqlite import SqliteSaver

from app.brain import build_brain, build_proposer
from app.config import ROOT, Settings, get_settings
from app.db import Store
from app.graph import build_graph, event
from app.investigation import Enterprise, Investigator
from app.kb import get_kb
from app.prompts import PROMPT_VERSION
from app.schemas import InboundEmail

EXPECTATIONS = Path(__file__).with_name("action_expectations.json")
SAMPLES = ROOT / "data" / "sample_emails.jsonl"
_MONEY = re.compile(r"(AED\s?\d|\d[\d,]*\s?(AED|dirham)|\b\d+\s?%)", re.IGNORECASE)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate proposed actions up to human review.")
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args(argv)
    if args.repeat < 1:
        parser.error("--repeat must be at least 1")
    return args


def _samples() -> dict[str, dict[str, Any]]:
    found: dict[str, dict[str, Any]] = {}
    for line in SAMPLES.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        found[row["id"].split("-", 1)[0]] = row
    return found


def _money(proposal: dict[str, Any]) -> bool:
    blob = proposal.get("rationale", "") + " " + " ".join(str(value) for value in (proposal.get("args") or {}).values())
    return _MONEY.search(blob) is not None


def _check(rule: dict[str, Any], proposals: list[dict[str, Any]]) -> list[str]:
    types = [item["type"] for item in proposals]
    problems: list[str] = []
    required = rule.get("must_include_any") or []
    if required and not any(name in types for name in required):
        problems.append(f"missing any of {required}; got {types or 'nothing'}")
    for banned in rule.get("must_not_include") or []:
        if banned in types:
            problems.append(f"included {banned}")
    if rule.get("no_amount_anywhere"):
        for proposal in proposals:
            if _money(proposal):
                problems.append(f"money amount in {proposal['type']}")
    return problems


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rules = json.loads(EXPECTATIONS.read_text(encoding="utf-8"))
    rows = _samples()
    base = get_settings()
    db_dir = ROOT / "data"
    settings = Settings(
        llm_provider=base.llm_provider,
        llm_model=base.llm_model,
        llm_temperature=base.llm_temperature,
        llm_timeout_s=base.llm_timeout_s,
        llm_max_retries=base.llm_max_retries,
        db_path=db_dir / "actions-eval.db",
        checkpoint_path=db_dir / "actions-eval-ckpt.db",
        kb_dir=base.kb_dir,
        enterprise_dir=base.enterprise_dir,
    )
    brain = build_brain(settings)
    kb = get_kb(settings.kb_dir)
    store = Store(settings.db_path)
    investigator = Investigator(settings, Enterprise.load(settings.enterprise_dir), store)
    proposer = build_proposer(settings)
    settings.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(settings.checkpoint_path, check_same_thread=False)
    graph = build_graph(brain, kb, investigator, proposer, SqliteSaver(connection), store=store)
    print(f"model={brain.name} prompt={PROMPT_VERSION} repeats={args.repeat}")
    failed = 0
    try:
        for key, rule in rules.items():
            row = rows[key]
            email = InboundEmail(sender=row["sender"], subject=row["subject"], body=row["body"])
            for run in range(args.repeat):
                case_id = str(uuid.uuid4())
                try:
                    graph.invoke(
                        {
                            "case_id": case_id,
                            "email": email.model_dump(),
                            "events": [event("intake", f"email received from {email.sender}", actor="system")],
                        },
                        {"configurable": {"thread_id": case_id}},
                    )
                    state = graph.get_state({"configurable": {"thread_id": case_id}}).values
                    proposals = list(state.get("proposals") or [])
                    problems = _check(rule, proposals)
                except Exception as exc:  # noqa: BLE001  # an eval failure records the type only, never the message
                    proposals = []
                    problems = [f"error: {type(exc).__name__}"]
                types = [item["type"] for item in proposals]
                if problems:
                    failed += 1
                    print(f"FAIL {key} run {run + 1}/{args.repeat}: {'; '.join(problems)} proposals={types}")
                else:
                    print(f"PASS {key} run {run + 1}/{args.repeat}: {types or 'none'}")
    finally:
        connection.close()
    print(f"failed={failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
