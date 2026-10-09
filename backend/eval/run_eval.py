"""Triage and guard evaluation against the labelled synthetic emails.

    python -m eval.run_eval                 # uses LLM_PROVIDER / LLM_MODEL from .env
    python -m eval.run_eval --file data/holdout_emails.jsonl --repeat 3
    LLM_PROVIDER=offline python -m eval.run_eval

The labels in data/sample_emails.jsonl were written by the author of this repo.
They are a fixture, not ground truth, and the report says which model and prompt
version produced each number so results can be compared over time.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.brain import Brain, build_brain
from app.completion import Completion
from app.config import ROOT, get_settings
from app.guards import screen_input
from app.prompts import PROMPT_VERSION
from app.schemas import SEVERITIES, InboundEmail, Triage

SAMPLES = ROOT / "data" / "sample_emails.jsonl"
RESULTS = ROOT / "eval" / "results"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate triage and the injection guard.")
    parser.add_argument("--repeat", type=int, default=1, help="How many times to triage each email.")
    parser.add_argument(
        "--file",
        default=str(SAMPLES),
        help="JSONL of labelled emails. Defaults to data/sample_emails.jsonl.",
    )
    args = parser.parse_args(argv)
    if args.repeat < 1:
        parser.error("--repeat must be at least 1")
    return args


def email_file(raw: str) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = ROOT / path
    return path


def _band(values: list[float]) -> dict[str, float]:
    return {"mean": round(sum(values) / len(values), 3), "min": round(min(values), 3), "max": round(max(values), 3)}


def _unwrap(result: object) -> object:
    if isinstance(result, Completion):
        return result.value
    return result


def evaluate(brain: Brain, rows: list[dict[str, Any]], repeats: int, source_name: str) -> dict[str, Any]:
    n = len(rows)
    cat_rates: list[float] = []
    sev_rates: list[float] = []
    sev_direction: Counter[str] = Counter()
    confusion: Counter[tuple[str, str]] = Counter()
    critical_under_calls = 0
    critical_under_call_ids: list[str] = []
    unstable = 0
    tp = fp = fn = 0
    details: list[dict[str, Any]] = []

    prepared: list[tuple[dict[str, Any], InboundEmail, bool]] = []
    for row in rows:
        email = InboundEmail(sender=row["sender"], subject=row["subject"], body=row["body"])
        guard = screen_input(email)
        prepared.append((row, email, guard.injection_suspected))
        tp += guard.injection_suspected and row["label_injection"]
        fp += guard.injection_suspected and not row["label_injection"]
        fn += (not guard.injection_suspected) and row["label_injection"]

    predictions: list[list[dict[str, str]]] = [[] for _ in prepared]
    for run in range(repeats):
        cat_ok = sev_ok = 0
        for index, (row, email, flagged) in enumerate(prepared):
            triage = _unwrap(brain.triage(email))
            if not isinstance(triage, Triage):
                raise TypeError(f"triage returned {type(triage).__name__}")
            predictions[index].append(
                {"category": triage.category, "severity": triage.severity, "rationale": triage.rationale}
            )
            cat_ok += triage.category == row["label_category"]
            sev_ok += triage.severity == row["label_severity"]
            if triage.category != row["label_category"]:
                confusion[(row["label_category"], triage.category)] += 1
            if triage.severity != row["label_severity"]:
                delta = SEVERITIES.index(triage.severity) - SEVERITIES.index(row["label_severity"])
                sev_direction["higher" if delta > 0 else "lower"] += 1
            label_is_critical = row["label_severity"] == "critical"
            predicted_lower = SEVERITIES.index(triage.severity) < SEVERITIES.index("critical")
            if label_is_critical and predicted_lower:
                critical_under_calls += 1
                if row["id"] not in critical_under_call_ids:
                    critical_under_call_ids.append(row["id"])
            print(
                f"run {run + 1}/{repeats} {row['id']:<22} "
                f"{row['label_category']:>18}/{row['label_severity']:<8} -> "
                f"{triage.category:>18}/{triage.severity:<8}"
                f"{'  [injection]' if flagged else ''}"
            )
        cat_rates.append(cat_ok / n)
        sev_rates.append(sev_ok / n)

    for (row, _, flagged), runs in zip(prepared, predictions, strict=True):
        categories = {item["category"] for item in runs}
        severities = {item["severity"] for item in runs}
        if len(categories) > 1 or len(severities) > 1:
            unstable += 1
        details.append(
            {
                "id": row["id"],
                "label": [row["label_category"], row["label_severity"]],
                "predicted": [[item["category"], item["severity"]] for item in runs],
                "rationales": [item["rationale"] for item in runs],
                "injection_flag": flagged,
            }
        )

    return {
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": brain.name,
        "prompt_version": PROMPT_VERSION,
        "file": source_name,
        "n": n,
        "repeats": repeats,
        "category_agreement": _band(cat_rates),
        "severity_agreement": _band(sev_rates),
        "unstable_emails": unstable,
        "severity_disagreements": dict(sev_direction),
        "critical_under_calls": critical_under_calls,
        "critical_under_call_ids": critical_under_call_ids,
        "category_confusions": {f"{a} -> {b}": c for (a, b), c in confusion.most_common()},
        "injection_guard": {"true_positive": int(tp), "false_positive": int(fp), "false_negative": int(fn)},
        "details": details,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    settings = get_settings()
    brain = build_brain(settings)
    path = email_file(args.file)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    report = evaluate(brain, rows, args.repeat, path.name)
    cat = report["category_agreement"]
    sev = report["severity_agreement"]
    guard = report["injection_guard"]
    print()
    print(f"model={report['model']} prompt={PROMPT_VERSION} n={report['n']} repeats={args.repeat}")
    print(f"category agreement: mean {cat['mean']:.1%} (min {cat['min']:.1%}, max {cat['max']:.1%})")
    print(f"severity agreement: mean {sev['mean']:.1%} (min {sev['min']:.1%}, max {sev['max']:.1%})")
    print(f"emails whose category or severity changed between runs: {report['unstable_emails']}")
    print(f"severity disagreements: {report['severity_disagreements']}")
    print(f"critical under-calls: {report['critical_under_calls']} ids={report['critical_under_call_ids']}")
    print(f"injection guard: TP={guard['true_positive']} FP={guard['false_positive']} FN={guard['false_negative']}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    slug = brain.name.replace(":", "_")
    out = RESULTS / f"{stamp}-{slug}-{PROMPT_VERSION}-x{args.repeat}.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"report: {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
